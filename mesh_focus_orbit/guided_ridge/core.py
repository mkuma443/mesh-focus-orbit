"""Guided Ridge component.

Loaded after foundation in the package namespace. The component boundary is
structural only; all route/prepare/curve state remains the existing state.
"""

import bpy
import bmesh
import blf
import copy
import gpu
import heapq
import hashlib
import json
import math
import os
import statistics
import struct
import tempfile
import time
import numpy as np
from array import array
from collections import deque
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
from mathutils.bvhtree import BVHTree

from .. import runtime as _runtime
from .. import lifecycle as _lifecycle
from . import curve_sculpt as _curve_sculpt

from ..config import (
    FACE_SET_ACTIVATION_OPERATOR_ID,
    FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD,
    FILL_PREVIEW_TERMINAL_GROWTH_STAGES,
    FILL_PREVIEW_TERMINAL_MAX_RADIUS_FACTOR,
    FILL_PREVIEW_TERMINAL_THRESHOLD_FACTOR,
    FILL_PREVIEW_WHEEL_DRAIN_SECONDS,
    GUIDED_RIDGE_CURVE_DEFAULT_SMOOTHING,
    GUIDED_RIDGE_CURVE_SCULPT_STEP1,
    GUIDED_RIDGE_CURVE_SCULPT_STEP2,
    GUIDED_RIDGE_DISTANCE_MAX_PAIRS,
    GUIDED_RIDGE_KEY,
    GUIDED_RIDGE_MAX_CANDIDATE_FACES,
    GUIDED_RIDGE_MAX_CONTROLS,
    GUIDED_RIDGE_PREPARE_WORK_CHUNK,
    GUIDED_RIDGE_UI_PROTOTYPE,
    GUIDED_RIDGE_WIDTH_ANCHOR_BAND_FRACTION,
    GUIDED_RIDGE_WIDTH_EDGE_MULTIPLIER,
    GUIDED_RIDGE_WIDTH_GUIDE_BAND_FRACTION,
    GUIDED_RIDGE_WIDTH_MAX_EXTENT_FRACTION,
    GUIDED_RIDGE_WIDTH_MIN_EDGE_MULTIPLIER,
    GUIDED_RIDGE_WIDTH_STEP_FACTOR,
    GUIDED_RIDGE_OPERATOR_ID,
    LOCAL_FACE_SET_GROW_KEY,
    LOCAL_FACE_SET_GROW_OPERATOR_ID,
    LOCAL_FEATURE_BRUSH_KEY,
    LOCAL_FEATURE_BRUSH_OPERATOR_ID,
    LOCAL_FEATURE_BRUSH_MARKER_CATALOG_ID,
    LOCAL_FEATURE_BRUSH_MARKER_PROPERTY,
    LOCAL_FEATURE_BRUSH_MARKER_VALUE,
    OPERATOR_ID,
    RECOVER_FACE_SET_STATE_OPERATOR_ID,
    TOOL_FACE_SET_OPERATOR_ID,
    TOOL_NORMAL_OPERATOR_ID,
    TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID,
    TOPOLOGY_COLOR_ATTRIBUTE_NAME,
    TOPOLOGY_COLOR_PALETTE,
    TOPOLOGY_COLOR_PANEL_CATEGORY,
    TUBE_SHAPE_KEY,
    TUBE_SHAPE_OPERATOR_ID,
    WATCHER_OPERATOR_ID,
)

# Keep this dependency local to the component. Tests may replace this
# reference without mutating Blender's shared view3d_utils module.
_guided_ridge_project_3d_to_region_2d = view3d_utils.location_3d_to_region_2d


def _tool_event_coordinate(context, event):
    """Resolve an explicit WorkSpaceTool click coordinate."""
    return _sculpt_cursor_region_coordinate(context, event)

class VIEW3D_OT_mesh_focus_orbit_tool(bpy.types.Operator):
    """Single-click entry point used by the resident Normal MFO tool."""

    bl_idname = TOOL_NORMAL_OPERATOR_ID
    bl_label = "Mesh Focus Orbit (Click)"
    bl_description = "Focus the orbit target on the mesh surface under the click"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
            and context.space_data is not None
            and context.mode in {"OBJECT", "EDIT_MESH", "SCULPT"}
        )

    def invoke(self, context, event):
        if not self.poll(context):
            return {"CANCELLED"}
        coordinate = _tool_event_coordinate(context, event)
        if coordinate is None:
            self.report({"WARNING"}, "MFO: click is outside the 3D View region")
            return {"CANCELLED"}
        if not _start_normal_tool_session(context, coordinate):
            self.report({"WARNING"}, "MFO: no visible mesh surface under click")
            return {"CANCELLED"}
        return {"FINISHED"}


class VIEW3D_OT_mesh_focus_face_set_tool(bpy.types.Operator):
    """Single-click strict/Face Set entry point used by the MFO tool."""

    bl_idname = TOOL_FACE_SET_OPERATOR_ID
    bl_label = "Mesh Focus Orbit (Strict Face Set)"
    bl_description = "Focus the configured Reference Object Face Set under the click"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
            and context.space_data is not None
            and context.mode in {"OBJECT", "EDIT_MESH"}
        )

    def invoke(self, context, event):
        if not self.poll(context):
            return {"CANCELLED"}
        coordinate = _tool_event_coordinate(context, event)
        if coordinate is None:
            self.report({"WARNING"}, "FSMFO: click is outside the 3D View region")
            return {"CANCELLED"}

        # OFF is a runtime-only teardown and must not create a second Undo
        # step.  ON is routed through the existing Activation operator below,
        # whose UNDO flag is the established Face Set contract.
        try:
            area_key = context.area.as_pointer()
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            return {"CANCELLED"}
        existing = _runtime.active_states.get(area_key)
        if existing is not None and existing.active:
            if isinstance(existing, _FaceSetOrbitState):
                _deactivate_state(existing)
                return {"FINISHED"}
            return {"CANCELLED"}

        result = bpy.ops.view3d.mesh_focus_face_set_activate(
            "EXEC_DEFAULT",
            True,
            mouse_region_x=int(round(coordinate.x)),
            mouse_region_y=int(round(coordinate.y)),
            use_click_coordinate=True,
        )
        if "FINISHED" not in result:
            self.report(
                {"WARNING"},
                "FSMFO: configured Reference Object has no visible Face Set under click",
            )
        return result


class VIEW3D_OT_mesh_focus_orbit_recover_face_set_state(bpy.types.Operator):
    """Emergency recovery for an orphaned Face Set MFO scene state."""

    bl_idname = RECOVER_FACE_SET_STATE_OPERATOR_ID
    bl_label = "Mesh Focus Orbit: Recover Face Set Mode"
    bl_description = (
        "Remove orphaned Face Set MFO proxies and restore the Reference viewport state"
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, _context):
        # The operation is intentionally independent of the active object and
        # mode so it remains available when those were changed by Undo.
        return True

    def execute(self, _context):
        _finish_face_set_states()
        removed_proxies, removed_meshes = _cleanup_orphan_face_set_proxies()
        self.report(
            {"INFO"},
            "MFO Face Set recovery: "
            f"{len(removed_proxies)} proxy object(s), "
            f"{len(removed_meshes)} mesh datablock(s) removed",
        )
        return {"FINISHED"}


class VIEW3D_OT_mesh_focus_orbit_watcher(bpy.types.Operator):
    """Monitor one owned FSMFO session without creating Undo data."""

    bl_idname = WATCHER_OPERATOR_ID
    bl_label = "Mesh Focus Orbit Watcher"
    bl_options = {"INTERNAL"}

    def invoke(self, context, _event):
        try:
            area_key = context.area.as_pointer()
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            return {"CANCELLED"}
        state = _runtime.active_states.get(area_key)
        if not _session_is_current(state):
            return {"CANCELLED"}
        if getattr(state, "watcher_operator_key", None) is not None:
            # A second Watcher must never replace the owner of a live session.
            return {"CANCELLED"}

        operator_key = _operator_key(self)
        try:
            timer = context.window_manager.event_timer_add(
                0.25,
                window=context.window,
            )
            self._mfo_area_key = area_key
            self._mfo_session_id = state.session_id
            self._mfo_timer = timer
            state.watcher_operator_key = operator_key
            state.watcher_timer = timer
            context.window_manager.modal_handler_add(self)
            _lifecycle.modal_register(
                self,
                "MFO Watcher",
                state.session_id,
                state=state,
            )
        except (
            AttributeError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            _detach_watcher_operator(self)
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def modal(self, _context, event):
        area_key = getattr(self, "_mfo_area_key", 0)
        session_id = getattr(self, "_mfo_session_id", 0)
        state = _runtime.active_states.get(area_key)
        if (
            state is None
            or not state.active
            or state.session_id != session_id
            or state.watcher_operator_key != _operator_key(self)
        ):
            # This watcher lost ownership.  It must not touch the current
            # session's Proxy, Reference, isolation, hooks, or timers.
            _detach_watcher_operator(self)
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}

        if event.type == "TIMER":
            if (
                isinstance(state, _FaceSetOrbitState)
                and not _face_set_activation_artifact_present(state)
            ):
                # Native Undo removed the Activation step.  This is a normal
                # end event, not a damaged session: do not rebuild anything.
                _finish_state(state, restore_retopo=False)
                _detach_watcher_operator(self)
                _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                return {"CANCELLED"}
            return {"PASS_THROUGH"}

        if event.type == "WINDOW_DEACTIVATE":
            prefs = _addon_preferences()
            if prefs and prefs.focus_loss_behavior == "EXIT":
                _finish_state(state)
                _detach_watcher_operator(self)
                _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                return {"CANCELLED"}

        # Passing events through preserves Navigation Gizmo, MMB orbit, and
        # RetopoFlow's own modal input.
        return {"PASS_THROUGH"}

    def __del__(self):
        try:
            _detach_watcher_operator(self)
        except (ReferenceError, RuntimeError, AttributeError):
            pass


class VIEW3D_OT_mesh_focus_face_set_activate(bpy.types.Operator):
    """Create one Face Set MFO activation Undo step, then finish."""

    bl_idname = FACE_SET_ACTIVATION_OPERATOR_ID
    bl_label = "Mesh Focus Orbit: Face Set Activation"
    bl_options = {"INTERNAL", "UNDO"}

    # These transient fields are populated only by the resident T-tool.  The
    # legacy key/double-tap path leaves them at their defaults and therefore
    # continues to use the viewport-center raycast.
    mouse_region_x: IntProperty(options={"SKIP_SAVE"}, default=0)
    mouse_region_y: IntProperty(options={"SKIP_SAVE"}, default=0)
    use_click_coordinate: BoolProperty(options={"SKIP_SAVE"}, default=False)

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
            and context.space_data is not None
            and context.mode in {"OBJECT", "EDIT_MESH"}
        )

    def invoke(self, context, event):
        """Handle FSMFO's Ctrl+double-tap without an outer trigger operator."""
        if not self.poll(context):
            return {"PASS_THROUGH"}

        try:
            area_key = context.area.as_pointer()
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            return {"PASS_THROUGH"}

        prefs = _addon_preferences()
        activation_key = prefs.activation_key if prefs else "RIGHT_SHIFT"
        double_tap_window = (
            prefs.double_tap_window if prefs else DEFAULT_DOUBLE_TAP_WINDOW
        )
        if event.type != activation_key or event.value != "PRESS":
            return {"PASS_THROUGH"}
        if not getattr(event, "ctrl", False):
            return {"PASS_THROUGH"}
        if getattr(event, "alt", False) or getattr(event, "oskey", False):
            return {"PASS_THROUGH"}

        tap_key = ("face_set", area_key)
        now = time.monotonic()
        previous_tap = _runtime.last_tap_times.get(tap_key)
        if previous_tap is None or now - previous_tap > double_tap_window:
            _runtime.last_tap_times[tap_key] = now
            return {"PASS_THROUGH"}

        _runtime.last_tap_times.pop(tap_key, None)
        return self.execute(context)

    def execute(self, context):
        if not self.poll(context):
            return {"CANCELLED"}
        try:
            area_key = context.area.as_pointer()
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            return {"CANCELLED"}
        if area_key in _runtime.session_cleanup_areas:
            return {"CANCELLED"}
        existing = _runtime.active_states.get(area_key)
        if existing is not None and existing.active:
            if isinstance(existing, _FaceSetOrbitState):
                # FSMFO uses this same direct Activation operator for OFF.
                # CANCELLED avoids creating a second UNDO step for the
                # runtime-only teardown.
                _deactivate_state(existing)
                return {"CANCELLED"}
            return {"CANCELLED"}

        coordinate = None
        if bool(self.use_click_coordinate):
            coordinate = Vector((float(self.mouse_region_x), float(self.mouse_region_y)))
        if not _activate_face_set_session(context, coordinate=coordinate):
            return {"CANCELLED"}
        # This operator must end here.  Only the Watcher remains modal.
        return {"FINISHED"}


class VIEW3D_OT_mesh_focus_orbit(bpy.types.Operator):
    """Double-tap the configured key to start or stop an MFO session.

    Normal MFO is also available while Sculpt Mode is active.  Face Set MFO
    keeps its separate Object/Edit-only activation operator and keymap item.
    """

    bl_idname = OPERATOR_ID
    bl_label = "Mesh Focus Orbit"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
            and context.space_data is not None
            and context.mode in {"OBJECT", "EDIT_MESH", "SCULPT"}
        )

    def invoke(self, context, event):
        if not self.poll(context):
            return {"PASS_THROUGH"}

        area_key = context.area.as_pointer()
        prefs = _addon_preferences()
        activation_key = prefs.activation_key if prefs else "RIGHT_SHIFT"
        double_tap_window = (
            prefs.double_tap_window if prefs else DEFAULT_DOUBLE_TAP_WINDOW
        )
        if event.type != activation_key or event.value != "PRESS":
            return {"PASS_THROUGH"}
        if getattr(event, "alt", False) or getattr(event, "oskey", False):
            return {"PASS_THROUGH"}

        # FSMFO has its own direct Ctrl keymap item and its own Activation
        # operator.  This operator is intentionally normal-MFO only.
        if getattr(event, "ctrl", False):
            return {"PASS_THROUGH"}

        tap_key = ("normal", area_key)
        now = time.monotonic()
        previous_tap = _runtime.last_tap_times.get(tap_key)
        if previous_tap is None or now - previous_tap > double_tap_window:
            _runtime.last_tap_times[tap_key] = now
            return {"PASS_THROUGH"}
        _runtime.last_tap_times.pop(tap_key, None)

        existing_state = _runtime.active_states.get(area_key)
        if existing_state is not None and existing_state.active:
            if not isinstance(existing_state, _FaceSetOrbitState):
                _deactivate_state(existing_state)
                # This trigger operator has no UNDO flag, so manual OFF does
                # not create an extra Undo boundary.
                return {"CANCELLED"}
            return {"PASS_THROUGH"}

        if area_key in _runtime.session_cleanup_areas:
            return {"CANCELLED"}

        # Recovery is limited to tagged stale artifacts.  It never rebuilds
        # the new session and is not part of Undo handling.
        _cleanup_orphan_face_set_proxies()

        hit = _find_center_hit(context)
        if hit is None:
            return {"PASS_THROUGH"}
        state = _TempOrbitState(context, hit[0], hit[1], activation_key)
        if not _start_session(context, state, face_set=False):
            return {"PASS_THROUGH"}
        return {"FINISHED"}


def _clip_sculpt_cursor_segment_to_planes(rv3d, start, end):
    """Clip a world-space cursor segment by RegionView3D user planes.

    ``RegionView3D.clip_planes`` stores the same four-component planes used
    by Blender's native ``ED_view3d_clip_segment`` path.  The positive side
    of each plane is inside the viewport.  Keeping this small slab clip in
    Python avoids turning a rejected near/far segment back into an infinite
    Object.ray_cast ray.
    """
    if not bool(getattr(rv3d, "use_clip_planes", False)):
        return start, end

    try:
        segment = end - start
        lower = 0.0
        upper = 1.0
        for plane in getattr(rv3d, "clip_planes", ()):
            if len(plane) < 4:
                continue
            normal = Vector((float(plane[0]), float(plane[1]), float(plane[2])))
            if normal.length_squared <= 1.0e-20:
                continue
            distance = float(plane[3])
            value_start = float(normal.dot(start)) + distance
            value_end = float(normal.dot(end)) + distance
            if value_start < 0.0 and value_end < 0.0:
                return None
            if value_start < 0.0 or value_end < 0.0:
                fraction = value_start / (value_start - value_end)
                if value_start < 0.0:
                    lower = max(lower, fraction)
                else:
                    upper = min(upper, fraction)
                if lower > upper:
                    return None
        return start + segment * lower, start + segment * upper
    except (AttributeError, IndexError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _sculpt_cursor_ray_segment(context, coord, direction):
    """Return the same clipped world segment rendered for one cursor pixel.

    ``RegionView3D.perspective_matrix`` is Blender's ``window_matrix *
    view_matrix``.  Unprojecting that matrix at NDC z=-1 and z=+1 gives the
    actual near/far world points for both perspective and orthographic views;
    unlike ``region_2d_to_origin_3d`` alone, this cannot start before the
    viewport's near plane or continue beyond its far plane.  The endpoints
    are ordered along the public view3d_utils ray direction and then clipped
    by optional custom RegionView3D planes.
    """
    try:
        region = context.region
        rv3d = context.space_data.region_3d
        if region is None or rv3d is None:
            return None
        width = float(region.width)
        height = float(region.height)
        if width <= 0.0 or height <= 0.0:
            return None

        perspective_inverse = rv3d.perspective_matrix.inverted_safe()
        ndc_x = (2.0 * float(coord.x) / width) - 1.0
        ndc_y = (2.0 * float(coord.y) / height) - 1.0

        def unproject(ndc_z):
            point = perspective_inverse @ Vector((ndc_x, ndc_y, ndc_z, 1.0))
            if abs(float(point.w)) <= 1.0e-12:
                return None
            return Vector((point.x, point.y, point.z)) / float(point.w)

        near_point = unproject(-1.0)
        far_point = unproject(1.0)
        if near_point is None or far_point is None:
            return None
        segment = far_point - near_point
        if segment.length_squared <= 1.0e-20:
            return None

        view_direction = Vector(direction)
        if view_direction.length_squared <= 1.0e-20:
            return None
        view_direction.normalize()
        if float(segment.dot(view_direction)) < 0.0:
            near_point, far_point = far_point, near_point
        return _clip_sculpt_cursor_segment_to_planes(rv3d, near_point, far_point)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _raycast_sculpt_face_set(context, coord):
    """Return the first visible active-mesh face under the cursor.

    The cursor ray is bounded to Blender's rendered near/far segment and
    optional custom RegionView3D clip planes.  Object.ray_cast does not honor
    Sculpt face hiding.  When it returns a hidden face, advance the local ray
    past that hit and continue until a visible face is found.  The returned
    polygon index remains the face index used by the Face Set attribute and
    preview adjacency graph.
    """
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return None

    face_set_attr = obj.data.attributes.get(".sculpt_face_set")
    if face_set_attr is None or face_set_attr.domain != "FACE":
        return None

    try:
        coord = Vector(coord)
        region = context.region
        rv3d = context.space_data.region_3d
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        if direction.length_squared == 0.0:
            return None
        direction.normalize()

        world_segment = _sculpt_cursor_ray_segment(context, coord, direction)
        if world_segment is None:
            return None
        world_start, world_end = world_segment
        world_ray = world_end - world_start
        world_ray_length_squared = float(world_ray.length_squared)
        if world_ray_length_squared <= 1.0e-20:
            return None
        world_direction = world_ray.normalized()

        inverse = obj.matrix_world.inverted_safe()
        local_origin = inverse @ world_start
        local_direction = inverse.to_3x3() @ world_direction
        if local_direction.length_squared == 0.0:
            return None
        local_direction.normalize()

        for _attempt in range(256):
            hit, location, _local_normal, face_index = obj.ray_cast(
                local_origin,
                local_direction,
            )
            if not hit or face_index < 0 or face_index >= len(face_set_attr.data):
                return None
            polygon = obj.data.polygons[int(face_index)]
            world_location = obj.matrix_world @ location
            hit_fraction = float((world_location - world_start).dot(world_ray)) / world_ray_length_squared
            if hit_fraction < -1.0e-6 or hit_fraction > 1.0 + 1.0e-6:
                return None
            # Match Sculpt's PBVH cursor ray: the nearest non-hidden triangle
            # is the cursor surface regardless of winding.  ``use_frontface``
            # / BRUSH_FRONTFACE filters the brush's affected vertices; it is
            # not a seed-ray backface cull.  Rejecting by polygon normal here
            # would make an inside-out but visible surface fall through to a
            # farther Face Set, unlike native Sculpt.  Occluded geometry is
            # already excluded by the nearest-hit rule, and hidden faces are
            # handled by the retry below.
            if not bool(polygon.hide):
                return (
                    obj,
                    int(face_index),
                    int(face_set_attr.data[face_index].value),
                    world_location,
                    coord,
                )
            # Blender's object ray cast includes hidden Sculpt faces.  Move
            # past this surface in local space before retrying, otherwise the
            # same hidden polygon would be returned indefinitely.  Keep the
            # step small relative to the hit distance and cap it so large
            # models do not skip a nearby visible layer.
            hit_distance = max((world_location - world_start).length, 1.0e-7)
            advance = min(max(hit_distance * 1.0e-6, 1.0e-7), 1.0e-3)
            local_origin = inverse @ (world_location + world_direction * advance)
        return None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _raycast_visible_mesh_face(context, coord):
    """Return the first visible active-mesh face under a viewport coordinate."""
    obj = getattr(context, "active_object", None)
    if obj is None or getattr(obj, "type", None) != "MESH":
        return None
    try:
        coord = Vector(coord)
        region = context.region
        rv3d = context.space_data.region_3d
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        if direction.length_squared <= 1.0e-20:
            return None
        direction.normalize()
        world_segment = _sculpt_cursor_ray_segment(context, coord, direction)
        if world_segment is None:
            return None
        world_start, world_end = world_segment
        world_ray = world_end - world_start
        ray_length_squared = float(world_ray.length_squared)
        if ray_length_squared <= 1.0e-20:
            return None
        world_direction = world_ray.normalized()
        inverse = obj.matrix_world.inverted_safe()
        local_origin = inverse @ world_start
        local_direction = inverse.to_3x3() @ world_direction
        if local_direction.length_squared <= 1.0e-20:
            return None
        local_direction.normalize()
        for _attempt in range(256):
            hit, local_location, _local_normal, face_index = obj.ray_cast(
                local_origin, local_direction
            )
            if not hit or face_index < 0 or face_index >= len(obj.data.polygons):
                return None
            polygon = obj.data.polygons[int(face_index)]
            world_location = obj.matrix_world @ local_location
            hit_fraction = float(
                (world_location - world_start).dot(world_ray)
            ) / ray_length_squared
            if hit_fraction < -1.0e-6 or hit_fraction > 1.0 + 1.0e-6:
                return None
            if not bool(polygon.hide):
                return (
                    obj,
                    int(face_index),
                    world_location,
                    Vector(local_location),
                    coord,
                )
            hit_distance = max((world_location - world_start).length, 1.0e-7)
            advance = min(max(hit_distance * 1.0e-6, 1.0e-7), 1.0e-3)
            local_origin = inverse @ (world_location + world_direction * advance)
        return None
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _vertex_paint_active_color_attribute(obj):
    """Return the active supported Vertex Paint color attribute and reason."""
    try:
        color_attributes = obj.data.color_attributes
        attribute = getattr(color_attributes, "active_color", None)
        if attribute is None:
            return None, "no active color attribute"
        domain = str(attribute.domain)
        data_type = str(attribute.data_type)
        if domain not in {"POINT", "CORNER"}:
            return None, f"unsupported color domain: {domain}"
        if data_type not in {"FLOAT_COLOR", "BYTE_COLOR"}:
            return None, f"unsupported color type: {data_type}"
        return attribute, None
    except (AttributeError, IndexError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None, "active color attribute is unavailable"


def _vertex_paint_geometry_compatibility(obj):
    """Reject visible/evaluated geometry whose hit cannot map to source colors.

    The color writer targets the original Mesh attribute.  Until an explicit
    evaluated-triangle-to-source mapping exists, viewport modifiers and
    non-basis shape keys can make a ray hit differ from that source mesh.
    Unmodified meshes (and a basis-only shape-key datablock) retain the native
    polygon/loop correspondence and are safe to sample.
    """
    try:
        for modifier in obj.modifiers:
            if bool(getattr(modifier, "show_viewport", True)):
                return False, "Vertex Paint Smart Fill requires modifiers disabled in the viewport"
        shape_keys = getattr(obj.data, "shape_keys", None)
        key_blocks = getattr(shape_keys, "key_blocks", None) if shape_keys is not None else None
        if key_blocks is not None and len(key_blocks) > 1:
            return False, "Vertex Paint Smart Fill does not support non-basis shape keys"
        return True, None
    except (AttributeError, IndexError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False, "Vertex Paint Smart Fill could not verify source/evaluated geometry"


def _vertex_paint_color_signature(obj, attribute):
    try:
        return (
            int(obj.data.as_pointer()),
            str(attribute.name),
            str(attribute.domain),
            str(attribute.data_type),
            int(len(attribute.data)),
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _vertex_paint_triangle_barycentric(point, tri_points):
    import numpy as np

    origin = np.asarray(tri_points[0], dtype=np.float64)
    edge_a = np.asarray(tri_points[1], dtype=np.float64) - origin
    edge_b = np.asarray(tri_points[2], dtype=np.float64) - origin
    offset = np.asarray(point, dtype=np.float64) - origin
    d00 = float(np.dot(edge_a, edge_a))
    d01 = float(np.dot(edge_a, edge_b))
    d11 = float(np.dot(edge_b, edge_b))
    d20 = float(np.dot(offset, edge_a))
    d21 = float(np.dot(offset, edge_b))
    denominator = d00 * d11 - d01 * d01
    if denominator <= 1.0e-20:
        return None
    value_b = (d11 * d20 - d01 * d21) / denominator
    value_c = (d00 * d21 - d01 * d20) / denominator
    value_a = 1.0 - value_b - value_c
    values = np.asarray((value_a, value_b, value_c), dtype=np.float64)
    if not np.all(np.isfinite(values)) or float(np.min(values)) < -1.0e-5:
        return None
    return tuple(float(value) for value in values)


def _vertex_paint_sample_color(obj, face_index, local_location, attribute):
    """Sample the actual active color attribute at a hit face location."""
    import numpy as np

    mesh = obj.data
    polygon = mesh.polygons[int(face_index)]
    mesh.calc_loop_triangles()
    candidates = [
        triangle
        for triangle in mesh.loop_triangles
        if int(triangle.polygon_index) == int(face_index)
    ]
    selected_values = None
    selected_weights = None
    selected_ids = None
    for triangle in candidates:
        loop_ids = tuple(int(value) for value in triangle.loops)
        vertex_ids = tuple(int(value) for value in triangle.vertices)
        points = [mesh.vertices[index].co for index in vertex_ids]
        weights = _vertex_paint_triangle_barycentric(local_location, points)
        if weights is not None:
            selected_weights = weights
            selected_ids = loop_ids if str(attribute.domain) == "CORNER" else vertex_ids
            break
    if selected_ids is None:
        selected_ids = tuple(
            int(loop_index) if str(attribute.domain) == "CORNER" else int(mesh.loops[loop_index].vertex_index)
            for loop_index in polygon.loop_indices
        )
        selected_weights = tuple(1.0 / max(len(selected_ids), 1) for _ in selected_ids)
    values = np.asarray(
        [tuple(attribute.data[index].color) for index in selected_ids],
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[1] != 4 or not len(values):
        return None
    weights = np.asarray(selected_weights, dtype=np.float64)
    weights /= max(float(np.sum(weights)), 1.0e-20)
    return tuple(float(value) for value in np.sum(values * weights[:, None], axis=0))


def _sculpt_cursor_region_coordinate(context, event):
    """Return the event cursor in the owning View3D WINDOW region.

    Blender exposes both window-space (``mouse_x/y``) and region-space
    (``mouse_region_x/y``) event fields.  The latter is normally correct, but
    it is not a sufficient ownership check when an operator is reached from a
    keymap while a sibling region (header/sidebar/asset shelf) is under the
    pointer.  Smart Fill must never silently reinterpret such an
    event as a different point in the 3D view.

    Use the absolute event position and the actual WINDOW region origin as the
    authoritative conversion.  The region-space fields are retained only as
    a compatibility fallback for synthetic events and older event shims that
    do not expose window coordinates.  This helper is deliberately cursor
    only; it does not call or share the viewport-center ray helpers used by
    Mesh Focus Orbit.
    """
    try:
        region = context.region
        if region is None or str(region.type) != "WINDOW":
            return None
        width = float(region.width)
        height = float(region.height)
        if width <= 0.0 or height <= 0.0:
            return None

        # ``Region.x/y`` are screen/window coordinates in Blender.  Prefer
        # this path even when mouse_region_* are present so a stale or
        # sibling-region relative value cannot select another surface.
        mouse_x = getattr(event, "mouse_x", None)
        mouse_y = getattr(event, "mouse_y", None)
        if mouse_x is not None and mouse_y is not None:
            x = float(mouse_x) - float(region.x)
            y = float(mouse_y) - float(region.y)
            if 0.0 <= x < width and 0.0 <= y < height:
                return Vector((x, y))
            # Absolute coordinates are available and prove that this event
            # belongs to a sibling region (or another area).  Do not fall
            # back to a stale region-relative pair in that case.
            return None

        # Keep direct unit tests and old event shims usable when absolute
        # window coordinates are unavailable.  Still reject out-of-region
        # values instead of raycasting a guessed/clamped point.
        region_x = getattr(event, "mouse_region_x", None)
        region_y = getattr(event, "mouse_region_y", None)
        if region_x is None or region_y is None:
            return None
        x = float(region_x)
        y = float(region_y)
        if 0.0 <= x < width and 0.0 <= y < height:
            return Vector((x, y))
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None
    return None


# ---------------------------------------------------------------------------
# Guided Ridge
# ---------------------------------------------------------------------------

def _guided_ridge_smoothstep(value):
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _guided_ridge_catmull_rom(points, samples_per_segment=8, max_points=256):
    """Interpolate every control point and bound the viewport preview size."""
    points = [Vector(point) for point in points]
    if len(points) < 2:
        return points[:]
    if len(points) > 64:
        raise ValueError("Guided Ridge accepts at most 64 control points")
    segments = len(points) - 1
    steps = max(2, min(int(samples_per_segment), max(2, (max_points - 1) // segments)))
    result = []
    for index in range(segments):
        p0 = points[max(0, index - 1)]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[min(len(points) - 1, index + 2)]
        for step in range(steps):
            t = step / float(steps)
            t2, t3 = t * t, t * t * t
            value = (
                p0 * (-0.5 * t3 + t2 - 0.5 * t)
                + p1 * (1.5 * t3 - 2.5 * t2 + 1.0)
                + p2 * (-1.5 * t3 + 2.0 * t2 + 0.5 * t)
                + p3 * (0.5 * t3 - 0.5 * t2)
            )
            # Avoid overshoot on tight hair sections before surface projection.
            value = Vector((
                max(min(q.x for q in (p0, p1, p2, p3)), min(max(q.x for q in (p0, p1, p2, p3)), value.x)),
                max(min(q.y for q in (p0, p1, p2, p3)), min(max(q.y for q in (p0, p1, p2, p3)), value.y)),
                max(min(q.z for q in (p0, p1, p2, p3)), min(max(q.z for q in (p0, p1, p2, p3)), value.z)),
            ))
            result.append(value)
    result.append(points[-1].copy())
    return result[:max_points]


def _guided_ridge_segment_distance(a, b, c, d):
    """Closest distance between two finite 3D segments."""
    u, v, w = b - a, d - c, a - c
    aa, bb, cc = u.dot(u), u.dot(v), v.dot(v)
    dd, ee = u.dot(w), v.dot(w)
    if aa <= 1.0e-24 and cc <= 1.0e-24:
        return (a - c).length
    if aa <= 1.0e-24:
        t = max(0.0, min(1.0, ee / cc if cc else 0.0))
        return (a - (c + v * t)).length
    if cc <= 1.0e-24:
        s = max(0.0, min(1.0, -dd / aa if aa else 0.0))
        return ((a + u * s) - c).length
    denom = aa * cc - bb * bb
    if abs(denom) <= 1.0e-12 * max(aa * cc, 1.0e-30):
        def point_segment(point, first, second):
            edge = second - first
            length_sq = edge.dot(edge)
            if length_sq <= 1.0e-24:
                return (point - first).length
            t = max(0.0, min(1.0, (point - first).dot(edge) / length_sq))
            return (point - (first + edge * t)).length
        return min(point_segment(a, c, d), point_segment(b, c, d),
                   point_segment(c, a, b), point_segment(d, a, b))
    s = (bb * ee - cc * dd) / denom
    t = (aa * ee - bb * dd) / denom
    if s < 0.0:
        s = 0.0
        t = max(0.0, min(1.0, ee / cc))
    elif s > 1.0:
        s = 1.0
        t = max(0.0, min(1.0, (bb + ee) / cc))
    elif t < 0.0:
        t = 0.0
        s = max(0.0, min(1.0, -dd / aa))
    elif t > 1.0:
        t = 1.0
        s = max(0.0, min(1.0, (bb - dd) / aa))
    return ((a + u * s) - (c + v * t)).length


def _guided_ridge_validate_curve(points, curve=None, min_distance=1.0e-7):
    points = [Vector(point) for point in points]
    if len(points) < 2:
        return {"ok": False, "reason": "at_least_two_points_required"}
    for first, second in zip(points, points[1:]):
        if (second - first).length <= min_distance:
            return {"ok": False, "reason": "duplicate_or_too_close_control_points"}
    curve = [Vector(point) for point in (curve or points)]
    if len(curve) < 2:
        return {"ok": False, "reason": "curve_has_too_few_points"}
    total = sum((b - a).length for a, b in zip(curve, curve[1:]))
    direct = (curve[-1] - curve[0]).length
    if direct > min_distance and total / direct > 18.0:
        return {"ok": False, "reason": "extreme_meandering"}
    cross_distance = max(min_distance * 0.1, total * 0.0001)
    for index in range(len(curve) - 1):
        for other in range(index + 2, len(curve) - 1):
            if _guided_ridge_segment_distance(
                curve[index], curve[index + 1], curve[other], curve[other + 1]
            ) <= cross_distance:
                return {"ok": False, "reason": "self_crossing_or_near_crossing"}
    return {"ok": True, "total_length": float(total), "direct_length": float(direct)}


def _guided_ridge_hash_array(values):
    try:
        import numpy as np
        return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()
    except (ImportError, TypeError, ValueError):
        return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def _guided_ridge_face_set_attribute(obj):
    try:
        attr = obj.data.attributes.get(".sculpt_face_set")
        if attr is not None and attr.domain == "FACE" and attr.data_type == "INT":
            return attr
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    return None


def _guided_ridge_prototype_route_enabled():
    """Whether Step 1 may collect a route without the legacy Face Set graph.

    Curve Sculpt Step 1 only records 3D route points and fits a current-view
    2D preview; it does not write mesh coordinates, colors, or Paint Curve data.  It therefore
    does not need the old 50,000-face candidate guard or a Face Set partition.
    Keep this check local to the prototype entry points; the legacy snapshot
    and deformation engine retain their original safety limits.
    """
    return bool(GUIDED_RIDGE_UI_PROTOTYPE and GUIDED_RIDGE_CURVE_SCULPT_STEP1)


def _guided_ridge_prototype_surface_hit(context, coord):
    """Return a generic visible-mesh hit for the non-mutating Step 1 route."""
    hit = _raycast_visible_mesh_face(context, coord)
    if hit is None:
        return None
    obj, face_index, world_location, _local_location, screen = hit
    try:
        polygon = obj.data.polygons[int(face_index)]
        normal_matrix = obj.matrix_world.inverted_safe().transposed().to_3x3()
        normal = (normal_matrix @ polygon.normal).normalized()
        attr = _guided_ridge_face_set_attribute(obj)
        face_set_id = int(attr.data[int(face_index)].value) if attr is not None else -1
        return obj, int(face_index), face_set_id, Vector(world_location), Vector(normal), screen
    except (AttributeError, IndexError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _guided_ridge_route_context_matches(context, state):
    """Validate the lightweight Step 1 route's launch context.

    The prototype intentionally skips the legacy Face Set snapshot, but its
    clicked 3D points still belong to one object and one View3D.  Keeping
    these identity checks here prevents a later click from silently sampling
    another active object after a mode, area, or datablock change.
    """
    if not ((state or {}).get("prototype_route_only") or (state or {}).get("phase") in {"curve_preview", "curve_sculpt"}):
        return True
    # Lightweight unit/fake contexts may represent only the preview arrays;
    # real sessions always carry obj from invoke and are checked below.
    if (state or {}).get("phase") in {"curve_preview", "curve_sculpt"} and (state or {}).get("obj") is None:
        return True
    try:
        obj = state.get("obj")
        mesh = obj.data
        if obj is None or getattr(obj, "type", "MESH") != "MESH":
            return False
        if context.active_object is not obj or str(context.mode) != "SCULPT":
            return False
        if int(obj.as_pointer()) != int(state.get("object_pointer", 0)):
            return False
        if int(mesh.as_pointer()) != int(state.get("mesh_pointer", 0)):
            return False
        area = context.area
        region = context.region
        window = context.window
        space = context.space_data
        if getattr(area, "type", None) != "VIEW_3D" or getattr(region, "type", None) != "WINDOW":
            return False
        if int(area.as_pointer()) != int(state.get("area_key", 0)):
            return False
        if int(region.as_pointer()) != int(state.get("region_key", 0)):
            return False
        if int(space.as_pointer()) != int(state.get("space_key", 0)):
            return False
        if window is None or int(window.as_pointer()) != int(state.get("window_key", 0)):
            return False
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _guided_ridge_prototype_hit_matches_state(hit, state):
    """Return True only for a hit on the route's original mesh datablock."""
    if not (state or {}).get("prototype_route_only"):
        return True
    if hit is None:
        # A miss is a normal no-op in route editing; an actual hit on another
        # object is the unsafe case handled by this helper.
        return True
    try:
        obj = hit[0] if hit is not None else None
        if obj is None or obj is not state.get("obj"):
            return False
        if getattr(obj, "type", "MESH") != "MESH":
            return False
        return int(obj.as_pointer()) == int(state.get("object_pointer", 0)) and int(
            obj.data.as_pointer()
        ) == int(state.get("mesh_pointer", 0))
    except (AttributeError, IndexError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _guided_ridge_route_monitor_start(context, state):
    """Start only the cheap Step 1 identity monitor timer."""
    if state is None or not state.get("prototype_route_only") or state.get("timer") is not None:
        return bool(state is not None and state.get("timer") is not None)
    try:
        state["timer"] = state["window_manager"].event_timer_add(0.10, window=context.window)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        state["timer"] = None
        return False
    return True


def _guided_ridge_safety_reason(obj):
    """Return a direct-coordinate-write rejection reason, if any."""
    if obj is None:
        return "active mesh is unavailable"
    try:
        mesh = obj.data
        if int(getattr(mesh, "users", 1)) > 1:
            return "shared mesh data is not a safe direct-write target"
        if getattr(obj, "library", None) is not None or getattr(mesh, "library", None) is not None:
            return "linked library mesh is read-only"
        if getattr(obj, "override_library", None) is not None or getattr(mesh, "override_library", None) is not None:
            return "library override mesh is not a safe direct-write target"
        if bool(getattr(obj, "is_evaluated", False)) or bool(getattr(mesh, "is_evaluated", False)):
            return "evaluated mesh is not a safe direct-write target"
        if getattr(mesh, "shape_keys", None) is not None:
            return "shape-key coordinates are not a safe direct-write target"
        if bool(getattr(obj, "use_dynamic_topology_sculpting", False)):
            return "Dyntopo is active; stable source vertex mapping is unavailable"
        for modifier in getattr(obj, "modifiers", ()):
            if str(getattr(modifier, "type", "")) == "MULTIRES":
                return "Multires modifier is active; stable source vertex mapping is unavailable"
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return "mesh safety state could not be read"
    return None


def _guided_ridge_matrix_signature(matrix):
    try:
        return tuple(float(value) for row in matrix for value in row)
    except (AttributeError, TypeError, ValueError):
        return None


def _guided_ridge_context_signature(context):
    try:
        area = context.area
        region = context.region
        window = context.window
        space = context.space_data
        region_3d = space.region_3d
        active_object = context.active_object
        ui_scale = float(
            getattr(
                getattr(getattr(context, "preferences", None), "system", None),
                "ui_scale",
                1.0,
            )
            or 1.0
        )
        return {
            "area_pointer": int(area.as_pointer()),
            "region_pointer": int(region.as_pointer()),
            "window_pointer": int(window.as_pointer()) if window is not None else 0,
            "space_pointer": int(space.as_pointer()),
            "scene_pointer": int(context.scene.as_pointer()),
            "active_object_pointer": int(active_object.as_pointer()),
            "mesh_pointer": int(active_object.data.as_pointer()),
            "mode": str(context.mode),
            "space_type": str(space.type),
            "region_size": (int(region.width), int(region.height)),
            "ui_scale": ui_scale,
            "view_matrix": _guided_ridge_matrix_signature(region_3d.view_matrix),
            "window_matrix": _guided_ridge_matrix_signature(
                getattr(region_3d, "window_matrix", None)
            ),
            "perspective_matrix": _guided_ridge_matrix_signature(
                getattr(region_3d, "perspective_matrix", None)
            ),
            "view_location": tuple(float(value) for value in region_3d.view_location),
            "view_distance": float(region_3d.view_distance),
        }
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


# Nonzero Curve Shape output is deliberately bounded.  Shape 0 is the raw
# click polyline and may retain all knots; generated Bezier previews use this
# cap for both drawing and validation so an unusually long click route cannot
# turn every idle redraw into an unbounded intersection scan.
GUIDED_RIDGE_CURVE_NONZERO_MAX_SAMPLES = 512


def _guided_ridge_curve_route_digest(state):
    """Return a stable route revision/digest without using screen coordinates."""
    if state is None:
        return (0, "")
    route = state.get("curve_route") or state.get("controls") or ()
    revision = int(state.get("curve_route_revision", 0) or 0)
    # Recompute the digest even when the UI revision is unchanged.  This is a
    # small O(route) identity check and protects isolated callers/tests that
    # replace a point in-place without going through the modal click path.
    values = []
    try:
        for point in route:
            values.extend((float(point.x), float(point.y), float(point.z)))
    except (AttributeError, IndexError, TypeError, ValueError):
        values = [repr(point) for point in route]
    payload = repr(tuple(values)).encode("utf-8", "replace")
    digest = hashlib.sha1(payload).hexdigest()
    result = (revision, digest)
    state["curve_route_digest_cache"] = (revision, len(route), result)
    return result


def _guided_ridge_curve_preview_signature(context, state):
    """Build a view/route/Shape signature for the screen-only preview cache."""
    context_signature = _guided_ridge_context_signature(context) if context is not None else None
    if context_signature is None:
        return None
    route_key = _guided_ridge_curve_route_digest(state)
    try:
        shape = float(state.get("curve_smoothing", _runtime.guided_ridge_curve_last_smoothing) or 0.0)
    except (AttributeError, TypeError, ValueError):
        shape = 0.0
    shape = min(max(shape, -100.0), 100.0)
    # repr keeps this tolerant of the lightweight isolated test contexts while
    # retaining all projection matrices, region size, object and mesh identity
    # fields supplied by Blender's context signature.
    context_key = repr(tuple(sorted(context_signature.items())))
    base = (route_key, context_key, int(GUIDED_RIDGE_CURVE_NONZERO_MAX_SAMPLES))
    return {
        "base": base,
        "full": base + (shape,),
        "shape": shape,
    }


def _guided_ridge_curve_bounded_parameters(total, sample_count):
    """Return a uniform, bounded parameterization for generated curves."""
    count = min(
        max(int(sample_count), 2),
        int(GUIDED_RIDGE_CURVE_NONZERO_MAX_SAMPLES),
    )
    return [
        float(total) * index / float(max(count - 1, 1))
        for index in range(count)
    ]


def _guided_ridge_context_matches(context, state):
    snapshot = (state or {}).get("snapshot") or {}
    expected = snapshot.get("context_signature")
    current = _guided_ridge_context_signature(context)
    if expected is None or current is None:
        return False
    stable_keys = (
        "area_pointer",
        "region_pointer",
        "window_pointer",
        "space_pointer",
        "scene_pointer",
        "active_object_pointer",
        "mesh_pointer",
        "mode",
        "space_type",
    )
    return all(current.get(key) == expected.get(key) for key in stable_keys)


def _guided_ridge_prepare_snapshot_sync(obj, seed_face, seed_face_set):
    """Synchronous reference implementation used by repeat/diagnostics.

    Guided Ridge entry uses the cooperative implementation below.  Keeping
    this exact implementation available makes it possible to compare the
    incremental snapshot byte-for-byte before changing the production path.
    """
    import numpy as np
    safety_reason = _guided_ridge_safety_reason(obj)
    if safety_reason is not None:
        return None, safety_reason
    attr = _guided_ridge_face_set_attribute(obj)
    if attr is None:
        return None, "active mesh has no .sculpt_face_set attribute"
    mesh = obj.data
    try:
        ids = np.empty(len(attr.data), dtype=np.int32)
        attr.data.foreach_get("value", ids)
        hidden_faces = np.zeros(len(mesh.polygons), dtype=bool)
        mesh.polygons.foreach_get("hide", hidden_faces)
        hidden_vertices = np.zeros(len(mesh.vertices), dtype=bool)
        mesh.vertices.foreach_get("hide", hidden_vertices)
        sculpt_mask = np.zeros(len(mesh.vertices), dtype=np.float64)
        mask_attr = mesh.attributes.get(".sculpt_mask")
        if mask_attr is not None and mask_attr.domain == "POINT" and len(mask_attr.data) == len(sculpt_mask):
            mask_attr.data.foreach_get("value", sculpt_mask)
        sculpt_mask = np.clip(sculpt_mask, 0.0, 1.0)
        if seed_face < 0 or seed_face >= len(ids) or int(ids[seed_face]) != int(seed_face_set):
            return None, "ray hit is outside the active Face Set"
        candidates = np.flatnonzero((ids == int(seed_face_set)) & ~hidden_faces)
        if len(candidates) > GUIDED_RIDGE_MAX_CANDIDATE_FACES:
            return None, "Face Set candidate exceeds 50,000 faces"
        edge_faces = {}
        face_vertices = {}
        for face_index in candidates:
            vertices = tuple(int(v) for v in mesh.polygons[int(face_index)].vertices)
            if len(vertices) < 3:
                continue
            face_vertices[int(face_index)] = vertices
            for offset, first in enumerate(vertices):
                second = vertices[(offset + 1) % len(vertices)]
                edge = (min(first, second), max(first, second))
                edge_faces.setdefault(edge, []).append(int(face_index))
        if int(seed_face) not in face_vertices:
            return None, "ray hit face is hidden or degenerate"
        component = {int(seed_face)}
        pending = [int(seed_face)]
        while pending:
            current = pending.pop()
            for offset, first in enumerate(face_vertices[current]):
                second = face_vertices[current][(offset + 1) % len(face_vertices[current])]
                for other in edge_faces.get((min(first, second), max(first, second)), ()):
                    if other not in component:
                        component.add(other)
                        pending.append(other)
                        if len(component) > GUIDED_RIDGE_MAX_CANDIDATE_FACES:
                            return None, "Face Set component exceeds 50,000 faces"
        face_indices = sorted(component)
        vertex_indices = sorted({v for face in face_indices for v in face_vertices[face]})
        component_face_set = set(face_indices)
        component_vertex_set = set(vertex_indices)
        # A vertex is not writable when any face outside the captured visible
        # Face Set component owns it.  This covers vertex-only contacts and
        # non-manifold third faces, including hidden faces that are excluded
        # from the candidate graph above.
        externally_shared_vertices = set()
        for outside_face_index, polygon in enumerate(mesh.polygons):
            if int(outside_face_index) in component_face_set:
                continue
            for vertex in polygon.vertices:
                vertex = int(vertex)
                if vertex in component_vertex_set:
                    externally_shared_vertices.add(vertex)
        protected_vertices = sorted(externally_shared_vertices)
        local_index = {global_index: index for index, global_index in enumerate(vertex_indices)}
        coords_local = np.asarray([mesh.vertices[index].co[:] for index in vertex_indices], dtype=np.float64)
        world_matrix = obj.matrix_world.copy()
        coords_world = np.asarray([world_matrix @ Vector(co) for co in coords_local], dtype=np.float64)
        triangles = []
        triangle_face_indices = []
        for face_index in face_indices:
            polygon = face_vertices[face_index]
            for offset in range(1, len(polygon) - 1):
                triangles.append((local_index[polygon[0]], local_index[polygon[offset]], local_index[polygon[offset + 1]]))
                triangle_face_indices.append(int(face_index))
        if not triangles:
            return None, "Face Set component has no triangles"
        edge_counts = {}
        for polygon in (face_vertices[index] for index in face_indices):
            for offset, first in enumerate(polygon):
                second = polygon[(offset + 1) % len(polygon)]
                edge = (min(first, second), max(first, second))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        boundary_vertices = sorted(
            {v for edge, count in edge_counts.items() if count == 1 for v in edge}
            | set(protected_vertices)
        )
        boundary_edges = [
            (local_index[first], local_index[second])
            for (first, second), count in edge_counts.items()
            if count == 1 and first in local_index and second in local_index
        ]
        adjacency = [[] for _ in vertex_indices]
        for first, second, third in triangles:
            adjacency[first].extend((second, third))
            adjacency[second].extend((first, third))
            adjacency[third].extend((first, second))
        adjacency = [tuple(sorted(set(neighbors))) for neighbors in adjacency]
        try:
            bvh = BVHTree.FromPolygons(coords_world.tolist(), triangles, all_triangles=True)
        except TypeError:
            bvh = BVHTree.FromPolygons(coords_world.tolist(), triangles)
        edge_lengths = [
            float(np.linalg.norm(coords_world[local_index[first]] - coords_world[local_index[second]]))
            for first, second in edge_counts
            if first in local_index and second in local_index
        ]
        normal_matrix = world_matrix.inverted_safe().transposed().to_3x3()
        normals_world = np.asarray([
            tuple((normal_matrix @ mesh.vertices[index].normal).normalized())
            for index in vertex_indices
        ], dtype=np.float64)
        signature = {
            "object_pointer": int(obj.as_pointer()),
            "mesh_pointer": int(mesh.as_pointer()),
            "counts": (len(mesh.vertices), len(mesh.edges), len(mesh.polygons), len(mesh.loops)),
            "face_indices": tuple(face_indices),
            "face_vertices": tuple(tuple(face_vertices[index]) for index in face_indices),
            "face_set_hash": _guided_ridge_hash_array(ids[face_indices]),
            "hidden_faces_hash": _guided_ridge_hash_array(hidden_faces[face_indices]),
            "hidden_vertices_hash": _guided_ridge_hash_array(hidden_vertices[vertex_indices]),
            "sculpt_mask_hash": _guided_ridge_hash_array(sculpt_mask[vertex_indices]),
            "protected_vertices_hash": _guided_ridge_hash_array(
                np.asarray(protected_vertices, dtype=np.int64)
            ),
            "coordinate_hash": _guided_ridge_hash_array(coords_local),
            "matrix_world": _guided_ridge_matrix_signature(world_matrix),
            "component_topology_signature": _guided_ridge_component_topology_signature(
                face_indices,
                [face_vertices[index] for index in face_indices],
            ),
        }
        snapshot = {
            "object_pointer": signature["object_pointer"],
            "mesh_pointer": signature["mesh_pointer"],
            "counts": signature["counts"],
            "face_set_id": int(seed_face_set),
            "face_indices": face_indices,
            "face_vertices": [face_vertices[index] for index in face_indices],
            "vertex_indices": vertex_indices,
            "triangles": triangles,
            "triangle_face_indices": triangle_face_indices,
            "boundary_vertices": boundary_vertices,
            "protected_vertices": protected_vertices,
            "boundary_edges": boundary_edges,
            "adjacency": adjacency,
            "coords_local": coords_local,
            "coords_world": coords_world,
            "normals_world": normals_world,
            "hidden_vertices": np.array(hidden_vertices[vertex_indices], dtype=bool, copy=True),
            "sculpt_mask": np.array(sculpt_mask[vertex_indices], dtype=np.float64, copy=True),
            "matrix_world": world_matrix.copy(),
            "bvh": bvh,
            "average_edge": float(sum(edge_lengths) / max(len(edge_lengths), 1)),
            "signature": signature,
        }
        return snapshot, None
    except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None, "could not prepare the Face Set component"


def _guided_ridge_prepare_progress(stage, stage_index, done=None, total=None, indeterminate=False):
    """Return a small, UI-safe progress token for cooperative preparation."""
    return {
        "stage": str(stage),
        "stage_index": int(stage_index),
        "stage_count": 13,
        "done": int(done or 0),
        "total": int(total or 0),
        "indeterminate": bool(indeterminate or not total),
    }


def _guided_ridge_prepare_snapshot_steps(obj, seed_face, seed_face_set):
    """Capture one component while yielding bounded UI-event-loop slices.

    The operations and ordering intentionally mirror
    ``_guided_ridge_prepare_snapshot_sync``.  This generator only inserts
    cooperative yields; it does not alter component selection, protection,
    BVH input, signatures, or snapshot values.
    """
    import numpy as np

    safety_reason = _guided_ridge_safety_reason(obj)
    if safety_reason is not None:
        return None, safety_reason
    attr = _guided_ridge_face_set_attribute(obj)
    if attr is None:
        return None, "active mesh has no .sculpt_face_set attribute"
    mesh = obj.data
    try:
        yield _guided_ridge_prepare_progress("reading Face Set data", 1, indeterminate=True)
        ids = np.empty(len(attr.data), dtype=np.int32)
        attr.data.foreach_get("value", ids)
        yield _guided_ridge_prepare_progress("reading hidden-face data", 1, indeterminate=True)
        hidden_faces = np.zeros(len(mesh.polygons), dtype=bool)
        mesh.polygons.foreach_get("hide", hidden_faces)
        yield _guided_ridge_prepare_progress("reading hidden-vertex data", 1, indeterminate=True)
        hidden_vertices = np.zeros(len(mesh.vertices), dtype=bool)
        mesh.vertices.foreach_get("hide", hidden_vertices)
        sculpt_mask = np.zeros(len(mesh.vertices), dtype=np.float64)
        mask_attr = mesh.attributes.get(".sculpt_mask")
        if mask_attr is not None and mask_attr.domain == "POINT" and len(mask_attr.data) == len(sculpt_mask):
            yield _guided_ridge_prepare_progress("reading sculpt mask", 1, indeterminate=True)
            mask_attr.data.foreach_get("value", sculpt_mask)
        sculpt_mask = np.clip(sculpt_mask, 0.0, 1.0)
        if seed_face < 0 or seed_face >= len(ids) or int(ids[seed_face]) != int(seed_face_set):
            return None, "ray hit is outside the active Face Set"
        candidates = np.flatnonzero((ids == int(seed_face_set)) & ~hidden_faces)
        if len(candidates) > GUIDED_RIDGE_MAX_CANDIDATE_FACES:
            return None, "Face Set candidate exceeds 50,000 faces"

        edge_faces = {}
        face_vertices = {}
        candidate_total = len(candidates)
        yield _guided_ridge_prepare_progress("building Face Set edge graph", 2, 0, candidate_total)
        for position, face_index in enumerate(candidates):
            vertices = tuple(int(v) for v in mesh.polygons[int(face_index)].vertices)
            if len(vertices) >= 3:
                face_vertices[int(face_index)] = vertices
                for offset, first in enumerate(vertices):
                    second = vertices[(offset + 1) % len(vertices)]
                    edge = (min(first, second), max(first, second))
                    edge_faces.setdefault(edge, []).append(int(face_index))
            if (position & 255) == 255 or position + 1 == candidate_total:
                yield _guided_ridge_prepare_progress(
                    "building Face Set edge graph", 2, position + 1, candidate_total
                )
        if int(seed_face) not in face_vertices:
            return None, "ray hit face is hidden or degenerate"

        component = {int(seed_face)}
        pending = [int(seed_face)]
        component_total = max(len(candidates), 1)
        component_pop_count = 0
        yield _guided_ridge_prepare_progress("finding connected component", 3, 1, component_total)
        while pending:
            current = pending.pop()
            component_pop_count += 1
            for offset, first in enumerate(face_vertices[current]):
                second = face_vertices[current][(offset + 1) % len(face_vertices[current])]
                for other in edge_faces.get((min(first, second), max(first, second)), ()):
                    if other not in component:
                        component.add(other)
                        pending.append(other)
                        if len(component) > GUIDED_RIDGE_MAX_CANDIDATE_FACES:
                            return None, "Face Set component exceeds 50,000 faces"
            if component_pop_count % GUIDED_RIDGE_PREPARE_WORK_CHUNK == 0 or not pending:
                yield _guided_ridge_prepare_progress(
                    "finding connected component", 3, component_pop_count, component_total
                )
        face_indices = sorted(component)
        vertex_indices = sorted({v for face in face_indices for v in face_vertices[face]})
        component_face_set = set(face_indices)
        component_vertex_set = set(vertex_indices)

        # Preserve the original safety rule: every mesh polygon is checked for
        # a vertex shared with the captured component, including hidden faces.
        externally_shared_vertices = set()
        polygon_total = len(mesh.polygons)
        yield _guided_ridge_prepare_progress(
            "checking externally shared vertices", 4, 0, polygon_total
        )
        for outside_face_index, polygon in enumerate(mesh.polygons):
            if int(outside_face_index) not in component_face_set:
                for vertex in polygon.vertices:
                    vertex = int(vertex)
                    if vertex in component_vertex_set:
                        externally_shared_vertices.add(vertex)
            if outside_face_index % 4096 == 4095 or outside_face_index + 1 == polygon_total:
                yield _guided_ridge_prepare_progress(
                    "checking externally shared vertices", 4, outside_face_index + 1, polygon_total
                )
        protected_vertices = sorted(externally_shared_vertices)
        local_index = {global_index: index for index, global_index in enumerate(vertex_indices)}

        yield _guided_ridge_prepare_progress("capturing component coordinates", 5, 0, len(vertex_indices))
        coords_local_values = []
        for position, index in enumerate(vertex_indices):
            coords_local_values.append(mesh.vertices[index].co[:])
            if (position & 255) == 255 or position + 1 == len(vertex_indices):
                yield _guided_ridge_prepare_progress(
                    "capturing component coordinates", 5, position + 1, len(vertex_indices)
                )
        coords_local = np.asarray(coords_local_values, dtype=np.float64)
        world_matrix = obj.matrix_world.copy()
        coords_world_values = []
        for position, co in enumerate(coords_local):
            coords_world_values.append(world_matrix @ Vector(co))
            if (position & 255) == 255 or position + 1 == len(coords_local):
                yield _guided_ridge_prepare_progress(
                    "transforming component coordinates", 5, position + 1, len(coords_local)
                )
        coords_world = np.asarray(coords_world_values, dtype=np.float64)

        triangles = []
        triangle_face_indices = []
        yield _guided_ridge_prepare_progress("triangulating component", 5, 0, len(face_indices))
        for position, face_index in enumerate(face_indices):
            polygon = face_vertices[face_index]
            for offset in range(1, len(polygon) - 1):
                triangles.append((local_index[polygon[0]], local_index[polygon[offset]], local_index[polygon[offset + 1]]))
                triangle_face_indices.append(int(face_index))
            if (position & 255) == 255 or position + 1 == len(face_indices):
                yield _guided_ridge_prepare_progress(
                    "triangulating component", 5, position + 1, len(face_indices)
                )
        if not triangles:
            return None, "Face Set component has no triangles"

        edge_counts = {}
        yield _guided_ridge_prepare_progress("building component boundaries", 6, 0, len(face_indices))
        for position, polygon in enumerate(face_vertices[index] for index in face_indices):
            for offset, first in enumerate(polygon):
                second = polygon[(offset + 1) % len(polygon)]
                edge = (min(first, second), max(first, second))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
            if (position & 255) == 255 or position + 1 == len(face_indices):
                yield _guided_ridge_prepare_progress(
                    "building component boundaries", 6, position + 1, len(face_indices)
                )
        boundary_vertex_set = set(protected_vertices)
        for position, (edge, count) in enumerate(edge_counts.items()):
            if count == 1:
                boundary_vertex_set.update(edge)
            if (position & 255) == 255 or position + 1 == len(edge_counts):
                yield _guided_ridge_prepare_progress(
                    "building component boundaries", 6, position + 1, len(edge_counts)
                )
        boundary_vertices = sorted(boundary_vertex_set)
        boundary_edges = []
        for position, ((first, second), count) in enumerate(edge_counts.items()):
            if count == 1 and first in local_index and second in local_index:
                boundary_edges.append((local_index[first], local_index[second]))
            if (position & 255) == 255 or position + 1 == len(edge_counts):
                yield _guided_ridge_prepare_progress(
                    "building component boundaries", 6, position + 1, len(edge_counts)
                )

        adjacency = [[] for _ in vertex_indices]
        yield _guided_ridge_prepare_progress("building surface adjacency", 7, 0, len(triangles))
        for position, (first, second, third) in enumerate(triangles):
            adjacency[first].extend((second, third))
            adjacency[second].extend((first, third))
            adjacency[third].extend((first, second))
            if (position & 255) == 255 or position + 1 == len(triangles):
                yield _guided_ridge_prepare_progress(
                    "building surface adjacency", 7, position + 1, len(triangles)
                )
        adjacency_result = []
        for position, neighbors in enumerate(adjacency):
            adjacency_result.append(tuple(sorted(set(neighbors))))
            if (position & 255) == 255 or position + 1 == len(adjacency):
                yield _guided_ridge_prepare_progress(
                    "deduplicating surface adjacency", 7, position + 1, len(adjacency)
                )
        adjacency = adjacency_result

        # BVH construction is one Blender C call and cannot be interrupted.
        yield _guided_ridge_prepare_progress("building surface BVH", 8, indeterminate=True)
        try:
            bvh = BVHTree.FromPolygons(coords_world.tolist(), triangles, all_triangles=True)
        except TypeError:
            bvh = BVHTree.FromPolygons(coords_world.tolist(), triangles)

        edge_lengths = []
        yield _guided_ridge_prepare_progress("measuring component edges", 9, 0, len(edge_counts))
        for position, (first, second) in enumerate(edge_counts):
            if first in local_index and second in local_index:
                edge_lengths.append(float(np.linalg.norm(coords_world[local_index[first]] - coords_world[local_index[second]])))
            if (position & 255) == 255 or position + 1 == len(edge_counts):
                yield _guided_ridge_prepare_progress(
                    "measuring component edges", 9, position + 1, len(edge_counts)
                )

        normal_matrix = world_matrix.inverted_safe().transposed().to_3x3()
        normals_world_values = []
        yield _guided_ridge_prepare_progress("capturing vertex normals", 10, 0, len(vertex_indices))
        for position, index in enumerate(vertex_indices):
            normals_world_values.append(tuple((normal_matrix @ mesh.vertices[index].normal).normalized()))
            if (position & 255) == 255 or position + 1 == len(vertex_indices):
                yield _guided_ridge_prepare_progress(
                    "capturing vertex normals", 10, position + 1, len(vertex_indices)
                )
        normals_world = np.asarray(normals_world_values, dtype=np.float64)

        yield _guided_ridge_prepare_progress("building component signature", 11, indeterminate=True)
        face_indices_tuple = tuple(face_indices)
        face_vertices_tuple = tuple(tuple(face_vertices[index]) for index in face_indices)
        face_set_hash = _guided_ridge_hash_array(ids[face_indices])
        yield _guided_ridge_prepare_progress("building component signature", 11, 1, 7)
        hidden_faces_hash = _guided_ridge_hash_array(hidden_faces[face_indices])
        yield _guided_ridge_prepare_progress("building component signature", 11, 2, 7)
        hidden_vertices_hash = _guided_ridge_hash_array(hidden_vertices[vertex_indices])
        yield _guided_ridge_prepare_progress("building component signature", 11, 3, 7)
        sculpt_mask_hash = _guided_ridge_hash_array(sculpt_mask[vertex_indices])
        yield _guided_ridge_prepare_progress("building component signature", 11, 4, 7)
        protected_vertices_hash = _guided_ridge_hash_array(np.asarray(protected_vertices, dtype=np.int64))
        yield _guided_ridge_prepare_progress("building component signature", 11, 5, 7)
        coordinate_hash = _guided_ridge_hash_array(coords_local)
        yield _guided_ridge_prepare_progress("building component signature", 11, 6, 7)
        topology_job = _guided_ridge_component_topology_signature_steps(
            face_indices, [face_vertices[index] for index in face_indices]
        )
        while True:
            try:
                progress = next(topology_job)
            except StopIteration as complete:
                topology_signature = complete.value
                break
            yield progress
        yield _guided_ridge_prepare_progress("building component signature", 11, 7, 7)
        signature = {
            "object_pointer": int(obj.as_pointer()),
            "mesh_pointer": int(mesh.as_pointer()),
            "counts": (len(mesh.vertices), len(mesh.edges), len(mesh.polygons), len(mesh.loops)),
            "face_indices": face_indices_tuple,
            "face_vertices": face_vertices_tuple,
            "face_set_hash": face_set_hash,
            "hidden_faces_hash": hidden_faces_hash,
            "hidden_vertices_hash": hidden_vertices_hash,
            "sculpt_mask_hash": sculpt_mask_hash,
            "protected_vertices_hash": protected_vertices_hash,
            "coordinate_hash": coordinate_hash,
            "matrix_world": _guided_ridge_matrix_signature(world_matrix),
            "component_topology_signature": topology_signature,
        }
        yield _guided_ridge_prepare_progress("finalizing snapshot", 12, indeterminate=True)
        snapshot = {
            "object_pointer": signature["object_pointer"],
            "mesh_pointer": signature["mesh_pointer"],
            "counts": signature["counts"],
            "face_set_id": int(seed_face_set),
            "face_indices": face_indices,
            "face_vertices": [face_vertices[index] for index in face_indices],
            "vertex_indices": vertex_indices,
            "triangles": triangles,
            "triangle_face_indices": triangle_face_indices,
            "boundary_vertices": boundary_vertices,
            "protected_vertices": protected_vertices,
            "boundary_edges": boundary_edges,
            "adjacency": adjacency,
            "coords_local": coords_local,
            "coords_world": coords_world,
            "normals_world": normals_world,
            "hidden_vertices": np.array(hidden_vertices[vertex_indices], dtype=bool, copy=True),
            "sculpt_mask": np.array(sculpt_mask[vertex_indices], dtype=np.float64, copy=True),
            "matrix_world": world_matrix.copy(),
            "bvh": bvh,
            "average_edge": float(sum(edge_lengths) / max(len(edge_lengths), 1)),
            "signature": signature,
        }
        return snapshot, None
    except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None, "could not prepare the Face Set component"


def _guided_ridge_prepare_snapshot(obj, seed_face, seed_face_set):
    """Synchronous compatibility wrapper that consumes cooperative slices."""
    job = _guided_ridge_prepare_snapshot_steps(obj, seed_face, seed_face_set)
    try:
        while True:
            next(job)
    except StopIteration as complete:
        return complete.value


def _guided_ridge_ray(context, coord):
    origin = view3d_utils.region_2d_to_origin_3d(
        context.region, context.space_data.region_3d, coord
    )
    direction = view3d_utils.region_2d_to_vector_3d(
        context.region, context.space_data.region_3d, coord
    )
    if direction.length_squared <= 1.0e-24:
        return None
    direction.normalize()
    return origin, direction


def _guided_ridge_snapshot_hit(context, snapshot, coord):
    ray = _guided_ridge_ray(context, coord)
    return _guided_ridge_snapshot_hit_ray(snapshot, ray)


def _guided_ridge_snapshot_hit_ray(snapshot, ray):
    """Ray-cast a saved world-space ray without consulting the current view."""
    if ray is None:
        return None
    try:
        location, normal, triangle_index, distance = snapshot["bvh"].ray_cast(*ray)
        if location is None or triangle_index is None or int(triangle_index) < 0:
            return None
        return Vector(location), Vector(normal).normalized(), int(triangle_index), float(distance)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _guided_ridge_project_curve(snapshot, controls):
    raw = _guided_ridge_catmull_rom(controls)
    projected = []
    max_distance = 0.0
    for point in raw:
        nearest = snapshot["bvh"].find_nearest(point)
        if not nearest or nearest[0] is None:
            return None, "guide projection failed"
        location, _normal, _triangle, distance = nearest
        max_distance = max(max_distance, float(distance or 0.0))
        projected.append(Vector(location))
    # Clicks were already on the component; this bound catches a curve that
    # leaves the component between controls without changing the source mesh.
    if max_distance > max(1.0e-7, snapshot.get("average_edge", 1.0e-4) * 4.0):
        return None, "guide leaves the Face Set component"
    result = _guided_ridge_validate_curve(controls, projected)
    if not result.get("ok"):
        return None, result.get("reason", "invalid guide")
    return projected, None


def _guided_ridge_barycentric(point, first, second, third):
    """Return finite triangle barycentrics for a surface anchor."""
    import numpy as np
    a = np.asarray(tuple(first), dtype=np.float64)
    b = np.asarray(tuple(second), dtype=np.float64)
    c = np.asarray(tuple(third), dtype=np.float64)
    p = np.asarray(tuple(point), dtype=np.float64)
    edge0 = b - a
    edge1 = c - a
    rel = p - a
    d00 = float(np.dot(edge0, edge0))
    d01 = float(np.dot(edge0, edge1))
    d11 = float(np.dot(edge1, edge1))
    d20 = float(np.dot(rel, edge0))
    d21 = float(np.dot(rel, edge1))
    denominator = d00 * d11 - d01 * d01
    if not np.isfinite(denominator) or abs(denominator) <= 1.0e-24:
        return None
    second_weight = (d11 * d20 - d01 * d21) / denominator
    third_weight = (d00 * d21 - d01 * d20) / denominator
    first_weight = 1.0 - second_weight - third_weight
    weights = (first_weight, second_weight, third_weight)
    if not np.all(np.isfinite(weights)) or min(weights) < -1.0e-5 or max(weights) > 1.00001:
        return None
    return tuple(float(max(0.0, min(1.0, weight))) for weight in weights)


def _guided_ridge_surface_anchors(snapshot, controls):
    """Encode guide points as bounded surface anchors without copying the mesh."""
    anchors = []
    try:
        for control in controls:
            nearest = snapshot["bvh"].find_nearest(control)
            if not nearest or nearest[0] is None or nearest[2] is None:
                return None, "guide anchor projection failed"
            location, _normal, triangle_index, _distance = nearest
            triangle_index = int(triangle_index)
            if triangle_index < 0 or triangle_index >= len(snapshot["triangles"]):
                return None, "guide anchor triangle is invalid"
            triangle = snapshot["triangles"][triangle_index]
            vertices_world = [snapshot["coords_world"][int(index)] for index in triangle]
            barycentric = _guided_ridge_barycentric(location, *vertices_world)
            if barycentric is None:
                return None, "guide anchor is on a degenerate triangle"
            anchors.append({
                "face_index": int(snapshot["triangle_face_indices"][triangle_index]),
                "vertex_indices": tuple(int(snapshot["vertex_indices"][int(index)]) for index in triangle),
                "barycentric": barycentric,
            })
    except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None, "guide anchor projection failed"
    return anchors, None


def _guided_ridge_component_topology_signature_steps(face_indices, face_vertices):
    """Yield bounded work while hashing component loops and edge membership."""
    digest = hashlib.sha256()
    component_faces = []
    total = len(face_indices)
    for position, (face_index, vertices) in enumerate(zip(face_indices, face_vertices)):
        component_faces.append((int(face_index), tuple(int(vertex) for vertex in vertices)))
        if (position & 63) == 63 or position + 1 == total:
            yield _guided_ridge_prepare_progress(
                "building topology signature", 11, position + 1, total
            )
    for position, (face_index, vertices) in enumerate(component_faces):
        digest.update(b"F")
        digest.update(face_index.to_bytes(8, "little", signed=True))
        digest.update(len(vertices).to_bytes(8, "little", signed=False))
        for vertex in vertices:
            digest.update(vertex.to_bytes(8, "little", signed=True))
        if (position & 63) == 63 or position + 1 == total:
            yield _guided_ridge_prepare_progress(
                "hashing topology loops", 11, position + 1, total
            )
    edge_membership = {}
    for position, (face_index, vertices) in enumerate(component_faces):
        for offset, first in enumerate(vertices):
            second = vertices[(offset + 1) % len(vertices)]
            edge = (min(first, second), max(first, second))
            edge_membership.setdefault(edge, []).append(face_index)
        if (position & 63) == 63 or position + 1 == total:
            yield _guided_ridge_prepare_progress(
                "building topology edge membership", 11, position + 1, total
            )
    sorted_edges = sorted(edge_membership)
    for position, edge in enumerate(sorted_edges):
        vertices = tuple(int(vertex) for vertex in edge)
        digest.update(b"E")
        for vertex in vertices:
            digest.update(vertex.to_bytes(8, "little", signed=True))
        members = tuple(sorted(edge_membership[edge]))
        digest.update(len(members).to_bytes(8, "little", signed=False))
        for face_index in members:
            digest.update(int(face_index).to_bytes(8, "little", signed=True))
        if (position & 63) == 63 or position + 1 == len(sorted_edges):
            yield _guided_ridge_prepare_progress(
                "hashing topology edges", 11, position + 1, len(sorted_edges)
            )
    return digest.hexdigest()


def _guided_ridge_component_topology_signature(face_indices, face_vertices):
    """Synchronous compatibility wrapper for the cooperative hash."""
    job = _guided_ridge_component_topology_signature_steps(face_indices, face_vertices)
    try:
        while True:
            next(job)
    except StopIteration as complete:
        return complete.value


def _guided_ridge_build_last_guide(state, anchors=None):
    """Build a bounded Repeat Last payload before mesh coordinates are written."""
    import numpy as np
    snapshot = state.get("snapshot") if state else None
    obj = state.get("obj") if state else None
    controls = state.get("controls") if state else None
    if snapshot is None or obj is None or not controls:
        return None, "last guide state is unavailable"
    if anchors is None:
        anchors, reason = _guided_ridge_surface_anchors(snapshot, controls)
        if anchors is None:
            return None, reason or "could not save surface anchors"
    signature = snapshot.get("signature", {})
    payload = {
        "object_name": str(getattr(obj, "name", "")),
        "data_name": str(getattr(obj.data, "name", "")),
        "object_pointer": int(snapshot.get("object_pointer", 0)),
        "mesh_pointer": int(snapshot.get("mesh_pointer", 0)),
        "counts": tuple(snapshot.get("counts", ())),
        "face_set_id": int(snapshot.get("face_set_id", 0)),
        "seed_face": int(snapshot.get("face_indices", [0])[0]),
        "face_indices_hash": _guided_ridge_hash_array(np.asarray(snapshot.get("face_indices", ()), dtype=np.int64)),
        "face_vertices_hash": _guided_ridge_hash_array(
            np.asarray([vertex for face in snapshot.get("face_vertices", ()) for vertex in face], dtype=np.int64)
        ),
        "face_set_hash": signature.get("face_set_hash"),
        "hidden_faces_hash": signature.get("hidden_faces_hash"),
        "hidden_vertices_hash": signature.get("hidden_vertices_hash"),
        "sculpt_mask_hash": signature.get("sculpt_mask_hash"),
        "protected_vertices_hash": signature.get("protected_vertices_hash"),
        "matrix_world": signature.get("matrix_world"),
        "component_topology_signature": signature.get("component_topology_signature"),
        "anchors": anchors,
        "half_width": float(state.get("half_width", 0.0) or 0.0),
    }
    return payload, None


def _guided_ridge_store_last_guide(state, anchors=None):
    """Publish a prebuilt surface-anchor payload for Repeat Last."""
    payload, reason = _guided_ridge_build_last_guide(state, anchors)
    if payload is None:
        return False, reason or "could not save surface anchors"
    _runtime.guided_ridge_last_guide = payload
    return True, None


def _guided_ridge_prepare_repeat_snapshot(obj, last_guide):
    """Prepare current geometry and accept coordinate changes from prior ridge commits."""
    import numpy as np
    snapshot, reason = _guided_ridge_prepare_snapshot(
        obj,
        int(last_guide.get("seed_face", -1)),
        int(last_guide.get("face_set_id", 0)),
    )
    if snapshot is None:
        return None, reason or "could not prepare the saved guide component"
    signature = snapshot.get("signature", {})
    try:
        if tuple(snapshot.get("counts", ())) != tuple(last_guide.get("counts", ())):
            return None, "Guided Ridge Repeat cancelled: mesh topology changed"
        if signature.get("face_set_hash") != last_guide.get("face_set_hash"):
            return None, "Guided Ridge Repeat cancelled: Face Set changed"
        if signature.get("hidden_faces_hash") != last_guide.get("hidden_faces_hash"):
            return None, "Guided Ridge Repeat cancelled: hidden faces changed"
        if signature.get("hidden_vertices_hash") != last_guide.get("hidden_vertices_hash"):
            return None, "Guided Ridge Repeat cancelled: hidden vertices changed"
        if signature.get("sculpt_mask_hash") != last_guide.get("sculpt_mask_hash"):
            return None, "Guided Ridge Repeat cancelled: sculpt mask changed"
        if signature.get("protected_vertices_hash") != last_guide.get("protected_vertices_hash"):
            return None, "Guided Ridge Repeat cancelled: protected boundary changed"
        if signature.get("matrix_world") != last_guide.get("matrix_world"):
            return None, "Guided Ridge Repeat cancelled: object transform changed"
        if signature.get("component_topology_signature") != last_guide.get("component_topology_signature"):
            return None, "Guided Ridge Repeat cancelled: Face Set component topology/connectivity changed"
        if _guided_ridge_hash_array(np.asarray(snapshot.get("face_indices", ()), dtype=np.int64)) != last_guide.get("face_indices_hash"):
            return None, "Guided Ridge Repeat cancelled: Face Set component topology changed"
        if _guided_ridge_hash_array(
            np.asarray([vertex for face in snapshot.get("face_vertices", ()) for vertex in face], dtype=np.int64)
        ) != last_guide.get("face_vertices_hash"):
            return None, "Guided Ridge Repeat cancelled: component faces changed"
    except (MemoryError, TypeError, ValueError):
        return None, "Guided Ridge Repeat cancelled: component signature is invalid"
    return snapshot, None


def _guided_ridge_repeat_controls(snapshot, last_guide):
    """Reconstruct control points and normals from current surface anchors."""
    import numpy as np
    local_index = {int(global_index): index for index, global_index in enumerate(snapshot["vertex_indices"])}
    controls = []
    normals = []
    try:
        for anchor in last_guide.get("anchors", ()):
            globals_for_triangle = tuple(int(index) for index in anchor["vertex_indices"])
            local_triangle = tuple(local_index[index] for index in globals_for_triangle)
            barycentric = np.asarray(anchor["barycentric"], dtype=np.float64)
            if len(local_triangle) != 3 or len(barycentric) != 3 or not np.all(np.isfinite(barycentric)):
                return None, None, "saved guide anchor is invalid"
            point = np.sum(
                snapshot["coords_world"][list(local_triangle)] * barycentric[:, None],
                axis=0,
            )
            normal = np.sum(
                snapshot["normals_world"][list(local_triangle)] * barycentric[:, None],
                axis=0,
            )
            norm = float(np.linalg.norm(normal))
            if norm <= 1.0e-12 or not np.all(np.isfinite(point)):
                return None, None, "saved guide normal is invalid"
            controls.append(Vector(tuple(point)))
            normals.append(Vector(tuple(normal / norm)))
    except (KeyError, IndexError, MemoryError, TypeError, ValueError):
        return None, None, "saved guide anchor no longer matches the component"
    if len(controls) < 2 or len(controls) > GUIDED_RIDGE_MAX_CONTROLS:
        return None, None, "saved guide point count is invalid"
    return controls, normals, None


def _guided_ridge_current_signature(obj, snapshot):
    import numpy as np
    mesh = obj.data
    try:
        if int(obj.as_pointer()) != int(snapshot["object_pointer"]) or int(mesh.as_pointer()) != int(snapshot["mesh_pointer"]):
            return None
        counts = (len(mesh.vertices), len(mesh.edges), len(mesh.polygons), len(mesh.loops))
        if counts != tuple(snapshot["counts"]):
            return None
        attr = _guided_ridge_face_set_attribute(obj)
        if attr is None:
            return None
        ids = np.empty(len(attr.data), dtype=np.int32)
        attr.data.foreach_get("value", ids)
        if _guided_ridge_hash_array(ids[snapshot["face_indices"]]) != snapshot["signature"]["face_set_hash"]:
            return None
        if _guided_ridge_matrix_signature(obj.matrix_world) != snapshot["signature"]["matrix_world"]:
            return None
        current_face_vertices = tuple(
            tuple(int(vertex) for vertex in mesh.polygons[index].vertices)
            for index in snapshot["face_indices"]
        )
        if current_face_vertices != snapshot["signature"]["face_vertices"]:
            return None
        coords_local = np.asarray([mesh.vertices[index].co[:] for index in snapshot["vertex_indices"]], dtype=np.float64)
        if _guided_ridge_hash_array(coords_local) != snapshot["signature"]["coordinate_hash"]:
            return None
        hidden = np.asarray([bool(mesh.polygons[index].hide) for index in snapshot["face_indices"]], dtype=bool)
        if _guided_ridge_hash_array(hidden) != snapshot["signature"]["hidden_faces_hash"]:
            return None
        hidden_vertices = np.zeros(len(mesh.vertices), dtype=bool)
        mesh.vertices.foreach_get("hide", hidden_vertices)
        if _guided_ridge_hash_array(hidden_vertices[snapshot["vertex_indices"]]) != snapshot["signature"]["hidden_vertices_hash"]:
            return None
        sculpt_mask = np.zeros(len(mesh.vertices), dtype=np.float64)
        mask_attr = mesh.attributes.get(".sculpt_mask")
        if mask_attr is not None and mask_attr.domain == "POINT" and len(mask_attr.data) == len(sculpt_mask):
            mask_attr.data.foreach_get("value", sculpt_mask)
        sculpt_mask = np.clip(sculpt_mask, 0.0, 1.0)
        if _guided_ridge_hash_array(sculpt_mask[snapshot["vertex_indices"]]) != snapshot["signature"]["sculpt_mask_hash"]:
            return None
        return coords_local
    except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _guided_ridge_restore_coordinates(context, state, coordinates_local):
    """Restore the immutable component coordinates after a failed write."""
    import numpy as np
    obj = state.get("obj") if state else None
    snapshot = state.get("snapshot") if state else None
    if obj is None or snapshot is None:
        return False, "rollback state is unavailable"
    try:
        mesh = obj.data
        values = np.asarray(coordinates_local, dtype=np.float64)
        vertex_indices = snapshot["vertex_indices"]
        if len(values) != len(vertex_indices):
            return False, "rollback coordinate count mismatch"
        for local_index, vertex_index in enumerate(vertex_indices):
            mesh.vertices[int(vertex_index)].co = tuple(values[local_index])
        mesh.update()
        obj.update_tag(refresh={"DATA"})
        context.view_layer.update()
        return True, None
    except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
        return False, str(error)


def _guided_ridge_build_candidate(snapshot, guide):
    """Build the bounded C-like ridge candidate in world space."""
    import numpy as np
    before = np.asarray(snapshot["coords_world"], dtype=np.float64)
    guide = np.asarray([tuple(point) for point in guide], dtype=np.float64)
    if len(guide) < 2:
        raise ValueError("at least two guide points are required")
    segment_lengths = np.linalg.norm(np.diff(guide, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    length = float(cumulative[-1])
    if length <= 1.0e-8:
        raise ValueError("guide is too short")
    tangents = np.zeros_like(guide)
    tangents[0] = guide[1] - guide[0]
    tangents[-1] = guide[-1] - guide[-2]
    if len(guide) > 2:
        tangents[1:-1] = guide[2:] - guide[:-2]
    tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(tangent_norm, 1.0e-12)
    normals = np.asarray(snapshot["normals_world"], dtype=np.float64)
    vertices = before
    distances = np.linalg.norm(vertices[:, None, :] - guide[None, :, :], axis=2)
    station_index = np.argmin(distances, axis=1)
    station = cumulative[station_index]
    tangent = tangents[station_index]
    normal = normals.copy()
    normal -= np.sum(normal * tangent, axis=1, keepdims=True) * tangent
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-12)
    lateral = np.cross(tangent, normal)
    lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1.0e-12)
    guide_at_vertex = guide[station_index]
    rel = vertices - guide_at_vertex
    q = np.sum(rel * lateral, axis=1)
    h = np.sum(rel * normal, axis=1)
    boundary_mask = np.zeros(len(vertices), dtype=bool)
    boundary_local = [snapshot["vertex_indices"].index(index) for index in snapshot["boundary_vertices"]]
    boundary_mask[boundary_local] = True
    # Estimate two actual side anchors at each guide station from boundary
    # vertices.  This keeps the cross section broad and preserves the edge.
    boundary_q = q[boundary_mask]
    boundary_h = h[boundary_mask]
    if len(boundary_q) < 2:
        raise ValueError("component boundary is unavailable")
    qlo = float(np.min(boundary_q))
    qhi = float(np.max(boundary_q))
    hlo = float(np.median(boundary_h[boundary_q <= qlo + max(abs(qlo) * 0.05, 1.0e-8)]))
    hhi = float(np.median(boundary_h[boundary_q >= qhi - max(abs(qhi) * 0.05, 1.0e-8)]))
    target = np.where(
        q <= 0.0,
        (q / min(qlo, -1.0e-10)) * hlo,
        (q / max(qhi, 1.0e-10)) * hhi,
    )
    root_guard = np.asarray([_guided_ridge_smoothstep((s / length - 0.29) / 0.13) for s in station])
    tip_hold = 0.08 * length
    tip_fade = 0.04 * length
    tip_guard = np.asarray([
        _guided_ridge_smoothstep((length - tip_fade - s) / max(tip_fade, 1.0e-12))
        for s in station
    ])
    boundary_distance = np.min(
        np.linalg.norm(vertices[:, None, :] - vertices[boundary_mask][None, :, :], axis=2), axis=1
    )
    boundary_width = max(length * 0.0092, 1.0e-7)
    boundary_guard = np.asarray([_guided_ridge_smoothstep(distance / boundary_width) for distance in boundary_distance])
    weight = 0.82 * root_guard * tip_guard * boundary_guard
    weight[boundary_mask] = 0.0
    weight[station <= 0.0] = 0.0
    delta_h = (target - h) * weight
    anchored = boundary_mask | (station <= 0.0)
    for _ in range(2):
        relaxed = delta_h.copy()
        for index, neighbors in enumerate(snapshot["adjacency"]):
            if anchored[index] or not neighbors:
                continue
            relaxed[index] = 0.72 * delta_h[index] + 0.28 * float(np.mean(delta_h[list(neighbors)]))
        relaxed[anchored] = 0.0
        delta_h = relaxed
    candidate = before + delta_h[:, None] * normal
    # Candidate-C-only correction: preserve A's broad field and add a limited
    # quarter-strength tip hold in the final 12% of the normalized guide.
    correction = np.asarray([
        0.25 * _guided_ridge_smoothstep((s - (length - 0.16 * length)) / (0.04 * length))
        for s in station
    ])
    candidate += (delta_h * correction)[:, None] * normal
    if not np.all(np.isfinite(candidate)):
        raise ValueError("candidate contains non-finite coordinates")
    return candidate, {
        "length": length,
        "station": station,
        "boundary_mask": boundary_mask,
        "weight": weight,
        "delta_h": delta_h,
        "normal": normal,
    }


def _guided_ridge_smooth_array(values, sigma=1.8):
    import numpy as np
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(np.asarray(values, dtype=np.float64), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _guided_ridge_unit_array(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1.0e-12)


def _guided_ridge_stable_frame_steps(before, faces, guide, controls, control_normals):
    # Publish the stage before touching the first potentially large array so a
    # guide edit gets one event-loop turn with visible progress as well as the
    # initial snapshot preparation.
    yield {
        "stage": "building guide frame",
        "stage_index": 0,
        "stage_count": 11,
        "done": 0,
        "total": 0,
        "indeterminate": True,
    }
    import numpy as np
    guide = np.asarray(guide, dtype=np.float64)
    segment = np.diff(guide, axis=0)
    segment_length = np.linalg.norm(segment, axis=1)
    if np.any(segment_length <= 1.0e-10):
        raise ValueError("guide contains a zero-length segment")
    arc = np.r_[0.0, np.cumsum(segment_length)]
    tangent = np.empty_like(guide)
    tangent[0] = segment[0] / segment_length[0]
    tangent[-1] = segment[-1] / segment_length[-1]
    tangent[1:-1] = _guided_ridge_unit_array(guide[2:] - guide[:-2])
    raw_tangent = tangent.copy()
    controls = np.asarray(controls, dtype=np.float64)
    control_normals = _guided_ridge_unit_array(np.asarray(control_normals, dtype=np.float64))
    control_arc = np.asarray([arc[int(np.argmin(np.linalg.norm(guide - point[None, :], axis=1)))] for point in controls])
    order = np.argsort(control_arc)
    control_arc = control_arc[order]
    control_normals = control_normals[order]
    keep = np.r_[True, np.diff(control_arc) > 1.0e-10]
    control_arc, control_normals = control_arc[keep], control_normals[keep]
    if len(control_arc) < 2:
        raise ValueError("guide controls do not span an arc")
    control_points_tangent = np.empty_like(controls)
    if len(controls) == 2:
        control_points_tangent[0] = controls[1] - controls[0]
        control_points_tangent[1] = controls[1] - controls[0]
    else:
        control_points_tangent[0] = controls[1] - controls[0]
        control_points_tangent[-1] = controls[-1] - controls[-2]
        control_points_tangent[1:-1] = controls[2:] - controls[:-2]
    control_points_tangent = _guided_ridge_unit_array(control_points_tangent)
    tangent = np.column_stack([
        np.interp(arc, control_arc, control_points_tangent[order][keep][:, axis]) for axis in range(3)
    ])
    tangent = _guided_ridge_unit_array(np.column_stack([_guided_ridge_smooth_array(tangent[:, axis], 1.8) for axis in range(3)]))
    tangent *= np.where(np.sum(tangent * raw_tangent, axis=1) < 0.0, -1.0, 1.0)[:, None]
    tangent = _guided_ridge_unit_array(tangent)
    base_normal = np.column_stack([
        np.interp(arc, control_arc, control_normals[:, axis]) for axis in range(3)
    ])
    base_normal = _guided_ridge_unit_array(base_normal)
    base_normal -= np.sum(base_normal * tangent, axis=1, keepdims=True) * tangent
    base_normal = _guided_ridge_unit_array(base_normal)
    base_normal = np.column_stack([_guided_ridge_smooth_array(base_normal[:, axis], 1.4) for axis in range(3)])
    vertex_normals = np.zeros_like(before)
    yield {
        "stage": "building guide frame",
        "stage_index": 0,
        "stage_count": 11,
        "done": 0,
        "total": int(math.ceil(len(faces) / 256.0)),
    }
    normal_total = int(math.ceil(len(faces) / 256.0))
    for normal_index, start in enumerate(range(0, len(faces), 256), 1):
        stop = min(start + 256, len(faces))
        tri = before[faces[start:stop]]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        local_faces = faces[start:stop]
        np.add.at(vertex_normals, local_faces[:, 0], cross)
        np.add.at(vertex_normals, local_faces[:, 1], cross)
        np.add.at(vertex_normals, local_faces[:, 2], cross)
        yield {
            "stage": "building guide frame",
            "stage_index": 0,
            "stage_count": 11,
            "done": normal_index,
            "total": normal_total,
        }
    vertex_normals = _guided_ridge_unit_array(vertex_normals)
    for index, point in enumerate(guide):
        distances = np.linalg.norm(before - point[None, :], axis=1)
        nearest_count = min(48, len(distances))
        nearest = np.argpartition(distances, nearest_count - 1)[:nearest_count]
        local = np.average(vertex_normals[nearest], axis=0, weights=1.0 / np.maximum(distances[nearest], 1.0e-9))
        if float(np.dot(base_normal[index], local)) < 0.0:
            base_normal[index] *= -1.0
        yield {
            "stage": "building guide frame",
            "stage_index": 0,
            "stage_count": 11,
            "done": index + 1,
            "total": len(guide),
        }
    base_normal = np.column_stack([_guided_ridge_smooth_array(base_normal[:, axis], 1.8) for axis in range(3)])
    base_normal -= np.sum(base_normal * tangent, axis=1, keepdims=True) * tangent
    normal = _guided_ridge_unit_array(base_normal)
    lateral = _guided_ridge_unit_array(np.cross(tangent, normal))
    normal = _guided_ridge_unit_array(np.cross(lateral, tangent))
    return {"guide": guide, "arc": arc, "tangent": tangent, "normal": normal, "lateral": lateral, "length": float(arc[-1])}


def _guided_ridge_stable_frame(before, faces, guide, controls, control_normals):
    """Synchronous compatibility wrapper around the cooperative frame job."""
    job = _guided_ridge_stable_frame_steps(before, faces, guide, controls, control_normals)
    try:
        while True:
            next(job)
    except StopIteration as complete:
        return complete.value


def _guided_ridge_frame_at(frame, station):
    import numpy as np
    index = int(np.clip(np.searchsorted(frame["arc"], station) - 1, 0, len(frame["arc"]) - 2))
    span = frame["arc"][index + 1] - frame["arc"][index]
    t = (station - frame["arc"][index]) / max(span, 1.0e-12)
    origin = frame["guide"][index] + t * (frame["guide"][index + 1] - frame["guide"][index])
    tangent = _guided_ridge_unit_array((1.0 - t) * frame["tangent"][index] + t * frame["tangent"][index + 1]).reshape(3)
    normal = _guided_ridge_unit_array((1.0 - t) * frame["normal"][index] + t * frame["normal"][index + 1]).reshape(3)
    normal = _guided_ridge_unit_array(normal - float(np.dot(normal, tangent)) * tangent).reshape(3)
    lateral = _guided_ridge_unit_array(np.cross(tangent, normal)).reshape(3)
    normal = _guided_ridge_unit_array(np.cross(lateral, tangent)).reshape(3)
    return origin, tangent, normal, lateral


def _guided_ridge_plane_boundary(points, faces, origin, plane_normal, boundary_edges):
    import numpy as np
    distance = (points - origin[None, :]) @ plane_normal
    boundary_points = []
    for first, second in boundary_edges:
        va, vb = float(distance[first]), float(distance[second])
        if abs(va) <= 1.0e-10:
            boundary_points.append(points[first].copy())
        if abs(vb) <= 1.0e-10:
            boundary_points.append(points[second].copy())
        if va * vb < -1.0e-20:
            boundary_points.append(points[first] + (va / (va - vb)) * (points[second] - points[first]))
    unique = []
    for point in boundary_points:
        if not any(np.linalg.norm(point - old) <= 1.0e-9 for old in unique):
            unique.append(point)
    return unique


def _guided_ridge_boundary_record(points, faces, boundary_edges, frame, station, previous_axis):
    import numpy as np
    origin, tangent, frame_normal, frame_lateral = _guided_ridge_frame_at(frame, station)
    cloud = np.asarray(_guided_ridge_plane_boundary(points, faces, origin, tangent, boundary_edges), dtype=np.float64).reshape(-1, 3)
    if len(cloud) < 2:
        return None
    orient = frame_lateral if previous_axis is None else previous_axis
    orient = _guided_ridge_unit_array(orient - np.dot(orient, tangent) * tangent).reshape(3)
    scalar = (cloud - origin[None, :]) @ orient
    lo_point, hi_point = cloud[int(np.argmin(scalar))], cloud[int(np.argmax(scalar))]
    axis = hi_point - lo_point
    axis -= np.dot(axis, tangent) * tangent
    if np.linalg.norm(axis) <= 1.0e-8:
        pair = max(((i, j) for i in range(len(cloud)) for j in range(i + 1, len(cloud))), key=lambda pair: np.linalg.norm(cloud[pair[1]] - cloud[pair[0]]))
        lo_point, hi_point = cloud[pair[0]], cloud[pair[1]]
        axis = hi_point - lo_point
        axis -= np.dot(axis, tangent) * tangent
    axis = _guided_ridge_unit_array(axis).reshape(3)
    if np.dot(axis, orient) < 0.0:
        axis *= -1.0
        lo_point, hi_point = hi_point, lo_point
    height_axis = _guided_ridge_unit_array(np.cross(tangent, axis)).reshape(3)
    if np.dot(height_axis, frame_normal) < 0.0:
        height_axis *= -1.0
    qlo = float(np.dot(lo_point - origin, axis))
    qhi = float(np.dot(hi_point - origin, axis))
    hlo = float(np.dot(lo_point - origin, height_axis))
    hhi = float(np.dot(hi_point - origin, height_axis))
    if not qlo < qhi:
        return None
    return {"station": station, "axis": axis, "height_axis": height_axis, "qlo": qlo, "qhi": qhi, "hlo": hlo, "hhi": hhi}


def _guided_ridge_bounded_pairwise_chunk_size(primary_count, secondary_count):
    return max(
        1,
        min(
            int(primary_count),
            GUIDED_RIDGE_DISTANCE_MAX_PAIRS // max(int(secondary_count), 1),
        ),
    )


def _guided_ridge_boundary_local_data(snapshot, vertex_count):
    """Convert snapshot-global boundary vertices to component-local indices.

    ``snapshot['triangles']`` and ``boundary_edges`` use component-local
    indices, while ``boundary_vertices`` is intentionally retained in the
    mesh-global index space for signatures and write protection.  Keeping the
    conversion explicit prevents an open section from silently losing its
    boundary endpoint when the component is a non-contiguous mesh island.
    """
    import numpy as np

    global_to_local = {
        int(global_index): int(local_index)
        for local_index, global_index in enumerate(snapshot.get("vertex_indices", ()))
    }
    local_vertices = []
    for global_index in snapshot.get("boundary_vertices", ()):
        local_index = global_to_local.get(int(global_index))
        if local_index is None:
            raise ValueError("boundary vertex is outside the captured component")
        local_vertices.append(local_index)
    local_vertex_set = set(local_vertices)
    boundary_edges = []
    for edge in snapshot.get("boundary_edges", ()):
        if len(edge) != 2:
            raise ValueError("boundary edge is malformed")
        first, second = (int(edge[0]), int(edge[1]))
        if not (0 <= first < int(vertex_count) and 0 <= second < int(vertex_count)):
            # A future/foreign snapshot may retain global edge indices.  Do
            # not guess for an in-range pair: only convert an unmistakably
            # global edge here.
            mapped_first = global_to_local.get(first)
            mapped_second = global_to_local.get(second)
            if mapped_first is None or mapped_second is None:
                raise ValueError("boundary edge is outside the captured component")
            first, second = mapped_first, mapped_second
        boundary_edges.append((first, second))
    return (
        np.asarray(sorted(local_vertex_set), dtype=np.int64),
        np.asarray(boundary_edges, dtype=np.int64).reshape((-1, 2)),
    )


def _guided_ridge_bounded_boundary_distance_steps(before, boundary_mask):
    """Yield one bounded boundary-distance chunk at a time.

    The synchronous wrapper below consumes the same generator, so the
    numeric operation and ordering remain identical for Repeat Last and the
    modal compute path while the latter returns to Blender between chunks.
    """
    import numpy as np

    boundary_points = np.asarray(before[boundary_mask], dtype=np.float64)
    if len(boundary_points) == 0:
        raise ValueError("component boundary is unavailable")
    result = np.empty(len(before), dtype=np.float64)
    chunk_size = min(
        _guided_ridge_bounded_pairwise_chunk_size(len(before), len(boundary_points)),
        1,
    )
    boundary_slice = 512
    vertex_total = int(math.ceil(len(before) / float(max(chunk_size, 1))))
    boundary_total = int(math.ceil(len(boundary_points) / float(boundary_slice)))
    total = vertex_total * boundary_total
    work_index = 0
    for start in range(0, len(before), chunk_size):
        stop = min(start + chunk_size, len(before))
        local_minimum = np.full(stop - start, np.inf, dtype=np.float64)
        for boundary_start in range(0, len(boundary_points), boundary_slice):
            boundary_stop = min(boundary_start + boundary_slice, len(boundary_points))
            delta = before[start:stop, None, :] - boundary_points[boundary_start:boundary_stop][None, :, :]
            local_minimum = np.minimum(
                local_minimum,
                np.min(np.linalg.norm(delta, axis=2), axis=1),
            )
            work_index += 1
            yield {
                "stage": "measuring boundary distances",
                "stage_index": 4,
                "stage_count": 5,
                "done": work_index,
                "total": total,
            }
        result[start:stop] = local_minimum
    return result, chunk_size, len(boundary_points)


def _guided_ridge_bounded_boundary_distance(before, boundary_mask):
    job = _guided_ridge_bounded_boundary_distance_steps(before, boundary_mask)
    try:
        while True:
            next(job)
    except StopIteration as complete:
        return complete.value


def _guided_ridge_surface_section_record_job(points, triangles, boundary_edges, boundary_vertices, frame, station, previous_axis=None):
    """Extract one deterministic surface cross-section around the guide crest.

    The old implementation intersected the station plane with the outer Face
    Set boundary.  That works only when the guide crosses the component from
    edge to edge.  A Guided Ridge guide instead follows an interior crest, so
    the section is built from the component surface triangles and only the
    crest-side arc is selected for deformation.

    This helper deliberately rejects ambiguous topology (coplanar triangles,
    branches, multiple equally-close sheets, and concave/ambiguous arcs).  It
    returns the same q/h anchor fields consumed by the existing profile code.
    """
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    boundary_edges = {tuple(sorted((int(a), int(b)))) for a, b in np.asarray(boundary_edges, dtype=np.int64)}
    boundary_vertices = {int(value) for value in boundary_vertices}
    origin, tangent, frame_normal, frame_lateral = _guided_ridge_frame_at(frame, float(station))
    eps = max(float(frame.get("average_edge", 0.0) or 0.0) * 1.0e-6, 1.0e-9)
    edge_pairs = ((0, 1), (1, 2), (2, 0))
    segments = {}
    point_by_id = {}
    segment_triangles = {}
    signed = np.empty((len(triangles), 3), dtype=np.float64)
    candidate_parts = []
    signed_chunk = 64
    signed_total = int(math.ceil(len(triangles) / float(signed_chunk)))
    for signed_position, signed_start in enumerate(range(0, len(triangles), signed_chunk), 1):
        signed_stop = min(signed_start + signed_chunk, len(triangles))
        tri_points = points[triangles[signed_start:signed_stop]]
        signed[signed_start:signed_stop] = np.sum(
            (tri_points - origin[None, None, :]) * tangent[None, None, :], axis=2
        )
        chunk_values = signed[signed_start:signed_stop]
        crossing = np.any(chunk_values < -eps, axis=1) & np.any(chunk_values > eps, axis=1)
        near = np.any(np.abs(chunk_values) <= eps, axis=1)
        local_candidates = np.flatnonzero(crossing | near)
        if len(local_candidates):
            candidate_parts.append(local_candidates + signed_start)
        yield {
            "stage": "classifying station triangles",
            "stage_index": 1,
            "stage_count": 8,
            "done": signed_position,
            "total": signed_total,
        }
    candidate_indices = (
        np.concatenate(candidate_parts).astype(np.int64, copy=False)
        if candidate_parts else np.empty(0, dtype=np.int64)
    )
    candidate_total = len(candidate_indices)
    for candidate_position, triangle_index in enumerate(candidate_indices, 1):
        values = signed[int(triangle_index)]
        if bool(np.all(np.abs(values) <= eps)):
            return None, "coplanar surface triangle is ambiguous"
        local_ids = []
        local_points = {}
        for first, second in edge_pairs:
            va, vb = float(values[first]), float(values[second])
            ia = int(triangles[int(triangle_index), first])
            ib = int(triangles[int(triangle_index), second])
            if abs(va) <= eps and abs(vb) <= eps:
                for vertex_index in (ia, ib):
                    key = ("v", vertex_index)
                    local_ids.append(key)
                    local_points[key] = points[vertex_index]
            elif abs(va) <= eps:
                key = ("v", ia)
                local_ids.append(key)
                local_points[key] = points[ia]
            elif abs(vb) <= eps:
                key = ("v", ib)
                local_ids.append(key)
                local_points[key] = points[ib]
            elif va * vb < -eps * eps:
                key = ("e", min(ia, ib), max(ia, ib))
                factor = va / (va - vb)
                local_ids.append(key)
                local_points[key] = points[ia] + factor * (points[ib] - points[ia])
        unique_ids = []
        for key in local_ids:
            if key not in unique_ids:
                unique_ids.append(key)
                point_by_id.setdefault(key, np.asarray(local_points[key], dtype=np.float64).copy())
        if len(unique_ids) < 2:
            continue
        if len(unique_ids) != 2:
            return None, "triangle-plane intersection is ambiguous"
        first, second = unique_ids
        if first == second or np.linalg.norm(point_by_id[first] - point_by_id[second]) <= 1.0e-10:
            return None, "zero-length surface section segment"
        segment_key = tuple(sorted((first, second)))
        segments[segment_key] = (first, second)
        segment_triangles.setdefault(segment_key, []).append(int(triangle_index))
        if candidate_position % 512 == 0 or candidate_position == candidate_total:
            yield {
                "stage": "extracting surface section",
                "stage_index": 1,
                "stage_count": 8,
                "done": candidate_position,
                "total": candidate_total,
            }
    if not segments:
        return None, "station has no surface intersection"

    graph = {}
    for first, second in segments.values():
        graph.setdefault(first, set()).add(second)
        graph.setdefault(second, set()).add(first)
    if any(len(neighbors) > 2 for neighbors in graph.values()):
        return None, "surface section has a branch or non-manifold degree"

    components = []
    unseen = set(graph)
    while unseen:
        root = min(unseen, key=repr)
        stack = [root]
        component = set()
        unseen.remove(root)
        while stack:
            current = stack.pop()
            component.add(current)
            for neighbor in graph[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        endpoints = [key for key in component if len(graph[key]) == 1]
        if len(endpoints) not in (0, 2):
            return None, "surface section has an invalid open polyline"
        if any(len(graph[key]) != 2 for key in component) and len(endpoints) != 2:
            return None, "surface section has an invalid loop degree"
        if len(component) < 2:
            continue
        components.append((component, endpoints))
    if not components:
        return None, "station has no usable surface polyline"

    def ordered_component(component, endpoints):
        if endpoints:
            start = min(endpoints, key=repr)
            ordered = [start]
            previous = None
            current = start
            while True:
                choices = sorted((value for value in graph[current] if value != previous), key=repr)
                if not choices:
                    break
                next_value = choices[0]
                ordered.append(next_value)
                previous, current = current, next_value
                if current in endpoints and current != start:
                    break
            return ordered, False
        start = min(component, key=repr)
        ordered = [start]
        previous = None
        current = start
        while True:
            choices = sorted((value for value in graph[current] if value != previous), key=repr)
            if not choices:
                break
            next_value = choices[0]
            if next_value == start:
                break
            ordered.append(next_value)
            previous, current = current, next_value
        return ordered, True

    def closest_on_polyline(ordered, closed):
        best = None
        count = len(ordered)
        limit = count if closed else count - 1
        for index in range(max(limit, 0)):
            first = np.asarray(point_by_id[ordered[index]], dtype=np.float64)
            second = np.asarray(point_by_id[ordered[(index + 1) % count]], dtype=np.float64)
            delta = second - first
            factor = float(np.dot(origin - first, delta) / max(np.dot(delta, delta), 1.0e-30))
            factor = min(max(factor, 0.0), 1.0)
            point = first + factor * delta
            distance = float(np.linalg.norm(point - origin))
            if best is None or distance < best[0]:
                best = (distance, index, factor, point)
        return best

    candidates = []
    for component, endpoints in components:
        ordered, closed = ordered_component(component, endpoints)
        closest = closest_on_polyline(ordered, closed)
        if closest is None:
            continue
        candidates.append((closest[0], ordered, closed, endpoints, closest))
    candidates.sort(key=lambda value: value[0])
    if not candidates:
        return None, "surface section has no crest candidate"
    crest_distance, ordered, closed, endpoints, closest = candidates[0]
    tie_epsilon = max(eps * 32.0, float(frame.get("average_edge", 0.0) or 0.0) * 0.05)
    if len(candidates) > 1 and candidates[1][0] - crest_distance <= tie_epsilon:
        return None, "multiple equally-close surface sheets"
    if crest_distance > max(float(frame.get("average_edge", 0.0) or 0.0) * 8.0, 1.0e-4):
        return None, "surface section crest is too far from the guide"

    segment_index, segment_factor, crest_point = closest[1], closest[2], closest[3]
    count = len(ordered)
    first_id = ordered[segment_index]
    second_id = ordered[(segment_index + 1) % count]
    forward = [crest_point]
    backward = [crest_point]
    # ``None`` denotes the interpolated crest point.  The remaining keys are
    # the topological provenance used later to keep the selected surface arc
    # separate from its complement.
    forward_keys = [None]
    backward_keys = [None]
    if segment_factor < 1.0 - 1.0e-6:
        forward.append(np.asarray(point_by_id[second_id], dtype=np.float64))
        forward_keys.append(second_id)
    if segment_factor > 1.0e-6:
        backward.append(np.asarray(point_by_id[first_id], dtype=np.float64))
        backward_keys.append(first_id)
    if closed:
        index = (segment_index + 1) % count
        while True:
            next_index = (index + 1) % count
            if next_index == (segment_index + 1) % count:
                break
            forward.append(np.asarray(point_by_id[ordered[next_index]], dtype=np.float64))
            forward_keys.append(ordered[next_index])
            index = next_index
        index = segment_index
        while True:
            next_index = (index - 1) % count
            if next_index == segment_index:
                break
            backward.append(np.asarray(point_by_id[ordered[next_index]], dtype=np.float64))
            backward_keys.append(ordered[next_index])
            index = next_index
    else:
        # Open polylines have real endpoints.  Do not use modulo arithmetic:
        # wrapping here would join the crest to the opposite endpoint and
        # accidentally turn a boundary strip into a closed loop.
        for next_index in range(segment_index + 2, count):
            forward.append(np.asarray(point_by_id[ordered[next_index]], dtype=np.float64))
            forward_keys.append(ordered[next_index])
        for next_index in range(segment_index - 1, -1, -1):
            backward.append(np.asarray(point_by_id[ordered[next_index]], dtype=np.float64))
            backward_keys.append(ordered[next_index])

    axis = np.asarray(frame_lateral, dtype=np.float64)
    if previous_axis is not None:
        previous_axis = np.asarray(previous_axis, dtype=np.float64)
        previous_axis -= np.dot(previous_axis, tangent) * tangent
        if np.linalg.norm(previous_axis) > 1.0e-8:
            axis = _guided_ridge_unit_array(previous_axis).reshape(3)
    axis = _guided_ridge_unit_array(axis - np.dot(axis, tangent) * tangent).reshape(3)
    height_axis = _guided_ridge_unit_array(np.cross(tangent, axis)).reshape(3)
    if np.dot(height_axis, frame_normal) < 0.0:
        height_axis *= -1.0

    def branch_values(branch):
        values = np.asarray([(point - origin) @ axis for point in branch], dtype=np.float64)
        heights = np.asarray([(point - origin) @ height_axis for point in branch], dtype=np.float64)
        return values, heights

    forward_q, forward_h = branch_values(forward)
    backward_q, backward_h = branch_values(backward)
    # A closed loop has two arcs from the crest.  Each arc must stop at the
    # first opposite-side crossing (normally the far-side point); otherwise a
    # full loop would incorrectly make both branches contain both signs.
    side_epsilon = max(eps * 16.0, 1.0e-7)
    def trim_at_opposite_side(branch, keys, values, heights):
        nonzero = np.flatnonzero(np.abs(values) > side_epsilon)
        if len(nonzero) == 0:
            return branch, keys, values, heights
        expected = 1.0 if values[int(nonzero[0])] > 0.0 else -1.0
        for index in range(int(nonzero[0]) + 1, len(values)):
            if expected * values[index] < -side_epsilon:
                stop = max(index, 2)
                return branch[:stop], keys[:stop], values[:stop], heights[:stop]
        return branch, keys, values, heights
    forward, forward_keys, forward_q, forward_h = trim_at_opposite_side(
        forward, forward_keys, forward_q, forward_h
    )
    backward, backward_keys, backward_q, backward_h = trim_at_opposite_side(
        backward, backward_keys, backward_q, backward_h
    )
    branch_data = (("forward", forward_q, forward_h), ("backward", backward_q, backward_h))
    negative = [item for item in branch_data if np.min(item[1]) < -side_epsilon]
    positive = [item for item in branch_data if np.max(item[1]) > side_epsilon]
    if len(negative) != 1 or len(positive) != 1 or negative[0][0] == positive[0][0]:
        return None, "crest-side arc has competing, folded, or ambiguous branches"
    if np.any(negative[0][1] > side_epsilon) or np.any(positive[0][1] < -side_epsilon):
        return None, "surface arc is concave or self-intersecting"
    left_q, left_h = negative[0][1], negative[0][2]
    right_q, right_h = positive[0][1], positive[0][2]
    qlo = float(np.min(left_q))
    qhi = float(np.max(right_q))
    hlo = float(left_h[int(np.argmin(left_q))])
    hhi = float(right_h[int(np.argmax(right_q))])
    if not (qlo < -side_epsilon < 0.0 < side_epsilon < qhi):
        return None, "surface crest section has insufficient lateral width"
    chord_at_crest = (hlo * qhi - hhi * qlo) / (qhi - qlo)
    crest_side = -1.0 if chord_at_crest > 0.0 else 1.0
    if abs(chord_at_crest) <= side_epsilon:
        return None, "crest-side arc is indistinguishable from its anchor chord"

    def endpoint_is_boundary(key):
        if key[0] == "v":
            return int(key[1]) in boundary_vertices
        return (int(key[1]), int(key[2])) in boundary_edges
    if not closed:
        if not endpoint_is_boundary(ordered[0]) or not endpoint_is_boundary(ordered[-1]):
            return None, "open surface section does not terminate at component boundaries"

    branch_by_name = {
        "forward": (forward, forward_keys, forward_q),
        "backward": (backward, backward_keys, backward_q),
    }
    selected_arc_keys = []
    selected_arc_triangles = set()
    selected_arc_vertices = set()
    selected_anchor_vertices = set()
    selected_segments = set()

    def add_key_vertices(key, target):
        if key is None:
            return
        if key[0] == "v":
            target.add(int(key[1]))
        elif key[0] == "e":
            target.update((int(key[1]), int(key[2])))

    def collect_branch(name, q_values):
        branch, keys, _branch_q = branch_by_name[name]
        selected_arc_keys.extend(key for key in keys if key is not None)
        for key in keys:
            add_key_vertices(key, selected_arc_vertices)
        for first_key, second_key in zip(keys, keys[1:]):
            if first_key is None or second_key is None:
                segment_key = tuple(sorted((first_id, second_id)))
            else:
                segment_key = tuple(sorted((first_key, second_key)))
            selected_segments.add(segment_key)
            selected_arc_triangles.update(segment_triangles.get(segment_key, ()))
        if len(q_values):
            add_key_vertices(keys[int(np.argmin(q_values))], selected_anchor_vertices)

    collect_branch(negative[0][0], left_q)
    collect_branch(positive[0][0], right_q)
    source_triangle = segment_triangles.get(tuple(sorted((first_id, second_id))), ())
    return {
        "station": float(station),
        "axis": axis,
        "height_axis": height_axis,
        "qlo": qlo,
        "qhi": qhi,
        "hlo": hlo,
        "hhi": hhi,
        "crest_side": float(crest_side),
        "crest_point": np.asarray(crest_point, dtype=np.float64),
        "source_triangle": int(source_triangle[0]) if source_triangle else -1,
        "arc_vertex_ids": tuple(sorted(selected_arc_vertices)),
        "arc_keys": tuple(repr(key) for key in selected_arc_keys),
        "arc_triangle_indices": tuple(sorted(int(value) for value in selected_arc_triangles)),
        "arc_segments": tuple(sorted(selected_segments, key=repr)),
        "anchor_vertex_ids": tuple(sorted(selected_anchor_vertices)),
        "closed": bool(closed),
    }, None


def _guided_ridge_surface_section_record(points, triangles, boundary_edges, boundary_vertices, frame, station, previous_axis=None):
    """Synchronous compatibility wrapper for one surface section."""
    job = _guided_ridge_surface_section_record_job(
        points, triangles, boundary_edges, boundary_vertices, frame, station, previous_axis
    )
    try:
        while True:
            next(job)
    except StopIteration as complete:
        return complete.value


def _guided_ridge_arc_candidate_steps_legacy(snapshot, guide, controls, control_normals):
    """Yield bounded cross-section work for the modal computing phase."""
    import numpy as np
    before = np.asarray(snapshot["coords_world"], dtype=np.float64)
    faces = np.asarray(snapshot["triangles"], dtype=np.int64)
    frame_job = _guided_ridge_stable_frame_steps(before, faces, guide, controls, control_normals)
    try:
        while True:
            yield next(frame_job)
    except StopIteration as complete:
        frame = complete.value
    frame["average_edge"] = float(snapshot.get("average_edge", 0.0) or 0.0)
    boundary_vertices, boundary_edges = _guided_ridge_boundary_local_data(snapshot, len(before))
    boundary_mask = np.zeros(len(before), dtype=bool)
    boundary_mask[boundary_vertices] = True
    stations = np.linspace(0.001, frame["length"] - 0.001, 64)
    records = []
    previous_axis = None
    invalid_reasons = {}
    yield {"stage": "extracting surface sections", "stage_index": 1, "stage_count": 5, "done": 0, "total": len(stations)}
    for index, station in enumerate(stations):
        section_job = _guided_ridge_surface_section_record_job(
            before, faces, boundary_edges, boundary_vertices, frame, float(station), previous_axis
        )
        try:
            while True:
                section_progress = next(section_job)
                section_progress["station_index"] = index
                yield section_progress
        except StopIteration as complete:
            record, reason = complete.value
        if record is not None:
            records.append(record)
            previous_axis = record["axis"]
        else:
            invalid_reasons[reason or "invalid section"] = invalid_reasons.get(reason or "invalid section", 0) + 1
        yield {
            "stage": "extracting surface sections",
            "stage_index": 1,
            "stage_count": 5,
            "done": index + 1,
            "total": len(stations),
        }
    boundary_job = _guided_ridge_bounded_boundary_distance_steps(before, boundary_mask)
    boundary_distance = None
    try:
        while True:
            progress = next(boundary_job)
            yield progress
    except StopIteration as complete:
        boundary_distance = complete.value
    yield {"stage": "building crest-side profile", "stage_index": 3, "stage_count": 5, "done": 0, "total": 1}
    candidate_job = _guided_ridge_arc_candidate_job_legacy(
        snapshot,
        guide,
        controls,
        control_normals,
        prepared_sections=(frame, stations, records, invalid_reasons, boundary_distance),
    )
    try:
        while True:
            progress = next(candidate_job)
            yield progress
    except StopIteration as complete:
        return complete.value


def _guided_ridge_arc_candidate_job_legacy(snapshot, guide, controls, control_normals, prepared_sections=None):
    import numpy as np
    before = np.asarray(snapshot["coords_world"], dtype=np.float64)
    faces = np.asarray(snapshot["triangles"], dtype=np.int64)
    prepared_boundary = []
    if prepared_sections is None:
        frame = _guided_ridge_stable_frame(before, faces, guide, controls, control_normals)
        frame["average_edge"] = float(snapshot.get("average_edge", 0.0) or 0.0)
        boundary_vertices, boundary_edges = _guided_ridge_boundary_local_data(snapshot, len(before))
        stations = np.linspace(0.001, frame["length"] - 0.001, 64)
        records = []
        previous_axis = None
        invalid_reasons = {}
        for station in stations:
            record, reason = _guided_ridge_surface_section_record(
                before, faces, boundary_edges, boundary_vertices, frame, float(station), previous_axis
            )
            if record is not None:
                records.append(record)
                previous_axis = record["axis"]
            else:
                invalid_reasons[reason or "invalid section"] = invalid_reasons.get(reason or "invalid section", 0) + 1
    else:
        prepared_frame, prepared_stations, records, invalid_reasons, *prepared_boundary = prepared_sections
        frame = prepared_frame
        stations = np.asarray(prepared_stations, dtype=np.float64)
        boundary_vertices, boundary_edges = _guided_ridge_boundary_local_data(snapshot, len(before))
    if len(records) < 8:
        reason = max(invalid_reasons, key=invalid_reasons.get) if invalid_reasons else "unknown section failure"
        raise ValueError(f"too few valid surface sections ({reason})")
    valid_indices = np.asarray([int(np.argmin(np.abs(stations - record["station"]))) for record in records], dtype=np.int64)
    if int(valid_indices[0]) > 12 or int(valid_indices[-1]) < len(stations) - 13:
        raise ValueError("surface sections do not cover the guide length")
    gaps = np.diff(valid_indices)
    # A dense, irregular surface can legitimately reject a short run of
    # folded stations.  Keep the missing run bounded without requiring every
    # plane to be usable; a gap of 24/64 still rejects endpoint-only samples.
    if len(gaps) and int(np.max(gaps)) > 24:
        raise ValueError("surface sections have an excessive missing run")
    for left, right in zip(records, records[1:]):
        if float(np.dot(left["axis"], right["axis"])) <= 0.0:
            raise ValueError("surface section orientation is discontinuous")
        if float(np.linalg.norm(left["crest_point"] - right["crest_point"])) > max(
            frame["average_edge"] * 12.0, 1.0e-4
        ):
            raise ValueError("surface section crest is discontinuous")
    valid = np.asarray([record["station"] for record in records], dtype=float)
    arrays = {
        key: np.asarray([record[key] for record in records], dtype=float)
        for key in ("axis", "height_axis", "qlo", "qhi", "hlo", "hhi", "crest_side")
    }
    anchors = {}
    for key, values in arrays.items():
        if values.ndim == 2:
            anchors[key] = np.column_stack([np.interp(stations, valid, values[:, axis]) for axis in range(3)])
        else:
            anchors[key] = np.interp(stations, valid, values)
    segment = np.diff(frame["guide"], axis=0)
    length2 = np.sum(segment * segment, axis=1)
    if not np.all(np.isfinite(length2)) or np.any(length2 <= 1.0e-16):
        raise ValueError("guide contains a degenerate segment")
    seg_index = np.empty(len(before), dtype=np.int32)
    t = np.empty(len(before), dtype=np.float64)
    chunk_size = min(
        _guided_ridge_bounded_pairwise_chunk_size(len(before), len(segment)),
        128,
    )
    guide_chunk_total = int(math.ceil(len(before) / float(max(chunk_size, 1))))
    for start in range(0, len(before), chunk_size):
        stop = min(start + chunk_size, len(before))
        block = before[start:stop, None, :] - frame["guide"][:-1][None, :, :]
        parameter = np.clip(np.sum(block * segment[None, :, :], axis=2) / length2[None, :], 0.0, 1.0)
        distance2 = np.sum((block - parameter[:, :, None] * segment[None, :, :]) ** 2, axis=2)
        seg_index[start:stop] = np.argmin(distance2, axis=1)
        t[start:stop] = parameter[np.arange(stop - start), seg_index[start:stop]]
        yield {
            "stage": "projecting vertices to guide",
            "stage_index": 4,
            "stage_count": 8,
            "done": int(stop // max(chunk_size, 1)),
            "total": guide_chunk_total,
        }
    origin = frame["guide"][seg_index] + t[:, None] * segment[seg_index]
    station = frame["arc"][seg_index] + t * (frame["arc"][seg_index + 1] - frame["arc"][seg_index])
    tangent = _guided_ridge_unit_array((1.0 - t)[:, None] * frame["tangent"][seg_index] + t[:, None] * frame["tangent"][seg_index + 1])
    u = np.clip(station / frame["length"] * (len(stations) - 1), 0.0, len(stations) - 1)
    i0 = np.floor(u).astype(int)
    i1 = np.minimum(i0 + 1, len(stations) - 1)
    ft = u - i0
    axis = _guided_ridge_unit_array((1.0 - ft)[:, None] * anchors["axis"][i0] + ft[:, None] * anchors["axis"][i1])
    raw_height_axis = _guided_ridge_unit_array((1.0 - ft)[:, None] * anchors["height_axis"][i0] + ft[:, None] * anchors["height_axis"][i1])
    axis -= np.sum(axis * tangent, axis=1, keepdims=True) * tangent
    axis = _guided_ridge_unit_array(axis)
    height_axis = _guided_ridge_unit_array(np.cross(tangent, axis))
    signs = np.sum(height_axis * raw_height_axis, axis=1)
    height_axis *= np.where(signs < 0.0, -1.0, 1.0)[:, None]
    q = np.sum((before - origin) * axis, axis=1)
    h = np.sum((before - origin) * height_axis, axis=1)
    qlo = (1.0 - ft) * anchors["qlo"][i0] + ft * anchors["qlo"][i1]
    qhi = (1.0 - ft) * anchors["qhi"][i0] + ft * anchors["qhi"][i1]
    hlo = (1.0 - ft) * anchors["hlo"][i0] + ft * anchors["hlo"][i1]
    hhi = (1.0 - ft) * anchors["hhi"][i0] + ft * anchors["hhi"][i1]
    target = np.where(q <= 0.0, (q / np.minimum(qlo, -1.0e-8)) * hlo, (q / np.maximum(qhi, 1.0e-8)) * hhi)
    crest_side = np.interp(station, valid, arrays["crest_side"])
    chord_height = np.where(
        q <= 0.0,
        hlo + (hhi - hlo) * np.clip((q - qlo) / np.maximum(qhi - qlo, 1.0e-12), 0.0, 1.0),
        hlo + (hhi - hlo) * np.clip((q - qlo) / np.maximum(qhi - qlo, 1.0e-12), 0.0, 1.0),
    )
    # A q/chord half-space is only a geometric guard.  The authoritative
    # membership comes from the selected section arc provenance; this keeps a
    # backside loop or a second sheet in the same half-space immutable.
    selected_arc_mask = np.zeros(len(before), dtype=bool)
    selected_section_anchor_mask = np.zeros(len(before), dtype=bool)
    selected_surface_triangle_mask = np.zeros(len(before), dtype=bool)
    for record in records:
        for local_index in record.get("arc_vertex_ids", ()):
            if 0 <= int(local_index) < len(before):
                selected_arc_mask[int(local_index)] = True
        for local_index in record.get("anchor_vertex_ids", ()):
            if 0 <= int(local_index) < len(before):
                selected_section_anchor_mask[int(local_index)] = True
        for triangle_index in record.get("arc_triangle_indices", ()):
            if 0 <= int(triangle_index) < len(faces):
                selected_surface_triangle_mask[faces[int(triangle_index)]] = True
    selected_arc_mask |= selected_surface_triangle_mask
    crest_side_mask = (
        (q >= qlo)
        & (q <= qhi)
        & (crest_side * (h - chord_height) >= -max(frame["average_edge"] * 0.25, 1.0e-7))
        & selected_arc_mask
    )
    surface_normals = np.asarray(snapshot.get("normals_world", ()), dtype=np.float64)
    if surface_normals.shape != before.shape:
        raise ValueError("surface normals are unavailable for planing direction")
    outward_component = np.sum(height_axis * surface_normals, axis=1)
    outward_component[np.abs(outward_component) <= 1.0e-6] = 0.0
    anchor_mask = selected_section_anchor_mask.copy()
    guide_anchor_mask = np.zeros(len(before), dtype=bool)
    anchors, anchor_reason = _guided_ridge_surface_anchors(snapshot, controls)
    if anchors is None:
        raise ValueError(anchor_reason or "guide anchors are unavailable")
    local_lookup = {int(value): index for index, value in enumerate(snapshot["vertex_indices"])}
    for anchor in anchors:
        for global_index in anchor["vertex_indices"]:
            local_index = local_lookup.get(int(global_index))
            if local_index is not None:
                guide_anchor_mask[local_index] = True
    anchor_mask |= guide_anchor_mask
    boundary_mask = np.zeros(len(before), dtype=bool)
    boundary_mask[boundary_vertices] = True
    hidden_vertices = np.asarray(
        snapshot.get("hidden_vertices", np.zeros(len(before), dtype=bool)), dtype=bool
    )
    sculpt_mask = np.clip(
        np.asarray(snapshot.get("sculpt_mask", np.zeros(len(before), dtype=np.float64)), dtype=np.float64),
        0.0,
        1.0,
    )
    if len(hidden_vertices) != len(before) or len(sculpt_mask) != len(before):
        raise ValueError("protected Sculpt state does not match the component")
    root_start, root_fade = 0.29 * frame["length"], 0.13 * frame["length"]
    old_tip_fade = max(0.010 / 0.07595313195548941 * frame["length"], 1.0e-9)
    new_tip_fade = max(0.003 / 0.07595313195548941 * frame["length"], 1.0e-9)
    boundary_width = max(0.00070 / 0.07595313195548941 * frame["length"], 1.0e-9)
    guard_chunk = 128
    guard_total = int(math.ceil(len(station) / float(guard_chunk)))
    root_guard = np.empty(len(station), dtype=np.float64)
    old_tip_guard = np.empty(len(station), dtype=np.float64)
    new_tip_guard = np.empty(len(station), dtype=np.float64)
    for guard_start in range(0, len(station), guard_chunk):
        guard_stop = min(guard_start + guard_chunk, len(station))
        root_guard[guard_start:guard_stop] = [
            _guided_ridge_smoothstep((value - root_start) / root_fade)
            for value in station[guard_start:guard_stop]
        ]
        old_tip_guard[guard_start:guard_stop] = [
            _guided_ridge_smoothstep((frame["length"] - old_tip_fade - value) / old_tip_fade)
            for value in station[guard_start:guard_stop]
        ]
        new_tip_guard[guard_start:guard_stop] = [
            _guided_ridge_smoothstep((frame["length"] - new_tip_fade - value) / new_tip_fade)
            for value in station[guard_start:guard_stop]
        ]
        yield {
            "stage": "building profile guards",
            "stage_index": 5,
            "stage_count": 11,
            "done": int(guard_stop // max(guard_chunk, 1)),
            "total": guard_total,
        }
    if prepared_boundary and prepared_boundary[0] is not None:
        boundary_distance, boundary_distance_chunk_size, boundary_count = prepared_boundary[0]
    else:
        boundary_distance, boundary_distance_chunk_size, boundary_count = _guided_ridge_bounded_boundary_distance(
            before, boundary_mask
        )
    boundary_guard = np.empty(len(boundary_distance), dtype=np.float64)
    smooth_chunk = 64
    smooth_total = int(math.ceil(len(boundary_distance) / float(smooth_chunk)))
    for smooth_index, start in enumerate(range(0, len(boundary_distance), smooth_chunk), 1):
        stop = min(start + smooth_chunk, len(boundary_distance))
        boundary_guard[start:stop] = [
            _guided_ridge_smoothstep(value / boundary_width)
            for value in boundary_distance[start:stop]
        ]
        yield {
            "stage": "building boundary falloff",
            "stage_index": 5,
            "stage_count": 8,
            "done": smooth_index,
            "total": smooth_total,
        }
    # Protected vertices are applied to their own final displacement below.
    # Keeping them out of the smoothing loop prevents a local mask/hidden
    # value from changing the candidate at neighboring unmasked vertices.
    anchored = boundary_mask | (station <= 0.0) | ~crest_side_mask | anchor_mask
    planing_delta = target - h
    planing_delta[(planing_delta * outward_component) > 0.0] = 0.0
    planing_delta[outward_component == 0.0] = 0.0
    # Only vertices that were already above (outside) their target plane are
    # eligible.  Smoothing may not turn a depression, a plane-level vertex, or
    # a backside vertex into a new material-adding displacement.
    protrusion_mask = crest_side_mask & (planing_delta * outward_component < -1.0e-12)
    planing_delta[~protrusion_mask] = 0.0
    safe_low = np.minimum(planing_delta, 0.0)
    # Build the flattened adjacency arrays cooperatively.  The previous
    # repeat/concatenate pair ran before the first falloff yield and could
    # create an opaque native pause on dense components.  Chunking this
    # mechanical preparation preserves list order and values while returning
    # to the modal timer between bounded pieces.
    adjacency_source_parts = []
    adjacency_target_parts = []
    adjacency_counts = np.asarray(
        [len(neighbors) for neighbors in snapshot["adjacency"]], dtype=np.float64
    )
    adjacency_prepare_chunk = 256
    adjacency_prepare_total = int(
        math.ceil(len(snapshot["adjacency"]) / float(adjacency_prepare_chunk))
    )
    for adjacency_start in range(0, len(snapshot["adjacency"]), adjacency_prepare_chunk):
        adjacency_stop = min(adjacency_start + adjacency_prepare_chunk, len(snapshot["adjacency"]))
        local_neighbors = snapshot["adjacency"][adjacency_start:adjacency_stop]
        local_targets = [
            np.asarray(neighbors, dtype=np.int64) for neighbors in local_neighbors if neighbors
        ]
        if local_targets:
            adjacency_target_parts.append(np.concatenate(local_targets))
            adjacency_source_parts.append(
                np.repeat(
                    np.arange(adjacency_start, adjacency_stop, dtype=np.int64),
                    np.asarray([len(neighbors) for neighbors in local_neighbors], dtype=np.int64),
                )
            )
        yield {
            "stage": "preparing adjacency smoothing",
            "stage_index": 5,
            "stage_count": 11,
            "done": int(adjacency_stop // max(adjacency_prepare_chunk, 1)),
            "total": adjacency_prepare_total,
        }
    # Avoid one final full-size concatenate: on the live Hair component it
    # was the longest opaque allocation in the first falloff slice.  Group a
    # few already bounded pieces at a time and consume those groups directly
    # in ``apply`` below.
    grouped_sources = []
    grouped_targets = []
    adjacency_group_size = 8
    adjacency_group_total = int(
        math.ceil(len(adjacency_source_parts) / float(adjacency_group_size))
    )
    for group_start in range(0, len(adjacency_source_parts), adjacency_group_size):
        group_stop = min(group_start + adjacency_group_size, len(adjacency_source_parts))
        grouped_sources.append(np.concatenate(adjacency_source_parts[group_start:group_stop]))
        grouped_targets.append(np.concatenate(adjacency_target_parts[group_start:group_stop]))
        yield {
            "stage": "preparing adjacency smoothing",
            "stage_index": 5,
            "stage_count": 11,
            "done": len(grouped_sources),
            "total": adjacency_group_total,
        }
    adjacency_source_parts = grouped_sources
    adjacency_target_parts = grouped_targets
    adjacency_count = adjacency_counts
    def apply(weight, delta_source=None):
        delta = planing_delta * (0.82 * weight if delta_source is None else delta_source)
        delta[~protrusion_mask] = 0.0
        for _ in range(2):
            relaxed = delta.copy()
            neighbor_sum = np.zeros(len(delta), dtype=np.float64)
            for source_part, target_part in zip(adjacency_source_parts, adjacency_target_parts):
                np.add.at(neighbor_sum, source_part, delta[target_part])
            movable = (adjacency_count > 0.0) & ~anchored
            relaxed[movable] = (
                0.72 * delta[movable]
                + 0.28 * neighbor_sum[movable] / adjacency_count[movable]
            )
            relaxed[~protrusion_mask] = 0.0
            relaxed = np.clip(relaxed, safe_low, 0.0)
            relaxed[anchored] = 0.0
            delta = relaxed
        delta[(delta * outward_component) > 0.0] = 0.0
        delta[outward_component == 0.0] = 0.0
        delta[~protrusion_mask] = 0.0
        delta = np.clip(delta, safe_low, 0.0)
        return delta
    yield {"stage": "building planing delta", "stage_index": 6, "stage_count": 10, "done": 0, "total": 4}
    delta_current = apply(root_guard * old_tip_guard * boundary_guard)
    current = before + delta_current[:, None] * height_axis
    yield {"stage": "building planing delta", "stage_index": 6, "stage_count": 10, "done": 1, "total": 4}
    active = (0.82 * root_guard * old_tip_guard * boundary_guard) > 0.05
    bins = np.linspace(0.0, frame["length"], 32 + 1)
    raw = np.full(32, np.nan, dtype=float)
    abs_delta = np.abs(delta_current)
    for index in range(32):
        mask = active & (station >= bins[index]) & (station < bins[index + 1])
        if np.any(mask):
            raw[index] = float(np.quantile(abs_delta[mask], 0.90))
    finite = np.isfinite(raw)
    center = float(np.median(raw[finite])) if np.any(finite) else 0.0
    filled = np.where(finite, raw, center)
    smooth_profile = _guided_ridge_smooth_array(filled, 1.8)
    scale_profile = np.clip((0.85 * smooth_profile + 0.15 * center) / np.maximum(filled, 1.0e-12), 0.70, 1.35)
    scale = np.interp(station, 0.5 * (bins[:-1] + bins[1:]), scale_profile, left=1.0, right=1.0)
    scale[~active] = 1.0
    variant_a = before + (delta_current * scale)[:, None] * height_axis
    current_new = before + apply(root_guard * new_tip_guard * boundary_guard)[:, None] * height_axis
    yield {"stage": "building planing delta", "stage_index": 6, "stage_count": 10, "done": 2, "total": 4}
    candidate = variant_a + 0.25 * (current_new - current) * scale[:, None]
    yield {"stage": "applying masks and planing clamp", "stage_index": 7, "stage_count": 10, "done": 0, "total": 2}
    protected_weight = np.clip(1.0 - sculpt_mask, 0.0, 1.0)
    protected_weight[hidden_vertices] = 0.0
    protected_weight[boundary_mask] = 0.0
    # Apply protection locally to each vertex's final displacement. The
    # unmasked/visible candidate stays unchanged when another vertex is
    # masked, hidden, or a boundary anchor.
    candidate_delta = candidate - before
    candidate_delta *= protected_weight[:, None]
    affected = (sculpt_mask > 1.0e-12) | hidden_vertices | boundary_mask | ~protrusion_mask
    if np.any(affected):
        final_delta_h = np.sum(candidate_delta * height_axis, axis=1)
        final_bound = np.abs(target - h) * 0.82 * root_guard * new_tip_guard * boundary_guard * protected_weight
        final_delta_h[affected] = np.clip(
            final_delta_h[affected], -final_bound[affected], final_bound[affected]
        )
        candidate_delta[affected] = final_delta_h[affected, None] * height_axis[affected]
    final_delta_h = np.sum(candidate_delta * height_axis, axis=1)
    final_delta_h[~protrusion_mask] = 0.0
    final_delta_h = np.clip(final_delta_h, safe_low, 0.0)
    final_delta_h[(final_delta_h * outward_component) > 0.0] = 0.0
    candidate_delta = final_delta_h[:, None] * height_axis
    candidate = before + candidate_delta
    yield {"stage": "applying masks and planing clamp", "stage_index": 7, "stage_count": 10, "done": 1, "total": 2}
    # Keep the established no-flip integrity contract while allowing a
    # valid surface section to produce a smaller, safe planing cut on dense
    # or sharply triangulated components.  Scaling only the already selected
    # crest-side, inward-only delta cannot move the guide, anchors, boundary,
    # or backside and does not introduce material.
    selected_scale = None
    integrity_scales = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    for scale_index, scale_factor in enumerate(integrity_scales, 1):
        trial = before + candidate_delta * scale_factor
        integrity = _guided_ridge_mesh_integrity(snapshot, before, trial)
        if integrity["finite"] and not integrity["normal_pair_flips"] and not integrity["new_degenerate_faces"]:
            candidate = trial
            selected_scale = float(scale_factor)
            break
        yield {
            "stage": "checking planing integrity",
            "stage_index": 8,
            "stage_count": 10,
            "done": scale_index,
            "total": len(integrity_scales),
        }
    if selected_scale is None:
        raise ValueError("surface planing candidate fails mesh integrity")
    return candidate, {
        "length": frame["length"],
        "station": station,
        "boundary_mask": boundary_mask,
        "crest_side_mask": crest_side_mask,
        "protrusion_mask": protrusion_mask,
        "anchor_mask": anchor_mask,
        "section_anchor_mask": selected_section_anchor_mask,
        "guide_anchor_mask": guide_anchor_mask,
        "selected_arc_mask": selected_arc_mask,
        "outward_component": outward_component,
        "section_invalid_reasons": invalid_reasons,
        "weight": 0.82 * root_guard * new_tip_guard * boundary_guard * protected_weight,
        "delta_h": np.sum((candidate - before) * height_axis, axis=1),
        "integrity_scale": selected_scale,
        "boundary_distance_chunk_size": int(boundary_distance_chunk_size),
        "boundary_count": int(boundary_count),
        "vertex_guide_chunk_size": int(chunk_size),
    }


def _guided_ridge_arc_candidate_legacy(snapshot, guide, controls, control_normals, prepared_sections=None):
    """Synchronous compatibility wrapper for Repeat Last and tests."""
    job = _guided_ridge_arc_candidate_job_legacy(
        snapshot, guide, controls, control_normals, prepared_sections=prepared_sections
    )
    try:
        while True:
            next(job)
    except StopIteration as complete:
        return complete.value


def _guided_ridge_width_limits(snapshot):
    """Return density-aware, component-safe initial/min/max half widths."""
    import numpy as np

    points = np.asarray(snapshot.get("coords_world", ()), dtype=np.float64)
    average_edge = max(float(snapshot.get("average_edge", 0.0) or 0.0), 1.0e-7)
    if len(points):
        extent = float(np.linalg.norm(np.ptp(points, axis=0)))
    else:
        extent = average_edge
    minimum = max(average_edge * GUIDED_RIDGE_WIDTH_MIN_EDGE_MULTIPLIER, 1.0e-7)
    maximum = max(minimum, extent * GUIDED_RIDGE_WIDTH_MAX_EXTENT_FRACTION)
    initial = min(max(average_edge * GUIDED_RIDGE_WIDTH_EDGE_MULTIPLIER, minimum), maximum)
    return float(initial), float(minimum), float(maximum), float(average_edge)


def _guided_ridge_align_frame_to_controls(frame, controls, control_normals):
    """Use the clicked outward normals to orient the planing hemisphere."""
    import numpy as np

    controls_array = np.asarray(controls, dtype=np.float64)
    normals_array = _guided_ridge_unit_array(np.asarray(control_normals, dtype=np.float64))
    if len(controls_array) == 0 or len(normals_array) == 0:
        return frame
    for index, point in enumerate(frame["guide"]):
        control_index = int(np.argmin(np.linalg.norm(controls_array - point[None, :], axis=1)))
        if float(np.dot(frame["normal"][index], normals_array[control_index])) < 0.0:
            frame["normal"][index] *= -1.0
    frame["normal"] = _guided_ridge_unit_array(frame["normal"])
    frame["lateral"] = _guided_ridge_unit_array(np.cross(frame["tangent"], frame["normal"]))
    frame["normal"] = _guided_ridge_unit_array(np.cross(frame["lateral"], frame["tangent"]))
    return frame


def _guided_ridge_finalize_width_frame(snapshot, frame, controls, control_normals):
    """Align a generated frame once before any layer/provenance work."""
    if frame is None:
        raise ValueError("guide frame is unavailable")
    frame = _guided_ridge_align_frame_to_controls(frame, controls, control_normals)
    frame["average_edge"] = float(snapshot.get("average_edge", 0.0) or 0.0)
    return frame


def _guided_ridge_width_frame(snapshot, guide, controls, control_normals):
    """Build the stable guide frame used by both width rails and candidate."""
    import numpy as np

    before = np.asarray(snapshot["coords_world"], dtype=np.float64)
    faces = np.asarray(snapshot["triangles"], dtype=np.int64)
    frame = _guided_ridge_stable_frame(before, faces, guide, controls, control_normals)
    return _guided_ridge_finalize_width_frame(
        snapshot, frame, controls, control_normals
    )


def _guided_ridge_width_cache_key(guide, controls, control_normals):
    """Stable-frame identity; half-width is deliberately not part of it."""
    import numpy as np

    def flatten(values):
        array = np.asarray(values, dtype=np.float64)
        return tuple(float(value) for value in array.reshape(-1))

    return (flatten(guide), flatten(controls), flatten(control_normals))


def _guided_ridge_surface_layer_steps(snapshot, frame):
    """Build station/side surface provenance once per guide-frame rebuild.

    This cache is intentionally independent of the requested width.  Wheel
    events can then query only the already selected local layer and never
    rebuild adjacency/BVH data or fall back to the component-global BVH.
    """
    import numpy as np

    points = np.asarray(snapshot.get("coords_world", ()), dtype=np.float64)
    triangles = np.asarray(snapshot.get("triangles", ()), dtype=np.int64)
    adjacency = snapshot.get("adjacency")
    normals_world = np.asarray(snapshot.get("normals_world", ()), dtype=np.float64)
    average_edge = max(float(snapshot.get("average_edge", 0.0) or 0.0), 1.0e-7)
    station_count = len(frame.get("guide", ()))
    empty = {"left": [None] * station_count, "right": [None] * station_count}
    reasons = {"left": [None] * station_count, "right": [None] * station_count}
    yield {"stage": "building width surface layers", "stage_index": 10, "stage_count": 12, "done": 0, "total": station_count * 2}
    if not adjacency or len(adjacency) != len(points):
        return {"entries": empty, "reasons": reasons, "available": False, "reason": "surface adjacency is unavailable"}
    try:
        from mathutils.bvhtree import BVHTree
    except (ImportError, RuntimeError):
        return {"entries": empty, "reasons": reasons, "available": False, "reason": "surface BVH is unavailable"}
    entries = {"left": [], "right": []}
    processed = 0
    station_radius = average_edge * 5.0
    for side, sign in (("left", -1.0), ("right", 1.0)):
        for station_index, (origin, tangent, lateral, normal) in enumerate(
            zip(frame["guide"], frame["tangent"], frame["lateral"], frame["normal"])
        ):
            offset = points - np.asarray(origin, dtype=np.float64)[None, :]
            along = np.dot(offset, np.asarray(tangent, dtype=np.float64))
            q_values = np.dot(offset, np.asarray(lateral, dtype=np.float64))
            h_values = np.dot(offset, np.asarray(normal, dtype=np.float64))
            if normals_world.shape == points.shape:
                alignment = np.dot(normals_world, np.asarray(normal, dtype=np.float64))
            else:
                alignment = np.ones(len(points), dtype=np.float64)
            local_mask = (
                np.isfinite(along) & np.isfinite(q_values) & np.isfinite(h_values)
                & (np.abs(along) <= station_radius)
                & (sign * q_values > max(average_edge * 0.05, 1.0e-7))
                & (alignment > -0.25)
            )
            local_indices = np.flatnonzero(local_mask)
            entry = None
            if len(local_indices) >= 3:
                local_set = set(int(index) for index in local_indices)
                components = []
                unseen = set(local_set)
                while unseen:
                    seed = unseen.pop()
                    component = {seed}
                    queue = [seed]
                    pop_count = 0
                    while queue:
                        current = queue.pop()
                        pop_count += 1
                        if pop_count % 256 == 0:
                            yield {
                                "stage": "building width surface layers",
                                "stage_index": 10,
                                "stage_count": 12,
                                "done": processed,
                                "total": station_count * 2,
                                "indeterminate": True,
                            }
                        try:
                            neighbors = adjacency[current]
                        except (IndexError, TypeError):
                            neighbors = ()
                        for neighbor in neighbors:
                            neighbor = int(neighbor)
                            if neighbor in unseen:
                                unseen.remove(neighbor)
                                component.add(neighbor)
                                queue.append(neighbor)
                    if len(component) >= 3:
                        values = np.asarray(sorted(component), dtype=np.int64)
                        radial = np.sqrt(q_values[values] ** 2 + (0.5 * h_values[values]) ** 2)
                        components.append((float(np.min(radial)), component))
                if components:
                    components.sort(key=lambda item: item[0])
                    if not (
                        len(components) > 1
                        and abs(components[1][0] - components[0][0]) <= max(average_edge * 0.25, 1.0e-7)
                    ):
                        chosen = set(components[0][1])
                        bridge_epsilon = max(average_edge * 0.25, 1.0e-7)
                        for index in range(len(points)):
                            if abs(float(q_values[index])) > bridge_epsilon:
                                continue
                            try:
                                if any(int(neighbor) in chosen for neighbor in adjacency[index]):
                                    chosen.add(int(index))
                            except (IndexError, TypeError):
                                continue
                        layer_vertex_list = list(sorted(chosen))
                        local_vertex_map = {value: index for index, value in enumerate(layer_vertex_list)}
                        layer_triangle_map = []
                        local_triangles = []
                        for triangle_index, triangle in enumerate(triangles):
                            triangle_values = tuple(int(value) for value in triangle)
                            if all(value in local_vertex_map for value in triangle_values):
                                local_triangles.append(tuple(local_vertex_map[value] for value in triangle_values))
                                layer_triangle_map.append(int(triangle_index))
                            if triangle_index % 256 == 255:
                                yield {
                                    "stage": "building width surface layers",
                                    "stage_index": 10,
                                    "stage_count": 12,
                                    "done": processed,
                                    "total": station_count * 2,
                                    "indeterminate": True,
                                }
                        if local_triangles:
                            try:
                                layer_bvh = BVHTree.FromPolygons(
                                    points[np.asarray(layer_vertex_list, dtype=np.int64)].tolist(),
                                    local_triangles,
                                )
                                entry = {
                                    "vertices": frozenset(layer_vertex_list),
                                    "bvh": layer_bvh,
                                    "triangle_map": tuple(layer_triangle_map),
                                }
                            except (AttributeError, IndexError, MemoryError, RuntimeError, TypeError, ValueError):
                                entry = None
                    elif len(components) > 1:
                        reasons[side][station_index] = "ambiguous local surface layer"
                elif len(local_indices) >= 3:
                    reasons[side][station_index] = "local surface layer is unavailable"
            else:
                reasons[side][station_index] = "local surface layer is unavailable"
            entries[side].append(entry)
            processed += 1
            yield {
                "stage": "building width surface layers",
                "stage_index": 10,
                "stage_count": 12,
                "done": processed,
                "total": station_count * 2,
            }
    return {"entries": entries, "reasons": reasons, "available": True, "reason": None}


def _guided_ridge_width_prepare_steps(
    snapshot, frame_job, controls=(), control_normals=()
):
    """Cooperatively finish a frame and build its width-independent layers."""
    try:
        while True:
            yield next(frame_job)
    except StopIteration as complete:
        frame = complete.value
    frame = _guided_ridge_finalize_width_frame(
        snapshot, frame, controls, control_normals
    )
    layer_job = _guided_ridge_surface_layer_steps(snapshot, frame)
    try:
        while True:
            yield next(layer_job)
    except StopIteration as complete:
        return frame, complete.value


def _guided_ridge_width_rails(snapshot, frame, half_width, layer_result=None):
    """Project both width rails to the captured surface; gaps remain explicit."""
    import numpy as np

    rails = {"left": [], "right": []}
    # Keep the complete hit provenance beside the display points.  The
    # candidate and the preview must share the same projected triangle/edge,
    # rather than independently choosing a nearest vertex later.
    rail_hits = {"left": [], "right": []}
    layer_bvh_cache = {}
    layer_entries = (layer_result or {}).get("entries", {}) if isinstance(layer_result, dict) else None
    bvh = snapshot.get("bvh")
    points = np.asarray(snapshot.get("coords_world", ()), dtype=np.float64)
    triangles = np.asarray(snapshot.get("triangles", ()), dtype=np.int64)
    average_edge = max(float(snapshot.get("average_edge", 0.0) or 0.0), 1.0e-7)
    max_projection_distance = max(average_edge * 8.0, float(half_width) * 0.5)

    def _local_layer_vertices(origin, tangent, lateral, normal, sign):
        """Choose the locally reachable surface layer for one station/side.

        The captured Face Set can contain nearby sheets that only connect
        somewhere far away.  Restrict the adjacency walk to a short station
        neighborhood and select the component closest to the guide; the
        global BVH is used only after this local topological filter.
        """
        adjacency = snapshot.get("adjacency")
        if not adjacency or len(adjacency) != len(points):
            return None
        offset = points - np.asarray(origin, dtype=np.float64)[None, :]
        along = np.dot(offset, np.asarray(tangent, dtype=np.float64))
        q_values = np.dot(offset, np.asarray(lateral, dtype=np.float64))
        h_values = np.dot(offset, np.asarray(normal, dtype=np.float64))
        normals_world = np.asarray(snapshot.get("normals_world", ()), dtype=np.float64)
        if normals_world.shape == points.shape:
            alignment = np.dot(normals_world, np.asarray(normal, dtype=np.float64))
        else:
            alignment = np.ones(len(points), dtype=np.float64)
        station_radius = average_edge * 5.0
        local_mask = (
            np.isfinite(along) & np.isfinite(q_values) & np.isfinite(h_values)
            & (np.abs(along) <= station_radius)
            & (sign * q_values > max(average_edge * 0.05, 1.0e-7))
            & (alignment > -0.25)
        )
        local_indices = np.flatnonzero(local_mask)
        if len(local_indices) < 3:
            return None
        local_set = set(int(index) for index in local_indices)
        components = []
        unseen = set(local_set)
        while unseen:
            seed = unseen.pop()
            component = {seed}
            queue = [seed]
            while queue:
                current = queue.pop()
                try:
                    neighbors = adjacency[current]
                except (IndexError, TypeError):
                    neighbors = ()
                for neighbor in neighbors:
                    neighbor = int(neighbor)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            if len(component) >= 3:
                values = np.asarray(sorted(component), dtype=np.int64)
                radial = np.sqrt(q_values[values] ** 2 + (0.5 * h_values[values]) ** 2)
                components.append((float(np.min(radial)), component))
        if not components:
            return None
        components.sort(key=lambda item: item[0])
        if len(components) > 1 and abs(components[1][0] - components[0][0]) <= max(average_edge * 0.25, 1.0e-7):
            # Two equally close local layers are ambiguous; never choose a
            # nominal-width hit merely because the other sheet is nearby.
            return None
        chosen = set(components[0][1])
        # Surface triangles at the crest can contain one q≈0 connector
        # vertex.  Admit only connectors directly adjacent to the selected
        # side, never an unrelated nearby sheet's crest vertex.
        bridge_epsilon = max(average_edge * 0.25, 1.0e-7)
        for index in range(len(points)):
            index = int(index)
            if abs(float(q_values[index])) > bridge_epsilon:
                continue
            try:
                if any(int(neighbor) in chosen for neighbor in adjacency[index]):
                    chosen.add(index)
            except (IndexError, TypeError):
                continue
        return chosen

    def _nearest_surface_rail(origin, tangent, lateral, normal, sign, precomputed_layer=None):
        """Find the closest same-side rail, tapering when the surface ends."""
        if bvh is None:
            return None, None, None, None
        if precomputed_layer is not None:
            layer_vertices = precomputed_layer.get("vertices")
            query_bvh = precomputed_layer.get("bvh")
            query_triangle_map = precomputed_layer.get("triangle_map")
            if not layer_vertices or query_bvh is None:
                return None, None, None, None
        else:
            layer_vertices = _local_layer_vertices(origin, tangent, lateral, normal, sign)
            # A real BVH must never fall back to a global nearest hit when the
            # local layer is unavailable or ambiguous.  Keep this station as a
            # visible gap; coverage checks decide whether the candidate can apply.
            if layer_vertices is None:
                return None, None, None, None
            query_bvh = None
            query_triangle_map = None
            # Do not ask the component-global BVH for a nominal-width hit and
            # discard it afterwards: that can hide the valid local layer when
            # a nearby sheet is closer.  Build a compact BVH from only the
            # locally connected layer and retain a map back to source faces.
            layer_key = tuple(sorted(int(value) for value in layer_vertices))
            cached_layer = layer_bvh_cache.get(layer_key)
            if cached_layer is None:
                layer_vertex_list = list(layer_key)
                local_vertex_map = {value: index for index, value in enumerate(layer_vertex_list)}
                layer_triangle_map = []
                local_triangles = []
                for triangle_index, triangle in enumerate(triangles):
                    triangle_values = tuple(int(value) for value in triangle)
                    if all(value in local_vertex_map for value in triangle_values):
                        local_triangles.append(tuple(local_vertex_map[value] for value in triangle_values))
                        layer_triangle_map.append(int(triangle_index))
                if local_triangles:
                    try:
                        from mathutils.bvhtree import BVHTree
                        query_bvh = BVHTree.FromPolygons(
                            points[np.asarray(layer_vertex_list, dtype=np.int64)].tolist(),
                            local_triangles,
                        )
                    except (AttributeError, IndexError, MemoryError, RuntimeError, TypeError, ValueError):
                        query_bvh = None
                else:
                    query_bvh = None
                cached_layer = (query_bvh, tuple(layer_triangle_map))
                layer_bvh_cache[layer_key] = cached_layer
            query_bvh, query_triangle_map = cached_layer
            if query_bvh is None:
                return None, None, None, None
        best = None
        # Query the requested maximum first, then move toward the crest.  A
        # tip that is narrower than the requested width therefore keeps its
        # actual support surface instead of jumping across the gap to a
        # nearby sheet.  The score prefers the widest valid same-side hit.
        for scale in (1.0, 0.92, 0.84, 0.76, 0.68, 0.60, 0.52, 0.44, 0.36, 0.28, 0.20, 0.12):
            requested = float(half_width) * float(scale)
            target = Vector(tuple(origin + sign * requested * lateral))
            try:
                nearest = query_bvh.find_nearest(target)
                if not nearest or nearest[0] is None or nearest[3] is None:
                    continue
                distance = float(nearest[3])
                if not math.isfinite(distance) or distance > max_projection_distance:
                    continue
                triangle_index = int(nearest[2]) if nearest[2] is not None else -1
                if query_triangle_map is not None:
                    if triangle_index < 0 or triangle_index >= len(query_triangle_map):
                        continue
                    triangle_index = int(query_triangle_map[triangle_index])
                location = np.asarray(nearest[0], dtype=np.float64)
                actual_q = float(np.dot(location - np.asarray(origin, dtype=np.float64), lateral))
                if not math.isfinite(actual_q) or sign * actual_q <= max(average_edge * 0.05, 1.0e-7):
                    continue
                # Overshooting the requested maximum is strongly penalized;
                # otherwise choose the widest available support on this side.
                overshoot = max(0.0, sign * actual_q - float(half_width))
                undershoot = max(0.0, float(half_width) - sign * actual_q)
                score = overshoot * 1000.0 + undershoot + distance * 0.01
                candidate = (score, nearest, target, actual_q, triangle_index)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                continue
        if best is None:
            return None, None, None, None
        return best[1], best[2], best[3], best[4]

    for side, sign in (("left", -1.0), ("right", 1.0)):
        for station_index, (origin, lateral) in enumerate(zip(frame["guide"], frame["lateral"])):
            target = Vector(tuple(origin + sign * float(half_width) * lateral))
            try:
                precomputed_layer = None
                if layer_entries is not None:
                    side_entries = layer_entries.get(side, ())
                    precomputed_layer = side_entries[station_index] if station_index < len(side_entries) else None
                if layer_entries is not None and precomputed_layer is None:
                    # ``None`` in a prepared layer table is an explicit local
                    # gap/invalid station.  Do not reinterpret it as a cold
                    # cache and rebuild BFS/triangles/BVH during Wheel.
                    nearest, projected_target, _actual_q, source_triangle_index = (
                        None, None, None, None
                    )
                else:
                    nearest, projected_target, _actual_q, source_triangle_index = _nearest_surface_rail(
                        origin,
                        frame["tangent"][station_index],
                        lateral,
                        frame["normal"][station_index],
                        sign,
                        precomputed_layer=precomputed_layer if layer_entries is not None else None,
                    )
                if projected_target is not None:
                    target = projected_target
                if nearest and nearest[0] is not None and float(nearest[3] or 0.0) <= max_projection_distance:
                    location = Vector(nearest[0])
                    rails[side].append(location)
                    triangle_index = int(source_triangle_index) if source_triangle_index is not None else -1
                    triangle_vertices = ()
                    barycentric = ()
                    support_vertices = ()
                    edge_support_vertices = ()
                    edge_index = None
                    if 0 <= triangle_index < len(triangles):
                        triangle_vertices = tuple(int(value) for value in triangles[triangle_index])
                        tri_points = points[np.asarray(triangle_vertices, dtype=np.int64)]
                        edge_a = tri_points[1] - tri_points[0]
                        edge_b = tri_points[2] - tri_points[0]
                        offset = np.asarray(location, dtype=np.float64) - tri_points[0]
                        d00 = float(np.dot(edge_a, edge_a))
                        d01 = float(np.dot(edge_a, edge_b))
                        d11 = float(np.dot(edge_b, edge_b))
                        d20 = float(np.dot(offset, edge_a))
                        d21 = float(np.dot(offset, edge_b))
                        denominator = d00 * d11 - d01 * d01
                        if denominator > 1.0e-20:
                            bary = (d11 * d20 - d01 * d21) / denominator
                            bary_c = (d00 * d21 - d01 * d20) / denominator
                            bary_a = 1.0 - bary - bary_c
                            barycentric = tuple(float(value) for value in (bary_a, bary, bary_c))
                            weight_epsilon = 1.0e-6
                            support_vertices = tuple(
                                triangle_vertices[index]
                                for index, value in enumerate(barycentric)
                                if value > weight_epsilon
                            )
                            if len(support_vertices) >= 2:
                                edge_support_vertices = support_vertices
                            elif support_vertices:
                                edge_support_vertices = support_vertices
                            zero_weights = [
                                index for index, value in enumerate(barycentric)
                                if abs(value) <= weight_epsilon
                            ]
                            if len(zero_weights) == 1:
                                edge_index = int(zero_weights[0])
                    rail_hits[side].append(
                        {
                            "location": location.copy(),
                            "triangle_index": triangle_index,
                            "triangle_vertices": triangle_vertices,
                            "barycentric_weights": barycentric,
                            "edge_index": edge_index,
                            "support_vertex_indices": support_vertices,
                            "edge_support_vertex_indices": edge_support_vertices,
                        }
                    )
                elif bvh is None and len(points):
                    distance = np.linalg.norm(points - np.asarray(target, dtype=np.float64)[None, :], axis=1)
                    index = int(np.argmin(distance))
                    if float(distance[index]) <= max_projection_distance:
                        location = Vector(tuple(points[index]))
                        rails[side].append(location)
                        rail_hits[side].append(
                            {
                                "location": location.copy(),
                                "triangle_index": -1,
                                "triangle_vertices": (int(index),),
                                "barycentric_weights": (1.0,),
                                "edge_index": None,
                                "support_vertex_indices": (int(index),),
                                "edge_support_vertex_indices": (int(index),),
                            }
                        )
                    else:
                        rails[side].append(None)
                        rail_hits[side].append(None)
                else:
                    rails[side].append(None)
                    rail_hits[side].append(None)
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                rails[side].append(None)
                rail_hits[side].append(None)
    rails["hits"] = rail_hits
    return rails


def _guided_ridge_width_anchor_data(snapshot, frame, width, rails):
    """Derive anchor heights/support vertices once from the shared rails."""
    import numpy as np

    points = np.asarray(snapshot.get("coords_world", ()), dtype=np.float64)
    triangles = np.asarray(snapshot.get("triangles", ()), dtype=np.int64)
    average_edge = max(float(snapshot.get("average_edge", 0.0) or 0.0), 1.0e-7)
    support_indices = {"left": [], "right": []}
    support_vertex_indices = {"left": [], "right": []}
    anchor_heights = {
        "left": np.full(len(frame["guide"]), np.nan, dtype=np.float64),
        "right": np.full(len(frame["guide"]), np.nan, dtype=np.float64),
    }
    anchor_lateral = {
        "left": np.full(len(frame["guide"]), np.nan, dtype=np.float64),
        "right": np.full(len(frame["guide"]), np.nan, dtype=np.float64),
    }
    support_mask = np.zeros(len(points), dtype=bool)
    rail_hits = rails.get("hits", {}) if isinstance(rails, dict) else {}
    rail_hit_ok = True
    for side in ("left", "right"):
        for index, point in enumerate(rails.get(side, ())):
            if point is None:
                support_indices[side].append(None)
                support_vertex_indices[side].append(())
                continue
            point_array = np.asarray(point, dtype=np.float64)
            hit = rail_hits.get(side, ())[index] if index < len(rail_hits.get(side, ())) else None
            if snapshot.get("bvh") is not None and (
                not hit
                or int((hit or {}).get("triangle_index", -1)) < 0
                or not (hit or {}).get("barycentric_weights")
                or not (hit or {}).get("support_vertex_indices")
            ):
                rail_hit_ok = False
            support_values = tuple(int(value) for value in (hit or {}).get("support_vertex_indices", ()))
            support_vertex_indices[side].append(support_values)
            for support_index in support_values:
                if 0 <= support_index < len(support_mask):
                    support_mask[support_index] = True
            # Preserve the old one-index summary for compatibility, while the
            # actual fixed mask uses every triangle/edge support vertex above.
            support_indices[side].append(support_values[0] if support_values else None)
            origin = np.asarray(frame["guide"][index], dtype=np.float64)
            normal = np.asarray(frame["normal"][index], dtype=np.float64)
            lateral = np.asarray(frame["lateral"][index], dtype=np.float64)
            anchor_heights[side][index] = float(np.dot(point_array - origin, normal))
            anchor_lateral[side][index] = float(np.dot(point_array - origin, lateral))
    projection_ok = all(
        point is not None and math.isfinite(float(anchor_heights[side][index]))
        for side in ("left", "right")
        for index, point in enumerate(rails.get(side, ()))
    )
    if not projection_ok:
        for side in ("left", "right"):
            values = anchor_heights[side]
            valid = np.flatnonzero(np.isfinite(values))
            if len(valid) >= 2:
                values[:] = np.interp(np.arange(len(values)), valid, values[valid])
    return {
        "rail_supports": support_indices,
        "rail_support_vertex_indices": support_vertex_indices,
        "rail_hits": rail_hits,
        "rail_anchor_height": anchor_heights,
        "rail_anchor_lateral": anchor_lateral,
        "anchor_vertex_indices": np.flatnonzero(support_mask).astype(np.int64),
        "anchor_projection_ok": bool(projection_ok),
        "rail_hit_ok": bool(rail_hit_ok),
    }


def _guided_ridge_width_frame_result(
    snapshot, guide, controls, control_normals, half_width, frame=None, layer_result=None
):
    """Shared frame/width/rail result used by preview, apply and Repeat Last."""
    initial, minimum, maximum, _average_edge = _guided_ridge_width_limits(snapshot)
    width = initial if half_width is None else float(half_width)
    if not math.isfinite(width):
        raise ValueError("planing width is not finite")
    width = min(max(width, minimum), maximum)
    if frame is None:
        frame = _guided_ridge_width_frame(snapshot, guide, controls, control_normals)
    if layer_result is None and snapshot.get("bvh") is not None:
        layer_job = _guided_ridge_surface_layer_steps(snapshot, frame)
        try:
            while True:
                next(layer_job)
        except StopIteration as complete:
            layer_result = complete.value
    rails = _guided_ridge_width_rails(snapshot, frame, width, layer_result=layer_result)
    anchor_data = _guided_ridge_width_anchor_data(snapshot, frame, width, rails)
    bvh = snapshot.get("bvh")
    projection_required = bvh is not None
    projection_ok = (
        not projection_required
        or all(point is not None for side in ("left", "right") for point in rails[side])
    )
    return {
        "frame": frame,
        "width": float(width),
        "width_min": float(minimum),
        "width_max": float(maximum),
        "rails": rails,
        "layer_result": layer_result,
        "projection_required": bool(projection_required),
        "projection_ok": bool(projection_ok),
        "anchor_projection_ok": bool(anchor_data["anchor_projection_ok"]),
        "rail_hit_ok": bool(anchor_data["rail_hit_ok"]),
        "rail_supports": anchor_data["rail_supports"],
        "rail_support_vertex_indices": anchor_data["rail_support_vertex_indices"],
        "rail_hits": anchor_data["rail_hits"],
        "rail_anchor_height": anchor_data["rail_anchor_height"],
        "rail_anchor_lateral": anchor_data["rail_anchor_lateral"],
        "anchor_vertex_indices": anchor_data["anchor_vertex_indices"],
        "anchor_band": min(
            max(width * GUIDED_RIDGE_WIDTH_ANCHOR_BAND_FRACTION, _average_edge * 0.5),
            max(width - min(max(width * GUIDED_RIDGE_WIDTH_GUIDE_BAND_FRACTION, _average_edge * 0.25), width * 0.22) - width * 0.05, width * 0.05),
        ),
        "guide_band": min(
            max(width * GUIDED_RIDGE_WIDTH_GUIDE_BAND_FRACTION, _average_edge * 0.25),
            width * 0.22,
        ),
        "cache_key": _guided_ridge_width_cache_key(guide, controls, control_normals),
    }


def _guided_ridge_interpolate_vertex_frame(frame, seg_index, seg_t):
    """Interpolate frame vectors on the captured guide segment parameter."""
    import numpy as np

    i0 = np.asarray(seg_index, dtype=np.int64)
    i1 = np.minimum(i0 + 1, len(frame["arc"]) - 1)
    blend = np.asarray(seg_t, dtype=np.float64)
    tangent = _guided_ridge_unit_array(
        (1.0 - blend)[:, None] * frame["tangent"][i0]
        + blend[:, None] * frame["tangent"][i1]
    )
    lateral = _guided_ridge_unit_array(
        (1.0 - blend)[:, None] * frame["lateral"][i0]
        + blend[:, None] * frame["lateral"][i1]
    )
    height_axis = _guided_ridge_unit_array(
        (1.0 - blend)[:, None] * frame["normal"][i0]
        + blend[:, None] * frame["normal"][i1]
    )
    lateral -= np.sum(lateral * tangent, axis=1, keepdims=True) * tangent
    lateral = _guided_ridge_unit_array(lateral)
    height_axis -= np.sum(height_axis * tangent, axis=1, keepdims=True) * tangent
    height_axis = _guided_ridge_unit_array(height_axis)
    return tangent, lateral, height_axis


def _guided_ridge_update_width_preview(state):
    """Refresh width limits and projected rails without touching mesh data."""
    snapshot = state.get("snapshot") if state else None
    guide = state.get("guide") if state else None
    if snapshot is None or "coords_world" not in snapshot or "triangles" not in snapshot:
        return False
    initial, minimum, maximum, _average_edge = _guided_ridge_width_limits(snapshot)
    width = float(state.get("half_width", initial) or initial)
    state["width_min"] = minimum
    state["width_max"] = maximum
    state["half_width"] = min(max(width, minimum), maximum)
    if not guide or len(guide) < 2:
        state["width_frame"] = None
        state["width_rails"] = {"left": [], "right": []}
        state["width_frame_result"] = None
        return True
    controls = state.get("controls", ())
    control_normals = state.get("control_normals", ())
    cache_key = _guided_ridge_width_cache_key(guide, controls, control_normals)
    result = state.get("width_frame_result")
    if result is None or result.get("cache_key") != cache_key:
        result = _guided_ridge_width_frame_result(
            snapshot, guide, controls, control_normals, state["half_width"]
        )
    else:
        result = dict(result)
        result["width"] = float(state["half_width"])
        if result.get("layer_result") is None and snapshot.get("bvh") is not None:
            layer_job = _guided_ridge_surface_layer_steps(snapshot, result["frame"])
            try:
                while True:
                    next(layer_job)
            except StopIteration as complete:
                result["layer_result"] = complete.value
        result["rails"] = _guided_ridge_width_rails(
            snapshot,
            result["frame"],
            result["width"],
            layer_result=result.get("layer_result"),
        )
        result.update(
            _guided_ridge_width_anchor_data(
                snapshot, result["frame"], result["width"], result["rails"]
            )
        )
        bvh = snapshot.get("bvh")
        result["projection_ok"] = (
            bvh is None
            or all(point is not None for side in ("left", "right") for point in result["rails"][side])
        )
    state["half_width"] = result["width"]
    state["width_frame_result"] = result
    state["width_frame"] = result["frame"]
    state["width_rails"] = result["rails"]
    state["width_cache_serial"] = int(state.get("width_cache_serial", 0)) + 1
    return True


def _guided_ridge_begin_width_frame_rebuild(context, state):
    """Start a cooperative frame rebuild after a guide edit.

    Width-only edits intentionally stay on the cached frame path above.  A
    changed guide, however, invalidates every frame vector and must not spend
    an unbounded synchronous interval inside the LMB handler.
    """
    if state is None or state.get("phase") != "ready":
        return False
    snapshot = state.get("snapshot") or {}
    guide = state.get("guide") or ()
    if len(guide) < 2:
        return _guided_ridge_update_width_preview(state)
    try:
        import numpy as np

        before = np.asarray(snapshot["coords_world"], dtype=np.float64)
        faces = np.asarray(snapshot["triangles"], dtype=np.int64)
        state["phase"] = "width_prepare"
        stable_frame_job = _guided_ridge_stable_frame_steps(
            before, faces, guide, state.get("controls", ()), state.get("control_normals", ())
        )
        state["width_frame_job"] = _guided_ridge_width_prepare_steps(
            snapshot,
            stable_frame_job,
            state.get("controls", ()),
            state.get("control_normals", ()),
        )
        state["width_frame_stage"] = "queued"
        state["width_frame_stage_index"] = 0
        state["width_frame_stage_count"] = 12
        state["width_frame_stage_done"] = 0
        state["width_frame_stage_total"] = 0
        state["width_frame_progress_fraction"] = None
        state["width_frame_progress_indeterminate"] = True
        state["width_frame_elapsed_seconds"] = 0.0
        state["width_frame_last_slice_seconds"] = 0.0
        state["width_frame_max_slice_seconds"] = 0.0
        state["width_frame_result"] = None
        state["width_frame"] = None
        state["width_rails"] = {"left": [], "right": []}
        if state.get("timer") is None:
            state["timer"] = context.window_manager.event_timer_add(
                0.01, window=context.window
            )
        _guided_ridge_overlay_tag(state)
        return True
    except (AttributeError, IndexError, KeyError, MemoryError, RuntimeError, TypeError, ValueError):
        state["phase"] = "ready"
        state["width_frame_job"] = None
        return False


def _guided_ridge_process_width_frame_timer(context, state):
    """Advance one bounded frame rebuild slice and return to Ready."""
    if state is None or not state.get("active") or state.get("phase") != "width_prepare":
        return False
    operator = state.get("operator")
    if not _guided_ridge_context_matches(context, state):
        if operator is not None:
            operator.report({"WARNING"}, "Guided Ridge: guide-frame context changed; no changes applied")
        _guided_ridge_cancel(state, "width-frame-context-changed")
        return False
    started = time.perf_counter()
    try:
        progress = None
        deadline = started + 0.008
        while True:
            progress = next(state["width_frame_job"])
            if isinstance(progress, dict):
                stage = str(progress.get("stage", "building guide frame"))
                if stage != state.get("width_frame_stage"):
                    break
            if time.perf_counter() >= deadline:
                break
    except StopIteration as complete:
        try:
            prepared = complete.value
            if not prepared or len(prepared) != 2:
                raise ValueError("guide frame is unavailable")
            frame, layer_result = prepared
            result = _guided_ridge_width_frame_result(
                state["snapshot"],
                state["guide"],
                state.get("controls", ()),
                state.get("control_normals", ()),
                state.get("half_width"),
                frame=frame,
                layer_result=layer_result,
            )
            state["width_frame_job"] = None
            state["width_frame_result"] = result
            state["width_frame"] = result["frame"]
            state["width_rails"] = result["rails"]
            state["half_width"] = result["width"]
            state["width_min"] = result["width_min"]
            state["width_max"] = result["width_max"]
            state["phase"] = "ready"
            state["width_frame_stage"] = "ready"
            state["width_frame_stage_done"] = 1
            state["width_frame_stage_total"] = 1
            state["width_frame_progress_fraction"] = 1.0
            state["width_frame_progress_indeterminate"] = False
            timer = state.get("timer")
            if timer is not None:
                try:
                    state["window_manager"].event_timer_remove(timer)
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    pass
                state["timer"] = None
            _guided_ridge_overlay_tag(state)
        except (AttributeError, IndexError, MemoryError, RuntimeError, TypeError, ValueError) as error:
            if operator is not None:
                operator.report({"WARNING"}, f"Guided Ridge: guide frame failed ({error})")
            _guided_ridge_cancel(state, "width-frame-failed")
            return False
    except (AttributeError, IndexError, MemoryError, RuntimeError, TypeError, ValueError) as error:
        if operator is not None:
            operator.report({"WARNING"}, f"Guided Ridge: guide frame failed ({error})")
        _guided_ridge_cancel(state, "width-frame-exception")
        return False
    else:
        if isinstance(progress, dict):
            stage = str(progress.get("stage", "building guide frame"))
            state["width_frame_stage"] = stage
            state["width_frame_stage_index"] = int(progress.get("stage_index", 0))
            state["width_frame_stage_count"] = int(progress.get("stage_count", 11))
            state["width_frame_stage_done"] = int(progress.get("done", 0))
            state["width_frame_stage_total"] = int(progress.get("total", 0))
            state["width_frame_progress_indeterminate"] = bool(progress.get("indeterminate", False))
            total = int(progress.get("total", 0))
            done = int(progress.get("done", 0))
            state["width_frame_progress_fraction"] = (
                min(max(done / float(total), 0.0), 1.0) if total > 0 else None
            )
            _guided_ridge_overlay_tag(state)
    finally:
        elapsed = time.perf_counter() - started
        state["width_frame_last_slice_seconds"] = float(elapsed)
        state["width_frame_max_slice_seconds"] = max(
            float(state.get("width_frame_max_slice_seconds", 0.0)), float(elapsed)
        )
        state["width_frame_elapsed_seconds"] = float(
            state.get("width_frame_elapsed_seconds", 0.0)
        ) + float(elapsed)
    return True


def _guided_ridge_width_adjacency_steps(snapshot):
    """Build adjacency chunks without an unbounded native concatenate."""
    import numpy as np

    adjacency = snapshot.get("adjacency", ())
    source_parts = []
    target_parts = []
    # Four source vertices keeps each concatenate/reduction bounded on the
    # large Hair component; the extra yields are preferable to a >16 ms UI
    # stall from one broad adjacency batch.
    chunk = 4
    total = int(math.ceil(len(adjacency) / float(chunk))) if adjacency else 0
    for start in range(0, len(adjacency), chunk):
        stop = min(start + chunk, len(adjacency))
        local = adjacency[start:stop]
        targets = [np.asarray(values, dtype=np.int64) for values in local if values]
        if targets:
            target_parts.append(np.concatenate(targets))
            source_parts.append(
                np.repeat(
                    np.arange(start, stop, dtype=np.int64),
                    np.asarray([len(values) for values in local], dtype=np.int64),
                )
            )
        yield {
            "stage": "preparing width smoothing",
            "stage_index": 7,
            "stage_count": 10,
            "done": int(stop // max(chunk, 1)),
            "total": total,
        }
    return source_parts, target_parts, np.asarray([len(values) for values in adjacency], dtype=np.float64)


def _guided_ridge_width_candidate_steps(
    snapshot, guide, controls, control_normals, half_width=None, width_result=None
):
    """Cooperatively compute a fixed-width, crest-side planing candidate."""
    yield {
        "stage": "queueing width candidate",
        "stage_index": 0,
        "stage_count": 10,
        "done": 0,
        "total": 0,
        "indeterminate": True,
    }
    import numpy as np

    before = np.asarray(snapshot["coords_world"], dtype=np.float64)
    faces = np.asarray(snapshot["triangles"], dtype=np.int64)
    if width_result is None:
        frame_job = _guided_ridge_stable_frame_steps(before, faces, guide, controls, control_normals)
        try:
            while True:
                yield next(frame_job)
        except StopIteration as complete:
            frame = complete.value
        frame = _guided_ridge_align_frame_to_controls(frame, controls, control_normals)
        frame["average_edge"] = float(snapshot.get("average_edge", 0.0) or 0.0)
        width_result = _guided_ridge_width_frame_result(
            snapshot, guide, controls, control_normals, half_width, frame=frame
        )
    frame = width_result["frame"]
    width = float(width_result["width"])
    minimum_width = float(width_result["width_min"])
    maximum_width = float(width_result["width_max"])
    average_edge = float(snapshot.get("average_edge", 0.0) or 0.0)
    if width_result.get("projection_required") and not width_result.get("projection_ok"):
        layer_result = width_result.get("layer_result") or {}
        reasons = layer_result.get("reasons", {}) if isinstance(layer_result, dict) else {}
        reason_values = [
            str(reason)
            for values in reasons.values()
            for reason in (values or ())
            if reason
        ]
        detail = reason_values[0] if reason_values else "local surface gap"
        raise ValueError(f"planing width rail projection is incomplete ({detail})")
    if not width_result.get("anchor_projection_ok", True):
        raise ValueError("planing width rail anchor support is incomplete")
    if not width_result.get("rail_hit_ok", True):
        raise ValueError("planing width rail triangle support is unavailable")
    boundary_vertices, _boundary_edges = _guided_ridge_boundary_local_data(snapshot, len(before))
    boundary_mask = np.zeros(len(before), dtype=bool)
    boundary_mask[boundary_vertices] = True
    segment = np.diff(frame["guide"], axis=0)
    length2 = np.sum(segment * segment, axis=1)
    if not np.all(np.isfinite(length2)) or np.any(length2 <= 1.0e-16):
        raise ValueError("guide contains a degenerate segment")
    segment_length = np.sqrt(length2)
    guide_arc = np.r_[0.0, np.cumsum(segment_length)]
    length = float(guide_arc[-1])
    if length <= 1.0e-9:
        raise ValueError("guide length is unavailable")
    seg_index = np.empty(len(before), dtype=np.int32)
    seg_t = np.empty(len(before), dtype=np.float64)
    project_chunk = 128
    project_total = int(math.ceil(len(before) / float(project_chunk)))
    for start in range(0, len(before), project_chunk):
        stop = min(start + project_chunk, len(before))
        block = before[start:stop, None, :] - frame["guide"][:-1][None, :, :]
        parameter = np.clip(np.sum(block * segment[None, :, :], axis=2) / length2[None, :], 0.0, 1.0)
        distance2 = np.sum((block - parameter[:, :, None] * segment[None, :, :]) ** 2, axis=2)
        seg_index[start:stop] = np.argmin(distance2, axis=1)
        seg_t[start:stop] = parameter[np.arange(stop - start), seg_index[start:stop]]
        yield {
            "stage": "projecting vertices to width frame",
            "stage_index": 2,
            "stage_count": 10,
            "done": int(stop // max(project_chunk, 1)),
            "total": project_total,
        }
    origin = frame["guide"][seg_index] + seg_t[:, None] * segment[seg_index]
    station = guide_arc[seg_index] + seg_t * (guide_arc[seg_index + 1] - guide_arc[seg_index])
    # ``seg_index``/``seg_t`` are already measured on the non-uniform guide
    # segments.  Do not remap station through a uniform sample index: that
    # changes the frame on curved guides whose points are unevenly spaced.
    tangent, lateral, height_axis = _guided_ridge_interpolate_vertex_frame(
        frame, seg_index, seg_t
    )
    surface_normals = np.asarray(snapshot.get("normals_world", ()), dtype=np.float64)
    if surface_normals.shape != before.shape:
        raise ValueError("surface normals are unavailable for planing direction")
    normal_alignment = np.sum(surface_normals * height_axis, axis=1)
    normal_epsilon = 0.05
    q = np.sum((before - origin) * lateral, axis=1)
    h = np.sum((before - origin) * height_axis, axis=1)
    rail_anchor_lateral = width_result.get("rail_anchor_lateral") or {}
    left_rail_q_values = np.asarray(rail_anchor_lateral.get("left", ()), dtype=np.float64)
    right_rail_q_values = np.asarray(rail_anchor_lateral.get("right", ()), dtype=np.float64)
    if len(left_rail_q_values) != len(frame["arc"]) or len(right_rail_q_values) != len(frame["arc"]):
        raise ValueError("planing width rail lateral supports are unavailable")
    if not np.all(np.isfinite(left_rail_q_values)) or not np.all(np.isfinite(right_rail_q_values)):
        raise ValueError("planing width rail lateral supports are non-finite")
    left_rail_q = np.interp(station, frame["arc"], left_rail_q_values)
    right_rail_q = np.interp(station, frame["arc"], right_rail_q_values)
    rail_q_epsilon = max(average_edge * 0.05, width * 0.001, 1.0e-7)
    if (
        np.any(left_rail_q >= -rail_q_epsilon)
        or np.any(right_rail_q <= rail_q_epsilon)
        or np.any(left_rail_q >= right_rail_q - rail_q_epsilon)
    ):
        raise ValueError("planing width rail lateral supports are degenerate")
    hidden_vertices = np.asarray(snapshot.get("hidden_vertices", np.zeros(len(before), dtype=bool)), dtype=bool)
    sculpt_mask = np.clip(
        np.asarray(snapshot.get("sculpt_mask", np.zeros(len(before), dtype=np.float64)), dtype=np.float64),
        0.0,
        1.0,
    )
    if len(hidden_vertices) != len(before) or len(sculpt_mask) != len(before):
        raise ValueError("protected Sculpt state does not match the component")
    normal_extent = np.abs(q[normal_alignment > normal_epsilon])
    if len(normal_extent) == 0 or float(np.max(normal_extent)) <= 1.0e-8:
        raise ValueError("planing width has no stable lateral surface extent")
    q_epsilon = max(average_edge * 0.35, width * 0.01, 1.0e-7)
    eligible = (
        np.isfinite(q)
        & np.isfinite(h)
        & (q >= left_rail_q - q_epsilon)
        & (q <= right_rail_q + q_epsilon)
        & (normal_alignment > normal_epsilon)
        & (station >= 0.0)
        & (station <= length)
        & ~hidden_vertices
    )
    if int(np.count_nonzero(eligible)) < 8:
        raise ValueError("planing width has too few eligible surface samples")
    # Keep enough longitudinal bins to distinguish a curved/anisotropic guide
    # from a broad first bin.  The previous edge-size-only lower bound merged
    # most of the Hair component into one station and made legitimate anchor
    # heights look like an ambiguous sheet.
    station_bins = min(64, max(16, int(math.ceil(length / max(average_edge * 4.0, 1.0e-7)))))
    station_bins = min(station_bins, max(16, len(before)))
    bin_index = np.clip((station / length * station_bins).astype(np.int64), 0, station_bins - 1)
    # A width/normal test alone can select two nearby, similarly oriented
    # sheets.  Group samples by station and a narrow lateral bin, then reject
    # a resolvable height split instead of averaging the layers together.
    lateral_bin_width = max(width / 8.0, average_edge * 2.0, 1.0e-8)
    lateral_bin = np.floor((q + width) / lateral_bin_width).astype(np.int64)
    competing_keys = np.stack((bin_index[eligible], lateral_bin[eligible]), axis=1)
    competing_values = h[eligible]
    competing_indices = np.flatnonzero(eligible)
    if len(competing_values):
        order = np.lexsort((competing_keys[:, 1], competing_keys[:, 0]))
        sorted_keys = competing_keys[order]
        sorted_values = competing_values[order]
        sorted_indices = competing_indices[order]
        run_start = 0
        while run_start < len(sorted_values):
            run_end = run_start + 1
            while run_end < len(sorted_values) and np.array_equal(sorted_keys[run_end], sorted_keys[run_start]):
                run_end += 1
            run_values = sorted_values[run_start:run_end]
            run_indices = sorted_indices[run_start:run_end]
            height_order = np.argsort(run_values, kind="stable")
            values = run_values[height_order]
            paired_indices = run_indices[height_order]
            if len(values) >= 2:
                gaps = np.diff(values)
                split = int(np.argmax(gaps))
                gap = float(gaps[split]) if len(gaps) else 0.0
                layer_gap_epsilon = max(average_edge * 2.0, width * 0.08, 1.0e-8)
                if gap > layer_gap_epsilon and gap <= width * 1.25:
                    low = set(int(value) for value in paired_indices[: split + 1])
                    high = set(int(value) for value in paired_indices[split + 1:])
                    adjacency = snapshot.get("adjacency", ())
                    station_key, lateral_key = (int(value) for value in sorted_keys[run_start])
                    local_candidates = np.flatnonzero(
                        eligible
                        & (np.abs(bin_index - station_key) <= 1)
                        & (np.abs(lateral_bin - lateral_key) <= 1)
                    )
                    local_set = set(int(value) for value in local_candidates)
                    frontier = list(low & local_set)
                    visited = set(frontier)
                    while frontier:
                        index = frontier.pop()
                        if index in high:
                            break
                        if 0 <= index < len(adjacency):
                            for neighbor in adjacency[index]:
                                neighbor = int(neighbor)
                                if neighbor in local_set and neighbor not in visited:
                                    visited.add(neighbor)
                                    frontier.append(neighbor)
                    connected = bool(visited & high)
                    if not connected:
                        raise ValueError(
                            f"planing width has competing surface layers at the same station "
                            f"(gap={gap:.6g}, samples={len(values)})"
                        )
            run_start = run_end
    bin_centers = (np.arange(station_bins, dtype=np.float64) + 0.5) * length / station_bins
    anchor_band = float(width_result.get("anchor_band", width * GUIDED_RIDGE_WIDTH_ANCHOR_BAND_FRACTION))
    guide_band = float(width_result.get("guide_band", width * GUIDED_RIDGE_WIDTH_GUIDE_BAND_FRACTION))
    anchor_masks = [
        eligible & (np.abs(q - left_rail_q) <= anchor_band),
        eligible & (np.abs(q - right_rail_q) <= anchor_band),
    ]
    anchor_mask = anchor_masks[0] | anchor_masks[1]
    # Rebuild this mask from the immutable rail provenance as well as the
    # compatibility summary.  The latter can be stale after a cached width
    # refresh; every barycentric triangle/edge support vertex must remain
    # fixed through smoothing, partial-mask falloff, and the final write.
    support_mask = _guided_ridge_width_support_mask(width_result, len(before))
    for support_index in np.asarray(width_result.get("anchor_vertex_indices", ()), dtype=np.int64):
        if 0 <= int(support_index) < len(support_mask):
            support_mask[int(support_index)] = True
    anchor_mask |= support_mask
    rail_anchor_height = width_result.get("rail_anchor_height") or {}
    left_anchor_values = np.asarray(rail_anchor_height.get("left", ()), dtype=np.float64)
    right_anchor_values = np.asarray(rail_anchor_height.get("right", ()), dtype=np.float64)
    if len(left_anchor_values) != len(frame["arc"]) or len(right_anchor_values) != len(frame["arc"]):
        raise ValueError("planing width rail anchor heights are unavailable")
    left_h = np.interp(station, frame["arc"], left_anchor_values)
    right_h = np.interp(station, frame["arc"], right_anchor_values)
    # Use the actual projected rail q values as the plane endpoints.  The
    # nominal UI width is only the request used to find the rail; it is not a
    # valid denominator when the surface projection lands short or long.
    target_h = np.where(
        q <= 0.0,
        (q / left_rail_q) * left_h,
        (q / right_rail_q) * right_h,
    )
    guide_anchor_mask = np.abs(q) <= guide_band
    crest_side_mask = eligible
    yield {"stage": "preparing width smoothing", "stage_index": 7, "stage_count": 10, "done": 0, "total": 4}
    boundary_distance_job = _guided_ridge_bounded_boundary_distance_steps(before, boundary_mask)
    try:
        while True:
            yield next(boundary_distance_job)
    except StopIteration as complete:
        boundary_distance, boundary_chunk_size, boundary_count = complete.value
    boundary_width = max(width * 0.12, average_edge * 2.0, 1.0e-7)
    boundary_guard = np.empty(len(boundary_distance), dtype=np.float64)
    falloff_chunk = 128
    falloff_total = int(math.ceil(len(boundary_distance) / float(falloff_chunk)))
    for start in range(0, len(boundary_distance), falloff_chunk):
        stop = min(start + falloff_chunk, len(boundary_distance))
        boundary_guard[start:stop] = [
            _guided_ridge_smoothstep(value / boundary_width)
            for value in boundary_distance[start:stop]
        ]
        yield {
            "stage": "building width boundary falloff",
            "stage_index": 5,
            "stage_count": 10,
            "done": int(stop // max(falloff_chunk, 1)),
            "total": falloff_total,
        }
    # Give the UI a turn before the final vectorized classification and
    # smoothing setup, which is larger than the subsequent per-chunk yields
    # on dense components.
    yield {"stage": "preparing width smoothing", "stage_index": 7, "stage_count": 10, "done": 0, "total": 2}
    root_fade = max(0.13 * length, 1.0e-7)
    root_guard = np.asarray([_guided_ridge_smoothstep(value / root_fade) for value in station], dtype=np.float64)
    tip_fade = max(0.003 / 0.07595313195548941 * length, 1.0e-9)
    tip_guard = np.asarray([
        _guided_ridge_smoothstep((length - tip_fade - value) / tip_fade) for value in station
    ], dtype=np.float64)
    protected_weight = np.clip(1.0 - sculpt_mask, 0.0, 1.0)
    protected_weight[hidden_vertices] = 0.0
    protected_weight[boundary_mask] = 0.0
    anchored = boundary_mask | guide_anchor_mask | anchor_mask
    yield {"stage": "preparing width smoothing", "stage_index": 7, "stage_count": 10, "done": 0, "total": 3}
    yield {"stage": "preparing width smoothing", "stage_index": 7, "stage_count": 10, "done": 1, "total": 4}
    planing_delta = target_h - h
    planing_delta[(planing_delta * normal_alignment) > 0.0] = 0.0
    planing_delta[normal_alignment <= normal_epsilon] = 0.0
    protrusion_mask = crest_side_mask & (planing_delta * normal_alignment < -1.0e-12)
    planing_delta[~protrusion_mask] = 0.0
    safe_low = np.minimum(planing_delta, 0.0)
    yield {"stage": "preparing width smoothing", "stage_index": 7, "stage_count": 10, "done": 0, "total": 1}
    adjacency_job = _guided_ridge_width_adjacency_steps(snapshot)
    try:
        while True:
            yield next(adjacency_job)
    except StopIteration as complete:
        adjacency_source_parts, adjacency_target_parts, adjacency_count = complete.value
    def apply_steps(weight):
        delta = planing_delta * (0.82 * weight)
        delta[~protrusion_mask] = 0.0
        yield {
            "stage": "building width planing delta",
            "stage_index": 8,
            "done": 0,
            "total": 4,
        }
        for iteration in range(2):
            relaxed = delta.copy()
            neighbor_sum = np.zeros(len(delta), dtype=np.float64)
            for part_index, (source_part, target_part) in enumerate(
                zip(adjacency_source_parts, adjacency_target_parts), 1
            ):
                np.add.at(neighbor_sum, source_part, delta[target_part])
                if part_index % 256 == 0:
                    yield {
                        "stage": "building width planing delta",
                        "stage_index": 8,
                        "done": 1 + iteration,
                        "total": 4,
                    }
            movable = (adjacency_count > 0.0) & ~anchored & protrusion_mask
            relaxed[movable] = 0.72 * delta[movable] + 0.28 * neighbor_sum[movable] / adjacency_count[movable]
            relaxed[~protrusion_mask] = 0.0
            relaxed[anchored] = 0.0
            relaxed = np.clip(relaxed, safe_low, 0.0)
            delta = relaxed
        delta[(delta * normal_alignment) > 0.0] = 0.0
        delta[~protrusion_mask] = 0.0
        return np.clip(delta, safe_low, 0.0)
    yield {"stage": "building width planing delta", "stage_index": 8, "stage_count": 10, "done": 0, "total": 2}
    apply_job = apply_steps(root_guard * tip_guard * boundary_guard)
    try:
        while True:
            yield next(apply_job)
    except StopIteration as complete:
        delta_h = complete.value
    yield {"stage": "building width planing delta", "stage_index": 8, "stage_count": 10, "done": 3, "total": 4}
    candidate_delta = delta_h[:, None] * height_axis * protected_weight[:, None]
    candidate_delta[anchored] = 0.0
    candidate_delta[support_mask] = 0.0
    candidate_delta[~protrusion_mask] = 0.0
    final_delta_h = np.sum(candidate_delta * height_axis, axis=1)
    final_delta_h[~protrusion_mask] = 0.0
    final_delta_h[support_mask] = 0.0
    final_delta_h = np.clip(final_delta_h, safe_low, 0.0)
    final_delta_h[(final_delta_h * normal_alignment) > 0.0] = 0.0
    candidate = before + final_delta_h[:, None] * height_axis
    yield {"stage": "building width planing delta", "stage_index": 8, "stage_count": 10, "done": 1, "total": 2}
    integrity_scales = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    selected_scale = None
    for scale_index, scale_factor in enumerate(integrity_scales, 1):
        yield {
            "stage": "checking width planing integrity",
            "stage_index": 9,
            "stage_count": 10,
            "done": scale_index - 1,
            "total": len(integrity_scales),
        }
        trial = before + (candidate - before) * scale_factor
        integrity_job = _guided_ridge_mesh_integrity_steps(snapshot, before, trial)
        try:
            while True:
                yield next(integrity_job)
        except StopIteration as complete:
            integrity = complete.value
        if integrity["finite"] and not integrity["normal_pair_flips"] and not integrity["new_degenerate_faces"]:
            candidate = trial
            selected_scale = float(scale_factor)
            break
        yield {
            "stage": "checking width planing integrity",
            "stage_index": 9,
            "stage_count": 10,
            "done": scale_index,
            "total": len(integrity_scales),
        }
    if selected_scale is None:
        raise ValueError("width planing candidate fails mesh integrity")
    return candidate, {
        "length": length,
        "station": station,
        "q": q,
        "left_rail_q": left_rail_q,
        "right_rail_q": right_rail_q,
        "h": h,
        "target_h": target_h,
        "width": width,
        "width_min": minimum_width,
        "width_max": maximum_width,
        "boundary_mask": boundary_mask,
        "crest_side_mask": crest_side_mask,
        "eligible_mask": eligible,
        "protrusion_mask": protrusion_mask,
        "anchor_mask": anchored,
        "width_anchor_mask": anchor_mask,
        "guide_anchor_mask": guide_anchor_mask,
        "selected_arc_mask": crest_side_mask,
        "outward_component": normal_alignment,
        "delta_h": np.sum((candidate - before) * height_axis, axis=1),
        "integrity_scale": selected_scale,
        "boundary_distance_chunk_size": int(boundary_chunk_size),
        "boundary_count": int(boundary_count),
        "vertex_guide_chunk_size": int(project_chunk),
        "frame_tangent": tangent,
        "frame_lateral": lateral,
        "frame_height_axis": height_axis,
        "rail_hits": width_result.get("rail_hits", {}),
        "rail_support_vertex_indices": width_result.get("rail_support_vertex_indices", {}),
        "rail_support_mask": support_mask,
    }


def _guided_ridge_exact_candidate_steps(
    snapshot, guide, controls, control_normals, half_width=None, width_result=None
):
    """Active width-based cooperative candidate path."""
    job = _guided_ridge_width_candidate_steps(
        snapshot, guide, controls, control_normals, half_width, width_result=width_result
    )
    try:
        while True:
            yield next(job)
    except StopIteration as complete:
        return complete.value


def _guided_ridge_exact_candidate(
    snapshot, guide, controls, control_normals, half_width=None, width_result=None
):
    """Synchronous compatibility wrapper for the active width candidate."""
    job = _guided_ridge_exact_candidate_steps(
        snapshot, guide, controls, control_normals, half_width, width_result=width_result
    )
    try:
        while True:
            next(job)
    except StopIteration as complete:
        return complete.value


def _guided_ridge_width_support_mask(width_result, vertex_count):
    """Return all fixed vertices supporting the shared projected rails."""
    import numpy as np

    mask = np.zeros(int(vertex_count), dtype=bool)
    if not isinstance(width_result, dict):
        return mask
    groups = width_result.get("rail_support_vertex_indices", {}) or {}
    for values in groups.values() if isinstance(groups, dict) else ():
        for support in values or ():
            for index in support or ():
                try:
                    index = int(index)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(mask):
                    mask[index] = True
    hits = width_result.get("rail_hits", {}) or {}
    for values in hits.values() if isinstance(hits, dict) else ():
        for hit in values or ():
            if not isinstance(hit, dict):
                continue
            # Include the complete hit triangle as a conservative support set:
            # BVH barycentric coordinates can carry a tiny non-zero weight on
            # the third vertex even when the hit is numerically on an edge.
            # Fixing it prevents the reconstructed rail point from drifting
            # across Blender's float32 mesh write/read round-trip.
            for key in ("support_vertex_indices", "edge_support_vertex_indices", "triangle_vertices"):
                for index in hit.get(key, ()) or ():
                    try:
                        index = int(index)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= index < len(mask):
                        mask[index] = True
    return mask


def _guided_ridge_rail_support_unchanged(snapshot, before, after, rail_hits, tolerance=1.0e-9):
    """Check projected rail points through their stored barycentric support."""
    import numpy as np

    if not rail_hits:
        return True
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    for side in ("left", "right"):
        for hit in rail_hits.get(side, ()):
            if not hit:
                continue
            vertices = tuple(int(value) for value in hit.get("triangle_vertices", ()))
            weights = np.asarray(hit.get("barycentric_weights", ()), dtype=np.float64)
            if not vertices or len(vertices) != len(weights):
                return False
            if any(value < 0 or value >= len(before) for value in vertices):
                return False
            if not np.all(np.isfinite(weights)) or abs(float(np.sum(weights)) - 1.0) > 1.0e-6:
                return False
            pre = np.sum(before[np.asarray(vertices, dtype=np.int64)] * weights[:, None], axis=0)
            post = np.sum(after[np.asarray(vertices, dtype=np.int64)] * weights[:, None], axis=0)
            location = np.asarray(tuple(hit.get("location", ())), dtype=np.float64)
            if len(location) == 3 and float(np.linalg.norm(pre - location)) > 1.0e-7:
                return False
            if float(np.linalg.norm(post - pre)) > tolerance:
                return False
    return True


def _guided_ridge_mesh_integrity(snapshot, before, after):
    import numpy as np
    faces = np.asarray(snapshot["triangles"], dtype=np.int64)
    before_tri = before[faces]
    after_tri = after[faces]
    cross_before = np.cross(before_tri[:, 1] - before_tri[:, 0], before_tri[:, 2] - before_tri[:, 0])
    cross_after = np.cross(after_tri[:, 1] - after_tri[:, 0], after_tri[:, 2] - after_tri[:, 0])
    area_before = np.linalg.norm(cross_before, axis=1) * 0.5
    area_after = np.linalg.norm(cross_after, axis=1) * 0.5
    dot = np.sum(cross_before * cross_after, axis=1) / np.maximum(
        np.linalg.norm(cross_before, axis=1) * np.linalg.norm(cross_after, axis=1), 1.0e-30
    )
    return {
        "finite": bool(np.all(np.isfinite(after))),
        "normal_pair_flips": int(np.count_nonzero(dot < 0.0)),
        "normal_pair_dot_min": float(np.min(dot)) if len(dot) else 1.0,
        "new_degenerate_faces": int(np.count_nonzero((area_after <= 1.0e-12) & (area_before > 1.0e-12))),
    }


def _guided_ridge_mesh_integrity_steps(snapshot, before, after, chunk_size=8):
    """Yield bounded integrity checks for the modal candidate phase.

    The synchronous helper above remains the public/reference check.  The
    modal path uses this mechanically equivalent chunked form so a dense
    component never spends a full frame in one vectorized triangle batch.
    """
    yield {
        "stage": "checking width planing integrity",
        "stage_index": 9,
        "stage_count": 10,
        "done": 0,
        "total": 0,
    }
    import numpy as np

    faces = np.asarray(snapshot["triangles"], dtype=np.int64)
    total = int(math.ceil(len(faces) / float(max(int(chunk_size), 1))))
    normal_flips = 0
    dot_min = 1.0
    new_degenerate = 0
    finite = bool(np.all(np.isfinite(after)))
    for index, start in enumerate(range(0, len(faces), max(int(chunk_size), 1)), 1):
        stop = min(start + max(int(chunk_size), 1), len(faces))
        local_faces = faces[start:stop]
        before_tri = before[local_faces]
        after_tri = after[local_faces]
        cross_before = np.cross(
            before_tri[:, 1] - before_tri[:, 0],
            before_tri[:, 2] - before_tri[:, 0],
        )
        cross_after = np.cross(
            after_tri[:, 1] - after_tri[:, 0],
            after_tri[:, 2] - after_tri[:, 0],
        )
        area_before = np.linalg.norm(cross_before, axis=1) * 0.5
        area_after = np.linalg.norm(cross_after, axis=1) * 0.5
        dot = np.sum(cross_before * cross_after, axis=1) / np.maximum(
            np.linalg.norm(cross_before, axis=1)
            * np.linalg.norm(cross_after, axis=1),
            1.0e-30,
        )
        normal_flips += int(np.count_nonzero(dot < 0.0))
        if len(dot):
            dot_min = min(dot_min, float(np.min(dot)))
        new_degenerate += int(
            np.count_nonzero((area_after <= 1.0e-12) & (area_before > 1.0e-12))
        )
        yield {
            "stage": "checking width planing integrity",
            "stage_index": 9,
            "stage_count": 10,
            "done": index,
            "total": total,
        }
    return {
        "finite": finite,
        "normal_pair_flips": normal_flips,
        "normal_pair_dot_min": dot_min,
        "new_degenerate_faces": new_degenerate,
    }


def _guided_ridge_overlay_tag(state):
    if state is not None:
        _tag_redraw(state.get("area"))


def _guided_ridge_preview_only_finish(operator, state, reason="confirm"):
    """Finish the Phase 1 UI session without entering the write path."""
    if operator is not None:
        operator.report({"INFO"}, "Guided Ridge UI Prototype: Preview Only: no geometry changes")
    if state is not None:
        state["preview_only_finished"] = True
        state["modal_result"] = "FINISHED"
    _guided_ridge_cancel(state, reason)
    return True


def _guided_ridge_curve_dedupe(points, epsilon=1.0e-7):
    """Copy route points while removing zero/near-zero length segments."""
    values = [Vector(point) for point in (points or ())]
    if not values:
        return []
    result = [values[0].copy()]
    route_length = sum(
        (second - first).length for first, second in zip(values, values[1:])
    )
    # Keep the click-noise policy scale-aware.  A fixed world-space epsilon
    # would erase an otherwise meaningful route when the same geometry is
    # uniformly scaled down to micron-sized coordinates.
    scale_floor = math.ulp(max(route_length, 1.0e-30)) * 16.0
    threshold = max(
        min(float(epsilon), float(epsilon) * route_length),
        scale_floor,
    )
    for point in values[1:]:
        if (point - result[-1]).length > threshold:
            result.append(point.copy())
    # Preserve the user supplied endpoint even when the last segment is very
    # short.  This keeps route editing reversible while still deduplicating
    # interior click noise.
    if len(values) > 1 and (values[-1] - result[-1]).length > scale_floor:
        result.append(values[-1].copy())
    return result


def _guided_ridge_curve_positive_smooth(samples, clean, amount=None):
    """Build a fixed target on a uniform arc-length grid.

    ``samples`` is deliberately a genuinely uniform arc-length grid.  The
    output route later contains the original knots as well, but smoothing the
    augmented (and therefore non-uniform) list would make the result depend on
    click density.  ``amount`` is retained as a compatibility argument for
    isolated callers; shape magnitude is applied only after this target has
    been built.
    """
    del amount
    for _iteration in range(5):
        alpha = 0.82
        previous = [point.copy() for point in samples]
        for index in range(1, len(samples) - 1):
            midpoint = (previous[index - 1] + previous[index + 1]) * 0.5
            candidate = previous[index].lerp(midpoint, alpha)
            # Component-wise neighbour bounds prevent an accidental large
            # overshoot or self-loop in the review-only curve preview.
            low = Vector((
                min(previous[index - 1][axis], previous[index + 1][axis])
                for axis in range(3)
            ))
            high = Vector((
                max(previous[index - 1][axis], previous[index + 1][axis])
                for axis in range(3)
            ))
            samples[index] = Vector((
                min(max(candidate[axis], low[axis]), high[axis])
                for axis in range(3)
            ))
        samples[0] = clean[0].copy()
        samples[-1] = clean[-1].copy()
    return samples


def _guided_ridge_curve_uniform_resample(clean, cumulative, total, sample_count):
    """Return only uniform arc-length samples for the smoothing calculation."""
    if sample_count <= 1:
        return [clean[0].copy()]
    samples = []
    segment = 0
    for index in range(sample_count):
        distance = total * index / float(sample_count - 1)
        while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] < distance:
            segment += 1
        if distance <= cumulative[segment] + 1.0e-12:
            samples.append(clean[segment].copy())
            continue
        if distance >= cumulative[segment + 1] - 1.0e-12:
            samples.append(clean[segment + 1].copy())
            continue
        span = cumulative[segment + 1] - cumulative[segment]
        factor = 0.0 if span <= 1.0e-12 else (distance - cumulative[segment]) / span
        samples.append(clean[segment].lerp(clean[segment + 1], factor))
    samples[0] = clean[0].copy()
    samples[-1] = clean[-1].copy()
    return samples


def _guided_ridge_curve_arc_tangent(clean, cumulative, distance):
    """Get a stable piecewise-linear raw-route tangent at arc distance."""
    if len(clean) <= 1:
        return Vector((0.0, 0.0, 0.0))
    segment = 0
    target = min(max(float(distance), 0.0), cumulative[-1])
    while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] < target - 1.0e-12:
        segment += 1
    tangent = clean[segment + 1] - clean[segment]
    if tangent.length <= 1.0e-12:
        for candidate in range(segment + 1, len(clean) - 1):
            tangent = clean[candidate + 1] - clean[candidate]
            if tangent.length > 1.0e-12:
                break
    if tangent.length <= 1.0e-12:
        for candidate in range(segment - 1, -1, -1):
            tangent = clean[candidate + 1] - clean[candidate]
            if tangent.length > 1.0e-12:
                break
    if tangent.length <= 1.0e-12:
        return Vector((0.0, 0.0, 0.0))
    return tangent.normalized()


def _guided_ridge_curve_interpolate_vectors(values, source_distances, target_distances):
    """Interpolate vectors by arc length without index-based resampling."""
    if not values:
        return []
    result = []
    segment = 0
    last = len(source_distances) - 1
    for distance in target_distances:
        target = min(max(float(distance), source_distances[0]), source_distances[-1])
        while segment + 1 < last and source_distances[segment + 1] < target:
            segment += 1
        if target <= source_distances[segment] + 1.0e-12:
            result.append(values[segment].copy())
            continue
        if target >= source_distances[segment + 1] - 1.0e-12:
            result.append(values[segment + 1].copy())
            continue
        span = source_distances[segment + 1] - source_distances[segment]
        factor = 0.0 if span <= 1.0e-12 else (target - source_distances[segment]) / span
        result.append(values[segment].lerp(values[segment + 1], factor))
    return result


def _guided_ridge_curve_progression_scale(raw_points, offsets, minimum_dot=1.0e-8):
    """Find a continuous safe scale that prevents adjacent segment reversal."""
    if len(raw_points) <= 2:
        return 1.0
    if not offsets or all(offset.length <= 1.0e-30 for offset in offsets):
        return 1.0

    raw_total = sum(
        (raw_points[index + 1] - raw_points[index]).length
        for index in range(len(raw_points) - 1)
    )
    # This tolerance is relative to the route and machine precision, never a
    # fixed world-space dot-product floor.  It lets scaled copies of the same
    # route take the same Shape path.
    route_tol = max(raw_total * 1.0e-12, math.ulp(max(raw_total, 1.0e-30)) * 16.0)

    def valid(scale):
        if scale <= 1.0e-15:
            # The raw route is valid by construction after deduplication.
            return True
        previous = raw_points[0]
        for index in range(1, len(raw_points)):
            current = raw_points[index] + offsets[index] * scale
            raw_step = raw_points[index] - raw_points[index - 1]
            candidate_step = current - previous
            raw_length = raw_step.length
            candidate_length = candidate_step.length
            if raw_length <= route_tol:
                # The knot-inclusive augmented list can contain a numerically
                # tiny edge.  It must not disable the entire route's Shape;
                # preserve its local ordering and let other edges constrain
                # the global scale.
                previous = current
                continue
            if candidate_length <= route_tol:
                return False
            directional_cosine = candidate_step.dot(raw_step) / (raw_length * candidate_length)
            if directional_cosine < 1.0e-7:
                return False
            previous = current
        return True

    if valid(1.0):
        return 1.0
    low, high = 0.0, 1.0
    for _iteration in range(28):
        middle = (low + high) * 0.5
        if valid(middle):
            low = middle
        else:
            high = middle
    return low


def _guided_ridge_curve_resample(clean, total, cumulative, sample_count):
    """Resample while retaining every original control knot exactly."""
    ordered = _guided_ridge_curve_output_parameters(
        total, cumulative, sample_count
    )
    samples = []
    segment = 0
    tolerance = max(total * 1.0e-9, 1.0e-12)
    for distance in ordered:
        while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] < distance - tolerance:
            segment += 1
        if abs(distance - cumulative[segment]) <= tolerance:
            samples.append(clean[segment].copy())
            continue
        if abs(distance - cumulative[segment + 1]) <= tolerance:
            samples.append(clean[segment + 1].copy())
            continue
        span = cumulative[segment + 1] - cumulative[segment]
        factor = 0.0 if span <= 1.0e-12 else (distance - cumulative[segment]) / span
        samples.append(clean[segment].lerp(clean[segment + 1], factor))
    samples[0] = clean[0].copy()
    samples[-1] = clean[-1].copy()
    return samples


def _guided_ridge_curve_output_parameters(total, cumulative, sample_count):
    """Return the shared uniform-plus-knot arc-length parameterization."""
    parameters = [
        total * index / float(max(sample_count - 1, 1))
        for index in range(sample_count)
    ]
    parameters.extend(float(distance) for distance in cumulative)
    tolerance = max(total * 1.0e-9, 1.0e-12)
    ordered = []
    for distance in sorted(parameters):
        if not ordered or distance - ordered[-1] > tolerance:
            ordered.append(distance)
    return ordered


def _guided_ridge_curve_preview_points(points, smoothing=50.0):
    """Build a bounded, endpoint-preserving signed-shape preview.

    Positive values preserve the existing low-frequency averaging.  Negative
    values reflect that filtered displacement through the arc-length route,
    producing an outward ridge emphasis without amplifying hand-click jitter.
    """
    clean = _guided_ridge_curve_dedupe(points)
    if len(clean) <= 1:
        return [point.copy() for point in clean]
    total = sum((b - a).length for a, b in zip(clean, clean[1:]))
    if total <= 1.0e-12:
        return [clean[0].copy(), clean[-1].copy()]
    sample_count = max(8, min(128, max(len(clean) * 4, 2)))
    cumulative = [0.0]
    for first, second in zip(clean, clean[1:]):
        cumulative.append(cumulative[-1] + (second - first).length)
    # The displayed route retains every original knot, while the smoothing
    # target is built exclusively on this uniform grid.  This prevents an
    # irregular click distribution from becoming an unintended smoothing
    # weight.
    samples = _guided_ridge_curve_resample(clean, total, cumulative, sample_count)
    uniform = _guided_ridge_curve_uniform_resample(
        clean, cumulative, total, sample_count
    )
    uniform_distances = [
        total * index / float(max(len(uniform) - 1, 1))
        for index in range(len(uniform))
    ]
    try:
        signed = min(max(float(smoothing), -100.0), 100.0) / 100.0
    except (TypeError, ValueError):
        signed = 0.0
    if abs(signed) <= 1.0e-12:
        return [point.copy() for point in samples]
    baseline = [point.copy() for point in samples]
    positive = _guided_ridge_curve_positive_smooth(
        [point.copy() for point in uniform], clean, 1.0
    )
    uniform_displacement = []
    for index, point in enumerate(positive):
        displacement = point - uniform[index]
        tangent = _guided_ridge_curve_arc_tangent(
            clean, cumulative, uniform_distances[index]
        )
        if tangent.length > 1.0e-12:
            displacement -= tangent * displacement.dot(tangent)
        uniform_displacement.append(displacement)
    # Reuse the exact parameter list used to build ``samples``.  Inferring
    # parameters from nearest geometry would be ambiguous for self-crossing
    # preview routes and could select a different arc branch.
    output_distances = _guided_ridge_curve_output_parameters(
        total, cumulative, sample_count
    )
    displacement_field = _guided_ridge_curve_interpolate_vectors(
        uniform_displacement, uniform_distances, output_distances
    )
    if signed > 0.0:
        offsets = [displacement * signed for displacement in displacement_field]
        safe_scale = _guided_ridge_curve_progression_scale(baseline, offsets)
        result = [baseline[index] + offsets[index] * safe_scale for index in range(len(baseline))]
        result[0] = clean[0].copy()
        result[-1] = clean[-1].copy()
        return result

    # Negative shape is R + amount * (R - S).  Filter that displacement once
    # more before extrapolation so hand-click jitter is not amplified.  Limit
    # each excursion by local arc length so sparse or sharply curved routes
    # cannot produce loops.
    # Filter only the uniform-grid displacement, then interpolate it by arc
    # length.  Never run a neighbour filter on the augmented knot-inclusive
    # output list.
    displacement_field = [-vector for vector in uniform_displacement]
    for _filter_pass in range(2):
        previous = [point.copy() for point in displacement_field]
        for index in range(1, len(previous) - 1):
            displacement_field[index] = (
                previous[index - 1] + previous[index] * 2.0 + previous[index + 1]
            ) * 0.25
        displacement_field[0] = Vector((0.0, 0.0, 0.0))
        displacement_field[-1] = Vector((0.0, 0.0, 0.0))
    displacement_field = _guided_ridge_curve_interpolate_vectors(
        displacement_field, uniform_distances, output_distances
    )
    average_step = total / float(max(len(baseline) - 1, 1))
    result = [point.copy() for point in baseline]
    offsets = [Vector((0.0, 0.0, 0.0)) for _point in baseline]
    for index in range(1, len(baseline) - 1):
        displacement = displacement_field[index]
        if displacement.length <= 1.0e-12:
            continue
        local_left = (baseline[index] - baseline[index - 1]).length
        local_right = (baseline[index + 1] - baseline[index]).length
        local_step = min(local_left, local_right)
        excursion_limit = max(average_step * 1.75, local_step * 1.5)
        scaled = displacement * abs(signed)
        normalized = scaled.length / max(excursion_limit, 1.0e-12)
        if normalized > 1.0e-12:
            # Smoothly approaches the local limit; there is no value-dependent
            # hard-clamp jump as Shape crosses an integer or threshold.
            scaled *= math.tanh(normalized) / normalized
        offsets[index] = scaled
    safe_scale = _guided_ridge_curve_progression_scale(baseline, offsets)
    for index in range(1, len(baseline) - 1):
        result[index] = baseline[index] + offsets[index] * safe_scale
    result[0] = clean[0].copy()
    result[-1] = clean[-1].copy()
    return result


def _guided_ridge_curve_2d_dedupe(points, epsilon=0.20):
    """Deduplicate projected screen points with a pixel-scale tolerance."""
    values = [Vector((float(point[0]), float(point[1]))) for point in (points or ())]
    if not values:
        return []
    route_length = sum(
        (second - first).length for first, second in zip(values, values[1:])
    )
    scale_floor = math.ulp(max(route_length, 1.0e-30)) * 16.0
    threshold = max(min(float(epsilon), float(epsilon) * route_length), scale_floor)
    result = [values[0].copy()]
    for point in values[1:]:
        if (point - result[-1]).length > threshold:
            result.append(point.copy())
    if len(values) > 1 and (values[-1] - result[-1]).length > scale_floor:
        result.append(values[-1].copy())
    return result


def _guided_ridge_curve_2d_arc_data(points):
    clean = _guided_ridge_curve_2d_dedupe(points)
    cumulative = [0.0]
    for first, second in zip(clean, clean[1:]):
        cumulative.append(cumulative[-1] + (second - first).length)
    return clean, cumulative, cumulative[-1] if cumulative else 0.0


def _guided_ridge_curve_2d_raw_chord_oracle(clean, cumulative, total):
    """Measure the raw route from every original projected knot.

    Generated curves are intentionally capped at 512 screen samples, but the
    raw reference must not use that cap: a short, tall lobe can peak exactly at
    a manually placed knot between uniform samples.  Chord distance is linear
    along each raw segment, so the deduplicated knots are sufficient to retain
    the extrema.  The returned normalized parameters are also used to compare
    each generated lobe against the same original route interval.
    """
    if len(clean) < 2 or len(clean) != len(cumulative) or total <= 1.0e-12:
        return {
            "parameters": [],
            "offsets": [],
            "normal": Vector((0.0, 0.0)),
            "route_scale": 0.0,
            "noise": 0.0,
            "lobes": [],
        }
    start = clean[0]
    end = clean[-1]
    chord = end - start
    chord_length = chord.length
    if chord_length <= 1.0e-12:
        normal = Vector((0.0, 0.0))
    else:
        normal = Vector((-chord.y, chord.x)).normalized()
    parameters = [float(distance) / float(total) for distance in cumulative]
    offsets = [
        (point - start.lerp(end, parameters[index])).dot(normal)
        for index, point in enumerate(clean)
    ]
    route_scale = max(
        chord_length,
        sum((clean[index + 1] - clean[index]).length for index in range(len(clean) - 1)),
        1.0e-12,
    )
    noise = max(route_scale * 1.0e-5, math.ulp(route_scale) * 64.0)
    return {
        "parameters": parameters,
        "offsets": offsets,
        "normal": normal,
        "route_scale": route_scale,
        "noise": noise,
        "lobes": [],
    }


def _guided_ridge_curve_2d_detect_lobes(values, parameters, oracle):
    """Return meaningful raw-knot arc intervals independent of the cleaned fit.

    Intervals are found on the bounded, uniform cleaned samples.  Original raw
    knots are deliberately consulted later for amplitude, so a small secondary
    lobe cannot disappear merely because its peak fell between draw samples.
    Same-sign peaks separated by a substantial valley become separate lobes;
    sub-noise extrema remain part of their surrounding interval.
    """
    # Production lobe ownership is based on the original projected knots, not
    # on the cleaned fit.  The raw knot parameters are an exact extrema oracle
    # for signed chord distance; this small scalar threshold is used only to
    # ignore sub-noise peaks during interval detection.
    raw_parameters = tuple(float(value) for value in oracle.get("parameters", ()))
    raw_offsets = tuple(float(value) for value in oracle.get("offsets", ()))
    if len(raw_parameters) >= 2 and len(raw_parameters) == len(raw_offsets):
        route_scale = max(float(oracle.get("route_scale", 0.0) or 0.0), 1.0e-12)
        raw_noise = max(
            float(oracle.get("noise", 0.0) or 0.0),
            route_scale * 1.0e-5,
            math.ulp(route_scale) * 64.0,
        )
        peak_noise = raw_noise * 1.5
        meaningful = [
            index for index, value in enumerate(raw_offsets)
            if abs(value) > peak_noise
        ]
        result = []
        if meaningful:
            runs = []
            first = previous = meaningful[0]
            side = 1.0 if raw_offsets[first] > 0.0 else -1.0
            for index in meaningful[1:]:
                current_side = 1.0 if raw_offsets[index] > 0.0 else -1.0
                if index != previous + 1 or current_side != side:
                    runs.append((first, previous, side))
                    first = index
                    side = current_side
                previous = index
            runs.append((first, previous, side))
            for run_first, run_last, run_side in runs:
                peaks = []
                for index in range(run_first, run_last + 1):
                    magnitude = abs(raw_offsets[index])
                    left = abs(raw_offsets[index - 1]) if index > run_first else magnitude
                    right = abs(raw_offsets[index + 1]) if index < run_last else magnitude
                    if (
                        magnitude > peak_noise
                        and magnitude >= left
                        and magnitude >= right
                    ):
                        peaks.append(index)
                if not peaks:
                    peaks = [
                        max(
                            range(run_first, run_last + 1),
                            key=lambda value: abs(raw_offsets[value]),
                        )
                    ]
                boundaries = [run_first]
                for left_peak, right_peak in zip(peaks, peaks[1:]):
                    peak_width = raw_parameters[right_peak] - raw_parameters[left_peak]
                    minimum_width = max(
                        raw_noise / route_scale * 2.0,
                        math.ulp(1.0) * 64.0,
                    )
                    if right_peak - left_peak < 2 or peak_width <= minimum_width:
                        continue
                    valley = min(
                        range(left_peak, right_peak + 1),
                        key=lambda value: abs(raw_offsets[value]),
                    )
                    valley_magnitude = abs(raw_offsets[valley])
                    peak_floor = min(
                        abs(raw_offsets[left_peak]),
                        abs(raw_offsets[right_peak]),
                    )
                    if (
                        valley_magnitude <= peak_floor * 0.78
                        and peak_floor > peak_noise * 2.0
                    ):
                        boundaries.append(valley)
                boundaries.append(run_last)
                for boundary_index, boundary in enumerate(boundaries[:-1]):
                    end_boundary = boundaries[boundary_index + 1]
                    start_parameter = raw_parameters[boundary]
                    end_parameter = raw_parameters[end_boundary]
                    if boundary == run_first and boundary > 0:
                        start_parameter = (
                            raw_parameters[boundary - 1] + start_parameter
                        ) * 0.5
                    if end_boundary == run_last and end_boundary + 1 < len(raw_parameters):
                        end_parameter = (
                            end_parameter + raw_parameters[end_boundary + 1]
                        ) * 0.5
                    if end_parameter <= start_parameter + 1.0e-12:
                        continue
                    result.append({
                        "start": float(start_parameter),
                        "end": float(end_parameter),
                        "side": float(run_side),
                        "raw_peak_indices": tuple(
                            peak for peak in peaks
                            if boundary <= peak <= end_boundary
                        ),
                    })
        result.sort(key=lambda item: (item["start"], item["end"], item["side"]))
        oracle["lobes"] = result
        oracle["lobe_detection"] = {
            "source": "raw-knots",
            "noise": raw_noise,
            "peak_noise": peak_noise,
            "prominence_ratio": 0.78,
            "minimum_width_normalized": max(
                raw_noise / route_scale * 2.0,
                math.ulp(1.0) * 64.0,
            ),
            "interval_count": len(result),
            "raw_knot_count": len(raw_offsets),
        }
        return result
    if not values or len(values) != len(parameters) or not oracle.get("normal"):
        return []
    normal = oracle["normal"]
    chord_start = None
    # The caller passes points through ``values``; offset extraction is done by
    # the companion helper below.  This function accepts scalar offsets in the
    # optional ``values`` form to keep the hot path allocation-free.
    scalar = values
    noise = float(oracle.get("noise", 0.0))
    meaningful = [index for index, value in enumerate(scalar) if abs(value) > noise]
    if not meaningful:
        return []
    runs = []
    run_start = meaningful[0]
    previous = meaningful[0]
    previous_sign = 1.0 if scalar[previous] > 0.0 else -1.0
    for index in meaningful[1:]:
        current_sign = 1.0 if scalar[index] > 0.0 else -1.0
        if index != previous + 1 or current_sign != previous_sign:
            runs.append((run_start, previous, previous_sign))
            run_start = index
        previous = index
        previous_sign = current_sign
    runs.append((run_start, previous, previous_sign))
    result = []
    for first, last, side in runs:
        if last <= first:
            continue
        peaks = []
        for index in range(max(first + 1, 1), min(last, len(scalar) - 2) + 1):
            magnitude = abs(scalar[index])
            if magnitude >= abs(scalar[index - 1]) and magnitude >= abs(scalar[index + 1]):
                if magnitude > noise:
                    peaks.append(index)
        if not peaks:
            peak = max(range(first, last + 1), key=lambda value: abs(scalar[value]))
            peaks = [peak]
        boundaries = [first]
        for left, right in zip(peaks, peaks[1:]):
            if right - left < 2:
                continue
            valley = min(range(left, right + 1), key=lambda value: abs(scalar[value]))
            valley_magnitude = abs(scalar[valley])
            peak_floor = min(abs(scalar[left]), abs(scalar[right]))
            if (
                valley_magnitude <= peak_floor * 0.72
                and peak_floor > noise * 4.0
            ):
                boundaries.append(valley)
        boundaries.append(last)
        for boundary_index, boundary in enumerate(boundaries[:-1]):
            end_boundary = boundaries[boundary_index + 1]
            if end_boundary <= boundary:
                continue
            result.append({
                "start": float(parameters[boundary]),
                "end": float(parameters[end_boundary]),
                "side": side,
            })
    # A low-frequency fit may smooth a small but meaningful raw lobe into its
    # neighbour.  Retain an arc interval for every original-knot sign run that
    # is above the scale-relative noise floor.  These are not interpolation
    # constraints: they only ensure normalization/validation compares the
    # generated waveform with the complete raw oracle.
    raw_parameters = tuple(oracle.get("parameters", ()))
    raw_offsets = tuple(oracle.get("offsets", ()))
    raw_noise = float(oracle.get("noise", noise))
    raw_runs = []
    meaningful_raw = [
        index for index, value in enumerate(raw_offsets)
        if abs(value) > raw_noise * 4.0
    ]
    if meaningful_raw:
        first = previous = meaningful_raw[0]
        side = 1.0 if raw_offsets[first] > 0.0 else -1.0
        for index in meaningful_raw[1:]:
            current_side = 1.0 if raw_offsets[index] > 0.0 else -1.0
            if index != previous + 1 or current_side != side:
                raw_runs.append((first, previous, side))
                first = index
                side = current_side
            previous = index
        raw_runs.append((first, previous, side))
    raw_hints = []
    for first, last, side in raw_runs:
        start = raw_parameters[first]
        end = raw_parameters[last]
        if first > 0:
            start = (raw_parameters[first - 1] + start) * 0.5
        if last + 1 < len(raw_parameters):
            end = (end + raw_parameters[last + 1]) * 0.5
        raw_hints.append({"start": start, "end": end, "side": side})
    # Discard a cleaned interval that spans a raw sign change; otherwise a
    # secondary lobe could still be judged using the wrong side.  Same-side
    # intervals remain useful for the smooth uniform waveform.
    filtered = []
    for item in result:
        incompatible = any(
            item["start"] <= (hint["start"] + hint["end"]) * 0.5 <= item["end"]
            and hint["side"] != item["side"]
            for hint in raw_hints
        )
        if not incompatible:
            filtered.append(item)
    result = filtered
    for hint in raw_hints:
        center = (hint["start"] + hint["end"]) * 0.5
        covered = any(
            item["side"] == hint["side"]
            and item["start"] - 1.0e-9 <= center <= item["end"] + 1.0e-9
            for item in result
        )
        if not covered:
            result.append(hint)
    result.sort(key=lambda item: (item["start"], item["end"]))
    oracle["lobes"] = result
    return result


def _guided_ridge_curve_2d_lobe_amplitude(oracle, interval):
    """Return the exact raw-knot amplitude for one normalized arc interval."""
    start = float(interval.get("start", 0.0))
    end = float(interval.get("end", 1.0))
    side = float(interval.get("side", 0.0))
    values = [
        abs(offset)
        for parameter, offset in zip(oracle.get("parameters", ()), oracle.get("offsets", ()))
        if start - 1.0e-9 <= parameter <= end + 1.0e-9 and offset * side > oracle.get("noise", 0.0)
    ]
    if not values:
        return 0.0
    return max(values)


def _guided_ridge_curve_2d_cubic_eval(segment, factor):
    first, control_a, control_b, last = segment
    value = min(max(float(factor), 0.0), 1.0)
    inverse = 1.0 - value
    return (
        first * (inverse * inverse * inverse)
        + control_a * (3.0 * inverse * inverse * value)
        + control_b * (3.0 * inverse * value * value)
        + last * (value * value * value)
    )


def _guided_ridge_curve_2d_historical_inscribed_fit(
    clean, cumulative, total, smoothing_passes=5
):
    """Reproduce the original Step-2 screen-space inscribed construction.

    This is the pre-outward-arc 3.3.53 construction: a uniform arc-length
    observation grid, broad robust neighbourhood smoothing, an endpoint-chord
    envelope, and a compact C1 Bezier fit.  It is intentionally kept as an
    isolated baseline so the positive/amplified waveform path cannot alter the
    default/inscribed geometry.
    """
    if len(clean) < 2 or total <= 1.0e-12:
        return []
    uniform_count = max(48, min(192, max(len(clean) * 8, 48)))
    uniform_distances = [
        total * index / float(uniform_count - 1)
        for index in range(uniform_count)
    ]
    uniform = _guided_ridge_curve_interpolate_vectors(
        clean, cumulative, uniform_distances
    )
    original = [point.copy() for point in uniform]
    radius = max(4, min(18, int(len(uniform) * 0.065)))
    target = [point.copy() for point in uniform]
    requested_passes = max(0.0, min(float(smoothing_passes), 5.0))
    whole_passes = int(requested_passes)
    fractional_pass = requested_passes - whole_passes
    pass_count = whole_passes + (1 if fractional_pass > 1.0e-9 else 0)
    for _iteration in range(pass_count):
        pass_alpha = 0.72
        if _iteration >= whole_passes:
            pass_alpha *= fractional_pass
        previous = [point.copy() for point in target]
        for index in range(1, len(previous) - 1):
            low_index = max(0, index - radius)
            high_index = min(len(previous), index + radius + 1)
            neighbourhood = previous[low_index:high_index]
            median = Vector((
                statistics.median(point[axis] for point in neighbourhood)
                for axis in range(2)
            ))
            residuals = [
                (point - median).length for point in neighbourhood
            ]
            residual_scale = max(
                statistics.median(residuals),
                total / max(len(previous), 1) * 0.35,
                1.0e-9,
            )
            weighted = Vector((0.0, 0.0))
            weight_total = 0.0
            for local_index, point in enumerate(neighbourhood):
                distance_from_center = abs(low_index + local_index - index)
                window_weight = float(radius + 1 - distance_from_center)
                huber_weight = min(
                    1.0,
                    residual_scale / max(residuals[local_index], residual_scale),
                )
                weight = window_weight * huber_weight
                weighted += point * weight
                weight_total += weight
            candidate = (
                weighted / weight_total
                if weight_total > 1.0e-12
                else previous[index].copy()
            )
            candidate = previous[index].lerp(candidate, pass_alpha)
            endpoint_chord = original[0].lerp(
                original[-1], index / float(len(original) - 1)
            )
            offset = candidate - endpoint_chord
            span = max(total * 0.20, 1.0e-9)
            if offset.length > span:
                candidate = endpoint_chord + offset.normalized() * span
            target[index] = candidate
        target[0] = clean[0].copy()
        target[-1] = clean[-1].copy()
    # Keep the compact segment count from the pre-outward implementation.
    # Manual knots remain soft observations rather than interpolation points.
    fit_count = max(4, min(12, int(total / 110.0) + 4))
    fit_distances = [
        total * index / float(fit_count - 1)
        for index in range(fit_count)
    ]
    fit_points = _guided_ridge_curve_interpolate_vectors(
        target, uniform_distances, fit_distances
    )
    tangents = []
    for index, point in enumerate(fit_points):
        if index == 0:
            tangent = fit_points[1] - point
        elif index == len(fit_points) - 1:
            tangent = point - fit_points[index - 1]
        else:
            tangent = (fit_points[index + 1] - fit_points[index - 1]) * 0.5
        previous_length = (
            (point - fit_points[index - 1]).length
            if index else tangent.length
        )
        next_length = (
            (fit_points[index + 1] - point).length
            if index + 1 < len(fit_points) else tangent.length
        )
        limit = max(min(previous_length, next_length) * 1.35, 1.0e-12)
        if tangent.length > limit:
            tangent = tangent.normalized() * limit
        tangents.append(tangent)
    return [
        (
            fit_points[index].copy(),
            fit_points[index] + tangents[index] / 3.0,
            fit_points[index + 1] - tangents[index + 1] / 3.0,
            fit_points[index + 1].copy(),
        )
        for index in range(len(fit_points) - 1)
    ]


def _guided_ridge_curve_2d_early_result(
    clean, cumulative, total, raw_oracle, parameters, signed
):
    """Return the raw/inscribed result before the positive waveform path.

    Shape zero and the restored historical negative construction intentionally
    do not build positive-wave lobes, normalization, or validation state.  This
    keeps their geometry and cost independent of the outward path.
    """
    raw = _guided_ridge_curve_interpolate_vectors(
        clean, cumulative, [float(value) * total for value in parameters]
    )
    chord_count = max(1, min(12, int(total / 110.0) + 4))
    chord_segments = _guided_ridge_curve_2d_endpoint_chord_segments(
        clean, chord_count
    )
    chord = _guided_ridge_curve_2d_bezier_samples(
        chord_segments, parameters
    ) if chord_segments else [point.copy() for point in raw]
    if signed < 0.0:
        historical_passes = min(5.0, max(1.0, 1.0 + abs(signed) * 4.0))
        historical_segments = _guided_ridge_curve_2d_historical_inscribed_fit(
            clean, cumulative, total, historical_passes
        )
        preview = _guided_ridge_curve_2d_bezier_samples(
            historical_segments, parameters
        ) if historical_segments else []
        status = "historical-inscribed"
        warning = None if preview else "inscribed fit unavailable"
        amplitude = max(
            (
                (preview[index] - raw[index]).length
                for index in range(1, min(len(raw), len(preview)) - 1)
            ),
            default=0.0,
        )
        validation = {
            "valid": bool(preview),
            "reason": warning,
            "construction": "pre-outward-3.3.53-screen-bezier",
            "smoothing_passes": historical_passes,
        }
        return {
            "raw": raw,
            "preview": preview,
            "smooth": [point.copy() for point in preview],
            "inscribed": [point.copy() for point in preview],
            "historical_inscribed": [point.copy() for point in preview],
            "historical_inscribed_bezier_segments": historical_segments,
            "historical_inscribed_passes": historical_passes,
            "chord": chord,
            "chord_bezier_segments": chord_segments,
            "detail_normalization": {},
            "raw_chord_oracle": raw_oracle,
            "lobe_intervals": [],
            "inscribed_bezier_segments": historical_segments,
            "bezier_segments": historical_segments,
            "smooth_bezier_segments": historical_segments,
            "preview_bezier_segments": historical_segments,
            "opposite": [point.copy() for point in preview],
            "emphasized_baseline": [],
            "opposite_bezier_segments": historical_segments,
            "opposite_amplitude": amplitude,
            "opposite_status": status,
            "opposite_warning": warning,
            "effective_smoothing": signed * 100.0,
            "wave_validation": validation,
            "parameters": list(parameters),
            "sample_count": len(preview),
            "fit_segment_count": len(historical_segments),
        }
    return {
        "raw": raw,
        "preview": [point.copy() for point in raw],
        "smooth": [point.copy() for point in raw],
        "inscribed": [point.copy() for point in raw],
        "historical_inscribed": [],
        "historical_inscribed_bezier_segments": [],
        "historical_inscribed_passes": 0.0,
        "chord": chord,
        "chord_bezier_segments": chord_segments,
        "detail_normalization": {},
        "raw_chord_oracle": raw_oracle,
        "lobe_intervals": [],
        "inscribed_bezier_segments": [],
        "bezier_segments": [],
        "smooth_bezier_segments": [],
        "preview_bezier_segments": [],
        "opposite": [],
        "emphasized_baseline": [],
        "opposite_bezier_segments": [],
        "opposite_amplitude": 0.0,
        "opposite_status": "neutral",
        "opposite_warning": None,
        "effective_smoothing": 0.0,
        "wave_validation": {"valid": True, "reason": None, "construction": "raw"},
        "parameters": list(parameters),
        "sample_count": len(raw),
        "fit_segment_count": 0,
    }


def _guided_ridge_curve_2d_bezier_fit(
    clean, cumulative, total, smoothing_passes=5, fit_count_override=None
):
    """Fit a small C1 cubic approximation from soft screen-space observations.

    The clicked interior knots are observations, not interpolation constraints.
    The fit is therefore assembled on a uniform arc-length grid, with a broad
    robust neighbourhood filter before the compact Bezier approximation.  This
    makes one manually placed corner/outlier influence the whole low-frequency
    shape instead of producing a cusp at that knot.  Only the two endpoints
    remain hard constraints.
    """
    if len(clean) < 2 or total <= 1.0e-12:
        return []
    # A sufficiently fine uniform grid makes the fit independent of the
    # original click spacing.  The generated curve is intentionally much
    # lower-order than the raw route, so it cannot chase every click.
    uniform_count = max(48, min(192, max(len(clean) * 8, 48)))
    uniform_distances = [
        total * index / float(uniform_count - 1)
        for index in range(uniform_count)
    ]
    uniform = _guided_ridge_curve_interpolate_vectors(
        clean, cumulative, uniform_distances
    )
    # Build a broad robust least-squares-like target.  A local coordinate
    # median is used only to estimate outlier residuals; the retained samples
    # are then a distance-weighted mean.  The fixed broad radius means a single
    # corner cannot become a high-weight interpolation constraint.  Repeating
    # the operation on the target itself gives a low-frequency curve while the
    # endpoints stay exact.
    original = [point.copy() for point in uniform]
    # Keep the smoothing envelope wide enough for the actual raw waveform.
    # A fixed total*0.20 cap clipped narrow/tall arches before the Shape gain
    # was applied, making their maximum preview smaller than the hand route.
    chord_delta = original[-1] - original[0]
    chord_length = chord_delta.length
    if chord_length > 1.0e-12:
        chord_normal = Vector((-chord_delta.y, chord_delta.x)).normalized()
        raw_envelope = max(
            abs((point - original[0].lerp(original[-1], index / float(len(original) - 1))).dot(chord_normal))
            for index, point in enumerate(original)
        )
    else:
        raw_envelope = max(
            (point - original[0]).length for point in original
        )
    envelope_span = max(raw_envelope * 1.10, total * 0.20, 1.0e-9)
    radius = max(4, min(18, int(len(uniform) * 0.065)))
    target = [point.copy() for point in uniform]
    for _iteration in range(max(0, int(smoothing_passes))):
        previous = [point.copy() for point in target]
        for index in range(1, len(previous) - 1):
            low_index = max(0, index - radius)
            high_index = min(len(previous), index + radius + 1)
            neighbourhood = previous[low_index:high_index]
            median = Vector((
                statistics.median(point[axis] for point in neighbourhood)
                for axis in range(2)
            ))
            residuals = [
                (point - median).length for point in neighbourhood
            ]
            residual_scale = max(
                statistics.median(residuals), total / max(len(previous), 1) * 0.35,
                1.0e-9,
            )
            weighted = Vector((0.0, 0.0))
            weight_total = 0.0
            for local_index, point in enumerate(neighbourhood):
                distance_from_center = abs(low_index + local_index - index)
                window_weight = float(radius + 1 - distance_from_center)
                huber_weight = min(1.0, residual_scale / max(residuals[local_index], residual_scale))
                weight = window_weight * huber_weight
                weighted += point * weight
                weight_total += weight
            if weight_total > 1.0e-12:
                candidate = weighted / weight_total
            else:
                candidate = previous[index].copy()
            # Blend gradually rather than snapping to a moving fit.  The
            # endpoint-to-endpoint chord envelope limits overshoot without
            # forcing the route through any interior observation.
            alpha = 0.72
            candidate = previous[index].lerp(candidate, alpha)
            endpoint_chord = original[0].lerp(original[-1], index / float(len(original) - 1))
            span = envelope_span
            offset = candidate - endpoint_chord
            if offset.length > span:
                candidate = endpoint_chord + offset.normalized() * span
            target[index] = candidate
        target[0] = clean[0].copy()
        target[-1] = clean[-1].copy()

    # Keep the generated approximation compact.  The number is intentionally
    # independent of manual knot count and never creates one segment per knot.
    fit_count = (
        max(4, int(fit_count_override))
        if fit_count_override is not None
        else max(
            4,
            min(
                12,
                # Keep the fitting grid uniform in arc length, while retaining
                # enough C1 degrees of freedom for independently meaningful
                # lobes.  The grid is not tied to knot positions: clicked
                # interiors remain soft observations, but an alternating
                # multi-lobe route must not be collapsed into one broad lobe
                # before the raw-knot oracle can compare it.
                max(int(total / 110.0) + 4, len(clean) + 1),
            ),
        )
    )
    fit_distances = [
        total * index / float(fit_count - 1)
        for index in range(fit_count)
    ]
    fit_points = _guided_ridge_curve_interpolate_vectors(
        target, uniform_distances, fit_distances
    )
    tangents = []
    for index, point in enumerate(fit_points):
        if index == 0:
            tangent = fit_points[1] - point
        elif index == len(fit_points) - 1:
            tangent = point - fit_points[index - 1]
        else:
            tangent = (fit_points[index + 1] - fit_points[index - 1]) * 0.5
        previous_length = (
            (point - fit_points[index - 1]).length if index else tangent.length
        )
        next_length = (
            (fit_points[index + 1] - point).length
            if index + 1 < len(fit_points) else tangent.length
        )
        limit = max(min(previous_length, next_length) * 1.35, 1.0e-12)
        if tangent.length > limit:
            tangent = tangent.normalized() * limit
        tangents.append(tangent)
    segments = []
    for index in range(len(fit_points) - 1):
        segments.append((
            fit_points[index].copy(),
            fit_points[index] + tangents[index] / 3.0,
            fit_points[index + 1] - tangents[index + 1] / 3.0,
            fit_points[index + 1].copy(),
        ))
    return segments


def _guided_ridge_curve_2d_bezier_samples(segments, parameters):
    if not segments:
        return []
    result = []
    segment_count = len(segments)
    for parameter in parameters:
        value = min(max(float(parameter), 0.0), 1.0)
        scaled = value * segment_count
        index = min(int(scaled), segment_count - 1)
        result.append(_guided_ridge_curve_2d_cubic_eval(
            segments[index], scaled - index
        ))
    return result


def _guided_ridge_curve_2d_progression_ok(raw, candidate):
    """Check candidate progression without collapsing it toward raw."""
    if len(raw) != len(candidate) or len(raw) <= 2:
        return False
    total = sum(
        (raw[index + 1] - raw[index]).length
        for index in range(len(raw) - 1)
    )
    route_tol = max(total * 1.0e-10, math.ulp(max(total, 1.0e-30)) * 16.0)
    for index in range(len(raw) - 1):
        raw_step = raw[index + 1] - raw[index]
        candidate_step = candidate[index + 1] - candidate[index]
        raw_length = raw_step.length
        candidate_length = candidate_step.length
        if raw_length <= route_tol:
            continue
        if candidate_length <= route_tol:
            return False
        cosine = raw_step.dot(candidate_step) / (raw_length * candidate_length)
        if cosine < 1.0e-5:
            return False
    return True


def _guided_ridge_curve_2d_is_straight(raw):
    """Classify only scale-relative collinear routes as neutral input."""
    if len(raw) <= 2:
        return True
    total = sum((raw[index + 1] - raw[index]).length for index in range(len(raw) - 1))
    if total <= 1.0e-30:
        return True
    chord = raw[-1] - raw[0]
    chord_length = chord.length
    scale = max(total, chord_length, math.ulp(max(total, 1.0e-300)) * 64.0)
    tolerance = max(scale * 1.0e-7, math.ulp(max(scale, 1.0e-300)) * 128.0)
    if chord_length <= tolerance:
        return max((point - raw[0]).length for point in raw) <= tolerance
    max_deviation = max(
        abs(chord.x * (point - raw[0]).y - chord.y * (point - raw[0]).x)
        / chord_length
        for point in raw[1:-1]
    )
    return max_deviation <= tolerance


def _guided_ridge_curve_2d_opposite_quality(raw, smooth, candidate):
    """Return deterministic signed/perpendicular evidence for an opposite fit."""
    if len(raw) != len(smooth) or len(raw) != len(candidate) or len(raw) <= 2:
        return {
            "valid": False,
            "opposite_dot": 0.0,
            "mean_perp": 0.0,
            "max_perp": 0.0,
            "local_opposite": False,
        }
    total = sum((raw[index + 1] - raw[index]).length for index in range(len(raw) - 1))
    scale = max(total, math.ulp(max(total, 1.0e-300)) * 64.0)
    tolerance = max(scale * 1.0e-7, math.ulp(max(scale, 1.0e-300)) * 128.0)
    opposite_dot = 0.0
    perpendicular = []
    local_opposite = False
    for index in range(1, len(raw) - 1):
        tangent = raw[index + 1] - raw[index - 1]
        if tangent.length <= tolerance:
            continue
        normal = Vector((-tangent.y, tangent.x)).normalized()
        positive = smooth[index] - raw[index]
        negative = candidate[index] - raw[index]
        opposite_dot += positive.dot(negative)
        positive_perp = positive.dot(normal)
        negative_perp = negative.dot(normal)
        perpendicular.append(abs(negative_perp))
        if abs(positive_perp) > tolerance and positive_perp * negative_perp < -tolerance * tolerance:
            local_opposite = True
    mean_perp = sum(perpendicular) / len(perpendicular) if perpendicular else 0.0
    max_perp = max(perpendicular) if perpendicular else 0.0
    euclidean_mean = sum(
        (candidate[index] - raw[index]).length
        for index in range(1, len(raw) - 1)
    ) / max(len(raw) - 2, 1)
    measurable = mean_perp > max(scale * 1.0e-5, tolerance * 4.0) or (
        euclidean_mean > max(scale * 1.0e-5, tolerance * 4.0)
    )
    opposite_threshold = max(scale * scale * 1.0e-8, tolerance * tolerance * 4.0)
    return {
        "valid": opposite_dot < -opposite_threshold and measurable and local_opposite,
        "opposite_dot": opposite_dot,
        "mean_perp": mean_perp,
        "max_perp": max_perp,
        "local_opposite": local_opposite,
    }


def _guided_ridge_curve_2d_opposite_bezier(
    raw, smooth, smooth_segments, cumulative, total, parameters
):
    """Build a single endpoint-only C1 arch for the positive emphasized side.

    Interior route points are used only to choose one global side and bulge;
    they are never interpolation constraints or individual Bezier controls.
    This keeps a sparse/outlier route from producing a knot-attracted cusp.
    """
    if _guided_ridge_curve_2d_is_straight(raw):
        return None, [], 0.0, "straight"
    if len(raw) < 3 or len(smooth) != len(raw):
        return None, [], 0.0, "failed"
    start = raw[0].copy()
    end = raw[-1].copy()
    chord = end - start
    chord_length = chord.length
    scale = max(
        chord_length,
        sum((raw[index + 1] - raw[index]).length for index in range(len(raw) - 1)),
        math.ulp(max(chord_length, 1.0e-300)) * 64.0,
    )
    if chord_length <= max(scale * 1.0e-7, math.ulp(scale) * 128.0):
        return None, [], 0.0, "failed"
    normal = Vector((-chord.y, chord.x)) / chord_length

    def signed_distances(points):
        return [(point - start).dot(normal) for point in points[1:-1]]

    smooth_offsets = signed_distances(smooth)
    raw_offsets = signed_distances(raw)
    tolerance = max(scale * 1.0e-7, math.ulp(scale) * 128.0)
    # Pick the raw route's dominant side only as a reference for the global
    # amplitude.  Ranking by absolute distance suppresses tiny near-chord noise.
    # If the strongest quarter contains comparable positive and negative mass,
    # this is an S/crossing route rather than a safe single-side arch.
    meaningful = [value for value in raw_offsets if abs(value) > tolerance]
    if not meaningful:
        return None, [], 0.0, "failed"
    ranked = sorted(meaningful, key=abs, reverse=True)
    top = ranked[:max(3, int(math.ceil(len(ranked) * 0.25)))]
    positive_mass = sum(abs(value) for value in top if value > 0.0)
    negative_mass = sum(abs(value) for value in top if value < 0.0)
    total_mass = positive_mass + negative_mass
    if (
        positive_mass > 0.0
        and negative_mass > 0.0
        and max(positive_mass, negative_mass) < total_mass * 0.68
    ):
        return None, [], 0.0, "failed"
    raw_side = 1.0 if positive_mass >= negative_mass else -1.0
    side_values = [value for value in raw_offsets if value * raw_side > tolerance]
    if not side_values:
        return None, [], 0.0, "failed"

    # Summarize the positive displacement S-R with one robust signed scalar.
    # The negative target is the opposite displacement from the raw reference:
    #   d_neg = d_raw + (d_raw - d_smooth) = d_raw - (S-R).
    # Interior points never become controls or interpolation constraints.
    magnitudes = sorted(abs(value) for value in side_values)
    percentile_index = min(
        len(magnitudes) - 1,
        max(0, int((len(magnitudes) - 1) * 0.75)),
    )
    raw_bulge = magnitudes[percentile_index] if magnitudes else 0.0
    delta_offsets = [
        smooth_offsets[index] - raw_offsets[index]
        for index in range(min(len(raw_offsets), len(smooth_offsets)))
    ]
    delta_values = [value for value in delta_offsets if abs(value) > tolerance]
    if not delta_values:
        return None, [], 0.0, "failed"
    delta_ranked = sorted(delta_values, key=abs, reverse=True)
    delta_top = delta_ranked[:max(3, int(math.ceil(len(delta_ranked) * 0.25)))]
    delta_positive = sum(abs(value) for value in delta_top if value > 0.0)
    delta_negative = sum(abs(value) for value in delta_top if value < 0.0)
    delta_total = delta_positive + delta_negative
    if (
        delta_positive > 0.0
        and delta_negative > 0.0
        and max(delta_positive, delta_negative) < delta_total * 0.68
    ):
        return None, [], 0.0, "failed"
    delta_side = 1.0 if delta_positive >= delta_negative else -1.0
    delta_side_values = [
        abs(value) for value in delta_values if value * delta_side > 0.0
    ]
    delta_magnitudes = sorted(delta_side_values)
    delta_index = min(
        len(delta_magnitudes) - 1,
        max(0, int((len(delta_magnitudes) - 1) * 0.75)),
    )
    delta_bulge = delta_magnitudes[delta_index]
    raw_signed = raw_side * raw_bulge
    delta_signed = delta_side * delta_bulge
    target_signed = raw_signed - delta_signed
    if abs(target_signed) <= max(tolerance, chord_length * 0.02):
        return None, [], 0.0, "failed"
    # Keep the opposite displacement visibly measurable while retaining a
    # chord-relative cap for sparse/outlier routes.
    target_signed = max(
        min(target_signed, chord_length * 0.75),
        -chord_length * 0.75,
    )
    target_side = 1.0 if target_signed >= 0.0 else -1.0
    bulge = abs(target_signed)
    control_offset = min(bulge / 0.75, chord_length * 1.0)
    control_side = normal * (target_side * control_offset)
    segments = [(
        start,
        start + chord / 3.0 + control_side,
        end - chord / 3.0 + control_side,
        end,
    )]
    candidate = _guided_ridge_curve_2d_bezier_samples(segments, parameters)
    if not candidate:
        return None, [], 0.0, "failed"
    candidate[0] = start.copy()
    candidate[-1] = end.copy()
    # Validate the actual tessellation.  One segment has no interior join and
    # the symmetric handles provide a bounded, smooth arch.
    candidate_offsets = signed_distances(candidate)
    tolerance = max(scale * 1.0e-8, math.ulp(scale) * 64.0)
    # Validate the actual tessellation against the robust positive displacement
    # summary.  The negative vector must point opposite S-R from the raw route.
    candidate_displacements = [
        candidate_offsets[index] - raw_offsets[index]
        for index in range(min(len(candidate_offsets), len(raw_offsets)))
    ]
    displacement_dot = sum(
        delta_offsets[index] * candidate_displacements[index]
        for index in range(min(len(delta_offsets), len(candidate_displacements)))
    )
    opposite_count = sum(
        1 for value in candidate_displacements if abs(value) > tolerance
    )
    mean_opposite = sum(abs(value) for value in candidate_displacements) / max(
        len(candidate_displacements), 1
    )
    if (
        opposite_count < max(1, len(candidate_offsets) // 8)
        or mean_opposite <= tolerance
        or displacement_dot >= -max(scale * scale * 1.0e-8, tolerance * tolerance)
    ):
        return None, [], 0.0, "failed"
    return candidate, segments, 1.0, "valid"


def _guided_ridge_curve_2d_emphasized_baseline(raw, segments, parameters):
    """Return a near-raw, endpoint-only C1 arch for Shape +1.

    The baseline is not a raw-polyline blend.  It reuses the validated
    emphasized segment's handles, scales their chord-relative displacement to
    a small amount beyond the robust raw bulge, and therefore remains a single
    smooth Bezier curve even at the first nonzero positive step.
    """
    if len(raw) < 3 or len(segments) != 1:
        return []
    start = raw[0].copy()
    end = raw[-1].copy()
    chord = end - start
    chord_length = chord.length
    if chord_length <= 1.0e-12:
        return []
    normal = Vector((-chord.y, chord.x)) / chord_length
    raw_offsets = [(point - start).dot(normal) for point in raw[1:-1]]
    scale = max(
        chord_length,
        sum((raw[index + 1] - raw[index]).length for index in range(len(raw) - 1)),
    )
    tolerance = max(scale * 1.0e-7, math.ulp(max(scale, 1.0e-300)) * 128.0)
    meaningful = [value for value in raw_offsets if abs(value) > tolerance]
    if not meaningful:
        return []
    ranked = sorted(meaningful, key=abs, reverse=True)
    top = ranked[:max(3, int(math.ceil(len(ranked) * 0.25)))]
    positive_mass = sum(abs(value) for value in top if value > 0.0)
    negative_mass = sum(abs(value) for value in top if value < 0.0)
    if positive_mass and negative_mass and max(positive_mass, negative_mass) < (
        (positive_mass + negative_mass) * 0.68
    ):
        return []
    side = 1.0 if positive_mass >= negative_mass else -1.0
    side_values = [abs(value) for value in meaningful if value * side > tolerance]
    if not side_values:
        return []
    raw_bulge = sorted(side_values)[min(len(side_values) - 1, max(0, int((len(side_values) - 1) * 0.75)))]
    full = segments[0]
    full_points = _guided_ridge_curve_2d_bezier_samples(segments, parameters)
    full_offsets = [(point - start).dot(normal) for point in full_points[1:-1]]
    full_side_values = [abs(value) for value in full_offsets if value * side > tolerance]
    if not full_side_values:
        return []
    full_bulge = sorted(full_side_values)[
        min(len(full_side_values) - 1, max(0, int((len(full_side_values) - 1) * 0.75)))
    ]
    # Stay just outside the robust raw envelope, but never exceed the already
    # validated full emphasized target.
    target_magnitude = min(full_bulge, max(raw_bulge * 1.01, tolerance * 8.0))
    if target_magnitude <= tolerance:
        return []
    ratio = min(1.0, target_magnitude / max(full_bulge, tolerance))
    line_one = start.lerp(end, 1.0 / 3.0)
    line_two = start.lerp(end, 2.0 / 3.0)
    baseline = (
        start,
        line_one + (full[1] - line_one) * ratio,
        line_two + (full[2] - line_two) * ratio,
        end,
    )
    return _guided_ridge_curve_2d_bezier_samples([baseline], parameters)


def _guided_ridge_curve_2d_endpoint_chord_segments(clean, segment_count):
    """Return a C1 cubic representation of the endpoint chord.

    The segment layout matches the compact detail fit so the two control-space
    paths can be blended without reintroducing manual-knot constraints.
    """
    if len(clean) < 2 or int(segment_count) < 1:
        return []
    start = clean[0].copy()
    end = clean[-1].copy()
    count = int(segment_count)
    result = []
    for index in range(count):
        first = start.lerp(end, index / float(count))
        last = start.lerp(end, (index + 1) / float(count))
        delta = last - first
        result.append((
            first,
            first + delta / 3.0,
            first + delta * (2.0 / 3.0),
            last,
        ))
    return result


def _guided_ridge_curve_2d_normalize_detail_segments(
    baseline_segments, detail_segments, raw, detail, parameters,
    raw_oracle=None, lobe_intervals=None
):
    """Preserve the measured raw chord envelope in the cleaned detail fit.

    The fit remains soft and C1, but its broad lobe is not allowed to collapse
    solely because repeated robust smoothing used a fixed absolute span.  A
    single scale factor is derived from the largest stable raw/detail lobe
    ratio; using one factor for all lobes keeps shared Bezier joins C1 and
    preserves S/multi-wave signs.  The route/bbox-relative cap is only a
    runaway guard, not a fixed world-space threshold.
    """
    if (
        not baseline_segments
        or len(baseline_segments) != len(detail_segments or ())
        or len(raw) != len(detail)
        or len(raw) != len(parameters)
        or len(raw) <= 2
    ):
        return detail_segments, {"factor": 1.0, "raw_amplitude": 0.0, "detail_amplitude": 0.0}
    start = raw[0]
    end = raw[-1]
    chord = end - start
    chord_length = chord.length
    if chord_length <= 1.0e-12:
        return detail_segments, {"factor": 1.0, "raw_amplitude": 0.0, "detail_amplitude": 0.0}
    normal = Vector((-chord.y, chord.x)).normalized()
    raw_wave = []
    detail_wave = []
    for index, parameter in enumerate(parameters):
        chord_point = start.lerp(end, float(parameter))
        raw_wave.append((raw[index] - chord_point).dot(normal))
        detail_wave.append((detail[index] - chord_point).dot(normal))
    measurement_parameters = sorted({
        float(parameter) for parameter in parameters
    } | {
        float(parameter)
        for parameter in (raw_oracle or {}).get("parameters", ())
    })
    measurement_detail = _guided_ridge_curve_2d_bezier_samples(
        detail_segments, measurement_parameters
    ) if measurement_parameters else []
    measurement_wave = [
        (
            point - start.lerp(end, float(parameter))
        ).dot(normal)
        for point, parameter in zip(measurement_detail, measurement_parameters)
    ]
    route_scale = max(
        chord_length,
        sum((raw[index + 1] - raw[index]).length for index in range(len(raw) - 1)),
        1.0e-12,
    )
    noise = max(route_scale * 1.0e-5, math.ulp(route_scale) * 64.0)
    oracle_offsets = tuple((raw_oracle or {}).get("offsets", ()))
    oracle_noise = float((raw_oracle or {}).get("noise", noise))
    raw_amplitudes = []
    detail_amplitudes = []
    side_factors = {}
    for side in (-1.0, 1.0):
        raw_amplitude = max(
            (
                abs(value)
                for value in (oracle_offsets or raw_wave)
                if value * side > (oracle_noise if oracle_offsets else noise)
            ),
            default=0.0,
        )
        detail_amplitude = max(
            (abs(value) for value in detail_wave if value * side > noise),
            default=0.0,
        )
        if raw_amplitude > noise and detail_amplitude > noise:
            raw_amplitudes.append(raw_amplitude)
            detail_amplitudes.append(detail_amplitude)
            side_factors[side] = raw_amplitude / max(detail_amplitude, noise)
    if not raw_amplitudes or not detail_amplitudes:
        return detail_segments, {"factor": 1.0, "raw_amplitude": 0.0, "detail_amplitude": 0.0}
    raw_reference = max(raw_amplitudes)
    detail_reference = max(detail_amplitudes)
    # Normalize each meaningful signed lobe to the measured raw envelope.  The
    # Shape gain is intentionally raw-relative: +1 is approximately 1.01x,
    # +100 is approximately 2x, while negative values attenuate toward the
    # endpoint trend.  Keeping the reference at 1.0 avoids labeling a still
    # smaller-than-raw result as "Amplified".
    envelope_fraction = 1.0
    factor = envelope_fraction * max(side_factors.values())
    factor = max(0.25, factor)
    # Keep the amplitude correction bounded by the actual route scale.  A
    # pathological tiny detail field will fail the normal validator honestly
    # instead of becoming an enormous screen-space excursion.
    factor = min(factor, max(4.0, route_scale / max(raw_reference, noise)))
    for side, side_factor in tuple(side_factors.items()):
        side_factors[side] = min(
            max(0.25, envelope_fraction * side_factor),
            max(4.0, route_scale / max(raw_reference, noise)),
        )

    # Preserve each meaningful lobe independently.  A global maximum is not
    # sufficient for an S/multi-wave route: a smaller, separated lobe must not
    # be normalized away by a taller one.  The interval came from the cleaned
    # uniform waveform, while raw amplitude is measured from every original
    # knot through ``raw_oracle``.
    lobe_factors = []
    for interval in lobe_intervals or ():
        side = float(interval.get("side", 0.0))
        raw_lobe = _guided_ridge_curve_2d_lobe_amplitude(
            raw_oracle or {}, interval
        )
        detail_lobe = max(
            (
                abs(value)
                for value, parameter in zip(measurement_wave, measurement_parameters)
                if float(interval.get("start", 0.0)) - 1.0e-9
                <= float(parameter)
                <= float(interval.get("end", 1.0)) + 1.0e-9
                and value * side > noise
            ),
            default=0.0,
        )
        if raw_lobe > noise and detail_lobe > noise:
            lobe_factors.append({
                "start": float(interval.get("start", 0.0)),
                "end": float(interval.get("end", 1.0)),
                "factor": min(
                    max(0.25, raw_lobe / max(detail_lobe, noise)),
                    max(4.0, route_scale / max(raw_reference, noise)),
                ),
            })

    def _factor_at(parameter, wave_value, fallback):
        matches = [
            item for item in lobe_factors
            if item["start"] - 1.0e-9 <= parameter <= item["end"] + 1.0e-9
        ]
        if matches:
            return min(matches, key=lambda item: item["end"] - item["start"])["factor"]
        if abs(wave_value) > noise:
            return side_factors.get(1.0 if wave_value > 0.0 else -1.0, fallback)
        return fallback

    def _join_factor(parameter, fallback):
        """Choose a per-lobe factor without breaking a Bezier join."""
        if not detail_wave or not parameters:
            return fallback
        nearest = min(
            range(len(parameters)),
            key=lambda index: abs(float(parameters[index]) - float(parameter)),
        )
        wave_value = detail_wave[nearest]
        if abs(wave_value) <= noise:
            return fallback
        return _factor_at(float(parameter), wave_value, fallback)
    normalized = []
    segment_count = len(detail_segments)
    for segment_index, (baseline_segment, detail_segment) in enumerate(
        zip(baseline_segments, detail_segments)
    ):
        controls = []
        for control_index, detail_point in enumerate(detail_segment):
            local = control_index / 3.0
            parameter = (segment_index + local) / float(segment_count)
            baseline_point = _guided_ridge_curve_2d_cubic_eval(
                baseline_segment, local
            )
            offset = detail_point - baseline_point
            if control_index == 1:
                control_factor = _join_factor(
                    segment_index / float(segment_count), factor
                )
            elif control_index == 2:
                control_factor = _join_factor(
                    (segment_index + 1) / float(segment_count), factor
                )
            else:
                control_factor = _factor_at(parameter, offset.dot(normal), factor)
            perpendicular = normal * offset.dot(normal) * control_factor
            tangential = offset - normal * offset.dot(normal)
            controls.append(baseline_point + tangential + perpendicular)
        normalized.append(tuple(controls))
    # Per-lobe scale selection can give neighbouring segments slightly
    # different normal factors. Reconcile each shared cubic tangent once so
    # the normalized detail remains genuinely C1 before the Shape gain is
    # applied; this is a control-space correction, not a raw-knot constraint.
    for join_index in range(len(normalized) - 1):
        first_segment = list(normalized[join_index])
        second_segment = list(normalized[join_index + 1])
        outgoing = first_segment[3] - first_segment[2]
        incoming = second_segment[1] - second_segment[0]
        tangent = (outgoing + incoming) * 0.5
        first_segment[2] = first_segment[3] - tangent
        second_segment[1] = second_segment[0] + tangent
        normalized[join_index] = tuple(first_segment)
        normalized[join_index + 1] = tuple(second_segment)
    # The join reconciliation can move a lobe by a few pixels when adjacent
    # lobes use different factors. Measure the resulting C1 samples once and
    # apply a bounded per-lobe correction so the shared raw-relative reference
    # is true after, rather than before, control-space reconciliation.
    max_factor = max(4.0, route_scale / max(raw_reference, noise))
    # Per-lobe corrections are followed through several bounded control-space
    # passes because C1 join reconciliation can otherwise leave a small lobe a
    # fraction below its all-knot raw reference.
    for _correction_pass in range(4):
        sampled = _guided_ridge_curve_2d_bezier_samples(
            normalized, measurement_parameters
        )
        sampled_wave = [
            (point - start.lerp(end, float(measurement_parameters[index]))).dot(normal)
            for index, point in enumerate(sampled)
        ]
        corrections = {}
        if lobe_intervals and oracle_offsets:
            for interval in lobe_intervals:
                side = float(interval.get("side", 0.0))
                start_parameter = float(interval.get("start", 0.0))
                end_parameter = float(interval.get("end", 1.0))
                raw_lobe = _guided_ridge_curve_2d_lobe_amplitude(
                    raw_oracle or {}, interval
                )
                sampled_lobe = max(
                    (
                        abs(value)
                        for value, parameter in zip(sampled_wave, measurement_parameters)
                        if start_parameter - 1.0e-9 <= float(parameter) <= end_parameter + 1.0e-9
                        and value * side > noise
                    ),
                    default=0.0,
                )
                if raw_lobe > noise and sampled_lobe > noise:
                    corrections[(start_parameter, end_parameter)] = min(
                        max(raw_lobe / sampled_lobe, 0.25), max_factor
                    )
        else:
            for side in (-1.0, 1.0):
                raw_lobe = max(
                    (abs(value) for value in raw_wave if value * side > noise),
                    default=0.0,
                )
                sampled_lobe = max(
                    (abs(value) for value in sampled_wave if value * side > noise),
                    default=0.0,
                )
                if raw_lobe > noise and sampled_lobe > noise:
                    corrections[side] = min(
                        max(raw_lobe / sampled_lobe, 0.25), max_factor
                    )
        if not corrections or all(abs(value - 1.0) <= 1.0e-4 for value in corrections.values()):
            break
        for segment_index, segment in enumerate(normalized):
            controls = []
            for control_index, point in enumerate(segment):
                local = control_index / 3.0
                # Recompute the normalized control parameter for every
                # correction pass.  Reusing the prior loop's final value
                # silently assigned one lobe factor to all controls.
                parameter = (segment_index + local) / float(segment_count)
                baseline_point = _guided_ridge_curve_2d_cubic_eval(
                    baseline_segments[segment_index], local
                )
                offset = point - baseline_point
                wave_value = offset.dot(normal)
                if abs(wave_value) <= noise:
                    correction = 1.0
                elif lobe_intervals and oracle_offsets:
                    correction = 1.0
                    matches = []
                    for interval in lobe_intervals:
                        start_parameter = float(interval.get("start", 0.0))
                        end_parameter = float(interval.get("end", 1.0))
                        if start_parameter - 1.0e-9 <= parameter <= end_parameter + 1.0e-9:
                            matches.append((
                                end_parameter - start_parameter,
                                corrections.get((start_parameter, end_parameter), 1.0),
                            ))
                    if matches:
                        correction = min(matches, key=lambda item: item[0])[1]
                else:
                    correction = corrections.get(
                        1.0 if wave_value > noise else -1.0,
                        1.0,
                    )
                tangential = offset - normal * wave_value
                controls.append(
                    baseline_point + tangential + normal * wave_value * correction
                )
            normalized[segment_index] = tuple(controls)
        for join_index in range(len(normalized) - 1):
            first_segment = list(normalized[join_index])
            second_segment = list(normalized[join_index + 1])
            tangent = (
                (first_segment[3] - first_segment[2])
                + (second_segment[1] - second_segment[0])
            ) * 0.5
            first_segment[2] = first_segment[3] - tangent
            second_segment[1] = second_segment[0] + tangent
            normalized[join_index] = tuple(first_segment)
            normalized[join_index + 1] = tuple(second_segment)
    return normalized, {
        "factor": factor,
        "raw_amplitude": raw_reference,
        "detail_amplitude": detail_reference,
    }


def _guided_ridge_curve_2d_wave_family(raw, baseline, detail, parameters, signed):
    """Evaluate the signed, band-limited Shape family in screen space.

    ``detail`` is a near-raw but already C1 Bezier approximation and
    ``baseline`` is the straight endpoint trend. Shape is a gain around that
    global trend: -100 reaches the trend, 0 is reserved for the exact raw
    polyline, and +100 doubles the cleaned waveform displacement. A smoothed
    copy of a long arc is deliberately not used as the baseline because it
    would absorb the very shape the user is asking to amplify.
    """
    if not raw or len(raw) != len(baseline) or len(raw) != len(detail):
        return None, "failed"
    try:
        gain = 1.0 + min(max(float(signed), -1.0), 1.0)
    except (TypeError, ValueError):
        return None, "failed"
    if signed > 0.0:
        # A tiny positive request must be visibly and numerically outside every
        # raw knot lobe, not only the largest one.  The bounded epsilon covers
        # the final C1 join/tessellation loss on small secondary lobes; it fades
        # to zero at +100 so the documented 2x endpoint remains exact.
        gain += 0.015 * (1.0 - min(max(float(signed), 0.0), 1.0))
    candidate = [
        baseline[index] + (detail[index] - baseline[index]) * gain
        for index in range(len(raw))
    ]
    candidate[0] = raw[0].copy()
    candidate[-1] = raw[-1].copy()
    if not all(
        all(math.isfinite(float(value)) for value in point)
        for point in candidate
    ):
        return None, "failed"
    return candidate, "valid"


def _guided_ridge_curve_2d_wave_segments(baseline_segments, detail_segments, signed):
    """Blend the two C1 Bezier fits in control space for one Shape gain."""
    if not baseline_segments or len(baseline_segments) != len(detail_segments or ()):
        return []
    try:
        gain = 1.0 + min(max(float(signed), -1.0), 1.0)
    except (TypeError, ValueError):
        return []
    if signed > 0.0:
        gain += 0.015 * (1.0 - min(max(float(signed), 0.0), 1.0))
    result = []
    for baseline, detail in zip(baseline_segments, detail_segments):
        if len(baseline) != 4 or len(detail) != 4:
            return []
        result.append(tuple(
            baseline[index] + (detail[index] - baseline[index]) * gain
            for index in range(4)
        ))
    return result


def _guided_ridge_curve_2d_segments_cross(a, b, c, d, tolerance):
    """Detect a proper or collinear 2D segment intersection."""
    def cross(first, second):
        return first.x * second.y - first.y * second.x

    def orient(first, second, third):
        return cross(second - first, third - first)

    def within(value, low, high):
        return low - tolerance <= value <= high + tolerance

    if (
        max(a.x, b.x) < min(c.x, d.x) - tolerance
        or max(c.x, d.x) < min(a.x, b.x) - tolerance
        or max(a.y, b.y) < min(c.y, d.y) - tolerance
        or max(c.y, d.y) < min(a.y, b.y) - tolerance
    ):
        return False
    first = orient(a, b, c)
    second = orient(a, b, d)
    third = orient(c, d, a)
    fourth = orient(c, d, b)
    if (
        ((first > tolerance and second < -tolerance) or (first < -tolerance and second > tolerance))
        and ((third > tolerance and fourth < -tolerance) or (third < -tolerance and fourth > tolerance))
    ):
        return True
    if abs(first) <= tolerance and within(c.x, min(a.x, b.x), max(a.x, b.x)) and within(c.y, min(a.y, b.y), max(a.y, b.y)):
        return True
    if abs(second) <= tolerance and within(d.x, min(a.x, b.x), max(a.x, b.x)) and within(d.y, min(a.y, b.y), max(a.y, b.y)):
        return True
    if abs(third) <= tolerance and within(a.x, min(c.x, d.x), max(c.x, d.x)) and within(a.y, min(c.y, d.y), max(c.y, d.y)):
        return True
    if abs(fourth) <= tolerance and within(b.x, min(c.x, d.x), max(c.x, d.x)) and within(b.y, min(c.y, d.y), max(c.y, d.y)):
        return True
    return False


def _guided_ridge_curve_2d_wave_validate(
    raw, baseline, detail, candidate, segments, signed,
    raw_oracle=None, lobe_intervals=None, parameters=None
):
    """Validate the actual tessellation/control joins of a wave candidate."""
    if (
        not raw
        or len(raw) != len(baseline)
        or len(raw) != len(detail)
        or len(raw) != len(candidate)
        or len(candidate) < 3
    ):
        return {"valid": False, "reason": "shape sample count mismatch"}
    route_length = sum(
        (raw[index + 1] - raw[index]).length for index in range(len(raw) - 1)
    )
    scale = max(route_length, math.ulp(max(route_length, 1.0e-300)) * 128.0)
    tolerance = max(scale * 1.0e-8, math.ulp(max(scale, 1.0e-300)) * 64.0)
    for point in candidate:
        if not all(math.isfinite(float(value)) for value in point):
            return {"valid": False, "reason": "non-finite wave sample"}
    min_tangent = float("inf")
    for index in range(len(candidate) - 1):
        step = candidate[index + 1] - candidate[index]
        min_tangent = min(min_tangent, step.length)
        if step.length <= tolerance:
            return {"valid": False, "reason": "wave tangent collapsed"}
        reference = baseline[index + 1] - baseline[index]
        if reference.length > tolerance:
            cosine = step.dot(reference) / (step.length * reference.length)
            if cosine <= -1.0e-6:
                return {"valid": False, "reason": "wave progression folded back"}

    # At most 512 tessellation samples are produced, so a bounded pairwise
    # check is deterministic and avoids accepting a new 2D self-intersection.
    intersection_tolerance = max(scale * 1.0e-9, tolerance)
    intersection_checks = 0
    for first_index in range(len(candidate) - 1):
        for second_index in range(first_index + 2, len(candidate) - 1):
            intersection_checks += 1
            if _guided_ridge_curve_2d_segments_cross(
                candidate[first_index], candidate[first_index + 1],
                candidate[second_index], candidate[second_index + 1],
                intersection_tolerance,
            ):
                return {
                    "valid": False,
                    "reason": "wave self-intersection",
                    "intersection_checks": intersection_checks,
                }

    expected_gain = 1.0 + float(signed)
    lobe_count = 0
    lobe_error = 0.0
    for index in range(1, len(candidate) - 1):
        tangent = baseline[index + 1] - baseline[index - 1]
        if tangent.length <= tolerance:
            continue
        normal = Vector((-tangent.y, tangent.x)).normalized()
        wave = (detail[index] - baseline[index]).dot(normal)
        delta = (candidate[index] - baseline[index]).dot(normal)
        if abs(wave) <= tolerance:
            continue
        lobe_count += 1
        ratio = delta / wave
        # Cubic tessellation and the baseline tangent field introduce a small
        # screen-space projection error; validate the expected gain relatively
        # rather than rejecting a valid lobe for sub-pixel noise.
        ratio_tolerance = max(3.0e-3, abs(expected_gain) * 3.0e-3)
        if signed > 0.0:
            if wave * delta <= 0.0 or ratio < expected_gain - ratio_tolerance:
                return {"valid": False, "reason": "amplified lobe changed orientation"}
        elif signed < 0.0:
            if wave * delta < -tolerance * tolerance or ratio < -ratio_tolerance or ratio > expected_gain + ratio_tolerance:
                return {"valid": False, "reason": "attenuated lobe changed orientation"}
        elif abs(delta) > tolerance:
            return {"valid": False, "reason": "neutral wave is not raw"}
        lobe_error = max(lobe_error, abs(ratio - expected_gain))
    if abs(signed) > 1.0e-12 and lobe_count == 0:
        # A straight/degenerate route is valid only when all wave displacement
        # is genuinely zero; nonzero gain must not silently invent a lobe.
        straight_tolerance = max(scale * 1.0e-6, tolerance * 16.0)
        if any(
            (detail[index] - baseline[index]).length > straight_tolerance
            for index in range(1, len(detail) - 1)
        ):
            return {"valid": False, "reason": "wave lobe orientation unavailable"}

    # Validate the final drawn samples against the same endpoint-chord metric
    # used by detail normalization.  A candidate is not allowed to claim
    # Amplified/Attenuated when its measured raw-relative envelope is on the
    # wrong side after any safety/backoff step.  Compare each meaningful signed
    # lobe independently so S and multi-wave routes retain their local intent.
    chord = raw[-1] - raw[0]
    if abs(signed) > 1.0e-12 and chord.length > tolerance:
        chord_normal = Vector((-chord.y, chord.x)).normalized()
        raw_offsets = []
        candidate_offsets = []
        sample_parameters = parameters or [
            index / float(len(raw) - 1) for index in range(len(raw))
        ]
        # Candidate amplitudes are measured from the bounded polyline that is
        # actually drawn.  Do not add hidden raw-knot parameters or re-tessellate
        # stored Bezier segments here: an unseen control-space peak cannot make
        # an under-amplified display pass validation.
        measurement_parameters = [
            float(parameter) for parameter in sample_parameters
        ]
        measurement_points = list(candidate)
        for index in range(1, len(raw) - 1):
            parameter = float(sample_parameters[index])
            chord_point = raw[0].lerp(raw[-1], parameter)
            candidate_offsets.append((candidate[index] - chord_point).dot(chord_normal))
        measurement_offsets = [
            (
                point - raw[0].lerp(raw[-1], float(parameter))
            ).dot(chord_normal)
            for point, parameter in zip(measurement_points, measurement_parameters)
        ]
        oracle_offsets = tuple((raw_oracle or {}).get("offsets", ()))
        if oracle_offsets:
            raw_offsets = list(oracle_offsets)
        else:
            raw_offsets = [
                (raw[index] - raw[0].lerp(raw[-1], float(sample_parameters[index]))).dot(chord_normal)
                for index in range(1, len(raw) - 1)
            ]
        envelope_noise = max(scale * 1.0e-5, tolerance * 8.0)
        intervals = lobe_intervals or ()
        def _drawn_interval_amplitude(start, end, side):
            # Signed chord distance is linear along each displayed segment.
            # Include interpolated interval boundaries so sparse draw samples
            # cannot hide a lobe extremum between samples.
            points = []
            for parameter, value in zip(measurement_parameters, measurement_offsets):
                if start - 1.0e-9 <= parameter <= end + 1.0e-9:
                    points.append(float(value))
            for boundary in (start, end):
                if boundary <= measurement_parameters[0]:
                    points.append(float(measurement_offsets[0]))
                    continue
                if boundary >= measurement_parameters[-1]:
                    points.append(float(measurement_offsets[-1]))
                    continue
                for left_index in range(len(measurement_parameters) - 1):
                    left_parameter = measurement_parameters[left_index]
                    right_parameter = measurement_parameters[left_index + 1]
                    if left_parameter <= boundary <= right_parameter:
                        span = max(right_parameter - left_parameter, 1.0e-12)
                        alpha = (boundary - left_parameter) / span
                        points.append(
                            measurement_offsets[left_index]
                            + (
                                measurement_offsets[left_index + 1]
                                - measurement_offsets[left_index]
                            ) * alpha
                        )
                        break
            return max(
                (
                    abs(value) for value in points
                    if value * side > envelope_noise
                ),
                default=0.0,
            )
        if intervals and oracle_offsets:
            raw_parameters = tuple((raw_oracle or {}).get("parameters", ()))
            for interval in intervals:
                side = float(interval.get("side", 0.0))
                start = float(interval.get("start", 0.0))
                end = float(interval.get("end", 1.0))
                raw_lobe = max(
                    (
                        abs(value)
                        for parameter, value in zip(raw_parameters, oracle_offsets)
                        if start - 1.0e-9 <= parameter <= end + 1.0e-9
                        and value * side > envelope_noise
                    ),
                    default=0.0,
                )
                candidate_lobe = _drawn_interval_amplitude(start, end, side)
                if raw_lobe <= envelope_noise:
                    continue
                margin = max(raw_lobe * 1.0e-4, envelope_noise)
                if signed > 0.0 and candidate_lobe <= raw_lobe + margin:
                    return {"valid": False, "reason": "amplified lobe did not exceed raw"}
                if signed < 0.0 and candidate_lobe >= raw_lobe - margin:
                    return {"valid": False, "reason": "attenuated lobe did not fall below raw"}
        else:
            for side in (-1.0, 1.0):
                raw_lobe = max(
                    (abs(value) for value in raw_offsets if value * side > envelope_noise),
                    default=0.0,
                )
                candidate_lobe = max(
                    (abs(value) for value in candidate_offsets if value * side > envelope_noise),
                    default=0.0,
                )
                if raw_lobe <= envelope_noise:
                    continue
                margin = max(raw_lobe * 1.0e-4, envelope_noise)
                if signed > 0.0 and candidate_lobe <= raw_lobe + margin:
                    return {"valid": False, "reason": "amplified envelope did not exceed raw"}
                if signed < 0.0 and candidate_lobe >= raw_lobe - margin:
                    return {"valid": False, "reason": "attenuated envelope did not fall below raw"}

    join_error = 0.0
    join_angle = 0.0
    if segments:
        for index in range(len(segments) - 1):
            outgoing = segments[index][3] - segments[index][2]
            incoming = segments[index + 1][1] - segments[index + 1][0]
            if outgoing.length <= tolerance or incoming.length <= tolerance:
                return {"valid": False, "reason": "Bezier join tangent collapsed"}
            join_error = max(join_error, (outgoing - incoming).length)
            join_angle = max(
                join_angle,
                math.acos(max(-1.0, min(1.0, outgoing.normalized().dot(incoming.normalized())))),
            )
            # Blender's float-backed Vector normalization quantizes very small
            # join angles (typically below 0.001 rad); reject only a visible
            # tangent break while still recording the measured C1 error.
            if join_error > scale * 1.0e-6 or join_angle > 1.0e-2:
                return {"valid": False, "reason": "Bezier join is not C1"}
    return {
        "valid": True,
        "reason": None,
        "min_tangent": min_tangent,
        "join_error": join_error,
        "join_angle": join_angle,
        "lobe_count": lobe_count,
        "lobe_error": lobe_error,
        "intersection_count": 0,
        "intersection_checks": intersection_checks,
        "expected_gain": expected_gain,
    }


def _guided_ridge_curve_2d_preview_points(points, smoothing=50.0):
    """Create the current-view, knot-inclusive 2D Bezier preview."""
    clean, cumulative, total = _guided_ridge_curve_2d_arc_data(points)
    if len(clean) <= 1 or total <= 1.0e-12:
        return {
            "raw": [point.copy() for point in clean],
            "preview": [point.copy() for point in clean],
            "smooth": [point.copy() for point in clean],
            "bezier_segments": [],
            "opposite": [],
            "opposite_bezier_segments": [],
            "opposite_amplitude": 0.0,
            "opposite_status": "straight",
            "opposite_warning": None,
            "sample_count": len(clean),
        }
    raw_oracle = _guided_ridge_curve_2d_raw_chord_oracle(
        clean, cumulative, total
    )
    # Pixel-length driven tessellation, capped to keep redraw work bounded.
    sample_count = max(64, min(512, int(total / 3.0) + len(clean) * 2))
    try:
        signed = min(max(float(smoothing), -100.0), 100.0) / 100.0
    except (TypeError, ValueError):
        signed = 0.0
    # Shape 0 deliberately retains every original knot.  Generated nonzero
    # previews use only the bounded uniform grid; raw knots remain soft input
    # to the fit and are not reintroduced into validation/drawing parameters.
    if abs(signed) > 1.0e-12:
        distances = _guided_ridge_curve_bounded_parameters(total, sample_count)
    else:
        distances = _guided_ridge_curve_output_parameters(total, cumulative, sample_count)
    parameters = [distance / total for distance in distances]
    raw = _guided_ridge_curve_interpolate_vectors(clean, cumulative, distances)
    if signed < 0.0 or abs(signed) <= 1.0e-12:
        return _guided_ridge_curve_2d_early_result(
            clean, cumulative, total, raw_oracle, parameters, signed
        )
    segments = _guided_ridge_curve_2d_bezier_fit(clean, cumulative, total)
    smooth = _guided_ridge_curve_2d_bezier_samples(segments, parameters)
    # Keep a second, still-C1 target close to the raw route.  Nonzero Shape
    # values must never linearly reintroduce the angular raw polyline: all
    # interpolation below is between Bezier-generated samples.  The one-pass
    # target is the near-raw C1 detail field, while the regular target is the
    # low-frequency baseline used for Shape -100.
    inscribed_segments = _guided_ridge_curve_2d_bezier_fit(
        clean, cumulative, total, smoothing_passes=1
    )
    inscribed = _guided_ridge_curve_2d_bezier_samples(
        inscribed_segments, parameters
    )
    # The negative/inscribed side is deliberately based on the original
    # pre-outward screen-space construction.  Its pass count varies only the
    # strength of that same robust arc filter; it never enters the positive
    # waveform/amplification path.
    historical_inscribed_passes = max(
        1, min(5, 1 + int(round(abs(signed) * 4.0)))
    )
    historical_inscribed_segments = (
        _guided_ridge_curve_2d_historical_inscribed_fit(
            clean, cumulative, total, historical_inscribed_passes
        )
        if signed < 0.0 else []
    )
    historical_inscribed = (
        _guided_ridge_curve_2d_bezier_samples(
            historical_inscribed_segments, parameters
        )
        if historical_inscribed_segments else []
    )
    chord_delta = clean[-1] - clean[0]
    chord_normal = (
        Vector((-chord_delta.y, chord_delta.x)).normalized()
        if chord_delta.length > 1.0e-12
        else Vector((0.0, 0.0))
    )
    inscribed_wave = [
        (
            point - clean[0].lerp(clean[-1], float(parameter))
        ).dot(chord_normal)
        for point, parameter in zip(inscribed, parameters)
    ]
    lobe_intervals = _guided_ridge_curve_2d_detect_lobes(
        inscribed_wave, parameters, raw_oracle
    )
    # Use the endpoint chord as the zero-amplitude global trend.  A long,
    # gentle arc can be almost fully absorbed by the five-pass smooth fit, so
    # using that fit as the baseline makes Shape +100 look nearly unchanged.
    # The chord keeps the route's broad bulge available to the gain family;
    # the one-pass C1 fit remains the soft, cleaned waveform observation.
    chord_segments = _guided_ridge_curve_2d_endpoint_chord_segments(
        clean, len(inscribed_segments)
    )
    chord = _guided_ridge_curve_2d_bezier_samples(chord_segments, parameters)
    inscribed_segments, detail_normalization = _guided_ridge_curve_2d_normalize_detail_segments(
        chord_segments, inscribed_segments, raw, inscribed, parameters,
        raw_oracle=raw_oracle, lobe_intervals=lobe_intervals
    )
    inscribed = _guided_ridge_curve_2d_bezier_samples(
        inscribed_segments, parameters
    )
    opposite = None
    opposite_segments = []
    preview_segments = None
    validation = {"valid": True, "reason": None}
    effective_signed = signed
    emphasized_baseline = []
    opposite_amplitude = 0.0
    opposite_status = "neutral"
    opposite_warning = None
    if signed < 0.0:
        # Restore the historical inscribed C1 construction for the negative
        # side.  This is an independent target, not a raw/chord morph and not
        # a fallback from the outward waveform validator.
        preview = [point.copy() for point in historical_inscribed]
        preview_segments = historical_inscribed_segments
        opposite = [point.copy() for point in historical_inscribed]
        opposite_segments = historical_inscribed_segments
        opposite_status = "historical-inscribed"
        opposite_amplitude = max(
            (
                (historical_inscribed[index] - raw[index]).length
                for index in range(1, min(len(raw), len(historical_inscribed)) - 1)
            ),
            default=0.0,
        )
        validation = {
            "valid": bool(historical_inscribed),
            "reason": None if historical_inscribed else "inscribed fit unavailable",
            "construction": "pre-outward-3.3.53-screen-bezier",
            "smoothing_passes": historical_inscribed_passes,
        }
    elif signed > 0.0:
        # Both signs use the same local waveform measurement.  The historical
        # endpoint/chord helper remains isolated for compatibility tests, but
        # is deliberately not used here: global S-curves are valid input.
        # Validate the actual sampled/control-space result.  If a large gain
        # introduces a fold or intersection, reduce only that gain
        # monotonically; never silently replace a requested shape with raw.
        preview_candidate = None
        for factor in (1.0, 0.75, 0.5, 0.25, 0.125):
            attempt_signed = signed * factor
            helper_candidate, helper_status = _guided_ridge_curve_2d_wave_family(
                raw, chord, inscribed, parameters, attempt_signed
            )
            if helper_status != "valid":
                continue
            attempt_segments = _guided_ridge_curve_2d_wave_segments(
                chord_segments, inscribed_segments, attempt_signed
            )
            if not attempt_segments:
                continue
            sampled_candidate = _guided_ridge_curve_2d_bezier_samples(
                attempt_segments, parameters
            )
            sampled_candidate[0] = clean[0].copy()
            sampled_candidate[-1] = clean[-1].copy()
            attempt_validation = _guided_ridge_curve_2d_wave_validate(
                raw, chord, inscribed, sampled_candidate,
                attempt_segments, attempt_signed,
                raw_oracle=raw_oracle,
                lobe_intervals=lobe_intervals,
                parameters=parameters,
            )
            if attempt_validation.get("valid"):
                preview_candidate = sampled_candidate
                preview_segments = attempt_segments
                validation = attempt_validation
                effective_signed = attempt_signed
                opposite_status = "valid"
                if factor < 1.0:
                    opposite_warning = (
                        f"Curve Shape gain reduced to {effective_signed * 100.0:.0f} "
                        "for a safe 2D preview"
                    )
                break
        if preview_candidate is not None:
            preview = preview_candidate
            if signed > 0.0:
                opposite_segments = _guided_ridge_curve_2d_wave_segments(
                    chord_segments, inscribed_segments, 1.0
                )
                opposite = _guided_ridge_curve_2d_bezier_samples(
                    opposite_segments, parameters
                ) if opposite_segments else []
                opposite_amplitude = max(0.0, 1.0 + effective_signed)
            else:
                opposite = []
                opposite_amplitude = max(0.0, 1.0 + effective_signed)
        else:
            opposite_status = "failed"
            preview = [point.copy() for point in smooth]
            opposite_warning = "Curve Shape generation failed; showing last valid preview"
    if abs(signed) <= 1.0e-12:
        preview = [point.copy() for point in raw]
    preview[0] = clean[0].copy()
    preview[-1] = clean[-1].copy()
    smooth[0] = clean[0].copy()
    smooth[-1] = clean[-1].copy()
    return {
        "raw": raw,
        "preview": preview,
        "smooth": smooth,
        "inscribed": inscribed,
        "historical_inscribed": historical_inscribed,
        "historical_inscribed_bezier_segments": historical_inscribed_segments,
        "historical_inscribed_passes": historical_inscribed_passes,
        "chord": chord,
        "chord_bezier_segments": chord_segments,
        "detail_normalization": detail_normalization,
        "raw_chord_oracle": raw_oracle,
        "lobe_intervals": lobe_intervals,
        "inscribed_bezier_segments": inscribed_segments,
        "bezier_segments": preview_segments or segments,
        "smooth_bezier_segments": segments,
        "preview_bezier_segments": preview_segments,
        "opposite": opposite or [],
        "emphasized_baseline": emphasized_baseline,
        "opposite_bezier_segments": opposite_segments,
        "opposite_amplitude": opposite_amplitude,
        "opposite_status": opposite_status,
        "opposite_warning": opposite_warning,
        "effective_smoothing": effective_signed * 100.0,
        "wave_validation": validation,
        "parameters": parameters,
        "sample_count": len(preview),
        "fit_segment_count": len(segments),
    }


def _guided_ridge_curve_preview_update(state):
    """Refresh Step 1's generated line without touching Blender data."""
    if state is None:
        return False
    route = _guided_ridge_curve_dedupe(state.get("curve_route") or state.get("controls") or ())
    state["curve_route"] = route
    state["curve_preview"] = _guided_ridge_curve_preview_points(
        route, state.get("curve_smoothing", _runtime.guided_ridge_curve_last_smoothing)
    )
    state["curve_preview_generation"] = int(state.get("curve_preview_generation", 0)) + 1
    _guided_ridge_overlay_tag(state)
    return bool(state["curve_preview"])


def _guided_ridge_curve_update_screen_preview(context, state):
    """Project the retained 3D route, then fit/render the curve in 2D."""
    if state is None or state.get("phase") not in {"curve_preview", "curve_sculpt"}:
        return False
    signature_info = _guided_ridge_curve_preview_signature(context, state)
    current_signature = signature_info.get("full") if signature_info else None
    current_base_signature = signature_info.get("base") if signature_info else None
    state["curve_screen_refresh_count"] = int(
        state.get("curve_screen_refresh_count", 0) or 0
    ) + 1
    route = state.get("curve_route") or state.get("controls") or ()
    projected = []
    projection_ok = True
    projected_complete = False
    outside_view = False
    try:
        region = context.region
        region_3d = context.space_data.region_3d
        ui_scale = float(
            getattr(getattr(getattr(context, "preferences", None), "system", None), "ui_scale", 1.0)
            or 1.0
        )
        tolerance = max(2.0, 3.0 * ui_scale)
        region_width = float(getattr(region, "width", 0) or 0)
        region_height = float(getattr(region, "height", 0) or 0)
        for point in route:
            screen_point = _guided_ridge_project_3d_to_region_2d(
                region, region_3d, point
            )
            if screen_point is None:
                projection_ok = False
                break
            projected.append(Vector((float(screen_point.x), float(screen_point.y))))
            if (
                float(screen_point.x) < -tolerance
                or float(screen_point.x) > region_width + tolerance
                or float(screen_point.y) < -tolerance
                or float(screen_point.y) > region_height + tolerance
            ):
                projection_ok = False
                outside_view = True
        projected_complete = len(projected) == len(route) and len(projected) >= 2
        state["curve_projection_raw_point_count"] = len(projected)
        state["curve_projection_unique_count"] = len(
            {(round(float(point.x), 6), round(float(point.y), 6)) for point in projected}
        )
        if not projected_complete:
            state["curve_screen_raw"] = []
            state["curve_screen_preview"] = []
            state["curve_screen_smooth"] = []
            state["curve_screen_bezier_segments"] = []
            state["curve_screen_sample_count"] = 0
            state["curve_screen_intersection_checks"] = 0
            state["curve_shape_warning"] = (
                "Curve unavailable in current view; no generated curve displayed"
            )
            state["curve_shape_status"] = "failed"
            state["curve_effective_smoothing"] = 0.0
        elif not projection_ok:
            # All points are still valid 2D projections, but one or more are
            # outside the current region.  Keep the newly projected raw route
            # visible instead of clearing it as if projection had failed.
            # Generated fitting is intentionally withheld until the route is
            # fully visible again.
            state["curve_screen_raw"] = [point.copy() for point in projected]
            state["curve_screen_preview"] = [point.copy() for point in projected]
            state["curve_screen_smooth"] = []
            state["curve_screen_bezier_segments"] = []
            state["curve_screen_sample_count"] = len(projected)
            state["curve_screen_intersection_checks"] = 0
            state["curve_screen_fit_segment_count"] = 0
            state["curve_shape_warning"] = (
                "Curve unavailable in current view; showing projected raw route"
            )
            state["curve_shape_status"] = "failed"
            state["curve_effective_smoothing"] = 0.0
        else:
            requested_shape = float(
                state.get("curve_smoothing", _runtime.guided_ridge_curve_last_smoothing) or 0.0
            )
            # A valid projection can still collapse the entire route to one
            # screen point (for example after a stale/foreign projection
            # helper has been installed).  Feed an explicit failed result
            # through the normal fallback path so every projected raw point is
            # retained and the HUD reports an actionable warning.
            if state["curve_projection_unique_count"] < 2:
                projection_ok = False
                result = {
                    "raw": [point.copy() for point in projected],
                    "preview": [point.copy() for point in projected],
                    "smooth": [],
                    "bezier_segments": [],
                    "preview_bezier_segments": [],
                    "opposite_status": "failed",
                    "opposite_warning": (
                        "Current-view projection collapsed the route; adjust the view and retry"
                    ),
                    "sample_count": len(projected),
                    "fit_segment_count": 0,
                    "wave_validation": {"intersection_checks": 0},
                    "effective_smoothing": 0.0,
                }
            else:
                result = _guided_ridge_curve_2d_preview_points(
                    projected,
                    requested_shape,
                )
            state["curve_screen_intersection_checks"] = int(
                (result.get("wave_validation") or {}).get("intersection_checks", 0) or 0
            )
            opposite_failed = (
                abs(requested_shape) > 1.0e-12
                and result.get("opposite_status") == "failed"
            )
            previous_preview = state.get("curve_last_valid_preview") or []
            previous_base_signature = state.get("curve_last_valid_base_signature")
            same_view_route = (
                previous_base_signature == current_base_signature
            )
            if opposite_failed and previous_preview and same_view_route:
                # Keep the exact last displayed generation.  In particular do
                # not substitute the current positive smooth fit for a failed
                # emphasized request; that would make the HUD claim a shape that
                # was never drawn.  New valid view/shape regeneration replaces
                # this snapshot below.
                state["curve_screen_raw"] = [
                    point.copy() for point in (state.get("curve_last_valid_raw") or [])
                ]
                state["curve_screen_preview"] = [point.copy() for point in previous_preview]
                state["curve_screen_smooth"] = [
                    point.copy() for point in (state.get("curve_last_valid_smooth") or [])
                ]
                state["curve_screen_bezier_segments"] = [
                    tuple(point.copy() for point in segment)
                    for segment in (state.get("curve_last_valid_segments") or [])
                ]
                state["curve_screen_sample_count"] = len(previous_preview)
                state["curve_screen_fit_segment_count"] = len(
                    state.get("curve_screen_bezier_segments") or []
                )
                effective_shape = float(
                    state.get("curve_last_valid_shape", 0.0) or 0.0
                )
                state["curve_effective_smoothing"] = effective_shape
                state["curve_shape_warning"] = (
                    f"Requested Shape {requested_shape:.0f} unavailable; "
                    f"showing Shape {effective_shape:.0f}"
                )
                state["curve_shape_status"] = "retained"
            elif opposite_failed:
                # A failed candidate from a new view/route must never display
                # screen coordinates from the previous view.  The current raw
                # projection is safe to show only when this view projected the
                # complete route; generated Shape output remains unavailable.
                fallback = [point.copy() for point in result["raw"]]
                state["curve_screen_raw"] = [point.copy() for point in result["raw"]]
                state["curve_screen_preview"] = fallback
                state["curve_screen_smooth"] = []
                state["curve_screen_bezier_segments"] = []
                state["curve_screen_sample_count"] = int(result.get("sample_count", 0))
                state["curve_screen_fit_segment_count"] = 0
                state["curve_effective_smoothing"] = 0.0
                failure_detail = result.get("opposite_warning")
                state["curve_shape_warning"] = (
                    failure_detail
                    if failure_detail and "collapsed" in str(failure_detail).lower()
                    else (
                        f"Requested Shape {requested_shape:.0f} unavailable in current view; "
                        "showing current raw route"
                    )
                )
                state["curve_shape_status"] = "failed"
                # The current raw projection is an explicit, honest fallback
                # baseline for a subsequent same-view failure.  It is never
                # labelled as a successful requested Shape and is scoped to
                # this exact route/view base signature.
                state["curve_last_valid_raw"] = [point.copy() for point in result["raw"]]
                state["curve_last_valid_preview"] = [point.copy() for point in fallback]
                state["curve_last_valid_smooth"] = []
                state["curve_last_valid_segments"] = []
                state["curve_last_valid_shape"] = 0.0
                state["curve_last_valid_status"] = "raw-fallback"
                state["curve_last_valid_base_signature"] = current_base_signature
                state["curve_last_valid_signature"] = current_signature
            else:
                display_segments = result.get("preview_bezier_segments")
                if display_segments is None:
                    display_segments = result.get("bezier_segments")
                display_segments = display_segments or []
                state["curve_screen_raw"] = result["raw"]
                state["curve_screen_preview"] = result["preview"]
                state["curve_screen_smooth"] = result["smooth"]
                state["curve_screen_bezier_segments"] = display_segments
                state["curve_screen_sample_count"] = int(result.get("sample_count", 0))
                state["curve_screen_fit_segment_count"] = int(result.get("fit_segment_count", 0))
                state["curve_shape_warning"] = result.get("opposite_warning")
                state["curve_shape_status"] = result.get("opposite_status", "neutral")
                effective_shape = float(
                    result.get("effective_smoothing", requested_shape)
                )
                state["curve_effective_smoothing"] = effective_shape
                # A successful draw is the new rollback/display baseline.  Use
                # detached vectors so later screen regeneration cannot mutate
                # the retained generation behind its warning semantics.
                state["curve_last_valid_raw"] = [point.copy() for point in result["raw"]]
                state["curve_last_valid_preview"] = [point.copy() for point in result["preview"]]
                state["curve_last_valid_smooth"] = [point.copy() for point in result["smooth"]]
                state["curve_last_valid_segments"] = [
                    tuple(point.copy() for point in segment)
                    for segment in display_segments
                ]
                state["curve_last_valid_shape"] = effective_shape
                state["curve_last_valid_status"] = result.get("opposite_status", "neutral")
                state["curve_last_valid_base_signature"] = current_base_signature
                state["curve_last_valid_signature"] = current_signature
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        projection_ok = False
        if projected_complete:
            # A complete 2D projection is still useful even when fitting or a
            # later view operation fails.  Keep that newly projected raw route
            # visible so Enter never leaves a blank preview.
            state["curve_screen_raw"] = [point.copy() for point in projected]
            state["curve_screen_preview"] = [point.copy() for point in projected]
            state["curve_screen_smooth"] = []
            state["curve_screen_bezier_segments"] = []
            state["curve_screen_sample_count"] = len(projected)
            state["curve_screen_intersection_checks"] = 0
            state["curve_screen_fit_segment_count"] = 0
            state["curve_shape_warning"] = (
                "Curve generation failed in current view; showing projected raw route"
            )
        else:
            state["curve_screen_raw"] = []
            state["curve_screen_preview"] = []
            state["curve_screen_smooth"] = []
            state["curve_screen_bezier_segments"] = []
            state["curve_screen_sample_count"] = 0
            state["curve_screen_intersection_checks"] = 0
            state["curve_shape_warning"] = (
                "Curve unavailable in current view; no generated curve displayed"
            )
        state["curve_shape_status"] = "failed"
        state["curve_effective_smoothing"] = 0.0
    state["curve_screen_signature"] = current_signature
    state["curve_screen_base_signature"] = current_base_signature
    state["curve_screen_cache_status"] = "valid" if projection_ok else "failed"
    state["curve_projection_warning"] = None if projection_ok else (
        "Curve is outside the current view; showing projected raw route"
        if projected_complete and outside_view
        else (
            "Current-view projection collapsed the route; adjust the view and retry"
            if projected_complete and int(state.get("curve_projection_unique_count", 0) or 0) < 2
            else (
                "Curve generation failed in the current view; showing projected raw route"
                if projected_complete
                else "Curve could not be projected from the current view; no changes applied"
            )
        )
    )
    state["curve_screen_generation"] = int(state.get("curve_screen_generation", 0)) + 1
    _guided_ridge_overlay_tag(state)
    return projection_ok


def _guided_ridge_begin_curve_preview(context, state):
    """Switch an edited snapped route into the non-destructive Step 1 view."""
    if state is None or state.get("phase") != "ready":
        return False
    route = _guided_ridge_curve_dedupe(state.get("controls") or ())
    if len(route) < 2:
        operator = state.get("operator")
        if operator is not None:
            operator.report({"WARNING"}, "Guided Ridge Curve Sculpt: add at least two route points")
        return False
    # Preview generation is display-only.  Keep the exact projected guide
    # and captured frame/cache objects so Backspace can return without
    # rebuilding the expensive cooperative width preparation.
    state["curve_route_restore"] = {
        key: state.get(key)
        for key in (
            "guide", "width_frame", "width_frame_result", "width_rails",
            "width_cache_serial", "width_frame_job",
        )
    }
    state["curve_route"] = route
    preferences = _addon_preferences()
    remembered_smoothing = getattr(
        preferences,
        "guided_ridge_curve_smoothing",
        _runtime.guided_ridge_curve_last_smoothing,
    ) if preferences is not None else _runtime.guided_ridge_curve_last_smoothing
    state["curve_smoothing"] = min(max(float(remembered_smoothing), -100.0), 100.0)
    # Every curve-preview session starts with no retained display generation;
    # the first successful screen redraw establishes the rollback baseline.
    for key in (
        "curve_last_valid_raw", "curve_last_valid_preview",
        "curve_last_valid_smooth", "curve_last_valid_segments",
        "curve_last_valid_shape", "curve_last_valid_status",
        "curve_last_valid_signature", "curve_last_valid_base_signature",
    ):
        state.pop(key, None)
    state["curve_effective_smoothing"] = state["curve_smoothing"]
    state["curve_projection_warning"] = None
    state["curve_shape_warning"] = None
    state["curve_shape_status"] = "neutral"
    state["curve_preview_notice"] = "Step 1 preview only; application comes in Step 3"
    state["phase"] = "curve_preview"
    state["curve_view_signature"] = _guided_ridge_context_signature(context) if context is not None else None
    state["curve_screen_signature"] = None
    state["curve_screen_base_signature"] = None
    state["curve_screen_cache_status"] = "empty"
    state["curve_idle_cache_hits"] = 0
    state["curve_route_digest_cache"] = None
    state["curve_route_revision"] = int(state.get("curve_route_revision", 0) or 0) + 1
    state["preview_context"] = context
    if not _guided_ridge_curve_preview_update(state):
        state["phase"] = "ready"
        return False
    _guided_ridge_curve_update_screen_preview(context, state)
    # Keep a lightweight timer so view changes can refresh the projection
    # warning and redraw opportunity without creating a temporary object.
    if context is not None and state.get("timer") is None:
        try:
            state["timer"] = context.window_manager.event_timer_add(0.05, window=context.window)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            state["timer"] = None
    _guided_ridge_overlay_tag(state)
    return True


def _guided_ridge_curve_refresh_projection(context, state):
    """Recheck the retained 3D route after a viewport change.

    The retained 3D route is projected into the current View3D and the
    non-destructive 2D Bezier preview is regenerated there.  A screen
    projection check provides a clear warning when the route is not fully
    visible in the current view.
    """
    if state is None or state.get("phase") not in {"curve_preview", "curve_sculpt"}:
        return False
    current = _guided_ridge_context_signature(context) if context is not None else None
    expected = state.get("context_signature") or {}
    stable_keys = (
        "area_pointer", "region_pointer", "window_pointer", "space_pointer",
        "scene_pointer", "active_object_pointer", "mesh_pointer", "mode", "space_type",
    )
    if current is None or any(current.get(key) != expected.get(key) for key in stable_keys):
        operator = state.get("operator")
        if operator is not None:
            operator.report({"WARNING"}, "Guided Ridge Curve Sculpt: viewport or object context changed")
        _guided_ridge_cancel(state, "curve-context-changed")
        return False
    signature_info = _guided_ridge_curve_preview_signature(context, state)
    if signature_info is None:
        return False
    requested_signature = signature_info.get("full")
    # TIMER remains a cheap identity/signature check.  Do not refit, run the
    # bounded intersection checker, or touch screen arrays when neither the
    # route/Shape nor the projection changed.
    screen_has_content = (
        len(state.get("curve_screen_raw") or ()) >= 2
        or len(state.get("curve_screen_preview") or ()) >= 2
    )
    if state.get("curve_screen_signature") == requested_signature and screen_has_content:
        state["curve_idle_cache_hits"] = int(state.get("curve_idle_cache_hits", 0) or 0) + 1
        return state.get("curve_screen_cache_status") != "failed"
    state["curve_view_signature"] = current
    projection_ok = _guided_ridge_curve_update_screen_preview(context, state)
    if state.get("phase") == "curve_sculpt" and state.get("curve_sculpt_active"):
        # Paint Curve points are 2D viewport data.  A changed view invalidates
        # the old temporary stroke; retain the mesh result and require the
        # next explicit Enter/Ctrl+Enter to create a fresh current-view curve.
        _curve_sculpt.cleanup(state)
        state["curve_sculpt_needs_rebuild"] = True
    return bool(projection_ok or state.get("curve_screen_cache_status") == "failed")


def _guided_ridge_curve_set_smoothing(state, value):
    if state is None:
        return False
    try:
        smoothing = min(max(float(value), -100.0), 100.0)
    except (TypeError, ValueError):
        smoothing = 0.0
    _runtime.guided_ridge_curve_last_smoothing = smoothing
    preferences = _addon_preferences()
    if preferences is not None and hasattr(preferences, "guided_ridge_curve_smoothing"):
        try:
            preferences.guided_ridge_curve_smoothing = smoothing
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    state["curve_smoothing"] = smoothing
    updated = _guided_ridge_curve_preview_update(state)
    preview_context = state.get("preview_context")
    if preview_context is not None:
        _guided_ridge_curve_update_screen_preview(preview_context, state)
    return updated


def _guided_ridge_curve_back_to_route(context, state):
    """Return from preview to the existing editable route, losslessly."""
    if state is None or state.get("phase") not in {"curve_preview", "curve_sculpt"}:
        return False
    if state.get("curve_sculpt_active") or state.get("curve_sculpt_restore"):
        _curve_sculpt.cleanup(state)
    timer = state.get("timer")
    if timer is not None:
        try:
            state["window_manager"].event_timer_remove(timer)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        state["timer"] = None
    restore = state.pop("curve_route_restore", None) or {}
    state["phase"] = "ready"
    # Restore object identity as well as values.  No width/frame/layer
    # preparation is needed because Step 1 never mutates those objects.
    for key, value in restore.items():
        state[key] = value
    if state.get("prototype_route_only"):
        _guided_ridge_route_monitor_start(context, state)
    state["curve_preview"] = []
    state["curve_screen_raw"] = []
    state["curve_screen_preview"] = []
    state["curve_screen_smooth"] = []
    state["curve_screen_bezier_segments"] = []
    for key in (
        "curve_last_valid_raw", "curve_last_valid_preview",
        "curve_last_valid_smooth", "curve_last_valid_segments",
        "curve_last_valid_shape", "curve_last_valid_status",
        "curve_last_valid_signature", "curve_last_valid_base_signature",
    ):
        state.pop(key, None)
    state["preview_context"] = None
    state["curve_projection_warning"] = None
    state["curve_screen_signature"] = None
    state["curve_screen_base_signature"] = None
    state["curve_screen_cache_status"] = "empty"
    _guided_ridge_overlay_tag(state)
    return True


def _guided_ridge_prepare_context_matches(context, state):
    """Check the stable context identity while preparation is yielding."""
    expected = (state or {}).get("context_signature")
    if expected is None:
        return False
    try:
        if context.active_object is not state.get("obj") or context.mode != "SCULPT":
            return False
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    current = _guided_ridge_context_signature(context)
    if current is None:
        return False
    stable_keys = (
        "area_pointer",
        "region_pointer",
        "window_pointer",
        "space_pointer",
        "scene_pointer",
        "active_object_pointer",
        "mesh_pointer",
        "mode",
        "space_type",
    )
    return all(current.get(key) == expected.get(key) for key in stable_keys)


def _guided_ridge_process_prepare_timer(context, state):
    """Advance one bounded snapshot slice and transition to guide-ready."""
    if state is None or not state.get("active") or state.get("phase") != "prepare":
        return False
    operator = state.get("operator")
    if not _guided_ridge_prepare_context_matches(context, state):
        if operator is not None:
            operator.report({"WARNING"}, "Guided Ridge: preparation context changed; no changes applied")
        _guided_ridge_cancel(state, "prepare-context-changed")
        return False
    started = time.perf_counter()
    state["prepare_slices"] = int(state.get("prepare_slices", 0)) + 1
    try:
        progress = None
        deadline = started + 0.008
        while True:
            progress = next(state["prepare_job"])
            # Publish a newly entered stage for one complete UI tick before
            # advancing into the stage's first expensive operation.  This
            # returns to the event loop and gives the overlay a redraw
            # opportunity before the first indivisible foreach_get/BVH call.
            if isinstance(progress, dict):
                progress_stage = str(progress.get("stage", "preparing"))
                if progress_stage != state.get("prepare_stage"):
                    break
            if time.perf_counter() >= deadline:
                break
    except StopIteration as complete:
        snapshot, reason = complete.value or (None, "could not prepare the Face Set component")
        state["prepare_job"] = None
        if snapshot is None:
            if operator is not None:
                operator.report({"WARNING"}, f"Guided Ridge: {reason or 'preparation failed'}")
            _guided_ridge_cancel(state, "prepare-failed")
            return False
        try:
            start_hit = _guided_ridge_snapshot_hit_ray(snapshot, state.get("start_ray"))
            if start_hit is None:
                if operator is not None:
                    operator.report({"WARNING"}, "Guided Ridge: saved start ray no longer hits the prepared Face Set component")
                _guided_ridge_cancel(state, "start-point-failed")
                return False
            context_signature = _guided_ridge_context_signature(context)
            if context_signature is None:
                if operator is not None:
                    operator.report({"WARNING"}, "Guided Ridge: viewport context is unavailable")
                _guided_ridge_cancel(state, "context-unavailable")
                return False
            snapshot["context_signature"] = context_signature
            state["snapshot"] = snapshot
            initial_width, minimum_width, maximum_width, _average_edge = _guided_ridge_width_limits(snapshot)
            state["half_width"] = min(
                max(float(state.get("half_width", initial_width) or initial_width), minimum_width),
                maximum_width,
            )
            state["width_min"] = minimum_width
            state["width_max"] = maximum_width
            state["width_frame"] = None
            state["width_frame_result"] = None
            state["width_rails"] = {"left": [], "right": []}
            state["controls"] = [start_hit[0]]
            state["control_normals"] = [start_hit[1]]
            state["guide"] = [start_hit[0]]
            state["last_cursor"] = start_hit[0]
            state["phase"] = "ready"
            state["prepare_stage"] = "ready"
            state["prepare_stage_index"] = int(state.get("prepare_stage_count", 13))
            state["prepare_stage_done"] = 1
            state["prepare_stage_total"] = 1
            state["prepare_progress_fraction"] = 1.0
            state["prepare_progress_indeterminate"] = False
            _guided_ridge_update_width_preview(state)
            timer = state.get("timer")
            if timer is not None:
                window_manager = state.get("window_manager")
                if window_manager is None:
                    window_manager = bpy.context.window_manager
                try:
                    window_manager.event_timer_remove(timer)
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    pass
                state["timer"] = None
            _guided_ridge_overlay_tag(state)
        except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
            if operator is not None:
                operator.report({"WARNING"}, f"Guided Ridge: preparation failed ({error})")
            _guided_ridge_cancel(state, "prepare-finalize-failed")
            return False
    except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
        if operator is not None:
            operator.report({"WARNING"}, f"Guided Ridge: preparation failed ({error})")
        _guided_ridge_cancel(state, "prepare-exception")
        return False
    else:
        if isinstance(progress, dict):
            stage = str(progress.get("stage", "preparing"))
            if stage != state.get("prepare_stage"):
                state["prepare_stage_index"] = max(
                    int(state.get("prepare_stage_index", 0)),
                    int(progress.get("stage_index", 0)),
                )
            state["prepare_stage"] = stage
            state["prepare_stage_count"] = int(progress.get("stage_count", 13))
            state["prepare_stage_done"] = int(progress.get("done", 0))
            state["prepare_stage_total"] = int(progress.get("total", 0))
            state["prepare_progress_indeterminate"] = bool(progress.get("indeterminate", False))
            total = int(progress.get("total", 0))
            done = int(progress.get("done", 0))
            state["prepare_progress_fraction"] = (
                min(max(done / float(total), 0.0), 1.0) if total > 0 else None
            )
            _guided_ridge_overlay_tag(state)
    finally:
        elapsed = time.perf_counter() - started
        state["prepare_last_slice_seconds"] = float(elapsed)
        state["prepare_max_slice_seconds"] = max(
            float(state.get("prepare_max_slice_seconds", 0.0)), float(elapsed)
        )
        state["prepare_elapsed_seconds"] = float(state.get("prepare_elapsed_seconds", 0.0)) + float(elapsed)
    return True


def _guided_ridge_begin_compute(context, state):
    """Enter the modal, cancellable candidate-computation phase."""
    if state is None or state.get("phase") != "ready":
        return False
    try:
        state["phase"] = "compute"
        state["compute_job"] = _guided_ridge_exact_candidate_steps(
            state["snapshot"],
            state["guide"],
            state["controls"],
            state["control_normals"],
            state.get("half_width"),
            width_result=state.get("width_frame_result"),
        )
        state["compute_stage"] = "queued"
        state["compute_stage_index"] = 0
        state["compute_stage_count"] = 4
        state["compute_stage_done"] = 0
        state["compute_stage_total"] = 0
        state["compute_progress_fraction"] = None
        state["compute_progress_indeterminate"] = True
        if state.get("timer") is None:
            state["timer"] = context.window_manager.event_timer_add(0.01, window=context.window)
        _guided_ridge_overlay_tag(state)
        return True
    except (AttributeError, MemoryError, RuntimeError, TypeError, ValueError):
        _guided_ridge_cancel(state, "compute-start-failed")
        return False


def _guided_ridge_finish_compute(context, state, result):
    """Consume a completed candidate through the established write/Undo path."""
    if state is None or not state.get("active"):
        return False
    candidate, info = result
    timer = state.get("timer")
    if timer is not None:
        try:
            state["window_manager"].event_timer_remove(timer)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        state["timer"] = None
    state["compute_job"] = None
    state["phase"] = "ready"
    operator = state.get("operator")
    if operator is None:
        _guided_ridge_cancel(state, "compute-no-operator")
        return False
    committed = bool(operator._commit(context, state, _compute_sync=True, _candidate_override=(candidate, info)))
    # _commit tears down the modal state on success.  Preserve the modal
    # operator result separately so Blender receives FINISHED rather than
    # mistaking the intentional cleanup (active=False) for cancellation.
    state["modal_result"] = "FINISHED" if committed and state.get("committed") else "CANCELLED"
    return committed


def _guided_ridge_process_compute_timer(context, state):
    """Advance candidate extraction for one bounded UI-event-loop slice."""
    if state is None or not state.get("active") or state.get("phase") != "compute":
        return False
    operator = state.get("operator")
    if not _guided_ridge_context_matches(context, state):
        if operator is not None:
            operator.report({"WARNING"}, "Guided Ridge: computation context changed; no changes applied")
        _guided_ridge_cancel(state, "compute-context-changed")
        return False
    started = time.perf_counter()
    try:
        deadline = started + 0.008
        progress = None
        while True:
            progress = next(state["compute_job"])
            if isinstance(progress, dict):
                stage = str(progress.get("stage", "computing"))
                if stage != state.get("compute_stage"):
                    break
            if time.perf_counter() >= deadline:
                break
    except StopIteration as complete:
        try:
            return _guided_ridge_finish_compute(context, state, complete.value)
        except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
            if operator is not None:
                operator.report({"WARNING"}, f"Guided Ridge: computation failed ({error})")
            _guided_ridge_cancel(state, "compute-failed")
            return False
    except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
        if operator is not None:
            operator.report({"WARNING"}, f"Guided Ridge: computation failed ({error})")
        _guided_ridge_cancel(state, "compute-failed")
        return False
    else:
        if isinstance(progress, dict):
            stage = str(progress.get("stage", "computing"))
            if stage != state.get("compute_stage"):
                state["compute_stage_index"] = max(
                    int(state.get("compute_stage_index", 0)), int(progress.get("stage_index", 0))
                )
            state["compute_stage"] = stage
            state["compute_stage_count"] = int(progress.get("stage_count", 4))
            state["compute_stage_done"] = int(progress.get("done", 0))
            state["compute_stage_total"] = int(progress.get("total", 0))
            total = int(progress.get("total", 0))
            done = int(progress.get("done", 0))
            state["compute_progress_indeterminate"] = bool(progress.get("indeterminate", False))
            state["compute_progress_fraction"] = min(max(done / float(total), 0.0), 1.0) if total > 0 else None
            _guided_ridge_overlay_tag(state)
    finally:
        elapsed = time.perf_counter() - started
        state["compute_last_slice_seconds"] = float(elapsed)
        state["compute_max_slice_seconds"] = max(float(state.get("compute_max_slice_seconds", 0.0)), float(elapsed))
        state["compute_elapsed_seconds"] = float(state.get("compute_elapsed_seconds", 0.0)) + float(elapsed)
    return True


def _guided_ridge_stop_draw():
    state = _runtime.guided_ridge_state
    if state is None:
        return
    for key in ("draw_handler", "text_draw_handler", "screen_curve_draw_handler"):
        handler = state.get(key)
        if handler is None:
            continue
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handler, "WINDOW")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        state[key] = None


def _guided_ridge_install_session_handlers(context, state):
    """Install the editor overlay/timer for an existing route session.

    This is shared by initial invoke and the short-lived native transaction
    resume path.  It intentionally does not rebuild the route or mesh
    snapshot.
    """
    if state is None:
        return False
    try:
        if state.get("draw_handler") is None:
            state["draw_handler"] = bpy.types.SpaceView3D.draw_handler_add(
                _guided_ridge_draw, (), "WINDOW", "POST_VIEW"
            )
        if state.get("text_draw_handler") is None:
            state["text_draw_handler"] = bpy.types.SpaceView3D.draw_handler_add(
                _guided_ridge_draw_text, (), "WINDOW", "POST_PIXEL"
            )
        if state.get("screen_curve_draw_handler") is None:
            state["screen_curve_draw_handler"] = bpy.types.SpaceView3D.draw_handler_add(
                _guided_ridge_draw_curve_screen, (), "WINDOW", "POST_PIXEL"
            )
        if state.get("timer") is None:
            if state.get("prototype_route_only"):
                if not _guided_ridge_route_monitor_start(context, state):
                    raise RuntimeError("route monitor timer could not be restored")
            else:
                state["timer"] = state["window_manager"].event_timer_add(
                    0.01, window=context.window
                )
        state["start_guard_active"] = False
        state["start_key_released"] = True
        _guided_ridge_overlay_tag(state)
        return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _guided_ridge_stop_draw()
        return False


def _guided_ridge_suspend_editor(context, state):
    """Suspend route editing before a one-shot native Sculpt operation."""
    if state is None:
        return False
    timer = state.get("timer")
    if timer is not None:
        try:
            state["window_manager"].event_timer_remove(timer)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        state["timer"] = None
    _guided_ridge_stop_draw()
    state["curve_sculpt_editor_suspended"] = True
    return True


def _guided_ridge_native_live_context(transaction, state):
    """Resolve saved UI identities through the current window manager.

    Application timers cannot trust ambient bpy.context.area/region.  Resolve
    the saved window, View3D area, window region, and active space from the
    live window manager, then let the caller create one temp override and
    validate the full launch identity inside it.
    """
    if (
        transaction is None
        or state is None
        or _runtime.guided_ridge_state is not state
        or not state.get("active")
        or transaction.get("token") != state.get("curve_sculpt_session_token")
    ):
        return None
    launch_identity = transaction.get("launch_identity") or {}
    identity_pairs = (
        ("window", "window_key"),
        ("area", "area_key"),
        ("region", "region_key"),
        ("object", "object_pointer"),
        ("mesh", "mesh_pointer"),
    )
    if any(launch_identity.get(left) != state.get(right) for left, right in identity_pairs):
        return None
    try:
        base_context = getattr(bpy, "context", None)
        window_manager = getattr(base_context, "window_manager", None)
        if base_context is None or window_manager is None:
            return None
        for window in tuple(window_manager.windows):
            if int(window.as_pointer()) != int(state.get("window_key", 0)):
                continue
            screen = window.screen
            for area in tuple(screen.areas):
                if int(area.as_pointer()) != int(state.get("area_key", 0)):
                    continue
                if getattr(area, "type", None) != "VIEW_3D":
                    return None
                space = getattr(area.spaces, "active", None)
                if space is None or int(space.as_pointer()) != int(state.get("space_key", 0)):
                    return None
                for region in tuple(area.regions):
                    if int(region.as_pointer()) != int(state.get("region_key", 0)):
                        continue
                    if getattr(region, "type", None) != "WINDOW":
                        return None
                    return base_context, window, area, region, space
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    return None


def _guided_ridge_native_in_override(transaction, state, callback):
    """Run one callback inside a freshly resolved launch-context override."""
    resolved = _guided_ridge_native_live_context(transaction, state)
    if resolved is None:
        raise RuntimeError("native transaction launch context is unavailable")
    base_context, window, area, region, space = resolved
    with base_context.temp_override(
        window=window,
        area=area,
        region=region,
        space_data=space,
    ):
        context = bpy.context
        if not _guided_ridge_route_context_matches(context, state):
            raise RuntimeError("native transaction launch context changed")
        return callback(context)


def _guided_ridge_native_transaction_tick():
    """Run exactly one native stroke, then re-enter the same editor session."""
    transaction = _runtime.guided_ridge_native_transaction
    if not transaction:
        _runtime.guided_ridge_native_transaction_timer = None
        return None
    state = transaction.get("state")
    if state is None or not state.get("active"):
        _runtime.guided_ridge_native_transaction = None
        _runtime.guided_ridge_native_transaction_timer = None
        return None
    if _guided_ridge_native_live_context(transaction, state) is None:
        if state.get("operator") is not None:
            state["operator"].report(
                {"WARNING"},
                "Guided Ridge: native transaction context changed; no stroke was applied",
            )
        _guided_ridge_cancel(state, "native-context-changed")
        _runtime.guided_ridge_native_transaction = None
        _runtime.guided_ridge_native_transaction_timer = None
        return None
    if transaction.get("status") == "pending":
        transaction["status"] = "running"
        mode = transaction.get("mode", "RIDGE")
        try:
            if not _guided_ridge_native_in_override(
                transaction,
                state,
                lambda live_context: _curve_sculpt.apply(live_context, state, mode=mode),
            ):
                transaction["status"] = "failed"
                if state.get("operator") is not None:
                    state["operator"].report(
                        {"WARNING"},
                        "Guided Ridge: native one-shot application was unavailable",
                    )
                state["curve_sculpt_editor_suspended"] = False
                # The original editor has already returned CANCELLED.  Keep
                # the transaction alive and route through the same fresh
                # operator re-entry used after success; reinstalling only the
                # draw handler/timer would leave the HUD visible with no modal
                # owner and make the session impossible to Esc-cancel.
                transaction["status"] = "resume"
                transaction["resume_warning"] = True
            state["curve_sculpt_editor_suspended"] = False
            state["curve_sculpt_native_last_mode"] = mode
            transaction["status"] = "resume"
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
            transaction["status"] = "failed"
            if state.get("operator") is not None:
                state["operator"].report({"WARNING"}, f"Guided Ridge: native transaction failed ({error})")
            state["curve_sculpt_editor_suspended"] = False
            transaction["status"] = "resume"
            transaction["resume_warning"] = True
    if transaction.get("status") == "resume":
        # A fresh operator instance owns the resumed modal handler.  The
        # existing route/session dictionary remains the canonical state.
        try:
            result = _guided_ridge_native_in_override(
                transaction,
                state,
                lambda _live_context: bpy.ops.view3d.mesh_focus_guided_ridge("INVOKE_DEFAULT"),
            )
            if "RUNNING_MODAL" in result:
                _runtime.guided_ridge_native_transaction = None
                _runtime.guided_ridge_native_transaction_timer = None
                return None
            raise RuntimeError(f"Guided Ridge resume did not enter modal: {result}")
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
            if state.get("operator") is not None:
                state["operator"].report({"WARNING"}, f"Guided Ridge: session resume failed ({error})")
            # Resume can fail after the new event timer has been installed.
            # Use the canonical cancel path so that timer, native transaction,
            # draw handlers, and the curve-sculpt restore state all close
            # together.
            _guided_ridge_cancel(state, "resume-failed")
            _runtime.guided_ridge_native_transaction = None
            _runtime.guided_ridge_native_transaction_timer = None
            return None
    return 0.0


def _guided_ridge_begin_native_transaction(context, state, mode):
    """Suspend the editor and schedule one independent native operation."""
    if state is None or not state.get("active"):
        return False
    if _runtime.guided_ridge_native_transaction is not None:
        return False
    if not _guided_ridge_suspend_editor(context, state):
        return False
    _runtime.guided_ridge_native_transaction = {
        "kind": "apply",
        "status": "pending",
        "mode": "GROOVE" if str(mode).upper() == "GROOVE" else "RIDGE",
        "state": state,
        "token": state.get("curve_sculpt_session_token"),
        "launch_identity": {
            "window": state.get("window_key"),
            "area": state.get("area_key"),
            "region": state.get("region_key"),
            "object": state.get("object_pointer"),
            "mesh": state.get("mesh_pointer"),
            "mode": (state.get("context_signature") or {}).get("mode"),
        },
    }
    try:
        bpy.app.timers.register(_guided_ridge_native_transaction_tick, first_interval=0.0)
        _runtime.guided_ridge_native_transaction_timer = _guided_ridge_native_transaction_tick
        return True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        _runtime.guided_ridge_native_transaction = None
        _runtime.guided_ridge_native_transaction_timer = None
        _guided_ridge_install_session_handlers(context, state)
        state["curve_sculpt_editor_suspended"] = False
        return False


def _guided_ridge_curve_2d_shader_get():
    if _runtime.guided_ridge_curve_2d_shader is None:
        for shader_name in ("2D_UNIFORM_COLOR", "UNIFORM_COLOR"):
            try:
                _runtime.guided_ridge_curve_2d_shader = gpu.shader.from_builtin(shader_name)
                break
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
    return _runtime.guided_ridge_curve_2d_shader


def _guided_ridge_draw_curve_screen():
    """Draw the current-view raw route and fitted Bezier in POST_PIXEL."""
    state = _runtime.guided_ridge_state
    if state is None or not state.get("active") or state.get("phase") not in {"curve_preview", "curve_sculpt"}:
        return
    try:
        context = bpy.context
        if context.area is None or int(context.area.as_pointer()) != int(state["area_key"]):
            return
        raw = state.get("curve_screen_raw") or []
        generated = state.get("curve_screen_preview") or []
        if len(raw) < 2 and len(generated) < 2:
            return
        shader = _guided_ridge_curve_2d_shader_get()
        if shader is None:
            return
        gpu.state.blend_set("ALPHA")
        if len(raw) >= 2:
            raw_batch = batch_for_shader(shader, "LINE_STRIP", {"pos": [tuple(point) for point in raw]})
            shader.bind()
            shader.uniform_float("color", (0.64, 0.72, 0.82, 0.34))
            try:
                gpu.state.line_width_set(1.0)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            raw_batch.draw(shader)
        if len(generated) >= 2:
            smooth_batch = batch_for_shader(
                shader, "LINE_STRIP", {"pos": [tuple(point) for point in generated]}
            )
            shader.bind()
            shader.uniform_float("color", (1.0, 0.28, 0.08, 0.98))
            try:
                gpu.state.line_width_set(2.5)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            smooth_batch.draw(shader)
        try:
            gpu.state.line_width_set(1.0)
            gpu.state.blend_set("NONE")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        try:
            gpu.state.blend_set("NONE")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


def _guided_ridge_cancel(state=None, reason="cancel"):
    current = _runtime.guided_ridge_state
    if state is not None and current is not state:
        return
    if current is None:
        return
    current["active"] = False
    timer = current.get("timer")
    if timer is not None:
        window_manager = current.get("window_manager")
        if window_manager is None:
            window_manager = bpy.context.window_manager
        try:
            window_manager.event_timer_remove(timer)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        current["timer"] = None
    current["prepare_job"] = None
    current["compute_job"] = None
    current["width_frame_job"] = None
    transaction = _runtime.guided_ridge_native_transaction
    if transaction is not None and transaction.get("state") is current:
        _runtime.guided_ridge_native_transaction = None
        _runtime.guided_ridge_native_transaction_timer = None
        try:
            callback = _guided_ridge_native_transaction_tick
            if bpy.app.timers.is_registered(callback):
                bpy.app.timers.unregister(callback)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    _curve_sculpt.cleanup(current)
    _guided_ridge_stop_draw()
    _runtime.guided_ridge_state = None
    _guided_ridge_overlay_tag(current)


def _guided_ridge_request_cancel(state=None, reason="external-change"):
    current = _runtime.guided_ridge_state
    if state is not None and current is not state:
        return
    if current is None:
        return
    entry = _lifecycle.modal_request_cancel(
        operator=current.get("operator"),
        reason=reason,
    )
    _guided_ridge_cancel(current, reason)
    _lifecycle.modal_schedule_terminal_event(
        operator=current.get("operator"),
        token=entry.get("token") if entry else None,
        state=current,
    )


def _guided_ridge_draw():
    state = _runtime.guided_ridge_state
    if state is None or not state.get("active"):
        return
    try:
        context = bpy.context
        if context.area is None or int(context.area.as_pointer()) != int(state["area_key"]):
            return
        shader = _fill_preview_shader_get()
        if shader is None:
            return
        guide = state.get("guide") or []
        curve_preview = state.get("phase") in {"curve_preview", "curve_sculpt"}
        if curve_preview:
            # Curve Preview is fitted and drawn in POST_PIXEL from the
            # current-view 2D projection.  Do not render the old world-space
            # smoothing path here; it can appear kinked after perspective.
            return
        if len(guide) < 1:
            return
        coords = [tuple(point) for point in guide]
        if len(coords) >= 2 and not curve_preview:
            batch = batch_for_shader(shader, "LINE_STRIP", {"pos": coords})
            gpu.state.blend_set("ALPHA")
            shader.bind()
            shader.uniform_float("color", (0.12, 0.82, 1.0, 0.95))
            batch.draw(shader)
        rail_colors = (
            ("left", (0.28, 0.70, 1.0, 0.72)),
            ("right", (0.28, 0.70, 1.0, 0.72)),
        )
        for side, color in rail_colors:
            rail = state.get("width_rails", {}).get(side, ())
            segment = []
            for point in list(rail) + [None]:
                if point is None:
                    if len(segment) >= 2:
                        rail_batch = batch_for_shader(shader, "LINE_STRIP", {"pos": segment})
                        shader.bind()
                        shader.uniform_float("color", color)
                        rail_batch.draw(shader)
                    segment = []
                else:
                    segment.append(tuple(point))
        if not curve_preview:
            points = batch_for_shader(shader, "POINTS", {"pos": coords})
            shader.bind()
            shader.uniform_float("color", (1.0, 0.35, 0.08, 1.0))
            try:
                gpu.state.point_size_set(8.0)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            points.draw(shader)
        gpu.state.blend_set("NONE")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        try:
            gpu.state.blend_set("NONE")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


def _guided_ridge_draw_text():
    state = _runtime.guided_ridge_state
    if state is None or not state.get("active"):
        return
    try:
        context = bpy.context
        if context.area is None or int(context.area.as_pointer()) != int(state["area_key"]):
            return
        width = int(getattr(state.get("region"), "width", 0))
        height = int(getattr(state.get("region"), "height", 0))
        font_id = 0
        blf.size(font_id, 13)
        if state.get("phase") == "prepare":
            stage = str(state.get("prepare_stage", "preparing"))
            done = int(state.get("prepare_stage_done", 0))
            total = int(state.get("prepare_stage_total", 0))
            fraction = state.get("prepare_progress_fraction")
            if state.get("prepare_progress_indeterminate") or fraction is None:
                progress_text = "processing..."
            else:
                progress_text = f"{done}/{max(total, 1)}"
            lines = [
                "Guided Ridge: preparing",
                f"{stage}  {progress_text}",
                "Esc: cancel",
            ]
        elif state.get("phase") == "width_prepare":
            stage = str(state.get("width_frame_stage", "building guide frame"))
            done = int(state.get("width_frame_stage_done", 0))
            total = int(state.get("width_frame_stage_total", 0))
            fraction = state.get("width_frame_progress_fraction")
            if state.get("width_frame_progress_indeterminate") or fraction is None:
                progress_text = "processing..."
            else:
                progress_text = f"{done}/{max(total, 1)}"
            lines = [
                "Guided Ridge: rebuilding guide frame",
                f"{stage}  {progress_text}",
                "Esc: cancel",
            ]
        elif state.get("phase") == "compute":
            stage = str(state.get("compute_stage", "computing"))
            done = int(state.get("compute_stage_done", 0))
            total = int(state.get("compute_stage_total", 0))
            fraction = state.get("compute_progress_fraction")
            if state.get("compute_progress_indeterminate") or fraction is None:
                progress_text = "processing..."
            else:
                progress_text = f"{done}/{max(total, 1)}"
            lines = [
                "Guided Ridge: computing",
                f"{stage}  {progress_text}",
                "Esc: cancel",
            ]
        elif state.get("phase") in {"curve_preview", "curve_sculpt"}:
            smoothing = float(state.get("curve_smoothing", _runtime.guided_ridge_curve_last_smoothing) or 0.0)
            warning = state.get("curve_projection_warning") or state.get("curve_shape_warning")
            effective = float(state.get("curve_effective_smoothing", smoothing) or 0.0)
            if state.get("curve_shape_status") == "straight":
                shape_label = "Neutral / straight"
            elif state.get("curve_shape_status") == "retained":
                shape_label = f"Showing Shape {effective:.0f}"
            else:
                shape_label = (
                    "Attenuated"
                    if smoothing < 0.0
                    else ("Raw" if abs(smoothing) < 1.0e-9 else "Amplified")
                )
                if abs(effective - smoothing) > 1.0e-6:
                    shape_label += f" (effective {effective:.0f})"
            if state.get("phase") == "curve_sculpt":
                lines = [
                    "GUIDED RIDGE - CURVE SCULPT / ACTIVE",
                    f"Curve Shape: {smoothing:.0f}/100 ({shape_label})   Ridge: {int(state.get('curve_sculpt_ridge_applications', 0))}  Groove: {int(state.get('curve_sculpt_groove_applications', 0))}",
                    "Enter: Pinch Ridge   Ctrl+Enter: Crease Polish Valley   Backspace: edit route",
                    "Tab: finish   Esc/RMB: exit (applied sculpt remains)",
                    str(warning) if warning else "Current-view Paint Curve is ready",
                ]
            else:
                lines = [
                    "GUIDED RIDGE - CURVE SCULPT / CURVE PREVIEW",
                    f"Curve Shape: {smoothing:.0f}/100 ({shape_label})   Wheel: adjust  Shift+Wheel: fine",
                    "Current-view 2D Bezier preview (dense adaptive tessellation)",
                    "Enter: Pinch Ridge   Ctrl+Enter: Crease Polish Valley",
                    "Backspace: edit route   Esc/RMB: cancel" if not warning else str(warning),
                ]
        else:
            if state.get("prototype_route_only"):
                lines = [
                    "GUIDED RIDGE - CURVE SCULPT STEP 1 / ROUTE EDIT",
                    "LMB: snapped route point   Enter: カーブプレビュー",
                    "Backspace: undo point   Esc/RMB: 取消",
                ]
            else:
                width_value = float(state.get("half_width", 0.0) or 0.0)
                width_min = float(state.get("width_min", 0.0) or 0.0)
                width_max = float(state.get("width_max", 0.0) or 0.0)
                lines = [
                    "Guided Ridge: ready",
                    f"幅 ±{width_value:.5g} (edge {width_value / max(float(state.get('snapshot', {}).get('average_edge', 1.0) or 1.0), 1.0e-12):.2f}x; {width_min:.3g}..{width_max:.3g})",
                    "Wheel/Shift+Wheel: 幅調整  LMB: 点追加  Enter: カーブプレビュー  Esc/RMB: 取消",
                ]
        x, y = max(18, width - 620), max(100, height - 54)
        for index, line in enumerate(lines):
            blf.position(font_id, x, y - index * 18, 0)
            blf.color(font_id, 0.92, 0.96, 1.0, 1.0)
            blf.draw(font_id, line)
        # Two screen-space buttons are deliberately kept away from the guide.
        for label, rect, color in (
            ("カーブプレビュー", state["apply_rect"], (0.16, 0.55, 0.25, 0.95)),
            ("取消", state["cancel_rect"], (0.55, 0.16, 0.16, 0.95)),
        ):
            x0, y0, x1, y1 = rect
            gpu.state.blend_set("ALPHA")
            shader = _fill_preview_shader_get()
            if shader is not None:
                shader.bind()
                shader.uniform_float("color", color)
                batch_for_shader(shader, "TRIS", {"pos": ((x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1))}).draw(shader)
            blf.position(font_id, x0 + 14, y0 + 10, 0)
            blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
            blf.draw(font_id, label)
        gpu.state.blend_set("NONE")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        try:
            gpu.state.blend_set("NONE")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


def _guided_ridge_event_in_rect(event, rect):
    x = float(getattr(event, "mouse_region_x", -1))
    y = float(getattr(event, "mouse_region_y", -1))
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def _guided_ridge_navigation_gizmo_rect(context, state=None):
    """Return a conservative top-right Navigation Gizmo hit zone.

    Blender reports mouse coordinates in region pixels while the visible
    gizmo scales with the UI scale.  The zone is intentionally limited to the
    upper-right corner so a normal guide click remains owned by this modal.
    """
    try:
        region = (state or {}).get("region") or context.region
        width = float(getattr(region, "width", 0) or 0)
        height = float(getattr(region, "height", 0) or 0)
        space = getattr(context, "space_data", None)
        if state is not None and state.get("space") is not None:
            space = state.get("space")
        # A top-right rectangle must never claim guide clicks when either the
        # general gizmo or its navigation subset is hidden.  Blender 5.2
        # exposes both properties on SpaceView3D; missing properties are a
        # conservative no-zone result for fake/older contexts.
        if space is None or not bool(getattr(space, "show_gizmo", False)):
            return None
        if not bool(getattr(space, "show_gizmo_navigate", False)):
            return None
        preferences = getattr(context, "preferences", None)
        system = getattr(preferences, "system", None)
        ui_scale = float(getattr(system, "ui_scale", 1.0) or 1.0)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    size = min(max(96.0 * ui_scale, 64.0), width * 0.26, height * 0.38)
    if size < 48.0:
        return None
    inset = max(4.0, 8.0 * ui_scale)
    return (max(0.0, width - size - inset), max(0.0, height - size - inset), width - inset, height - inset)


def _guided_ridge_navigation_passthrough(context, event, state):
    """Recognize viewport navigation before guide point/confirm handling.

    The modal remains the owner of Wheel width changes.  Middle mouse, NDOF,
    numpad view events, and the Navigation Gizmo are handed back to Blender
    so orbit/pan/zoom can update the view while the captured world-space
    snapshot and guide remain intact.
    """
    if state is None:
        return False
    event_type = str(getattr(event, "type", "") or "")
    event_value = getattr(event, "value", None)
    gizmo_rect = _guided_ridge_navigation_gizmo_rect(context, state)
    in_gizmo = gizmo_rect is not None and _guided_ridge_event_in_rect(event, gizmo_rect)
    gizmo_active = bool(state.get("navigation_gizmo_active"))
    mmb_active = bool(state.get("navigation_mmb_active"))
    if gizmo_active:
        if event_type == "LEFTMOUSE" and event_value == "RELEASE":
            state["navigation_gizmo_active"] = False
        if event_type in {"LEFTMOUSE", "MOUSEMOVE"}:
            return True
    if event_type == "LEFTMOUSE" and event_value == "PRESS" and in_gizmo:
        state["navigation_gizmo_active"] = True
        return True
    if event_type == "MOUSEMOVE" and in_gizmo:
        return True
    if event_type == "MIDDLEMOUSE":
        state["navigation_mmb_active"] = event_value != "RELEASE"
        return True
    if mmb_active and event_type == "MOUSEMOVE":
        return True
    if event_type.startswith("NDOF_"):
        return True
    if event_type in {
        "NUMPAD_0", "NUMPAD_1", "NUMPAD_2", "NUMPAD_3", "NUMPAD_4",
        "NUMPAD_5", "NUMPAD_6", "NUMPAD_7", "NUMPAD_8", "NUMPAD_9",
        "NUMPAD_PERIOD", "NUMPAD_SLASH", "NUMPAD_ASTERIX",
    }:
        return True
    if event_type in {"HOME", "END", "PAGEUP", "PAGEDOWN"}:
        return True
    return False


def _guided_ridge_conflict_reason(context):
    if _runtime.guided_ridge_state is not None and _runtime.guided_ridge_state.get("active"):
        return "Guided Ridge is already active"
    if _runtime.fill_preview_state is not None and _runtime.fill_preview_state.get("active"):
        return "Smart Fill is active"
    if _runtime.tube_preview_state is not None and _runtime.tube_preview_state.get("active"):
        return "Tube Shape is active"
    try:
        area_key = int(context.area.as_pointer())
        existing = _runtime.active_states.get(area_key)
        if existing is not None and getattr(existing, "active", False):
            # MFO's orbit/focus state is deliberately independent from the
            # Guided Ridge modal.  Keep its camera, visibility proxy, and
            # teardown ownership untouched while the active Sculpt object is
            # used by Guided Ridge.  Other modal tools still remain exclusive.
            if isinstance(existing, (_TempOrbitState, _FaceSetOrbitState)):
                return None
            return "another MFO modal session is active"
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return "MFO viewport state is unavailable"
    return None


class VIEW3D_OT_mesh_focus_guided_ridge(bpy.types.Operator):
    """Place a snapped route and review the Step 1 Curve Sculpt preview."""

    bl_idname = GUIDED_RIDGE_OPERATOR_ID
    bl_label = "Guided Ridge"
    bl_description = "Place a snapped route and preview a smoothed Curve Sculpt line"
    # Keep the modal route operator out of Blender's own undo stack.  The
    # native Paint Curve operator invoked by the explicit apply action owns the
    # actual mesh edit and its standard Undo step.
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "active_object", None)
        attr = _guided_ridge_face_set_attribute(obj) if obj is not None else None
        prototype_route = _guided_ridge_prototype_route_enabled()
        conflict = _guided_ridge_conflict_reason(context)
        # An active state is accepted only so EXEC_DEFAULT can route through
        # the same production commit path.  invoke() rejects a second modal
        # session explicitly below; a state-free EXEC_DEFAULT remains denied.
        active_guided_state = _runtime.guided_ridge_state is not None and _runtime.guided_ridge_state.get("active")
        if conflict is not None and not (active_guided_state and conflict == "Guided Ridge is already active"):
            return False
        return bool(
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
            and context.mode == "SCULPT"
            and obj is not None
            and obj.type == "MESH"
            and (prototype_route or attr is not None)
        )

    def _commit(self, context, state, _compute_sync=False, _candidate_override=None):
        import numpy as np
        # Curve Sculpt owns the accepted screen-space route.  The legacy
        # width/deformation candidate remains available for non-curve sessions
        # and isolated reference tests, but an interactive curve session uses
        # the dedicated native Paint Curve service instead.
        if GUIDED_RIDGE_UI_PROTOTYPE and state.get("curve_preview_session"):
            operator = self
            if state.get("phase") in {"curve_preview", "curve_sculpt"}:
                operator.report({"INFO"}, "Guided Ridge Curve Sculpt: use the Curve Sculpt apply path")
                return False
            if state.get("phase") == "ready":
                return _guided_ridge_begin_curve_preview(context, state)
            return _guided_ridge_preview_only_finish(operator, state, "curve-preview-guard")
        # Phase 1 is intentionally a UI-only build.  Keep this guard ahead of
        # candidate construction, integrity checks, and all mesh assignments so
        # neither Enter nor any legacy/internal commit route can reach the
        # deformation engine.
        if state.get("phase") == "ready" and not _compute_sync and _candidate_override is None and not state.get("repeat"):
            return _guided_ridge_begin_compute(context, state) and False
        if state.get("phase") == "compute" and not _compute_sync:
            return False
        if len(state.get("controls", ())) < 2:
            self.report({"WARNING"}, "Guided Ridge: at least two guide points are required")
            return False
        safety_reason = _guided_ridge_safety_reason(state.get("obj"))
        if safety_reason is not None:
            self.report({"WARNING"}, f"Guided Ridge: {safety_reason}; no changes applied")
            _guided_ridge_cancel(state, "unsafe-commit")
            return False
        if not _guided_ridge_context_matches(context, state):
            self.report({"WARNING"}, "Guided Ridge: viewport/object context changed; no changes applied")
            _guided_ridge_cancel(state, "context-changed")
            return False
        pending_anchors, anchor_reason = _guided_ridge_surface_anchors(
            state["snapshot"], state.get("controls", ())
        )
        if pending_anchors is None:
            self.report({"WARNING"}, f"Guided Ridge: {anchor_reason or 'surface anchors are invalid'}")
            return False
        try:
            pending_last_guide, guide_reason = _guided_ridge_build_last_guide(state, pending_anchors)
        except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as guide_error:
            pending_last_guide, guide_reason = None, str(guide_error)
        if pending_last_guide is None:
            self.report({"WARNING"}, f"Guided Ridge: guide save preparation failed ({guide_reason})")
            return False
        if state.get("repeat"):
            current = np.asarray(state["snapshot"].get("coords_local", ()), dtype=np.float64).copy()
            if len(current) != len(state["snapshot"].get("vertex_indices", ())):
                self.report({"WARNING"}, "Guided Ridge Repeat cancelled: current coordinates are unavailable")
                return False
        else:
            current = _guided_ridge_current_signature(state["obj"], state["snapshot"])
        if current is None:
            self.report({"WARNING"}, "Guided Ridge: source mesh, Face Set, or coordinates changed")
            return False
        try:
            if _candidate_override is None:
                projected, reason = _guided_ridge_project_curve(state["snapshot"], state["controls"])
                if projected is None:
                    self.report({"WARNING"}, f"Guided Ridge: {reason}")
                    return False
                candidate, info = _guided_ridge_exact_candidate(
                    state["snapshot"],
                    projected,
                    state["controls"],
                    state["control_normals"],
                    state.get("half_width"),
                    width_result=state.get("width_frame_result"),
                )
            else:
                projected = state.get("guide")
                candidate, info = _candidate_override
            integrity = _guided_ridge_mesh_integrity(state["snapshot"], state["snapshot"]["coords_world"], candidate)
            if not integrity["finite"] or integrity["normal_pair_flips"] or integrity["new_degenerate_faces"]:
                self.report({"WARNING"}, "Guided Ridge: unsafe geometry candidate; no changes applied")
                return False
            delta = np.asarray(candidate, dtype=np.float64) - np.asarray(state["snapshot"]["coords_world"], dtype=np.float64)
            outward_component = np.asarray(info.get("outward_component", np.zeros(len(delta))), dtype=np.float64)
            if "delta_h" in info and np.any(np.asarray(info["delta_h"]) * outward_component > 1.0e-10):
                self.report({"WARNING"}, "Guided Ridge: planing direction is unsafe; no changes applied")
                return False
            anchor_mask = np.asarray(info.get("anchor_mask", np.zeros(len(delta), dtype=bool)), dtype=bool)
            crest_mask = np.asarray(info.get("crest_side_mask", np.ones(len(delta), dtype=bool)), dtype=bool)
            if np.any(np.linalg.norm(delta[anchor_mask], axis=1) > 1.0e-9) or np.any(np.linalg.norm(delta[~crest_mask], axis=1) > 1.0e-9):
                self.report({"WARNING"}, "Guided Ridge: fixed crest or backside moved; no changes applied")
                return False
            rail_support_mask = np.asarray(
                info.get("rail_support_mask", np.zeros(len(delta), dtype=bool)), dtype=bool
            )
            if len(rail_support_mask) != len(delta) or np.any(
                np.linalg.norm(delta[rail_support_mask], axis=1) > 1.0e-9
            ):
                self.report({"WARNING"}, "Guided Ridge: projected rail support moved; no changes applied")
                return False
            if not _guided_ridge_rail_support_unchanged(
                state["snapshot"],
                state["snapshot"]["coords_world"],
                candidate,
                info.get("rail_hits", {}),
            ):
                self.report({"WARNING"}, "Guided Ridge: projected rail support moved; no changes applied")
                return False
            obj = state["obj"]
            inverse = obj.matrix_world.inverted_safe()
            candidate_world = np.asarray(candidate, dtype=np.float64)
            snapshot_world = np.asarray(state["snapshot"]["coords_world"], dtype=np.float64)
            changed_mask = np.linalg.norm(candidate_world - snapshot_world, axis=1) > 1.0e-12
            changed_indices = np.flatnonzero(changed_mask).astype(np.int64)
            changed_local = {
                int(local_index): tuple(inverse @ Vector(candidate_world[int(local_index)]))
                for local_index in changed_indices
            }
            if changed_local and not np.all(np.isfinite(np.asarray(tuple(changed_local.values()), dtype=np.float64))):
                self.report({"WARNING"}, "Guided Ridge: non-finite local coordinates")
                return False
            before_local = np.asarray(current, dtype=np.float64).copy()
            mesh = obj.data
            try:
                for write_index, local_index in enumerate(changed_indices):
                    vertex_index = state["snapshot"]["vertex_indices"][int(local_index)]
                    mesh.vertices[int(vertex_index)].co = changed_local[int(local_index)]
                    failure_after = state.get("_test_failure_after")
                    if failure_after is not None and write_index >= int(failure_after):
                        raise RuntimeError("injected Guided Ridge write failure")
                mesh.update()
                obj.update_tag(refresh={"DATA"})
                context.view_layer.update()
            except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as write_error:
                rollback_ok, rollback_error = _guided_ridge_restore_coordinates(context, state, before_local)
                if rollback_ok:
                    self.report({"WARNING"}, f"Guided Ridge: commit failed; coordinates rolled back ({write_error})")
                else:
                    self.report({"ERROR"}, f"Guided Ridge: commit and rollback failed ({write_error}; {rollback_error})")
                    _guided_ridge_cancel(state, "rollback-failed")
                return False
            # The complete Repeat Last payload was prepared before writing.  The
            # assignment is intentionally the only post-write bookkeeping step.
            _runtime.guided_ridge_last_guide = pending_last_guide
            state["committed"] = True
            self.report({"INFO"}, f"Guided Ridge: applied to {len(state['snapshot']['vertex_indices'])} vertices")
            _guided_ridge_cancel(state, "confirm")
            return True
        except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
            self.report({"WARNING"}, f"Guided Ridge: commit failed ({error})")
            return False

    def _execute_repeat(self, context, last_guide):
        """Rebuild a current component snapshot and route Repeat Last through _commit."""
        if GUIDED_RIDGE_CURVE_SCULPT_STEP1:
            self.report({"INFO"}, "Guided Ridge Curve Sculpt Step 1: Repeat Last is preview-only; no geometry changes")
            return True
        if GUIDED_RIDGE_UI_PROTOTYPE:
            self.report({"INFO"}, "Guided Ridge UI Prototype: Repeat Last preview only; no geometry changes")
            return True
        def reject(message):
            if _runtime.guided_ridge_last_guide is last_guide:
                _runtime.guided_ridge_last_guide = None
            self.report({"WARNING"}, message)
            return False

        if not self.poll(context):
            return reject("Guided Ridge Repeat cancelled: SCULPT Face Set context required")
        obj = context.active_object
        try:
            object_pointer = int(obj.as_pointer())
            mesh_pointer = int(obj.data.as_pointer())
            if object_pointer != int(last_guide.get("object_pointer", 0)):
                return reject("Guided Ridge Repeat cancelled: object RNA datablock changed")
            if mesh_pointer != int(last_guide.get("mesh_pointer", 0)):
                return reject("Guided Ridge Repeat cancelled: mesh RNA datablock changed")
            if str(getattr(obj, "name", "")) != str(last_guide.get("object_name", "")):
                return reject("Guided Ridge Repeat cancelled: active object changed")
            if str(getattr(obj.data, "name", "")) != str(last_guide.get("data_name", "")):
                return reject("Guided Ridge Repeat cancelled: active mesh changed")
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            return reject("Guided Ridge Repeat cancelled: active object is unavailable")
        safety_reason = _guided_ridge_safety_reason(obj)
        if safety_reason is not None:
            return reject(f"Guided Ridge Repeat cancelled: {safety_reason}; no changes applied")
        snapshot, reason = _guided_ridge_prepare_repeat_snapshot(obj, last_guide)
        if snapshot is None:
            return reject(reason or "Guided Ridge Repeat cancelled: component changed")
        controls, control_normals, reason = _guided_ridge_repeat_controls(snapshot, last_guide)
        if controls is None:
            return reject(reason or "Guided Ridge Repeat cancelled: guide anchors are invalid")
        context_signature = _guided_ridge_context_signature(context)
        if context_signature is None:
            return reject("Guided Ridge Repeat cancelled: viewport context is unavailable")
        snapshot["context_signature"] = context_signature
        repeat_state = {
            "active": True,
            "repeat": True,
            "operator": self,
            "obj": obj,
            "snapshot": snapshot,
            "controls": controls,
            "control_normals": control_normals,
            "guide": list(controls),
            "half_width": float(last_guide.get("half_width", 0.0) or 0.0),
            "width_frame": None,
            "width_frame_result": None,
            "width_frame_job": None,
            "width_rails": {"left": [], "right": []},
            "last_cursor": controls[-1],
            "area": context.area,
            "area_key": int(context.area.as_pointer()),
            "region": context.region,
            "space": getattr(context, "space_data", None),
            "window_manager": context.window_manager,
            "window_key": int(context.window.as_pointer()) if context.window else 0,
            "draw_handler": None,
            "text_draw_handler": None,
            "committed": False,
        }
        _runtime.guided_ridge_state = repeat_state
        try:
            result = bool(self._commit(context, repeat_state))
            if not result:
                _runtime.guided_ridge_last_guide = None
            return result
        finally:
            if _runtime.guided_ridge_state is repeat_state:
                _guided_ridge_cancel(repeat_state, "repeat-finished")

    def execute(self, context):
        """Execute the active modal commit or Blender's standard Repeat Last path."""
        state = _runtime.guided_ridge_state
        if state is None or not state.get("active"):
            if _runtime.guided_ridge_last_guide is None:
                self.report({"WARNING"}, "Guided Ridge: no saved guide is available for Repeat Last")
                return {"CANCELLED"}
            return {"FINISHED"} if self._execute_repeat(context, _runtime.guided_ridge_last_guide) else {"CANCELLED"}
        try:
            if context.area is None or int(context.area.as_pointer()) != int(state["area_key"]):
                self.report({"WARNING"}, "Guided Ridge: EXEC_DEFAULT context does not match the active guide")
                return {"CANCELLED"}
            if context.active_object is not state["obj"] or context.mode != "SCULPT":
                self.report({"WARNING"}, "Guided Ridge: EXEC_DEFAULT active object does not match the guide")
                return {"CANCELLED"}
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            self.report({"WARNING"}, "Guided Ridge: EXEC_DEFAULT context is unavailable")
            return {"CANCELLED"}
        # A WorkSpaceTool keymap may dispatch an additional EXEC_DEFAULT while
        # the preparation timer is still active.  Never let that direct path
        # reach _commit (or Repeat Last); the modal timer owns preparation and
        # subsequent point/confirm events.
        if state.get("phase") == "prepare":
            return {"CANCELLED"}
        if state.get("phase") == "compute":
            return {"CANCELLED"}
        if GUIDED_RIDGE_CURVE_SCULPT_STEP1 and state.get("curve_preview_session"):
            if state.get("phase") == "ready":
                return {"FINISHED"} if _guided_ridge_begin_curve_preview(context, state) else {"CANCELLED"}
            if state.get("phase") == "curve_preview":
                self.report({"INFO"}, "Guided Ridge Curve Sculpt: preview only; application comes in Step 3")
                return {"FINISHED"}
        started = self._commit(context, state)
        if state.get("phase") == "compute":
            return {"FINISHED"}
        return {"FINISHED"} if started else {"CANCELLED"}

    def invoke(self, context, event):
        prototype_route = _guided_ridge_prototype_route_enabled()
        resume_transaction = _runtime.guided_ridge_native_transaction
        existing_state = _runtime.guided_ridge_state
        if (
            resume_transaction is not None
            and resume_transaction.get("status") == "resume"
            and existing_state is resume_transaction.get("state")
            and existing_state is not None
        ):
            existing_state["operator"] = self
            existing_state["active"] = True
            existing_state["curve_sculpt_editor_suspended"] = False
            if not _guided_ridge_install_session_handlers(context, existing_state):
                self.report({"WARNING"}, "Guided Ridge: editor session could not be resumed")
                _guided_ridge_cancel(existing_state, "resume-install-failed")
                return {"CANCELLED"}
            try:
                context.window_manager.modal_handler_add(self)
                _lifecycle.modal_register(
                    self,
                    "Guided Ridge",
                    existing_state.get("curve_sculpt_session_token"),
                    state=existing_state,
                )
                _runtime.guided_ridge_native_transaction = None
                _runtime.guided_ridge_native_transaction_timer = None
                return {"RUNNING_MODAL"}
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                _guided_ridge_cancel(existing_state, "resume-modal-failed")
                return {"CANCELLED"}
        if _runtime.guided_ridge_state is not None and _runtime.guided_ridge_state.get("active"):
            self.report({"WARNING"}, "Guided Ridge is already active")
            return {"CANCELLED"}
        if not self.poll(context):
            reason = _guided_ridge_conflict_reason(context)
            self.report({"WARNING"}, f"Guided Ridge: {reason or 'SCULPT Face Set context required'}")
            return {"CANCELLED"}
        coord = _sculpt_cursor_region_coordinate(context, event)
        if coord is None:
            self.report({"WARNING"}, "Guided Ridge: cursor is outside the View3D window region")
            return {"CANCELLED"}
        hit = (
            _guided_ridge_prototype_surface_hit(context, coord)
            if prototype_route
            else _raycast_sculpt_face_set(context, coord)
        )
        if hit is None:
            self.report(
                {"WARNING"},
                "Guided Ridge: visible mesh surface not found under cursor"
                if prototype_route
                else "Guided Ridge: visible Face Set surface not found under cursor",
            )
            return {"CANCELLED"}
        if prototype_route:
            # The route belongs to the launch context's active Sculpt mesh;
            # do not start a session on another visible object merely because
            # it was closer to the click ray.
            if hit[0] is not context.active_object:
                self.report({"WARNING"}, "Guided Ridge: visible hit is outside the active launch mesh")
                return {"CANCELLED"}
            obj, face_index, face_set_id, start_location, start_normal, _screen = hit
        else:
            obj, face_index, face_set_id, _location, _screen = hit
            start_location = None
            start_normal = None
        safety_reason = None if prototype_route else _guided_ridge_safety_reason(obj)
        if safety_reason is not None:
            self.report({"WARNING"}, f"Guided Ridge: {safety_reason}; no changes applied")
            return {"CANCELLED"}
        context_signature = _guided_ridge_context_signature(context)
        if context_signature is None:
            self.report({"WARNING"}, "Guided Ridge: viewport context is unavailable")
            return {"CANCELLED"}
        start_ray = _guided_ridge_ray(context, coord)
        if start_ray is None:
            self.report({"WARNING"}, "Guided Ridge: click ray is unavailable")
            return {"CANCELLED"}
        width = int(getattr(context.region, "width", 0))
        height = int(getattr(context.region, "height", 0))
        _runtime.guided_ridge_state = {
            "active": True,
            "operator": self,
            "obj": obj,
            "snapshot": None,
            "controls": [start_location] if prototype_route else [],
            "control_normals": [start_normal] if prototype_route else [],
            "guide": [start_location] if prototype_route else [],
            "curve_preview_session": bool(GUIDED_RIDGE_CURVE_SCULPT_STEP1),
            "prototype_route_only": prototype_route,
            "curve_route": [],
            "curve_preview": [],
            "curve_screen_raw": [],
            "curve_screen_preview": [],
            "curve_screen_smooth": [],
            "curve_screen_bezier_segments": [],
            "curve_screen_sample_count": 0,
            "curve_screen_fit_segment_count": 0,
            "curve_screen_intersection_checks": 0,
            "curve_screen_generation": 0,
            "curve_screen_signature": None,
            "curve_screen_base_signature": None,
            "curve_screen_cache_status": "empty",
            "curve_screen_refresh_count": 0,
            "curve_idle_cache_hits": 0,
            "curve_route_revision": 0,
            "curve_route_digest_cache": None,
            "preview_context": None,
            "curve_smoothing": float(_runtime.guided_ridge_curve_last_smoothing),
            "curve_preview_generation": 0,
            "curve_projection_warning": None,
            "curve_sculpt_active": False,
            "curve_sculpt_mode": None,
            "curve_sculpt_restore": None,
            "curve_sculpt_restore_asset_reference": None,
            "curve_sculpt_paint_curve": None,
            "curve_sculpt_session_token": f"guided-ridge-{id(self):x}-{int(time.time_ns())}",
            "curve_sculpt_owned_brushes": {},
            "curve_sculpt_managed_brushes": {},
            "curve_sculpt_managed_brush_settings": {},
            "curve_sculpt_managed_preexisting_curves": {},
            "curve_sculpt_owned_curves": {},
            "curve_sculpt_brushes_by_mode": {},
            "curve_sculpt_created_brushes": [],
            "curve_sculpt_created_curves": [],
            "curve_sculpt_last_apply": None,
            "curve_sculpt_ridge_applications": 0,
            "curve_sculpt_groove_applications": 0,
            "curve_sculpt_apply_generation": 0,
            "curve_sculpt_apply_active": False,
            "curve_sculpt_expected_update_generation": None,
            "curve_sculpt_expected_update_budget": 0,
            "curve_sculpt_expected_mesh_signature": None,
            "curve_sculpt_target_object_pointer": None,
            "curve_sculpt_target_mesh_pointer": None,
            "curve_view_signature": None,
            "half_width": 0.0,
            "width_min": 0.0,
            "width_max": 0.0,
            "width_frame": None,
            "width_frame_result": None,
            "width_frame_job": None,
            "width_rails": {"left": [], "right": []},
            "width_cache_serial": 0,
            "last_cursor": None,
            "phase": "ready" if prototype_route else "prepare",
            "start_coordinate": coord.copy(),
            "start_ray": (start_ray[0].copy(), start_ray[1].copy()),
            "start_key": str(getattr(event, "type", "LEFTMOUSE") or "LEFTMOUSE"),
            "start_key_released": False,
            "start_guard_active": True,
            "context_signature": context_signature,
            # Step 1 keeps the launch identities separately from the legacy
            # snapshot signature because its route has no Face Set snapshot.
            "object_pointer": int(obj.as_pointer()) if hasattr(obj, "as_pointer") else 0,
            "mesh_pointer": int(obj.data.as_pointer()) if hasattr(obj.data, "as_pointer") else 0,
            "prepare_job": None if prototype_route else _guided_ridge_prepare_snapshot_steps(obj, face_index, face_set_id),
            "prepare_stage": "ready" if prototype_route else "queued",
            "prepare_stage_index": 0,
            "prepare_stage_count": 13,
            "prepare_stage_done": 1 if prototype_route else 0,
            "prepare_stage_total": 1 if prototype_route else 0,
            "prepare_progress_fraction": 1.0 if prototype_route else None,
            "prepare_progress_indeterminate": False if prototype_route else True,
            "prepare_slices": 0,
            "prepare_elapsed_seconds": 0.0,
            "prepare_last_slice_seconds": 0.0,
            "prepare_max_slice_seconds": 0.0,
            "compute_job": None,
            "compute_stage": "idle",
            "compute_stage_index": 0,
            "compute_stage_count": 4,
            "compute_stage_done": 0,
            "compute_stage_total": 0,
            "compute_progress_fraction": None,
            "compute_progress_indeterminate": True,
            "compute_elapsed_seconds": 0.0,
            "compute_last_slice_seconds": 0.0,
            "compute_max_slice_seconds": 0.0,
            "modal_result": None,
            "timer": None,
            "area": context.area,
            "area_key": int(context.area.as_pointer()),
            "region": context.region,
            "region_key": int(context.region.as_pointer()) if hasattr(context.region, "as_pointer") else 0,
            "space": getattr(context, "space_data", None),
            "space_key": int(context.space_data.as_pointer())
            if getattr(context, "space_data", None) is not None and hasattr(context.space_data, "as_pointer")
            else 0,
            "window_manager": context.window_manager,
            "window_key": int(context.window.as_pointer()) if context.window else 0,
            "draw_handler": None,
            "text_draw_handler": None,
            "screen_curve_draw_handler": None,
            "navigation_gizmo_active": False,
            "navigation_mmb_active": False,
            "apply_rect": (max(18, width - 230), 22, max(19, width - 140), 54),
            "cancel_rect": (max(24, width - 130), 22, max(25, width - 40), 54),
            "committed": False,
        }
        state = _runtime.guided_ridge_state
        if prototype_route:
            # The first hit is the route's initial snapped point.  There is no
            # snapshot/BVH preparation in Step 1: subsequent clicks use the
            # same read-only visible-mesh raycast and the preview curve works
            # directly from these retained world-space points.
            state["curve_route"] = [start_location]
            state["last_cursor"] = start_location
        try:
            if not _guided_ridge_install_session_handlers(context, state):
                raise RuntimeError("Guided Ridge editor handlers could not be installed")
            context.window_manager.modal_handler_add(self)
            _lifecycle.modal_register(
                self,
                "Guided Ridge",
                state.get("curve_sculpt_session_token"),
                state=state,
            )
            _guided_ridge_overlay_tag(state)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _guided_ridge_cancel(state, "start")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        state = _runtime.guided_ridge_state
        entry = _lifecycle.modal_entry(operator=self)
        if entry is not None and (
            entry.get("cancel_requested") or entry.get("teardown_requested")
        ):
            if state is not None:
                _guided_ridge_cancel(state, entry.get("cancel_reason") or "external-change")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        if state is None or state.get("operator") is not self:
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        event_type = getattr(event, "type", "")
        event_value = getattr(event, "value", None)
        history_gate = "none"
        if state.get("phase") in {"curve_preview", "curve_sculpt"}:
            history_gate = _guided_ridge_history_modal_gate(context, state, event_type)
            if history_gate == "cancel":
                self.report({"WARNING"}, "Guided Ridge: Undo/Redo context could not be reacquired")
                _guided_ridge_request_cancel(state, "history-context-changed")
                _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                return {"CANCELLED"}
            if history_gate == "wait":
                if event_type in {
                    "MIDDLEMOUSE",
                    "WHEELUPMOUSE",
                    "WHEELDOWNMOUSE",
                    "NDOF_MOTION",
                    "MOUSEMOVE",
                }:
                    return {"PASS_THROUGH"}
                if event_type not in {"ESC", "RIGHTMOUSE"}:
                    return {"RUNNING_MODAL"}
        if (state.get("prototype_route_only") or state.get("phase") in {"curve_preview", "curve_sculpt"}) and not _guided_ridge_route_context_matches(context, state):
            self.report({"WARNING"}, "Guided Ridge: launch object or View3D context changed; no changes applied")
            _guided_ridge_request_cancel(state, "route-context-changed")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        if event_type in {"ESC", "RIGHTMOUSE"} and event_value in {None, "PRESS"}:
            _guided_ridge_cancel(state, "cancel")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        if event_type == "TAB" and event_value in {None, "PRESS"} and state.get("phase") in {"curve_preview", "curve_sculpt"}:
            self.report({"INFO"}, "Guided Ridge: Curve Sculpt finished; use the active brush for manual Scrape finish")
            _guided_ridge_cancel(state, "curve-sculpt-finish")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        if state.get("start_guard_active") and event_type == state.get("start_key"):
            if event_value == "RELEASE":
                state["start_key_released"] = True
                state["start_guard_active"] = False
            # The initial PRESS/RELEASE pair is never a guide point or a
            # confirmation event.  This is especially important for the
            # LEFTMOUSE WorkSpaceTool entry.
            return {"RUNNING_MODAL"}
        if event_type == "TIMER":
            expected_timer = state.get("timer")
            actual_timer = getattr(event, "timer", None)
            if actual_timer is not None and expected_timer is not None and actual_timer is not expected_timer:
                return {"RUNNING_MODAL"}
            if state.get("phase") == "prepare":
                _guided_ridge_process_prepare_timer(context, state)
                if not state.get("active"):
                    _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                    return {"CANCELLED"}
                return {"CANCELLED" if not state.get("active") else "RUNNING_MODAL"}
            if state.get("phase") == "width_prepare":
                _guided_ridge_process_width_frame_timer(context, state)
                if not state.get("active"):
                    _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                    return {"CANCELLED"}
                return {"CANCELLED" if not state.get("active") else "RUNNING_MODAL"}
            if state.get("phase") == "compute":
                _guided_ridge_process_compute_timer(context, state)
                if state.get("modal_result") == "FINISHED":
                    _lifecycle.modal_terminal(operator=self, status="FINISHED")
                    return {"FINISHED"}
                if not state.get("active"):
                    _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                    return {"CANCELLED"}
                return {"CANCELLED" if not state.get("active") else "RUNNING_MODAL"}
            if state.get("phase") == "curve_preview":
                if state.get("curve_sculpt_restore_pending"):
                    _curve_sculpt.retry_restore(context, state)
                _guided_ridge_curve_refresh_projection(context, state)
                if not state.get("active"):
                    _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                    return {"CANCELLED"}
                return {"CANCELLED" if not state.get("active") else "RUNNING_MODAL"}
            if state.get("phase") == "curve_sculpt":
                if state.get("curve_sculpt_restore_pending"):
                    _curve_sculpt.retry_restore(context, state)
                _guided_ridge_curve_refresh_projection(context, state)
                if not state.get("active"):
                    _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                    return {"CANCELLED"}
                return {"CANCELLED" if not state.get("active") else "RUNNING_MODAL"}
            return {"RUNNING_MODAL"}
        # Navigation must be checked before any phase-specific guide or
        # confirm branch.  In particular a gizmo LMB is not a guide point.
        if _guided_ridge_navigation_passthrough(context, event, state):
            return {"PASS_THROUGH"}
        if state.get("phase") == "prepare":
            # Preparation owns no point/confirm input.  Keep viewport
            # navigation pass-through compatible with the established modal.
            if event_type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
                return {"PASS_THROUGH"}
            return {"RUNNING_MODAL"}
        if state.get("phase") == "width_prepare":
            # A guide edit owns the frame rebuild until its cached rails and
            # width result are complete.  Do not let another LMB or Enter
            # commit against a half-built frame.
            if event_type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
                return {"PASS_THROUGH"}
            return {"RUNNING_MODAL"}
        if state.get("phase") == "compute":
            # The computing timer owns all point/confirm input.  Esc/RMB was
            # handled above and cancels the generator immediately.
            if event_type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
                return {"PASS_THROUGH"}
            return {"RUNNING_MODAL"}
        if event_type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event_value in {None, "PRESS"}:
            if state.get("phase") == "curve_preview":
                current = float(state.get("curve_smoothing", _runtime.guided_ridge_curve_last_smoothing) or 0.0)
                step = 10.0 if not bool(getattr(event, "shift", False)) else 2.0
                if event_type == "WHEELDOWNMOUSE":
                    step = -step
                _guided_ridge_curve_set_smoothing(state, current + step)
                return {"RUNNING_MODAL"}
            if state.get("phase") == "curve_sculpt":
                self.report({"INFO"}, "Guided Ridge: press Backspace to edit the route before changing Shape")
                return {"RUNNING_MODAL"}
            current_width = float(state.get("half_width", state.get("width_min", 0.0)) or 0.0)
            factor = GUIDED_RIDGE_WIDTH_STEP_FACTOR
            if bool(getattr(event, "shift", False)):
                factor = math.sqrt(factor)
            if event_type == "WHEELDOWNMOUSE":
                factor = 1.0 / factor
            minimum = float(state.get("width_min", 0.0) or 0.0)
            maximum = float(state.get("width_max", 0.0) or 0.0)
            state["half_width"] = min(max(current_width * factor, minimum), maximum)
            try:
                _guided_ridge_update_width_preview(state)
            except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
                self.report({"WARNING"}, f"Guided Ridge: width preview unavailable ({error})")
                state["width_rails"] = {"left": [], "right": []}
            _guided_ridge_overlay_tag(state)
            return {"RUNNING_MODAL"}
        if event_type == "BACK_SPACE" and event_value in {None, "PRESS"}:
            if state.get("phase") in {"curve_preview", "curve_sculpt"}:
                _guided_ridge_curve_back_to_route(context, state)
                return {"RUNNING_MODAL"}
            if state.get("prototype_route_only"):
                if len(state.get("controls", ())) > 1:
                    state["controls"].pop()
                    if state.get("control_normals"):
                        state["control_normals"].pop()
                    state["guide"] = list(state["controls"])
                    state["curve_route"] = _guided_ridge_curve_dedupe(state["controls"])
                    state["curve_route_revision"] = int(state.get("curve_route_revision", 0) or 0) + 1
                    state["curve_route_digest_cache"] = None
                    state["last_cursor"] = state["controls"][-1]
                    _guided_ridge_overlay_tag(state)
                return {"RUNNING_MODAL"}
            if len(state["controls"]) > 1:
                state["controls"].pop()
                state["control_normals"].pop()
                state["guide"], _reason = _guided_ridge_project_curve(state["snapshot"], state["controls"])
                if state["guide"] is None:
                    state["guide"] = list(state["controls"])
                try:
                    if not _guided_ridge_begin_width_frame_rebuild(context, state):
                        raise ValueError("guide frame rebuild could not start")
                except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError):
                    state["width_rails"] = {"left": [], "right": []}
                _guided_ridge_overlay_tag(state)
            return {"RUNNING_MODAL"}
        if (
            event_type == "Z"
            and event_value in {None, "PRESS"}
            and bool(getattr(event, "ctrl", False))
            and state.get("phase") == "curve_sculpt"
        ):
            direction = "redo" if bool(getattr(event, "shift", False)) else "undo"
            if _curve_sculpt.request_history_change(state, direction):
                self.report(
                    {"INFO"},
                    "Guided Ridge: Blender standard Redo requested"
                    if direction == "redo"
                    else "Guided Ridge: Blender standard Undo requested",
                )
                # Never call bpy.ops.ed.undo from this long-lived editor
                # modal.  PASS_THROUGH lets Blender own the native history
                # operation; depsgraph validation below refreshes or cancels
                # the session when the resulting signature arrives.
                return {"PASS_THROUGH"}
            return {"RUNNING_MODAL"}
        if event_type in {"RET", "NUMPAD_ENTER", "ENTER"} and event_value in {None, "PRESS"}:
            if state.get("phase") in {"curve_preview", "curve_sculpt"}:
                if not GUIDED_RIDGE_CURVE_SCULPT_STEP2:
                    self.report({"INFO"}, "Guided Ridge Curve Sculpt: application is disabled in this build")
                    return {"RUNNING_MODAL"}
                mode = "GROOVE" if bool(getattr(event, "ctrl", False)) else "RIDGE"
                _guided_ridge_curve_refresh_projection(context, state)
                if _guided_ridge_begin_native_transaction(context, state, mode):
                    self.report(
                        {"INFO"},
                        "Guided Ridge: native one-shot queued; editor will resume after the stroke",
                    )
                    _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                    return {"CANCELLED"}
                self.report({"WARNING"}, "Guided Ridge: native transaction could not be queued")
                return {"RUNNING_MODAL"}
            if state.get("phase") == "ready" and GUIDED_RIDGE_CURVE_SCULPT_STEP1:
                _guided_ridge_begin_curve_preview(context, state)
                return {"RUNNING_MODAL" if state.get("active") else "CANCELLED"}
            self._commit(context, state)
            result = {
                "FINISHED"
                if state.get("committed") or state.get("modal_result") == "FINISHED"
                else "RUNNING_MODAL"
            }
            if result == {"FINISHED"}:
                _lifecycle.modal_terminal(operator=self, status="FINISHED")
            return result
        if event_type == "LEFTMOUSE" and event_value == "PRESS":
            if _guided_ridge_event_in_rect(event, state["apply_rect"]):
                if state.get("phase") in {"curve_preview", "curve_sculpt"}:
                    if not GUIDED_RIDGE_CURVE_SCULPT_STEP2:
                        self.report({"INFO"}, "Guided Ridge Curve Sculpt: application is disabled in this build")
                        return {"RUNNING_MODAL"}
                    mode = "GROOVE" if bool(getattr(event, "ctrl", False)) else "RIDGE"
                    _guided_ridge_curve_refresh_projection(context, state)
                    if _guided_ridge_begin_native_transaction(context, state, mode):
                        self.report(
                            {"INFO"},
                            "Guided Ridge: native one-shot queued; editor will resume after the stroke",
                        )
                        _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                        return {"CANCELLED"}
                    self.report({"WARNING"}, "Guided Ridge: native transaction could not be queued")
                    return {"RUNNING_MODAL"}
                if state.get("phase") == "ready" and GUIDED_RIDGE_CURVE_SCULPT_STEP1:
                    _guided_ridge_begin_curve_preview(context, state)
                    return {"RUNNING_MODAL" if state.get("active") else "CANCELLED"}
                self._commit(context, state)
                result = {
                    "FINISHED"
                    if state.get("committed") or state.get("modal_result") == "FINISHED"
                    else "RUNNING_MODAL"
                }
                if result == {"FINISHED"}:
                    _lifecycle.modal_terminal(operator=self, status="FINISHED")
                return result
            if _guided_ridge_event_in_rect(event, state["cancel_rect"]):
                _guided_ridge_cancel(state, "cancel-button")
                _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                return {"CANCELLED"}
            if state.get("phase") in {"curve_preview", "curve_sculpt"}:
                # Once the route has entered preview or Curve Sculpt, LMB is
                # consumed by the modal.  Only the explicit apply/cancel
                # rectangles above may act; normal route editing resumes via
                # Backspace.  This also prevents a native stroke's LMB from
                # adding a new route point.
                return {"RUNNING_MODAL"}
            coord = _sculpt_cursor_region_coordinate(context, event)
            if coord is None:
                return {"RUNNING_MODAL"}
            if state.get("prototype_route_only"):
                hit = _guided_ridge_prototype_surface_hit(context, coord)
                if not _guided_ridge_prototype_hit_matches_state(hit, state):
                    self.report({"WARNING"}, "Guided Ridge: surface hit is outside the launch mesh; route cancelled")
                    _guided_ridge_request_cancel(state, "route-hit-object-changed")
                    _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                    return {"CANCELLED"}
                if hit is None:
                    self.report({"WARNING"}, "Guided Ridge: visible mesh surface not found under cursor")
                    return {"RUNNING_MODAL"}
                point = hit[3]
                normal = hit[4]
                if state["controls"] and (point - state["controls"][-1]).length <= 1.0e-7:
                    return {"RUNNING_MODAL"}
                if len(state["controls"]) >= GUIDED_RIDGE_MAX_CONTROLS:
                    self.report({"WARNING"}, "Guided Ridge: maximum 64 guide points reached")
                    return {"RUNNING_MODAL"}
                state["controls"] = state["controls"] + [point]
                state["control_normals"] = state.get("control_normals", []) + [normal]
                state["guide"] = list(state["controls"])
                state["curve_route"] = _guided_ridge_curve_dedupe(state["controls"])
                state["curve_route_revision"] = int(state.get("curve_route_revision", 0) or 0) + 1
                state["curve_route_digest_cache"] = None
                state["last_cursor"] = point
                _guided_ridge_overlay_tag(state)
                return {"RUNNING_MODAL"}
            hit = _guided_ridge_snapshot_hit(context, state["snapshot"], coord)
            if hit is None:
                self.report({"WARNING"}, "Guided Ridge: click must hit the same Face Set component")
                return {"RUNNING_MODAL"}
            point = hit[0]
            if state["controls"] and (point - state["controls"][-1]).length <= 1.0e-7:
                return {"RUNNING_MODAL"}
            if len(state["controls"]) >= GUIDED_RIDGE_MAX_CONTROLS:
                self.report({"WARNING"}, "Guided Ridge: maximum 64 guide points reached")
                return {"RUNNING_MODAL"}
            candidate_controls = state["controls"] + [point]
            projected, reason = _guided_ridge_project_curve(state["snapshot"], candidate_controls)
            if projected is None:
                self.report({"WARNING"}, f"Guided Ridge: {reason}")
                return {"RUNNING_MODAL"}
            state["controls"] = candidate_controls
            state["control_normals"] = state["control_normals"] + [hit[1]]
            state["guide"] = projected
            try:
                if not _guided_ridge_begin_width_frame_rebuild(context, state):
                    raise ValueError("guide frame rebuild could not start")
            except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
                self.report({"WARNING"}, f"Guided Ridge: width rail projection unavailable ({error})")
                state["width_rails"] = {"left": [], "right": []}
            _guided_ridge_overlay_tag(state)
            return {"RUNNING_MODAL"}
        if event_type == "MOUSEMOVE":
            coord = _sculpt_cursor_region_coordinate(context, event)
            if coord is None:
                return {"RUNNING_MODAL"}
            hit = _guided_ridge_snapshot_hit(context, state["snapshot"], coord)
            if hit is not None:
                state["last_cursor"] = hit[0]
                _guided_ridge_overlay_tag(state)
            return {"RUNNING_MODAL"}
        if event_type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}


def _guided_ridge_history_resolve_live_object(context, state):
    """Replace a stale saved Object reference with its live RNA object."""
    try:
        expected_pointer = int(state.get("object_pointer", 0) or 0)
        if expected_pointer <= 0:
            return None
        active_object = getattr(context, "active_object", None)
        if active_object is not None and int(active_object.as_pointer()) == expected_pointer:
            live_object = active_object
        else:
            live_object = None
            data = getattr(bpy, "data", None)
            for candidate in tuple(getattr(data, "objects", ())):
                if int(candidate.as_pointer()) == expected_pointer:
                    live_object = candidate
                    break
        if live_object is None or getattr(live_object, "type", "MESH") != "MESH":
            return None
        live_mesh = live_object.data
        if live_mesh is None:
            return None
        state["obj"] = live_object
        state["object_pointer"] = expected_pointer
        state["mesh_pointer"] = int(live_mesh.as_pointer())
        return live_object, live_mesh
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _guided_ridge_history_modal_gate(context, state, event_type):
    """Consume a native Undo/Redo only after Blender's post marker arrives."""
    transaction = state.get("curve_sculpt_transaction") or {}
    if transaction.get("pending_history") is None:
        return "none"
    live_identity = _guided_ridge_history_resolve_live_object(context, state)
    if live_identity is None:
        return "cancel"
    current_signature = _guided_ridge_mesh_revision_signature(state)
    history_status = _curve_sculpt.observe_history_change(
        state,
        current_signature,
        advance_wait=(event_type == "TIMER"),
    )
    if history_status == "pending":
        return "wait"
    if history_status == "external":
        return "cancel"
    if history_status not in {"undo", "redo"}:
        return "none"
    try:
        # Reacquire the datablock identity before the ordinary route check.
        # Undo may restore a different Mesh RNA pointer while preserving the
        # same launch object and View3D.
        if context.active_object is not state.get("obj") or str(context.mode) != "SCULPT":
            return "cancel"
        signature = _guided_ridge_context_signature(context)
        obj = context.active_object
        mesh = obj.data
        if signature is None or getattr(obj, "type", "MESH") != "MESH":
            return "cancel"
        state["object_pointer"] = int(obj.as_pointer())
        state["mesh_pointer"] = int(mesh.as_pointer())
        state["curve_sculpt_target_object_pointer"] = state["object_pointer"]
        state["curve_sculpt_target_mesh_pointer"] = state["mesh_pointer"]
        state["area_key"] = int(context.area.as_pointer())
        state["region_key"] = int(context.region.as_pointer())
        state["window_key"] = int(context.window.as_pointer()) if context.window else 0
        space = getattr(context, "space_data", None)
        state["space_key"] = int(space.as_pointer()) if space is not None else 0
        state["context_signature"] = signature
        snapshot = state.get("snapshot")
        if isinstance(snapshot, dict):
            snapshot["context_signature"] = signature
        state["curve_sculpt_expected_mesh_signature"] = current_signature
        state["curve_sculpt_history_reacquire_pending"] = False
        return "reacquired"
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return "cancel"


def _guided_ridge_mesh_revision_signature(state):
    """Return a cheap, bounded signature for the active native stroke.

    The signature is deliberately sampled rather than a full mesh hash: it is
    only used to bound the short native Paint Curve notification window.  The
    generation token and finite update budget ensure external edits resume
    cancellation even when Blender reports the same object datablock.
    """
    return _curve_sculpt.mesh_revision_signature(state)


def _guided_ridge_history_handler(direction, phase):
    """Record Blender's native Undo/Redo lifecycle for modal reacquisition."""
    state = _runtime.guided_ridge_state
    if state is None or not state.get("active"):
        return
    transaction = state.get("curve_sculpt_transaction") or {}
    pending = transaction.get("pending_history")
    if pending is None or pending.get("direction") != direction:
        return
    transaction[f"history_{phase}_direction"] = direction
    transaction["history_last_phase"] = phase
    if phase == "post":
        # The modal TIMER consumes the expected signature after Blender has
        # rebuilt the datablock.  This marker distinguishes that reacquisition
        # from an unrelated mesh update while keeping the callback non-invasive.
        state["curve_sculpt_history_reacquire_pending"] = True


@persistent
def _on_guided_ridge_undo_pre(_dummy):
    _guided_ridge_history_handler("undo", "pre")


@persistent
def _on_guided_ridge_undo_post(_dummy):
    _guided_ridge_history_handler("undo", "post")


@persistent
def _on_guided_ridge_redo_pre(_dummy):
    _guided_ridge_history_handler("redo", "pre")


@persistent
def _on_guided_ridge_redo_post(_dummy):
    _guided_ridge_history_handler("redo", "post")


@persistent
def _on_guided_ridge_depsgraph_update(_scene, depsgraph):
    state = _runtime.guided_ridge_state
    if state is None or not state.get("active"):
        return
    try:
        pointers = set()
        for update in depsgraph.updates:
            data = update.id
            pointers.add(int(data.as_pointer()))
            original = getattr(data, "original", None)
            if original is not None:
                pointers.add(int(original.as_pointer()))
        target_pointers = {
            int(state["obj"].as_pointer()),
            int(state["obj"].data.as_pointer()),
        }
        if not pointers.intersection(target_pointers):
            return
        # The deferred native transaction owns the target update window.  Its
        # timer validates the live context before applying and before resume;
        # a depsgraph callback here must not race that ownership and cancel the
        # session between the native operator and the fresh modal handler.
        native_transaction = _runtime.guided_ridge_native_transaction
        if native_transaction is not None and native_transaction.get("state") is state:
            return
        # Ctrl+Z/Ctrl+Shift+Z deliberately pass through to Blender.  Blender
        # emits the target depsgraph update before the modal TIMER can compare
        # the expected pre/post signature, so defer all decisions to that
        # observer while a history request is armed.
        history_transaction = state.get("curve_sculpt_transaction") or {}
        if history_transaction.get("pending_history") is not None:
            state["curve_sculpt_history_depsgraph_pending"] = True
            return
        # The native stroke is synchronous.  Notifications are ignored only
        # while this exact apply generation is active; once the operator and
        # explicit view-layer update return, every later target update is
        # external and must cancel the session.
        generation = state.get("curve_sculpt_apply_generation")
        expected_generation = state.get("curve_sculpt_expected_update_generation")
        if (
            bool(state.get("curve_sculpt_apply_active"))
            and generation is not None
            and generation == expected_generation
        ):
            return
        _guided_ridge_request_cancel(state, "stale")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _guided_ridge_request_cancel(state, "stale")


@persistent
def _on_guided_ridge_load_pre(_scene):
    _runtime.guided_ridge_last_guide = None
    _guided_ridge_request_cancel(reason="load-pre")


@persistent
def _on_guided_ridge_load_post(_scene):
    _runtime.guided_ridge_last_guide = None
    _guided_ridge_request_cancel(reason="load-post")


# Foundation is imported after its runtime singleton declarations.  These
# explicit references replace the former shared exec namespace while keeping
# Guided Ridge's legacy session/state contracts intact.
from ..foundation import (
    DEFAULT_DOUBLE_TAP_WINDOW,
    _FaceSetOrbitState,
    _TempOrbitState,
    _activate_face_set_session,
    _addon_preferences,
    _cleanup_orphan_face_set_proxies,
    _deactivate_state,
    _detach_watcher_operator,
    _face_set_activation_artifact_present,
    _fill_preview_shader_get,
    _find_center_hit,
    _finish_face_set_states,
    _finish_state,
    _operator_key,
    _session_is_current,
    _start_normal_tool_session,
    _start_session,
    _tag_redraw,
)
