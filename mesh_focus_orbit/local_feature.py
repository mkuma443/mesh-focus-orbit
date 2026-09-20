"""Local Feature component.

Loaded by the package entry point in dependency order. This module owns the
Local Feature operators and imports shared runtime state explicitly.
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

from . import runtime as _runtime
from . import lifecycle as _lifecycle
from .foundation import _addon_preferences, _operator_key
from .config import LOCAL_FEATURE_BRUSH_CATALOG_ID

# ---------------------------------------------------------------------------
from .config import (
    FACE_SET_ACTIVATION_OPERATOR_ID,
    FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD,
    FILL_PREVIEW_TERMINAL_GROWTH_STAGES,
    FILL_PREVIEW_TERMINAL_MAX_RADIUS_FACTOR,
    FILL_PREVIEW_TERMINAL_THRESHOLD_FACTOR,
    FILL_PREVIEW_WHEEL_DRAIN_SECONDS,
    GUIDED_RIDGE_CURVE_DEFAULT_SMOOTHING,
    GUIDED_RIDGE_CURVE_SCULPT_STEP1,
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
# Local feature brush
# ---------------------------------------------------------------------------

def _local_feature_signature(obj):
    """Return a cheap cache identity without hashing all vertex coordinates."""
    try:
        mesh = obj.data
        matrix = tuple(
            round(float(value), 12)
            for row in obj.matrix_world
            for value in row
        )
        return (
            int(obj.as_pointer()),
            int(mesh.as_pointer()),
            int(len(mesh.vertices)),
            int(len(mesh.edges)),
            int(len(mesh.polygons)),
            matrix,
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _local_feature_safety_reason(obj):
    """Reject mesh states where direct geometry writes would be ambiguous."""
    if obj is None or obj.type != "MESH":
        return "active object is not a mesh"
    try:
        mesh = obj.data
        if mesh is None or mesh.library is not None:
            return "linked mesh is read-only"
        if int(mesh.users) > 1:
            return "shared mesh data is not supported"
        if mesh.shape_keys is not None:
            return "shape keys are not supported"
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return "mesh data is unavailable"
    try:
        for modifier in obj.modifiers:
            if modifier.type == "MULTIRES":
                return "Multires modifier is not supported"
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return "modifier state is unavailable"
    try:
        if bool(getattr(obj, "use_dynamic_topology_sculpting", False)):
            return "Dyntopo is not supported"
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return "Dyntopo state is unavailable"
    try:
        linear = obj.matrix_world.to_3x3()
        lengths = [float(linear.col[index].length) for index in range(3)]
        scale = max(lengths)
        if scale <= 1.0e-12 or max(lengths) - min(lengths) > max(scale * 1.0e-5, 1.0e-7):
            return "non-uniform object scale is not supported"
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return "object transform is unavailable"
    return None


def _local_feature_brush_settings(context):
    """Read brush diameter and pressure without changing the active brush."""
    prefs = _addon_preferences()
    tool_settings = getattr(context, "tool_settings", None)
    sculpt = getattr(tool_settings, "sculpt", None)
    brush = getattr(sculpt, "brush", None)
    sculpt_unified = getattr(sculpt, "unified_paint_settings", None)
    if brush is None:
        return None
    try:
        use_unified_size = bool(getattr(sculpt_unified, "use_unified_size", False))
        diameter = float(sculpt_unified.size if use_unified_size else brush.size)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None
    if not math.isfinite(diameter) or diameter <= 0.0:
        return None
    return {
        "diameter_px": max(1.0, diameter),
        "strength": max(0.0, min(1.0, float(getattr(prefs, "local_feature_strength", 0.35))))
        if prefs is not None else 0.35,
        "feature_scale": max(0.10, min(1.0, float(getattr(prefs, "local_feature_scale", 0.35))))
        if prefs is not None else 0.35,
        "radius_factor": max(0.10, min(4.0, float(getattr(prefs, "local_feature_radius", 1.0))))
        if prefs is not None else 1.0,
        "pressure_enabled": bool(getattr(brush, "use_pressure_strength", False)),
    }


def _local_feature_brush_is_active(context):
    """Return whether the selected Brush is the dedicated local asset.

    A stable saved marker and AssetMetaData are required.  Name and pointer
    comparisons are intentionally omitted because Asset Shelf activation
    creates a new datablock and users may rename it.
    """
    try:
        sculpt = context.tool_settings.sculpt
        brush = getattr(sculpt, "brush", None)
        asset_data = getattr(brush, "asset_data", None)
        if brush is None or asset_data is None:
            return False
        if str(brush.get(LOCAL_FEATURE_BRUSH_MARKER_PROPERTY, "")) != (
            LOCAL_FEATURE_BRUSH_MARKER_VALUE
        ):
            return False
        description = str(getattr(asset_data, "description", ""))
        catalog_id = str(getattr(asset_data, "catalog_id", ""))
        return bool(
            description == LOCAL_FEATURE_BRUSH_MARKER_VALUE
            or catalog_id == LOCAL_FEATURE_BRUSH_CATALOG_ID
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _local_feature_event_in_non_window_region(context, event):
    """Reject screen points belonging to a sibling UI/Asset Shelf region."""
    try:
        area = context.area
        x = int(getattr(event, "mouse_x", -1))
        y = int(getattr(event, "mouse_y", -1))
        if area is None or x < 0 or y < 0:
            return False
        for region in area.regions:
            if region.type == "WINDOW":
                continue
            if (
                int(region.x) <= x < int(region.x + region.width)
                and int(region.y) <= y < int(region.y + region.height)
            ):
                return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        # A missing screen coordinate must not make the operator consume an
        # unrelated UI event.  The keymap poll remains the primary boundary.
        return False
    return False


def _local_feature_event_pressure(event):
    """Read tablet pressure without converting a deliberate zero to one."""
    try:
        value = getattr(event, "pressure", None)
        if value is None:
            return 1.0
        value = float(value)
        return max(0.0, min(1.0, value)) if math.isfinite(value) else 1.0
    except (AttributeError, TypeError, ValueError):
        return 1.0


def _local_feature_selection_signature(context):
    """Identify the standard asset/tool that was active at selection time."""
    try:
        sculpt = context.tool_settings.sculpt
        brush = sculpt.brush
        tool = context.workspace.tools.from_space_view3d_mode(context.mode)
        return (
            int(brush.as_pointer()),
            getattr(brush, "name", None),
            getattr(tool, "idname", None),
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _local_feature_context_identity(context):
    try:
        return (
            int(context.window.as_pointer()),
            int(context.area.as_pointer()),
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return (id(getattr(context, "window", None)), id(getattr(context, "area", None)))


def _local_feature_drop_cache(state):
    """Drop cache references after restore or a mesh lifecycle boundary."""
    if state is None:
        return
    cache = state.get("cache")
    selector = state.get("selector")
    if cache is None and selector is not None:
        cache = selector.get("cache")
    if cache is not None:
        for signature, candidate in list(_runtime.local_feature_brush_cache.items()):
            if candidate is cache:
                _runtime.local_feature_brush_cache.pop(signature, None)
    state["cache"] = None
    state["cache_invalidated"] = True
    if selector is not None:
        selector["cache"] = None


def _local_feature_notify_owned_update(state):
    """Publish a custom write while marking its depsgraph update as owned."""
    obj = state.get("obj")
    if obj is None or obj.type != "MESH":
        return
    state["owned_update_pending"] = True
    state["owned_update_serial"] = int(state.get("owned_update_serial", 0)) + 1
    try:
        obj.data.update(calc_edges=False, calc_edges_loose=False)
        obj.update_tag(refresh={"DATA"})
        view_layer = bpy.context.view_layer
        if view_layer is not None:
            # Force the owner boundary through Blender's normal dependency
            # update path.  The handler consumes only this marked update;
            # external updates take the invalidation path below.
            view_layer.update()
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    finally:
        # Ownership is valid only for this synchronous update boundary.  A
        # later depsgraph update is external and must invalidate the cache.
        state["owned_update_pending"] = False
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _local_feature_mesh_identity(state, obj):
    try:
        return (
            int(obj.data.as_pointer()) == int(state.get("mesh_ptr", -1))
            and _local_feature_signature(obj) == state.get("signature")
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


_LOCAL_FEATURE_NATIVE_BRUSH_FIELDS = (
    "sculpt_brush_type",
    "size",
    "strength",
    "auto_smooth_factor",
    "deform_target",
    "use_pressure_strength",
    "use_pressure_size",
    "use_pressure_masking",
    "use_smooth_stroke",
    "use_space_attenuation",
    "use_inverse_smooth_pressure",
    "use_frontface",
    "use_frontface_falloff",
    "gravity",
    "gravity_factor",
    "use_gravity",
    "use_locked_size",
    "unprojected_size",
)
_LOCAL_FEATURE_NATIVE_SCULPT_FIELDS = ("gravity",)
_LOCAL_FEATURE_NATIVE_AUTOMASK_FIELDS = (
    "use_automasking_topology",
    "use_automasking_face_sets",
    "use_automasking_boundary_edges",
    "use_automasking_boundary_face_sets",
    "use_automasking_cavity",
    "use_automasking_cavity_inverted",
    "use_automasking_custom_cavity_curve",
    "use_automasking_start_normal",
    "use_automasking_view_normal",
    "use_automasking_view_occlusion",
)
_LOCAL_FEATURE_NATIVE_UNIFIED_FIELDS = (
    "use_unified_size",
    "use_unified_strength",
    "size",
    "strength",
    "use_locked_size",
    "unprojected_size",
)
_LOCAL_FEATURE_NATIVE_STROKE_FIELDS = (
    ("name", "STRING", 0),
    ("location", "FLOAT", 3),
    ("mouse", "FLOAT", 2),
    ("mouse_event", "FLOAT", 2),
    ("pressure", "FLOAT", 0),
    ("size", "FLOAT", 0),
    ("x_tilt", "FLOAT", 0),
    ("y_tilt", "FLOAT", 0),
    ("time", "FLOAT", 0),
    ("is_start", "BOOLEAN", 0),
)


def _local_feature_native_rna_property(owner, name, required=True):
    try:
        prop = owner.bl_rna.properties.get(name)
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        prop = None
    if prop is None and required:
        raise RuntimeError(f"native Sculpt RNA property is unavailable: {name}")
    return prop


def _local_feature_native_rna_snapshot(owner, names):
    values = {}
    for name in names:
        prop = _local_feature_native_rna_property(owner, name, required=False)
        if prop is None:
            continue
        try:
            value = getattr(owner, name)
            values[name] = tuple(value) if bool(getattr(prop, "is_array", False)) else value
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(f"native Sculpt setting read failed: {name}: {error}")
    return values


def _local_feature_native_rna_set(owner, name, value, required=True):
    prop = _local_feature_native_rna_property(owner, name, required=required)
    if prop is None:
        return False
    if bool(getattr(prop, "is_readonly", False)):
        raise RuntimeError(f"native Sculpt setting is read-only: {name}")
    try:
        setattr(owner, name, value)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"native Sculpt setting write failed: {name}: {error}")
    return True


def _local_feature_native_rna_restore(owner, values):
    errors = []
    for name, value in values.items():
        try:
            _local_feature_native_rna_set(owner, name, value)
        except (RuntimeError, TypeError, ValueError) as error:
            errors.append(f"{name}: {error}")
    return errors


def _local_feature_native_schema():
    """Validate the public 5.2 brush_stroke and OperatorStrokeElement RNA."""
    operator_rna = bpy.ops.sculpt.brush_stroke.get_rna_type()
    operator_props = {}
    for name, expected_type in (
        ("stroke", "COLLECTION"),
        ("mode", "ENUM"),
        ("brush_toggle", "ENUM"),
        ("pen_flip", "BOOLEAN"),
        ("override_location", "BOOLEAN"),
    ):
        prop = _local_feature_native_rna_property(operator_rna, name)
        if prop.type != expected_type:
            raise RuntimeError(f"native Sculpt RNA type mismatch: {name}={prop.type}")
        operator_props[name] = prop
    for name, expected in (("mode", "NORMAL"), ("brush_toggle", "None")):
        items = getattr(operator_props[name], "enum_items", ())
        if not any(getattr(item, "identifier", None) == expected for item in items):
            raise RuntimeError(f"native Sculpt enum is unavailable: {name}={expected}")
    element_rna = getattr(operator_props["stroke"], "fixed_type", None)
    if element_rna is None:
        raise RuntimeError("native Sculpt OperatorStrokeElement RNA is unavailable")
    element_props = {}
    for name, expected_type, expected_length in _LOCAL_FEATURE_NATIVE_STROKE_FIELDS:
        prop = _local_feature_native_rna_property(element_rna, name)
        if prop.type != expected_type:
            raise RuntimeError(f"OperatorStrokeElement type mismatch: {name}={prop.type}")
        actual_length = int(getattr(prop, "array_length", 0) or 0)
        if actual_length != expected_length:
            raise RuntimeError(
                f"OperatorStrokeElement array mismatch: {name}={actual_length}, expected={expected_length}"
            )
        element_props[name] = prop
    return {"operator": operator_props, "element": element_props}


def _local_feature_native_stroke_points(coord, hit_location, size_value, schema):
    try:
        x = float(coord.x)
        y = float(coord.y)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(f"native Sculpt stroke coordinate is invalid: {error}")
    location = tuple(float(value) for value in hit_location)
    if len(location) != 3 or not all(math.isfinite(value) for value in location):
        raise RuntimeError("native Sculpt stroke hit location is not finite")
    common = {
        "location": location,
        "mouse": (x, y),
        "mouse_event": (x, y),
        "pressure": 1.0,
        "size": float(size_value),
        "x_tilt": 0.0,
        "y_tilt": 0.0,
        "time": 0.0,
    }
    points = []
    for index, is_start in enumerate((True, False)):
        point = dict(common)
        point["name"] = "MFO_LOCAL_FEATURE_PREP_START" if is_start else "MFO_LOCAL_FEATURE_PREP_END"
        point["is_start"] = is_start
        for name, expected_type, expected_length in _LOCAL_FEATURE_NATIVE_STROKE_FIELDS:
            value = point.get(name)
            if value is None:
                raise RuntimeError(f"native Sculpt stroke field is missing: {name}")
            values = value if expected_length else (value,)
            if expected_length and len(values) != expected_length:
                raise RuntimeError(f"native Sculpt stroke field length is invalid: {name}")
            prop = schema["element"][name]
            for component in values:
                if expected_type == "BOOLEAN":
                    if not isinstance(component, bool):
                        raise RuntimeError(f"native Sculpt stroke boolean is invalid: {name}")
                elif expected_type == "STRING":
                    if not isinstance(component, str):
                        raise RuntimeError(f"native Sculpt stroke string is invalid: {name}")
                else:
                    checked = float(component)
                    if not math.isfinite(checked):
                        raise RuntimeError(f"native Sculpt stroke float is not finite: {name}")
                    lower = getattr(prop, "hard_min", None)
                    upper = getattr(prop, "hard_max", None)
                    if lower is not None and checked < float(lower) - 1.0e-6:
                        raise RuntimeError(f"native Sculpt stroke value is below range: {name}")
                    if upper is not None and checked > float(upper) + 1.0e-6:
                        raise RuntimeError(f"native Sculpt stroke value is above range: {name}")
        points.append(point)
    return points


def _local_feature_native_geometry_guard(obj):
    """Use a bounded sample and topology counts, not a full mesh hash."""
    mesh = obj.data
    count = len(mesh.vertices)
    if count <= 0:
        raise RuntimeError("native Sculpt prep requires at least one vertex")
    indices = sorted({0, count - 1, count // 2, count // 3, (2 * count) // 3})
    samples = {
        int(index): tuple(float(value) for value in mesh.vertices[index].co)
        for index in indices
    }
    return {
        "mesh_ptr": int(mesh.as_pointer()),
        "vertex_count": int(count),
        "edge_count": int(len(mesh.edges)),
        "polygon_count": int(len(mesh.polygons)),
        "samples": samples,
    }


def _local_feature_native_coverage(
    obj,
    world_hit,
    width,
    height,
    coord,
    diameter_px,
    radius_factor,
    hard_min,
    hard_max,
):
    """Return a finite world-space sphere plus a viewport contract."""
    try:
        width = max(1.0, float(width))
        height = max(1.0, float(height))
        cx = float(coord.x)
        cy = float(coord.y)
        diameter_px = float(diameter_px)
        radius_factor = max(0.10, float(radius_factor))
        hard_min = float(hard_min)
        hard_max = float(hard_max)
        center = Vector(tuple(float(value) for value in world_hit))
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(f"native Sculpt coverage projection failed: {error}")
    if not all(math.isfinite(value) for value in (cx, cy, diameter_px, radius_factor, *center)):
        raise RuntimeError("native Sculpt coverage contains a non-finite value")
    if diameter_px <= 0.0 or hard_max <= 0.0:
        raise RuntimeError("native Sculpt coverage has an invalid brush range")
    custom_radius_px = max(0.5, 0.5 * diameter_px * radius_factor)
    try:
        world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"native Sculpt object bounds are unavailable: {error}")
    if len(world_corners) != 8:
        raise RuntimeError("native Sculpt object bounds do not contain eight corners")
    world_radius = max(float((corner - center).length) for corner in world_corners)
    if not math.isfinite(world_radius) or world_radius <= 1.0e-12:
        raise RuntimeError("native Sculpt object bounds are degenerate")
    world_margin = max(1.0e-7, world_radius * 1.0e-6)
    native_size = max(int(math.ceil(diameter_px)), int(math.ceil(hard_min)))
    if native_size > hard_max:
        raise RuntimeError(
            "native Sculpt coverage exceeds Brush.size hard limit "
            f"({native_size} > {hard_max:g})"
        )
    return {
        "diameter_px": int(native_size),
        "radius_px": float(native_size) * 0.5,
        "custom_radius_px": float(custom_radius_px),
        "origin_coord": (cx, cy),
        "region_size": (width, height),
        "world_center": tuple(float(value) for value in center),
        "world_radius": float(world_radius + world_margin),
        "world_bounds_corners": int(len(world_corners)),
        "projection_contract": "all mesh points inside the world-space bounds sphere; cursor/view remain in this unchanged VIEW_3D region",
    }


def _local_feature_native_view_signature(context):
    try:
        region_3d = context.space_data.region_3d
        values = []
        for matrix_name in ("view_matrix", "perspective_matrix"):
            matrix = getattr(region_3d, matrix_name)
            values.extend(round(float(value), 12) for row in matrix for value in row)
        values.extend(
            round(float(value), 12)
            for value in tuple(region_3d.view_location)
        )
        values.append(round(float(region_3d.view_distance), 12))
        return tuple(values)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _local_feature_native_coverage_valid(state, context, coord):
    coverage = state.get("native_prep", {}).get("coverage")
    if not coverage:
        return True
    try:
        width, height = coverage["region_size"]
        x = float(coord.x)
        y = float(coord.y)
        if not (0.0 <= x <= float(width) and 0.0 <= y <= float(height)):
            return False
        settings = state["settings"]
        expected_view = state.get("native_prep", {}).get("view_signature")
        if expected_view is not None and _local_feature_native_view_signature(context) != expected_view:
            return False
        return True
    except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _local_feature_native_world_coverage_valid(state, world_hit):
    try:
        coverage = state.get("native_prep", {}).get("coverage") or {}
        center = Vector(coverage["world_center"])
        radius = float(coverage["world_radius"])
        hit = Vector(tuple(float(value) for value in world_hit))
        return math.isfinite(radius) and radius > 0.0 and float((hit - center).length) <= radius + 1.0e-7
    except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _local_feature_native_undo_prep(context, state, coord, world_hit):
    """Create one public native Sculpt Undo step before Python setters."""
    obj = state.get("obj")
    if obj is None or obj.type != "MESH":
        raise RuntimeError("native Sculpt prep has no active mesh")
    if context.area is None or context.area.type != "VIEW_3D" or context.region.type != "WINDOW":
        raise RuntimeError("native Sculpt prep requires a VIEW_3D WINDOW context")
    if context.window is None:
        raise RuntimeError("native Sculpt prep requires a window context")
    tool_settings = getattr(context, "tool_settings", None)
    sculpt = getattr(tool_settings, "sculpt", None)
    brush = getattr(sculpt, "brush", None)
    unified = getattr(sculpt, "unified_paint_settings", None)
    if sculpt is None or brush is None or unified is None:
        raise RuntimeError("native Sculpt brush settings are unavailable")
    schema = _local_feature_native_schema()
    settings = state.get("settings") or {}
    diameter_px = float(settings.get("diameter_px", 0.0))
    radius_factor = float(state.get("radius_factor", settings.get("radius_factor", 1.0)))
    if not math.isfinite(diameter_px) or diameter_px <= 0.0:
        raise RuntimeError("native Sculpt coverage has invalid brush diameter")
    region = context.region
    try:
        width = max(1.0, float(region.width))
        height = max(1.0, float(region.height))
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"native Sculpt coverage projection failed: {error}")
    size_prop = _local_feature_native_rna_property(brush, "size")
    hard_min = float(getattr(size_prop, "hard_min", 1.0))
    hard_max = float(getattr(size_prop, "hard_max", 0.0))
    coverage = _local_feature_native_coverage(
        obj,
        world_hit,
        width,
        height,
        coord,
        diameter_px,
        radius_factor,
        hard_min,
        hard_max,
    )
    native_size = coverage["diameter_px"]
    before = _local_feature_native_geometry_guard(obj)
    brush_saved = _local_feature_native_rna_snapshot(brush, _LOCAL_FEATURE_NATIVE_BRUSH_FIELDS)
    sculpt_saved = _local_feature_native_rna_snapshot(sculpt, _LOCAL_FEATURE_NATIVE_SCULPT_FIELDS)
    unified_saved = _local_feature_native_rna_snapshot(unified, _LOCAL_FEATURE_NATIVE_UNIFIED_FIELDS)
    automask = getattr(brush, "mesh_automasking_settings", None)
    if automask is None:
        automask = getattr(sculpt, "mesh_automasking_settings", None)
    if automask is None:
        raise RuntimeError("native Sculpt automasking settings are unavailable")
    automask_saved = _local_feature_native_rna_snapshot(automask, _LOCAL_FEATURE_NATIVE_AUTOMASK_FIELDS)
    restore_errors = []
    result = None
    try:
        draw_prop = _local_feature_native_rna_property(brush, "sculpt_brush_type")
        if not any(getattr(item, "identifier", None) == "DRAW" for item in getattr(draw_prop, "enum_items", ())):
            raise RuntimeError("active Sculpt brush RNA does not expose DRAW")
        _local_feature_native_rna_set(brush, "sculpt_brush_type", "DRAW")
        _local_feature_native_rna_set(brush, "size", native_size)
        _local_feature_native_rna_set(brush, "use_locked_size", "SCENE")
        world_diameter = float(coverage["world_radius"]) * 2.0
        unprojected_prop = _local_feature_native_rna_property(brush, "unprojected_size")
        unprojected_max = float(getattr(unprojected_prop, "hard_max", 0.0))
        if unprojected_max <= 0.0 or world_diameter > unprojected_max:
            raise RuntimeError("native Sculpt world coverage exceeds unprojected_size range")
        _local_feature_native_rna_set(brush, "unprojected_size", world_diameter)
        _local_feature_native_rna_set(brush, "strength", 0.0)
        _local_feature_native_rna_set(brush, "auto_smooth_factor", 0.0)
        _local_feature_native_rna_set(brush, "deform_target", "GEOMETRY")
        for name in (
            "use_pressure_strength",
            "use_pressure_size",
            "use_pressure_masking",
            "use_smooth_stroke",
            "use_space_attenuation",
            "use_inverse_smooth_pressure",
            "use_frontface",
            "use_frontface_falloff",
            "use_gravity",
        ):
            prop = _local_feature_native_rna_property(brush, name, required=False)
            if prop is None:
                continue
            if getattr(prop, "type", None) == "ENUM":
                disabled = next(
                    (
                        getattr(item, "identifier", None)
                        for item in getattr(prop, "enum_items", ())
                        if getattr(item, "identifier", None) in {"NONE", "OFF", "DISABLED"}
                    ),
                    None,
                )
                if disabled is None:
                    continue
                _local_feature_native_rna_set(brush, name, disabled, required=False)
            else:
                _local_feature_native_rna_set(brush, name, False, required=False)
        for name in ("gravity", "gravity_factor"):
            _local_feature_native_rna_set(brush, name, 0.0, required=False)
        _local_feature_native_rna_set(sculpt, "gravity", 0.0, required=False)
        _local_feature_native_rna_set(unified, "use_unified_size", False)
        _local_feature_native_rna_set(unified, "use_unified_strength", False)
        _local_feature_native_rna_set(unified, "use_locked_size", "SCENE", required=False)
        _local_feature_native_rna_set(unified, "unprojected_size", world_diameter, required=False)
        for name in _LOCAL_FEATURE_NATIVE_AUTOMASK_FIELDS:
            _local_feature_native_rna_set(automask, name, False, required=False)
        stroke = _local_feature_native_stroke_points(coord, world_hit, native_size, schema)
        with bpy.context.temp_override(
            window=context.window,
            area=context.area,
            region=context.region,
            space_data=context.space_data,
        ):
            # The strength-zero native preparation itself publishes a
            # synchronous mesh/depsgraph update in Blender 5.2.  Keep the
            # existing cache owned only for this nested call so the handler
            # cannot mistake that preparation for an external edit.  The
            # handler consumes the marker at its update boundary; the
            # finally clause clears it when no callback was delivered.
            state["owned_update_pending"] = True
            try:
                result = bpy.ops.sculpt.brush_stroke(
                    "EXEC_DEFAULT",
                    stroke=stroke,
                    mode="NORMAL",
                    brush_toggle="None",
                    pen_flip=False,
                    override_location=False,
                )
            finally:
                if state.get("owned_update_pending"):
                    state["owned_update_pending"] = False
        if list(result) != ["FINISHED"]:
            raise RuntimeError(f"native Sculpt prep did not finish: {list(result)}")
    finally:
        restore_errors.extend(_local_feature_native_rna_restore(brush, brush_saved))
        restore_errors.extend(_local_feature_native_rna_restore(sculpt, sculpt_saved))
        restore_errors.extend(_local_feature_native_rna_restore(unified, unified_saved))
        restore_errors.extend(_local_feature_native_rna_restore(automask, automask_saved))
    if restore_errors:
        raise RuntimeError("native Sculpt prep settings restore failed: " + "; ".join(restore_errors))
    after = _local_feature_native_geometry_guard(obj)
    if before != after:
        raise RuntimeError("native Sculpt prep changed bounded geometry/topology guard")
    return {
        "nested_operator": "bpy.ops.sculpt.brush_stroke",
        "nested_context": "EXEC_DEFAULT",
        "nested_call_count": 1,
        "result": list(result),
        "schema_verified": True,
        "settings_restored": True,
        "geometry_guard": True,
        "coverage": coverage,
        "view_signature": _local_feature_native_view_signature(context),
        "mesh_counts": before,
    }


def _local_feature_raycast(context, coord):
    """Return an active-mesh visible hit and a world-space surface normal."""
    obj = getattr(context, "active_object", None)
    if obj is None or obj.type != "MESH":
        return None
    try:
        region = context.region
        rv3d = context.space_data.region_3d
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        if direction.length_squared <= 1.0e-20:
            return None
        direction.normalize()
        inverse = obj.matrix_world.inverted_safe()
        local_origin = inverse @ origin
        local_direction = inverse.to_3x3() @ direction
        if local_direction.length_squared <= 1.0e-20:
            return None
        local_direction.normalize()
        for _attempt in range(256):
            hit, location, local_normal, face_index = obj.ray_cast(
                local_origin, local_direction
            )
            if not hit or face_index < 0 or face_index >= len(obj.data.polygons):
                return None
            polygon = obj.data.polygons[int(face_index)]
            if not bool(polygon.hide):
                hidden_vertex = False
                try:
                    hidden_vertex = any(
                        bool(obj.data.vertices[int(index)].hide)
                        for index in polygon.vertices
                    )
                except (AttributeError, ReferenceError, RuntimeError, TypeError):
                    hidden_vertex = False
                if not hidden_vertex:
                    world_location = obj.matrix_world @ location
                    world_normal = obj.matrix_world.to_3x3().inverted().transposed() @ local_normal
                    if world_normal.length_squared <= 1.0e-20:
                        return None
                    world_normal.normalize()
                    return obj, int(face_index), world_location, world_normal
            advance = min(max((location - local_origin).length * 1.0e-6, 1.0e-7), 1.0e-3)
            local_origin = location + local_direction * advance
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None
    return None


def _local_feature_world_radius(context, coord, hit, diameter_px, radius_factor):
    """Convert the native brush diameter (5.2: pixels) to a world radius."""
    try:
        region = context.region
        rv3d = context.space_data.region_3d
        half_px = max(0.5, float(diameter_px) * 0.5 * float(radius_factor))
        probe = Vector((float(coord.x) + half_px, float(coord.y)))
        world_probe = view3d_utils.region_2d_to_location_3d(region, rv3d, probe, hit)
        radius = float((world_probe - hit).length)
        if radius > 1.0e-8 and math.isfinite(radius):
            return radius
        distance = float(getattr(rv3d, "view_distance", 1.0))
        width = max(1.0, float(region.width))
        return max(distance * half_px / width, 1.0e-5)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _local_feature_read_mask_hidden(mesh):
    """Read Sculpt mask/hidden state once; missing mask means fully editable."""
    import numpy as np

    count = len(mesh.vertices)
    hidden = np.zeros(count, dtype=bool)
    mask = np.zeros(count, dtype=np.float32)
    try:
        mesh.vertices.foreach_get("hide", hidden)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        for index, vertex in enumerate(mesh.vertices):
            hidden[index] = bool(getattr(vertex, "hide", False))
    attribute = mesh.attributes.get(".sculpt_mask")
    if attribute is not None and attribute.domain == "POINT":
        try:
            attribute.data.foreach_get("value", mask)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            for index, item in enumerate(attribute.data):
                mask[index] = float(getattr(item, "value", 0.0))
    # A hidden polygon can still expose vertices that are not individually
    # hidden.  Mark those vertices protected too, so a nearby visible face's
    # halo cannot move geometry belonging to a hidden face.
    try:
        for polygon in mesh.polygons:
            if bool(polygon.hide):
                for index in polygon.vertices:
                    hidden[int(index)] = True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    return hidden, np.clip(mask, 0.0, 1.0)


def _local_feature_build_cache_steps(obj):
    """Yield through one reusable world-space triangle/BVH preparation."""
    import numpy as np

    signature = _local_feature_signature(obj)
    if signature is None:
        raise RuntimeError("mesh signature unavailable")
    cached = _runtime.local_feature_brush_cache.get(signature)
    if cached is not None:
        return cached
    mesh = obj.data
    if len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
        raise RuntimeError("mesh has no surface")
    mesh.calc_loop_triangles()
    coordinates_local = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    yield "allocate"
    mesh.vertices.foreach_get("co", coordinates_local.ravel())
    yield "vertices"
    transform = np.asarray(obj.matrix_world, dtype=np.float64)
    coordinates_world = coordinates_local @ transform[:3, :3].T + transform[:3, 3]
    triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", triangles.ravel())
    yield "triangles"
    if len(triangles) == 0:
        raise RuntimeError("mesh has no triangles")
    bvh_points = []
    for start in range(0, len(coordinates_world), 4096):
        bvh_points.extend(
            tuple(float(value) for value in point)
            for point in coordinates_world[start : start + 4096]
        )
        yield f"bvh-points-{min(start + 4096, len(coordinates_world))}"
    bvh_faces = []
    for start in range(0, len(triangles), 4096):
        bvh_faces.extend(
            tuple(int(value) for value in face)
            for face in triangles[start : start + 4096]
        )
        yield f"bvh-faces-{min(start + 4096, len(triangles))}"
    try:
        bvh = BVHTree.FromPolygons(
            bvh_points,
            bvh_faces,
            all_triangles=True,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, MemoryError) as error:
        raise RuntimeError(f"surface index build failed: {error}")
    yield "bvh"
    hidden, mask = _local_feature_read_mask_hidden(mesh)
    yield "mask-hidden"
    cache = {
        "signature": signature,
        "obj_ptr": int(obj.as_pointer()),
        "mesh_ptr": int(mesh.as_pointer()),
        "points_world": coordinates_world,
        "source_points_world": coordinates_world.copy(),
        "points_local": coordinates_local,
        "triangles": triangles,
        "bvh": bvh,
        "hidden": hidden,
        "mask": mask,
        "max_displacement_world": 0.0,
        "created_at": time.monotonic(),
    }
    _runtime.local_feature_brush_cache[signature] = cache
    return cache


def _local_feature_build_cache(obj):
    """Synchronous wrapper used by isolated callers; modal uses the steps."""
    job = _local_feature_build_cache_steps(obj)
    while True:
        try:
            next(job)
        except StopIteration as complete:
            return complete.value


def _local_feature_dab_arrays(
    points_world,
    triangles,
    center,
    normal,
    radius,
    strength=0.35,
    feature_scale=0.35,
    target_indices=None,
    hidden=None,
    mask=None,
):
    """Return local ridge/valley deltas using only a candidate+halo patch.

    The operation is intentionally independent of world axes and stroke
    tangent.  A long low-pass before the residual gate is what rejects the
    known fine-ripple failure of the earlier four-pass prototype.
    """
    import numpy as np

    points = np.asarray(points_world, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    if len(points) == 0 or len(faces) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1.0e-12:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    normal /= normal_length
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 1.0e-10:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    scale = max(0.10, min(1.0, float(feature_scale)))
    strength = max(0.0, min(1.0, float(strength)))
    distances = np.linalg.norm(points - center[None, :], axis=1)
    local_ids = np.flatnonzero(distances < radius * 2.25).astype(np.int32)
    if target_indices is None:
        target = distances < radius
    else:
        target = np.zeros(len(points), dtype=bool)
        target[np.asarray(target_indices, dtype=np.int32)] = True
        target &= distances < radius
    if hidden is not None:
        target &= ~np.asarray(hidden, dtype=bool)
    if mask is not None:
        target &= np.asarray(mask, dtype=np.float64) < (1.0 - 1.0e-6)
    if not np.any(target) or len(local_ids) < 4:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    local_lookup = np.full(len(points), -1, dtype=np.int32)
    local_lookup[local_ids] = np.arange(len(local_ids), dtype=np.int32)
    inside = np.all(np.isin(faces, local_ids), axis=1)
    local_faces = local_lookup[faces[inside]]
    if len(local_faces) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    edges = np.concatenate(
        (local_faces[:, (0, 1)], local_faces[:, (1, 2)], local_faces[:, (2, 0)]),
        axis=0,
    )
    edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    degree = np.bincount(edges[:, 0], minlength=len(local_ids)).astype(np.float64)
    if not np.any(degree > 0.0):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    heights = (points[local_ids] - center[None, :]) @ normal
    iterations = max(8, min(24, int(round(8.0 + 16.0 * scale))))
    smoothed = heights.copy()
    # Jacobi low-pass uses the same local graph for every pass.  Halo vertices
    # keep boundary values from leaking into the target patch.
    for _index in range(iterations):
        sums = np.zeros(len(local_ids), dtype=np.float64)
        np.add.at(sums, edges[:, 0], smoothed[edges[:, 1]])
        neighbour = sums / np.maximum(degree, 1.0)
        smoothed = 0.30 * smoothed + 0.70 * neighbour
    sums = np.zeros(len(local_ids), dtype=np.float64)
    np.add.at(sums, edges[:, 0], smoothed[edges[:, 1]])
    neighbour = sums / np.maximum(degree, 1.0)
    residual = smoothed - neighbour
    edge_lengths = np.linalg.norm(
        points[local_ids[edges[:, 0]]] - points[local_ids[edges[:, 1]]], axis=1
    )
    edge_scale = float(np.median(edge_lengths[edge_lengths > 1.0e-12])) if np.any(edge_lengths > 1.0e-12) else 0.0
    if edge_scale <= 1.0e-12:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    curvature = residual / (edge_scale * edge_scale)
    # A curvature gate is deliberately applied after the 16-pass low-pass.
    # This leaves a flat plane and a pure fine ripple below the gate while
    # retaining both signs of a broad existing ridge/valley.
    gate_start = 0.20 / (0.55 + scale)
    gate_end = 0.80 / (0.55 + scale)
    gate = np.clip((np.abs(curvature) - gate_start) / max(gate_end - gate_start, 1.0e-9), 0.0, 1.0)
    target_local = local_lookup[np.flatnonzero(target)]
    target_local = target_local[target_local >= 0]
    if len(target_local) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    t = np.clip(distances[local_ids[target_local]] / radius, 0.0, 1.0)
    falloff = (1.0 - t * t) ** 2
    if mask is None:
        editable = np.ones(len(target_local), dtype=np.float64)
    else:
        editable = 1.0 - np.clip(np.asarray(mask, dtype=np.float64)[local_ids[target_local]], 0.0, 1.0)
    gain = 0.55 * strength
    deltas = residual[target_local] * gain * gate[target_local] * falloff * editable
    max_delta = max(edge_scale * 0.35, radius * 0.025) * max(strength, 0.05)
    deltas = np.clip(deltas, -max_delta, max_delta)
    keep = np.abs(deltas) > 1.0e-12
    return local_ids[target_local[keep]].astype(np.int32), deltas[keep].astype(np.float64)


def _local_feature_pick_candidate(cache, center, radius):
    """Use the one-time BVH to obtain a local triangle candidate set."""
    import numpy as np

    try:
        displacement_bound = max(0.0, float(cache.get("max_displacement_world", 0.0)))
        query_radius = float(radius * 2.25) + displacement_bound
        hits = cache["bvh"].find_nearest_range(
            tuple(float(value) for value in center), query_radius
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        hits = []
    triangle_ids = []
    for item in hits:
        try:
            triangle_ids.append(int(item[2]))
        except (IndexError, TypeError, ValueError):
            continue
    if not triangle_ids:
        return np.empty(0, dtype=np.int32), np.empty((0, 3), dtype=np.int32)
    triangles = cache["triangles"][np.unique(np.asarray(triangle_ids, dtype=np.int32))]
    return np.unique(triangles.reshape(-1)).astype(np.int32), triangles.astype(np.int32, copy=False)


def _local_feature_apply_dab(state, context, coord, pressure=1.0):
    """Ray-hit one dab, compute local deltas, and write only changed vertices."""
    import numpy as np

    obj = state.get("obj")
    cache = state.get("cache")
    if obj is None or cache is None:
        return 0
    if state.get("native_prepared") and not _local_feature_native_coverage_valid(
        state, context, coord
    ):
        raise RuntimeError(
            "local feature stroke left the one-time native Sculpt coverage contract"
        )
    hit = _local_feature_raycast(context, coord)
    if hit is None or hit[0] is not obj:
        return 0
    _hit_obj, _face_index, world_hit, world_normal = hit
    if state.get("native_prepared") and not _local_feature_native_world_coverage_valid(
        state, world_hit
    ):
        raise RuntimeError("local feature hit left the one-time native Sculpt world coverage")
    settings = state["settings"]
    radius = _local_feature_world_radius(
        context,
        coord,
        world_hit,
        settings["diameter_px"],
        state.get("radius_factor", settings["radius_factor"]),
    )
    if radius is None:
        return 0
    candidate, candidate_triangles = _local_feature_pick_candidate(cache, world_hit, radius)
    if len(candidate) == 0:
        return 0
    # Compress to the BVH-returned candidate+halo before any NumPy operation.
    # Passing the full mesh here would turn every dab into an O(total-N) scan.
    candidate = np.asarray(candidate, dtype=np.int32)
    # ``candidate`` is sorted by np.unique, so searchsorted maps global
    # triangle indices to the compact patch without allocating an N-sized map.
    local_triangles = np.searchsorted(candidate, candidate_triangles).astype(np.int32)
    if len(local_triangles) == 0:
        return 0
    local_ids, deltas = _local_feature_dab_arrays(
        cache["points_world"][candidate],
        local_triangles,
        world_hit,
        world_normal,
        radius,
        strength=settings["strength"]
        * float(state.get("strength_factor", 1.0))
        * (float(pressure) if settings["pressure_enabled"] else 1.0),
        feature_scale=settings["feature_scale"],
        target_indices=np.arange(len(candidate), dtype=np.int32),
        hidden=cache["hidden"][candidate],
        mask=cache["mask"][candidate],
    )
    if len(local_ids) == 0:
        return 0
    # The first usable dab is the only point at which the native Sculpt
    # position node may be prepared.  Compute deltas first so a flat/no-op
    # dab does not create an empty undo step, then prepare before any setter.
    if not state.get("native_prepared"):
        state["native_prep"] = _local_feature_native_undo_prep(
            context, state, coord, world_hit
        )
        state["native_prepared"] = True
    transform = np.asarray(obj.matrix_world, dtype=np.float64)
    inverse_linear = np.linalg.inv(transform[:3, :3])
    mesh = obj.data
    changed = 0
    for vertex_index, delta in zip(local_ids, deltas):
        index = int(candidate[int(vertex_index)])
        if bool(cache["hidden"][index]) or float(cache["mask"][index]) >= 1.0 - 1.0e-6:
            continue
        if index not in state["saved_coords"]:
            state["saved_coords"][index] = tuple(float(value) for value in mesh.vertices[index].co)
        world_delta = np.asarray(world_normal, dtype=np.float64) * float(delta)
        local_delta = inverse_linear @ world_delta
        vertex = mesh.vertices[index]
        current = np.asarray(vertex.co, dtype=np.float64)
        new_value = current + local_delta
        vertex.co = tuple(float(value) for value in new_value)
        cache["points_local"][index] = new_value
        current_world = new_value @ transform[:3, :3].T + transform[:3, 3]
        cache["points_world"][index] = current_world
        source_world = cache.get("source_points_world")
        if source_world is not None:
            displacement = float(np.linalg.norm(current_world - source_world[index]))
            cache["max_displacement_world"] = max(
                float(cache.get("max_displacement_world", 0.0)), displacement
            )
        changed += 1
    if changed:
        state["changed"] = True
        state["last_hit"] = tuple(float(value) for value in world_hit)
        _local_feature_notify_owned_update(state)
    return changed


def _local_feature_finalize_mesh(state):
    try:
        obj = state.get("obj")
        if obj is not None and obj.type == "MESH":
            obj.data.update(calc_edges=False, calc_edges_loose=False)
            obj.update_tag(refresh={"DATA"})
            for area in bpy.context.screen.areas if bpy.context.screen else ():
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _local_feature_restore_state(state):
    try:
        obj = state.get("obj")
        if obj is None or obj.type != "MESH":
            _local_feature_drop_cache(state)
            return False
        if not _local_feature_mesh_identity(state, obj):
            # Never write saved indices into a replacement mesh or changed
            # topology.  The runtime/cache is disposable at this boundary.
            _local_feature_drop_cache(state)
            state["saved_coords"] = {}
            return False
        mesh = obj.data
        for index, coordinate in state.get("saved_coords", {}).items():
            if 0 <= int(index) < len(mesh.vertices):
                mesh.vertices[int(index)].co = coordinate
        if state.get("saved_coords"):
            _local_feature_finalize_mesh(state)
        _local_feature_drop_cache(state)
        return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _local_feature_drop_cache(state)
        return False


def _local_feature_dispose_state(state):
    if state is None:
        return
    timer = state.get("prepare_timer")
    if timer is not None:
        window_manager = state.get("window_manager")
        try:
            if window_manager is not None:
                window_manager.event_timer_remove(timer)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        state["prepare_timer"] = None
    state["prepare_job"] = None
    key = state.get("operator_key")
    if key is not None:
        _runtime.local_feature_brush_states.pop(key, None)


def _local_feature_cancel_all(reason="lifecycle", restore=True):
    for state in list(_runtime.local_feature_brush_states.values()):
        if restore and state.get("changed"):
            _local_feature_restore_state(state)
        selector = state.get("selector")
        if selector is not None:
            selector["cache"] = None
        _local_feature_dispose_state(state)
    _runtime.local_feature_brush_cache.clear()
    _runtime.local_feature_brush_pending_stroke = None
    _runtime.local_feature_brush_stroke_operator = None


def _local_feature_request_cancel_all(reason="external-change", restore=True):
    entries = []
    for state in tuple(_runtime.local_feature_brush_states.values()):
        entry = _lifecycle.modal_request_cancel(
            operator=state.get("operator"),
            reason=reason,
        )
        entries.append((entry, state))
    _local_feature_cancel_all(reason, restore=restore)
    for entry, state in entries:
        if entry is not None:
            _lifecycle.modal_schedule_terminal_event(
                token=entry.get("token"),
                state=state,
            )


def _on_local_feature_brush_depsgraph_update(_scene, depsgraph):
    """Invalidate reusable geometry after an external mesh update."""
    updated = set()
    try:
        for update in depsgraph.updates:
            data = update.id
            updated.add(int(data.as_pointer()))
            original = getattr(data, "original", None)
            if original is not None:
                updated.add(int(original.as_pointer()))
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _runtime.local_feature_brush_cache.clear()
        for state in _runtime.local_feature_brush_states.values():
            state["cache"] = None
            state["cache_invalidated"] = True
            selector = state.get("selector")
            if selector is not None:
                selector["cache"] = None
                selector["cache_invalidated"] = True
        return
    for signature, cache in list(_runtime.local_feature_brush_cache.items()):
        if cache.get("obj_ptr") not in updated and cache.get("mesh_ptr") not in updated:
            continue
        owned = False
        for state in _runtime.local_feature_brush_states.values():
            if state.get("cache") is not cache:
                continue
            state_obj = state.get("obj")
            try:
                state_obj_ptr = int(state_obj.as_pointer()) if state_obj is not None else -1
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                state_obj_ptr = -1
            cache_obj_ptr = int(cache.get("obj_ptr", -1))
            if (
                state.get("owned_update_pending")
                and not state.get("external_update_expected")
                and (
                    state.get("mesh_ptr") in updated
                    or cache_obj_ptr in updated
                    or state_obj_ptr in updated
                )
            ):
                # Mesh.update() from our own write is consumed at this exact
                # depsgraph boundary.  A native Shift Smooth marks the
                # external flag before PASS_THROUGH and therefore cannot be
                # mistaken for this update.
                state["owned_update_pending"] = False
                owned = True
            else:
                state["owned_update_pending"] = False
                state["external_update_expected"] = False
        if owned:
            continue
        _runtime.local_feature_brush_cache.pop(signature, None)
        for state in _runtime.local_feature_brush_states.values():
            if state.get("cache") is cache:
                state["cache"] = None
                state["cache_invalidated"] = True
            selector = state.get("selector")
            if selector is not None and selector.get("cache") is cache:
                selector["cache"] = None
                selector["cache_invalidated"] = True


def _on_local_feature_brush_undo_post(_scene):
    # Native Undo has already restored its own coordinates.  Never write an
    # old modal snapshot over that authoritative history result.
    _local_feature_request_cancel_all("undo", restore=False)


def _on_local_feature_brush_redo_post(_scene):
    _local_feature_request_cancel_all("redo", restore=False)


def _on_local_feature_brush_load_pre(_scene):
    _local_feature_request_cancel_all("load-pre", restore=False)


def _on_local_feature_brush_load_post(_scene):
    _local_feature_request_cancel_all("load-post", restore=False)


class VIEW3D_OT_mesh_focus_local_feature_brush(bpy.types.Operator):
    """Dispatch one stroke when the dedicated Brush Asset is active.

    This operator is bound only to an unmodified LMB press in the Sculpt
    keymap.  It is not a selector and never installs a persistent modal
    handler; Blender's Asset Shelf remains the selection UI.
    """

    bl_idname = LOCAL_FEATURE_BRUSH_OPERATOR_ID
    bl_label = "Mesh Focus: Local Feature Brush"
    bl_description = "With the MFO Brush Asset selected, drag LMB to enhance existing ridges and valleys"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
            and context.space_data is not None
            and context.mode == "SCULPT"
            and context.active_object is not None
            and context.active_object.type == "MESH"
            and _local_feature_brush_is_active(context)
        )

    def invoke(self, context, event):
        if not self.poll(context):
            return {"PASS_THROUGH"}
        # The keymap already excludes modifiers, but retain this guard for
        # direct invocation and for Blender keymap precedence differences.
        if (
            bool(getattr(event, "shift", False))
            or bool(getattr(event, "ctrl", False))
            or bool(getattr(event, "alt", False))
            or _local_feature_event_in_non_window_region(context, event)
        ):
            return {"PASS_THROUGH"}
        obj = context.active_object
        reason = _local_feature_safety_reason(obj)
        if reason:
            self.report({"WARNING"}, f"Local Feature Brush: {reason}")
            return {"CANCELLED"}
        settings = _local_feature_brush_settings(context)
        if settings is None:
            self.report({"WARNING"}, "Local Feature Brush: Sculpt brush settings unavailable")
            return {"CANCELLED"}
        # This transient selector record belongs to the child stroke only;
        # it is intentionally not placed in _runtime.local_feature_brush_states and
        # therefore cannot consume idle UI/navigation events between strokes.
        selector_state = {
            "operator": self,
            "obj": obj,
            "mesh_ptr": int(obj.data.as_pointer()),
            "signature": _local_feature_signature(obj),
            "settings": settings,
            "selection_signature": _local_feature_selection_signature(context),
            "radius_factor": 1.0,
            "strength_factor": 1.0,
            "selected": True,
            "cache": None,
        }
        _runtime.local_feature_brush_pending_stroke = {
            "selector": selector_state,
            "coord": Vector((event.mouse_region_x, event.mouse_region_y)),
            "pressure": _local_feature_event_pressure(event),
        }
        try:
            result = bpy.ops.view3d.mesh_focus_local_feature_brush_stroke("INVOKE_DEFAULT")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _runtime.local_feature_brush_pending_stroke = None
            self.report({"WARNING"}, "Local Feature Brush: stroke could not start")
            # The dedicated asset owns this LMB press.  A failed start must
            # not fall through to the asset's native Draw behaviour.
            return {"CANCELLED"}
        # The child modal owns only this active stroke.  Returning FINISHED
        # here is what leaves the viewport and all sibling UI regions free
        # between strokes while the selected Asset remains active.
        if isinstance(result, set) and "CANCELLED" in result:
            return {"CANCELLED"}
        return {"FINISHED"}


class VIEW3D_OT_mesh_focus_local_feature_brush_stroke(bpy.types.Operator):
    """Own one LMB stroke around one native Sculpt Undo preparation."""

    bl_idname = "view3d.mesh_focus_local_feature_brush_stroke"
    bl_label = "Mesh Focus: Local Feature Brush Stroke"
    # The outer stroke operator must retain its UNDO depth until FINISHED so
    # the nested public Sculpt position step stays pending until LMB release.
    # The selector remains non-UNDO; only native-prepared strokes are allowed.
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return VIEW3D_OT_mesh_focus_local_feature_brush.poll(context)

    def invoke(self, context, _event):
        pending = _runtime.local_feature_brush_pending_stroke
        _runtime.local_feature_brush_pending_stroke = None
        if pending is None or pending.get("selector") is None:
            return {"CANCELLED"}
        selector_state = pending["selector"]
        if selector_state.get("operator") is None or not self.poll(context):
            return {"CANCELLED"}
        obj = selector_state.get("obj")
        reason = _local_feature_safety_reason(obj)
        if reason:
            self.report({"WARNING"}, f"Local Feature Brush: {reason}")
            return {"CANCELLED"}
        # Preferences are editable from the Sculpt N-panel while the selector
        # remains active.  Re-read them at every stroke boundary so the next
        # stroke uses the values currently shown in the panel; the selector's
        # wheel multipliers remain intentionally persistent between strokes.
        settings = _local_feature_brush_settings(context)
        if settings is None:
            self.report({"WARNING"}, "Local Feature Brush: Sculpt brush settings unavailable")
            return {"CANCELLED"}
        selector_state["settings"] = settings
        key = _operator_key(self)
        state = {
            "operator": self,
            "operator_key": key,
            "selector": selector_state,
            "obj": obj,
            "window_manager": context.window_manager,
            "mesh_ptr": int(obj.data.as_pointer()),
            "signature": _local_feature_signature(obj),
            "settings": settings,
            "radius_factor": max(
                0.10,
                min(
                    4.0,
                    float(settings["radius_factor"])
                    * float(selector_state.get("radius_factor", 1.0)),
                ),
            ),
            "strength_factor": selector_state.get("strength_factor", 1.0),
            "stroke_active": True,
            "changed": False,
            "saved_coords": {},
            "native_prepared": False,
            "native_prep": None,
            "cache_invalidated": False,
            "owned_update_pending": False,
            "owned_update_serial": 0,
            "external_update_expected": False,
            "cache": selector_state.get("cache"),
            "last_mouse": pending["coord"],
            "ignore_first_press": True,
            "pending_pressure": pending["pressure"],
            "prepare_job": None,
            "prepare_timer": None,
            "prepare_timer_duration": 0.0,
            "prepare_stage": "queued",
        }
        try:
            if state["cache"] is None:
                state["prepare_job"] = _local_feature_build_cache_steps(obj)
                state["prepare_timer"] = context.window_manager.event_timer_add(
                    0.01, window=context.window
                )
            _runtime.local_feature_brush_states[key] = state
            _runtime.local_feature_brush_stroke_operator = self
            context.window_manager.modal_handler_add(self)
            _lifecycle.modal_register(self, "Local Feature", key, state=state)
            if state["cache"] is not None:
                _local_feature_apply_dab(state, context, pending["coord"], pressure=pending["pressure"])
        except (MemoryError, RuntimeError, TypeError, ValueError) as error:
            self.report({"WARNING"}, f"Local Feature Brush: preparation failed ({error})")
            _runtime.local_feature_brush_stroke_operator = None
            _local_feature_dispose_state(state)
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _state(self):
        return _runtime.local_feature_brush_states.get(_operator_key(self))

    def _finish(self, state, cancelled=False):
        timer = state.get("prepare_timer")
        if timer is not None:
            try:
                state.get("window_manager", bpy.context.window_manager).event_timer_remove(timer)
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                pass
            state["prepare_timer"] = None
        if cancelled:
            _local_feature_restore_state(state)
        elif state.get("changed"):
            _local_feature_finalize_mesh(state)
        selector = state.get("selector")
        if cancelled and state.get("drop_selector_on_cancel") and selector is not None:
            _local_feature_dispose_state(selector)
        _local_feature_dispose_state(state)
        _runtime.local_feature_brush_stroke_operator = None
        terminal_status = "CANCELLED" if cancelled else "FINISHED"
        _lifecycle.modal_terminal(operator=self, status=terminal_status)
        return {terminal_status}

    def modal(self, context, event):
        state = self._state()
        entry = _lifecycle.modal_entry(operator=self)
        if entry is not None and (
            entry.get("cancel_requested") or entry.get("teardown_requested")
        ):
            if state is not None:
                state["drop_selector_on_cancel"] = True
                result = self._finish(state, cancelled=True)
            else:
                _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                result = {"CANCELLED"}
            return result
        if state is None or state.get("operator") is not self:
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        if not self.poll(context) or context.active_object is not state.get("obj"):
            state["drop_selector_on_cancel"] = True
            return self._finish(state, cancelled=True)
        if _local_feature_signature(state["obj"]) != state.get("signature"):
            state["drop_selector_on_cancel"] = True
            return self._finish(state, cancelled=True)
        if state.get("cache_invalidated") and state.get("prepare_job") is None:
            return self._finish(state, cancelled=True)
        event_type = getattr(event, "type", "")
        event_value = getattr(event, "value", None)
        if event_type in {"LEFT_SHIFT", "RIGHT_SHIFT"}:
            return {"PASS_THROUGH"}
        if event_type == "ESC" and event_value in {None, "PRESS"}:
            return self._finish(state, cancelled=True)
        if event_type == "LEFTMOUSE" and event_value == "RELEASE" and state.get("prepare_job") is not None:
            # The user released before the staged source cache was ready; do
            # not apply a deferred dab after the gesture has ended.
            return self._finish(state, cancelled=True)
        if event_type == "TIMER" and state.get("prepare_job") is not None:
            if state.get("cache_invalidated"):
                return self._finish(state, cancelled=True)
            expected_timer = state.get("prepare_timer")
            event_timer = getattr(event, "timer", None)
            if event_timer is not None:
                # Future Blender versions may expose the owner directly.
                if event_timer is not expected_timer:
                    return {"PASS_THROUGH"}
            else:
                # Blender 5.2's public Event exposes TIMER type/value but no
                # timer pointer.  Use the saved WM Timer's monotonic public
                # duration as the progress/ownership boundary.
                if expected_timer is None:
                    return {"PASS_THROUGH"}
                try:
                    duration = float(getattr(expected_timer, "time_duration"))
                    previous = float(state.get("prepare_timer_duration", 0.0))
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    return {"PASS_THROUGH"}
                if not math.isfinite(duration) or duration <= previous:
                    return {"PASS_THROUGH"}
                state["prepare_timer_duration"] = duration
            try:
                state["prepare_stage"] = next(state["prepare_job"])
                return {"RUNNING_MODAL"}
            except StopIteration as complete:
                state["prepare_job"] = None
                state["prepare_timer"] = None
                state["cache"] = complete.value
                state["selector"]["cache"] = state["cache"]
                try:
                    if expected_timer is not None:
                        bpy.context.window_manager.event_timer_remove(expected_timer)
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    pass
                try:
                    _local_feature_apply_dab(
                        state,
                        context,
                        state["last_mouse"],
                        pressure=state.get("pending_pressure", 1.0),
                    )
                except (MemoryError, RuntimeError, TypeError, ValueError) as error:
                    self.report({"WARNING"}, f"Local Feature Brush: dab failed ({error})")
                    return self._finish(state, cancelled=True)
                return {"RUNNING_MODAL"}
            except (MemoryError, RuntimeError, TypeError, ValueError) as error:
                self.report({"WARNING"}, f"Local Feature Brush: preparation failed ({error})")
                return self._finish(state, cancelled=True)
        if event_type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event_value in {None, "PRESS"}:
            factor = 1.15 if event_type == "WHEELUPMOUSE" else 1.0 / 1.15
            if bool(getattr(event, "shift", False)):
                state["strength_factor"] = max(0.10, min(2.0, state["strength_factor"] * factor))
            else:
                state["radius_factor"] = max(0.10, min(4.0, state["radius_factor"] * factor))
            return {"RUNNING_MODAL"}
        if event_type == "LEFTMOUSE" and event_value == "PRESS":
            if state.get("ignore_first_press"):
                state["ignore_first_press"] = False
                if bool(getattr(event, "shift", False)):
                    state["external_update_expected"] = True
                return {"PASS_THROUGH"}
            if bool(getattr(event, "shift", False)):
                state["external_update_expected"] = True
                return {"PASS_THROUGH"}
            state["last_mouse"] = Vector((event.mouse_region_x, event.mouse_region_y))
            pressure = _local_feature_event_pressure(event)
            try:
                _local_feature_apply_dab(state, context, state["last_mouse"], pressure=pressure)
            except (MemoryError, RuntimeError, TypeError, ValueError) as error:
                self.report({"WARNING"}, f"Local Feature Brush: dab failed ({error})")
                return self._finish(state, cancelled=True)
            return {"RUNNING_MODAL"}
        if event_type == "MOUSEMOVE" and state.get("stroke_active"):
            if bool(getattr(event, "shift", False)):
                return {"PASS_THROUGH"}
            coordinate = Vector((event.mouse_region_x, event.mouse_region_y))
            state["last_mouse"] = coordinate
            pressure = _local_feature_event_pressure(event)
            try:
                _local_feature_apply_dab(state, context, coordinate, pressure=pressure)
            except (MemoryError, RuntimeError, TypeError, ValueError) as error:
                self.report({"WARNING"}, f"Local Feature Brush: dab failed ({error})")
                return self._finish(state, cancelled=True)
            return {"RUNNING_MODAL"}
        if event_type == "LEFTMOUSE" and event_value == "RELEASE":
            state["stroke_active"] = False
            if not state.get("changed") and not state.get("native_prepared"):
                # A flat/masked/no-op stroke must not leave an empty generic
                # REGISTER,UNDO record behind.
                return self._finish(state, cancelled=True)
            return self._finish(state, cancelled=False)
        if event_type in {"RIGHTMOUSE", "WINDOW_DEACTIVATE"}:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}
