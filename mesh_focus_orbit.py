"""Mesh Focus Orbit.

Double-tap the configured key in a 3D Viewport to make the first visible mesh
surface under the viewport center the temporary orbit target.  The built-in
viewport navigation, including the Navigation Gizmo, remains responsible for
rotating the view.
"""

bl_info = {
    "name": "Mesh Focus Orbit",
    "author": "OpenAI",
    "version": (3, 2, 17),
    "blender": (5, 2, 0),
    "location": "3D View",
    "description": "Temporary mesh-centered orbit and local Smart Face Set Fill preview",
    "category": "3D View",
}

import bpy
import bmesh
import blf
import copy
import gpu
import heapq
import json
import math
import os
import statistics
import tempfile
import time
from array import array
from collections import deque

from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.geometry import intersect_ray_tri, tessellate_polygon
from mathutils.bvhtree import BVHTree


OPERATOR_ID = "view3d.mesh_focus_orbit"
FACE_SET_ACTIVATION_OPERATOR_ID = "view3d.mesh_focus_face_set_activate"
WATCHER_OPERATOR_ID = "view3d.mesh_focus_orbit_watcher"
RECOVER_FACE_SET_STATE_OPERATOR_ID = "view3d.mesh_focus_orbit_recover_face_set_state"
LOCAL_FACE_SET_GROW_OPERATOR_ID = "view3d.mesh_focus_local_face_set_grow_v2"
LOCAL_FACE_SET_GROW_KEY = "E"
TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID = "view3d.mesh_focus_topology_color_assign"
TOPOLOGY_COLOR_ATTRIBUTE_NAME = "mfo_topology_color"
TOPOLOGY_COLOR_PANEL_CATEGORY = "MFO"
TOPOLOGY_COLOR_PALETTE = (
    (0.93, 0.20, 0.20),
    (0.98, 0.57, 0.12),
    (0.95, 0.88, 0.10),
    (0.22, 0.80, 0.34),
    (0.16, 0.65, 0.96),
    (0.65, 0.32, 0.94),
)

_addon_keymaps = []
_active_states = {}
_last_tap_times = {}
_session_serial = 0
_session_cleanup_areas = set()
_undo_orphan_cleanup_pending = False
_undo_orphan_cleanup_retries = 0
_last_undo_post_perf = None
_retopo_undo_retry_pending = False
_retopo_undo_tombstones = {}
_local_face_set_adjacency_cache = {}
_fill_preview_adjacency_cache = {}
_fill_preview_cursor_cache = {}
_fill_preview_state = None
_fill_preview_draw_handler = None
_fill_preview_text_draw_handler = None
_fill_preview_shader = None
_FILL_PREVIEW_WHEEL_DRAIN_SECONDS = 0.12
_is_registered = False
_polyquilt_qsnap_class = None
_polyquilt_qsnap_original_snap_objects = None
_polyquilt_qsnap_filter_installed = False
_retopoflow_nearest_filter_class = None
_retopoflow_nearest_filter_original_update = None
_retopoflow_nearest_filter_installed = False
_retopoflow_automerge_class = None
_retopoflow_automerge_original = None
_retopoflow_automerge_installed = False
_retopoflow_automerge_nearest_bmvert = None
_retopoflow_automerge_filter = None
_retopoflow_automerge_session_id = None
_retopoflow_automerge_retopo_session_id = None
_retopoflow_polypen_class = None
_retopoflow_polypen_original_update = None
_retopoflow_polypen_installed = False
_retopoflow_polypen_owner = None
_retopoflow_polypen_nearest_bmvert = None
_retopoflow_polypen_filter = None
_retopoflow_polypen_session_id = None
_retopoflow_polypen_retopo_session_id = None
_retopoflow_translate_preview_class = None
_retopoflow_translate_preview_original = None
_retopoflow_translate_preview_installed = False
_retopoflow_translate_preview_owner = None
_retopoflow_translate_preview_nearest_bmvert = None
_retopoflow_translate_preview_filter = None
_retopoflow_translate_preview_session_id = None
_retopoflow_translate_preview_retopo_session_id = None
_retopo_isolation_serial = 0

# Topology-color drawing owns no BMesh elements.  Cache entries contain only
# copied world-space coordinates, so Undo, file load, and mode changes can
# invalidate them without retaining a stale Edit BMesh.
_topology_color_draw_handler = None
_topology_color_cache = {}
_topology_color_cache_dirty = set()
_topology_color_depth_shader_cache = None
_topology_color_fallback_shader = None

_TOPOLOGY_COLOR_DEPTH_VERTEX_SOURCE = """
void main()
{
    vec4 clip = ModelViewProjectionMatrix * vec4(pos, 1.0);
    clip.z -= depth_bias * clip.w;
    gl_Position = clip;
}
"""

_TOPOLOGY_COLOR_DEPTH_FRAGMENT_SOURCE = """
void main()
{
    fragColor = color;
}
"""

# ``importlib.reload`` replaces these functions while Blender keeps handler
# objects from the previous module instance.  Remove only our old handlers so
# reload never doubles the draw/update callbacks.
_TOPOLOGY_COLOR_HANDLER_NAMES = {
    "_on_topology_color_depsgraph_update",
    "_on_topology_color_undo_post",
    "_on_topology_color_redo_post",
    "_on_topology_color_load_pre",
    "_on_topology_color_load_post",
}
_FILL_PREVIEW_HANDLER_NAMES = {
    "_on_fill_preview_depsgraph_update",
}
for _handler_list_name in (
    "depsgraph_update_post",
    "undo_post",
    "redo_post",
    "load_pre",
    "load_post",
):
    try:
        _handler_list = getattr(bpy.app.handlers, _handler_list_name)
        for _old_handler in list(_handler_list):
            if (
                getattr(_old_handler, "__name__", "") in _TOPOLOGY_COLOR_HANDLER_NAMES
                or getattr(_old_handler, "__name__", "") in _FILL_PREVIEW_HANDLER_NAMES
            ):
                _handler_list.remove(_old_handler)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
del _handler_list_name
_RETOPO_DEBUG_LOG_PATH = os.path.join(
    tempfile.gettempdir(),
    "mesh_focus_orbit_debug.jsonl",
)
_RETOPO_DEBUG_LOG_BACKUP = _RETOPO_DEBUG_LOG_PATH + ".1"
_RETOPO_DEBUG_SCHEMA_VERSION = 1
_RETOPO_DEBUG_MAX_BYTES = 2 * 1024 * 1024
_RETOPO_DEBUG_MAX_RECORDS = 320
_RETOPO_DEBUG_BUCKET_LIMITS = {
    "normal": 128,
    "priority": 64,
    "release": 32,
    "lifecycle": 32,
    "diagnostic": 64,
}
_retopo_debug_sessions = {}
_retopo_debug_retired_sessions = {}
_retopo_debug_last_session_id = None
_retopo_debug_file_size = None
_retopo_debug_file_records = None

# ``importlib.reload`` keeps names that disappeared from the source module.
# Remove the retired Undo-resync helpers before the new module body is used so
# an add-on reload cannot accidentally expose or call stale recovery code.
for _stale_callback_name in (
    "_retopo_debug_followup_01",
    "_retopo_debug_followup_05",
):
    _stale_callback = globals().get(_stale_callback_name)
    if callable(_stale_callback):
        try:
            if bpy.app.timers.is_registered(_stale_callback):
                bpy.app.timers.unregister(_stale_callback)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
del _stale_callback_name, _stale_callback

for _stale_name in (
    "_on_undo_pre",
    "_on_undo_post",
    "_resync_active_face_set_states_after_undo",
    "_retopo_rebuild_isolation_after_undo",
    "_debug_undo_snapshot",
    "_debug_undo_log",
    "_modal_operator_areas",
    "_UNDO_DEBUG_OBSERVATION",
    "_UNDO_DEBUG_TEXT_NAME",
    "_UNDO_BOUNDARY_ACTIVE",
    "_UNDO_BOUNDARY_CROSSED",
    "_UNDO_BOUNDARY_PENDING",
    "_UNDO_BOUNDARY_INVALID",
    "_retopo_debug_followup_01",
    "_retopo_debug_followup_05",
    "_retopo_debug_schedule_followups",
    "_retopo_debug_cancel_followups",
    "_retopo_debug_followup_targets",
    "_retopo_debug_followup_pending",
):
    globals().pop(_stale_name, None)
del _stale_name

DEFAULT_DOUBLE_TAP_WINDOW = 0.28

# Face Set MFO creates a temporary scene object because Blender's Face Set
# overlay is not reliable while a different object remains active in Edit Mode.
# These namespaced markers let us recover if Blender invalidates the Python
# modal state (for example after Undo) before the normal finish path runs.
_FACE_SET_PROXY_NAME_TOKEN = "__MFO_FACE_SET_PROXY__"
_FACE_SET_PROXY_MESH_NAME_TOKEN = "__MFO_FACE_SET_PROXY_MESH"
_FACE_SET_PROXY_TAG = "mesh_focus_orbit.face_set_proxy"
_FACE_SET_PROXY_REFERENCE = "mesh_focus_orbit.reference_name"
_FACE_SET_PROXY_PREVIOUS_HIDE = "mesh_focus_orbit.reference_previous_hide"
_FACE_SET_PROXY_MESH_TAG = "mesh_focus_orbit.face_set_proxy_mesh"
_FACE_SET_REFERENCE_TAG = "mesh_focus_orbit.reference_hidden_by_proxy"
_FACE_SET_REFERENCE_PREVIOUS_HIDE = "mesh_focus_orbit.reference_previous_hide"
_FACE_SET_REFERENCE_PROXY_COUNT = "mesh_focus_orbit.proxy_count"

# These are temporary, narrowly-scoped runtime compatibility hooks.  They
# never edit the RetopoFlow installation permanently; the markers let a later
# reload or cleanup restore only wrappers created by Mesh Focus Orbit.
_RETOPOFLOW_NEAREST_FILTER_TAG = "mesh_focus_orbit.retopoflow_nearest_filter"
_RETOPOFLOW_NEAREST_FILTER_ORIGINAL = (
    "mesh_focus_orbit.retopoflow_original_nearest_bmvert_update"
)
_RETOPOFLOW_AUTOMERGE_TAG = "mesh_focus_orbit.retopoflow_automerge_hook"
_RETOPOFLOW_AUTOMERGE_ORIGINAL = (
    "mesh_focus_orbit.retopoflow_original_automerge"
)
_RETOPOFLOW_POLYPEN_TAG = "mesh_focus_orbit.retopoflow_polypen_hook"
_RETOPOFLOW_POLYPEN_ORIGINAL = (
    "mesh_focus_orbit.retopoflow_original_polypen_update"
)
_RETOPOFLOW_TRANSLATE_PREVIEW_TAG = (
    "mesh_focus_orbit.retopoflow_translate_preview_hook"
)
_RETOPOFLOW_TRANSLATE_PREVIEW_ORIGINAL = (
    "mesh_focus_orbit.retopoflow_original_translate_preview"
)

# Face Set MFO may also isolate the active Edit Mesh.  These markers are
# temporary recovery metadata only; they are removed when the mode finishes.
_RETOPO_ISOLATION_TAG = "mesh_focus_orbit.retopo_isolation"
_RETOPO_ISOLATION_SESSION = "mesh_focus_orbit.retopo_isolation_session"
_RETOPO_ISOLATION_MARKER_LAYER = "mesh_focus_orbit.retopo_marker_layer"
_RETOPO_ISOLATION_HIDE_LAYER = "mesh_focus_orbit.retopo_hide_layer"
_RETOPO_ISOLATION_SELECT_LAYER = "mesh_focus_orbit.retopo_select_layer"
_RETOPO_ISOLATION_TARGET_LAYER = "mesh_focus_orbit.retopo_target_layer"
_RETOPO_ISOLATION_VERT_MARKER_LAYER = "mesh_focus_orbit.retopo_vert_marker_layer"
_RETOPO_ISOLATION_VERT_TARGET_LAYER = "mesh_focus_orbit.retopo_vert_target_layer"
_RETOPO_ISOLATION_VERT_ORIGIN_LAYER = "mesh_focus_orbit.retopo_vert_origin_layer"
_RETOPO_ISOLATION_VERT_CREATED_LAYER = "mesh_focus_orbit.retopo_vert_created_layer"
_RETOPO_ISOLATION_VERT_HIDE_LAYER = "mesh_focus_orbit.retopo_vert_hide_layer"
_RETOPO_ISOLATION_VERT_SELECT_LAYER = "mesh_focus_orbit.retopo_vert_select_layer"
_RETOPO_ISOLATION_EDGE_MARKER_LAYER = "mesh_focus_orbit.retopo_edge_marker_layer"
_RETOPO_ISOLATION_EDGE_HIDE_LAYER = "mesh_focus_orbit.retopo_edge_hide_layer"
_RETOPO_ISOLATION_EDGE_SELECT_LAYER = "mesh_focus_orbit.retopo_edge_select_layer"


# Automatic Retopo island classification thresholds and score parameters.
# Distance scales are deliberately explicit so they can later be adapted to
# the model scale without changing the interaction design.
RETOPO_ISOLATION_DISTANCE_TOLERANCE = 0.005
RETOPO_ISOLATION_MAX_MEDIAN_DISTANCE = 0.010
RETOPO_ISOLATION_MIN_NEAR_RATIO = 0.65
RETOPO_ISOLATION_MIN_CONFIDENCE_GAP = 0.15
RETOPO_ISOLATION_MIN_SCORE = 0.50
RETOPO_ISOLATION_MEDIAN_DECAY_DISTANCE = 0.002
RETOPO_ISOLATION_P90_DECAY_DISTANCE = 0.005
RETOPO_ISOLATION_NEAR_RATIO_WEIGHT = 0.60
RETOPO_ISOLATION_MEDIAN_QUALITY_WEIGHT = 0.30
RETOPO_ISOLATION_P90_QUALITY_WEIGHT = 0.10
RETOPO_ISOLATION_SAMPLE_LIMIT = 200


ACTIVATION_ITEMS = (
    ("LEFT_CTRL", "Left Ctrl", "Double-tap Left Ctrl to toggle temporary orbit"),
    ("RIGHT_CTRL", "Right Ctrl", "Double-tap Right Ctrl to toggle temporary orbit"),
    ("LEFT_SHIFT", "Left Shift", "Double-tap Left Shift to toggle temporary orbit"),
    ("RIGHT_SHIFT", "Right Shift", "Double-tap Right Shift to toggle temporary orbit"),
    ("LEFT_ALT", "Left Alt", "Double-tap Left Alt to toggle temporary orbit"),
    ("RIGHT_ALT", "Right Alt", "Double-tap Right Alt to toggle temporary orbit"),
)


FOCUS_LOSS_ITEMS = (
    (
        "KEEP",
        "Keep Mode",
        "Keep Mesh Focus Orbit active when the Blender window loses focus",
    ),
    (
        "EXIT",
        "Exit Mode",
        "Exit Mesh Focus Orbit when the Blender window loses focus",
    ),
)


def _addon_preferences():
    """Return this add-on's preferences when Blender has created them."""
    addon_id = __package__ or __name__
    addon = bpy.context.preferences.addons.get(addon_id)
    return addon.preferences if addon else None


def _tag_redraw(area):
    try:
        if area and area.type == "VIEW_3D":
            area.tag_redraw()
    except (ReferenceError, AttributeError, TypeError):
        pass


def _topology_color_tag_redraw_all():
    """Redraw every visible 3D View without requiring a context override."""
    try:
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _invalidate_topology_color_cache(obj=None):
    """Invalidate copied overlay geometry, never retaining BMesh elements."""
    if obj is None:
        # Clearing is both cheaper and safer than a process-wide dirty marker:
        # the next viewport rebuilds only the active Edit Mesh.
        _topology_color_cache.clear()
        _topology_color_cache_dirty.clear()
    else:
        try:
            _topology_color_cache_dirty.add(int(obj.as_pointer()))
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            _topology_color_cache.clear()
            _topology_color_cache_dirty.clear()
    _topology_color_tag_redraw_all()


def _topology_color_object(context):
    """Return the active edit mesh used by the overlay and operators."""
    try:
        obj = context.edit_object
    except (AttributeError, ReferenceError, RuntimeError):
        obj = None
    if obj is None:
        try:
            obj = context.active_object
        except (AttributeError, ReferenceError, RuntimeError):
            obj = None
    if obj is None or getattr(obj, "type", None) != "MESH":
        return None
    if getattr(context, "mode", None) != "EDIT_MESH":
        return None
    return obj


def _next_session_id():
    """Return a process-unique FSMFO session generation."""
    global _session_serial
    _session_serial += 1
    return _session_serial


def _session_is_current(state):
    """Return whether *state* still owns its viewport session."""
    if state is None:
        return False
    try:
        return bool(
            state.active
            and state.session_id > 0
            and _active_states.get(state.area_key) is state
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _register_session(state):
    """Install one state as the sole owner of its viewport."""
    if state is None:
        return False
    try:
        area_key = int(state.area_key)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    if not area_key or area_key in _session_cleanup_areas:
        return False
    existing = _active_states.get(area_key)
    if existing is not None and getattr(existing, "active", False):
        return False
    state.session_id = _next_session_id()
    _active_states[area_key] = state
    return True


def _unregister_session(state):
    """Remove *state* only if it still owns its viewport registry entry."""
    if state is None:
        return
    try:
        if _active_states.get(state.area_key) is state:
            _active_states.pop(state.area_key, None)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


class _TempOrbitState:
    """State for one active temporary orbit in one viewport."""

    __slots__ = (
        "area",
        "region",
        "rv3d",
        "original_view_location",
        "original_view_distance",
        "original_view_rotation",
        "original_view_perspective",
        "hit_location",
        "hit_distance",
        "temporary_start_distance",
        "is_perspective",
        "activation_key",
        "draw_handler",
        "indicator_draw_handler",
        "active",
        "indicator_text",
        "session_id",
        "area_key",
        "watcher_operator_key",
        "watcher_timer",
    )

    def __init__(self, context, hit_location, hit_distance, activation_key):
        self.area = context.area
        self.region = context.region
        self.rv3d = context.space_data.region_3d
        self.original_view_location = self.rv3d.view_location.copy()
        self.original_view_distance = self.rv3d.view_distance
        self.original_view_rotation = self.rv3d.view_rotation.copy()
        self.original_view_perspective = self.rv3d.view_perspective
        self.hit_location = hit_location.copy()
        self.hit_distance = hit_distance
        self.temporary_start_distance = (
            hit_distance if self.rv3d.view_perspective in {"PERSP", "CAMERA"}
            else self.original_view_distance
        )
        self.is_perspective = self.rv3d.view_perspective in {"PERSP", "CAMERA"}
        self.activation_key = activation_key
        self.draw_handler = None
        self.indicator_draw_handler = None
        self.active = True
        self.indicator_text = "MESH FOCUS ORBIT ON"
        self.session_id = 0
        try:
            self.area_key = context.area.as_pointer()
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            self.area_key = 0
        self.watcher_operator_key = None
        self.watcher_timer = None


class _FaceSetOrbitState(_TempOrbitState):
    """Temporary orbit state with a reversible Face Set display proxy."""

    __slots__ = (
        "reference_object",
        "reference_object_name",
        "face_set_id",
        "proxy_object",
        "proxy_object_name",
        "reference_was_hidden",
        "retopo_isolation",
        "retopo_restore_snapshot",
    )

    def __init__(
        self,
        context,
        hit_location,
        hit_distance,
        activation_key,
        reference_object,
        face_set_id,
    ):
        super().__init__(context, hit_location, hit_distance, activation_key)
        self.reference_object = reference_object
        try:
            self.reference_object_name = str(reference_object.name)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            self.reference_object_name = ""
        self.face_set_id = int(face_set_id)
        try:
            self.reference_was_hidden = bool(
                reference_object.hide_get(view_layer=context.view_layer)
            )
        except (AttributeError, RuntimeError, TypeError):
            try:
                self.reference_was_hidden = bool(reference_object.hide_get())
            except (AttributeError, RuntimeError, TypeError):
                self.reference_was_hidden = False
        self.proxy_object = None
        self.proxy_object_name = ""
        self.retopo_isolation = None
        # This Python-side snapshot is deliberately independent from the
        # temporary BMesh layers.  Undo may remove those layers or rebuild the
        # Edit BMesh before undo_post runs, but the original visibility state
        # is still needed when FSMFO eventually ends.
        self.retopo_restore_snapshot = None
        self.indicator_text = "FACE SET MFO ON"


def _retopo_edit_object(context, reference_object):
    """Return the active Edit Mesh that can be isolated alongside FSMFO."""
    try:
        obj = context.edit_object
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        obj = None
    if obj is None:
        try:
            obj = context.active_object
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            obj = None
    if (
        obj is None
        or obj.type != "MESH"
        or obj.mode != "EDIT"
        or obj == reference_object
    ):
        return None
    return obj


def _retopo_face_is_valid(face):
    try:
        return bool(face.is_valid)
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False


def _retopo_connected_components(bm):
    """Return Ctrl+L-style valid Vert/Edge connectivity components.

    Faces remain attached to the same connectivity record for Face Set/BVH
    ranking, while wire and open-topology edges are retained in the component
    itself.  Visibility is deliberately not consulted.
    """
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    visited_vertices = set()
    components = []
    for start in bm.verts:
        if not _retopo_valid_element(start) or start in visited_vertices:
            continue
        component_vertices = {start}
        component_edges = set()
        component_faces = set()
        visited_vertices.add(start)
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            try:
                linked_edges = tuple(vertex.link_edges)
            except (AttributeError, ReferenceError, RuntimeError, TypeError):
                linked_edges = ()
            for edge in linked_edges:
                if not _retopo_valid_element(edge):
                    continue
                component_edges.add(edge)
                try:
                    edge_vertices = tuple(edge.verts)
                except (AttributeError, ReferenceError, RuntimeError, TypeError):
                    edge_vertices = ()
                for linked_vertex in edge_vertices:
                    if not _retopo_valid_element(linked_vertex):
                        continue
                    component_vertices.add(linked_vertex)
                    if linked_vertex not in visited_vertices:
                        visited_vertices.add(linked_vertex)
                        queue.append(linked_vertex)
                try:
                    linked_faces = tuple(edge.link_faces)
                except (AttributeError, ReferenceError, RuntimeError, TypeError):
                    linked_faces = ()
                for face in linked_faces:
                    if not _retopo_face_is_valid(face):
                        continue
                    component_faces.add(face)
                    try:
                        face_edges = tuple(face.edges)
                    except (AttributeError, ReferenceError, RuntimeError, TypeError):
                        face_edges = ()
                    for face_edge in face_edges:
                        if not _retopo_valid_element(face_edge):
                            continue
                        component_edges.add(face_edge)
                        try:
                            face_vertices = tuple(face_edge.verts)
                        except (AttributeError, ReferenceError, RuntimeError, TypeError):
                            face_vertices = ()
                        for face_vertex in face_vertices:
                            if not _retopo_valid_element(face_vertex):
                                continue
                            component_vertices.add(face_vertex)
                            if face_vertex not in visited_vertices:
                                visited_vertices.add(face_vertex)
                                queue.append(face_vertex)
        components.append(
            {
                "faces": component_faces,
                "verts": component_vertices,
                "edges": component_edges,
            }
        )
    return components


def _retopo_even_sample(items, limit=RETOPO_ISOLATION_SAMPLE_LIMIT):
    if len(items) <= limit:
        return list(items)
    return [
        items[round(index * (len(items) - 1) / (limit - 1))]
        for index in range(limit)
    ]


def _retopo_percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(
        ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    )


def _retopo_proxy_bvh(proxy):
    if proxy is None or proxy.type != "MESH":
        return None
    try:
        vertices = [
            tuple(proxy.matrix_world @ vertex.co)
            for vertex in proxy.data.vertices
        ]
        polygons = [tuple(polygon.vertices) for polygon in proxy.data.polygons]
        if not vertices or not polygons:
            return None
        return BVHTree.FromPolygons(
            vertices,
            polygons,
            all_triangles=False,
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _retopo_island_metrics(retopo_object, component, proxy_bvh):
    faces = component.get("faces", ()) if isinstance(component, dict) else component
    vertices = []
    seen_vertices = set()
    for face in faces:
        for vertex in face.verts:
            if vertex not in seen_vertices:
                seen_vertices.add(vertex)
                vertices.append(vertex)
    samples = _retopo_even_sample(vertices)
    distances = []
    for vertex in samples:
        try:
            nearest = proxy_bvh.find_nearest(
                retopo_object.matrix_world @ vertex.co
            )
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            nearest = None
        if nearest is not None and nearest[3] is not None:
            distances.append(float(nearest[3]))

    near_ratio = None
    median_distance = None
    p90_distance = None
    median_quality = None
    p90_quality = None
    score = None
    if distances:
        near_ratio = sum(
            distance <= RETOPO_ISOLATION_DISTANCE_TOLERANCE
            for distance in distances
        ) / len(distances)
        median_distance = statistics.median(distances)
        p90_distance = _retopo_percentile(distances, 0.90)
        median_quality = math.exp(
            -max(0.0, median_distance)
            / RETOPO_ISOLATION_MEDIAN_DECAY_DISTANCE
        )
        p90_quality = math.exp(
            -max(0.0, p90_distance)
            / RETOPO_ISOLATION_P90_DECAY_DISTANCE
        )
        score = (
            RETOPO_ISOLATION_NEAR_RATIO_WEIGHT * near_ratio
            + RETOPO_ISOLATION_MEDIAN_QUALITY_WEIGHT * median_quality
            + RETOPO_ISOLATION_P90_QUALITY_WEIGHT * p90_quality
        )
    return {
        "component": component,
        "face_count": len(faces),
        "vertex_count": len(vertices),
        "sample_count": len(samples),
        "near_ratio": near_ratio,
        "median_distance": median_distance,
        "p90_distance": p90_distance,
        "median_quality": median_quality,
        "p90_quality": p90_quality,
        "score": score,
    }


def _retopo_candidate_is_acceptable(metrics):
    return (
        metrics is not None
        and metrics["near_ratio"] is not None
        and metrics["median_distance"] is not None
        and metrics["score"] is not None
        and metrics["near_ratio"] >= RETOPO_ISOLATION_MIN_NEAR_RATIO
        and metrics["median_distance"] <= RETOPO_ISOLATION_MAX_MEDIAN_DISTANCE
        and metrics["score"] >= RETOPO_ISOLATION_MIN_SCORE
    )


def _retopo_selected_element(bm):
    """Return the preferred selected Face, Edge, or Vertex for fallback."""
    active = bm.select_history.active
    selected_faces = [face for face in bm.faces if face.select]
    if isinstance(active, bmesh.types.BMFace) and _retopo_face_is_valid(active):
        if active.select:
            return active
    if selected_faces:
        return selected_faces[0]

    selected_edges = [edge for edge in bm.edges if edge.select and edge.is_valid]
    if isinstance(active, bmesh.types.BMEdge) and active.is_valid and active.select:
        return active
    if selected_edges:
        return selected_edges[0]

    selected_verts = [vertex for vertex in bm.verts if vertex.select and vertex.is_valid]
    if isinstance(active, bmesh.types.BMVert) and active.is_valid and active.select:
        return active
    if selected_verts:
        return selected_verts[0]
    return None


def _retopo_component_matches_element(component, element):
    if isinstance(component, dict):
        if isinstance(element, bmesh.types.BMFace):
            return element in component.get("faces", ())
        if isinstance(element, bmesh.types.BMEdge):
            return element in component.get("edges", ())
        if isinstance(element, bmesh.types.BMVert):
            return element in component.get("verts", ())
        return False
    if isinstance(element, bmesh.types.BMFace):
        return any(face is element for face in component)
    if isinstance(element, bmesh.types.BMEdge):
        return any(element in face.edges for face in component)
    if isinstance(element, bmesh.types.BMVert):
        return any(element in face.verts for face in component)
    return False


def _retopo_select_island(context, reference_object, proxy):
    """Select the best Retopo island using Proxy proximity, not selection."""
    retopo_object = _retopo_edit_object(context, reference_object)
    if retopo_object is None:
        return None

    try:
        bm = bmesh.from_edit_mesh(retopo_object.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.verts.index_update()
        bm.edges.index_update()
        bm.faces.index_update()
        components = _retopo_connected_components(bm)
        proxy_bvh = _retopo_proxy_bvh(proxy)
        if not components:
            return None
        if proxy_bvh is None:
            return None

        metrics = [
            _retopo_island_metrics(retopo_object, component, proxy_bvh)
            for component in components
        ]
        ranked = sorted(
            metrics,
            key=lambda item: (
                item["score"] if item["score"] is not None else -1.0,
                item["near_ratio"] if item["near_ratio"] is not None else -1.0,
                -(
                    item["median_distance"]
                    if item["median_distance"] is not None
                    else float("inf")
                ),
            ),
            reverse=True,
        )
        if not ranked or not _retopo_candidate_is_acceptable(ranked[0]):
            best = None
        else:
            best = ranked[0]

        auto_confident = False
        if best is not None:
            if len(ranked) == 1:
                auto_confident = True
            else:
                next_best = ranked[1]
                score_gap = best["score"] - (
                    next_best["score"]
                    if next_best["score"] is not None
                    else -1.0
                )
                auto_confident = score_gap >= RETOPO_ISOLATION_MIN_CONFIDENCE_GAP

        if auto_confident:
            chosen = best
        else:
            # Selection is only a fallback when proximity confidence is not
            # sufficient.  It is never allowed to override a clear automatic
            # result.
            selected_seed = _retopo_selected_element(bm)
            selected_component = None
            if selected_seed is not None:
                matching_components = [
                    item
                    for item in metrics
                    if _retopo_component_matches_element(
                        item["component"], selected_seed
                    )
                ]
                # A vertex can touch multiple edge-connected components at a
                # point.  Do not isolate an arbitrary one in that ambiguous
                # case.
                if len(matching_components) == 1:
                    selected_component = matching_components[0]
            # An explicit Edit Mode selection is the only fallback that may
            # bypass the proximity score.  Automatic detection remains the
            # first choice; selection is consulted only after its confidence
            # test has failed.
            chosen = selected_component

        if chosen is None:
            return None
        chosen_component = chosen["component"]
        return {
            "object": retopo_object,
            "component": list(chosen_component.get("faces", ())),
            "component_verts": list(chosen_component.get("verts", ())),
            "component_edges": list(chosen_component.get("edges", ())),
            "component_vert_indices": [
                int(vertex.index)
                for vertex in chosen_component.get("verts", ())
                if _retopo_valid_element(vertex)
            ],
            "component_edge_indices": [
                int(edge.index)
                for edge in chosen_component.get("edges", ())
                if _retopo_valid_element(edge)
            ],
            # This is only a hand-off between two immediate BMesh reads during
            # activation.  It is not used for restoration after topology
            # changes; direct face refs and the temporary marker layer are
            # used for that purpose.
            "component_indices": [
                int(face.index)
                for face in chosen_component.get("faces", ())
                if _retopo_face_is_valid(face)
            ],
            "metrics": metrics,
        }
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _retopo_next_session_id():
    global _retopo_isolation_serial
    _retopo_isolation_serial = (_retopo_isolation_serial + 1) & 0x7FFFFFFF
    if _retopo_isolation_serial == 0:
        _retopo_isolation_serial = 1
    # The time component avoids collisions after an add-on reload while old
    # recovery markers are still present in the current .blend.
    return ((int(time.time() * 1000000) & 0x7FFFFFFF) ^ _retopo_isolation_serial) or 1


def _retopo_element_token(element):
    """Return a stable BMesh element token across Python wrapper reads.

    Blender 5.2 BMVert/BMEdge/BMFace wrappers do not expose ``as_pointer``.
    Their hash is backed by the underlying BMesh element identity and remains
    stable when ``bmesh.from_edit_mesh`` returns another Python wrapper.
    """
    try:
        return int(hash(element))
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _retopo_layer_names(session_id):
    prefix = f"__MFO_RETOPO_{session_id:08x}"
    return {
        "marker": f"{prefix}_MARKER",
        "hide": f"{prefix}_HIDE",
        "select": f"{prefix}_SELECT",
        "target": f"{prefix}_TARGET",
        "vert_marker": f"{prefix}_VERT_MARKER",
        "vert_target": f"{prefix}_VERT_TARGET",
        "vert_origin": f"{prefix}_VERT_ORIGIN",
        "vert_created": f"{prefix}_VERT_CREATED",
        "vert_hide": f"{prefix}_VERT_HIDE",
        "vert_select": f"{prefix}_VERT_SELECT",
        "edge_marker": f"{prefix}_EDGE_MARKER",
        "edge_hide": f"{prefix}_EDGE_HIDE",
        "edge_select": f"{prefix}_EDGE_SELECT",
    }


def _retopo_set_object_markers(obj, session_id, layer_names):
    obj[_RETOPO_ISOLATION_TAG] = True
    obj[_RETOPO_ISOLATION_SESSION] = int(session_id)
    obj[_RETOPO_ISOLATION_MARKER_LAYER] = layer_names["marker"]
    obj[_RETOPO_ISOLATION_HIDE_LAYER] = layer_names["hide"]
    obj[_RETOPO_ISOLATION_SELECT_LAYER] = layer_names["select"]
    obj[_RETOPO_ISOLATION_TARGET_LAYER] = layer_names["target"]
    obj[_RETOPO_ISOLATION_VERT_MARKER_LAYER] = layer_names["vert_marker"]
    obj[_RETOPO_ISOLATION_VERT_TARGET_LAYER] = layer_names["vert_target"]
    obj[_RETOPO_ISOLATION_VERT_ORIGIN_LAYER] = layer_names["vert_origin"]
    obj[_RETOPO_ISOLATION_VERT_CREATED_LAYER] = layer_names["vert_created"]
    obj[_RETOPO_ISOLATION_VERT_HIDE_LAYER] = layer_names["vert_hide"]
    obj[_RETOPO_ISOLATION_VERT_SELECT_LAYER] = layer_names["vert_select"]
    obj[_RETOPO_ISOLATION_EDGE_MARKER_LAYER] = layer_names["edge_marker"]
    obj[_RETOPO_ISOLATION_EDGE_HIDE_LAYER] = layer_names["edge_hide"]
    obj[_RETOPO_ISOLATION_EDGE_SELECT_LAYER] = layer_names["edge_select"]


def _retopo_clear_object_markers(obj):
    if obj is None:
        return
    for key in (
        _RETOPO_ISOLATION_TAG,
        _RETOPO_ISOLATION_SESSION,
        _RETOPO_ISOLATION_MARKER_LAYER,
        _RETOPO_ISOLATION_HIDE_LAYER,
        _RETOPO_ISOLATION_SELECT_LAYER,
        _RETOPO_ISOLATION_TARGET_LAYER,
        _RETOPO_ISOLATION_VERT_MARKER_LAYER,
        _RETOPO_ISOLATION_VERT_TARGET_LAYER,
        _RETOPO_ISOLATION_VERT_ORIGIN_LAYER,
        _RETOPO_ISOLATION_VERT_CREATED_LAYER,
        _RETOPO_ISOLATION_VERT_HIDE_LAYER,
        _RETOPO_ISOLATION_VERT_SELECT_LAYER,
        _RETOPO_ISOLATION_EDGE_MARKER_LAYER,
        _RETOPO_ISOLATION_EDGE_HIDE_LAYER,
        _RETOPO_ISOLATION_EDGE_SELECT_LAYER,
    ):
        try:
            if key in obj:
                del obj[key]
        except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError):
            pass


def _retopo_begin_isolation(context, reference_object, proxy):
    """Create a reversible Retopo island isolation state, if confident."""
    chosen = _retopo_select_island(context, reference_object, proxy)
    if chosen is None:
        return None

    retopo_object = chosen["object"]
    _retopo_discard_undo_tombstones_for_object(retopo_object.name)
    component = chosen["component"]
    component_vertices = list(chosen.get("component_verts", ()))
    component_edges = list(chosen.get("component_edges", ()))
    component_vertex_indices = set(
        int(index)
        for index in chosen.get("component_vert_indices", ())
        if isinstance(index, int) or isinstance(index, float)
    )
    component_edge_indices = set(
        int(index)
        for index in chosen.get("component_edge_indices", ())
        if isinstance(index, int) or isinstance(index, float)
    )
    if "component_vert_indices" not in chosen:
        component_vertex_indices = {
            int(vertex.index)
            for vertex in component_vertices
            if _retopo_valid_element(vertex)
        }
    if "component_edge_indices" not in chosen:
        component_edge_indices = {
            int(edge.index)
            for edge in component_edges
            if _retopo_valid_element(edge)
        }
    try:
        bm = bmesh.from_edit_mesh(retopo_object.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        component_indices = chosen.get("component_indices", ())
        if component_indices:
            component_index_set = {
                int(index)
                for index in component_indices
                if 0 <= index < len(bm.faces)
            }
            component = [
                bm.faces[index]
                for index in component_indices
                if 0 <= index < len(bm.faces)
                and _retopo_face_is_valid(bm.faces[index])
            ]
        else:
            component = [
                face
                for face in component
                if _retopo_face_is_valid(face)
            ]
            component_index_set = {
                int(face.index)
                for face in component
                if _retopo_face_is_valid(face)
            }
        component_vertices = [
            vertex
            for vertex in component_vertices
            if _retopo_valid_element(vertex)
        ]
        component_edges = [
            edge
            for edge in component_edges
            if _retopo_valid_element(edge)
        ]
        # A Ctrl+L connectivity component may consist entirely of a loose
        # vertex/edge chain.  Faces are only a ranking/proxy derivative; they
        # are not required for membership or isolation.
        if (
            not component
            and not component_vertices
            and not component_edges
            and not component_vertex_indices
            and not component_edge_indices
        ):
            return None
        try:
            bm.faces.index_update()
            bm.edges.index_update()
            bm.verts.index_update()
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            return None
        component_vertices = [
            bm.verts[index]
            for index in sorted(component_vertex_indices)
            if 0 <= index < len(bm.verts)
            and _retopo_valid_element(bm.verts[index])
        ]
        component_edges = [
            bm.edges[index]
            for index in sorted(component_edge_indices)
            if 0 <= index < len(bm.edges)
            and _retopo_valid_element(bm.edges[index])
        ]
        session_id = _retopo_next_session_id()
        layer_names = _retopo_layer_names(session_id)
        marker_layer = bm.faces.layers.int.new(layer_names["marker"])
        hide_layer = bm.faces.layers.int.new(layer_names["hide"])
        select_layer = bm.faces.layers.int.new(layer_names["select"])
        target_layer = bm.faces.layers.int.new(layer_names["target"])
        vert_marker_layer = bm.verts.layers.int.new(layer_names["vert_marker"])
        vert_target_layer = bm.verts.layers.int.new(layer_names["vert_target"])
        vert_origin_layer = bm.verts.layers.int.new(layer_names["vert_origin"])
        vert_created_layer = bm.verts.layers.int.new(layer_names["vert_created"])
        vert_hide_layer = bm.verts.layers.int.new(layer_names["vert_hide"])
        vert_select_layer = bm.verts.layers.int.new(layer_names["vert_select"])
        edge_marker_layer = bm.edges.layers.int.new(layer_names["edge_marker"])
        edge_hide_layer = bm.edges.layers.int.new(layer_names["edge_hide"])
        edge_select_layer = bm.edges.layers.int.new(layer_names["edge_select"])

        # Creating BMesh custom layers can invalidate element wrappers even
        # when the underlying BMesh token is unchanged.  Rebind from the
        # index hand-off after all layers exist, and keep the already captured
        # index sets as the authoritative hide/target masks.
        component_vertices = [
            bm.verts[index]
            for index in sorted(component_vertex_indices)
            if 0 <= index < len(bm.verts)
            and _retopo_valid_element(bm.verts[index])
        ]
        component_edges = [
            bm.edges[index]
            for index in sorted(component_edge_indices)
            if 0 <= index < len(bm.edges)
            and _retopo_valid_element(bm.edges[index])
        ]
        target_vertex_indices = set(component_vertex_indices)
        target_edge_indices = set(component_edge_indices)

        active = bm.select_history.active
        active_face = (
            active
            if isinstance(active, bmesh.types.BMFace)
            and _retopo_face_is_valid(active)
            else None
        )
        original_records = []
        initial_faces = []
        for face in bm.faces:
            if not _retopo_face_is_valid(face):
                continue
            initial_faces.append(face)
            original_records.append(
                (face, bool(face.hide), bool(face.select))
            )
            face[marker_layer] = session_id
            face[hide_layer] = int(face.hide)
            face[select_layer] = int(face.select)
            face[target_layer] = int(face.index in component_index_set)

        try:
            bm.verts.index_update()
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass
        activation_bmesh_token = _retopo_element_token(bm)
        if activation_bmesh_token is None:
            raise RuntimeError("Retopo BMesh identity is unavailable")

        original_vert_records = []
        initial_verts = []
        initial_vert_indices = []
        initial_vert_hide_values = []
        initial_vertex_origin_ids = []
        initial_vertex_tokens = []
        try:
            bm.verts.index_update()
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass
        for vertex in bm.verts:
            initial_verts.append(vertex)
            try:
                vertex_index = int(vertex.index)
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                vertex_index = -1
            initial_vert_indices.append(vertex_index)
            initial_vert_hide_values.append(int(vertex.hide))
            original_vert_records.append(
                (vertex, bool(vertex.hide), bool(vertex.select))
            )
            vertex[vert_marker_layer] = session_id
            vertex[vert_target_layer] = int(vertex_index in target_vertex_indices)
            origin_id = vertex_index + 1 if vertex_index >= 0 else len(initial_verts)
            if origin_id <= 0:
                origin_id = len(initial_verts)
            vertex[vert_origin_layer] = int(origin_id)
            vertex[vert_created_layer] = 0
            initial_vertex_origin_ids.append(int(origin_id))
            token = _retopo_element_token(vertex)
            if token is None:
                raise RuntimeError("Retopo vertex identity is unavailable")
            initial_vertex_tokens.append(token)
            vertex[vert_hide_layer] = int(vertex.hide)
            vertex[vert_select_layer] = int(vertex.select)

        original_edge_records = []
        initial_edges = []
        for edge in bm.edges:
            initial_edges.append(edge)
            original_edge_records.append(
                (edge, bool(edge.hide), bool(edge.select))
            )
            edge[edge_marker_layer] = session_id
            edge[edge_hide_layer] = int(edge.hide)
            edge[edge_select_layer] = int(edge.select)

        for face in initial_faces:
            if face.index not in component_index_set:
                face.hide = True
                face.select = False

        for vertex in initial_verts:
            if int(vertex.index) not in target_vertex_indices:
                vertex.hide = True
                vertex.select = False

        for edge in initial_edges:
            if int(edge.index) not in target_edge_indices:
                edge.hide = True
                edge.select = False

        info = {
            "object_name": retopo_object.name,
            "session_id": session_id,
            "layer_names": layer_names,
            "initial_faces": initial_faces,
            "original_records": original_records,
            "component_faces": list(component),
            "component_vertices": list(component_vertices),
            "component_edges": list(component_edges),
            "target_members": set(component_vertices),
            "target_member_edges": set(component_edges),
            "target_member_faces": set(component),
            "target_members_bmesh_token": activation_bmesh_token,
            "initial_verts": initial_verts,
            "initial_vert_indices": initial_vert_indices,
            "initial_vert_hide_values": initial_vert_hide_values,
            "initial_vertex_origin_ids": initial_vertex_origin_ids,
            "initial_vertex_tokens": initial_vertex_tokens,
            "activation_bmesh_token": activation_bmesh_token,
            "initial_vertex_count": len(initial_verts),
            "original_vert_records": original_vert_records,
            "initial_edges": initial_edges,
            "original_edge_records": original_edge_records,
            "active_face": active_face,
        }
        # Capture before update_edit_mesh can invalidate the temporary
        # BMesh-element wrappers retained in the info dictionary.
        info["restore_snapshot"] = _retopo_capture_restore_snapshot(info)
        bmesh.update_edit_mesh(
            retopo_object.data,
            loop_triangles=False,
            destructive=False,
        )
        # ``update_edit_mesh`` may refresh face wrappers independently of
        # verts/edges.  Rebind all three component domains once more so the
        # runtime isolation record never carries a stale face-only view while
        # the underlying BMesh token is unchanged.
        try:
            bm_post = bmesh.from_edit_mesh(retopo_object.data)
            bm_post.verts.ensure_lookup_table()
            bm_post.edges.ensure_lookup_table()
            bm_post.faces.ensure_lookup_table()
            bm_post.verts.index_update()
            bm_post.edges.index_update()
            bm_post.faces.index_update()
            rebound_faces = [
                bm_post.faces[index]
                for index in sorted(component_index_set)
                if 0 <= index < len(bm_post.faces)
                and _retopo_face_is_valid(bm_post.faces[index])
            ]
            rebound_vertices = [
                bm_post.verts[index]
                for index in sorted(target_vertex_indices)
                if 0 <= index < len(bm_post.verts)
                and _retopo_valid_element(bm_post.verts[index])
            ]
            rebound_edges = [
                bm_post.edges[index]
                for index in sorted(target_edge_indices)
                if 0 <= index < len(bm_post.edges)
                and _retopo_valid_element(bm_post.edges[index])
            ]
            info["component_faces"] = rebound_faces
            info["component_vertices"] = rebound_vertices
            info["component_edges"] = rebound_edges
            info["target_members"] = set(rebound_vertices)
            info["target_member_edges"] = set(rebound_edges)
            info["target_member_faces"] = set(rebound_faces)
            info["target_members_bmesh_token"] = _retopo_element_token(bm_post)
        except (
            AttributeError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            pass
        _retopo_set_object_markers(retopo_object, session_id, layer_names)
        return info
    except (
        AttributeError,
        ReferenceError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        try:
            if "bm" in locals():
                for layer in (
                    locals().get("marker_layer"),
                    locals().get("hide_layer"),
                    locals().get("select_layer"),
                    locals().get("target_layer"),
                ):
                    if layer is not None:
                        bm.faces.layers.int.remove(layer)
                for layer in (
                    locals().get("vert_marker_layer"),
                    locals().get("vert_target_layer"),
                    locals().get("vert_origin_layer"),
                    locals().get("vert_created_layer"),
                    locals().get("vert_hide_layer"),
                    locals().get("vert_select_layer"),
                ):
                    if layer is not None:
                        bm.verts.layers.int.remove(layer)
                for layer in (
                    locals().get("edge_marker_layer"),
                    locals().get("edge_hide_layer"),
                    locals().get("edge_select_layer"),
                ):
                    if layer is not None:
                        bm.edges.layers.int.remove(layer)
                bmesh.update_edit_mesh(
                    retopo_object.data,
                    loop_triangles=False,
                    destructive=False,
                )
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        _retopo_clear_object_markers(retopo_object)
        return None


def _retopo_valid_element(element):
    try:
        return bool(element.is_valid)
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False


def _retopo_coordinate_signature(coordinate):
    try:
        return tuple(round(float(value), 9) for value in coordinate)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _retopo_element_geometry_signature(element):
    """Return a rebuild-tolerant local-space signature for a BMesh element."""
    try:
        if isinstance(element, bmesh.types.BMVert):
            return ("V", _retopo_coordinate_signature(element.co))
        if isinstance(element, bmesh.types.BMEdge):
            coordinates = sorted(
                _retopo_coordinate_signature(vertex.co)
                for vertex in element.verts
            )
            return ("E", tuple(coordinates))
        if isinstance(element, bmesh.types.BMFace):
            coordinates = sorted(
                _retopo_coordinate_signature(vertex.co)
                for vertex in element.verts
            )
            return ("F", tuple(coordinates))
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None
    return None


def _retopo_capture_restore_snapshot(info):
    """Capture pre-FSMFO visibility without retaining BMElement references.

    BMesh layers are the authoritative *active-session* membership store, but
    they are not a durable undo boundary.  This compact snapshot is only for
    restoring the user's pre-FSMFO hide/select state after a later rebuild;
    it contains no live Blender RNA or BMesh references.
    """
    if not info:
        return None
    snapshot = {
        "object_name": str(info.get("object_name", "")),
        "vertices": [],
        "edges": [],
        "faces": [],
    }

    def capture(elements_key, records_key, destination):
        elements = list(info.get(elements_key, ()))
        records = list(info.get(records_key, ()))
        for index, element in enumerate(elements):
            if not _retopo_valid_element(element):
                continue
            try:
                original_hide, original_select = records[index][1:3]
            except (IndexError, TypeError, ValueError):
                continue
            try:
                element_index = int(element.index)
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                element_index = -1
            destination.append({
                "token": _retopo_element_token(element),
                "index": element_index,
                "geometry": _retopo_element_geometry_signature(element),
                "hide": bool(original_hide),
                "select": bool(original_select),
            })

    capture("initial_verts", "original_vert_records", snapshot["vertices"])
    capture("initial_edges", "original_edge_records", snapshot["edges"])
    capture("initial_faces", "original_records", snapshot["faces"])
    return snapshot


def _retopo_match_restore_snapshot(bm, snapshot):
    """Match current BMesh elements to a Python-only activation snapshot."""
    if not snapshot:
        return {"verts": [], "edges": [], "faces": []}

    def match(elements, entries):
        entries = list(entries or ())
        by_token = {
            entry.get("token"): entry
            for entry in entries
            if entry.get("token") is not None
        }
        by_index_geometry = {}
        by_geometry = {}
        for entry in entries:
            geometry = entry.get("geometry")
            if geometry is None:
                continue
            by_index_geometry.setdefault(
                (int(entry.get("index", -1)), geometry), []
            ).append(entry)
            by_geometry.setdefault(geometry, []).append(entry)

        matched = []
        used = set()
        for element in elements:
            if not _retopo_valid_element(element):
                continue
            token = _retopo_element_token(element)
            geometry = _retopo_element_geometry_signature(element)
            try:
                element_index = int(element.index)
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                element_index = -1

            entry = by_token.get(token)
            if (
                entry is not None
                and entry.get("geometry") is not None
                and geometry != entry.get("geometry")
            ):
                entry = None
            if entry is None and geometry is not None:
                indexed = by_index_geometry.get((element_index, geometry), ())
                entry = next(
                    (candidate for candidate in indexed if id(candidate) not in used),
                    None,
                )
            if entry is None and geometry is not None:
                by_geometry_candidates = by_geometry.get(geometry, ())
                unused = [
                    candidate
                    for candidate in by_geometry_candidates
                    if id(candidate) not in used
                ]
                if len(unused) == 1:
                    entry = unused[0]
            if entry is None or id(entry) in used:
                continue
            used.add(id(entry))
            matched.append((element, entry))
        return matched

    return {
        "verts": match(bm.verts, snapshot.get("vertices")),
        "edges": match(bm.edges, snapshot.get("edges")),
        "faces": match(bm.faces, snapshot.get("faces")),
    }


def _retopo_apply_restore_snapshot(info, snapshot):
    """Rebind a fresh isolation session to the original activation elements.

    Elements absent from the activation snapshot are treated as topology
    created during FSMFO: they are made visible and have no trusted old-member
    marker.  They can be confirmed by the normal target-connectivity path when
    RetopoFlow creates a new snap candidate later.
    """
    if not info or not snapshot:
        return True
    object_name = str(info.get("object_name", ""))
    retopo_object = bpy.data.objects.get(object_name)
    if retopo_object is None or retopo_object.type != "MESH" or retopo_object.mode != "EDIT":
        return False

    try:
        bm = bmesh.from_edit_mesh(retopo_object.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        matched = _retopo_match_restore_snapshot(bm, snapshot)
        if not matched["faces"] and not matched["verts"]:
            return False

        names = info.get("layer_names", {})
        face_marker = bm.faces.layers.int.get(names.get("marker", ""))
        face_hide = bm.faces.layers.int.get(names.get("hide", ""))
        face_select = bm.faces.layers.int.get(names.get("select", ""))
        face_target = bm.faces.layers.int.get(names.get("target", ""))
        vert_marker = bm.verts.layers.int.get(names.get("vert_marker", ""))
        vert_target = bm.verts.layers.int.get(names.get("vert_target", ""))
        vert_origin = bm.verts.layers.int.get(names.get("vert_origin", ""))
        vert_created = bm.verts.layers.int.get(names.get("vert_created", ""))
        vert_hide = bm.verts.layers.int.get(names.get("vert_hide", ""))
        vert_select = bm.verts.layers.int.get(names.get("vert_select", ""))
        edge_marker = bm.edges.layers.int.get(names.get("edge_marker", ""))
        edge_hide = bm.edges.layers.int.get(names.get("edge_hide", ""))
        edge_select = bm.edges.layers.int.get(names.get("edge_select", ""))
        session_id = int(info.get("session_id", 0))

        def element_key(element):
            return _retopo_element_token(element) or id(element)

        face_matches = {
            element_key(element): entry for element, entry in matched["faces"]
        }
        vert_matches = {
            element_key(element): entry for element, entry in matched["verts"]
        }
        edge_matches = {
            element_key(element): entry for element, entry in matched["edges"]
        }

        for face in bm.faces:
            entry = face_matches.get(element_key(face))
            if entry is None:
                if face_marker is not None:
                    face[face_marker] = 0
                if face_target is not None:
                    face[face_target] = 0
                face.hide = False
                continue
            if face_marker is not None:
                face[face_marker] = session_id

        for vertex in bm.verts:
            entry = vert_matches.get(element_key(vertex))
            if entry is None:
                if vert_marker is not None:
                    vertex[vert_marker] = 0
                if vert_target is not None:
                    vertex[vert_target] = 0
                if vert_origin is not None:
                    vertex[vert_origin] = 0
                if vert_created is not None:
                    vertex[vert_created] = 0
                vertex.hide = False
                continue
            if vert_marker is not None:
                vertex[vert_marker] = session_id
            if vert_origin is not None:
                vert_index = int(entry.get("index", -1))
                vertex[vert_origin] = max(1, vert_index + 1)

        for edge in bm.edges:
            entry = edge_matches.get(element_key(edge))
            if entry is None:
                if edge_marker is not None:
                    edge[edge_marker] = 0
                edge.hide = False
                continue
            if edge_marker is not None:
                edge[edge_marker] = session_id

        info["initial_faces"] = [element for element, _entry in matched["faces"]]
        info["original_records"] = [
            (element, bool(entry.get("hide")), bool(entry.get("select")))
            for element, entry in matched["faces"]
        ]
        info["initial_verts"] = [element for element, _entry in matched["verts"]]
        info["original_vert_records"] = [
            (element, bool(entry.get("hide")), bool(entry.get("select")))
            for element, entry in matched["verts"]
        ]
        info["initial_vert_indices"] = [
            int(entry.get("index", -1)) for _element, entry in matched["verts"]
        ]
        info["initial_vert_hide_values"] = [
            int(bool(entry.get("hide"))) for _element, entry in matched["verts"]
        ]
        info["initial_vertex_origin_ids"] = [
            max(1, int(entry.get("index", -1)) + 1)
            for _element, entry in matched["verts"]
        ]
        info["initial_vertex_tokens"] = [
            _retopo_element_token(element) for element, _entry in matched["verts"]
        ]
        info["initial_vertex_count"] = len(matched["verts"])
        info["initial_edges"] = [element for element, _entry in matched["edges"]]
        info["original_edge_records"] = [
            (element, bool(entry.get("hide")), bool(entry.get("select")))
            for element, entry in matched["edges"]
        ]
        active_face = info.get("active_face")
        if (
            not _retopo_valid_element(active_face)
            or element_key(active_face) not in face_matches
        ):
            info["active_face"] = None

        bmesh.update_edit_mesh(
            retopo_object.data,
            loop_triangles=False,
            destructive=False,
        )
        return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _retopo_discard_isolation_layers(info):
    """Remove a superseded session's layers without changing hide state."""
    if not info:
        return
    object_name = str(info.get("object_name", ""))
    retopo_object = bpy.data.objects.get(object_name)
    if retopo_object is None or retopo_object.type != "MESH" or retopo_object.mode != "EDIT":
        return
    try:
        bm = bmesh.from_edit_mesh(retopo_object.data)
        names = info.get("layer_names", {})
        for domain, keys in (
            (bm.faces.layers.int, ("marker", "hide", "select", "target")),
            (bm.verts.layers.int, (
                "vert_marker", "vert_target", "vert_origin", "vert_created",
                "vert_hide", "vert_select",
            )),
            (bm.edges.layers.int, ("edge_marker", "edge_hide", "edge_select")),
        ):
            for key in keys:
                layer = domain.get(names.get(key, ""))
                if layer is not None:
                    domain.remove(layer)
        bmesh.update_edit_mesh(
            retopo_object.data,
            loop_triangles=False,
            destructive=False,
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _retopo_layer_handles(bm, info):
    names = info.get("layer_names", {})
    try:
        return (
            bm.faces.layers.int.get(names.get("marker", "")),
            bm.faces.layers.int.get(names.get("hide", "")),
            bm.faces.layers.int.get(names.get("select", "")),
            bm.faces.layers.int.get(names.get("target", "")),
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return None, None, None, None


def _retopo_element_layer_handles(bm, info, domain):
    names = info.get("layer_names", {})
    try:
        elements = getattr(bm, domain)
        layers = elements.layers.int
        return (
            layers.get(names.get(f"{domain[:-1]}_marker", "")),
            layers.get(names.get(f"{domain[:-1]}_hide", "")),
            layers.get(names.get(f"{domain[:-1]}_select", "")),
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return None, None, None


def _retopo_remove_named_layers(bm, domain, names):
    """Remove owned integer layers by name, reacquiring after each removal."""
    for name in names:
        if not name:
            continue
        try:
            layer = domain.get(name)
            if layer is not None:
                domain.remove(layer)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass


def _retopo_restore_info(info):
    """Restore one active or orphaned Retopo isolation without mode changes."""
    if not info:
        return False
    # Never trust the long-lived Object wrapper stored in modal state.  Undo,
    # RetopoFlow, and Edit Mesh rebuilds can remove that StructRNA while a new
    # Object with the same name remains in bpy.data.
    object_name = info.get("object_name", "")
    if not object_name:
        return False
    try:
        retopo_object = bpy.data.objects.get(str(object_name))
        if retopo_object is None or retopo_object.type != "MESH":
            return False
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False

    bm = None
    owns_bmesh = False
    try:
        if retopo_object.mode == "EDIT":
            bm = bmesh.from_edit_mesh(retopo_object.data)
        else:
            bm = bmesh.new()
            bm.from_mesh(retopo_object.data)
            owns_bmesh = True
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        marker_layer, hide_layer, select_layer, _target_layer = (
            _retopo_layer_handles(bm, info)
        )
        vert_marker_layer, vert_hide_layer, vert_select_layer = (
            _retopo_element_layer_handles(bm, info, "verts")
        )
        names = info.get("layer_names", {})
        vert_target_layer = bm.verts.layers.int.get(
            names.get("vert_target", "")
        )
        vert_origin_layer = bm.verts.layers.int.get(
            names.get("vert_origin", "")
        )
        vert_created_layer = bm.verts.layers.int.get(
            names.get("vert_created", "")
        )
        edge_marker_layer, edge_hide_layer, edge_select_layer = (
            _retopo_element_layer_handles(bm, info, "edges")
        )
        restored_direct = False

        for face, original_hide, original_select in info.get(
            "original_records", ()
        ):
            if not _retopo_face_is_valid(face):
                continue
            face.hide = bool(original_hide)
            face.select = bool(original_select)
            restored_direct = True

        for vertex, original_hide, original_select in info.get(
            "original_vert_records", ()
        ):
            if not _retopo_face_is_valid(vertex):
                continue
            vertex.hide = bool(original_hide)
            vertex.select = bool(original_select)
            restored_direct = True

        for edge, original_hide, original_select in info.get(
            "original_edge_records", ()
        ):
            if not _retopo_face_is_valid(edge):
                continue
            edge.hide = bool(original_hide)
            edge.select = bool(original_select)
            restored_direct = True

        # The layer fallback covers an Undo-rebuilt BMesh or a lost modal
        # Python state.  New elements have no session marker, so they remain
        # untouched and are not restored as old elements here.
        if marker_layer is not None and hide_layer is not None:
            session_id = int(info["session_id"])
            for face in bm.faces:
                if int(face[marker_layer]) != session_id:
                    continue
                face.hide = bool(face[hide_layer])
                if select_layer is not None:
                    face.select = bool(face[select_layer])

        if vert_marker_layer is not None and vert_hide_layer is not None:
            session_id = int(info["session_id"])
            for vertex in bm.verts:
                if int(vertex[vert_marker_layer]) != session_id:
                    continue
                vertex.hide = bool(vertex[vert_hide_layer])
                if vert_select_layer is not None:
                    vertex.select = bool(vertex[vert_select_layer])

        if edge_marker_layer is not None and edge_hide_layer is not None:
            session_id = int(info["session_id"])
            for edge in bm.edges:
                if int(edge[edge_marker_layer]) != session_id:
                    continue
                edge.hide = bool(edge[edge_hide_layer])
                if edge_select_layer is not None:
                    edge.select = bool(edge[edge_select_layer])

        active_face = info.get("active_face")
        if _retopo_face_is_valid(active_face) and active_face.select:
            try:
                bm.select_history.add(active_face)
            except (AttributeError, ReferenceError, RuntimeError, TypeError):
                pass

        _retopo_remove_named_layers(
            bm,
            bm.faces.layers.int,
            (names.get("marker", ""), names.get("hide", ""),
             names.get("select", ""), names.get("target", "")),
        )
        _retopo_remove_named_layers(
            bm,
            bm.verts.layers.int,
            (names.get("vert_marker", ""), names.get("vert_target", ""),
             names.get("vert_origin", ""), names.get("vert_created", ""),
             names.get("vert_hide", ""), names.get("vert_select", "")),
        )
        _retopo_remove_named_layers(
            bm,
            bm.edges.layers.int,
            (names.get("edge_marker", ""), names.get("edge_hide", ""),
             names.get("edge_select", "")),
        )

        if owns_bmesh:
            bm.to_mesh(retopo_object.data)
            retopo_object.data.update()
        else:
            bmesh.update_edit_mesh(
                retopo_object.data,
                loop_triangles=False,
                destructive=False,
            )
        _retopo_clear_object_markers(retopo_object)
        return (
            restored_direct
            or marker_layer is not None
            or vert_marker_layer is not None
            or edge_marker_layer is not None
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    finally:
        if owns_bmesh and bm is not None:
            try:
                bm.free()
            except (ReferenceError, RuntimeError, AttributeError):
                pass


def _retopo_revalidate_isolation(info):
    """Revalidate and reapply active FSMFO hide state after Undo.

    This is a fast path only.  If the session layers are absent, the caller
    falls back to fresh current-BMesh island detection; it never guesses from
    indices, Python BMElement references, or current visibility.
    """
    if not info:
        return False
    object_name = str(info.get("object_name", ""))
    session_id = int(info.get("session_id", 0))
    if not object_name or session_id <= 0:
        return False
    try:
        retopo_object = bpy.data.objects.get(object_name)
        if retopo_object is None or retopo_object.type != "MESH":
            return False
        if retopo_object.mode != "EDIT":
            return False
        bm = bmesh.from_edit_mesh(retopo_object.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        names = info.get("layer_names") or {}
        face_marker_layer = bm.faces.layers.int.get(names.get("marker", ""))
        face_target_layer = bm.faces.layers.int.get(names.get("target", ""))
        vert_marker_layer = bm.verts.layers.int.get(
            names.get("vert_marker", "")
        )
        vert_target_layer = bm.verts.layers.int.get(
            names.get("vert_target", "")
        )
        vert_created_layer = bm.verts.layers.int.get(
            names.get("vert_created", "")
        )
        edge_marker_layer = bm.edges.layers.int.get(
            names.get("edge_marker", "")
        )
        if any(
            layer is None
            for layer in (
                face_marker_layer,
                face_target_layer,
                vert_marker_layer,
                vert_target_layer,
                vert_created_layer,
                edge_marker_layer,
            )
        ):
            return False

        session_face_count = 0
        for face in bm.faces:
            if int(face[face_marker_layer]) != session_id:
                continue
            target = int(face[face_target_layer])
            if target not in (0, 1):
                return False
            session_face_count += 1
        if session_face_count <= 0:
            return False

        for vertex in bm.verts:
            if int(vertex[vert_marker_layer]) != session_id:
                continue
            target = int(vertex[vert_target_layer])
            created = int(vertex[vert_created_layer])
            if target not in (0, 1):
                return False
            if created not in (0, session_id):
                return False

        # Reapply only the explicit target boundary.  Elements created after
        # activation have no session marker and are deliberately left alone.
        for face in bm.faces:
            if int(face[face_marker_layer]) != session_id:
                continue
            face.hide = not bool(int(face[face_target_layer]))
            if face.hide:
                face.select = False

        for vertex in bm.verts:
            if int(vertex[vert_marker_layer]) != session_id:
                continue
            vertex.hide = not bool(int(vertex[vert_target_layer]))
            if vertex.hide:
                vertex.select = False

        for edge in bm.edges:
            if int(edge[edge_marker_layer]) != session_id:
                continue
            edge_is_target = any(
                int(face[face_marker_layer]) == session_id
                and int(face[face_target_layer]) == 1
                for face in edge.link_faces
            )
            edge.hide = not edge_is_target
            if edge.hide:
                edge.select = False

        bmesh.update_edit_mesh(
            retopo_object.data,
            loop_triangles=False,
            destructive=False,
        )
        return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _retopo_capture_undo_visibility(obj):
    """Capture the exact ON-state hide/select surface for safe fallback use."""
    if obj is None or obj.type != "MESH" or obj.mode != "EDIT":
        return None
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        return {
            domain_name: [
                {
                    "geometry": _retopo_element_geometry_signature(element),
                    "hide": bool(element.hide),
                    "select": bool(element.select),
                }
                for element in elements
                if _retopo_element_geometry_signature(element) is not None
            ]
            for domain_name, elements in (
                ("verts", bm.verts),
                ("edges", bm.edges),
                ("faces", bm.faces),
            )
        }
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _retopo_visibility_match_status(obj, expected):
    """Classify the recorded ON-state without conflating mismatch and retry."""
    if obj is None or obj.type != "MESH" or obj.mode != "EDIT" or not expected:
        return "mismatch"
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        for domain_name, elements in (
            ("verts", bm.verts),
            ("edges", bm.edges),
            ("faces", bm.faces),
        ):
            candidates = {}
            for element in elements:
                geometry = _retopo_element_geometry_signature(element)
                if geometry is not None:
                    candidates.setdefault(geometry, []).append(element)
            for entry in expected.get(domain_name, ()):
                geometry = entry.get("geometry")
                matches = candidates.get(geometry, [])
                match_index = next(
                    (
                        index
                        for index, element in enumerate(matches)
                        if bool(element.hide) == bool(entry.get("hide"))
                        and bool(element.select) == bool(entry.get("select"))
                    ),
                    None,
                )
                if match_index is None:
                    return "mismatch"
                matches.pop(match_index)
        return "matched"
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return "unavailable"


def _retopo_visibility_matches_tombstone(obj, expected):
    """Verify native Undo restored the recorded ON-state before fallback."""
    return _retopo_visibility_match_status(obj, expected) == "matched"


def _retopo_make_undo_tombstone(info, snapshot=None):
    """Capture ended-session identity before normal OFF clears native markers."""
    if not info:
        return None
    object_name = str(info.get("object_name", ""))
    retopo_object = bpy.data.objects.get(object_name)
    if (
        not object_name
        or retopo_object is None
        or retopo_object.type != "MESH"
        or retopo_object.data is None
    ):
        return None
    session_id = int(info.get("session_id", 0))
    layer_names = dict(info.get("layer_names") or {})
    if session_id <= 0 or not layer_names.get("marker"):
        return None
    if snapshot is None:
        snapshot = _retopo_capture_restore_snapshot(info)
    return {
        "object_name": object_name,
        "mesh_name": str(retopo_object.data.name),
        "session_id": session_id,
        "layer_names": layer_names,
        "snapshot": copy.deepcopy(snapshot) if snapshot else None,
        "undo_visibility": _retopo_capture_undo_visibility(retopo_object),
        # Set only for a transient BMesh access failure in the current
        # undo_post event.  A normal state mismatch is a dormant tombstone.
        "retry_pending": False,
        # ``armed`` permits the one initial snapshot fallback.  Once a
        # session-layer restore succeeds, retain this tombstone as a
        # fail-closed history guard for later native Undo states from the
        # same ended FSMFO session.
        "guard_mode": "armed",
        "successful_restore_count": 0,
        "last_restore_method": None,
    }


def _retopo_record_undo_tombstone(info, snapshot=None):
    """Remember one ended session for the next native Undo reconciliation."""
    tombstone = _retopo_make_undo_tombstone(info, snapshot=snapshot)
    if tombstone is None:
        return None
    key = (
        tombstone["object_name"],
        tombstone["mesh_name"],
        tombstone["session_id"],
    )
    existing = _retopo_undo_tombstones.get(key)
    if existing is not None:
        if not existing.get("snapshot") and tombstone.get("snapshot"):
            existing["snapshot"] = tombstone["snapshot"]
        if not existing.get("undo_visibility") and tombstone.get("undo_visibility"):
            existing["undo_visibility"] = tombstone["undo_visibility"]
        return existing
    _retopo_undo_tombstones[key] = tombstone
    return tombstone


def _retopo_discard_undo_tombstones_for_object(object_name):
    object_name = str(object_name or "")
    for key in list(_retopo_undo_tombstones):
        if key[0] == object_name:
            _retopo_undo_tombstones.pop(key, None)


def _retopo_tombstone_info(tombstone, obj):
    return {
        "object": obj,
        "object_name": tombstone["object_name"],
        "session_id": tombstone["session_id"],
        "layer_names": dict(tombstone["layer_names"]),
        "original_records": [],
        "original_vert_records": [],
        "original_edge_records": [],
        "active_face": None,
    }


def _retopo_tombstone_session_layer_status(obj, tombstone):
    """Classify temporary layers while preserving transient BMesh failures."""
    if obj is None or obj.type != "MESH" or obj.mode != "EDIT":
        return "mismatch"
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        session_id = int(tombstone.get("session_id", 0))
        names = tombstone.get("layer_names") or {}
        for elements, key in (
            (bm.faces, "marker"),
            (bm.verts, "vert_marker"),
            (bm.edges, "edge_marker"),
        ):
            layer = elements.layers.int.get(names.get(key, ""))
            if layer is not None and any(
                int(element[layer]) == session_id for element in elements
            ):
                return "matched"
        return "missing"
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return "unavailable"


def _retopo_tombstone_has_session_layers(obj, tombstone):
    """Require an actual session marker before consuming temporary layers."""
    return _retopo_tombstone_session_layer_status(obj, tombstone) == "matched"


def _retopo_restore_snapshot_visibility(info, snapshot):
    """Restore only exact snapshot-matched elements when layers are absent."""
    if not info or not snapshot:
        return False
    object_name = str(info.get("object_name", ""))
    retopo_object = bpy.data.objects.get(object_name)
    if (
        retopo_object is None
        or retopo_object.type != "MESH"
        or retopo_object.mode != "EDIT"
    ):
        return False
    try:
        bm = bmesh.from_edit_mesh(retopo_object.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        matched = _retopo_match_restore_snapshot(bm, snapshot)
        matched_count = 0
        for elements in matched.values():
            for element, entry in elements:
                element.select = bool(entry.get("select", False))
                element.hide = bool(entry.get("hide", False))
                if not bool(entry.get("select", False)):
                    try:
                        element.select_set(False)
                    except (AttributeError, ReferenceError, RuntimeError, TypeError):
                        element.select = False
                matched_count += 1
        if not matched_count:
            return False
        bmesh.update_edit_mesh(
            retopo_object.data,
            loop_triangles=False,
            destructive=False,
        )
        return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _retopo_reconcile_undo_tombstones():
    """Reconcile ended sessions whose Object markers vanished during Undo."""
    global _retopo_undo_retry_pending
    reconcile_started = time.perf_counter()
    _retopo_undo_retry_pending = False
    active_object_names = {
        state.retopo_isolation.get("object_name")
        for state in _active_states.values()
        if isinstance(state, _FaceSetOrbitState)
        and state.active
        and state.retopo_isolation
    }
    pending = False
    for key, tombstone in list(_retopo_undo_tombstones.items()):
        object_name = tombstone["object_name"]
        if object_name in active_object_names:
            _retopo_debug_emit(
                "undo_reconcile_result",
                {
                    "tombstone_key": key,
                    "decision": "skip_active_state",
                    "reason": "active_object_exclusion",
                },
            )
            continue
        obj = bpy.data.objects.get(object_name)
        _retopo_debug_emit(
            "undo_reconcile_probe",
            {
                "tombstone_key": key,
                "active_object_excluded": False,
                "object_name": object_name,
                "mesh_name": str(tombstone.get("mesh_name", "")),
                "guard_mode": str(tombstone.get("guard_mode", "armed")),
            },
        )
        if obj is None or obj.type != "MESH":
            _retopo_undo_tombstones.pop(key, None)
            _retopo_debug_emit(
                "undo_reconcile_result",
                {
                    "tombstone_key": key,
                    "decision": "discard",
                    "reason": "object_missing_or_wrong_type",
                },
            )
            continue
        if str(obj.data.name) != tombstone["mesh_name"]:
            _retopo_undo_tombstones.pop(key, None)
            _retopo_debug_emit(
                "undo_reconcile_result",
                {
                    "tombstone_key": key,
                    "decision": "discard",
                    "reason": "mesh_identity_mismatch",
                    "expected_mesh_name": str(tombstone["mesh_name"]),
                    "actual_mesh_name": str(obj.data.name),
                },
            )
            continue
        tombstone["retry_pending"] = False
        if obj.mode != "EDIT":
            _retopo_debug_emit(
                "undo_reconcile_result",
                {
                    "tombstone_key": key,
                    "decision": "dormant",
                    "reason": "object_not_edit_mode",
                },
            )
            continue
        info = _retopo_tombstone_info(tombstone, obj)
        restored = False
        restore_method = None
        guard_mode = str(tombstone.get("guard_mode", "armed"))
        layer_status = _retopo_tombstone_session_layer_status(obj, tombstone)
        if layer_status == "unavailable":
            tombstone["retry_pending"] = True
            _retopo_undo_retry_pending = True
            pending = True
            _retopo_debug_emit(
                "undo_reconcile_result",
                {
                    "tombstone_key": key,
                    "decision": "retry",
                    "reason": "bmesh_unavailable_layer_probe",
                    "guard_mode": guard_mode,
                },
            )
            continue
        if layer_status == "matched":
            restore_method = "layers"
            restored = _retopo_restore_info(info)
        if not restored and layer_status == "missing":
            if guard_mode != "armed":
                # A successful restore has converted this tombstone into a
                # history guard.  Fingerprint-only matching is no longer
                # trusted: later unrelated user states must remain untouched.
                _retopo_debug_emit(
                    "undo_reconcile_result",
                    {
                        "tombstone_key": key,
                        "decision": "dormant",
                        "reason": "history_guard_requires_exact_session_layers",
                        "guard_mode": guard_mode,
                    },
                )
                continue
            visibility_status = _retopo_visibility_match_status(
                obj,
                tombstone.get("undo_visibility"),
            )
            if visibility_status == "unavailable":
                tombstone["retry_pending"] = True
                _retopo_undo_retry_pending = True
                pending = True
                _retopo_debug_emit(
                    "undo_reconcile_result",
                    {
                        "tombstone_key": key,
                        "decision": "retry",
                        "reason": "bmesh_unavailable_visibility_probe",
                        "guard_mode": guard_mode,
                    },
                )
                continue
            if visibility_status == "matched":
                restore_method = "snapshot"
                restored = _retopo_restore_snapshot_visibility(
                    info,
                    tombstone.get("snapshot"),
                )
        elif not restored and layer_status == "matched":
            # _retopo_restore_info() deliberately converts its own failures
            # to False.  Retry only when a follow-up BMesh probe confirms the
            # data became temporarily unavailable; a stable failed/mismatch
            # state is dormant and must wait for a later undo_post.
            if _retopo_tombstone_session_layer_status(obj, tombstone) == "unavailable":
                tombstone["retry_pending"] = True
                _retopo_undo_retry_pending = True
                pending = True
                _retopo_debug_emit(
                    "undo_reconcile_result",
                    {
                        "tombstone_key": key,
                        "decision": "retry",
                        "reason": "bmesh_unavailable_after_restore_info_false",
                        "guard_mode": guard_mode,
                    },
                )
                continue
        if restored:
            _retopo_clear_object_markers(obj)
            tombstone["retry_pending"] = False
            tombstone["last_restore_method"] = restore_method
            tombstone["successful_restore_count"] = int(
                tombstone.get("successful_restore_count", 0)
            ) + 1
            tombstone["guard_mode"] = (
                "layer_guard" if restore_method == "layers" else "snapshot_guard"
            )
            _retopo_debug_emit(
                "undo_reconcile_result",
                {
                    "tombstone_key": key,
                    "decision": "guarded",
                    "reason": "restore_success_guard_retained",
                    "restore_method": restore_method,
                    "guard_mode": tombstone.get("guard_mode"),
                    "successful_restore_count": tombstone.get(
                        "successful_restore_count", 0
                    ),
                    "restore_info_result": bool(restore_method == "layers"),
                    "snapshot_fallback_result": bool(restore_method == "snapshot"),
                },
            )
        else:
            _retopo_debug_emit(
                "undo_reconcile_result",
                {
                    "tombstone_key": key,
                    "decision": "dormant",
                    "reason": "stable_layer_or_visibility_mismatch",
                    "restore_method": restore_method,
                    "restore_info_result": False if restore_method == "layers" else None,
                    "snapshot_fallback_result": False if restore_method == "snapshot" else None,
                    "guard_mode": guard_mode,
                },
            )
        # Missing layers plus a nonmatching fingerprint is deliberately
        # dormant.  A later native undo may restore the ended ON state.
    _retopo_debug_emit(
        "undo_reconcile_timing",
        {
            "duration_ms": round(
                (time.perf_counter() - reconcile_started) * 1000.0,
                3,
            ),
            "pending": bool(pending),
            "tombstone_count": len(_retopo_undo_tombstones),
        },
    )
    return pending


def _retopo_restore_isolation(state):
    if not isinstance(state, _FaceSetOrbitState):
        return
    info = state.retopo_isolation
    if not info:
        return
    # Native Undo may restore the temporary BMesh layers but not the Object
    # custom properties that identify them.  Keep a runtime-only tombstone
    # before normal OFF removes those properties and layers.
    snapshot = (
        state.retopo_restore_snapshot
        or info.get("restore_snapshot")
        or _retopo_capture_restore_snapshot(info)
    )
    tombstone = _retopo_record_undo_tombstone(info, snapshot=snapshot)
    session_id = getattr(state, "session_id", None)
    retopo_object = bpy.data.objects.get(str(info.get("object_name", "")))
    restore_started = time.perf_counter()
    _retopo_debug_emit(
        "undo_off_before_restore",
        {
            "area_key": str(getattr(state, "area_key", "")),
            "state_session_id": session_id,
            "object_name": str(info.get("object_name", "")),
            "mesh_name": (
                str(retopo_object.data.name)
                if retopo_object is not None
                and getattr(retopo_object, "type", None) == "MESH"
                and getattr(retopo_object, "data", None) is not None
                else None
            ),
            "tombstone_created": bool(tombstone is not None),
            "tombstone_key": (
                [
                    str(tombstone["object_name"]),
                    str(tombstone["mesh_name"]),
                    str(tombstone["session_id"]),
                ]
                if tombstone is not None
                else None
            ),
        },
        session_id=session_id,
    )
    # Restore only elements marked at activation.  New topology is deliberately
    # left untouched so RetopoFlow edits survive FSMFO teardown.
    restore_result = _retopo_restore_info(info)
    _retopo_debug_emit(
        "undo_off_after_restore",
        {
            "restore_info_result": bool(restore_result),
            "duration_ms": round(
                (time.perf_counter() - restore_started) * 1000.0,
                3,
            ),
            "tombstone_count": len(_retopo_undo_tombstones),
            "tombstone_key": (
                [
                    str(tombstone["object_name"]),
                    str(tombstone["mesh_name"]),
                    str(tombstone["session_id"]),
                ]
                if tombstone is not None
                else None
            ),
        },
        session_id=session_id,
    )
    state.retopo_isolation = None


def _cleanup_orphan_retopo_isolations():
    """Restore tagged Retopo hide state when FSMFO Python state is gone."""
    _retopo_debug_emit(
        "tagged_retopo_cleanup_before",
        {
            "active_state_count": len(_active_states),
            "tombstone_count": len(_retopo_undo_tombstones),
        },
    )
    active_object_names = {
        state.retopo_isolation.get("object_name")
        for state in _active_states.values()
        if isinstance(state, _FaceSetOrbitState)
        and state.active
        and state.retopo_isolation
    }
    restored_names = []
    for obj in list(bpy.data.objects):
        try:
            if not obj.get(_RETOPO_ISOLATION_TAG, False):
                continue
            if obj.name in active_object_names:
                continue
            info = {
                "object": obj,
                "object_name": obj.name,
                "session_id": int(obj.get(_RETOPO_ISOLATION_SESSION, 0)),
                "layer_names": {
                    "marker": str(obj.get(_RETOPO_ISOLATION_MARKER_LAYER, "")),
                    "hide": str(obj.get(_RETOPO_ISOLATION_HIDE_LAYER, "")),
                    "select": str(obj.get(_RETOPO_ISOLATION_SELECT_LAYER, "")),
                    "target": str(obj.get(_RETOPO_ISOLATION_TARGET_LAYER, "")),
                    "vert_marker": str(
                        obj.get(_RETOPO_ISOLATION_VERT_MARKER_LAYER, "")
                    ),
                    "vert_target": str(
                        obj.get(_RETOPO_ISOLATION_VERT_TARGET_LAYER, "")
                    ),
                    "vert_origin": str(
                        obj.get(_RETOPO_ISOLATION_VERT_ORIGIN_LAYER, "")
                    ),
                    "vert_created": str(
                        obj.get(_RETOPO_ISOLATION_VERT_CREATED_LAYER, "")
                    ),
                    "vert_hide": str(
                        obj.get(_RETOPO_ISOLATION_VERT_HIDE_LAYER, "")
                    ),
                    "vert_select": str(
                        obj.get(_RETOPO_ISOLATION_VERT_SELECT_LAYER, "")
                    ),
                    "edge_marker": str(
                        obj.get(_RETOPO_ISOLATION_EDGE_MARKER_LAYER, "")
                    ),
                    "edge_hide": str(
                        obj.get(_RETOPO_ISOLATION_EDGE_HIDE_LAYER, "")
                    ),
                    "edge_select": str(
                        obj.get(_RETOPO_ISOLATION_EDGE_SELECT_LAYER, "")
                    ),
                },
                "original_records": [],
                "active_face": None,
            }
            restore_result = _retopo_restore_info(info)
            _retopo_debug_emit(
                "tagged_retopo_restore",
                {
                    "object_name": str(obj.name),
                    "restore_info_result": bool(restore_result),
                },
            )
            if restore_result:
                restored_names.append(obj.name)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
    _retopo_debug_emit(
        "tagged_retopo_cleanup_after",
        {
            "restored_names": [str(name) for name in restored_names],
            "restored_count": len(restored_names),
            "active_state_count": len(_active_states),
            "tombstone_count": len(_retopo_undo_tombstones),
        },
    )
    return restored_names


def _world_ray_from_view_center(context):
    """Return (origin, direction) for the exact center of the 3D View region."""
    region = context.region
    rv3d = context.space_data.region_3d
    center_2d = Vector((region.width * 0.5, region.height * 0.5))

    try:
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, center_2d)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, center_2d)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None

    if direction.length_squared == 0.0:
        return None
    return origin, direction.normalized()


def _object_is_visible(context, obj):
    """Filter objects using viewport visibility, not selection state."""
    try:
        if obj.hide_get(view_layer=context.view_layer):
            return False
    except (AttributeError, RuntimeError, TypeError):
        try:
            if obj.hide_get():
                return False
        except (AttributeError, RuntimeError, TypeError):
            pass

    try:
        if not obj.visible_get(
            view_layer=context.view_layer,
            viewport=context.space_data,
        ):
            return False
    except (AttributeError, RuntimeError, TypeError):
        try:
            if not obj.visible_get():
                return False
        except (AttributeError, RuntimeError, TypeError):
            pass

    # Wire and bounds display has no visible surface to use as an orbit target.
    return obj.display_type not in {"WIRE", "BOUNDS"}


def _reference_object_poll(_self, obj):
    return obj is not None and obj.type == "MESH"


def _get_reference_object(context):
    """Return only the explicitly configured Reference Object."""
    try:
        obj = context.scene.mfo_reference_object
    except (AttributeError, ReferenceError, RuntimeError):
        return None
    if obj is None or obj.type != "MESH":
        return None
    return obj if _object_is_visible(context, obj) else None


def _distance_along_ray(origin, direction, point):
    distance = (point - origin).dot(direction)
    return distance if distance > 1.0e-7 else None


def _raycast_object(obj, depsgraph, origin, direction):
    """Ray cast an evaluated object and return (world_point, distance)."""
    try:
        evaluated = obj.evaluated_get(depsgraph)
        matrix_world = evaluated.matrix_world
        inverse = matrix_world.inverted_safe()
        local_origin = inverse @ origin
        local_direction = (inverse.to_3x3() @ direction)
        if local_direction.length_squared == 0.0:
            return None
        local_direction.normalize()

        hit, local_location, _normal, _face_index = evaluated.ray_cast(
            local_origin,
            local_direction,
        )
        if not hit:
            return None

        world_location = matrix_world @ local_location
        distance = _distance_along_ray(origin, direction, world_location)
        return (world_location, distance) if distance is not None else None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _raycast_edit_object(obj, origin, direction):
    """Ray cast the live BMesh so unsaved Edit Mode changes are included."""
    try:
        bm = bmesh.from_edit_mesh(obj.data)
    except (AttributeError, RuntimeError, TypeError):
        return None

    matrix_world = obj.matrix_world
    best = None

    for face in bm.faces:
        if not face.is_valid or face.hide:
            continue

        local_vertices = [loop.vert.co.copy() for loop in face.loops]
        if len(local_vertices) < 3 or any(loop.vert.hide for loop in face.loops):
            continue

        if len(local_vertices) == 3:
            triangles = (local_vertices,)
        else:
            try:
                triangle_indices = tessellate_polygon([local_vertices])
                triangles = tuple(
                    tuple(local_vertices[index] for index in triangle)
                    for triangle in triangle_indices
                )
            except (RuntimeError, TypeError, ValueError):
                # A fan is a useful fallback for unusual or temporarily
                # invalid n-gons while the user is editing.
                triangles = tuple(
                    (local_vertices[0], local_vertices[index], local_vertices[index + 1])
                    for index in range(1, len(local_vertices) - 1)
                )

        for triangle in triangles:
            world_triangle = tuple(matrix_world @ vertex for vertex in triangle)
            hit_location = intersect_ray_tri(
                world_triangle[0],
                world_triangle[1],
                world_triangle[2],
                direction,
                origin,
                True,
            )
            if hit_location is None:
                continue

            distance = _distance_along_ray(origin, direction, hit_location)
            if distance is not None and (best is None or distance < best[1]):
                best = (hit_location.copy(), distance)

    return best


def _find_center_hit(context):
    """Find the Reference Object surface under the viewport center."""
    ray = _world_ray_from_view_center(context)
    if ray is None:
        return None
    origin, direction = ray
    obj = _get_reference_object(context)
    if obj is None:
        return None
    depsgraph = context.evaluated_depsgraph_get()
    if context.mode == "EDIT_MESH" and obj.mode == "EDIT":
        return _raycast_edit_object(obj, origin, direction)
    return _raycast_object(obj, depsgraph, origin, direction)


def _raycast_reference_face_set(context):
    """Return the center hit and Face Set ID from the configured reference."""
    obj = _get_reference_object(context)
    if obj is None:
        return None

    face_set_attr = obj.data.attributes.get(".sculpt_face_set")
    if face_set_attr is None or face_set_attr.domain != "FACE":
        return None

    ray = _world_ray_from_view_center(context)
    if ray is None:
        return None
    origin, direction = ray

    try:
        inverse = obj.matrix_world.inverted_safe()
        local_origin = inverse @ origin
        local_direction = inverse.to_3x3() @ direction
        if local_direction.length_squared == 0.0:
            return None
        local_direction.normalize()
        hit, local_location, _normal, face_index = obj.ray_cast(
            local_origin,
            local_direction,
        )
        if not hit or face_index < 0 or face_index >= len(face_set_attr.data):
            return None
        world_location = obj.matrix_world @ local_location
        distance = _distance_along_ray(origin, direction, world_location)
        if distance is None:
            return None
        return (
            obj,
            int(face_index),
            int(face_set_attr.data[face_index].value),
            world_location,
            distance,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _set_object_hidden(obj, hidden):
    """Set per-view-layer viewport visibility without changing hide_viewport."""
    if obj is None:
        return False
    try:
        obj.hide_set(bool(hidden))
        try:
            return bool(obj.hide_get()) == bool(hidden)
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False


def _is_face_set_proxy_object(obj):
    """Return whether *obj* is a proxy created by Face Set MFO.

    The name check intentionally remains as a compatibility fallback for
    proxies created by version 2.1.0, before persistent custom markers were
    added.  Newly-created proxies always receive the explicit marker.
    """
    try:
        return bool(
            obj
            and obj.type == "MESH"
            and (
                bool(obj.get(_FACE_SET_PROXY_TAG, False))
                or _FACE_SET_PROXY_NAME_TOKEN in obj.name
            )
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False


def _is_face_set_proxy_mesh(mesh):
    try:
        return bool(
            mesh
            and (
                bool(mesh.get(_FACE_SET_PROXY_MESH_TAG, False))
                or _FACE_SET_PROXY_MESH_NAME_TOKEN in mesh.name
            )
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False


def _proxy_reference_name(proxy):
    """Return the Reference Object name stored on a proxy.

    Legacy proxies did not store custom properties, so their generated name
    is also parsed as a fallback.
    """
    try:
        reference_name = proxy.get(_FACE_SET_PROXY_REFERENCE)
        if reference_name:
            return str(reference_name)
        if _FACE_SET_PROXY_NAME_TOKEN in proxy.name:
            return proxy.name.split(_FACE_SET_PROXY_NAME_TOKEN, 1)[0]
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        pass
    return ""


def _proxy_previous_hide(proxy, fallback=False):
    try:
        return bool(proxy.get(_FACE_SET_PROXY_PREVIOUS_HIDE, fallback))
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return bool(fallback)


def _get_polyquilt_qsnap():
    """Return PolyQuilt's optional QSnap class without making it a dependency."""
    try:
        from bl_ext.blender_org.PolyQuilt_Fork.QMesh.QSnap import QSnap

        return QSnap
    except (ImportError, AttributeError, RuntimeError, TypeError):
        return None


def _has_active_face_set_state():
    return any(
        isinstance(state, _FaceSetOrbitState) and state.active
        for state in _active_states.values()
    )


def _retopoflow_snap_weld_filter_enabled():
    prefs = _addon_preferences()
    return bool(
        getattr(prefs, "retopoflow_target_island_filter", False)
        if prefs is not None
        else False
    )


def _retopoflow_filter_info(context):
    """Return the active FSMFO Retopo isolation info for *context*."""
    try:
        edit_object = context.edit_object
        if edit_object is None or edit_object.mode != "EDIT":
            return None
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return None

    for state in _active_states.values():
        if not isinstance(state, _FaceSetOrbitState) or not state.active:
            continue
        info = state.retopo_isolation
        if not info:
            continue
        try:
            if info.get("object_name") == edit_object.name:
                return info
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            continue
    return None


class _RetopoFilterUnavailable(RuntimeError):
    """Raised when FSMFO membership data cannot be trusted safely."""


def _retopoflow_reject_all_filter(_vertex):
    """Reject every candidate after FSMFO membership became untrusted."""
    raise _RetopoFilterUnavailable()


def _retopoflow_abort_untrusted_filter(info):
    """End FSMFO instead of allowing an unclassifiable Retopo candidate."""
    if not info or info.get("filter_invalid"):
        return
    info["filter_invalid"] = True
    for state in list(_active_states.values()):
        try:
            if (
                isinstance(state, _FaceSetOrbitState)
                and state.retopo_isolation is info
                and state.active
            ):
                _finish_state(state)
                return
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            continue


def _retopoflow_reject_active_isolations_if_context_missing():
    """Reject a context-less hooked call without ending active FSMFO."""
    found = False
    for state in list(_active_states.values()):
        try:
            if (
                isinstance(state, _FaceSetOrbitState)
                and state.active
                and state.retopo_isolation
            ):
                found = True
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            continue
    return _retopoflow_reject_all_filter if found else None


def _retopo_debug_enabled():
    """Return the existing Debug Display preference without touching the log."""
    try:
        prefs = _addon_preferences()
        return bool(prefs and getattr(prefs, "debug_display", False))
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _retopo_debug_coordinate(coordinate):
    try:
        return [round(float(value), 9) for value in coordinate]
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _retopo_debug_element(element, include_hide=False):
    """Capture one already-known element without enumerating its BMesh."""
    if element is None:
        return None
    result = {"id": id(element)}
    try:
        result["hash"] = hash(element)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        result["hash"] = None
    try:
        result["index"] = int(element.index)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        result["index"] = None
    try:
        result["co"] = _retopo_debug_coordinate(element.co)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        result["co"] = None
    if include_hide:
        try:
            result["hide"] = bool(element.hide)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            result["hide"] = None
    return result


def _retopo_debug_bmv(nearest):
    try:
        return getattr(nearest, "bmv", None)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _retopo_debug_valid_session_id(session_id):
    try:
        return int(session_id) > 0
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _retopo_debug_new_state(
    mfo_session_id,
    phase,
    object_name=None,
    retopo_session_id=None,
):
    global _retopo_debug_file_size, _retopo_debug_file_records
    if not _retopo_debug_valid_session_id(mfo_session_id):
        return None
    try:
        if _retopo_debug_file_size is None or _retopo_debug_file_records is None:
            _retopo_debug_file_size = os.path.getsize(_RETOPO_DEBUG_LOG_PATH)
            with open(_RETOPO_DEBUG_LOG_PATH, "rb") as handle:
                _retopo_debug_file_records = sum(1 for _line in handle)
        size = int(_retopo_debug_file_size)
    except FileNotFoundError:
        size = 0
        _retopo_debug_file_size = 0
        _retopo_debug_file_records = 0
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return {
        "mfo_session_id": mfo_session_id,
        "retopo_session_id": retopo_session_id,
        "phase": phase,
        "object_name": object_name,
        "active": True,
        "sequence": 0,
        "file_size": size,
        "disabled": False,
        "counts": {bucket: 0 for bucket in _RETOPO_DEBUG_BUCKET_LIMITS},
    }


def _retopo_debug_write(
    mfo_session_id,
    event,
    payload,
    bucket="normal",
    allow_inactive=False,
):
    """Append one bounded record; never affect addon behavior."""
    global _retopo_debug_file_size, _retopo_debug_file_records
    if not _retopo_debug_valid_session_id(mfo_session_id):
        return False
    state = _retopo_debug_sessions.get(mfo_session_id)
    if state is None and allow_inactive:
        state = _retopo_debug_retired_sessions.get(mfo_session_id)
    if (
        not state
        or (not state.get("active") and not allow_inactive)
        or state.get("disabled")
    ):
        return False
    if state.get("mfo_session_id") != mfo_session_id:
        return False
    if bucket not in _RETOPO_DEBUG_BUCKET_LIMITS:
        bucket = "normal"
    counts = state.get("counts") or {}
    if counts.get(bucket, 0) >= _RETOPO_DEBUG_BUCKET_LIMITS[bucket]:
        return False
    if sum(counts.values()) >= _RETOPO_DEBUG_MAX_RECORDS:
        return False
    try:
        record = {
            "event": event,
            "version": list(bl_info.get("version", ())),
            "timestamp": time.time(),
            "sequence": int(state.get("sequence", 0)) + 1,
            # ``session_id`` is retained as the logger's stable public key,
            # and is always the MFO state session.  RetopoFlow's independent
            # isolation/session id is carried separately and never indexes
            # ``_retopo_debug_sessions``.
            "session_id": mfo_session_id,
            "mfo_session_id": mfo_session_id,
            "retopo_session_id": state.get("retopo_session_id"),
        }
        record.update(payload or {})
        # Event payloads cannot relabel a record onto the Retopo session.
        record["session_id"] = mfo_session_id
        record["mfo_session_id"] = mfo_session_id
        record["retopo_session_id"] = state.get("retopo_session_id")
        line = json.dumps(
            record,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ) + "\n"
        line_size = len(line.encode("utf-8"))
        if line_size > _RETOPO_DEBUG_MAX_BYTES:
            state["disabled"] = True
            return False
        if (
            int(_retopo_debug_file_size or 0) + line_size > _RETOPO_DEBUG_MAX_BYTES
            or int(_retopo_debug_file_records or 0) >= _RETOPO_DEBUG_MAX_RECORDS
        ):
            if os.path.exists(_RETOPO_DEBUG_LOG_PATH):
                os.replace(_RETOPO_DEBUG_LOG_PATH, _RETOPO_DEBUG_LOG_BACKUP)
            _retopo_debug_file_size = 0
            _retopo_debug_file_records = 0
        with open(_RETOPO_DEBUG_LOG_PATH, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
        state["sequence"] = int(state.get("sequence", 0)) + 1
        _retopo_debug_file_size = int(_retopo_debug_file_size or 0) + line_size
        _retopo_debug_file_records = int(_retopo_debug_file_records or 0) + 1
        state["file_size"] = _retopo_debug_file_size
        counts[bucket] = counts.get(bucket, 0) + 1
        state["counts"] = counts
        return True
    except Exception:
        state["disabled"] = True
        return False


def _retopo_debug_begin_session(
    mfo_session_id,
    phase,
    object_name=None,
    retopo_session_id=None,
):
    """Activate logging only after one MFO session is fully established."""
    global _retopo_debug_last_session_id
    if not _retopo_debug_valid_session_id(mfo_session_id):
        return False
    if not _retopo_debug_enabled():
        return False
    existing = _retopo_debug_sessions.get(mfo_session_id)
    if existing and existing.get("active") and not existing.get("disabled"):
        return True
    state = _retopo_debug_new_state(
        mfo_session_id,
        phase,
        object_name,
        retopo_session_id,
    )
    if state is None:
        return False
    _retopo_debug_last_session_id = int(mfo_session_id)
    _retopo_debug_retired_sessions.pop(int(mfo_session_id), None)
    _retopo_debug_sessions[mfo_session_id] = state
    if _retopo_debug_write(
        mfo_session_id,
        "session_start",
        {
            "schema_version": _RETOPO_DEBUG_SCHEMA_VERSION,
            "phase": phase,
            "object_name": object_name,
        },
        bucket="lifecycle",
    ):
        return True
    state["active"] = False
    _retopo_debug_sessions.pop(mfo_session_id, None)
    return False


def _retopo_debug_is_active(mfo_session_id):
    if not _retopo_debug_valid_session_id(mfo_session_id):
        return False
    state = _retopo_debug_sessions.get(mfo_session_id)
    if not state or not state.get("active") or state.get("disabled"):
        return False
    # Debug preference is sampled once by _retopo_debug_begin_session.  An
    # active map entry remains the sole session association until its end
    # record is attempted and the entry is retired.
    return True


def _retopo_debug_lifecycle(mfo_session_id, phase, object_name=None):
    if not _retopo_debug_is_active(mfo_session_id):
        return
    _retopo_debug_write(
        mfo_session_id,
        "lifecycle",
        {
            "phase": phase,
            "object_name": object_name,
        },
        bucket="lifecycle",
    )


def _retopo_debug_end_session(mfo_session_id, reason):
    global _retopo_debug_last_session_id
    if not _retopo_debug_valid_session_id(mfo_session_id):
        return
    state = _retopo_debug_sessions.get(mfo_session_id)
    if not state:
        return
    try:
        _retopo_debug_write(
            mfo_session_id,
            "session_end",
            {"reason": reason},
            bucket="lifecycle",
        )
    finally:
        state["active"] = False
        _retopo_debug_sessions.pop(mfo_session_id, None)
        _retopo_debug_last_session_id = int(mfo_session_id)
        _retopo_debug_retired_sessions[int(mfo_session_id)] = state
        while len(_retopo_debug_retired_sessions) > 8:
            oldest = next(iter(_retopo_debug_retired_sessions))
            _retopo_debug_retired_sessions.pop(oldest, None)


def _retopo_debug_exception(error):
    try:
        return {
            "type": type(error).__name__,
            "message": str(error)[:160],
        }
    except Exception:
        return {"type": "Unknown", "message": "unavailable"}


def _retopo_debug_object_identity(obj):
    if obj is None:
        return {"found": False}
    result = {"found": True, "name": None, "pointer": None, "mode": None}
    try:
        result["name"] = str(obj.name)
    except Exception as error:
        result["name_error"] = _retopo_debug_exception(error)
    try:
        result["pointer"] = int(obj.as_pointer())
    except Exception as error:
        result["pointer_error"] = _retopo_debug_exception(error)
    try:
        result["mode"] = str(obj.mode)
    except Exception as error:
        result["mode_error"] = _retopo_debug_exception(error)
    try:
        mesh = obj.data if obj.type == "MESH" else None
        result["mesh"] = {
            "found": mesh is not None,
            "name": str(mesh.name) if mesh is not None else None,
            "pointer": int(mesh.as_pointer()) if mesh is not None else None,
        }
    except Exception as error:
        result["mesh_error"] = _retopo_debug_exception(error)
    return result


def _retopo_debug_marker_summary(obj):
    if obj is None:
        return {}
    names = (
        _RETOPO_ISOLATION_TAG,
        _RETOPO_ISOLATION_SESSION,
        _RETOPO_ISOLATION_MARKER_LAYER,
        _RETOPO_ISOLATION_HIDE_LAYER,
        _RETOPO_ISOLATION_SELECT_LAYER,
        _RETOPO_ISOLATION_TARGET_LAYER,
        _RETOPO_ISOLATION_VERT_MARKER_LAYER,
        _RETOPO_ISOLATION_VERT_TARGET_LAYER,
        _RETOPO_ISOLATION_VERT_ORIGIN_LAYER,
        _RETOPO_ISOLATION_VERT_CREATED_LAYER,
        _RETOPO_ISOLATION_VERT_HIDE_LAYER,
        _RETOPO_ISOLATION_VERT_SELECT_LAYER,
        _RETOPO_ISOLATION_EDGE_MARKER_LAYER,
        _RETOPO_ISOLATION_EDGE_HIDE_LAYER,
        _RETOPO_ISOLATION_EDGE_SELECT_LAYER,
    )
    result = {}
    for name in names:
        try:
            if name in obj:
                result[name] = obj[name]
        except Exception:
            continue
    return result


def _retopo_debug_layer_summary(obj, layer_names=None, session_id=None):
    """Read-only BMesh counts; never writes layers, hide, or selection."""
    result = {
        "bmesh_available": False,
        "mode": None,
        "domains": {},
    }
    if obj is None:
        return result
    try:
        result["mode"] = str(obj.mode)
    except Exception:
        pass
    if getattr(obj, "type", None) != "MESH" or result.get("mode") != "EDIT":
        return result
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        result["bmesh_available"] = True
        names = layer_names or {}
        domain_specs = (
            ("faces", bm.faces, "marker"),
            ("verts", bm.verts, "vert_marker"),
            ("edges", bm.edges, "edge_marker"),
        )
        for domain_name, elements, marker_key in domain_specs:
            domain = elements.layers.int
            expected = []
            for key, value in names.items():
                if value and key not in expected and value not in expected:
                    expected.append(value)
            present = []
            layer_details = {}
            for name in expected:
                try:
                    layer = domain.get(name)
                    layer_details[name] = {"present": layer is not None}
                    if layer is not None:
                        present.append(name)
                except Exception as error:
                    layer_details[name] = {
                        "present": False,
                        "error": _retopo_debug_exception(error),
                    }
            marker_name = names.get(marker_key, "")
            marker_layer = domain.get(marker_name) if marker_name else None
            marker_count = 0
            marker_nonzero_count = 0
            marker_session_count = 0
            hide_count = 0
            select_count = 0
            for element in elements:
                try:
                    marker_value = int(element[marker_layer]) if marker_layer is not None else None
                    if marker_layer is not None:
                        marker_count += 1
                        if marker_value:
                            marker_nonzero_count += 1
                        if session_id is not None and marker_value == int(session_id):
                            marker_session_count += 1
                except Exception:
                    pass
                try:
                    hide_count += int(bool(element.hide))
                except Exception:
                    pass
                try:
                    select_count += int(bool(element.select))
                except Exception:
                    pass
            result["domains"][domain_name] = {
                "count": len(elements),
                "hide_count": hide_count,
                "select_count": select_count,
                "layer_names_present": present,
                "layer_details": layer_details,
                "marker_name": marker_name or None,
                "marker_value_count": marker_count,
                "marker_nonzero_count": marker_nonzero_count,
                "marker_session_match_count": marker_session_count,
            }
    except Exception as error:
        result["error"] = _retopo_debug_exception(error)
    return result


def _retopo_debug_visibility_counts(visibility):
    result = {}
    for domain_name, entries in (visibility or {}).items():
        try:
            entries = list(entries or ())
            result[domain_name] = {
                "count": len(entries),
                "hide_count": sum(int(bool(entry.get("hide", False))) for entry in entries),
                "select_count": sum(int(bool(entry.get("select", False))) for entry in entries),
            }
        except Exception as error:
            result[domain_name] = {"error": _retopo_debug_exception(error)}
    return result


def _retopo_debug_snapshot_counts(snapshot):
    result = {}
    for domain_name, entries in (snapshot or {}).items():
        try:
            result[domain_name] = len(entries or ())
        except Exception:
            result[domain_name] = None
    return result


def _retopo_debug_restore_counts(obj, snapshot):
    """Count snapshot entries and exact post-restore hide/select matches."""
    result = {}
    if obj is None or getattr(obj, "type", None) != "MESH" or not snapshot:
        return result
    try:
        if obj.mode != "EDIT":
            return {name: {"snapshot_count": len(snapshot.get(name, ())), "matched_count": 0,
                           "state_match_count": 0} for name in snapshot}
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        matched = _retopo_match_restore_snapshot(bm, snapshot)
        for domain_name, entries in snapshot.items():
            pairs = matched.get(domain_name, ())
            state_matches = sum(
                int(
                    bool(element.hide) == bool(entry.get("hide", False))
                    and bool(element.select) == bool(entry.get("select", False))
                )
                for element, entry in pairs
            )
            result[domain_name] = {
                "snapshot_count": len(entries or ()),
                "matched_count": len(pairs),
                "state_match_count": state_matches,
            }
    except Exception as error:
        result["error"] = _retopo_debug_exception(error)
    return result


def _retopo_debug_record_followup_target(tombstone=None, obj=None):
    try:
        if tombstone:
            object_name = str(tombstone.get("object_name", ""))
            mesh_name = str(tombstone.get("mesh_name", ""))
            session_id = int(tombstone.get("session_id", 0))
            layer_names = dict(tombstone.get("layer_names") or {})
        elif obj is not None:
            object_name = str(obj.name)
            mesh_name = str(obj.data.name) if obj.type == "MESH" else ""
            session_id = 0
            layer_names = {}
        else:
            return
        if object_name and mesh_name:
            _retopo_debug_followup_targets[object_name] = {
                "object_name": object_name,
                "mesh_name": mesh_name,
                "session_id": session_id,
                "layer_names": layer_names,
            }
            while len(_retopo_debug_followup_targets) > 8:
                oldest = next(iter(_retopo_debug_followup_targets))
                _retopo_debug_followup_targets.pop(oldest, None)
    except Exception:
        pass


def _retopo_debug_followup_target_snapshots():
    result = []
    for target in list(_retopo_debug_followup_targets.values()):
        try:
            obj = bpy.data.objects.get(target.get("object_name", ""))
            result.append({
                "object": _retopo_debug_object_identity(obj),
                "object_name_match": bool(obj is not None and obj.name == target.get("object_name")),
                "mesh_name_match": bool(
                    obj is not None
                    and obj.type == "MESH"
                    and obj.data is not None
                    and obj.data.name == target.get("mesh_name")
                ),
                "layer_summary": _retopo_debug_layer_summary(
                    obj,
                    target.get("layer_names"),
                    target.get("session_id"),
                ),
                "markers": _retopo_debug_marker_summary(obj),
            })
        except Exception as error:
            result.append({"error": _retopo_debug_exception(error)})
    return result


def _retopo_debug_active_states():
    result = []
    for area_key, state in list(_active_states.items()):
        try:
            info = state.retopo_isolation or {}
            result.append({
                "area_key": str(area_key),
                "session_id": int(getattr(state, "session_id", 0)),
                "active": bool(getattr(state, "active", False)),
                "proxy_object_name": getattr(state, "proxy_object_name", None),
                "retopo_object_name": info.get("object_name"),
                "retopo_session_id": info.get("session_id"),
            })
        except Exception as error:
            result.append({"area_key": str(area_key), "error": _retopo_debug_exception(error)})
    return result


def _retopo_debug_tombstones():
    result = []
    for key, tombstone in list(_retopo_undo_tombstones.items()):
        try:
            object_name = str(tombstone.get("object_name", ""))
            obj = bpy.data.objects.get(object_name)
            result.append({
                "key": [str(value) for value in key],
                "object_name": object_name,
                "mesh_name": str(tombstone.get("mesh_name", "")),
                "session_id": tombstone.get("session_id"),
                "retry_pending": bool(tombstone.get("retry_pending", False)),
                "guard_mode": str(tombstone.get("guard_mode", "armed")),
                "successful_restore_count": int(
                    tombstone.get("successful_restore_count", 0)
                ),
                "last_restore_method": tombstone.get("last_restore_method"),
                "object": _retopo_debug_object_identity(obj),
            })
        except Exception as error:
            result.append({"key": [str(value) for value in key], "error": _retopo_debug_exception(error)})
    return result


def _retopo_debug_probe_tombstone(tombstone):
    try:
        object_name = str(tombstone.get("object_name", "")) if tombstone else ""
        try:
            obj = bpy.data.objects.get(object_name) if object_name else None
        except Exception:
            obj = None
        layer_names = (tombstone or {}).get("layer_names") or {}
        object_identity = _retopo_debug_object_identity(obj)
        mesh_identity_match = False
        if obj is not None:
            try:
                mesh_identity_match = str(obj.data.name) == str(tombstone.get("mesh_name", ""))
            except Exception:
                mesh_identity_match = False
        try:
            layer_status = _retopo_tombstone_session_layer_status(obj, tombstone)
        except Exception as error:
            layer_status = "probe_exception"
            layer_error = _retopo_debug_exception(error)
        else:
            layer_error = None
        try:
            visibility_status = _retopo_visibility_match_status(
                obj,
                (tombstone or {}).get("undo_visibility"),
            )
        except Exception as error:
            visibility_status = "probe_exception"
            visibility_error = _retopo_debug_exception(error)
        else:
            visibility_error = None
        result = {
            "object_lookup": object_identity,
            "object_name_match": bool(obj is not None and object_identity.get("name") == object_name),
            "mesh_name_match": mesh_identity_match,
            "mode": object_identity.get("mode"),
            "guard_mode": str((tombstone or {}).get("guard_mode", "armed")),
            "successful_restore_count": int(
                (tombstone or {}).get("successful_restore_count", 0)
            ),
            "layer_status": layer_status,
            "visibility_status": visibility_status,
            "layer_summary": _retopo_debug_layer_summary(
                obj,
                layer_names,
                (tombstone or {}).get("session_id"),
            ),
            "markers": _retopo_debug_marker_summary(obj),
        }
        if layer_error is not None:
            result["layer_error"] = layer_error
        if visibility_error is not None:
            result["visibility_error"] = visibility_error
        return result
    except Exception as error:
        return {"probe_error": _retopo_debug_exception(error)}


def _retopo_debug_diagnostic(event, payload=None, session_id=None):
    """Write diagnostics through the existing bounded JSONL infrastructure."""
    if not _retopo_debug_enabled():
        return False
    try:
        candidate = session_id
        if not _retopo_debug_valid_session_id(candidate):
            candidate = _retopo_debug_last_session_id
        if not _retopo_debug_valid_session_id(candidate):
            return False
        candidate = int(candidate)
        state = _retopo_debug_sessions.get(candidate)
        if state is None:
            state = _retopo_debug_retired_sessions.get(candidate)
        if state is None:
            return False
        if candidate not in _retopo_debug_sessions:
            _retopo_debug_retired_sessions[candidate] = state
        return _retopo_debug_write(
            candidate,
            event,
            payload or {},
            bucket="diagnostic",
            allow_inactive=True,
        )
    except Exception:
        return False


def _retopo_debug_emit(event, payload=None, session_id=None):
    if not _retopo_debug_enabled():
        return False
    session_ids = []
    if _retopo_debug_valid_session_id(session_id):
        session_ids.append(int(session_id))
    # Keep the hot path independent from diagnostic BMesh/object scans.  The
    # session map is already the ownership source of truth for the bounded
    # JSONL logger, so iterating its keys is sufficient here.
    for value in tuple(_retopo_debug_sessions.keys()):
        if _retopo_debug_valid_session_id(value) and int(value) not in session_ids:
            session_ids.append(int(value))
    if not session_ids and _retopo_debug_valid_session_id(_retopo_debug_last_session_id):
        session_ids.append(int(_retopo_debug_last_session_id))
    written = False
    for value in session_ids:
        written = _retopo_debug_diagnostic(event, payload, session_id=value) or written
    return written


def _retopo_debug_predicate(
    mfo_session_id,
    mode,
    candidate,
    decision,
    direct_source,
    source_vertices,
    source_ambiguous,
    reached_source,
    hop_distance,
    visited_count,
    source_kind=None,
    source_count=None,
    source_summary=None,
    target_member_count=None,
    local_hit=False,
    full_fallback=False,
):
    if not _retopo_debug_is_active(mfo_session_id):
        return
    if source_count is None:
        try:
            source_count = len(source_vertices)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            source_count = 0
    if source_summary is None:
        source_summary = []
        try:
            for vertex in source_vertices:
                source_summary.append(_retopo_debug_element(vertex))
                if len(source_summary) >= 4:
                    break
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            source_summary = []
    if target_member_count is None:
        target_member_count = source_count
    priority = bool(source_ambiguous) or hop_distance == 12 or bool(full_fallback)
    _retopo_debug_write(
        mfo_session_id,
        "membership",
        {
            "mode": mode,
            "source_kind": source_kind or mode,
            "candidate": _retopo_debug_element(candidate, include_hide=True),
            "direct_before": bool(direct_source),
            "direct_before_basis": "source_bmverts",
            "decision": bool(decision),
            "direct_after": bool(direct_source),
            "cache_len_before": None,
            "cache_len_after": None,
            "cache_updated": False,
            "reached_seed": None,
            "source_count": source_count,
            "source_summary": source_summary,
            "source_ambiguous": bool(source_ambiguous),
            "reached_source": _retopo_debug_element(reached_source),
            "hop_distance": hop_distance,
            "visited_count": visited_count,
            "component_faces_valid": None,
            "target_members_count": target_member_count,
            "local_hit": bool(local_hit),
            "full_fallback": bool(full_fallback),
        },
        bucket="priority" if priority else "normal",
    )


def _retopo_debug_nearest(
    mfo_session_id,
    mode,
    nearest,
    coordinate,
    caller_filter,
    filter_selected,
    before,
    after,
):
    if not _retopo_debug_is_active(mfo_session_id):
        return
    before_nonnull = before is not None
    after_nonnull = after is not None
    priority = before_nonnull != after_nonnull
    _retopo_debug_write(
        mfo_session_id,
        "nearest_update",
        {
            "mode": mode,
            "self": id(nearest),
            "input_co": _retopo_debug_coordinate(coordinate),
            "caller_filter": caller_filter is not None,
            "filter_selected": bool(filter_selected),
            "bmv_before": _retopo_debug_element(before),
            "bmv_after": _retopo_debug_element(after),
        },
        bucket="priority" if priority else "normal",
    )


def _retopo_debug_polypen_tick(mfo_session_id, owner, nearest, result=None):
    if not _retopo_debug_is_active(mfo_session_id):
        return
    try:
        state = str(getattr(owner, "state", None))
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        state = None
    try:
        hit = _retopo_debug_coordinate(owner.hit)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        hit = None
    before = _retopo_debug_bmv(nearest)
    _retopo_debug_write(
        mfo_session_id,
        "polypen_tick",
        {
            "mode": "polypen",
            "owner": id(owner),
            "state": state,
            "result": result,
            "source": _retopo_debug_element(_retopo_debug_bmv(owner)),
            "nearest_bmv": _retopo_debug_element(before),
            "hit": hit,
        },
    )


def _retopo_debug_release(mfo_session_id, operator, nearest, bmv_before, result):
    if not _retopo_debug_is_active(mfo_session_id):
        return
    bmv_after = _retopo_debug_bmv(nearest)
    _retopo_debug_write(
        mfo_session_id,
        "automerge_release",
        {
            "mode": "automerge",
            "owner": id(operator),
            "result": result,
            "nearest": id(nearest) if nearest is not None else None,
            "bmv_before": _retopo_debug_element(bmv_before),
            "bmv_after": _retopo_debug_element(bmv_after),
        },
        bucket="release",
    )


def _retopoflow_valid_element(element):
    try:
        return bool(element.is_valid)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _retopoflow_refresh_target_members(info):
    """Drop dead wrappers and rebind members after a live BMesh rebuild.

    Normal preview calls retain the session set and do not traverse the
    BMesh.  A changed BMesh token is the explicit exception: the existing
    activation vertex-target layer is used once to bind current wrappers, so
    Undo/rebuild cannot revive a deleted identity or keep an old wrapper as a
    positive membership witness.
    """
    if not isinstance(info, dict):
        return False
    target_members = info.get("target_members")
    if not isinstance(target_members, set):
        return False
    try:
        live_members = set()
        for member in tuple(target_members):
            if _retopoflow_valid_element(member):
                try:
                    live_members.add(member)
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    pass
        target_members.clear()
        target_members.update(live_members)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False

    object_name = str(info.get("object_name", ""))
    if not object_name:
        return False
    try:
        retopo_object = bpy.data.objects.get(object_name)
        if (
            retopo_object is None
            or retopo_object.type != "MESH"
            or retopo_object.mode != "EDIT"
        ):
            return False
        bm = bmesh.from_edit_mesh(retopo_object.data)
        bm.verts.ensure_lookup_table()
        current_token = _retopo_element_token(bm)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    if current_token is None:
        return False
    previous_token = info.get("target_members_bmesh_token")
    if previous_token == current_token and live_members:
        return True

    names = info.get("layer_names") or {}
    try:
        target_layer = bm.verts.layers.int.get(names.get("vert_target", ""))
        marker_layer = bm.verts.layers.int.get(names.get("vert_marker", ""))
        session_id = int(info.get("session_id", 0))
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    if target_layer is None:
        return False

    rebound = set()
    try:
        for vertex in bm.verts:
            if not _retopoflow_valid_element(vertex):
                continue
            if marker_layer is not None and int(vertex[marker_layer]) != session_id:
                continue
            if int(vertex[target_layer]) == 1:
                rebound.add(vertex)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    if not rebound:
        return False
    target_members.clear()
    target_members.update(rebound)
    info["component_vertices"] = list(rebound)
    info["target_members_bmesh_token"] = current_token
    return True


def _retopoflow_append_source_vertex(vertices, seen, vertex):
    if not _retopoflow_valid_element(vertex):
        return False
    identity = id(vertex)
    if identity in seen:
        return True
    seen.add(identity)
    vertices.append(vertex)
    return True


def _retopoflow_polypen_source(owner):
    """Resolve only PP_Logic's live selected editing source, lazily."""
    vertices = []
    seen = set()
    invalid = False
    try:
        selected = getattr(owner, "selected")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return {
            "vertices": (),
            "ambiguous": True,
            "kind": "polypen_selected",
        }
    if selected is None or not hasattr(selected, "get"):
        return {
            "vertices": (),
            "ambiguous": True,
            "kind": "polypen_selected",
        }

    def add_values(key, include_vertices=False):
        nonlocal invalid
        try:
            values = selected.get(key, ()) or ()
            values = tuple(values)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            invalid = True
            return
        for element in values:
            if not _retopoflow_valid_element(element):
                invalid = True
                continue
            if include_vertices:
                try:
                    element_vertices = tuple(element.verts)
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    invalid = True
                    continue
                for vertex in element_vertices:
                    if not _retopoflow_append_source_vertex(vertices, seen, vertex):
                        invalid = True
            elif not _retopoflow_append_source_vertex(vertices, seen, element):
                invalid = True

    add_values(bmesh.types.BMVert)
    add_values(bmesh.types.BMEdge, include_vertices=True)
    add_values(bmesh.types.BMFace, include_vertices=True)
    # RetopoFlow has already established this as the live edit source.  Do
    # not add a second bounded connectivity witness here: a valid source
    # group may span more than twelve edges, and session membership is
    # intentionally monotonic until FSMFO ends.
    ambiguous = invalid or not vertices
    return {
        "vertices": tuple(vertices),
        "ambiguous": ambiguous,
        "kind": "polypen_selected",
    }


def _retopoflow_translate_source(operator):
    """Resolve RFOperator_Translate.bmvs without reading its nearest helper."""
    vertices = []
    seen = set()
    invalid = False
    try:
        values = getattr(operator, "bmvs")
        values = tuple(values)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return {
            "vertices": (),
            "ambiguous": True,
            "kind": "translate_bmvs",
        }
    for vertex in values:
        if not _retopoflow_append_source_vertex(vertices, seen, vertex):
            invalid = True
    ambiguous = invalid or not vertices
    return {
        "vertices": tuple(vertices),
        "ambiguous": ambiguous,
        "kind": "translate_bmvs",
    }


def _retopoflow_target_island_filter(context, source_provider=None, mode=None):
    """Return the shared source-first, two-stage connectivity predicate.

    RetopoFlow's live editing source is resolved lazily, after the caller has
    established its current selection.  The normal path walks only from the
    candidate to those source vertices, up to twelve edge hops.  It never
    scans the session member set.  Only when that local source walk misses do
    we continue the same queue through the candidate's complete live
    BMVert/BMEdge component and test that component against the confirmed
    activation/source members.  Candidates and BFS-visited vertices are not
    promoted; only the live source group is recorded as confirmed membership.
    """
    info = _retopoflow_filter_info(context)
    if info is None:
        return _retopoflow_reject_active_isolations_if_context_missing()
    # RetopoFlow's isolation/session id is intentionally separate from the
    # MFO state session id that owns the logger.  Only the latter can activate
    # diagnostic I/O or index the active logger map.
    mfo_session_id = info.get("mfo_session_id")
    retopo_session_id = info.get("session_id")
    diag_enabled = _retopo_debug_is_active(mfo_session_id)
    if info.get("filter_invalid"):
        return _retopoflow_reject_all_filter

    target_members = info.get("target_members")
    if not isinstance(target_members, set):
        return _retopoflow_reject_all_filter
    if not _retopoflow_refresh_target_members(info):
        return _retopoflow_reject_all_filter

    try:
        edit_object = context.edit_object
        if edit_object is None or edit_object.mode != "EDIT":
            raise _RetopoFilterUnavailable()
        if str(info.get("object_name", "")) != str(edit_object.name):
            raise _RetopoFilterUnavailable()

        operation_mode = mode or "unknown"
        source_resolved = False
        source_vertices = ()
        source_set = set()
        source_ambiguous = True
        source_kind = "source_unresolved"

        def resolve_source():
            nonlocal source_resolved
            nonlocal source_vertices
            nonlocal source_set
            nonlocal source_ambiguous
            nonlocal source_kind
            if source_resolved:
                return
            source_resolved = True
            payload = None
            try:
                payload = source_provider() if callable(source_provider) else None
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                source_kind = str(payload.get("kind") or operation_mode)
                source_ambiguous = bool(payload.get("ambiguous", False))
                payload = payload.get("vertices", ())
            else:
                source_kind = operation_mode
            try:
                values = tuple(payload or ())
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                values = ()
                source_ambiguous = True
            valid_sources = []
            seen_sources = set()
            for vertex in values:
                if not _retopoflow_valid_element(vertex):
                    source_ambiguous = True
                    continue
                try:
                    if vertex in seen_sources:
                        continue
                    seen_sources.add(vertex)
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    source_ambiguous = True
                    continue
                valid_sources.append(vertex)
            if not valid_sources:
                source_ambiguous = True
            if source_ambiguous:
                source_vertices = tuple(valid_sources)
                source_set = set()
                return
            source_vertices = tuple(valid_sources)
            try:
                source_set = set(source_vertices)
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                source_set = set()
                source_ambiguous = True
                return
            # A source is an already visible RetopoFlow edit member.  Record
            # that confirmed fact once for later slow fallbacks; do not add
            # the candidate or any source-walk intermediates here.
            for vertex in source_vertices:
                try:
                    target_members.add(vertex)
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    source_ambiguous = True
                    return

        def _member_hit(vertex):
            try:
                return vertex in target_members
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                return False

        def connected_to_target(candidate):
            resolve_source()
            source_count = len(source_vertices)
            if not _retopoflow_valid_element(candidate):
                if diag_enabled:
                    _retopo_debug_predicate(
                        mfo_session_id,
                        operation_mode,
                        candidate,
                        False,
                        False,
                        source_vertices,
                        source_ambiguous,
                        None,
                        None,
                        0,
                        source_kind,
                        source_count=source_count,
                        target_member_count=len(target_members),
                        local_hit=False,
                        full_fallback=False,
                    )
                return False

            if source_ambiguous or not source_set:
                if diag_enabled:
                    _retopo_debug_predicate(
                        mfo_session_id,
                        operation_mode,
                        candidate,
                        False,
                        False,
                        source_vertices,
                        True,
                        None,
                        None,
                        0,
                        source_kind,
                        source_count=source_count,
                        target_member_count=len(target_members),
                        local_hit=False,
                        full_fallback=False,
                    )
                return False

            try:
                direct_source = candidate in source_set
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                direct_source = False
            if direct_source:
                decision = True
                reached_source = candidate
                hop_distance = 0
                visited_count = 1
                local_hit = True
                full_fallback = False
            else:
                decision = False
                reached_source = None
                hop_distance = None
                local_hit = False
                full_fallback = False
                try:
                    visited = {candidate: 0}
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    visited = {}
                if not visited:
                    if diag_enabled:
                        _retopo_debug_predicate(
                            mfo_session_id,
                            operation_mode,
                            candidate,
                            False,
                            False,
                            source_vertices,
                            True,
                            None,
                            None,
                            0,
                            source_kind,
                            source_count=source_count,
                            target_member_count=len(target_members),
                            local_hit=False,
                            full_fallback=False,
                        )
                    return False
                pending = deque([(candidate, 0)])
                # Fast path: BFS in nondecreasing distance order, stopping at
                # the first depth-12 frontier.  The frontier remains in the
                # same deque for the slow fallback below.
                while pending and pending[0][1] < 12:
                    vertex, depth = pending.popleft()
                    try:
                        edges = tuple(vertex.link_edges)
                    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                        continue
                    for edge in edges:
                        if not _retopoflow_valid_element(edge):
                            continue
                        try:
                            edge_vertices = tuple(edge.verts)
                        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                            continue
                        for linked in edge_vertices:
                            if linked is vertex or not _retopoflow_valid_element(linked):
                                continue
                            next_depth = depth + 1
                            try:
                                if linked in visited:
                                    continue
                                visited[linked] = next_depth
                            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                                continue
                            try:
                                linked_is_source = linked in source_set
                            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                                linked_is_source = False
                            if linked_is_source:
                                decision = True
                                reached_source = linked
                                hop_distance = next_depth
                                local_hit = True
                                pending.clear()
                                break
                            if next_depth >= 12:
                                # Preserve the frontier for full fallback;
                                # no further fast-path expansion is allowed.
                                pending.append((linked, next_depth))
                                continue
                            pending.append((linked, next_depth))
                        if decision:
                            break
                    if decision:
                        break

                if not decision:
                    full_fallback = True
                    # First inspect the already visited local ball.  This is
                    # the first point where session-membership lookup is
                    # permitted; the normal source fast path never performs
                    # this lookup.
                    for vertex, depth in tuple(visited.items()):
                        if _member_hit(vertex):
                            decision = True
                            reached_source = vertex
                            hop_distance = depth
                            break
                    while not decision and pending:
                        vertex, depth = pending.popleft()
                        try:
                            edges = tuple(vertex.link_edges)
                        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                            continue
                        for edge in edges:
                            if not _retopoflow_valid_element(edge):
                                continue
                            try:
                                edge_vertices = tuple(edge.verts)
                            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                                continue
                            for linked in edge_vertices:
                                if linked is vertex or not _retopoflow_valid_element(linked):
                                    continue
                                next_depth = depth + 1
                                try:
                                    if linked in visited:
                                        continue
                                    visited[linked] = next_depth
                                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                                    continue
                                if _member_hit(linked):
                                    decision = True
                                    reached_source = linked
                                    hop_distance = next_depth
                                    break
                                pending.append((linked, next_depth))
                            if decision:
                                break
                        if decision:
                            break
                visited_count = len(visited)
            if diag_enabled:
                _retopo_debug_predicate(
                    mfo_session_id,
                    operation_mode,
                    candidate,
                    decision,
                    direct_source,
                    source_vertices,
                    source_ambiguous,
                    reached_source,
                    hop_distance,
                    visited_count,
                    source_kind,
                    source_count=source_count,
                    target_member_count=len(target_members),
                    local_hit=local_hit,
                    full_fallback=full_fallback,
                )
            return decision

        if diag_enabled:
            try:
                setattr(
                    connected_to_target,
                    "mesh_focus_orbit_mfo_session_id",
                    mfo_session_id,
                )
                setattr(
                    connected_to_target,
                    "mesh_focus_orbit_retopo_session_id",
                    retopo_session_id,
                )
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                pass
        return connected_to_target
    except _RetopoFilterUnavailable:
        return _retopoflow_reject_all_filter
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return _retopoflow_reject_all_filter
def _get_retopoflow_nearest_bmvert_class():
    """Return RetopoFlow's nearest-vertex helper when RetopoFlow is present."""
    try:
        from bl_ext.superhivemarket_com.retopoflow.retopoflow.common.bmesh import (
            NearestBMVert,
        )

        return NearestBMVert
    except (ImportError, AttributeError, RuntimeError, TypeError):
        return None


def _get_retopoflow_translate_class():
    """Return RetopoFlow's translate operator class when available."""
    try:
        from bl_ext.superhivemarket_com.retopoflow.retopoflow.rfoperators.transform import (
            RFOperator_Translate,
        )

        return RFOperator_Translate
    except (ImportError, AttributeError, RuntimeError, TypeError):
        return None


def _get_retopoflow_polypen_logic_class():
    """Return RetopoFlow's PolyPen logic class when RetopoFlow is present."""
    try:
        from bl_ext.superhivemarket_com.retopoflow.retopoflow.rftool_polypen.polypen_logic import (
            PP_Logic,
        )

        return PP_Logic
    except (ImportError, AttributeError, RuntimeError, TypeError):
        return None


def _restore_retopoflow_hooks():
    """Restore only MFO-owned, narrowly-scoped RetopoFlow wrappers."""
    global _retopoflow_nearest_filter_class
    global _retopoflow_nearest_filter_original_update
    global _retopoflow_nearest_filter_installed
    global _retopoflow_automerge_class
    global _retopoflow_automerge_original
    global _retopoflow_automerge_installed
    global _retopoflow_automerge_nearest_bmvert
    global _retopoflow_automerge_filter
    global _retopoflow_automerge_session_id
    global _retopoflow_automerge_retopo_session_id
    global _retopoflow_polypen_class
    global _retopoflow_polypen_original_update
    global _retopoflow_polypen_installed
    global _retopoflow_polypen_owner
    global _retopoflow_polypen_nearest_bmvert
    global _retopoflow_polypen_filter
    global _retopoflow_polypen_session_id
    global _retopoflow_polypen_retopo_session_id
    global _retopoflow_translate_preview_class
    global _retopoflow_translate_preview_original
    global _retopoflow_translate_preview_installed
    global _retopoflow_translate_preview_owner
    global _retopoflow_translate_preview_nearest_bmvert
    global _retopoflow_translate_preview_filter
    global _retopoflow_translate_preview_session_id
    global _retopoflow_translate_preview_retopo_session_id

    translate_classes = []
    if _retopoflow_automerge_class is not None:
        translate_classes.append(_retopoflow_automerge_class)
    if (
        _retopoflow_translate_preview_class is not None
        and _retopoflow_translate_preview_class not in translate_classes
    ):
        translate_classes.append(_retopoflow_translate_preview_class)
    current_translate_class = _get_retopoflow_translate_class()
    if current_translate_class is not None and current_translate_class not in translate_classes:
        translate_classes.append(current_translate_class)
    for translate_class in translate_classes:
        try:
            current = translate_class.translate
            if getattr(current, _RETOPOFLOW_TRANSLATE_PREVIEW_TAG, False):
                original = getattr(
                    current,
                    _RETOPOFLOW_TRANSLATE_PREVIEW_ORIGINAL,
                    None,
                )
                if original is not None:
                    translate_class.translate = original
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass
        try:
            current = translate_class.automerge
            if getattr(current, _RETOPOFLOW_AUTOMERGE_TAG, False):
                original = getattr(current, _RETOPOFLOW_AUTOMERGE_ORIGINAL, None)
                if original is not None:
                    translate_class.automerge = original
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass

    polypen_classes = []
    if _retopoflow_polypen_class is not None:
        polypen_classes.append(_retopoflow_polypen_class)
    current_polypen_class = _get_retopoflow_polypen_logic_class()
    if current_polypen_class is not None and current_polypen_class not in polypen_classes:
        polypen_classes.append(current_polypen_class)
    for polypen_class in polypen_classes:
        try:
            current = polypen_class.update
            if getattr(current, _RETOPOFLOW_POLYPEN_TAG, False):
                original = getattr(current, _RETOPOFLOW_POLYPEN_ORIGINAL, None)
                if original is not None:
                    polypen_class.update = original
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass

    nearest_classes = []
    if _retopoflow_nearest_filter_class is not None:
        nearest_classes.append(_retopoflow_nearest_filter_class)
    current_nearest_class = _get_retopoflow_nearest_bmvert_class()
    if current_nearest_class is not None and current_nearest_class not in nearest_classes:
        nearest_classes.append(current_nearest_class)
    for nearest_class in nearest_classes:
        try:
            current = nearest_class.update
            original = None
            if getattr(current, _RETOPOFLOW_NEAREST_FILTER_TAG, False):
                original = getattr(current, _RETOPOFLOW_NEAREST_FILTER_ORIGINAL, None)
            # Remove wrappers from the previous pre-membership implementation
            # as well, so an add-on reload cannot leave that hook active.
            if original is None and getattr(
                current,
                "mesh_focus_orbit.retopoflow_hidden_vertex_filter",
                False,
            ):
                original = getattr(
                    current,
                    "mesh_focus_orbit.retopoflow_original_nearest_bmvert_update",
                    None,
                )
            if original is not None:
                nearest_class.update = original
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass

    _retopoflow_nearest_filter_class = None
    _retopoflow_nearest_filter_original_update = None
    _retopoflow_nearest_filter_installed = False
    _retopoflow_automerge_class = None
    _retopoflow_automerge_original = None
    _retopoflow_automerge_installed = False
    _retopoflow_automerge_nearest_bmvert = None
    _retopoflow_automerge_filter = None
    _retopoflow_automerge_session_id = None
    _retopoflow_automerge_retopo_session_id = None
    _retopoflow_polypen_class = None
    _retopoflow_polypen_original_update = None
    _retopoflow_polypen_installed = False
    _retopoflow_polypen_owner = None
    _retopoflow_polypen_nearest_bmvert = None
    _retopoflow_polypen_filter = None
    _retopoflow_polypen_session_id = None
    _retopoflow_polypen_retopo_session_id = None
    _retopoflow_translate_preview_class = None
    _retopoflow_translate_preview_original = None
    _retopoflow_translate_preview_installed = False
    _retopoflow_translate_preview_owner = None
    _retopoflow_translate_preview_nearest_bmvert = None
    _retopoflow_translate_preview_filter = None
    _retopoflow_translate_preview_session_id = None
    _retopoflow_translate_preview_retopo_session_id = None


def _install_retopoflow_automerge_hook():
    """Scope membership filtering to Translate.automerge only."""
    global _retopoflow_automerge_class
    global _retopoflow_automerge_original
    global _retopoflow_automerge_installed

    translate_class = _get_retopoflow_translate_class()
    if translate_class is None:
        return False
    try:
        original_automerge = translate_class.automerge
        if getattr(original_automerge, _RETOPOFLOW_AUTOMERGE_TAG, False):
            original_automerge = getattr(
                original_automerge,
                _RETOPOFLOW_AUTOMERGE_ORIGINAL,
                None,
            )
            if original_automerge is None:
                return False
            translate_class.automerge = original_automerge

        def filtered_automerge(self, context, event, *args, **kwargs):
            global _retopoflow_automerge_nearest_bmvert
            global _retopoflow_automerge_filter
            global _retopoflow_automerge_session_id
            global _retopoflow_automerge_retopo_session_id
            previous_nearest = _retopoflow_automerge_nearest_bmvert
            previous_filter = _retopoflow_automerge_filter
            previous_session_id = _retopoflow_automerge_session_id
            previous_retopo_session_id = _retopoflow_automerge_retopo_session_id
            try:
                nearest = getattr(self, "nearest_bmv", None)
                _retopoflow_automerge_nearest_bmvert = nearest
                _retopoflow_automerge_filter = _retopoflow_target_island_filter(
                    context,
                    source_provider=lambda: _retopoflow_translate_source(self),
                    mode="automerge",
                )
                _retopoflow_automerge_session_id = (
                    getattr(
                        _retopoflow_automerge_filter,
                        "mesh_focus_orbit_mfo_session_id",
                        None,
                    )
                )
                _retopoflow_automerge_retopo_session_id = (
                    getattr(
                        _retopoflow_automerge_filter,
                        "mesh_focus_orbit_retopo_session_id",
                        None,
                    )
                )
                bmv_before = (
                    _retopo_debug_bmv(nearest)
                    if _retopo_debug_is_active(_retopoflow_automerge_session_id)
                    else None
                )
                result = original_automerge(
                    self,
                    context,
                    event,
                    *args,
                    **kwargs,
                )
                if _retopo_debug_is_active(_retopoflow_automerge_session_id):
                    _retopo_debug_release(
                        _retopoflow_automerge_session_id,
                        self,
                        nearest,
                        bmv_before,
                        result,
                    )
                return result
            finally:
                _retopoflow_automerge_nearest_bmvert = previous_nearest
                _retopoflow_automerge_filter = previous_filter
                _retopoflow_automerge_session_id = previous_session_id
                _retopoflow_automerge_retopo_session_id = previous_retopo_session_id

        setattr(filtered_automerge, _RETOPOFLOW_AUTOMERGE_TAG, True)
        setattr(filtered_automerge, _RETOPOFLOW_AUTOMERGE_ORIGINAL, original_automerge)
        translate_class.automerge = filtered_automerge
        _retopoflow_automerge_class = translate_class
        _retopoflow_automerge_original = original_automerge
        _retopoflow_automerge_installed = True
        return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _restore_retopoflow_hooks()
        return False


def _install_retopoflow_translate_preview_hook():
    """Scope membership filtering to RFOperator_Translate.translate only."""
    global _retopoflow_translate_preview_class
    global _retopoflow_translate_preview_original
    global _retopoflow_translate_preview_installed

    translate_class = _get_retopoflow_translate_class()
    if translate_class is None:
        return False
    try:
        original_translate = translate_class.translate
        if getattr(original_translate, _RETOPOFLOW_TRANSLATE_PREVIEW_TAG, False):
            original_translate = getattr(
                original_translate,
                _RETOPOFLOW_TRANSLATE_PREVIEW_ORIGINAL,
                None,
            )
            if original_translate is None:
                return False
            translate_class.translate = original_translate

        def filtered_translate(self, context, event, *args, **kwargs):
            global _retopoflow_translate_preview_owner
            global _retopoflow_translate_preview_nearest_bmvert
            global _retopoflow_translate_preview_filter
            global _retopoflow_translate_preview_session_id
            global _retopoflow_translate_preview_retopo_session_id
            previous_owner = _retopoflow_translate_preview_owner
            previous_nearest = _retopoflow_translate_preview_nearest_bmvert
            previous_filter = _retopoflow_translate_preview_filter
            previous_session_id = _retopoflow_translate_preview_session_id
            previous_retopo_session_id = (
                _retopoflow_translate_preview_retopo_session_id
            )
            try:
                nearest = getattr(self, "nearest_bmv", None)
                mfo_filter = (
                    _retopoflow_target_island_filter(
                        context,
                        source_provider=lambda: _retopoflow_translate_source(self),
                        mode="translate_preview",
                    )
                    if context is not None
                    else _retopoflow_reject_active_isolations_if_context_missing()
                )
                if mfo_filter is None:
                    return original_translate(self, context, event, *args, **kwargs)
                _retopoflow_translate_preview_owner = self
                _retopoflow_translate_preview_nearest_bmvert = nearest
                _retopoflow_translate_preview_filter = mfo_filter
                _retopoflow_translate_preview_session_id = getattr(
                    mfo_filter,
                    "mesh_focus_orbit_mfo_session_id",
                    None,
                )
                _retopoflow_translate_preview_retopo_session_id = getattr(
                    mfo_filter,
                    "mesh_focus_orbit_retopo_session_id",
                    None,
                )
                return original_translate(self, context, event, *args, **kwargs)
            finally:
                _retopoflow_translate_preview_owner = previous_owner
                _retopoflow_translate_preview_nearest_bmvert = previous_nearest
                _retopoflow_translate_preview_filter = previous_filter
                _retopoflow_translate_preview_session_id = previous_session_id
                _retopoflow_translate_preview_retopo_session_id = (
                    previous_retopo_session_id
                )

        setattr(filtered_translate, _RETOPOFLOW_TRANSLATE_PREVIEW_TAG, True)
        setattr(
            filtered_translate,
            _RETOPOFLOW_TRANSLATE_PREVIEW_ORIGINAL,
            original_translate,
        )
        translate_class.translate = filtered_translate
        _retopoflow_translate_preview_class = translate_class
        _retopoflow_translate_preview_original = original_translate
        _retopoflow_translate_preview_installed = True
        return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _restore_retopoflow_hooks()
        return False


def _install_retopoflow_polypen_hook():
    """Scope membership filtering to PolyPen's own nearest BMVert instance."""
    global _retopoflow_polypen_class
    global _retopoflow_polypen_original_update
    global _retopoflow_polypen_installed

    polypen_class = _get_retopoflow_polypen_logic_class()
    if polypen_class is None:
        return False
    try:
        original_update = polypen_class.update
        if getattr(original_update, _RETOPOFLOW_POLYPEN_TAG, False):
            original_update = getattr(
                original_update,
                _RETOPOFLOW_POLYPEN_ORIGINAL,
                None,
            )
            if original_update is None:
                return False
            polypen_class.update = original_update

        def filtered_polypen_update(self, *args, **kwargs):
            global _retopoflow_polypen_owner
            global _retopoflow_polypen_nearest_bmvert
            global _retopoflow_polypen_filter
            global _retopoflow_polypen_session_id
            global _retopoflow_polypen_retopo_session_id
            context = kwargs.get("context")
            if context is None and args:
                context = args[0]
            previous_nearest = _retopoflow_polypen_nearest_bmvert
            previous_filter = _retopoflow_polypen_filter
            previous_session_id = _retopoflow_polypen_session_id
            previous_retopo_session_id = _retopoflow_polypen_retopo_session_id
            previous_owner = _retopoflow_polypen_owner
            try:
                nearest = getattr(self, "nearest", None)
                mfo_filter = (
                    _retopoflow_target_island_filter(
                        context,
                        source_provider=lambda: _retopoflow_polypen_source(self),
                        mode="polypen",
                    )
                    if context is not None
                    else _retopoflow_reject_active_isolations_if_context_missing()
                )
                _retopoflow_polypen_owner = self
                _retopoflow_polypen_nearest_bmvert = nearest
                _retopoflow_polypen_filter = mfo_filter
                _retopoflow_polypen_session_id = (
                    getattr(
                        mfo_filter,
                        "mesh_focus_orbit_mfo_session_id",
                        None,
                    )
                )
                _retopoflow_polypen_retopo_session_id = (
                    getattr(
                        mfo_filter,
                        "mesh_focus_orbit_retopo_session_id",
                        None,
                    )
                )
                result = original_update(self, *args, **kwargs)
                post_nearest = getattr(self, "nearest", None)
                if _retopo_debug_is_active(_retopoflow_polypen_session_id):
                    _retopo_debug_polypen_tick(
                        _retopoflow_polypen_session_id,
                        self,
                        post_nearest,
                        result,
                    )
                return result
            finally:
                _retopoflow_polypen_owner = previous_owner
                _retopoflow_polypen_nearest_bmvert = previous_nearest
                _retopoflow_polypen_filter = previous_filter
                _retopoflow_polypen_session_id = previous_session_id
                _retopoflow_polypen_retopo_session_id = previous_retopo_session_id

        setattr(filtered_polypen_update, _RETOPOFLOW_POLYPEN_TAG, True)
        setattr(filtered_polypen_update, _RETOPOFLOW_POLYPEN_ORIGINAL, original_update)
        polypen_class.update = filtered_polypen_update
        _retopoflow_polypen_class = polypen_class
        _retopoflow_polypen_original_update = original_update
        _retopoflow_polypen_installed = True
        return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _restore_retopoflow_hooks()
        return False


def _install_retopoflow_nearest_filter_hook():
    """Add membership only to active PolyPen/Translate/Auto Merge calls."""
    global _retopoflow_nearest_filter_class
    global _retopoflow_nearest_filter_original_update
    global _retopoflow_nearest_filter_installed

    nearest_class = _get_retopoflow_nearest_bmvert_class()
    if nearest_class is None:
        return False
    try:
        original_update = nearest_class.update
        if getattr(original_update, _RETOPOFLOW_NEAREST_FILTER_TAG, False):
            original_update = getattr(
                original_update,
                _RETOPOFLOW_NEAREST_FILTER_ORIGINAL,
                None,
            )
            if original_update is None:
                return False
            nearest_class.update = original_update
        elif getattr(
            original_update,
            "mesh_focus_orbit.retopoflow_hidden_vertex_filter",
            False,
        ):
            original_update = getattr(
                original_update,
                "mesh_focus_orbit.retopoflow_original_nearest_bmvert_update",
                None,
            )
            if original_update is None:
                return False
            nearest_class.update = original_update
        import inspect

        update_signature = inspect.signature(original_update)
        if "filter_fn" not in update_signature.parameters:
            return False

        def filtered_update(self, context, co, *args, **kwargs):
            try:
                bound = update_signature.bind(
                    self,
                    context,
                    co,
                    *args,
                    **kwargs,
                )
            except TypeError:
                return original_update(self, context, co, *args, **kwargs)

            caller_filter = bound.arguments.get("filter_fn")
            mode = None
            mfo_filter = None
            if (
                self is _retopoflow_polypen_nearest_bmvert
                or (
                    _retopoflow_polypen_owner is not None
                    and getattr(_retopoflow_polypen_owner, "nearest", None)
                    is self
                )
            ):
                mode = "polypen"
                mfo_filter = _retopoflow_polypen_filter
            elif self is _retopoflow_automerge_nearest_bmvert:
                mode = "automerge"
                mfo_filter = _retopoflow_automerge_filter
            elif (
                self is _retopoflow_translate_preview_nearest_bmvert
                or (
                    _retopoflow_translate_preview_owner is not None
                    and getattr(
                        _retopoflow_translate_preview_owner,
                        "nearest_bmv",
                        None,
                    )
                    is self
                )
            ):
                mode = "translate_preview"
                mfo_filter = _retopoflow_translate_preview_filter
            filter_selected = bool(bound.arguments.get("filter_selected", True))
            if mode is None or mfo_filter is None:
                if _active_states and _retopo_debug_sessions:
                    info = _retopoflow_filter_info(context)
                    mfo_session_id = (
                        info.get("mfo_session_id") if info is not None else None
                    )
                    if _retopo_debug_is_active(mfo_session_id):
                        before = _retopo_debug_bmv(self)
                        result = original_update(*bound.args, **bound.kwargs)
                        after = _retopo_debug_bmv(self)
                        _retopo_debug_nearest(
                            mfo_session_id,
                            "other",
                            self,
                            co,
                            caller_filter,
                            filter_selected,
                            before,
                            after,
                        )
                        return result
                return original_update(*bound.args, **bound.kwargs)

            mfo_session_id = (
                _retopoflow_polypen_session_id
                if mode == "polypen"
                else (
                    _retopoflow_automerge_session_id
                    if mode == "automerge"
                    else _retopoflow_translate_preview_session_id
                )
            )
            debug_enabled = _retopo_debug_is_active(mfo_session_id)
            before = _retopo_debug_bmv(self) if debug_enabled else None

            def combined_filter(vertex):
                try:
                    if caller_filter is not None and not caller_filter(vertex):
                        return False
                    # When the native call supplied no filter_fn, preserve the
                    # exclusive filter_selected branch that RF would have
                    # taken.  PolyPen's own filter_fn path stays exactly as RF
                    # provided it.
                    if caller_filter is None and filter_selected and bool(vertex.select):
                        return False
                    return bool(mfo_filter(vertex))
                except _RetopoFilterUnavailable:
                    raise
                except (
                    AttributeError,
                    ReferenceError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise _RetopoFilterUnavailable() from exc

            bound.arguments["filter_fn"] = combined_filter
            try:
                result = original_update(*bound.args, **bound.kwargs)
            except _RetopoFilterUnavailable:
                # The membership session was no longer trustworthy.  Never
                # retry the native call without a filter: that would turn a
                # fail-closed membership failure into a fail-open weld.
                try:
                    self.bmv = None
                except (AttributeError, ReferenceError, RuntimeError, TypeError):
                    pass
                result = None
            if debug_enabled:
                _retopo_debug_nearest(
                    mfo_session_id,
                    mode,
                    self,
                    co,
                    caller_filter,
                    filter_selected,
                    before,
                    _retopo_debug_bmv(self),
                )
            return result

        setattr(filtered_update, _RETOPOFLOW_NEAREST_FILTER_TAG, True)
        setattr(filtered_update, _RETOPOFLOW_NEAREST_FILTER_ORIGINAL, original_update)
        nearest_class.update = filtered_update
        _retopoflow_nearest_filter_class = nearest_class
        _retopoflow_nearest_filter_original_update = original_update
        _retopoflow_nearest_filter_installed = True
        return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _restore_retopoflow_hooks()
        return False


def _install_retopoflow_hooks():
    """Install only the PolyPen and Translate candidate-boundary hooks."""
    if not _install_retopoflow_nearest_filter_hook():
        return False
    if not _install_retopoflow_polypen_hook():
        _restore_retopoflow_hooks()
        return False
    if not _install_retopoflow_automerge_hook():
        _restore_retopoflow_hooks()
        return False
    if not _install_retopoflow_translate_preview_hook():
        _restore_retopoflow_hooks()
        return False
    return True


def _release_retopoflow_hooks():
    """Release the temporary RetopoFlow hooks after FSMFO finishes."""
    if not _has_active_face_set_state():
        _restore_retopoflow_hooks()


def _state_reference_object(state):
    """Resolve a Face Set state Reference by name after Undo/rebuilds."""
    if not isinstance(state, _FaceSetOrbitState):
        return None
    try:
        name = str(state.reference_object_name)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        name = ""
    if name:
        try:
            reference = bpy.data.objects.get(name)
            if reference is not None and reference.type == "MESH":
                state.reference_object = reference
                return reference
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
    try:
        reference = state.reference_object
        if reference is not None and reference.type == "MESH":
            return reference
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    return None


def _polyquilt_filtered_snap_objects(context):
    """Keep Reference snapping while excluding the visible MFO proxy."""
    original = _polyquilt_qsnap_original_snap_objects
    if original is None:
        return []

    try:
        objects = [
            obj for obj in original(context)
            if not _is_face_set_proxy_object(obj)
        ]
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        objects = []

    active_reference_names = {
        state.reference_object_name
        for state in _active_states.values()
        if isinstance(state, _FaceSetOrbitState)
        and state.active
        and state.reference_object_name
    }
    for reference_name in active_reference_names:
        reference = bpy.data.objects.get(reference_name)
        if (
            reference is not None
            and reference.type == "MESH"
            and reference != context.active_object
            and reference not in objects
        ):
            objects.append(reference)
    return objects


def _refresh_polyquilt_qsnap(context=None, state=None):
    """Refresh PolyQuilt's cached BVH after MFO visibility changes."""
    qsnap = _polyquilt_qsnap_class or _get_polyquilt_qsnap()
    if qsnap is None or qsnap.instance is None:
        return

    try:
        if context is not None:
            qsnap.update(context)
            return

        if state is not None and state.area is not None:
            window = bpy.context.window
            region = state.region
            area = state.area
            space_data = area.spaces.active
            with bpy.context.temp_override(
                window=window,
                area=area,
                region=region,
                space_data=space_data,
            ):
                qsnap.update(bpy.context)
            return

        qsnap.update(bpy.context)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        # QSnap is optional and may be in the middle of its own teardown.
        pass


def _install_polyquilt_qsnap_filter(context=None):
    """Temporarily make PolyQuilt snap to Reference, not the display proxy."""
    global _polyquilt_qsnap_class
    global _polyquilt_qsnap_original_snap_objects
    global _polyquilt_qsnap_filter_installed

    if _polyquilt_qsnap_filter_installed:
        _refresh_polyquilt_qsnap(context=context)
        return

    qsnap = _get_polyquilt_qsnap()
    if qsnap is None:
        return

    try:
        original = qsnap.snap_objects
        if original is None:
            return
        _polyquilt_qsnap_class = qsnap
        _polyquilt_qsnap_original_snap_objects = original
        qsnap.snap_objects = staticmethod(_polyquilt_filtered_snap_objects)
        _polyquilt_qsnap_filter_installed = True
        _refresh_polyquilt_qsnap(context=context)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _polyquilt_qsnap_class = None
        _polyquilt_qsnap_original_snap_objects = None
        _polyquilt_qsnap_filter_installed = False


def _restore_polyquilt_qsnap_filter(state=None):
    """Restore PolyQuilt's original snap object provider and cached BVH."""
    global _polyquilt_qsnap_class
    global _polyquilt_qsnap_original_snap_objects
    global _polyquilt_qsnap_filter_installed

    if not _polyquilt_qsnap_filter_installed:
        return

    qsnap = _polyquilt_qsnap_class
    original = _polyquilt_qsnap_original_snap_objects
    try:
        if qsnap is not None and original is not None:
            if qsnap.snap_objects is _polyquilt_filtered_snap_objects:
                qsnap.snap_objects = staticmethod(original)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass

    _polyquilt_qsnap_class = None
    _polyquilt_qsnap_original_snap_objects = None
    _polyquilt_qsnap_filter_installed = False
    _refresh_polyquilt_qsnap(state=state)


def _release_polyquilt_qsnap_filter(state=None):
    if not _has_active_face_set_state():
        _restore_polyquilt_qsnap_filter(state=state)


def _clear_face_set_reference_marker(reference):
    if reference is None:
        return
    for key in (
        _FACE_SET_REFERENCE_TAG,
        _FACE_SET_REFERENCE_PREVIOUS_HIDE,
        _FACE_SET_REFERENCE_PROXY_COUNT,
    ):
        try:
            if key in reference:
                del reference[key]
        except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError):
            pass


def _remaining_face_set_proxies(reference_name):
    if not reference_name:
        return []
    return [
        obj
        for obj in bpy.data.objects
        if _is_face_set_proxy_object(obj)
        and _proxy_reference_name(obj) == reference_name
    ]


def _restore_reference_if_unused(reference, fallback_hide=False):
    """Restore a Reference only after its last temporary proxy is gone."""
    if reference is None:
        return
    try:
        if _remaining_face_set_proxies(reference.name):
            return

        previous_hide = bool(
            reference.get(_FACE_SET_REFERENCE_PREVIOUS_HIDE, fallback_hide)
        )
        _set_object_hidden(reference, previous_hide)
        _clear_face_set_reference_marker(reference)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _remove_face_set_proxy_object(proxy):
    """Remove one generated proxy and its unique mesh datablock."""
    if proxy is None:
        return
    try:
        proxy_mesh = proxy.data if proxy.type == "MESH" else None
        bpy.data.objects.remove(proxy, do_unlink=True)
        if proxy_mesh is not None and proxy_mesh.users == 0:
            bpy.data.meshes.remove(proxy_mesh)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _remove_face_set_proxy(state):
    """Delete the temporary Face Set proxy and restore Reference visibility."""
    if not isinstance(state, _FaceSetOrbitState):
        return

    proxy = state.proxy_object
    if proxy is None:
        try:
            proxy = bpy.data.objects.get(state.proxy_object_name)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            proxy = None
    state.proxy_object = None
    state.proxy_object_name = ""
    _remove_face_set_proxy_object(proxy)
    _restore_reference_if_unused(
        _state_reference_object(state),
        state.reference_was_hidden,
    )
    _refresh_polyquilt_qsnap(state=state)
    _tag_redraw(state.area)


def _build_face_set_proxy(state):
    """Create a temporary mesh containing only the hit Face Set."""
    if not isinstance(state, _FaceSetOrbitState):
        return False

    reference = _state_reference_object(state)
    if reference is None:
        return False
    source_mesh = reference.data
    face_set_attr = source_mesh.attributes.get(".sculpt_face_set")
    if face_set_attr is None or face_set_attr.domain != "FACE":
        return False

    proxy = None
    proxy_mesh = None
    try:
        face_set_values = array("i", [0]) * len(face_set_attr.data)
        face_set_attr.data.foreach_get("value", face_set_values)
        target_polygons = [
            polygon
            for polygon in source_mesh.polygons
            if int(face_set_values[polygon.index]) == state.face_set_id
        ]
        if not target_polygons:
            return False

        used_vertex_indices = sorted(
            {
                vertex_index
                for polygon in target_polygons
                for vertex_index in polygon.vertices
            }
        )
        if not used_vertex_indices:
            return False

        vertex_index_map = {
            old_index: new_index
            for new_index, old_index in enumerate(used_vertex_indices)
        }
        vertices = [
            tuple(source_mesh.vertices[index].co)
            for index in used_vertex_indices
        ]
        faces = [
            tuple(vertex_index_map[index] for index in polygon.vertices)
            for polygon in target_polygons
        ]

        proxy_mesh = bpy.data.meshes.new(
            f"{reference.name}__MFO_FACE_SET_PROXY_MESH"
        )
        proxy_mesh[_FACE_SET_PROXY_MESH_TAG] = True
        proxy_mesh.from_pydata(vertices, [], faces)
        proxy_mesh.update()

        # Do not share the Reference material datablocks with this temporary
        # proxy.  The proxy can survive an Undo while the source mesh/material
        # relation is being rebuilt; sharing those slots makes Blender's
        # dependency-graph material builder observe stale ownership during
        # that boundary.  The proxy's object color/display settings below are
        # sufficient for the solid viewport isolation display.
        for proxy_polygon, source_polygon in zip(
            proxy_mesh.polygons, target_polygons
        ):
            proxy_polygon.use_smooth = bool(source_polygon.use_smooth)

        proxy = bpy.data.objects.new(
            f"{reference.name}__MFO_FACE_SET_PROXY__{state.area.as_pointer()}",
            proxy_mesh,
        )
        proxy.matrix_world = reference.matrix_world.copy()
        proxy.display_type = reference.display_type
        proxy.color = reference.color
        proxy.hide_select = True
        proxy.hide_viewport = False
        proxy.hide_render = True
        proxy[_FACE_SET_PROXY_TAG] = True
        proxy[_FACE_SET_PROXY_REFERENCE] = reference.name
        proxy[_FACE_SET_PROXY_PREVIOUS_HIDE] = bool(state.reference_was_hidden)

        visible_collections = [
            collection
            for collection in reference.users_collection
            if not collection.hide_viewport
        ]
        if not visible_collections:
            visible_collections = [bpy.context.scene.collection]
        for collection in visible_collections:
            collection.objects.link(proxy)

        state.proxy_object = proxy
        state.proxy_object_name = proxy.name
        if not reference.get(_FACE_SET_REFERENCE_TAG, False):
            reference[_FACE_SET_REFERENCE_PREVIOUS_HIDE] = bool(
                state.reference_was_hidden
            )
        reference[_FACE_SET_REFERENCE_TAG] = True
        reference[_FACE_SET_REFERENCE_PROXY_COUNT] = int(
            reference.get(_FACE_SET_REFERENCE_PROXY_COUNT, 0)
        ) + 1
        if not _set_object_hidden(reference, True):
            raise RuntimeError("Could not hide the Reference Object in the viewport")
        _tag_redraw(state.area)
        return True
    except (
        AttributeError,
        ReferenceError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        proxy = state.proxy_object or proxy
        state.proxy_object = None
        state.proxy_object_name = ""
        _remove_face_set_proxy_object(proxy)
        _restore_reference_if_unused(reference, state.reference_was_hidden)
        return False


def _is_live_active_state(state):
    """Return True only while this exact state is registered and active."""
    try:
        if state is None or not state.active:
            return False
        return any(candidate is state for candidate in _active_states.values())
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _topology_color_matrix_key(obj):
    try:
        return tuple(
            float(value)
            for row in obj.matrix_world
            for value in row
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _topology_color_overlay_offset():
    """Return the current RetopoFlow depth setting without moving vertices."""
    try:
        overlay = getattr(bpy.context.space_data, "overlay", None)
        if overlay is None or not bool(
            getattr(overlay, "show_retopology", False)
        ):
            return 0.0
        return max(
            0.0,
            float(getattr(overlay, "retopology_offset", 0.0)),
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return 0.0


def _topology_color_depth_bias(overlay_offset):
    """Map Blender's overlay setting to a small clip-space depth bias."""
    try:
        configured = max(0.0, float(overlay_offset))
    except (TypeError, ValueError):
        configured = 0.0
    # Keep the same shared vertex positions for every face.  The configured
    # retopology offset is a display-depth hint, so apply it in clip space and
    # preserve clip X/Y/W rather than translating each face along its normal.
    return max(1.0e-5, min(2.0e-2, configured + 1.0e-3))


def _topology_color_depth_shader():
    """Create one cached shader that biases only clip-space Z."""
    global _topology_color_depth_shader_cache
    if _topology_color_depth_shader_cache is not None:
        return _topology_color_depth_shader_cache
    try:
        from gpu.types import GPUShaderCreateInfo

        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("FLOAT", "depth_bias")
        info.push_constant("VEC4", "color")
        info.vertex_in(0, "VEC3", "pos")
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_TOPOLOGY_COLOR_DEPTH_VERTEX_SOURCE)
        info.fragment_source(_TOPOLOGY_COLOR_DEPTH_FRAGMENT_SOURCE)
        _topology_color_depth_shader_cache = gpu.shader.create_from_info(info)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        _topology_color_depth_shader_cache = None
    return _topology_color_depth_shader_cache


def _topology_color_draw_shader():
    """Return the clip-depth shader, with a cached built-in fallback."""
    global _topology_color_fallback_shader
    depth_shader = _topology_color_depth_shader()
    if depth_shader is not None:
        return depth_shader, True
    try:
        if _topology_color_fallback_shader is None:
            _topology_color_fallback_shader = gpu.shader.from_builtin(
                "UNIFORM_COLOR"
            )
        return _topology_color_fallback_shader, False
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None, False


def _topology_color_batch(shader, primitive, vertices, custom_shader):
    """Build a batch for either the custom or built-in shader path."""
    if not custom_shader:
        return batch_for_shader(shader, primitive, {"pos": vertices})
    fmt = gpu.types.GPUVertFormat()
    fmt.attr_add(
        id="pos",
        comp_type="F32",
        len=3,
        fetch_mode="FLOAT",
    )
    vertex_buffer = gpu.types.GPUVertBuf(
        len=len(vertices),
        format=fmt,
    )
    vertex_buffer.attr_fill(id="pos", data=vertices)
    return gpu.types.GPUBatch(type=primitive, buf=vertex_buffer)


def _build_topology_color_cache(obj, overlay_offset=None):
    """Copy only visible colored Edit BMesh faces into draw-ready buffers."""
    cache = {
        "object_pointer": 0,
        "mesh_pointer": 0,
        "matrix_key": None,
        "overlay_offset": 0.0,
        "face_count": 0,
        "vert_count": 0,
        "triangles": {index: [] for index in range(1, 7)},
        "wire": [],
        "gpu_batches": None,
        "gpu_wire_batch": None,
        "gpu_shader": None,
    }
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        layer = bm.faces.layers.int.get(TOPOLOGY_COLOR_ATTRIBUTE_NAME)
        cache["object_pointer"] = int(obj.as_pointer())
        cache["mesh_pointer"] = int(obj.data.as_pointer())
        cache["matrix_key"] = _topology_color_matrix_key(obj)
        cache["overlay_offset"] = (
            _topology_color_overlay_offset()
            if overlay_offset is None
            else float(overlay_offset)
        )
        cache["face_count"] = len(bm.faces)
        cache["vert_count"] = len(bm.verts)
        if layer is None:
            return cache

        matrix = obj.matrix_world
        seen_edges = set()
        for face in bm.faces:
            if face.hide:
                continue
            try:
                color_index = int(face[layer])
            except (ReferenceError, RuntimeError, TypeError, ValueError):
                continue
            if color_index < 1 or color_index > 6:
                continue

            # ``BMFace.calc_tessellation`` returns BMVerts on Blender 5.2.
            # Keep a fallback for polygon APIs that return plain Vectors.
            try:
                tessellation = face.calc_tessellation()
                tessellation_vertices = None
            except (AttributeError, RuntimeError, TypeError, ValueError):
                tessellation_vertices = [
                    vertex.co.copy()
                    for vertex in face.verts
                ]
                tessellation = tessellate_polygon(
                    [tessellation_vertices]
                )
            for triangle in tessellation:
                if len(triangle) != 3:
                    continue
                triangle_points = []
                for vertex in triangle:
                    if (
                        tessellation_vertices is not None
                        and isinstance(vertex, int)
                    ):
                        coordinate = tessellation_vertices[vertex]
                    else:
                        coordinate = getattr(vertex, "co", vertex)
                    triangle_points.append(tuple(matrix @ coordinate))
                cache["triangles"][color_index].extend(triangle_points)

            for edge in face.edges:
                try:
                    edge_index = int(edge.index)
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    edge_index = -1
                if edge_index >= 0:
                    edge_key = ("index", edge_index)
                else:
                    fallback_vertices = []
                    for vertex in edge.verts:
                        try:
                            vertex_index = int(vertex.index)
                        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                            vertex_index = -1
                        if vertex_index >= 0:
                            fallback_vertices.append(("index", vertex_index))
                        else:
                            # ``id(BMVert)`` is stable for this one build and
                            # avoids collapsing every unindexed edge together.
                            fallback_vertices.append(("identity", id(vertex)))
                    edge_key = (
                        "verts",
                        tuple(sorted(fallback_vertices, key=repr)),
                    )
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                first, second = edge.verts
                line = (
                    tuple(matrix @ first.co),
                    tuple(matrix @ second.co),
                )
                cache["wire"].extend(line)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        # A mode switch or Undo can invalidate the Edit BMesh between the draw
        # callback and this read.  The next redraw will retry from scratch.
        return cache
    return cache


def _topology_color_cache_for(obj):
    try:
        object_pointer = int(obj.as_pointer())
        mesh_pointer = int(obj.data.as_pointer())
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None
    cache = _topology_color_cache.get(object_pointer)
    matrix_key = _topology_color_matrix_key(obj)
    overlay_offset = _topology_color_overlay_offset()
    stale = (
        cache is None
        or object_pointer in _topology_color_cache_dirty
        or cache.get("mesh_pointer") != mesh_pointer
        or cache.get("matrix_key") != matrix_key
        or cache.get("overlay_offset") != overlay_offset
    )
    if stale:
        cache = _build_topology_color_cache(obj, overlay_offset)
        _topology_color_cache[object_pointer] = cache
        _topology_color_cache_dirty.discard(object_pointer)
    return cache


def _draw_topology_colors():
    """Draw translucent face colors for the active Edit Mesh only."""
    prefs = _addon_preferences()
    if prefs is not None:
        if not prefs.enabled or not prefs.topology_colors_enabled:
            return
    try:
        context = bpy.context
        if (
            context.area is None
            or context.area.type != "VIEW_3D"
            or context.region is None
            or context.region.type != "WINDOW"
        ):
            return
        overlay = getattr(context.space_data, "overlay", None)
        if overlay is not None and not bool(
            getattr(overlay, "show_overlays", True)
        ):
            return
        obj = _topology_color_object(context)
        if obj is None:
            return
        cache = _topology_color_cache_for(obj)
        if cache is None:
            return
        opacity = float(prefs.topology_color_opacity) if prefs else 0.35
        opacity = max(0.0, min(1.0, opacity))
        shader, uses_depth_bias = _topology_color_draw_shader()
        if shader is None:
            return
        depth_bias = _topology_color_depth_bias(cache.get("overlay_offset", 0.0))
        depth_set = False
        depth_mask_changed = False
        gpu.state.blend_set("ALPHA")
        try:
            try:
                gpu.state.depth_test_set("LESS_EQUAL")
                depth_set = True
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            try:
                gpu.state.depth_mask_set(False)
                depth_mask_changed = True
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            if uses_depth_bias:
                shader.bind()
                shader.uniform_float(
                    "ModelViewProjectionMatrix",
                    gpu.matrix.get_projection_matrix()
                    @ gpu.matrix.get_model_view_matrix(),
                )
                shader.uniform_float("depth_bias", depth_bias)
            gpu_batches = cache.get("gpu_batches")
            if cache.get("gpu_shader") is not shader:
                gpu_batches = None
                cache["gpu_wire_batch"] = None
                cache["gpu_shader"] = shader
            if gpu_batches is None:
                gpu_batches = {
                    color_index: _topology_color_batch(
                        shader,
                        "TRIS",
                        vertices,
                        uses_depth_bias,
                    )
                    for color_index, vertices in cache["triangles"].items()
                    if vertices
                }
                cache["gpu_batches"] = gpu_batches
            for color_index, vertices in cache["triangles"].items():
                if not vertices:
                    continue
                batch = gpu_batches[color_index]
                shader.bind()
                shader.uniform_float(
                    "color",
                    (*TOPOLOGY_COLOR_PALETTE[color_index - 1], opacity),
                )
                batch.draw(shader)

            if cache["wire"]:
                wire_batch = cache.get("gpu_wire_batch")
                if wire_batch is None:
                    wire_batch = _topology_color_batch(
                        shader,
                        "LINES",
                        cache["wire"],
                        uses_depth_bias,
                    )
                    cache["gpu_wire_batch"] = wire_batch
                shader.bind()
                shader.uniform_float("color", (0.01, 0.01, 0.01, 0.45))
                wire_batch.draw(shader)
        finally:
            if depth_mask_changed:
                try:
                    gpu.state.depth_mask_set(True)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            if depth_set:
                try:
                    gpu.state.depth_test_set("NONE")
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            try:
                gpu.state.line_width_set(1.0)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            gpu.state.blend_set("NONE")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        # Drawing is optional and must never interrupt RF4 or viewport input.
        pass


def _start_topology_color_draw():
    global _topology_color_draw_handler
    if _topology_color_draw_handler is not None:
        return
    try:
        _topology_color_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_topology_colors,
            (),
            "WINDOW",
            "POST_VIEW",
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        _topology_color_draw_handler = None


def _stop_topology_color_draw():
    global _topology_color_draw_handler
    if _topology_color_draw_handler is None:
        return
    try:
        bpy.types.SpaceView3D.draw_handler_remove(
            _topology_color_draw_handler,
            "WINDOW",
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    _topology_color_draw_handler = None
    _topology_color_cache.clear()
    _topology_color_cache_dirty.clear()


def _draw_debug_point(state):
    if not _is_live_active_state(state):
        return

    try:
        shader = gpu.shader.from_builtin("POINT_UNIFORM_COLOR")
        batch = batch_for_shader(
            shader,
            "POINTS",
            {"pos": [state.hit_location]},
        )
        shader.bind()
        shader.uniform_float("color", (0.05, 0.55, 1.0, 1.0))
        shader.uniform_float("size", 10.0)
        batch.draw(shader)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # Debug drawing is optional and must never affect viewport navigation.
        pass


def _rounded_rect_points(x_min, y_min, x_max, y_max, radius, segments=6):
    radius = max(0.0, min(radius, (x_max - x_min) * 0.5, (y_max - y_min) * 0.5))
    corners = (
        (x_min + radius, y_min + radius, math.pi, math.pi * 1.5),
        (x_max - radius, y_min + radius, math.pi * 1.5, math.pi * 2.0),
        (x_max - radius, y_max - radius, 0.0, math.pi * 0.5),
        (x_min + radius, y_max - radius, math.pi * 0.5, math.pi),
    )
    points = []
    for center_x, center_y, start_angle, end_angle in corners:
        for index in range(segments + 1):
            factor = index / segments
            angle = start_angle + (end_angle - start_angle) * factor
            points.append(
                (
                    center_x + math.cos(angle) * radius,
                    center_y + math.sin(angle) * radius,
                )
            )
    return points


def _draw_rounded_indicator_box(x_min, y_min, x_max, y_max):
    points = _rounded_rect_points(x_min, y_min, x_max, y_max, 10.0)
    center = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5)
    fill_vertices = []
    for index, point in enumerate(points):
        fill_vertices.extend(
            (center, point, points[(index + 1) % len(points)])
        )

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    fill_batch = batch_for_shader(shader, "TRIS", {"pos": fill_vertices})
    shader.bind()
    shader.uniform_float("color", (0.015, 0.045, 0.035, 0.78))
    fill_batch.draw(shader)

    outline_batch = batch_for_shader(
        shader,
        "LINE_STRIP",
        {"pos": points + [points[0]]},
    )
    shader.bind()
    shader.uniform_float("color", (0.35, 1.0, 0.75, 0.9))
    outline_batch.draw(shader)


def _draw_indicator_text(state):
    """Draw the MFO indicator without using the shared area header text."""
    if not _is_live_active_state(state):
        return

    try:
        context_area = bpy.context.area
        context_region = bpy.context.region
        if context_area is None or context_region is None:
            return
        if context_area.as_pointer() != state.area.as_pointer():
            return
        if context_region.as_pointer() != state.region.as_pointer():
            return
        if context_region.type != "WINDOW":
            return

        font_id = 0
        # This is the top edge of the indicator box.  RetopoFlow can add a
        # second tool row inside the top of the viewport, so keep the box just
        # below that area while staying close to Blender's own overlay text.
        box_top = max(24, context_region.height - 90)
        gpu.state.blend_set("ALPHA")
        try:
            blf.size(font_id, 18)
            blf.color(font_id, 0.35, 1.0, 0.75, 1.0)
            text_width, text_height = blf.dimensions(
                font_id,
                state.indicator_text,
            )
            padding_x = 18.0
            padding_top = 10.0
            padding_bottom = 9.0
            box_width = text_width + padding_x * 2.0
            box_height = text_height + padding_top + padding_bottom
            box_left = max(8.0, (context_region.width - box_width) * 0.5)
            box_right = min(
                context_region.width - 8.0,
                box_left + box_width,
            )
            box_bottom = box_top - box_height

            try:
                _draw_rounded_indicator_box(
                    box_left,
                    box_bottom,
                    box_right,
                    box_top,
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # The panel is optional; keep the text visible if GPU drawing
                # is unavailable in a particular viewport context.
                pass

            text_x = box_left + padding_x
            text_y = box_bottom + padding_bottom
            blf.position(font_id, text_x, text_y, 0)
            blf.draw(font_id, state.indicator_text)
        finally:
            gpu.state.blend_set("NONE")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        # The indicator is optional and must never affect viewport operation.
        pass


def _start_indicator_draw(state):
    prefs = _addon_preferences()
    if prefs is not None and not prefs.show_indicator:
        return
    if state.indicator_draw_handler is not None:
        return
    try:
        state.indicator_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_indicator_text,
            (state,),
            "WINDOW",
            "POST_PIXEL",
        )
    except (AttributeError, RuntimeError, TypeError):
        state.indicator_draw_handler = None


def _stop_indicator_draw(state):
    if state.indicator_draw_handler is None:
        return
    try:
        bpy.types.SpaceView3D.draw_handler_remove(
            state.indicator_draw_handler,
            "WINDOW",
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    state.indicator_draw_handler = None


def _start_debug_draw(state):
    prefs = _addon_preferences()
    if not prefs or not prefs.debug_display:
        return
    try:
        state.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_debug_point,
            (state,),
            "WINDOW",
            "POST_VIEW",
        )
    except (AttributeError, RuntimeError, TypeError):
        state.draw_handler = None


def _stop_debug_draw(state):
    if state.draw_handler is None:
        return
    try:
        bpy.types.SpaceView3D.draw_handler_remove(state.draw_handler, "WINDOW")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    state.draw_handler = None


def _activate_state(state):
    """Move only the orbit target, keeping the camera fixed at activation."""
    rv3d = state.rv3d
    if state.is_perspective and state.hit_distance is not None:
        # Changing view_location alone would move the camera.  Match the
        # camera-to-hit distance so Ctrl activation itself does not jump.
        rv3d.view_distance = max(state.hit_distance, 1.0e-6)

    rv3d.view_location = state.hit_location
    # Navigation Gizmo may consume the cached view matrix immediately after
    # activation.  Force Blender to rebuild it before the next native orbit.
    try:
        rv3d.update()
    except (AttributeError, RuntimeError, TypeError):
        pass
    _start_indicator_draw(state)
    _start_debug_draw(state)
    _tag_redraw(state.area)


def _face_set_activation_artifact_present(state):
    """Use the native Proxy Object as the FSMFO activation sentinel."""
    if not isinstance(state, _FaceSetOrbitState):
        return True
    try:
        proxy_name = str(state.proxy_object_name)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    if not proxy_name:
        return False
    proxy = bpy.data.objects.get(proxy_name)
    return _is_face_set_proxy_object(proxy)


def _stop_watcher_timer(state):
    """Remove a session timer without touching any other session."""
    if state is None:
        return
    timer = getattr(state, "watcher_timer", None)
    if timer is None:
        return
    try:
        bpy.context.window_manager.event_timer_remove(timer)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    state.watcher_timer = None


def _detach_watcher_operator(operator):
    """Detach one watcher instance, never another session's watcher."""
    if operator is None:
        return
    try:
        operator_key = operator.as_pointer()
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        operator_key = id(operator)
    try:
        timer = getattr(operator, "_mfo_timer", None)
        if timer is not None:
            try:
                bpy.context.window_manager.event_timer_remove(timer)
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                pass
        area_key = int(getattr(operator, "_mfo_area_key", 0))
        session_id = int(getattr(operator, "_mfo_session_id", 0))
        state = _active_states.get(area_key)
        if (
            state is not None
            and getattr(state, "session_id", 0) == session_id
            and getattr(state, "watcher_operator_key", None) == operator_key
        ):
            state.watcher_operator_key = None
            state.watcher_timer = None
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    try:
        operator._mfo_timer = None
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        pass


def _restore_original_view(state):
    """Restore the complete viewport transform saved at activation."""
    if not state:
        return
    try:
        rv3d = state.rv3d
        rv3d.view_perspective = state.original_view_perspective
        rv3d.view_location = state.original_view_location
        rv3d.view_distance = state.original_view_distance
        rv3d.view_rotation = state.original_view_rotation
        rv3d.update()
    except (ReferenceError, RuntimeError, AttributeError, TypeError, ValueError):
        pass


def _deactivate_state(state):
    """Exit temporary mode and restore the complete saved viewport view."""
    if not state or not state.active:
        return
    _finish_state(state)


def _finish_state(state, restore_retopo=True):
    """End one owned session with idempotent, ownership-safe cleanup."""
    if state is None:
        return

    try:
        area_key = int(state.area_key)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        area_key = 0
    try:
        session_id = int(state.session_id)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        session_id = None
    restore_error = False
    if area_key and area_key in _session_cleanup_areas:
        return
    if area_key:
        _session_cleanup_areas.add(area_key)

    # Deactivate and detach drawing immediately.  Cleanup below is deliberately
    # best-effort: a stale Object, Proxy, Area, or View3D handler must not leave
    # the ON indicator visible or prevent the remaining cleanup steps.
    try:
        state.active = False
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        pass
    _unregister_session(state)
    try:
        _stop_indicator_draw(state)
    except Exception:
        state.indicator_draw_handler = None
    try:
        _stop_debug_draw(state)
    except Exception:
        state.draw_handler = None

    # The Watcher owns the periodic Timer.  Remove only this session's timer
    # before touching native data so queued TIMER events cannot affect a new
    # session that reuses the same viewport.
    _stop_watcher_timer(state)
    try:
        state.watcher_operator_key = None
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        pass

    if restore_retopo:
        try:
            _retopo_restore_isolation(state)
        except Exception:
            restore_error = True
            _retopo_debug_lifecycle(session_id, "fsmfo_restore_exception")
            pass
    elif isinstance(state, _FaceSetOrbitState):
        # Blender has already restored the pre-FSMFO Undo state.  The old
        # BMesh elements/layers may no longer exist, so deliberately discard
        # the Python reference without attempting to restore it.
        state.retopo_isolation = None
    try:
        _remove_face_set_proxy(state)
    except Exception:
        pass
    try:
        _restore_original_view(state)
    except Exception:
        pass

    # These hooks are shared with RetopoFlow/PolyQuilt.  Always release them
    # even when one of the scene/view restoration steps failed.
    try:
        _release_retopoflow_hooks()
    except Exception:
        restore_error = True
        _retopo_debug_lifecycle(session_id, "retopoflow_hook_release_exception")
        pass
    try:
        _release_polyquilt_qsnap_filter(state=state)
    except Exception:
        pass
    try:
        _tag_redraw(state.area)
    except Exception:
        pass
    _retopo_debug_end_session(
        session_id,
        "restore_error"
        if restore_error
        else "undo_restore"
        if not restore_retopo
        else "restore",
    )
    if area_key:
        _session_cleanup_areas.discard(area_key)


def _finish_all_states():
    _fill_preview_cancel(reason="finish-all")
    for key, state in list(_active_states.items()):
        _finish_state(state)
        _active_states.pop(key, None)
    _last_tap_times.clear()
    _local_face_set_adjacency_cache.clear()


def _finish_face_set_states():
    """Finish only Face Set MFO states, leaving normal MFO untouched."""
    face_set_area_keys = set()
    for area_key, state in list(_active_states.items()):
        if not isinstance(state, _FaceSetOrbitState):
            continue
        face_set_area_keys.add(area_key)
        _finish_state(state)
        _active_states.pop(area_key, None)

    _session_cleanup_areas.difference_update(face_set_area_keys)


def _cleanup_orphan_face_set_proxies(reconcile_undo_tombstones=False):
    """Remove Face Set MFO scene remnants whose modal state no longer exists.

    Blender's Undo and file/window lifecycle can invalidate Python operator
    instances without running their ``__del__`` method.  This function is
    deliberately limited to tagged/generated proxy objects and the Reference
    visibility marker owned by those proxies.  The optional tombstone
    reconciliation is the only path that may restore owned Retopo mesh data;
    generic proxy cleanup never changes selection, active object, or mode.
    """
    _retopo_debug_emit(
        "generic_cleanup_before",
        {
            "reconcile_undo_tombstones": bool(reconcile_undo_tombstones),
            "tombstone_count": len(_retopo_undo_tombstones),
            "active_state_count": len(_active_states),
        },
    )
    if reconcile_undo_tombstones:
        _retopo_reconcile_undo_tombstones()
    # Reconcile the runtime tombstone before the generic tagged cleanup.  On
    # native Undo the Object markers and BMesh layers may return together; the
    # tombstone must consume that ON state before the generic path removes it.
    _cleanup_orphan_retopo_isolations()

    active_proxy_pointers = set()
    active_proxy_names = set()
    for state in list(_active_states.values()):
        if not isinstance(state, _FaceSetOrbitState) or not state.active:
            continue
        try:
            if state.proxy_object_name:
                active_proxy_names.add(str(state.proxy_object_name))
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        proxy = state.proxy_object
        if proxy is None:
            continue
        try:
            active_proxy_pointers.add(proxy.as_pointer())
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass

    affected_references = {}
    removed_proxy_names = []
    for proxy in list(bpy.data.objects):
        if not _is_face_set_proxy_object(proxy):
            continue
        if proxy.name in active_proxy_names:
            continue
        try:
            if proxy.as_pointer() in active_proxy_pointers:
                continue
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass

        reference_name = _proxy_reference_name(proxy)
        affected_references.setdefault(
            reference_name,
            _proxy_previous_hide(proxy, False),
        )
        removed_proxy_names.append(proxy.name)
        _remove_face_set_proxy_object(proxy)

    for reference_name, previous_hide in affected_references.items():
        reference = bpy.data.objects.get(reference_name)
        _restore_reference_if_unused(reference, previous_hide)

    # A previous crash may have removed the proxy but left the Reference
    # marker behind.  Clear that marker only when no proxy remains.
    for reference in list(bpy.data.objects):
        try:
            if not reference.get(_FACE_SET_REFERENCE_TAG, False):
                continue
            if _remaining_face_set_proxies(reference.name):
                continue
            previous_hide = bool(
                reference.get(_FACE_SET_REFERENCE_PREVIOUS_HIDE, False)
            )
            _set_object_hidden(reference, previous_hide)
            _clear_face_set_reference_marker(reference)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass

    removed_proxy_meshes = []
    for mesh in list(bpy.data.meshes):
        try:
            if mesh.users == 0 and _is_face_set_proxy_mesh(mesh):
                removed_proxy_meshes.append(mesh.name)
                bpy.data.meshes.remove(mesh)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass

    _release_polyquilt_qsnap_filter()
    _release_retopoflow_hooks()

    if removed_proxy_names or removed_proxy_meshes:
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            if screen:
                for area in screen.areas:
                    _tag_redraw(area)

    _retopo_debug_emit(
        "generic_cleanup_after",
        {
            "reconcile_undo_tombstones": bool(reconcile_undo_tombstones),
            "removed_proxy_names": [str(name) for name in removed_proxy_names],
            "removed_proxy_meshes": [str(name) for name in removed_proxy_meshes],
            "removed_proxy_count": len(removed_proxy_names),
            "removed_mesh_count": len(removed_proxy_meshes),
            "tombstone_count": len(_retopo_undo_tombstones),
            "active_state_count": len(_active_states),
        },
    )
    return removed_proxy_names, removed_proxy_meshes


def _is_live_face_set_proxy(proxy):
    """Return whether a state still points at a live generated proxy."""
    if not _is_face_set_proxy_object(proxy):
        return False
    try:
        return bpy.data.objects.get(proxy.name) is proxy
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False


def _has_reclaimable_orphan_artifacts():
    """Return whether tagged, non-active FSMFO artifacts still need cleanup."""
    active_proxy_names = set()
    active_retopo_names = set()
    for state in list(_active_states.values()):
        if not isinstance(state, _FaceSetOrbitState) or not state.active:
            continue
        try:
            if state.proxy_object_name:
                active_proxy_names.add(str(state.proxy_object_name))
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        try:
            info = state.retopo_isolation
            if info and info.get("object_name"):
                active_retopo_names.add(str(info["object_name"]))
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass

    for tombstone in _retopo_undo_tombstones.values():
        if (
            tombstone.get("object_name") not in active_retopo_names
            and tombstone.get("retry_pending", False)
        ):
            return True

    for obj in list(bpy.data.objects):
        try:
            if _is_face_set_proxy_object(obj) and obj.name not in active_proxy_names:
                return True
            if (
                obj.get(_FACE_SET_REFERENCE_TAG, False)
                and not _remaining_face_set_proxies(obj.name)
            ):
                return True
            if (
                obj.get(_RETOPO_ISOLATION_TAG, False)
                and obj.name not in active_retopo_names
            ):
                return True
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            continue
    for mesh in list(bpy.data.meshes):
        try:
            if mesh.users == 0 and _is_face_set_proxy_mesh(mesh):
                return True
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            continue
    return False


def _deferred_orphan_cleanup():
    """Run registration-time recovery after Blender leaves RestrictedData."""
    if not _is_registered:
        return None
    try:
        _cleanup_orphan_face_set_proxies()
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        # A short retry handles the brief interval during workspace startup.
        return 0.25
    return None


def _schedule_orphan_cleanup():
    try:
        bpy.app.timers.register(_deferred_orphan_cleanup, first_interval=0.1)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass


def _deferred_undo_orphan_cleanup():
    """Reconcile tagged FSMFO remnants after Blender finishes an Undo step.

    ``undo_post`` can run while Edit BMesh data is still being rebuilt.  Keep
    the handler lightweight and perform the ownership-tagged cleanup from a
    short timer instead.  Active sessions remain protected by the existing
    session/generation checks in the orphan cleanup functions.
    """
    global _undo_orphan_cleanup_pending, _undo_orphan_cleanup_retries
    global _last_undo_post_perf
    global _retopo_undo_retry_pending
    deferred_started = time.perf_counter()
    undo_post_to_callback_ms = None
    if _last_undo_post_perf is not None:
        undo_post_to_callback_ms = round(
            (deferred_started - _last_undo_post_perf) * 1000.0,
            3,
        )
    _last_undo_post_perf = None
    _undo_orphan_cleanup_pending = False
    _retopo_debug_emit(
        "deferred_start",
        {
            "retries": int(_undo_orphan_cleanup_retries),
            "undo_post_to_callback_ms": undo_post_to_callback_ms,
            "tombstone_count": len(_retopo_undo_tombstones),
            "active_state_count": len(_active_states),
        },
    )
    if not _is_registered:
        _retopo_debug_emit(
            "deferred_end",
            {
                "decision": "skip_unregistered",
                "tombstone_count": len(_retopo_undo_tombstones),
                "duration_ms": round(
                    (time.perf_counter() - deferred_started) * 1000.0,
                    3,
                ),
            },
        )
        return None
    try:
        _cleanup_orphan_face_set_proxies(reconcile_undo_tombstones=True)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        # BMesh may still be temporarily unavailable after undo_post.
        if _undo_orphan_cleanup_retries < 3:
            _undo_orphan_cleanup_retries += 1
            _undo_orphan_cleanup_pending = True
            _retopo_debug_emit(
                "deferred_end",
                {
                    "decision": "retry",
                    "reason": "cleanup_exception",
                    "retries": int(_undo_orphan_cleanup_retries),
                    "tombstone_count": len(_retopo_undo_tombstones),
                    "duration_ms": round(
                        (time.perf_counter() - deferred_started) * 1000.0,
                        3,
                    ),
                },
            )
            return 0.1
        _retopo_undo_retry_pending = False
        for tombstone in _retopo_undo_tombstones.values():
            tombstone["retry_pending"] = False
        _undo_orphan_cleanup_retries = 0
        _retopo_debug_emit(
            "deferred_end",
            {
                "decision": "stop_retries_retain_tombstones",
                "reason": "cleanup_exception_retry_limit",
                "tombstone_count": len(_retopo_undo_tombstones),
                "duration_ms": round(
                    (time.perf_counter() - deferred_started) * 1000.0,
                    3,
                ),
            },
        )
        return None
    if _has_reclaimable_orphan_artifacts() and _undo_orphan_cleanup_retries < 3:
        _undo_orphan_cleanup_retries += 1
        _undo_orphan_cleanup_pending = True
        _retopo_debug_emit(
            "deferred_end",
            {
                "decision": "retry",
                "reason": "reclaimable_transient_artifact",
                "retries": int(_undo_orphan_cleanup_retries),
                "tombstone_count": len(_retopo_undo_tombstones),
                "duration_ms": round(
                    (time.perf_counter() - deferred_started) * 1000.0,
                    3,
                ),
            },
        )
        return 0.1
    if _has_reclaimable_orphan_artifacts():
        # Stop retrying this undo event, but retain dormant tombstones.  A
        # later undo_post must get another chance to reach the native ON
        # state.  Only identity/session/load lifecycle boundaries discard
        # tombstones.
        _retopo_undo_retry_pending = False
        for tombstone in _retopo_undo_tombstones.values():
            tombstone["retry_pending"] = False
    _undo_orphan_cleanup_retries = 0
    _retopo_debug_emit(
        "deferred_end",
        {
            "decision": "complete",
            "tombstone_count": len(_retopo_undo_tombstones),
            "active_state_count": len(_active_states),
            "duration_ms": round(
                (time.perf_counter() - deferred_started) * 1000.0,
                3,
            ),
        },
    )
    return None


def _schedule_undo_orphan_cleanup():
    """Schedule at most one post-Undo reconciliation for this module."""
    global _undo_orphan_cleanup_pending, _undo_orphan_cleanup_retries
    global _retopo_undo_retry_pending
    if not _is_registered or _undo_orphan_cleanup_pending:
        return
    try:
        bpy.app.timers.register(_deferred_undo_orphan_cleanup, first_interval=0.1)
        _undo_orphan_cleanup_pending = True
        _undo_orphan_cleanup_retries = 0
        _retopo_undo_retry_pending = False
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass


def _cancel_undo_orphan_cleanup():
    """Remove a queued callback during unregister/reload lifecycle cleanup."""
    global _undo_orphan_cleanup_pending, _undo_orphan_cleanup_retries
    global _last_undo_post_perf
    global _retopo_undo_retry_pending
    try:
        if bpy.app.timers.is_registered(_deferred_undo_orphan_cleanup):
            bpy.app.timers.unregister(_deferred_undo_orphan_cleanup)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    _undo_orphan_cleanup_pending = False
    _undo_orphan_cleanup_retries = 0
    _last_undo_post_perf = None
    _retopo_undo_retry_pending = False
    for tombstone in _retopo_undo_tombstones.values():
        tombstone["retry_pending"] = False


@persistent
def _on_undo_post(_dummy):
    """Queue ownership-safe cleanup after native Undo has rebuilt the scene."""
    global _last_undo_post_perf
    _fill_preview_cancel(reason="undo")
    if not _undo_orphan_cleanup_pending:
        _last_undo_post_perf = time.perf_counter()
    _retopo_debug_emit(
        "undo_post",
        {
            "tombstone_count": len(_retopo_undo_tombstones),
            "active_state_count": len(_active_states),
            "cleanup_timer_pending": bool(_undo_orphan_cleanup_pending),
        },
    )
    _schedule_undo_orphan_cleanup()


@persistent
def _on_load_pre(_dummy):
    """Clear viewport-bound state before Blender replaces the current file."""
    _fill_preview_cancel(reason="load")
    _finish_all_states()
    _cleanup_orphan_face_set_proxies()
    _retopo_undo_tombstones.clear()
    _local_face_set_adjacency_cache.clear()
    _fill_preview_adjacency_cache.clear()
    _fill_preview_cursor_cache.clear()


@persistent
def _on_load_post(_dummy):
    """Recover remnants loaded from a file saved during Face Set MFO."""
    _fill_preview_cancel(reason="load-post")
    _cleanup_orphan_face_set_proxies()


def _clear_topology_color_draw_cache():
    """Drop copied coordinates at every history/file lifetime boundary."""
    _topology_color_cache.clear()
    _topology_color_cache_dirty.clear()
    _topology_color_tag_redraw_all()


@persistent
def _on_topology_color_depsgraph_update(_scene, depsgraph):
    """Invalidate only meshes/objects changed by the dependency graph."""
    try:
        for update in depsgraph.updates:
            data = update.id
            if isinstance(data, bpy.types.Mesh):
                data_pointer = int(data.as_pointer())
                for object_pointer, cache in _topology_color_cache.items():
                    if cache.get("mesh_pointer") == data_pointer:
                        _topology_color_cache_dirty.add(object_pointer)
            elif isinstance(data, bpy.types.Object) and data.type == "MESH":
                _topology_color_cache_dirty.add(int(data.as_pointer()))
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


@persistent
def _on_topology_color_undo_post(_dummy):
    _clear_topology_color_draw_cache()


@persistent
def _on_topology_color_redo_post(_dummy):
    _clear_topology_color_draw_cache()


@persistent
def _on_topology_color_load_pre(_dummy):
    _clear_topology_color_draw_cache()


@persistent
def _on_topology_color_load_post(_dummy):
    _clear_topology_color_draw_cache()


def _operator_key(operator):
    """Get a stable key for session ownership diagnostics."""
    try:
        return operator.as_pointer()
    except (AttributeError, ReferenceError, RuntimeError):
        return id(operator)


def _start_session(context, state, face_set=False):
    """Create one native activation and attach exactly one Watcher."""
    if state is None or not _register_session(state):
        return False
    try:
        if face_set:
            if not _build_face_set_proxy(state):
                raise RuntimeError("Unable to create Face Set proxy")
            state.retopo_isolation = _retopo_begin_isolation(
                context,
                state.reference_object,
                state.proxy_object,
            )
            if state.retopo_isolation:
                state.retopo_restore_snapshot = (
                    state.retopo_isolation.get("restore_snapshot")
                    or _retopo_capture_restore_snapshot(state.retopo_isolation)
                )

        if face_set:
            _install_polyquilt_qsnap_filter(context=context)
            if (
                _retopoflow_snap_weld_filter_enabled()
                and state.retopo_isolation
            ):
                # RetopoFlow is optional.  Its absence must not prevent
                # normal Face Set MFO activation; when present, the hook
                # installer owns its own all-or-nothing rollback.
                _install_retopoflow_hooks()

        _activate_state(state)
        watcher_result = bpy.ops.view3d.mesh_focus_orbit_watcher(
            "INVOKE_DEFAULT"
        )
        if "RUNNING_MODAL" not in watcher_result:
            raise RuntimeError("FSMFO Watcher did not start")
        if _retopo_debug_enabled():
            try:
                info = getattr(state, "retopo_isolation", None)
                object_name = info.get("object_name") if info else None
                if object_name is None:
                    object_name = getattr(state, "reference_object_name", None)
                retopo_session_id = info.get("session_id") if info else None
                # This is a Python-only association on the live isolation
                # dictionary.  It is never copied to BMesh layers, objects,
                # or scene data, and the logger remains keyed only by MFO's
                # state.session_id.
                if info is not None:
                    info["mfo_session_id"] = state.session_id
                _retopo_debug_begin_session(
                    state.session_id,
                    "fsmfo_session" if face_set else "mfo_session",
                    object_name,
                    retopo_session_id,
                )
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                pass
        return True
    except (
        AttributeError,
        ReferenceError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        _finish_state(state)
        return False


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
        state = _active_states.get(area_key)
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
        except (
            AttributeError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            _detach_watcher_operator(self)
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def modal(self, _context, event):
        area_key = getattr(self, "_mfo_area_key", 0)
        session_id = getattr(self, "_mfo_session_id", 0)
        state = _active_states.get(area_key)
        if (
            state is None
            or not state.active
            or state.session_id != session_id
            or state.watcher_operator_key != _operator_key(self)
        ):
            # This watcher lost ownership.  It must not touch the current
            # session's Proxy, Reference, isolation, hooks, or timers.
            _detach_watcher_operator(self)
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
                return {"CANCELLED"}
            return {"PASS_THROUGH"}

        if event.type == "WINDOW_DEACTIVATE":
            prefs = _addon_preferences()
            if prefs and prefs.focus_loss_behavior == "EXIT":
                _finish_state(state)
                _detach_watcher_operator(self)
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
        previous_tap = _last_tap_times.get(tap_key)
        if previous_tap is None or now - previous_tap > double_tap_window:
            _last_tap_times[tap_key] = now
            return {"PASS_THROUGH"}

        _last_tap_times.pop(tap_key, None)
        return self.execute(context)

    def execute(self, context):
        if not self.poll(context):
            return {"CANCELLED"}
        try:
            area_key = context.area.as_pointer()
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            return {"CANCELLED"}
        if area_key in _session_cleanup_areas:
            return {"CANCELLED"}
        existing = _active_states.get(area_key)
        if existing is not None and existing.active:
            if isinstance(existing, _FaceSetOrbitState):
                # FSMFO uses this same direct Activation operator for OFF.
                # CANCELLED avoids creating a second UNDO step for the
                # runtime-only teardown.
                _deactivate_state(existing)
                return {"CANCELLED"}
            return {"CANCELLED"}

        hit = _raycast_reference_face_set(context)
        if hit is None:
            return {"CANCELLED"}
        reference_object, _face_index, face_set_id, hit_location, hit_distance = hit
        prefs = _addon_preferences()
        activation_key = prefs.activation_key if prefs else "RIGHT_SHIFT"
        state = _FaceSetOrbitState(
            context,
            hit_location,
            hit_distance,
            activation_key,
            reference_object,
            face_set_id,
        )
        if not _start_session(context, state, face_set=True):
            return {"CANCELLED"}
        # This operator must end here.  Only the Watcher remains modal.
        return {"FINISHED"}


class VIEW3D_OT_mesh_focus_orbit(bpy.types.Operator):
    """Double-tap the configured key to start or stop an MFO session."""

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
            and context.mode in {"OBJECT", "EDIT_MESH"}
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
        previous_tap = _last_tap_times.get(tap_key)
        if previous_tap is None or now - previous_tap > double_tap_window:
            _last_tap_times[tap_key] = now
            return {"PASS_THROUGH"}
        _last_tap_times.pop(tap_key, None)

        existing_state = _active_states.get(area_key)
        if existing_state is not None and existing_state.active:
            if not isinstance(existing_state, _FaceSetOrbitState):
                _deactivate_state(existing_state)
                # This trigger operator has no UNDO flag, so manual OFF does
                # not create an extra Undo boundary.
                return {"CANCELLED"}
            return {"PASS_THROUGH"}

        if area_key in _session_cleanup_areas:
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


def _raycast_sculpt_face_set(context, coord):
    """Return the first visible active-mesh face under the cursor.

    Object.ray_cast does not honor Sculpt face hiding.  When it returns a
    hidden face, advance the local ray past that hit and continue until a
    visible face is found.  The returned polygon index remains the original
    mesh index used by the Face Set attribute and preview adjacency graph.
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
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        if direction.length_squared == 0.0:
            return None
        direction.normalize()

        inverse = obj.matrix_world.inverted_safe()
        local_origin = inverse @ origin
        local_direction = inverse.to_3x3() @ direction
        if local_direction.length_squared == 0.0:
            return None
        local_direction.normalize()

        for _attempt in range(256):
            hit, location, _normal, face_index = obj.ray_cast(
                local_origin,
                local_direction,
            )
            if not hit or face_index < 0 or face_index >= len(face_set_attr.data):
                return None
            polygon = obj.data.polygons[int(face_index)]
            if not bool(polygon.hide):
                return (
                    obj,
                    int(face_index),
                    int(face_set_attr.data[face_index].value),
                    obj.matrix_world @ location,
                    coord,
                )
            # Blender's object ray cast includes hidden Sculpt faces.  Move
            # past this surface in local space before retrying, otherwise the
            # same hidden polygon would be returned indefinitely.  Keep the
            # step small relative to the hit distance and cap it so large
            # models do not skip a nearby visible layer.
            hit_distance = max((location - local_origin).length, 1.0e-7)
            advance = min(max(hit_distance * 1.0e-6, 1.0e-7), 1.0e-3)
            local_origin = location + local_direction * advance
        return None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _fill_preview_signature(obj):
    """Return a cheap identity signature for one preview target.

    Geometry changes are invalidated by the dependency-graph callback below;
    this signature is deliberately limited to ownership, topology counts and
    transform so that a wheel event never performs a full mesh digest.
    """
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


def _fill_preview_adjacency_steps(obj):
    """Yield between large mesh reads used by a modal preview preparation."""
    import numpy as np

    mesh = obj.data
    signature = _fill_preview_signature(obj)
    if signature is None:
        raise RuntimeError("preview target is unavailable")
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.polygons)
    coordinates = np.empty((vertex_count, 3), dtype=np.float32)
    loop_edges = np.empty(len(mesh.loops), dtype=np.int32)
    loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
    totals = np.empty(face_count, dtype=np.int32)
    hidden = np.empty(face_count, dtype=bool)
    yield "allocate"
    mesh.vertices.foreach_get("co", coordinates.ravel())
    yield "vertices"
    mesh.loops.foreach_get("edge_index", loop_edges)
    yield "loop_edges"
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    yield "loop_vertices"
    mesh.polygons.foreach_get("loop_total", totals)
    yield "totals"
    mesh.polygons.foreach_get("hide", hidden)
    yield "hidden"
    centers = np.empty((face_count, 3), dtype=np.float32)
    normals = np.empty((face_count, 3), dtype=np.float32)
    mesh.polygons.foreach_get("center", centers.ravel())
    yield "centers"
    mesh.polygons.foreach_get("normal", normals.ravel())
    yield "normals"
    return {
        "signature": signature,
        "coordinates": coordinates,
        "loop_edges": loop_edges,
        "loop_vertices": loop_vertices,
        "totals": totals,
        "hidden": hidden,
        "centers": centers,
        "normals": normals,
    }


def _fill_preview_cursor_initial_radius(obj, cache, seed_face, geometry=None):
    """Choose the existing model-scaled radius without a full graph.

    A provisional edge-length estimate is used only to select the first crop.
    Once that compact graph exists, the exact value uses the seed's adjacent
    face-center distances, matching the previous full-graph path.
    """
    import numpy as np

    mesh = obj.data
    seed_face = int(seed_face)
    matrix = obj.matrix_world
    edge_lengths = []
    if geometry is not None:
        edge_lengths = list(geometry.get("seed_neighbor_lengths", ()))
    if not edge_lengths:
        # Recover the seed's existing face-center scale from cached loop-edge
        # ids.  This avoids endpoint RNA reads and keeps the first crop close
        # to the exact center-neighbor radius used after graph preparation.
        loop_edges = np.asarray(cache.get("loop_edges", ()), dtype=np.int32)
        totals = np.asarray(cache.get("totals", ()), dtype=np.int32)
        if len(loop_edges) and len(totals):
            starts = np.r_[0, np.cumsum(totals[:-1], dtype=np.int64)]
            seed_loops = np.arange(
                int(starts[seed_face]),
                int(starts[seed_face] + totals[seed_face]),
                dtype=np.int64,
            )
            seed_edges = np.unique(loop_edges[seed_loops])
            matching_loops = np.flatnonzero(np.isin(loop_edges, seed_edges))
            matching_faces = np.searchsorted(starts, matching_loops, side="right") - 1
            other_faces = matching_faces[matching_faces != seed_face]
            if len(other_faces):
                edge_lengths = np.linalg.norm(
                    cache["world_centers"][other_faces]
                    - cache["world_centers"][seed_face],
                    axis=1,
                ).tolist()
        if not edge_lengths:
            polygon = mesh.polygons[seed_face]
            world_points = [matrix @ mesh.vertices[int(vertex)].co for vertex in polygon.vertices]
            edge_lengths = [
                (world_points[index] - world_points[(index + 1) % len(world_points)]).length
                for index in range(len(world_points))
            ]
    local_scale = float(np.median(edge_lengths)) if edge_lengths else 0.0
    extent = np.ptp(cache["world_centers"], axis=0)
    diagonal = float(np.linalg.norm(extent))
    if not diagonal:
        diagonal = max(local_scale, 1.0e-3)
    return max(
        min(max(local_scale * 8.0, diagonal * 0.01), diagonal * 0.20),
        diagonal * 1.0e-4,
        1.0e-8,
    )


def _fill_preview_cursor_build_shading_geometry(obj, cache, spatial_radius):
    """Build the fixed-light cursor graph with bulk face reads.

    Centers, loop edge ids, and loop totals are already cached.  Manifold
    adjacency is recovered from those arrays without per-loop RNA calls;
    vertex endpoints are fetched only for globally open edges that may form a
    validated seam.  Draw-time endpoint reads are deferred until a boundary
    segment is actually emitted.
    """
    import numpy as np

    mesh = obj.data
    centers_all = cache["world_centers"]
    seed_face = int(cache["seed_face"])
    seed_center = centers_all[seed_face]
    spatial_ids = np.flatnonzero(
        np.linalg.norm(centers_all - seed_center, axis=1) <= float(spatial_radius)
    ).astype(np.int32)
    hidden_all = np.empty(len(mesh.polygons), dtype=bool)
    normals_all = np.empty((len(mesh.polygons), 3), dtype=np.float32)
    mesh.polygons.foreach_get("hide", hidden_all)
    mesh.polygons.foreach_get("normal", normals_all.ravel())
    face_ids = spatial_ids[~hidden_all[spatial_ids]].astype(np.int32, copy=False)
    if seed_face not in set(int(face) for face in face_ids):
        raise RuntimeError("preview seed is hidden")
    totals = np.asarray(cache["totals"], dtype=np.int32)
    loop_edges = np.asarray(cache["loop_edges"], dtype=np.int32)
    face_starts = np.r_[0, np.cumsum(totals[:-1], dtype=np.int64)]
    local_counts = totals[face_ids]
    face_count = int(len(face_ids))
    local_face = np.repeat(np.arange(face_count, dtype=np.int32), local_counts)
    local_start = np.repeat(
        np.r_[0, np.cumsum(local_counts[:-1], dtype=np.int64)], local_counts
    )
    local_loops = (
        np.repeat(face_starts[face_ids], local_counts)
        + np.arange(int(np.sum(local_counts)), dtype=np.int64)
        - local_start
    )
    local_global_faces = face_ids[local_face]
    local_next_loops = face_starts[local_global_faces] + (
        (local_loops - face_starts[local_global_faces] + 1)
        % np.maximum(totals[local_global_faces], 1)
    )
    edge_values = loop_edges[local_loops]
    order = np.argsort(edge_values, kind="stable")
    sorted_edges = edge_values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_edges)) + 1]
    lengths = np.diff(np.r_[starts, len(order)])
    group_edges = sorted_edges[starts]
    edge_degree = np.asarray(cache["edge_degree"], dtype=np.int32)
    manifold = (lengths == 2) & (edge_degree[group_edges] == 2)
    pair_starts = starts[manifold]
    first_loop = order[pair_starts]
    second_loop = order[pair_starts + 1]
    first = local_face[first_loop]
    second = local_face[second_loop]
    keep = first != second
    first = first[keep].astype(np.int32, copy=False)
    second = second[keep].astype(np.int32, copy=False)
    pair_edges = group_edges[manifold][keep].astype(np.int32, copy=False)
    pair_kind = np.zeros(len(first), dtype=np.int8)

    boundary_mask = (lengths == 1) & (edge_degree[group_edges] == 1)
    boundary_positions = order[starts[boundary_mask]]
    boundary_faces = local_face[boundary_positions]
    boundary_edges = group_edges[boundary_mask].astype(np.int32, copy=False)
    boundary_records = []
    for local_position, face, edge_index in zip(
        boundary_positions, boundary_faces, boundary_edges
    ):
        loop_index = int(local_loops[int(local_position)])
        next_loop_index = int(local_next_loops[int(local_position)])
        v0 = int(mesh.loops[loop_index].vertex_index)
        v1 = int(mesh.loops[next_loop_index].vertex_index)
        p0 = np.asarray(obj.matrix_world @ mesh.vertices[v0].co, dtype=np.float64)
        p1 = np.asarray(obj.matrix_world @ mesh.vertices[v1].co, dtype=np.float64)
        boundary_records.append((int(face), int(edge_index), v0, v1, p0, p1))

    seam_records = []
    seam_count = 0
    if boundary_records:
        lengths_world = [
            float(np.linalg.norm(record[5] - record[4]))
            for record in boundary_records
        ]
        tolerance = max(float(np.median(lengths_world)) * 1.0e-4, 1.0e-12)
        buckets = {}
        for record in boundary_records:
            _face, _edge, _v0, _v1, p0, p1 = record
            qa = tuple(np.rint(p0 / tolerance).astype(np.int64))
            qb = tuple(np.rint(p1 / tolerance).astype(np.int64))
            buckets.setdefault(tuple(sorted((qa, qb))), []).append(record)
        normals = normals_all[face_ids].astype(np.float64) @ np.linalg.inv(
            np.asarray(obj.matrix_world, dtype=np.float64)[:3, :3]
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-20)
        for records in buckets.values():
            if len(records) != 2:
                continue
            first_record, second_record = records
            first_face, first_edge, _v0, _v1, first_a, first_b = first_record
            second_face, second_edge, _v2, _v3, second_a, second_b = second_record
            if (
                first_face == second_face
                or np.linalg.norm(first_a - second_b) > tolerance
                or np.linalg.norm(first_b - second_a) > tolerance
                or float(np.dot(normals[first_face], normals[second_face])) <= 0.5
            ):
                continue
            seam_records.append((first_face, second_face, first_edge))
            seam_count += 1
    if seam_records:
        seam_first = np.asarray([record[0] for record in seam_records], dtype=np.int32)
        seam_second = np.asarray([record[1] for record in seam_records], dtype=np.int32)
        seam_edges = np.asarray([record[2] for record in seam_records], dtype=np.int32)
        first = np.r_[first, seam_first]
        second = np.r_[second, seam_second]
        pair_edges = np.r_[pair_edges, seam_edges]
        pair_kind = np.r_[pair_kind, np.ones(len(seam_records), dtype=np.int8)]

    centers = centers_all[face_ids].astype(np.float64, copy=True)
    transform = np.asarray(obj.matrix_world, dtype=np.float64)[:3, :3]
    normals = normals_all[face_ids].astype(np.float64) @ np.linalg.inv(transform)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-20)
    delta = centers[second] - centers[first]
    pair_lengths = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-20)
    sources = np.r_[first, second]
    destinations = np.r_[second, first]
    directed_lengths = np.r_[pair_lengths, pair_lengths]
    directed_edges = np.r_[pair_edges, pair_edges]
    graph_order = np.argsort(sources, kind="stable")
    degree = np.bincount(sources, minlength=face_count)
    offsets = np.r_[0, np.cumsum(degree)].astype(np.int64)
    return {
        "signature": cache["signature"],
        "count": face_count,
        "face_edge_counts": local_counts.astype(np.int32, copy=False),
        "centers": centers,
        "normals": normals,
        "hidden": np.zeros(face_count, dtype=bool),
        "world_vertices": np.empty((0, 3), dtype=np.float64),
        "offsets": offsets,
        "neighbors": destinations[graph_order].astype(np.int32, copy=False),
        "neighbor_lengths": directed_lengths[graph_order],
        "edge_v0": directed_edges[graph_order].astype(np.int32, copy=False),
        "edge_v1": directed_edges[graph_order].astype(np.int32, copy=False),
        "edge_indices": directed_edges[graph_order].astype(np.int32, copy=False),
        "first": first,
        "second": second,
        "pair_lengths": pair_lengths,
        "pair_v0": pair_edges,
        "pair_v1": pair_edges,
        "pair_edge_indices": pair_edges,
        "pair_kind": pair_kind,
        "seam_count": int(seam_count),
        "face_ids": face_ids,
        "spatial_radius": float(spatial_radius),
        "spatial_faces": int(len(spatial_ids)),
        "seed_neighbor_lengths": pair_lengths[
            (first == int(np.flatnonzero(face_ids == seed_face)[0]))
            | (second == int(np.flatnonzero(face_ids == seed_face)[0]))
        ],
        "cursor_local": True,
        "shading_approx": True,
    }


def _fill_preview_cursor_build_geometry(obj, cache, spatial_radius):
    """Build a compact graph from a cursor-centered spatial crop.

    The full center and global edge-degree arrays are the only full-mesh reads.
    Polygon RNA is read only for faces inside the crop.  A crop boundary is
    never treated as an open seam: seam bridges are considered only for edges
    whose global degree is one and whose two open counterparts are both in the
    crop.
    """
    import numpy as np

    mesh = obj.data
    centers_all = cache["world_centers"]
    seed_face = int(cache["seed_face"])
    seed_center = centers_all[seed_face]
    spatial_ids = np.flatnonzero(
        np.linalg.norm(centers_all - seed_center, axis=1)
        <= float(spatial_radius)
    ).astype(np.int32)
    visible_ids = [
        int(face_id)
        for face_id in spatial_ids
        if not bool(mesh.polygons[int(face_id)].hide)
    ]
    if seed_face not in visible_ids:
        raise RuntimeError("preview seed is hidden")
    face_ids = np.asarray(visible_ids, dtype=np.int32)
    local_index = {int(face_id): index for index, face_id in enumerate(face_ids)}
    matrix = obj.matrix_world
    transform = np.asarray(matrix.to_3x3(), dtype=np.float64)
    inverse_transform = np.linalg.inv(transform)
    centers = centers_all[face_ids].astype(np.float64, copy=True)
    normals = np.empty((len(face_ids), 3), dtype=np.float64)
    vertex_index = {}
    vertex_world = []
    edge_records = {}
    boundary_records = []
    edge_degree = cache["edge_degree"]

    def local_vertex(global_vertex):
        global_vertex = int(global_vertex)
        local = vertex_index.get(global_vertex)
        if local is None:
            local = len(vertex_world)
            vertex_index[global_vertex] = local
            vertex_world.append(
                tuple(float(value) for value in (matrix @ mesh.vertices[global_vertex].co))
            )
        return local

    for local_face, global_face in enumerate(face_ids):
        polygon = mesh.polygons[int(global_face)]
        raw_normal = np.asarray(polygon.normal, dtype=np.float64)
        world_normal = raw_normal @ inverse_transform
        normals[local_face] = world_normal / max(
            float(np.linalg.norm(world_normal)), 1.0e-20
        )
        loop_indices = tuple(polygon.loop_indices)
        for loop_position, loop_index in enumerate(loop_indices):
            edge_index = int(mesh.loops[int(loop_index)].edge_index)
            v0_global = int(mesh.loops[int(loop_index)].vertex_index)
            v1_global = int(
                mesh.loops[int(loop_indices[(loop_position + 1) % len(loop_indices)])].vertex_index
            )
            v0 = local_vertex(v0_global)
            v1 = local_vertex(v1_global)
            record = (local_face, v0, v1, (v0_global, v1_global))
            if int(edge_degree[edge_index]) == 2:
                edge_records.setdefault(edge_index, []).append(record)
            elif int(edge_degree[edge_index]) == 1:
                boundary_records.append(record)

    pair_records = []
    pair_kind = []
    pair_seen = set()
    for edge_index, records in edge_records.items():
        if len(records) != 2:
            continue
        first_record, second_record = records
        first_face, first_v0, first_v1, _first_key = first_record
        second_face, second_v0, second_v1, _second_key = second_record
        if first_face == second_face:
            continue
        pair_key = (int(edge_index), min(first_face, second_face), max(first_face, second_face))
        if pair_key in pair_seen:
            continue
        pair_seen.add(pair_key)
        pair_records.append((first_face, second_face, first_v0, first_v1))
        pair_kind.append(0)

    seam_count = 0
    if boundary_records:
        lengths = [
            float(
                np.linalg.norm(
                    np.asarray(vertex_world[v1]) - np.asarray(vertex_world[v0])
                )
            )
            for _face, v0, v1, _key in boundary_records
        ]
        tolerance = max(float(np.median(lengths)) * 1.0e-4, 1.0e-12)
        buckets = {}
        for record in boundary_records:
            face, v0, v1, _key = record
            point_a = np.asarray(vertex_world[v0], dtype=np.float64)
            point_b = np.asarray(vertex_world[v1], dtype=np.float64)
            qa = tuple(np.rint(point_a / tolerance).astype(np.int64))
            qb = tuple(np.rint(point_b / tolerance).astype(np.int64))
            buckets.setdefault(tuple(sorted((qa, qb))), []).append(record)
        for records in buckets.values():
            if len(records) != 2:
                continue
            first_record, second_record = records
            first_face, first_v0, first_v1, _first_key = first_record
            second_face, second_v0, second_v1, _second_key = second_record
            if first_face == second_face:
                continue
            first_a = np.asarray(vertex_world[first_v0], dtype=np.float64)
            first_b = np.asarray(vertex_world[first_v1], dtype=np.float64)
            second_a = np.asarray(vertex_world[second_v0], dtype=np.float64)
            second_b = np.asarray(vertex_world[second_v1], dtype=np.float64)
            if (
                np.linalg.norm(first_a - second_b) > tolerance
                or np.linalg.norm(first_b - second_a) > tolerance
                or float(np.dot(normals[first_face], normals[second_face])) <= 0.5
            ):
                continue
            pair_key = ("seam", min(first_face, second_face), max(first_face, second_face), first_v0, first_v1)
            if pair_key in pair_seen:
                continue
            pair_seen.add(pair_key)
            pair_records.append((first_face, second_face, first_v0, first_v1))
            pair_kind.append(1)
            seam_count += 1

    if pair_records:
        first = np.asarray([record[0] for record in pair_records], dtype=np.int32)
        second = np.asarray([record[1] for record in pair_records], dtype=np.int32)
        pair_v0 = np.asarray([record[2] for record in pair_records], dtype=np.int32)
        pair_v1 = np.asarray([record[3] for record in pair_records], dtype=np.int32)
    else:
        first = second = pair_v0 = pair_v1 = np.empty(0, dtype=np.int32)
        pair_kind = np.empty(0, dtype=np.int8)
    if pair_records:
        pair_kind = np.asarray(pair_kind, dtype=np.int8)
    delta = centers[second] - centers[first]
    pair_lengths = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-20)
    sources = np.r_[first, second]
    destinations = np.r_[second, first]
    directed_lengths = np.r_[pair_lengths, pair_lengths]
    directed_v0 = np.r_[pair_v0, pair_v0]
    directed_v1 = np.r_[pair_v1, pair_v1]
    order = np.argsort(sources, kind="stable")
    degree = np.bincount(sources, minlength=len(face_ids))
    offsets = np.r_[0, np.cumsum(degree)].astype(np.int64)
    seed_local = local_index[seed_face]
    seed_neighbor_lengths = pair_lengths[
        (first == seed_local) | (second == seed_local)
    ]
    return {
        "signature": cache["signature"],
        "count": int(len(face_ids)),
        "face_edge_counts": totals[face_ids].astype(np.int32, copy=False),
        "centers": centers,
        "normals": normals,
        "hidden": np.zeros(len(face_ids), dtype=bool),
        "world_vertices": np.asarray(vertex_world, dtype=np.float64),
        "offsets": offsets,
        "neighbors": destinations[order].astype(np.int32, copy=False),
        "neighbor_lengths": directed_lengths[order],
        "edge_v0": directed_v0[order].astype(np.int32, copy=False),
        "edge_v1": directed_v1[order].astype(np.int32, copy=False),
        "first": first,
        "second": second,
        "pair_lengths": pair_lengths,
        "pair_v0": pair_v0,
        "pair_v1": pair_v1,
        "pair_kind": pair_kind,
        "seam_count": int(seam_count),
        "face_ids": face_ids,
        "spatial_radius": float(spatial_radius),
        "spatial_faces": int(len(spatial_ids)),
        "seed_neighbor_lengths": seed_neighbor_lengths,
        "cursor_local": True,
    }


def _fill_preview_cursor_prepare_steps(obj, seed_face):
    """Yield cursor-first reads before returning one compact adjacency graph."""
    import numpy as np

    global _fill_preview_cursor_cache
    mesh = obj.data
    signature = _fill_preview_signature(obj)
    if signature is None:
        raise RuntimeError("preview target is unavailable")
    cache = _fill_preview_cursor_cache.get(signature)
    if cache is None:
        face_count = len(mesh.polygons)
        centers_local = np.empty((face_count, 3), dtype=np.float32)
        totals = np.empty(face_count, dtype=np.int32)
        loop_edges = np.empty(len(mesh.loops), dtype=np.int32)
        yield "cursor-allocate"
        mesh.polygons.foreach_get("center", centers_local.ravel())
        matrix = np.asarray(obj.matrix_world, dtype=np.float64)
        world_centers = centers_local.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
        yield "cursor-centers"
        mesh.polygons.foreach_get("loop_total", totals)
        yield "cursor-totals"
        mesh.loops.foreach_get("edge_index", loop_edges)
        edge_degree = np.bincount(loop_edges, minlength=len(mesh.edges)).astype(np.int32, copy=False)
        cache = {
            "signature": signature,
            "world_centers": world_centers,
            "edge_degree": edge_degree,
            "loop_edges": loop_edges,
            "totals": totals,
        }
        _fill_preview_cursor_cache[signature] = cache
        yield "cursor-edge-degree"
    else:
        yield "cursor-cache"
    cache["seed_face"] = int(seed_face)
    provisional_radius = _fill_preview_cursor_initial_radius(obj, cache, seed_face)
    # Prepare only the initial preview crop.  Wider wheel ranges are extended
    # on demand so the first Ready time does not hide a 4x cold preparation.
    spatial_radius = max(
        provisional_radius * 1.5,
        provisional_radius * 1.0,
    )
    geometry = _fill_preview_cursor_build_shading_geometry(
        obj, cache, spatial_radius
    )
    initial_radius = _fill_preview_cursor_initial_radius(
        obj, cache, seed_face, geometry=geometry
    )
    required_radius = max(
        initial_radius * 1.5,
        initial_radius * 1.0,
    )
    if required_radius > spatial_radius * (1.0 + 1.0e-9):
        geometry = _fill_preview_cursor_build_shading_geometry(
            obj, cache, required_radius
        )
    geometry["cursor_cache"] = cache
    geometry["initial_radius"] = float(initial_radius)
    yield "cursor-local-graph"
    return geometry


def _fill_preview_build_adjacency(obj, prepared=None):
    """Build only the reusable surface graph needed by preview sessions.

    This preparation reads all loops once, but intentionally does not compute
    curvature, valley bands, partition ownership, or contour relaxation.  All
    geometry features are recomputed on the session's candidate+halo patch.
    """
    import numpy as np

    mesh = obj.data
    signature = _fill_preview_signature(obj)
    if signature is None:
        raise RuntimeError("preview target is unavailable")
    cached = _fill_preview_adjacency_cache.get(signature)
    if cached is not None:
        return cached

    if prepared is None:
        vertex_count = len(mesh.vertices)
        face_count = len(mesh.polygons)
        coordinates = np.empty((vertex_count, 3), dtype=np.float32)
        loop_edges = np.empty(len(mesh.loops), dtype=np.int32)
        loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
        totals = np.empty(face_count, dtype=np.int32)
        hidden = np.empty(face_count, dtype=bool)
        mesh.vertices.foreach_get("co", coordinates.ravel())
        mesh.loops.foreach_get("edge_index", loop_edges)
        mesh.loops.foreach_get("vertex_index", loop_vertices)
        mesh.polygons.foreach_get("loop_total", totals)
        mesh.polygons.foreach_get("hide", hidden)
        centers = np.empty((face_count, 3), dtype=np.float32)
        normals = np.empty((face_count, 3), dtype=np.float32)
        mesh.polygons.foreach_get("center", centers.ravel())
        mesh.polygons.foreach_get("normal", normals.ravel())
    else:
        coordinates = prepared["coordinates"]
        loop_edges = prepared["loop_edges"]
        loop_vertices = prepared["loop_vertices"]
        totals = prepared["totals"]
        hidden = prepared["hidden"]
        centers = prepared["centers"]
        normals = prepared["normals"]
        vertex_count = len(coordinates)
        face_count = len(totals)

    transform = np.asarray(obj.matrix_world.to_3x3(), dtype=np.float64)
    world_matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    translation = world_matrix[:3, 3]
    world_vertices = coordinates.astype(np.float64) @ transform.T + translation
    centers = centers.astype(np.float64) @ transform.T + translation
    normals = normals.astype(np.float64) @ np.linalg.inv(transform)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-20)

    face_ids = np.repeat(np.arange(face_count, dtype=np.int32), totals)
    order = np.argsort(loop_edges, kind="stable")
    sorted_edges = loop_edges[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_edges)) + 1]
    lengths = np.diff(np.r_[starts, len(order)])
    pair_starts = starts[lengths == 2]
    paired_loops = order[pair_starts]
    first = face_ids[paired_loops]
    second = face_ids[order[pair_starts + 1]]
    face_starts = np.r_[0, np.cumsum(totals)[:-1]]
    paired_next = face_starts[first] + (
        (paired_loops - face_starts[first] + 1) % totals[first]
    )
    edge_v0 = loop_vertices[paired_loops]
    edge_v1 = loop_vertices[paired_next]

    # Preserve the existing unwelded-seam behavior.  Only coincident open
    # edges with opposite winding and compatible normals become bridges.
    border_loops = order[starts[lengths == 1]]
    seam_count = 0
    if len(border_loops):
        border_faces = face_ids[border_loops]
        next_loops = face_starts[border_faces] + (
            (border_loops - face_starts[border_faces] + 1) % totals[border_faces]
        )
        points_a = world_vertices[loop_vertices[border_loops]]
        points_b = world_vertices[loop_vertices[next_loops]]
        edge_lengths = np.linalg.norm(points_b - points_a, axis=1)
        tolerance = max(float(np.median(edge_lengths)) * 1.0e-4, 1.0e-12)
        quantized = np.rint(np.vstack((points_a, points_b)) / tolerance).astype(np.int64)
        _, point_ids = np.unique(quantized, axis=0, return_inverse=True)
        pa, pb = np.split(point_ids, 2)
        signatures = np.column_stack((np.minimum(pa, pb), np.maximum(pa, pb)))
        _, inverse, counts = np.unique(
            signatures, axis=0, return_inverse=True, return_counts=True
        )
        seam_order = np.argsort(inverse, kind="stable")
        seam_starts = np.r_[0, np.cumsum(counts)[:-1]][counts == 2]
        ia, ib = seam_order[seam_starts], seam_order[seam_starts + 1]
        fa, fb = border_faces[ia], border_faces[ib]
        match = (
            (pa[ia] == pb[ib])
            & (pb[ia] == pa[ib])
            & (pa[ia] != pb[ia])
            & (fa != fb)
            & (np.sum(normals[fa] * normals[fb], axis=1) > 0.5)
            & (np.linalg.norm(points_a[ia] - points_b[ib], axis=1) <= tolerance)
            & (np.linalg.norm(points_b[ia] - points_a[ib], axis=1) <= tolerance)
        )
        first = np.r_[first, fa[match]]
        second = np.r_[second, fb[match]]
        edge_v0 = np.r_[edge_v0, loop_vertices[border_loops[ia[match]]]]
        edge_v1 = np.r_[edge_v1, loop_vertices[next_loops[ia[match]]]]
        seam_count = int(np.count_nonzero(match))

    valid = ~(hidden[first] | hidden[second]) & (first != second)
    first, second = first[valid], second[valid]
    edge_v0, edge_v1 = edge_v0[valid], edge_v1[valid]
    delta = centers[second] - centers[first]
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-20)
    sources = np.r_[first, second]
    destinations = np.r_[second, first]
    source_edges = np.r_[np.arange(len(first)), np.arange(len(first))]
    edge_v0_directed = np.r_[edge_v0, edge_v0]
    edge_v1_directed = np.r_[edge_v1, edge_v1]
    order = np.argsort(sources, kind="stable")
    degree = np.bincount(sources, minlength=face_count)
    offsets = np.r_[0, np.cumsum(degree)].astype(np.int64)
    cached = {
        "signature": signature,
        "count": int(face_count),
        "face_edge_counts": totals.astype(np.int32, copy=False),
        "centers": centers,
        "normals": normals,
        "hidden": hidden,
        "world_vertices": world_vertices,
        "offsets": offsets,
        "neighbors": destinations[order].astype(np.int32, copy=False),
        "neighbor_lengths": np.r_[distance, distance][order],
        "edge_v0": edge_v0_directed[order].astype(np.int32, copy=False),
        "edge_v1": edge_v1_directed[order].astype(np.int32, copy=False),
        "first": first,
        "second": second,
        "pair_lengths": distance,
        "pair_v0": edge_v0,
        "pair_v1": edge_v1,
        "seam_count": seam_count,
    }
    # Drop obsolete revisions for this mesh while retaining unrelated meshes.
    mesh_pointer = signature[1]
    for key in list(_fill_preview_adjacency_cache):
        if key[1] == mesh_pointer and key != signature:
            _fill_preview_adjacency_cache.pop(key, None)
    _fill_preview_adjacency_cache[signature] = cached
    return cached


def _fill_preview_dijkstra(geometry, seed_face, max_distance):
    """Return surface distances and ids within one candidate+halo radius."""
    import heapq
    import numpy as np

    count = int(geometry["count"])
    distances = np.full(count, np.inf, dtype=np.float64)
    seed_face = int(seed_face)
    distances[seed_face] = 0.0
    heap = [(0.0, seed_face)]
    offsets = geometry["offsets"]
    neighbors = geometry["neighbors"]
    lengths = geometry["neighbor_lengths"]
    popped = 0
    while heap:
        current, face = heapq.heappop(heap)
        if current > float(distances[face]) + 1.0e-15:
            continue
        if current > float(max_distance):
            break
        popped += 1
        for edge in range(int(offsets[face]), int(offsets[face + 1])):
            other = int(neighbors[edge])
            candidate = current + float(lengths[edge])
            if candidate < distances[other] and candidate <= float(max_distance):
                distances[other] = candidate
                heapq.heappush(heap, (candidate, other))
    return distances, np.flatnonzero(np.isfinite(distances)).astype(np.int32), popped


def _fill_preview_dijkstra_incremental(geometry, seed_face, max_distance, state):
    """Extend one Dijkstra frontier and reuse it for shrink/revisit events."""
    import heapq
    import numpy as np

    if state is None:
        count = int(geometry["count"])
        distances = np.full(count, np.inf, dtype=np.float64)
        distances[int(seed_face)] = 0.0
        state = {
            "distances": distances,
            "heap": [(0.0, int(seed_face))],
            "max_distance": -1.0,
            "popped": 0,
        }
    if float(max_distance) > float(state["max_distance"]):
        offsets = geometry["offsets"]
        neighbors = geometry["neighbors"]
        lengths = geometry["neighbor_lengths"]
        distances = state["distances"]
        heap = state["heap"]
        while heap:
            current, face = heapq.heappop(heap)
            if current > float(distances[face]) + 1.0e-15:
                continue
            if current > float(max_distance):
                heapq.heappush(heap, (current, face))
                break
            state["popped"] += 1
            for edge in range(int(offsets[face]), int(offsets[face + 1])):
                other = int(neighbors[edge])
                candidate = current + float(lengths[edge])
                # Keep the first frontier beyond the current radius so an
                # expanded wheel distance can continue without restarting.
                if candidate < distances[other]:
                    distances[other] = candidate
                    heapq.heappush(heap, (candidate, other))
        state["max_distance"] = float(max_distance)
    distances = state["distances"]
    ids = np.flatnonzero(distances <= float(max_distance)).astype(np.int32)
    return distances, ids, int(state["popped"]), state


def _fill_preview_shading_proxy(geometry, patch_ids, distances, target_radius, seed_local):
    """Choose a coarse candidate plus a narrow correction band.

    The cursor graph already contains centers, normals, and manifold pairs.
    Fixed Lambert samples are evaluated only on the current distance patch.
    The signed valley barriers then split the original face graph, and the
    seed-connected component is retained.  The returned ids deliberately
    include one graph ring and adjacent signed barriers so the existing exact
    partition can inspect the original boundary without deciding the radius.
    """
    import time
    import numpy as np

    started = time.perf_counter()
    patch_ids = np.asarray(patch_ids, dtype=np.int32)
    count = int(len(patch_ids))
    if count == 0:
        return patch_ids, {"proxy_seconds": 0.0, "proxy_nodes": 0}
    global_count = int(geometry["count"])
    local_id = np.full(global_count, -1, dtype=np.int32)
    local_id[patch_ids] = np.arange(count, dtype=np.int32)
    graph_first = np.asarray(geometry["first"], dtype=np.int32)
    graph_second = np.asarray(geometry["second"], dtype=np.int32)
    keep = (local_id[graph_first] >= 0) & (local_id[graph_second] >= 0)
    first = local_id[graph_first[keep]]
    second = local_id[graph_second[keep]]
    pair_lengths = np.asarray(geometry["pair_lengths"], dtype=np.float64)[keep]
    if len(first) == 0:
        return patch_ids, {
            "proxy_seconds": float(time.perf_counter() - started),
            "proxy_nodes": count,
            "proxy_candidate_faces": count,
            "proxy_band_faces": count,
            "proxy_barrier_edges": 0,
        }

    lights = np.asarray(
        (
            (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
            (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    lights /= np.maximum(np.linalg.norm(lights, axis=1, keepdims=True), 1.0e-20)
    normals = np.asarray(geometry["normals"], dtype=np.float64)[patch_ids]
    centers = np.asarray(geometry["centers"], dtype=np.float64)[patch_ids]
    shading = np.clip(normals @ lights.T, 0.0, 1.0)
    delta = centers[second] - centers[first]
    lengths = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-20)
    direction = delta / lengths[:, None]
    shade_delta = shading[second] - shading[first]
    signed_turn = np.mean(shade_delta * (direction @ lights.T), axis=1)
    local_scale = max(float(np.median(pair_lengths)), 1.0e-12)
    normalized_turn = -signed_turn / np.maximum(pair_lengths / local_scale, 1.0e-12)
    face_valley = np.zeros(count, dtype=np.float64)
    np.maximum.at(face_valley, first, np.maximum(normalized_turn, 0.0))
    np.maximum.at(face_valley, second, np.maximum(normalized_turn, 0.0))
    face_degree = np.bincount(first, minlength=count) + np.bincount(second, minlength=count)
    for _ in range(2):
        face_valley = (
            face_valley
            + np.bincount(first, weights=face_valley[second], minlength=count)
            + np.bincount(second, weights=face_valley[first], minlength=count)
        ) / np.maximum(face_degree + 1, 1)
    center = float(np.median(face_valley))
    mad = float(np.median(np.abs(face_valley - center)))
    valley_threshold = max(0.025, center + 2.0 * mad)
    edge_valley = np.maximum(face_valley[first], face_valley[second])
    barrier = edge_valley >= valley_threshold
    pair_kind = np.asarray(geometry.get("pair_kind", np.zeros(len(first), dtype=np.int8)), dtype=np.int8)
    if len(pair_kind) == len(graph_first):
        pair_kind = pair_kind[keep]
    else:
        pair_kind = np.zeros(len(first), dtype=np.int8)

    # Split the original face graph at the signed valley barriers.  True seams
    # remain traversable here; they are not geometric valley barriers.  This
    # component uses only the requested target radius.  The halo is retained
    # for valley detection and fine boundary inspection, but it cannot provide
    # a route around a valley that lies outside the requested display range.
    patch_distances = np.asarray(distances, dtype=np.float64)[patch_ids]
    target_mask = np.isfinite(patch_distances) & (
        patch_distances <= float(target_radius) + 1.0e-9
    )
    parent = np.arange(count, dtype=np.int32)
    sizes = np.ones(count, dtype=np.int32)

    def find(value):
        value = int(value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for edge, (left, right) in enumerate(zip(first, second)):
        if barrier[edge] or not (target_mask[left] and target_mask[right]):
            continue
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]
    roots = np.asarray([find(index) for index in range(count)], dtype=np.int32)
    seed_local = int(seed_local)
    seed_root = int(roots[seed_local])
    candidate = (roots == seed_root) & target_mask
    if not np.any(candidate):
        candidate[seed_local] = True

    # One unrestricted ring supplies the fine correction band.  The band may
    # contain the opposite side so the exact local routine can inspect the
    # original faces, but the final candidate ids stay in the seed component.
    band = candidate.copy()
    adjacent = candidate[first] | candidate[second]
    band[first[adjacent]] = True
    band[second[adjacent]] = True
    barrier_adjacent = barrier & adjacent
    band[first[barrier_adjacent]] = True
    band[second[barrier_adjacent]] = True
    band[seed_local] = True
    frontier = np.isfinite(patch_distances) & (
        np.abs(patch_distances - float(target_radius))
        <= max(local_scale * 2.0, 1.0e-9)
    )
    band |= frontier
    selected = np.flatnonzero(band).astype(np.int32)
    proxy_candidate_ids = patch_ids[np.flatnonzero(candidate)].astype(np.int32, copy=False)
    return patch_ids[selected], {
        "proxy_seconds": float(time.perf_counter() - started),
        "proxy_nodes": count,
        "proxy_candidate_faces": int(np.count_nonzero(candidate)),
        "proxy_component_faces": int(np.count_nonzero(candidate)),
        "proxy_correction_faces": 0,
        "proxy_band_faces": int(len(selected)),
        "proxy_barrier_edges": int(np.count_nonzero(barrier)),
        "proxy_valley_threshold": float(valley_threshold),
        "proxy_candidate_ids": proxy_candidate_ids,
    }


def _fill_preview_local_geometry(geometry, face_ids):
    """Slice adjacency and recompute all scalar features on the local patch."""
    import numpy as np

    face_ids = np.asarray(face_ids, dtype=np.int32)
    count = int(geometry["count"])
    local_id = np.full(count, -1, dtype=np.int32)
    local_id[face_ids] = np.arange(len(face_ids), dtype=np.int32)
    offsets_global = geometry["offsets"]
    degree = offsets_global[face_ids + 1] - offsets_global[face_ids]
    total = int(np.sum(degree))
    if total:
        starts = np.repeat(offsets_global[face_ids], degree)
        segment_starts = np.repeat(np.r_[0, np.cumsum(degree)[:-1]], degree)
        edge_ids = (
            starts + np.arange(total, dtype=np.int64) - segment_starts
        )
        source_global = np.repeat(face_ids, degree)
        destination_global = geometry["neighbors"][edge_ids]
        keep = (
            (local_id[destination_global] >= 0)
            & (source_global < destination_global)
        )
        first = local_id[source_global[keep]]
        second = local_id[destination_global[keep]]
        pair_lengths = geometry["neighbor_lengths"][edge_ids][keep]
        pair_v0 = geometry["edge_v0"][edge_ids][keep]
        pair_v1 = geometry["edge_v1"][edge_ids][keep]
        pair_edge_indices = geometry.get("edge_indices")
        if pair_edge_indices is not None:
            pair_edge_indices = pair_edge_indices[edge_ids][keep]
        else:
            pair_edge_indices = np.full(len(pair_v0), -1, dtype=np.int32)
    else:
        first = second = np.empty(0, dtype=np.int32)
        pair_lengths = np.empty(0, dtype=np.float64)
        pair_v0 = pair_v1 = np.empty(0, dtype=np.int32)
        pair_edge_indices = np.empty(0, dtype=np.int32)

    sources = np.r_[first, second]
    destinations = np.r_[second, first]
    directed_lengths = np.r_[pair_lengths, pair_lengths]
    directed_v0 = np.r_[pair_v0, pair_v0]
    directed_v1 = np.r_[pair_v1, pair_v1]
    degree = np.bincount(sources, minlength=len(face_ids))
    order = np.argsort(sources, kind="stable")
    offsets = np.r_[0, np.cumsum(degree)].astype(np.int64)
    local = {
        "first": first,
        "second": second,
        "offsets": offsets,
        "centers": geometry["centers"][face_ids],
        "normals": geometry["normals"][face_ids],
        "hidden": geometry["hidden"][face_ids],
        "world_vertices": geometry["world_vertices"],
        "neighbors": destinations[order].astype(np.int32, copy=False),
        "neighbor_lengths": directed_lengths[order],
        "edge_v0": directed_v0[order].astype(np.int32, copy=False),
        "edge_v1": directed_v1[order].astype(np.int32, copy=False),
        "pair_v0": pair_v0.astype(np.int32, copy=False),
        "pair_v1": pair_v1.astype(np.int32, copy=False),
        "pair_edge_indices": pair_edge_indices.astype(np.int32, copy=False),
        "count": int(len(face_ids)),
        "seam_count": int(geometry.get("seam_count", 0)),
        "partitions": {},
        "_global_face_ids": face_ids,
    }
    count = local["count"]
    local["scale"] = (
        np.bincount(first, weights=pair_lengths, minlength=count)
        + np.bincount(second, weights=pair_lengths, minlength=count)
    ) / np.maximum(
        np.bincount(first, minlength=count) + np.bincount(second, minlength=count),
        1,
    )
    if geometry.get("shading_approx"):
        # Fixed, view-independent Lambert samples are used as a coarse shape
        # signal.  Normalize the signed change by local edge scale, aggregate
        # it over two face-neighborhoods, and retain only the concave sign as
        # a barrier.  Convex crests therefore remain traversable.
        lights = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
                (1.0, 1.0, 1.0),
                (-1.0, 1.0, 1.0),
            ),
            dtype=np.float64,
        )
        lights /= np.maximum(np.linalg.norm(lights, axis=1, keepdims=True), 1.0e-20)
        shading = np.clip(local["normals"] @ lights.T, 0.0, 1.0)
        shade_delta = shading[second] - shading[first]
        delta = local["centers"][second] - local["centers"][first]
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-20)
        direction = delta / distance[:, None]
        signed_turn = np.mean(shade_delta * (direction @ lights.T), axis=1)
        local_scale = max(float(np.median(pair_lengths)), 1.0e-12)
        normalized_turn = -signed_turn / np.maximum(pair_lengths / local_scale, 1.0e-12)
        face_valley = np.zeros(count, dtype=np.float64)
        np.maximum.at(face_valley, first, np.maximum(normalized_turn, 0.0))
        np.maximum.at(face_valley, second, np.maximum(normalized_turn, 0.0))
        degree = np.bincount(first, minlength=count) + np.bincount(second, minlength=count)
        for _ in range(2):
            face_valley = (
                face_valley
                + np.bincount(first, weights=face_valley[second], minlength=count)
                + np.bincount(second, weights=face_valley[first], minlength=count)
            ) / np.maximum(degree + 1, 1)
        robust_center = float(np.median(face_valley))
        robust_mad = float(np.median(np.abs(face_valley - robust_center)))
        valley_threshold = max(0.025, robust_center + 2.0 * robust_mad)
        normalized_face_valley = np.maximum(
            face_valley / max(valley_threshold, 1.0e-12) * 0.10,
            0.0,
        )
        edge_valley = np.maximum(
            normalized_face_valley[first], normalized_face_valley[second]
        )
        contour_cost = pair_lengths / (1.0 + (edge_valley / 0.10) ** 2)
        local["contrast"] = np.zeros(count, dtype=np.float64)
        local["concavity"] = normalized_face_valley
        local["directional_valley"] = normalized_face_valley
        local["crease"] = np.zeros(count, dtype=np.float64)
        local["contour_cost"] = np.r_[contour_cost, contour_cost][order]
        local["shading_threshold"] = float(valley_threshold)
        return local
    normals = local["normals"]
    smooth = _fill_average(normals, first, second, count, 2)
    smooth /= np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1.0e-20)
    delta = local["centers"][second] - local["centers"][first]
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-20)
    curvature_edge = np.sum((smooth[second] - smooth[first]) * delta, axis=1)
    inward = np.maximum(-curvature_edge / distance, 0.0) * 3.0
    directional_valley = np.zeros(count)
    np.maximum.at(directional_valley, first, inward)
    np.maximum.at(directional_valley, second, inward)
    directional_valley = _fill_average(
        directional_valley, first, second, count, 3
    )
    metric = (
        np.bincount(first, weights=distance**2, minlength=count)
        + np.bincount(second, weights=distance**2, minlength=count)
    )
    curvature = (
        np.bincount(first, weights=curvature_edge, minlength=count)
        + np.bincount(second, weights=curvature_edge, minlength=count)
    ) / np.maximum(metric, 1.0e-30)
    curvature = _fill_average(curvature, first, second, count, 3)
    broad = _fill_average(curvature, first, second, count, 24)
    contrast = np.where(
        curvature < 0.0,
        np.maximum(broad - curvature, 0.0) * local["scale"] * 6.0,
        0.0,
    )
    concavity = np.maximum(-curvature, 0.0) * local["scale"] * 6.0
    valley_line = _fill_average(
        np.maximum(concavity, directional_valley), first, second, count, 6
    )
    edge_valley = (valley_line[first] + valley_line[second]) * 0.5
    contour_cost = pair_lengths / (1.0 + (edge_valley / 0.10) ** 2)
    raw_angle = np.arccos(
        np.clip(np.sum(normals[first] * normals[second], axis=1), -1, 1)
    )
    raw_turn = np.sum((normals[second] - normals[first]) * delta, axis=1)
    raw_angle = np.where(raw_turn < -distance * 1.0e-6, raw_angle, 0.0)
    crease = np.zeros(count)
    np.maximum.at(crease, first, raw_angle)
    np.maximum.at(crease, second, raw_angle)
    local["contrast"] = contrast
    local["concavity"] = concavity
    local["directional_valley"] = directional_valley
    local["crease"] = crease
    local["contour_cost"] = np.r_[contour_cost, contour_cost][order]
    return local


def _fill_preview_region(local, seed_local, strict_mode):
    """Run the production partition semantics on one local patch."""
    import numpy as np

    partition = _fill_partition(local, strict_mode)
    core, owner = partition["core"], partition["owner"]
    seed_local = int(seed_local)
    root = int(owner[seed_local])
    if root == int(local["count"]):
        return np.asarray([seed_local], dtype=np.int32), partition
    selected = np.zeros(int(local["count"]) + 1, dtype=bool)
    selected[root] = True
    pending = deque([root])
    offsets, neighbors = local["offsets"], local["neighbors"]
    while pending:
        face = pending.popleft()
        for neighbor in neighbors[offsets[face] : offsets[face + 1]]:
            neighbor = int(neighbor)
            if core[neighbor] and not selected[neighbor]:
                selected[neighbor] = True
                pending.append(neighbor)
    region = selected[owner]
    first, second = local["first"], local["second"]
    relaxed = region.copy()
    editable = partition["band"] & ~partition["protected"] & (
        owner < local["count"]
    )
    weights = local["contour_cost"]
    for iteration in range(6):
        crossing = relaxed[first] != relaxed[second]
        fringe = np.unique(np.r_[first[crossing], second[crossing]])
        fringe = fringe[editable[fringe]]
        if iteration % 2:
            fringe = fringe[::-1]
        changed = False
        for face in fringe:
            if int(face) == seed_local:
                continue
            start, end = offsets[face], offsets[face + 1]
            linked = neighbors[start:end]
            costs = weights[start:end]
            old_cost = float(costs[relaxed[linked] != relaxed[face]].sum())
            new_cost = float(costs[relaxed[linked] == relaxed[face]].sum())
            if new_cost < old_cost - max(old_cost, new_cost, 1.0e-20) * 1.0e-8:
                relaxed[face] = not relaxed[face]
                changed = True
        if not changed:
            break
    if relaxed[seed_local]:
        connected = np.zeros(local["count"], dtype=bool)
        connected[root] = True
        pending = deque([root])
        while pending:
            face = pending.popleft()
            for neighbor in neighbors[offsets[face] : offsets[face + 1]]:
                neighbor = int(neighbor)
                if relaxed[neighbor] and not connected[neighbor]:
                    connected[neighbor] = True
                    pending.append(neighbor)
        if connected[seed_local]:
            region = connected
    return np.flatnonzero(region).astype(np.int32), partition


def _fill_preview_initial_radius(geometry, seed_face):
    """Choose a model-scaled starting distance without a fixed world radius."""
    import numpy as np

    seed_face = int(seed_face)
    start, end = geometry["offsets"][seed_face : seed_face + 2]
    if end > start:
        local_scale = float(np.median(geometry["neighbor_lengths"][start:end]))
    else:
        local_scale = 0.0
    extent = np.ptp(geometry["centers"], axis=0)
    diagonal = float(np.linalg.norm(extent))
    if not diagonal:
        diagonal = max(local_scale, 1.0e-3)
    return max(
        min(max(local_scale * 8.0, diagonal * 0.01), diagonal * 0.20),
        diagonal * 1.0e-4,
        1.0e-8,
    )


def _fill_preview_cursor_expand(state, radius):
    """Extend a cursor graph only when the requested radius needs more halo."""
    import numpy as np

    geometry = state["adjacency"]
    if not geometry.get("cursor_local"):
        return False
    halo = max(float(radius) * 0.5, float(state["initial_radius"]) * 0.5)
    required = float(radius) + halo
    if required <= float(geometry.get("spatial_radius", 0.0)) * (1.0 + 1.0e-9):
        return False
    cache = geometry.get("cursor_cache")
    if cache is None:
        raise RuntimeError("preview cursor cache is unavailable")
    cache["seed_face"] = int(state["seed_face"])
    build_geometry = (
        _fill_preview_cursor_build_shading_geometry
        if geometry.get("shading_approx")
        else _fill_preview_cursor_build_geometry
    )
    expanded = build_geometry(state["obj"], cache, required)
    expanded["cursor_cache"] = cache
    expanded["initial_radius"] = float(state["initial_radius"])
    state["adjacency"] = expanded
    state["seed_local"] = int(
        np.flatnonzero(expanded["face_ids"] == int(state["seed_face"]))[0]
    )
    state["distance_state"] = None
    # Result faces are stored in stable mesh-face ids and their draw batches
    # own copied vertices, so prior radii remain valid across a crop growth.
    # Keep only the current result plus the two immediately preceding stages;
    # the timer path performs the bounded eviction after adding a new radius.
    state["expansion_count"] = int(state.get("expansion_count", 0)) + 1
    return True


def _fill_preview_enclosed_gap_faces(geometry, distances, target_radius, selected_ids):
    """Return visible distance-domain components enclosed by the candidate.

    This is a topological completion pass for holes left by a valley band.  It
    only fills an unselected component when it touches the current candidate,
    has no route to the requested-distance boundary, and every source face
    edge is represented by a valid graph pair.  The last condition keeps open
    mesh boundaries, non-manifold edges, hidden neighbors, crop cuts, and
    unmatched seams classified as outside instead of as holes.
    """
    import numpy as np

    count = int(geometry["count"])
    distances = np.asarray(distances, dtype=np.float64)
    if len(distances) != count:
        return np.empty(0, dtype=np.int32), {
            "enclosed_gap_faces": 0,
            "enclosed_gap_components": 0,
            "enclosed_gap_skipped": "distance-size",
        }
    edge_counts = geometry.get("face_edge_counts")
    if edge_counts is None or len(edge_counts) != count:
        return np.empty(0, dtype=np.int32), {
            "enclosed_gap_faces": 0,
            "enclosed_gap_components": 0,
            "enclosed_gap_skipped": "edge-counts-unavailable",
        }
    edge_counts = np.asarray(edge_counts, dtype=np.int32)
    hidden = np.asarray(geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool)
    domain = np.isfinite(distances) & (
        distances <= float(target_radius) + max(float(target_radius) * 1.0e-8, 1.0e-9)
    ) & ~hidden
    selected = np.zeros(count, dtype=bool)
    selected_ids = np.asarray(selected_ids, dtype=np.int32).reshape(-1)
    selected_ids = selected_ids[(selected_ids >= 0) & (selected_ids < count)]
    selected[selected_ids] = True
    missing = domain & ~selected
    if not np.any(missing):
        return np.empty(0, dtype=np.int32), {
            "enclosed_gap_faces": 0,
            "enclosed_gap_components": 0,
        }

    offsets = np.asarray(geometry["offsets"], dtype=np.int64)
    neighbors = np.asarray(geometry["neighbors"], dtype=np.int32)
    degree = np.diff(offsets)
    # A face with an unpaired original edge is connected to a real mesh/crop
    # boundary, non-manifold edge, hidden face, or an unmatched open seam.
    outside = missing & (degree < edge_counts)
    touches_selected = np.zeros(count, dtype=bool)
    for face in np.flatnonzero(missing):
        start, end = int(offsets[face]), int(offsets[face + 1])
        for neighbor in neighbors[start:end]:
            neighbor = int(neighbor)
            if selected[neighbor]:
                touches_selected[face] = True
            elif not domain[neighbor]:
                outside[face] = True

    fills = []
    enclosed_components = 0
    visited = np.zeros(count, dtype=bool)
    for start_face in np.flatnonzero(missing):
        start_face = int(start_face)
        if visited[start_face]:
            continue
        pending = [start_face]
        visited[start_face] = True
        component = []
        reaches_outside = False
        reaches_selected = False
        while pending:
            face = pending.pop()
            component.append(face)
            reaches_outside |= bool(outside[face])
            reaches_selected |= bool(touches_selected[face])
            for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
                neighbor = int(neighbor)
                if missing[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    pending.append(neighbor)
        if not reaches_outside and reaches_selected:
            fills.extend(component)
            enclosed_components += 1
    fills = np.asarray(fills, dtype=np.int32)
    return np.unique(fills), {
        "enclosed_gap_faces": int(len(fills)),
        "enclosed_gap_components": int(enclosed_components),
    }


def _fill_preview_make_result(state, radius):
    """Compute one immutable candidate result for the current wheel distance."""
    import numpy as np

    geometry = state["adjacency"]
    if state.get("cursor_local"):
        _fill_preview_cursor_expand(state, radius)
        geometry = state["adjacency"]
    seed_face = int(state.get("seed_local", state["seed_face"]))
    if seed_face < 0 or seed_face >= len(geometry["hidden"]):
        raise RuntimeError("preview seed is outside the prepared mesh")
    if bool(geometry["hidden"][seed_face]):
        raise RuntimeError("preview seed is hidden")
    halo = max(float(radius) * 0.5, float(state["initial_radius"]) * 0.5)
    patch_radius = float(radius) + halo
    started = time.perf_counter()
    distances, patch_ids, popped, distance_state = _fill_preview_dijkstra_incremental(
        geometry, seed_face, patch_radius, state.get("distance_state")
    )
    state["distance_state"] = distance_state
    if len(patch_ids) == 0 or not np.isfinite(distances[seed_face]):
        raise RuntimeError("preview seed is outside the prepared patch")
    proxy_metrics = {}
    analysis_ids = patch_ids
    if geometry.get("shading_approx"):
        analysis_ids, proxy_metrics = _fill_preview_shading_proxy(
            geometry,
            patch_ids,
            distances,
            radius,
            int(np.flatnonzero(patch_ids == seed_face)[0]),
        )
    local = _fill_preview_local_geometry(geometry, analysis_ids)
    seed_local = int(np.flatnonzero(analysis_ids == seed_face)[0])
    local_region, partition = _fill_preview_region(
        local, seed_local, bool(state["strict_mode"])
    )
    # Let the exact local partition inspect the original boundary band while
    # retaining only the current distance-first seed component.  The component
    # is recomputed for every radius, so finite valleys can reconnect at their
    # visible ends without carrying a blacklist between wheel stages.
    coarse_candidate_ids = proxy_metrics.get("proxy_candidate_ids")
    if coarse_candidate_ids is not None and len(coarse_candidate_ids):
        # The exact local partition may see both sides of a newly detected
        # valley because the correction band intentionally includes the
        # adjacent original faces.  Keep that fine inspection, but constrain
        # the accepted region to the distance-first seed component returned by
        # the current patch.  A later radius recomputes this component from
        # the newly acquired distance range, so a finite valley can reconnect
        # naturally without a persistent blacklist.
        allowed_local = np.isin(analysis_ids, coarse_candidate_ids)
        local_region = local_region[allowed_local[local_region]]
        local_region = np.unique(local_region).astype(np.int32)
    local_global = analysis_ids[local_region]
    candidate_mask = (
        (distances[local_global] <= float(radius) + max(radius * 1.e-8, 1.e-9))
        & ~geometry["hidden"][local_global]
    )
    preview_local_ids = local_global[candidate_mask].astype(np.int32, copy=False)
    enclosed_gap_ids, enclosed_gap_metrics = _fill_preview_enclosed_gap_faces(
        geometry, distances, radius, preview_local_ids
    )
    if len(enclosed_gap_ids):
        preview_local_ids = np.unique(
            np.r_[preview_local_ids, enclosed_gap_ids]
        ).astype(np.int32, copy=False)
        expanded_analysis_ids = np.unique(
            np.r_[analysis_ids, enclosed_gap_ids]
        ).astype(np.int32, copy=False)
        if len(expanded_analysis_ids) != len(analysis_ids):
            analysis_ids = expanded_analysis_ids
            local = _fill_preview_local_geometry(geometry, analysis_ids)
            seed_local = int(np.flatnonzero(analysis_ids == seed_face)[0])
            # Recompute only the local partition metadata used for boundary
            # classification.  The selected mask below remains the exact
            # candidate plus the enclosed components found above.
            _unused_region, partition = _fill_preview_region(
                local, seed_local, bool(state["strict_mode"])
            )
    preview_faces = preview_local_ids
    face_ids = geometry.get("face_ids")
    if face_ids is not None:
        preview_faces = face_ids[preview_faces].astype(np.int32, copy=False)
    if len(preview_faces) == 0:
        raise RuntimeError("preview produced no visible candidate faces")
    preview_set = np.isin(analysis_ids, preview_local_ids)
    first, second = local["first"], local["second"]
    shape_segments = []
    distance_segments = []
    mesh = state["obj"].data
    boundary = preview_set[first] != preview_set[second]
    for edge in np.flatnonzero(boundary):
        left, right = int(first[edge]), int(second[edge])
        inside = left if preview_set[left] else right
        outside = right if preview_set[left] else left
        v0 = int(local["pair_v0"][edge])
        v1 = int(local["pair_v1"][edge])
        edge_indices = local.get("pair_edge_indices")
        if edge_indices is not None and int(edge_indices[edge]) >= 0:
            edge_index = int(edge_indices[edge])
            edge_vertices = tuple(int(value) for value in mesh.edges[edge_index].vertices)
            matrix = state["obj"].matrix_world
            segment = tuple(
                tuple(float(value) for value in (matrix @ mesh.vertices[vertex].co))
                for vertex in edge_vertices
            )
        else:
            segment = (
                tuple(float(value) for value in geometry["world_vertices"][v0]),
                tuple(float(value) for value in geometry["world_vertices"][v1]),
            )
        outside_distance = float(distances[analysis_ids[outside]])
        shape_boundary = (
            np.isfinite(outside_distance)
            and outside_distance < float(radius) - max(radius * 0.01, 1.e-8)
            and bool(
                partition["barrier"][inside] or partition["barrier"][outside]
            )
        )
        (shape_segments if shape_boundary else distance_segments).append(segment)
    patch_edge_reached = bool(
        len(patch_ids)
        and np.any(distances[patch_ids] >= patch_radius - max(patch_radius * 0.01, 1.e-8))
    )
    elapsed = time.perf_counter() - started
    proxy_metrics_for_result = dict(proxy_metrics)
    proxy_metrics_for_result.pop("proxy_candidate_ids", None)
    return {
        "radius": float(radius),
        "faces": preview_faces,
        "candidate_count": int(len(preview_faces)),
        "analysis_faces": int(len(analysis_ids)),
        "popped_faces": int(popped),
        "shape_segments": shape_segments,
        "distance_segments": distance_segments,
        "patch_edge_reached": patch_edge_reached,
        "compute_seconds": float(elapsed),
        "geometry": local,
        "partition": partition,
        "created_generation": int(state["generation"]),
        **enclosed_gap_metrics,
        **proxy_metrics_for_result,
    }


def _fill_preview_build_draw_batches(state, result):
    """Create copied vertices and GPU batches once for one candidate result."""
    import numpy as np

    if result.get("triangles") is not None and result.get("gpu_batches") is not None:
        return
    if result.get("triangles") is None:
        obj = state["obj"]
        mesh = obj.data
        matrix = obj.matrix_world
        triangles = []
        try:
            for face_index in result["faces"]:
                polygon = mesh.polygons[int(face_index)]
                points = [
                    Vector(matrix @ mesh.vertices[int(vertex)].co)
                    for vertex in polygon.vertices
                ]
                if len(points) < 3:
                    continue
                for triangle in tessellate_polygon([points]):
                    if len(triangle) == 3:
                        triangle_vertices = []
                        for vertex in triangle:
                            # Blender 5.2 returns polygon-local integer indices;
                            # older builds returned Vector objects.
                            if isinstance(vertex, (int, np.integer)):
                                vertex = points[int(vertex)]
                            triangle_vertices.append(
                                tuple(float(value) for value in vertex)
                            )
                        # Some Sculpt viewport configurations cull the back side
                        # of a face.  Keep the original coordinates and depth,
                        # but submit the same triangle with both winding orders.
                        triangles.extend(triangle_vertices)
                        triangles.extend(reversed(triangle_vertices))
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError, IndexError):
            triangles = []
        result["triangles"] = triangles
        result["shape_lines"] = [value for segment in result["shape_segments"] for value in segment]
        result["distance_lines"] = [value for segment in result["distance_segments"] for value in segment]
        result["triangles_np"] = np.asarray(triangles, dtype=np.float32) if triangles else np.empty((0, 3), dtype=np.float32)
        result["shape_lines_np"] = np.asarray(result["shape_lines"], dtype=np.float32) if result["shape_lines"] else np.empty((0, 3), dtype=np.float32)
        result["distance_lines_np"] = np.asarray(result["distance_lines"], dtype=np.float32) if result["distance_lines"] else np.empty((0, 3), dtype=np.float32)
    if result.get("gpu_batches") is None:
        shader = _fill_preview_shader_get()
        batches = {}
        if shader is not None:
            if len(result["triangles_np"]):
                batches["triangles"] = batch_for_shader(
                    shader, "TRIS", {"pos": result["triangles_np"]}
                )
            if len(result["distance_lines_np"]):
                batches["distance_lines"] = batch_for_shader(
                    shader, "LINES", {"pos": result["distance_lines_np"]}
                )
            if len(result["shape_lines_np"]):
                batches["shape_lines"] = batch_for_shader(
                    shader, "LINES", {"pos": result["shape_lines_np"]}
                )
        result["gpu_batches"] = batches


def _fill_preview_draw_text(state, result):
    """Draw short, ASCII-safe status text in the owning viewport."""
    try:
        font_id = 0
        blf.size(font_id, 13)
        region = state["region"]
        width = int(getattr(region, "width", 0))
        height = int(getattr(region, "height", 0))
        # Keep the status HUD in the unobstructed viewport corner.  In
        # Blender's POST_PIXEL space the lower shelf occupies the last rows.
        x, y = max(18, width - 520), max(120, height - 100)
        if state["phase"] in {"prepare", "prepare_finalize"}:
            cancel_line = (
                "Esc queued; waiting for current processing step"
                if state["phase"] == "prepare_finalize"
                else "Esc cancel"
            )
            lines = [
                "Smart Fill Preview - preparing surface data...",
                f"Prep {float(state.get('prepare_seconds', 0.0)):.2f}s",
                f"{cancel_line}; E again/Enter apply",
            ]
        elif state["phase"] == "compute":
            lines = [
                "Smart Fill Preview - computing local patch...",
                f"Distance {float(state['desired_radius']):.4g} m",
                "E again/Enter apply when ready; Esc cancel",
            ]
        else:
            edge = "shape boundary" if result.get("shape_segments") else "distance boundary"
            if result.get("patch_edge_reached"):
                edge += " / analysis limit"
            lines = [
                "Smart Fill Preview",
                f"Distance {float(result['radius']):.4g} m  Candidates {int(result['candidate_count'])}",
                f"Ready - {edge}  Prep {float(state.get('prepare_seconds', 0.0)):.2f}s",
                "E again: apply   Enter: apply   Esc: cancel",
            ]
        for index, line in enumerate(lines):
            blf.position(font_id, x, y - index * 18, 0)
            blf.color(font_id, 0.92, 0.96, 1.0, 1.0)
            blf.draw(font_id, line)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass


def _fill_preview_draw():
    """Draw one session's copied candidate overlay and boundary lines."""
    state = _fill_preview_state
    if state is None or not state.get("active"):
        return
    try:
        context = bpy.context
        if context.area is None or int(context.area.as_pointer()) != int(state["area_key"]):
            return
        result = state.get("result")
        shader = _fill_preview_shader
        if shader is None:
            return
        if result is not None:
            _fill_preview_build_draw_batches(state, result)
            if (
                state.get("phase") == "ready"
                and result is state.get("result")
                and state.get("drawn_generation") != state.get("generation")
            ):
                # The modal gate does not open merely because a compute timer
                # returned.  Mark the first callback that actually draws the
                # newest result, then drain already queued wheel input briefly.
                state["drawn_generation"] = int(state.get("generation", 0))
                state["wheel_drain_until"] = (
                    time.perf_counter() + _FILL_PREVIEW_WHEEL_DRAIN_SECONDS
                )
            gpu.state.blend_set("ALPHA")
            depth_set = False
            depth_mask = False
            try:
                try:
                    gpu.state.depth_test_set("LESS_EQUAL")
                    depth_set = True
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                try:
                    gpu.state.depth_mask_set(False)
                    depth_mask = True
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                if len(result["triangles_np"]):
                    shader.bind()
                    shader.uniform_float("color", (0.16, 0.72, 0.96, 0.20))
                    batch = result.get("gpu_batches", {}).get("triangles")
                    if batch is not None:
                        batch.draw(shader)
                for key, color, batch_key in (
                    ("distance_lines_np", (0.20, 0.86, 1.0, 0.95), "distance_lines"),
                    ("shape_lines_np", (1.0, 0.38, 0.08, 0.95), "shape_lines"),
                ):
                    if len(result[key]):
                        shader.bind()
                        shader.uniform_float("color", color)
                        gpu.state.line_width_set(2.0)
                        batch = result.get("gpu_batches", {}).get(batch_key)
                        if batch is not None:
                            batch.draw(shader)
            finally:
                if depth_mask:
                    try:
                        gpu.state.depth_mask_set(True)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass
                if depth_set:
                    try:
                        gpu.state.depth_test_set("NONE")
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass
                try:
                    gpu.state.line_width_set(1.0)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                gpu.state.blend_set("NONE")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        try:
            gpu.state.blend_set("NONE")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


def _fill_preview_draw_text_handler():
    """Draw the 2D status HUD in POST_PIXEL, after the 3D overlay pass."""
    state = _fill_preview_state
    if state is None or not state.get("active"):
        return
    try:
        context = bpy.context
        if context.area is None or int(context.area.as_pointer()) != int(state["area_key"]):
            return
        _fill_preview_draw_text(state, state.get("result") or {})
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _fill_preview_tag_redraw(state=None):
    if state is not None:
        _tag_redraw(state.get("area"))
    _topology_color_tag_redraw_all()


def _fill_preview_stop_draw():
    global _fill_preview_draw_handler, _fill_preview_text_draw_handler
    for handler in (_fill_preview_draw_handler, _fill_preview_text_draw_handler):
        if handler is None:
            continue
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handler, "WINDOW")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    _fill_preview_draw_handler = None
    _fill_preview_text_draw_handler = None


def _fill_preview_cancel(state=None, reason="cancel"):
    global _fill_preview_state
    current = _fill_preview_state
    if state is not None and current is not state:
        return
    if current is None:
        return
    current["active"] = False
    timer = current.get("timer")
    if timer is not None:
        try:
            bpy.context.window_manager.event_timer_remove(timer)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
    current["timer"] = None
    _fill_preview_stop_draw()
    _fill_preview_state = None
    _fill_preview_tag_redraw(current)


def _fill_preview_valid(state, context):
    if state is None or not state.get("active"):
        return False
    try:
        obj = state["obj"]
        window = getattr(context, "window", None)
        window_key = int(state.get("window_key", 0) or 0)
        return (
            context.area is not None
            and int(context.area.as_pointer()) == int(state["area_key"])
            and (
                not window_key
                or (window is not None and int(window.as_pointer()) == window_key)
            )
            and context.mode == "SCULPT"
            and context.active_object is obj
            and _fill_preview_signature(obj) == state["signature"]
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _fill_preview_shader_get():
    global _fill_preview_shader
    if _fill_preview_shader is None:
        try:
            _fill_preview_shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _fill_preview_shader = None
    return _fill_preview_shader


def _on_fill_preview_depsgraph_update(_scene, depsgraph):
    """Invalidate cached graph data for updated objects/meshes.

    Cache ownership is independent of an active preview session: a geometry or
    visibility update while idle must not leave a reusable graph from the old
    revision.  Blender exposes many Face Set writes as mesh updates too, so
    those updates conservatively drop the cache; this costs a rebuild but never
    permits stale geometry to reach a later E invocation.
    """
    state = _fill_preview_state
    try:
        updated_pointers = set()
        for update in depsgraph.updates:
            data = update.id
            updated_pointers.add(int(data.as_pointer()))
            # depsgraph updates commonly expose evaluated Object/Mesh copies;
            # cache keys are owned by the original datablocks.
            original = getattr(data, "original", None)
            if original is not None:
                updated_pointers.add(int(original.as_pointer()))
        if not updated_pointers:
            return
        affected_session = False
        if state is not None and state.get("active"):
            obj = state["obj"]
            affected_session = bool(
                int(obj.as_pointer()) in updated_pointers
                or int(obj.data.as_pointer()) in updated_pointers
            )
        for key in list(_fill_preview_adjacency_cache):
            if key[0] in updated_pointers or key[1] in updated_pointers:
                _fill_preview_adjacency_cache.pop(key, None)
        for key in list(_fill_preview_cursor_cache):
            if key[0] in updated_pointers or key[1] in updated_pointers:
                _fill_preview_cursor_cache.pop(key, None)
        if affected_session:
            _fill_preview_cancel(state, "stale")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        if state is not None and state.get("active"):
            _fill_preview_cancel(state, "stale")


def _fill_average(values, first, second, count, iterations):
    """Diffuse over manifold face adjacency, without changing mesh geometry."""
    import numpy as np
    degree = np.bincount(first, minlength=count) + np.bincount(second, minlength=count)
    divisor = degree + 1
    current = values.copy()
    for _ in range(iterations):
        if current.ndim == 1:
            current = (current + np.bincount(first, weights=current[second], minlength=count)
                       + np.bincount(second, weights=current[first], minlength=count)) / divisor
        else:
            current = np.column_stack([
                (current[:, axis] + np.bincount(first, weights=current[second, axis], minlength=count)
                 + np.bincount(second, weights=current[first, axis], minlength=count)) / divisor
                for axis in range(current.shape[1])])
    return current


def _fill_geometry(obj):
    """Bulk geometry; content-keyed so sculpting, Undo and rewiring invalidate it."""
    import hashlib
    import numpy as np
    mesh = obj.data
    count = len(mesh.polygons)
    coordinates = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    loop_edges = np.empty(len(mesh.loops), dtype=np.int32)
    loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
    totals = np.empty(count, dtype=np.int32)
    hidden = np.empty(count, dtype=bool)
    mesh.vertices.foreach_get('co', coordinates.ravel())
    mesh.loops.foreach_get('edge_index', loop_edges)
    mesh.loops.foreach_get('vertex_index', loop_vertices)
    mesh.polygons.foreach_get('loop_total', totals)
    mesh.polygons.foreach_get('hide', hidden)
    transform = np.asarray(obj.matrix_world.to_3x3(), dtype=np.float64)
    digest = hashlib.blake2b(digest_size=16)
    for value in (coordinates, loop_edges, loop_vertices, totals, hidden, transform):
        digest.update(value.tobytes())
    key = (mesh.as_pointer(), digest.digest())
    cached = _local_face_set_adjacency_cache.get(key)
    if cached is not None:
        return cached

    centers = np.empty((count, 3), dtype=np.float32)
    normals = np.empty((count, 3), dtype=np.float32)
    mesh.polygons.foreach_get('center', centers.ravel())
    mesh.polygons.foreach_get('normal', normals.ravel())
    centers = centers @ transform.T
    normals = normals @ np.linalg.inv(transform)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.e-20)
    face_ids = np.repeat(np.arange(count, dtype=np.int32), totals)
    order = np.argsort(loop_edges, kind='stable')
    sorted_edges = loop_edges[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_edges)) + 1]
    lengths = np.diff(np.r_[starts, len(order)])
    pairs = starts[lengths == 2]
    first, second = face_ids[order[pairs]], face_ids[order[pairs + 1]]
    face_starts = np.r_[0, np.cumsum(totals)[:-1]]
    paired_loops = order[pairs]
    paired_next = face_starts[first] + ((paired_loops - face_starts[first] + 1) % totals[first])
    edge_vectors = (coordinates[loop_vertices[paired_next]].astype(np.float64)
                    - coordinates[loop_vertices[paired_loops]]) @ transform.T
    shared_lengths = np.linalg.norm(edge_vectors, axis=1)
    # Imported meshes can have coincident but unwelded ring seams. Match only
    # two open edges with both endpoints coincident, opposite winding, and
    # compatible normals. This changes the search graph, never the mesh.
    border_loops = order[starts[lengths == 1]]
    seam_count = 0
    if len(border_loops):
        face_starts = np.r_[0, np.cumsum(totals)[:-1]]
        border_faces = face_ids[border_loops]
        next_loops = face_starts[border_faces] + (
            (border_loops - face_starts[border_faces] + 1) % totals[border_faces])
        points_a = coordinates[loop_vertices[border_loops]].astype(np.float64) @ transform.T
        points_b = coordinates[loop_vertices[next_loops]].astype(np.float64) @ transform.T
        edge_lengths = np.linalg.norm(points_b - points_a, axis=1)
        tolerance = max(float(np.median(edge_lengths)) * 1.e-4, 1.e-12)
        quantized = np.rint(np.vstack((points_a, points_b)) / tolerance).astype(np.int64)
        _, point_ids = np.unique(quantized, axis=0, return_inverse=True)
        pa, pb = np.split(point_ids, 2)
        signatures = np.column_stack((np.minimum(pa, pb), np.maximum(pa, pb)))
        _, inverse, counts = np.unique(signatures, axis=0, return_inverse=True, return_counts=True)
        seam_order = np.argsort(inverse, kind='stable')
        seam_starts = np.r_[0, np.cumsum(counts)[:-1]][counts == 2]
        ia, ib = seam_order[seam_starts], seam_order[seam_starts + 1]
        fa, fb = border_faces[ia], border_faces[ib]
        match = ((pa[ia] == pb[ib]) & (pb[ia] == pa[ib]) & (pa[ia] != pb[ia])
                 & (fa != fb) & (np.sum(normals[fa] * normals[fb], axis=1) > 0.5)
                 & (np.linalg.norm(points_a[ia] - points_b[ib], axis=1) <= tolerance)
                 & (np.linalg.norm(points_b[ia] - points_a[ib], axis=1) <= tolerance))
        first, second = np.r_[first, fa[match]], np.r_[second, fb[match]]
        shared_lengths = np.r_[shared_lengths, (edge_lengths[ia[match]] + edge_lengths[ib[match]]) * 0.5]
        seam_count = int(np.count_nonzero(match))
    valid = ~(hidden[first] | hidden[second]) & (first != second)
    first, second = first[valid], second[valid]
    shared_lengths = shared_lengths[valid]
    # Open and non-manifold edges are never bridges to another sheet.
    delta = centers[second] - centers[first]
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1.e-20)
    degree = np.bincount(first, minlength=count) + np.bincount(second, minlength=count)
    scale = (np.bincount(first, weights=distance, minlength=count)
             + np.bincount(second, weights=distance, minlength=count)) / np.maximum(degree, 1)
    smooth = _fill_average(normals, first, second, count, 2)
    smooth /= np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1.e-20)
    # Signed curvature per unit length, not raw dihedral per polygon: varying
    # tessellation density must not create a ring-shaped stopping boundary.
    curvature_edge = np.sum((smooth[second] - smooth[first]) * delta, axis=1)
    # A ridge can curve outward along its length while its foot curves inward
    # across it. A mean curvature would cancel these two directions and miss
    # the foot. Preserve the strongest inward directional turn separately.
    inward = np.maximum(-curvature_edge / distance, 0.0) * 3.0
    directional_valley = np.zeros(count)
    np.maximum.at(directional_valley, first, inward)
    np.maximum.at(directional_valley, second, inward)
    directional_valley = _fill_average(directional_valley, first, second, count, 3)
    metric = (np.bincount(first, weights=distance**2, minlength=count)
              + np.bincount(second, weights=distance**2, minlength=count))
    curvature = (np.bincount(first, weights=curvature_edge, minlength=count)
                 + np.bincount(second, weights=curvature_edge, minlength=count)) / np.maximum(metric, 1.e-30)
    curvature = _fill_average(curvature, first, second, count, 3)
    broad = _fill_average(curvature, first, second, count, 24)
    # Positive curvature is a convex crest: it must remain traversable even
    # when its radius changes abruptly. Only concave departures form barriers.
    contrast = np.where(curvature < 0.0,
                        np.maximum(broad - curvature, 0.0) * scale * 6.0, 0.0)
    concavity = np.maximum(-curvature, 0.0) * scale * 6.0
    # A stable approximation of the broad concave normal change that appears
    # as a dark foot under studio lighting. Never sample screen brightness:
    # the inferred line must not move with the camera, lights or Face Set color.
    valley_line = _fill_average(np.maximum(concavity, directional_valley),
                               first, second, count, 6)
    edge_valley = (valley_line[first] + valley_line[second]) * 0.5
    contour_cost = shared_lengths / (1.0 + (edge_valley / 0.10)**2)
    raw_angle = np.arccos(np.clip(np.sum(normals[first] * normals[second], axis=1), -1, 1))
    raw_turn = np.sum((normals[second] - normals[first]) * delta, axis=1)
    raw_angle = np.where(raw_turn < -distance * 1.e-6, raw_angle, 0.0)
    crease = np.zeros(count)
    np.maximum.at(crease, first, raw_angle)
    np.maximum.at(crease, second, raw_angle)
    # CSR is compact for triangles and also supports arbitrary polygons.
    sources = np.r_[first, second]
    destinations = np.r_[second, first]
    order = np.argsort(sources, kind='stable')
    offsets = np.r_[0, np.cumsum(degree)].astype(np.int64)
    cached = dict(first=first, second=second, offsets=offsets,
                  centers=centers, normals=normals, scale=scale,
                  neighbors=destinations[order], neighbor_lengths=np.r_[distance, distance][order],
                  contour_cost=np.r_[contour_cost, contour_cost][order],
                  hidden=hidden, contrast=contrast, concavity=concavity,
                  crease=crease, directional_valley=directional_valley,
                  count=count, seam_count=seam_count, partitions={})
    _local_face_set_adjacency_cache.clear()
    _local_face_set_adjacency_cache[key] = cached
    return cached


def _fill_partition(geometry, strict_mode):
    """Close short boundary gaps, then assign the band to its nearest core.

    Owners propagate only through the boundary band. They cannot use that
    band to merge separate smooth cores. Equal-distance ties use face index,
    independent of the clicked seed or previously painted Face Set values.
    """
    import numpy as np
    cached = geometry['partitions'].get(bool(strict_mode))
    if cached is not None:
        return cached
    first, second = geometry['first'], geometry['second']
    count, hidden = geometry['count'], geometry['hidden']
    barrier = ((geometry['contrast'] > (0.09 if strict_mode else 0.10))
               | (geometry['concavity'] > (0.09 if strict_mode else 0.10))
               | (geometry['directional_valley'] > (0.09 if strict_mode else 0.10))
               | (geometry['crease'] > math.radians(50 if strict_mode else 65))) & ~hidden
    band = barrier.copy()
    # Close passages only near detected shape boundaries; an isolated smooth
    # narrow strip has no barrier and is therefore not cut just for being thin.
    for _ in range(1):
        expanded = band.copy()
        np.logical_or.at(expanded, first, band[second])
        np.logical_or.at(expanded, second, band[first])
        band = expanded
    core = ~band & ~hidden
    owner = np.full(count, count, dtype=np.int32)
    owner[core] = np.flatnonzero(core)
    # Use physical surface distance instead of polygon-step count. This
    # reduces tessellation-shaped zigzags without moving any mesh vertices.
    import heapq
    offsets, neighbors = geometry['offsets'], geometry['neighbors']
    lengths = geometry['neighbor_lengths']
    # Reserve tangent-continuous extensions of each core before sharing the
    # ambiguous valley band. In particular, a flat base must keep its own
    # continuation up to the foot; nearest-core distance alone can give that
    # flat strip to the raised part. Anchoring both normal AND tangent-plane
    # height to the source prevents walking across a rounded foot in tiny steps.
    centers, normals = geometry['centers'], geometry['normals']
    protected = core.copy()
    extension_distance = np.full(count, np.inf)
    extension_distance[core] = 0.0
    extension_owner = owner.copy()
    normal_limit = math.cos(math.radians(6.0))

    def offer_extension(face, source, distance):
        if core[face] or hidden[face]:
            return
        if float(np.dot(normals[face], normals[source])) < normal_limit:
            return
        tolerance = max(float(geometry['scale'][source]) * 0.10, 1.e-12)
        if abs(float(np.dot(centers[face] - centers[source], normals[source]))) > tolerance:
            return
        if (distance < extension_distance[face]
                or (distance == extension_distance[face] and source < extension_owner[face])):
            extension_distance[face], extension_owner[face] = distance, source
            heapq.heappush(extension_heap, (distance, source, face))

    extension_heap = []
    rim = np.zeros(count, dtype=bool)
    rim[first[band[first] & core[second]]] = True
    rim[second[band[second] & core[first]]] = True
    for face in np.flatnonzero(rim):
        for edge in range(offsets[face], offsets[face + 1]):
            source = int(neighbors[edge])
            if core[source]:
                offer_extension(int(face), source, float(lengths[edge]))
    while extension_heap:
        distance, source, face = heapq.heappop(extension_heap)
        if distance != extension_distance[face] or source != extension_owner[face]:
            continue
        protected[face] = True
        for edge in range(offsets[face], offsets[face + 1]):
            neighbor = int(neighbors[edge])
            if band[neighbor]:
                offer_extension(neighbor, source, distance + float(lengths[edge]))
    owner[protected] = extension_owner[protected]
    best = np.full(count, np.inf)
    best[protected] = 0.0
    fringe = np.zeros(count, dtype=bool)
    fringe[first[~protected[first] & protected[second]]] = True
    fringe[second[~protected[second] & protected[first]]] = True
    heap = []
    for face in np.flatnonzero(fringe):
        start, end = offsets[face], offsets[face + 1]
        for edge in range(start, end):
            neighbor = int(neighbors[edge])
            if not protected[neighbor]:
                continue
            candidate = float(lengths[edge])
            source = int(owner[neighbor])
            if candidate < best[face] or (candidate == best[face] and source < owner[face]):
                best[face], owner[face] = candidate, source
        heap.append((float(best[face]), int(owner[face]), int(face)))
    heapq.heapify(heap)
    while heap:
        distance, source, face = heapq.heappop(heap)
        if distance != best[face] or source != owner[face]:
            continue
        for edge in range(offsets[face], offsets[face + 1]):
            neighbor = int(neighbors[edge])
            if protected[neighbor] or hidden[neighbor]:
                continue
            candidate = distance + float(lengths[edge])
            if candidate < best[neighbor] or (candidate == best[neighbor] and source < owner[neighbor]):
                best[neighbor], owner[neighbor] = candidate, source
                heapq.heappush(heap, (candidate, source, neighbor))
    cached = dict(core=core, owner=owner, barrier=barrier, band=band, protected=protected)
    geometry['partitions'][bool(strict_mode)] = cached
    return cached


def _smart_face_set_region(obj, seed_face, strict_mode=False):
    """Return a complete smooth core and its owned valley/step boundary band."""
    import numpy as np
    geometry = _fill_geometry(obj)
    count = geometry['count']
    if seed_face < 0 or seed_face >= count or geometry['hidden'][seed_face]:
        return np.empty(0, dtype=np.int32)
    partition = _fill_partition(geometry, strict_mode)
    core, owner = partition['core'], partition['owner']
    root = int(owner[seed_face])
    if root == count:
        # A component consisting entirely of sharp boundary has no smooth
        # core. Do not guess and fill the whole component.
        return np.asarray([seed_face], dtype=np.int32)
    selected = np.zeros(count + 1, dtype=bool)
    selected[root] = True
    pending = deque([root])
    neighbors, offsets = geometry['neighbors'], geometry['offsets']
    while pending:
        face = pending.popleft()
        for neighbor in neighbors[offsets[face]:offsets[face + 1]]:
            if core[neighbor] and not selected[neighbor]:
                selected[neighbor] = True
                pending.append(int(neighbor))
    region = selected[owner]
    # Minimize a valley-weighted physical boundary length inside the band.
    # Strong, broadly supported concave normal changes attract the contour;
    # length penalizes mesh-scale zigzags. Every sequential flip must lower
    # that energy. This is a bounded local relaxation, not a global graph cut.
    # Cores stay fixed, so another part's interior cannot be absorbed.
    first, second = geometry['first'], geometry['second']
    relaxed = region.copy()
    editable = partition['band'] & ~partition['protected'] & (owner < count)
    weights = geometry['contour_cost']
    for iteration in range(6):
        crossing = relaxed[first] != relaxed[second]
        fringe = np.unique(np.r_[first[crossing], second[crossing]])
        fringe = fringe[editable[fringe]]
        if iteration % 2:
            fringe = fringe[::-1]
        changed = False
        for face in fringe:
            if face == seed_face:
                continue
            start, end = offsets[face], offsets[face + 1]
            linked = neighbors[start:end]
            costs = weights[start:end]
            old_cost = float(costs[relaxed[linked] != relaxed[face]].sum())
            new_cost = float(costs[relaxed[linked] == relaxed[face]].sum())
            if new_cost < old_cost - max(old_cost, new_cost, 1.e-20) * 1.e-8:
                relaxed[face] = not relaxed[face]
                changed = True
        if not changed:
            break
    if relaxed[seed_face]:
        # Discard detached tips created by the contour relaxation. If it
        # would disconnect the clicked face, retain the unsmoothed region.
        connected = np.zeros(count, dtype=bool)
        connected[root] = True
        pending = deque([root])
        while pending:
            face = pending.popleft()
            for neighbor in neighbors[offsets[face]:offsets[face + 1]]:
                if relaxed[neighbor] and not connected[neighbor]:
                    connected[neighbor] = True
                    pending.append(int(neighbor))
        if connected[seed_face]:
            region = connected
    return np.flatnonzero(region).astype(np.int32)


def _smart_face_set_fill(context, coord, strict_mode=False):
    """Find the region before making one undoable Face Set write."""
    import numpy as np
    hit = _raycast_sculpt_face_set(context, coord)
    if hit is None:
        return None
    obj, seed_face, seed_face_set, _seed_location, _screen_position = hit
    attr = obj.data.attributes.get('.sculpt_face_set')
    candidates = _smart_face_set_region(obj, seed_face, strict_mode)
    values = np.empty(len(attr.data), dtype=np.int32)
    attr.data.foreach_get('value', values)
    changed = int(np.count_nonzero(values[candidates] != seed_face_set))
    if changed:
        values[candidates] = seed_face_set
        attr.data.foreach_set('value', values)
        obj.data.update()
    return len(candidates), changed



class VIEW3D_OT_mesh_focus_local_face_set_grow(bpy.types.Operator):
    """Preview and apply one geometry-aware local Face Set region."""

    bl_idname = LOCAL_FACE_SET_GROW_OPERATOR_ID
    bl_label = "Mesh Focus: Smart Face Set Fill"
    bl_description = (
        "Fill the connected smooth region under the cursor with the seed Face Set"
    )
    bl_options = {"REGISTER", "UNDO"}

    mouse_region_x: IntProperty(options={"SKIP_SAVE"})
    mouse_region_y: IntProperty(options={"SKIP_SAVE"})
    strict_mode: BoolProperty(options={"SKIP_SAVE"}, default=False)

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
        )

    def invoke(self, context, event):
        global _fill_preview_state, _fill_preview_draw_handler, _fill_preview_text_draw_handler
        if not self.poll(context):
            return {"PASS_THROUGH"}

        mouse_x = getattr(event, "mouse_region_x", None)
        mouse_y = getattr(event, "mouse_region_y", None)
        if mouse_x is None or mouse_y is None:
            return {"CANCELLED"}

        self.mouse_region_x = int(mouse_x)
        self.mouse_region_y = int(mouse_y)
        self.strict_mode = bool(
            self.strict_mode or getattr(event, "ctrl", False)
        )
        hit = _raycast_sculpt_face_set(
            context, Vector((self.mouse_region_x, self.mouse_region_y))
        )
        if hit is None:
            self.report({"WARNING"}, "Smart Face Set Fill: no visible face under cursor")
            return {"CANCELLED"}
        if _fill_preview_state is not None:
            _fill_preview_cancel(_fill_preview_state, "replaced")
        obj, seed_face, seed_face_set, _location, _screen = hit
        signature = _fill_preview_signature(obj)
        if signature is None:
            return {"CANCELLED"}
        self._preview_id = _next_session_id()
        state = {
            "active": True,
            "phase": "prepare",
            "operator": self,
            "session_id": int(self._preview_id),
            "area": context.area,
            "area_key": int(context.area.as_pointer()),
            "window_key": int(context.window.as_pointer()) if context.window else 0,
            "region": context.region,
            "obj": obj,
            "obj_pointer": int(obj.as_pointer()),
            "mesh_pointer": int(obj.data.as_pointer()),
            "signature": signature,
            "mode": str(context.mode),
            "seed_face": int(seed_face),
            "seed_face_set": int(seed_face_set),
            "strict_mode": bool(self.strict_mode),
            "cursor_prepare": True,
            "cursor_local": True,
            "seed_local": None,
            "start_key": str(getattr(event, "type", "E") or "E"),
            "start_key_released": False,
            "adjacency": None,
            "prepare_job": None,
            "prepare_finalize_job": None,
            "prepare_raw": None,
            "prepare_stage": "queued",
            "prepare_tick_times": [],
            "distance_state": None,
            "initial_radius": None,
            "desired_radius": None,
            "processed_radius": None,
            "pending": True,
            "wheel_armed": False,
            "wheel_gate": False,
            "dropped_wheel_events": 0,
            "drawn_generation": None,
            "wheel_drain_until": 0.0,
            "result": None,
            "results": {},
            "generation": 0,
            "prepare_seconds": 0.0,
            "last_tick_seconds": 0.0,
            "max_tick_seconds": 0.0,
            "timer": None,
            "last_timer_dispatch": 0.0,
        }
        _fill_preview_state = state
        shader = _fill_preview_shader_get()
        if shader is None:
            _fill_preview_cancel(state, "shader")
            return {"CANCELLED"}
        try:
            state["timer"] = context.window_manager.event_timer_add(
                0.01, window=context.window
            )
            context.window_manager.modal_handler_add(self)
            _fill_preview_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                _fill_preview_draw, (), "WINDOW", "POST_VIEW"
            )
            _fill_preview_text_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                _fill_preview_draw_text_handler, (), "WINDOW", "POST_PIXEL"
            )
            _fill_preview_tag_redraw(state)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _fill_preview_cancel(state, "start")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _process_timer(self, context, state):
        import numpy as np

        if not _fill_preview_valid(state, context):
            _fill_preview_cancel(state, "stale")
            return False
        try:
            if state["phase"] == "prepare":
                started = time.perf_counter()
                if state["prepare_job"] is None and state["prepare_raw"] is None:
                    signature = state["signature"]
                    cached = None if state.get("cursor_prepare") else _fill_preview_adjacency_cache.get(signature)
                    if cached is not None:
                        state["adjacency"] = cached
                        state["initial_radius"] = _fill_preview_initial_radius(
                            cached, state["seed_face"]
                        )
                        state["desired_radius"] = state["initial_radius"]
                        state["phase"] = "compute"
                        state["pending"] = True
                        state["wheel_armed"] = False
                        state["wheel_gate"] = False
                        state["prepare_stage"] = "cache"
                        return True
                    if state.get("cursor_prepare"):
                        state["prepare_job"] = _fill_preview_cursor_prepare_steps(
                            state["obj"], state["seed_face"]
                        )
                    else:
                        state["prepare_job"] = _fill_preview_adjacency_steps(state["obj"])
                try:
                    state["prepare_stage"] = next(state["prepare_job"])
                    state["prepare_job_tick"] = time.perf_counter() - started
                    state["prepare_tick_times"].append(
                        {"stage": state["prepare_stage"], "seconds": state["prepare_job_tick"]}
                    )
                except StopIteration as complete:
                    state["prepare_job"] = None
                    if state.get("cursor_prepare"):
                        adjacency = complete.value
                        state["adjacency"] = adjacency
                        state["initial_radius"] = float(adjacency["initial_radius"])
                        state["desired_radius"] = state["initial_radius"]
                        state["processed_radius"] = None
                        state["seed_local"] = int(
                            np.flatnonzero(
                                adjacency["face_ids"] == int(state["seed_face"])
                            )[0]
                        )
                        state["prepare_stage"] = "cursor-ready"
                        state["phase"] = "compute"
                        state["pending"] = True
                        state["wheel_armed"] = False
                        state["wheel_gate"] = False
                    else:
                        state["prepare_raw"] = complete.value
                        state["prepare_stage"] = "arrays-ready"
                        state["phase"] = "prepare_finalize"
                elapsed = time.perf_counter() - started
                state["last_tick_seconds"] = elapsed
                state["max_tick_seconds"] = max(
                    float(state["max_tick_seconds"]), elapsed
                )
                state["prepare_seconds"] += elapsed
                _fill_preview_tag_redraw(state)
                return True
            if state["phase"] == "prepare_finalize":
                started = time.perf_counter()
                if state["prepare_finalize_job"] is None:
                    state["prepare_finalize_job"] = _fill_preview_build_adjacency_cooperative(
                        state["obj"], prepared=state["prepare_raw"]
                    )
                try:
                    state["prepare_stage"] = next(state["prepare_finalize_job"])
                    state["prepare_finalize_tick"] = time.perf_counter() - started
                except StopIteration as complete:
                    state["prepare_finalize_job"] = None
                    adjacency = complete.value
                    state["adjacency"] = adjacency
                    state["prepare_raw"] = None
                    state["initial_radius"] = _fill_preview_initial_radius(
                        adjacency, state["seed_face"]
                    )
                    state["desired_radius"] = state["initial_radius"]
                    state["prepare_stage"] = "adjacency-ready"
                elapsed = time.perf_counter() - started
                state["prepare_tick_times"].append(
                    {"stage": state["prepare_stage"], "seconds": elapsed}
                )
                state["last_tick_seconds"] = elapsed
                state["max_tick_seconds"] = max(
                    float(state["max_tick_seconds"]), elapsed
                )
                state["prepare_seconds"] += elapsed
                if state["prepare_finalize_job"] is not None:
                    _fill_preview_tag_redraw(state)
                    return True
                state["phase"] = "compute"
                state["pending"] = True
                state["wheel_armed"] = False
                state["wheel_gate"] = False
                _fill_preview_tag_redraw(state)
                return True
            if state["phase"] == "compute" and state["pending"]:
                radius = float(state["desired_radius"])
                state["result"] = None
                state["generation"] += 1
                key = round(radius, 10)
                cached = state["results"].get(key)
                if cached is None:
                    cached = _fill_preview_make_result(state, radius)
                    state["results"][key] = cached
                    while len(state["results"]) > 3:
                        state["results"].pop(next(iter(state["results"])), None)
                else:
                    # Treat a shrink/revisit as the current stage so eviction
                    # preserves the current radius and its two predecessors.
                    state["results"].pop(key, None)
                    state["results"][key] = cached
                if int(cached.get("created_generation", -1)) != int(state["generation"]):
                    cached = dict(cached)
                    cached["created_generation"] = int(state["generation"])
                _fill_preview_build_draw_batches(state, cached)
                state["last_tick_seconds"] = float(cached.get("compute_seconds", 0.0))
                state["max_tick_seconds"] = max(
                    float(state["max_tick_seconds"]),
                    float(state["last_tick_seconds"]),
                )
                state["result"] = cached
                state["processed_radius"] = radius
                state["pending"] = False
                state["phase"] = "ready"
                state["wheel_armed"] = False
                state["wheel_gate"] = True
                state["drawn_generation"] = None
                state["wheel_drain_until"] = 0.0
                _fill_preview_tag_redraw(state)
                return True
        except (
            AttributeError,
            IndexError,
            MemoryError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            try:
                message = str(error).strip() or "preview calculation failed"
                self.report({"WARNING"}, f"Smart Face Set Fill: {message}")
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            _fill_preview_cancel(state, "compute-error")
            return False
        return True

    def _finish_confirm(self, context, state):
        import numpy as np

        result = state.get("result")
        if state.get("phase") != "ready" or result is None or state.get("pending"):
            return {"RUNNING_MODAL"}
        if not _fill_preview_valid(state, context):
            _fill_preview_cancel(state, "stale")
            return {"CANCELLED"}
        if int(result.get("created_generation", -1)) != int(state["generation"]):
            _fill_preview_cancel(state, "stale-result")
            return {"CANCELLED"}
        obj = state["obj"]
        attr = obj.data.attributes.get(".sculpt_face_set")
        if attr is None or attr.domain != "FACE":
            _fill_preview_cancel(state, "face-set-layer")
            return {"CANCELLED"}
        try:
            values = np.empty(len(attr.data), dtype=np.int32)
            attr.data.foreach_get("value", values)
            faces = np.asarray(result["faces"], dtype=np.int32)
            changed = int(np.count_nonzero(values[faces] != int(state["seed_face_set"])))
            if changed:
                values[faces] = int(state["seed_face_set"])
                attr.data.foreach_set("value", values)
                obj.data.update()
        except (
            AttributeError,
            IndexError,
            MemoryError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            _fill_preview_cancel(state, "write-error")
            return {"CANCELLED"}
        candidate_count = int(result.get("candidate_count", len(faces)))
        _fill_preview_cancel(state, "confirm")
        self.report(
            {"INFO"},
            f"Smart Face Set Fill: {changed} faces changed ({candidate_count} candidates)",
        )
        return {"FINISHED"}

    def modal(self, context, event):
        state = _fill_preview_state
        if state is None or state.get("operator") is not self:
            return {"CANCELLED"}
        event_type = getattr(event, "type", "")
        event_value = getattr(event, "value", None)
        if event_type == "ESC" and event_value in {None, "PRESS"}:
            _fill_preview_cancel(state, "escape")
            return {"CANCELLED"}
        if event_type == state.get("start_key", "E"):
            if event_value == "RELEASE":
                state["start_key_released"] = True
                return {"RUNNING_MODAL"}
            if (
                event_value == "PRESS"
                and state.get("start_key_released")
                and not bool(getattr(event, "is_repeat", False))
            ):
                return self._finish_confirm(context, state)
            # The initial press, auto-repeat, and a press held through the
            # preparation phase must never confirm the old/partial result.
            return {"RUNNING_MODAL"}
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            # Wheel events cannot carry an input timestamp through Blender's
            # Python modal API.  Drop every event until the latest generation
            # has drawn and its short queue-drain interval has elapsed.  A
            # queued burst therefore cannot mutate desired_radius immediately
            # after a synchronous compute returns Ready.
            if (
                state.get("initial_radius") is None
                or state.get("phase") != "ready"
                or state.get("pending")
                or not state.get("wheel_armed")
            ):
                state["dropped_wheel_events"] = int(
                    state.get("dropped_wheel_events", 0)
                ) + 1
                if (
                    state.get("wheel_gate")
                    and state.get("drawn_generation") == state.get("generation")
                ):
                    state["wheel_drain_until"] = (
                        time.perf_counter() + _FILL_PREVIEW_WHEEL_DRAIN_SECONDS
                    )
                return {"RUNNING_MODAL"}
            factor = 1.25 if event.type == "WHEELUPMOUSE" else 1.0 / 1.25
            base = float(state["desired_radius"] or state["initial_radius"])
            state["desired_radius"] = max(
                float(state["initial_radius"]) * 0.125,
                min(float(state["initial_radius"]) * 16.0, base * factor),
            )
            state["wheel_armed"] = False
            state["wheel_gate"] = False
            state["pending"] = True
            state["phase"] = "compute"
            state["result"] = None
            _fill_preview_tag_redraw(state)
            return {"RUNNING_MODAL"}
        if event.type in {"RET", "NUMPAD_ENTER", "ENTER"}:
            return self._finish_confirm(context, state)
        if event_type in {"LEFTMOUSE", "RIGHTMOUSE"}:
            # Consume selection/stroke clicks while the preview owns the
            # modal handler.  Neither button confirms nor cancels this tool.
            return {"RUNNING_MODAL"}
        if event.type == "TIMER":
            # Blender 5.2.1 emits Event(type='TIMER', value='NOTHING') without
            # an Event.timer property.  Newer builds may expose the timer and
            # must still be identity-checked.  For the timer-less event, the
            # modal's owning area/window and a short cadence guard provide the
            # available ownership boundary and avoid duplicate foreign ticks.
            event_timer = getattr(event, "timer", None)
            expected_timer = state.get("timer")
            if event_timer is not None and event_timer is not expected_timer:
                return {"RUNNING_MODAL"}
            now = time.perf_counter()
            if (
                event_timer is None
                and now - float(state.get("last_timer_dispatch", 0.0)) < 0.005
            ):
                return {"RUNNING_MODAL"}
            state["last_timer_dispatch"] = now
            if (
                state.get("phase") == "ready"
                and state.get("wheel_gate")
                and state.get("drawn_generation") == state.get("generation")
                and now >= float(state.get("wheel_drain_until", 0.0))
            ):
                # This timer is the first modal boundary after the real draw
                # and the short drain interval; the next wheel is new input.
                state["wheel_gate"] = False
                state["wheel_armed"] = True
            if not self._process_timer(context, state):
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}
        if not _fill_preview_valid(state, context):
            _fill_preview_cancel(state, "stale")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def execute(self, context):
        # Blender can call execute from a scripted invocation. Route it through
        # the same non-destructive modal entry point using the current cursor.
        if not self.poll(context):
            return {"CANCELLED"}
        return self.invoke(
            context,
            type(
                "_PreviewEvent",
                (),
                {
                    "mouse_region_x": self.mouse_region_x,
                    "mouse_region_y": self.mouse_region_y,
                    "ctrl": bool(self.strict_mode),
                },
            )(),
        )


class VIEW3D_OT_mesh_focus_topology_color_assign(bpy.types.Operator):
    """Assign or clear one topology guide color on selected visible faces."""

    bl_idname = TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID
    bl_label = "Mesh Focus: Assign Topology Color"
    bl_description = (
        "Assign the selected visible faces a topology guide color; zero clears it"
    )
    bl_options = {"REGISTER", "UNDO"}

    color_index: IntProperty(
        name="Color",
        description="Stored topology guide color number (0 clears the color)",
        min=0,
        max=6,
        default=0,
        options={"SKIP_SAVE"},
    )

    @classmethod
    def poll(cls, context):
        obj = _topology_color_object(context)
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and obj is not None
        )

    def execute(self, context):
        if not self.poll(context):
            return {"CANCELLED"}
        obj = _topology_color_object(context)
        try:
            bm = bmesh.from_edit_mesh(obj.data)
            selected_faces = [
                face
                for face in bm.faces
                if face.select and not face.hide
            ]
            # A no-selection keypress is intentionally a no-op and does not
            # create the attribute or an unnecessary Undo step.
            if not selected_faces:
                return {"FINISHED"}
            layer = bm.faces.layers.int.get(TOPOLOGY_COLOR_ATTRIBUTE_NAME)
            if layer is None:
                if int(self.color_index) == 0:
                    return {"FINISHED"}
                layer = bm.faces.layers.int.new(TOPOLOGY_COLOR_ATTRIBUTE_NAME)
                # Creating the first custom-data layer can rebuild the
                # BMFace wrappers.  Re-read the selected faces so the write
                # never retains invalid elements across that boundary.
                selected_faces = [
                    face
                    for face in bm.faces
                    if face.select and not face.hide
                ]
            value = int(self.color_index)
            changed = 0
            for face in selected_faces:
                if int(face[layer]) != value:
                    face[layer] = value
                    changed += 1
            if changed:
                bmesh.update_edit_mesh(
                    obj.data,
                    loop_triangles=False,
                    destructive=False,
                )
                obj.data.update_tag()
                _invalidate_topology_color_cache(obj)
            return {"FINISHED"}
        except (
            AttributeError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return {"CANCELLED"}


class VIEW3D_PT_mesh_focus_topology_colors(bpy.types.Panel):
    """N-panel controls for the six stored topology guide colors."""

    bl_idname = "VIEW3D_PT_mesh_focus_topology_colors"
    bl_label = "Topology Colors"
    bl_category = TOPOLOGY_COLOR_PANEL_CATEGORY
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context):
        return _topology_color_object(context) is not None

    def draw(self, context):
        layout = self.layout
        prefs = _addon_preferences()
        if prefs is not None:
            layout.prop(prefs, "topology_colors_enabled", text="表示")
            layout.prop(prefs, "topology_color_opacity", text="透明度")
        layout.label(text="選択面へ割り当て")
        for row_start in (1, 4):
            row = layout.row(align=True)
            for color_index in range(row_start, row_start + 3):
                operator = row.operator(
                    TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID,
                    text=str(color_index),
                )
                operator.color_index = color_index
        clear_operator = layout.operator(
            TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID,
            text="0 解除",
        )
        clear_operator.color_index = 0


def _remove_keymaps():
    for keymap, keymap_item in _addon_keymaps:
        try:
            keymap.keymap_items.remove(keymap_item)
        except (ReferenceError, RuntimeError, ValueError):
            pass
    _addon_keymaps.clear()

    # A module reload replaces this Python list, while Blender keeps the old
    # Addon KeyMapItems.  Remove every stale item owned by this add-on so one
    # physical tap cannot invoke the operator twice and look like a double tap.
    try:
        keyconfig = bpy.context.window_manager.keyconfigs.addon
        if keyconfig is None:
            return
        owned_operator_ids = {
            OPERATOR_ID,
            FACE_SET_ACTIVATION_OPERATOR_ID,
            LOCAL_FACE_SET_GROW_OPERATOR_ID,
            TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID,
        }
        for keymap in keyconfig.keymaps:
            for keymap_item in list(keymap.keymap_items):
                if keymap_item.idname not in owned_operator_ids:
                    continue
                try:
                    keymap.keymap_items.remove(keymap_item)
                except (ReferenceError, RuntimeError, ValueError):
                    pass
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _rebuild_keymaps():
    if not _is_registered:
        return

    _remove_keymaps()
    prefs = _addon_preferences()
    if prefs is not None and not prefs.enabled:
        return

    try:
        keyconfig = bpy.context.window_manager.keyconfigs.addon
        if keyconfig is None:
            return
        keymap = keyconfig.keymaps.new(
            name="3D View",
            space_type="VIEW_3D",
            region_type="WINDOW",
        )
        activation_key = prefs.activation_key if prefs else "RIGHT_SHIFT"
        # Keep normal MFO and Face Set MFO in separate keymap items.  The
        # normal item must not use ``any=True``: otherwise Ctrl+activation_key
        # also reaches this operator and competes with the direct FSMFO item.
        keymap_item = keymap.keymap_items.new(
            OPERATOR_ID,
            activation_key,
            "PRESS",
            any=False,
            ctrl=False,
        )
        _addon_keymaps.append((keymap, keymap_item))

        face_set_item = keymap.keymap_items.new(
            FACE_SET_ACTIVATION_OPERATOR_ID,
            activation_key,
            "PRESS",
            any=False,
            ctrl=True,
        )
        _addon_keymaps.append((keymap, face_set_item))

        local_grow_item = keymap.keymap_items.new(
            LOCAL_FACE_SET_GROW_OPERATOR_ID,
            LOCAL_FACE_SET_GROW_KEY,
            "PRESS",
        )
        _addon_keymaps.append((keymap, local_grow_item))

        strict_grow_item = keymap.keymap_items.new(
            LOCAL_FACE_SET_GROW_OPERATOR_ID,
            LOCAL_FACE_SET_GROW_KEY,
            "PRESS",
            ctrl=True,
        )
        strict_grow_item.properties.strict_mode = True
        _addon_keymaps.append((keymap, strict_grow_item))

        for color_index, key in enumerate(
            ("ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "ZERO"),
            start=1,
        ):
            color_item = keymap.keymap_items.new(
                TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID,
                key,
                "PRESS",
                any=False,
                ctrl=True,
                alt=True,
            )
            color_item.properties.color_index = 0 if key == "ZERO" else color_index
            _addon_keymaps.append((keymap, color_item))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        _remove_keymaps()


def _preferences_changed(_self, _context):
    if _is_registered:
        _rebuild_keymaps()
        _invalidate_topology_color_cache()


class MESH_FOCUS_ORBIT_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__ or __name__

    enabled: BoolProperty(
        name="Enable",
        description="Enable the temporary orbit modifier",
        default=True,
        update=_preferences_changed,
    )
    activation_key: EnumProperty(
        name="Activation Key",
        description="Key to double-tap for entering or leaving temporary orbit",
        items=ACTIVATION_ITEMS,
        default="RIGHT_SHIFT",
        update=_preferences_changed,
    )
    focus_loss_behavior: EnumProperty(
        name="Focus Loss Behavior",
        description="Choose what happens when the Blender window loses focus",
        items=FOCUS_LOSS_ITEMS,
        default="KEEP",
    )
    double_tap_window: FloatProperty(
        name="Double-tap Window",
        description="Maximum time in seconds between the two taps",
        default=DEFAULT_DOUBLE_TAP_WINDOW,
        min=0.15,
        max=0.60,
        precision=2,
    )
    debug_display: BoolProperty(
        name="Debug Display",
        description="Draw a point at the current temporary orbit center",
        default=False,
    )
    show_indicator: BoolProperty(
        name="Show Mode Indicator",
        description="Show MESH FOCUS ORBIT ON in the 3D Viewport",
        default=True,
    )
    retopoflow_target_island_filter: BoolProperty(
        name="RetopoFlow Focus-Island Snap/Weld Filter",
        description=(
            "During Face Set MFO, restrict PolyPen and Auto Merge targets to "
            "the recorded target Retopo island"
        ),
        default=False,
    )
    topology_colors_enabled: BoolProperty(
        name="Topology Colors",
        description="Show the stored six-color topology guide overlay",
        default=True,
        update=_preferences_changed,
    )
    topology_color_opacity: FloatProperty(
        name="Topology Color Opacity",
        description="Opacity of the topology guide face overlay",
        default=0.35,
        min=0.05,
        max=1.0,
        precision=2,
        subtype="FACTOR",
        update=_preferences_changed,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "enabled")
        layout.prop(self, "activation_key")
        layout.prop(context.scene, "mfo_reference_object", text="Reference Object")
        layout.prop(self, "focus_loss_behavior")
        layout.prop(self, "debug_display")
        layout.prop(self, "show_indicator")
        layout.prop(self, "retopoflow_target_island_filter")
        layout.prop(self, "topology_colors_enabled")
        layout.prop(self, "topology_color_opacity")
        layout.prop(self, "double_tap_window")
        layout.separator()
        layout.label(text="Double-tap the activation key in a 3D Viewport.")
        layout.label(text="Ctrl + double-tap uses Face Set MFO on the Reference Object.")
        layout.label(text="The viewport center is ray-cast once on activation.")
        layout.label(text="Ctrl + Alt + 1..6 assigns selected faces; 0 clears.")


CLASSES = (
    VIEW3D_OT_mesh_focus_orbit_recover_face_set_state,
    VIEW3D_OT_mesh_focus_orbit_watcher,
    VIEW3D_OT_mesh_focus_face_set_activate,
    VIEW3D_OT_mesh_focus_orbit,
    VIEW3D_OT_mesh_focus_local_face_set_grow,
    VIEW3D_OT_mesh_focus_topology_color_assign,
    VIEW3D_PT_mesh_focus_topology_colors,
    MESH_FOCUS_ORBIT_AddonPreferences,
)


def register():
    global _is_registered
    if _is_registered:
        return
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    if not hasattr(bpy.types.Scene, "mfo_reference_object"):
        bpy.types.Scene.mfo_reference_object = PointerProperty(
            name="MFO Reference Object",
            description=(
                "Mesh used by Mesh Focus Orbit and Face Set MFO for center ray casting"
            ),
            type=bpy.types.Object,
            poll=_reference_object_poll,
        )
    _is_registered = True
    # A script reload can leave an old MFO wrapper on RetopoFlow's class even
    # though the previous module no longer has Python state for it.
    _restore_retopoflow_hooks()
    _schedule_orphan_cleanup()
    _start_topology_color_draw()
    if _on_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_load_pre)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    if _on_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_on_undo_post)
    if _on_topology_color_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(
            _on_topology_color_depsgraph_update
        )
    if _on_topology_color_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_on_topology_color_undo_post)
    if _on_topology_color_redo_post not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(_on_topology_color_redo_post)
    if _on_topology_color_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_topology_color_load_pre)
    if _on_topology_color_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_topology_color_load_post)
    if _on_fill_preview_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_fill_preview_depsgraph_update)
    _rebuild_keymaps()


def unregister():
    global _is_registered
    if not _is_registered:
        _fill_preview_cancel(reason="unregister")
        _cancel_undo_orphan_cleanup()
        _retopo_undo_tombstones.clear()
        _retopo_debug_sessions.clear()
        _retopo_debug_retired_sessions.clear()
        if _on_undo_post in bpy.app.handlers.undo_post:
            bpy.app.handlers.undo_post.remove(_on_undo_post)
        for _handler_list_name, _handler in (
            ("depsgraph_update_post", _on_topology_color_depsgraph_update),
            ("undo_post", _on_topology_color_undo_post),
            ("redo_post", _on_topology_color_redo_post),
            ("load_pre", _on_topology_color_load_pre),
            ("load_post", _on_topology_color_load_post),
        ):
            _handler_list = getattr(bpy.app.handlers, _handler_list_name)
            if _handler in _handler_list:
                _handler_list.remove(_handler)
        if _on_fill_preview_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(_on_fill_preview_depsgraph_update)
        _restore_retopoflow_hooks()
        _stop_topology_color_draw()
        return
    _finish_all_states()
    _fill_preview_cancel(reason="unregister")
    _cancel_undo_orphan_cleanup()
    _cleanup_orphan_face_set_proxies()
    _retopo_undo_tombstones.clear()
    _retopo_debug_sessions.clear()
    _retopo_debug_retired_sessions.clear()
    _restore_retopoflow_hooks()
    if _on_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_on_load_pre)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    if _on_undo_post in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(_on_undo_post)
    for _handler_list_name, _handler in (
        ("depsgraph_update_post", _on_topology_color_depsgraph_update),
        ("undo_post", _on_topology_color_undo_post),
        ("redo_post", _on_topology_color_redo_post),
        ("load_pre", _on_topology_color_load_pre),
        ("load_post", _on_topology_color_load_post),
    ):
        _handler_list = getattr(bpy.app.handlers, _handler_list_name)
        if _handler in _handler_list:
            _handler_list.remove(_handler)
    if _on_fill_preview_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_fill_preview_depsgraph_update)
    _stop_topology_color_draw()
    _remove_keymaps()
    _local_face_set_adjacency_cache.clear()
    _fill_preview_adjacency_cache.clear()
    _fill_preview_cursor_cache.clear()
    _is_registered = False
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, "mfo_reference_object"):
        del bpy.types.Scene.mfo_reference_object


if __name__ == "__main__":
    register()
def _fill_preview_build_adjacency_cooperative(obj, prepared=None):
    """Build only the reusable surface graph needed by preview sessions.

    This preparation reads all loops once, but intentionally does not compute
    curvature, valley bands, partition ownership, or contour relaxation.  All
    geometry features are recomputed on the session's candidate+halo patch.
    """
    import numpy as np

    mesh = obj.data
    signature = _fill_preview_signature(obj)
    if signature is None:
        raise RuntimeError("preview target is unavailable")
    cached = _fill_preview_adjacency_cache.get(signature)
    if cached is not None:
        return cached

    if prepared is None:
        vertex_count = len(mesh.vertices)
        face_count = len(mesh.polygons)
        coordinates = np.empty((vertex_count, 3), dtype=np.float32)
        loop_edges = np.empty(len(mesh.loops), dtype=np.int32)
        loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
        totals = np.empty(face_count, dtype=np.int32)
        hidden = np.empty(face_count, dtype=bool)
        mesh.vertices.foreach_get("co", coordinates.ravel())
        mesh.loops.foreach_get("edge_index", loop_edges)
        mesh.loops.foreach_get("vertex_index", loop_vertices)
        mesh.polygons.foreach_get("loop_total", totals)
        mesh.polygons.foreach_get("hide", hidden)
        centers = np.empty((face_count, 3), dtype=np.float32)
        normals = np.empty((face_count, 3), dtype=np.float32)
        mesh.polygons.foreach_get("center", centers.ravel())
        mesh.polygons.foreach_get("normal", normals.ravel())
    else:
        coordinates = prepared["coordinates"]
        loop_edges = prepared["loop_edges"]
        loop_vertices = prepared["loop_vertices"]
        totals = prepared["totals"]
        hidden = prepared["hidden"]
        centers = prepared["centers"]
        normals = prepared["normals"]
        vertex_count = len(coordinates)
        face_count = len(totals)

    transform = np.asarray(obj.matrix_world.to_3x3(), dtype=np.float64)
    world_matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    translation = world_matrix[:3, 3]
    world_vertices = coordinates.astype(np.float64) @ transform.T + translation
    centers = centers.astype(np.float64) @ transform.T + translation
    normals = normals.astype(np.float64) @ np.linalg.inv(transform)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-20)
    yield "world-space"

    face_ids = np.repeat(np.arange(face_count, dtype=np.int32), totals)
    order = np.argsort(loop_edges, kind="stable")
    sorted_edges = loop_edges[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_edges)) + 1]
    lengths = np.diff(np.r_[starts, len(order)])
    pair_starts = starts[lengths == 2]
    paired_loops = order[pair_starts]
    first = face_ids[paired_loops]
    second = face_ids[order[pair_starts + 1]]
    face_starts = np.r_[0, np.cumsum(totals)[:-1]]
    paired_next = face_starts[first] + (
        (paired_loops - face_starts[first] + 1) % totals[first]
    )
    edge_v0 = loop_vertices[paired_loops]
    edge_v1 = loop_vertices[paired_next]
    yield "edge-pairs"

    yield "seam-before"
    # Preserve the existing unwelded-seam behavior.  Only coincident open
    # edges with opposite winding and compatible normals become bridges.
    border_loops = order[starts[lengths == 1]]
    seam_count = 0
    if len(border_loops):
        border_faces = face_ids[border_loops]
        next_loops = face_starts[border_faces] + (
            (border_loops - face_starts[border_faces] + 1) % totals[border_faces]
        )
        points_a = world_vertices[loop_vertices[border_loops]]
        points_b = world_vertices[loop_vertices[next_loops]]
        edge_lengths = np.linalg.norm(points_b - points_a, axis=1)
        tolerance = max(float(np.median(edge_lengths)) * 1.0e-4, 1.0e-12)
        quantized = np.rint(np.vstack((points_a, points_b)) / tolerance).astype(np.int64)
        _, point_ids = np.unique(quantized, axis=0, return_inverse=True)
        pa, pb = np.split(point_ids, 2)
        signatures = np.column_stack((np.minimum(pa, pb), np.maximum(pa, pb)))
        _, inverse, counts = np.unique(
            signatures, axis=0, return_inverse=True, return_counts=True
        )
        seam_order = np.argsort(inverse, kind="stable")
        seam_starts = np.r_[0, np.cumsum(counts)[:-1]][counts == 2]
        ia, ib = seam_order[seam_starts], seam_order[seam_starts + 1]
        fa, fb = border_faces[ia], border_faces[ib]
        match = (
            (pa[ia] == pb[ib])
            & (pb[ia] == pa[ib])
            & (pa[ia] != pb[ia])
            & (fa != fb)
            & (np.sum(normals[fa] * normals[fb], axis=1) > 0.5)
            & (np.linalg.norm(points_a[ia] - points_b[ib], axis=1) <= tolerance)
            & (np.linalg.norm(points_b[ia] - points_a[ib], axis=1) <= tolerance)
        )
        first = np.r_[first, fa[match]]
        second = np.r_[second, fb[match]]
        edge_v0 = np.r_[edge_v0, loop_vertices[border_loops[ia[match]]]]
        edge_v1 = np.r_[edge_v1, loop_vertices[next_loops[ia[match]]]]
        seam_count = int(np.count_nonzero(match))
        yield "seam-after"

    valid = ~(hidden[first] | hidden[second]) & (first != second)
    first, second = first[valid], second[valid]
    edge_v0, edge_v1 = edge_v0[valid], edge_v1[valid]
    delta = centers[second] - centers[first]
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-20)
    sources = np.r_[first, second]
    destinations = np.r_[second, first]
    source_edges = np.r_[np.arange(len(first)), np.arange(len(first))]
    edge_v0_directed = np.r_[edge_v0, edge_v0]
    edge_v1_directed = np.r_[edge_v1, edge_v1]
    order = np.argsort(sources, kind="stable")
    yield "csr-before"
    degree = np.bincount(sources, minlength=face_count)
    offsets = np.r_[0, np.cumsum(degree)].astype(np.int64)
    yield "csr-after"
    cached = {
        "signature": signature,
        "count": int(face_count),
        "centers": centers,
        "normals": normals,
        "hidden": hidden,
        "world_vertices": world_vertices,
        "offsets": offsets,
        "neighbors": destinations[order].astype(np.int32, copy=False),
        "neighbor_lengths": np.r_[distance, distance][order],
        "edge_v0": edge_v0_directed[order].astype(np.int32, copy=False),
        "edge_v1": edge_v1_directed[order].astype(np.int32, copy=False),
        "first": first,
        "second": second,
        "pair_lengths": distance,
        "pair_v0": edge_v0,
        "pair_v1": edge_v1,
        "seam_count": seam_count,
    }
    # Drop obsolete revisions for this mesh while retaining unrelated meshes.
    mesh_pointer = signature[1]
    for key in list(_fill_preview_adjacency_cache):
        if key[1] == mesh_pointer and key != signature:
            _fill_preview_adjacency_cache.pop(key, None)
    _fill_preview_adjacency_cache[signature] = cached
    return cached
