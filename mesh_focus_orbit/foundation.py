"""MFO foundation component.

Loaded by the package entry point into the single legacy namespace. This keeps
existing mutable-state and monkeypatch contracts while physically owning the
foundation implementation.
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
from mathutils import Matrix, Vector
from mathutils.geometry import tessellate_polygon
from mathutils.bvhtree import BVHTree

OPERATOR_ID = "view3d.mesh_focus_orbit"
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
from . import runtime as _runtime
FACE_SET_ACTIVATION_OPERATOR_ID = "view3d.mesh_focus_face_set_activate"
WATCHER_OPERATOR_ID = "view3d.mesh_focus_orbit_watcher"
RECOVER_FACE_SET_STATE_OPERATOR_ID = "view3d.mesh_focus_orbit_recover_face_set_state"
LOCAL_FACE_SET_GROW_OPERATOR_ID = "view3d.mesh_focus_local_face_set_grow_v2"
LOCAL_FACE_SET_GROW_KEY = "E"
TUBE_SHAPE_OPERATOR_ID = "view3d.mesh_focus_tube_shape"
TUBE_SHAPE_KEY = "T"
LOCAL_FEATURE_BRUSH_OPERATOR_ID = "view3d.mesh_focus_local_feature_brush"
LOCAL_FEATURE_BRUSH_KEY = "F"
GUIDED_RIDGE_OPERATOR_ID = "view3d.mesh_focus_guided_ridge"
GUIDED_RIDGE_KEY = "G"
TOOL_NORMAL_OPERATOR_ID = "view3d.mesh_focus_orbit_tool"
TOOL_FACE_SET_OPERATOR_ID = "view3d.mesh_focus_face_set_tool"
GUIDED_RIDGE_MAX_CONTROLS = 64
GUIDED_RIDGE_MAX_CANDIDATE_FACES = 50000
GUIDED_RIDGE_DISTANCE_MAX_PAIRS = 262144
GUIDED_RIDGE_PREPARE_WORK_CHUNK = 256
# Phase 1 deliberately exposes only the non-destructive guide/rail UI.  Keep
# this switch beside the Guided Ridge contract so the future deformation
# engine can be reconnected in one explicit, reviewable place.
GUIDED_RIDGE_UI_PROTOTYPE = True
# The local-feature implementation is selected as a normal Sculpt Brush
# asset.  The marker is an ID property saved with the dedicated asset; its
# name and datablock pointer are deliberately not part of the identity.
LOCAL_FEATURE_BRUSH_MARKER_PROPERTY = "mfo_local_feature_marker"
LOCAL_FEATURE_BRUSH_MARKER_VALUE = "local-feature-brush-v1"
LOCAL_FEATURE_BRUSH_CATALOG_ID = "2c0f6e95-2f2a-4bd9-8b8f-3b8f741d8e3a"
_FILL_PREVIEW_ANALYSIS_SHADER_PROFILE = {
    "name": "MFO shadow analysis: toon_dark matcap",
    "light": "MATCAP",
    "matcap_candidates": ("toon_dark.exr", "basic_dark.exr"),
}
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

_addon_keymaps = _runtime.addon_keymaps
_shadow_analysis_view_tokens = _runtime.shadow_analysis_view_tokens
_active_states = _runtime.active_states
_last_tap_times = _runtime.last_tap_times
_session_cleanup_areas = _runtime.session_cleanup_areas
_retopo_undo_tombstones = _runtime.retopo_undo_tombstones
_local_face_set_adjacency_cache = _runtime.local_face_set_adjacency_cache
_fill_preview_adjacency_cache = _runtime.fill_preview_adjacency_cache
_fill_preview_cursor_cache = _runtime.fill_preview_cursor_cache
# Curve Sculpt Step 1 keeps the existing Guided Ridge route editor but stops
# before every geometry/paint-curve operation.  This explicit gate is checked
# by the interactive entry points; the older candidate engine remains in the
# module for a later step and for isolated reference tests.
GUIDED_RIDGE_CURVE_SCULPT_STEP1 = True

from .viewport import sculpt_cursor_region_coordinate as _sculpt_cursor_region_coordinate
# A new route enters the preview as a close, clean inscribed curve.  Keep this
# as a named policy value rather than burying a magic number in the Enter path:
# Shape 0 remains the exact angular route, while this small negative gain keeps
# the first C1 result near the hand-placed envelope instead of drifting toward
# the endpoint chord or accidentally starting in the amplified direction.
GUIDED_RIDGE_CURVE_DEFAULT_SMOOTHING = -12.0
GUIDED_RIDGE_WIDTH_EDGE_MULTIPLIER = 8.0
GUIDED_RIDGE_WIDTH_MIN_EDGE_MULTIPLIER = 4.0
GUIDED_RIDGE_WIDTH_MAX_EXTENT_FRACTION = 0.45
GUIDED_RIDGE_WIDTH_STEP_FACTOR = 1.18
GUIDED_RIDGE_WIDTH_GUIDE_BAND_FRACTION = 0.08
GUIDED_RIDGE_WIDTH_ANCHOR_BAND_FRACTION = 0.14
_FILL_PREVIEW_WHEEL_DRAIN_SECONDS = 0.12
_FILL_PREVIEW_TERMINAL_GROWTH_STAGES = 3
_FILL_PREVIEW_TERMINAL_MAX_RADIUS_FACTOR = 16.0
_FILL_PREVIEW_TERMINAL_THRESHOLD_FACTOR = (
    _FILL_PREVIEW_TERMINAL_MAX_RADIUS_FACTOR
    / (1.25 ** _FILL_PREVIEW_TERMINAL_GROWTH_STAGES)
)
# Normal-mode frontier-delta begins at this selected-face count.  The value
# is deliberately count based so large selections can use differential edge
# maintenance before the radius-only terminal stages.
_FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = 100_000
_FILL_PREVIEW_BOUNDARY_ROUTE_SHAPE_PAIR_BUDGET = 128
_FILL_PREVIEW_BOUNDARY_ROUTE_COMPONENT_BUDGET = 8
_FILL_PREVIEW_BOUNDARY_ROUTE_COMPONENT_PAIR_BUDGET = 512
_FILL_PREVIEW_BOUNDARY_ROUTE_TOTAL_PAIR_BUDGET = 2048
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
    vec4 view_pos = ModelViewMatrix * vec4(pos, 1.0);
    vec4 clip = ProjectionMatrix * view_pos;
    if (retopology_offset > 0.0) {
        if (ProjectionMatrix[3][3] == 0.0) {
            float offset = min(retopology_offset, -view_pos.z * 0.5);
            clip.z += ProjectionMatrix[3][2]
                * (offset / (view_pos.z * (view_pos.z + offset)))
                * clip.w;
        }
        else {
            clip.z += ProjectionMatrix[2][2] * retopology_offset * clip.w;
        }
    }
    gl_Position = clip;
}
"""

_TOPOLOGY_COLOR_DEPTH_FRAGMENT_SOURCE = """
void main()
{
    fragColor = color;
}
"""

_TOPOLOGY_COLOR_HANDLER_NAMES = {
    "_on_topology_color_depsgraph_update",
    "_on_topology_color_undo_post",
    "_on_topology_color_redo_post",
    "_on_topology_color_load_pre",
    "_on_topology_color_load_post",
}
_FILL_PREVIEW_HANDLER_NAMES = {
    "_on_fill_preview_depsgraph_update",
    "_on_fill_preview_redo_post",
}
_TUBE_PREVIEW_HANDLER_NAMES = {
    "_on_tube_preview_depsgraph_update",
}
_LOCAL_FEATURE_BRUSH_HANDLER_NAMES = {
    "_on_local_feature_brush_depsgraph_update",
    "_on_local_feature_brush_undo_post",
    "_on_local_feature_brush_redo_post",
    "_on_local_feature_brush_load_pre",
    "_on_local_feature_brush_load_post",
}
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


def _fill_preview_shader_get():
    """Return the shared preview shader owned by the runtime singleton."""
    if _runtime.fill_preview_shader is None:
        try:
            _runtime.fill_preview_shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _runtime.fill_preview_shader = None
    return _runtime.fill_preview_shader


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
    _runtime.session_serial += 1
    return _runtime.session_serial


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
    reconcile_started = time.perf_counter()
    _runtime.retopo_undo_retry_pending = False
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
            _runtime.retopo_undo_retry_pending = True
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
                _runtime.retopo_undo_retry_pending = True
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
                _runtime.retopo_undo_retry_pending = True
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

    return _world_ray_from_region_coordinate(context, center_2d)


def _world_ray_from_region_coordinate(context, coordinate):
    """Return a world ray for one owning WINDOW-region coordinate.

    Tool activation must use the actual click location.  Keeping this helper
    separate from the center-ray wrapper makes it difficult for a toolbar
    event to accidentally regress to the old viewport-center behaviour.
    """
    region = getattr(context, "region", None)
    space_data = getattr(context, "space_data", None)
    rv3d = getattr(space_data, "region_3d", None)
    if region is None or rv3d is None:
        return None

    try:
        coordinate = Vector(coordinate)
        if not (0.0 <= float(coordinate.x) < float(region.width)):
            return None
        if not (0.0 <= float(coordinate.y) < float(region.height)):
            return None
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coordinate)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coordinate)
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


_FLOAT32_SIGN_MASK = 0x80000000
_FLOAT32_MIN_SUBNORMAL = 1.401298464324817e-45


def _float32_step_toward(value, direction):
    """Return the next float32 spacing from ``value`` along ``direction``."""
    value = float(value)
    direction = float(direction)
    if not math.isfinite(value) or direction == 0.0:
        return None
    if value == 0.0:
        return _FLOAT32_MIN_SUBNORMAL

    try:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
    except (OverflowError, struct.error):
        return None

    if direction > 0.0:
        next_bits = bits - 1 if bits & _FLOAT32_SIGN_MASK else bits + 1
    else:
        next_bits = bits + 1 if bits & _FLOAT32_SIGN_MASK else bits - 1

    try:
        next_value = struct.unpack(
            "<f", struct.pack("<I", next_bits)
        )[0]
    except (OverflowError, struct.error):
        return None
    spacing = abs(next_value - value)
    return spacing if math.isfinite(spacing) and spacing > 0.0 else None


def _edit_ray_advance_epsilon(local_location, local_direction, local_normal):
    """Find the smallest float32-safe advance that exits a hidden hit."""
    candidates = []
    for coordinate, component in zip(local_location, local_direction):
        component = float(component)
        if not math.isfinite(component) or component == 0.0:
            continue
        spacing = _float32_step_toward(coordinate, component)
        if spacing is None:
            continue
        epsilon = spacing / abs(component)
        if math.isfinite(epsilon) and epsilon > 0.0:
            candidates.append(epsilon)

    if not candidates:
        return None

    try:
        normal_length = local_normal.length
    except (AttributeError, TypeError, ValueError):
        normal_length = 0.0
    ray_normal = (
        local_normal.dot(local_direction) if normal_length > 1.0e-12 else 0.0
    )

    # Test candidates in ascending order using mathutils' actual float32
    # vector arithmetic.  For a slanted ray, a tangent coordinate can be the
    # first ULP candidate while the normal coordinate still rounds unchanged;
    # reject that candidate and retain the smallest one that moves through the
    # hit surface.  This avoids both same-face re-hits and over-large gaps.
    for epsilon in sorted(set(candidates)):
        epsilon = math.nextafter(epsilon, math.inf)
        advanced = local_location + local_direction * epsilon
        delta = advanced - local_location
        if not any(float(value) != 0.0 for value in delta):
            continue
        if ray_normal != 0.0 and local_normal.dot(delta) * ray_normal <= 0.0:
            continue
        return epsilon

    # A degenerate/zero BVH normal still gets the smallest candidate that
    # actually changes a float32 component.  The caller's bounded retry loop
    # remains the final guard for malformed geometry.
    for epsilon in reversed(sorted(set(candidates))):
        epsilon = math.nextafter(epsilon, math.inf)
        advanced = local_location + local_direction * epsilon
        if any(float(value) != 0.0 for value in advanced - local_location):
            return epsilon
    return None


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


def _raycast_sculpt_visible_object(obj, origin, direction):
    """Skip hidden Sculpt faces when finding the clicked orbit surface.

    Evaluated Mesh polygon hide flags do not reliably reflect Sculpt's live
    Face Set visibility.  Use the source mesh ray cast, as the Sculpt cursor
    does, and advance past each hidden face before accepting a hit.
    """
    try:
        matrix_world = obj.matrix_world
        inverse = matrix_world.inverted_safe()
        local_origin = inverse @ origin
        local_direction = inverse.to_3x3() @ direction
        if local_direction.length_squared == 0.0:
            return None
        local_direction.normalize()
        polygons = obj.data.polygons
        last_hidden_face = None
        repeated_hidden_hits = 0

        for _attempt in range(256):
            hit, local_location, local_normal, face_index = obj.ray_cast(
                local_origin, local_direction
            )
            if not hit or face_index < 0 or face_index >= len(polygons):
                return None
            world_location = matrix_world @ local_location
            distance = _distance_along_ray(origin, direction, world_location)
            if distance is None:
                return None
            if not polygons[int(face_index)].hide:
                return world_location, distance

            if face_index == last_hidden_face:
                repeated_hidden_hits += 1
            else:
                last_hidden_face = face_index
                repeated_hidden_hits = 1
            if repeated_hidden_hits > 8:
                return None
            advance = _edit_ray_advance_epsilon(
                local_location, local_direction, local_normal
            )
            if advance is None:
                return None
            local_origin = local_location + local_direction * advance
        return None
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _raycast_edit_object(obj, origin, direction):
    """Ray cast the live BMesh so unsaved Edit Mode changes are included."""
    try:
        bm = bmesh.from_edit_mesh(obj.data)
    except (AttributeError, RuntimeError, TypeError):
        return None

    # FromBMesh consumes the live edit BMesh directly.  Refreshing the face
    # lookup/index tables is a C-side operation and makes the BVH hit index
    # map back to ``bm.faces[index]`` without copying millions of faces into
    # Python.  The tree is deliberately local to this activation: an edit
    # operation can invalidate both the BMesh and its spatial index.
    try:
        bm.faces.ensure_lookup_table()
        bm.faces.index_update()
        bvh = BVHTree.FromBMesh(bm)
        matrix_world = obj.matrix_world
        inverse = matrix_world.inverted_safe()
        local_origin = inverse @ origin
        local_direction = inverse.to_3x3() @ direction
        if local_direction.length_squared == 0.0:
            return None
        local_direction.normalize()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None

    # Blender's BVH includes hidden BMesh faces/vertices.  A hit on one is
    # skipped by starting the next query just beyond the hit.  The bounded
    # loop handles hidden layers and prevents malformed edit geometry from
    # causing an unbounded sequence of ray casts.
    ray_origin = local_origin
    max_hidden_hits = 64
    last_hidden_face_index = None
    same_hidden_face_hits = 0
    for _ in range(max_hidden_hits + 1):
        try:
            hit = bvh.ray_cast(ray_origin, local_direction)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

        local_location, local_normal, face_index, _local_distance = hit
        if local_location is None or face_index is None:
            return None

        try:
            face = bm.faces[int(face_index)]
        except (IndexError, ReferenceError, RuntimeError, TypeError, ValueError):
            return None

        try:
            visible = (
                face.is_valid
                and not face.hide
                and not any(loop.vert.hide for loop in face.loops)
            )
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            return None

        try:
            if local_normal is None or local_normal.length <= 1.0e-12:
                local_normal = face.normal.copy()
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            local_normal = None

        if visible:
            world_location = matrix_world @ local_location
            distance = _distance_along_ray(origin, direction, world_location)
            return (
                (world_location.copy(), distance)
                if distance is not None
                else None
            )

        if face_index == last_hidden_face_index:
            same_hidden_face_hits += 1
        else:
            last_hidden_face_index = face_index
            same_hidden_face_hits = 1
        if same_hidden_face_hits > max_hidden_hits:
            return None

        # Advance by the smallest float32-safe ULP along the ray.  The helper
        # checks the hit normal so a slanted ray cannot choose a tangent-axis
        # ULP that leaves the normal coordinate unchanged.
        epsilon = _edit_ray_advance_epsilon(
            local_location,
            local_direction,
            local_normal,
        )
        if epsilon is None:
            return None
        ray_origin = local_location + local_direction * epsilon

    return None


def _find_mesh_hit(context, ray):
    """Find the nearest visible mesh surface along an already chosen ray."""
    if ray is None:
        return None
    origin, direction = ray
    depsgraph = context.evaluated_depsgraph_get()
    best = None

    # Normal MFO follows what is actually visible in this viewport.  In
    # particular, do not use the preference Reference Object here: it may be
    # unset, hidden, or farther away than another visible mesh.  The existing
    # visibility helper also excludes collection-hidden and local-view objects.
    try:
        candidates = context.view_layer.objects
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return None

    for obj in candidates:
        if obj.type != "MESH" or not _object_is_visible(context, obj):
            continue

        if context.mode == "EDIT_MESH" and obj.mode == "EDIT":
            hit = _raycast_edit_object(obj, origin, direction)
        elif context.mode == "SCULPT" and obj is context.active_object:
            hit = _raycast_sculpt_visible_object(obj, origin, direction)
        else:
            hit = _raycast_object(obj, depsgraph, origin, direction)
        if hit is None:
            continue
        if best is None or hit[1] < best[1]:
            best = hit

    return best


def _find_center_hit(context):
    """Find the nearest visible mesh surface under the viewport center."""
    return _find_mesh_hit(context, _world_ray_from_view_center(context))


def _find_mesh_hit_at_coordinate(context, coordinate):
    """Find the nearest visible mesh surface under *coordinate*."""
    return _find_mesh_hit(
        context,
        _world_ray_from_region_coordinate(context, coordinate),
    )


def _raycast_reference_face_set_at_coordinate(context, coordinate):
    """Return the configured Reference Face Set hit at *coordinate*."""
    obj = _get_reference_object(context)
    if obj is None:
        return None

    face_set_attr = obj.data.attributes.get(".sculpt_face_set")
    if face_set_attr is None or face_set_attr.domain != "FACE":
        return None

    ray = _world_ray_from_region_coordinate(context, coordinate)
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


def _raycast_reference_face_set(context):
    """Return the center hit and Face Set ID from the configured reference."""
    region = getattr(context, "region", None)
    if region is None:
        return None
    return _raycast_reference_face_set_at_coordinate(
        context,
        Vector((float(region.width) * 0.5, float(region.height) * 0.5)),
    )


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


def _topology_color_depth_shader():
    """Create one cached shader matching Retopology overlay depth offset."""
    global _topology_color_depth_shader_cache
    if _topology_color_depth_shader_cache is not None:
        return _topology_color_depth_shader_cache
    try:
        from gpu.types import GPUShaderCreateInfo

        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewMatrix")
        info.push_constant("MAT4", "ProjectionMatrix")
        info.push_constant("FLOAT", "retopology_offset")
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


def _topology_color_canonical_cycle(vertex_slots):
    """Normalize a face cycle's start slot without changing its winding."""
    vertex_slots = tuple(vertex_slots)
    if not vertex_slots:
        return ()
    start = min(range(len(vertex_slots)), key=vertex_slots.__getitem__)
    return vertex_slots[start:] + vertex_slots[:start]


def _topology_color_face_slots(face, expected_cycle, bm_vertices):
    """Resolve a cached face cycle through BMesh sequence slots only."""
    try:
        face_vertices = tuple(face.verts)
        if len(face_vertices) != len(expected_cycle):
            return None
        current_cycle = []
        for vertex in face_vertices:
            matched_slot = None
            for slot in expected_cycle:
                if slot < 0 or slot >= len(bm_vertices):
                    return None
                if bm_vertices[slot] == vertex:
                    matched_slot = slot
                    break
            if matched_slot is None:
                return None
            current_cycle.append(matched_slot)
        current_cycle = tuple(current_cycle)
        if _topology_color_canonical_cycle(current_cycle) != tuple(expected_cycle):
            return None
        return current_cycle
    except (AttributeError, IndexError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _topology_color_face_triangle_slots(
    face,
    face_vertex_slots,
    coordinate_overrides=None,
):
    """Tessellate one colored face into copied BMesh sequence slots."""
    try:
        face_vertices = tuple(face.verts)
        if len(face_vertices) != len(face_vertex_slots):
            return None
        if len(face_vertices) == 4:
            # Match Blender's native fast quad tessellation in the original
            # BMesh loop order. Keep the strict predicate so diagonal choice
            # follows the same rule as the edit-mesh depth prepass.
            if coordinate_overrides is None:
                v0, v1, v2, v3 = face_vertices
                coordinates = (v0.co, v1.co, v2.co, v3.co)
            else:
                coordinates = tuple(
                    Vector(coordinate_overrides[slot])
                    for slot in face_vertex_slots
                )
            v0, v1, v2, v3 = coordinates
            a = v1 - v0
            b = v2 - v0
            c = v3 - v0
            if a.cross(b).dot(c.cross(b)) > 0.0:
                triangle_indices = ((0, 1, 3), (1, 2, 3))
            else:
                triangle_indices = ((0, 1, 2), (0, 2, 3))
            return tuple(
                tuple(face_vertex_slots[index] for index in triangle)
                for triangle in triangle_indices
            )
        vertex_slots = {
            vertex: slot
            for vertex, slot in zip(face_vertices, face_vertex_slots)
        }
        try:
            tessellation = face.calc_tessellation()
            tessellation_vertices = None
        except (AttributeError, RuntimeError, TypeError, ValueError):
            tessellation_vertices = [
                vertex.co.copy()
                for vertex in face_vertices
            ]
            tessellation = tessellate_polygon([tessellation_vertices])

        triangles = []
        for triangle in tessellation:
            if len(triangle) != 3:
                continue
            triangle_slots = []
            for vertex in triangle:
                if isinstance(vertex, int):
                    if vertex < 0 or vertex >= len(face_vertex_slots):
                        return None
                    slot = face_vertex_slots[vertex]
                else:
                    slot = vertex_slots.get(vertex)
                    if slot is None:
                        return None
                triangle_slots.append(slot)
            triangles.append(tuple(triangle_slots))
        return tuple(triangles)
    except (AttributeError, IndexError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _topology_color_cache_face_states(cache, bm):
    """Check only copied colored faces, including hidden faces, for layout changes."""
    if (
        cache.get("vert_count") != len(bm.verts)
        or cache.get("edge_count") != len(bm.edges)
        or cache.get("face_count") != len(bm.faces)
    ):
        return None

    layer = bm.faces.layers.int.get(TOPOLOGY_COLOR_ATTRIBUTE_NAME)
    if bool(cache.get("topology_color_layer_present")) != (layer is not None):
        return None
    face_layout = cache.get("colored_face_layout")
    if face_layout is None:
        return None
    if layer is None:
        return [] if not face_layout else None

    try:
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        face_states = []
        for record in face_layout:
            face_slot = int(record["face_slot"])
            if face_slot < 0 or face_slot >= len(bm.faces):
                return None
            face = bm.faces[face_slot]
            if (
                int(face[layer]) != record["color_index"]
                or bool(face.hide) != record["hidden"]
            ):
                return None
            face_vertex_slots = _topology_color_face_slots(
                face,
                record["vertex_cycle"],
                bm.verts,
            )
            if face_vertex_slots is None:
                return None
            if tuple(face_vertex_slots) != tuple(
                record.get("ordered_vertex_cycle", ())
            ):
                return None
            face_states.append((record, face, face_vertex_slots))
        return face_states
    except (AttributeError, IndexError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _topology_color_cache_current_coordinates(cache, bm):
    """Copy coordinates for cached colored vertices without visiting uncolored vertices."""
    try:
        bm.verts.ensure_lookup_table()
        coordinates = {}
        for slot in cache.get("colored_vertex_slots", ()):
            if slot < 0 or slot >= len(bm.verts):
                return None
            coordinates[slot] = tuple(
                float(value)
                for value in bm.verts[slot].co
            )
        return coordinates
    except (AttributeError, IndexError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _topology_color_cache_geometry_matches(cache, bm):
    """Check colored-face layout and coordinates without full-mesh walks or index writes."""
    if _topology_color_cache_face_states(cache, bm) is None:
        return False
    coordinates = _topology_color_cache_current_coordinates(cache, bm)
    return (
        coordinates is not None
        and coordinates == cache.get("colored_vertex_snapshot")
    )


def _topology_color_mirror_configuration(obj, include_transforms=True):
    """Return a value-only Mirror signature and supported raw-copy settings."""
    modifiers = tuple(obj.modifiers)
    signature = []
    visible_mirrors = []
    for index, modifier in enumerate(modifiers):
        try:
            modifier_type = str(modifier.type)
            show_viewport = bool(modifier.show_viewport)
            show_in_editmode = bool(getattr(modifier, "show_in_editmode", False))
            show_on_cage = bool(getattr(modifier, "show_on_cage", False))
            modifier_pointer = int(modifier.as_pointer())
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            return None, None

        row = (
            index,
            modifier_pointer,
            str(getattr(modifier, "name", "")),
            modifier_type,
            show_viewport,
            show_in_editmode,
            show_on_cage,
        )
        if modifier_type == "MIRROR":
            try:
                mirror_object = modifier.mirror_object
                axes = tuple(bool(value) for value in modifier.use_axis)
                bisect_axes = tuple(bool(value) for value in modifier.use_bisect_axis)
                bisect_flips = tuple(bool(value) for value in modifier.use_bisect_flip_axis)
                mirror_object_pointer = (
                    int(mirror_object.as_pointer()) if mirror_object is not None else 0
                )
                mirror_object_matrix = (
                    _topology_color_matrix_key(mirror_object)
                    if mirror_object is not None
                    else None
                )
                use_clip = bool(modifier.use_clip)
                use_merge = bool(modifier.use_mirror_merge)
                merge_threshold = float(modifier.merge_threshold)
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                return None, None
            row += (
                axes,
                bisect_axes,
                bisect_flips,
                use_clip,
                use_merge,
                merge_threshold,
                mirror_object_pointer,
                mirror_object_matrix,
            )
            if show_viewport and show_in_editmode:
                visible_mirrors.append((index, modifier, mirror_object, axes, bisect_axes))
        signature.append(row)

    signature = tuple(signature)
    # Multiple visible Mirrors and bisected/pre-deformed inputs need a fuller
    # evaluated topology map. Keep the raw source guide and fail closed here.
    if len(visible_mirrors) != 1:
        return signature, None
    mirror_index, modifier, mirror_object, axes, bisect_axes = visible_mirrors[0]
    if any(bisect_axes) or any(
        bool(getattr(previous, "show_viewport", False))
        for previous in modifiers[:mirror_index]
    ):
        return signature, None
    active_axes = tuple(index for index, enabled in enumerate(axes) if enabled)
    if not active_axes:
        return signature, None

    try:
        copy_masks = tuple(
            sum(1 << axis for bit, axis in enumerate(active_axes) if subset & (1 << bit))
            for subset in range(1, 1 << len(active_axes))
        )
        configuration = {
            "axes": active_axes,
            "copy_masks": copy_masks,
            "use_merge": use_merge,
            "merge_threshold": merge_threshold,
        }
        if not include_transforms:
            return signature, configuration
        object_matrix = obj.matrix_world.copy()
        object_inverse = object_matrix.inverted()
        mirror_basis = (
            mirror_object.matrix_world.copy()
            if mirror_object is not None
            else object_matrix.copy()
        )
        mirror_basis_inverse = mirror_basis.inverted()
        axis_transforms = []
        for axis in active_axes:
            reflection = Matrix.Identity(4)
            reflection[axis][axis] = -1.0
            local_reflection = (
                object_inverse
                @ mirror_basis
                @ reflection
                @ mirror_basis_inverse
                @ object_matrix
            )
            axis_transforms.append(
                (
                    axis,
                    tuple(tuple(float(value) for value in row) for row in local_reflection),
                )
            )
        configuration["axis_transforms"] = tuple(axis_transforms)
        return signature, configuration
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return signature, None


def _topology_color_mirror_positions_equal(first, second):
    """Compare copied mirror points within transform round-off only."""
    try:
        scale = max(1.0, *(abs(float(value)) for point in (first, second) for value in point))
        tolerance = scale * 1.0e-7
        return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(first, second))
    except (TypeError, ValueError):
        return False


def _topology_color_mirror_face_positions_equal(first, second):
    """Compare the small vertex sets of two generated faces without ordering."""
    if len(first) != len(second):
        return False
    unmatched = list(second)
    for point in first:
        for index, candidate in enumerate(unmatched):
            if _topology_color_mirror_positions_equal(point, candidate):
                del unmatched[index]
                break
        else:
            return False
    return not unmatched


def _topology_color_cache_refresh_geometry(
    cache,
    obj,
    bm,
    face_states,
    coordinates,
    mirror_configuration=None,
):
    """Refresh copied draw positions for the cached colored region only."""
    try:
        matrix = obj.matrix_world
        world_coordinates = {}
        variants = {}
        local_variants = {}
        if mirror_configuration is not None:
            object_matrix = matrix.copy()
            axis_transforms = {
                axis: Matrix(matrix_values)
                for axis, matrix_values in mirror_configuration["axis_transforms"]
            }
            for copy_mask in (0, *mirror_configuration["copy_masks"]):
                variant = {}
                local_variant = {}
                for slot in cache.get("colored_vertex_slots", ()):
                    local = Vector(coordinates[slot])
                    for axis in mirror_configuration["axes"]:
                        reflected = axis_transforms[axis] @ local
                        if copy_mask & (1 << axis):
                            if (
                                mirror_configuration["use_merge"]
                                and (reflected - local).length
                                < mirror_configuration["merge_threshold"]
                            ):
                                local = (local + reflected) * 0.5
                            else:
                                local = reflected
                        elif (
                            mirror_configuration["use_merge"]
                            and (reflected - local).length
                            < mirror_configuration["merge_threshold"]
                        ):
                            local = (local + reflected) * 0.5
                    local_variant[slot] = tuple(local)
                    variant[slot] = tuple(object_matrix @ local)
                variants[copy_mask] = variant
                local_variants[copy_mask] = local_variant
            world_coordinates = variants.pop(0)
            source_local_coordinates = local_variants.pop(0)
        else:
            world_coordinates = {
                slot: tuple(matrix @ bm.verts[slot].co)
                for slot in cache.get("colored_vertex_slots", ())
            }
            source_local_coordinates = coordinates
        triangles = {index: [] for index in range(1, 7)}
        for record, face, face_vertex_slots in face_states:
            if record["hidden"]:
                continue
            if len(record["vertex_cycle"]) >= 4:
                triangle_slots = _topology_color_face_triangle_slots(
                    face,
                    face_vertex_slots,
                    source_local_coordinates
                    if mirror_configuration is not None
                    else None,
                )
                if triangle_slots is None:
                    return False
                record["triangle_slots"] = triangle_slots
            for triangle in record["triangle_slots"]:
                triangles[record["color_index"]].extend(
                    world_coordinates[slot]
                    for slot in triangle
                )

        wire = []
        for first_slot, second_slot in cache.get("wire_layout", ()):
            wire.extend(
                (
                    world_coordinates[first_slot],
                    world_coordinates[second_slot],
                )
            )

        cache["colored_vertex_snapshot"] = dict(coordinates)
        cache["triangles"] = triangles
        cache["wire"] = wire
        mirror_triangles = {index: [] for index in range(1, 7)}
        mirror_wire = []
        if mirror_configuration is not None:
            object_matrix = obj.matrix_world.copy()
            for record, _face, face_vertex_slots in face_states:
                if record["hidden"] or not record["triangle_slots"]:
                    continue
                seen_face_positions = [tuple(
                    world_coordinates[slot] for slot in face_vertex_slots
                )]
                for copy_mask, variant in variants.items():
                    face_positions = tuple(
                        variant[slot] for slot in face_vertex_slots
                    )
                    duplicate_face = any(
                        _topology_color_mirror_face_positions_equal(
                            face_positions,
                            previous,
                        )
                        for previous in seen_face_positions
                    )
                    if duplicate_face:
                        continue
                    seen_face_positions.append(face_positions)
                    reversed_winding = bool(copy_mask.bit_count() % 2)
                    copy_triangle_slots = record["triangle_slots"]
                    if len(face_vertex_slots) == 4:
                        copy_face_slots = (
                            tuple(face_vertex_slots[:1])
                            + tuple(reversed(face_vertex_slots[1:]))
                            if reversed_winding
                            else face_vertex_slots
                        )
                        copy_triangle_slots = _topology_color_face_triangle_slots(
                            _face,
                            copy_face_slots,
                            local_variants[copy_mask],
                        )
                        if copy_triangle_slots is None:
                            return False
                    for triangle in copy_triangle_slots:
                        slots = (
                            (triangle[0], triangle[2], triangle[1])
                            if reversed_winding and len(face_vertex_slots) != 4
                            else triangle
                        )
                        mirror_triangles[record["color_index"]].extend(
                            variant[slot] for slot in slots
                        )

            for first_slot, second_slot in cache.get("wire_layout", ()):
                original_edge = (
                    world_coordinates[first_slot],
                    world_coordinates[second_slot],
                )
                seen_edges = [original_edge]
                for variant in variants.values():
                    edge = (
                        variant[first_slot],
                        variant[second_slot],
                    )
                    duplicate = any(
                        (
                            _topology_color_mirror_positions_equal(edge[0], prior[0])
                            and _topology_color_mirror_positions_equal(edge[1], prior[1])
                        )
                        or (
                            _topology_color_mirror_positions_equal(edge[0], prior[1])
                            and _topology_color_mirror_positions_equal(edge[1], prior[0])
                        )
                        for prior in seen_edges
                    )
                    if duplicate:
                        continue
                    seen_edges.append(edge)
                    mirror_wire.extend(
                        (
                            variant[first_slot],
                            variant[second_slot],
                        )
                    )

        cache["mirror_triangles"] = mirror_triangles
        cache["mirror_wire"] = mirror_wire
        cache["mirror_configuration"] = mirror_configuration
        cache["gpu_batches"] = None
        cache["gpu_wire_batch"] = None
        cache["gpu_mirror_batches"] = None
        cache["gpu_mirror_wire_batch"] = None
        cache["layout_valid"] = True
        return True
    except (AttributeError, IndexError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _build_topology_color_cache(
    obj,
    overlay_offset=None,
    mirror_signature=None,
    mirror_configuration=None,
):
    """Copy only visible colored Edit BMesh faces into draw-ready buffers."""
    cache = {
        "object_pointer": 0,
        "mesh_pointer": 0,
        "matrix_key": None,
        "overlay_offset": 0.0,
        "mirror_signature": mirror_signature,
        "mirror_configuration": mirror_configuration,
        "face_count": 0,
        "vert_count": 0,
        "edge_count": 0,
        "topology_color_layer_present": False,
        "layout_valid": False,
        "colored_vertex_slots": (),
        "colored_face_layout": (),
        "wire_layout": (),
        "colored_vertex_snapshot": {},
        "triangles": {index: [] for index in range(1, 7)},
        "wire": [],
        "mirror_triangles": {index: [] for index in range(1, 7)},
        "mirror_wire": [],
        "gpu_batches": None,
        "gpu_wire_batch": None,
        "gpu_mirror_batches": None,
        "gpu_mirror_wire_batch": None,
        "gpu_shader": None,
    }
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        layer = bm.faces.layers.int.get(TOPOLOGY_COLOR_ATTRIBUTE_NAME)
        cache["topology_color_layer_present"] = layer is not None
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
        cache["edge_count"] = len(bm.edges)
        if layer is None:
            cache["layout_valid"] = True
            return cache

        vertex_slots = {
            vertex: slot
            for slot, vertex in enumerate(bm.verts)
        }
        colored_vertex_slots = set()
        colored_face_layout = []
        wire_layout = set()
        for face_slot, face in enumerate(bm.faces):
            try:
                color_index = int(face[layer])
            except (ReferenceError, RuntimeError, TypeError, ValueError):
                continue
            if color_index < 1 or color_index > 6:
                continue
            face_vertices = tuple(face.verts)
            face_vertex_slots = tuple(vertex_slots[vertex] for vertex in face_vertices)
            vertex_cycle = _topology_color_canonical_cycle(face_vertex_slots)
            hidden = bool(face.hide)
            triangle_slots = ()
            if not hidden:
                triangle_slots = _topology_color_face_triangle_slots(
                    face,
                    face_vertex_slots,
                )
                if triangle_slots is None:
                    return cache
                for edge in face.edges:
                    edge_slots = tuple(vertex_slots[vertex] for vertex in edge.verts)
                    if len(edge_slots) == 2:
                        wire_layout.add(tuple(sorted(edge_slots)))
            colored_vertex_slots.update(face_vertex_slots)
            colored_face_layout.append(
                {
                    "face_slot": face_slot,
                    "color_index": color_index,
                    "hidden": hidden,
                    "vertex_cycle": vertex_cycle,
                    "ordered_vertex_cycle": tuple(face_vertex_slots),
                    "triangle_slots": triangle_slots,
                }
            )

        cache["colored_vertex_slots"] = tuple(sorted(colored_vertex_slots))
        cache["colored_face_layout"] = tuple(colored_face_layout)
        cache["wire_layout"] = tuple(sorted(wire_layout))
        face_states = _topology_color_cache_face_states(cache, bm)
        coordinates = _topology_color_cache_current_coordinates(cache, bm)
        if face_states is None or coordinates is None:
            return cache
        _topology_color_cache_refresh_geometry(
            cache,
            obj,
            bm,
            face_states,
            coordinates,
            mirror_configuration,
        )
    except (AttributeError, IndexError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
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
    mirror_signature, mirror_configuration = _topology_color_mirror_configuration(
        obj,
        include_transforms=False,
    )
    stale = (
        cache is None
        or object_pointer in _topology_color_cache_dirty
        or cache.get("mesh_pointer") != mesh_pointer
        or cache.get("matrix_key") != matrix_key
        or cache.get("overlay_offset") != overlay_offset
        or cache.get("mirror_signature") != mirror_signature
        or not cache.get("layout_valid", False)
    )
    if not stale:
        try:
            bm = bmesh.from_edit_mesh(obj.data)
            face_states = _topology_color_cache_face_states(cache, bm)
            if face_states is None:
                stale = True
            else:
                coordinates = _topology_color_cache_current_coordinates(cache, bm)
                if coordinates is None:
                    stale = True
                elif coordinates != cache.get("colored_vertex_snapshot"):
                    stale = not _topology_color_cache_refresh_geometry(
                        cache,
                        obj,
                        bm,
                        face_states,
                        coordinates,
                        cache.get("mirror_configuration"),
                    )
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            stale = True
    if stale:
        if mirror_configuration is not None:
            mirror_signature, mirror_configuration = _topology_color_mirror_configuration(
                obj,
                include_transforms=True,
            )
        cache = _build_topology_color_cache(
            obj,
            overlay_offset,
            mirror_signature,
            mirror_configuration,
        )
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
            if _topology_color_cache:
                _clear_topology_color_draw_cache()
            return
        cache = _topology_color_cache_for(obj)
        if cache is None:
            return
        opacity = float(prefs.topology_color_opacity) if prefs else 0.35
        opacity = max(0.0, min(1.0, opacity))
        shader, uses_depth_shader = _topology_color_draw_shader()
        if shader is None:
            return
        retopology_offset = max(
            0.0,
            float(cache.get("overlay_offset", 0.0)),
        )
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
            if uses_depth_shader:
                shader.bind()
                shader.uniform_float(
                    "ModelViewMatrix",
                    gpu.matrix.get_model_view_matrix(),
                )
                shader.uniform_float(
                    "ProjectionMatrix",
                    gpu.matrix.get_projection_matrix(),
                )
                shader.uniform_float("retopology_offset", retopology_offset)
            gpu_batches = cache.get("gpu_batches")
            if cache.get("gpu_shader") is not shader:
                gpu_batches = None
                cache["gpu_wire_batch"] = None
                cache["gpu_mirror_batches"] = None
                cache["gpu_mirror_wire_batch"] = None
                cache["gpu_shader"] = shader
            if gpu_batches is None:
                gpu_batches = {
                    color_index: _topology_color_batch(
                        shader,
                        "TRIS",
                        vertices,
                        uses_depth_shader,
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

            gpu_mirror_batches = cache.get("gpu_mirror_batches")
            if gpu_mirror_batches is None:
                gpu_mirror_batches = {
                    color_index: _topology_color_batch(
                        shader,
                        "TRIS",
                        vertices,
                        uses_depth_shader,
                    )
                    for color_index, vertices in cache["mirror_triangles"].items()
                    if vertices
                }
                cache["gpu_mirror_batches"] = gpu_mirror_batches
            for color_index, vertices in cache["mirror_triangles"].items():
                if not vertices:
                    continue
                batch = gpu_mirror_batches[color_index]
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
                        uses_depth_shader,
                    )
                    cache["gpu_wire_batch"] = wire_batch
                shader.bind()
                shader.uniform_float("color", (0.01, 0.01, 0.01, 0.45))
                wire_batch.draw(shader)
            if cache["mirror_wire"]:
                mirror_wire_batch = cache.get("gpu_mirror_wire_batch")
                if mirror_wire_batch is None:
                    mirror_wire_batch = _topology_color_batch(
                        shader,
                        "LINES",
                        cache["mirror_wire"],
                        uses_depth_shader,
                    )
                    cache["gpu_mirror_wire_batch"] = mirror_wire_batch
                shader.bind()
                shader.uniform_float("color", (0.01, 0.01, 0.01, 0.45))
                mirror_wire_batch.draw(shader)
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


def _fill_preview_cancel(*args, **kwargs):
    """Lazy bridge to Smart Fill so foundation remains importable first."""
    from .smart_fill.preview import _fill_preview_cancel as cancel
    return cancel(*args, **kwargs)


def _fill_preview_request_cancel(*args, **kwargs):
    from .smart_fill.preview import _fill_preview_request_cancel as request
    return request(*args, **kwargs)


def _fill_preview_restore_manual_analysis_profiles(*args, **kwargs):
    """Lazy bridge to the Smart Fill analysis-profile owner."""
    from .smart_fill.geometry import (
        _fill_preview_restore_manual_analysis_profiles as restore,
    )
    return restore(*args, **kwargs)


def _tube_preview_cancel(*args, **kwargs):
    """Lazy bridge to Tube Shape teardown."""
    from .tube_shape import _tube_preview_cancel as cancel
    return cancel(*args, **kwargs)


def _tube_preview_request_cancel(*args, **kwargs):
    from .tube_shape import _tube_preview_request_cancel as request
    return request(*args, **kwargs)


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
    if not _runtime.is_registered:
        _runtime.deferred_orphan_cleanup = None
        return None
    try:
        _cleanup_orphan_face_set_proxies()
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        # A short retry handles the brief interval during workspace startup.
        return 0.25
    _runtime.deferred_orphan_cleanup = None
    return None


def _schedule_orphan_cleanup():
    callback = _runtime.deferred_orphan_cleanup
    try:
        if callback is not None and bpy.app.timers.is_registered(callback):
            return
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    _runtime.deferred_orphan_cleanup = _deferred_orphan_cleanup
    try:
        bpy.app.timers.register(_deferred_orphan_cleanup, first_interval=0.1)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        _runtime.deferred_orphan_cleanup = None


def _cancel_orphan_cleanup():
    """Unregister the exact orphan callback, including one from an old module."""
    callback = _runtime.deferred_orphan_cleanup
    try:
        if callback is not None and bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    _runtime.deferred_orphan_cleanup = None


def _deferred_undo_orphan_cleanup():
    """Reconcile tagged FSMFO remnants after Blender finishes an Undo step.

    ``undo_post`` can run while Edit BMesh data is still being rebuilt.  Keep
    the handler lightweight and perform the ownership-tagged cleanup from a
    short timer instead.  Active sessions remain protected by the existing
    session/generation checks in the orphan cleanup functions.
    """
    deferred_started = time.perf_counter()
    undo_post_to_callback_ms = None
    if _runtime.last_undo_post_perf is not None:
        undo_post_to_callback_ms = round(
            (deferred_started - _runtime.last_undo_post_perf) * 1000.0,
            3,
        )
    _runtime.last_undo_post_perf = None
    _runtime.undo_orphan_cleanup_pending = False
    _retopo_debug_emit(
        "deferred_start",
        {
            "retries": int(_runtime.undo_orphan_cleanup_retries),
            "undo_post_to_callback_ms": undo_post_to_callback_ms,
            "tombstone_count": len(_retopo_undo_tombstones),
            "active_state_count": len(_active_states),
        },
    )
    if not _runtime.is_registered:
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
        _runtime.deferred_undo_orphan_cleanup = None
        return None
    try:
        _cleanup_orphan_face_set_proxies(reconcile_undo_tombstones=True)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        # BMesh may still be temporarily unavailable after undo_post.
        if _runtime.undo_orphan_cleanup_retries < 3:
            _runtime.undo_orphan_cleanup_retries += 1
            _runtime.undo_orphan_cleanup_pending = True
            _retopo_debug_emit(
                "deferred_end",
                {
                    "decision": "retry",
                    "reason": "cleanup_exception",
                    "retries": int(_runtime.undo_orphan_cleanup_retries),
                    "tombstone_count": len(_retopo_undo_tombstones),
                    "duration_ms": round(
                        (time.perf_counter() - deferred_started) * 1000.0,
                        3,
                    ),
                },
            )
            return 0.1
        _runtime.retopo_undo_retry_pending = False
        for tombstone in _retopo_undo_tombstones.values():
            tombstone["retry_pending"] = False
        _runtime.undo_orphan_cleanup_retries = 0
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
        _runtime.deferred_undo_orphan_cleanup = None
        return None
    if _has_reclaimable_orphan_artifacts() and _runtime.undo_orphan_cleanup_retries < 3:
        _runtime.undo_orphan_cleanup_retries += 1
        _runtime.undo_orphan_cleanup_pending = True
        _retopo_debug_emit(
            "deferred_end",
            {
                "decision": "retry",
                "reason": "reclaimable_transient_artifact",
                "retries": int(_runtime.undo_orphan_cleanup_retries),
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
        _runtime.retopo_undo_retry_pending = False
        for tombstone in _retopo_undo_tombstones.values():
            tombstone["retry_pending"] = False
    _runtime.undo_orphan_cleanup_retries = 0
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
    _runtime.deferred_undo_orphan_cleanup = None
    return None


def _schedule_undo_orphan_cleanup():
    """Schedule at most one post-Undo reconciliation for this module."""
    if not _runtime.is_registered or _runtime.undo_orphan_cleanup_pending:
        return
    callback = _runtime.deferred_undo_orphan_cleanup
    try:
        if callback is not None and bpy.app.timers.is_registered(callback):
            return
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    _runtime.deferred_undo_orphan_cleanup = _deferred_undo_orphan_cleanup
    try:
        bpy.app.timers.register(_deferred_undo_orphan_cleanup, first_interval=0.1)
        _runtime.undo_orphan_cleanup_pending = True
        _runtime.undo_orphan_cleanup_retries = 0
        _runtime.retopo_undo_retry_pending = False
    except (AttributeError, RuntimeError, TypeError, ValueError):
        _runtime.deferred_undo_orphan_cleanup = None


def _cancel_undo_orphan_cleanup():
    """Remove a queued callback during unregister/reload lifecycle cleanup."""
    callback = _runtime.deferred_undo_orphan_cleanup
    try:
        if callback is not None and bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    _runtime.undo_orphan_cleanup_pending = False
    _runtime.deferred_undo_orphan_cleanup = None
    _runtime.undo_orphan_cleanup_retries = 0
    _runtime.last_undo_post_perf = None
    _runtime.retopo_undo_retry_pending = False
    for tombstone in _retopo_undo_tombstones.values():
        tombstone["retry_pending"] = False


@persistent
def _on_undo_post(_dummy):
    """Queue ownership-safe cleanup after native Undo has rebuilt the scene."""
    _fill_preview_request_cancel(reason="undo")
    _tube_preview_request_cancel(reason="undo")
    # Undo can restore coordinates/topology/visibility and may not expose a
    # target Mesh update in the same callback turn.  Never let a pre-Undo
    # surface graph survive on that assumption.
    _fill_preview_adjacency_cache.clear()
    _fill_preview_cursor_cache.clear()
    if not _runtime.undo_orphan_cleanup_pending:
        _runtime.last_undo_post_perf = time.perf_counter()
    _retopo_debug_emit(
        "undo_post",
        {
            "tombstone_count": len(_retopo_undo_tombstones),
            "active_state_count": len(_active_states),
            "cleanup_timer_pending": bool(_runtime.undo_orphan_cleanup_pending),
        },
    )
    _schedule_undo_orphan_cleanup()


@persistent
def _on_fill_preview_redo_post(_dummy):
    """Invalidate preview graphs after native Redo restores mesh state."""
    _fill_preview_request_cancel(reason="redo")
    _fill_preview_adjacency_cache.clear()
    _fill_preview_cursor_cache.clear()


@persistent
def _on_load_pre(_dummy):
    """Clear viewport-bound state before Blender replaces the current file."""
    _fill_preview_request_cancel(reason="load")
    _fill_preview_restore_manual_analysis_profiles()
    _tube_preview_request_cancel(reason="load")
    _finish_all_states()
    _cleanup_orphan_face_set_proxies()
    _retopo_undo_tombstones.clear()
    _local_face_set_adjacency_cache.clear()
    _fill_preview_adjacency_cache.clear()
    _fill_preview_cursor_cache.clear()


@persistent
def _on_load_post(_dummy):
    """Recover remnants loaded from a file saved during Face Set MFO."""
    _fill_preview_request_cancel(reason="load-post")
    _fill_preview_restore_manual_analysis_profiles()
    _tube_preview_request_cancel(reason="load-post")
    _cleanup_orphan_face_set_proxies()
    # File loading can lose an app-timer callback while a retired modal is in
    # its grace window. Keep the durable pending record and re-arm safely.
    try:
        from . import lifecycle as _lifecycle
        _lifecycle.modal_quiescence_ensure()
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        pass


def _clear_topology_color_draw_cache():
    """Drop copied coordinates at every history/file lifetime boundary."""
    _topology_color_cache.clear()
    _topology_color_cache_dirty.clear()
    _topology_color_tag_redraw_all()


def _topology_color_id_chain(data):
    """Return an evaluated ID and its original without retaining either."""
    chain = []
    seen = set()
    current = data
    for _index in range(4):
        try:
            pointer = int(current.as_pointer())
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            break
        if pointer in seen:
            break
        seen.add(pointer)
        chain.append(current)
        try:
            original = getattr(current, "original", None)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            break
        if original is None or original is current:
            break
        current = original
    return chain


def _topology_color_update_requires_forced_invalidation(update):
    """Use region snapshots for geometry/transform updates when available."""
    try:
        return not (
            bool(update.is_updated_geometry)
            or bool(update.is_updated_transform)
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return True


@persistent
def _on_topology_color_depsgraph_update(_scene, depsgraph):
    """Resolve original IDs, redraw matches, and invalidate unknown updates."""
    try:
        matched_cache = False
        for update in depsgraph.updates:
            id_chain = _topology_color_id_chain(update.id)
            force_cache_invalidation = (
                _topology_color_update_requires_forced_invalidation(update)
            )
            object_pointers = set()
            mesh_pointers = set()
            for data in id_chain:
                try:
                    data_pointer = int(data.as_pointer())
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    continue
                if isinstance(data, bpy.types.Mesh):
                    mesh_pointers.add(data_pointer)
                elif isinstance(data, bpy.types.Object) and data.type == "MESH":
                    object_pointers.add(data_pointer)
                    mesh_chain = _topology_color_id_chain(data.data)
                    for mesh in mesh_chain:
                        try:
                            mesh_pointers.add(int(mesh.as_pointer()))
                        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                            continue

            if not object_pointers and not mesh_pointers:
                continue
            for object_pointer, cache in tuple(_topology_color_cache.items()):
                if (
                    object_pointer in object_pointers
                    or cache.get("mesh_pointer") in mesh_pointers
                ):
                    if force_cache_invalidation:
                        _topology_color_cache_dirty.add(object_pointer)
                    matched_cache = True
        if matched_cache:
            _topology_color_tag_redraw_all()
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


def _tool_event_coordinate(context, event):
    """Resolve a WorkSpaceTool click without ever substituting the center."""
    # This helper is intentionally shared with the Sculpt click operators.  It
    # prefers absolute mouse coordinates and rejects sibling regions, which
    # prevents a toolbar/sidebar click from being interpreted as a viewport
    # target.
    return _sculpt_cursor_region_coordinate(context, event)


def _start_normal_tool_session(context, coordinate):
    """Start or stop normal MFO using one explicit region coordinate."""
    try:
        area_key = context.area.as_pointer()
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False
    existing = _active_states.get(area_key)
    if existing is not None and existing.active:
        if not isinstance(existing, _FaceSetOrbitState):
            _deactivate_state(existing)
            return True
        return False
    if area_key in _session_cleanup_areas:
        return False

    _cleanup_orphan_face_set_proxies()
    hit = _find_mesh_hit_at_coordinate(context, coordinate)
    if hit is None:
        return False
    prefs = _addon_preferences()
    activation_key = prefs.activation_key if prefs else "RIGHT_SHIFT"
    state = _TempOrbitState(context, hit[0], hit[1], activation_key)
    return bool(_start_session(context, state, face_set=False))


def _activate_face_set_session(context, coordinate=None):
    """Start Face Set MFO, optionally using an explicit region coordinate.

    The click tool deliberately calls the public Activation operator below so
    Blender creates the same single ``UNDO`` boundary as the legacy
    double-tap/Ctrl path.  This helper owns only the coordinate-aware ON path;
    OFF remains the existing direct runtime teardown in ``execute``.
    """
    try:
        area_key = context.area.as_pointer()
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False
    if area_key in _session_cleanup_areas:
        return False

    hit = (
        _raycast_reference_face_set(context)
        if coordinate is None
        else _raycast_reference_face_set_at_coordinate(context, coordinate)
    )
    if hit is None:
        return False
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
    return bool(_start_session(context, state, face_set=True))
