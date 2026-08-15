"""Mesh Focus Orbit.

Double-tap the configured key in a 3D Viewport to make the first visible mesh
surface under the viewport center the temporary orbit target.  The built-in
viewport navigation, including the Navigation Gizmo, remains responsible for
rotating the view.
"""

bl_info = {
    "name": "Mesh Focus Orbit",
    "author": "OpenAI",
    "version": (2, 1, 19),
    "blender": (5, 2, 0),
    "location": "3D View",
    "description": "Temporary mesh-centered orbit and one-click Smart Face Set Fill",
    "category": "3D View",
}

import bpy
import bmesh
import blf
import gpu
import heapq
import math
import statistics
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
RECOVER_FACE_SET_STATE_OPERATOR_ID = "view3d.mesh_focus_orbit_recover_face_set_state"
LOCAL_FACE_SET_GROW_OPERATOR_ID = "view3d.mesh_focus_local_face_set_grow_v2"
LOCAL_FACE_SET_GROW_KEY = "E"

_addon_keymaps = []
_active_states = {}
_modal_operator_areas = {}
_last_tap_times = {}
_local_face_set_adjacency_cache = {}
_is_registered = False
_polyquilt_qsnap_class = None
_polyquilt_qsnap_original_snap_objects = None
_polyquilt_qsnap_filter_installed = False
_retopoflow_hidden_vertex_filter_class = None
_retopoflow_hidden_vertex_filter_original_update = None
_retopoflow_hidden_vertex_filter_installed = False
_retopo_isolation_serial = 0

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

# This is a temporary runtime compatibility hook.  It never edits the
# RetopoFlow installation; the marker lets a later reload or cleanup restore
# only a wrapper created by Mesh Focus Orbit.
_RETOPOFLOW_HIDDEN_VERTEX_FILTER_TAG = (
    "mesh_focus_orbit.retopoflow_hidden_vertex_filter"
)
_RETOPOFLOW_HIDDEN_VERTEX_FILTER_ORIGINAL = (
    "mesh_focus_orbit.retopoflow_original_nearest_bmvert_update"
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
_RETOPO_ISOLATION_VERT_HIDE_LAYER = "mesh_focus_orbit.retopo_vert_hide_layer"
_RETOPO_ISOLATION_VERT_SELECT_LAYER = "mesh_focus_orbit.retopo_vert_select_layer"
_RETOPO_ISOLATION_EDGE_MARKER_LAYER = "mesh_focus_orbit.retopo_edge_marker_layer"
_RETOPO_ISOLATION_EDGE_HIDE_LAYER = "mesh_focus_orbit.retopo_edge_hide_layer"
_RETOPO_ISOLATION_EDGE_SELECT_LAYER = "mesh_focus_orbit.retopo_edge_select_layer"


SMART_FACE_SET_FILL_NORMAL_SMOOTH_RADIUS_FACTOR = 8.0
SMART_FACE_SET_FILL_NORMAL_VARIATION_WEIGHT = 0.25
SMART_FACE_SET_FILL_NORMAL_ANGLE_LIMIT = math.radians(75.0)
SMART_FACE_SET_FILL_NORMAL_RAW_ANGLE_LIMIT = math.radians(85.0)
SMART_FACE_SET_FILL_RAW_EDGE_WEIGHT = 0.65
SMART_FACE_SET_FILL_RAW_EDGE_ANGLE_LIMIT = math.radians(30.0)
SMART_FACE_SET_FILL_CONCAVITY_PENALTY = 2.5
SMART_FACE_SET_FILL_ACCEPTANCE_THRESHOLD = 0.60
SMART_FACE_SET_FILL_MAX_FACES = 100000
SMART_FACE_SET_FILL_MAX_GEODESIC_SCALE = 350.0
SMART_FACE_SET_FILL_MAX_GEODESIC_FACTOR = 0.45
SMART_FACE_SET_FILL_MIN_GEODESIC_FACTOR = 0.03
SMART_FACE_SET_FILL_NORMAL_SAMPLE_LIMIT = 8

# Strict Smart Face Set Fill parameters. The normal E shortcut keeps the
# values above; Shift+E uses these tighter local limits to stop at valleys
# that normal mode is allowed to cross.
SMART_FACE_SET_FILL_STRICT_NORMAL_SMOOTH_RADIUS_FACTOR = 4.0
SMART_FACE_SET_FILL_STRICT_NORMAL_VARIATION_WEIGHT = 0.35
SMART_FACE_SET_FILL_STRICT_NORMAL_ANGLE_LIMIT = math.radians(55.0)
SMART_FACE_SET_FILL_STRICT_NORMAL_RAW_ANGLE_LIMIT = math.radians(70.0)
SMART_FACE_SET_FILL_STRICT_RAW_EDGE_WEIGHT = 0.90
SMART_FACE_SET_FILL_STRICT_RAW_EDGE_ANGLE_LIMIT = math.radians(20.0)
SMART_FACE_SET_FILL_STRICT_CONCAVITY_PENALTY = 4.0
SMART_FACE_SET_FILL_STRICT_ACCEPTANCE_THRESHOLD = 0.40
SMART_FACE_SET_FILL_STRICT_MAX_GEODESIC_SCALE = 160.0
SMART_FACE_SET_FILL_STRICT_MAX_GEODESIC_FACTOR = 0.12
SMART_FACE_SET_FILL_STRICT_MIN_GEODESIC_FACTOR = 0.01


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


class _FaceSetOrbitState(_TempOrbitState):
    """Temporary orbit state with a reversible Face Set display proxy."""

    __slots__ = (
        "reference_object",
        "face_set_id",
        "proxy_object",
        "reference_was_hidden",
        "retopo_isolation",
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
        self.retopo_isolation = None
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
    """Return all BMFace edge-connected components in the Edit Mesh."""
    bm.faces.ensure_lookup_table()
    visited = set()
    components = []
    for face in bm.faces:
        if not _retopo_face_is_valid(face) or face in visited:
            continue
        component = {face}
        visited.add(face)
        queue = deque([face])
        while queue:
            current = queue.popleft()
            for edge in current.edges:
                for linked_face in edge.link_faces:
                    if (
                        _retopo_face_is_valid(linked_face)
                        and linked_face not in visited
                    ):
                        visited.add(linked_face)
                        component.add(linked_face)
                        queue.append(linked_face)
        components.append(component)
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
    vertices = []
    seen_vertices = set()
    for face in component:
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
        except (ReferenceError, RuntimeError, TypeError, ValueError):
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
        "face_count": len(component),
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
        components = _retopo_connected_components(bm)
        proxy_bvh = _retopo_proxy_bvh(proxy)
        if not components or proxy_bvh is None:
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
        return {
            "object": retopo_object,
            "component": list(chosen["component"]),
            # This is only a hand-off between two immediate BMesh reads during
            # activation.  It is not used for restoration after topology
            # changes; direct face refs and the temporary marker layer are
            # used for that purpose.
            "component_indices": [
                int(face.index)
                for face in chosen["component"]
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


def _retopo_layer_names(session_id):
    prefix = f"__MFO_RETOPO_{session_id:08x}"
    return {
        "marker": f"{prefix}_MARKER",
        "hide": f"{prefix}_HIDE",
        "select": f"{prefix}_SELECT",
        "target": f"{prefix}_TARGET",
        "vert_marker": f"{prefix}_VERT_MARKER",
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
    component = chosen["component"]
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
        if not component:
            return None
        session_id = _retopo_next_session_id()
        layer_names = _retopo_layer_names(session_id)
        marker_layer = bm.faces.layers.int.new(layer_names["marker"])
        hide_layer = bm.faces.layers.int.new(layer_names["hide"])
        select_layer = bm.faces.layers.int.new(layer_names["select"])
        target_layer = bm.faces.layers.int.new(layer_names["target"])
        vert_marker_layer = bm.verts.layers.int.new(layer_names["vert_marker"])
        vert_hide_layer = bm.verts.layers.int.new(layer_names["vert_hide"])
        vert_select_layer = bm.verts.layers.int.new(layer_names["vert_select"])
        edge_marker_layer = bm.edges.layers.int.new(layer_names["edge_marker"])
        edge_hide_layer = bm.edges.layers.int.new(layer_names["edge_hide"])
        edge_select_layer = bm.edges.layers.int.new(layer_names["edge_select"])

        target_vertex_indices = {
            int(vertex.index)
            for face in bm.faces
            if int(face.index) in component_index_set
            for vertex in face.verts
        }
        target_edge_indices = {
            int(edge.index)
            for face in bm.faces
            if int(face.index) in component_index_set
            for edge in face.edges
        }

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

        original_vert_records = []
        initial_verts = []
        initial_vert_indices = []
        initial_vert_hide_values = []
        for vertex in bm.verts:
            initial_verts.append(vertex)
            try:
                initial_vert_indices.append(int(vertex.index))
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                initial_vert_indices.append(-1)
            initial_vert_hide_values.append(int(vertex.hide))
            original_vert_records.append(
                (vertex, bool(vertex.hide), bool(vertex.select))
            )
            vertex[vert_marker_layer] = session_id
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

        bmesh.update_edit_mesh(
            retopo_object.data,
            loop_triangles=False,
            destructive=False,
        )
        _retopo_set_object_markers(retopo_object, session_id, layer_names)
        return {
            "object": retopo_object,
            "object_name": retopo_object.name,
            "mesh": retopo_object.data,
            "session_id": session_id,
            "layer_names": layer_names,
            "initial_faces": initial_faces,
            "original_records": original_records,
            "component_faces": list(component),
            "initial_verts": initial_verts,
            "initial_vert_indices": initial_vert_indices,
            "initial_vert_hide_values": initial_vert_hide_values,
            "original_vert_records": original_vert_records,
            "initial_edges": initial_edges,
            "original_edge_records": original_edge_records,
            "active_face": active_face,
        }
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


def _retopo_restore_info(info):
    """Restore one active or orphaned Retopo isolation without mode changes."""
    if not info:
        return False
    retopo_object = info.get("object")
    if retopo_object is None:
        retopo_object = bpy.data.objects.get(info.get("object_name", ""))
    if retopo_object is None or retopo_object.type != "MESH":
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

        for layer in (
            marker_layer,
            hide_layer,
            select_layer,
            _target_layer,
        ):
            if layer is not None:
                try:
                    bm.faces.layers.int.remove(layer)
                except (AttributeError, ReferenceError, RuntimeError, TypeError):
                    pass

        for layer in (
            vert_marker_layer,
            vert_hide_layer,
            vert_select_layer,
        ):
            if layer is not None:
                try:
                    bm.verts.layers.int.remove(layer)
                except (AttributeError, ReferenceError, RuntimeError, TypeError):
                    pass

        for layer in (
            edge_marker_layer,
            edge_hide_layer,
            edge_select_layer,
        ):
            if layer is not None:
                try:
                    bm.edges.layers.int.remove(layer)
                except (AttributeError, ReferenceError, RuntimeError, TypeError):
                    pass

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


def _retopo_restore_isolation(state):
    if not isinstance(state, _FaceSetOrbitState):
        return
    info = state.retopo_isolation
    if not info:
        return
    # Restore only elements marked at activation.  New topology is deliberately
    # left untouched so RetopoFlow edits survive FSMFO teardown.
    _retopo_restore_info(info)
    state.retopo_isolation = None


def _cleanup_orphan_retopo_isolations():
    """Restore tagged Retopo hide state when FSMFO Python state is gone."""
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
            if _retopo_restore_info(info):
                restored_names.append(obj.name)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
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


def _retopoflow_hidden_vertex_filter_enabled():
    prefs = _addon_preferences()
    return bool(
        getattr(prefs, "retopoflow_hidden_vertex_filter", False)
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
            if info.get("object") is edit_object:
                return info
            if info.get("object_name") == edit_object.name:
                return info
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            continue
    return None


def _retopoflow_mfo_vertex_filter(context):
    """Build a filter for only pre-existing vertices hidden by this session.

    A plain ``vertex.hide`` check is too broad: it also classifies any vertex
    created after FSMFO started.  The isolation marker is initialized on every
    vertex that existed at activation.  Directly-created vertices use the
    layer default, while split/copy operations may inherit custom data; the
    saved element identity/index guard keeps those new vertices outside this
    filter as well.
    """
    info = _retopoflow_filter_info(context)
    if info is None:
        return None

    try:
        edit_object = context.edit_object
        layer_names = info.get("layer_names") or {}
        marker_name = layer_names.get("vert_marker")
        hide_name = layer_names.get("vert_hide")
        if not marker_name or not hide_name:
            return None
        bm = bmesh.from_edit_mesh(edit_object.data)
        marker_layer = bm.verts.layers.int.get(marker_name)
        hide_layer = bm.verts.layers.int.get(hide_name)
        session_id = int(info["session_id"])
        initial_vertex_pointers = set()
        for initial_vertex in info.get("initial_verts", ()):
            try:
                if initial_vertex.is_valid:
                    pointer = initial_vertex.as_pointer()
                    if pointer:
                        initial_vertex_pointers.add(pointer)
            except (AttributeError, ReferenceError, RuntimeError, TypeError):
                continue
        initial_vertex_indices = {
            int(index)
            for index in info.get("initial_vert_indices", ())
            if isinstance(index, int) and index >= 0
        }
        try:
            bm.verts.index_update()
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass

        initial_vertex_hide_by_index = {
            int(index): int(hide_value)
            for index, hide_value in zip(
                info.get("initial_vert_indices", ()),
                info.get("initial_vert_hide_values", ()),
            )
            if isinstance(index, int) and index >= 0
        }

        if marker_layer is None or hide_layer is None:
            # RetopoFlow can rebuild the Edit BMesh and discard MFO's
            # temporary layers.  Keep a narrow identity-based fallback so
            # hidden pre-existing vertices remain protected without reverting
            # to the unsafe broad ``vertex.hide`` test.
            if not initial_vertex_hide_by_index:
                return None

            def is_allowed_without_layers(vertex):
                try:
                    index = int(vertex.index)
                    if index not in initial_vertex_hide_by_index:
                        return True
                    return not (
                        initial_vertex_hide_by_index[index] == 0
                        and bool(vertex.hide)
                    )
                except (
                    AttributeError,
                    ReferenceError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    return True

            return is_allowed_without_layers
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None

    def is_allowed(vertex):
        try:
            if initial_vertex_pointers:
                try:
                    if vertex.as_pointer() not in initial_vertex_pointers:
                        return True
                except (AttributeError, ReferenceError, RuntimeError, TypeError):
                    if initial_vertex_indices:
                        if int(vertex.index) not in initial_vertex_indices:
                            return True
                    else:
                        return True
            elif initial_vertex_indices:
                if int(vertex.index) not in initial_vertex_indices:
                    return True
            else:
                # No reliable identity data means this vertex cannot be
                # safely classified as an FSMFO pre-existing vertex.
                return True
            return not (
                int(vertex[marker_layer]) == session_id
                and int(vertex[hide_layer]) == 0
                and bool(vertex.hide)
            )
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            # A stale/invalid element must not make the wrapper fail.  The
            # original RetopoFlow routine still owns final validity checks.
            return True

    return is_allowed


def _get_retopoflow_nearest_bmvert_class():
    """Return RetopoFlow's nearest-vertex helper when RetopoFlow is present."""
    try:
        from bl_ext.superhivemarket_com.retopoflow.retopoflow.common.bmesh import (
            NearestBMVert,
        )

        return NearestBMVert
    except (ImportError, AttributeError, RuntimeError, TypeError):
        return None


def _restore_retopoflow_hidden_vertex_filter():
    """Restore only an MFO-owned RetopoFlow update wrapper, if present."""
    global _retopoflow_hidden_vertex_filter_class
    global _retopoflow_hidden_vertex_filter_original_update
    global _retopoflow_hidden_vertex_filter_installed

    classes = []
    if _retopoflow_hidden_vertex_filter_class is not None:
        classes.append(_retopoflow_hidden_vertex_filter_class)
    current_class = _get_retopoflow_nearest_bmvert_class()
    if current_class is not None and current_class not in classes:
        classes.append(current_class)

    for nearest_class in classes:
        try:
            current_update = nearest_class.update
            if not getattr(
                current_update,
                _RETOPOFLOW_HIDDEN_VERTEX_FILTER_TAG,
                False,
            ):
                continue
            original_update = getattr(
                current_update,
                _RETOPOFLOW_HIDDEN_VERTEX_FILTER_ORIGINAL,
                None,
            )
            if original_update is not None:
                nearest_class.update = original_update
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass

    _retopoflow_hidden_vertex_filter_class = None
    _retopoflow_hidden_vertex_filter_original_update = None
    _retopoflow_hidden_vertex_filter_installed = False


def _install_retopoflow_hidden_vertex_filter():
    """Temporarily exclude hidden BMesh vertices from RetopoFlow nearest snaps."""
    global _retopoflow_hidden_vertex_filter_class
    global _retopoflow_hidden_vertex_filter_original_update
    global _retopoflow_hidden_vertex_filter_installed

    if _retopoflow_hidden_vertex_filter_installed:
        current_class = _get_retopoflow_nearest_bmvert_class()
        try:
            if (
                current_class is _retopoflow_hidden_vertex_filter_class
                and current_class is not None
                and getattr(
                    current_class.update,
                    _RETOPOFLOW_HIDDEN_VERTEX_FILTER_TAG,
                    False,
                )
            ):
                return True
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass
        _restore_retopoflow_hidden_vertex_filter()

    nearest_class = _get_retopoflow_nearest_bmvert_class()
    if nearest_class is None:
        return False

    try:
        original_update = nearest_class.update
        if getattr(
            original_update,
            _RETOPOFLOW_HIDDEN_VERTEX_FILTER_TAG,
            False,
        ):
            stale_original = getattr(
                original_update,
                _RETOPOFLOW_HIDDEN_VERTEX_FILTER_ORIGINAL,
                None,
            )
            if stale_original is None:
                return False
            # A module reload can leave the previous MFO wrapper on the class.
            # Remove that stale wrapper before installing the current one.
            nearest_class.update = stale_original
            original_update = stale_original

        try:
            import inspect

            if "filter_fn" not in inspect.signature(original_update).parameters:
                return False
        except (ImportError, TypeError, ValueError):
            return False

        update_signature = inspect.signature(original_update)

        def filtered_update(self, context, co, *args, **kwargs):
            mfo_filter = _retopoflow_mfo_vertex_filter(context)
            if mfo_filter is None:
                return original_update(self, context, co, *args, **kwargs)

            try:
                bound = update_signature.bind(
                    self,
                    context,
                    co,
                    *args,
                    **kwargs,
                )
            except TypeError:
                # Preserve RetopoFlow's original argument validation instead
                # of changing the error behavior of an invalid call.
                return original_update(self, context, co, *args, **kwargs)

            caller_filter = bound.arguments.get("filter_fn")
            if caller_filter is None:
                combined_filter = mfo_filter
            else:
                combined_filter = lambda vertex: (
                    bool(caller_filter(vertex)) and bool(mfo_filter(vertex))
                )
            bound.arguments["filter_fn"] = combined_filter

            # BoundArguments reconstructs positional and keyword arguments in
            # the form required by the actual signature.  This avoids adding
            # filter_fn to kwargs when it was supplied positionally.
            return original_update(*bound.args, **bound.kwargs)

        setattr(filtered_update, _RETOPOFLOW_HIDDEN_VERTEX_FILTER_TAG, True)
        setattr(
            filtered_update,
            _RETOPOFLOW_HIDDEN_VERTEX_FILTER_ORIGINAL,
            original_update,
        )
        nearest_class.update = filtered_update
        _retopoflow_hidden_vertex_filter_class = nearest_class
        _retopoflow_hidden_vertex_filter_original_update = original_update
        _retopoflow_hidden_vertex_filter_installed = True
        return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _restore_retopoflow_hidden_vertex_filter()
        return False


def _release_retopoflow_hidden_vertex_filter():
    """Release the temporary hook after the last FSMFO state finishes."""
    if not _has_active_face_set_state():
        _restore_retopoflow_hidden_vertex_filter()


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
        state.reference_object.name
        for state in _active_states.values()
        if isinstance(state, _FaceSetOrbitState)
        and state.active
        and state.reference_object is not None
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
    state.proxy_object = None
    _remove_face_set_proxy_object(proxy)
    _restore_reference_if_unused(state.reference_object, state.reference_was_hidden)
    _refresh_polyquilt_qsnap(state=state)
    _tag_redraw(state.area)


def _build_face_set_proxy(state):
    """Create a temporary mesh containing only the hit Face Set."""
    if not isinstance(state, _FaceSetOrbitState):
        return False

    reference = state.reference_object
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

        # Preserve the source material slots and smooth shading so the proxy
        # remains visually close to the Reference Object.
        for material in source_mesh.materials:
            if material is not None:
                proxy_mesh.materials.append(material)
        for proxy_polygon, source_polygon in zip(
            proxy_mesh.polygons, target_polygons
        ):
            proxy_polygon.use_smooth = bool(source_polygon.use_smooth)
            if 0 <= source_polygon.material_index < len(proxy_mesh.materials):
                proxy_polygon.material_index = source_polygon.material_index

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
        _remove_face_set_proxy_object(proxy)
        _restore_reference_if_unused(reference, state.reference_was_hidden)
        return False


def _draw_debug_point(state):
    if not state.active:
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
    if not state or not state.active:
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
    except (AttributeError, RuntimeError, TypeError):
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
    except (AttributeError, RuntimeError, TypeError):
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
    area_key = state.area.as_pointer()
    _finish_state(state)
    _active_states.pop(area_key, None)
    for operator_key, mapped_area_key in list(_modal_operator_areas.items()):
        if mapped_area_key == area_key:
            _modal_operator_areas.pop(operator_key, None)


def _finish_state(state):
    """End the modal watcher and restore the saved viewport transform."""
    if not state or not state.active:
        return
    _retopo_restore_isolation(state)
    _remove_face_set_proxy(state)
    _restore_original_view(state)
    state.active = False
    _release_retopoflow_hidden_vertex_filter()
    _release_polyquilt_qsnap_filter(state=state)
    _stop_indicator_draw(state)
    _stop_debug_draw(state)
    _tag_redraw(state.area)


def _finish_all_states():
    for key, state in list(_active_states.items()):
        _finish_state(state)
        _active_states.pop(key, None)
    _modal_operator_areas.clear()
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

    for operator_key, area_key in list(_modal_operator_areas.items()):
        if area_key in face_set_area_keys:
            _modal_operator_areas.pop(operator_key, None)


def _cleanup_orphan_face_set_proxies():
    """Remove Face Set MFO scene remnants whose modal state no longer exists.

    Blender's Undo and file/window lifecycle can invalidate Python operator
    instances without running their ``__del__`` method.  This function is
    deliberately limited to tagged/generated proxy objects and the Reference
    visibility marker owned by those proxies.  It never changes mesh data,
    selection, active object, or mode.
    """
    _cleanup_orphan_retopo_isolations()

    active_proxy_pointers = set()
    for state in list(_active_states.values()):
        if not isinstance(state, _FaceSetOrbitState) or not state.active:
            continue
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
    _release_retopoflow_hidden_vertex_filter()

    if removed_proxy_names or removed_proxy_meshes:
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            if screen:
                for area in screen.areas:
                    _tag_redraw(area)

    return removed_proxy_names, removed_proxy_meshes


def _is_live_face_set_proxy(proxy):
    """Return whether a state still points at a live generated proxy."""
    if not _is_face_set_proxy_object(proxy):
        return False
    try:
        return bpy.data.objects.get(proxy.name) is proxy
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False


def _resync_active_face_set_states_after_undo():
    """Keep Face Set MFO active while repairing an undo-invalidated proxy.

    Undo is a normal editing operation while MFO is active.  Blender can still
    invalidate data-API-created proxy objects as part of the undo transaction,
    so repair only the missing display proxy here.  The viewport transform,
    active object, mode, and selection are intentionally left untouched.
    """
    active_face_set_states = [
        state
        for state in _active_states.values()
        if isinstance(state, _FaceSetOrbitState) and state.active
    ]
    if not active_face_set_states:
        return

    for state in active_face_set_states:
        try:
            reference_name = state.reference_object.name
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            continue

        reference = bpy.data.objects.get(reference_name)
        if reference is None or reference.type != "MESH":
            continue
        # Undo may replace the datablock wrapper while keeping the object name.
        # Keep the active mode state attached to the current Reference object.
        state.reference_object = reference

        if _is_live_face_set_proxy(state.proxy_object):
            _set_object_hidden(reference, True)
        else:
            # The normal orphan cleanup above has already removed any stale
            # tagged proxy and restored the saved Reference visibility.  Rebuild
            # only the proxy belonging to this still-active FSMFO state.
            state.proxy_object = None
            _build_face_set_proxy(state)

    _install_polyquilt_qsnap_filter()
    if _retopoflow_hidden_vertex_filter_enabled():
        _install_retopoflow_hidden_vertex_filter()
    for state in active_face_set_states:
        if state.active:
            _refresh_polyquilt_qsnap(state=state)
            _tag_redraw(state.area)


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


@persistent
def _on_load_pre(_dummy):
    """Clear viewport-bound state before Blender replaces the current file."""
    _finish_all_states()
    _cleanup_orphan_face_set_proxies()
    _local_face_set_adjacency_cache.clear()


@persistent
def _on_load_post(_dummy):
    """Recover remnants loaded from a file saved during Face Set MFO."""
    _cleanup_orphan_face_set_proxies()


@persistent
def _on_undo_pre(_dummy):
    """Leave both temporary modes active while Blender performs Undo.

    MFO is used while editing.  Undo therefore must undo the user's edit, not
    terminate the orbit mode or restore the viewport from before activation.
    The post handler repairs only a proxy that Blender actually invalidated.
    """
    return None


@persistent
def _on_undo_post(_dummy):
    """Clean orphaned remnants without ending an active temporary mode."""
    _cleanup_orphan_face_set_proxies()
    _resync_active_face_set_states_after_undo()


def _operator_key(operator):
    """Get a stable key without storing Python attributes on bpy.types.Operator."""
    try:
        return operator.as_pointer()
    except (AttributeError, ReferenceError, RuntimeError):
        return id(operator)


def _finish_operator(operator):
    operator_key = _operator_key(operator)
    area_key = _modal_operator_areas.pop(operator_key, None)
    if area_key is None:
        return
    state = _active_states.pop(area_key, None)
    _finish_state(state)


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


class VIEW3D_OT_mesh_focus_orbit(bpy.types.Operator):
    """Double-tap the configured key to toggle the viewport-center orbit target."""

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

        # The configured key keeps the normal MFO shortcut.  Holding Ctrl while
        # double-tapping it selects the independent Face Set MFO mode.
        face_set_mode = bool(getattr(event, "ctrl", False))
        tap_key = (area_key, face_set_mode)

        now = time.monotonic()
        previous_tap = _last_tap_times.get(tap_key)
        if previous_tap is None or now - previous_tap > double_tap_window:
            _last_tap_times[tap_key] = now
            # A single tap must remain a normal Blender modifier event.
            return {"PASS_THROUGH"}

        _last_tap_times.pop(tap_key, None)
        existing_state = _active_states.get(area_key)
        if existing_state and existing_state.active:
            same_mode = (
                isinstance(existing_state, _FaceSetOrbitState)
                if face_set_mode
                else not isinstance(existing_state, _FaceSetOrbitState)
            )
            if same_mode:
                _deactivate_state(existing_state)
                return {"FINISHED"}
            return {"PASS_THROUGH"}

        # A previous modal instance may have been invalidated by Undo or
        # window/file lifecycle events.  Recover scene remnants before a new
        # Face Set MFO activation can create another proxy.
        _cleanup_orphan_face_set_proxies()

        if face_set_mode:
            hit = _raycast_reference_face_set(context)
            if hit is None:
                # No Reference Object surface or Face Set at the center.
                return {"PASS_THROUGH"}
            reference_object, _face_index, face_set_id, hit_location, hit_distance = hit
            state = _FaceSetOrbitState(
                context,
                hit_location,
                hit_distance,
                activation_key,
                reference_object,
                face_set_id,
            )
        else:
            hit = _find_center_hit(context)
            if hit is None:
                # No Reference Object surface at the center: do nothing.
                return {"PASS_THROUGH"}
            state = _TempOrbitState(context, hit[0], hit[1], activation_key)

        operator_key = _operator_key(self)

        try:
            if face_set_mode and not _build_face_set_proxy(state):
                state.active = False
                return {"PASS_THROUGH"}
            if face_set_mode:
                state.retopo_isolation = _retopo_begin_isolation(
                    context,
                    state.reference_object,
                    state.proxy_object,
                )
            _active_states[area_key] = state
            _modal_operator_areas[operator_key] = area_key
            if face_set_mode:
                _install_polyquilt_qsnap_filter(context=context)
                if _retopoflow_hidden_vertex_filter_enabled():
                    _install_retopoflow_hidden_vertex_filter()
            _activate_state(state)
        except (ReferenceError, RuntimeError, AttributeError, TypeError, ValueError):
            _active_states.pop(area_key, None)
            _modal_operator_areas.pop(operator_key, None)
            _retopo_restore_isolation(state)
            _remove_face_set_proxy(state)
            state.active = False
            _release_retopoflow_hidden_vertex_filter()
            _release_polyquilt_qsnap_filter(state=state)
            return {"PASS_THROUGH"}

        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, _context, event):
        area_key = _modal_operator_areas.get(_operator_key(self))
        state = _active_states.get(area_key)
        if state is None or not state.active:
            return {"CANCELLED"}

        if event.type == "WINDOW_DEACTIVATE":
            prefs = _addon_preferences()
            if prefs and prefs.focus_loss_behavior == "EXIT":
                _finish_operator(self)
                return {"CANCELLED"}
            return {"PASS_THROUGH"}

        # Passing all events through is what lets Blender's native Navigation
        # Gizmo and MMB orbit continue to handle the gesture.
        return {"PASS_THROUGH"}

    def __del__(self):
        # Covers cancellation during workspace/window teardown.
        try:
            _finish_operator(self)
        except (ReferenceError, RuntimeError, AttributeError):
            pass


def _raycast_sculpt_face_set(context, coord):
    """Return the active mesh, seed face, Face Set ID and cursor position."""
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

        hit, location, _normal, face_index = obj.ray_cast(
            local_origin,
            local_direction,
        )
        if not hit or face_index < 0 or face_index >= len(face_set_attr.data):
            return None

        return (
            obj,
            int(face_index),
            int(face_set_attr.data[face_index].value),
            obj.matrix_world @ location,
            coord,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _build_local_face_set_adjacency(mesh):
    """Build compact edge-to-face incidence data for one mesh topology."""
    face_count = len(mesh.polygons)
    edge_count = len(mesh.edges)
    loop_count = len(mesh.loops)

    loop_edges = array("i", [0]) * loop_count
    loop_vertices = array("i", [0]) * loop_count
    face_loop_starts = array("i", [0]) * face_count
    face_loop_totals = array("i", [0]) * face_count

    if loop_count:
        mesh.loops.foreach_get("edge_index", loop_edges)
        mesh.loops.foreach_get("vertex_index", loop_vertices)
    if face_count:
        mesh.polygons.foreach_get("loop_start", face_loop_starts)
        mesh.polygons.foreach_get("loop_total", face_loop_totals)

    edge_face_a = array("i", [-1]) * edge_count
    edge_face_b = array("i", [-1]) * edge_count
    non_manifold_faces = {}

    for face_index in range(face_count):
        start = face_loop_starts[face_index]
        end = start + face_loop_totals[face_index]
        for loop_index in range(start, end):
            edge_index = loop_edges[loop_index]
            if edge_index < 0 or edge_index >= edge_count:
                continue
            if edge_face_a[edge_index] < 0:
                edge_face_a[edge_index] = face_index
            elif edge_face_b[edge_index] < 0:
                edge_face_b[edge_index] = face_index
            else:
                non_manifold_faces.setdefault(
                    edge_index,
                    [edge_face_a[edge_index], edge_face_b[edge_index]],
                ).append(face_index)

    return {
        "loop_edges": loop_edges,
        "loop_vertices": loop_vertices,
        "face_loop_starts": face_loop_starts,
        "face_loop_totals": face_loop_totals,
        "edge_face_a": edge_face_a,
        "edge_face_b": edge_face_b,
        "non_manifold_faces": non_manifold_faces,
    }


def _get_cached_local_face_set_adjacency(mesh):
    """Reuse topology data while the mesh topology remains unchanged."""
    key = (
        mesh.as_pointer(),
        len(mesh.vertices),
        len(mesh.edges),
        len(mesh.polygons),
        len(mesh.loops),
    )
    cached = _local_face_set_adjacency_cache.get(key)
    if cached is None:
        cached = _build_local_face_set_adjacency(mesh)
        _local_face_set_adjacency_cache.clear()
        _local_face_set_adjacency_cache[key] = cached
    return cached


def _local_face_geometry(state, face_index):
    cached = state["geometry_cache"].get(face_index)
    if cached is not None:
        return cached

    polygon = state["mesh"].polygons[face_index]
    center = polygon.center.copy()
    normal = polygon.normal.copy()
    if normal.length_squared:
        normal.normalize()
    else:
        normal = Vector((0.0, 0.0, 1.0))

    geometry = {
        "center": center,
        "normal": normal,
        "area": max(float(polygon.area), 1.0e-12),
        "scale": max(math.sqrt(float(polygon.area)), 1.0e-8),
    }
    state["geometry_cache"][face_index] = geometry
    return geometry


def _local_face_neighbors(state, face_index):
    """Yield (neighbor face, shared edge) pairs for edge-connected faces."""
    adjacency = state["adjacency"]
    loop_edges = adjacency["loop_edges"]
    starts = adjacency["face_loop_starts"]
    totals = adjacency["face_loop_totals"]
    edge_face_a = adjacency["edge_face_a"]
    edge_face_b = adjacency["edge_face_b"]
    non_manifold_faces = adjacency["non_manifold_faces"]

    start = starts[face_index]
    end = start + totals[face_index]
    yielded = set()

    for loop_index in range(start, end):
        edge_index = loop_edges[loop_index]
        if edge_index < 0 or edge_index >= len(edge_face_a):
            continue

        neighbors = []
        first = edge_face_a[edge_index]
        second = edge_face_b[edge_index]
        if first == face_index:
            neighbors.append(second)
        elif second == face_index:
            neighbors.append(first)

        for neighbor in non_manifold_faces.get(edge_index, ()):
            neighbors.append(neighbor)

        for neighbor in neighbors:
            if neighbor >= 0 and neighbor != face_index and neighbor not in yielded:
                yielded.add(neighbor)
                yield neighbor, edge_index


def _local_face_edge_direction(state, face_index, edge_index):
    """Return the current face's oriented direction along a shared edge."""
    adjacency = state["adjacency"]
    starts = adjacency["face_loop_starts"]
    totals = adjacency["face_loop_totals"]
    loop_edges = adjacency["loop_edges"]
    loop_vertices = adjacency["loop_vertices"]
    start = starts[face_index]
    total = totals[face_index]

    if total <= 0:
        return None

    mesh = state["mesh"]
    for offset in range(total):
        loop_index = start + offset
        if loop_edges[loop_index] != edge_index:
            continue
        next_loop_index = start + ((offset + 1) % total)
        vertex_a = loop_vertices[loop_index]
        vertex_b = loop_vertices[next_loop_index]
        try:
            direction = mesh.vertices[vertex_b].co - mesh.vertices[vertex_a].co
        except (IndexError, ReferenceError, RuntimeError):
            return None
        if direction.length_squared == 0.0:
            return None
        direction.normalize()
        return direction
    return None


def _local_face_smoothed_normal(state, face_index):
    """Fast area-weighted, spatially limited smoothing around one face.

    A one-ring edge neighborhood is intentional here.  On an AI-generated
    high-density mesh, recursively collecting many rings for every traversed
    face becomes the dominant cost and smooths across the very valleys that
    are supposed to stop the fill.  The radius still rejects neighbors from a
    wildly different local scale, while the area weighting suppresses tiny
    triangle-normal noise.
    """
    cached = state["smoothed_normals"].get(face_index)
    if cached is not None:
        return cached

    center_geometry = _local_face_geometry(state, face_index)
    center = center_geometry["center"]
    base_normal = center_geometry["normal"]
    radius = max(
        state["normal_smoothing_radius"],
        center_geometry["scale"] * 3.0,
    )
    radius_squared = radius * radius
    raw_angle_limit = math.cos(state["normal_raw_angle_limit"])

    weighted_normal = base_normal * center_geometry["area"]
    sample_count = 1
    for neighbor, _edge_index in _local_face_neighbors(state, face_index):
        if sample_count >= state["normal_sample_limit"]:
            break
        geometry = _local_face_geometry(state, neighbor)
        offset = geometry["center"] - center
        if offset.length_squared > radius_squared:
            continue
        if base_normal.dot(geometry["normal"]) < raw_angle_limit:
            continue
        distance = math.sqrt(max(offset.length_squared, 0.0))
        falloff = 1.0 / (1.0 + distance / max(radius, 1.0e-8))
        weighted_normal += geometry["normal"] * max(
            geometry["area"] * falloff,
            1.0e-12,
        )
        sample_count += 1

    if weighted_normal.length_squared == 0.0:
        weighted_normal = base_normal.copy()
    else:
        weighted_normal.normalize()

    state["smoothed_normals"][face_index] = weighted_normal
    return weighted_normal


def _local_face_transition_cost(state, face_index, neighbor, edge_index):
    """Return a local shape transition cost for bottleneck region growing."""
    cache_key = (face_index, neighbor, edge_index)
    cached = state["transition_cache"].get(cache_key)
    if cached is not None:
        return cached

    current_geometry = _local_face_geometry(state, face_index)
    neighbor_geometry = _local_face_geometry(state, neighbor)
    current_normal = _local_face_smoothed_normal(state, face_index)
    neighbor_normal = _local_face_smoothed_normal(state, neighbor)
    dot_normal = max(-1.0, min(1.0, current_normal.dot(neighbor_normal)))
    normal_angle = math.acos(dot_normal)
    normal_cost = min(
        normal_angle / state["normal_angle_limit"],
        2.0,
    ) * state["normal_variation_weight"]

    raw_dot = max(
        -1.0,
        min(1.0, current_geometry["normal"].dot(neighbor_geometry["normal"])),
    )
    raw_angle = math.acos(raw_dot)
    raw_edge_cost = min(
        raw_angle / state["raw_edge_angle_limit"],
        2.0,
    ) * state["raw_edge_weight"]

    concavity = 0.0
    edge_direction = _local_face_edge_direction(state, face_index, edge_index)
    if edge_direction is not None:
        # Use raw normals for the sign so a smoothed normal cannot erase the
        # narrow concave valley that should separate adjacent hair bundles.
        turn = current_geometry["normal"].cross(
            neighbor_geometry["normal"]
        ).dot(edge_direction)
        concavity = max(0.0, min(1.0, (-turn - 0.15) / 0.65))

    cost = normal_cost + raw_edge_cost + (
        concavity * state["concavity_penalty"]
    )
    state["transition_cache"][cache_key] = cost
    return cost


def _smart_face_set_fill(context, coord, strict_mode=False):
    """Find and apply one geometry-aware Face Set region."""
    hit = _raycast_sculpt_face_set(context, coord)
    if hit is None:
        return None

    obj, seed_face, seed_face_set, _seed_location, _screen_position = hit
    mesh = obj.data
    face_set_attr = mesh.attributes.get(".sculpt_face_set")
    if face_set_attr is None or face_set_attr.domain != "FACE":
        return None

    if strict_mode:
        parameters = {
            "normal_smoothing_radius_factor": SMART_FACE_SET_FILL_STRICT_NORMAL_SMOOTH_RADIUS_FACTOR,
            "normal_variation_weight": SMART_FACE_SET_FILL_STRICT_NORMAL_VARIATION_WEIGHT,
            "normal_angle_limit": SMART_FACE_SET_FILL_STRICT_NORMAL_ANGLE_LIMIT,
            "normal_raw_angle_limit": SMART_FACE_SET_FILL_STRICT_NORMAL_RAW_ANGLE_LIMIT,
            "normal_sample_limit": SMART_FACE_SET_FILL_NORMAL_SAMPLE_LIMIT,
            "raw_edge_weight": SMART_FACE_SET_FILL_STRICT_RAW_EDGE_WEIGHT,
            "raw_edge_angle_limit": SMART_FACE_SET_FILL_STRICT_RAW_EDGE_ANGLE_LIMIT,
            "concavity_penalty": SMART_FACE_SET_FILL_STRICT_CONCAVITY_PENALTY,
            "acceptance_threshold": SMART_FACE_SET_FILL_STRICT_ACCEPTANCE_THRESHOLD,
            "max_faces": SMART_FACE_SET_FILL_MAX_FACES,
            "max_geodesic_scale": SMART_FACE_SET_FILL_STRICT_MAX_GEODESIC_SCALE,
            "max_geodesic_factor": SMART_FACE_SET_FILL_STRICT_MAX_GEODESIC_FACTOR,
            "min_geodesic_factor": SMART_FACE_SET_FILL_STRICT_MIN_GEODESIC_FACTOR,
        }
    else:
        parameters = {
            "normal_smoothing_radius_factor": SMART_FACE_SET_FILL_NORMAL_SMOOTH_RADIUS_FACTOR,
            "normal_variation_weight": SMART_FACE_SET_FILL_NORMAL_VARIATION_WEIGHT,
            "normal_angle_limit": SMART_FACE_SET_FILL_NORMAL_ANGLE_LIMIT,
            "normal_raw_angle_limit": SMART_FACE_SET_FILL_NORMAL_RAW_ANGLE_LIMIT,
            "normal_sample_limit": SMART_FACE_SET_FILL_NORMAL_SAMPLE_LIMIT,
            "raw_edge_weight": SMART_FACE_SET_FILL_RAW_EDGE_WEIGHT,
            "raw_edge_angle_limit": SMART_FACE_SET_FILL_RAW_EDGE_ANGLE_LIMIT,
            "concavity_penalty": SMART_FACE_SET_FILL_CONCAVITY_PENALTY,
            "acceptance_threshold": SMART_FACE_SET_FILL_ACCEPTANCE_THRESHOLD,
            "max_faces": SMART_FACE_SET_FILL_MAX_FACES,
            "max_geodesic_scale": SMART_FACE_SET_FILL_MAX_GEODESIC_SCALE,
            "max_geodesic_factor": SMART_FACE_SET_FILL_MAX_GEODESIC_FACTOR,
            "min_geodesic_factor": SMART_FACE_SET_FILL_MIN_GEODESIC_FACTOR,
        }

    state = {
        "object": obj,
        "mesh": mesh,
        "adjacency": _get_cached_local_face_set_adjacency(mesh),
        "geometry_cache": {},
        "smoothed_normals": {},
        "transition_cache": {},
        "strict_mode": bool(strict_mode),
        **parameters,
    }

    seed_geometry = _local_face_geometry(state, seed_face)
    seed_scale = seed_geometry["scale"]
    state["normal_smoothing_radius"] = max(
        seed_scale * state["normal_smoothing_radius_factor"],
        1.0e-8,
    )
    object_diagonal = max(Vector(obj.dimensions).length, 1.0e-8)
    state["max_geodesic_distance"] = max(
        object_diagonal * state["min_geodesic_factor"],
        min(
            seed_scale * state["max_geodesic_scale"],
            object_diagonal * state["max_geodesic_factor"],
        ),
    )

    best_cost = {seed_face: 0.0}
    best_distance = {seed_face: 0.0}
    heap = [(0.0, 0.0, seed_face)]
    candidates = set()

    while heap and len(candidates) < state["max_faces"]:
        path_cost, path_distance, face_index = heapq.heappop(heap)
        known_cost = best_cost.get(face_index, float("inf"))
        known_distance = best_distance.get(face_index, float("inf"))
        if path_cost > known_cost + 1.0e-9:
            continue
        if (
            abs(path_cost - known_cost) <= 1.0e-9
            and path_distance > known_distance + 1.0e-9
        ):
            continue

        candidates.add(face_index)
        current_geometry = _local_face_geometry(state, face_index)

        for neighbor, edge_index in _local_face_neighbors(state, face_index):
            neighbor_geometry = _local_face_geometry(state, neighbor)
            step_distance = (
                neighbor_geometry["center"] - current_geometry["center"]
            ).length
            next_distance = path_distance + step_distance
            if next_distance > state["max_geodesic_distance"]:
                continue

            transition_cost = _local_face_transition_cost(
                state,
                face_index,
                neighbor,
                edge_index,
            )
            next_cost = max(path_cost, transition_cost)
            if next_cost > state["acceptance_threshold"]:
                continue

            previous_cost = best_cost.get(neighbor, float("inf"))
            previous_distance = best_distance.get(neighbor, float("inf"))
            if (
                next_cost < previous_cost - 1.0e-9
                or (
                    abs(next_cost - previous_cost) <= 1.0e-9
                    and next_distance < previous_distance
                )
            ):
                best_cost[neighbor] = next_cost
                best_distance[neighbor] = next_distance
                heapq.heappush(heap, (next_cost, next_distance, neighbor))

    changed_faces = 0
    for face_index in candidates:
        if face_set_attr.data[face_index].value != seed_face_set:
            face_set_attr.data[face_index].value = seed_face_set
            changed_faces += 1

    if changed_faces:
        mesh.update()
    return len(candidates), changed_faces


class VIEW3D_OT_mesh_focus_local_face_set_grow(bpy.types.Operator):
    """Apply the seed Face Set to one geometry-aware connected region."""

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
        if not self.poll(context):
            return {"PASS_THROUGH"}

        mouse_x = getattr(event, "mouse_region_x", None)
        mouse_y = getattr(event, "mouse_region_y", None)
        if mouse_x is None or mouse_y is None:
            return {"CANCELLED"}

        self.mouse_region_x = int(mouse_x)
        self.mouse_region_y = int(mouse_y)
        self.strict_mode = bool(
            self.strict_mode or getattr(event, "shift", False)
        )
        return self.execute(context)

    def execute(self, context):
        if not self.poll(context):
            return {"CANCELLED"}

        try:
            result = _smart_face_set_fill(
                context,
                Vector((self.mouse_region_x, self.mouse_region_y)),
                strict_mode=bool(self.strict_mode),
            )
        except (
            AttributeError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
            MemoryError,
        ):
            result = None

        if result is None:
            return {"CANCELLED"}

        candidate_count, changed_faces = result
        self.report(
            {"INFO"},
            f"Smart Face Set Fill: {changed_faces} faces changed "
            f"({candidate_count} candidates)",
        )
        return {"FINISHED"}


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
            LOCAL_FACE_SET_GROW_OPERATOR_ID,
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
        keymap_item = keymap.keymap_items.new(OPERATOR_ID, activation_key, "PRESS")
        # The configured event itself is a modifier key.  Accept Ctrl here so
        # Ctrl+double-tap can select Face Set MFO; Alt/OSKey are rejected in
        # invoke() because they are not part of either activation gesture.
        keymap_item.any = True
        _addon_keymaps.append((keymap, keymap_item))

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
            shift=True,
        )
        strict_grow_item.properties.strict_mode = True
        _addon_keymaps.append((keymap, strict_grow_item))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        _remove_keymaps()


def _preferences_changed(_self, _context):
    if _is_registered:
        _rebuild_keymaps()


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
    retopoflow_hidden_vertex_filter: BoolProperty(
        name="RetopoFlow Hidden Vertex Filter",
        description=(
            "During Face Set MFO, temporarily exclude hidden Retopo vertices "
            "from RetopoFlow nearest-vertex snapping"
        ),
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "enabled")
        layout.prop(self, "activation_key")
        layout.prop(context.scene, "mfo_reference_object", text="Reference Object")
        layout.prop(self, "focus_loss_behavior")
        layout.prop(self, "debug_display")
        layout.prop(self, "show_indicator")
        layout.prop(self, "retopoflow_hidden_vertex_filter")
        layout.prop(self, "double_tap_window")
        layout.separator()
        layout.label(text="Double-tap the activation key in a 3D Viewport.")
        layout.label(text="Ctrl + double-tap uses Face Set MFO on the Reference Object.")
        layout.label(text="The viewport center is ray-cast once on activation.")


CLASSES = (
    VIEW3D_OT_mesh_focus_orbit_recover_face_set_state,
    VIEW3D_OT_mesh_focus_orbit,
    VIEW3D_OT_mesh_focus_local_face_set_grow,
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
    _restore_retopoflow_hidden_vertex_filter()
    _schedule_orphan_cleanup()
    if _on_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_load_pre)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    if _on_undo_pre not in bpy.app.handlers.undo_pre:
        bpy.app.handlers.undo_pre.append(_on_undo_pre)
    if _on_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_on_undo_post)
    _rebuild_keymaps()


def unregister():
    global _is_registered
    if not _is_registered:
        _restore_retopoflow_hidden_vertex_filter()
        return
    _finish_all_states()
    _cleanup_orphan_face_set_proxies()
    _restore_retopoflow_hidden_vertex_filter()
    if _on_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_on_load_pre)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    if _on_undo_pre in bpy.app.handlers.undo_pre:
        bpy.app.handlers.undo_pre.remove(_on_undo_pre)
    if _on_undo_post in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(_on_undo_post)
    _remove_keymaps()
    _local_face_set_adjacency_cache.clear()
    _is_registered = False
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, "mfo_reference_object"):
        del bpy.types.Scene.mfo_reference_object


if __name__ == "__main__":
    register()
