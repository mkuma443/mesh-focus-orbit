"""Smart Fill preview/operator component.

Loaded by the package entry point in dependency order. This module owns Smart
Fill preview and modal behavior and imports cross-domain state explicitly.
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

from ..config import (
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
from .. import runtime as _runtime
from ..foundation import (
    _FILL_PREVIEW_BOUNDARY_ROUTE_COMPONENT_BUDGET,
    _FILL_PREVIEW_BOUNDARY_ROUTE_COMPONENT_PAIR_BUDGET,
    _FILL_PREVIEW_BOUNDARY_ROUTE_SHAPE_PAIR_BUDGET,
    _FILL_PREVIEW_BOUNDARY_ROUTE_TOTAL_PAIR_BUDGET,
    _tag_redraw,
    _topology_color_tag_redraw_all,
)
from .geometry import (
    _fill_preview_array_fingerprint,
    _fill_preview_cursor_build_geometry,
    _fill_preview_cursor_build_shading_geometry,
    _fill_preview_face_set_initial_component,
    _fill_preview_face_vertex_sequence,
    _fill_preview_local_geometry,
    _fill_preview_multi_source_metric_shell,
    _fill_preview_prepare_token,
    _fill_preview_preserve_initial_ids,
    _fill_preview_progressive_range_step,
    _fill_preview_region,
    _fill_preview_restore_visible_analysis_profile,
    _fill_preview_shading_proxy,
    _fill_preview_shadow_view_key,
    _fill_preview_signature,
    _fill_preview_valley_component_is_crossing,
    _fill_preview_visible_analysis_profile,
)
from .invariants import (
    monotonic_visible_selection,
    visible_seed_reached_component,
    visible_floor_enabled,
)
from ..guided_ridge.core import (
    _raycast_sculpt_face_set,
    _vertex_paint_active_color_attribute,
    _vertex_paint_color_signature,
    _vertex_paint_sample_color,
)

_FILL_PREVIEW_WHEEL_DRAIN_SECONDS = FILL_PREVIEW_WHEEL_DRAIN_SECONDS
_FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD


def _fill_preview_establish_visible_universe(state, geometry):
    """Capture the immutable visible seed component for this session.

    The progressive radius is allowed to change, but visibility and seed
    connectivity are not radius-dependent.  Hidden faces are deliberately
    omitted from the snapshot and become stable graph barriers for every
    subsequent wheel stage.
    """
    import numpy as np

    if not isinstance(geometry, dict):
        raise RuntimeError("preview visibility graph is unavailable")
    count = int(geometry.get("count", 0))
    face_ids = np.asarray(
        geometry.get("face_ids", np.arange(count, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    hidden = np.asarray(
        geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool
    ).reshape(-1)
    if len(face_ids) != count or len(hidden) != count:
        raise RuntimeError("preview visibility graph schema is invalid")
    seed_face = int(state.get("seed_face", -1))
    seed_rows = np.flatnonzero(face_ids == seed_face)
    if len(seed_rows) != 1:
        raise RuntimeError("preview seed is outside the visible graph")
    reached_ids = state.get("visible_reached_ids")
    if reached_ids is None:
        # A defensive first-stage fallback uses the current candidate only.
        # Normal production calls populate this from the incremental Dijkstra
        # distances before the floor is established.
        reached_ids = state.get("visible_candidate_ids", (int(seed_rows[0]),))
    visible, component, reason = visible_seed_reached_component(
        count,
        hidden,
        int(seed_rows[0]),
        reached_ids,
    )
    if reason != "ok":
        raise RuntimeError(f"preview visible component unavailable: {reason}")
    state["visible_universe"] = np.array(visible, dtype=bool, copy=True)
    state["visible_component"] = np.array(component, dtype=bool, copy=True)
    state["visible_face_ids"] = np.array(face_ids[visible], dtype=np.int32, copy=True)
    state["visible_component_face_ids"] = np.array(
        face_ids[component], dtype=np.int32, copy=True
    )
    state["visible_hidden_snapshot"] = np.array(hidden, dtype=bool, copy=True)
    state["visible_topology_fingerprint"] = geometry.get("topology_fingerprint")
    state["visible_seed_local"] = int(seed_rows[0])
    state["visible_snapshot_ready"] = True


def _fill_preview_apply_visible_selection_floor(state, geometry, local_ids, radius):
    """Return a visible, seed-connected candidate with monotonic growth.

    Only increasing radius stages union the previously accepted selection. A
    shrink therefore remains an exact cache lookup or a fresh bounded result;
    it never inherits an additive frontier from a larger radius.
    """
    import numpy as np

    if not state.get("visible_snapshot_ready"):
        state["visible_candidate_ids"] = np.asarray(local_ids, dtype=np.int32).reshape(-1)
        _fill_preview_establish_visible_universe(state, geometry)
    reached_ids = state.get("visible_reached_ids")
    if reached_ids is not None:
        reached = np.asarray(reached_ids, dtype=np.int64).reshape(-1)
        in_range = (reached >= 0) & (reached < len(geometry.get("hidden", ())))
        reached = np.unique(reached[in_range])
        if len(reached):
            visible_component = np.asarray(
                state.get("visible_component", ()), dtype=bool
            ).reshape(-1)
            visible_component[reached] = ~np.asarray(
                geometry.get("hidden", ()), dtype=bool
            ).reshape(-1)[reached]
            state["visible_component"] = visible_component
    hidden = np.asarray(geometry.get("hidden", ()), dtype=bool).reshape(-1)
    snapshot_hidden = np.asarray(
        state.get("visible_hidden_snapshot", ()), dtype=bool
    ).reshape(-1)
    if len(hidden) != len(snapshot_hidden) or not np.array_equal(hidden, snapshot_hidden):
        raise RuntimeError("Smart Fill visibility changed during preview")
    snapshot_topology = state.get("visible_topology_fingerprint")
    current_topology = geometry.get("topology_fingerprint")
    if snapshot_topology != current_topology:
        raise RuntimeError("Smart Fill topology changed during preview")
    face_ids = np.asarray(
        geometry.get("face_ids", np.arange(len(hidden), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    visible_universe = np.asarray(
        state.get("visible_universe", ()), dtype=bool
    ).reshape(-1)
    visible_component = np.asarray(
        state.get("visible_component", ()), dtype=bool
    ).reshape(-1)
    if (
        len(face_ids) != len(hidden)
        or len(visible_universe) != len(face_ids)
        or len(visible_component) != len(face_ids)
    ):
        raise RuntimeError("Smart Fill visible graph changed during preview")
    candidate = np.asarray(local_ids, dtype=np.int32).reshape(-1)
    candidate = candidate[(candidate >= 0) & (candidate < len(face_ids))]
    candidate = candidate[visible_component[candidate]]
    previous_radius = state.get("processed_radius")
    accepted = state.get("accepted_visible_ids", np.empty(0, dtype=np.int32))
    growing = previous_radius is not None and float(radius) >= float(previous_radius)
    if growing:
        merged_ids, reason = monotonic_visible_selection(
            accepted,
            candidate,
            visible_universe,
            visible_component,
            int(state["visible_seed_local"]),
        )
        if reason != "ok" or merged_ids is None:
            raise RuntimeError(f"Smart Fill monotonic selection unavailable: {reason}")
        candidate = np.asarray(merged_ids, dtype=np.int32).reshape(-1)
    seed_local = int(state["visible_seed_local"])
    if seed_local not in set(int(value) for value in candidate):
        candidate = np.unique(np.r_[candidate, np.asarray([seed_local], dtype=np.int32)])
    return np.unique(candidate).astype(np.int32, copy=False)

class VIEW3D_OT_mesh_focus_shadow_analysis_toggle(bpy.types.Operator):
    """Toggle the toon_dark + Face Set + wire analysis look in this View3D."""

    bl_idname = "view3d.mesh_focus_shadow_analysis_toggle"
    bl_label = "Mesh Focus: Toggle Shadow Analysis View"
    bl_description = (
        "Show toon_dark with Face Sets and wire overlay, or restore this View3D"
    )

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.space_data is not None
            and context.mode == "SCULPT"
        )

    def execute(self, context):
        if not self.poll(context):
            return {"CANCELLED"}
        key = _fill_preview_shadow_view_key(context)
        if not key:
            self.report({"WARNING"}, "Shadow analysis view: no View3D space")
            return {"CANCELLED"}
        token = _runtime.shadow_analysis_view_tokens.get(key)
        if isinstance(token, dict) and token.get("active"):
            restored = _fill_preview_restore_visible_analysis_profile(token)
            _runtime.shadow_analysis_view_tokens.pop(key, None)
            if restored:
                self.report(
                    {"INFO"},
                    "Analysis View: OFF (restored; Shift+Alt+E to enable)",
                )
                _tag_redraw(context.area)
                return {"FINISHED"}
            self.report({"WARNING"}, "Shadow analysis view: restore failed")
            return {"CANCELLED"}
        token = _fill_preview_visible_analysis_profile(context)
        if not token.get("applied"):
            self.report(
                {"WARNING"},
                "Shadow analysis view: " + str(token.get("reason", "apply failed")),
            )
            return {"CANCELLED"}
        token["manual"] = True
        _runtime.shadow_analysis_view_tokens[key] = token
        self.report(
            {"INFO"},
            "Analysis View: toon_dark + Face Sets + Wire ON "
            "(Shift+Alt+E to restore)",
        )
        return {"FINISHED"}


def _fill_preview_shadow_pair_signal(state, local, pair_index, with_reason=False):
    """Sample the captured screen-space shadow line at one physical edge."""
    import numpy as np

    capture = state.get("shadow_capture")
    if not isinstance(capture, dict) or not capture.get("ok"):
        value = (0.0, 0, "capture-unavailable")
        return value if with_reason else value[:2]
    line_strength = np.asarray(capture.get("line_strength", ()), dtype=np.float32)
    line_map = np.asarray(capture.get("line_map", ()), dtype=bool)
    projection = np.asarray(capture.get("view_projection"), dtype=np.float64)
    if (
        line_strength.ndim != 2 or line_map.shape != line_strength.shape
        or projection.shape != (4, 4)
    ):
        value = (0.0, 0, "capture-schema")
        return value if with_reason else value[:2]
    try:
        pair_index = int(pair_index)
        vertices = np.asarray(local["world_vertices"], dtype=np.float64)
        v0 = int(local["pair_v0"][pair_index])
        v1 = int(local["pair_v1"][pair_index])
        if local.get("vertex_id_space") == "mesh-global":
            mesh = state["obj"].data
            matrix = state["obj"].matrix_world
            a = np.asarray(matrix @ mesh.vertices[v0].co, dtype=np.float64)
            b = np.asarray(matrix @ mesh.vertices[v1].co, dtype=np.float64)
        else:
            a = vertices[v0]
            b = vertices[v1]
        if a.shape != (3,) or b.shape != (3,):
            value = (0.0, 0, "endpoint-schema")
            return value if with_reason else value[:2]
        samples = []
        projected_samples = 0
        for fraction in (0.25, 0.5, 0.75):
            point = a * (1.0 - fraction) + b * fraction
            clip = projection @ np.r_[point, 1.0]
            if not np.all(np.isfinite(clip)) or abs(float(clip[3])) <= 1.0e-12:
                continue
            ndc = clip[:3] / float(clip[3])
            x = int(round((float(ndc[0]) * 0.5 + 0.5) * (line_strength.shape[1] - 1)))
            y = int(round((float(ndc[1]) * 0.5 + 0.5) * (line_strength.shape[0] - 1)))
            if 1 <= x < line_strength.shape[1] - 1 and 1 <= y < line_strength.shape[0] - 1:
                projected_samples += 1
                crop = line_strength[y - 1:y + 2, x - 1:x + 2]
                mask = line_map[y - 1:y + 2, x - 1:x + 2]
                if np.any(mask):
                    samples.append(float(np.max(crop[mask])))
        if projected_samples == 0:
            value = (0.0, 0, "projection-no-sample")
            return value if with_reason else value[:2]
        if not samples:
            value = (0.0, projected_samples, "line-not-hit")
            return value if with_reason else value[:2]
        value = (float(max(samples)), projected_samples, "line-hit")
        return value if with_reason else value[:2]
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
        value = (0.0, 0, "endpoint-schema")
        return value if with_reason else value[:2]


def _fill_preview_shadow_edge_key(local, pair_index):
    """Return a direction-independent physical-edge cache key."""
    import numpy as np

    try:
        pair_index = int(pair_index)
        edge_ids = np.asarray(local.get("pair_edge_indices", ()), dtype=np.int64).reshape(-1)
        if 0 <= pair_index < len(edge_ids) and int(edge_ids[pair_index]) >= 0:
            return ("edge", int(edge_ids[pair_index]))
        v0 = np.asarray(local.get("pair_v0", ()), dtype=np.int64).reshape(-1)
        v1 = np.asarray(local.get("pair_v1", ()), dtype=np.int64).reshape(-1)
        if 0 <= pair_index < len(v0) and 0 <= pair_index < len(v1):
            return ("vertices", min(int(v0[pair_index]), int(v1[pair_index])), max(int(v0[pair_index]), int(v1[pair_index])))
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return None


def _fill_preview_shadow_pair_cached_signal(state, local, pair_index):
    """Read one cached physical-edge signal, with a safe uncached fallback."""
    cache = state.get("shadow_edge_cache")
    key = _fill_preview_shadow_edge_key(local, pair_index)
    if isinstance(cache, dict) and key is not None:
        values = cache.get("edges", {}).get(key)
        if isinstance(values, dict):
            return float(values.get("signal", 0.0)), int(values.get("tested", 0)), str(values.get("reason", "cached"))
    return _fill_preview_shadow_pair_signal(state, local, pair_index, with_reason=True)


def _fill_preview_shadow_edge_cache(state, geometry):
    """Build/update an immutable-per-invoke physical-edge shadow map."""
    import numpy as np

    cache = state.get("shadow_edge_cache")
    if not isinstance(cache, dict) or cache.get("generation") != state.get("shadow_capture_generation"):
        cache = {
            "generation": int(state.get("shadow_capture_generation", 0)),
            "edges": {},
            "total_pairs": 0,
            "tested": 0,
            "skipped_hard": 0,
            "invalid_endpoint": 0,
            "projection_no_sample": 0,
            "line_no_hit": 0,
        }
    first = np.asarray(geometry.get("first", ()), dtype=np.int32).reshape(-1)
    second = np.asarray(geometry.get("second", ()), dtype=np.int32).reshape(-1)
    hidden = np.asarray(geometry.get("hidden", np.zeros(int(geometry.get("count", 0)), dtype=bool)), dtype=bool).reshape(-1)
    source_hard = np.asarray(geometry.get("source_hard", np.zeros(len(hidden), dtype=bool)), dtype=bool).reshape(-1)
    if len(source_hard) != len(hidden):
        source_hard = np.zeros(len(hidden), dtype=bool)
    for pair_index in range(min(len(first), len(second))):
        key = _fill_preview_shadow_edge_key(geometry, pair_index)
        if key is None or key in cache["edges"]:
            continue
        left, right = int(first[pair_index]), int(second[pair_index])
        cache["total_pairs"] += 1
        if (
            left < 0 or right < 0 or left >= len(hidden) or right >= len(hidden)
            or hidden[left] or hidden[right] or source_hard[left] or source_hard[right]
        ):
            cache["skipped_hard"] += 1
            cache["edges"][key] = {"signal": 0.0, "tested": 0, "reason": "hard-safety"}
            continue
        signal, tested, reason = _fill_preview_shadow_pair_signal(state, geometry, pair_index, with_reason=True)
        cache["tested"] += int(tested > 0)
        if reason == "projection-no-sample":
            cache["projection_no_sample"] += 1
        elif reason in {"endpoint-schema", "capture-schema"}:
            cache["invalid_endpoint"] += 1
        elif reason == "line-not-hit":
            cache["line_no_hit"] += 1
        cache["edges"][key] = {"signal": float(signal), "tested": int(tested), "reason": reason}
    state["shadow_edge_cache"] = cache
    return cache


def _fill_preview_shadow_face_key(geometry, face_index):
    """Return a stable mesh-face key for one cursor/local geometry face."""
    try:
        face_ids = geometry.get("face_ids")
        if face_ids is not None:
            value = int(face_ids[int(face_index)])
        else:
            value = int(face_index)
        return ("face", value)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _fill_preview_shadow_face_luminance_cache(state, geometry):
    """Sample neutral viewport luminance at every source face once per invoke.

    The cache deliberately stores face samples rather than screen edge hits.
    A wheel stage can therefore reuse the same lighting observation while the
    graph radius changes, and two neighbouring faces with a shared edge still
    receive independent near/far values.
    """
    import numpy as np

    capture = state.get("shadow_capture")
    generation = int(state.get("shadow_capture_generation", 0))
    cached = state.get("shadow_face_luminance_cache")
    if (
        isinstance(cached, dict)
        and cached.get("generation") == generation
        and cached.get("geometry_signature") == geometry.get("signature")
    ):
        return cached
    cache = {
        "generation": generation,
        "geometry_signature": geometry.get("signature"),
        "values": {},
        "sampled": 0,
        "visible": 0,
        "unsampled": 0,
        "patch_radius": 1,
    }
    if not isinstance(capture, dict) or not capture.get("ok"):
        state["shadow_face_luminance_cache"] = cache
        return cache
    luminance = np.asarray(capture.get("luminance", ()), dtype=np.float32)
    projection = np.asarray(capture.get("view_projection"), dtype=np.float64)
    centers = np.asarray(geometry.get("centers", ()), dtype=np.float64)
    count = int(geometry.get("count", len(centers)))
    if luminance.ndim != 2 or projection.shape != (4, 4) or len(centers) < count:
        state["shadow_face_luminance_cache"] = cache
        return cache
    height, width = luminance.shape
    radius = int(cache["patch_radius"])
    for face_index in range(count):
        key = _fill_preview_shadow_face_key(geometry, face_index)
        if key is None:
            cache["unsampled"] += 1
            continue
        try:
            clip = projection @ np.r_[centers[face_index], 1.0]
            if not np.all(np.isfinite(clip)) or abs(float(clip[3])) <= 1.0e-12:
                cache["values"][key] = {
                    "luminance": None, "sampled": False, "reason": "projection-invalid"
                }
                cache["unsampled"] += 1
                continue
            ndc = clip[:3] / float(clip[3])
            if not np.all(np.isfinite(ndc)):
                raise ValueError("projection-invalid")
            x = int(round((float(ndc[0]) * 0.5 + 0.5) * (width - 1)))
            y = int(round((float(ndc[1]) * 0.5 + 0.5) * (height - 1)))
            if x < 0 or x >= width or y < 0 or y >= height:
                cache["values"][key] = {
                    "luminance": None, "sampled": False, "reason": "outside-capture"
                }
                cache["unsampled"] += 1
                continue
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            patch = np.asarray(luminance[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
            patch = patch[np.isfinite(patch)]
            if len(patch) == 0:
                raise ValueError("empty-luminance-patch")
            value = float(np.median(patch))
            cache["values"][key] = {
                "luminance": value, "sampled": True, "reason": "face-patch-median"
            }
            cache["sampled"] += 1
            cache["visible"] += 1
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            cache["values"][key] = {
                "luminance": None, "sampled": False, "reason": "face-sample-failed"
            }
            cache["unsampled"] += 1
    # Remove triangle/cavity speckle before any edge subtraction.  This is a
    # fixed physical one-ring median on the source graph (not a candidate-size
    # filter), so a later radius cannot change a face's observed luminance.
    raw_by_face = np.full(count, np.nan, dtype=np.float64)
    for face_index in range(count):
        key = _fill_preview_shadow_face_key(geometry, face_index)
        entry = cache["values"].get(key, {}) if key is not None else {}
        try:
            value = float(entry.get("luminance"))
        except (AttributeError, TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            raw_by_face[face_index] = value
    offsets = np.asarray(geometry.get("offsets", ()), dtype=np.int64).reshape(-1)
    neighbors = np.asarray(geometry.get("neighbors", ()), dtype=np.int32).reshape(-1)
    smoothed_by_face = np.array(raw_by_face, copy=True)
    if len(offsets) == count + 1 and int(offsets[-1]) <= len(neighbors):
        for face_index in range(count):
            ring_ids = np.r_[
                np.asarray([face_index], dtype=np.int32),
                neighbors[int(offsets[face_index]):int(offsets[face_index + 1])],
            ]
            ring_values = raw_by_face[ring_ids]
            ring_values = ring_values[np.isfinite(ring_values)]
            if len(ring_values):
                smoothed_by_face[face_index] = float(np.median(ring_values))
    for face_index in range(count):
        key = _fill_preview_shadow_face_key(geometry, face_index)
        if key is None or key not in cache["values"]:
            continue
        entry = cache["values"][key]
        if np.isfinite(smoothed_by_face[face_index]):
            entry["smoothed_luminance"] = float(smoothed_by_face[face_index])
        else:
            entry["smoothed_luminance"] = None
    cache["smoothed_count"] = int(np.count_nonzero(np.isfinite(smoothed_by_face)))
    cache["raw_values_by_face"] = raw_by_face
    cache["smoothed_values_by_face"] = smoothed_by_face
    state["shadow_face_luminance_cache"] = cache
    return cache


def _fill_preview_shadow_luminance_edge_cache_legacy(
    state, geometry, distances, seed_geometry_face
):
    """Build a direction-aware, seed-relative shadow knee map.

    This is intentionally a small robust edge calculation, not a second
    region solver.  The existing graph flood still owns extent and topology;
    this map only classifies physical shared edges as a shadow onset.  The
    context for a threshold is the one-ring of the edge, so candidate size
    and the current radius cannot alter a previously observed decision.
    """
    import numpy as np

    generation = int(state.get("shadow_capture_generation", 0))
    signature = geometry.get("signature")
    cached = state.get("shadow_luminance_edge_cache")
    if (
        isinstance(cached, dict)
        and cached.get("generation") == generation
        and cached.get("geometry_signature") == signature
    ):
        return cached
    face_cache = _fill_preview_shadow_face_luminance_cache(state, geometry)
    cache = {
        "generation": generation,
        "geometry_signature": signature,
        "edges": {},
        "total_pairs": 0,
        "tested": 0,
        "skipped_hard": 0,
        "unsampled": 0,
        "gentle": 0,
        "brightening": 0,
        "knee_candidates": 0,
        "knee_adopted": 0,
        "no_context_rejected": 0,
        "component_count": 0,
        "post_close_component_count": 0,
        "component_lengths": (),
        "traced_sequence_count": 0,
        "junction_pair_count": 0,
        "junction_split_count": 0,
        "junction_ambiguous_count": 0,
        "junction_fallback_count": 0,
        "closing_window_physical": 0.0,
        "closing_window_n": 0,
        "closing_filled_physical": 0.0,
        "free_endpoint_count": 0,
        "closed_chain_count": 0,
        "domain_spanning_chain_count": 0,
        "adopted_edge_count": 0,
        "adopted_after_coherence": 0,
        "plateau_median": None,
        "plateau_mad": 0.0,
        "smoothed_count": int(face_cache.get("smoothed_count", 0)),
        "local_trend_median": 0.0,
        "local_trend_mad": 0.0,
        "seed_luminance": None,
        "seed_luminance_used": False,
        "reason": "capture-unavailable",
    }
    first = np.asarray(geometry.get("first", ()), dtype=np.int32).reshape(-1)
    second = np.asarray(geometry.get("second", ()), dtype=np.int32).reshape(-1)
    count = int(geometry.get("count", 0))
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    source_offsets = np.asarray(
        geometry.get("offsets", ()), dtype=np.int64
    ).reshape(-1)
    source_neighbors = np.asarray(
        geometry.get("neighbors", ()), dtype=np.int32
    ).reshape(-1)
    hidden = np.asarray(
        geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool
    ).reshape(-1)
    source_hard = np.asarray(
        geometry.get("source_hard", np.zeros(count, dtype=bool)), dtype=bool
    ).reshape(-1)
    if len(source_hard) != count:
        physical_degree = np.asarray(
            geometry.get("physical_degree", ()), dtype=np.int32
        ).reshape(-1)
        edge_counts = np.asarray(
            geometry.get("face_edge_counts", ()), dtype=np.int32
        ).reshape(-1)
        if len(physical_degree) == count and len(edge_counts) == count:
            source_hard = hidden | (physical_degree < edge_counts)
        else:
            source_hard = np.zeros(count, dtype=bool)
    if (
        not face_cache.get("values")
        or len(first) != len(second)
        or len(distances) < count
    ):
        state["shadow_luminance_edge_cache"] = cache
        return cache
    smoothed_by_face = np.asarray(
        face_cache.get("smoothed_values_by_face", ()), dtype=np.float64
    ).reshape(-1)
    if len(smoothed_by_face) != count:
        smoothed_by_face = np.full(count, np.nan, dtype=np.float64)
    seed_key = _fill_preview_shadow_face_key(geometry, seed_geometry_face)
    seed_entry = face_cache.get("values", {}).get(seed_key, {})
    try:
        seed_luminance = float(seed_entry.get("smoothed_luminance"))
    except (AttributeError, TypeError, ValueError):
        seed_luminance = float("nan")
    # Establish the plateau from a fixed one-ring around the seed.  This is
    # the measured local variation/noise floor; a global capture MAD is not a
    # reliable noise estimate on a large viewport with a real shadow.
    plateau_ids = np.asarray([int(seed_geometry_face)], dtype=np.int32)
    if (
        0 <= int(seed_geometry_face) < count
        and len(source_offsets) == count + 1
    ):
        seed_start = int(source_offsets[int(seed_geometry_face)])
        seed_end = int(source_offsets[int(seed_geometry_face) + 1])
        plateau_ids = np.r_[
            plateau_ids,
            source_neighbors[seed_start:seed_end],
        ]
    plateau_ids = plateau_ids[(plateau_ids >= 0) & (plateau_ids < count)]
    plateau_values = smoothed_by_face[plateau_ids]
    plateau_values = plateau_values[np.isfinite(plateau_values)]
    if len(plateau_values):
        plateau_median = float(np.median(plateau_values))
        plateau_mad = float(
            1.4826 * np.median(np.abs(plateau_values - plateau_median))
        )
        cache["plateau_median"] = plateau_median
        cache["plateau_mad"] = plateau_mad
        if not np.isfinite(seed_luminance):
            seed_luminance = plateau_median
    if np.isfinite(seed_luminance):
        cache["seed_luminance"] = seed_luminance
        cache["seed_luminance_used"] = True
    finite_distance = np.isfinite(distances[:count])
    tolerance = max(
        float(np.median(np.asarray(geometry.get("pair_lengths", (1.0,)), dtype=np.float64)))
        * 1.0e-8,
        1.0e-9,
    )
    incident = [[] for _ in range(count)]
    raw = np.zeros(len(first), dtype=np.float64)
    tested = np.zeros(len(first), dtype=bool)
    near_face = np.full(len(first), -1, dtype=np.int32)
    far_face = np.full(len(first), -1, dtype=np.int32)
    near_lum = np.full(len(first), np.nan, dtype=np.float64)
    far_lum = np.full(len(first), np.nan, dtype=np.float64)
    for pair_index, (left_value, right_value) in enumerate(zip(first, second)):
        left, right = int(left_value), int(right_value)
        cache["total_pairs"] += 1
        if (
            left < 0 or right < 0 or left >= count or right >= count
            or hidden[left] or hidden[right] or source_hard[left] or source_hard[right]
            or not finite_distance[left] or not finite_distance[right]
        ):
            cache["skipped_hard"] += 1
            continue
        left_key = _fill_preview_shadow_face_key(geometry, left)
        right_key = _fill_preview_shadow_face_key(geometry, right)
        left_lum = (
            float(smoothed_by_face[left]) if 0 <= left < len(smoothed_by_face) else float("nan")
        )
        right_lum = (
            float(smoothed_by_face[right]) if 0 <= right < len(smoothed_by_face) else float("nan")
        )
        if not np.isfinite(left_lum) or not np.isfinite(right_lum):
            cache["unsampled"] += 1
            continue
        if distances[left] < distances[right] - tolerance:
            near, far = left, right
            lnear, lfar = left_lum, right_lum
        elif distances[right] < distances[left] - tolerance:
            near, far = right, left
            lnear, lfar = right_lum, left_lum
        else:
            # Tangential/equal-distance edges have no outward direction and
            # must not become barriers from a bright/dark ordering accident.
            continue
        near_face[pair_index], far_face[pair_index] = near, far
        near_lum[pair_index], far_lum[pair_index] = lnear, lfar
        raw[pair_index] = max(0.0, float(lnear - lfar))
        tested[pair_index] = True
        cache["tested"] += 1
        incident[near].append(pair_index)
        incident[far].append(pair_index)
    # The plateau supplies a relative SNR floor.  It scales with the captured
    # seed luminance and measured local variation, so a flat area with tiny
    # raster noise cannot become a field of orange edges.
    plateau_reference = (
        float(cache["plateau_median"])
        if cache["plateau_median"] is not None
        else float(seed_luminance) if np.isfinite(seed_luminance) else 0.0
    )
    plateau_mad = float(cache.get("plateau_mad", 0.0))
    noise_floor = max(
        2.0 * plateau_mad,
        0.015 * max(abs(plateau_reference), 1.0e-2),
    )
    trend_values = []
    raw_candidate_indices = []
    for pair_index in np.flatnonzero(tested):
        near, far = int(near_face[pair_index]), int(far_face[pair_index])
        # Only predecessor edges whose far face is this edge's near face are
        # valid trend context.  Mixing the near/far incident fan makes a
        # nearly flat 2-D patch look like a zero-trend discontinuity.
        predecessors = [
            other for other in incident[near]
            if other != pair_index
            and tested[other]
            and int(far_face[other]) == near
            and float(distances[int(near_face[other])])
            < float(distances[near]) - tolerance
        ]
        trend_source = [float(raw[other]) for other in predecessors]
        drop = float(raw[pair_index])
        far_value = float(far_lum[pair_index])
        dark_from_plateau = bool(
            np.isfinite(seed_luminance)
            and max(0.0, plateau_reference - far_value) > noise_floor
        )
        if trend_source:
            trend = float(np.median(trend_source))
            mad = float(
                1.4826 * np.median(np.abs(np.asarray(trend_source) - trend))
            )
            trend_values.extend(trend_source)
            margin = max(2.0 * mad, 0.75 * noise_floor)
            knee = bool(
                dark_from_plateau
                and drop > noise_floor
                and drop > trend + margin
            )
        else:
            cache["no_context_rejected"] += 1
            trend = 0.0
            mad = 0.0
            # A seed-rim edge has no predecessor.  It may be an entrance only
            # when its drop is large relative to the measured plateau and the
            # other seed-rim crossings provide line context.
            seed_rim = near == int(seed_geometry_face)
            knee = bool(
                seed_rim
                and dark_from_plateau
                and drop > noise_floor
                and len(incident[near]) >= 2
            )
        if knee:
            cache["knee_candidates"] += 1
            raw_candidate_indices.append(int(pair_index))
        elif drop > 0.0:
            if far_value >= float(near_lum[pair_index]):
                cache["brightening"] += 1
            else:
                cache["gentle"] += 1
        key = _fill_preview_shadow_edge_key(geometry, pair_index)
        if key is not None:
            cache["edges"][key] = {
                "signal": drop,
                "tested": 1,
                "barrier": False,
                "candidate": bool(knee),
                "reason": "shadow-knee-candidate" if knee else (
                    "brightening-pass" if far_value >= float(near_lum[pair_index])
                    else "gentle-darkening-pass"
                ),
                "near_luminance": float(near_lum[pair_index]),
                "far_luminance": far_value,
                "drop": drop,
                "trend": trend,
                "mad": mad,
                "dark_from_seed": bool(dark_from_plateau),
                "context_count": int(len(predecessors)),
            }
    # A knee candidate is not a barrier until it forms a finite physical line.
    # Connected components use the shared face and, when available, the shared
    # mesh vertex of the physical edges.  This removes isolated face-raster
    # noise without changing the scalar luminance calculation itself.
    candidate_set = set(int(value) for value in raw_candidate_indices)
    face_candidates = {}
    for pair_index in raw_candidate_indices:
        face_candidates.setdefault(int(first[pair_index]), []).append(int(pair_index))
        face_candidates.setdefault(int(second[pair_index]), []).append(int(pair_index))
    pair_v0 = np.asarray(geometry.get("pair_v0", ()), dtype=np.int32).reshape(-1)
    pair_v1 = np.asarray(geometry.get("pair_v1", ()), dtype=np.int32).reshape(-1)

    def shares_physical_vertex(left_pair, right_pair):
        if (
            len(pair_v0) == len(first) and len(pair_v1) == len(first)
            and pair_v0[left_pair] >= 0 and pair_v1[left_pair] >= 0
            and pair_v0[right_pair] >= 0 and pair_v1[right_pair] >= 0
        ):
            return bool(
                pair_v0[left_pair] in (pair_v0[right_pair], pair_v1[right_pair])
                or pair_v1[left_pair] in (pair_v0[right_pair], pair_v1[right_pair])
            )
        return True

    remaining = set(candidate_set)
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = [start]
        pending = [start]
        while pending:
            current = pending.pop()
            shared_faces = (int(first[current]), int(second[current]))
            neighbors_in_line = set()
            for face in shared_faces:
                neighbors_in_line.update(face_candidates.get(face, ()))
            for other in sorted(neighbors_in_line):
                if other in remaining and shares_physical_vertex(current, other):
                    remaining.remove(other)
                    pending.append(other)
                    component.append(other)
        components.append(tuple(sorted(component)))
    cache["component_count"] = int(len(components))
    component_lengths = np.asarray(
        geometry.get("pair_lengths", np.ones(len(first))), dtype=np.float64
    ).reshape(-1)
    finite_lengths = component_lengths[tested & np.isfinite(component_lengths)]
    median_length = float(np.median(finite_lengths)) if len(finite_lengths) else 1.0
    pair_points = np.asarray(geometry.get("pair_edge_points", ()), dtype=np.float64)
    projection = np.asarray(
        state.get("shadow_capture", {}).get("view_projection"), dtype=np.float64
    )
    adopted = set()
    accepted_lengths = []
    orientation_rejected = 0
    for component in components:
        length = float(np.sum(component_lengths[list(component)])) if len(component_lengths) else 0.0
        orientation_ok = True
        projected_directions = []
        if pair_points.ndim == 3 and pair_points.shape[1:] == (2, 3) and projection.shape == (4, 4):
            for pair_index in component:
                if pair_index >= len(pair_points) or not np.all(np.isfinite(pair_points[pair_index])):
                    continue
                clip = np.column_stack((pair_points[pair_index], np.ones(2))) @ projection.T
                if not np.all(np.isfinite(clip)) or np.any(np.abs(clip[:, 3]) <= 1.0e-12):
                    continue
                ndc = clip[:, :2] / clip[:, 3:4]
                direction = ndc[1] - ndc[0]
                norm = float(np.linalg.norm(direction))
                if norm > 1.0e-8:
                    projected_directions.append(direction / norm)
            if len(projected_directions) >= 2:
                reference = projected_directions[0]
                alignment = [abs(float(np.dot(reference, direction))) for direction in projected_directions[1:]]
                orientation_ok = bool(float(np.median(alignment)) >= 0.10)
        coherent = bool(
            len(component) >= 2
            and length >= max(2.0 * median_length, 1.0e-12)
            and orientation_ok
        )
        if coherent:
            adopted.update(component)
            accepted_lengths.append(length)
        elif not orientation_ok:
            orientation_rejected += 1
    cache["component_lengths"] = tuple(float(value) for value in accepted_lengths)
    cache["orientation_rejected"] = int(orientation_rejected)
    cache["adopted_after_coherence"] = int(len(adopted))
    cache["knee_adopted"] = int(len(adopted))
    for pair_index in raw_candidate_indices:
        key = _fill_preview_shadow_edge_key(geometry, pair_index)
        if key is None or key not in cache["edges"]:
            continue
        entry = cache["edges"][key]
        if pair_index in adopted:
            entry["barrier"] = True
            entry["reason"] = "shadow-knee-coherent-line"
        else:
            entry["reason"] = "knee-no-coherence"
    if trend_values:
        trend_array = np.asarray(trend_values, dtype=np.float64)
        trend_median = float(np.median(trend_array))
        cache["local_trend_median"] = trend_median
        cache["local_trend_mad"] = float(
            1.4826 * np.median(np.abs(trend_array - trend_median))
        )
    cache["reason"] = "seed-relative-shadow-knee" if cache["seed_luminance_used"] else "seed-unsampled"
    state["shadow_luminance_edge_cache"] = cache
    return cache


def _fill_preview_shadow_luminance_edge_cache(
    state, geometry, distances, seed_geometry_face
):
    """Classify a fresh luminance field with metric grayscale morphology.

    ``S_view`` is an absolute luminance contrast per physical edge length.
    The seed plateau supplies its relative tolerance and contrast scale.  A
    local scalar max/min closing is applied before a physical-line coherence
    check, so isolated raster noise cannot become an orange boundary while a
    short gap in a real shadow line can be recovered.
    """
    import numpy as np

    generation = int(state.get("shadow_capture_generation", 0))
    signature = geometry.get("signature")
    cached = state.get("shadow_luminance_edge_cache")
    if (
        isinstance(cached, dict)
        and cached.get("generation") == generation
        and cached.get("geometry_signature") == signature
    ):
        return cached
    face_cache = _fill_preview_shadow_face_luminance_cache(state, geometry)
    first = np.asarray(geometry.get("first", ()), dtype=np.int32).reshape(-1)
    second = np.asarray(geometry.get("second", ()), dtype=np.int32).reshape(-1)
    count = int(geometry.get("count", 0))
    edge_count = min(len(first), len(second))
    cache = {
        "generation": generation,
        "geometry_signature": signature,
        "edges": {},
        "total_pairs": int(edge_count),
        "tested": 0,
        "skipped_hard": 0,
        "unsampled": 0,
        "gentle": 0,
        "brightening": 0,
        "knee_candidates": 0,
        "knee_adopted": 0,
        "no_context_rejected": 0,
        "component_count": 0,
        "post_close_component_count": 0,
        "component_lengths": (),
        "traced_sequence_count": 0,
        "junction_pair_count": 0,
        "junction_split_count": 0,
        "junction_ambiguous_count": 0,
        "junction_fallback_count": 0,
        "closing_window_physical": 0.0,
        "closing_window_n": 0,
        "closing_filled_physical": 0.0,
        "free_endpoint_count": 0,
        "closed_chain_count": 0,
        "domain_spanning_chain_count": 0,
        "adopted_edge_count": 0,
        "adopted_after_coherence": 0,
        "closing_filled_gaps": 0,
        "raw_high_edges": 0,
        "raw_s_min": 0.0,
        "raw_s_median": 0.0,
        "raw_s_max": 0.0,
        "raw_s_mad": 0.0,
        "plateau_median": None,
        "plateau_mad": 0.0,
        "baseline_tolerance": 0.0,
        "contrast_scale": 0.0,
        "smoothed_count": int(face_cache.get("smoothed_count", 0)),
        "local_trend_median": 0.0,
        "local_trend_mad": 0.0,
        "seed_luminance": None,
        "seed_luminance_used": False,
        "reason": "capture-unavailable",
    }
    if not face_cache.get("values") or edge_count == 0:
        state["shadow_luminance_edge_cache"] = cache
        return cache
    smoothed = np.asarray(
        face_cache.get("smoothed_values_by_face", ()), dtype=np.float64
    ).reshape(-1)
    if len(smoothed) != count:
        smoothed = np.full(count, np.nan, dtype=np.float64)
    offsets = np.asarray(geometry.get("offsets", ()), dtype=np.int64).reshape(-1)
    neighbors = np.asarray(geometry.get("neighbors", ()), dtype=np.int32).reshape(-1)
    seed_index = int(seed_geometry_face)
    seed_ring = [seed_index]
    if 0 <= seed_index < count and len(offsets) == count + 1:
        seed_ring.extend(
            int(value) for value in neighbors[offsets[seed_index]:offsets[seed_index + 1]]
        )
    seed_ring = np.asarray(seed_ring, dtype=np.int32)
    seed_ring = seed_ring[(seed_ring >= 0) & (seed_ring < count)]
    plateau_values = smoothed[seed_ring]
    plateau_values = plateau_values[np.isfinite(plateau_values)]
    if len(plateau_values):
        plateau_median = float(np.median(plateau_values))
        plateau_mad = float(
            1.4826 * np.median(np.abs(plateau_values - plateau_median))
        )
        cache["plateau_median"] = plateau_median
        cache["plateau_mad"] = plateau_mad
    else:
        plateau_median = float("nan")
        plateau_mad = 0.0
    seed_key = _fill_preview_shadow_face_key(geometry, seed_index)
    seed_entry = face_cache.get("values", {}).get(seed_key, {})
    try:
        seed_luminance = float(seed_entry.get("smoothed_luminance"))
    except (AttributeError, TypeError, ValueError):
        seed_luminance = float("nan")
    if not np.isfinite(seed_luminance) and np.isfinite(plateau_median):
        seed_luminance = plateau_median
    if np.isfinite(seed_luminance):
        cache["seed_luminance"] = seed_luminance
        cache["seed_luminance_used"] = True
    plateau_reference = (
        plateau_median if np.isfinite(plateau_median) else seed_luminance
    )
    baseline_tolerance = max(
        2.0 * plateau_mad,
        0.015 * max(abs(float(plateau_reference)), 1.0e-2)
        if np.isfinite(plateau_reference) else 0.0,
    )
    cache["baseline_tolerance"] = float(baseline_tolerance)
    pair_lengths = np.asarray(
        geometry.get("pair_lengths", ()), dtype=np.float64
    ).reshape(-1)
    if len(pair_lengths) != edge_count:
        pair_lengths = np.linalg.norm(
            np.asarray(geometry.get("centers", np.empty((0, 3))), dtype=np.float64)[second]
            - np.asarray(geometry.get("centers", np.empty((0, 3))), dtype=np.float64)[first],
            axis=1,
        ) if edge_count and len(geometry.get("centers", ())) else np.ones(edge_count)
    finite_lengths = pair_lengths[np.isfinite(pair_lengths) & (pair_lengths > 1.0e-12)]
    edge_scale = float(np.median(finite_lengths)) if len(finite_lengths) else 1.0
    contrast_scale = max(
        3.0 * plateau_mad,
        0.02 * max(abs(float(plateau_reference)), 1.0e-2)
        if np.isfinite(plateau_reference) else 0.02,
        1.0e-4,
    )
    cache["contrast_scale"] = float(contrast_scale)
    hidden = np.asarray(
        geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool
    ).reshape(-1)
    source_hard = np.asarray(
        geometry.get("source_hard", np.zeros(count, dtype=bool)), dtype=bool
    ).reshape(-1)
    if len(source_hard) != count:
        physical_degree = np.asarray(geometry.get("physical_degree", ()), dtype=np.int32).reshape(-1)
        edge_counts = np.asarray(geometry.get("face_edge_counts", ()), dtype=np.int32).reshape(-1)
        source_hard = (
            hidden | (physical_degree < edge_counts)
            if len(physical_degree) == count and len(edge_counts) == count
            else np.zeros(count, dtype=bool)
        )
    incident = [[] for _ in range(count)]
    raw_s = np.zeros(edge_count, dtype=np.float64)
    departures = np.zeros(edge_count, dtype=np.float64)
    tested = np.zeros(edge_count, dtype=bool)
    for pair_index in range(edge_count):
        left, right = int(first[pair_index]), int(second[pair_index])
        if (
            left < 0 or right < 0 or left >= count or right >= count
            or hidden[left] or hidden[right] or source_hard[left] or source_hard[right]
        ):
            cache["skipped_hard"] += 1
            continue
        left_lum = float(smoothed[left]) if np.isfinite(smoothed[left]) else float("nan")
        right_lum = float(smoothed[right]) if np.isfinite(smoothed[right]) else float("nan")
        if not np.isfinite(left_lum) or not np.isfinite(right_lum):
            cache["unsampled"] += 1
            continue
        length = max(float(pair_lengths[pair_index]), 0.25 * edge_scale, 1.0e-12)
        # S_view = abs(I(A)-I(B))/d * Weight_contrast.  The edge-scale factor
        # keeps the result dimensionless while retaining physical-length
        # behavior across nonuniform tessellation.
        score = abs(left_lum - right_lum) / length * edge_scale / contrast_scale
        raw_s[pair_index] = float(score)
        departures[pair_index] = max(
            abs(left_lum - float(plateau_reference)),
            abs(right_lum - float(plateau_reference)),
        ) if np.isfinite(plateau_reference) else 0.0
        tested[pair_index] = True
        cache["tested"] += 1
        incident[left].append(pair_index)
        incident[right].append(pair_index)
    valid_scores = raw_s[tested]
    if len(valid_scores):
        cache["raw_s_min"] = float(np.min(valid_scores))
        cache["raw_s_median"] = float(np.median(valid_scores))
        cache["raw_s_max"] = float(np.max(valid_scores))
        cache["raw_s_mad"] = float(
            1.4826 * np.median(np.abs(valid_scores - cache["raw_s_median"]))
        )
    raw_high = np.zeros(edge_count, dtype=bool)
    local_threshold = np.zeros(edge_count, dtype=np.float64)
    context_values = []
    for pair_index in np.flatnonzero(tested):
        left, right = int(first[pair_index]), int(second[pair_index])
        context_ids = set(incident[left]) | set(incident[right])
        context_ids.discard(int(pair_index))
        values = [float(raw_s[other]) for other in sorted(context_ids) if tested[other]]
        context_values.extend(values)
        if not values:
            cache["no_context_rejected"] += 1
        median = float(np.median(values)) if values else 0.0
        mad = (
            float(1.4826 * np.median(np.abs(np.asarray(values) - median)))
            if values else float(cache["raw_s_mad"])
        )
        threshold = median + max(1.0 * mad, 0.5) if values else float("inf")
        local_threshold[pair_index] = threshold
        departed = departures[pair_index] > baseline_tolerance
        raw_high[pair_index] = bool(departed and raw_s[pair_index] > threshold)
        if not raw_high[pair_index]:
            if raw_s[pair_index] < threshold:
                cache["gentle"] += 1
            else:
                cache["brightening"] += 1
        key = _fill_preview_shadow_edge_key(geometry, pair_index)
        if key is not None:
            left_luminance = float(smoothed[left])
            right_luminance = float(smoothed[right])
            cache["edges"][key] = {
                "signal": float(raw_s[pair_index]),
                "raw_signal": float(raw_s[pair_index]),
                "tested": 1,
                "barrier": False,
                "candidate": False,
                "near_luminance": left_luminance,
                "far_luminance": right_luminance,
                "drop": float(abs(left_luminance - right_luminance)),
                "departure": float(departures[pair_index]),
                "local_threshold": float(threshold),
                "reason": "raw-high" if raw_high[pair_index] else "below-local-threshold",
            }
    cache["raw_high_edges"] = int(np.count_nonzero(raw_high))
    if context_values:
        context_array = np.asarray(context_values, dtype=np.float64)
        cache["local_trend_median"] = float(np.median(context_array))
        cache["local_trend_mad"] = float(
            1.4826 * np.median(np.abs(context_array - cache["local_trend_median"]))
        )
    # Trace maximal simple physical-edge sequences before filtering.  The
    # previous implementation connected an entire face fan and ran one-hop
    # max/min filters, which fragmented a long shadow line and allowed the
    # flood to walk around its short pieces.  Endpoints are therefore the
    # primary topology, while a junction is paired only through the most
    # straight continuation.  A near tie is deliberately split.
    pair_v0 = np.asarray(geometry.get("pair_v0", ()), dtype=np.int32).reshape(-1)
    pair_v1 = np.asarray(geometry.get("pair_v1", ()), dtype=np.int32).reshape(-1)
    world_vertices = np.asarray(geometry.get("world_vertices", ()), dtype=np.float64)
    endpoint_ok = len(pair_v0) == edge_count and len(pair_v1) == edge_count
    line_neighbors = [set() for _ in range(edge_count)]
    cache["junction_pair_count"] = 0
    cache["junction_split_count"] = 0
    cache["junction_ambiguous_count"] = 0
    cache["junction_fallback_count"] = 0
    for face in range(count):
        members = [int(value) for value in incident[face] if tested[int(value)]]
        if len(members) < 2:
            continue
        shared = {pair_index: [] for pair_index in members}
        for offset, left_pair in enumerate(members):
            for right_pair in members[offset + 1:]:
                shares = True
                if endpoint_ok:
                    lv = (int(pair_v0[left_pair]), int(pair_v1[left_pair]))
                    rv = (int(pair_v0[right_pair]), int(pair_v1[right_pair]))
                    shares = min(lv + rv) < 0 or bool(set(lv) & set(rv))
                if shares:
                    shared[left_pair].append(right_pair)
                    shared[right_pair].append(left_pair)
        # Faces with a single contour crossing do not contribute a line pair;
        # missing endpoint metadata uses the old deterministic face-fan
        # fallback, but is counted for diagnostics.
        for left_pair in members:
            options = sorted(set(shared[left_pair]))
            if not options:
                continue
            if not endpoint_ok:
                cache["junction_fallback_count"] += len(options)
            if len(options) == 1:
                right_pair = options[0]
                line_neighbors[left_pair].add(right_pair)
                line_neighbors[right_pair].add(left_pair)
                cache["junction_pair_count"] += 1
                continue
            def edge_direction(pair_index):
                if (
                    not endpoint_ok or len(world_vertices) == 0
                    or int(pair_v0[pair_index]) < 0
                    or int(pair_v1[pair_index]) < 0
                    or int(pair_v0[pair_index]) >= len(world_vertices)
                    or int(pair_v1[pair_index]) >= len(world_vertices)
                ):
                    return None
                vector = world_vertices[int(pair_v1[pair_index])] - world_vertices[int(pair_v0[pair_index])]
                length = float(np.linalg.norm(vector))
                return vector / length if np.isfinite(length) and length > 1.0e-12 else None
            direction = edge_direction(left_pair)
            scored = []
            for right_pair in options:
                other_direction = edge_direction(right_pair)
                score = (
                    abs(float(np.dot(direction, other_direction)))
                    if direction is not None and other_direction is not None else -1.0
                )
                scored.append((score, int(right_pair)))
            scored.sort(key=lambda item: (-item[0], item[1]))
            best_score, right_pair = scored[0]
            next_score = scored[1][0]
            if best_score < 0.0 or best_score - next_score < 0.10:
                cache["junction_split_count"] += 1
                cache["junction_ambiguous_count"] += 1
                continue
            line_neighbors[left_pair].add(right_pair)
            line_neighbors[right_pair].add(left_pair)
            cache["junction_pair_count"] += 1

    tested_ids = [int(value) for value in np.flatnonzero(tested)]
    # Extract ordered maximal simple paths.  A node of degree != 2 is a
    # deterministic interval boundary; all-degree-2 components are closed
    # loops.  This keeps branches and unrelated parallel lines separate.
    sequences = []
    visited_line = set()
    for start in tested_ids:
        if start in visited_line or len(line_neighbors[start]) == 2:
            continue
        for first_neighbor in sorted(line_neighbors[start]):
            if first_neighbor in visited_line:
                continue
            sequence = [start]
            previous, current = start, first_neighbor
            while True:
                sequence.append(current)
                visited_line.add(current)
                choices = sorted(line_neighbors[current] - {previous})
                if len(line_neighbors[current]) != 2 or not choices:
                    break
                previous, current = current, choices[0]
                if current in sequence:
                    break
            visited_line.update(sequence)
            if len(sequence) >= 2:
                sequences.append(tuple(sequence))
    for start in tested_ids:
        if start in visited_line:
            continue
        # Remaining nodes are either isolated or closed degree-two loops.
        sequence = [start]
        visited_line.add(start)
        previous, current = start, None
        if line_neighbors[start]:
            current = min(line_neighbors[start])
        while current is not None and current != start and current not in sequence:
            sequence.append(current)
            visited_line.add(current)
            choices = sorted(line_neighbors[current] - ({previous} if previous is not None else set()))
            previous, current = current, (choices[0] if choices else None)
        if len(sequence) >= 2:
            sequences.append(tuple(sequence))
    cache["traced_sequence_count"] = int(len(sequences))
    cache["closed_chain_count"] = int(
        sum(bool(len(line_neighbors[path[0]]) == 2 and len(line_neighbors[path[-1]]) == 2)
            for path in sequences if path)
    )
    cache["free_endpoint_count"] = int(
        sum(1 for node in tested_ids if len(line_neighbors[node]) <= 1)
    )

    # True grayscale Closing on each ordered sequence.  The neighborhood is
    # selected by accumulated physical length, never by a fixed edge count.
    # Two edge-scale gaps are the smallest useful default and the upper bound
    # is intentionally small to keep the live E path bounded.
    window_radius = min(8.0 * edge_scale, max(2.0 * edge_scale, 1.0e-12))
    window_n = 2
    cache["closing_window_physical"] = float(window_radius)
    cache["closing_window_n"] = int(window_n)
    closed_signal = np.array(raw_s, copy=True)
    closing_candidate = np.zeros(edge_count, dtype=bool)
    raw_components = 0
    post_components = 0
    adopted = set()
    accepted_lengths = []
    component_lengths = []
    for path in sequences:
        path = tuple(int(value) for value in path)
        if len(path) < 2:
            continue
        values = np.asarray([raw_s[node] for node in path], dtype=np.float64)
        lengths = np.asarray([max(float(pair_lengths[node]), 1.0e-12) for node in path], dtype=np.float64)
        centers = np.cumsum(lengths) - 0.5 * lengths
        high = np.asarray([raw_high[node] for node in path], dtype=bool)
        raw_support = int(np.count_nonzero(high))
        if raw_support:
            raw_components += 1
        if raw_support < 2:
            continue
        quiet_values = values[~high]
        if len(quiet_values):
            quiet_median = float(np.median(quiet_values))
            quiet_mad = float(
                1.4826 * np.median(np.abs(quiet_values - quiet_median))
            )
            sequence_threshold = max(0.5, quiet_median + 2.0 * quiet_mad)
        else:
            sequence_threshold = max(0.5, float(np.median(values)))
        # Fixed metric window, with N <= 8 and a cumulative physical radius.
        dilation = np.array(values, copy=True)
        for index in range(len(path)):
            ids = np.flatnonzero(np.abs(centers - centers[index]) <= window_radius + 0.5 * lengths[index])
            if len(ids):
                dilation[index] = float(np.max(values[ids]))
        erosion = np.array(dilation, copy=True)
        for index in range(len(path)):
            ids = np.flatnonzero(np.abs(centers - centers[index]) <= window_radius + 0.5 * lengths[index])
            if len(ids):
                erosion[index] = float(np.min(dilation[ids]))
        path_candidate = np.zeros(len(path), dtype=bool)
        for index, node in enumerate(path):
            # The per-edge context threshold is useful for raw knee
            # detection, but it contains the adjacent high support and would
            # reject the very gap that Closing is intended to recover.  Use a
            # robust quiet-side threshold for the post-close scalar instead.
            threshold = float(sequence_threshold)
            path_candidate[index] = bool(
                erosion[index] > threshold
                and departures[node] > 0.5 * baseline_tolerance
            )
            closed_signal[node] = float(erosion[index])
            closing_candidate[node] = bool(path_candidate[index])
        if not np.any(path_candidate):
            continue
        post_components += 1
        candidate_nodes = [path[index] for index in np.flatnonzero(path_candidate)]
        component_length = float(np.sum(pair_lengths[candidate_nodes]))
        component_lengths.append(component_length)
        # A line must retain at least two raw supports.  The closing may add
        # intermediate edges but never invents a disconnected endpoint.
        if len(candidate_nodes) >= 2 and component_length >= 2.0 * edge_scale:
            adopted.update(candidate_nodes)
            accepted_lengths.append(component_length)
            cache["closing_filled_gaps"] += int(
                np.count_nonzero(path_candidate & ~high)
            )
            cache["closing_filled_physical"] = float(
                cache.get("closing_filled_physical", 0.0)
                + np.sum(lengths[path_candidate & ~high])
            )
            for node in candidate_nodes:
                key = _fill_preview_shadow_edge_key(geometry, node)
                if key is not None and key in cache["edges"]:
                    cache["edges"][key]["candidate"] = True
                    cache["edges"][key]["barrier"] = True
                    cache["edges"][key]["closed_signal"] = float(closed_signal[node])
                    cache["edges"][key]["reason"] = "shadow-chain-metric-closing"
        else:
            for node in candidate_nodes:
                key = _fill_preview_shadow_edge_key(geometry, node)
                if key is not None and key in cache["edges"]:
                    cache["edges"][key]["closed_signal"] = float(closed_signal[node])
                    cache["edges"][key]["candidate"] = True
                    cache["edges"][key]["reason"] = "closing-short-chain"
    cache["component_count"] = int(raw_components)
    cache["post_close_component_count"] = int(post_components)
    cache["component_lengths"] = tuple(float(value) for value in accepted_lengths)
    cache["adopted_after_coherence"] = int(len(adopted))
    cache["knee_candidates"] = int(np.count_nonzero(closing_candidate))
    cache["knee_adopted"] = int(len(adopted))
    cache["closing_barrier_count"] = int(len(adopted))
    cache["adopted_edge_count"] = int(len(adopted))
    domain_spanning = 0
    for path in sequences:
        if not path:
            continue
        path_faces = np.asarray(
            [int(face) for node in path for face in (first[node], second[node])],
            dtype=np.int32,
        )
        path_faces = path_faces[(path_faces >= 0) & (path_faces < len(distances))]
        if len(path_faces) and np.isfinite(distances[path_faces]).any():
            domain_spanning += 1
    cache["domain_spanning_chain_count"] = int(domain_spanning)
    cache["reason"] = "seed-relative-view-contrast" if cache["seed_luminance_used"] else "seed-unsampled"
    state["shadow_luminance_edge_cache"] = cache
    return cache


def _fill_preview_shadow_luminance_region(
    state, local, distances, radius, seed_local, source_geometry=None,
    source_distances=None,
):
    """Flood the connected face class under the visible analysis shader.

    Ordinary ``E`` is intentionally a face-luminance classifier.  Distance is
    retained only as a secondary halo/cursor acquisition mechanism; it never
    decides whether a sampled neighbour belongs to the candidate.  The
    classifier uses one immutable threshold per capture generation so wheel
    stages cannot turn a dark face into a pass merely by growing the radius.
    """
    import numpy as np

    count = int(local.get("count", 0))
    local_distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    metrics = {
        "shadow_classifier_mode": "face-luminance-region",
        "shadow_classifier_face_set_independent": True,
        "shadow_classifier_seed_luminance": 0.0,
        "shadow_classifier_threshold": 0.0,
        "shadow_classifier_base_threshold": 0.0,
        "shadow_classifier_threshold_delta": 0.0,
        "shadow_classifier_step": 0,
        "shadow_classifier_monotonic_action": "initial",
        "shadow_classifier_initial_count": 0,
        "shadow_classifier_previous_count": 0,
        "shadow_classifier_preserved_count": 0,
        "shadow_classifier_prevented_removal_count": 0,
        "shadow_classifier_orientation": "unknown",
        "shadow_classifier_sampled_count": 0,
        "shadow_classifier_unsampled_count": 0,
        "shadow_classifier_sampled_min": 0.0,
        "shadow_classifier_sampled_max": 0.0,
        "shadow_classifier_sampled_q10": 0.0,
        "shadow_classifier_sampled_q90": 0.0,
        "shadow_classifier_accepted_count": 0,
        "shadow_classifier_accepted_luminance_min": 0.0,
        "shadow_classifier_accepted_luminance_max": 0.0,
        "shadow_classifier_rejected_neighbor_count": 0,
        "shadow_classifier_rejected_neighbor_min": 0.0,
        "shadow_classifier_rejected_neighbor_max": 0.0,
        "shadow_classifier_boundary_count": 0,
        "shadow_classifier_pass_count": 0,
        "shadow_classifier_reject_count": 0,
        "shadow_classifier_reason": "capture-unavailable",
        "shadow_classifier_threshold_source": "none",
        "shadow_classifier_radius_secondary": float(radius),
    }
    state["shadow_luminance_region_edges"] = set()
    if count <= 0 or seed_local < 0 or seed_local >= count:
        metrics["shadow_classifier_reason"] = "classifier-schema"
        return np.empty(0, dtype=np.int32), metrics
    source_geometry = source_geometry if source_geometry is not None else local
    try:
        face_cache = _fill_preview_shadow_face_luminance_cache(
            state, source_geometry
        )
        smoothed = np.asarray(
            face_cache.get("smoothed_values_by_face", ()), dtype=np.float64
        ).reshape(-1)
        local_global_ids = np.asarray(
            local.get("_global_face_ids", np.arange(count, dtype=np.int32)),
            dtype=np.int64,
        ).reshape(-1)
        if len(local_global_ids) != count:
            raise ValueError("classifier-face-id-schema")
        local_values = np.full(count, np.nan, dtype=np.float64)
        valid_ids = (
            (local_global_ids >= 0)
            & (local_global_ids < len(smoothed))
        )
        local_values[valid_ids] = smoothed[local_global_ids[valid_ids]]
        sampled = np.isfinite(local_values)
        metrics["shadow_classifier_sampled_count"] = int(np.count_nonzero(sampled))
        metrics["shadow_classifier_unsampled_count"] = int(np.count_nonzero(~sampled))
        finite_values = local_values[sampled]
        if not len(finite_values) or not sampled[int(seed_local)]:
            metrics["shadow_classifier_reason"] = "seed-unsampled"
            return np.asarray([int(seed_local)], dtype=np.int32), metrics
        seed_luminance = float(local_values[int(seed_local)])
        metrics["shadow_classifier_seed_luminance"] = seed_luminance
        metrics["shadow_classifier_sampled_min"] = float(np.min(finite_values))
        metrics["shadow_classifier_sampled_max"] = float(np.max(finite_values))
        metrics["shadow_classifier_sampled_q10"] = float(np.percentile(finite_values, 10.0))
        metrics["shadow_classifier_sampled_q90"] = float(np.percentile(finite_values, 90.0))
        generation = int(state.get("shadow_capture_generation", 0))
        classifier_cache = state.get("shadow_luminance_classifier_cache")
        cache_valid = (
            isinstance(classifier_cache, dict)
            and classifier_cache.get("generation") == generation
        )
        if cache_valid:
            base_threshold = float(classifier_cache.get("threshold", seed_luminance))
            bright = bool(classifier_cache.get("bright", True))
            threshold_source = str(classifier_cache.get("threshold_source", "cached"))
        else:
            # Estimate a robust seed plateau from its immediate graph ring.
            offsets = np.asarray(local.get("offsets", ()), dtype=np.int64).reshape(-1)
            neighbors = np.asarray(local.get("neighbors", ()), dtype=np.int32).reshape(-1)
            ring_ids = [int(seed_local)]
            if len(offsets) == count + 1:
                ring_ids.extend(
                    int(v) for v in neighbors[offsets[int(seed_local)]:offsets[int(seed_local) + 1]]
                )
            ring_ids = np.asarray(ring_ids, dtype=np.int32)
            ring_ids = ring_ids[(ring_ids >= 0) & (ring_ids < count)]
            ring_values = local_values[ring_ids]
            ring_values = ring_values[np.isfinite(ring_values)]
            plateau = float(np.median(ring_values)) if len(ring_values) else seed_luminance
            plateau_mad = float(
                1.4826 * np.median(np.abs(ring_values - plateau))
            ) if len(ring_values) else 0.0
            global_median = float(np.median(finite_values))
            bright = bool(seed_luminance >= global_median)
            sorted_values = np.sort(np.unique(finite_values))
            if bright:
                side = sorted_values[sorted_values <= seed_luminance + 1.0e-9]
                gaps = np.diff(side) if len(side) >= 2 else np.empty(0)
                tolerance = max(3.0 * plateau_mad, 0.02 * max(abs(plateau), 1.0e-2), 1.0e-3)
                if len(gaps):
                    gap_index = int(np.argmax(gaps))
                    if float(gaps[gap_index]) > max(2.0 * tolerance, 0.01):
                        threshold = float((side[gap_index] + side[gap_index + 1]) * 0.5)
                        threshold_source = "largest-luminance-gap-below-seed"
                    else:
                        threshold = float(plateau - tolerance)
                        threshold_source = "seed-plateau-relative"
                else:
                    threshold = float(plateau - tolerance)
                    threshold_source = "seed-plateau-relative"
            else:
                side = sorted_values[sorted_values >= seed_luminance - 1.0e-9]
                gaps = np.diff(side) if len(side) >= 2 else np.empty(0)
                tolerance = max(3.0 * plateau_mad, 0.02 * max(abs(plateau), 1.0e-2), 1.0e-3)
                if len(gaps):
                    gap_index = int(np.argmax(gaps))
                    if float(gaps[gap_index]) > max(2.0 * tolerance, 0.01):
                        threshold = float((side[gap_index] + side[gap_index + 1]) * 0.5)
                        threshold_source = "largest-luminance-gap-above-seed"
                    else:
                        threshold = float(plateau + tolerance)
                        threshold_source = "seed-plateau-relative"
                else:
                    threshold = float(plateau + tolerance)
                    threshold_source = "seed-plateau-relative"
            classifier_cache = {
                "generation": generation,
                "threshold": float(threshold),
                "bright": bool(bright),
                "threshold_source": threshold_source,
                "seed_luminance": seed_luminance,
            }
            state["shadow_luminance_classifier_cache"] = classifier_cache
            base_threshold = float(threshold)
        # The threshold is anchored to the first capture, not to the mutable
        # result from the previous wheel stage.  This makes wheel semantics a
        # deterministic one-dimensional control: for a bright seed, expand
        # lowers T and shrink raises T (with the inverse operation returning
        # exactly to the stored step-zero result).
        step = int(state.get("luminance_step", 0))
        sample_min = float(np.min(finite_values))
        sample_max = float(np.max(finite_values))
        sample_span = max(sample_max - sample_min, abs(seed_luminance) * 0.05, 0.02)
        threshold_delta = max(sample_span * 0.05, 0.01)
        threshold = float(base_threshold + (-step if bright else step) * threshold_delta)
        threshold = float(np.clip(threshold, sample_min, sample_max))
        metrics["shadow_classifier_step"] = step
        metrics["shadow_classifier_base_threshold"] = float(base_threshold)
        metrics["shadow_classifier_threshold_delta"] = float(threshold_delta)
        metrics["shadow_classifier_threshold"] = float(threshold)
        metrics["shadow_classifier_orientation"] = "bright" if bright else "dark"
        metrics["shadow_classifier_threshold_source"] = threshold_source
        offsets = np.asarray(local.get("offsets", ()), dtype=np.int64).reshape(-1)
        neighbors = np.asarray(local.get("neighbors", ()), dtype=np.int32).reshape(-1)
        first = np.asarray(local.get("first", ()), dtype=np.int32).reshape(-1)
        second = np.asarray(local.get("second", ()), dtype=np.int32).reshape(-1)
        hidden = np.asarray(local.get("hidden", np.zeros(count, dtype=bool)), dtype=bool).reshape(-1)
        source_hard = np.asarray(local.get("source_hard", np.zeros(count, dtype=bool)), dtype=bool).reshape(-1)
        if len(hidden) != count:
            hidden = np.zeros(count, dtype=bool)
        if len(source_hard) != count:
            source_hard = np.zeros(count, dtype=bool)
        hard = hidden | source_hard
        accepted = np.zeros(count, dtype=bool)
        rejected = np.zeros(count, dtype=bool)
        accepted[int(seed_local)] = True
        pending = deque([int(seed_local)])
        while pending:
            face = int(pending.popleft())
            if len(offsets) != count + 1:
                break
            for row in range(int(offsets[face]), int(offsets[face + 1])):
                neighbor = int(neighbors[row])
                if neighbor < 0 or neighbor >= count or accepted[neighbor] or rejected[neighbor]:
                    continue
                if hard[neighbor] or not sampled[neighbor]:
                    rejected[neighbor] = True
                    continue
                passes = bool(local_values[neighbor] >= threshold) if bright else bool(local_values[neighbor] <= threshold)
                if passes:
                    accepted[neighbor] = True
                    pending.append(neighbor)
                    metrics["shadow_classifier_pass_count"] += 1
                else:
                    rejected[neighbor] = True
                    metrics["shadow_classifier_reject_count"] += 1
        accepted[int(seed_local)] = True
        selected_ids = np.flatnonzero(accepted).astype(np.int32, copy=False)
        # Enforce inclusion/exclusion in stable mesh-face-id space.  Cursor
        # halo growth can reorder local indices, so comparing local arrays is
        # insufficient.  Normal E stores its own immutable step-zero set;
        # strict Ctrl+E never enters this helper and remains untouched.
        source_face_ids = np.asarray(
            source_geometry.get(
                "face_ids", np.arange(len(smoothed), dtype=np.int32)
            ),
            dtype=np.int32,
        ).reshape(-1)
        if len(source_face_ids) != len(smoothed):
            source_face_ids = np.arange(len(smoothed), dtype=np.int32)
        local_stable_ids = np.full(count, -1, dtype=np.int32)
        valid_stable = (
            (local_global_ids >= 0)
            & (local_global_ids < len(source_face_ids))
        )
        local_stable_ids[valid_stable] = source_face_ids[local_global_ids[valid_stable]]
        candidate_stable = np.unique(
            local_stable_ids[selected_ids][local_stable_ids[selected_ids] >= 0]
        ).astype(np.int32, copy=False)
        initial_stable = np.asarray(
            state.get("shadow_luminance_initial_ids", ()), dtype=np.int32
        ).reshape(-1)
        previous_stable = np.asarray(
            state.get("shadow_luminance_previous_ids", ()), dtype=np.int32
        ).reshape(-1)
        action = "initial"
        final_stable = candidate_stable
        if step == 0:
            if len(initial_stable) == 0:
                initial_stable = candidate_stable.copy()
                state["shadow_luminance_initial_ids"] = initial_stable.copy()
            else:
                final_stable = initial_stable.copy()
            action = "initial" if len(previous_stable) == 0 else "return-to-initial"
        elif step > 0:
            parts = [candidate_stable]
            if len(initial_stable):
                parts.append(initial_stable)
            if len(previous_stable) and int(state.get("shadow_luminance_previous_step", 0)) < step:
                parts.append(previous_stable)
            final_stable = np.unique(np.concatenate(parts)).astype(np.int32, copy=False)
            action = "expand-union"
        else:
            final_stable = np.intersect1d(candidate_stable, initial_stable)
            if len(previous_stable) and int(state.get("shadow_luminance_previous_step", 0)) > step:
                final_stable = np.intersect1d(final_stable, previous_stable)
            action = "shrink-intersection"
        if len(local_stable_ids) and len(final_stable):
            accepted = np.isin(local_stable_ids, final_stable)
            accepted &= local_stable_ids >= 0
        else:
            accepted = np.zeros(count, dtype=bool)
        # Stable-ID projection is only a monotonic luminance policy.  It must
        # never override the physical safety mask for a face that is hidden
        # or marked source-hard in the current graph.
        accepted &= ~hard
        accepted[int(seed_local)] = True
        selected_ids = np.flatnonzero(accepted).astype(np.int32, copy=False)
        previous_count = int(len(previous_stable))
        state["shadow_luminance_previous_ids"] = final_stable.copy()
        state["shadow_luminance_previous_step"] = step
        metrics["shadow_classifier_monotonic_action"] = action
        metrics["shadow_classifier_initial_count"] = int(len(initial_stable))
        metrics["shadow_classifier_previous_count"] = previous_count
        metrics["shadow_classifier_preserved_count"] = int(len(np.intersect1d(final_stable, initial_stable)))
        metrics["shadow_classifier_prevented_removal_count"] = int(
            max(0, len(initial_stable) - len(candidate_stable)) if step > 0 else 0
        )
        metrics["shadow_classifier_accepted_count"] = int(len(selected_ids))
        accepted_values = local_values[accepted & sampled]
        if len(accepted_values):
            metrics["shadow_classifier_accepted_luminance_min"] = float(np.min(accepted_values))
            metrics["shadow_classifier_accepted_luminance_max"] = float(np.max(accepted_values))
        rejected_values = []
        edge_keys = set()
        for pair_index, (left, right) in enumerate(zip(first, second)):
            left, right = int(left), int(right)
            if left < 0 or right < 0 or left >= count or right >= count:
                continue
            if bool(accepted[left]) == bool(accepted[right]):
                continue
            key = _fill_preview_shadow_edge_key(local, pair_index)
            if key is not None:
                edge_keys.add(key)
            outside = right if accepted[left] else left
            if 0 <= outside < count and sampled[outside]:
                rejected_values.append(float(local_values[outside]))
        state["shadow_luminance_region_edges"] = edge_keys
        metrics["shadow_classifier_boundary_count"] = int(len(edge_keys))
        metrics["shadow_classifier_rejected_neighbor_count"] = int(len(rejected_values))
        if rejected_values:
            metrics["shadow_classifier_rejected_neighbor_min"] = float(min(rejected_values))
            metrics["shadow_classifier_rejected_neighbor_max"] = float(max(rejected_values))
        metrics["shadow_classifier_reason"] = "connected-luminance-region"
        return selected_ids, metrics
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError, ZeroDivisionError):
        metrics["shadow_classifier_reason"] = "classifier-schema"
        return np.asarray([int(seed_local)], dtype=np.int32), metrics


def _fill_preview_shadow_pre_shadow_snap(
    state, geometry, buffers, metrics, candidate, seed_face
):
    """Choose a coherent mesh chain on the seed side of a shadow onset.

    The screen flood remains authoritative.  This bounded pass only moves an
    interface inward by one mesh ring when the current accepted face is
    already contaminated by the directed luminance transition and an
    adjacent accepted plateau face is available.  All evidence is taken from
    the same sharpened luminance field used by the flood; topology merely
    supplies the local ring and continuity checks.
    """
    import numpy as np
    from collections import deque

    defaults = {
        "shadow_screen_pre_shadow_oriented_chain_count": 0,
        "shadow_screen_pre_shadow_orientation": "unknown",
        "shadow_screen_pre_shadow_shift_count": 0,
        "shadow_screen_pre_shadow_ring_distance": 0,
        "shadow_screen_pre_shadow_accepted_deviation": 0.0,
        "shadow_screen_pre_shadow_rejected_drop": 0.0,
        "shadow_screen_pre_shadow_unchanged_segments": 0,
        "shadow_screen_pre_shadow_no_valid_direction": 0,
        "shadow_screen_pre_shadow_coherent_chain_count": 0,
        "shadow_screen_pre_shadow_seed_connected": False,
        "shadow_screen_pre_shadow_reason": "not-run",
    }

    def done(reason, result, **updates):
        out = dict(defaults)
        out.update(updates)
        out["shadow_screen_pre_shadow_reason"] = str(reason)
        return np.asarray(result, dtype=bool).copy(), out

    refined = np.asarray(candidate, dtype=bool).reshape(-1).copy()
    count = int(geometry.get("count", len(refined)))
    if count <= 0 or len(refined) != count:
        return done("schema", refined)
    first = np.asarray(geometry.get("first", ()), dtype=np.int32).reshape(-1)
    second = np.asarray(geometry.get("second", ()), dtype=np.int32).reshape(-1)
    if len(first) != len(second):
        return done("edge-schema", refined)
    valid = (
        (first >= 0) & (second >= 0)
        & (first < count) & (second < count)
    )
    boundary = valid & (refined[first] != refined[second])
    rows = np.flatnonzero(boundary).astype(np.int32, copy=False)
    if not len(rows):
        return done(
            "no-boundary", refined,
            shadow_screen_pre_shadow_seed_connected=bool(
                0 <= int(seed_face) < count and refined[int(seed_face)]
            ),
        )
    field = np.asarray(buffers.get("denoised", ()), dtype=np.float32)
    cx = np.asarray(buffers.get("center_x", ()), dtype=np.int64).reshape(-1)
    cy = np.asarray(buffers.get("center_y", ()), dtype=np.int64).reshape(-1)
    if field.ndim != 2 or len(cx) != count or len(cy) != count:
        return done("buffer-schema", refined)
    h, w = field.shape

    def face_luminance(face):
        face = int(face)
        if not (0 <= face < count):
            return None
        x, y = int(cx[face]), int(cy[face])
        if not (0 <= x < w and 0 <= y < h):
            return None
        lo_x, hi_x = max(0, x - 1), min(w, x + 2)
        lo_y, hi_y = max(0, y - 1), min(h, y + 2)
        values = np.asarray(field[lo_y:hi_y, lo_x:hi_x], dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if len(values) else None

    seed_value = face_luminance(seed_face)
    finite = field[np.isfinite(field)]
    if seed_value is None or not len(finite):
        return done("no-luminance", refined)
    cache = state.get("shadow_screen_classifier_cache")
    bright = bool(cache.get("bright")) if isinstance(cache, dict) else bool(
        seed_value >= float(np.median(finite))
    )
    seed_mad = float(cache.get("tolerance", 0.0)) if isinstance(cache, dict) else 0.0
    seed_patch = field[
        max(0, int(cy[int(seed_face)]) - 1):min(h, int(cy[int(seed_face)]) + 2),
        max(0, int(cx[int(seed_face)]) - 1):min(w, int(cx[int(seed_face)]) + 2),
    ]
    seed_patch = seed_patch[np.isfinite(seed_patch)]
    if len(seed_patch):
        patch_median = float(np.median(seed_patch))
        patch_mad = float(1.4826 * np.median(np.abs(seed_patch - patch_median)))
    else:
        patch_median, patch_mad = seed_value, 0.0
    seed_value = patch_median
    tolerance = max(seed_mad, 3.0 * patch_mad, 0.02 * max(abs(seed_value), 1.0e-2), 1.0e-3)
    span = max(float(np.percentile(finite, 90.0) - np.percentile(finite, 10.0)), tolerance)
    # The direction is deliberately luminance-only.  The seed side is the
    # bright plateau for a bright seed and the dark plateau for a dark seed.
    orientation = "bright-to-dark" if bright else "dark-to-bright"

    offsets = np.asarray(geometry.get("offsets", ()), dtype=np.int64).reshape(-1)
    neighbors = np.asarray(geometry.get("neighbors", ()), dtype=np.int32).reshape(-1)
    edge_indices = np.asarray(geometry.get("edge_indices", ()), dtype=np.int32).reshape(-1)
    if len(offsets) != count + 1 or int(offsets[-1]) != len(neighbors):
        return done("graph-schema", refined, shadow_screen_pre_shadow_orientation=orientation)
    pair_lookup = {}
    for directed, pair in enumerate(edge_indices if len(edge_indices) == len(neighbors) else ()):
        left, right = int(first[int(pair)]), int(second[int(pair)])
        pair_lookup.setdefault((min(left, right), max(left, right)), int(pair))

    # One face ring is sufficient for the preferred pre-shadow edge.  A row
    # is eligible only if its accepted side is transition-contaminated while
    # one accepted neighbour remains plateau-like.  This prevents blind
    # nearest-edge shifts and leaves weak/top/bottom segments untouched.
    proposals = []
    failures = 0
    accepted_deviations = []
    rejected_drops = []
    for row in rows:
        left, right = int(first[row]), int(second[row])
        accepted = left if refined[left] else right
        rejected = right if accepted == left else left
        ia, ib = face_luminance(accepted), face_luminance(rejected)
        if ia is None or ib is None:
            failures += 1
            continue
        if bright:
            directed_drop = ia - ib
            accepted_deviation = max(0.0, seed_value - ia)
            rejected_drop = max(0.0, seed_value - ib)
        else:
            directed_drop = ib - ia
            accepted_deviation = max(0.0, ia - seed_value)
            rejected_drop = max(0.0, ib - seed_value)
        accepted_deviations.append(accepted_deviation)
        rejected_drops.append(rejected_drop)
        # Require an actual directed onset.  A gentle gradient or a random
        # low-amplitude edge cannot qualify merely because local MAD is zero.
        drop_floor = max(0.75 * tolerance, 0.025 * span)
        if directed_drop <= drop_floor or rejected_drop <= max(tolerance, 0.025 * span):
            continue
        accepted_neighbors = []
        start, stop = int(offsets[accepted]), int(offsets[accepted + 1])
        for neighbor in neighbors[start:stop]:
            neighbor = int(neighbor)
            if not (0 <= neighbor < count) or not refined[neighbor]:
                continue
            value = face_luminance(neighbor)
            if value is None:
                continue
            if bright:
                deviation = max(0.0, seed_value - value)
            else:
                deviation = max(0.0, value - seed_value)
            accepted_neighbors.append((deviation, neighbor, value))
        if not accepted_neighbors:
            continue
        accepted_neighbors.sort(key=lambda item: (item[0], item[1]))
        plateau_deviation, predecessor, _predecessor_value = accepted_neighbors[0]
        # If the currently accepted face is already plateau-like, this is the
        # correct pre-shadow chain and must remain unchanged.
        if accepted_deviation <= max(1.25 * tolerance, 0.02 * span):
            continue
        if plateau_deviation > max(1.25 * tolerance, 0.02 * span):
            continue
        proposals.append({
            "row": int(row),
            "accepted": int(accepted),
            "rejected": int(rejected),
            "predecessor": int(predecessor),
            "drop": float(directed_drop),
            "accepted_deviation": float(accepted_deviation),
            "rejected_drop": float(rejected_drop),
        })

    if not proposals:
        return done(
            "no-valid-direction", refined,
            shadow_screen_pre_shadow_orientation=orientation,
            shadow_screen_pre_shadow_no_valid_direction=int(len(rows) + failures),
            shadow_screen_pre_shadow_unchanged_segments=int(len(rows)),
        )

    # Connect rows by accepted-side topology first, then by the projected
    # physical contact geometry.  Adjacent rows on a straight boundary often
    # do not share a face (each edge separates a different pair of rows), so
    # requiring a common face would fragment the very chain we want.  The
    # projected midpoint test remains bounded and rejects parallel chains by
    # requiring progression perpendicular to the accepted/rejected contact.
    by_face = {}
    for index, proposal in enumerate(proposals):
        for face in (proposal["accepted"], proposal["predecessor"]):
            by_face.setdefault(face, []).append(index)
    midpoint_data = {}
    centers = np.asarray(geometry.get("centers", ()), dtype=np.float64)
    capture = state.get("shadow_capture")
    projection = np.asarray(
        capture.get("view_projection") if isinstance(capture, dict) else (),
        dtype=np.float64,
    )
    full_width = int(metrics.get("shadow_screen_width", 0))
    full_height = int(metrics.get("shadow_screen_height", 0))

    def project(point):
        if projection.shape != (4, 4):
            return None
        clip = projection @ np.r_[np.asarray(point, dtype=np.float64), 1.0]
        if not np.all(np.isfinite(clip)) or abs(float(clip[3])) <= 1.0e-12:
            return None
        ndc = clip[:3] / float(clip[3])
        px = (float(ndc[0]) * 0.5 + 0.5) * max(1.0, full_width - 1.0)
        py = (float(ndc[1]) * 0.5 + 0.5) * max(1.0, full_height - 1.0)
        return np.asarray((px, py), dtype=np.float64)

    if centers.shape == (count, 3):
        for index, proposal in enumerate(proposals):
            a = project(centers[proposal["accepted"]])
            b = project(centers[proposal["rejected"]])
            if a is not None and b is not None:
                edge = b - a
                length = float(np.linalg.norm(edge))
                if np.isfinite(length) and length > 1.0e-6:
                    midpoint_data[index] = (
                        (a + b) * 0.5,
                        edge / length,
                    )
    nearest = []
    if len(midpoint_data) >= 2:
        points = tuple(midpoint_data[index][0] for index in sorted(midpoint_data))
        for index in sorted(midpoint_data):
            point = midpoint_data[index][0]
            distances_to_other = [
                float(np.linalg.norm(point - other))
                for other_index, other in midpoint_data.items()
                if other_index != index
            ]
            if distances_to_other:
                nearest.append(min(distances_to_other))
    typical_spacing = float(np.median(nearest)) if nearest else 0.0
    factor = max(1, int(metrics.get("shadow_screen_downsample_factor", 1)))
    link_distance = max(2.25 * float(factor), 1.8 * typical_spacing)
    unseen = set(range(len(proposals)))
    components = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        component = [root]
        pending = [root]
        while pending:
            index = pending.pop()
            proposal = proposals[index]
            for face in (proposal["accepted"], proposal["predecessor"]):
                for other in by_face.get(face, ()):
                    if other in unseen:
                        unseen.remove(other)
                        pending.append(other)
                        component.append(other)
            if index in midpoint_data:
                point, direction = midpoint_data[index]
                for other in tuple(unseen):
                    if other not in midpoint_data:
                        continue
                    other_point, other_direction = midpoint_data[other]
                    delta = other_point - point
                    distance = float(np.linalg.norm(delta))
                    if not np.isfinite(distance) or distance > link_distance or distance <= 1.0e-6:
                        continue
                    delta_direction = delta / distance
                    # Same oriented edge family and a step along the edge
                    # chain, not across a parallel shadow line.
                    if abs(float(np.dot(direction, other_direction))) < 0.5:
                        continue
                    if abs(float(np.dot(delta_direction, direction))) > 0.75:
                        continue
                    unseen.remove(other)
                    pending.append(other)
                    component.append(other)
        components.append(tuple(sorted(component)))
    coherent = [component for component in components if len(component) >= 3]
    if not coherent:
        return done(
            "no-coherent-chain", refined,
            shadow_screen_pre_shadow_orientation=orientation,
            shadow_screen_pre_shadow_no_valid_direction=int(len(proposals)),
            shadow_screen_pre_shadow_unchanged_segments=int(len(rows)),
        )

    remove_faces = set()
    for component in coherent:
        for index in component:
            remove_faces.add(int(proposals[index]["accepted"]))
    remove_faces.discard(int(seed_face))
    if not remove_faces:
        return done(
            "seed-protected", refined,
            shadow_screen_pre_shadow_orientation=orientation,
            shadow_screen_pre_shadow_oriented_chain_count=int(len(coherent)),
            shadow_screen_pre_shadow_coherent_chain_count=int(len(coherent)),
            shadow_screen_pre_shadow_unchanged_segments=int(len(rows)),
        )
    trial = refined.copy()
    trial[np.asarray(sorted(remove_faces), dtype=np.int32)] = False
    if not (0 <= int(seed_face) < count and trial[int(seed_face)]):
        return done("seed-protected", refined, shadow_screen_pre_shadow_orientation=orientation)
    # Keep the accepted component connected.  The topology check is safety
    # only; luminance remains the sole orientation and shift criterion.
    connected = np.zeros(count, dtype=bool)
    connected[int(seed_face)] = True
    q = deque((int(seed_face),))
    while q:
        face = int(q.popleft())
        for neighbor in neighbors[int(offsets[face]):int(offsets[face + 1])]:
            neighbor = int(neighbor)
            if 0 <= neighbor < count and trial[neighbor] and not connected[neighbor]:
                connected[neighbor] = True
                q.append(neighbor)
    if not np.all(connected[trial]):
        return done(
            "seed-disconnected", refined,
            shadow_screen_pre_shadow_orientation=orientation,
            shadow_screen_pre_shadow_oriented_chain_count=int(len(coherent)),
            shadow_screen_pre_shadow_coherent_chain_count=int(len(coherent)),
        )
    return done(
        "applied",
        trial,
        shadow_screen_pre_shadow_orientation=orientation,
        shadow_screen_pre_shadow_oriented_chain_count=int(len(coherent)),
        shadow_screen_pre_shadow_coherent_chain_count=int(len(coherent)),
        shadow_screen_pre_shadow_shift_count=int(len(remove_faces)),
        shadow_screen_pre_shadow_ring_distance=1,
        shadow_screen_pre_shadow_accepted_deviation=float(np.median(accepted_deviations)) if accepted_deviations else 0.0,
        shadow_screen_pre_shadow_rejected_drop=float(np.median(rejected_drops)) if rejected_drops else 0.0,
        shadow_screen_pre_shadow_unchanged_segments=int(max(0, len(rows) - len(proposals))),
        shadow_screen_pre_shadow_no_valid_direction=int(failures),
        shadow_screen_pre_shadow_seed_connected=True,
    )


def _fill_preview_shadow_refine_boundary(
    state, geometry, buffers, metrics, final, seed_face
):
    """Refine a screen-classifier interface inside a bounded face band.

    The ROI flood is the authoritative coarse result.  This pass is allowed
    to move that interface by at most two adjacency rings, and it reads only
    the immutable analysis luminance/gradient buffers.  It deliberately does
    not inspect normals, curvature, Face Sets, distances, or any geometry
    score: mesh topology is used only to enumerate the local edge band and to
    keep the accepted component connected.
    """
    import numpy as np
    from collections import deque

    def empty_result(reason):
        return np.asarray(final, dtype=bool).copy(), {
            "shadow_screen_refine_reason": str(reason),
            "shadow_screen_boundary_before": 0,
            "shadow_screen_boundary_after": 0,
            "shadow_screen_refine_scored_edges": 0,
            "shadow_screen_refine_score_min": 0.0,
            "shadow_screen_refine_score_median": 0.0,
            "shadow_screen_refine_score_max": 0.0,
            "shadow_screen_refine_score_threshold": 0.0,
            "shadow_screen_refine_moved_faces": 0,
            "shadow_screen_refine_max_ring_shift": 0,
            "shadow_screen_refine_sample_failures": 0,
            "shadow_screen_refine_chain_count": 0,
            "shadow_screen_refine_spikes_removed": 0,
            "shadow_screen_refine_seed_connected": False,
        }

    candidate = np.asarray(final, dtype=bool).reshape(-1).copy()
    count = int(geometry.get("count", len(candidate)))
    if len(candidate) != count or count <= 0:
        return empty_result("schema")
    first = np.asarray(geometry.get("first", ()), dtype=np.int32).reshape(-1)
    second = np.asarray(geometry.get("second", ()), dtype=np.int32).reshape(-1)
    if len(first) != len(second):
        return empty_result("edge-schema")
    boundary = (first >= 0) & (second >= 0) & (first < count) & (second < count)
    boundary &= candidate[first] != candidate[second]
    boundary_rows = np.flatnonzero(boundary).astype(np.int32, copy=False)
    base_metrics = {
        "shadow_screen_refine_reason": "not-run",
        "shadow_screen_boundary_before": int(len(boundary_rows)),
        "shadow_screen_boundary_after": int(len(boundary_rows)),
        "shadow_screen_refine_scored_edges": 0,
        "shadow_screen_refine_score_min": 0.0,
        "shadow_screen_refine_score_median": 0.0,
        "shadow_screen_refine_score_max": 0.0,
        "shadow_screen_refine_score_threshold": 0.0,
        "shadow_screen_refine_moved_faces": 0,
        "shadow_screen_refine_max_ring_shift": 0,
        "shadow_screen_refine_sample_failures": 0,
        "shadow_screen_refine_chain_count": 0,
        "shadow_screen_refine_spikes_removed": 0,
        "shadow_screen_refine_seed_connected": False,
        "shadow_screen_pre_shadow_oriented_chain_count": 0,
        "shadow_screen_pre_shadow_orientation": "unknown",
        "shadow_screen_pre_shadow_shift_count": 0,
        "shadow_screen_pre_shadow_ring_distance": 0,
        "shadow_screen_pre_shadow_accepted_deviation": 0.0,
        "shadow_screen_pre_shadow_rejected_drop": 0.0,
        "shadow_screen_pre_shadow_unchanged_segments": 0,
        "shadow_screen_pre_shadow_no_valid_direction": 0,
        "shadow_screen_pre_shadow_coherent_chain_count": 0,
        "shadow_screen_pre_shadow_seed_connected": False,
        "shadow_screen_pre_shadow_reason": "not-run",
    }
    if not len(boundary_rows):
        base_metrics["shadow_screen_refine_reason"] = "no-boundary"
        base_metrics["shadow_screen_refine_seed_connected"] = bool(
            0 <= int(seed_face) < count and candidate[int(seed_face)]
        )
        return candidate, base_metrics
    offsets = np.asarray(geometry.get("offsets", ()), dtype=np.int64).reshape(-1)
    neighbors = np.asarray(geometry.get("neighbors", ()), dtype=np.int32).reshape(-1)
    edge_indices = np.asarray(geometry.get("edge_indices", ()), dtype=np.int32).reshape(-1)
    if (
        len(offsets) != count + 1
        or int(offsets[-1]) != len(neighbors)
        or (len(edge_indices) not in (0, len(neighbors)))
    ):
        return candidate, {**base_metrics, "shadow_screen_refine_reason": "graph-schema"}

    # Build the bounded one/two-ring face band without traversing unrelated
    # mesh faces.  The CSR graph is already the cached physical adjacency.
    ring = np.full(count, -1, dtype=np.int8)
    band_faces = set()
    queue = deque()
    for row in boundary_rows:
        for face in (int(first[row]), int(second[row])):
            if ring[face] < 0:
                ring[face] = 0
                band_faces.add(face)
                queue.append(face)
    while queue:
        face = int(queue.popleft())
        depth = int(ring[face])
        if depth >= 2:
            continue
        start, stop = int(offsets[face]), int(offsets[face + 1])
        for neighbor in neighbors[start:stop]:
            neighbor = int(neighbor)
            if 0 <= neighbor < count and ring[neighbor] < 0:
                ring[neighbor] = depth + 1
                band_faces.add(neighbor)
                queue.append(neighbor)
    band_mask = np.zeros(count, dtype=bool)
    if band_faces:
        band_mask[np.asarray(sorted(band_faces), dtype=np.int32)] = True

    # Gather physical edge rows incident to the local face band.  New full
    # graphs expose edge_indices in the directed CSR; the fallback is kept
    # for old/synthetic geometry and is still limited to the band predicate.
    edge_rows = set()
    for face in sorted(band_faces):
        start, stop = int(offsets[face]), int(offsets[face + 1])
        if len(edge_indices) == len(neighbors):
            for directed in range(start, stop):
                pair = int(edge_indices[directed])
                if 0 <= pair < len(first):
                    edge_rows.add(pair)
    if not edge_rows:
        edge_rows.update(
            int(row)
            for row in np.flatnonzero(
                (first >= 0) & (second >= 0)
                & (first < count) & (second < count)
                & (band_mask[first] | band_mask[second])
            )
        )
    if not edge_rows:
        return candidate, {**base_metrics, "shadow_screen_refine_reason": "band-empty"}

    capture = state.get("shadow_capture")
    projection = np.asarray(
        capture.get("view_projection") if isinstance(capture, dict) else (),
        dtype=np.float64,
    )
    denoised = np.asarray(buffers.get("denoised", ()), dtype=np.float32)
    gradient = np.asarray(buffers.get("gradient", ()), dtype=np.float32)
    if (
        projection.shape != (4, 4)
        or denoised.ndim != 2
        or gradient.shape != denoised.shape
    ):
        return candidate, {**base_metrics, "shadow_screen_refine_reason": "buffer-schema"}
    analysis_h, analysis_w = denoised.shape
    origin = metrics.get("shadow_screen_roi_origin", (0, 0))
    factor = max(1, int(metrics.get("shadow_screen_downsample_factor", 1)))
    full_width = int(metrics.get("shadow_screen_width", 0))
    full_height = int(metrics.get("shadow_screen_height", 0))
    if full_width < 2 or full_height < 2:
        return candidate, {**base_metrics, "shadow_screen_refine_reason": "capture-size"}
    try:
        origin_x, origin_y = float(origin[0]), float(origin[1])
    except (IndexError, TypeError, ValueError):
        return candidate, {**base_metrics, "shadow_screen_refine_reason": "roi-origin"}

    pair_v0 = np.asarray(geometry.get("pair_v0", ()), dtype=np.int32).reshape(-1)
    pair_v1 = np.asarray(geometry.get("pair_v1", ()), dtype=np.int32).reshape(-1)
    centers = np.asarray(geometry.get("centers", ()), dtype=np.float64)
    vertices = np.asarray(geometry.get("world_vertices", ()), dtype=np.float64)
    vertex_space = geometry.get("vertex_id_space")
    cache_key = (
        int(state.get("shadow_capture_generation", 0)),
        buffers.get("key"),
    )
    refine_cache = state.get("shadow_screen_boundary_refine_cache")
    if not isinstance(refine_cache, dict) or refine_cache.get("key") != cache_key:
        refine_cache = {"key": cache_key, "scores": {}}
        state["shadow_screen_boundary_refine_cache"] = refine_cache
    score_cache = refine_cache["scores"]
    score_by_pair = {}
    failed = 0

    def project(point):
        clip = projection @ np.r_[np.asarray(point, dtype=np.float64), 1.0]
        if not np.all(np.isfinite(clip)) or abs(float(clip[3])) <= 1.0e-12:
            return None
        ndc = clip[:3] / float(clip[3])
        x = (float(ndc[0]) * 0.5 + 0.5) * (full_width - 1.0)
        y = (float(ndc[1]) * 0.5 + 0.5) * (full_height - 1.0)
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return np.asarray((x, y), dtype=np.float64)

    def sample(x, y, field):
        # A 3x3 median on the immutable reduced grid makes the boundary score
        # insensitive to one-pixel shader noise while retaining a thin line.
        cx, cy = int(round(float((x - origin_x) / factor))), int(
            round(float((y - origin_y) / factor))
        )
        if not (0 <= cx < analysis_w and 0 <= cy < analysis_h):
            return None
        xlo, xhi = max(0, cx - 1), min(analysis_w, cx + 2)
        ylo, yhi = max(0, cy - 1), min(analysis_h, cy + 2)
        values = np.asarray(field[ylo:yhi, xlo:xhi], dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if len(values) else None

    # The physical edge midpoint/side samples are intentionally cached by
    # pair row for all wheel stages in one capture generation.
    for pair in sorted(edge_rows):
        if pair in score_cache:
            value = score_cache[pair]
            if isinstance(value, dict) and bool(value.get("ok")):
                score_by_pair[pair] = float(value.get("score", 0.0))
            else:
                failed += 1
            continue
        left, right = int(first[pair]), int(second[pair])
        if not (0 <= left < count and 0 <= right < count):
            score_cache[pair] = {"ok": False, "reason": "face-schema"}
            failed += 1
            continue
        a = b = None
        if (
            len(pair_v0) == len(first) == len(pair_v1)
            and int(pair_v0[pair]) >= 0
            and int(pair_v1[pair]) >= 0
            and len(vertices) > max(int(pair_v0[pair]), int(pair_v1[pair]))
        ):
            a = vertices[int(pair_v0[pair])]
            b = vertices[int(pair_v1[pair])]
        elif centers.shape == (count, 3):
            # Synthetic/legacy graphs without endpoints use the two face
            # centers as a safe projected contact approximation.
            a, b = centers[left], centers[right]
        pa, pb = project(a), project(b)
        if pa is None or pb is None:
            score_cache[pair] = {"ok": False, "reason": "projection"}
            failed += 1
            continue
        edge_vec = pb - pa
        edge_len = float(np.linalg.norm(edge_vec))
        if not np.isfinite(edge_len) or edge_len <= 0.5:
            score_cache[pair] = {"ok": False, "reason": "degenerate"}
            failed += 1
            continue
        midpoint = (pa + pb) * 0.5
        perp = np.asarray((-edge_vec[1], edge_vec[0]), dtype=np.float64) / edge_len
        offset = max(0.75, min(2.5, 0.75 * float(factor)))
        plus = sample(*(midpoint + perp * offset), denoised)
        minus = sample(*(midpoint - perp * offset), denoised)
        gmid = sample(float(midpoint[0]), float(midpoint[1]), gradient)
        if plus is None or minus is None or gmid is None:
            score_cache[pair] = {"ok": False, "reason": "side-sample"}
            failed += 1
            continue
        value = float(abs(plus - minus) + 0.5 * max(0.0, gmid))
        score_cache[pair] = {"ok": True, "score": value}
        score_by_pair[pair] = value
    scores = np.asarray(tuple(score_by_pair.values()), dtype=np.float64)
    base_metrics["shadow_screen_refine_scored_edges"] = int(len(scores))
    base_metrics["shadow_screen_refine_sample_failures"] = int(failed)
    if not len(scores):
        base_metrics["shadow_screen_refine_reason"] = "no-edge-samples"
        return candidate, base_metrics
    score_median = float(np.median(scores))
    score_mad = float(1.4826 * np.median(np.abs(scores - score_median)))
    score_min, score_max = float(np.min(scores)), float(np.max(scores))
    span = max(score_max - score_min, 1.0e-8)
    threshold = max(score_median + max(1.5 * score_mad, 0.10 * span), score_min + 0.25 * span)
    high_pairs = {
        int(pair) for pair, value in score_by_pair.items()
        if float(value) >= threshold and float(value) > score_min + 1.0e-8
    }
    base_metrics.update({
        "shadow_screen_refine_score_min": score_min,
        "shadow_screen_refine_score_median": score_median,
        "shadow_screen_refine_score_max": score_max,
        "shadow_screen_refine_score_threshold": float(threshold),
    })

    # Grow only from the existing accepted side into the two-ring band.  A
    # strong sampled edge is a true barrier, while unscored edges are closed
    # conservatively.  This moves a staircase toward a continuous high-score
    # chain without touching faces outside the band.
    refined = candidate.copy()
    local_sources = np.flatnonzero(candidate & band_mask).astype(np.int32, copy=False)
    grow = deque(int(value) for value in local_sources)
    pair_lookup = {}
    for pair in edge_rows:
        pair_lookup[(min(int(first[pair]), int(second[pair])), max(int(first[pair]), int(second[pair])))] = int(pair)
    while grow:
        face = int(grow.popleft())
        for directed in range(int(offsets[face]), int(offsets[face + 1])):
            neighbor = int(neighbors[directed])
            if not (0 <= neighbor < count) or not band_mask[neighbor] or refined[neighbor]:
                continue
            if len(edge_indices) == len(neighbors):
                pair = int(edge_indices[directed])
            else:
                pair = pair_lookup.get((min(face, neighbor), max(face, neighbor)), -1)
            if pair < 0 or pair not in score_by_pair or pair in high_pairs:
                continue
            refined[neighbor] = True
            grow.append(neighbor)

    # Bounded opening: remove only a genuine one-face accepted leaf.  This is
    # the local counterpart to Closing above and cannot disconnect the seed
    # component because a leaf has at most one original accepted neighbor.
    spikes_removed = 0
    for face in np.flatnonzero(candidate & band_mask):
        face = int(face)
        if face == int(seed_face):
            continue
        start, stop = int(offsets[face]), int(offsets[face + 1])
        adjacent = [int(value) for value in neighbors[start:stop] if 0 <= int(value) < count]
        accepted_neighbors = sum(bool(candidate[value]) for value in adjacent)
        outside_neighbors = sum(not bool(candidate[value]) for value in adjacent)
        if accepted_neighbors > 1 or outside_neighbors < 2:
            continue
        incident = []
        for directed in range(start, stop):
            neighbor = int(neighbors[directed])
            pair = (
                int(edge_indices[directed]) if len(edge_indices) == len(neighbors)
                else pair_lookup.get((min(face, neighbor), max(face, neighbor)), -1)
            )
            if pair in score_by_pair:
                incident.append(float(score_by_pair[pair]))
        if incident and max(incident) >= threshold:
            continue
        refined[face] = False
        spikes_removed += 1

    # Seed connectivity is checked on the refined candidate itself.  The
    # graph walk is bounded by the existing candidate plus the two-ring band.
    seed_connected = False
    if 0 <= int(seed_face) < count and refined[int(seed_face)]:
        connected = np.zeros(count, dtype=bool)
        connected[int(seed_face)] = True
        q = deque((int(seed_face),))
        while q:
            face = int(q.popleft())
            for neighbor in neighbors[int(offsets[face]):int(offsets[face + 1])]:
                neighbor = int(neighbor)
                if 0 <= neighbor < count and refined[neighbor] and not connected[neighbor]:
                    connected[neighbor] = True
                    q.append(neighbor)
        # Only the original accepted component must remain connected.  Newly
        # added local faces are allowed to form the snapped boundary component.
        seed_connected = bool(np.all(connected[candidate]))
        if not seed_connected:
            refined = candidate.copy()
            spikes_removed = 0
    after_boundary = (first >= 0) & (second >= 0) & (first < count) & (second < count)
    after_boundary &= refined[first] != refined[second]
    changed = np.flatnonzero(refined != candidate)
    shifts = ring[changed]
    base_metrics.update({
        "shadow_screen_refine_boundary_after": int(np.count_nonzero(after_boundary)),
        "shadow_screen_refine_moved_faces": int(len(changed)),
        "shadow_screen_refine_max_ring_shift": int(np.max(shifts)) if len(shifts) else 0,
        "shadow_screen_refine_spikes_removed": int(spikes_removed),
        "shadow_screen_refine_seed_connected": bool(seed_connected),
        "shadow_screen_refine_chain_count": 0,
        "shadow_screen_refine_reason": "applied" if len(changed) else "no-interface-move",
    })
    # Count high-score chains through shared faces.  This is a bounded
    # diagnostic; it never decides membership by itself.
    if high_pairs:
        by_face = {}
        for pair in high_pairs:
            by_face.setdefault(int(first[pair]), set()).add(pair)
            by_face.setdefault(int(second[pair]), set()).add(pair)
        unseen = set(high_pairs)
        chains = 0
        while unseen:
            start = min(unseen)
            unseen.remove(start)
            pending = [start]
            chains += 1
            while pending:
                pair = pending.pop()
                for face in (int(first[pair]), int(second[pair])):
                    for other in by_face.get(face, ()):
                        if other in unseen:
                            unseen.remove(other)
                            pending.append(other)
        base_metrics["shadow_screen_refine_chain_count"] = int(chains)
    # The generic bounded pass above removes isolated teeth and can move a
    # staircase toward a strong chain.  Apply the luminance-directed
    # pre-shadow choice last so a boundary that sits inside the transition is
    # pulled back to the seed/plateau side.  The caller still enforces wheel
    # monotonicity for expand/shrink stages.
    try:
        pre_shadow, pre_metrics = _fill_preview_shadow_pre_shadow_snap(
            state, geometry, buffers, metrics, refined, seed_face
        )
        refined = pre_shadow
        base_metrics.update(pre_metrics)
        if int(pre_metrics.get("shadow_screen_pre_shadow_shift_count", 0)):
            base_metrics["shadow_screen_refine_moved_faces"] = int(
                base_metrics.get("shadow_screen_refine_moved_faces", 0)
            ) + int(pre_metrics["shadow_screen_pre_shadow_shift_count"])
            post_boundary = (
                (first >= 0) & (second >= 0)
                & (first < count) & (second < count)
                & (refined[first] != refined[second])
            )
            base_metrics["shadow_screen_refine_boundary_after"] = int(
                np.count_nonzero(post_boundary)
            )
            base_metrics["shadow_screen_refine_reason"] = "applied-pre-shadow"
    except (
        AttributeError,
        IndexError,
        KeyError,
        MemoryError,
        RuntimeError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        base_metrics.update({
            "shadow_screen_pre_shadow_reason": "exception",
            "shadow_screen_pre_shadow_no_valid_direction": int(len(boundary_rows)),
        })
    return refined, base_metrics


def _fill_preview_shadow_screen_region(state, geometry, seed_face, step=0):
    """Classify ordinary-E from one immutable screen-space grayscale image.

    The analysis capture is deliberately separate from the geometry-strict
    path.  A temporary int32 center-coverage raster maps visible face centers
    to local face indices; the graph is used only for the final seed-connected
    safety check and for emitting the exact accepted/rejected interface.  No
    distance, radius, Face Set, normal, or curvature score participates in
    this classifier.
    """
    return _fill_preview_shadow_screen_region_roi(
        state, geometry, int(seed_face), int(step)
    )
    import numpy as np

    count = int(geometry.get("count", 0))
    capture = state.get("shadow_capture")
    metrics = {
        "shadow_screen_classifier_mode": "screen-grayscale-morphology",
        "shadow_screen_fallback_reason": "capture-unavailable",
        "shadow_screen_width": 0,
        "shadow_screen_height": 0,
        "shadow_screen_luminance_min": 0.0,
        "shadow_screen_luminance_max": 0.0,
        "shadow_screen_seed_luminance": 0.0,
        "shadow_screen_raw_luminance_min": 0.0,
        "shadow_screen_raw_luminance_max": 0.0,
        "shadow_screen_raw_luminance_quantiles": (),
        "shadow_screen_sharpened_luminance_min": 0.0,
        "shadow_screen_sharpened_luminance_max": 0.0,
        "shadow_screen_sharpened_luminance_quantiles": (),
        "shadow_screen_sharpened_field_used": False,
        "shadow_screen_sharpen_knee": 0.0,
        "shadow_screen_sharpen_steepness": 0.0,
        "shadow_screen_sharpen_curve_scale": 0.0,
        "shadow_screen_transition_width_before_pixels": 0,
        "shadow_screen_transition_width_after_pixels": 0,
        "shadow_screen_transition_face_rows_before": 0,
        "shadow_screen_transition_face_rows_after": 0,
        "shadow_screen_sharpened_overlay_active": False,
        "shadow_screen_sharpened_overlay_alpha": 0.0,
        "shadow_screen_sharpened_overlay_restored": False,
        "shadow_screen_denoise_kernel": 3,
        "shadow_screen_gradient_kernel": 3,
        "shadow_screen_closing_kernel": 3,
        "shadow_screen_strong_threshold": 0.0,
        "shadow_screen_weak_threshold": 0.0,
        "shadow_screen_threshold": 0.0,
        "shadow_screen_base_threshold": 0.0,
        "shadow_screen_step": int(step),
        "shadow_screen_mask_pixel_count": 0,
        "shadow_screen_face_pixel_count": 0,
        "shadow_screen_face_coverage_min": 0.0,
        "shadow_screen_face_coverage_max": 0.0,
        "shadow_screen_accepted_count": 0,
        "shadow_screen_accepted_luminance_min": 0.0,
        "shadow_screen_accepted_luminance_max": 0.0,
        "shadow_screen_rejected_neighbor_count": 0,
        "shadow_screen_boundary_count": 0,
        "shadow_screen_raw_gradient_min": 0.0,
        "shadow_screen_raw_gradient_median": 0.0,
        "shadow_screen_raw_gradient_max": 0.0,
        "shadow_screen_gradient_mad": 0.0,
        "shadow_screen_closing_filled_pixels": 0,
        "shadow_screen_top_hat_used": False,
        "shadow_screen_black_hat_used": False,
        "shadow_screen_monotonic_action": "initial",
        "shadow_screen_initial_count": 0,
        "shadow_screen_previous_count": 0,
        "shadow_screen_preserved_count": 0,
        "shadow_screen_face_set_independent": True,
        "shadow_screen_id_raster_mode": "int32-face-center-depth-order",
        "shadow_screen_occlusion_depth_used": False,
    }
    state["shadow_luminance_region_edges"] = set()
    if count <= 0 or not isinstance(capture, dict) or not capture.get("ok"):
        metrics["shadow_screen_fallback_reason"] = "capture-unavailable"
        return np.asarray([int(seed_face)], dtype=np.int32), metrics
    try:
        image = np.asarray(capture.get("luminance", ()), dtype=np.float32)
        if image.ndim != 2 or image.shape[0] < 8 or image.shape[1] < 8:
            raise ValueError("screen-image-schema")
        height, width = (int(image.shape[0]), int(image.shape[1]))
        metrics["shadow_screen_width"] = width
        metrics["shadow_screen_height"] = height
        finite = np.isfinite(image)
        if not np.any(finite):
            raise ValueError("screen-image-empty")
        image = np.where(finite, image, float(np.nanmedian(image))).astype(
            np.float32, copy=False
        )
        metrics["shadow_screen_luminance_min"] = float(np.min(image))
        metrics["shadow_screen_luminance_max"] = float(np.max(image))

        # Small fixed median removes one-pixel shader/readback noise while
        # retaining the visible toon-dark band.  The morphology is image
        # based and independent of candidate size or wheel radius.
        padded = np.pad(image, 1, mode="edge")
        samples = np.stack(
            tuple(
                padded[dy:dy + height, dx:dx + width]
                for dy in range(3) for dx in range(3)
            ),
            axis=0,
        )
        denoised = np.median(samples, axis=0).astype(np.float32, copy=False)
        dilated = np.max(samples, axis=0)
        eroded = np.min(samples, axis=0)
        gradient = np.maximum(dilated - eroded, 0.0)
        finite_gradient = gradient[np.isfinite(gradient)]
        gradient_median = float(np.median(finite_gradient)) if len(finite_gradient) else 0.0
        gradient_mad = float(
            1.4826 * np.median(np.abs(finite_gradient - gradient_median))
        ) if len(finite_gradient) else 0.0
        metrics["shadow_screen_raw_gradient_min"] = float(np.min(gradient))
        metrics["shadow_screen_raw_gradient_median"] = gradient_median
        metrics["shadow_screen_raw_gradient_max"] = float(np.max(gradient))
        metrics["shadow_screen_gradient_mad"] = gradient_mad
        gradient_span = max(float(np.max(denoised) - np.min(denoised)), 1.0e-3)
        strong_threshold = max(
            gradient_median + 3.0 * max(gradient_mad, 1.0e-4),
            gradient_span * 0.04,
        )
        weak_threshold = max(
            gradient_median + 1.5 * max(gradient_mad, 1.0e-4),
            strong_threshold * 0.5,
        )
        metrics["shadow_screen_strong_threshold"] = float(strong_threshold)
        metrics["shadow_screen_weak_threshold"] = float(weak_threshold)
        strong = gradient >= strong_threshold
        weak = gradient >= weak_threshold
        # Hysteresis: only weak pixels connected to a strong seed are barriers.
        hysteresis = np.zeros_like(strong, dtype=bool)
        strong_rows, strong_cols = np.nonzero(strong)
        frontier = [(int(y), int(x)) for y, x in zip(strong_rows, strong_cols)]
        hysteresis[strong] = True
        while frontier:
            y, x = frontier.pop()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and weak[ny, nx] and not hysteresis[ny, nx]:
                    hysteresis[ny, nx] = True
                    frontier.append((ny, nx))
        # Closing bridges a one/two pixel gap in one continuous line.  It is
        # applied to the scalar-derived barrier, never to selected faces.
        barrier_pad = np.pad(hysteresis, 1, mode="constant")
        barrier_samples = np.stack(
            tuple(
                barrier_pad[dy:dy + height, dx:dx + width]
                for dy in range(3) for dx in range(3)
            ),
            axis=0,
        )
        dilated_barrier = np.max(barrier_samples, axis=0)
        closed_pad = np.pad(dilated_barrier, 1, mode="constant")
        closed_samples = np.stack(
            tuple(
                closed_pad[dy:dy + height, dx:dx + width]
                for dy in range(3) for dx in range(3)
            ),
            axis=0,
        )
        closed_barrier = np.min(closed_samples, axis=0).astype(bool)
        metrics["shadow_screen_closing_filled_pixels"] = int(
            np.count_nonzero(closed_barrier & ~hysteresis)
        )

        projection = np.asarray(capture.get("view_projection"), dtype=np.float64)
        centers = np.asarray(geometry.get("centers", ()), dtype=np.float64)
        hidden = np.asarray(
            geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool
        ).reshape(-1)
        source_hard = np.asarray(
            geometry.get("source_hard", np.zeros(count, dtype=bool)), dtype=bool
        ).reshape(-1)
        if projection.shape != (4, 4) or centers.shape != (count, 3):
            raise ValueError("screen-projection-schema")
        if len(hidden) != count:
            hidden = np.zeros(count, dtype=bool)
        if len(source_hard) != count:
            source_hard = np.zeros(count, dtype=bool)
        clip = np.c_[centers, np.ones(count, dtype=np.float64)] @ projection.T
        valid = np.all(np.isfinite(clip), axis=1) & (np.abs(clip[:, 3]) > 1.0e-12)
        ndc = np.zeros((count, 3), dtype=np.float64)
        ndc[valid] = clip[valid, :3] / clip[valid, 3, None]
        xs = np.rint((ndc[:, 0] * 0.5 + 0.5) * (width - 1)).astype(np.int64)
        ys = np.rint((ndc[:, 1] * 0.5 + 0.5) * (height - 1)).astype(np.int64)
        inside = (
            valid & ~hidden & ~source_hard
            & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        )
        depth_image = np.asarray(capture.get("depth", ()), dtype=np.float32)
        if depth_image.shape == (height, width):
            projected_depth = ndc[:, 2] * 0.5 + 0.5
            sampled_depth = np.full(count, np.nan, dtype=np.float64)
            sampled_depth[inside] = depth_image[ys[inside], xs[inside]]
            depth_valid = inside & np.isfinite(sampled_depth)
            # GPU depth readback is only used when the buffer has the normal
            # [0,1] range.  A conservative tolerance avoids discarding a
            # visible center due to polygon-center interpolation differences.
            plausible = depth_valid & (sampled_depth >= -1.0e-4) & (sampled_depth <= 1.0001)
            if np.any(plausible):
                inside &= ~plausible | (
                    np.abs(sampled_depth - projected_depth) <= 0.05
                )
                metrics["shadow_screen_occlusion_depth_used"] = True
        id_raster = np.full((height, width), -1, dtype=np.int32)
        depth_raster = np.full((height, width), np.inf, dtype=np.float64)
        valid_ids = np.flatnonzero(inside)
        if len(valid_ids) == 0:
            raise ValueError("screen-id-raster-empty")
        flat = ys[valid_ids] * width + xs[valid_ids]
        # OpenGL NDC z is ordered from near to far in increasing depth for
        # this comparison.  Stable sorting makes ties deterministic.
        order = np.argsort(ndc[valid_ids, 2], kind="stable")
        flat_ordered = flat[order]
        id_flat = id_raster.ravel()
        depth_flat = depth_raster.ravel()
        id_flat[flat_ordered] = valid_ids[order].astype(np.int32, copy=False)
        depth_flat[flat_ordered] = ndc[valid_ids[order], 2]
        metrics["shadow_screen_face_pixel_count"] = int(np.count_nonzero(id_raster >= 0))
        coverage = np.bincount(id_raster[id_raster >= 0], minlength=count)
        nonzero_coverage = coverage[coverage > 0]
        if len(nonzero_coverage):
            metrics["shadow_screen_face_coverage_min"] = float(np.min(nonzero_coverage))
            metrics["shadow_screen_face_coverage_max"] = float(np.max(nonzero_coverage))

        seed = int(seed_face)
        if seed < 0 or seed >= count or not inside[seed]:
            raise ValueError("screen-seed-raster-missing")
        seed_x, seed_y = int(xs[seed]), int(ys[seed])
        seed_patch = image[max(0, seed_y - 1):min(height, seed_y + 2), max(0, seed_x - 1):min(width, seed_x + 2)]
        seed_luminance = float(np.median(seed_patch)) if seed_patch.size else float(image[seed_y, seed_x])
        metrics["shadow_screen_seed_luminance"] = seed_luminance
        all_values = denoised[np.isfinite(denoised)]
        global_median = float(np.median(all_values)) if len(all_values) else seed_luminance
        plateau_values = seed_patch[np.isfinite(seed_patch)]
        plateau_median = float(np.median(plateau_values)) if len(plateau_values) else seed_luminance
        plateau_mad = float(1.4826 * np.median(np.abs(plateau_values - plateau_median))) if len(plateau_values) else 0.0
        bright = bool(seed_luminance >= global_median)
        tolerance = max(3.0 * plateau_mad, 0.02 * max(abs(plateau_median), 1.0e-2), 1.0e-3)
        cache = state.get("shadow_screen_classifier_cache")
        generation = int(state.get("shadow_capture_generation", 0))
        if not isinstance(cache, dict) or cache.get("generation") != generation:
            values_sorted = np.sort(np.unique(all_values))
            if bright:
                side = values_sorted[values_sorted <= seed_luminance + 1.0e-9]
                gaps = np.diff(side) if len(side) >= 2 else np.empty(0)
                if len(gaps) and float(np.max(gaps)) > max(2.0 * tolerance, 0.01):
                    idx = int(np.argmax(gaps)); base_threshold = float((side[idx] + side[idx + 1]) * 0.5)
                else:
                    base_threshold = float(plateau_median - tolerance)
            else:
                side = values_sorted[values_sorted >= seed_luminance - 1.0e-9]
                gaps = np.diff(side) if len(side) >= 2 else np.empty(0)
                if len(gaps) and float(np.max(gaps)) > max(2.0 * tolerance, 0.01):
                    idx = int(np.argmax(gaps)); base_threshold = float((side[idx] + side[idx + 1]) * 0.5)
                else:
                    base_threshold = float(plateau_median + tolerance)
            cache = {
                "generation": generation,
                "base_threshold": float(base_threshold),
                "bright": bool(bright),
                "seed_luminance": float(seed_luminance),
                "tolerance": float(tolerance),
            }
            state["shadow_screen_classifier_cache"] = cache
        base_threshold = float(cache.get("base_threshold", seed_luminance))
        bright = bool(cache.get("bright", bright))
        threshold_delta = max(
            float(cache.get("tolerance", tolerance)) * 0.5,
            (float(np.percentile(all_values, 90.0)) - float(np.percentile(all_values, 10.0))) * 0.05
            if len(all_values) else 0.01,
            0.01,
        )
        threshold = base_threshold + ((-int(step)) if bright else int(step)) * threshold_delta
        threshold = float(np.clip(threshold, float(np.min(all_values)), float(np.max(all_values))))
        metrics["shadow_screen_base_threshold"] = base_threshold
        metrics["shadow_screen_threshold"] = threshold
        metrics["shadow_screen_seed_luminance"] = float(cache.get("seed_luminance", seed_luminance))
        class_mask = denoised >= threshold if bright else denoised <= threshold
        # 2-D seed flood through the luminance class; closed strong gradients
        # are the only image-space barriers.  No screen edge overlap test is
        # used as the primary classifier.
        region = np.zeros((height, width), dtype=bool)
        if not class_mask[seed_y, seed_x]:
            class_mask[seed_y, seed_x] = True
        region[seed_y, seed_x] = True
        frontier = [(seed_y, seed_x)]
        while frontier:
            y, x = frontier.pop()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                if region[ny, nx] or not class_mask[ny, nx]:
                    continue
                # Luminance class membership is the primary barrier.  A
                # strong gradient pixel inside the same class must not split
                # a wide bright plateau into one-pixel islands; the closed
                # hysteresis map is retained for diagnostics and for
                # opposite-class transitions only.
                if (
                    (closed_barrier[ny, nx] or closed_barrier[y, x])
                    and bool(class_mask[ny, nx]) != bool(class_mask[y, x])
                ):
                    continue
                region[ny, nx] = True
                frontier.append((ny, nx))
        metrics["shadow_screen_mask_pixel_count"] = int(np.count_nonzero(region))
        pixel_faces = id_raster[region]
        pixel_faces = pixel_faces[pixel_faces >= 0]
        candidates = np.zeros(count, dtype=bool)
        if len(pixel_faces):
            candidates[np.unique(pixel_faces)] = True
        candidates[seed] = True
        # Graph connectivity is a topology safety check only.  It does not
        # add any geometry score or radius condition to normal E.
        offsets = np.asarray(geometry.get("offsets", ()), dtype=np.int64).reshape(-1)
        neighbors = np.asarray(geometry.get("neighbors", ()), dtype=np.int32).reshape(-1)
        connected = np.zeros(count, dtype=bool)
        if len(offsets) == count + 1 and int(offsets[-1]) == len(neighbors):
            connected[seed] = True
            frontier_faces = [seed]
            while frontier_faces:
                face = int(frontier_faces.pop())
                for neighbor in neighbors[int(offsets[face]):int(offsets[face + 1])]:
                    neighbor = int(neighbor)
                    if 0 <= neighbor < count and candidates[neighbor] and not connected[neighbor]:
                        connected[neighbor] = True
                        frontier_faces.append(neighbor)
        else:
            connected = candidates
            connected[seed] = True
        connected &= ~hidden & ~source_hard
        connected[seed] = True
        candidate_ids = np.flatnonzero(connected).astype(np.int32, copy=False)

        face_ids = np.asarray(
            geometry.get("face_ids", np.arange(count, dtype=np.int32)), dtype=np.int32
        ).reshape(-1)
        if len(face_ids) != count:
            face_ids = np.arange(count, dtype=np.int32)
        candidate_stable = np.unique(face_ids[candidate_ids]).astype(np.int32, copy=False)
        initial_stable = np.asarray(state.get("shadow_screen_initial_ids", ()), dtype=np.int32).reshape(-1)
        previous_stable = np.asarray(state.get("shadow_screen_previous_ids", ()), dtype=np.int32).reshape(-1)
        if int(step) == 0:
            if len(initial_stable) == 0:
                initial_stable = candidate_stable.copy()
                state["shadow_screen_initial_ids"] = initial_stable.copy()
                action = "initial"
            else:
                candidate_stable = initial_stable.copy()
                action = "return-to-initial"
        elif int(step) > 0:
            candidate_stable = np.unique(np.concatenate((candidate_stable, initial_stable, previous_stable))).astype(np.int32, copy=False)
            action = "expand-union"
        else:
            candidate_stable = np.intersect1d(candidate_stable, initial_stable)
            if len(previous_stable) and int(state.get("shadow_screen_previous_step", 0)) > int(step):
                candidate_stable = np.intersect1d(candidate_stable, previous_stable)
            action = "shrink-intersection"
        final = np.isin(face_ids, candidate_stable) & ~hidden & ~source_hard
        final[seed] = True
        selected_ids = np.flatnonzero(final).astype(np.int32, copy=False)
        state["shadow_screen_previous_ids"] = candidate_stable.copy()
        state["shadow_screen_previous_step"] = int(step)
        metrics["shadow_screen_monotonic_action"] = action
        metrics["shadow_screen_initial_count"] = int(len(initial_stable))
        metrics["shadow_screen_previous_count"] = int(len(previous_stable))
        metrics["shadow_screen_preserved_count"] = int(len(np.intersect1d(candidate_stable, initial_stable)))
        metrics["shadow_screen_accepted_count"] = int(len(selected_ids))
        accepted_values = denoised[ys[selected_ids], xs[selected_ids]] if len(selected_ids) else np.empty(0)
        if len(accepted_values):
            metrics["shadow_screen_accepted_luminance_min"] = float(np.min(accepted_values))
            metrics["shadow_screen_accepted_luminance_max"] = float(np.max(accepted_values))
        first = np.asarray(geometry.get("first", ()), dtype=np.int32).reshape(-1)
        second = np.asarray(geometry.get("second", ()), dtype=np.int32).reshape(-1)
        edge_keys = set()
        for pair_index, (left, right) in enumerate(zip(first, second)):
            left, right = int(left), int(right)
            if 0 <= left < count and 0 <= right < count and bool(final[left]) != bool(final[right]):
                key = _fill_preview_shadow_edge_key(geometry, pair_index)
                if key is not None:
                    edge_keys.add(key)
        state["shadow_luminance_region_edges"] = edge_keys
        metrics["shadow_screen_boundary_count"] = int(len(edge_keys))
        metrics["shadow_screen_rejected_neighbor_count"] = int(max(0, len(edge_keys)))
        metrics["shadow_screen_fallback_reason"] = "ok"
        return selected_ids, metrics
    except (AttributeError, IndexError, KeyError, MemoryError, RuntimeError, TypeError, ValueError, ZeroDivisionError):
        metrics["shadow_screen_fallback_reason"] = "screen-raster-or-morphology-failed"
        return np.asarray([int(seed_face)], dtype=np.int32), metrics


def _fill_preview_shadow_screen_region_roi(state, geometry, seed_face, step=0):
    """Bounded ordinary-E screen classifier for one immutable cursor ROI.

    Capture and projection are still performed once per invoke, but all image
    morphology and flood work is cropped to a brush-sized circle.  The mesh
    map is deliberately center-sample based: each visible projected face
    center owns one ROI pixel and the final interface is the physical edge
    between accepted and rejected center samples.  This keeps the experimental
    path predictable on dense meshes without pretending to be polygon coverage.
    """
    import numpy as np
    from collections import deque

    count = int(geometry.get("count", 0))
    seed = int(seed_face)
    generation = int(state.get("shadow_capture_generation", 0))
    metrics = {
        "shadow_screen_classifier_mode": "screen-grayscale-morphology-roi",
        "shadow_screen_fallback_reason": "capture-unavailable",
        "shadow_screen_width": 0,
        "shadow_screen_height": 0,
        "shadow_screen_roi_origin": (0, 0),
        "shadow_screen_roi_bounds": (0, 0, 0, 0),
        "shadow_screen_roi_radius": int(state.get("shadow_screen_roi_radius", 320)),
        "shadow_screen_roi_pixel_count": 0,
        "shadow_screen_roi_circle_pixels": 0,
        "shadow_screen_analysis_width": 0,
        "shadow_screen_analysis_height": 0,
        "shadow_screen_downsample_factor": 1,
        "shadow_screen_projected_median_edge_spacing": 0.0,
        "shadow_screen_analysis_pixel_count": 0,
        "shadow_screen_luminance_min": 0.0,
        "shadow_screen_luminance_max": 0.0,
        "shadow_screen_seed_luminance": 0.0,
        "shadow_screen_raw_luminance_min": 0.0,
        "shadow_screen_raw_luminance_max": 0.0,
        "shadow_screen_raw_luminance_quantiles": (),
        "shadow_screen_sharpened_luminance_min": 0.0,
        "shadow_screen_sharpened_luminance_max": 0.0,
        "shadow_screen_sharpened_luminance_quantiles": (),
        "shadow_screen_sharpened_field_used": False,
        "shadow_screen_sharpen_knee": 0.0,
        "shadow_screen_sharpen_steepness": 0.0,
        "shadow_screen_sharpen_curve_scale": 0.0,
        "shadow_screen_transition_width_before_pixels": 0,
        "shadow_screen_transition_width_after_pixels": 0,
        "shadow_screen_transition_face_rows_before": 0,
        "shadow_screen_transition_face_rows_after": 0,
        "shadow_screen_sharpened_overlay_active": False,
        "shadow_screen_sharpened_overlay_alpha": 0.0,
        "shadow_screen_sharpened_overlay_restored": False,
        "shadow_screen_denoise_kernel": 3,
        "shadow_screen_gradient_kernel": 3,
        "shadow_screen_closing_kernel": 3,
        "shadow_screen_strong_threshold": 0.0,
        "shadow_screen_weak_threshold": 0.0,
        "shadow_screen_threshold": 0.0,
        "shadow_screen_base_threshold": 0.0,
        "shadow_screen_step": int(step),
        "shadow_screen_mask_pixel_count": 0,
        "shadow_screen_face_pixel_count": 0,
        "shadow_screen_face_coverage_min": 0.0,
        "shadow_screen_face_coverage_max": 0.0,
        "shadow_screen_accepted_count": 0,
        "shadow_screen_accepted_luminance_min": 0.0,
        "shadow_screen_accepted_luminance_max": 0.0,
        "shadow_screen_rejected_neighbor_count": 0,
        "shadow_screen_boundary_count": 0,
        "shadow_screen_raw_gradient_min": 0.0,
        "shadow_screen_raw_gradient_median": 0.0,
        "shadow_screen_raw_gradient_max": 0.0,
        "shadow_screen_gradient_mad": 0.0,
        "shadow_screen_closing_filled_pixels": 0,
        "shadow_screen_top_hat_used": False,
        "shadow_screen_black_hat_used": False,
        "shadow_screen_monotonic_action": "initial",
        "shadow_screen_initial_count": 0,
        "shadow_screen_previous_count": 0,
        "shadow_screen_preserved_count": 0,
        "shadow_screen_face_set_independent": True,
        "shadow_screen_id_raster_mode": "int32-visible-face-center-depth-order",
        "shadow_screen_center_sample_mode": True,
        "shadow_screen_centers_considered": 0,
        "shadow_screen_centers_visible": 0,
        "shadow_screen_centers_in_region": 0,
        "shadow_screen_duplicate_center_pixel_count": 0,
        "shadow_screen_max_centers_per_pixel": 0,
        "shadow_screen_candidates_after_connectivity": 0,
        "shadow_screen_collapse_reason": "none",
        "shadow_screen_pixels_visited": 0,
        "shadow_screen_traversal_chunks": 0,
        "shadow_screen_work_budget": 0,
        "shadow_screen_work_budget_cap": 0,
        "shadow_screen_cancel_state": "not-requested",
        "shadow_screen_barrier_pixels_rejected": 0,
        "shadow_screen_barrier_crossings_rejected": 0,
        "shadow_screen_frontier_count": 0,
        "shadow_screen_refine_reason": "not-run",
        "shadow_screen_boundary_before": 0,
        "shadow_screen_boundary_after": 0,
        "shadow_screen_refine_scored_edges": 0,
        "shadow_screen_refine_score_min": 0.0,
        "shadow_screen_refine_score_median": 0.0,
        "shadow_screen_refine_score_max": 0.0,
        "shadow_screen_refine_score_threshold": 0.0,
        "shadow_screen_refine_moved_faces": 0,
        "shadow_screen_refine_max_ring_shift": 0,
        "shadow_screen_refine_sample_failures": 0,
        "shadow_screen_refine_chain_count": 0,
        "shadow_screen_refine_spikes_removed": 0,
        "shadow_screen_refine_seed_connected": False,
    }
    state["shadow_luminance_region_edges"] = set()

    def seed_only(reason):
        metrics["shadow_screen_fallback_reason"] = str(reason)
        metrics["shadow_screen_accepted_count"] = 1 if 0 <= seed < count else 0
        return np.asarray([seed], dtype=np.int32) if 0 <= seed < count else np.empty(0, dtype=np.int32), metrics

    capture = state.get("shadow_capture")
    if count <= 0 or not isinstance(capture, dict) or not capture.get("ok"):
        return seed_only("capture-unavailable")
    try:
        full = np.asarray(capture.get("luminance", ()), dtype=np.float32)
        if full.ndim != 2 or full.shape[0] < 8 or full.shape[1] < 8:
            return seed_only("screen-image-schema")
        height, width = int(full.shape[0]), int(full.shape[1])
        metrics["shadow_screen_width"] = width
        metrics["shadow_screen_height"] = height
        seed_screen = state.get("seed_screen", capture.get("seed_screen"))
        try:
            sx, sy = int(round(float(seed_screen[0]))), int(round(float(seed_screen[1])))
        except (IndexError, TypeError, ValueError):
            sx, sy = width // 2, height // 2
        sx = max(0, min(width - 1, sx))
        sy = max(0, min(height - 1, sy))
        roi_radius = 320
        x0, x1 = max(0, sx - roi_radius), min(width, sx + roi_radius + 1)
        y0, y1 = max(0, sy - roi_radius), min(height, sy + roi_radius + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return seed_only("roi-too-small")
        metrics["shadow_screen_roi_origin"] = (int(x0), int(y0))
        metrics["shadow_screen_roi_bounds"] = (int(x0), int(y0), int(x1), int(y1))
        roi_w, roi_h = int(x1 - x0), int(y1 - y0)
        yy, xx = np.ogrid[:roi_h, :roi_w]
        circle = ((xx + x0 - sx) ** 2 + (yy + y0 - sy) ** 2) <= roi_radius ** 2
        metrics["shadow_screen_roi_pixel_count"] = int(circle.size)
        metrics["shadow_screen_roi_circle_pixels"] = int(np.count_nonzero(circle))

        buffers = state.get("shadow_screen_buffers")
        # Estimate a robust projected face-center spacing before building the
        # image buffers.  Two-to-four analysis samples per typical face keeps
        # a 640px context while avoiding a Python walk over hundreds of
        # thousands of mostly redundant pixels.
        projection_probe = np.asarray(capture.get("view_projection"), dtype=np.float64)
        centers_probe = np.asarray(geometry.get("centers", ()), dtype=np.float64)
        edge_spacing = np.empty(0, dtype=np.float64)
        if projection_probe.shape == (4, 4) and centers_probe.shape == (count, 3):
            clip_probe = np.c_[centers_probe, np.ones(count, dtype=np.float64)] @ projection_probe.T
            valid_probe = np.all(np.isfinite(clip_probe), axis=1) & (np.abs(clip_probe[:, 3]) > 1.0e-12)
            ndc_probe = np.zeros((count, 3), dtype=np.float64)
            ndc_probe[valid_probe] = clip_probe[valid_probe, :3] / clip_probe[valid_probe, 3, None]
            px_probe = np.rint((ndc_probe[:, 0] * 0.5 + 0.5) * (width - 1)).astype(np.int64)
            py_probe = np.rint((ndc_probe[:, 1] * 0.5 + 0.5) * (height - 1)).astype(np.int64)
            first_probe = np.asarray(geometry.get("first", ()), dtype=np.int32).reshape(-1)
            second_probe = np.asarray(geometry.get("second", ()), dtype=np.int32).reshape(-1)
            valid_pairs = (
                (first_probe >= 0) & (second_probe >= 0)
                & (first_probe < count) & (second_probe < count)
                & valid_probe[first_probe] & valid_probe[second_probe]
            ) if len(first_probe) == len(second_probe) else np.zeros(0, dtype=bool)
            if np.any(valid_pairs):
                dx = px_probe[first_probe[valid_pairs]] - px_probe[second_probe[valid_pairs]]
                dy = py_probe[first_probe[valid_pairs]] - py_probe[second_probe[valid_pairs]]
                edge_spacing = np.hypot(dx, dy).astype(np.float64)
                edge_spacing = edge_spacing[np.isfinite(edge_spacing) & (edge_spacing > 0.0)]
        median_spacing = float(np.median(edge_spacing)) if len(edge_spacing) else 4.0
        downsample = int(max(2, min(8, round(median_spacing / 3.0))))
        metrics["shadow_screen_projected_median_edge_spacing"] = median_spacing
        buffer_key = (generation, int(x0), int(y0), int(x1), int(y1), int(roi_radius), downsample)
        if not isinstance(buffers, dict) or buffers.get("key") != buffer_key:
            source_h, source_w = int(y1 - y0), int(x1 - x0)
            source_circle = circle
            image = full[y0:y1, x0:x1].copy()
            finite = np.isfinite(image)
            if not np.any(finite):
                return seed_only("screen-image-empty")
            fill = float(np.median(image[finite]))
            image = np.where(finite, image, fill).astype(np.float32, copy=False)
            analysis_h = int((source_h + downsample - 1) // downsample)
            analysis_w = int((source_w + downsample - 1) // downsample)
            padded_image = np.pad(
                image,
                ((0, analysis_h * downsample - source_h),
                 (0, analysis_w * downsample - source_w)),
                mode="edge",
            )
            image = np.median(
                padded_image.reshape(analysis_h, downsample, analysis_w, downsample),
                axis=(1, 3),
            ).astype(np.float32, copy=False)
            roi_h, roi_w = analysis_h, analysis_w
            analysis_x = x0 + (np.arange(roi_w, dtype=np.float64) + 0.5) * downsample
            analysis_y = y0 + (np.arange(roi_h, dtype=np.float64) + 0.5) * downsample
            ay, ax = np.meshgrid(analysis_y, analysis_x, indexing="ij")
            circle = ((ax - sx) ** 2 + (ay - sy) ** 2) <= roi_radius ** 2
            metrics["shadow_screen_analysis_width"] = int(roi_w)
            metrics["shadow_screen_analysis_height"] = int(roi_h)
            metrics["shadow_screen_analysis_pixel_count"] = int(circle.size)
            pad = np.pad(image, 1, mode="edge")
            samples = np.stack(tuple(
                pad[dy:dy + roi_h, dx:dx + roi_w]
                for dy in range(3) for dx in range(3)
            ), axis=0)
            raw_denoised = np.median(samples, axis=0).astype(np.float32, copy=False)
            # The visible toon_dark field can contain a broad studio-light
            # ramp.  Keep its ordering but apply a monotonic, adaptive sigmoid
            # around the robust seed-class knee so morphology and boundary
            # scoring see the same narrow transition instead of a wide gray
            # shelf.  Raw values remain available for diagnostics.
            raw_values = raw_denoised[circle]
            raw_values = raw_values[np.isfinite(raw_values)]
            seed_cell_x = int(np.clip((sx - x0) // downsample, 0, roi_w - 1))
            seed_cell_y = int(np.clip((sy - y0) // downsample, 0, roi_h - 1))
            raw_seed = float(raw_denoised[seed_cell_y, seed_cell_x])
            raw_q05 = float(np.percentile(raw_values, 5.0)) if len(raw_values) else raw_seed
            raw_q95 = float(np.percentile(raw_values, 95.0)) if len(raw_values) else raw_seed
            raw_span = max(raw_q95 - raw_q05, 1.0e-6)
            raw_median = float(np.median(raw_values)) if len(raw_values) else raw_seed
            seed_window = raw_denoised[
                max(0, seed_cell_y - 1):min(roi_h, seed_cell_y + 2),
                max(0, seed_cell_x - 1):min(roi_w, seed_cell_x + 2),
            ]
            seed_window = seed_window[np.isfinite(seed_window)]
            seed_ref = float(np.median(seed_window)) if len(seed_window) else raw_seed
            seed_mad = float(1.4826 * np.median(np.abs(seed_window - seed_ref))) if len(seed_window) else 0.0
            seed_tolerance = max(3.0 * seed_mad, 0.02 * max(abs(seed_ref), 1.0e-2), 1.0e-3)
            raw_sorted = np.sort(np.unique(raw_values)) if len(raw_values) else np.empty(0)
            raw_bright = bool(seed_ref >= raw_median)
            raw_side = (
                raw_sorted[raw_sorted <= seed_ref + 1.0e-9]
                if raw_bright else raw_sorted[raw_sorted >= seed_ref - 1.0e-9]
            )
            raw_gaps = np.diff(raw_side) if len(raw_side) >= 2 else np.empty(0)
            raw_gap_index = int(np.argmax(raw_gaps)) if len(raw_gaps) else -1
            if raw_gap_index >= 0 and float(raw_gaps[raw_gap_index]) > max(2.0 * seed_tolerance, 0.01):
                raw_knee = float((raw_side[raw_gap_index] + raw_side[raw_gap_index + 1]) * 0.5)
            else:
                raw_knee = float(seed_ref - seed_tolerance if raw_bright else seed_ref + seed_tolerance)
            raw_knee = float(np.clip(raw_knee, raw_q05, raw_q95))
            curve_scale = max(0.06 * raw_span, 2.0 * seed_tolerance, 1.0e-3)
            steepness = float(np.clip(raw_span / curve_scale * 0.85, 3.0, 14.0))
            if raw_span > max(1.0e-4, 4.0 * seed_tolerance):
                normalized = np.clip((raw_denoised.astype(np.float64) - raw_q05) / raw_span, 0.0, 1.0)
                knee_normalized = float(np.clip((raw_knee - raw_q05) / raw_span, 0.0, 1.0))
                exponent = np.clip(-steepness * (normalized - knee_normalized), -40.0, 40.0)
                sigmoid = 1.0 / (1.0 + np.exp(exponent))
                denoised = (raw_q05 + raw_span * sigmoid).astype(np.float32, copy=False)
                sharpened_active = True
                sharpened_knee = float(raw_q05 + raw_span * 0.5)
            else:
                denoised = raw_denoised.copy()
                sharpened_active = False
                sharpened_knee = raw_seed
            denoised_pad = np.pad(denoised, 1, mode="edge")
            denoised_samples = np.stack(tuple(
                denoised_pad[dy:dy + roi_h, dx:dx + roi_w]
                for dy in range(3) for dx in range(3)
            ), axis=0)
            gradient = (
                np.max(denoised_samples, axis=0)
                - np.min(denoised_samples, axis=0)
            ).astype(np.float32, copy=False)
            values = denoised[circle]
            values = values[np.isfinite(values)]
            raw_transition = (
                (raw_denoised >= raw_q05 + 0.10 * raw_span)
                & (raw_denoised <= raw_q05 + 0.90 * raw_span)
                & circle
            )
            sharpened_transition = (
                (denoised >= raw_q05 + 0.10 * raw_span)
                & (denoised <= raw_q05 + 0.90 * raw_span)
                & circle
            )
            def longest_run(mask):
                best = 0
                for axis in (0, 1):
                    rows = np.moveaxis(mask, axis, 0)
                    for row in rows:
                        indices = np.flatnonzero(row)
                        if not len(indices):
                            continue
                        cuts = np.flatnonzero(np.diff(indices) > 1)
                        starts = np.r_[0, cuts + 1]
                        ends = np.r_[cuts + 1, len(indices)]
                        best = max(best, int(np.max(ends - starts)))
                return int(best)
            transition_before = longest_run(raw_transition)
            sharpened_values = denoised[circle]
            sharpened_values = sharpened_values[np.isfinite(sharpened_values)]
            sharpened_q05 = float(np.percentile(sharpened_values, 5.0)) if len(sharpened_values) else raw_q05
            sharpened_q95 = float(np.percentile(sharpened_values, 95.0)) if len(sharpened_values) else raw_q95
            sharpened_span = max(sharpened_q95 - sharpened_q05, 1.0e-6)
            sharpened_transition = (
                (denoised >= sharpened_q05 + 0.10 * sharpened_span)
                & (denoised <= sharpened_q05 + 0.90 * sharpened_span)
                & circle
            )
            transition_after = longest_run(sharpened_transition)
            gvalues = gradient[circle]
            gradient_median = float(np.median(gvalues)) if len(gvalues) else 0.0
            gradient_mad = float(1.4826 * np.median(np.abs(gvalues - gradient_median))) if len(gvalues) else 0.0
            value_span = max(float(np.max(values) - np.min(values)), 1.0e-3) if len(values) else 1.0e-3
            strong_threshold = max(gradient_median + 3.0 * max(gradient_mad, 1.0e-4), value_span * 0.04)
            weak_threshold = max(gradient_median + 1.5 * max(gradient_mad, 1.0e-4), strong_threshold * 0.5)
            strong = (gradient >= strong_threshold) & circle
            weak = (gradient >= weak_threshold) & circle
            hysteresis = np.zeros_like(strong, dtype=bool)
            q = deque((int(y), int(x)) for y, x in zip(*np.nonzero(strong)))
            hysteresis[strong] = True
            while q:
                cy, cx = q.popleft()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < roi_h and 0 <= nx < roi_w and weak[ny, nx] and not hysteresis[ny, nx]:
                        hysteresis[ny, nx] = True
                        q.append((ny, nx))
            # Closing is causal: it bridges short gaps in a detected line,
            # and the resulting pixels are non-traversable in the flood.
            bpad = np.pad(hysteresis, 1, mode="constant")
            bd = np.max(np.stack(tuple(bpad[dy:dy + roi_h, dx:dx + roi_w] for dy in range(3) for dx in range(3)), axis=0), axis=0)
            cpad = np.pad(bd, 1, mode="constant")
            closed = np.min(np.stack(tuple(cpad[dy:dy + roi_h, dx:dx + roi_w] for dy in range(3) for dx in range(3)), axis=0), axis=0).astype(bool)
            closed &= circle

            projection = np.asarray(capture.get("view_projection"), dtype=np.float64)
            centers = np.asarray(geometry.get("centers", ()), dtype=np.float64)
            hidden = np.asarray(geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool).reshape(-1)
            source_hard = np.asarray(geometry.get("source_hard", np.zeros(count, dtype=bool)), dtype=bool).reshape(-1)
            if projection.shape != (4, 4) or centers.shape != (count, 3):
                return seed_only("screen-projection-schema")
            if len(hidden) != count:
                hidden = np.zeros(count, dtype=bool)
            if len(source_hard) != count:
                source_hard = np.zeros(count, dtype=bool)
            clip = np.c_[centers, np.ones(count, dtype=np.float64)] @ projection.T
            valid = np.all(np.isfinite(clip), axis=1) & (np.abs(clip[:, 3]) > 1.0e-12)
            ndc = np.zeros((count, 3), dtype=np.float64)
            ndc[valid] = clip[valid, :3] / clip[valid, 3, None]
            px = np.rint((ndc[:, 0] * 0.5 + 0.5) * (width - 1)).astype(np.int64)
            py = np.rint((ndc[:, 1] * 0.5 + 0.5) * (height - 1)).astype(np.int64)
            center_inside = valid & ~hidden & ~source_hard & (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
            local_x = np.floor((px - x0) / float(downsample)).astype(np.int64)
            local_y = np.floor((py - y0) / float(downsample)).astype(np.int64)
            center_inside &= circle[np.clip(local_y, 0, roi_h - 1), np.clip(local_x, 0, roi_w - 1)]
            depth_image = np.asarray(capture.get("depth", ()), dtype=np.float32)
            if depth_image.shape == (height, width):
                sampled = np.full(count, np.nan, dtype=np.float64)
                sampled[center_inside] = depth_image[py[center_inside], px[center_inside]]
                plausible = center_inside & np.isfinite(sampled) & (sampled >= -1.0e-4) & (sampled <= 1.0001)
                center_inside &= ~plausible | (np.abs(sampled - (ndc[:, 2] * 0.5 + 0.5)) <= 0.05)
            id_raster = np.full((roi_h, roi_w), -1, dtype=np.int32)
            depth_raster = np.full((roi_h, roi_w), np.inf, dtype=np.float64)
            ids = np.flatnonzero(center_inside)
            if len(ids):
                flat = local_y[ids] * roi_w + local_x[ids]
                order = np.argsort(ndc[ids, 2], kind="stable")
                flat_ordered = flat[order]
                id_flat, depth_flat = id_raster.ravel(), depth_raster.ravel()
                id_flat[flat_ordered] = ids[order].astype(np.int32, copy=False)
                depth_flat[flat_ordered] = ndc[ids[order], 2]
            buffers = {
                "key": buffer_key,
                "generation": generation,
                "image": image,
                "raw_denoised": raw_denoised,
                "denoised": denoised,
                "gradient": gradient,
                "hysteresis": hysteresis,
                "closed_barrier": closed,
                "raw_transition": raw_transition,
                "sharpened_transition": sharpened_transition,
                "circle": circle,
                "id_raster": id_raster,
                "center_inside": center_inside,
                "center_x": local_x,
                "center_y": local_y,
                "strong_threshold": float(strong_threshold),
                "weak_threshold": float(weak_threshold),
                "gradient_median": gradient_median,
                "gradient_mad": gradient_mad,
                "seed_xy": (int((sx - x0) // downsample), int((sy - y0) // downsample)),
                "downsample_factor": int(downsample),
                "source_roi_size": (int(source_w), int(source_h)),
                "sharpened_field_used": bool(sharpened_active),
                "sharpen_raw_q05": float(raw_q05),
                "sharpen_raw_q95": float(raw_q95),
                "sharpen_raw_knee": float(raw_knee),
                "sharpen_knee": float(sharpened_knee),
                "sharpen_steepness": float(steepness),
                "sharpen_curve_scale": float(curve_scale),
                "transition_width_before": int(transition_before),
                "transition_width_after": int(transition_after),
                "centers_considered": int(np.count_nonzero(valid)),
                "centers_visible": int(len(ids)),
            }
            state["shadow_screen_buffers"] = buffers
        else:
            image = buffers["image"]
            denoised = buffers["denoised"]
            gradient = buffers["gradient"]
            closed = buffers["closed_barrier"]
            circle = buffers["circle"]
            id_raster = buffers["id_raster"]
            center_inside = buffers["center_inside"]
            local_x, local_y = buffers["center_x"], buffers["center_y"]
            roi_h, roi_w = int(image.shape[0]), int(image.shape[1])
        metrics["shadow_screen_analysis_width"] = int(roi_w)
        metrics["shadow_screen_analysis_height"] = int(roi_h)
        metrics["shadow_screen_analysis_pixel_count"] = int(circle.size)
        metrics["shadow_screen_downsample_factor"] = int(buffers.get("downsample_factor", 1))
        metrics["shadow_screen_strong_threshold"] = float(buffers.get("strong_threshold", 0.0))
        metrics["shadow_screen_weak_threshold"] = float(buffers.get("weak_threshold", 0.0))
        metrics["shadow_screen_raw_gradient_min"] = float(np.min(gradient[circle])) if np.any(circle) else 0.0
        metrics["shadow_screen_raw_gradient_median"] = float(np.median(gradient[circle])) if np.any(circle) else 0.0
        metrics["shadow_screen_raw_gradient_max"] = float(np.max(gradient[circle])) if np.any(circle) else 0.0
        metrics["shadow_screen_gradient_mad"] = float(buffers.get("gradient_mad", 0.0))
        metrics["shadow_screen_closing_filled_pixels"] = int(np.count_nonzero(buffers["closed_barrier"] & ~buffers["hysteresis"]))
        metrics["shadow_screen_centers_considered"] = int(buffers.get("centers_considered", 0))
        metrics["shadow_screen_centers_visible"] = int(buffers.get("centers_visible", 0))
        visible_for_transition = np.flatnonzero(center_inside)
        if len(visible_for_transition):
            tx = np.clip(local_x[visible_for_transition], 0, roi_w - 1)
            ty = np.clip(local_y[visible_for_transition], 0, roi_h - 1)
            raw_transition_field = np.asarray(
                buffers.get("raw_transition", np.zeros_like(circle)), dtype=bool
            )
            sharp_transition_field = np.asarray(
                buffers.get("sharpened_transition", np.zeros_like(circle)), dtype=bool
            )
            if raw_transition_field.shape == circle.shape:
                metrics["shadow_screen_transition_face_rows_before"] = int(
                    np.count_nonzero(raw_transition_field[ty, tx])
                )
            if sharp_transition_field.shape == circle.shape:
                metrics["shadow_screen_transition_face_rows_after"] = int(
                    np.count_nonzero(sharp_transition_field[ty, tx])
                )
        metrics["shadow_screen_face_pixel_count"] = int(
            np.count_nonzero(np.asarray(buffers.get("id_raster"), dtype=np.int32) >= 0)
        )
        # This is center-sample occupancy, not polygon coverage.  Keep the
        # legacy field populated only as a clearly documented 0/1 sample
        # count so diagnostics cannot mistake it for rasterized area.
        metrics["shadow_screen_face_coverage_min"] = 1.0 if metrics["shadow_screen_face_pixel_count"] else 0.0
        metrics["shadow_screen_face_coverage_max"] = 1.0 if metrics["shadow_screen_face_pixel_count"] else 0.0
        raw_field = np.asarray(
            buffers.get("raw_denoised", denoised), dtype=np.float32
        )
        raw_field_values = raw_field[circle] if raw_field.shape == denoised.shape else np.empty(0)
        raw_field_values = raw_field_values[np.isfinite(raw_field_values)]
        sharp_values = denoised[circle]
        sharp_values = sharp_values[np.isfinite(sharp_values)]
        if len(raw_field_values):
            metrics["shadow_screen_raw_luminance_min"] = float(np.min(raw_field_values))
            metrics["shadow_screen_raw_luminance_max"] = float(np.max(raw_field_values))
            metrics["shadow_screen_raw_luminance_quantiles"] = tuple(
                float(value) for value in np.percentile(raw_field_values, (5.0, 25.0, 50.0, 75.0, 95.0))
            )
        if len(sharp_values):
            metrics["shadow_screen_sharpened_luminance_min"] = float(np.min(sharp_values))
            metrics["shadow_screen_sharpened_luminance_max"] = float(np.max(sharp_values))
            metrics["shadow_screen_sharpened_luminance_quantiles"] = tuple(
                float(value) for value in np.percentile(sharp_values, (5.0, 25.0, 50.0, 75.0, 95.0))
            )
        metrics["shadow_screen_sharpened_field_used"] = bool(
            buffers.get("sharpened_field_used", False)
        )
        metrics["shadow_screen_sharpen_knee"] = float(
            buffers.get("sharpen_knee", 0.0)
        )
        metrics["shadow_screen_sharpen_steepness"] = float(
            buffers.get("sharpen_steepness", 0.0)
        )
        metrics["shadow_screen_sharpen_curve_scale"] = float(
            buffers.get("sharpen_curve_scale", 0.0)
        )
        metrics["shadow_screen_transition_width_before_pixels"] = int(
            buffers.get("transition_width_before", 0)
        )
        metrics["shadow_screen_transition_width_after_pixels"] = int(
            buffers.get("transition_width_after", 0)
        )
        roi_values = denoised[circle]
        if len(roi_values):
            metrics["shadow_screen_luminance_min"] = float(np.min(roi_values))
            metrics["shadow_screen_luminance_max"] = float(np.max(roi_values))
        seed_x, seed_y = (int(v) for v in buffers.get("seed_xy", (0, 0)))
        if not (0 <= seed_x < roi_w and 0 <= seed_y < roi_h):
            return seed_only("screen-seed-roi-missing")
        seed_luminance = float(denoised[seed_y, seed_x])
        metrics["shadow_screen_seed_luminance"] = seed_luminance
        values = denoised[circle]
        values = values[np.isfinite(values)]
        if not len(values):
            return seed_only("screen-roi-empty")
        roi_median = float(np.median(values))
        seed_patch = denoised[max(0, seed_y - 1):min(roi_h, seed_y + 2), max(0, seed_x - 1):min(roi_w, seed_x + 2)]
        seed_patch = seed_patch[np.isfinite(seed_patch)]
        seed_ref = float(np.median(seed_patch)) if len(seed_patch) else seed_luminance
        seed_mad = float(1.4826 * np.median(np.abs(seed_patch - seed_ref))) if len(seed_patch) else 0.0
        tolerance = max(3.0 * seed_mad, 0.02 * max(abs(seed_ref), 1.0e-2), 1.0e-3)
        cache = state.get("shadow_screen_classifier_cache")
        if not isinstance(cache, dict) or cache.get("generation") != generation or cache.get("roi_key") != buffers.get("key"):
            sorted_values = np.sort(np.unique(values))
            bright = bool(seed_ref >= roi_median)
            side = sorted_values[sorted_values <= seed_ref + 1.0e-9] if bright else sorted_values[sorted_values >= seed_ref - 1.0e-9]
            gaps = np.diff(side) if len(side) >= 2 else np.empty(0)
            largest = int(np.argmax(gaps)) if len(gaps) else -1
            if largest >= 0 and float(gaps[largest]) > max(2.0 * tolerance, 0.01):
                base_threshold = float((side[largest] + side[largest + 1]) * 0.5)
            else:
                base_threshold = float(seed_ref - tolerance if bright else seed_ref + tolerance)
            cache = {"generation": generation, "roi_key": buffers.get("key"), "base_threshold": base_threshold, "bright": bright, "seed_luminance": seed_luminance, "tolerance": tolerance}
            state["shadow_screen_classifier_cache"] = cache
        base_threshold = float(cache.get("base_threshold", seed_luminance))
        bright = bool(cache.get("bright", seed_ref >= roi_median))
        threshold_delta = max(float(cache.get("tolerance", tolerance)) * 0.5, float(np.percentile(values, 90.0) - np.percentile(values, 10.0)) * 0.05, 0.01)
        threshold = base_threshold + ((-int(step)) if bright else int(step)) * threshold_delta
        threshold = float(np.clip(threshold, float(np.min(values)), float(np.max(values))))
        metrics["shadow_screen_base_threshold"] = base_threshold
        metrics["shadow_screen_threshold"] = threshold
        class_mask = (denoised >= threshold) if bright else (denoised <= threshold)
        class_mask &= circle
        if not class_mask[seed_y, seed_x]:
            class_mask[seed_y, seed_x] = True

        region = np.zeros((roi_h, roi_w), dtype=bool)
        region[seed_y, seed_x] = True
        q = deque(((seed_y, seed_x),))
        circle_pixels = int(np.count_nonzero(circle))
        # The fixed 320px radius has about 322k pixels.  Permit the whole
        # circle plus a small margin, while retaining a deterministic hard
        # cap for malformed/oversized future configurations.
        work_cap = 380000
        work_budget = min(int(circle.size), work_cap, circle_pixels + 8192)
        metrics["shadow_screen_work_budget"] = int(work_budget)
        metrics["shadow_screen_work_budget_cap"] = int(work_cap)
        visited = 0
        barrier_rejected = 0
        barrier_crossings = 0
        cancelled = False
        while q and visited < work_budget:
            y, x = q.popleft()
            visited += 1
            if visited % 4096 == 0 and bool(state.get("cancel_requested", False)):
                cancelled = True
                break
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < roi_h and 0 <= nx < roi_w) or region[ny, nx] or not circle[ny, nx] or not class_mask[ny, nx]:
                    continue
                # A closed morphology barrier is a real non-traversable
                # image boundary, even when both pixels share the class.
                if ((closed[y, x] or closed[ny, nx]) and (y, x) != (seed_y, seed_x) and (ny, nx) != (seed_y, seed_x)):
                    barrier_rejected += 1
                    barrier_crossings += 1
                    continue
                region[ny, nx] = True
                q.append((ny, nx))
        metrics["shadow_screen_pixels_visited"] = int(visited)
        metrics["shadow_screen_traversal_chunks"] = int((visited + 4095) // 4096)
        metrics["shadow_screen_barrier_pixels_rejected"] = int(barrier_rejected)
        metrics["shadow_screen_barrier_crossings_rejected"] = int(barrier_crossings)
        metrics["shadow_screen_cancel_state"] = "cancelled" if cancelled else ("work-budget" if q else "complete")
        metrics["shadow_screen_mask_pixel_count"] = int(np.count_nonzero(region))
        metrics["shadow_screen_frontier_count"] = int(len(q))
        # The diagnostic ID raster keeps one deterministic winner per pixel,
        # but it is not authoritative: dense meshes routinely place several
        # visible centers on the same pixel.  Sample every cached center
        # directly so a collision cannot erase faces before graph safety.
        visible_ids = np.flatnonzero(center_inside)
        if len(visible_ids):
            center_flat = local_y[visible_ids] * roi_w + local_x[visible_ids]
            center_counts = np.bincount(center_flat, minlength=roi_h * roi_w)
            metrics["shadow_screen_duplicate_center_pixel_count"] = int(
                np.count_nonzero(center_counts > 1)
            )
            metrics["shadow_screen_max_centers_per_pixel"] = int(
                np.max(center_counts) if len(center_counts) else 0
            )
            center_in_region = region[local_y[visible_ids], local_x[visible_ids]]
            region_ids = visible_ids[center_in_region]
        else:
            region_ids = np.empty(0, dtype=np.int32)
        metrics["shadow_screen_centers_in_region"] = int(len(region_ids))
        candidates = np.zeros(count, dtype=bool)
        if len(region_ids):
            candidates[region_ids] = True
        if 0 <= seed < count:
            candidates[seed] = True
        hidden = np.asarray(geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool).reshape(-1)
        source_hard = np.asarray(geometry.get("source_hard", np.zeros(count, dtype=bool)), dtype=bool).reshape(-1)
        offsets = np.asarray(geometry.get("offsets", ()), dtype=np.int64).reshape(-1)
        neighbors = np.asarray(geometry.get("neighbors", ()), dtype=np.int32).reshape(-1)
        connected = np.zeros(count, dtype=bool)
        if len(offsets) == count + 1 and int(offsets[-1]) == len(neighbors):
            connected[seed] = True
            face_q = deque((seed,))
            while face_q:
                face = int(face_q.popleft())
                for neighbor in neighbors[int(offsets[face]):int(offsets[face + 1])]:
                    neighbor = int(neighbor)
                    if 0 <= neighbor < count and candidates[neighbor] and not connected[neighbor] and center_inside[neighbor]:
                        connected[neighbor] = True
                        face_q.append(neighbor)
        else:
            connected = candidates
        connected &= ~hidden & ~source_hard
        if 0 <= seed < count:
            connected[seed] = True
        metrics["shadow_screen_candidates_after_connectivity"] = int(
            np.count_nonzero(connected)
        )
        if len(region_ids) and not np.count_nonzero(connected):
            metrics["shadow_screen_collapse_reason"] = "connectivity-empty"
        elif len(region_ids) and np.count_nonzero(connected) < len(region_ids):
            metrics["shadow_screen_collapse_reason"] = "connectivity-pruned"
        else:
            metrics["shadow_screen_collapse_reason"] = "none"
        face_ids = np.asarray(geometry.get("face_ids", np.arange(count, dtype=np.int32)), dtype=np.int32).reshape(-1)
        if len(face_ids) != count:
            face_ids = np.arange(count, dtype=np.int32)
        candidate_stable = np.unique(face_ids[np.flatnonzero(connected)]).astype(np.int32, copy=False)
        initial_stable = np.asarray(state.get("shadow_screen_initial_ids", ()), dtype=np.int32).reshape(-1)
        previous_stable = np.asarray(state.get("shadow_screen_previous_ids", ()), dtype=np.int32).reshape(-1)
        if int(step) == 0:
            if len(initial_stable) == 0:
                initial_stable = candidate_stable.copy()
                state["shadow_screen_initial_ids"] = initial_stable.copy()
                action = "initial"
            else:
                candidate_stable = initial_stable.copy()
                action = "return-to-initial"
        elif int(step) > 0:
            candidate_stable = np.unique(np.concatenate((candidate_stable, initial_stable, previous_stable))).astype(np.int32, copy=False)
            action = "expand-union"
        else:
            candidate_stable = np.intersect1d(candidate_stable, initial_stable)
            if len(previous_stable) and int(state.get("shadow_screen_previous_step", 0)) > int(step):
                candidate_stable = np.intersect1d(candidate_stable, previous_stable)
            action = "shrink-intersection"
        final = np.isin(face_ids, candidate_stable) & ~hidden & ~source_hard
        if 0 <= seed < count:
            final[seed] = True
        # Refine only the current physical interface using the same immutable
        # reduced grayscale buffers.  Monotonic wheel semantics remain the
        # authority: expand may add but never remove, shrink may remove but
        # never add, and step zero records the refined baseline.
        refine_metrics = {}
        try:
            buffers_for_refine = state.get("shadow_screen_buffers")
            if isinstance(buffers_for_refine, dict):
                refined_final, refine_metrics = _fill_preview_shadow_refine_boundary(
                    state,
                    geometry,
                    buffers_for_refine,
                    metrics,
                    final,
                    seed,
                )
                if int(step) > 0:
                    refined_final |= final
                elif int(step) < 0:
                    refined_final &= final
                refined_final[seed] = True
                final = refined_final
                candidate_stable = np.unique(
                    face_ids[np.flatnonzero(final)]
                ).astype(np.int32, copy=False)
                if int(step) == 0:
                    initial_stable = candidate_stable.copy()
                    state["shadow_screen_initial_ids"] = initial_stable.copy()
        except (
            AttributeError,
            IndexError,
            KeyError,
            MemoryError,
            RuntimeError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            refine_metrics = {
                "shadow_screen_refine_reason": "refine-exception",
                "shadow_screen_refine_seed_connected": bool(final[seed]),
            }
        metrics.update(refine_metrics)
        selected_ids = np.flatnonzero(final).astype(np.int32, copy=False)
        state["shadow_screen_previous_ids"] = candidate_stable.copy()
        state["shadow_screen_previous_step"] = int(step)
        metrics["shadow_screen_monotonic_action"] = action
        metrics["shadow_screen_initial_count"] = int(len(initial_stable))
        metrics["shadow_screen_previous_count"] = int(len(previous_stable))
        metrics["shadow_screen_preserved_count"] = int(len(np.intersect1d(candidate_stable, initial_stable)))
        metrics["shadow_screen_accepted_count"] = int(len(selected_ids))
        if len(selected_ids):
            selected_x = np.clip(local_x[selected_ids], 0, roi_w - 1)
            selected_y = np.clip(local_y[selected_ids], 0, roi_h - 1)
            accepted_values = denoised[selected_y, selected_x]
            metrics["shadow_screen_accepted_luminance_min"] = float(np.min(accepted_values))
            metrics["shadow_screen_accepted_luminance_max"] = float(np.max(accepted_values))
        edge_keys = set()
        first = np.asarray(geometry.get("first", ()), dtype=np.int32).reshape(-1)
        second = np.asarray(geometry.get("second", ()), dtype=np.int32).reshape(-1)
        for pair_index, (left, right) in enumerate(zip(first, second)):
            left, right = int(left), int(right)
            if 0 <= left < count and 0 <= right < count and bool(final[left]) != bool(final[right]):
                key = _fill_preview_shadow_edge_key(geometry, pair_index)
                if key is not None:
                    edge_keys.add(key)
        state["shadow_luminance_region_edges"] = edge_keys
        metrics["shadow_screen_boundary_count"] = int(len(edge_keys))
        metrics["shadow_screen_rejected_neighbor_count"] = int(len(edge_keys))
        metrics["shadow_screen_fallback_reason"] = "roi-work-budget-provisional" if q else ("roi-cancelled" if cancelled else "ok")
        return selected_ids, metrics
    except (AttributeError, IndexError, KeyError, MemoryError, RuntimeError, TypeError, ValueError, ZeroDivisionError):
        return seed_only("screen-roi-morphology-failed")


def _fill_preview_shadow_region(
    state, local, distances, radius, seed_local, source_geometry=None,
    source_distances=None
):
    """Grow through distance-domain faces, stopping only at shadow line edges."""
    import numpy as np

    count = int(local.get("count", 0))
    metrics = {
        "shadow_capture_ok": bool(
            isinstance(state.get("shadow_capture"), dict)
            and state["shadow_capture"].get("ok")
        ),
        "shadow_capture_reason": (
            state.get("shadow_capture", {}).get("reason", "not-captured")
            if isinstance(state.get("shadow_capture"), dict)
            else "not-captured"
        ),
        "shadow_capture_size": tuple(
            state.get("shadow_capture", {}).get("capture_size", (0, 0))
            if isinstance(state.get("shadow_capture"), dict) else (0, 0)
        ),
        "shadow_luminance_min": float(state.get("shadow_capture", {}).get("luminance_min", 0.0))
        if isinstance(state.get("shadow_capture"), dict) else 0.0,
        "shadow_luminance_max": float(state.get("shadow_capture", {}).get("luminance_max", 0.0))
        if isinstance(state.get("shadow_capture"), dict) else 0.0,
        "shadow_seed_luminance": float(state.get("shadow_capture", {}).get("luminance_seed", 0.0))
        if isinstance(state.get("shadow_capture"), dict) else 0.0,
        "shadow_noise_scale": float(state.get("shadow_capture", {}).get("noise_scale", 0.0))
        if isinstance(state.get("shadow_capture"), dict) else 0.0,
        "shadow_line_pixel_count": int(state.get("shadow_capture", {}).get("line_pixel_count", 0))
        if isinstance(state.get("shadow_capture"), dict) else 0,
        "shadow_total_pairs": 0,
        "shadow_skipped_hard": 0,
        "shadow_skipped_domain": 0,
        "shadow_invalid_endpoint": 0,
        "shadow_projection_no_sample": 0,
        "shadow_line_no_hit": 0,
        "shadow_cache_edge_count": 0,
        "shadow_cache_generation": int(state.get("shadow_capture_generation", 0)),
        "shadow_tested_crossings": 0,
        "shadow_barrier_count": 0,
        "shadow_gentle_crossings_passed": 0,
        "shadow_brightening_passed": 0,
        "shadow_knee_crossings": 0,
        "shadow_knee_candidates": 0,
        "shadow_knee_adopted": 0,
        "shadow_raw_high_edges": 0,
        "shadow_raw_s_min": 0.0,
        "shadow_raw_s_median": 0.0,
        "shadow_raw_s_max": 0.0,
        "shadow_raw_s_mad": 0.0,
        "shadow_baseline_tolerance": 0.0,
        "shadow_plateau_median": 0.0,
        "shadow_plateau_mad": 0.0,
        "shadow_smoothed_face_count": 0,
        "shadow_closing_filled_gaps": 0,
        "shadow_component_count": 0,
        "shadow_post_close_component_count": 0,
        "shadow_component_lengths": (),
        "shadow_traced_sequence_count": 0,
        "shadow_junction_pair_count": 0,
        "shadow_junction_split_count": 0,
        "shadow_junction_ambiguous_count": 0,
        "shadow_junction_fallback_count": 0,
        "shadow_closing_window_physical": 0.0,
        "shadow_closing_window_n": 0,
        "shadow_closing_filled_physical": 0.0,
        "shadow_free_endpoint_count": 0,
        "shadow_closed_chain_count": 0,
        "shadow_domain_spanning_chain_count": 0,
        "shadow_adopted_edge_count": 0,
        "shadow_adopted_after_coherence": 0,
        "shadow_no_context_rejected": 0,
        "shadow_sampled_face_count": 0,
        "shadow_visible_face_count": 0,
        "shadow_unsampled_face_count": 0,
        "shadow_seed_luminance_used": False,
        "shadow_local_trend_median": 0.0,
        "shadow_local_trend_mad": 0.0,
        "shadow_connected_line_count": 0,
        "shadow_face_set_neutral": True,
        "shadow_restore_verified": bool(state.get("shadow_capture", {}).get("restore_verified", False))
        if isinstance(state.get("shadow_capture"), dict) else False,
        "shadow_reason": "distance-only-capture-unavailable",
    }
    if count <= 0:
        return np.empty(0, dtype=np.int32), metrics
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    offsets = np.asarray(local.get("offsets", ()), dtype=np.int64).reshape(-1)
    neighbors = np.asarray(local.get("neighbors", ()), dtype=np.int32).reshape(-1)
    first = np.asarray(local.get("first", ()), dtype=np.int32).reshape(-1)
    second = np.asarray(local.get("second", ()), dtype=np.int32).reshape(-1)
    if (
        len(distances) < count or len(offsets) != count + 1
        or int(offsets[-1]) != len(neighbors)
        or len(first) != len(second)
    ):
        metrics["shadow_reason"] = "shadow-graph-schema"
        return np.asarray([int(seed_local)], dtype=np.int32), metrics
    local_distances = distances[:count]
    tolerance = max(float(radius) * 1.0e-8, 1.0e-9)
    domain = np.isfinite(local_distances) & (local_distances <= float(radius) + tolerance)
    hidden = np.asarray(local.get("hidden", np.zeros(count, dtype=bool)), dtype=bool).reshape(-1)
    source_hard = np.asarray(local.get("source_hard", np.zeros(count, dtype=bool)), dtype=bool).reshape(-1)
    if len(hidden) != count:
        hidden = np.zeros(count, dtype=bool)
    if len(source_hard) != count:
        source_hard = np.zeros(count, dtype=bool)
    safety_hard = hidden | source_hard
    hard = safety_hard | ~domain
    protected = np.zeros(count, dtype=bool)
    edge_barrier = np.zeros(len(first), dtype=bool)
    shadow_luminance_cache = None
    if metrics["shadow_capture_ok"]:
        source_geometry = source_geometry if source_geometry is not None else local
        source_count = int(source_geometry.get("count", count))
        cache_distances = np.full(source_count, np.inf, dtype=np.float64)
        if source_distances is not None:
            source_values = np.asarray(source_distances, dtype=np.float64).reshape(-1)
            if len(source_values) >= source_count:
                cache_distances[:] = source_values[:source_count]
        local_global_ids = np.asarray(
            local.get("_global_face_ids", np.arange(count, dtype=np.int32)),
            dtype=np.int64,
        ).reshape(-1)
        valid_map = np.zeros(count, dtype=bool)
        map_limit = min(count, len(local_global_ids))
        if map_limit:
            valid_map[:map_limit] = (
                (local_global_ids[:map_limit] >= 0)
                & (local_global_ids[:map_limit] < source_count)
            )
        if np.any(valid_map):
            cache_distances[local_global_ids[:count][valid_map]] = local_distances[valid_map]
        else:
            cache_distances[: min(source_count, count)] = local_distances[: min(source_count, count)]
        seed_geometry_face = (
            int(local_global_ids[int(seed_local)])
            if 0 <= int(seed_local) < len(local_global_ids)
            else int(seed_local)
        )
        shadow_luminance_cache = _fill_preview_shadow_luminance_edge_cache(
            state, source_geometry, cache_distances, seed_geometry_face
        )
        metrics["shadow_total_pairs"] = int(shadow_luminance_cache.get("total_pairs", 0))
        metrics["shadow_cache_edge_count"] = int(
            len(shadow_luminance_cache.get("edges", {}))
        )
        metrics["shadow_skipped_hard"] = int(shadow_luminance_cache.get("skipped_hard", 0))
        metrics["shadow_tested_crossings"] = int(shadow_luminance_cache.get("tested", 0))
        metrics["shadow_sampled_face_count"] = int(
            _fill_preview_shadow_face_luminance_cache(state, source_geometry).get("sampled", 0)
        )
        metrics["shadow_visible_face_count"] = int(
            _fill_preview_shadow_face_luminance_cache(state, source_geometry).get("visible", 0)
        )
        metrics["shadow_unsampled_face_count"] = int(
            _fill_preview_shadow_face_luminance_cache(state, source_geometry).get("unsampled", 0)
        )
        metrics["shadow_seed_luminance_used"] = bool(
            shadow_luminance_cache.get("seed_luminance_used", False)
        )
        if shadow_luminance_cache.get("seed_luminance") is not None:
            metrics["shadow_seed_luminance"] = float(
                shadow_luminance_cache["seed_luminance"]
            )
        metrics["shadow_gentle_crossings_passed"] = int(
            shadow_luminance_cache.get("gentle", 0)
        )
        metrics["shadow_brightening_passed"] = int(
            shadow_luminance_cache.get("brightening", 0)
        )
        metrics["shadow_knee_candidates"] = int(
            shadow_luminance_cache.get("knee_candidates", 0)
        )
        metrics["shadow_knee_adopted"] = int(
            shadow_luminance_cache.get("knee_adopted", 0)
        )
        metrics["shadow_knee_crossings"] = int(
            shadow_luminance_cache.get("knee_adopted", 0)
        )
        metrics["shadow_raw_high_edges"] = int(
            shadow_luminance_cache.get("raw_high_edges", 0)
        )
        metrics["shadow_raw_s_min"] = float(
            shadow_luminance_cache.get("raw_s_min", 0.0)
        )
        metrics["shadow_raw_s_median"] = float(
            shadow_luminance_cache.get("raw_s_median", 0.0)
        )
        metrics["shadow_raw_s_max"] = float(
            shadow_luminance_cache.get("raw_s_max", 0.0)
        )
        metrics["shadow_raw_s_mad"] = float(
            shadow_luminance_cache.get("raw_s_mad", 0.0)
        )
        metrics["shadow_baseline_tolerance"] = float(
            shadow_luminance_cache.get("baseline_tolerance", 0.0)
        )
        metrics["shadow_plateau_median"] = float(
            shadow_luminance_cache.get("plateau_median", 0.0) or 0.0
        )
        metrics["shadow_plateau_mad"] = float(
            shadow_luminance_cache.get("plateau_mad", 0.0)
        )
        metrics["shadow_smoothed_face_count"] = int(
            shadow_luminance_cache.get("smoothed_count", 0)
        )
        metrics["shadow_closing_filled_gaps"] = int(
            shadow_luminance_cache.get("closing_filled_gaps", 0)
        )
        metrics["shadow_component_count"] = int(
            shadow_luminance_cache.get("component_count", 0)
        )
        metrics["shadow_post_close_component_count"] = int(
            shadow_luminance_cache.get("post_close_component_count", 0)
        )
        metrics["shadow_component_lengths"] = tuple(
            float(value) for value in shadow_luminance_cache.get("component_lengths", ())
        )
        metrics["shadow_adopted_after_coherence"] = int(
            shadow_luminance_cache.get("adopted_after_coherence", 0)
        )
        metrics["shadow_no_context_rejected"] = int(
            shadow_luminance_cache.get("no_context_rejected", 0)
        )
        metrics["shadow_traced_sequence_count"] = int(
            shadow_luminance_cache.get("traced_sequence_count", 0)
        )
        metrics["shadow_junction_pair_count"] = int(
            shadow_luminance_cache.get("junction_pair_count", 0)
        )
        metrics["shadow_junction_split_count"] = int(
            shadow_luminance_cache.get("junction_split_count", 0)
        )
        metrics["shadow_junction_ambiguous_count"] = int(
            shadow_luminance_cache.get("junction_ambiguous_count", 0)
        )
        metrics["shadow_junction_fallback_count"] = int(
            shadow_luminance_cache.get("junction_fallback_count", 0)
        )
        metrics["shadow_closing_window_physical"] = float(
            shadow_luminance_cache.get("closing_window_physical", 0.0)
        )
        metrics["shadow_closing_window_n"] = int(
            shadow_luminance_cache.get("closing_window_n", 0)
        )
        metrics["shadow_closing_filled_physical"] = float(
            shadow_luminance_cache.get("closing_filled_physical", 0.0)
        )
        metrics["shadow_free_endpoint_count"] = int(
            shadow_luminance_cache.get("free_endpoint_count", 0)
        )
        metrics["shadow_closed_chain_count"] = int(
            shadow_luminance_cache.get("closed_chain_count", 0)
        )
        metrics["shadow_domain_spanning_chain_count"] = int(
            shadow_luminance_cache.get("domain_spanning_chain_count", 0)
        )
        metrics["shadow_adopted_edge_count"] = int(
            shadow_luminance_cache.get("adopted_edge_count", 0)
        )
        metrics["shadow_local_trend_median"] = float(
            shadow_luminance_cache.get("local_trend_median", 0.0)
        )
        metrics["shadow_local_trend_mad"] = float(
            shadow_luminance_cache.get("local_trend_mad", 0.0)
        )
        for pair_index in range(len(first)):
            left, right = int(first[pair_index]), int(second[pair_index])
            if safety_hard[left] or safety_hard[right]:
                metrics["shadow_skipped_hard"] += 1
                continue
            key = _fill_preview_shadow_edge_key(local, pair_index)
            entry = (
                shadow_luminance_cache.get("edges", {}).get(key, {})
                if isinstance(shadow_luminance_cache, dict)
                else {}
            )
            tested = int(entry.get("tested", 0))
            if tested and bool(entry.get("barrier", False)):
                edge_barrier[pair_index] = True
                metrics["shadow_barrier_count"] += 1
            # Gentle/brightening totals are computed once by the immutable
            # source edge map above; the local slice must not count them again.
    cache = state.get("shadow_luminance_edge_cache")
    selected = np.zeros(count, dtype=bool)
    seed_local = int(seed_local)
    if seed_local < 0 or seed_local >= count or hard[seed_local]:
        metrics["shadow_reason"] = "shadow-seed-unsafe"
        return np.empty(0, dtype=np.int32), metrics
    selected[seed_local] = True
    pair_lookup = {}
    for pair_index, (left, right) in enumerate(zip(first, second)):
        pair_lookup.setdefault((int(left), int(right)), []).append(pair_index)
        pair_lookup.setdefault((int(right), int(left)), []).append(pair_index)
    pending = [seed_local]
    while pending:
        face = int(pending.pop())
        for edge_position in range(int(offsets[face]), int(offsets[face + 1])):
            neighbor = int(neighbors[edge_position])
            if neighbor < 0 or neighbor >= count or selected[neighbor] or hard[neighbor]:
                continue
            blocked = any(edge_barrier[p] for p in pair_lookup.get((face, neighbor), ()))
            if blocked or protected[neighbor]:
                continue
            selected[neighbor] = True
            pending.append(neighbor)
    metrics["shadow_reason"] = "shadow-barrier-region" if metrics["shadow_capture_ok"] else "distance-only-provisional"
    metrics["shadow_connected_line_count"] = int(metrics["shadow_barrier_count"] > 0)
    return np.flatnonzero(selected).astype(np.int32), metrics


def _fill_preview_gray_local_threshold(values):
    """Return fixed-window robust hysteresis values for one local context."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.08, 0.06, 0.0, 0.0, "insufficient-local-window"
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    scale = max(mad, 1.0e-3)
    if mad <= 1.0e-6:
        high = max(median, 0.08)
        low = max(0.90 * median, 0.06)
    else:
        high = max(median + 2.0 * scale, 0.08)
        low = max(median + scale, 0.06)
        signal_range = float(np.ptp(values))
        if signal_range > 0.0:
            high = min(high, median + 0.75 * signal_range)
            low = min(low, high)
    return high, low, median, mad, "full-local-window" if len(values) >= 6 else "insufficient-local-window"


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


def _fill_preview_shape_boundary_mask(crossing, outside_distance, radius, edge_indices):
    """Classify valid candidate crossings using the shared distance tolerance.

    An edge crossing whose outside face is inside the current candidate
    distance is a shape/region boundary.  The fine partition barrier is a
    shape signal for region ownership, but it must not demote an otherwise
    valid in-range crossing to the distance-boundary color.
    """
    import numpy as np

    crossing = np.asarray(crossing, dtype=bool).reshape(-1)
    outside_distance = np.asarray(outside_distance, dtype=np.float64).reshape(-1)
    edge_indices = np.asarray(edge_indices, dtype=np.int32).reshape(-1)
    if (
        len(crossing) == 0
        or len(crossing) != len(outside_distance)
        or len(crossing) != len(edge_indices)
    ):
        return np.zeros(len(crossing), dtype=bool)
    tolerance = max(float(radius) * 1.0e-8, 1.0e-9)
    return (
        crossing
        & (edge_indices >= 0)
        & np.isfinite(outside_distance)
        & (outside_distance <= float(radius) + tolerance)
    )


def _fill_preview_should_use_green_boundary(
    shape_boundary_mask,
    boundary_mask=None,
):
    """Return whether every currently displayed boundary segment is orange.

    This is a display-only predicate over the existing cyan/orange
    classification.  It deliberately does not inspect candidate faces,
    wheel limits, or a future stage: the green state only communicates that
    the current visible boundary has no cyan segment.
    """
    import numpy as np

    try:
        shape = np.asarray(shape_boundary_mask, dtype=bool).reshape(-1)
        if boundary_mask is not None:
            boundary = np.asarray(boundary_mask, dtype=bool).reshape(-1)
            if len(boundary) != len(shape):
                return False
            shape = shape[boundary]
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(len(shape) and np.all(shape))


def _fill_preview_resolve_enclosed_components(
    geometry, distances, radius, selected_ids
):
    """Fill only enclosed missing components with at most 100 outer vertices.

    This is the one compact-domain resolver used by the preview.  It has no
    terminal/"tip" classification and does not use boundary colors.  Any
    route to a distance, hidden, crop, mesh, non-manifold, or unmatched graph
    boundary marks the complete missing component as open.
    """
    import numpy as np

    metrics = {
        "enclosed_component_faces": 0,
        "enclosed_component_count": 0,
        "enclosed_component_filled_faces": 0,
        "enclosed_component_filled_components": 0,
        "enclosed_component_max_outer_vertices": 0,
        "enclosed_component_rejected_large_components": 0,
        "enclosed_component_rejected_face_ids": (),
        "enclosed_component_reason": "ok",
    }
    try:
        count = int(geometry["count"])
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        edge_counts = np.asarray(geometry["face_edge_counts"], dtype=np.int32).reshape(-1)
        hidden = np.asarray(geometry["hidden"], dtype=bool).reshape(-1)
        offsets = np.asarray(geometry["offsets"], dtype=np.int64).reshape(-1)
        neighbors = np.asarray(geometry["neighbors"], dtype=np.int32).reshape(-1)
        face_vertices = _fill_preview_face_vertex_sequence(geometry)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        metrics["enclosed_component_reason"] = "schema"
        return np.empty(0, dtype=np.int32), metrics
    if (
        count <= 0
        or len(distances) != count
        or len(edge_counts) != count
        or len(hidden) != count
        or len(offsets) != count + 1
        or int(offsets[-1]) != len(neighbors)
        or len(face_vertices) != count
        or np.any(edge_counts <= 0)
        or np.any(offsets[1:] < offsets[:-1])
        or np.any(neighbors < 0)
        or np.any(neighbors >= count)
    ):
        metrics["enclosed_component_reason"] = "schema"
        return np.empty(0, dtype=np.int32), metrics
    tolerance = max(float(radius) * 1.0e-8, 1.0e-9)
    domain = (
        np.isfinite(distances)
        & (distances <= float(radius) + tolerance)
        & ~hidden
    )

    selected = np.zeros(count, dtype=bool)
    try:
        selected_ids = np.asarray(selected_ids, dtype=np.int32).reshape(-1)
    except (TypeError, ValueError):
        metrics["enclosed_component_reason"] = "selected-schema"
        return np.empty(0, dtype=np.int32), metrics
    selected_ids = selected_ids[(selected_ids >= 0) & (selected_ids < count)]
    selected[selected_ids] = True
    missing = domain & ~selected
    metrics["enclosed_component_faces"] = int(np.count_nonzero(missing))
    if not np.any(missing):
        return np.empty(0, dtype=np.int32), metrics
    degree = np.diff(offsets)
    outside = missing & (degree < edge_counts)
    touches_selected = np.zeros(count, dtype=bool)
    for face in np.flatnonzero(missing):
        face = int(face)
        for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
            neighbor = int(neighbor)
            if selected[neighbor]:
                touches_selected[face] = True
            elif not domain[neighbor]:
                outside[face] = True
    # Propagate protected-boundary reachability through missing components.
    external = np.zeros(count, dtype=bool)
    pending = deque(int(face) for face in np.flatnonzero(outside))
    external[outside] = True
    while pending:
        face = pending.popleft()
        for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
            neighbor = int(neighbor)
            if missing[neighbor] and not external[neighbor]:
                external[neighbor] = True
                pending.append(neighbor)
    enclosed = missing & ~external
    fills = []
    rejected_ids = []
    visited = np.zeros(count, dtype=bool)
    component_count = 0
    filled_components = 0
    max_outer = 0
    rejected_large_components = 0
    for start_face in np.flatnonzero(enclosed):
        start_face = int(start_face)
        if visited[start_face]:
            continue
        component_count += 1
        visited[start_face] = True
        pending = [start_face]
        component = []
        interface_faces = set()
        while pending:
            face = int(pending.pop())
            component.append(face)
            for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
                neighbor = int(neighbor)
                if selected[neighbor]:
                    interface_faces.add(neighbor)
                elif enclosed[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    pending.append(neighbor)
        if not interface_faces:
            continue
        component_vertices = set()
        interface_vertices = set()
        valid_vertices = True
        try:
            for face in component:
                values = tuple(int(value) for value in face_vertices[face])
                if not values:
                    valid_vertices = False
                    break
                component_vertices.update(values)
            if valid_vertices:
                for face in interface_faces:
                    interface_vertices.update(
                        int(value) for value in face_vertices[face]
                    )
        except (IndexError, TypeError, ValueError):
            valid_vertices = False
        if not valid_vertices:
            continue
        outer_count = int(len(component_vertices - interface_vertices))
        max_outer = max(max_outer, outer_count)
        if outer_count <= 100:
            fills.extend(component)
            filled_components += 1
        else:
            rejected_ids.extend(component)
            rejected_large_components += 1
    metrics.update(
        {
            "enclosed_component_count": int(component_count),
            "enclosed_component_filled_faces": int(len(fills)),
            "enclosed_component_filled_components": int(filled_components),
            "enclosed_component_max_outer_vertices": int(max_outer),
            "enclosed_component_rejected_large_components": int(
                rejected_large_components
            ),
            "enclosed_component_rejected_face_ids": tuple(sorted(set(rejected_ids))),
        }
    )
    return np.unique(np.asarray(fills, dtype=np.int32)), metrics


def _fill_preview_boundary_topology_metrics(
    mesh, pair_edge_indices, boundary, pair_v0=None, pair_v1=None
):
    """Summarize the copied physical boundary without changing selection.

    The normal progressive path draws from the full prepared graph.  The
    retained pair endpoint arrays are the authoritative physical-edge
    identity; ``pair_edge_indices`` is only a legacy mesh-edge hint and may
    be stale after a cursor-expanded graph is rebuilt.  Invalid/crop rows are
    counted explicitly rather than silently dropped.
    """
    try:
        import numpy as np

        pair_edge_indices = np.asarray(pair_edge_indices, dtype=np.int32).reshape(-1)
        boundary = np.asarray(boundary, dtype=bool).reshape(-1)
        pair_v0 = np.asarray(
            pair_v0 if pair_v0 is not None else (), dtype=np.int32
        ).reshape(-1)
        pair_v1 = np.asarray(
            pair_v1 if pair_v1 is not None else (), dtype=np.int32
        ).reshape(-1)
        use_pair_vertices = len(pair_v0) == len(boundary) and len(pair_v1) == len(boundary)
        if len(pair_edge_indices) != len(boundary) and not use_pair_vertices:
            return {
                "boundary_components": 0,
                "boundary_free_endpoints": 0,
                "boundary_missing_neighbor_fallback_count": int(np.count_nonzero(boundary)),
                "boundary_topology_reason": "schema",
            }
        rows = []
        for row in np.flatnonzero(boundary):
            row = int(row)
            if use_pair_vertices:
                if int(pair_v0[row]) >= 0 and int(pair_v1[row]) >= 0:
                    rows.append(row)
            elif int(pair_edge_indices[row]) >= 0:
                rows.append(row)
        missing = int(np.count_nonzero(boundary)) - len(rows)
        if not rows or (mesh is None and not use_pair_vertices):
            return {
                "boundary_components": 0,
                "boundary_free_endpoints": 0,
                "boundary_missing_neighbor_fallback_count": missing,
                "boundary_topology_reason": "no-physical-edge-rows",
            }
        parent = {row: row for row in rows}

        def find(value):
            value = int(value)
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left, right):
            left, right = find(left), find(right)
            if left != right:
                parent[right] = left

        vertex_rows = {}
        for row in rows:
            if use_pair_vertices:
                vertices = (int(pair_v0[row]), int(pair_v1[row]))
            else:
                edge = mesh.edges[int(pair_edge_indices[row])]
                vertices = tuple(int(value) for value in edge.vertices)
            for vertex in vertices:
                vertex_rows.setdefault(vertex, []).append(row)
        for incident in vertex_rows.values():
            for row in incident[1:]:
                union(incident[0], row)
        endpoint_count = sum(1 for incident in vertex_rows.values() if len(incident) == 1)
        return {
            "boundary_components": int(len({find(row) for row in rows})),
            "boundary_free_endpoints": int(endpoint_count),
            "boundary_missing_neighbor_fallback_count": int(max(0, missing)),
            "boundary_topology_reason": (
                "pair-vertex-physical-interface"
                if use_pair_vertices
                else "validated-mesh-edge-interface"
            ),
        }
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return {
            "boundary_components": 0,
            "boundary_free_endpoints": 0,
            "boundary_missing_neighbor_fallback_count": int(
                np.count_nonzero(boundary) if "np" in locals() else 0
            ),
            "boundary_topology_reason": "schema-exception",
        }


def _fill_preview_resolve_sandwiched_bands(
    geometry, distances, radius, selected_ids
):
    """Close narrow missing face-graph bands without changing the base.

    The resolver intentionally does not classify a terminal, a valley, or a
    boundary colour.  It computes graph morphology from the immutable base
    candidate in the current radius domain.  A newly created component is
    accepted only when its original base interface consists of at least two
    spatially separated boundary arcs.  Thus a one-sided bulge cannot be
    mistaken for a shore pair, while a long narrow gap is closed independently
    of its length.  Each scale is recomputed from the same base; accepted
    faces never seed a later scale.
    """
    import numpy as np

    metrics = {
        "sandwiched_band_faces": 0,
        "sandwiched_band_components": 0,
        "sandwiched_band_interface_edges": 0,
        "sandwiched_band_closing_attempts": 0,
        "sandwiched_band_closing_components": 0,
        "sandwiched_band_closing_added_faces": 0,
        "sandwiched_band_filled_faces": 0,
        "sandwiched_band_filled_components": 0,
        "sandwiched_band_contact_arcs": 0,
        "sandwiched_band_rejected_protected": 0,
        "sandwiched_band_rejected_single_arc": 0,
        "sandwiched_band_rejected_hard_distance": 0,
        "sandwiched_band_closing_max_n": 0,
        "sandwiched_band_graph_walks": 0,
        "sandwiched_band_reason": "ok",
    }
    try:
        count = int(geometry["count"])
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        edge_counts = np.asarray(
            geometry["face_edge_counts"], dtype=np.int32
        ).reshape(-1)
        hidden = np.asarray(geometry["hidden"], dtype=bool).reshape(-1)
        offsets = np.asarray(geometry["offsets"], dtype=np.int64).reshape(-1)
        neighbors = np.asarray(geometry["neighbors"], dtype=np.int32).reshape(-1)
        first = np.asarray(geometry["first"], dtype=np.int32).reshape(-1)
        second = np.asarray(geometry["second"], dtype=np.int32).reshape(-1)
        pair_points = np.asarray(
            geometry["pair_edge_points"], dtype=np.float64
        )
        pair_edge_indices = np.asarray(
            geometry["pair_edge_indices"], dtype=np.int32
        ).reshape(-1)
        centers = np.asarray(geometry["centers"], dtype=np.float64)
        normals = np.asarray(geometry["normals"], dtype=np.float64)
        face_vertices = _fill_preview_face_vertex_sequence(geometry)
        world_vertices = np.asarray(
            geometry["world_vertices"], dtype=np.float64
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        metrics["sandwiched_band_reason"] = "schema"
        return np.empty(0, dtype=np.int32), metrics
    if (
        count <= 0
        or len(distances) != count
        or len(edge_counts) != count
        or len(hidden) != count
        or len(offsets) != count + 1
        or np.any(offsets[1:] < offsets[:-1])
        or int(offsets[-1]) != len(neighbors)
        or len(first) != len(second)
        or pair_points.shape != (len(first), 2, 3)
        or len(pair_edge_indices) != len(first)
        or np.any(pair_edge_indices < 0)
        or len(set(int(value) for value in pair_edge_indices)) != len(pair_edge_indices)
        or len(centers) != count
        or normals.shape != (count, 3)
        or len(face_vertices) != count
        or world_vertices.ndim != 2
        or world_vertices.shape[1] != 3
        or np.any(neighbors < 0)
        or np.any(neighbors >= count)
        or np.any(first < 0)
        or np.any(second < 0)
        or np.any(first >= count)
        or np.any(second >= count)
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(normals))
        or not np.all(np.isfinite(pair_points))
        or not np.all(np.isfinite(world_vertices))
    ):
        metrics["sandwiched_band_reason"] = "schema"
        return np.empty(0, dtype=np.int32), metrics
    try:
        selected_ids = np.asarray(selected_ids, dtype=np.int32).reshape(-1)
    except (TypeError, ValueError):
        metrics["sandwiched_band_reason"] = "selected-schema"
        return np.empty(0, dtype=np.int32), metrics

    selected = np.zeros(count, dtype=bool)
    selected_ids = selected_ids[
        (selected_ids >= 0) & (selected_ids < count)
    ]
    selected[selected_ids] = True
    tolerance = max(float(radius) * 1.0e-8, 1.0e-9)
    domain = (
        np.isfinite(distances)
        & (distances <= float(radius) + tolerance)
        & ~hidden
    )
    # Closing is defined on the induced graph of the requested radius.  A
    # finite face beyond that radius is deliberately not a dilation target and
    # is not an erosion requirement: this is a neutral/reflecting radius cut,
    # rather than a zero-valued boundary.  Crop/unknown boundaries are kept
    # distinct and fail closed below using the original polygon edge count.
    base = selected & domain
    missing = domain & ~base
    metrics["sandwiched_band_faces"] = int(np.count_nonzero(missing))
    if not np.any(missing) or not np.any(base):
        return np.empty(0, dtype=np.int32), metrics

    degree = np.diff(offsets).astype(np.int64, copy=False)
    pair_map = {}
    for pair_index, (left, right) in enumerate(zip(first, second)):
        left = int(left)
        right = int(right)
        if left == right:
            continue
        key = (min(left, right), max(left, right))
        # Duplicate graph rows are malformed for boundary evidence.  Keep the
        # first source-of-truth edge and reject duplicate interfaces below.
        pair_map.setdefault(key, int(pair_index))

    # A finite face outside radius is a clipping endpoint, not a hard
    # protection boundary.  Hidden, non-finite, missing-edge, and malformed
    # graph links remain barriers and cannot participate in Closing.  A degree
    # deficit on any adjacent prepared face is also unknown: it may be a crop,
    # non-manifold, or unmatched seam, so it is never crossed or treated as a
    # soft radius cut.
    hard_face = (
        hidden
        | ~np.isfinite(distances)
        | (degree < edge_counts)
    )
    protected = missing & hard_face
    for face in np.flatnonzero(missing):
        face = int(face)
        for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
            neighbor = int(neighbor)
            if bool(hard_face[neighbor]):
                protected[face] = True

    # Closing is evaluated only for N=1..max_n.  Compute shortest graph-hop
    # distance to a hard/unknown boundary once, from all relevant finite
    # sources.  A finite face outside the radius is traversable for this
    # *diagnostic* only; it remains outside the induced Closing graph.  Thus a
    # normal radius clip contributes no source, while a crop/non-manifold or
    # hidden/non-finite continuation is found at its true shortest depth.
    max_n = 8
    finite_faces = np.isfinite(distances)
    hard_distance = np.full(count, max_n + 1, dtype=np.int32)
    hard_queue = deque()
    for face in np.flatnonzero(finite_faces & hard_face):
        face = int(face)
        hard_distance[face] = 0
        hard_queue.append(face)
    # Non-finite/hidden hard faces are not traversed, but their finite
    # neighbors still need a distance-one source so a two-hop crop is not
    # mistaken for a soft radius cut.
    for face in np.flatnonzero(finite_faces):
        face = int(face)
        if hard_distance[face] == 0:
            continue
        start = int(offsets[face])
        end = int(offsets[face + 1])
        if any(bool(hard_face[int(neighbor)]) and not bool(finite_faces[int(neighbor)])
               for neighbor in neighbors[start:end]):
            hard_distance[face] = 1
            hard_queue.append(face)
    while hard_queue:
        face = int(hard_queue.popleft())
        depth = int(hard_distance[face])
        if depth >= max_n:
            continue
        next_depth = depth + 1
        start = int(offsets[face])
        end = int(offsets[face + 1])
        for neighbor in neighbors[start:end]:
            neighbor = int(neighbor)
            if not finite_faces[neighbor] or next_depth >= int(hard_distance[neighbor]):
                continue
            hard_distance[neighbor] = next_depth
            hard_queue.append(neighbor)

    component_of = np.full(count, -1, dtype=np.int32)
    components = []
    for start_face in np.flatnonzero(missing):
        start_face = int(start_face)
        if component_of[start_face] >= 0:
            continue
        component_id = len(components)
        component_of[start_face] = component_id
        pending = [start_face]
        component = []
        while pending:
            face = int(pending.pop())
            component.append(face)
            for neighbor in neighbors[
                int(offsets[face]) : int(offsets[face + 1])
            ]:
                neighbor = int(neighbor)
                if missing[neighbor] and component_of[neighbor] < 0:
                    component_of[neighbor] = component_id
                    pending.append(neighbor)
        components.append(np.asarray(component, dtype=np.int32))
    metrics["sandwiched_band_components"] = int(len(components))
    component_protected = np.asarray(
        [bool(np.any(protected[component])) for component in components],
        dtype=bool,
    )
    component_hard_distance = np.asarray(
        [
            int(np.min(hard_distance[component])) if len(component)
            else max_n + 1
            for component in components
        ],
        dtype=np.int32,
    )
    # Cache the component-level hard decision once; the N loop below only
    # checks the candidate component's precomputed per-face distance.
    component_protected |= component_hard_distance <= 0
    # Per-face local scale is used only to choose a stable endpoint tolerance.
    # It never creates a global acceptance threshold shared by components.
    face_scale_cache = {}

    def face_scale(face):
        face = int(face)
        cached = face_scale_cache.get(face)
        if cached is not None:
            return cached
        scale = 0.0
        try:
            ids = np.asarray(
                tuple(int(value) for value in face_vertices[face]),
                dtype=np.int64,
            )
            if len(ids) >= 2 and not np.any(ids < 0) and not np.any(
                ids >= len(world_vertices)
            ):
                points = world_vertices[ids]
                lengths = np.linalg.norm(
                    points - np.roll(points, -1, axis=0), axis=1
                )
                lengths = lengths[
                    np.isfinite(lengths) & (lengths > 1.0e-12)
                ]
                if len(lengths):
                    scale = float(np.percentile(lengths, 25.0))
        except (IndexError, TypeError, ValueError):
            scale = 0.0
        face_scale_cache[face] = scale
        return scale

    component_interfaces = []
    total_interfaces = 0
    rejected_protected = 0
    for component_index, component in enumerate(components):
        records = []
        malformed = False
        for missing_face in component:
            missing_face = int(missing_face)
            for neighbor in neighbors[
                int(offsets[missing_face]) : int(offsets[missing_face + 1])
            ]:
                neighbor = int(neighbor)
                if not base[neighbor]:
                    continue
                pair_index = pair_map.get(
                    (min(missing_face, neighbor), max(missing_face, neighbor))
                )
                if pair_index is None:
                    malformed = True
                    continue
                edge = pair_points[pair_index]
                if edge.shape != (2, 3) or not np.all(np.isfinite(edge)):
                    malformed = True
                    continue
                edge_length = float(np.linalg.norm(edge[1] - edge[0]))
                if edge_length <= 1.0e-12:
                    malformed = True
                    continue
                records.append(
                    {
                        "missing_face": missing_face,
                        "selected_face": neighbor,
                        "pair_index": int(pair_index),
                        "edge": edge.copy(),
                        "edge_length": edge_length,
                    }
                )
        total_interfaces += len(records)
        if bool(component_protected[component_index]):
            rejected_protected += 1
            component_interfaces.append(())
        elif malformed or len(records) < 2:
            component_interfaces.append(())
        else:
            component_interfaces.append(tuple(records))
    metrics["sandwiched_band_interface_edges"] = int(total_interfaces)
    metrics["sandwiched_band_rejected_protected"] = int(rejected_protected)

    eligible = [
        index
        for index, records in enumerate(component_interfaces)
        if records and not bool(component_protected[index])
    ]
    if not eligible:
        metrics["sandwiched_band_reason"] = (
            "protected-or-no-interface"
            if rejected_protected
            else "no-interface"
        )
        return np.empty(0, dtype=np.int32), metrics

    allowed = domain & ~hard_face

    graph_walks = 0

    def dilate_once(seed):
        nonlocal graph_walks
        graph_walks += 1
        result = np.asarray(seed, dtype=bool)
        expanded = result.copy()
        # Protected faces may remain in the immutable base for the sake of
        # base preservation, but they must never act as dilation seeds.
        for face in np.flatnonzero(result & ~hard_face):
            start = int(offsets[int(face)])
            end = int(offsets[int(face) + 1])
            for neighbor in neighbors[start:end]:
                neighbor = int(neighbor)
                if allowed[neighbor]:
                    expanded[neighbor] = True
        return expanded

    def erode(mask, steps):
        nonlocal graph_walks
        result = np.asarray(mask, dtype=bool).copy()
        for _ in range(int(steps)):
            graph_walks += 1
            reduced = result.copy()
            for face in np.flatnonzero(result):
                start = int(offsets[int(face)])
                end = int(offsets[int(face) + 1])
                for neighbor in neighbors[start:end]:
                    neighbor = int(neighbor)
                    # Radius-cut neighbors are outside the induced graph and
                    # therefore neutral for erosion.  Every domain neighbor
                    # must still survive and be allowed.
                    if domain[neighbor] and (
                        not allowed[neighbor] or not result[neighbor]
                    ):
                        reduced[int(face)] = False
                        break
            result = reduced
        return result

    def endpoint_key(point, quantum):
        return tuple(
            int(round(float(value) / quantum)) for value in point
        )

    def contact_arcs(new_faces, records):
        """Group original-base contact edges into separated arcs."""
        if not records:
            return (), "none"
        segments = []
        for face in new_faces:
            face = int(face)
            for neighbor in neighbors[
                int(offsets[face]) : int(offsets[face + 1])
            ]:
                neighbor = int(neighbor)
                if not base[neighbor]:
                    continue
                pair_index = pair_map.get(
                    (min(face, neighbor), max(face, neighbor))
                )
                if pair_index is None:
                    return (), "malformed"
                edge = pair_points[pair_index]
                if edge.shape != (2, 3) or not np.all(np.isfinite(edge)):
                    return (), "malformed"
                edge_length = float(np.linalg.norm(edge[1] - edge[0]))
                if edge_length <= 1.0e-12:
                    return (), "malformed"
                segments.append(
                    {
                        "p0": edge[0],
                        "p1": edge[1],
                        "selected_face": neighbor,
                        "length": edge_length,
                    }
                )
        if not segments:
            return (), "none"
        scale_values = np.asarray(
            [
                max(
                    item["length"],
                    face_scale(item["selected_face"]),
                )
                for item in segments
            ],
            dtype=np.float64,
        )
        scale_values = scale_values[
            np.isfinite(scale_values) & (scale_values > 1.0e-12)
        ]
        quantum = max(
            (float(np.median(scale_values)) if len(scale_values) else 1.0)
            * 1.0e-6,
            1.0e-9,
        )
        parent = list(range(len(segments)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left, right):
            left = find(left)
            right = find(right)
            if left != right:
                parent[right] = left

        endpoint_map = {}
        for index, item in enumerate(segments):
            for point in (item["p0"], item["p1"]):
                endpoint_map.setdefault(
                    endpoint_key(point, quantum), []
                ).append(index)
        for indices in endpoint_map.values():
            if indices:
                first_index = indices[0]
                for index in indices[1:]:
                    union(first_index, index)
        groups = {}
        for index in range(len(segments)):
            groups.setdefault(find(index), []).append(index)
        arcs = []
        for indices in groups.values():
            selected_faces = set(
                int(segments[index]["selected_face"]) for index in indices
            )
            points = np.asarray(
                [
                    point
                    for index in indices
                    for point in (segments[index]["p0"], segments[index]["p1"])
                ],
                dtype=np.float64,
            )
            arcs.append(
                {
                    "indices": tuple(indices),
                    "selected_faces": selected_faces,
                    "center": np.mean(points, axis=0),
                }
            )
        if len(arcs) < 2:
            return tuple(arcs), "single"
        separated = []
        for first_index, first_arc in enumerate(arcs):
            for second_index in range(first_index + 1, len(arcs)):
                second_arc = arcs[second_index]
                span = float(
                    np.linalg.norm(
                        second_arc["center"] - first_arc["center"]
                    )
                )
                # Endpoint connectivity already defines one contact arc.  Do
                # not compare arc spacing with the long edge of an
                # anisotropic quad; only discard numerical duplicate arcs.
                if span <= quantum * 4.0:
                    continue
                direct_adjacency = False
                for first_face in first_arc["selected_faces"]:
                    start = int(offsets[first_face])
                    end = int(offsets[first_face + 1])
                    if any(
                        int(value) in second_arc["selected_faces"]
                        for value in neighbors[start:end]
                    ):
                        direct_adjacency = True
                        break
                if not direct_adjacency:
                    separated.append((first_index, second_index, span))
        if not separated:
            return tuple(arcs), "single"
        return tuple(arcs), "separated"

    accepted = np.zeros(count, dtype=bool)
    accepted_component_count = 0
    rejected_single_arc = 0
    contact_arc_count = 0
    closing_components = 0
    attempts = 0
    visit_marks = np.zeros(count, dtype=np.int32)
    candidate_marks = np.zeros(count, dtype=np.int32)
    visit_token = 0
    candidate_token = 0
    # Build the dilation frontier once.  Each erosion still starts from the
    # immutable-base dilation at this scale, so scales remain independent and
    # accepted faces never seed a later one.  This reduces the former
    # 2*(1+...+8)=72 graph walks to 8+36=44 without changing the Closing.
    dilation = base.copy()
    rejected_hard_distance = 0
    for steps in range(1, max_n + 1):
        dilation = dilate_once(dilation)
        closing = erode(dilation, steps)
        # Each scale starts from the immutable base.  Existing accepted faces
        # remain output-only state; they are deliberately not removed from
        # this Closing result, so a wider band in the same missing component
        # can be validated and added at a later N.
        new_faces = closing & missing & domain
        metrics["sandwiched_band_closing_max_n"] = int(steps)
        for component_index in eligible:
            component = components[component_index]
            component_new_faces = component[new_faces[component]]
            if len(component_new_faces) == 0:
                continue
            attempts += 1
            candidate_token += 1
            candidate_marks[component_new_faces] = candidate_token
            for start_face in component_new_faces:
                start_face = int(start_face)
                if visit_marks[start_face] == candidate_token:
                    continue
                closing_components += 1
                visit_marks[start_face] = candidate_token
                pending = [start_face]
                new_component = []
                while pending:
                    face = int(pending.pop())
                    new_component.append(face)
                    for neighbor in neighbors[
                        int(offsets[face]) : int(offsets[face + 1])
                    ]:
                        neighbor = int(neighbor)
                        if (
                            candidate_marks[neighbor] == candidate_token
                            and visit_marks[neighbor] != candidate_token
                        ):
                            visit_marks[neighbor] = candidate_token
                            pending.append(neighbor)
                arcs, arc_status = contact_arcs(
                    np.asarray(new_component, dtype=np.int32),
                    component_interfaces[component_index],
                )
                if arc_status in {"malformed", "none"}:
                    continue
                contact_arc_count = max(contact_arc_count, len(arcs))
                if arc_status != "separated":
                    rejected_single_arc += 1
                    continue
                new_component = np.asarray(new_component, dtype=np.int32)
                # The accepted Closing component must not be within N graph
                # hops of a hard/unknown boundary.  The per-face shortest
                # distance makes this scale-sensitive: a one-step closing can
                # survive a four-hop crop, while an eight-step closing cannot.
                if int(np.min(hard_distance[new_component])) <= int(steps):
                    rejected_hard_distance += 1
                    continue
                additions = new_component[~accepted[new_component]]
                if len(additions) == 0:
                    continue
                accepted[additions] = True
                accepted_component_count += 1
    fills = np.flatnonzero(accepted & missing & domain).astype(np.int32)
    metrics.update(
        {
            "sandwiched_band_closing_attempts": int(attempts),
            "sandwiched_band_closing_components": int(closing_components),
            "sandwiched_band_closing_added_faces": int(len(fills)),
            "sandwiched_band_filled_faces": int(len(fills)),
            "sandwiched_band_filled_components": int(accepted_component_count),
            "sandwiched_band_contact_arcs": int(contact_arc_count),
            "sandwiched_band_rejected_single_arc": int(rejected_single_arc),
            "sandwiched_band_rejected_hard_distance": int(rejected_hard_distance),
            "sandwiched_band_graph_walks": int(graph_walks),
        }
    )
    if not len(fills):
        metrics["sandwiched_band_reason"] = (
            "no-separated-base-arcs"
            if rejected_single_arc
            else "no-closing-addition"
        )
        return fills, metrics
    return np.unique(fills), metrics

def _fill_preview_compact_external_faces(
    geometry, distances, radius, selected_ids, blocked_ids=()
):
    """Compatibility entry point for the unified enclosed-component resolver."""
    return _fill_preview_resolve_enclosed_components(
        geometry, distances, radius, selected_ids
    )


def _fill_preview_tip_external_faces(geometry, distances, radius, selected_ids):
    """Compatibility entry point; no separate terminal-tip pass remains."""
    return _fill_preview_resolve_enclosed_components(
        geometry, distances, radius, selected_ids
    )


def _fill_preview_boundary_route(
    state, geometry, local, distances, analysis_ids, radius,
    preview_local_ids, partition, protected_geometry_ids=(),
):
    """Prefer a long, straight current shape interval's nearby edge route.

    This is deliberately a small post-pass over the current candidate.  It
    does not discover a valley independently: only a connected shape boundary
    already produced for this radius can enter the route search.  The chosen
    primal edge path is converted back into a subset of the current face mask,
    so the drawn and confirmed boundaries remain the same.
    """
    import numpy as np

    fallback = {
        "boundary_route_applied": False,
        "boundary_route_reason": "not-run",
        "boundary_route_components": 0,
        "boundary_route_edges": 0,
        "boundary_route_changed_faces": 0,
        "_boundary_route_edge_indices": (),
    }
    original_preview_ids = np.asarray(preview_local_ids, dtype=np.int32).reshape(-1)
    try:
        mesh = state["obj"].data
        matrix = state["obj"].matrix_world
        count = int(local["count"])
        first = np.asarray(local["first"], dtype=np.int32)
        second = np.asarray(local["second"], dtype=np.int32)
        edge_indices = np.asarray(local.get("pair_edge_indices", ()), dtype=np.int32)
        if count <= 0 or len(first) == 0 or len(edge_indices) != len(first):
            fallback["boundary_route_reason"] = "edge-data-unavailable"
            return original_preview_ids, fallback
        analysis_ids = np.asarray(analysis_ids, dtype=np.int32)
        if len(analysis_ids) != count:
            fallback["boundary_route_reason"] = "analysis-size"
            return original_preview_ids, fallback
        local_global_ids = np.asarray(local.get("_global_face_ids", ()), dtype=np.int32)
        if len(local_global_ids) != count:
            fallback["boundary_route_reason"] = "local-id-data-unavailable"
            return original_preview_ids, fallback
        # make_result stores geometry-local ids in preview_local_ids.  Convert
        # them back to this compact local slice before using them as a mask.
        selected = np.isin(local_global_ids, original_preview_ids)
        preview_local_ids = np.flatnonzero(selected).astype(np.int32)
        local_distances = np.asarray(distances, dtype=np.float64)[analysis_ids]
        crossing = selected[first] != selected[second]
        inside = np.where(selected[first], first, second)
        outside = np.where(selected[first], second, first)
        outside_distance = local_distances[outside]
        shape = _fill_preview_shape_boundary_mask(
            crossing, outside_distance, radius, edge_indices
        )
        shape_pairs = np.flatnonzero(shape & (edge_indices >= 0)).astype(np.int32)
        if len(shape_pairs) > _FILL_PREVIEW_BOUNDARY_ROUTE_SHAPE_PAIR_BUDGET:
            fallback["boundary_route_reason"] = "shape-budget"
            return original_preview_ids, fallback
        if len(shape_pairs) < 6:
            fallback["boundary_route_reason"] = "shape-interval-short"
            return original_preview_ids, fallback

        point_cache = {}

        def point(vertex):
            vertex = int(vertex)
            value = point_cache.get(vertex)
            if value is None:
                value = np.asarray(matrix @ mesh.vertices[vertex].co, dtype=np.float64)
                point_cache[vertex] = value
            return value

        edge_vertices = {}

        def vertices_for_edge(edge):
            edge = int(edge)
            value = edge_vertices.get(edge)
            if value is None:
                values = tuple(int(v) for v in mesh.edges[edge].vertices)
                if len(values) != 2:
                    return None
                edge_vertices[edge] = values
                value = values
            return value

        # Split the current shape edges into simple primal chains. Branches and
        # cycles are kept unchanged, because a single fit would erase their
        # topology.
        shape_edges = sorted(set(int(edge_indices[pair]) for pair in shape_pairs))
        vertex_edges = {}
        for edge in shape_edges:
            values = vertices_for_edge(edge)
            if values is None:
                continue
            for vertex in values:
                vertex_edges.setdefault(vertex, []).append(edge)
        remaining = set(shape_edges)
        components = []
        while remaining:
            start_edge = min(remaining)
            pending = [start_edge]
            remaining.remove(start_edge)
            component = []
            while pending:
                edge = pending.pop()
                component.append(edge)
                values = vertices_for_edge(edge) or ()
                for vertex in values:
                    for neighbor_edge in vertex_edges.get(vertex, ()):
                        if neighbor_edge in remaining:
                            remaining.remove(neighbor_edge)
                            pending.append(neighbor_edge)
            components.append(component)
        fallback["boundary_route_components"] = int(len(components))
        if len(components) > _FILL_PREVIEW_BOUNDARY_ROUTE_COMPONENT_BUDGET:
            fallback["boundary_route_reason"] = "component-budget"
            return original_preview_ids, fallback

        pair_lengths = np.asarray(
            local.get("pair_lengths", local.get("neighbor_lengths", ())),
            dtype=np.float64,
        )
        spacing = float(np.median(pair_lengths)) if len(pair_lengths) else 0.0
        if not np.isfinite(spacing) or spacing <= 1.0e-12:
            fallback["boundary_route_reason"] = "spacing-unavailable"
            return original_preview_ids, fallback
        neighbors = np.asarray(local["neighbors"], dtype=np.int32)
        offsets = np.asarray(local["offsets"], dtype=np.int64)
        protected_ids = np.asarray(tuple(protected_geometry_ids), dtype=np.int32)
        protected = (
            np.isin(local_global_ids, protected_ids)
            if len(local_global_ids) and len(protected_ids)
            else np.zeros(count, dtype=bool)
        )
        best = None
        candidate_pair_total = 0

        def chain_turn(edge_list):
            incident = {}
            for edge in edge_list:
                values = vertices_for_edge(edge)
                if values is None:
                    continue
                a, b = values
                incident.setdefault(a, []).append(b)
                incident.setdefault(b, []).append(a)
            total = 0.0
            for vertex, values in incident.items():
                if len(values) != 2:
                    continue
                va = point(values[0]) - point(vertex)
                vb = point(values[1]) - point(vertex)
                denominator = max(float(np.linalg.norm(va) * np.linalg.norm(vb)), 1.0e-20)
                total += math.pi - math.acos(
                    float(np.clip(np.dot(va, vb) / denominator, -1.0, 1.0))
                )
            return float(total)

        for component in components:
            if len(component) < 6:
                continue
            incident = {}
            for edge in component:
                values = vertices_for_edge(edge)
                if values is None:
                    continue
                for vertex in values:
                    incident.setdefault(vertex, []).append(edge)
            degrees = {vertex: len(edges) for vertex, edges in incident.items()}
            endpoints = [vertex for vertex, degree in degrees.items() if degree == 1]
            if len(endpoints) != 2 or any(degree > 2 for degree in degrees.values()):
                continue
            component_vertices = tuple(incident.keys())
            points = np.asarray([point(vertex) for vertex in component_vertices])
            center = points.mean(axis=0)
            _u, _s, vh = np.linalg.svd(points - center, full_matrices=False)
            direction = np.asarray(vh[0], dtype=np.float64)
            direction /= max(float(np.linalg.norm(direction)), 1.0e-20)
            projections = (points - center) @ direction
            lower, upper = float(projections.min()), float(projections.max())
            fit_error = float(
                np.mean(
                    np.linalg.norm(
                        (points - center)
                        - np.outer(projections, direction),
                        axis=1,
                    )
                )
            )
            # A current orange interval can be a visibly zigzagged edge chain
            # before the longer-radius re-evaluation.  Keep the fit gate
            # conservative enough to reject broad curves, while allowing the
            # interval to reach the nearby straight-route comparison.
            if fit_error > spacing * 1.50:
                continue
            shape_turn = chain_turn(component)
            if shape_turn <= 1.0e-6:
                continue

            component_pairs = shape_pairs[np.isin(edge_indices[shape_pairs], component)]
            boundary_faces = np.unique(
                np.r_[first[component_pairs], second[component_pairs]]
            ).astype(np.int32)
            neighborhood = np.zeros(count, dtype=bool)
            neighborhood[boundary_faces] = True
            for face in boundary_faces:
                neighborhood[
                    neighbors[int(offsets[face]) : int(offsets[face + 1])]
                ] = True
            candidate_pairs = np.flatnonzero(
                (neighborhood[first] | neighborhood[second]) & (edge_indices >= 0)
            )
            if len(candidate_pairs) > _FILL_PREVIEW_BOUNDARY_ROUTE_COMPONENT_PAIR_BUDGET:
                continue
            if (
                candidate_pair_total + len(candidate_pairs)
                > _FILL_PREVIEW_BOUNDARY_ROUTE_TOTAL_PAIR_BUDGET
            ):
                break
            candidate_pair_total += len(candidate_pairs)
            candidate_edges = sorted(set(int(edge_indices[pair]) for pair in candidate_pairs))
            band = max(spacing * 2.5, 1.0e-8)
            graph = {}
            for edge in candidate_edges:
                values = vertices_for_edge(edge)
                if values is None:
                    continue
                a, b = values
                pa, pb = point(a), point(b)
                midpoint = (pa + pb) * 0.5
                midpoint_delta = midpoint - center
                midpoint_projection = float(np.dot(midpoint_delta, direction))
                if midpoint_projection < lower - spacing or midpoint_projection > upper + spacing:
                    continue
                endpoint_distances = []
                for value in (pa, pb):
                    delta = value - center
                    endpoint_distances.append(
                        float(np.linalg.norm(delta - direction * np.dot(delta, direction)))
                    )
                if max(endpoint_distances) > band:
                    continue
                edge_vector = pb - pa
                edge_length = max(float(np.linalg.norm(edge_vector)), 1.0e-20)
                alignment = abs(float(np.dot(edge_vector / edge_length, direction)))
                if alignment < 0.75:
                    continue
                distance_cost = endpoint_distances[0] + endpoint_distances[1]
                cost = (
                    (distance_cost / max(2.0 * spacing, 1.0e-20)) ** 2
                    + 4.0 * (1.0 - alignment)
                    + 0.05 * (edge_length / spacing)
                )
                graph.setdefault(a, []).append((b, edge, cost))
                graph.setdefault(b, []).append((a, edge, cost))
            if not graph:
                continue
            start_projection = lower
            end_projection = upper
            starts = [
                vertex
                for vertex in graph
                if abs(float(np.dot(point(vertex) - center, direction)) - start_projection)
                <= spacing * 0.25
            ]
            ends = {
                vertex
                for vertex in graph
                if abs(float(np.dot(point(vertex) - center, direction)) - end_projection)
                <= spacing * 0.25
            }
            if not starts or not ends:
                continue
            scores = {}
            previous = {}
            pending = []
            for vertex in starts:
                scores[vertex] = 0.0
                previous[vertex] = None
                heapq.heappush(pending, (0.0, int(vertex)))
            target = None
            while pending:
                score, vertex = heapq.heappop(pending)
                if score != scores.get(vertex):
                    continue
                if vertex in ends:
                    target = vertex
                    break
                for neighbor, edge, cost in graph.get(vertex, ()):
                    if float(np.dot(point(neighbor) - point(vertex), direction)) <= 1.0e-8:
                        continue
                    candidate_score = score + cost
                    if candidate_score < scores.get(neighbor, float("inf")):
                        scores[neighbor] = candidate_score
                        previous[neighbor] = (vertex, edge)
                        heapq.heappush(pending, (candidate_score, int(neighbor)))
            if target is None:
                continue
            route_edges = []
            route_vertices = [target]
            vertex = target
            while previous.get(vertex) is not None:
                old_vertex, edge = previous[vertex]
                route_edges.append(int(edge))
                vertex = old_vertex
                route_vertices.append(vertex)
            route_edges.reverse()
            route_vertices.reverse()
            if len(route_edges) < 6 or len(route_edges) < int(len(component) * 0.60):
                continue
            route_turn = chain_turn(route_edges)
            if route_turn >= shape_turn * 0.85:
                continue
            route_points = np.asarray([point(vertex) for vertex in route_vertices])
            route_projection = (route_points - center) @ direction
            route_fit = float(
                np.mean(
                    np.linalg.norm(
                        (route_points - center)
                        - np.outer(route_projection, direction),
                        axis=1,
                    )
                )
            )
            if route_fit > fit_error + spacing * 0.30:
                continue
            endpoint_distance = max(
                min(float(np.linalg.norm(point(route_vertices[0]) - point(value))) for value in endpoints),
                min(float(np.linalg.norm(point(route_vertices[-1]) - point(value))) for value in endpoints),
            )
            # The replacement route may sit one local row away from the
            # current orange chain.  Treat that short offset as an implicit
            # terminal connector; the long middle section is still required
            # to be an existing mesh-edge route.
            if endpoint_distance > spacing * 3.0:
                continue
            seed_matches = np.flatnonzero(local_global_ids == int(state.get("seed_local", -1)))
            if len(seed_matches) == 0:
                continue
            seed_center = local["centers"][int(seed_matches[0])]
            side = np.asarray(seed_center, dtype=np.float64) - center
            side -= direction * np.dot(side, direction)
            side_norm = float(np.linalg.norm(side))
            if side_norm <= 1.0e-12:
                continue
            side /= side_norm
            route_midpoints = np.asarray(
                [(point(vertices_for_edge(edge)[0]) + point(vertices_for_edge(edge)[1])) * 0.5 for edge in route_edges]
            )
            route_level = float(np.median((route_midpoints - center) @ side))
            seed_level = float(np.dot(seed_center - center, side))
            if route_level <= spacing * 0.05 or seed_level <= route_level + spacing * 0.50:
                continue
            signed = (local["centers"] - center) @ side
            local_projection = (local["centers"] - center) @ direction
            # Distance-boundary faces are fixed for this radius.  Keep both
            # sides of those crossings out of the local mask move so a nearby
            # route cannot shorten or redraw the radial boundary.
            distance_boundary = crossing & ~shape
            distance_boundary_faces = np.zeros(count, dtype=bool)
            if np.any(distance_boundary):
                distance_boundary_faces[first[distance_boundary]] = True
                distance_boundary_faces[second[distance_boundary]] = True
            remove = selected & ~protected & (
                signed < route_level + spacing * 0.25
            ) & (signed > -band) & (
                local_projection >= lower - spacing
            ) & (local_projection <= upper + spacing)
            remove &= neighborhood
            remove &= ~distance_boundary_faces
            if not np.any(remove) or bool(remove[int(seed_matches[0])]):
                continue
            candidate_mask = selected.copy()
            candidate_mask[remove] = False
            after_crossing = candidate_mask[first] != candidate_mask[second]
            # A route may replace the shape crossing, but an existing distance
            # boundary must remain a boundary at the same radius.
            distance_boundary = crossing & ~shape
            if np.any(~after_crossing[distance_boundary]):
                continue
            route_pairs = np.flatnonzero(np.isin(edge_indices, route_edges))
            if not len(route_pairs) or not np.all(
                after_crossing[route_pairs]
            ):
                continue
            shape_pairs_component = component_pairs
            if np.any(after_crossing[shape_pairs_component]):
                continue
            # A route one mesh row away can expose one short existing edge at
            # each end between the route endpoint and the old interval end.
            # Keep those terminal connectors in the same shape boundary so
            # the final candidate and its orange classification agree.
            route_boundary_edge_ids = set(int(edge) for edge in route_edges)
            route_endpoint_vertices = {
                int(route_vertices[0]), int(route_vertices[-1])
            }
            shape_endpoint_vertices = {int(vertex) for vertex in endpoints}
            for pair in np.flatnonzero(after_crossing & (edge_indices >= 0)):
                edge = int(edge_indices[pair])
                values = vertices_for_edge(edge)
                if values is None:
                    continue
                first_vertex, second_vertex = values
                if (
                    (
                        first_vertex in route_endpoint_vertices
                        and second_vertex in shape_endpoint_vertices
                    )
                    or (
                        second_vertex in route_endpoint_vertices
                        and first_vertex in shape_endpoint_vertices
                    )
                ):
                    route_boundary_edge_ids.add(edge)
            connected = np.zeros(count, dtype=bool)
            seed_local = int(seed_matches[0])
            connected[seed_local] = True
            pending_faces = [seed_local]
            while pending_faces:
                face = pending_faces.pop()
                for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
                    neighbor = int(neighbor)
                    if candidate_mask[neighbor] and not connected[neighbor]:
                        connected[neighbor] = True
                        pending_faces.append(neighbor)
            if not np.all(connected[candidate_mask]):
                continue
            corrected_geometry_ids = local_global_ids[np.flatnonzero(candidate_mask)].astype(
                np.int32
            )
            new_gap_ids, _gap_metrics = _fill_preview_enclosed_gap_faces(
                geometry, distances, radius, corrected_geometry_ids
            )
            if len(new_gap_ids):
                continue
            if best is None or route_turn < best[0]:
                best = (
                    route_turn,
                    candidate_mask,
                    len(route_edges),
                    len(np.flatnonzero(selected & ~candidate_mask)),
                    shape_turn,
                    route_fit,
                    fit_error,
                    tuple(sorted(route_boundary_edge_ids)),
                )
        if best is None:
            fallback["boundary_route_reason"] = "no-clear-route"
            return original_preview_ids, fallback
        (
            _route_turn,
            candidate_mask,
            route_count,
            changed,
            before_turn,
            route_fit,
            fit_error,
            route_boundary_edge_ids,
        ) = best
        corrected = local_global_ids[np.flatnonzero(candidate_mask)].astype(np.int32)
        fallback.update(
            {
                "boundary_route_applied": True,
                "boundary_route_reason": "applied",
                "boundary_route_edges": int(route_count),
                "boundary_route_changed_faces": int(changed),
                "_boundary_route_edge_indices": tuple(
                    int(edge) for edge in route_boundary_edge_ids
                ),
                "boundary_route_before_turn": float(before_turn),
                "boundary_route_after_turn": float(_route_turn),
                "boundary_route_fit_error": float(fit_error),
                "boundary_route_route_fit": float(route_fit),
            }
        )
        return corrected, fallback
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError):
        fallback["boundary_route_reason"] = "error-fallback"
        return original_preview_ids, fallback


def _fill_preview_boundary_one_hop(
    state, geometry, local, distances, analysis_ids, radius,
    preview_local_ids, protected_geometry_ids=(), allowed_shape_rows=None,
):
    """Propose a bounded one-row Closing, with gap fallback.

    This is a bounded post-pass over the candidate already produced for this
    radius.  It does not expand from an added face, and it never adds a face
    outside the current distance domain.  If local topological completion
    finds a newly enclosed gap, only the newly added faces adjacent to that
    gap are removed for up to three passes; an unresolved gap restores the
    original candidate.
    """
    import numpy as np

    original_ids = np.asarray(preview_local_ids, dtype=np.int32).reshape(-1)
    fallback = {
        "one_hop_applied": False,
        "one_hop_reason": "not-run",
        "one_hop_added_faces": 0,
        "one_hop_removed_faces": 0,
        "one_hop_gap_passes": 0,
        "one_hop_gap_faces": 0,
        "one_hop_distance_boundary_kept": True,
        "one_hop_radius_outside_faces": 0,
    }
    try:
        count = int(local["count"])
        first = np.asarray(local["first"], dtype=np.int32)
        second = np.asarray(local["second"], dtype=np.int32)
        edge_indices = np.asarray(local.get("pair_edge_indices", ()), dtype=np.int32)
        local_global_ids = np.asarray(local.get("_global_face_ids", ()), dtype=np.int32)
        analysis_ids = np.asarray(analysis_ids, dtype=np.int32)
        distances = np.asarray(distances, dtype=np.float64)
        if (
            count <= 0
            or len(first) == 0
            or len(first) != len(second)
            or len(first) != len(edge_indices)
            or len(local_global_ids) != count
            or len(analysis_ids) != count
            or len(distances) <= int(np.max(analysis_ids))
        ):
            fallback["one_hop_reason"] = "edge-data-unavailable"
            return original_ids, fallback
        selected = np.isin(local_global_ids, original_ids)
        local_distances = distances[analysis_ids]
        crossing = selected[first] != selected[second]
        outside = np.where(selected[first], second, first)
        shape = _fill_preview_shape_boundary_mask(
            crossing,
            local_distances[outside],
            float(radius),
            edge_indices,
        )
        shape_pairs = np.flatnonzero(shape & (edge_indices >= 0)).astype(np.int32)
        if allowed_shape_rows is not None:
            allowed_shape_rows = np.asarray(allowed_shape_rows, dtype=bool).reshape(-1)
            if len(allowed_shape_rows) != len(first):
                fallback["one_hop_reason"] = "interval-schema"
                return original_ids, fallback
            shape_pairs = shape_pairs[allowed_shape_rows[shape_pairs]]
        fallback["one_hop_shape_pairs"] = int(len(shape_pairs))
        if len(shape_pairs) == 0:
            fallback["one_hop_reason"] = "shape-interval-short"
            return original_ids, fallback

        hidden = np.asarray(
            local.get("hidden", np.zeros(count, dtype=bool)), dtype=bool
        )
        mesh = state["obj"].data
        geometry_face_ids = np.asarray(geometry["face_ids"], dtype=np.int32)
        outside_shape = outside[shape_pairs]
        visible = np.asarray(
            [
                not bool(
                    mesh.polygons[
                        int(geometry_face_ids[int(local_global_ids[int(face_id)])])
                    ].hide
                )
                for face_id in outside_shape
            ],
            dtype=bool,
        )
        tolerance = max(float(radius) * 1.0e-8, 1.0e-9)
        protected_ids = np.asarray(
            tuple(protected_geometry_ids), dtype=np.int32
        )
        eligible = shape_pairs[
            (~selected[outside_shape])
            & np.isfinite(local_distances[outside_shape])
            & (local_distances[outside_shape] <= float(radius) + tolerance)
            & ~hidden[outside_shape]
            & visible
        ]
        if len(protected_ids):
            eligible = eligible[
                ~np.isin(local_global_ids[outside[eligible]], protected_ids)
            ]
        added_local = np.unique(outside[eligible]).astype(np.int32)
        fallback["one_hop_eligible_pairs"] = int(len(eligible))
        if len(added_local) == 0:
            fallback["one_hop_reason"] = "no-eligible-face"
            return original_ids, fallback
        fallback["one_hop_radius_outside_faces"] = int(
            np.count_nonzero(
                ~np.isfinite(local_distances[added_local])
                | (local_distances[added_local] > float(radius) + tolerance)
            )
        )
        if fallback["one_hop_radius_outside_faces"]:
            fallback["one_hop_reason"] = "radius-outside"
            return original_ids, fallback

        neighbors = np.asarray(local["neighbors"], dtype=np.int32)
        offsets = np.asarray(local["offsets"], dtype=np.int64)
        added_mask = np.zeros(count, dtype=bool)
        added_mask[added_local] = True
        working = selected.copy()
        working[added_local] = True
        geometry_neighbors = np.asarray(geometry["neighbors"], dtype=np.int32)
        geometry_offsets = np.asarray(geometry["offsets"], dtype=np.int64)
        id_to_local = {int(value): index for index, value in enumerate(local_global_ids)}
        gap_passes = 0
        removed_faces = 0
        final_gap_count = None
        for _pass in range(3):
            selected_ids = local_global_ids[np.flatnonzero(working)].astype(np.int32)
            gap_ids, _gap_metrics = _fill_preview_enclosed_gap_faces(
                geometry, distances, radius, selected_ids
            )
            if len(gap_ids) == 0:
                final_gap_count = 0
                break
            gap_passes += 1
            remove = set()
            for gap_id in np.asarray(gap_ids, dtype=np.int32):
                gap_id = int(gap_id)
                if gap_id < 0 or gap_id + 1 >= len(geometry_offsets):
                    continue
                for neighbor in geometry_neighbors[
                    int(geometry_offsets[gap_id]) : int(geometry_offsets[gap_id + 1])
                ]:
                    local_neighbor = id_to_local.get(int(neighbor))
                    if (
                        local_neighbor is not None
                        and added_mask[local_neighbor]
                        and working[local_neighbor]
                    ):
                        remove.add(int(local_neighbor))
            if not remove:
                final_gap_count = int(len(gap_ids))
                break
            remove_ids = np.asarray(sorted(remove), dtype=np.int32)
            working[remove_ids] = False
            removed_faces += int(len(remove_ids))
        if final_gap_count is None:
            selected_ids = local_global_ids[np.flatnonzero(working)].astype(np.int32)
            final_gap_ids, _gap_metrics = _fill_preview_enclosed_gap_faces(
                geometry, distances, radius, selected_ids
            )
            final_gap_count = int(len(final_gap_ids))
        fallback["one_hop_gap_passes"] = int(gap_passes)
        fallback["one_hop_gap_faces"] = int(final_gap_count)
        fallback["one_hop_removed_faces"] = int(removed_faces)
        if final_gap_count:
            fallback["one_hop_reason"] = "gap-fallback"
            return original_ids, fallback

        base_distance = crossing & ~shape
        after_crossing = working[first] != working[second]
        distance_kept = bool(np.all(after_crossing[base_distance]))
        fallback["one_hop_distance_boundary_kept"] = distance_kept
        if not distance_kept:
            fallback["one_hop_reason"] = "distance-boundary"
            return original_ids, fallback

        seed_matches = np.flatnonzero(
            local_global_ids == int(state.get("seed_local", -1))
        )
        if len(seed_matches) == 0:
            fallback["one_hop_reason"] = "seed-missing"
            return original_ids, fallback
        seed_local = int(seed_matches[0])
        connected = np.zeros(count, dtype=bool)
        connected[seed_local] = True
        pending = [seed_local]
        while pending:
            face = pending.pop()
            for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
                neighbor = int(neighbor)
                if working[neighbor] and not connected[neighbor]:
                    connected[neighbor] = True
                    pending.append(neighbor)
        if not np.all(connected[working]):
            fallback["one_hop_reason"] = "disconnected"
            return original_ids, fallback

        corrected = local_global_ids[np.flatnonzero(working)].astype(np.int32)
        fallback.update(
            {
                "one_hop_applied": True,
                "one_hop_reason": "applied",
                "one_hop_added_faces": int(np.count_nonzero(working & ~selected)),
            }
        )
        return corrected, fallback
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError):
        fallback["one_hop_reason"] = "error-fallback"
        return original_ids, fallback


def _fill_preview_boundary_endpoint_schema(
    state, geometry, local, edge_indices, pair_v0, pair_v1
):
    """Validate the declared endpoint id space before any boundary scoring.

    Cursor graphs use compact indices into ``world_vertices``; the full mesh
    graph uses explicitly declared mesh-global indices.  An omitted or
    inconsistent schema is not recoverable by guessing, because an integer
    that is valid in one space can silently name an unrelated vertex in the
    other space.  Return a short rejection reason instead of allowing a
    proposal to be generated from ambiguous coordinates.
    """
    import numpy as np

    vertex_id_space = local.get("vertex_id_space")
    if vertex_id_space not in {"compact", "mesh-global"}:
        return "endpoint-schema"
    edge_indices = np.asarray(edge_indices, dtype=np.int32).reshape(-1)
    pair_v0 = np.asarray(pair_v0, dtype=np.int32).reshape(-1)
    pair_v1 = np.asarray(pair_v1, dtype=np.int32).reshape(-1)
    if (
        len(edge_indices) == 0
        or len(pair_v0) != len(edge_indices)
        or len(pair_v1) != len(edge_indices)
        or np.any(pair_v0 < 0)
        or np.any(pair_v1 < 0)
    ):
        return "endpoint-schema"
    if vertex_id_space == "compact":
        world_vertices = np.asarray(
            local.get("world_vertices", geometry.get("world_vertices", ())),
            dtype=np.float64,
        )
        if (
            world_vertices.ndim != 2
            or world_vertices.shape[1] != 3
            or not np.all(np.isfinite(world_vertices))
            or np.any(pair_v0 >= len(world_vertices))
            or np.any(pair_v1 >= len(world_vertices))
        ):
            return "endpoint-schema"
        return None
    mesh = getattr(state.get("obj"), "data", None)
    matrix = getattr(state.get("obj"), "matrix_world", None)
    if mesh is None or matrix is None:
        return "endpoint-schema"
    try:
        vertex_count = int(len(mesh.vertices))
    except (AttributeError, TypeError, ValueError):
        return "endpoint-schema"
    if np.any(pair_v0 >= vertex_count) or np.any(pair_v1 >= vertex_count):
        return "endpoint-schema"
    return None


def _fill_preview_boundary_trim_bulge(
    state, geometry, local, distances, analysis_ids, radius,
    preview_local_ids, protected_geometry_ids=(), allowed_shape_rows=None,
):
    """Propose a bounded complement Opening for one off-level bulge.

    The complement of Closing is useful only as a proposal here: this helper
    does not open the whole candidate.  It first fits each existing orange
    edge chain, finds a repeated one/two-ring edge plateau that sits away from
    the chain's robust median level, and returns only the selected face
    component touching that plateau.  The caller still applies the shared
    shape score and all seed/cyan/barrier/connectivity gates.
    """
    import numpy as np

    original_ids = np.asarray(preview_local_ids, dtype=np.int32).reshape(-1)
    fallback = {
        "boundary_trim_applied": False,
        "boundary_trim_reason": "not-run",
        "boundary_trim_changed_faces": 0,
        "boundary_trim_edges": 0,
    }
    try:
        count = int(local["count"])
        first = np.asarray(local["first"], dtype=np.int32).reshape(-1)
        second = np.asarray(local["second"], dtype=np.int32).reshape(-1)
        edge_indices = np.asarray(
            local.get("pair_edge_indices", ()), dtype=np.int32
        ).reshape(-1)
        local_global_ids = np.asarray(
            local.get("_global_face_ids", ()), dtype=np.int32
        ).reshape(-1)
        pair_v0 = np.asarray(local.get("pair_v0", ()), dtype=np.int32).reshape(-1)
        pair_v1 = np.asarray(local.get("pair_v1", ()), dtype=np.int32).reshape(-1)
        analysis_ids = np.asarray(analysis_ids, dtype=np.int32).reshape(-1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        offsets = np.asarray(local["offsets"], dtype=np.int64).reshape(-1)
        neighbors = np.asarray(local["neighbors"], dtype=np.int32).reshape(-1)
        if (
            count <= 0
            or len(first) == 0
            or len(first) != len(second)
            or len(first) != len(edge_indices)
            or len(local_global_ids) != count
            or len(analysis_ids) != count
            or len(offsets) != count + 1
            or int(offsets[-1]) != len(neighbors)
            or np.any(first < 0)
            or np.any(second < 0)
            or np.any(first >= count)
            or np.any(second >= count)
            or np.any(analysis_ids < 0)
            or np.any(analysis_ids >= len(distances))
        ):
            fallback["boundary_trim_reason"] = "schema"
            return original_ids, fallback
        local_distances = distances[analysis_ids]
        selected = np.isin(local_global_ids, original_ids)
        crossing = selected[first] != selected[second]
        outside = np.where(selected[first], second, first)
        shape = _fill_preview_shape_boundary_mask(
            crossing, local_distances[outside], float(radius), edge_indices
        )
        shape_pairs = np.flatnonzero(shape & (edge_indices >= 0)).astype(np.int32)
        if allowed_shape_rows is not None:
            allowed_shape_rows = np.asarray(allowed_shape_rows, dtype=bool).reshape(-1)
            if len(allowed_shape_rows) != len(first):
                fallback["boundary_trim_reason"] = "interval-schema"
                return original_ids, fallback
            shape_pairs = shape_pairs[allowed_shape_rows[shape_pairs]]
        if len(shape_pairs) < 6:
            fallback["boundary_trim_reason"] = "shape-interval-short"
            return original_ids, fallback

        endpoint_reason = _fill_preview_boundary_endpoint_schema(
            state, geometry, local, edge_indices, pair_v0, pair_v1
        )
        if endpoint_reason is not None:
            fallback["boundary_trim_reason"] = endpoint_reason
            return original_ids, fallback

        world_vertices = np.asarray(
            local.get("world_vertices", geometry.get("world_vertices", ())),
            dtype=np.float64,
        )
        mesh = getattr(state.get("obj"), "data", None)
        matrix = getattr(state.get("obj"), "matrix_world", None)
        vertex_id_space = str(local.get("vertex_id_space"))
        point_cache = {}

        def point(vertex):
            vertex = int(vertex)
            if vertex in point_cache:
                return point_cache[vertex]
            if vertex_id_space == "mesh-global" and mesh is not None and matrix is not None:
                value = np.asarray(matrix @ mesh.vertices[vertex].co, dtype=np.float64)
            elif vertex_id_space == "compact" and (
                world_vertices.ndim == 2
                and world_vertices.shape[1] == 3
                and 0 <= vertex < len(world_vertices)
            ):
                value = np.asarray(world_vertices[vertex], dtype=np.float64)
            else:
                raise ValueError("edge endpoint data unavailable")
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("edge endpoint data unavailable")
            point_cache[vertex] = value
            return value

        edge_vertices = {}

        def vertices_for_edge(pair):
            edge = int(edge_indices[int(pair)])
            if edge in edge_vertices:
                return edge_vertices[edge]
            if (
                0 <= int(pair) < len(pair_v0)
                and len(pair_v0) == len(edge_indices)
                and len(pair_v1) == len(edge_indices)
                and int(pair_v0[int(pair)]) >= 0
                and int(pair_v1[int(pair)]) >= 0
            ):
                values = int(pair_v0[int(pair)]), int(pair_v1[int(pair)])
                edge_vertices[edge] = values
                return values
            return None

        # Split shape edges into the same physical chains used by the score.
        edge_to_pairs = {}
        vertex_edges = {}
        for pair in shape_pairs:
            edge = int(edge_indices[int(pair)])
            values = vertices_for_edge(pair)
            if values is None:
                continue
            edge_to_pairs.setdefault(edge, []).append(int(pair))
            for vertex in values:
                vertex_edges.setdefault(int(vertex), []).append(edge)
        remaining = set(edge_to_pairs)
        components = []
        while remaining:
            edge = min(remaining)
            remaining.remove(edge)
            pending = [edge]
            component = []
            while pending:
                edge = pending.pop()
                component.append(edge)
                values = edge_vertices.get(edge, ())
                for vertex in values:
                    for other in vertex_edges.get(vertex, ()):
                        if other in remaining:
                            remaining.remove(other)
                            pending.append(other)
            components.append(component)
        if not components:
            fallback["boundary_trim_reason"] = "edge-data-unavailable"
            return original_ids, fallback

        centers = np.asarray(local.get("centers"), dtype=np.float64)
        if centers.shape != (count, 3):
            fallback["boundary_trim_reason"] = "center-schema"
            return original_ids, fallback
        protected_ids = np.asarray(tuple(protected_geometry_ids), dtype=np.int32)
        protected = np.isin(local_global_ids, protected_ids)
        seed_matches = np.flatnonzero(
            local_global_ids == int(state.get("seed_local", -1))
        )
        if len(seed_matches) == 0:
            fallback["boundary_trim_reason"] = "seed-missing"
            return original_ids, fallback
        seed_local = int(seed_matches[0])
        spacing_values = np.asarray(
            local.get("pair_lengths", local.get("neighbor_lengths", ())),
            dtype=np.float64,
        )
        spacing = float(np.median(spacing_values)) if len(spacing_values) else 0.0
        if not np.isfinite(spacing) or spacing <= 1.0e-12:
            fallback["boundary_trim_reason"] = "spacing-unavailable"
            return original_ids, fallback

        # Prefer the first repeated plateau that has at least two physical
        # edges.  A single isolated edge, a plane, and a wide basin therefore
        # never become an opening proposal.
        best_remove = None
        for component in components:
            if len(component) < 6:
                continue
            pairs = [pair for edge in component for pair in edge_to_pairs.get(edge, ())]
            if len(pairs) < 6:
                continue
            vertices = sorted(
                {
                    vertex
                    for edge in component
                    for vertex in edge_vertices.get(edge, ())
                }
            )
            if len(vertices) < 4:
                continue
            points = np.asarray([point(vertex) for vertex in vertices], dtype=np.float64)
            center = points.mean(axis=0)
            _u, _s, vh = np.linalg.svd(points - center, full_matrices=False)
            direction = np.asarray(vh[0], dtype=np.float64)
            direction /= max(float(np.linalg.norm(direction)), 1.0e-20)
            normal = np.asarray(vh[1], dtype=np.float64)
            normal -= direction * np.dot(normal, direction)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1.0e-12:
                continue
            normal /= normal_norm
            edge_offsets = []
            edge_pairs = {}
            for edge in component:
                pair = edge_to_pairs[edge][0]
                values = edge_vertices[edge]
                midpoint = (point(values[0]) + point(values[1])) * 0.5
                edge_offsets.append(float(np.dot(midpoint - center, normal)))
                edge_pairs[edge] = pair
            edge_offsets = np.asarray(edge_offsets, dtype=np.float64)
            baseline = float(np.median(edge_offsets))
            outlier_edges = [
                edge
                for edge, offset in zip(component, edge_offsets)
                if abs(float(offset) - baseline) >= spacing * 0.40
            ]
            if len(outlier_edges) < 2:
                continue
            outlier_delta = float(
                np.median(
                    [
                        edge_offsets[index]
                        for index, edge in enumerate(component)
                        if edge in outlier_edges
                    ]
                )
                - baseline
            )
            outlier_side = 1.0 if outlier_delta >= 0.0 else -1.0
            # The selected face immediately behind an outlier edge seeds a
            # local component.  Expand only through faces with the same
            # off-level signal; this is the one/two-ring bulge corridor.
            face_offsets = (centers - center) @ normal
            high_face = (
                (face_offsets - baseline) * outlier_side >= spacing * 0.20
            )
            starts = set()
            boundary_inside_faces = set()
            for edge in outlier_edges:
                pair = edge_pairs[edge]
                inside = int(first[pair]) if selected[int(first[pair])] else int(second[pair])
                boundary_inside_faces.add(inside)
                if (
                    selected[inside]
                    and not protected[inside]
                    and inside != seed_local
                    and high_face[inside]
                ):
                    starts.add(inside)
            if not starts:
                continue
            candidate_faces = set()
            trim_corridor = np.zeros(count, dtype=bool)
            boundary_inside_array = np.asarray(sorted(boundary_inside_faces), dtype=np.int32)
            trim_corridor[boundary_inside_array] = True
            for face in boundary_inside_array:
                trim_corridor[
                    neighbors[int(offsets[face]) : int(offsets[face + 1])]
                ] = True
            pending = list(starts)
            while pending:
                face = int(pending.pop())
                if (
                    face in candidate_faces
                    or not trim_corridor[face]
                    or not selected[face]
                    or protected[face]
                    or face == seed_local
                ):
                    continue
                if not high_face[face]:
                    continue
                candidate_faces.add(face)
                # Opening is intentionally one graph layer.  Do not flood
                # through an arbitrary high-face component: the caller's
                # complete-geometry validation remains the final safety gate.
            if len(candidate_faces) < 2:
                continue
            if best_remove is None or len(candidate_faces) > len(best_remove):
                best_remove = candidate_faces
                fallback["boundary_trim_edges"] = int(len(outlier_edges))
        if not best_remove:
            fallback["boundary_trim_reason"] = "no-local-bulge"
            return original_ids, fallback
        candidate_mask = selected.copy()
        candidate_mask[np.asarray(sorted(best_remove), dtype=np.int32)] = False
        corrected = local_global_ids[np.flatnonzero(candidate_mask)].astype(np.int32)
        fallback.update(
            {
                "boundary_trim_applied": True,
                "boundary_trim_reason": "proposal",
                "boundary_trim_changed_faces": int(len(best_remove)),
            }
        )
        return corrected, fallback
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        fallback["boundary_trim_reason"] = "error-fallback"
        return original_ids, fallback


def _fill_preview_gray_partition_intervals(
    edge_ids, edge_vertices, edge_signal, edge_face_normals, point,
):
    """Partition physical edges into deterministic simple intervals.

    Proposal generation and candidate scoring must agree on where a physical
    edge chain starts, ends, and crosses a junction.  The local tangent plane
    is therefore resolved in this one helper.  A degenerate averaged normal
    is deliberately ambiguous: falling back to 3D directions would make the
    two callers disagree on folded surfaces.
    """
    import numpy as np

    edge_ids = tuple(sorted(int(edge) for edge in edge_ids))
    if not edge_ids:
        return (), 0, 0, 0
    vertex_edges = {}
    for edge in edge_ids:
        values = edge_vertices.get(edge)
        if values is None or len(values) != 2:
            continue
        a, b = int(values[0]), int(values[1])
        vertex_edges.setdefault(a, []).append(edge)
        vertex_edges.setdefault(b, []).append(edge)
    remaining = set(edge_ids)
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        pending = [start]
        component = []
        while pending:
            edge = pending.pop()
            component.append(edge)
            values = edge_vertices.get(edge)
            if values is None or len(values) != 2:
                continue
            for vertex in (int(values[0]), int(values[1])):
                for other in vertex_edges.get(vertex, ()):
                    if other in remaining:
                        remaining.remove(other)
                        pending.append(other)
        components.append(tuple(sorted(component)))

    intervals = []
    junction_count = 0
    junction_pair_count = 0
    junction_ambiguous_count = 0
    for component in components:
        component_set = set(component)
        incident = {}
        for edge in component:
            values = edge_vertices.get(edge)
            if values is None or len(values) != 2:
                continue
            incident.setdefault(int(values[0]), []).append(edge)
            incident.setdefault(int(values[1]), []).append(edge)
        junctions = sorted(
            vertex for vertex, values in incident.items() if len(values) > 2
        )
        junction_count += len(junctions)
        transition = {}
        for vertex, raw_values in incident.items():
            values = tuple(sorted(int(value) for value in raw_values))
            if len(values) == 2:
                transition[(int(vertex), values[0])] = values[1]
                transition[(int(vertex), values[1])] = values[0]
                continue
            if len(values) <= 2:
                continue
            origin = point(vertex)
            directions = {}
            normal_values = []
            for edge in values:
                edge_points = edge_vertices.get(edge)
                if edge_points is None or len(edge_points) != 2:
                    continue
                a, b = int(edge_points[0]), int(edge_points[1])
                other = b if vertex == a else a
                direction = np.asarray(point(other), dtype=np.float64) - origin
                length = float(np.linalg.norm(direction))
                if not np.isfinite(length) or length <= 1.0e-12:
                    continue
                directions[edge] = direction / length
                normals = edge_face_normals.get(edge, ())
                normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
                if len(normals):
                    normal_values.extend(normals)
            if len(directions) < 2 or len(normal_values) == 0:
                junction_ambiguous_count += 1
                continue
            normal = np.sum(np.asarray(normal_values, dtype=np.float64), axis=0)
            normal_length = float(np.linalg.norm(normal))
            if not np.isfinite(normal_length) or normal_length <= 1.0e-12:
                # No 3D fallback: the tangent plane is undefined, so this
                # junction is intentionally split as ambiguous.
                junction_ambiguous_count += 1
                continue
            normal /= normal_length
            pair_scores = []
            ordered_values = sorted(directions)
            for left_index, left_edge in enumerate(ordered_values):
                for right_edge in ordered_values[left_index + 1:]:
                    left_direction = directions[left_edge] - normal * float(
                        np.dot(directions[left_edge], normal)
                    )
                    right_direction = directions[right_edge] - normal * float(
                        np.dot(directions[right_edge], normal)
                    )
                    left_length = float(np.linalg.norm(left_direction))
                    right_length = float(np.linalg.norm(right_direction))
                    if left_length <= 1.0e-12 or right_length <= 1.0e-12:
                        continue
                    left_direction /= left_length
                    right_direction /= right_length
                    dot = float(np.clip(np.dot(left_direction, right_direction), -1.0, 1.0))
                    straightness = 0.5 * (1.0 - dot)
                    signal_continuity = 1.0 - abs(
                        float(edge_signal.get(left_edge, 0.0))
                        - float(edge_signal.get(right_edge, 0.0))
                    )
                    pair_scores.append((
                        float(0.75 * straightness + 0.25 * signal_continuity),
                        left_edge,
                        right_edge,
                    ))
            pair_scores.sort(key=lambda value: (-value[0], value[1], value[2]))
            used_at_junction = set()
            accepted_at_junction = 0
            for pair_score, left_edge, right_edge in pair_scores:
                if pair_score < 0.72:
                    continue
                if left_edge in used_at_junction or right_edge in used_at_junction:
                    continue
                alternatives = [
                    value[0]
                    for value in pair_scores
                    if (value[1], value[2]) != (left_edge, right_edge)
                    and (value[1] in (left_edge, right_edge) or value[2] in (left_edge, right_edge))
                ]
                if alternatives and pair_score - max(alternatives) < 0.08:
                    continue
                transition[(int(vertex), left_edge)] = right_edge
                transition[(int(vertex), right_edge)] = left_edge
                used_at_junction.update((left_edge, right_edge))
                accepted_at_junction += 1
            junction_pair_count += accepted_at_junction
            if accepted_at_junction == 0:
                junction_ambiguous_count += 1

        starts = []
        for edge in sorted(component_set):
            values = edge_vertices.get(edge)
            if values is None or len(values) != 2:
                continue
            for vertex in (int(values[0]), int(values[1])):
                if (vertex, edge) not in transition:
                    starts.append((edge, vertex))
        used_edges = set()
        for start_edge, start_vertex in sorted(starts):
            if start_edge in used_edges:
                continue
            path = []
            edge = int(start_edge)
            vertex = int(start_vertex)
            while edge not in used_edges and edge in component_set:
                used_edges.add(edge)
                path.append(edge)
                values = edge_vertices.get(edge)
                if values is None or len(values) != 2:
                    break
                a, b = int(values[0]), int(values[1])
                next_vertex = b if vertex == a else a
                next_edge = transition.get((next_vertex, edge))
                if next_edge is None or next_edge in used_edges:
                    break
                vertex = next_vertex
                edge = next_edge
            if path:
                intervals.append(tuple(path))
    return tuple(intervals), int(junction_count), int(junction_pair_count), int(junction_ambiguous_count)


def _fill_preview_boundary_grayscale_proposals(
    state, geometry, local, distances, analysis_ids, radius,
    preview_local_ids, protected_geometry_ids=(), usable_shape_rows=None,
):
    """Generate bounded grayscale Closing/Opening proposals.

    The signal is attached to physical mesh edges and filtered only along
    metric-length bounded chains.  A proposal is still a face-set delta from
    the immutable input; all full-geometry safety checks remain in the caller.
    """
    import math
    import numpy as np

    original_ids = np.unique(np.asarray(preview_local_ids, dtype=np.int32).reshape(-1))
    metrics = {
        "shape_refine_gray_interval_count": 0,
        "shape_refine_gray_usable_edges": 0,
        "shape_refine_gray_guarded_edges": 0,
        "shape_refine_gray_signal_median": 0.0,
        "shape_refine_gray_signal_mad": 0.0,
        "shape_refine_gray_close_response_max": 0.0,
        "shape_refine_gray_open_response_max": 0.0,
        "shape_refine_gray_add_edges": 0,
        "shape_refine_gray_trim_edges": 0,
        "shape_refine_gray_add_faces": 0,
        "shape_refine_gray_corridor_band_faces": 0,
        "shape_refine_gray_corridor_band_components": 0,
        "shape_refine_gray_trim_faces": 0,
        "shape_refine_gray_hysteresis_high": 0.0,
        "shape_refine_gray_hysteresis_low": 0.0,
        "shape_refine_gray_window_factor_min": 0.0,
        "shape_refine_gray_window_factor_max": 0.0,
        "shape_refine_gray_window_distance_min": 0.0,
        "shape_refine_gray_window_distance_max": 0.0,
        "shape_refine_gray_target_chain": (),
        "shape_refine_gray_proposal_edge_signatures": (),
        "shape_refine_gray_final_edge_signature": (),
        "shape_refine_gray_adopted_proposal_edge_signature": (),
        "shape_refine_gray_junction_count": 0,
        "shape_refine_gray_junction_pairs": 0,
        "shape_refine_gray_junction_ambiguous": 0,
        "shape_refine_gray_rejection_reason": "not-run",
        "shape_refine_gray_context_edge_count": 0,
        "shape_refine_gray_context_window": 0.0,
        "shape_refine_gray_context_status": "not-run",
        "shape_refine_provisional": False,
        "shape_refine_add_generated_faces": 0,
        "shape_refine_add_reason": "not-run",
        "shape_refine_add_eligible_pairs": 0,
        "shape_refine_add_gap_faces": 0,
        "shape_refine_add_proposal_count": 0,
        "shape_refine_trim_generated_faces": 0,
        "shape_refine_trim_reason": "not-run",
        "shape_refine_trim_outlier_edges": 0,
        "shape_refine_trim_gap_faces": 0,
        "shape_refine_trim_proposal_count": 0,
        "shape_refine_proposal_validation": (),
        "shadow_tested_crossings": 0,
        "shadow_barrier_count": 0,
        "shadow_gentle_crossings_passed": 0,
        "shadow_knee_crossings": 0,
        "shadow_connected_line_count": 0,
        "shadow_face_set_neutral": True,
    }
    empty = ((), (), metrics)
    try:
        count = int(local["count"])
        first = np.asarray(local["first"], dtype=np.int32).reshape(-1)
        second = np.asarray(local["second"], dtype=np.int32).reshape(-1)
        edge_indices = np.asarray(local.get("pair_edge_indices", ()), dtype=np.int32).reshape(-1)
        pair_v0 = np.asarray(local.get("pair_v0", ()), dtype=np.int32).reshape(-1)
        pair_v1 = np.asarray(local.get("pair_v1", ()), dtype=np.int32).reshape(-1)
        local_global_ids = np.asarray(local.get("_global_face_ids", ()), dtype=np.int32).reshape(-1)
        analysis_ids = np.asarray(analysis_ids, dtype=np.int32).reshape(-1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        if (
            count <= 0 or len(first) != len(second) or len(first) != len(edge_indices)
            or len(first) != len(pair_v0) or len(first) != len(pair_v1)
            or len(local_global_ids) != count or len(analysis_ids) != count
            or np.any(first < 0) or np.any(second < 0)
            or np.any(first >= count) or np.any(second >= count)
            or np.any(analysis_ids < 0) or np.any(analysis_ids >= len(distances))
        ):
            metrics["shape_refine_gray_rejection_reason"] = "schema"
            metrics["shape_refine_add_reason"] = "schema"
            metrics["shape_refine_trim_reason"] = "schema"
            return empty
        allowed = np.asarray(
            usable_shape_rows if usable_shape_rows is not None else (), dtype=bool
        ).reshape(-1)
        if len(allowed) != len(first):
            metrics["shape_refine_gray_rejection_reason"] = "interval-schema"
            metrics["shape_refine_add_reason"] = "interval-schema"
            metrics["shape_refine_trim_reason"] = "interval-schema"
            return empty
        endpoint_reason = _fill_preview_boundary_endpoint_schema(
            state, geometry, local, edge_indices, pair_v0, pair_v1
        )
        if endpoint_reason is not None:
            metrics["shape_refine_gray_rejection_reason"] = endpoint_reason
            metrics["shape_refine_add_reason"] = endpoint_reason
            metrics["shape_refine_trim_reason"] = endpoint_reason
            return empty
        selected = np.isin(local_global_ids, original_ids)
        local_distances = distances[analysis_ids]
        crossing = selected[first] != selected[second]
        outside = np.where(selected[first], second, first)
        shape = _fill_preview_shape_boundary_mask(
            crossing, local_distances[outside], float(radius), edge_indices
        )
        shape_rows = np.flatnonzero(shape & (edge_indices >= 0) & allowed).astype(np.int32)
        metrics["shape_refine_gray_usable_edges"] = int(len(shape_rows))
        if len(shape_rows) < 6:
            metrics["shape_refine_gray_rejection_reason"] = "usable-interval-short"
            metrics["shape_refine_add_reason"] = "usable-interval-short"
            metrics["shape_refine_trim_reason"] = "usable-interval-short"
            return empty

        offsets = np.asarray(local["offsets"], dtype=np.int64).reshape(-1)
        neighbors = np.asarray(local["neighbors"], dtype=np.int32).reshape(-1)
        hidden = np.asarray(local.get("hidden", np.zeros(count, dtype=bool)), dtype=bool).reshape(-1)
        degree = np.diff(offsets) if len(offsets) == count + 1 else np.zeros(count, dtype=np.int32)
        edge_counts = np.asarray(
            local.get("face_edge_counts", degree), dtype=np.int32
        ).reshape(-1)
        if (
            len(offsets) != count + 1 or int(offsets[-1]) != len(neighbors)
            or len(hidden) != count or len(edge_counts) != count
        ):
            metrics["shape_refine_gray_rejection_reason"] = "graph-schema"
            metrics["shape_refine_add_reason"] = "graph-schema"
            metrics["shape_refine_trim_reason"] = "graph-schema"
            return empty
        hard = hidden | ~np.isfinite(local_distances) | (degree < edge_counts)
        protected = np.isin(local_global_ids, np.asarray(tuple(protected_geometry_ids), dtype=np.int32))
        protected |= hard
        tolerance = max(float(radius) * 1.0e-8, 1.0e-9)
        legal_face = (
            np.isfinite(local_distances)
            & (local_distances <= float(radius) + tolerance)
            & ~hidden & ~hard
        )

        # Corridor is measured in graph faces only to bound the physical-edge
        # field.  It is not a global N-ring morphology.
        corridor = np.zeros(count, dtype=bool)
        corridor[first[shape_rows]] = True
        corridor[second[shape_rows]] = True
        frontier = corridor.copy()
        for _ in range(2):
            grown = frontier.copy()
            for face in np.flatnonzero(frontier):
                grown[neighbors[int(offsets[face]) : int(offsets[face + 1])]] = True
            corridor |= grown
            frontier = grown

        vertex_id_space = str(local.get("vertex_id_space"))
        world_vertices = np.asarray(
            local.get("world_vertices", geometry.get("world_vertices", ())), dtype=np.float64
        )
        mesh = getattr(state.get("obj"), "data", None)
        matrix = getattr(state.get("obj"), "matrix_world", None)
        point_cache = {}

        def point(vertex):
            vertex = int(vertex)
            if vertex in point_cache:
                return point_cache[vertex]
            if vertex_id_space == "compact":
                if world_vertices.ndim != 2 or world_vertices.shape[1] != 3 or not (0 <= vertex < len(world_vertices)):
                    raise ValueError("endpoint-schema")
                value = np.asarray(world_vertices[vertex], dtype=np.float64)
            elif vertex_id_space == "mesh-global" and mesh is not None and matrix is not None:
                if vertex < 0 or vertex >= len(mesh.vertices):
                    raise ValueError("endpoint-schema")
                value = np.asarray(matrix @ mesh.vertices[vertex].co, dtype=np.float64)
            else:
                raise ValueError("endpoint-schema")
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("endpoint-schema")
            point_cache[vertex] = value
            return value

        # Build a bounded physical-edge field.  A duplicate edge row is
        # ambiguous and fails closed rather than mixing face-local identities.
        row_by_edge = {}
        shape_edges = set()
        corridor_rows = np.flatnonzero(
            (edge_indices >= 0) & (corridor[first] | corridor[second])
        ).astype(np.int32)
        for row in corridor_rows:
            edge = int(edge_indices[int(row)])
            if edge in row_by_edge:
                metrics["shape_refine_gray_rejection_reason"] = "duplicate-edge"
                metrics["shape_refine_add_reason"] = "duplicate-edge"
                metrics["shape_refine_trim_reason"] = "duplicate-edge"
                return empty
            if not (legal_face[int(first[row])] and legal_face[int(second[row])]):
                continue
            row_by_edge[edge] = int(row)
        for row in shape_rows:
            if int(edge_indices[int(row)]) in row_by_edge:
                shape_edges.add(int(edge_indices[int(row)]))
        if not shape_edges:
            metrics["shape_refine_gray_rejection_reason"] = "no-legal-edge"
            metrics["shape_refine_add_reason"] = "no-legal-edge"
            metrics["shape_refine_trim_reason"] = "no-legal-edge"
            return empty

        # Feature signal is evaluated for every corridor edge, while current
        # shape rows determine the interval baseline and face delta direction.
        normals = np.asarray(local.get("normals"), dtype=np.float64)
        contrast = np.asarray(local.get("contrast", np.zeros(count)), dtype=np.float64).reshape(-1)
        concavity = np.asarray(local.get("concavity", np.zeros(count)), dtype=np.float64).reshape(-1)
        directional = np.asarray(local.get("directional_valley", np.zeros(count)), dtype=np.float64).reshape(-1)
        crease = np.asarray(local.get("crease", np.zeros(count)), dtype=np.float64).reshape(-1)
        if (
            normals.shape != (count, 3)
            or any(len(values) != count for values in (contrast, concavity, directional, crease))
            or not all(np.all(np.isfinite(values)) for values in (normals, contrast, concavity, directional, crease))
        ):
            metrics["shape_refine_gray_rejection_reason"] = "feature-schema"
            metrics["shape_refine_add_reason"] = "feature-schema"
            metrics["shape_refine_trim_reason"] = "feature-schema"
            return empty
        row_signal = {}
        row_length = {}
        row_midpoint = {}
        for edge, row in row_by_edge.items():
            left, right = int(first[row]), int(second[row])
            dot = float(np.clip(np.dot(normals[left], normals[right]), -1.0, 1.0))
            normal_signal = math.acos(dot) / math.pi
            crease_signal = float(np.clip(max(crease[left], crease[right]) / math.pi, 0.0, 1.0))
            scalar_signal = float(np.clip(max(contrast[left], contrast[right], concavity[left], directional[left], contrast[right], concavity[right], directional[right]) / 0.10, 0.0, 1.0))
            values = (int(pair_v0[row]), int(pair_v1[row]))
            point_a = point(values[0])
            point_b = point(values[1])
            length = float(np.linalg.norm(point_b - point_a))
            if not np.isfinite(length) or length <= 1.0e-12:
                continue
            if bool(state.get("strict_mode", False)):
                outside_distance = float(min(local_distances[left], local_distances[right]))
                radius_prior = float(np.clip(1.0 - outside_distance / max(float(radius), 1.0e-20), 0.0, 1.0))
                geodesic_prior = float(np.clip(1.0 - abs(local_distances[left] - local_distances[right]) / max(float(radius), 1.0e-20), 0.0, 1.0))
                row_value = (
                    0.45 * normal_signal
                    + 0.20 * crease_signal
                    + 0.25 * scalar_signal
                    + 0.05 * radius_prior
                    + 0.05 * geodesic_prior
                )
            else:
                # Geometry-only signal. Radius/distance is a boundary
                # classifier, never a feature-strength prior. Ordinary E
                # uses only the fresh overlay-free viewport shadow sample;
                # capture failure is intentionally a zero signal rather than
                # a silent geometry fallback.
                row_value, tested = _fill_preview_shadow_pair_signal(
                    state, local, row
                )
                metrics["shadow_tested_crossings"] += int(tested > 0)
                if tested and row_value >= 1.0:
                    metrics["shadow_barrier_count"] += 1
                    metrics["shadow_knee_crossings"] += 1
                elif tested:
                    metrics["shadow_gentle_crossings_passed"] += 1
                metrics["shadow_face_set_neutral"] = True
            row_signal[edge] = float(np.clip(row_value, 0.0, 1.0))
            row_length[edge] = length
            row_midpoint[edge] = (point_a + point_b) * 0.5
        if not row_signal:
            metrics["shape_refine_gray_rejection_reason"] = "edge-data-unavailable"
            metrics["shape_refine_add_reason"] = "edge-data-unavailable"
            metrics["shape_refine_trim_reason"] = "edge-data-unavailable"
            return empty
        shape_lengths = np.asarray(
            [row_length[edge] for edge in sorted(shape_edges) if edge in row_length],
            dtype=np.float64,
        )
        context_spacing = float(np.median(shape_lengths)) if len(shape_lengths) else 0.0
        context_window = max(4.0 * context_spacing, context_spacing, 1.0e-12)
        shape_points = [row_midpoint[edge] for edge in shape_edges if edge in row_midpoint]
        context_edges = []
        if shape_points:
            for edge, midpoint in row_midpoint.items():
                if any(
                    float(np.linalg.norm(midpoint - anchor)) <= context_window + 1.0e-12
                    for anchor in shape_points
                ):
                    context_edges.append(edge)
        signal_values = np.asarray(
            [row_signal[edge] for edge in sorted(set(context_edges)) if edge in row_signal],
            dtype=np.float64,
        )
        metrics["shape_refine_gray_context_edge_count"] = int(len(signal_values))
        metrics["shape_refine_gray_context_window"] = float(context_window)
        metrics["shape_refine_gray_context_status"] = (
            "full-local-window" if len(signal_values) >= 6 else "insufficient-local-window"
        )
        if len(signal_values) < 6:
            # The normal E path keeps this boundary provisional so bootstrap
            # may explore a bounded cyan shell. Ctrl+E remains conservative.
            if not bool(state.get("strict_mode", False)):
                metrics["shape_refine_provisional"] = True
                metrics["shape_refine_gray_rejection_reason"] = "provisional-context-short"
                metrics["shape_refine_add_reason"] = "provisional-context-short"
                metrics["shape_refine_trim_reason"] = "provisional-context-short"
            else:
                metrics["shape_refine_gray_rejection_reason"] = "signal-short"
                metrics["shape_refine_add_reason"] = "signal-short"
                metrics["shape_refine_trim_reason"] = "signal-short"
            return empty
        median = float(np.median(signal_values))
        mad = float(1.4826 * np.median(np.abs(signal_values - median)))
        scale = max(mad, 1.0e-3)
        if mad <= 1.0e-6:
            # A constant strong edge field is valid, but it still needs a
            # non-zero Closing/Opening response before it can move a face.
            high = max(median, 0.08)
            low = max(0.90 * median, 0.06)
        else:
            high = max(median + 2.0 * scale, 0.08)
            low = max(median + scale, 0.06)
            # A two-level field can have a large MAD even when its upper
            # cluster is a legitimate short edge chain.  Cap the robust
            # threshold by the observed signal range; this is data-derived,
            # not a fixed feature-strength relaxation.
            signal_range = float(np.ptp(signal_values))
            if signal_range > 0.0:
                high = min(high, median + 0.75 * signal_range)
                low = min(low, high)
        metrics.update({
            "shape_refine_gray_signal_median": median,
            "shape_refine_gray_signal_mad": mad,
            "shape_refine_gray_hysteresis_high": high,
            "shape_refine_gray_hysteresis_low": low,
        })

        # Physical-edge components are the intervals.  Only components that
        # contain a current usable shape edge may produce a proposal.  The
        # exact partition helper is shared with candidate scoring below.
        edge_vertices = {}
        edge_face_normals = {}
        for edge in shape_edges:
            row = row_by_edge[edge]
            if edge not in row_signal:
                continue
            values = (int(pair_v0[row]), int(pair_v1[row]))
            edge_vertices[edge] = values
            left, right = int(first[row]), int(second[row])
            edge_face_normals[edge] = (normals[left], normals[right])
        intervals, junction_count, junction_pair_count, junction_ambiguous_count = (
            _fill_preview_gray_partition_intervals(
                tuple(edge_vertices),
                edge_vertices,
                row_signal,
                edge_face_normals,
                point,
            )
        )
        metrics["shape_refine_gray_junction_count"] = int(junction_count)
        metrics["shape_refine_gray_junction_pairs"] = int(junction_pair_count)
        metrics["shape_refine_gray_junction_ambiguous"] = int(junction_ambiguous_count)
        metrics["shape_refine_gray_interval_count"] = int(len(intervals))
        add_proposals = []
        trim_proposals = []
        add_reasons = []
        trim_reasons = []
        target_signature = []
        proposal_signatures = []
        window_factors = []
        window_distances = []
        for interval_index, component in enumerate(intervals):
            current_edges = [edge for edge in component if edge in shape_edges]
            if len(current_edges) < 6:
                add_reasons.append("interval-%d:short" % interval_index)
                trim_reasons.append("interval-%d:short" % interval_index)
                continue
            lengths = np.asarray([row_length[edge] for edge in component], dtype=np.float64)
            spacing = float(np.median(lengths))
            if not np.isfinite(spacing) or spacing <= 1.0e-12:
                continue
            # A simple path is required for deterministic chain morphology.
            incident = {}
            for edge in component:
                for vertex in edge_vertices[edge]:
                    incident.setdefault(vertex, []).append(edge)
            endpoints = [vertex for vertex, values in incident.items() if len(values) == 1]
            if len(endpoints) != 2 or any(len(values) > 2 for values in incident.values()):
                add_reasons.append("interval-%d:branch-or-cycle" % interval_index)
                trim_reasons.append("interval-%d:branch-or-cycle" % interval_index)
                continue
            ordered = []
            used = set()
            vertex = min(endpoints)
            while True:
                options = [edge for edge in incident.get(vertex, ()) if edge not in used]
                if not options:
                    break
                edge = min(options)
                used.add(edge)
                ordered.append(edge)
                a, b = edge_vertices[edge]
                vertex = b if vertex == a else a
            if len(ordered) != len(component):
                continue
            values = np.asarray([row_signal[edge] for edge in ordered], dtype=np.float64)
            ordered_lengths = np.asarray(
                [row_length[edge] for edge in ordered], dtype=np.float64
            )
            positions = np.zeros(len(ordered), dtype=np.float64)
            if len(ordered) > 1:
                positions[1:] = np.cumsum(ordered_lengths[:-1])
            # Derive a robust threshold independently at every physical
            # edge from a fixed world-distance neighborhood.  This prevents
            # extending the candidate, adding a remote ridge, or changing
            # the wheel radius from changing the threshold of an unchanged
            # local boundary.  Context comes from both sides of the current
            # candidate whenever the prepared source graph contains it.
            local_high = np.empty(len(values), dtype=np.float64)
            local_low = np.empty(len(values), dtype=np.float64)
            local_scale = np.empty(len(values), dtype=np.float64)
            local_medians = []
            local_mads = []
            for index, edge in enumerate(ordered):
                if bool(state.get("strict_mode", False)):
                    local_high[index] = high
                    local_low[index] = low
                    local_scale[index] = scale
                    local_medians.append(median)
                    local_mads.append(mad)
                    continue
                anchor = row_midpoint.get(edge)
                context_values = [
                    row_signal[other]
                    for other, midpoint in row_midpoint.items()
                    if anchor is not None
                    and float(np.linalg.norm(midpoint - anchor))
                    <= context_window + 1.0e-12
                    and other in row_signal
                ]
                if not context_values:
                    context_values = [float(values[index])]
                context_array = np.asarray(context_values, dtype=np.float64)
                local_high[index], local_low[index], local_median, local_mad, _context_status = (
                    _fill_preview_gray_local_threshold(context_array)
                )
                local_scale[index] = max(local_mad, 1.0e-3)
                local_medians.append(local_median)
                local_mads.append(local_mad)
            if local_medians:
                metrics["shape_refine_gray_signal_median"] = float(
                    np.median(np.asarray(local_medians, dtype=np.float64))
                )
                metrics["shape_refine_gray_signal_mad"] = float(
                    np.median(np.asarray(local_mads, dtype=np.float64))
                )
                metrics["shape_refine_gray_hysteresis_high"] = float(
                    np.median(local_high)
                )
                metrics["shape_refine_gray_hysteresis_low"] = float(
                    np.median(local_low)
                )
            # Keep the neighborhood metric-bounded, but adapt its width to
            # the local field rather than collapsing to a fixed two-edge
            # window.  A noisy/curved signal gets a little more context;
            # strongly non-uniform edge lengths trim the span so a long edge
            # cannot leap across a short local feature.  Both terms are
            # dimensionless and are clamped to the documented 1.25--4.0x
            # median-edge metric range.
            interval_median = float(np.median(local_medians)) if local_medians else median
            interval_mad = float(np.median(local_mads)) if local_mads else mad
            signal_variation = float(
                np.clip(interval_mad / max(abs(interval_median), 1.0e-6), 0.0, 2.0)
            )
            length_variation = float(
                np.clip(np.std(ordered_lengths) / max(spacing, 1.0e-12), 0.0, 2.0)
            )
            window_factor = float(
                np.clip(2.0 + 0.75 * signal_variation - 0.35 * length_variation, 1.25, 4.0)
            )
            window = window_factor * spacing
            window_factors.append(window_factor)
            window_distances.append(window)
            dilation = np.empty(len(values), dtype=np.float64)
            erosion = np.empty(len(values), dtype=np.float64)
            for index in range(len(values)):
                near = np.flatnonzero(np.abs(positions - positions[index]) <= window + 1.0e-12)
                dilation[index] = float(np.max(values[near]))
                erosion[index] = float(np.min(values[near]))
            close = np.empty(len(values), dtype=np.float64)
            opened = np.empty(len(values), dtype=np.float64)
            for index in range(len(values)):
                near = np.flatnonzero(np.abs(positions - positions[index]) <= window + 1.0e-12)
                close[index] = float(np.min(dilation[near]))
                opened[index] = float(np.max(erosion[near]))
            # Explicit grayscale closing/opening responses.  Hysteresis keeps
            # only a connected strong run and prevents isolated face spikes.
            close_response = close - values
            open_response = values - opened
            metrics["shape_refine_gray_close_response_max"] = max(metrics["shape_refine_gray_close_response_max"], float(np.max(close_response)))
            metrics["shape_refine_gray_open_response_max"] = max(metrics["shape_refine_gray_open_response_max"], float(np.max(open_response)))

            def hysteresis(field, high_values, low_values):
                strong = field >= high_values
                weak = field >= low_values
                keep = np.zeros(len(field), dtype=bool)
                pending = list(np.flatnonzero(strong))
                keep[strong] = True
                while pending:
                    index = int(pending.pop())
                    for other in (index - 1, index + 1):
                        if 0 <= other < len(field) and weak[other] and not keep[other]:
                            keep[other] = True
                            pending.append(other)
                return keep

            add_keep = hysteresis(close, local_high, local_low)
            trim_keep = hysteresis(values, local_high, local_low)
            # ``ordered.index`` makes a long physical chain quadratic.  Keep
            # the same deterministic edge ordering while resolving current
            # boundary positions in linear time.
            ordered_position = {
                edge: index for index, edge in enumerate(ordered)
            }
            shape_positions = {
                ordered_position[edge]
                for edge in current_edges
                if edge in ordered_position
            }
            # Only response-supported current boundary rows become face deltas.
            add_response_positions = {
                index for index in shape_positions
                if add_keep[index]
                and close_response[index] >= max(0.10 * local_scale[index], 0.01)
            }
            opening_anchor_positions = {
                index for index in shape_positions
                if open_response[index] >= max(0.10 * local_scale[index], 0.01)
            }
            # A complete hysteretic run with multiple Opening anchors is a
            # narrow overrun interval, even when an interior plateau has zero
            # point-wise response.  Permit the full current run as one
            # connected trim proposal; score/validator still decide whether
            # the immutable full candidate may adopt it.
            trim_full_interval = (
                len(current_edges) >= 6
                and shape_positions
                and shape_positions <= set(np.flatnonzero(trim_keep))
                and len(opening_anchor_positions) >= 2
            )
            # A Closing response may be concentrated in a short valley.  Once
            # it has two anchors, carry the hysteretic run across the whole
            # physical interval so the proposal remains one continuous chain.
            add_rows = [
                row_by_edge[ordered[index]] for index in shape_positions
                if len(add_response_positions) >= 2 and add_keep[index]
            ]
            trim_rows = [
                row_by_edge[ordered[index]] for index in shape_positions
                if trim_keep[index]
                and (
                    open_response[index] >= max(0.10 * local_scale[index], 0.01)
                    or trim_full_interval
                )
            ]
            if len(add_rows) >= 2:
                add_faces = []
                for row in add_rows:
                    out = int(outside[row])
                    if not selected[out] and legal_face[out] and not protected[out]:
                        add_faces.append(out)
                add_faces = np.unique(np.asarray(add_faces, dtype=np.int32))
                if len(add_faces) >= 2:
                    # Reconstruct a bounded local corridor band from the
                    # sparse boundary seeds.  The old implementation used
                    # one outside face per row, which can leave a valid
                    # physical chain represented by two disconnected face
                    # islands on a real mesh.  Flood only the already-built
                    # two-ring corridor and stop at the same immutable
                    # barriers as validation; this is a local graph
                    # reconstruction, not a global N-ring morphology.
                    # Include every face directly opposite the complete
                    # hysteretic target chain as a seed.  Then flood only
                    # through faces that do not sit on a further
                    # out-of-radius front; this prevents a local band from
                    # leaking into a remote sheet/escape branch while still
                    # bridging sparse real-mesh boundary rows.
                    target_band_faces = []
                    for index in np.flatnonzero(add_keep):
                        target_row = row_by_edge[ordered[int(index)]]
                        out = int(outside[target_row])
                        if (
                            not selected[out]
                            and legal_face[out]
                            and not protected[out]
                        ):
                            target_band_faces.append(out)
                    target_band_faces = np.unique(
                        np.asarray(target_band_faces, dtype=np.int32)
                    )
                    band_allowed = corridor & ~selected & legal_face & ~protected
                    near_external_front = np.zeros(count, dtype=bool)
                    for face in np.flatnonzero(band_allowed):
                        for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
                            neighbor = int(neighbor)
                            if (
                                not np.isfinite(local_distances[neighbor])
                                or local_distances[neighbor] > float(radius) + tolerance
                                or hidden[neighbor]
                            ):
                                near_external_front[face] = True
                                break
                    band_allowed &= ~near_external_front
                    band_allowed[target_band_faces] = True
                    seed_mask = np.zeros(count, dtype=bool)
                    seed_mask[np.unique(np.r_[add_faces, target_band_faces])] = True
                    band_seen = np.zeros(count, dtype=bool)
                    band_components = []
                    for seed in np.flatnonzero(seed_mask & band_allowed):
                        seed = int(seed)
                        if band_seen[seed]:
                            continue
                        band_seen[seed] = True
                        pending = [seed]
                        component_faces = []
                        while pending:
                            face = int(pending.pop())
                            component_faces.append(face)
                            for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
                                neighbor = int(neighbor)
                                if (
                                    band_allowed[neighbor]
                                    and not band_seen[neighbor]
                                ):
                                    band_seen[neighbor] = True
                                    pending.append(neighbor)
                        if component_faces:
                            band_components.append(np.asarray(sorted(component_faces), dtype=np.int32))
                    metrics["shape_refine_gray_corridor_band_components"] += int(len(band_components))
                    metrics["shape_refine_gray_corridor_band_faces"] += int(
                        sum(len(component_faces) for component_faces in band_components)
                    )
                    target_edges = tuple(
                        sorted(int(ordered[index]) for index in np.flatnonzero(add_keep))
                    )
                    # Validate/score each contiguous component independently.
                    # A disconnected side branch therefore cannot poison an
                    # otherwise useful target-chain proposal.
                    for component_faces in band_components:
                        if len(component_faces) < 2:
                            continue
                        component_mask = np.zeros(count, dtype=bool)
                        component_mask[component_faces] = True
                        candidate = local_global_ids[
                            np.flatnonzero(selected | component_mask)
                        ].astype(np.int32)
                        add_proposals.append((candidate, {
                            "gray_interval_index": int(interval_index),
                            "gray_target_edges": target_edges,
                            "gray_supported_edges": target_edges,
                            "gray_response": float(np.max(close_response)),
                            "gray_corridor_band_faces": int(len(component_faces)),
                        }))
                        metrics["shape_refine_gray_add_edges"] += int(np.count_nonzero(add_keep))
                        metrics["shape_refine_gray_add_faces"] += int(len(component_faces))
                        metrics["shape_refine_add_generated_faces"] += int(len(component_faces))
                        metrics["shape_refine_add_proposal_count"] += 1
                        target_signature.append(target_edges)
                        proposal_signatures.append((
                            "add", int(interval_index), target_edges,
                        ))
            if len(trim_rows) >= 2:
                trim_faces = []
                seed_value = int(state.get("seed_local", -1))
                for row in trim_rows:
                    inside = int(first[row]) if selected[int(first[row])] else int(second[row])
                    if selected[inside] and legal_face[inside] and not protected[inside] and inside != seed_value:
                        trim_faces.append(inside)
                trim_faces = np.unique(np.asarray(trim_faces, dtype=np.int32))
                if len(trim_faces) >= 2:
                    candidate_mask = selected.copy()
                    candidate_mask[trim_faces] = False
                    candidate = local_global_ids[np.flatnonzero(candidate_mask)].astype(np.int32)
                    trim_proposals.append((candidate, {
                        "gray_interval_index": int(interval_index),
                        "gray_target_edges": tuple(sorted(int(ordered[index]) for index in np.flatnonzero(trim_keep))),
                        "gray_supported_edges": tuple(sorted(int(ordered[index]) for index in np.flatnonzero(trim_keep))),
                        "gray_response": float(np.max(open_response)),
                    }))
                    metrics["shape_refine_gray_trim_edges"] += int(np.count_nonzero(trim_keep))
                    metrics["shape_refine_gray_trim_faces"] += int(len(trim_faces))
                    metrics["shape_refine_trim_generated_faces"] += int(len(trim_faces))
                    metrics["shape_refine_trim_proposal_count"] += 1
                    proposal_signatures.append((
                        "trim", int(interval_index),
                        tuple(sorted(int(ordered[index]) for index in np.flatnonzero(trim_keep))),
                    ))
            add_reasons.append("interval-%d:%s" % (interval_index, "proposal" if add_rows else "no-closing-response"))
            trim_reasons.append("interval-%d:%s" % (interval_index, "proposal" if trim_rows else "no-opening-response"))
        metrics["shape_refine_gray_target_chain"] = tuple(target_signature)
        metrics["shape_refine_gray_proposal_edge_signatures"] = tuple(proposal_signatures)
        if window_factors:
            metrics["shape_refine_gray_window_factor_min"] = float(min(window_factors))
            metrics["shape_refine_gray_window_factor_max"] = float(max(window_factors))
        if window_distances:
            metrics["shape_refine_gray_window_distance_min"] = float(min(window_distances))
            metrics["shape_refine_gray_window_distance_max"] = float(max(window_distances))
        metrics["shape_refine_add_reason"] = ";".join(add_reasons) if add_reasons else "no-interval"
        metrics["shape_refine_trim_reason"] = ";".join(trim_reasons) if trim_reasons else "no-interval"
        metrics["shape_refine_gray_guarded_edges"] = int(np.count_nonzero(~allowed & (edge_indices >= 0)))
        metrics["shape_refine_gray_rejection_reason"] = "proposal-generated" if (add_proposals or trim_proposals) else "no-clear-response"
        return tuple(trim_proposals), tuple(add_proposals), metrics
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        metrics["shape_refine_gray_rejection_reason"] = "error-fallback"
        metrics["shape_refine_add_reason"] = "error-fallback"
        metrics["shape_refine_trim_reason"] = "error-fallback"
        return empty


def _fill_preview_boundary_morphology_proposals(
    state, geometry, local, distances, analysis_ids, radius,
    preview_local_ids, protected_geometry_ids=(), usable_shape_rows=None,
):
    """Generate independent one-layer Closing/Opening proposals per edge interval.

    The add and trim helpers each receive the same immutable base candidate.
    An interval is a connected component of physical mesh edges, not a raw
    graph-face ring.  This keeps the morphology local and lets the caller
    compare every proposal with the unchanged candidate under one score.
    """
    import numpy as np

    # The former binary helper body is retained below only as a compatibility
    # reference for old diagnostics; all live callers use the grayscale path.
    return _fill_preview_boundary_grayscale_proposals(
        state, geometry, local, distances, analysis_ids, radius,
        preview_local_ids, protected_geometry_ids, usable_shape_rows,
    )

    original_ids = np.asarray(preview_local_ids, dtype=np.int32).reshape(-1)
    metrics = {
        "shape_refine_add_generated_faces": 0,
        "shape_refine_add_reason": "not-run",
        "shape_refine_add_eligible_pairs": 0,
        "shape_refine_add_gap_faces": 0,
        "shape_refine_add_proposal_count": 0,
        "shape_refine_trim_generated_faces": 0,
        "shape_refine_trim_reason": "not-run",
        "shape_refine_trim_outlier_edges": 0,
        "shape_refine_trim_gap_faces": 0,
        "shape_refine_trim_proposal_count": 0,
        "shape_refine_proposal_validation": (),
    }
    empty = ((), (), metrics)
    try:
        count = int(local["count"])
        first = np.asarray(local["first"], dtype=np.int32).reshape(-1)
        second = np.asarray(local["second"], dtype=np.int32).reshape(-1)
        edge_indices = np.asarray(
            local.get("pair_edge_indices", ()), dtype=np.int32
        ).reshape(-1)
        pair_v0 = np.asarray(local.get("pair_v0", ()), dtype=np.int32).reshape(-1)
        pair_v1 = np.asarray(local.get("pair_v1", ()), dtype=np.int32).reshape(-1)
        local_global_ids = np.asarray(
            local.get("_global_face_ids", ()), dtype=np.int32
        ).reshape(-1)
        analysis_ids = np.asarray(analysis_ids, dtype=np.int32).reshape(-1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        if (
            count <= 0
            or len(first) == 0
            or len(first) != len(second)
            or len(first) != len(edge_indices)
            or len(first) != len(pair_v0)
            or len(first) != len(pair_v1)
            or len(local_global_ids) != count
            or len(analysis_ids) != count
            or np.any(first < 0)
            or np.any(second < 0)
            or np.any(first >= count)
            or np.any(second >= count)
            or np.any(analysis_ids < 0)
            or np.any(analysis_ids >= len(distances))
        ):
            metrics["shape_refine_add_reason"] = "schema"
            metrics["shape_refine_trim_reason"] = "schema"
            return empty
        usable_shape_rows = np.asarray(
            usable_shape_rows if usable_shape_rows is not None else (),
            dtype=bool,
        ).reshape(-1)
        if len(usable_shape_rows) != len(first):
            metrics["shape_refine_add_reason"] = "interval-schema"
            metrics["shape_refine_trim_reason"] = "interval-schema"
            return empty
        endpoint_reason = _fill_preview_boundary_endpoint_schema(
            state, geometry, local, edge_indices, pair_v0, pair_v1
        )
        if endpoint_reason is not None:
            metrics["shape_refine_add_reason"] = endpoint_reason
            metrics["shape_refine_trim_reason"] = endpoint_reason
            return empty
        selected = np.isin(local_global_ids, original_ids)
        local_distances = distances[analysis_ids]
        crossing = selected[first] != selected[second]
        outside = np.where(selected[first], second, first)
        shape = _fill_preview_shape_boundary_mask(
            crossing, local_distances[outside], float(radius), edge_indices
        )
        shape_pairs = np.flatnonzero(
            shape & (edge_indices >= 0) & usable_shape_rows
        ).astype(np.int32)
        if len(shape_pairs) < 6:
            metrics["shape_refine_add_reason"] = "usable-interval-short"
            metrics["shape_refine_trim_reason"] = "usable-interval-short"
            return empty

        # Build physical-edge components from the explicit endpoint arrays.
        # Duplicate edge ids are ambiguous and are never turned into a
        # morphology proposal.
        edge_to_pair = {}
        vertex_to_edges = {}
        for pair in shape_pairs:
            edge = int(edge_indices[int(pair)])
            if edge in edge_to_pair:
                metrics["shape_refine_add_reason"] = "duplicate-edge"
                metrics["shape_refine_trim_reason"] = "duplicate-edge"
                return empty
            values = (int(pair_v0[int(pair)]), int(pair_v1[int(pair)]))
            edge_to_pair[edge] = int(pair)
            for vertex in values:
                vertex_to_edges.setdefault(vertex, []).append(edge)
        remaining = set(edge_to_pair)
        components = []
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            pending = [start]
            component = []
            while pending:
                edge = pending.pop()
                component.append(edge)
                pair = edge_to_pair[edge]
                for vertex in (int(pair_v0[pair]), int(pair_v1[pair])):
                    for other in vertex_to_edges.get(vertex, ()):
                        if other in remaining:
                            remaining.remove(other)
                            pending.append(other)
            components.append(tuple(sorted(component)))
        metrics["shape_refine_interval_count"] = int(len(components))
        add_proposals = []
        trim_proposals = []
        add_reasons = []
        trim_reasons = []
        validation_records = []
        protected_ids = np.asarray(tuple(protected_geometry_ids), dtype=np.int32)
        for interval_index, component in enumerate(components):
            if len(component) < 6:
                add_reasons.append("interval-%d:short" % interval_index)
                trim_reasons.append("interval-%d:short" % interval_index)
                continue
            allowed = np.zeros(len(first), dtype=bool)
            component_pairs = [edge_to_pair[edge] for edge in component]
            allowed[np.asarray(component_pairs, dtype=np.int32)] = True
            add_ids, add_details = _fill_preview_boundary_one_hop(
                state, geometry, local, distances, analysis_ids, radius,
                original_ids, protected_ids, allowed_shape_rows=allowed,
            )
            trim_ids, trim_details = _fill_preview_boundary_trim_bulge(
                state, geometry, local, distances, analysis_ids, radius,
                original_ids, protected_ids, allowed_shape_rows=allowed,
            )
            add_ids = np.unique(np.asarray(add_ids, dtype=np.int32).reshape(-1))
            trim_ids = np.unique(np.asarray(trim_ids, dtype=np.int32).reshape(-1))
            add_reasons.append(
                "interval-%d:%s" % (interval_index, add_details.get("one_hop_reason", "unknown"))
            )
            trim_reasons.append(
                "interval-%d:%s" % (interval_index, trim_details.get("boundary_trim_reason", "unknown"))
            )
            metrics["shape_refine_add_eligible_pairs"] += int(
                add_details.get("one_hop_eligible_pairs", 0)
            )
            metrics["shape_refine_add_gap_faces"] += int(
                add_details.get("one_hop_gap_faces", 0)
            )
            metrics["shape_refine_trim_outlier_edges"] += int(
                trim_details.get("boundary_trim_edges", 0)
            )
            if not np.array_equal(add_ids, original_ids):
                add_proposals.append(
                    (add_ids, dict(add_details, interval_index=int(interval_index)))
                )
                metrics["shape_refine_add_proposal_count"] += 1
                metrics["shape_refine_add_generated_faces"] += int(
                    np.count_nonzero(~np.isin(original_ids, add_ids))
                )
            if not np.array_equal(trim_ids, original_ids):
                trim_proposals.append(
                    (trim_ids, dict(trim_details, interval_index=int(interval_index)))
                )
                metrics["shape_refine_trim_proposal_count"] += 1
                metrics["shape_refine_trim_generated_faces"] += int(
                    np.count_nonzero(np.isin(original_ids, trim_ids))
                )
        metrics["shape_refine_add_reason"] = ";".join(add_reasons) if add_reasons else "no-interval"
        metrics["shape_refine_trim_reason"] = ";".join(trim_reasons) if trim_reasons else "no-interval"
        return tuple(trim_proposals), tuple(add_proposals), metrics
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        metrics["shape_refine_add_reason"] = "error-fallback"
        metrics["shape_refine_trim_reason"] = "error-fallback"
        return empty


def _fill_preview_refine_shape_boundary(
    state,
    geometry,
    local,
    distances,
    analysis_ids,
    radius,
    preview_local_ids,
    partition=None,
    full_candidate_ids=None,
):
    """Compare bounded add/trim proposals for one orange boundary corridor.

    The face graph resolver deliberately has one polarity: its Closing may
    add a narrow band, but it never removes from the immutable base.  This
    post-pass is the small, symmetric correction that was missing for the
    opposite seed direction.  The bounded bulge-trim and one-hop add helpers
    only *propose* candidates; this function gives current/add/trim the same
    boundary score and accepts a proposal only when it is a clear, connected
    improvement.
    Distance crossings, hard graph barriers, and all faces outside the
    one/two-ring corridor are immutable here.
    """
    import numpy as np

    metrics = {
        "shape_refine_attempted": False,
        "shape_refine_applied": False,
        "shape_refine_reason": "not-run",
        "shape_refine_candidates": 0,
        "shape_refine_proposal": "none",
        "shape_refine_gray_proposal_edge_signatures": (),
        "shape_refine_gray_final_edge_signature": (),
        "shape_refine_gray_adopted_proposal_edge_signature": (),
        "shape_refine_before_score": 0.0,
        "shape_refine_after_score": 0.0,
        "shape_refine_score_margin": 0.0,
        "shape_refine_added_faces": 0,
        "shape_refine_removed_faces": 0,
        "shape_refine_boundary_edges_before": 0,
        "shape_refine_boundary_edges_after": 0,
        "shadow_tested_crossings": 0,
        "shadow_barrier_count": 0,
        "shadow_gentle_crossings_passed": 0,
        "shadow_knee_crossings": 0,
        "shadow_connected_line_count": 0,
        "shadow_face_set_neutral": True,
        "shape_refine_signal_edges": 0,
        "shape_refine_cyan_guard_pairs": 0,
        "shape_refine_cyan_guard_faces": 0,
        "shape_refine_usable_pairs": 0,
        "shape_refine_cyan_skipped_pairs": 0,
        "shape_refine_interval_count": 0,
        "shape_refine_gray_interval_count": 0,
        "shape_refine_gray_usable_edges": 0,
        "shape_refine_gray_guarded_edges": 0,
        "shape_refine_gray_signal_median": 0.0,
        "shape_refine_gray_signal_mad": 0.0,
        "shape_refine_gray_close_response_max": 0.0,
        "shape_refine_gray_open_response_max": 0.0,
        "shape_refine_gray_add_edges": 0,
        "shape_refine_gray_trim_edges": 0,
        "shape_refine_gray_add_faces": 0,
        "shape_refine_gray_trim_faces": 0,
        "shape_refine_gray_hysteresis_high": 0.0,
        "shape_refine_gray_hysteresis_low": 0.0,
        "shape_refine_gray_window_factor_min": 0.0,
        "shape_refine_gray_window_factor_max": 0.0,
        "shape_refine_gray_window_distance_min": 0.0,
        "shape_refine_gray_window_distance_max": 0.0,
        "shape_refine_gray_target_chain": (),
        "shape_refine_gray_junction_count": 0,
        "shape_refine_gray_junction_pairs": 0,
        "shape_refine_gray_junction_ambiguous": 0,
        "shape_refine_gray_rejection_reason": "not-run",
        "shape_refine_gray_context_edge_count": 0,
        "shape_refine_gray_context_window": 0.0,
        "shape_refine_gray_context_status": "not-run",
        "shape_refine_provisional": False,
        "shape_refine_add_generated_faces": 0,
        "shape_refine_add_reason": "not-run",
        "shape_refine_add_eligible_pairs": 0,
        "shape_refine_add_gap_faces": 0,
        "shape_refine_add_proposal_count": 0,
        "shape_refine_trim_generated_faces": 0,
        "shape_refine_trim_reason": "not-run",
        "shape_refine_trim_outlier_edges": 0,
        "shape_refine_trim_gap_faces": 0,
        "shape_refine_trim_proposal_count": 0,
        "shape_refine_proposal_validation": (),
    }
    original_ids = np.unique(
        np.asarray(
            preview_local_ids if full_candidate_ids is None else full_candidate_ids,
            dtype=np.int32,
        ).reshape(-1)
    )
    try:
        full_count = int(geometry["count"])
        full_first = np.asarray(geometry["first"], dtype=np.int32).reshape(-1)
        full_second = np.asarray(geometry["second"], dtype=np.int32).reshape(-1)
        full_edge_indices = np.asarray(
            geometry.get("pair_edge_indices", ()), dtype=np.int32
        ).reshape(-1)
        full_offsets = np.asarray(geometry["offsets"], dtype=np.int64).reshape(-1)
        full_neighbors = np.asarray(geometry["neighbors"], dtype=np.int32).reshape(-1)
        full_hidden = np.asarray(geometry["hidden"], dtype=bool).reshape(-1)
        full_edge_counts = np.asarray(
            geometry.get("face_edge_counts", ()), dtype=np.int32
        ).reshape(-1)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        metrics["shape_refine_reason"] = "schema"
        return original_ids, metrics
    try:
        count = int(local["count"])
        first = np.asarray(local["first"], dtype=np.int32).reshape(-1)
        second = np.asarray(local["second"], dtype=np.int32).reshape(-1)
        edge_indices = np.asarray(
            local.get("pair_edge_indices", ()), dtype=np.int32
        ).reshape(-1)
        local_global_ids = np.asarray(
            local.get("_global_face_ids", ()), dtype=np.int32
        ).reshape(-1)
        analysis_ids = np.asarray(analysis_ids, dtype=np.int32).reshape(-1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        offsets = np.asarray(local["offsets"], dtype=np.int64).reshape(-1)
        neighbors = np.asarray(local["neighbors"], dtype=np.int32).reshape(-1)
        if (
            count <= 0
            or len(first) == 0
            or len(first) != len(second)
            or len(first) != len(edge_indices)
            or len(local_global_ids) != count
            or len(analysis_ids) != count
            or len(offsets) != count + 1
            or int(offsets[-1]) != len(neighbors)
            or np.any(first < 0)
            or np.any(second < 0)
            or np.any(first >= count)
            or np.any(second >= count)
            or np.any(analysis_ids < 0)
            or np.any(analysis_ids >= len(distances))
            or full_count <= 0
            or len(full_first) != len(full_second)
            or len(full_first) != len(full_edge_indices)
            or len(full_offsets) != full_count + 1
            or int(full_offsets[-1]) != len(full_neighbors)
            or len(full_hidden) != full_count
            or len(full_edge_counts) != full_count
            or len(distances) != full_count
            or np.any(full_first < 0)
            or np.any(full_second < 0)
            or np.any(full_first >= full_count)
            or np.any(full_second >= full_count)
            or np.any(full_neighbors < 0)
            or np.any(full_neighbors >= full_count)
            or np.any(full_offsets[1:] < full_offsets[:-1])
        ):
            metrics["shape_refine_reason"] = "schema"
            return original_ids, metrics
        if np.any(original_ids < 0) or np.any(original_ids >= full_count):
            metrics["shape_refine_reason"] = "full-candidate-schema"
            return original_ids, metrics
        if np.any(local_global_ids < 0) or np.any(local_global_ids >= full_count):
            metrics["shape_refine_reason"] = "local-id-schema"
            return original_ids, metrics
        if len(np.unique(local_global_ids)) != count:
            metrics["shape_refine_reason"] = "local-id-schema"
            return original_ids, metrics
        full_pair_lookup = {}
        for full_pair_index, (left, right) in enumerate(zip(full_first, full_second)):
            key = (
                min(int(left), int(right)),
                max(int(left), int(right)),
                int(full_edge_indices[full_pair_index]),
            )
            full_pair_lookup.setdefault(key, []).append(int(full_pair_index))
        original_full_mask = np.zeros(full_count, dtype=bool)
        original_full_mask[original_ids] = True
        seed_full = int(state.get("seed_local", -1))
        if seed_full < 0 or seed_full >= full_count:
            metrics["shape_refine_reason"] = "seed-schema"
            return original_ids, metrics
        local_distances = distances[analysis_ids]
        selected = np.isin(local_global_ids, original_ids)
        local_original_ids = local_global_ids[selected].astype(np.int32, copy=False)
        crossing = selected[first] != selected[second]
        outside = np.where(selected[first], second, first)
        shape = _fill_preview_shape_boundary_mask(
            crossing,
            local_distances[outside],
            float(radius),
            edge_indices,
        )
        shape_pairs = np.flatnonzero(shape & (edge_indices >= 0)).astype(np.int32)
        metrics["shape_refine_boundary_edges_before"] = int(len(shape_pairs))
        if len(shape_pairs) < 6 or len(shape_pairs) > _FILL_PREVIEW_BOUNDARY_ROUTE_SHAPE_PAIR_BUDGET:
            if len(shape_pairs) < 6 and not bool(state.get("strict_mode", False)):
                # Ordinary E treats a short/under-context boundary as
                # provisional.  The preview bootstrap may advance a bounded
                # cyan shell; it must not silently turn uncertainty into a
                # permanent orange terminal.  Ctrl+E retains the strict,
                # conservative early return.
                metrics["shape_refine_provisional"] = True
                metrics["shape_refine_reason"] = "provisional-short-context"
            else:
                metrics["shape_refine_reason"] = "shape-interval-short-or-budget"
            return original_ids, metrics

        # The correction corridor is the current orange boundary plus at most
        # two face rings.  The add proposal is a bounded local Closing and the
        # trim proposal is its bounded complement Opening; neither is a raw
        # N-ring morphology over the full candidate.  Keeping this compact is
        # important: a broad opening can otherwise look numerically attractive
        # on a dense patch.
        corridor = np.zeros(count, dtype=bool)
        corridor[np.unique(np.r_[first[shape_pairs], second[shape_pairs]])] = True
        frontier = corridor.copy()
        for _ in range(2):
            grown = frontier.copy()
            for face in np.flatnonzero(frontier):
                grown[neighbors[int(offsets[face]) : int(offsets[face + 1])]] = True
            corridor |= grown
            frontier = grown
        distance_boundary = crossing & ~shape
        # A cyan crossing at one terminal of a long orange interval should
        # not suppress the whole correction.  Find only shape-pair rows that
        # actually touch that terminal, protect that row plus one adjacent
        # shape row, and keep the interior interval in the shared score.
        # Invalid/non-edge crossings remain a hard comparison gap and retain
        # the old all-or-nothing behavior.
        shape_pair_by_face = {}
        for pair_position, pair in enumerate(shape_pairs):
            for face in (int(first[pair]), int(second[pair])):
                shape_pair_by_face.setdefault(face, []).append(int(pair_position))

        def count_shape_intervals(pair_mask):
            pair_mask = np.asarray(pair_mask, dtype=bool).reshape(-1)
            remaining = set(
                int(value)
                for value in np.flatnonzero(pair_mask[shape_pairs])
            )
            interval_count = 0
            while remaining:
                interval_count += 1
                pending = [remaining.pop()]
                while pending:
                    pair_position = pending.pop()
                    pair = int(shape_pairs[pair_position])
                    for face in (int(first[pair]), int(second[pair])):
                        for neighbor_position in shape_pair_by_face.get(face, ()):
                            if neighbor_position in remaining:
                                remaining.remove(neighbor_position)
                                pending.append(neighbor_position)
            return int(interval_count)

        distance_contact_positions = set()
        valid_distance_rows = distance_boundary & (edge_indices >= 0)
        for pair in np.flatnonzero(valid_distance_rows):
            for face in (int(first[pair]), int(second[pair])):
                distance_contact_positions.update(shape_pair_by_face.get(face, ()))
        guard_pair_positions = set(distance_contact_positions)
        if guard_pair_positions:
            # One pair hop is deliberately the minimum terminal guard.  It
            # protects two pairs at an ordinary endpoint while avoiding a
            # broad face-ring exclusion on dense real meshes.
            adjacent_positions = set()
            for pair_position in guard_pair_positions:
                pair = int(shape_pairs[pair_position])
                pair_faces = (int(first[pair]), int(second[pair]))
                for face in pair_faces:
                    adjacent_positions.update(shape_pair_by_face.get(face, ()))
            guard_pair_positions.update(adjacent_positions)
            guard_pair_rows = np.zeros(len(first), dtype=bool)
            guard_pair_rows[shape_pairs[np.asarray(sorted(guard_pair_positions), dtype=np.int32)]] = True
            guard_faces = np.zeros(count, dtype=bool)
            guard_pairs_array = shape_pairs[np.asarray(sorted(guard_pair_positions), dtype=np.int32)]
            guard_faces[first[guard_pairs_array]] = True
            guard_faces[second[guard_pairs_array]] = True
            for pair in np.flatnonzero(valid_distance_rows):
                if guard_pair_rows[pair] or any(
                    guard_faces[int(face)] for face in (int(first[pair]), int(second[pair]))
                ):
                    guard_faces[first[pair]] = True
                    guard_faces[second[pair]] = True
            # One graph ring around the terminal faces blocks a proposal from
            # redrawing a connector immediately next to the protected cyan
            # endpoint.  The full cyan crossing equality check below remains
            # authoritative for every geometry-local pair.
            guard_frontier = guard_faces.copy()
            guard_ring = guard_faces.copy()
            for face in np.flatnonzero(guard_frontier):
                guard_ring[
                    neighbors[int(offsets[face]) : int(offsets[face + 1])]
                ] = True
            guard_faces = guard_ring
            # The guard face ring is immutable, but its first interior
            # transition edge is still allowed to become an orange shape
            # crossing when a proposal starts just beyond the guard.  Freeze
            # only the contacted cyan rows and the terminal shape rows; a
            # blanket ``touches guard face`` rule would make every valid
            # interior add impossible.
            guard_rows = guard_pair_rows | valid_distance_rows
            usable_shape_rows = (
                shape & (edge_indices >= 0) & ~guard_rows
            )
            metrics["shape_refine_cyan_guard_pairs"] = int(len(guard_pair_positions))
            metrics["shape_refine_cyan_guard_faces"] = int(np.count_nonzero(guard_faces))
            metrics["shape_refine_cyan_skipped_pairs"] = int(
                np.count_nonzero(shape & ~usable_shape_rows)
            )
            metrics["shape_refine_usable_pairs"] = int(np.count_nonzero(usable_shape_rows))
            # Count contiguous usable runs in the original row adjacency.  A
            # disconnected interior is scored independently by the common
            # chain evaluator, but the metric makes endpoint-only skips
            # visible in the preview diagnostics.
            usable_positions = np.flatnonzero(usable_shape_rows).astype(np.int32)
            metrics["shape_refine_interval_count"] = count_shape_intervals(usable_shape_rows)
            if len(usable_positions) < 6:
                metrics["shape_refine_reason"] = "distance-boundary-endpoint-only"
                return original_ids, metrics
        elif np.any(
            distance_boundary
            & (corridor[first] | corridor[second])
        ):
            # A nearby cyan/non-edge crossing that is not a terminal contact
            # cannot be oriented safely relative to this shape interval.
            metrics["shape_refine_reason"] = "distance-boundary-near-shape"
            return original_ids, metrics
        else:
            guard_faces = np.zeros(count, dtype=bool)
            guard_rows = np.zeros(len(first), dtype=bool)
            usable_shape_rows = shape & (edge_indices >= 0)
            metrics["shape_refine_usable_pairs"] = int(np.count_nonzero(usable_shape_rows))
            metrics["shape_refine_interval_count"] = count_shape_intervals(usable_shape_rows)
        protected = np.zeros(count, dtype=bool)
        if partition is not None:
            raw_protected = np.asarray(
                partition.get("protected", ()), dtype=bool
            ).reshape(-1)
            if len(raw_protected) == count:
                protected |= raw_protected
        protected |= guard_faces
        protected_full = np.zeros(full_count, dtype=bool)
        if np.any(protected):
            protected_full[local_global_ids[protected]] = True
        protected_ids = local_global_ids[protected]
        metrics["shape_refine_attempted"] = True

        # Boundary score -------------------------------------------------
        # The score is shared by all three masks.  It rewards an existing
        # mesh-edge chain with a normal/crease/valley signal, while charging
        # chain turning and endpoint gaps.  The absolute threshold below is
        # intentionally low; the improvement margin and a multi-edge signal
        # gate reject planes, broad basins, and single-edge bulges.
        normals = np.asarray(local.get("normals"), dtype=np.float64)
        if normals.shape != (count, 3):
            metrics["shape_refine_reason"] = "normal-schema"
            return original_ids, metrics
        contrast = np.asarray(
            local.get("contrast", np.zeros(count)), dtype=np.float64
        ).reshape(-1)
        concavity = np.asarray(
            local.get("concavity", np.zeros(count)), dtype=np.float64
        ).reshape(-1)
        directional = np.asarray(
            local.get("directional_valley", np.zeros(count)), dtype=np.float64
        ).reshape(-1)
        crease = np.asarray(
            local.get("crease", np.zeros(count)), dtype=np.float64
        ).reshape(-1)
        if any(len(values) != count for values in (contrast, concavity, directional, crease)):
            metrics["shape_refine_reason"] = "feature-schema"
            return original_ids, metrics
        finite_features = all(
            np.all(np.isfinite(values))
            for values in (normals, contrast, concavity, directional, crease)
        )
        if not finite_features:
            metrics["shape_refine_reason"] = "nonfinite-feature"
            return original_ids, metrics

        # Use the already prepared world endpoints in the explicitly declared
        # id space.  Compact cursor slices do not consult mesh-global ids;
        # tests can supply world_vertices without constructing a bpy mesh.
        mesh = getattr(state.get("obj"), "data", None)
        matrix = getattr(state.get("obj"), "matrix_world", None)
        world_vertices = np.asarray(
            local.get("world_vertices", geometry.get("world_vertices", ())),
            dtype=np.float64,
        )
        pair_v0 = np.asarray(local.get("pair_v0", ()), dtype=np.int32).reshape(-1)
        pair_v1 = np.asarray(local.get("pair_v1", ()), dtype=np.int32).reshape(-1)
        endpoint_reason = _fill_preview_boundary_endpoint_schema(
            state, geometry, local, edge_indices, pair_v0, pair_v1
        )
        if endpoint_reason is not None:
            metrics["shape_refine_reason"] = endpoint_reason
            return original_ids, metrics
        vertex_id_space = str(local.get("vertex_id_space"))
        point_cache = {}

        def point(vertex):
            vertex = int(vertex)
            cached = point_cache.get(vertex)
            if cached is not None:
                return cached
            if vertex_id_space == "mesh-global" and mesh is not None and matrix is not None:
                value = np.asarray(matrix @ mesh.vertices[vertex].co, dtype=np.float64)
            elif vertex_id_space == "compact" and (
                world_vertices.ndim == 2
                and world_vertices.shape[1] == 3
                and 0 <= vertex < len(world_vertices)
            ):
                value = np.asarray(world_vertices[vertex], dtype=np.float64)
            else:
                raise ValueError("edge endpoint data unavailable")
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("edge endpoint data unavailable")
            point_cache[vertex] = value
            return value

        def edge_vertices(pair):
            edge = int(edge_indices[int(pair)])
            if (
                0 <= int(pair) < len(pair_v0)
                and len(pair_v0) == len(edge_indices)
                and len(pair_v1) == len(edge_indices)
                and int(pair_v0[int(pair)]) >= 0
                and int(pair_v1[int(pair)]) >= 0
            ):
                return int(pair_v0[int(pair)]), int(pair_v1[int(pair)])
            return None

        def candidate_score(candidate_ids):
            candidate_ids = np.asarray(candidate_ids, dtype=np.int32).reshape(-1)
            mask = np.isin(local_global_ids, candidate_ids)
            candidate_crossing = mask[first] != mask[second]
            candidate_outside = np.where(mask[first], second, first)
            candidate_shape = _fill_preview_shape_boundary_mask(
                candidate_crossing,
                local_distances[candidate_outside],
                float(radius),
                edge_indices,
            ) & (edge_indices >= 0)
            candidate_shape &= corridor[first] | corridor[second]
            candidate_shape &= ~guard_rows
            pairs = np.flatnonzero(candidate_shape).astype(np.int32)
            if len(pairs) == 0:
                return -float("inf"), 0, 0, False
            edge_to_pairs = {}
            for pair in pairs:
                edge_to_pairs.setdefault(int(edge_indices[pair]), []).append(int(pair))
            # Multiple graph rows for one mesh edge are malformed; score the
            # physical edge once and refuse its ambiguous duplicate.
            if any(len(values) != 1 for values in edge_to_pairs.values()):
                return -float("inf"), len(pairs), 0, False
            edge_ids = sorted(edge_to_pairs)
            edge_points = {}
            edge_signal = {}
            edge_face_normals = {}
            for edge in edge_ids:
                pair = edge_to_pairs[edge][0]
                values = edge_vertices(pair)
                if values is None:
                    return -float("inf"), len(pairs), 0, False
                a, b = values
                pa, pb = point(a), point(b)
                length = float(np.linalg.norm(pb - pa))
                if not np.isfinite(length) or length <= 1.0e-12:
                    return -float("inf"), len(pairs), 0, False
                edge_points[edge] = (a, b, pa, pb, length)
                left, right = int(first[pair]), int(second[pair])
                edge_face_normals[edge] = (normals[left], normals[right])
                dot = float(np.clip(np.dot(normals[left], normals[right]), -1.0, 1.0))
                normal_signal = math.acos(dot) / math.pi
                crease_signal = float(np.clip(max(crease[left], crease[right]) / math.pi, 0.0, 1.0))
                scalar_signal = max(
                    float(contrast[left]), float(contrast[right]),
                    float(concavity[left]), float(concavity[right]),
                    float(directional[left]), float(directional[right]),
                )
                scalar_signal = float(np.clip(scalar_signal / 0.10, 0.0, 1.0))
                if bool(state.get("strict_mode", False)):
                    edge_value = (
                        0.55 * normal_signal
                        + 0.20 * crease_signal
                        + 0.25 * scalar_signal
                    )
                else:
                    edge_value, tested = _fill_preview_shadow_pair_signal(
                        state, local, pair
                    )
                    metrics["shadow_tested_crossings"] += int(tested > 0)
                    if tested and edge_value >= 1.0:
                        metrics["shadow_barrier_count"] += 1
                        metrics["shadow_knee_crossings"] += 1
                    elif tested:
                        metrics["shadow_gentle_crossings_passed"] += 1
                    metrics["shadow_face_set_neutral"] = True
                edge_signal[edge] = float(np.clip(edge_value, 0.0, 1.0))
            chains, _, _, _ = _fill_preview_gray_partition_intervals(
                edge_ids,
                {edge: edge_points[edge][:2] for edge in edge_ids},
                edge_signal,
                edge_face_normals,
                point,
            )
            if not chains:
                return -float("inf"), len(pairs), 0, False
            spacing = float(np.median([item[4] for item in edge_points.values()]))
            best_score = -float("inf")
            best_signal_edges = 0
            best_count = 0
            for chain in chains:
                if len(chain) < 6:
                    continue
                total_length = sum(edge_points[edge][4] for edge in chain)
                weighted_signal = sum(edge_signal[edge] * edge_points[edge][4] for edge in chain)
                mean_signal = weighted_signal / max(total_length, 1.0e-20)
                if bool(state.get("strict_mode", False)):
                    strong_edges = sum(edge_signal[edge] >= 0.08 for edge in chain)
                else:
                    chain_signal_values = np.asarray(
                        [edge_signal[edge] for edge in chain], dtype=np.float64
                    )
                    chain_median = float(np.median(chain_signal_values))
                    chain_mad = float(
                        1.4826
                        * np.median(np.abs(chain_signal_values - chain_median))
                    )
                    strong_threshold = max(chain_median + chain_mad, 0.08)
                    strong_edges = sum(
                        edge_signal[edge] >= strong_threshold for edge in chain
                    )
                # Order a simple chain by walking from its endpoint.  A cycle
                # is intentionally not a correction candidate.
                incident = {}
                for edge in chain:
                    a, b = edge_points[edge][:2]
                    incident.setdefault(a, []).append(edge)
                    incident.setdefault(b, []).append(edge)
                endpoints = [vertex for vertex, values in incident.items() if len(values) == 1]
                if len(endpoints) != 2:
                    continue
                ordered = []
                used = set()
                vertex = endpoints[0]
                while True:
                    next_edges = [edge for edge in incident.get(vertex, ()) if edge not in used]
                    if not next_edges:
                        break
                    edge = min(next_edges)
                    used.add(edge)
                    ordered.append(edge)
                    a, b = edge_points[edge][:2]
                    vertex = b if vertex == a else a
                if len(ordered) != len(chain):
                    continue
                turns = 0.0
                for left_edge, right_edge in zip(ordered, ordered[1:]):
                    left_a, left_b = edge_points[left_edge][:2]
                    right_a, right_b = edge_points[right_edge][:2]
                    shared = left_a if left_a in (right_a, right_b) else left_b
                    left_other = left_b if shared == left_a else left_a
                    right_other = right_b if shared == right_a else right_a
                    va = point(left_other) - point(shared)
                    vb = point(right_other) - point(shared)
                    denominator = max(float(np.linalg.norm(va) * np.linalg.norm(vb)), 1.0e-20)
                    # The vectors point from the shared vertex to each edge's
                    # outer endpoint.  A straight chain therefore has an
                    # angle of pi, while a fold-back has zero; use the
                    # supplementary angle as the actual turn so continuity
                    # scores straight=1 and fold-back=0.
                    angle = math.acos(
                        float(np.clip(np.dot(va, vb) / denominator, -1.0, 1.0))
                    )
                    turns += math.pi - angle
                # Smooth long curves are valid: turn is normalized by the
                # number of intervals and only gently reduces the score.
                mean_turn = turns / max(len(chain) - 1, 1)
                continuity = float(np.clip(1.0 - mean_turn / math.pi, 0.0, 1.0))
                if bool(state.get("strict_mode", False)):
                    score = math.log1p(total_length / max(spacing, 1.0e-20)) * (
                        0.08 + mean_signal
                    ) * (0.55 + 0.45 * continuity)
                else:
                    # A fixed local metric window caps the length
                    # contribution; enlarging the candidate/radius cannot
                    # make a remote chain look like a stronger improvement.
                    score_window = max(4.0 * spacing, spacing, 1.0e-20)
                    score = math.log1p(
                        min(total_length, score_window) / max(spacing, 1.0e-20)
                    ) * (0.08 + mean_signal) * (0.55 + 0.45 * continuity)
                if score > best_score:
                    best_score = score
                    best_signal_edges = strong_edges
                    best_count = len(chain)
            signal_ok = best_signal_edges >= 4 and best_count >= 6 and best_score > -float("inf")
            return float(best_score), len(pairs), int(best_signal_edges), signal_ok

        before_score, before_edges, before_signal_edges, before_signal_ok = candidate_score(original_ids)
        metrics["shape_refine_before_score"] = float(before_score if np.isfinite(before_score) else 0.0)
        metrics["shape_refine_signal_edges"] = int(before_signal_edges)
        # A weak current boundary is not itself a reason to stop.  In the
        # add-only polarity the missing band is precisely what can carry the
        # stronger physical-edge signal.  Candidate scoring below still
        # requires a multi-edge signal and a clear margin, so a plane or a
        # broad basin remains unchanged; this gate only lets a strong add or
        # trim proposal be compared with that weak baseline.
        if not before_signal_ok and not np.isfinite(before_score):
            metrics["shape_refine_reason"] = "weak-shape-signal"
            return original_ids, metrics

        # Proposals are intentionally generated independently from the same
        # immutable current mask.  The live path is the grayscale physical
        # edge morphology helper; it does not seed a second pass from an add.
        trim_proposals, add_proposals, morphology_metrics = _fill_preview_boundary_morphology_proposals(
            state, geometry, local, distances, analysis_ids, radius,
            local_original_ids, protected_ids, usable_shape_rows,
        )
        metrics.update(morphology_metrics)
        proposals = []
        for ids, details in trim_proposals:
            proposals.append(("trim", np.asarray(ids, dtype=np.int32).reshape(-1), details))
        for ids, details in add_proposals:
            proposals.append(("add", np.asarray(ids, dtype=np.int32).reshape(-1), details))
        metrics["shape_refine_candidates"] = int(1 + len(proposals))

        def validate(candidate_ids):
            candidate_ids = np.unique(
                np.asarray(candidate_ids, dtype=np.int32).reshape(-1)
            )
            if len(candidate_ids) == 0:
                return False, "empty", None
            if np.any(~np.isin(candidate_ids, local_global_ids)):
                return False, "outside-local", None
            # Proposals are analysis-local, but the candidate state is a
            # complete geometry-local mask.  Only analysis faces may change;
            # analysis-outside Closing/valley faces are copied verbatim.
            candidate = np.isin(local_global_ids, candidate_ids)
            candidate_full = original_full_mask.copy()
            candidate_full[local_global_ids] = candidate
            seed_matches = np.flatnonzero(
                local_global_ids == int(state.get("seed_local", -1))
            )
            if len(seed_matches) == 0 or not candidate[int(seed_matches[0])]:
                return False, "seed", None
            if not candidate_full[seed_full]:
                return False, "seed", None
            tolerance = max(float(radius) * 1.0e-8, 1.0e-9)
            local_domain = np.isfinite(local_distances) & (
                local_distances <= float(radius) + tolerance
            )
            if np.any(candidate & ~local_domain):
                return False, "radius-outside", None
            full_domain = np.isfinite(distances) & (
                distances <= float(radius) + tolerance
            ) & ~full_hidden
            if np.any(candidate_full & ~full_domain):
                return False, "radius-outside", None
            changed = candidate != selected
            changed_full = candidate_full != original_full_mask
            if not np.any(changed_full):
                return False, "unchanged", None
            # The hard-face mask is geometry-local and shares the same
            # source of truth as Closing: hidden/non-finite faces and every
            # degree-deficit face are immutable.  A trim may preserve every
            # hard-edge crossing and still delete a hard face itself, so
            # reject that proposal before the crossing XOR check.
            full_degree = np.diff(full_offsets)
            hard_full = (
                full_hidden
                | ~np.isfinite(distances)
                | (full_degree < full_edge_counts)
            )
            if np.any(changed_full & hard_full):
                return False, "changed-hard-face", None
            if np.any(changed & protected) or np.any(changed_full & protected_full):
                return False, "protected", None
            if np.any(changed & ~corridor):
                return False, "corridor", None
            analysis_mask = np.zeros(full_count, dtype=bool)
            analysis_mask[local_global_ids] = True
            if np.any(candidate_full[~analysis_mask] != original_full_mask[~analysis_mask]):
                return False, "outside-analysis", None

            # Compare the complete geometry-local cyan crossing set, not just
            # the compact analysis slice.
            full_crossing = original_full_mask[full_first] != original_full_mask[full_second]
            full_outside = np.where(original_full_mask[full_first], full_second, full_first)
            full_shape = _fill_preview_shape_boundary_mask(
                full_crossing,
                distances[full_outside],
                float(radius),
                full_edge_indices,
            )
            distance_pairs = full_crossing & ~full_shape
            candidate_crossing = candidate_full[full_first] != candidate_full[full_second]
            candidate_outside = np.where(candidate_full[full_first], full_second, full_first)
            candidate_shape = _fill_preview_shape_boundary_mask(
                candidate_crossing,
                distances[candidate_outside],
                float(radius),
                full_edge_indices,
            )
            candidate_distance_pairs = candidate_crossing & ~candidate_shape
            if np.any(candidate_distance_pairs != distance_pairs):
                return False, "distance-boundary", None
            guard_full_rows = np.zeros(len(full_first), dtype=bool)
            for local_pair_index in np.flatnonzero(guard_rows):
                left = int(local_global_ids[int(first[local_pair_index])])
                right = int(local_global_ids[int(second[local_pair_index])])
                key = (
                    min(left, right), max(left, right),
                    int(edge_indices[int(local_pair_index)]),
                )
                full_matches = full_pair_lookup.get(key)
                if not full_matches:
                    return False, "cyan-guard-schema", None
                guard_full_rows[np.asarray(full_matches, dtype=np.int32)] = True
            if np.any(candidate_crossing[guard_full_rows] != full_crossing[guard_full_rows]):
                return False, "cyan-guard", None

            # Hard barriers are geometry-local.  Do not look up a local id in
            # geometry["face_ids"]: that array is the canonical mesh-face map,
            # not a geometry-local index map.
            hard_pairs = hard_full[full_first] | hard_full[full_second]
            if np.any(candidate_crossing[hard_pairs] != full_crossing[hard_pairs]):
                return False, "hard-barrier", None

            # Seed connectivity and changed-interval checks use the complete
            # geometry graph so a preserved analysis-outside component cannot
            # be accidentally discarded or disconnected.
            connected = np.zeros(full_count, dtype=bool)
            connected[seed_full] = True
            pending = [seed_full]
            while pending:
                face = pending.pop()
                for neighbor in full_neighbors[int(full_offsets[face]) : int(full_offsets[face + 1])]:
                    neighbor = int(neighbor)
                    if candidate_full[neighbor] and not connected[neighbor]:
                        connected[neighbor] = True
                        pending.append(neighbor)
            if not np.all(connected[candidate_full]):
                return False, "disconnected", None
            changed_seen = np.zeros(full_count, dtype=bool)
            changed_components = 0
            for start_face in np.flatnonzero(changed_full):
                start_face = int(start_face)
                if changed_seen[start_face]:
                    continue
                changed_components += 1
                changed_seen[start_face] = True
                pending = [start_face]
                while pending:
                    face = pending.pop()
                    for neighbor in full_neighbors[int(full_offsets[face]) : int(full_offsets[face + 1])]:
                        neighbor = int(neighbor)
                        if changed_full[neighbor] and not changed_seen[neighbor]:
                            changed_seen[neighbor] = True
                            pending.append(neighbor)
            if changed_components != 1:
                return False, "non-contiguous", None
            try:
                gap_ids, _gap_metrics = _fill_preview_enclosed_gap_faces(
                    geometry, distances, float(radius),
                    np.flatnonzero(candidate_full).astype(np.int32),
                )
                if len(gap_ids):
                    return False, "gap", None
            except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError):
                return False, "gap-check", None
            return True, "ok", (candidate, candidate_full)

        best = None
        rejected = []
        for name, candidate_ids, details in proposals:
            valid, reason, candidate_masks = validate(candidate_ids)
            if not valid:
                rejected.append(f"{name}:{reason}")
                continue
            score, edge_count, signal_edges, signal_ok = candidate_score(candidate_ids)
            if not signal_ok or not np.isfinite(score):
                rejected.append(f"{name}:weak-score")
                continue
            if best is None or score > best[0]:
                best = (float(score), name, candidate_ids, candidate_masks, int(edge_count))
        margin = max(abs(float(before_score)), 0.05) * 0.08
        if best is None or best[0] <= float(before_score) + margin:
            metrics["shape_refine_reason"] = (
                "changed-hard-face"
                if any(
                    str(reason).endswith(":changed-hard-face")
                    for reason in rejected
                )
                else "no-clear-improvement"
            )
            if rejected:
                metrics["shape_refine_rejections"] = tuple(rejected)
                metrics["shape_refine_proposal_validation"] = tuple(rejected)
            return original_ids, metrics
        _score, name, candidate_ids, candidate_masks, after_edges = best
        _candidate_mask, candidate_full = candidate_masks
        changed = candidate_full != original_full_mask
        # Adoption evidence is extracted from the validated complete mask,
        # never copied from a rejected or pre-add proposal.  Sorting the
        # physical edge ids makes this signature direction-independent and
        # preserves the complete chain (a single common edge is insufficient).
        final_crossing = candidate_full[full_first] != candidate_full[full_second]
        final_outside = np.where(candidate_full[full_first], full_second, full_first)
        final_shape = _fill_preview_shape_boundary_mask(
            final_crossing,
            distances[final_outside],
            float(radius),
            full_edge_indices,
        ) & (full_edge_indices >= 0)
        final_signature = tuple(sorted(
            set(int(edge) for edge in full_edge_indices[final_shape])
        ))
        adopted_signature = tuple()
        for proposal_name, proposal_ids, proposal_details in proposals:
            if proposal_name == name and np.array_equal(
                np.unique(np.asarray(proposal_ids, dtype=np.int32)),
                np.unique(np.asarray(candidate_ids, dtype=np.int32)),
            ):
                adopted_signature = tuple(sorted(
                    int(edge)
                    for edge in proposal_details.get("gray_supported_edges", ())
                ))
                break
        metrics.update(
            {
                "shape_refine_applied": True,
                "shape_refine_reason": "applied",
                "shape_refine_proposal": str(name),
                "shape_refine_after_score": float(_score),
                "shape_refine_score_margin": float(_score - before_score),
                "shape_refine_added_faces": int(np.count_nonzero(changed & candidate_full)),
                "shape_refine_removed_faces": int(np.count_nonzero(changed & original_full_mask)),
                "shape_refine_boundary_edges_after": int(after_edges),
                "shape_refine_gray_final_edge_signature": final_signature,
                "shape_refine_gray_adopted_proposal_edge_signature": adopted_signature,
            }
        )
        return np.flatnonzero(candidate_full).astype(np.int32), metrics
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        metrics["shape_refine_reason"] = "error-fallback"
        return original_ids, metrics


def _fill_preview_revalidate_valley_components(
    geometry, distances, radius, analysis_ids, fine_selected_ids, records
):
    """Revalidate and return complete valley components for the fine pass.

    Records use mesh polygon ids.  The current geometry is mapped through its
    own ``face_ids`` once, then all component faces are checked together; the
    valley faces do not have to be present in the precise partition that
    supplied the shore faces.
    """
    import numpy as np

    if not records:
        return np.empty(0, dtype=np.int32), "none"
    try:
        count = int(geometry["count"])
        canonical = np.asarray(geometry["face_ids"], dtype=np.int64).reshape(-1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        hidden = np.asarray(geometry["hidden"], dtype=bool).reshape(-1)
        offsets = np.asarray(geometry["offsets"], dtype=np.int64).reshape(-1)
        neighbors = np.asarray(geometry["neighbors"], dtype=np.int32).reshape(-1)
        edge_counts = np.asarray(geometry["face_edge_counts"], dtype=np.int32).reshape(-1)
        first = np.asarray(geometry["first"], dtype=np.int32).reshape(-1)
        second = np.asarray(geometry["second"], dtype=np.int32).reshape(-1)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return np.empty(0, dtype=np.int32), "schema"
    if (
        len(canonical) != count
        or len(distances) < count
        or len(hidden) != count
        or len(offsets) != count + 1
        or int(offsets[-1]) != len(neighbors)
        or len(edge_counts) != count
        or len(first) != len(second)
        or len(set(int(value) for value in canonical)) != count
    ):
        return np.empty(0, dtype=np.int32), "schema"
    if np.any(neighbors < 0) or np.any(neighbors >= count):
        return np.empty(0, dtype=np.int32), "graph"
    degree = np.diff(offsets).astype(np.int64, copy=False)
    local_by_mesh = {int(mesh_id): index for index, mesh_id in enumerate(canonical)}
    analysis_ids = np.asarray(analysis_ids, dtype=np.int32).reshape(-1)
    fine_selected_ids = np.asarray(fine_selected_ids, dtype=np.int32).reshape(-1)
    analysis_set = set(int(value) for value in analysis_ids)
    selected_set = set(int(value) for value in fine_selected_ids)
    pair_set = {
        (int(left), int(right))
        for left, right in zip(first, second)
    }
    pair_set |= {(right, left) for left, right in tuple(pair_set)}
    tolerance = max(float(radius) * 1.0e-8, 1.0e-9)
    fills = []
    for record in records:
        try:
            valley_ids = tuple(int(value) for value in record["valley_face_ids"])
            shore_ids = tuple(int(value) for value in record["shore_face_ids"])
            interface_edges = tuple(record["interface_edges"])
            scale = float(record.get("scale", 1.0))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if not valley_ids or len(set(valley_ids)) != len(valley_ids):
            continue
        if any(value not in local_by_mesh for value in valley_ids + shore_ids):
            continue
        valley_local = tuple(local_by_mesh[value] for value in valley_ids)
        shore_local = tuple(local_by_mesh[value] for value in shore_ids)
        surviving_shore_ids = tuple(
            shore_id
            for shore_id, shore_index in zip(shore_ids, shore_local)
            if shore_index in selected_set
        )
        if len(surviving_shore_ids) < 2:
            continue
        if any(value not in analysis_set for value in valley_local):
            continue
        if any(
            not np.isfinite(distances[value])
            or distances[value] > float(radius) + tolerance
            or bool(hidden[value])
            or int(degree[value]) < int(edge_counts[value])
            for value in valley_local
        ):
            continue
        # Component continuity and interface protection are checked against
        # the current full graph, not against proxy patch-local ids.
        component_set = set(valley_local)
        seen = {valley_local[0]}
        pending = [valley_local[0]]
        protected = False
        while pending:
            face = pending.pop()
            for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
                neighbor = int(neighbor)
                if neighbor in component_set and neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
                # Distance clipping is handled by the component-face checks
                # above; a neighbor outside the requested radius alone is
                # not a protected boundary.  Hidden neighbors remain hard
                # stops during fine revalidation.
                elif neighbor not in component_set and bool(hidden[neighbor]):
                    protected = True
        if protected or len(seen) != len(component_set):
            continue
        valid_edges = []
        for edge in interface_edges:
            try:
                valley_id, shore_id, point_a, point_b = edge
                valley_id = int(valley_id)
                shore_id = int(shore_id)
                valley_local_id = local_by_mesh[valley_id]
                shore_local_id = local_by_mesh[shore_id]
                if (
                    shore_id not in surviving_shore_ids
                    or (valley_local_id, shore_local_id) not in pair_set
                ):
                    continue
                valid_edges.append((valley_id, shore_id, point_a, point_b))
            except (KeyError, TypeError, ValueError):
                continue
        if not valid_edges:
            continue
        if not _fill_preview_valley_component_is_crossing(
            geometry,
            valley_ids,
            surviving_shore_ids,
            valid_edges,
            scale=scale,
        ):
            continue
        fills.extend(valley_local)
    return np.unique(np.asarray(fills, dtype=np.int32)), "ok"


def _fill_preview_terminal_reset(state):
    """Drop the normal-mode terminal frontier when its provenance is stale."""
    if not isinstance(state, dict):
        return
    state["terminal_base"] = None
    state["terminal_base_radius"] = None
    state["terminal_base_signature"] = None
    state["terminal_frontier"] = None
    state["terminal_expansion_count"] = 0
    state["terminal_coverage_certificate"] = None


def _fill_preview_terminal_barriers(result):
    """Return shape-boundary transitions as canonical local face pairs."""
    barriers = set()
    if not isinstance(result, dict):
        return barriers
    for record in result.get("boundary_records", ()) or ():
        if not isinstance(record, dict) or not bool(record.get("shape")):
            continue
        try:
            first = int(record["geometry_face_a"])
            second = int(record["geometry_face_b"])
        except (KeyError, TypeError, ValueError):
            continue
        if first != second and first >= 0 and second >= 0:
            barriers.add((min(first, second), max(first, second)))
    return barriers


def _fill_preview_terminal_coverage_certified(state, geometry, requested_radius):
    """Prove the retained graph covers every face through this radius.

    The terminal walk is not allowed to treat an exhausted heap as proof that
    the requested range is complete: a cursor-local/cropped graph can exhaust
    before reaching the real mesh boundary.  Normal mode adjacency is a full
    mesh graph, while diagnostics and future cropped builders can explicitly
    mark their coverage state.
    """
    import numpy as np

    if not isinstance(geometry, dict):
        return False
    try:
        requested_radius = float(requested_radius)
    except (TypeError, ValueError):
        return False
    certificate = state.get("terminal_coverage_certificate")
    signature = state.get("signature")
    if (
        isinstance(certificate, dict)
        and certificate.get("signature") == signature
        and certificate.get("geometry_id") == id(geometry)
    ):
        state["terminal_coverage_reuse_count"] = int(
            state.get("terminal_coverage_reuse_count", 0)
        ) + 1
        return bool(certificate.get("certified")) and requested_radius >= 0.0
    if "terminal_coverage_certified" in geometry:
        certified = bool(geometry.get("terminal_coverage_certified"))
        state["terminal_coverage_certificate"] = {
            "signature": signature,
            "geometry_id": id(geometry),
            "certified": certified,
        }
        state["terminal_coverage_scan_count"] = int(
            state.get("terminal_coverage_scan_count", 0)
        ) + 1
        return certified and requested_radius >= 0.0
    try:
        count = int(geometry.get("count", 0))
        face_ids = np.asarray(
            geometry.get("face_ids", np.arange(count, dtype=np.int32)),
            dtype=np.int64,
        ).reshape(-1)
        offsets = np.asarray(geometry["offsets"], dtype=np.int64).reshape(-1)
        neighbors = np.asarray(geometry["neighbors"], dtype=np.int32).reshape(-1)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if count <= 0 or len(face_ids) != count or len(offsets) != count + 1:
        return False
    if len(np.unique(face_ids)) != count or len(neighbors) != int(offsets[-1]):
        return False
    obj = state.get("obj")
    mesh = getattr(obj, "data", None)
    polygons = getattr(mesh, "polygons", None)
    try:
        mesh_count = len(polygons) if polygons is not None else None
    except (ReferenceError, TypeError):
        mesh_count = None
    if mesh_count is not None:
        if int(mesh_count) != count:
            return False
        if not np.array_equal(np.sort(face_ids), np.arange(count, dtype=np.int64)):
            return False
    # A full graph's exact adjacency is the coverage certificate.  The
    # requested radius is intentionally accepted only after that certificate,
    # never because the retained heap happened to become empty.
    state["terminal_coverage_certificate"] = {
        "signature": signature,
        "geometry_id": id(geometry),
        "certified": True,
        "count": count,
        "mesh_count": mesh_count,
    }
    state["terminal_coverage_scan_count"] = int(
        state.get("terminal_coverage_scan_count", 0)
    ) + 1
    return bool(requested_radius >= 0.0)


def _fill_preview_terminal_boundary_cache(geometry, result, local_ids, barriers):
    """Build the edge ownership cache used by large-region frontier growth.

    This is intentionally a one-time full pass when a terminal base is
    captured.  Subsequent normal-mode growth updates only edges incident to
    newly reached faces.  Duplicate face pairs are treated as non-manifold
    provenance and disable the differential path rather than guessing which
    physical edge owns a transition.
    """
    import numpy as np

    try:
        count = int(geometry["count"])
        first = np.asarray(geometry["first"], dtype=np.int32).reshape(-1)
        second = np.asarray(geometry["second"], dtype=np.int32).reshape(-1)
        pair_v0 = np.asarray(geometry["pair_v0"], dtype=np.int32).reshape(-1)
        pair_v1 = np.asarray(geometry["pair_v1"], dtype=np.int32).reshape(-1)
        vertices = np.asarray(geometry["world_vertices"], dtype=np.float64).reshape((-1, 3))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if (
        count <= 0
        or len(first) != len(second)
        or len(pair_v0) != len(first)
        or len(pair_v1) != len(first)
        or not np.all((first >= 0) & (first < count))
        or not np.all((second >= 0) & (second < count))
    ):
        return None
    pair_to_edge = {}
    for edge, (face_a, face_b) in enumerate(zip(first, second)):
        key = (min(int(face_a), int(face_b)), max(int(face_a), int(face_b)))
        if key[0] == key[1] or key in pair_to_edge:
            # A repeated face pair is ambiguous for differential ownership.
            return None
        v0, v1 = int(pair_v0[edge]), int(pair_v1[edge])
        if min(v0, v1) < 0 or max(v0, v1) >= len(vertices):
            return None
        pair_to_edge[key] = int(edge)
    selected = {int(value) for value in np.asarray(local_ids, dtype=np.int32).reshape(-1)}
    boundary_edges = {}
    for edge, (face_a, face_b) in enumerate(zip(first, second)):
        inside_a = int(face_a) in selected
        inside_b = int(face_b) in selected
        if inside_a != inside_b:
            key = (min(int(face_a), int(face_b)), max(int(face_a), int(face_b)))
            boundary_edges[int(edge)] = bool(key in barriers)
    # Existing records are a useful provenance assertion.  Records which do
    # not map to a retained graph edge make the differential result unsafe.
    for record in result.get("boundary_records", ()) or ():
        if not isinstance(record, dict):
            return None
        try:
            key = (
                min(int(record["geometry_face_a"]), int(record["geometry_face_b"])),
                max(int(record["geometry_face_a"]), int(record["geometry_face_b"])),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if key not in pair_to_edge:
            return None
    try:
        confirm_domain = {
            int(value)
            for value in np.asarray(
                result.get("confirm_domain_ids", ()), dtype=np.int32
            ).reshape(-1)
            if 0 <= int(value) < count
        }
    except (AttributeError, TypeError, ValueError):
        return None
    confirm_domain.update(int(value) for value in selected)
    return {
        "pair_to_edge": pair_to_edge,
        "boundary_edges": boundary_edges,
        "confirm_domain": confirm_domain,
        "edge_count": int(len(first)),
    }


def _fill_preview_terminal_store(state, result, radius):
    """Retain one exact normal result as the base of the final range stages.

    The retained frontier starts with the already selected faces and the
    distances from the ordinary Dijkstra pass.  Future growth can then walk
    only new faces while respecting the shape (orange) transitions captured by
    the base result.  Missing provenance deliberately disables this fast path.
    """
    import heapq
    import numpy as np

    _fill_preview_terminal_reset(state)
    if bool(state.get("strict_mode", False)) or not isinstance(result, dict):
        return False
    geometry = state.get("adjacency")
    if not isinstance(geometry, dict) or state.get("signature") is None:
        return False
    if not _fill_preview_terminal_coverage_certified(state, geometry, radius):
        return False
    try:
        count = int(geometry.get("count", 0))
        face_ids = np.asarray(
            geometry.get("face_ids", np.arange(count, dtype=np.int32)),
            dtype=np.int32,
        ).reshape(-1)
        faces = np.asarray(result.get("faces", ()), dtype=np.int32).reshape(-1)
        distances = np.asarray(
            state.get("distance_state", {}).get("distances"), dtype=np.float64
        ).reshape(-1)
        offsets = np.asarray(geometry["offsets"], dtype=np.int64).reshape(-1)
        neighbors = np.asarray(geometry["neighbors"], dtype=np.int32).reshape(-1)
        lengths = np.asarray(geometry["neighbor_lengths"], dtype=np.float64).reshape(-1)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if (
        count <= 0 or len(face_ids) != count or len(distances) < count
        or len(offsets) != count + 1 or len(neighbors) != len(lengths)
        or len(faces) == 0
    ):
        return False
    local_ids = np.flatnonzero(np.isin(face_ids, faces)).astype(np.int32, copy=False)
    if len(local_ids) != len(faces):
        return False
    hidden = np.asarray(geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool).reshape(-1)
    if len(hidden) != count or np.any(hidden[local_ids]):
        return False
    seed_matches = np.flatnonzero(face_ids == int(state.get("seed_face", -1)))
    if len(seed_matches) != 1 or int(seed_matches[0]) not in set(int(value) for value in local_ids):
        return False
    # The terminal walk may start from every exact selected face, but the base
    # itself must still be one connected selected island.  Otherwise a full
    # resolver is required to prove that no disconnected island is being
    # joined by the lightweight frontier.
    selected_set = set(int(value) for value in local_ids)
    reachable = {int(seed_matches[0])}
    stack = [int(seed_matches[0])]
    offsets = np.asarray(geometry["offsets"], dtype=np.int64).reshape(-1)
    neighbors = np.asarray(geometry["neighbors"], dtype=np.int32).reshape(-1)
    if len(offsets) != count + 1:
        return False
    while stack:
        current = stack.pop()
        for edge in range(int(offsets[current]), int(offsets[current + 1])):
            other = int(neighbors[edge])
            if other in selected_set and other not in reachable:
                reachable.add(other)
                stack.append(other)
    if reachable != selected_set:
        return False
    barriers = _fill_preview_terminal_barriers(result)
    # A result with a boundary but no pair provenance cannot safely expand.
    if result.get("boundary_records") and not barriers and result.get("shape_segments"):
        return False
    boundary_cache = _fill_preview_terminal_boundary_cache(
        geometry, result, local_ids, barriers
    )
    if boundary_cache is None:
        return False
    frontier_distances = np.full(count, np.inf, dtype=np.float64)
    heap = []
    # Only selected faces on an unblocked exterior transition can seed future
    # growth.  Interior selected faces are already behind those boundary
    # neighbors and need not be pushed into the first delta queue.
    local_id_set = {int(value) for value in local_ids}
    seed_locals = []
    for local in local_ids:
        local = int(local)
        distance = float(distances[local])
        if not np.isfinite(distance):
            return False
        for edge in range(int(offsets[local]), int(offsets[local + 1])):
            other = int(neighbors[edge])
            pair = (min(local, other), max(local, other))
            if other not in local_id_set and pair not in barriers:
                seed_locals.append(local)
                break
    for local in seed_locals:
        distance = float(distances[local])
        frontier_distances[local] = distance
        heapq.heappush(heap, (distance, local))
    state["terminal_base"] = {
        "result": dict(result),
        "face_ids": np.array(face_ids, dtype=np.int32, copy=True),
        "base_local_ids": np.array(local_ids, dtype=np.int32, copy=True),
        "barriers": frozenset(barriers),
        "signature": state.get("signature"),
        "seed_face": int(state.get("seed_face", -1)),
    }
    state["terminal_base_radius"] = float(radius)
    state["terminal_base_signature"] = state.get("signature")
    state["terminal_frontier"] = {
        "geometry": geometry,
        "distances": frontier_distances,
        "heap": heap,
        "popped": 0,
        "selected": set(int(value) for value in local_ids),
        "barriers": frozenset(barriers),
        "last_radius": float(radius),
        "pair_to_edge": boundary_cache["pair_to_edge"],
        "boundary_edges": boundary_cache["boundary_edges"],
        "confirm_domain": boundary_cache["confirm_domain"],
        "boundary_cache_valid": True,
        "last_new_faces": (),
        "last_delta_result": None,
        "frontier_seed_count": int(len(seed_locals)),
    }
    state["terminal_expansion_count"] = 0
    return True


def _fill_preview_terminal_delta_result(
    state, base, frontier, radius, newly_reached, popped_before, base_radius
):
    """Publish a frontier-only result for a large normal-mode selection.

    The selected set and confirm domain are mutable only inside ``frontier``;
    the returned candidate owns fresh face/domain arrays for this generation.
    No all-edge selected-mask scan is performed here.  The one unavoidable
    O(N) operation is materialising the immutable face array consumed by the
    renderer/confirm writer.
    """
    import numpy as np

    geometry = frontier.get("geometry")
    if not isinstance(geometry, dict) or not frontier.get("boundary_cache_valid"):
        return None
    try:
        count = int(geometry["count"])
        first = np.asarray(geometry["first"], dtype=np.int32).reshape(-1)
        second = np.asarray(geometry["second"], dtype=np.int32).reshape(-1)
        pair_v0 = np.asarray(geometry["pair_v0"], dtype=np.int32).reshape(-1)
        pair_v1 = np.asarray(geometry["pair_v1"], dtype=np.int32).reshape(-1)
        world_vertices = np.asarray(
            geometry["world_vertices"], dtype=np.float64
        ).reshape((-1, 3))
        face_ids = np.asarray(base["face_ids"], dtype=np.int32).reshape(-1)
        selected = frontier["selected"]
        pair_to_edge = frontier["pair_to_edge"]
        boundary_edges = frontier["boundary_edges"]
        barriers = frontier.get("barriers", frozenset())
        confirm_domain = frontier["confirm_domain"]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if (
        len(face_ids) != count
        or len(first) != len(second)
        or len(pair_v0) != len(first)
        or len(pair_v1) != len(first)
        or len(selected) == 0
        or not isinstance(confirm_domain, set)
    ):
        return None
    previous_delta = frontier.get("last_delta_result")
    if not newly_reached and isinstance(previous_delta, dict):
        # A wheel tick can increase the requested radius without crossing a
        # new face.  Reuse the already materialised immutable arrays instead
        # of sorting the large selected set again.
        result = dict(previous_delta)
        result.update({
            "radius": float(radius),
            "patch_edge_reached": bool(frontier.get("heap")),
            "progressive_range_cache_hit": False,
            "progressive_range_terminal_expansion": True,
            "progressive_range_newly_processed_faces": 0,
            "progressive_range_reused_faces": int(len(selected)),
            "progressive_range_selected_sort_count": 0,
            "created_generation": int(state.get("generation", 0)),
        })
        return result
    # A newly reached face can only change ownership on its incident edges.
    # Updating the canonical pair map also makes an unexpected adjacency
    # mismatch a safe full-compute fallback instead of a guessed boundary.
    offsets = np.asarray(geometry.get("offsets"), dtype=np.int64).reshape(-1)
    neighbors = np.asarray(geometry.get("neighbors"), dtype=np.int32).reshape(-1)
    if len(offsets) != count + 1:
        return None
    newly_set = {int(value) for value in newly_reached}
    working_boundary_edges = dict(boundary_edges)
    for face in newly_reached:
        face = int(face)
        if face < 0 or face >= count:
            return None
        for adjacency_edge in range(int(offsets[face]), int(offsets[face + 1])):
            other = int(neighbors[adjacency_edge])
            key = (min(face, other), max(face, other))
            edge = pair_to_edge.get(key)
            if edge is None or edge < 0 or edge >= len(first):
                return None
            inside = (
                (int(first[edge]) in selected or int(first[edge]) in newly_set)
                != (int(second[edge]) in selected or int(second[edge]) in newly_set)
            )
            if inside:
                working_boundary_edges[int(edge)] = bool(key in barriers)
            else:
                working_boundary_edges.pop(int(edge), None)
    # Materialize the selected union once.  In particular, do not sort the
    # retained selected set and then sort/concatenate it again for a new ring.
    selected_union = set(int(value) for value in selected)
    selected_union.update(newly_set)
    local_selected = np.asarray(sorted(selected_union), dtype=np.int32)
    if len(local_selected) == 0:
        return None
    # Build only the currently exposed edge rows, not a mask over all graph
    # edges.  Non-manifold/invalid support is rejected here and falls back to
    # the ordinary exact resolver in the caller.
    geometry_face_ids = np.asarray(
        geometry.get("face_ids", np.arange(count, dtype=np.int32)), dtype=np.int32
    ).reshape(-1)
    if len(geometry_face_ids) != count:
        return None
    shape_segments, distance_segments, boundary_records = [], [], []
    shape_boundary_flags = []
    for edge, shape in sorted(working_boundary_edges.items()):
        first_face, second_face = int(first[edge]), int(second[edge])
        v0, v1 = int(pair_v0[edge]), int(pair_v1[edge])
        if min(v0, v1) < 0 or max(v0, v1) >= len(world_vertices):
            return None
        segment = (
            tuple(float(value) for value in world_vertices[v0]),
            tuple(float(value) for value in world_vertices[v1]),
        )
        shape = bool(shape)
        shape_boundary_flags.append(shape)
        boundary_records.append({
            "geometry_face_a": first_face,
            "geometry_face_b": second_face,
            "mesh_face_a": int(geometry_face_ids[first_face]),
            "mesh_face_b": int(geometry_face_ids[second_face]),
            "mesh_edge": -1,
            "shape": shape,
        })
        (shape_segments if shape else distance_segments).append(segment)
    base_result = base.get("result") if isinstance(base, dict) else None
    if not isinstance(base_result, dict):
        return None
    confirm_geometry = base_result.get("confirm_geometry")
    if not isinstance(confirm_geometry, dict):
        return None
    # Confirm-domain provenance may already contain a face that is entering
    # the selected set on this tick (for example a radius+epsilon support
    # face).  Form a true set union before the single immutable sort so no
    # duplicate domain ids reach either backend writer.
    confirm_union = set(int(value) for value in confirm_domain)
    confirm_union.update(newly_set)
    confirm_domain_ids = np.asarray(sorted(confirm_union), dtype=np.int32)
    seed_matches = np.flatnonzero(face_ids == int(state.get("seed_face", -1)))
    if len(seed_matches) != 1:
        return None
    result = dict(base_result)
    result.update({
        "radius": float(radius),
        "faces": np.asarray(face_ids[local_selected], dtype=np.int32).copy(),
        "candidate_count": int(len(local_selected)),
        "boundary_edge_count": int(len(boundary_records)),
        "boundary_all_orange": bool(
            boundary_records
            and _fill_preview_should_use_green_boundary(
                np.asarray(shape_boundary_flags, dtype=bool),
                np.ones(len(shape_boundary_flags), dtype=bool),
            )
        ),
        "shape_segments": shape_segments,
        "distance_segments": distance_segments,
        "boundary_records": tuple(boundary_records),
        "confirm_geometry": confirm_geometry,
        "confirm_seed_local": int(seed_matches[0]),
        "confirm_domain_ids": confirm_domain_ids,
        "confirm_signature": state.get("signature"),
        "patch_edge_reached": bool(frontier.get("heap")),
        "compute_seconds": 0.0,
        "geometry": geometry,
        "progressive_range_cache_hit": False,
        "progressive_range_terminal_expansion": True,
        "progressive_range_terminal_mode": "frontier-delta",
        "progressive_range_terminal_base_radius": float(base_radius),
        "progressive_range_newly_processed_faces": int(len(newly_reached)),
        "progressive_range_reused_faces": int(max(0, len(local_selected) - len(newly_reached))),
        "progressive_range_full_boundary_scan": False,
        "progressive_range_confirm_geometry_reused": True,
        "progressive_range_selected_sort_count": 1,
        "progressive_range_confirm_sort_count": 1,
        "progressive_range_new_face_dedupe_mode": "set",
        "progressive_range_new_face_unique_count": int(len(newly_set)),
        "progressive_range_delta_boundary_edges": int(len(boundary_records)),
        "created_generation": int(state.get("generation", 0)),
    })
    # Commit all mutable frontier state only after the complete result has
    # passed provenance/schema validation.  A failure above leaves the
    # caller's frontier untouched so it can discard it and run the ordinary
    # full resolver.
    selected.update(newly_set)
    confirm_domain.update(newly_set)
    frontier["boundary_edges"].clear()
    frontier["boundary_edges"].update(working_boundary_edges)
    frontier["last_new_faces"] = tuple(int(value) for value in newly_reached)
    frontier["last_delta_result"] = result
    frontier["delta_face_materializations"] = int(
        frontier.get("delta_face_materializations", 0)
    ) + 1
    return result


def _fill_preview_terminal_result(state, radius):
    """Expand a stored normal result without rerunning shape/refinement passes."""
    import heapq
    import numpy as np

    if bool(state.get("strict_mode", False)):
        return None
    base = state.get("terminal_base")
    frontier = state.get("terminal_frontier")
    if not isinstance(base, dict) or not isinstance(frontier, dict):
        return None
    if state.get("terminal_base_signature") != state.get("signature"):
        _fill_preview_terminal_reset(state)
        return None
    try:
        base_radius = float(state.get("terminal_base_radius"))
        radius = float(radius)
        geometry = frontier["geometry"]
        count = int(geometry["count"])
        offsets = np.asarray(geometry["offsets"], dtype=np.int64).reshape(-1)
        neighbors = np.asarray(geometry["neighbors"], dtype=np.int32).reshape(-1)
        lengths = np.asarray(geometry["neighbor_lengths"], dtype=np.float64).reshape(-1)
        distances = frontier["distances"]
        heap = frontier["heap"]
        selected = frontier["selected"]
        barriers = frontier.get("barriers", frozenset())
        face_ids = np.asarray(base["face_ids"], dtype=np.int32).reshape(-1)
    except (KeyError, TypeError, ValueError, AttributeError):
        _fill_preview_terminal_reset(state)
        return None
    last_radius = float(frontier.get("last_radius", base_radius))
    if (
        radius <= last_radius
        or len(face_ids) != count
        or len(distances) != count
        or not _fill_preview_terminal_coverage_certified(state, geometry, radius)
    ):
        if radius < last_radius:
            _fill_preview_terminal_reset(state)
        return None
    popped_before = int(frontier.get("popped", 0))
    selected_before = len(selected)
    newly_reached = []
    newly_reached_set = set()
    while heap:
        current, face = heapq.heappop(heap)
        face = int(face)
        if current > float(distances[face]) + 1.0e-15:
            continue
        if current > radius:
            heapq.heappush(heap, (current, face))
            break
        frontier["popped"] = int(frontier.get("popped", 0)) + 1
        if face not in selected and face not in newly_reached_set:
            newly_reached.append(face)
            newly_reached_set.add(face)
        for edge in range(int(offsets[face]), int(offsets[face + 1])):
            other = int(neighbors[edge])
            if (min(face, other), max(face, other)) in barriers:
                continue
            candidate = current + float(lengths[edge])
            # Retain the first frontier beyond this radius; the next terminal
            # wheel stage can consume it without restarting the walk.
            if candidate < float(distances[other]):
                distances[other] = candidate
                heapq.heappush(heap, (candidate, other))
    # The old terminal path rescanned the entire distance array here.  The
    # heap is ordered, so every face newly reached for this radius is already
    # observed by the bounded walk above; retaining those ids makes growth
    # proportional to the frontier rather than the selected population.
    delta_attempted = selected_before >= _FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD
    # If this tick crosses the count threshold, keep the full terminal
    # boundary path for this tick.  It refreshes the cache from the complete
    # selected result below; the next tick is the first safe delta update.
    # Applying newly_reached faces to a base-era cache would leave an
    # internalized edge (for example the previous 1-2 frontier) behind.
    if delta_attempted and frontier.get("boundary_cache_valid"):
        delta_result = _fill_preview_terminal_delta_result(
            state,
            base,
            frontier,
            radius,
            newly_reached,
            popped_before,
            base_radius,
        )
        if delta_result is not None:
            frontier["last_radius"] = radius
            state["terminal_expansion_count"] = int(
                state.get("terminal_expansion_count", 0)
            ) + 1
            return delta_result
        # A delta attempt that fails provenance validation must not continue
        # with its partially walked heap/cache.  Discard the terminal state
        # so the caller takes the ordinary exact full-compute path.
        _fill_preview_terminal_reset(state)
        return None
    if delta_attempted:
        _fill_preview_terminal_reset(state)
        return None
    selected.update(int(value) for value in newly_reached)
    local_selected = np.asarray(sorted(selected), dtype=np.int32)
    if len(local_selected) == 0:
        return None
    # Keep the base candidate in the exact confirm domain even where its
    # shape resolver selected a face just beyond the raw Dijkstra radius.
    confirm_distances = np.asarray(distances, dtype=np.float64).copy()
    confirm_geometry, confirm_domain_ids = _fill_preview_confirm_graph_snapshot(
        geometry, confirm_distances, radius
    )
    if confirm_geometry is None:
        return None
    confirm_domain_ids = np.unique(
        np.r_[confirm_domain_ids, local_selected]
    ).astype(np.int32, copy=False)

    draw_first = np.asarray(geometry.get("first", ()), dtype=np.int32).reshape(-1)
    draw_second = np.asarray(geometry.get("second", ()), dtype=np.int32).reshape(-1)
    pair_v0 = np.asarray(geometry.get("pair_v0", ()), dtype=np.int32).reshape(-1)
    pair_v1 = np.asarray(geometry.get("pair_v1", ()), dtype=np.int32).reshape(-1)
    world_vertices = np.asarray(geometry.get("world_vertices", ()), dtype=np.float64).reshape((-1, 3))
    if len(draw_first) != len(draw_second) or len(pair_v0) != len(draw_first) or len(pair_v1) != len(draw_first):
        return None
    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[local_selected] = True
    boundary = selected_mask[draw_first] != selected_mask[draw_second]
    shape_segments, distance_segments, boundary_records = [], [], []
    shape_boundary_mask = np.zeros(len(boundary), dtype=bool)
    geometry_face_ids = np.asarray(
        geometry.get("face_ids", np.arange(count, dtype=np.int32)), dtype=np.int32
    ).reshape(-1)
    for edge in np.flatnonzero(boundary):
        edge = int(edge)
        first, second = int(draw_first[edge]), int(draw_second[edge])
        v0, v1 = int(pair_v0[edge]), int(pair_v1[edge])
        if min(v0, v1) < 0 or max(v0, v1) >= len(world_vertices):
            return None
        segment = (
            tuple(float(value) for value in world_vertices[v0]),
            tuple(float(value) for value in world_vertices[v1]),
        )
        shape = (min(first, second), max(first, second)) in barriers
        shape_boundary_mask[edge] = bool(shape)
        boundary_records.append({
            "geometry_face_a": first,
            "geometry_face_b": second,
            "mesh_face_a": int(geometry_face_ids[first]),
            "mesh_face_b": int(geometry_face_ids[second]),
            "mesh_edge": -1,
            "shape": bool(shape),
        })
        (shape_segments if shape else distance_segments).append(segment)
    result = dict(base["result"])
    result.update({
        "radius": radius,
        "faces": np.asarray(face_ids[local_selected], dtype=np.int32).copy(),
        "candidate_count": int(len(local_selected)),
        "boundary_edge_count": int(len(boundary_records)),
        "boundary_all_orange": bool(
            boundary_records and len(boundary_records) == int(np.count_nonzero(boundary))
            and _fill_preview_should_use_green_boundary(
                shape_boundary_mask,
                boundary,
            )
        ),
        "shape_segments": shape_segments,
        "distance_segments": distance_segments,
        "boundary_records": tuple(boundary_records),
        "confirm_geometry": confirm_geometry,
        "confirm_seed_local": int(np.flatnonzero(face_ids == int(state["seed_face"]))[0])
        if np.any(face_ids == int(state["seed_face"])) else -1,
        "confirm_domain_ids": confirm_domain_ids,
        "confirm_signature": state["signature"],
        "patch_edge_reached": bool(len(heap) > 0),
        "compute_seconds": 0.0,
        "geometry": geometry,
        "progressive_range_cache_hit": False,
        "progressive_range_terminal_expansion": True,
        "progressive_range_terminal_mode": "frontier",
        "progressive_range_terminal_base_radius": base_radius,
        "progressive_range_newly_processed_faces": int(frontier.get("popped", 0)) - popped_before,
        "progressive_range_reused_faces": int(len(local_selected)),
        "progressive_range_full_boundary_scan": True,
        "progressive_range_confirm_geometry_reused": False,
        "progressive_range_selected_sort_count": 1,
        "created_generation": int(state.get("generation", 0)),
    })
    if len(local_selected) >= _FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD:
        refreshed_cache = _fill_preview_terminal_boundary_cache(
            geometry, result, local_selected, barriers
        )
        if refreshed_cache is None:
            frontier["boundary_cache_valid"] = False
        else:
            frontier.update(refreshed_cache)
            frontier["boundary_cache_valid"] = True
            frontier["last_delta_result"] = None
    frontier["last_radius"] = radius
    state["terminal_expansion_count"] = int(state.get("terminal_expansion_count", 0)) + 1
    return result


def _fill_preview_expand_only_selection(distances, hidden, radius):
    """Return the visible topology-distance stage for Expand Only."""
    import numpy as np

    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    hidden = np.asarray(hidden, dtype=bool).reshape(-1)
    if len(distances) != len(hidden):
        raise ValueError("expand-only distance/visibility schema is invalid")
    tolerance = max(float(radius) * 1.0e-8, 1.0e-9)
    return (
        np.isfinite(distances)
        & (distances <= float(radius) + tolerance)
        & ~hidden
    )


def _fill_preview_make_expand_only_result(state, radius):
    """Build a lightweight feature-cost candidate for Shift+E Smart Fill.

    This path uses only topology distance plus a binary adjacent-normal cost:
    clear ridges and valleys cost twice as much to cross as flat adjacency.
    It does not call partition, contour, shadow, or boundary-refinement
    analysis.
    """
    import numpy as np

    started = time.perf_counter()
    progressive = _fill_preview_progressive_range_step(state, radius)
    geometry = progressive["geometry"]
    distances = np.asarray(progressive["distances"], dtype=np.float64).reshape(-1)
    count = int(geometry.get("count", 0))
    if count <= 0 or len(distances) < count:
        raise RuntimeError("expand-only adjacency graph is unavailable")
    hidden = np.asarray(
        geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool
    ).reshape(-1)
    if len(hidden) != count:
        raise RuntimeError("expand-only visibility schema is invalid")
    seed_local = int(progressive["seed_face"])
    if seed_local < 0 or seed_local >= count or bool(hidden[seed_local]):
        raise RuntimeError("expand-only seed is hidden or outside the graph")
    selected = _fill_preview_expand_only_selection(
        distances[:count], hidden, radius
    )
    selected_local = np.flatnonzero(selected).astype(np.int32, copy=False)
    if len(selected_local) == 0 or not bool(selected[seed_local]):
        raise RuntimeError("expand-only produced no visible candidate")
    face_ids = np.asarray(
        geometry.get("face_ids", np.arange(count, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    if len(face_ids) != count:
        raise RuntimeError("expand-only face-id schema is invalid")
    preview_faces = face_ids[selected_local].astype(np.int32, copy=True)
    first = np.asarray(geometry.get("first", ()), dtype=np.int32).reshape(-1)
    second = np.asarray(geometry.get("second", ()), dtype=np.int32).reshape(-1)
    pair_v0 = np.asarray(geometry.get("pair_v0", ()), dtype=np.int32).reshape(-1)
    pair_v1 = np.asarray(geometry.get("pair_v1", ()), dtype=np.int32).reshape(-1)
    if not (len(first) == len(second) == len(pair_v0) == len(pair_v1)):
        raise RuntimeError("expand-only boundary schema is invalid")
    selected_mask = selected
    crossing = selected_mask[first] != selected_mask[second]
    world_vertices = np.asarray(
        geometry.get("world_vertices", ()), dtype=np.float64
    ).reshape((-1, 3))
    if len(pair_v0) and (
        np.any(pair_v0 < 0)
        or np.any(pair_v1 < 0)
        or np.any(pair_v0 >= len(world_vertices))
        or np.any(pair_v1 >= len(world_vertices))
    ):
        raise RuntimeError("expand-only edge vertex schema is invalid")
    distance_segments = [
        (
            tuple(float(value) for value in world_vertices[int(pair_v0[index])]),
            tuple(float(value) for value in world_vertices[int(pair_v1[index])]),
        )
        for index in np.flatnonzero(crossing)
    ]
    boundary_records = tuple(
        {
            "geometry_face_a": int(first[index]),
            "geometry_face_b": int(second[index]),
            "mesh_face_a": int(face_ids[int(first[index])]),
            "mesh_face_b": int(face_ids[int(second[index])]),
            "mesh_edge": -1,
            "shape": False,
        }
        for index in np.flatnonzero(crossing)
    )
    confirm_snapshot, confirm_domain_ids = _fill_preview_confirm_graph_snapshot(
        geometry, distances, radius
    )
    if confirm_snapshot is None or len(confirm_domain_ids) == 0:
        raise RuntimeError("expand-only confirmation graph is unavailable")
    seed_matches = np.flatnonzero(face_ids == int(state.get("seed_face", -1)))
    if len(seed_matches) != 1:
        raise RuntimeError("expand-only seed mapping is invalid")
    elapsed = time.perf_counter() - started
    return {
        "radius": float(radius),
        "faces": preview_faces,
        "candidate_count": int(len(preview_faces)),
        "boundary_edge_count": int(len(boundary_records)),
        "boundary_all_orange": False,
        "analysis_faces": 0,
        "popped_faces": int(progressive.get("popped", 0)),
        "shape_segments": [],
        "distance_segments": distance_segments,
        "boundary_records": boundary_records,
        "confirm_geometry": confirm_snapshot,
        "confirm_seed_local": int(seed_matches[0]),
        "confirm_domain_ids": confirm_domain_ids,
        "confirm_signature": state["signature"],
        "patch_edge_reached": bool(progressive.get("patch_ids") is not None),
        "compute_seconds": float(elapsed),
        "geometry": geometry,
        "partition": None,
        "created_generation": int(state["generation"]),
        "progressive_range_mode": True,
        "progressive_range_newly_processed_faces": int(
            progressive.get("newly_processed_faces", 0)
        ),
        "progressive_range_reused_faces": int(progressive.get("reused_faces", 0)),
        "progressive_range_wheel_compute_seconds": float(elapsed),
        "expand_only_mode": True,
        "surface_evaluation_bypassed": False,
        "full_surface_analysis_bypassed": True,
        "surface_analysis": "simple-dihedral-cost",
    }


def _fill_preview_make_result(state, radius):
    """Compute one immutable candidate result for the current wheel distance."""
    import numpy as np

    # The active normal-E implementation is deliberately the fast,
    # incremental geometry-range resolver.  The former screen-space
    # morphology implementation remains in this module as dormant diagnostic
    # code, but neither normal E nor Ctrl+E enters it.
    shadow_mode = False
    shadow_capture = None
    screen_mode = False
    geometry = state["adjacency"]
    seed_value = state.get("seed_local")
    if seed_value is None:
        seed_value = state["seed_face"]
    seed_face = int(state["seed_face"]) if screen_mode else int(seed_value)
    if seed_face < 0 or seed_face >= len(geometry["hidden"]):
        raise RuntimeError("preview seed is outside the prepared mesh")
    if bool(geometry["hidden"][seed_face]):
        raise RuntimeError("preview seed is hidden")
    started = time.perf_counter()
    progressive_step = None
    proxy_metrics = {}
    if screen_mode:
        # The image classifier owns ordinary E end-to-end.  The full graph is
        # retained only for hidden/non-manifold safety and exact interfaces;
        # there is no Dijkstra/radius/proxy gate in this branch.
        count = int(geometry.get("count", 0))
        analysis_ids = np.arange(count, dtype=np.int32)
        patch_ids = analysis_ids
        distances = np.zeros(count, dtype=np.float64)
        popped = 0
        state["distance_state"] = None
        local = geometry
        seed_local = seed_face
    else:
        progressive_step = _fill_preview_progressive_range_step(state, radius)
        geometry = progressive_step["geometry"]
        seed_face = int(progressive_step["seed_face"])
        distances = progressive_step["distances"]
        patch_ids = progressive_step["patch_ids"]
        # The prepared Dijkstra frontier is the canonical topological
        # reachability source for the session component. It is already paid
        # for by progressive Smart Fill; do not run a second full-mesh BFS.
        # patch_ids is the Dijkstra walk's already-materialized reached set.
        # Reuse it instead of scanning the full distance array again.
        state["visible_reached_ids"] = np.asarray(
            patch_ids, dtype=np.int32
        ).reshape(-1)
        popped = int(progressive_step["popped"])
        patch_radius = float(progressive_step["patch_radius"])
        if len(patch_ids) == 0 or not np.isfinite(distances[seed_face]):
            raise RuntimeError("preview seed is outside the prepared patch")
        analysis_ids = patch_ids
        if geometry.get("shading_approx") and bool(state.get("strict_mode", False)):
            analysis_ids, proxy_metrics = _fill_preview_shading_proxy(
                geometry,
                patch_ids,
                distances,
                radius,
                int(np.flatnonzero(patch_ids == seed_face)[0]),
            )
        local = _fill_preview_local_geometry(
            geometry, analysis_ids, bool(state.get("strict_mode", False))
        )
        seed_local = int(np.flatnonzero(analysis_ids == seed_face)[0])
    visible_profile = state.get("visible_analysis_profile") if not bool(
        state.get("strict_mode", False)
    ) else None
    shadow_metrics = {
        "shadow_capture_ok": bool(
            isinstance(shadow_capture, dict) and shadow_capture.get("ok")
        ),
        "shadow_capture_reason": (
            shadow_capture.get("reason", "not-captured")
            if isinstance(shadow_capture, dict)
            else (
                "progressive-range-no-capture"
                if not bool(state.get("strict_mode", False))
                else "strict-mode"
            )
        ),
        "shadow_capture_generation": int(state.get("shadow_capture_generation", 0)),
        "shadow_capture_fresh": bool(shadow_mode and shadow_capture is not None),
        "shadow_capture_size": tuple(
            shadow_capture.get("capture_size", (0, 0))
            if isinstance(shadow_capture, dict) else (0, 0)
        ),
        "shadow_luminance_min": float(
            shadow_capture.get("luminance_min", 0.0)
            if isinstance(shadow_capture, dict) else 0.0
        ),
        "shadow_luminance_max": float(
            shadow_capture.get("luminance_max", 0.0)
            if isinstance(shadow_capture, dict) else 0.0
        ),
        "shadow_seed_luminance": float(
            shadow_capture.get("luminance_seed", 0.0)
            if isinstance(shadow_capture, dict) else 0.0
        ),
        "shadow_noise_scale": float(
            shadow_capture.get("noise_scale", 0.0)
            if isinstance(shadow_capture, dict) else 0.0
        ),
        "shadow_line_pixel_count": int(
            shadow_capture.get("line_pixel_count", 0)
            if isinstance(shadow_capture, dict) else 0
        ),
        "shadow_face_set_neutral": True,
        "shadow_analysis_shader_name": str(
            shadow_capture.get("analysis_shader_name", "")
            if isinstance(shadow_capture, dict) else ""
        ),
        "shadow_analysis_shader_profile_registered": bool(
            shadow_capture.get("analysis_shader_profile_registered", False)
            if isinstance(shadow_capture, dict) else False
        ),
        "shadow_analysis_shader_matcap": str(
            shadow_capture.get("analysis_shader_matcap", "")
            if isinstance(shadow_capture, dict) else ""
        ),
        "shadow_faceset_suppressed": bool(
            shadow_capture.get("faceset_suppressed", False)
            if isinstance(shadow_capture, dict) else False
        ),
        "shadow_mask_suppressed": bool(
            shadow_capture.get("mask_suppressed", False)
            if isinstance(shadow_capture, dict) else False
        ),
        "shadow_shading_restore_verified": bool(
            shadow_capture.get("shading_restore_verified", False)
            if isinstance(shadow_capture, dict) else False
        ),
        "shadow_overlay_restore_verified": bool(
            shadow_capture.get("overlay_restore_verified", False)
            if isinstance(shadow_capture, dict) else False
        ),
        "visible_analysis_profile_active": bool(
            isinstance(visible_profile, dict) and visible_profile.get("active", False)
        ),
        "visible_analysis_profile_applied": bool(
            isinstance(visible_profile, dict) and visible_profile.get("applied", False)
        ),
        "visible_analysis_profile_restore_verified": bool(
            isinstance(visible_profile, dict) and visible_profile.get("restore_verified", False)
        ),
        "visible_analysis_profile_reason": str(
            visible_profile.get("reason", "not-attempted")
            if isinstance(visible_profile, dict) else "strict-mode"
        ),
        "visible_analysis_profile_manual_owner": bool(
            (not bool(state.get("strict_mode", False)))
            and state.get("visible_analysis_profile_manual", False)
        ),
        "analysis_profile_restored_before_ready": bool(
            shadow_mode and state.get("analysis_profile_restored_before_ready", False)
        ),
        "analysis_profile_capture_applied": bool(
            shadow_mode and isinstance(visible_profile, dict)
            and visible_profile.get("applied", False)
        ),
        "shadow_screen_sharpened_overlay_active": False,
        "shadow_screen_sharpened_overlay_mode": "hidden-after-capture",
        "shadow_screen_sharpened_overlay_restored": bool(shadow_mode),
        "progressive_range_mode": not bool(state.get("strict_mode", False)),
        "progressive_range_active_path": (
            "progressive-range"
            if not bool(state.get("strict_mode", False))
            else "geometry-strict"
        ),
        "progressive_range_initial_radius": float(
            state.get("initial_radius") or 0.0
        ),
        "progressive_range_current_radius": float(radius),
        "progressive_range_newly_processed_faces": 0,
        "progressive_range_reused_faces": 0,
        "progressive_range_cache_hit": bool(
            state.get("active_result_cache_hit", False)
        ),
        "progressive_range_wheel_compute_seconds": 0.0,
        "progressive_range_full_capture_called": False,
        "progressive_range_screen_morph_called": False,
    }
    if shadow_mode:
        # Keep the existing graph/partition machinery, but remove every
        # geometry signal from ordinary E.  The grayscale boundary pass below
        # receives only the fresh screen-space shadow signal; a failed capture
        # therefore remains distance-only rather than silently reverting to
        # normal/crease/valley barriers.
        local = dict(local)
        hard_faces = np.asarray(
            local.get("source_hard", np.zeros(int(local["count"]), dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if len(hard_faces) != int(local["count"]):
            hard_faces = np.zeros(int(local["count"]), dtype=bool)
        hidden_faces = np.asarray(
            local.get("hidden", np.zeros(int(local["count"]), dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if len(hidden_faces) != int(local["count"]):
            hidden_faces = np.zeros(int(local["count"]), dtype=bool)
        if len(hard_faces) == int(local["count"]):
            hard_faces |= hidden_faces
        else:
            hard_faces = np.zeros(int(local["count"]), dtype=bool)
        for key in ("contrast", "concavity", "directional_valley", "crease"):
            local[key] = np.zeros(int(local["count"]), dtype=np.float64)
        # Physical hard/open/non-manifold faces remain safety barriers even
        # though ordinary E does not use geometry as a shape signal.
        local["contrast"][hard_faces] = 1.0
        local["concavity"][hard_faces] = 1.0
        local["directional_valley"][hard_faces] = 1.0
        local["crease"][hard_faces] = 2.0
        local["contour_cost"] = np.asarray(
            local.get("neighbor_lengths", np.ones(len(local.get("neighbors", ())))),
            dtype=np.float64,
        ).copy()
        # The normal path classifies all source physical edges in the
        # seed-relative luminance cache inside ``_fill_preview_shadow_region``.
        # No screen-overlap edge map is built here; that older map could lose
        # edges outside the current radius and let a later wheel stage differ.
        local["partitions"] = {}
    # Face Set assistance belongs to the geometry strict initial phase only;
    # ordinary E is entirely shadow/graph based and never reads the ID layer.
    # Vertex Paint uses only geometry/shading adjacency plus the sampled
    # active color; Sculpt Face Set IDs are never a region prior or seed
    # constraint for that backend.
    face_set_initial_phase = (
        state.get("backend", "SCULPT") == "SCULPT"
        and state.get("processed_radius") is None
        and not shadow_mode
    )
    face_set_metrics = {
        "face_set_prior_phase": "initial" if face_set_initial_phase else "disabled-after-initial",
        "face_set_enabled": False,
        "face_set_seed_face_set_id": -1,
        "face_set_trusted_enabled": False,
        "face_set_seed_id": -1,
        "face_set_relaxed_pair_count": 0,
        "face_set_reached_same_set_faces": 0,
        "face_set_full_component_faces": 0,
        "face_set_expanded_analysis_faces": 0,
        "face_set_different_id_neutral_crossing_count": 0,
        "face_set_compact_cut_relaxed_faces": 0,
        "face_set_source_hard_faces": 0,
        "face_set_prior_safety_reason": "disabled-after-initial" if not face_set_initial_phase else "attribute-missing",
        "face_set_prior_applied_reason": "disabled-after-initial" if not face_set_initial_phase else "attribute-missing",
        "face_set_reason": "disabled-after-initial" if not face_set_initial_phase else "attribute-missing",
        "face_set_geometry_baseline_count": 0,
        "face_set_same_id_bonus_count": 0,
        "face_set_different_id_baseline_preserved_count": 0,
        "geometry_baseline_count": 0,
        "same_id_bonus_count": 0,
        "different_id_baseline_preserved_count": 0,
    }
    face_set_expanded_analysis_ids = np.empty(0, dtype=np.int32)
    # Face Set values are a read-only prior for preview traversal.  The
    # compact cursor graph indexes geometry faces, while ``face_ids`` maps
    # those entries back to mesh polygon ids.
    try:
        if not face_set_initial_phase:
            raise RuntimeError("face-set-prior-disabled")
        mesh = state["obj"].data
        face_set_attribute = mesh.attributes.get(".sculpt_face_set")
        mesh_face_ids = np.asarray(
            geometry.get("face_ids", np.arange(int(geometry["count"]), dtype=np.int32)),
            dtype=np.int32,
        ).reshape(-1)
        if (
            face_set_attribute is None
            or getattr(face_set_attribute, "domain", None) != "FACE"
            or len(face_set_attribute.data) != len(mesh.polygons)
            or len(mesh_face_ids) != int(geometry["count"])
            or np.any(mesh_face_ids < 0)
            or np.any(mesh_face_ids >= len(mesh.polygons))
        ):
            face_set_metrics["face_set_reason"] = "attribute-invalid"
        else:
            all_face_sets = np.empty(len(mesh.polygons), dtype=np.int32)
            face_set_attribute.data.foreach_get("value", all_face_sets)
            # This is the live Face Set snapshot already required for the
            # initial prior.  Reuse it as the cache's first change baseline so
            # cold preparation does not perform a second full attribute read.
            try:
                adjacency_cache = _runtime.fill_preview_adjacency_cache.get(
                    state.get("signature")
                )
                if isinstance(adjacency_cache, dict):
                    adjacency_cache["face_set_fingerprint"] = (
                        _fill_preview_array_fingerprint(all_face_sets)
                    )
            except (AttributeError, TypeError, ValueError):
                pass
            seed_mesh_face = int(state.get("seed_face", -1))
            if seed_mesh_face < 0 or seed_mesh_face >= len(all_face_sets):
                raise ValueError("seed-face-schema")
            seed_face_set_id = int(all_face_sets[seed_mesh_face])
            face_set_metrics["face_set_seed_face_set_id"] = seed_face_set_id
            face_set_metrics["face_set_seed_id"] = seed_face_set_id
            if seed_face_set_id < 0:
                face_set_metrics["face_set_reason"] = "seed-id-invalid"
            else:
                trusted_geometry_ids, source_hard, prior_safety_reason = (
                    _fill_preview_face_set_initial_component(
                        geometry,
                        distances,
                        radius,
                        int(state.get("seed_face", seed_face)),
                        all_face_sets,
                    )
                )
                if len(trusted_geometry_ids):
                    face_set_expanded_analysis_ids = trusted_geometry_ids[
                        ~np.isin(trusted_geometry_ids, analysis_ids)
                    ]
                    face_set_metrics["face_set_full_component_faces"] = int(
                        len(trusted_geometry_ids)
                    )
                    face_set_metrics["face_set_expanded_analysis_faces"] = int(
                        len(face_set_expanded_analysis_ids)
                    )
                    if len(face_set_expanded_analysis_ids):
                        analysis_ids = np.unique(
                            np.r_[analysis_ids, face_set_expanded_analysis_ids]
                        ).astype(np.int32, copy=False)
                        local = _fill_preview_local_geometry(
                            geometry,
                            analysis_ids,
                            bool(state.get("strict_mode", False)),
                        )
                        seed_local = int(
                            np.flatnonzero(analysis_ids == seed_face)[0]
                        )
                # ``local`` carries the full-graph source safety projection;
                # compact degree is never used as a physical barrier.
                local_face_sets = all_face_sets[mesh_face_ids[analysis_ids]]
                local["face_set_values"] = local_face_sets.astype(np.int32, copy=False)
                local["face_set_seed_id"] = seed_face_set_id
                local["face_set_prior_enabled"] = True
                face_set_metrics["face_set_enabled"] = True
                face_set_metrics["face_set_trusted_enabled"] = True
                face_set_metrics["face_set_reason"] = "same-id-geodesic-bonus"
                face_set_metrics["face_set_prior_applied_reason"] = "same-id-geodesic-bonus"
                face_set_metrics["face_set_prior_safety_reason"] = prior_safety_reason
    except (AttributeError, IndexError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
        if face_set_initial_phase:
            face_set_metrics["face_set_reason"] = "attribute-invalid"
            face_set_metrics["face_set_prior_applied_reason"] = "attribute-invalid"
    # Compute the geometry-only candidate first.  Face Set prior is an
    # additive initial hint, never a reason to remove a face or bend the
    # baseline boundary toward a different-ID outline.  Keep the assisted
    # partition separate so all later scoring/refinement sees geometry costs.
    geometry_local = dict(local)
    for _key in (
        "face_set_values",
        "face_set_seed_id",
        "face_set_prior_enabled",
    ):
        geometry_local.pop(_key, None)
    geometry_local["partitions"] = {}
    if shadow_mode:
        # Ordinary E bypasses the legacy geometry/Face Set partition entirely.
        # With a valid fresh analysis capture, face luminance is the primary
        # classifier: connected bright/dark membership is decided before any
        # secondary edge Closing.  If capture is unavailable, retain the
        # explicit distance-only provisional safety path.
        if screen_mode:
            local_region, shadow_region_metrics = _fill_preview_shadow_screen_region(
                state, geometry, seed_local, int(state.get("luminance_step", 0))
            )
        elif shadow_metrics.get("shadow_capture_ok"):
            local_region, shadow_region_metrics = _fill_preview_shadow_luminance_region(
                state, local, distances[analysis_ids], radius, seed_local,
                geometry, distances,
            )
        else:
            local_region, shadow_region_metrics = _fill_preview_shadow_region(
                state,
                local,
                distances[analysis_ids],
                radius,
                seed_local,
                geometry,
                distances,
            )
        local_region = np.unique(
            np.asarray(local_region, dtype=np.int32).reshape(-1)
        )
        geometry_region = local_region.copy()
        assisted_region = np.empty(0, dtype=np.int32)
        geometry_partition = {
            "protected": np.zeros(int(local["count"]), dtype=bool),
        }
        assisted_partition = {}
    else:
        geometry_region, geometry_partition = _fill_preview_region(
            geometry_local, seed_local, bool(state["strict_mode"])
        )
        assisted_region, assisted_partition = _fill_preview_region(
            local, seed_local, bool(state["strict_mode"])
        )
        local_region = np.unique(
            np.r_[geometry_region, assisted_region]
        ).astype(np.int32, copy=False)
        shadow_region_metrics = {}
    # Do not let the Face Set-adjusted graph leak into the boundary resolver.
    local = geometry_local
    partition = geometry_partition
    if isinstance(assisted_partition, dict):
        face_set_metrics["face_set_relaxed_pair_count"] = int(
            assisted_partition.get("face_set_relaxed_pair_count", 0)
        )
        face_set_metrics["face_set_reached_same_set_faces"] = int(
            assisted_partition.get("face_set_reached_same_set_faces", 0)
        )
        face_set_metrics["face_set_different_id_neutral_crossing_count"] = int(
            assisted_partition.get("face_set_different_id_neutral_crossing_count", 0)
        )
        face_set_metrics["face_set_compact_cut_relaxed_faces"] = int(
            assisted_partition.get("face_set_compact_cut_relaxed_faces", 0)
        )
        face_set_metrics["face_set_source_hard_faces"] = int(
            assisted_partition.get("face_set_source_hard_faces", 0)
        )
        face_set_metrics["face_set_prior_safety_reason"] = str(
            assisted_partition.get("face_set_prior_safety_reason", "unknown")
        )
        if (
            face_set_initial_phase
            and face_set_metrics["face_set_enabled"]
            and not face_set_metrics["face_set_relaxed_pair_count"]
        ):
            face_set_metrics["face_set_prior_applied_reason"] = "same-id-component-no-shared-pair"
        face_set_metrics["face_set_geometry_baseline_count"] = int(
            len(geometry_region)
        )
        face_set_metrics["face_set_same_id_bonus_count"] = int(
            len(np.setdiff1d(assisted_region, geometry_region, assume_unique=False))
        )
        face_set_metrics["face_set_different_id_baseline_preserved_count"] = int(
            len(geometry_region)
        )
        face_set_metrics["geometry_baseline_count"] = int(len(geometry_region))
        face_set_metrics["same_id_bonus_count"] = int(
            len(np.setdiff1d(assisted_region, geometry_region, assume_unique=False))
        )
        face_set_metrics["different_id_baseline_preserved_count"] = int(
            len(geometry_region)
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
        if len(face_set_expanded_analysis_ids):
            allowed_local |= np.isin(analysis_ids, face_set_expanded_analysis_ids)
        local_region = local_region[allowed_local[local_region]]
        local_region = np.unique(local_region).astype(np.int32)
    valley_components = proxy_metrics.get("proxy_valley_components", ())
    if valley_components:
        # Revalidate one component record at a time.  The exact partition may
        # exclude every valley face while retaining both shores; that is not a
        # circular prerequisite for the component rescue.
        # ``local_region`` is analysis-local, while the revalidator returns
        # geometry-local ids.  Convert through an explicit geometry-local
        # union before mapping accepted ids back to analysis-local indices.
        fine_geometry_ids = analysis_ids[local_region]
        accepted_valley_geometry_ids, valley_fine_reason = _fill_preview_revalidate_valley_components(
            geometry,
            distances,
            radius,
            analysis_ids,
            fine_geometry_ids,
            valley_components,
        )
        if len(accepted_valley_geometry_ids):
            accepted_valley_geometry_ids = np.asarray(
                accepted_valley_geometry_ids, dtype=np.int32
            ).reshape(-1)
            analysis_inverse = {
                int(geometry_id): analysis_index
                for analysis_index, geometry_id in enumerate(analysis_ids)
            }
            accepted_analysis_local = [
                analysis_inverse.get(int(geometry_id), -1)
                for geometry_id in accepted_valley_geometry_ids
            ]
            if (
                any(index < 0 for index in accepted_analysis_local)
                or len(set(accepted_analysis_local)) != len(accepted_analysis_local)
            ):
                # A revalidator result outside the current analysis partition
                # is a schema failure, never an invitation to add another
                # geometry face to the candidate.
                valley_fine_reason = "analysis-id-schema"
            else:
                fine_geometry_ids = np.unique(
                    np.r_[fine_geometry_ids, accepted_valley_geometry_ids]
                ).astype(np.int32)
                local_region = np.unique(
                    np.r_[
                        local_region,
                        np.asarray(accepted_analysis_local, dtype=np.int32),
                    ]
                ).astype(np.int32)
    else:
        valley_fine_reason = "none"
    # Use the geometry-local union directly for the final distance/hidden
    # mask.  This keeps the accepted valley component in the same ID space
    # that the revalidator returned; analysis-local indices are only used for
    # partition bookkeeping above.
    local_global = (
        fine_geometry_ids
        if valley_components
        else analysis_ids[local_region]
    )
    if screen_mode:
        # Screen classification already owns extent.  The full graph is
        # deliberately not clipped by the legacy radius candidate mask.
        candidate_mask = ~geometry["hidden"][local_global]
    else:
        candidate_mask = (
            (distances[local_global] <= float(radius) + max(radius * 1.e-8, 1.e-9))
            & ~geometry["hidden"][local_global]
        )
    preview_local_ids = local_global[candidate_mask].astype(np.int32, copy=False)
    # The enclosed and sandwiched-band resolvers both inspect this same base
    # candidate.  They are intentionally independent: band evidence must not
    # turn a large open component into a compact hole, and the 100/101 rule is
    # never rerun after band faces have been added.
    base_preview_local_ids = np.unique(preview_local_ids).astype(
        np.int32, copy=False
    )
    if screen_mode:
        enclosed_component_ids = np.empty(0, dtype=np.int32)
        sandwiched_band_ids = np.empty(0, dtype=np.int32)
        enclosed_component_metrics = {"screen_classifier_skipped": True}
        sandwiched_band_metrics = {"screen_classifier_skipped": True}
    else:
        enclosed_component_ids, enclosed_component_metrics = _fill_preview_resolve_enclosed_components(
            geometry, distances, radius, base_preview_local_ids,
        )
        sandwiched_band_ids, sandwiched_band_metrics = _fill_preview_resolve_sandwiched_bands(
            geometry, distances, radius, base_preview_local_ids,
        )
    preview_parts = [base_preview_local_ids]
    if len(enclosed_component_ids):
        preview_parts.append(enclosed_component_ids)
    if len(sandwiched_band_ids):
        preview_parts.append(sandwiched_band_ids)
    preview_local_ids = np.unique(np.concatenate(preview_parts)).astype(
        np.int32, copy=False
    )
    # An edge-adjacent seed can initially occupy the whole visible component:
    # every crossing is orange, while the first cyan/distance front is still
    # one safe geodesic step away.  Bootstrap only this provisional case;
    # candidate face count is deliberately not used as a density heuristic.
    # The added faces are selected from the already prepared local patch, so
    # no hidden/non-manifold/protected face or unrelated sheet can be crossed.
    bootstrap_metrics = {
        "shape_refine_bootstrap_attempted": False,
        "shape_refine_bootstrap_applied": False,
        "shape_refine_bootstrap_reason": "not-triggered",
        "shape_refine_bootstrap_faces": 0,
        "shape_refine_bootstrap_selected_faces": int(len(preview_local_ids)),
        "shape_refine_bootstrap_orange_pairs": 0,
        "shape_refine_bootstrap_low_signal_threshold": 0.0,
        "shape_refine_bootstrap_eligible_pairs": 0,
        "shape_refine_bootstrap_run_count": 0,
        "shape_refine_bootstrap_front_faces": 0,
        "shape_refine_bootstrap_multi_front_source_count": 0,
        "shape_refine_bootstrap_shells_advanced": 0,
        "shape_refine_bootstrap_cyan_front_created": False,
        "shape_refine_bootstrap_target_line_reached": False,
        "shape_refine_bootstrap_target_line_strength": 0.0,
    }
    try:
        local_global_ids = np.asarray(
            local.get("_global_face_ids", ()), dtype=np.int32
        ).reshape(-1)
        local_first = np.asarray(local.get("first", ()), dtype=np.int32).reshape(-1)
        local_second = np.asarray(local.get("second", ()), dtype=np.int32).reshape(-1)
        local_edges = np.asarray(
            local.get("pair_edge_indices", ()), dtype=np.int32
        ).reshape(-1)
        if bool(state.get("strict_mode", False)) and (
            len(local_global_ids) == int(local.get("count", 0))
            and len(local_first) == len(local_second) == len(local_edges)
            and len(local_global_ids) > 0
        ):
            local_distances = distances[analysis_ids]
            # A later radius may shrink the geometry resolver's raw region,
            # but the accepted initial cache remains part of the provisional
            # source range for bootstrap.  The cache is stable mesh-id space;
            # project it through the current geometry face-id map here.
            geometry_face_ids = np.asarray(
                geometry.get("face_ids", np.arange(int(geometry["count"]), dtype=np.int32)),
                dtype=np.int32,
            ).reshape(-1)
            cached_mesh_ids = np.asarray(
                state.get("initial_accepted_ids", ()), dtype=np.int32
            ).reshape(-1)
            cached_geometry_ids = np.flatnonzero(
                np.isin(geometry_face_ids, np.unique(cached_mesh_ids))
            ).astype(np.int32)
            bootstrap_source_ids = np.unique(
                np.r_[preview_local_ids, cached_geometry_ids]
            ).astype(np.int32, copy=False)
            local_selected = np.isin(local_global_ids, bootstrap_source_ids)
            local_crossing = local_selected[local_first] != local_selected[local_second]
            local_outside = np.where(local_selected[local_first], local_second, local_first)
            local_shape = _fill_preview_shape_boundary_mask(
                local_crossing,
                local_distances[local_outside],
                float(radius),
                local_edges,
            )
            orange_pairs = local_crossing & (local_edges >= 0)
            bootstrap_metrics["shape_refine_bootstrap_orange_pairs"] = int(
                np.count_nonzero(orange_pairs)
            )
            orange_minimum = 4 if bool(state.get("strict_mode", False)) else 1
            all_orange = bool(
                np.count_nonzero(orange_pairs) >= orange_minimum
                and not np.any(local_crossing & ~local_shape)
            )
            if all_orange:
                bootstrap_metrics["shape_refine_bootstrap_attempted"] = True
                bootstrap_metrics["shape_refine_bootstrap_reason"] = "all-orange-edge-adjacent"
                offsets = np.asarray(local.get("offsets", ()), dtype=np.int64).reshape(-1)
                neighbors = np.asarray(local.get("neighbors", ()), dtype=np.int32).reshape(-1)
                hidden_local = np.asarray(
                    local.get("hidden", np.zeros(len(local_global_ids), dtype=bool)),
                    dtype=bool,
                ).reshape(-1)
                source_hard_local = np.asarray(
                    local.get("source_hard", np.zeros(len(local_global_ids), dtype=bool)),
                    dtype=bool,
                ).reshape(-1)
                if len(source_hard_local) != len(local_global_ids):
                    source_hard_local = np.zeros(len(local_global_ids), dtype=bool)
                protected_local = np.zeros(len(local_global_ids), dtype=bool)
                if isinstance(partition, dict):
                    raw_protected = np.asarray(partition.get("protected", ()), dtype=bool).reshape(-1)
                    if len(raw_protected) == len(protected_local):
                        protected_local |= raw_protected
                # Compact analysis degree is allowed to be deficient at a
                # crop edge.  Only full-geometry source_hard is physical
                # non-manifold/open evidence; never reintroduce the old
                # compact-degree hard stop here.
                hard_local = hidden_local | ~np.isfinite(local_distances) | source_hard_local
                normals_local = np.asarray(local.get("normals", ()), dtype=np.float64)
                contrast_local = np.asarray(
                    local.get("contrast", np.zeros(len(local_global_ids))),
                    dtype=np.float64,
                ).reshape(-1)
                concavity_local = np.asarray(
                    local.get("concavity", np.zeros(len(local_global_ids))),
                    dtype=np.float64,
                ).reshape(-1)
                directional_local = np.asarray(
                    local.get("directional_valley", np.zeros(len(local_global_ids))),
                    dtype=np.float64,
                ).reshape(-1)
                crease_local = np.asarray(
                    local.get("crease", np.zeros(len(local_global_ids))),
                    dtype=np.float64,
                ).reshape(-1)
                orange_rows = np.flatnonzero(orange_pairs).astype(np.int32)
                orange_signal = {}
                if (
                    normals_local.shape == (len(local_global_ids), 3)
                    and all(
                        len(values) == len(local_global_ids)
                        for values in (
                            contrast_local,
                            concavity_local,
                            directional_local,
                            crease_local,
                        )
                    )
                ):
                    for pair in orange_rows:
                        pair = int(pair)
                        left, right = int(local_first[pair]), int(local_second[pair])
                        dot = float(np.clip(np.dot(normals_local[left], normals_local[right]), -1.0, 1.0))
                        normal_signal = math.acos(dot) / math.pi
                        crease_signal = float(
                            np.clip(max(crease_local[left], crease_local[right]) / math.pi, 0.0, 1.0)
                        )
                        scalar_signal = float(
                            np.clip(
                                max(
                                    abs(float(contrast_local[left])),
                                    abs(float(contrast_local[right])),
                                    abs(float(concavity_local[left])),
                                    abs(float(concavity_local[right])),
                                    abs(float(directional_local[left])),
                                    abs(float(directional_local[right])),
                                ) / 0.10,
                                0.0,
                                1.0,
                            )
                        )
                        geodesic_signal = float(
                            np.clip(
                                1.0
                                - abs(float(local_distances[left]) - float(local_distances[right]))
                                / max(float(radius), 1.0e-20),
                                0.0,
                                1.0,
                            )
                        )
                        orange_signal[pair] = float(
                            np.clip(
                                0.55 * normal_signal
                                + 0.20 * crease_signal
                                + 0.20 * scalar_signal
                                + 0.05 * geodesic_signal,
                                0.0,
                                1.0,
                            )
                        )
                signal_values = np.asarray(tuple(orange_signal.values()), dtype=np.float64)
                if len(signal_values):
                    signal_median = float(np.median(signal_values))
                    signal_mad = float(1.4826 * np.median(np.abs(signal_values - signal_median)))
                    low_signal_threshold = signal_median - max(0.5 * signal_mad, 0.03)
                else:
                    low_signal_threshold = -float("inf")
                bootstrap_metrics["shape_refine_bootstrap_low_signal_threshold"] = float(
                    low_signal_threshold if np.isfinite(low_signal_threshold) else 0.0
                )
                eligible_rows = {
                    pair
                    for pair, signal in orange_signal.items()
                    if signal <= low_signal_threshold
                }
                bootstrap_metrics["shape_refine_bootstrap_eligible_pairs"] = int(len(eligible_rows))
                # Connected runs are formed from the same face incidence as
                # the boundary graph.  A distant low-signal edge cannot be
                # used as an unrelated bootstrap exit.
                face_to_orange_rows = {}
                for pair in eligible_rows:
                    for face in (int(local_first[pair]), int(local_second[pair])):
                        face_to_orange_rows.setdefault(face, []).append(pair)
                remaining_rows = set(eligible_rows)
                eligible_runs = []
                while remaining_rows:
                    start = min(remaining_rows)
                    remaining_rows.remove(start)
                    pending = [start]
                    run = []
                    while pending:
                        pair = int(pending.pop())
                        run.append(pair)
                        for face in (int(local_first[pair]), int(local_second[pair])):
                            for other in face_to_orange_rows.get(face, ()):
                                if other in remaining_rows:
                                    remaining_rows.remove(other)
                                    pending.append(other)
                    eligible_runs.append(tuple(sorted(run)))
                bootstrap_metrics["shape_refine_bootstrap_run_count"] = int(len(eligible_runs))
                # Use every safe orange boundary face as a multi-source wave,
                # rather than selecting one low-signal run.  This keeps the
                # bootstrap geometry-only and lets a 2-D front branch around
                # a ridge before reconnecting.  Face Set values are not read
                # here: only the cached initial source range above matters.
                safe_sources = set()
                if (
                    len(offsets) == len(local_global_ids) + 1
                    and int(offsets[-1]) == len(neighbors)
                ):
                    for pair in orange_rows:
                        outside = int(local_outside[int(pair)])
                        if (
                            0 <= outside < len(local_global_ids)
                            and not local_selected[outside]
                            and not hard_local[outside]
                            and not protected_local[outside]
                            and np.isfinite(local_distances[outside])
                        ):
                            safe_sources.add(outside)
                bootstrap_metrics["shape_refine_bootstrap_multi_front_source_count"] = int(
                    len(safe_sources)
                )
                front, bootstrap_seed_faces, shells_advanced, shell_reason = (
                    _fill_preview_multi_source_metric_shell(
                        safe_sources,
                        local_selected,
                        offsets,
                        neighbors,
                        local_distances,
                        radius,
                        hard_local,
                        protected_local,
                        max_shells=6,
                    )
                )
                bootstrap_metrics["shape_refine_bootstrap_shells_advanced"] = int(
                    shells_advanced
                )
                bootstrap_metrics["shape_refine_bootstrap_cyan_front_created"] = bool(
                    front
                )
                target_line_strength = float(
                    max(orange_signal.values()) if orange_signal else 0.0
                )
                bootstrap_metrics["shape_refine_bootstrap_target_line_strength"] = (
                    target_line_strength
                )
                bootstrap_metrics["shape_refine_bootstrap_target_line_reached"] = bool(
                    front and target_line_strength >= 0.08
                )
                bootstrap_metrics["shape_refine_bootstrap_front_faces"] = int(len(front))
                if front:
                    front_values = np.asarray(
                        sorted(front), dtype=np.int32
                    )
                    minimum_distance = float(np.min(local_distances[front_values]))
                    front_values = front_values[
                        local_distances[front_values]
                        <= minimum_distance + max(minimum_distance * 1.e-3, 1.e-8)
                    ]
                    if len(front_values) > 64:
                        front_values = front_values[:64]
                    if len(front_values):
                        bootstrap_radius = max(
                            float(radius),
                            float(np.max(local_distances[front_values]))
                            + max(float(radius) * 1.e-8, 1.e-9),
                        )
                        bootstrap_ids = local_global_ids[
                            np.unique(
                                np.r_[
                                    front_values,
                                    np.asarray(sorted(bootstrap_seed_faces), dtype=np.int32),
                                ]
                            )
                        ]
                        preview_local_ids = np.unique(
                            np.r_[preview_local_ids, bootstrap_ids]
                        ).astype(np.int32, copy=False)
                        radius = bootstrap_radius
                        bootstrap_metrics.update(
                            {
                                "shape_refine_bootstrap_applied": True,
                                "shape_refine_bootstrap_reason": "multi-source-metric-shell",
                                "shape_refine_bootstrap_faces": int(len(front_values)),
                            }
                        )
                    else:
                        bootstrap_metrics["shape_refine_bootstrap_reason"] = "front-distance-schema"
                elif not safe_sources:
                    bootstrap_metrics["shape_refine_bootstrap_reason"] = "no-safe-boundary-source"
                else:
                    bootstrap_metrics["shape_refine_bootstrap_reason"] = shell_reason
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        bootstrap_metrics["shape_refine_bootstrap_reason"] = "bootstrap-schema"
    # Closing is intentionally preserved as the immutable-base resolver.  A
    # separate bounded post-pass compares its add proposal with the existing
    # candidate and the complementary trim proposal, using one local shape
    # score for all three.  This is what makes the same physical boundary
    # independent of whether the cursor reached it from above or below.
    if shadow_mode:
        # Shadow-only E has already resolved the candidate through the
        # dedicated graph walk. Do not let legacy geometry refinement or its
        # bootstrap overwrite that result.
        shape_refined_ids = preview_local_ids
        shape_refine_metrics = {
            "shape_refine_attempted": False,
            "shape_refine_applied": False,
            "shape_refine_reason": "shadow-only",
            "shape_refine_candidates": 1,
            "shape_refine_proposal": "none",
            "shape_refine_added_faces": 0,
            "shape_refine_removed_faces": 0,
            "shape_refine_provisional": not bool(
                shadow_metrics.get("shadow_capture_ok")
            ),
            **shadow_region_metrics,
        }
    else:
        shape_refined_ids, shape_refine_metrics = _fill_preview_refine_shape_boundary(
            state,
            geometry,
            local,
            distances,
            analysis_ids,
            radius,
            preview_local_ids,
            partition,
            full_candidate_ids=preview_local_ids,
        )
    preview_local_ids = np.unique(shape_refined_ids).astype(
        np.int32, copy=False
    )
    # The immutable visible-component floor belongs to Normal E only.  Strict
    # Ctrl+E deliberately owns a cursor-local crop that can grow between
    # stages; applying the Normal floor there compares arrays from different
    # local graphs and falsely reports a visibility change.  Strict still uses
    # its established crop/Face Set validation below, without this global
    # monotonic merge.
    if visible_floor_enabled(state.get("strict_mode", False)):
        preview_local_ids = _fill_preview_apply_visible_selection_floor(
            state, geometry, preview_local_ids, radius
        )
    preview_local_ids, monotonic_metrics, cached_local_ids = (
        _fill_preview_preserve_initial_ids(
            state,
            geometry,
            preview_local_ids,
            face_set_initial_phase,
            bool(face_set_metrics.get("face_set_enabled", False)),
        )
    )
    preview_faces = preview_local_ids
    face_ids = geometry.get("face_ids")
    if face_ids is not None:
        preview_faces = face_ids[preview_faces].astype(np.int32, copy=False)
    if len(preview_faces) == 0:
        raise RuntimeError("preview produced no visible candidate faces")
    shape_segments = []
    distance_segments = []
    boundary_records = []
    mesh = state["obj"].data
    draw_full_geometry = bool(
        (
            not bool(state.get("strict_mode", False))
            and not screen_mode
        )
        or (
            geometry.get("shading_approx")
            and bool(state.get("strict_mode", False))
        )
    ) and bool(
        len(geometry.get("first", ())) == len(geometry.get("second", ()))
        and len(geometry.get("first", ())) == len(geometry.get("pair_edge_indices", ()))
        and len(distances) >= int(geometry.get("count", 0))
    )
    if draw_full_geometry:
        draw_first = np.asarray(geometry["first"], dtype=np.int32)
        draw_second = np.asarray(geometry["second"], dtype=np.int32)
        pair_edge_indices = np.asarray(geometry["pair_edge_indices"], dtype=np.int32)
        draw_pair_v0 = np.asarray(geometry.get("pair_v0", ()), dtype=np.int32).reshape(-1)
        draw_pair_v1 = np.asarray(geometry.get("pair_v1", ()), dtype=np.int32).reshape(-1)
        draw_world_vertices = np.asarray(
            geometry.get("world_vertices", ()), dtype=np.float64
        ).reshape((-1, 3))
        draw_ids = np.arange(int(geometry["count"]), dtype=np.int32)
        draw_selected = np.isin(draw_ids, preview_local_ids)
        draw_outside = np.where(
            draw_selected[draw_first], draw_second, draw_first
        )
        draw_outside_distances = distances[draw_outside]
    else:
        draw_first = np.asarray(local["first"], dtype=np.int32)
        draw_second = np.asarray(local["second"], dtype=np.int32)
        draw_pair_v0 = np.asarray(local.get("pair_v0", ()), dtype=np.int32).reshape(-1)
        draw_pair_v1 = np.asarray(local.get("pair_v1", ()), dtype=np.int32).reshape(-1)
        draw_world_vertices = np.asarray(
            geometry.get("world_vertices", ()), dtype=np.float64
        ).reshape((-1, 3))
        draw_selected = np.isin(analysis_ids, preview_local_ids)
        pair_edge_indices = local.get("pair_edge_indices")
        draw_outside = np.where(
            draw_selected[draw_first], draw_second, draw_first
        )
        draw_outside_distances = distances[analysis_ids[draw_outside]]
    if pair_edge_indices is None:
        pair_edge_indices = np.full(len(draw_first), -1, dtype=np.int32)
    else:
        pair_edge_indices = np.asarray(pair_edge_indices, dtype=np.int32).reshape(-1)
        if len(pair_edge_indices) != len(draw_first):
            pair_edge_indices = np.full(len(draw_first), -1, dtype=np.int32)
    boundary = draw_selected[draw_first] != draw_selected[draw_second]
    boundary_topology_metrics = _fill_preview_boundary_topology_metrics(
        mesh, pair_edge_indices, boundary, draw_pair_v0, draw_pair_v1
    )
    if shadow_mode:
        # In ordinary E, orange is exclusively a sampled shadow crossing;
        # every other radius crossing remains cyan/distance-only.
        shape_boundary_mask = np.zeros(len(boundary), dtype=bool)
        if shadow_metrics.get("shadow_capture_ok"):
            luminance_region_edges = state.get(
                "shadow_luminance_region_edges", set()
            )
            luminance_edge_cache = state.get("shadow_luminance_edge_cache")
            for edge in np.flatnonzero(boundary):
                key = _fill_preview_shadow_edge_key(local, int(edge))
                entry = (
                    luminance_edge_cache.get("edges", {}).get(key, {})
                    if isinstance(luminance_edge_cache, dict)
                    else {}
                )
                if key in luminance_region_edges or (
                    int(entry.get("tested", 0))
                    and bool(entry.get("barrier", False))
                ):
                    shape_boundary_mask[int(edge)] = True
    else:
        shape_boundary_mask = _fill_preview_shape_boundary_mask(
            boundary, draw_outside_distances, radius, pair_edge_indices
        )
    patch_edge_reached = False if screen_mode else bool(
        len(patch_ids)
        and np.any(distances[patch_ids] >= patch_radius - max(patch_radius * 0.01, 1.e-8))
    )
    for edge in np.flatnonzero(boundary):
        edge_index = -1
        # pair_v0/pair_v1 are generated together with the face pair and
        # world_vertices.  They remain valid when cursor expansion replaces
        # the compact graph, while pair_edge_indices can refer to an older
        # mesh-edge ordering.  Use the physical endpoint arrays for every
        # preview segment; never draw from an unvalidated mesh edge hint.
        if (
            edge >= len(draw_pair_v0)
            or edge >= len(draw_pair_v1)
            or int(draw_pair_v0[edge]) < 0
            or int(draw_pair_v1[edge]) < 0
            or int(draw_pair_v0[edge]) >= len(draw_world_vertices)
            or int(draw_pair_v1[edge]) >= len(draw_world_vertices)
        ):
            continue
        v0 = int(draw_pair_v0[edge])
        v1 = int(draw_pair_v1[edge])
        segment = (
            tuple(float(value) for value in draw_world_vertices[v0]),
            tuple(float(value) for value in draw_world_vertices[v1]),
        )
        shape_boundary = bool(shape_boundary_mask[edge])
        geometry_face_a = int(
            draw_first[edge]
            if draw_full_geometry
            else analysis_ids[int(draw_first[edge])]
        )
        geometry_face_b = int(
            draw_second[edge]
            if draw_full_geometry
            else analysis_ids[int(draw_second[edge])]
        )
        boundary_records.append(
            {
                "geometry_face_a": geometry_face_a,
                "geometry_face_b": geometry_face_b,
                "mesh_face_a": int(
                    geometry["face_ids"][geometry_face_a]
                ) if geometry.get("face_ids") is not None else -1,
                "mesh_face_b": int(
                    geometry["face_ids"][geometry_face_b]
                ) if geometry.get("face_ids") is not None else -1,
                "mesh_edge": int(edge_index),
                "shape": bool(shape_boundary),
            }
        )
        (shape_segments if shape_boundary else distance_segments).append(segment)
    confirm_snapshot, confirm_domain_ids = _fill_preview_confirm_graph_snapshot(
        geometry, distances, radius
    )
    if len(cached_local_ids):
        # Confirmation must retain the same stable cache floor as drawing;
        # the current radius domain alone is allowed to be smaller.
        confirm_domain_ids = np.unique(
            np.r_[confirm_domain_ids, cached_local_ids]
        ).astype(np.int32, copy=False)
    if confirm_snapshot is None or len(confirm_domain_ids) == 0:
        raise RuntimeError("preview confirmation graph is unavailable")
    elapsed = time.perf_counter() - started
    if isinstance(progressive_step, dict):
        newly_processed = int(progressive_step.get("newly_processed_faces", 0))
        reused_faces = int(progressive_step.get("reused_faces", 0))
    else:
        newly_processed = 0
        reused_faces = int(len(patch_ids))
    shadow_metrics["progressive_range_newly_processed_faces"] = int(
        newly_processed
    )
    shadow_metrics["progressive_range_reused_faces"] = int(reused_faces)
    shadow_metrics["progressive_range_wheel_compute_seconds"] = float(elapsed)
    proxy_metrics_for_result = dict(proxy_metrics)
    proxy_metrics_for_result.pop("proxy_candidate_ids", None)
    proxy_metrics_for_result.pop("proxy_valley_merge_ids", None)
    proxy_metrics_for_result.pop("proxy_valley_components", None)
    return {
        "radius": float(radius),
        "faces": preview_faces,
        "candidate_count": int(len(preview_faces)),
        "boundary_edge_count": int(len(boundary_records)),
        "boundary_all_orange": bool(
            len(boundary_records)
            and len(boundary_records) == int(np.count_nonzero(boundary))
            and _fill_preview_should_use_green_boundary(
                shape_boundary_mask, boundary
            )
        ),
        "analysis_faces": int(len(analysis_ids)),
        "popped_faces": int(popped),
        "shape_segments": shape_segments,
        "distance_segments": distance_segments,
        "boundary_records": tuple(boundary_records),
        "confirm_geometry": confirm_snapshot,
        "confirm_seed_local": int(seed_face),
        "confirm_domain_ids": confirm_domain_ids,
        "confirm_signature": state["signature"],
        "patch_edge_reached": patch_edge_reached,
        "compute_seconds": float(elapsed),
        "geometry": local,
        "partition": partition,
        "created_generation": int(state["generation"]),
        "valley_fine_reason": valley_fine_reason,
        "visibility_cache_checked": bool(
            state.get("visibility_cache_checked", False)
        ),
        "visibility_cache_invalidated": bool(
            state.get("visibility_cache_invalidated", False)
        ),
        "visibility_cache_reason": str(
            state.get("visibility_cache_reason", "not-checked")
        ),
        **enclosed_component_metrics,
        **sandwiched_band_metrics,
        **shadow_metrics,
        **shape_refine_metrics,
        **bootstrap_metrics,
        **monotonic_metrics,
        **face_set_metrics,
        **boundary_topology_metrics,
        **proxy_metrics_for_result,
    }


def _fill_preview_build_draw_batches(state, result):
    """Create copied boundary-line vertices and GPU batches once on draw.

    Preview candidates are boundary-first: no face tessellation or face GPU
    batch is created.  The draw handler may create the two line batches lazily
    after the compute timer has produced the immutable boundary snapshot.
    """
    import numpy as np

    if result.get("line_geometry_ready") and result.get("gpu_batches") is not None:
        return
    if not result.get("line_geometry_ready"):
        result["triangles"] = None
        result["shape_lines"] = [value for segment in result["shape_segments"] for value in segment]
        result["distance_lines"] = [value for segment in result["distance_segments"] for value in segment]
        result["triangles_np"] = np.empty((0, 3), dtype=np.float32)
        result["shape_lines_np"] = np.asarray(result["shape_lines"], dtype=np.float32) if result["shape_lines"] else np.empty((0, 3), dtype=np.float32)
        result["distance_lines_np"] = np.asarray(result["distance_lines"], dtype=np.float32) if result["distance_lines"] else np.empty((0, 3), dtype=np.float32)
        result["line_geometry_ready"] = True
    if result.get("gpu_batches") is None:
        shader = _fill_preview_shader_get()
        batches = {}
        if shader is not None:
            if len(result["distance_lines_np"]):
                batches["distance_lines"] = batch_for_shader(
                    shader, "LINES", {"pos": result["distance_lines_np"]}
                )
            if len(result["shape_lines_np"]):
                batches["shape_lines"] = batch_for_shader(
                    shader, "LINES", {"pos": result["shape_lines_np"]}
                )
        result["gpu_batches"] = batches


def _fill_preview_record_confirm_metrics(result, metrics):
    """Persist confirmation diagnostics after modal state is cleared."""
    payload = dict(metrics)
    _runtime.fill_preview_last_confirm_metrics = payload
    if isinstance(result, dict):
        result["confirm_metrics"] = dict(payload)
        result["confirm_reason"] = str(payload.get("reason", "unknown"))
    return payload


def _fill_preview_confirm_flood(state, result):
    """Validate and return the ready result's complete mesh-global candidate.

    ``result['faces']`` is the authority for confirmation.  It is already the
    immutable, mesh-global face set represented by the ready preview, so a
    second seed-connected flood would silently drop disconnected candidate
    islands.  The compact graph remains useful for drawing and diagnostics,
    but is deliberately not a confirmation filter.
    """
    import numpy as np

    metrics = {
        "reason": "started",
        "visible_domain_count": 0,
        "hidden_excluded_count": 0,
        "outside_domain_visibility_change_ignored": 0,
        "outside_candidate_visibility_change_ignored": 0,
        "candidate_count": 0,
        "candidate_domain_count": 0,
        "candidate_snapshot_hidden_count": 0,
        "candidate_current_hidden_count": 0,
    }

    def fail(reason, **updates):
        metrics["reason"] = str(reason)
        metrics.update(updates)
        _fill_preview_record_confirm_metrics(result, metrics)
        return None

    geometry = result.get("confirm_geometry")
    if geometry is None:
        return fail("missing-confirm-geometry")
    if result.get("confirm_signature") != state.get("signature"):
        return fail("confirm-signature-mismatch")
    try:
        count = int(geometry.get("count", 0))
    except (AttributeError, TypeError, ValueError):
        return fail("schema-count")
    if count <= 0:
        return fail("domain-empty")
    face_ids = geometry.get("face_ids")
    if face_ids is None:
        face_ids = np.arange(count, dtype=np.int32)
    else:
        try:
            raw_face_ids = np.asarray(face_ids)
        except (TypeError, ValueError):
            return fail("schema-face-ids")
        if raw_face_ids.ndim != 1 or raw_face_ids.dtype.kind not in "iu":
            return fail("schema-face-ids")
        face_ids = raw_face_ids.astype(np.int64, copy=False)
    if len(face_ids) != count:
        return fail("schema-face-ids")
    if np.any(face_ids < 0):
        return fail("schema-face-id-range")
    if len(np.unique(face_ids)) != count:
        return fail("schema-face-ids-duplicate")
    try:
        mesh = state["obj"].data
        current_hidden = np.empty(len(mesh.polygons), dtype=bool)
        mesh.polygons.foreach_get("hide", current_hidden)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return fail("visibility-read-error")
    try:
        raw_snapshot_hidden = np.asarray(
            geometry.get("hidden", np.zeros(count, dtype=bool))
        )
    except (TypeError, ValueError):
        return fail("schema-hidden")
    if raw_snapshot_hidden.ndim != 1 or raw_snapshot_hidden.dtype.kind != "b":
        return fail("schema-hidden")
    snapshot_hidden = raw_snapshot_hidden.astype(bool, copy=False)
    if len(snapshot_hidden) != count:
        return fail("schema-face-id-range")
    if np.any(face_ids >= len(current_hidden)):
        return fail("schema-face-id-range")

    candidate_raw = result.get("faces")
    if candidate_raw is None:
        return fail("missing-candidate")
    try:
        candidate_array = np.asarray(candidate_raw)
    except (TypeError, ValueError):
        return fail("schema-candidate")
    if candidate_array.ndim != 1 or candidate_array.dtype.kind not in "iu":
        return fail("schema-candidate")
    if len(candidate_array) == 0:
        return fail("candidate-empty")
    candidate = candidate_array.astype(np.int64, copy=False)
    if np.any(candidate < 0) or np.any(candidate >= len(current_hidden)):
        return fail("candidate-face-id-range")
    if len(np.unique(candidate)) != len(candidate):
        return fail("candidate-face-id-duplicate")
    candidate_local = np.flatnonzero(np.isin(face_ids, candidate)).astype(
        np.int64, copy=False
    )
    if len(candidate_local) != len(candidate):
        return fail("candidate-not-in-confirm-snapshot")
    candidate_mask = np.zeros(count, dtype=bool)
    candidate_mask[candidate_local] = True
    metrics["candidate_count"] = int(len(candidate))

    try:
        raw_domain_ids = np.asarray(result.get("confirm_domain_ids", ()))
    except (TypeError, ValueError):
        return fail("schema-confirm-domain")
    if raw_domain_ids.ndim != 1 or raw_domain_ids.dtype.kind not in "iu":
        return fail("schema-confirm-domain")
    domain_ids = raw_domain_ids.astype(np.int64, copy=False)
    if len(domain_ids) == 0:
        return fail("domain-empty")
    if np.any(domain_ids < 0) or np.any(domain_ids >= count):
        return fail("confirm-domain-range")
    if len(np.unique(domain_ids)) != len(domain_ids):
        return fail("confirm-domain-duplicate")
    domain = np.zeros(count, dtype=bool)
    domain_ids = domain_ids[(domain_ids >= 0) & (domain_ids < count)]
    domain[domain_ids] = True
    current_hidden_local = current_hidden[face_ids]
    changed_hidden = current_hidden_local != snapshot_hidden
    metrics["outside_domain_visibility_change_ignored"] = int(
        np.count_nonzero(changed_hidden & ~domain)
    )
    metrics["outside_candidate_visibility_change_ignored"] = int(
        np.count_nonzero(changed_hidden & domain & ~candidate_mask)
    )
    hidden_in_domain = domain & (snapshot_hidden | current_hidden_local)
    visible_domain = domain & ~snapshot_hidden & ~current_hidden_local
    metrics["hidden_excluded_count"] = int(np.count_nonzero(hidden_in_domain))
    metrics["visible_domain_count"] = int(np.count_nonzero(visible_domain))
    candidate_in_domain = domain[candidate_local]
    metrics["candidate_domain_count"] = int(np.count_nonzero(candidate_in_domain))
    if not np.all(candidate_in_domain):
        return fail(
            "candidate-outside-confirm-domain",
            candidate_domain_count=int(np.count_nonzero(candidate_in_domain)),
        )
    candidate_snapshot_hidden = snapshot_hidden[candidate_local]
    candidate_current_hidden = current_hidden_local[candidate_local]
    metrics["candidate_snapshot_hidden_count"] = int(
        np.count_nonzero(candidate_snapshot_hidden)
    )
    metrics["candidate_current_hidden_count"] = int(
        np.count_nonzero(candidate_current_hidden)
    )
    if np.any(candidate_snapshot_hidden):
        return fail("candidate-snapshot-hidden")
    if np.any(candidate_current_hidden):
        return fail("candidate-hidden-changed")
    seed = int(result.get("confirm_seed_local", -1))
    if seed < 0 or seed >= count:
        return fail("seed-invalid")
    if not candidate_mask[seed]:
        return fail("seed-outside-candidate")
    if snapshot_hidden[seed] or current_hidden_local[seed]:
        return fail("seed-hidden")
    if not domain[seed] or not visible_domain[seed]:
        return fail("seed-outside-visible-domain")
    if not np.all(visible_domain[candidate_local]):
        return fail("candidate-outside-visible-domain")
    if not np.any(candidate_mask):
        return fail("domain-empty")
    candidate_ids = candidate.astype(np.int32, copy=True)
    metrics["reason"] = "ok"
    _fill_preview_record_confirm_metrics(result, metrics)
    return candidate_ids


def _fill_preview_confirm_graph_snapshot(geometry, distances, radius):
    """Copy only the compact graph arrays needed by a later confirmation."""
    import numpy as np

    count = int(geometry.get("count", 0))
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    if count <= 0 or len(distances) < count:
        return None, np.empty(0, dtype=np.int32)
    hidden = np.asarray(
        geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool
    ).reshape(-1)
    if len(hidden) != count:
        return None, np.empty(0, dtype=np.int32)
    domain_ids = np.flatnonzero(
        np.isfinite(distances[:count])
        & (
            distances[:count]
            <= float(radius) + max(float(radius) * 1.0e-8, 1.0e-9)
        )
        & ~hidden
    ).astype(np.int32)
    face_ids = geometry.get("face_ids")
    if face_ids is None:
        face_ids = np.arange(count, dtype=np.int32)
    else:
        face_ids = np.asarray(face_ids, dtype=np.int32).reshape(-1)
    if len(face_ids) != count:
        return None, domain_ids
    snapshot = {
        "count": count,
        "face_ids": np.array(face_ids, dtype=np.int32, copy=True),
        "hidden": np.array(hidden, dtype=bool, copy=True),
        "offsets": np.array(geometry.get("offsets"), dtype=np.int64, copy=True),
        "neighbors": np.array(geometry.get("neighbors"), dtype=np.int32, copy=True),
    }
    return snapshot, np.array(domain_ids, dtype=np.int32, copy=True)


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
            fraction = state.get("prepare_progress_fraction")
            if state.get("prepare_progress_indeterminate") or fraction is None:
                tick = int(time.perf_counter() * 8.0) % 4
                progress_line = "Progress [" + ("." * tick) + ">" + ("." * (3 - tick)) + "] working"
                eta_line = "ETA approximate: calculating"
            else:
                fraction = min(max(float(fraction), 0.0), 1.0)
                filled = int(round(fraction * 20.0))
                progress_line = f"Progress [{'#' * filled}{'-' * (20 - filled)}] {fraction * 100.0:5.1f}%"
                eta = state.get("prepare_eta_seconds")
                eta_line = (
                    f"ETA approximate: {float(eta):.1f}s"
                    if eta is not None
                    else "ETA approximate: --"
                )
            lines = [
                "Smart Fill Preview - preparing surface data...",
                f"Stage {int(state.get('prepare_stage_index', 0)) + 1}/{int(state.get('prepare_stage_count', 1))}: {state.get('prepare_stage', 'working')}",
                progress_line,
                f"Prep {float(state.get('prepare_seconds', 0.0)):.2f}s",
                eta_line,
                "Esc cancel (accepted between slices)",
            ]
        elif (
            not bool(state.get("strict_mode", False))
            and state.get("phase") == "ready"
            and state.get("drawn_generation") != state.get("generation")
        ):
            lines = [
                "Smart Fill Preview - waiting for visible boundary...",
                f"Distance {float(result.get('radius', state.get('desired_radius') or 0.0)):.4g} m",
                "Wheel is accepted after the boundary is drawn",
                "Enter/E apply when ready; Esc cancel",
            ]
        elif state["phase"] == "compute":
            lines = [
                "Smart Fill Preview - computing local patch...",
                f"Distance {float(state['desired_radius']):.4g} m",
                "E again/Enter apply when ready; Esc cancel",
            ]
        else:
            screen_roi = bool(result.get("shadow_screen_classifier_mode"))
            edge = (
                "screen luminance boundary"
                if screen_roi
                else ("shape boundary" if result.get("shape_segments") else "distance boundary")
            )
            if result.get("patch_edge_reached"):
                edge += " / analysis limit"
            if screen_roi:
                roi_radius = int(result.get("shadow_screen_roi_radius", state.get("shadow_screen_roi_radius", 320)))
                roi_bounds = result.get("shadow_screen_roi_bounds", (0, 0, 0, 0))
                try:
                    roi_w = max(0, int(roi_bounds[2]) - int(roi_bounds[0]))
                    roi_h = max(0, int(roi_bounds[3]) - int(roi_bounds[1]))
                except (TypeError, ValueError, IndexError):
                    roi_w = int(result.get("shadow_screen_width", 0))
                    roi_h = int(result.get("shadow_screen_height", 0))
                lines = [
                    "Smart Fill Preview - screen morphology",
                    f"ROI r{roi_radius}px ({roi_w}x{roi_h})  Boundary edges {int(result.get('boundary_edge_count', 0))}",
                    f"Field {'sharpened' if result.get('shadow_screen_sharpened_field_used') else 'raw'}  Analysis restored {'yes' if result.get('analysis_profile_restored_before_ready') else 'manual/unknown'}",
                    f"Ready - {edge}  Prep {float(state.get('prepare_seconds', 0.0)):.2f}s",
                    "E again: apply   Enter: apply   Esc: cancel",
                ]
            else:
                lines = [
                    (
                    "Smart Fill Preview - Simple Ridge/Valley Cost"
                        if result.get("expand_only_mode")
                        else (
                            "Smart Fill Preview - progressive range"
                            if result.get("progressive_range_mode")
                            else "Smart Fill Preview"
                        )
                    ),
                    f"Distance {float(result['radius']):.4g} m  Boundary edges {int(result.get('boundary_edge_count', 0))}",
                    (
                        f"New {int(result.get('progressive_range_newly_processed_faces', 0))} "
                        f"/ reused {int(result.get('progressive_range_reused_faces', 0))} "
                        f"({'frontier-delta' if result.get('progressive_range_terminal_mode') == 'frontier-delta' else ('cache' if result.get('progressive_range_cache_hit') else 'frontier')})"
                        if result.get("progressive_range_mode")
                        else ""
                    ),
                    f"Ready - {edge}  Prep {float(state.get('prepare_seconds', 0.0)):.2f}s",
                    "E again: apply   Enter: apply   Esc: cancel",
                ]
        for index, line in enumerate(lines):
            blf.position(font_id, x, y - index * 18, 0)
            blf.color(font_id, 0.92, 0.96, 1.0, 1.0)
            blf.draw(font_id, line)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass


def _fill_preview_build_shadow_field_overlay(state, result):
    """Build a bounded face-tone view of the sharpened ROI field.

    A native 2D image shader would require version-specific texture upload
    state.  The experimental overlay therefore uses one center point per
    reduced ROI cell, colored from the exact sharpened luminance buffer that
    drives the classifier.  It is an explicitly center-sample view, not a
    polygon-coverage claim, and is rebuilt only when the immutable capture
    generation/buffer changes.
    """
    import numpy as np

    if bool(state.get("strict_mode", False)):
        return False
    buffers = state.get("shadow_screen_buffers")
    geometry = result.get("geometry") if isinstance(result, dict) else None
    if not isinstance(buffers, dict) or not isinstance(geometry, dict):
        return False
    cache_key = (state.get("shadow_capture_generation"), buffers.get("key"))
    if (
        state.get("shadow_screen_overlay_batches") is not None
        and state.get("shadow_screen_overlay_cache_key") == cache_key
    ):
        if isinstance(result, dict):
            result["shadow_screen_sharpened_overlay_active"] = True
            result["shadow_screen_sharpened_overlay_alpha"] = float(
                state.get("shadow_screen_overlay_alpha", 0.30)
            )
            result["shadow_screen_sharpened_overlay_mode"] = "center-sample-face-tone"
        return True
    shader = _fill_preview_shader_get()
    if shader is None:
        return False
    try:
        center_inside = np.asarray(buffers.get("center_inside"), dtype=bool).reshape(-1)
        center_x = np.asarray(buffers.get("center_x"), dtype=np.int64).reshape(-1)
        center_y = np.asarray(buffers.get("center_y"), dtype=np.int64).reshape(-1)
        field = np.asarray(buffers.get("denoised"), dtype=np.float32)
        centers = np.asarray(geometry.get("centers"), dtype=np.float64)
        if (
            field.ndim != 2
            or centers.ndim != 2
            or centers.shape[1] != 3
            or len(center_inside) != len(centers)
            or len(center_x) != len(centers)
            or len(center_y) != len(centers)
        ):
            return False
        height, width = field.shape
        valid = center_inside.copy()
        valid &= center_x >= 0
        valid &= center_x < width
        valid &= center_y >= 0
        valid &= center_y < height
        ids = np.flatnonzero(valid)
        if not len(ids):
            return False
        # One deterministic center per reduced cell keeps the overlay bounded
        # on dense meshes while the classifier still evaluates every center.
        flat = center_y[ids] * width + center_x[ids]
        _, first_index = np.unique(flat, return_index=True)
        ids = ids[np.sort(first_index)]
        values = field[center_y[ids], center_x[ids]].astype(np.float64, copy=False)
        finite = np.isfinite(values)
        ids, values = ids[finite], values[finite]
        if not len(ids):
            return False
        vmin, vmax = float(np.min(values)), float(np.max(values))
        span = max(vmax - vmin, 1.0e-6)
        bins = np.clip(((values - vmin) / span * 8.0).astype(np.int32), 0, 7)
        batches = []
        for index in range(8):
            selected = ids[bins == index]
            if not len(selected):
                continue
            batches.append(
                (
                    float(vmin + (index + 0.5) * span / 8.0),
                    batch_for_shader(shader, "POINTS", {"pos": centers[selected].astype(np.float32, copy=False)}),
                )
            )
        if not batches:
            return False
        state["shadow_screen_overlay_batches"] = tuple(batches)
        state["shadow_screen_overlay_cache_key"] = cache_key
        if isinstance(result, dict):
            result["shadow_screen_sharpened_overlay_active"] = True
            result["shadow_screen_sharpened_overlay_alpha"] = float(
                state.get("shadow_screen_overlay_alpha", 0.30)
            )
            result["shadow_screen_sharpened_overlay_mode"] = "center-sample-face-tone"
        return True
    except (AttributeError, IndexError, MemoryError, RuntimeError, TypeError, ValueError):
        state["shadow_screen_overlay_batches"] = None
        state["shadow_screen_overlay_cache_key"] = None
        return False


def _fill_preview_draw_shadow_field_overlay(state, result, shader):
    """Draw the sharpened center-sample field beneath orange/cyan lines."""
    if not isinstance(state, dict) or bool(state.get("strict_mode", False)):
        return False
    if not _fill_preview_build_shadow_field_overlay(state, result):
        return False
    batches = state.get("shadow_screen_overlay_batches") or ()
    try:
        gpu.state.point_size_set(
            max(2.0, min(9.0, 0.75 * float(
                (state.get("shadow_screen_buffers") or {}).get("downsample_factor", 2)
            )))
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    alpha = float(state.get("shadow_screen_overlay_alpha", 0.30))
    for value, batch in batches:
        shader.bind()
        shader.uniform_float("color", (float(value), float(value), float(value), alpha))
        batch.draw(shader)
    return True


def _fill_preview_draw():
    """Draw one session's copied candidate overlay and boundary lines."""
    state = _runtime.fill_preview_state
    if state is None or not state.get("active"):
        return
    try:
        context = bpy.context
        if context.area is None or int(context.area.as_pointer()) != int(state["area_key"]):
            return
        result = state.get("result")
        shader = _runtime.fill_preview_shader
        if shader is None:
            return
        if result is not None:
            _fill_preview_build_draw_batches(state, result)
            current_generation = int(state.get("generation", 0))
            first_draw_for_generation = (
                state.get("drawn_generation") != current_generation
            )
            draw_succeeded = False
            gpu.state.blend_set("ALPHA")
            depth_set = False
            depth_mask = False
            draw_succeeded = True
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
                # The processed field is capture-time diagnostic only.  Once
                # a candidate is ready, restore the user's normal viewport
                # and draw only the orange/cyan interface; keeping the tone
                # overlay here would hide the user's Face Set colors and
                # make the preview differ from the editable scene.
                if len(result["triangles_np"]):
                    shader.bind()
                    shader.uniform_float("color", (0.16, 0.72, 0.96, 0.20))
                    batch = result.get("gpu_batches", {}).get("triangles")
                    if batch is not None:
                        batch.draw(shader)
                    else:
                        draw_succeeded = False
                green_boundary = bool(result.get("boundary_all_orange", False))
                for key, color, batch_key in (
                    ("distance_lines_np", (0.20, 0.86, 1.0, 0.95), "distance_lines"),
                    ("shape_lines_np", (1.0, 0.38, 0.08, 0.95), "shape_lines"),
                ):
                    if len(result[key]):
                        if green_boundary:
                            color = (0.16, 1.0, 0.34, 0.95)
                        shader.bind()
                        shader.uniform_float("color", color)
                        gpu.state.line_width_set(2.0)
                        batch = result.get("gpu_batches", {}).get(batch_key)
                        if batch is not None:
                            batch.draw(shader)
                        else:
                            draw_succeeded = False
                # Do not open the confirm gate yet.  The state reset below is
                # part of the draw transaction: an exception from any bind,
                # uniform, batch draw, or required reset leaves the generation
                # stale and confirmation blocked until a later successful
                # viewport redraw.
            finally:
                if depth_mask:
                    try:
                        gpu.state.depth_mask_set(True)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        draw_succeeded = False
                if depth_set:
                    try:
                        gpu.state.depth_test_set("NONE")
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        draw_succeeded = False
                try:
                    gpu.state.line_width_set(1.0)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    draw_succeeded = False
                try:
                    gpu.state.point_size_set(1.0)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    draw_succeeded = False
                try:
                    gpu.state.blend_set("NONE")
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    draw_succeeded = False
            if (
                draw_succeeded
                and state.get("phase") == "ready"
                and result is state.get("result")
                and current_generation == int(state.get("generation", 0))
                and first_draw_for_generation
            ):
                # The modal gate opens only after every required operation for
                # this exact result generation has succeeded.
                state["drawn_generation"] = current_generation
                state["wheel_drain_until"] = (
                    time.perf_counter() + _FILL_PREVIEW_WHEEL_DRAIN_SECONDS
                )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        try:
            gpu.state.blend_set("NONE")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


def _fill_preview_draw_text_handler():
    """Draw the 2D status HUD in POST_PIXEL, after the 3D overlay pass."""
    state = _runtime.fill_preview_state
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


def _fill_preview_update_prepare_progress(state, token):
    """Publish honest per-stage progress and an approximate ETA."""
    if not isinstance(token, dict):
        token = _fill_preview_prepare_token(str(token), phase="prepare-finalize")
    stage = str(token.get("stage", "working"))
    previous_stage = str(state.get("prepare_stage", ""))
    if previous_stage != stage:
        state["prepare_stage_started_at"] = time.perf_counter()
    stage_order = {
        "allocate": 0,
        "vertices": 1,
        "loop_edges": 2,
        "loop_vertices": 3,
        "totals": 4,
        "hidden": 5,
        "centers": 6,
        "normals": 7,
        "world-space": 8,
        "world-space-ready": 8,
        "edge-pairs": 9,
        "edge-pairs-ready": 9,
        "seam-detection": 10,
        "seam-detection-ready": 10,
        "topology-filter": 11,
        "topology-filter-ready": 11,
        "topology-fingerprint": 12,
        "topology-fingerprint-ready": 12,
        "topology-validity": 13,
        "topology-validity-ready": 13,
        "edge-metrics": 14,
        "edge-metrics-ready": 14,
        "face-vertex-schema": 15,
        "face-vertex-schema-ready": 15,
        "csr-order": 16,
        "csr-order-ready": 16,
        "csr-ready": 17,
        "cache-fingerprints": 18,
        "cache-fingerprints-ready": 18,
        "cache-build": 19,
        "cache-ready": 19,
    }
    if token.get("stage_index") is None and stage in stage_order:
        token["stage_index"] = stage_order[stage]
    state["prepare_stage"] = stage
    state["prepare_progress_phase"] = str(token.get("phase", "prepare"))
    state["prepare_progress_indeterminate"] = bool(token.get("indeterminate", True))
    state["prepare_stage_done"] = int(token.get("done", 0))
    state["prepare_stage_total"] = int(token.get("total", 0))
    if token.get("stage_index") is not None:
        state["prepare_stage_index"] = int(token["stage_index"])
    if token.get("stage_count") is not None:
        state["prepare_stage_count"] = int(token["stage_count"])
    if state["prepare_progress_indeterminate"]:
        state["prepare_progress_fraction"] = None
        state["prepare_eta_seconds"] = None
        return
    total = max(int(state.get("prepare_stage_total", 0)), 1)
    done = min(max(int(state.get("prepare_stage_done", 0)), 0), total)
    fraction = float(done) / float(total)
    state["prepare_progress_fraction"] = fraction
    started = float(state.get("prepare_stage_started_at", time.perf_counter()))
    elapsed = max(0.0, time.perf_counter() - started)
    if fraction > 1.0e-6 and done < total:
        state["prepare_eta_seconds"] = max(0.0, elapsed * (1.0 - fraction) / fraction)
    else:
        state["prepare_eta_seconds"] = None


def _fill_preview_stop_draw():
    for handler in (_runtime.fill_preview_draw_handler, _runtime.fill_preview_text_draw_handler):
        if handler is None:
            continue
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handler, "WINDOW")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    _runtime.fill_preview_draw_handler = None
    _runtime.fill_preview_text_draw_handler = None


def _fill_preview_cancel(state=None, reason="cancel", terminal=True):
    cleanup_started = time.perf_counter()
    current = _runtime.fill_preview_state
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
    # The normal-E visible analysis profile is purely transient.  Restore it
    # before removing draw handlers on every exit path (confirm, Esc, stale,
    # replacement, load/unregister, and modal exceptions).  The helper is
    # idempotent so repeated cancellation is safe.
    visible_profile = current.get("visible_analysis_profile")
    if visible_profile is not None and not current.get(
        "visible_analysis_profile_manual", False
    ):
        try:
            _fill_preview_restore_visible_analysis_profile(visible_profile)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            visible_profile["restore_verified"] = False
        result = current.get("result")
        if isinstance(result, dict):
            result["visible_analysis_profile_active"] = bool(
                visible_profile.get("active", False)
            )
            result["visible_analysis_profile_restore_verified"] = bool(
                visible_profile.get("restore_verified", False)
            )
            result["visible_analysis_profile_reason"] = str(
                visible_profile.get("reason", "restored")
            )
    result = current.get("result")
    if isinstance(result, dict) and result.get("shadow_screen_classifier_mode"):
        result["shadow_screen_sharpened_overlay_active"] = False
        result["shadow_screen_sharpened_overlay_restored"] = True
    current["shadow_screen_overlay_batches"] = None
    current["shadow_screen_overlay_cache_key"] = None
    _fill_preview_stop_draw()
    cleanup_seconds = time.perf_counter() - cleanup_started
    current["metrics"]["cleanup_seconds"] = cleanup_seconds
    cancel_requested_at = current["metrics"].get("cancel_requested_at")
    if cancel_requested_at is not None:
        current["metrics"]["cancel_finished_at"] = time.perf_counter()
        current["metrics"]["cancel_finish_seconds"] = max(
            0.0, current["metrics"]["cancel_finished_at"] - cancel_requested_at
        )
    # Modal ownership is cleared by registration immediately before the
    # owning modal() returns its terminal status.  External callbacks use
    # terminal=False and must leave the registry/handler liveness intact.
    _runtime.fill_preview_state = None
    _fill_preview_tag_redraw(current)


def _fill_preview_request_cancel(state=None, reason="external-change"):
    """Cancel visible Smart Fill state while retaining the live modal owner.

    Blender has no public modal-handler removal API.  External callbacks may
    tear down preview resources, but the old operator must receive one more
    event and return CANCELLED before its RNA class is eligible for teardown.
    """
    current = _runtime.fill_preview_state
    if state is not None and current is not state:
        return
    if current is None:
        return
    _runtime.fill_preview_modal_cancel_requested = True
    _runtime.fill_preview_modal_cancel_reason = str(reason)
    _fill_preview_cancel(current, reason, terminal=False)
    try:
        from .. import lifecycle
        lifecycle.modal_request_cancel(
            operator=_runtime.fill_preview_modal_operator,
            reason=reason,
        )
        lifecycle.modal_schedule_terminal_event(
            operator=_runtime.fill_preview_modal_operator,
            state=current,
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _fill_preview_modal_context_matches(state, context):
    """Check the cheap modal ownership boundary before handling any event.

    This intentionally avoids the expensive mesh/signature validation used by
    timer processing.  A mode/object/window/area/region change must terminate
    synchronously before the event can reach Blender's normal handlers.
    """
    if state is None or not state.get("active"):
        return False
    # Older in-memory test/proxy sessions may predate the context-bound
    # session fields.  Real invoke() always writes region_key, so production
    # sessions never take this compatibility branch.
    if "region_key" not in state:
        return True
    try:
        area = getattr(context, "area", None)
        window = getattr(context, "window", None)
        region = getattr(context, "region", None)
        obj = getattr(context, "active_object", None)
        if area is None or region is None or obj is None:
            return False
        area_key = int(area.as_pointer())
        region_key = int(region.as_pointer())
        window_key = int(window.as_pointer()) if window is not None else 0
        obj_key = int(obj.as_pointer())
        mesh = getattr(obj, "data", None)
        mesh_key = int(mesh.as_pointer()) if mesh is not None else 0
        expected_region = int(state.get("region_key", 0) or 0)
        expected_obj = int(state.get("obj_pointer", 0) or 0)
        expected_mesh = int(state.get("mesh_pointer", 0) or 0)
        return (
            area_key == int(state.get("area_key", 0) or 0)
            and region_key == expected_region
            and (
                not int(state.get("window_key", 0) or 0)
                or window_key == int(state.get("window_key", 0) or 0)
            )
            and str(getattr(context, "mode", "")) == str(state.get("mode", ""))
            and obj_key == expected_obj
            and mesh_key == expected_mesh
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


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
            and context.mode == str(state.get("mode", "SCULPT"))
            and context.active_object is obj
            and _fill_preview_signature(obj) == state["signature"]
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _fill_preview_shader_get():
    if _runtime.fill_preview_shader is None:
        try:
            _runtime.fill_preview_shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _runtime.fill_preview_shader = None
    return _runtime.fill_preview_shader


def _on_fill_preview_depsgraph_update(_scene, depsgraph):
    """Mark volatile cache layers dirty without evicting topology indices.

    Blender reports Face Set attribute writes and sculpt geometry edits through
    the same Mesh/Object update flag.  The graph therefore survives this
    callback; the next E either takes the Face Set-only fast path or refreshes
    geometry/visibility after a topology guard.  Undo/redo/load handlers still
    clear all caches because those operations can restore an earlier datablock
    revision without a reliable per-ID update batch.
    """
    state = _runtime.fill_preview_state
    try:
        updates = tuple(depsgraph.updates)
        # Vertex Paint color edits are shading-only updates.  They do not
        # alter polygon topology or the reusable adjacency graph, so retain
        # that graph and let the confirm writer's seed/signature validation
        # decide whether a stale preview may be committed.  Geometry updates
        # remain fully invalidating and are still handled conservatively.
        if state is not None and state.get("active") and state.get("backend") == "PAINT_VERTEX":
            updates = tuple(
                update
                for update in updates
                if not (
                    getattr(update, "is_updated_shading", False)
                    and getattr(update, "is_updated_geometry", False) is False
                )
            )
        updated_pointers = _fill_preview_update_pointers(updates)
        if not updated_pointers:
            return
        affected_session = False
        if state is not None and state.get("active"):
            obj = state["obj"]
            affected_session = bool(
                int(obj.as_pointer()) in updated_pointers
                or int(obj.data.as_pointer()) in updated_pointers
            )
        _fill_preview_mark_cache_pointers_dirty(updated_pointers)
        if affected_session:
            _fill_preview_request_cancel(state, "stale")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        if state is not None and state.get("active"):
            _fill_preview_request_cancel(state, "stale")


def _fill_preview_update_pointers(updates):
    """Return original and evaluated IDs from a depsgraph update batch."""
    pointers = set()
    for update in updates:
        data = update.id
        pointers.add(int(data.as_pointer()))
        # Depsgraph updates commonly expose evaluated Object/Mesh copies;
        # cache keys are owned by the original datablocks.
        original = getattr(data, "original", None)
        if original is not None:
            pointers.add(int(original.as_pointer()))
    return pointers


def _fill_preview_mark_cache_pointers_dirty(updated_pointers):
    """Mark volatile fields dirty while retaining the expensive graph."""
    for key in list(_runtime.fill_preview_adjacency_cache):
        if key[0] in updated_pointers or key[1] in updated_pointers:
            cached = _runtime.fill_preview_adjacency_cache.get(key)
            if isinstance(cached, dict):
                cached["geometry_dirty"] = True
                cached["geometry_dirty_reason"] = "depsgraph-update"
    for key in list(_runtime.fill_preview_cursor_cache):
        if key[0] in updated_pointers or key[1] in updated_pointers:
            cached = _runtime.fill_preview_cursor_cache.get(key)
            if isinstance(cached, dict):
                cached["geometry_dirty"] = True
                cached["geometry_dirty_reason"] = "depsgraph-update"


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
    cached = _runtime.local_face_set_adjacency_cache.get(key)
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
                  physical_degree=degree.astype(np.int32, copy=False),
                  hidden=hidden, contrast=contrast, concavity=concavity,
                  crease=crease, directional_valley=directional_valley,
                  count=count, seam_count=seam_count, partitions={})
    _runtime.local_face_set_adjacency_cache.clear()
    _runtime.local_face_set_adjacency_cache[key] = cached
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



def _fill_preview_write_vertex_paint(state, result):
    """Write the ready Smart Fill candidate through the active color layer."""
    import numpy as np

    obj = state.get("obj")
    attribute, reason = _vertex_paint_active_color_attribute(obj)
    if attribute is None:
        return None, reason or "active color attribute is unavailable"
    signature = _vertex_paint_color_signature(obj, attribute)
    if signature != state.get("color_attribute_signature"):
        return None, "active color attribute changed during preview"
    seed_location = state.get("seed_local_location")
    seed_color = state.get("seed_color")
    if seed_location is None or seed_color is None:
        return None, "seed color sample is unavailable"
    current_seed = _vertex_paint_sample_color(
        obj,
        int(state.get("seed_face", -1)),
        Vector(seed_location),
        attribute,
    )
    if current_seed is None or not np.allclose(
        np.asarray(current_seed, dtype=np.float64),
        np.asarray(seed_color, dtype=np.float64),
        atol=2.0e-5,
        rtol=0.0,
    ):
        return None, "seed color changed during preview"
    faces = _fill_preview_confirm_flood(state, result)
    if faces is None or len(faces) == 0:
        return None, "confirm graph is unavailable"
    mesh = obj.data
    domain = str(attribute.domain)
    if domain == "CORNER":
        target_indices = np.asarray(
            [
                int(loop_index)
                for face_index in faces
                for loop_index in mesh.polygons[int(face_index)].loop_indices
            ],
            dtype=np.int64,
        )
    elif domain == "POINT":
        target_indices = np.asarray(
            sorted(
                {
                    int(mesh.loops[int(loop_index)].vertex_index)
                    for face_index in faces
                    for loop_index in mesh.polygons[int(face_index)].loop_indices
                }
            ),
            dtype=np.int64,
        )
    else:
        return None, f"unsupported color domain: {domain}"
    if len(target_indices) == 0:
        return None, "candidate has no writable color elements"
    colors = np.empty((len(attribute.data), 4), dtype=np.float32)
    attribute.data.foreach_get("color", colors.ravel())
    original_colors = np.array(colors, copy=True)
    target = np.asarray(seed_color, dtype=np.float32)
    changed_mask = np.any(
        np.abs(colors[target_indices].astype(np.float64) - target.astype(np.float64))
        > 2.0e-5,
        axis=1,
    )
    changed = int(np.count_nonzero(changed_mask))
    if changed:
        colors[target_indices[changed_mask]] = target
        try:
            attribute.data.foreach_set("color", colors.ravel())
            mesh.update()
        except (
            AttributeError,
            IndexError,
            MemoryError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            restore_error = None
            try:
                attribute.data.foreach_set("color", original_colors.ravel())
                mesh.update()
            except (
                AttributeError,
                IndexError,
                MemoryError,
                ReferenceError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as restore:
                restore_error = restore
            if restore_error is not None:
                return None, f"color write failed and restoration failed: {restore_error}"
            return None, f"color write failed; original colors restored: {error}"
    return (changed, int(len(faces)), domain), None


def _fill_preview_accept_normal_wheel(state, event):
    """Accept exactly one normal-mode wheel step after a visible result.

    Wheel events arriving during preparation, computation, draw-gating, or the
    short post-draw drain are deliberately dropped.  There is no debounce
    queue: one accepted event produces one radius step, so rapid input and
    slow input have identical stage semantics.
    """
    now = time.perf_counter()
    if (
        state.get("initial_radius") is None
        or state.get("phase") != "ready"
        or state.get("pending")
        or not state.get("wheel_armed")
        or state.get("wheel_gate")
        or state.get("drawn_generation") != state.get("generation")
        or now < float(state.get("wheel_drain_until", 0.0))
    ):
        state["dropped_wheel_events"] = int(
            state.get("dropped_wheel_events", 0)
        ) + 1
        return False
    factor = 1.25 if event.type == "WHEELUPMOUSE" else 1.0 / 1.25
    initial = float(state["initial_radius"])
    base = state.get("desired_radius") or initial
    desired = max(initial * 0.125, min(initial * 16.0, float(base) * factor))
    state["desired_radius"] = desired
    state["wheel_armed"] = False
    state["wheel_gate"] = False
    state["pending"] = True
    state["phase"] = "compute"
    state["result"] = None
    _fill_preview_tag_redraw(state)
    return True
