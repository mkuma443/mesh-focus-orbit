"""Smart Fill geometry component.

Loaded by the package entry point in dependency order. This module owns Smart
Fill geometry and imports cross-domain state explicitly.
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
from ..foundation import _FILL_PREVIEW_ANALYSIS_SHADER_PROFILE, _tag_redraw


def _fill_average(values, first, second, count, iterations):
    """Diffuse a geometry array over adjacency without preview-module coupling."""
    degree = np.bincount(first, minlength=count) + np.bincount(second, minlength=count)
    divisor = degree + 1
    current = values.copy()
    for _ in range(iterations):
        if current.ndim == 1:
            current = (
                current
                + np.bincount(first, weights=current[second], minlength=count)
                + np.bincount(second, weights=current[first], minlength=count)
            ) / divisor
        else:
            current = np.column_stack(
                [
                    (
                        current[:, axis]
                        + np.bincount(first, weights=current[second, axis], minlength=count)
                        + np.bincount(second, weights=current[first, axis], minlength=count)
                    )
                    / divisor
                    for axis in range(current.shape[1])
                ]
            )
    return current


def _fill_preview_cursor_expand(state, radius):
    """Extend a cursor graph only when the requested radius needs more halo."""
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
    state["expansion_count"] = int(state.get("expansion_count", 0)) + 1
    return True


def _fill_partition(geometry, strict_mode):
    """Resolve partitioning through the Smart Fill preview owner at runtime.

    The full partition policy is UI/session-facing and remains in preview.py;
    this explicit adapter avoids an undeclared exec-global dependency while
    keeping geometry importable independently.
    """
    from . import preview as _preview
    return _preview._fill_partition(geometry, strict_mode)

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

def _fill_preview_signature(obj):
    """Return a cheap identity signature for one preview target.

    Geometry changes are invalidated by the dependency-graph callback below;
    this signature is deliberately limited to ownership, topology counts and
    transform so that a wheel event never performs a full mesh digest.  The
    live ``.sculpt_face_set`` values are intentionally not part of either
    surface-cache key; a new session reads the current attribute for its seed
    Face Set prior.
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


def _fill_preview_cached_visibility_check(obj, cached):
    """Compare a reusable full graph's hidden mask with the live mesh once.

    The adjacency signature intentionally omits visibility because it is a
    cheap ownership/topology key.  Visibility is therefore validated at cache
    handoff; a mismatch invalidates the complete graph so hidden filtering is
    rebuilt instead of patched in place.
    """
    import numpy as np

    try:
        mesh = obj.data
        cached_hidden = np.asarray(cached.get("hidden"), dtype=bool).reshape(-1)
        if len(cached_hidden) != len(mesh.polygons):
            return {"matches": False, "reason": "hidden-schema", "hidden_count": 0}
        current_hidden = np.empty(len(mesh.polygons), dtype=bool)
        mesh.polygons.foreach_get("hide", current_hidden)
        matches = bool(np.array_equal(current_hidden, cached_hidden))
        return {
            "matches": matches,
            "reason": "match" if matches else "hidden-changed",
            "hidden_count": int(np.count_nonzero(current_hidden)),
        }
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return {"matches": False, "reason": "visibility-read-error", "hidden_count": 0}


def _fill_preview_prepare_token(stage, done=None, total=None, stage_index=None, stage_count=None, phase="prepare"):
    """Describe one bounded preparation slice for the modal progress HUD."""
    token = {
        "stage": str(stage),
        "phase": str(phase),
        "indeterminate": done is None or total is None or int(total) <= 0,
    }
    if done is not None:
        token["done"] = int(done)
    if total is not None:
        token["total"] = int(total)
    if stage_index is not None:
        token["stage_index"] = int(stage_index)
    if stage_count is not None:
        token["stage_count"] = int(stage_count)
    return token


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
    yield _fill_preview_prepare_token("allocate", 0, 1, 0, 8)
    mesh.vertices.foreach_get("co", coordinates.ravel())
    yield _fill_preview_prepare_token("vertices", 1, 1, 1, 8)
    mesh.loops.foreach_get("edge_index", loop_edges)
    yield _fill_preview_prepare_token("loop_edges", 1, 1, 2, 8)
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    yield _fill_preview_prepare_token("loop_vertices", 1, 1, 3, 8)
    mesh.polygons.foreach_get("loop_total", totals)
    yield _fill_preview_prepare_token("totals", 1, 1, 4, 8)
    mesh.polygons.foreach_get("hide", hidden)
    yield _fill_preview_prepare_token("hidden", 1, 1, 5, 8)
    centers = np.empty((face_count, 3), dtype=np.float32)
    normals = np.empty((face_count, 3), dtype=np.float32)
    mesh.polygons.foreach_get("center", centers.ravel())
    yield _fill_preview_prepare_token("centers", 1, 1, 6, 8)
    mesh.polygons.foreach_get("normal", normals.ravel())
    yield _fill_preview_prepare_token("normals", 1, 1, 7, 8)
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
    face_vertex_ids_global = tuple(
        tuple(
            int(mesh.loops[int(loop_index)].vertex_index)
            for loop_index in mesh.polygons[int(face_id)].loop_indices
        )
        for face_id in face_ids
    )
    vertex_values = sorted(
        set(int(value) for values in face_vertex_ids_global for value in values)
    )
    vertex_local = {value: index for index, value in enumerate(vertex_values)}
    world_vertices = np.asarray(
        [
            tuple(
                float(value)
                for value in (obj.matrix_world @ mesh.vertices[vertex_id].co)
            )
            for vertex_id in vertex_values
        ],
        dtype=np.float64,
    )
    face_vertex_ids = tuple(
        tuple(vertex_local[int(value)] for value in values)
        for values in face_vertex_ids_global
    )
    # ``world_vertices`` is compact for a cursor crop.  Keep pair endpoints
    # in that same compact id space; mesh.edges[].vertices are mesh-global
    # ids and must never be mixed with these indices.
    pair_v0 = np.empty(len(pair_edges), dtype=np.int32)
    pair_v1 = np.empty(len(pair_edges), dtype=np.int32)
    for pair_index, edge_index in enumerate(pair_edges):
        edge_vertices = tuple(int(value) for value in mesh.edges[int(edge_index)].vertices)
        if len(edge_vertices) != 2:
            pair_v0[pair_index] = -1
            pair_v1[pair_index] = -1
            continue
        try:
            pair_v0[pair_index] = int(vertex_local[edge_vertices[0]])
            pair_v1[pair_index] = int(vertex_local[edge_vertices[1]])
        except KeyError:
            pair_v0[pair_index] = -1
            pair_v1[pair_index] = -1
    # Keep the actual world-space endpoints for every face pair.  The
    # shading proxy uses this local edge direction to distinguish two shores
    # from two contacts on the same shore; an edge index alone is not enough
    # after the crop-local graph has been filtered.
    pair_edge_points = np.empty((len(pair_edges), 2, 3), dtype=np.float64)
    for pair_index, edge_index in enumerate(pair_edges):
        if pair_v0[pair_index] < 0 or pair_v1[pair_index] < 0:
            pair_edge_points[pair_index] = np.nan
            continue
        pair_edge_points[pair_index, 0] = np.asarray(
            world_vertices[pair_v0[pair_index]],
            dtype=np.float64,
        )
        pair_edge_points[pair_index, 1] = np.asarray(
            world_vertices[pair_v1[pair_index]],
            dtype=np.float64,
        )
    delta = centers[second] - centers[first]
    pair_lengths = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-20)
    sources = np.r_[first, second]
    destinations = np.r_[second, first]
    directed_lengths = np.r_[pair_lengths, pair_lengths]
    directed_v0 = np.r_[pair_v0, pair_v0]
    directed_v1 = np.r_[pair_v1, pair_v1]
    directed_edges = np.r_[pair_edges, pair_edges]
    graph_order = np.argsort(sources, kind="stable")
    degree = np.bincount(sources, minlength=face_count)
    offsets = np.r_[0, np.cumsum(degree)].astype(np.int64)
    face_of_loop = np.repeat(
        np.arange(len(totals), dtype=np.int32), totals
    )
    physical_loop = (
        (edge_degree[loop_edges] == 2)
        & ~hidden_all[face_of_loop]
    ).astype(np.int32, copy=False)
    physical_degree_all = np.add.reduceat(physical_loop, face_starts)
    return {
        "signature": cache["signature"],
        "count": face_count,
        "face_edge_counts": local_counts.astype(np.int32, copy=False),
        "physical_degree": physical_degree_all[face_ids].astype(
            np.int32, copy=False
        ),
        "centers": centers,
        "normals": normals,
        "hidden": np.zeros(face_count, dtype=bool),
        "world_vertices": world_vertices,
        "offsets": offsets,
        "neighbors": destinations[graph_order].astype(np.int32, copy=False),
        "neighbor_lengths": directed_lengths[graph_order],
        "edge_v0": directed_v0[graph_order].astype(np.int32, copy=False),
        "edge_v1": directed_v1[graph_order].astype(np.int32, copy=False),
        "edge_indices": directed_edges[graph_order].astype(np.int32, copy=False),
        "first": first,
        "second": second,
        "pair_lengths": pair_lengths,
        "pair_v0": pair_v0,
        "pair_v1": pair_v1,
        "pair_edge_indices": pair_edges,
        "pair_edge_points": pair_edge_points,
        "pair_kind": pair_kind,
        "face_vertex_ids": face_vertex_ids,
        "seam_count": int(seam_count),
        "face_ids": face_ids,
        "spatial_radius": float(spatial_radius),
        "spatial_faces": int(len(spatial_ids)),
        "seed_neighbor_lengths": pair_lengths[
            (first == int(np.flatnonzero(face_ids == seed_face)[0]))
            | (second == int(np.flatnonzero(face_ids == seed_face)[0]))
        ],
        "cursor_local": True,
        "vertex_id_space": "compact",
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
    face_vertex_ids = []
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
        # Keep vertex ids local to ``world_vertices``.  This makes the same
        # face-vertex schema usable by the enclosed-component resolver and by
        # the valley proof after a cursor crop.
        face_vertex_ids.append(
            tuple(
                local_vertex(int(mesh.loops[int(loop_index)].vertex_index))
                for loop_index in loop_indices
            )
        )
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
    pair_edge_points = (
        np.stack(
            (np.asarray(vertex_world, dtype=np.float64)[pair_v0],
             np.asarray(vertex_world, dtype=np.float64)[pair_v1]),
            axis=1,
        )
        if len(pair_v0)
        else np.empty((0, 2, 3), dtype=np.float64)
    )
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
        "pair_edge_points": pair_edge_points,
        "pair_kind": pair_kind,
        "face_vertex_ids": tuple(face_vertex_ids),
        "seam_count": int(seam_count),
        "face_ids": face_ids,
        "spatial_radius": float(spatial_radius),
        "spatial_faces": int(len(spatial_ids)),
        "seed_neighbor_lengths": seed_neighbor_lengths,
        "cursor_local": True,
        "vertex_id_space": "compact",
    }


def _fill_preview_cursor_prepare_steps(obj, seed_face):
    """Yield cursor-first reads before returning one compact adjacency graph."""
    import numpy as np
    mesh = obj.data
    signature = _fill_preview_signature(obj)
    if signature is None:
        raise RuntimeError("preview target is unavailable")
    cache = _runtime.fill_preview_cursor_cache.get(signature)
    if cache is not None and cache.get("geometry_dirty"):
        # Cursor preparation has its own small reusable index.  Refresh the
        # volatile centers/loop metadata in place; Face Set IDs are not part
        # of this cache and therefore never cause an eviction.
        face_count = int(len(mesh.polygons))
        loop_count = int(len(mesh.loops))
        if (
            len(np.asarray(cache.get("world_centers", ()))) != face_count
            or len(np.asarray(cache.get("loop_edges", ()))) != loop_count
            or len(np.asarray(cache.get("totals", ()))) != face_count
        ):
            _runtime.fill_preview_cursor_cache.pop(signature, None)
            cache = None
        else:
            centers_local = np.empty((face_count, 3), dtype=np.float32)
            totals = np.empty(face_count, dtype=np.int32)
            loop_edges = np.empty(loop_count, dtype=np.int32)
            mesh.polygons.foreach_get("center", centers_local.ravel())
            matrix = np.asarray(obj.matrix_world, dtype=np.float64)
            cache["world_centers"] = (
                centers_local.astype(np.float64) @ matrix[:3, :3].T
                + matrix[:3, 3]
            )
            mesh.polygons.foreach_get("loop_total", totals)
            mesh.loops.foreach_get("edge_index", loop_edges)
            cache["edge_degree"] = np.bincount(
                loop_edges, minlength=len(mesh.edges)
            ).astype(np.int32, copy=False)
            cache["loop_edges"] = loop_edges
            cache["totals"] = totals
            cache["geometry_dirty"] = False
            cache["geometry_refresh_count"] = int(
                cache.get("geometry_refresh_count", 0)
            ) + 1
            cache["geometry_refresh_reason"] = "depsgraph-dirty"
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
        _runtime.fill_preview_cursor_cache[signature] = cache
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


def _fill_preview_topology_fingerprint(loop_edges, loop_vertices, totals):
    """Return a compact fingerprint for the connectivity-only cache layer.

    Face Set values are intentionally absent.  The Smart Fill graph
    is an expensive connectivity index; a Face Set paint changes only the
    live classification read by each E invocation and must not evict this
    index.  The loop arrays are retained only as a compact integrity guard so
    a same-count rewiring still forces a rebuild.
    """
    import numpy as np

    digest = hashlib.blake2b(digest_size=16)
    for values in (loop_edges, loop_vertices, totals):
        array = np.ascontiguousarray(values)
        digest.update(array.tobytes())
    return (
        int(len(loop_edges)),
        int(len(loop_vertices)),
        int(len(totals)),
        digest.digest(),
    )


def _fill_preview_array_fingerprint(values):
    """Hash one numeric array without retaining its contents as cache state."""
    import numpy as np

    array = np.ascontiguousarray(values)
    return (tuple(int(value) for value in array.shape), hashlib.blake2b(
        array.tobytes(), digest_size=16
    ).digest())


def _fill_preview_face_set_fingerprint(obj):
    """Return a change hint for live Face Set values, never a Face Set cache."""
    import numpy as np

    try:
        attribute = obj.data.attributes.get(".sculpt_face_set")
        if attribute is None or attribute.domain != "FACE":
            return None
        values = np.empty(len(attribute.data), dtype=np.int32)
        attribute.data.foreach_get("value", values)
        return _fill_preview_array_fingerprint(values)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _fill_preview_refresh_cached_adjacency(obj, cached):
    """Refresh volatile geometry/visibility in a retained topology graph.

    Depsgraph cannot distinguish a ``.sculpt_face_set`` write from a sculpt
    coordinate edit: both are commonly reported as ``is_updated_geometry``.
    Therefore the handler marks the graph dirty instead of deleting it.  On
    the next E, this function rereads the current coordinates, normals and
    hidden flags, validates loop topology, and rebuilds only the filtered CSR
    views and edge weights.  Face Set values are only fingerprinted as a
    volatile-update hint; the actual values remain live-read by each preview
    result and are never used to build this graph.

    Returning ``False`` means the topology guard failed (or the entry is an
    older cache without the split-layer fields); the caller must perform a
    cold graph build.  A visibility change remains correct because the
    visible CSR is regenerated from the retained raw pair list.
    """
    import numpy as np

    if not isinstance(cached, dict):
        return False
    required = (
        "topology_first",
        "topology_second",
        "topology_edge_v0",
        "topology_edge_v1",
    )
    if any(key not in cached for key in required):
        return False
    try:
        cached["visibility_refresh_verified"] = False
        mesh = obj.data
        vertex_count = int(len(mesh.vertices))
        face_count = int(len(mesh.polygons))
        loop_count = int(len(mesh.loops))
        if int(cached.get("count", -1)) != face_count:
            return False

        # Face Set Paint and Smart Fill confirmation are the common dirty path.  The
        # graph deliberately excludes Face Set IDs, so identify that path from
        # the live attribute first and avoid touching the large loop/coordinate
        # arrays.  Hidden state is still checked because visibility is a hard
        # traversal boundary.  A simultaneous Face Set+geometry edit is
        # conservatively handled by the normal geometry path when the Face Set
        # fingerprint did not change; Blender does not expose a more granular
        # per-attribute depsgraph revision.
        face_set_fingerprint = _fill_preview_face_set_fingerprint(obj)
        if (
            face_set_fingerprint is not None
            and cached.get("face_set_fingerprint") is not None
            and face_set_fingerprint != cached.get("face_set_fingerprint")
        ):
            hidden = np.empty(face_count, dtype=bool)
            mesh.polygons.foreach_get("hide", hidden)
            hidden_matches = bool(
                np.array_equal(
                    hidden,
                    np.asarray(cached.get("hidden", ()), dtype=bool).reshape(-1),
                )
            )
            if hidden_matches:
                cached["face_set_fingerprint"] = face_set_fingerprint
                cached["geometry_dirty"] = False
                cached["geometry_refresh_count"] = int(
                    cached.get("geometry_refresh_count", 0)
                ) + 1
                cached["geometry_refresh_reason"] = "face-set-only"
                cached["visibility_refresh_verified"] = True
                cached["visibility_refresh_hidden_count"] = int(
                    np.count_nonzero(hidden)
                )
                return True

        loop_edges = np.empty(loop_count, dtype=np.int32)
        loop_vertices = np.empty(loop_count, dtype=np.int32)
        totals = np.empty(face_count, dtype=np.int32)
        mesh.loops.foreach_get("edge_index", loop_edges)
        mesh.loops.foreach_get("vertex_index", loop_vertices)
        mesh.polygons.foreach_get("loop_total", totals)
        fingerprint = _fill_preview_topology_fingerprint(
            loop_edges, loop_vertices, totals
        )
        if fingerprint != cached.get("topology_fingerprint"):
            return False

        coordinates = np.empty((vertex_count, 3), dtype=np.float32)
        hidden = np.empty(face_count, dtype=bool)
        mesh.vertices.foreach_get("co", coordinates.ravel())
        mesh.polygons.foreach_get("hide", hidden)
        coordinate_fingerprint = _fill_preview_array_fingerprint(coordinates)
        hidden_matches = bool(
            np.array_equal(
                hidden,
                np.asarray(cached.get("hidden", ()), dtype=bool).reshape(-1),
            )
        )
        if (
            fingerprint == cached.get("topology_fingerprint")
            and hidden_matches
            and coordinate_fingerprint == cached.get("coordinate_fingerprint")
        ):
            # This is the common Face Set Paint/Smart Fill-confirm case.  The
            # topology graph and its geometry are still valid; only the live
            # Face Set attribute changed.  Clear the dirty marker without
            # sorting or rebuilding the CSR graph.
            cached["face_set_fingerprint"] = face_set_fingerprint
            cached["coordinate_fingerprint"] = coordinate_fingerprint
            cached["geometry_dirty"] = False
            cached["geometry_refresh_count"] = int(
                cached.get("geometry_refresh_count", 0)
            ) + 1
            cached["geometry_refresh_reason"] = "face-set-only"
            cached["visibility_refresh_verified"] = True
            cached["visibility_refresh_hidden_count"] = int(
                np.count_nonzero(hidden)
            )
            return True
        # Unwelded-seam bridges are inferred from world-space coincidence and
        # normal compatibility, so a deformation can change whether a seam
        # exists even when the loop topology is unchanged.  Rebuild those
        # uncommon graphs instead of retaining a stale seam bridge.
        if int(cached.get("seam_count", 0)):
            return False
        centers_local = np.empty((face_count, 3), dtype=np.float32)
        normals_local = np.empty((face_count, 3), dtype=np.float32)
        mesh.polygons.foreach_get("center", centers_local.ravel())
        mesh.polygons.foreach_get("normal", normals_local.ravel())

        world_matrix = np.asarray(obj.matrix_world, dtype=np.float64)
        transform = world_matrix[:3, :3]
        translation = world_matrix[:3, 3]
        world_vertices = coordinates.astype(np.float64) @ transform.T + translation
        centers = centers_local.astype(np.float64) @ transform.T + translation
        normals = normals_local.astype(np.float64) @ np.linalg.inv(transform)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-20)

        raw_first = np.asarray(cached["topology_first"], dtype=np.int32).reshape(-1)
        raw_second = np.asarray(cached["topology_second"], dtype=np.int32).reshape(-1)
        raw_v0 = np.asarray(cached["topology_edge_v0"], dtype=np.int32).reshape(-1)
        raw_v1 = np.asarray(cached["topology_edge_v1"], dtype=np.int32).reshape(-1)
        if not (
            len(raw_first) == len(raw_second)
            and len(raw_first) == len(raw_v0)
            and len(raw_first) == len(raw_v1)
        ):
            return False
        if (
            (len(raw_first) and (
                int(np.min(raw_first)) < 0
                or int(np.max(raw_first)) >= face_count
                or int(np.min(raw_second)) < 0
                or int(np.max(raw_second)) >= face_count
            ))
            or (len(raw_v0) and (
                int(np.min(raw_v0)) < 0
                or int(np.max(raw_v0)) >= vertex_count
                or int(np.min(raw_v1)) < 0
                or int(np.max(raw_v1)) >= vertex_count
            ))
        ):
            return False

        valid = ~(hidden[raw_first] | hidden[raw_second]) & (
            raw_first != raw_second
        )
        first = raw_first[valid]
        second = raw_second[valid]
        edge_v0 = raw_v0[valid]
        edge_v1 = raw_v1[valid]
        delta = centers[second] - centers[first]
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-20)
        sources = np.r_[first, second]
        destinations = np.r_[second, first]
        source_edges = np.r_[np.arange(len(first)), np.arange(len(first))]
        directed_v0 = np.r_[edge_v0, edge_v0]
        directed_v1 = np.r_[edge_v1, edge_v1]
        order = np.argsort(sources, kind="stable")
        degree = np.bincount(sources, minlength=face_count)
        offsets = np.r_[0, np.cumsum(degree)].astype(np.int64)
        pair_edge_points = (
            np.stack((world_vertices[edge_v0], world_vertices[edge_v1]), axis=1)
            if len(edge_v0)
            else np.empty((0, 2, 3), dtype=np.float64)
        )
        cached.update(
            {
                "face_edge_counts": totals.astype(np.int32, copy=False),
                "physical_degree": degree.astype(np.int32, copy=False),
                "centers": centers,
                "normals": normals,
                "hidden": hidden,
                "world_vertices": world_vertices,
                "offsets": offsets,
                "neighbors": destinations[order].astype(np.int32, copy=False),
                "neighbor_lengths": np.r_[distance, distance][order],
                "edge_indices": source_edges[order].astype(np.int32, copy=False),
                "edge_v0": directed_v0[order].astype(np.int32, copy=False),
                "edge_v1": directed_v1[order].astype(np.int32, copy=False),
                "first": first,
                "second": second,
                "pair_lengths": distance,
                "pair_v0": edge_v0,
                "pair_v1": edge_v1,
                "pair_edge_points": pair_edge_points,
                "coordinate_fingerprint": coordinate_fingerprint,
                "face_set_fingerprint": face_set_fingerprint,
                "geometry_dirty": False,
                "geometry_refresh_count": int(cached.get("geometry_refresh_count", 0)) + 1,
                "geometry_refresh_reason": "depsgraph-dirty",
                "visibility_refresh_verified": True,
                "visibility_refresh_hidden_count": int(np.count_nonzero(hidden)),
            }
        )
        return True
    except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


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
    cached = _runtime.fill_preview_adjacency_cache.get(signature)
    if cached is not None:
        refreshed = False
        if cached.get("geometry_dirty"):
            refreshed = _fill_preview_refresh_cached_adjacency(obj, cached)
            if refreshed:
                cached.pop("visibility_refresh_verified", None)
                cached.pop("visibility_refresh_hidden_count", None)
                return cached
            _runtime.fill_preview_adjacency_cache.pop(signature, None)
            cached = None
        if cached is None:
            visibility = None
        else:
            visibility = _fill_preview_cached_visibility_check(obj, cached)
        if visibility is not None and visibility.get("matches"):
            return cached
        if cached is not None:
            _runtime.fill_preview_adjacency_cache.pop(signature, None)

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

    topology_first = np.asarray(first, dtype=np.int32)
    topology_second = np.asarray(second, dtype=np.int32)
    topology_edge_v0 = np.asarray(edge_v0, dtype=np.int32)
    topology_edge_v1 = np.asarray(edge_v1, dtype=np.int32)
    topology_fingerprint = _fill_preview_topology_fingerprint(
        loop_edges, loop_vertices, totals
    )
    valid = ~(hidden[first] | hidden[second]) & (first != second)
    if not bool(np.all(valid)):
        # Hidden faces need an immutable raw pair list so visibility changes
        # can rebuild the filtered CSR without re-sorting every loop.  When
        # the common all-visible case applies, share the arrays with the
        # visible graph and avoid four large duplicate allocations.
        topology_first = topology_first.copy()
        topology_second = topology_second.copy()
        topology_edge_v0 = topology_edge_v0.copy()
        topology_edge_v1 = topology_edge_v1.copy()
    first, second = first[valid], second[valid]
    edge_v0, edge_v1 = edge_v0[valid], edge_v1[valid]
    delta = centers[second] - centers[first]
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-20)
    sources = np.r_[first, second]
    destinations = np.r_[second, first]
    source_edges = np.r_[np.arange(len(first)), np.arange(len(first))]
    edge_v0_directed = np.r_[edge_v0, edge_v0]
    edge_v1_directed = np.r_[edge_v1, edge_v1]
    face_vertex_ids = tuple(
        tuple(
            int(value)
            for value in loop_vertices[
                int(face_starts[face]):int(face_starts[face]) + int(totals[face])
            ]
        )
        for face in range(face_count)
    )
    order = np.argsort(sources, kind="stable")
    degree = np.bincount(sources, minlength=face_count)
    offsets = np.r_[0, np.cumsum(degree)].astype(np.int64)
    pair_edge_points = (
        np.stack((world_vertices[edge_v0], world_vertices[edge_v1]), axis=1)
        if len(edge_v0)
        else np.empty((0, 2, 3), dtype=np.float64)
    )
    cached = {
        "signature": signature,
        "count": int(face_count),
        "face_edge_counts": totals.astype(np.int32, copy=False),
        "physical_degree": degree.astype(np.int32, copy=False),
        "centers": centers,
        "normals": normals,
        "hidden": hidden,
        "world_vertices": world_vertices,
        "offsets": offsets,
        "neighbors": destinations[order].astype(np.int32, copy=False),
        "neighbor_lengths": np.r_[distance, distance][order],
        # Physical pair id for each directed CSR neighbor.  Boundary-local
        # passes use this index to gather only a one/two-ring edge band.
        "edge_indices": source_edges[order].astype(np.int32, copy=False),
        "edge_v0": edge_v0_directed[order].astype(np.int32, copy=False),
        "edge_v1": edge_v1_directed[order].astype(np.int32, copy=False),
        "first": first,
        "second": second,
        "pair_lengths": distance,
        "pair_v0": edge_v0,
        "pair_v1": edge_v1,
        "pair_edge_points": pair_edge_points,
        "face_vertex_ids": face_vertex_ids,
        # Canonical mesh polygon ids are present even for the full graph.  The
        # proxy/fine valley record uses this field instead of patch-local ids.
        "face_ids": np.arange(face_count, dtype=np.int32),
        "seam_count": seam_count,
        "topology_fingerprint": topology_fingerprint,
        "topology_first": topology_first,
        "topology_second": topology_second,
        "topology_edge_v0": topology_edge_v0,
        "topology_edge_v1": topology_edge_v1,
        "coordinate_fingerprint": _fill_preview_array_fingerprint(coordinates),
        "face_set_fingerprint": _fill_preview_face_set_fingerprint(obj),
        "geometry_dirty": False,
        "geometry_refresh_count": 0,
    }
    # Drop obsolete revisions for this mesh while retaining unrelated meshes.
    mesh_pointer = signature[1]
    for key in list(_runtime.fill_preview_adjacency_cache):
        if key[1] == mesh_pointer and key != signature:
            _runtime.fill_preview_adjacency_cache.pop(key, None)
    _runtime.fill_preview_adjacency_cache[signature] = cached
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


def _fill_preview_simple_feature_neighbor_lengths(
    geometry, feature_angle_radians=math.radians(12.0)
):
    """Double traversal cost across a clear ridge or valley.

    The Shift+E path deliberately uses only the unsigned angle between
    adjacent face normals.  Convex and concave folds therefore receive the
    same lightweight cost, while flat adjacency keeps its physical length.
    """
    import numpy as np

    base = np.asarray(geometry.get("neighbor_lengths", ()), dtype=np.float64).reshape(-1)
    first = np.asarray(geometry.get("first", ()), dtype=np.int32).reshape(-1)
    second = np.asarray(geometry.get("second", ()), dtype=np.int32).reshape(-1)
    normals = np.asarray(geometry.get("normals", ()), dtype=np.float64)
    count = int(geometry.get("count", 0))
    if (
        normals.shape != (count, 3)
        or len(first) != len(second)
        or len(base) != 2 * len(first)
        or np.any(first < 0)
        or np.any(second < 0)
        or np.any(first >= count)
        or np.any(second >= count)
    ):
        raise RuntimeError("simple feature-cost geometry schema is invalid")
    dots = np.einsum("ij,ij->i", normals[first], normals[second])
    threshold_dot = math.cos(max(float(feature_angle_radians), 0.0))
    feature_pairs = np.isfinite(dots) & (
        np.clip(dots, -1.0, 1.0) <= threshold_dot
    )
    directed_pairs = None
    if not geometry.get("cursor_local"):
        candidate = np.asarray(
            geometry.get("edge_indices", ()), dtype=np.int64
        ).reshape(-1)
        if (
            len(candidate) == len(base)
            and (len(candidate) == 0 or np.min(candidate) >= 0)
            and (len(candidate) == 0 or np.max(candidate) < len(first))
        ):
            directed_pairs = candidate
    if directed_pairs is None:
        sources = np.r_[first, second]
        graph_order = np.argsort(sources, kind="stable")
        directed_pairs = np.r_[
            np.arange(len(first), dtype=np.int64),
            np.arange(len(first), dtype=np.int64),
        ][graph_order]
    weighted = np.array(base, copy=True)
    weighted[feature_pairs[directed_pairs]] *= 2.0
    return weighted, feature_pairs


def _fill_preview_progressive_range_step(state, radius):
    """Advance the active normal-E physical range by one bounded stage.

    This small entry point is intentionally shared by the production result
    builder and diagnostics.  It owns only the reusable Dijkstra frontier;
    viewport capture, screen morphology, and projection helpers are not part
    of this active path.
    """
    import numpy as np

    geometry = state["adjacency"]
    if state.get("cursor_local"):
        _fill_preview_cursor_expand(state, float(radius))
        geometry = state["adjacency"]
    seed_value = state.get("seed_local")
    if seed_value is None:
        seed_value = state.get("seed_face", 0)
    seed_face = int(seed_value)
    halo = max(float(radius) * 0.5, float(state.get("initial_radius") or radius) * 0.5)
    patch_radius = float(radius) + halo
    walk_geometry = geometry
    if bool(state.get("expand_only", False)):
        feature_cache_key = (
            id(geometry),
            geometry.get("coordinate_fingerprint"),
            len(geometry.get("neighbor_lengths", ())),
        )
        feature_cache = state.get("simple_feature_cost_cache")
        if (
            not isinstance(feature_cache, dict)
            or feature_cache.get("key") != feature_cache_key
        ):
            weighted_lengths, feature_pairs = (
                _fill_preview_simple_feature_neighbor_lengths(geometry)
            )
            feature_cache = {
                "key": feature_cache_key,
                "neighbor_lengths": weighted_lengths,
                "feature_pairs": feature_pairs,
            }
            state["simple_feature_cost_cache"] = feature_cache
        else:
            weighted_lengths = feature_cache["neighbor_lengths"]
        walk_geometry = dict(geometry)
        walk_geometry["neighbor_lengths"] = weighted_lengths
    previous = state.get("distance_state")
    popped_before = int(previous.get("popped", 0)) if isinstance(previous, dict) else 0
    distances, patch_ids, popped, distance_state = _fill_preview_dijkstra_incremental(
        walk_geometry, seed_face, patch_radius, previous
    )
    state["distance_state"] = distance_state
    newly_processed = max(
        0, int(distance_state.get("popped", popped)) - popped_before
    )
    return {
        "geometry": geometry,
        "seed_face": seed_face,
        "distances": distances,
        "patch_ids": patch_ids,
        "popped": int(popped),
        "patch_radius": float(patch_radius),
        "newly_processed_faces": int(newly_processed),
        "reused_faces": int(max(0, len(patch_ids) - newly_processed)),
    }


def _fill_preview_valley_contact_pair_is_crossing(
    geometry,
    valley_face,
    first_contact,
    second_contact,
    scale=1.0,
    contact_edge_points=None,
):
    """Prove that two contacts lie on opposite shores of one valley face.

    A center projection alone cannot distinguish opposite shores from two
    contacts on the same shore (or from a pair running along the valley).
    Use the actual shared-edge midpoint vectors in the local tangent plane:
    the shore vectors must oppose one another and the contact vector must
    align with their across-valley axis.  Missing/ambiguous edge geometry is
    deliberately rejected so a crop or malformed graph cannot turn a normal
    valley boundary into a rescue.
    """
    import numpy as np

    try:
        valley_faces = np.asarray(valley_face, dtype=np.int32).reshape(-1)
        first_contact = int(first_contact)
        second_contact = int(second_contact)
        centers = np.asarray(geometry["centers"], dtype=np.float64)
        normals = np.asarray(geometry["normals"], dtype=np.float64)
        first = np.asarray(geometry["first"], dtype=np.int32).reshape(-1)
        second = np.asarray(geometry["second"], dtype=np.int32).reshape(-1)
        pair_points = np.asarray(
            geometry.get("pair_edge_points"), dtype=np.float64
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False
    if (
        centers.ndim != 2
        or centers.shape[1] != 3
        or normals.ndim != 2
        or normals.shape[1] != 3
        or len(first) != len(second)
        or pair_points.shape != (len(first), 2, 3)
        or len(valley_faces) == 0
        or np.any(valley_faces < 0)
        or min(first_contact, second_contact) < 0
        or max(int(np.max(valley_faces)), first_contact, second_contact) >= len(centers)
        or first_contact == second_contact
    ):
        return False
    valley_faces = np.unique(valley_faces)

    def shared_pair(contact):
        if contact_edge_points is not None:
            for edge in contact_edge_points.get(int(contact), ()):
                edge = np.asarray(edge, dtype=np.float64)
                if edge.shape == (2, 3) and np.all(np.isfinite(edge)):
                    return edge
            return None
        matches = np.flatnonzero(
            (np.isin(first, valley_faces) & (second == contact))
            | ((first == contact) & np.isin(second, valley_faces))
        )
        for index in matches:
            edge = pair_points[int(index)]
            if np.all(np.isfinite(edge)):
                return edge
        return None

    edge_first = shared_pair(first_contact)
    edge_second = shared_pair(second_contact)
    if edge_first is None or edge_second is None:
        return False
    valley_normal = np.mean(normals[valley_faces], axis=0)
    normal_length = float(np.linalg.norm(valley_normal))
    if not np.isfinite(normal_length) or normal_length <= 1.0e-12:
        return False
    valley_normal = valley_normal / normal_length
    side_vectors = []
    edge_directions = []
    for contact, edge in (
        (first_contact, edge_first),
        (second_contact, edge_second),
    ):
        edge_vector = np.asarray(edge[1] - edge[0], dtype=np.float64)
        edge_length = float(np.linalg.norm(edge_vector))
        if not np.isfinite(edge_length) or edge_length <= 1.0e-12:
            return False
        edge_directions.append(edge_vector / edge_length)
        side = centers[contact] - np.mean(edge, axis=0)
        side = side - valley_normal * float(np.dot(side, valley_normal))
        side_length = float(np.linalg.norm(side))
        if not np.isfinite(side_length) or side_length <= max(float(scale) * 1.0e-6, 1.0e-12):
            return False
        side_vectors.append(side / side_length)
    side_first, side_second = side_vectors
    if float(np.dot(side_first, side_second)) >= -0.15:
        return False
    # The two shore edges should describe the same long direction.  This gate
    # is intentionally conservative for one/two-face components.
    if abs(float(np.dot(edge_directions[0], edge_directions[1]))) < 0.20:
        return False
    cross = centers[second_contact] - centers[first_contact]
    cross = cross - valley_normal * float(np.dot(cross, valley_normal))
    cross_length = float(np.linalg.norm(cross))
    across = side_second - side_first
    across = across - valley_normal * float(np.dot(across, valley_normal))
    across_length = float(np.linalg.norm(across))
    if cross_length <= max(float(scale) * 0.35, 1.0e-12) or across_length <= 1.0e-12:
        return False
    alignment = abs(float(np.dot(cross / cross_length, across / across_length)))
    return bool(alignment >= 0.45)


class _FillPreviewFaceVertexSequence:
    """Tuple-compatible view over the full graph's flat loop schema."""

    __slots__ = ("_flat", "_offsets", "_counts")

    def __init__(self, flat, offsets, counts):
        self._flat = flat
        self._offsets = offsets
        self._counts = counts

    def __len__(self):
        return len(self._counts)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = int(self._offsets[index])
        end = start + int(self._counts[index])
        return self._flat[start:end]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def _fill_preview_face_vertex_sequence(geometry):
    """Return tuple geometry or a zero-copy flat-array compatibility view."""
    stored = geometry.get("face_vertex_ids")
    if stored is not None:
        return stored
    try:
        import numpy as np

        flat = np.asarray(geometry["face_vertex_flat"], dtype=np.int32).reshape(-1)
        offsets = np.asarray(
            geometry["face_vertex_offsets"], dtype=np.int64
        ).reshape(-1)
        counts = np.asarray(
            geometry.get("face_vertex_counts", geometry["face_edge_counts"]),
            dtype=np.int32,
        ).reshape(-1)
        if len(offsets) != len(counts) or len(offsets) == 0:
            return None
        if int(offsets[0]) != 0 or np.any(offsets[1:] < offsets[:-1]):
            return None
        if np.any(counts < 0) or np.any(offsets + counts > len(flat)):
            return None
        return _FillPreviewFaceVertexSequence(flat, offsets, counts)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


def _fill_preview_valley_component_is_crossing(
    geometry, valley_face_ids, shore_face_ids, interface_edges, scale=1.0
):
    """Validate one complete valley component using canonical mesh ids.

    ``interface_edges`` contains ``(valley_id, shore_id, p0, p1)`` records.
    The record is deliberately component-wide: a middle valley face need not
    share an edge with both shores itself.  Every candidate shore pair is
    compared, and a pair is accepted only when its edge directions agree,
    its side vectors oppose, and its span is transverse to the component
    tangent.  Ambiguous one-face polygons (for example a square) fail closed.
    """
    import numpy as np

    try:
        canonical = np.asarray(geometry["face_ids"], dtype=np.int64).reshape(-1)
        centers = np.asarray(geometry["centers"], dtype=np.float64)
        normals = np.asarray(geometry["normals"], dtype=np.float64)
        face_vertices = _fill_preview_face_vertex_sequence(geometry)
        world_vertices = np.asarray(
            geometry.get("world_vertices"), dtype=np.float64
        )
        valley_ids = tuple(dict.fromkeys(int(value) for value in valley_face_ids))
        shore_ids = tuple(dict.fromkeys(int(value) for value in shore_face_ids))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False
    if (
        len(canonical) != len(centers)
        or centers.ndim != 2
        or centers.shape[1] != 3
        or normals.shape != centers.shape
        or len(valley_ids) == 0
        or len(shore_ids) < 2
        or face_vertices is None
        or len(face_vertices) != len(canonical)
        or world_vertices.ndim != 2
        or world_vertices.shape[1] != 3
        or len(set(int(value) for value in canonical)) != len(canonical)
    ):
        return False
    local_by_mesh = {int(mesh_id): index for index, mesh_id in enumerate(canonical)}
    if any(value not in local_by_mesh for value in valley_ids + shore_ids):
        return False
    valley_local = np.asarray([local_by_mesh[value] for value in valley_ids], dtype=np.int32)
    shore_local = {value: local_by_mesh[value] for value in shore_ids}
    normal = np.mean(normals[valley_local], axis=0)
    normal_length = float(np.linalg.norm(normal))
    if not np.isfinite(normal_length) or normal_length <= 1.0e-12:
        return False
    normal /= normal_length
    component_center = np.mean(centers[valley_local], axis=0)
    # The component tangent is derived from the component's complete polygon
    # footprint for every component size.  Vertex ids are deduplicated before
    # PCA so shared vertices do not receive extra weight merely because they
    # occur on two adjacent valley faces.  Project onto the mean-normal
    # tangent plane first; face-center differences can point across the valley
    # and must never become a two-face tangent shortcut.
    def derive_tangent(local_faces):
        vertex_ids = set()
        try:
            for local_face in tuple(int(value) for value in local_faces):
                for vertex_id in face_vertices[local_face]:
                    vertex_id = int(vertex_id)
                    if vertex_id < 0 or vertex_id >= len(world_vertices):
                        return None
                    vertex_ids.add(vertex_id)
        except (IndexError, TypeError, ValueError):
            return None
        if len(vertex_ids) < 2:
            return None
        points = world_vertices[np.asarray(sorted(vertex_ids), dtype=np.int32)]
        points = points - component_center
        points = points - normal * np.sum(points * normal, axis=1, keepdims=True)
        points = points - np.mean(points, axis=0)
        if len(points) < 2:
            return None
        try:
            _u, singular, vh = np.linalg.svd(points, full_matrices=False)
            if (
                not len(singular)
                or float(singular[0]) <= max(float(scale) * 1.0e-8, 1.0e-12)
                or (len(singular) >= 2 and float(singular[0]) <= float(singular[1]) * 1.15)
            ):
                return None
            tangent = vh[0]
        except (np.linalg.LinAlgError, TypeError, ValueError):
            return None
        tangent = tangent - normal * float(np.dot(tangent, normal))
        tangent_length = float(np.linalg.norm(tangent))
        return tangent / tangent_length if tangent_length > 1.0e-12 else None

    tangent = derive_tangent(valley_local)
    if tangent is None:
        # A single face without a geometrically dominant long direction is
        # indistinguishable from a square/ambiguous contact arrangement.
        return False

    by_shore = {shore_id: [] for shore_id in shore_ids}
    try:
        for valley_id, shore_id, point_a, point_b in interface_edges:
            valley_id = int(valley_id)
            shore_id = int(shore_id)
            if valley_id not in valley_ids or shore_id not in by_shore:
                continue
            edge = np.asarray((point_a, point_b), dtype=np.float64)
            if edge.shape != (2, 3) or not np.all(np.isfinite(edge)):
                return False
            by_shore[shore_id].append((valley_id, edge))
    except (TypeError, ValueError):
        return False
    if any(not edges for edges in by_shore.values()):
        return False

    # Compare interface edges at the same local station along the component,
    # rather than requiring every cross-section to sit near the component's
    # global centroid.  The latter rejects a valid long valley as soon as its
    # shores are distributed along the full component.  Keep the entries
    # sorted so the pair search below visits only a bounded local window.
    station_entries = {}
    all_station_entries = []
    max_edge_length = 0.0
    try:
        for shore_id, edges in by_shore.items():
            prepared = []
            for valley_id, edge in edges:
                edge_vector = np.asarray(edge[1] - edge[0], dtype=np.float64)
                edge_length = float(np.linalg.norm(edge_vector))
                edge_midpoint = np.mean(edge, axis=0)
                station = float(np.dot(edge_midpoint, tangent))
                if (
                    not np.isfinite(edge_length)
                    or edge_length <= 1.0e-12
                    or not np.isfinite(station)
                ):
                    return False
                prepared.append(
                    (station, valley_id, edge, edge_length)
                )
                max_edge_length = max(max_edge_length, edge_length)
            prepared.sort(key=lambda value: value[0])
            station_entries[shore_id] = tuple(prepared)
            all_station_entries.extend(
                (float(station), int(shore_id), int(valley_id), edge, float(edge_length))
                for station, valley_id, edge, edge_length in prepared
            )
    except (TypeError, ValueError):
        return False
    if not station_entries or not all_station_entries or max_edge_length <= 1.0e-12:
        return False

    # Compare only entries in one local station window.  The previous
    # shore-pair outer loop still visited every contact pair even though most
    # pairs had no nearby interface edge; indexing all edges once keeps the
    # hot path bounded by local interface density instead of shore^2.
    all_station_entries.sort(key=lambda value: value[0])
    all_stations = np.asarray(
        [entry[0] for entry in all_station_entries], dtype=np.float64
    )
    station_search_window = max(
        float(scale) * 4.0,
        max_edge_length + float(scale) * 2.0,
    )
    pair_geometry = {}
    candidates = []
    for first_index, (
        station_first,
        shore_first_raw,
        valley_first,
        edge_first,
        edge_length_first,
    ) in enumerate(all_station_entries):
        start = int(
            np.searchsorted(
                all_stations,
                station_first - station_search_window,
                side="left",
            )
        )
        end = int(
            np.searchsorted(
                all_stations,
                station_first + station_search_window,
                side="right",
            )
        )
        for second_index in range(max(first_index + 1, start), end):
            (
                station_second,
                shore_second_raw,
                valley_second,
                edge_second,
                edge_length_second,
            ) = all_station_entries[second_index]
            shore_first = int(shore_first_raw)
            shore_second = int(shore_second_raw)
            if shore_first == shore_second:
                continue
            # Keep each unordered contact pair in one orientation.  All
            # gates below are sign-invariant except the side vectors, whose
            # opposing-dot test is symmetric as well.
            if shore_first > shore_second:
                shore_first, shore_second = shore_second, shore_first
                edge_first, edge_second = edge_second, edge_first
                edge_length_first, edge_length_second = (
                    edge_length_second,
                    edge_length_first,
                )
            pair_key = (shore_first, shore_second)
            cached = pair_geometry.get(pair_key)
            if cached is None:
                first_center = centers[shore_local[shore_first]]
                second_center = centers[shore_local[shore_second]]
                cross = second_center - first_center
                cross = cross - normal * float(np.dot(cross, normal))
                cross_length = float(np.linalg.norm(cross))
                pair_geometry[pair_key] = (cross, cross_length)
            else:
                cross, cross_length = cached
            if cross_length <= max(float(scale) * 0.35, 1.0e-12):
                continue
            # The permitted station span is derived from the local graph
            # scale, the two interface edge lengths, and a bounded fraction
            # of the across-valley span.  It is intentionally independent of
            # the component's total length and global center.
            if abs(float(station_second) - float(station_first)) > max(
                float(scale) * 2.0,
                (edge_length_first + edge_length_second) * 0.5,
                min(float(cross_length) * 0.25, float(scale) * 2.0),
            ):
                continue
            dirs = []
            sides = []
            valid = True
            first_center = centers[shore_local[shore_first]]
            second_center = centers[shore_local[shore_second]]
            for contact_center, edge in (
                (first_center, edge_first),
                (second_center, edge_second),
            ):
                edge_vector = edge[1] - edge[0]
                edge_length = float(np.linalg.norm(edge_vector))
                if edge_length <= 1.0e-12:
                    valid = False
                    break
                dirs.append(edge_vector / edge_length)
                side = contact_center - np.mean(edge, axis=0)
                side = side - normal * float(np.dot(side, normal))
                side_length = float(np.linalg.norm(side))
                if side_length <= max(float(scale) * 1.0e-6, 1.0e-12):
                    valid = False
                    break
                sides.append(side / side_length)
            if not valid:
                continue
            # Always use the component tangent.  Re-deriving it from the two
            # contact faces turns a two-face cross-valley center difference
            # into a false longitudinal tangent.
            if abs(float(np.dot(cross, tangent))) / cross_length > 0.78:
                continue
            if float(np.dot(sides[0], sides[1])) >= -0.25:
                continue
            if abs(float(np.dot(dirs[0], dirs[1]))) < 0.45:
                continue
            # A shore edge is longitudinal; a pair whose center span follows
            # it is a long-way/end contact, not a cross-section.
            if max(
                abs(float(np.dot(cross / cross_length, direction)))
                for direction in dirs
            ) > 0.70:
                continue
            across = sides[1] - sides[0]
            across_length = float(np.linalg.norm(across))
            if across_length <= 1.0e-12:
                continue
            alignment = abs(
                float(np.dot(cross / cross_length, across / across_length))
            )
            if alignment < 0.45:
                continue
            candidates.append(cross_length)
    # At least one stable section is required.  The component-wide tangent,
    # opposing side vectors, and complete interface scan make a lone accepted
    # pair a proof of this component rather than a scalar face shortcut.
    if not candidates:
        return False
    return True


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
    if np.any(patch_ids < 0) or np.any(patch_ids >= global_count):
        return patch_ids, {
            "proxy_seconds": float(time.perf_counter() - started),
            "proxy_nodes": count,
            "proxy_candidate_faces": count,
            "proxy_band_faces": count,
            "proxy_barrier_edges": 0,
            "proxy_valley_merge_reason": "patch-id-schema",
        }
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
    raw_pair_points = geometry.get("pair_edge_points")
    pair_edge_points = None
    if raw_pair_points is not None:
        raw_pair_points = np.asarray(raw_pair_points, dtype=np.float64)
        if raw_pair_points.shape == (len(graph_first), 2, 3):
            pair_edge_points = raw_pair_points[keep]

    # Split the original face graph at the signed valley barriers.  True seams
    # remain traversable here; they are not geometric valley barriers.  This
    # component uses only the requested target radius.  The halo is retained
    # for valley detection and fine boundary inspection, but it cannot provide
    # a route around a valley that lies outside the requested display range.
    patch_distances = np.asarray(distances, dtype=np.float64)[patch_ids]
    target_mask = np.isfinite(patch_distances) & (
        patch_distances <= float(target_radius) + 1.0e-9
    )
    hidden = np.asarray(
        geometry.get("hidden", np.zeros(global_count, dtype=bool)), dtype=bool
    )
    if len(hidden) == global_count:
        target_mask &= ~hidden[patch_ids]
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

    # A valley can be several faces wide and can run for many face rows.  Keep
    # one component-wide record so the fine stage can revalidate the same
    # proof even when the exact partition excludes every valley face.
    valley_merge = np.zeros(count, dtype=bool)
    valley_components = []
    valley_faces = (face_valley >= valley_threshold) & target_mask
    incidence_sources = np.r_[first, second]
    incidence_edges = np.r_[
        np.arange(len(first), dtype=np.int32),
        np.arange(len(first), dtype=np.int32),
    ]
    incidence_order = np.argsort(incidence_sources, kind="stable")
    incidence_faces = incidence_sources[incidence_order]
    incidence_edges = incidence_edges[incidence_order]
    incidence_counts = np.bincount(incidence_sources, minlength=count)
    raw_edge_counts = geometry.get("face_edge_counts")
    edge_counts = np.empty(0, dtype=np.int32)
    valley_protection_ready = False
    if raw_edge_counts is not None:
        raw_edge_counts = np.asarray(raw_edge_counts, dtype=np.int32).reshape(-1)
        if len(raw_edge_counts) == global_count and np.all(raw_edge_counts > 0):
            edge_counts = raw_edge_counts[patch_ids]
            valley_protection_ready = True
        elif (
            len(raw_edge_counts) == count
            and global_count == count
            and np.all(raw_edge_counts > 0)
        ):
            # A genuinely patch-local geometry is valid only when its count
            # cannot be confused with a larger global geometry.
            edge_counts = raw_edge_counts
            valley_protection_ready = True
    if len(hidden) != global_count:
        valley_protection_ready = False
    try:
        mesh_face_ids = np.asarray(
            geometry.get("face_ids"), dtype=np.int64
        ).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        mesh_face_ids = np.empty(0, dtype=np.int64)
    canonical_ready = (
        len(mesh_face_ids) == global_count
        and len(np.unique(mesh_face_ids)) == global_count
    )
    valley_seen = np.zeros(count, dtype=bool)
    for valley_start in (
        np.flatnonzero(valley_faces) if valley_protection_ready else ()
    ):
        valley_start = int(valley_start)
        if valley_seen[valley_start]:
            continue
        valley_seen[valley_start] = True
        pending = [valley_start]
        component = []
        while pending:
            valley_face = int(pending.pop())
            component.append(valley_face)
            incidence_start = int(
                np.searchsorted(incidence_faces, valley_face, side="left")
            )
            incidence_end = int(
                np.searchsorted(incidence_faces, valley_face, side="right")
            )
            for edge in incidence_edges[incidence_start:incidence_end]:
                edge = int(edge)
                left = int(first[edge])
                right = int(second[edge])
                other = right if left == valley_face else left
                if valley_faces[other] and not valley_seen[other]:
                    valley_seen[other] = True
                    pending.append(other)

        component = np.asarray(component, dtype=np.int32)
        component_mask = np.zeros(count, dtype=bool)
        component_mask[component] = True
        component_open = False
        contacts = set()
        interface_edges = []
        for valley_face in component:
            valley_face = int(valley_face)
            # ``face_edge_counts`` is the original polygon edge count.  A
            # lower graph degree means an open/non-manifold/crop boundary.
            expected_edges = int(edge_counts[valley_face]) if len(edge_counts) == count else 0
            if expected_edges and int(incidence_counts[valley_face]) < expected_edges:
                component_open = True
            incidence_start = int(
                np.searchsorted(incidence_faces, valley_face, side="left")
            )
            incidence_end = int(
                np.searchsorted(incidence_faces, valley_face, side="right")
            )
            for edge in incidence_edges[incidence_start:incidence_end]:
                edge = int(edge)
                left = int(first[edge])
                right = int(second[edge])
                other = right if left == valley_face else left
                # A neighbor beyond the requested radius is a legitimate
                # clipping endpoint.  Do not reject the in-radius valley
                # component merely because the valley continues outside the
                # current brush domain; hidden faces remain a hard boundary.
                if not target_mask[other] and bool(hidden[patch_ids[other]]):
                    component_open = True
                elif not component_mask[other] and candidate[other]:
                    contacts.add(other)
                    if pair_edge_points is not None:
                        # Explicit ID conversion: patch-local -> geometry
                        # local via patch_ids -> canonical mesh polygon id via
                        # geometry["face_ids"].  The edge points come from the
                        # same retained graph edge.
                        valley_geometry_id = int(patch_ids[valley_face])
                        shore_geometry_id = int(patch_ids[other])
                        interface_edges.append(
                            (
                                int(mesh_face_ids[valley_geometry_id]),
                                int(mesh_face_ids[shore_geometry_id]),
                                tuple(float(value) for value in pair_edge_points[edge, 0]),
                                tuple(float(value) for value in pair_edge_points[edge, 1]),
                            )
                        )
        if component_open or len(contacts) < 2:
            continue
        if not canonical_ready or pair_edge_points is None:
            continue
        component_geometry_ids = tuple(
            int(patch_ids[int(face)]) for face in component
        )
        contact_geometry_ids = tuple(
            int(patch_ids[int(face)]) for face in sorted(contacts)
        )
        component_mesh_ids = tuple(
            int(mesh_face_ids[geometry_id])
            for geometry_id in component_geometry_ids
        )
        shore_mesh_ids = tuple(
            int(mesh_face_ids[geometry_id])
            for geometry_id in contact_geometry_ids
        )
        if not _fill_preview_valley_component_is_crossing(
            # The helper receives the complete geometry-local arrays and the
            # canonical mesh ids above.  Slicing only face_ids would make the
            # canonical-to-local map disagree with centers/normals/vertices.
            geometry,
            component_mesh_ids,
            shore_mesh_ids,
            interface_edges,
            scale=local_scale,
        ):
            continue
        valley_merge[component] = True
        valley_components.append(
            {
                "valley_face_ids": component_mesh_ids,
                "shore_face_ids": shore_mesh_ids,
                "interface_edges": tuple(interface_edges),
                "scale": float(local_scale),
            }
        )
    if np.any(valley_merge):
        candidate |= valley_merge

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
    proxy_valley_merge_ids = patch_ids[np.flatnonzero(valley_merge)].astype(
        np.int32, copy=False
    )
    return patch_ids[selected], {
        "proxy_seconds": float(time.perf_counter() - started),
        "proxy_nodes": count,
        "proxy_candidate_faces": int(np.count_nonzero(candidate)),
        "proxy_component_faces": int(np.count_nonzero(candidate)),
        "proxy_correction_faces": 0,
        "proxy_band_faces": int(len(selected)),
        "proxy_barrier_edges": int(np.count_nonzero(barrier)),
        "proxy_valley_threshold": float(valley_threshold),
        "proxy_valley_merge_protection": bool(
            valley_protection_ready and pair_edge_points is not None
        ),
        "proxy_valley_merge_reason": (
            "edge-count-schema"
            if not valley_protection_ready
            else "pair-edge-schema"
            if pair_edge_points is None
            else "ok"
        ),
        "proxy_valley_merge_faces": int(np.count_nonzero(valley_merge)),
        "proxy_valley_merge_ids": proxy_valley_merge_ids,
        "proxy_valley_components": tuple(valley_components),
        "proxy_candidate_ids": proxy_candidate_ids,
    }


def _fill_preview_local_geometry(geometry, face_ids, strict_mode=False):
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
        "face_edge_counts": np.asarray(
            geometry.get("face_edge_counts", np.zeros(int(len(face_ids)), dtype=np.int32)),
            dtype=np.int32,
        )[face_ids].astype(np.int32, copy=False),
        "count": int(len(face_ids)),
        "seam_count": int(geometry.get("seam_count", 0)),
        "partitions": {},
        "_global_face_ids": face_ids,
        # The endpoint ids and ``world_vertices`` are meaningful only when
        # their id space is explicitly declared by the geometry builder.
        # Never infer mesh-global ids for a compact cursor slice.
        "vertex_id_space": geometry.get("vertex_id_space"),
    }
    geometry_count = int(geometry.get("count", len(geometry.get("hidden", ()))))
    physical_degree = np.asarray(
        geometry.get("physical_degree", ()), dtype=np.int32
    ).reshape(-1)
    full_edge_counts = np.asarray(
        geometry.get("face_edge_counts", ()), dtype=np.int32
    ).reshape(-1)
    source_hard = np.asarray(
        geometry.get("source_hard", ()), dtype=bool
    ).reshape(-1)
    if len(physical_degree) == geometry_count and len(full_edge_counts) == geometry_count:
        local["physical_degree"] = physical_degree[face_ids].copy()
        if len(source_hard) != geometry_count:
            source_hard = (
                geometry["hidden"]
                | (physical_degree < full_edge_counts)
            )
    if len(source_hard) == geometry_count:
        local["source_hard"] = source_hard[face_ids].copy()
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
        if bool(strict_mode):
            # Preserve the conservative Ctrl+E geometry path byte-for-byte
            # in its feature normalization; only ordinary E uses the
            # candidate-size-independent local normalization below.
            local_scale = max(float(np.median(pair_lengths)), 1.0e-12)
            normalized_turn = -signed_turn / np.maximum(
                pair_lengths / local_scale, 1.0e-12
            )
        else:
            # Normalize each edge against its incident physical edge scale. A
            # larger wheel stage therefore cannot change the same local valley
            # by changing the median of a much wider candidate patch.
            face_degree = np.bincount(first, minlength=count) + np.bincount(
                second, minlength=count
            )
            face_edge_scale = (
                np.bincount(first, weights=pair_lengths, minlength=count)
                + np.bincount(second, weights=pair_lengths, minlength=count)
            ) / np.maximum(face_degree, 1)
            pair_edge_scale = 0.5 * (
                face_edge_scale[first] + face_edge_scale[second]
            )
            normalized_turn = -signed_turn / np.maximum(
                pair_lengths / np.maximum(pair_edge_scale, 1.0e-12), 1.0e-12
            )
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
        if bool(strict_mode):
            robust_center = float(np.median(face_valley))
            robust_mad = float(np.median(np.abs(face_valley - robust_center)))
            valley_threshold = max(0.025, robust_center + 2.0 * robust_mad)
            normalized_face_valley = np.maximum(
                face_valley / max(valley_threshold, 1.0e-12) * 0.10,
                0.0,
            )
        else:
            # Keep the scalar's physical meaning fixed. Robust local
            # hysteresis is applied later by the grayscale edge window; this
            # stage must not recompute a candidate-wide baseline that changes
            # with radius.
            valley_threshold = 0.025
            normalized_face_valley = np.maximum(face_valley * 0.10, 0.0)
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

    seed_local = int(seed_local)
    working_local = local
    face_set_values = np.asarray(
        local.get("face_set_values", ()), dtype=np.int32
    ).reshape(-1)
    face_set_seed_id = int(local.get("face_set_seed_id", -1))
    prior_enabled = bool(local.get("face_set_prior_enabled", False))
    trusted_same_set = np.zeros(int(local["count"]), dtype=bool)
    relaxed_pair_count = 0
    neutral_crossing_count = 0
    compact_cut_relaxed_faces = 0
    source_hard_faces = 0
    prior_safety_reason = "disabled"
    prior_reason = "disabled"
    if (
        prior_enabled
        and face_set_seed_id >= 0
        and len(face_set_values) == int(local["count"])
        and 0 <= seed_local < int(local["count"])
    ):
        count = int(local["count"])
        offsets = np.asarray(local["offsets"], dtype=np.int64).reshape(-1)
        neighbors = np.asarray(local["neighbors"], dtype=np.int32).reshape(-1)
        hidden = np.asarray(
            local.get("hidden", np.zeros(count, dtype=bool)), dtype=bool
        ).reshape(-1)
        degree = np.diff(offsets) if len(offsets) == count + 1 else np.zeros(0, dtype=np.int32)
        edge_counts = np.asarray(
            local.get("face_edge_counts", degree), dtype=np.int32
        ).reshape(-1)
        source_hard = np.asarray(
            local.get("source_hard", ()), dtype=bool
        ).reshape(-1)
        if (
            len(hidden) == count
            and len(edge_counts) == count
            and len(degree) == count
            and int(offsets[-1]) == len(neighbors)
        ):
            if len(source_hard) == count:
                # ``degree`` belongs to the compact analysis slice and may
                # be smaller solely because its neighboring faces were
                # cropped.  Only the full-geometry source_hard mask can mark
                # a true open/non-manifold/hidden barrier.
                hard = source_hard | hidden
                source_hard_faces = int(np.count_nonzero(hard))
                compact_cut_mask = (degree < edge_counts) & ~hard
                same_id = face_set_values == face_set_seed_id
                if same_id[seed_local] and not hard[seed_local]:
                    trusted_same_set[seed_local] = True
                    pending_same = deque([seed_local])
                    while pending_same:
                        face = int(pending_same.popleft())
                        for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
                            neighbor = int(neighbor)
                            if (
                                not trusted_same_set[neighbor]
                                and same_id[neighbor]
                                and not hard[neighbor]
                            ):
                                trusted_same_set[neighbor] = True
                                pending_same.append(neighbor)
                compact_cut_relaxed_faces = int(
                    np.count_nonzero(trusted_same_set & compact_cut_mask)
                )
                prior_safety_reason = (
                    "compact-cut-relaxed"
                    if compact_cut_relaxed_faces
                    else "full-geometry-source-hard"
                )
                first = np.asarray(local["first"], dtype=np.int32).reshape(-1)
                second = np.asarray(local["second"], dtype=np.int32).reshape(-1)
                safe_pair = ~hard[first] & ~hard[second]
                if len(first) == len(second):
                    relaxed_pair_count = int(
                        np.count_nonzero(
                            trusted_same_set[first]
                            & trusted_same_set[second]
                            & safe_pair
                        )
                    )
                    neutral_crossing_count = int(
                        np.count_nonzero(
                            (face_set_values[first] != face_set_values[second])
                            & safe_pair
                        )
                    )
            else:
                # No full-geometry safety metadata means the prior is
                # disabled; never infer physical safety from compact degree.
                hard = hidden
                source_hard_faces = int(np.count_nonzero(hard))
                prior_safety_reason = "source-hard-schema-missing"
                prior_reason = "prior-physical-safety-schema"
            # Apply only a geodesic/traversal bonus to the connected same-ID
            # component.  Different IDs are deliberately untouched and can
            # still be reached by the ordinary geometry resolver.
            neighbor_lengths = np.asarray(
                local.get("neighbor_lengths", np.ones(len(neighbors))),
                dtype=np.float64,
            ).reshape(-1)
            contour_cost = np.asarray(
                local.get("contour_cost", np.ones(len(neighbors))),
                dtype=np.float64,
            ).reshape(-1)
            if (
                len(source_hard) == count
                and len(neighbor_lengths) == len(neighbors)
                and len(contour_cost) == len(neighbors)
            ):
                adjusted_lengths = neighbor_lengths.copy()
                adjusted_contour = contour_cost.copy()
                for face in range(count):
                    if not trusted_same_set[face]:
                        continue
                    for edge_position in range(int(offsets[face]), int(offsets[face + 1])):
                        neighbor = int(neighbors[edge_position])
                        if trusted_same_set[neighbor]:
                            adjusted_lengths[edge_position] *= 0.5
                            adjusted_contour[edge_position] *= 0.5
                working_local = dict(local)
                working_local["neighbor_lengths"] = adjusted_lengths
                working_local["contour_cost"] = adjusted_contour
                working_local["partitions"] = {}
                prior_reason = (
                    "same-id-geodesic-discount"
                    if relaxed_pair_count
                    else "same-id-component-no-shared-pair"
                )
            else:
                prior_reason = "prior-cost-schema"
        else:
            prior_reason = "prior-graph-schema"
    elif prior_enabled:
        prior_reason = "prior-attribute-schema"
    partition = _fill_partition(working_local, strict_mode)
    partition["face_set_prior_applied_reason"] = prior_reason
    partition["face_set_relaxed_pair_count"] = int(relaxed_pair_count)
    partition["face_set_reached_same_set_faces"] = int(
        np.count_nonzero(trusted_same_set)
    )
    partition["face_set_different_id_neutral_crossing_count"] = int(
        neutral_crossing_count
    )
    partition["face_set_compact_cut_relaxed_faces"] = int(
        compact_cut_relaxed_faces
    )
    partition["face_set_source_hard_faces"] = int(source_hard_faces)
    partition["face_set_prior_safety_reason"] = prior_safety_reason
    core, owner = partition["core"], partition["owner"]
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
    first, second = working_local["first"], working_local["second"]
    relaxed = region.copy()
    editable = partition["band"] & ~partition["protected"] & (
        owner < working_local["count"]
    )
    weights = working_local["contour_cost"]
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


def _fill_preview_face_set_initial_component(
    geometry, distances, radius, seed_mesh_face, face_sets
):
    """Return the safe same-ID component from the unsliced preview graph.

    ``_fill_preview_local_geometry`` intentionally drops faces outside the
    current analysis slice.  Its compact degree is therefore not a physical
    manifold test.  Initial Face Set prior traversal uses this wider graph
    and its full physical-safety metadata before projecting back to the local
    candidate.
    """
    import numpy as np

    count = int(geometry.get("count", 0))
    face_ids = np.asarray(
        geometry.get("face_ids", np.arange(count, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    if count <= 0 or len(face_ids) != count:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=bool), "graph-schema"
    face_sets = np.asarray(face_sets, dtype=np.int32).reshape(-1)
    if (
        len(face_sets) == 0
        or np.any(face_ids < 0)
        or np.any(face_ids >= len(face_sets))
    ):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=bool), "face-set-schema"
    offsets = np.asarray(geometry.get("offsets", ()), dtype=np.int64).reshape(-1)
    neighbors = np.asarray(geometry.get("neighbors", ()), dtype=np.int32).reshape(-1)
    hidden = np.asarray(
        geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool
    ).reshape(-1)
    edge_counts = np.asarray(
        geometry.get("face_edge_counts", ()), dtype=np.int32
    ).reshape(-1)
    physical_degree = np.asarray(
        geometry.get("physical_degree", ()), dtype=np.int32
    ).reshape(-1)
    source_hard = np.asarray(
        geometry.get("source_hard", ()), dtype=bool
    ).reshape(-1)
    if len(source_hard) != count:
        if len(physical_degree) != count or len(edge_counts) != count:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=bool), "source-hard-schema"
        source_hard = hidden | (physical_degree < edge_counts)
    if len(hidden) != count or len(offsets) != count + 1 or int(offsets[-1]) != len(neighbors):
        return np.empty(0, dtype=np.int32), source_hard, "graph-schema"
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    if len(distances) != count:
        return np.empty(0, dtype=np.int32), source_hard, "distance-schema"
    seed_rows = np.flatnonzero(face_ids == int(seed_mesh_face))
    if len(seed_rows) != 1:
        return np.empty(0, dtype=np.int32), source_hard, "seed-schema"
    seed = int(seed_rows[0])
    values = face_sets[face_ids]
    target_id = int(face_sets[int(seed_mesh_face)])
    limit = float(radius) + max(float(radius) * 1.0e-8, 1.0e-9)
    eligible = (~source_hard) & np.isfinite(distances) & (distances <= limit)
    if not eligible[seed] or int(values[seed]) != target_id:
        return np.empty(0, dtype=np.int32), source_hard, "seed-not-eligible"
    reached = np.zeros(count, dtype=bool)
    reached[seed] = True
    pending = deque([seed])
    while pending:
        face = int(pending.popleft())
        for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
            neighbor = int(neighbor)
            if (
                not reached[neighbor]
                and eligible[neighbor]
                and int(values[neighbor]) == target_id
            ):
                reached[neighbor] = True
                pending.append(neighbor)
    return np.flatnonzero(reached).astype(np.int32), source_hard, "full-geometry-safe-component"


def _fill_preview_preserve_initial_ids(
    state, geometry, preview_local_ids, initial_phase, prior_enabled
):
    """Keep the accepted initial Face Set range as a monotonic floor.

    The cache is stored in stable mesh-face-id space rather than compact
    cursor-local indices.  Cursor halo growth can therefore reorder or add
    local rows without allowing a later wheel result to remove an accepted
    face.  This helper deliberately does not inspect Face Set values after
    the initial phase; it only projects the already accepted ids.
    """
    import numpy as np

    local_ids = np.unique(
        np.asarray(preview_local_ids, dtype=np.int32).reshape(-1)
    )
    count = int(geometry.get("count", 0))
    face_ids = np.asarray(
        geometry.get("face_ids", np.arange(count, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    hidden = np.asarray(
        geometry.get("hidden", np.zeros(count, dtype=bool)), dtype=bool
    ).reshape(-1)
    if len(face_ids) != count:
        face_ids = np.arange(count, dtype=np.int32)
    if len(hidden) != count:
        hidden = np.zeros(count, dtype=bool)
    local_ids = local_ids[(local_ids >= 0) & (local_ids < count)]
    local_ids = local_ids[~hidden[local_ids]]
    metrics = {
        "initial_base_count": 0,
        "preserved_count": 0,
        "prevented_removal_count": 0,
        "monotonic_reason": "no-initial-cache",
    }
    if bool(initial_phase) and bool(prior_enabled):
        stable_ids = np.unique(face_ids[local_ids]).astype(np.int32, copy=False)
        state["initial_accepted_ids"] = stable_ids.copy()
        state["initial_base_count"] = int(len(stable_ids))
        metrics.update(
            {
                "initial_base_count": int(len(stable_ids)),
                "preserved_count": int(len(stable_ids)),
                "monotonic_reason": "initial-cache-created",
            }
        )
        return local_ids.astype(np.int32, copy=False), metrics, local_ids

    cached = np.asarray(
        state.get("initial_accepted_ids", ()), dtype=np.int32
    ).reshape(-1)
    if len(cached) == 0:
        return local_ids.astype(np.int32, copy=False), metrics, np.empty(0, dtype=np.int32)
    cached_local = np.flatnonzero(
        np.isin(face_ids, np.unique(cached)) & ~hidden
    ).astype(np.int32)
    visible_component = np.asarray(
        state.get("visible_component", ()), dtype=bool
    ).reshape(-1)
    if len(visible_component) == count:
        cached_local = cached_local[visible_component[cached_local]]
    before = np.unique(local_ids)
    merged = np.unique(np.r_[before, cached_local]).astype(np.int32, copy=False)
    prevented = int(len(np.setdiff1d(cached_local, before, assume_unique=True)))
    metrics.update(
        {
            "initial_base_count": int(state.get("initial_base_count", len(cached))),
            "preserved_count": int(len(cached_local)),
            "prevented_removal_count": prevented,
            "monotonic_reason": "initial-cache-preserved",
        }
    )
    return merged, metrics, cached_local


def _fill_preview_multi_source_metric_shell(
    source_faces,
    selected,
    offsets,
    neighbors,
    distances,
    radius,
    hard,
    protected,
    max_shells=6,
):
    """Advance all safe boundary sources until a bounded cyan front exists."""
    import numpy as np

    selected = np.asarray(selected, dtype=bool).reshape(-1)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    hard = np.asarray(hard, dtype=bool).reshape(-1)
    protected = np.asarray(protected, dtype=bool).reshape(-1)
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
    neighbors = np.asarray(neighbors, dtype=np.int32).reshape(-1)
    count = len(selected)
    if (
        len(distances) != count
        or len(hard) != count
        or len(protected) != count
        or len(offsets) != count + 1
        or int(offsets[-1]) != len(neighbors)
    ):
        return set(), set(), 0, "shell-schema"
    limit = float(radius) + max(float(radius) * 1.0e-8, 1.0e-9)
    sources = {
        int(face)
        for face in source_faces
        if 0 <= int(face) < count
        and not selected[int(face)]
        and not hard[int(face)]
        and not protected[int(face)]
        and np.isfinite(distances[int(face)])
    }
    if not sources:
        return set(), set(), 0, "no-safe-boundary-source"
    front = {face for face in sources if distances[face] > limit}
    visited = set(sources)
    frontier = set(sources)
    shells = 0
    for shell_index in range(1, max(int(max_shells), 1) + 1):
        if front:
            break
        next_frontier = set()
        for face in sorted(frontier):
            for neighbor in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
                neighbor = int(neighbor)
                if (
                    neighbor < 0
                    or neighbor >= count
                    or neighbor in visited
                    or selected[neighbor]
                    or hard[neighbor]
                    or protected[neighbor]
                    or not np.isfinite(distances[neighbor])
                ):
                    continue
                visited.add(neighbor)
                next_frontier.add(neighbor)
                if distances[neighbor] > limit:
                    front.add(neighbor)
        shells = shell_index
        frontier = next_frontier
    return front, sources, shells, "cyan-front-created" if front else "no-safe-front"


def _fill_preview_analysis_shader_path():
    """Resolve the bundled matcap without creating a material datablock."""
    profile = _FILL_PREVIEW_ANALYSIS_SHADER_PROFILE
    try:
        import bpy
        roots = (
            bpy.utils.system_resource("DATAFILES"),
            bpy.utils.resource_path("LOCAL"),
            bpy.utils.resource_path("SYSTEM"),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        roots = ()
    for root in roots:
        for filename in profile["matcap_candidates"]:
            path = os.path.join(root, "studiolights", "matcap", filename)
            if os.path.isfile(path):
                return filename, path, True
    return "", "", False


def _fill_preview_shadow_screen_roi_radius(context):
    """Return the fixed 320px experimental ROI radius.

    The 3.2.85 experiment intentionally decouples the image context from the
    sculpt brush size.  The crop is clipped only by the View3D edges, giving a
    roughly 640x640 window (or smaller near a viewport edge) for repeatable
    screen-space evaluation.
    """
    del context
    return 320


def _fill_preview_capture_shadow_signal(context, seed_screen=None):
    """Capture a temporary, overlay-free viewport luminance line map.

    This is deliberately best-effort.  Ordinary ``E`` may use the capture
    only when the current View3D can render it; a failed capture is reported
    to the preview as an explicit seed-only provisional.  The
    strict Ctrl+E path never calls this function.
    """
    import numpy as np

    result = {
        "ok": False,
        "reason": "capture-not-attempted",
        "restore_verified": False,
        "width": 0,
        "height": 0,
        "luminance": np.empty((0, 0), dtype=np.float32),
        "depth": np.empty((0, 0), dtype=np.float32),
        "depth_available": False,
        "line_strength": np.empty((0, 0), dtype=np.float32),
        "line_map": np.empty((0, 0), dtype=bool),
        "view_projection": None,
        "luminance_min": 0.0,
        "luminance_max": 0.0,
        "luminance_seed": 0.0,
        "seed_screen": (0.0, 0.0),
        "noise_scale": 0.0,
        "line_pixel_count": 0,
        "capture_size": (0, 0),
        "analysis_shader_name": _FILL_PREVIEW_ANALYSIS_SHADER_PROFILE["name"],
        "analysis_shader_profile_registered": False,
        "analysis_shader_matcap": "",
        "faceset_suppressed": False,
        "mask_suppressed": False,
        "shading_restore_verified": False,
        "overlay_restore_verified": False,
        "analysis_state_snapshot_count": 0,
    }
    offscreen = None
    overlay = getattr(context.space_data, "overlay", None)
    shading = getattr(context.space_data, "shading", None)
    saved = {"overlay": {}, "shading": {}}
    profile_errors = []
    try:
        region = context.region
        space = context.space_data
        region_3d = getattr(space, "region_3d", None)
        width = int(getattr(region, "width", 0))
        height = int(getattr(region, "height", 0))
        if region_3d is None or width < 8 or height < 8:
            result["reason"] = "invalid-view3d-region"
            return result
        result["width"], result["height"] = width, height
        result["capture_size"] = (width, height)
        overlay_properties = (
            "show_overlays", "show_sculpt_face_sets",
            "sculpt_mode_face_sets_opacity", "show_sculpt_mask",
            "sculpt_mode_mask_opacity",
        )
        shading_properties = (
            "type", "light", "studio_light", "use_world_space_lighting",
            "use_studiolight_view_rotation", "studiolight_rotate_z", "intensity",
            "color_type", "single_color", "show_shadows", "shadow_intensity",
            "show_specular_highlight", "show_cavity", "cavity_type",
            "curvature_ridge_factor", "curvature_valley_factor", "background_type",
            "background_color", "background_strength", "show_xray",
            "show_backface_culling",
        )
        for owner, properties in ((overlay, overlay_properties), (shading, shading_properties)):
            if owner is None:
                continue
            target = saved["overlay" if owner is overlay else "shading"]
            for prop_name in properties:
                if not hasattr(owner, prop_name):
                    continue
                try:
                    target[prop_name] = copy.deepcopy(getattr(owner, prop_name))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    continue
        result["analysis_state_snapshot_count"] = int(
            len(saved["overlay"]) + len(saved["shading"])
        )
        matcap_name, matcap_path, profile_registered = _fill_preview_analysis_shader_path()
        result["analysis_shader_matcap"] = matcap_name
        result["analysis_shader_profile_registered"] = bool(profile_registered)
        if not profile_registered:
            result["reason"] = "analysis-shader-not-found"
            return result
        def apply_property(owner, prop_name, value):
            if owner is None or not hasattr(owner, prop_name):
                return False
            try:
                setattr(owner, prop_name, value)
                return True
            except (AttributeError, RuntimeError, TypeError, ValueError):
                profile_errors.append(prop_name)
                return False
        # Hide all user-facing color overlays and masks during the temporary
        # draw.  The values are restored in finally, including enum choices
        # which may differ between Blender point releases.
        overlay_ok = True
        if overlay is not None:
            overlay_ok &= apply_property(overlay, "show_overlays", False)
            overlay_ok &= apply_property(overlay, "show_sculpt_face_sets", False)
            overlay_ok &= apply_property(overlay, "sculpt_mode_face_sets_opacity", 0.0)
            overlay_ok &= apply_property(overlay, "show_sculpt_mask", False)
            overlay_ok &= apply_property(overlay, "sculpt_mode_mask_opacity", 0.0)
        result["faceset_suppressed"] = bool(overlay_ok)
        result["mask_suppressed"] = bool(overlay_ok)
        profile_ok = True
        if shading is not None:
            profile_ok &= apply_property(shading, "type", "SOLID")
            profile_ok &= apply_property(shading, "light", "MATCAP")
            profile_ok &= apply_property(shading, "studio_light", matcap_name)
            profile_ok &= apply_property(shading, "color_type", "SINGLE")
            profile_ok &= apply_property(shading, "single_color", (0.72, 0.72, 0.72))
            profile_ok &= apply_property(shading, "show_specular_highlight", False)
            profile_ok &= apply_property(shading, "show_shadows", False)
            profile_ok &= apply_property(shading, "show_cavity", False)
        if profile_errors or not profile_ok or not overlay_ok:
            result["reason"] = "analysis-profile-apply-failed"
            return result
        from gpu.types import GPUOffScreen
        offscreen = GPUOffScreen(width, height)
        view_matrix = region_3d.view_matrix.copy()
        projection_matrix = region_3d.window_matrix.copy()
        offscreen.draw_view3d(
            context.scene,
            context.view_layer,
            space,
            region,
            view_matrix,
            projection_matrix,
            do_color_management=False,
            draw_background=True,
        )
        offscreen.bind()
        try:
            color_buffer = offscreen.read_color(0, 0, width, height, 4, 0, "UBYTE")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            color_buffer = gpu.state.active_framebuffer_get().read_color(
                0, 0, width, height, 4, 0, "UBYTE"
            )
        try:
            raw_color = np.frombuffer(bytes(color_buffer), dtype=np.uint8)
        except (TypeError, ValueError):
            raw_color = np.asarray(color_buffer.to_list(), dtype=np.uint8)
        expected = width * height * 4
        if raw_color.size < expected:
            raise RuntimeError("color-readback-short")
        rgba = raw_color[:expected].reshape((height, width, 4)).astype(np.float32) / 255.0
        depth = np.empty((0, 0), dtype=np.float32)
        try:
            depth_buffer = offscreen.read_depth(0, 0, width, height)
            raw_depth = np.frombuffer(bytes(depth_buffer), dtype=np.float32)
            if raw_depth.size >= width * height:
                depth = raw_depth[: width * height].reshape((height, width))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            depth = np.empty((0, 0), dtype=np.float32)
        luminance = (
            0.2126 * rgba[..., 0]
            + 0.7152 * rgba[..., 1]
            + 0.0722 * rgba[..., 2]
        )
        # Ordinary E performs denoise, gradient hysteresis, and Closing only
        # inside the bounded cursor ROI.  Keep these legacy fields empty so a
        # future caller cannot accidentally interpret a full-viewport edge
        # map as the normal classifier's input.
        line_strength = np.empty((0, 0), dtype=np.float32)
        line_map = np.empty((0, 0), dtype=bool)
        noise = 0.0
        seed_x, seed_y = width // 2, height // 2
        if seed_screen is not None:
            try:
                seed_x = int(round(float(seed_screen[0])))
                seed_y = int(round(float(seed_screen[1])))
            except (IndexError, TypeError, ValueError):
                try:
                    seed_x = int(round(float(seed_screen.x)))
                    seed_y = int(round(float(seed_screen.y)))
                except (AttributeError, TypeError, ValueError):
                    seed_x, seed_y = width // 2, height // 2
        seed_x = max(0, min(width - 1, seed_x))
        seed_y = max(0, min(height - 1, seed_y))
        result.update(
            {
                "ok": True,
                "reason": "ok",
                "luminance": luminance.astype(np.float32, copy=False),
                "depth": depth,
                "depth_available": bool(depth.shape == (height, width)),
                "line_strength": line_strength,
                "line_map": line_map,
                "view_projection": np.asarray(
                    region_3d.perspective_matrix, dtype=np.float64
                ),
                "luminance_min": float(np.min(luminance)),
                "luminance_max": float(np.max(luminance)),
                "luminance_seed": float(luminance[seed_y, seed_x]),
                "seed_screen": (int(seed_x), int(seed_y)),
                # Full-viewport gradient noise is intentionally not computed
                # in the ROI design.  ``noise`` is the remaining capture
                # noise estimate; the ROI helper records its own gradient
                # MAD, so no stale ``gradient_noise`` local belongs here.
                "noise_scale": float(noise),
                "line_pixel_count": 0,
            }
        )
    except (AttributeError, ImportError, MemoryError, RuntimeError, TypeError, ValueError):
        result["reason"] = "offscreen-readback-failed"
    finally:
        try:
            if offscreen is not None:
                offscreen.unbind()
        except (AttributeError, RuntimeError):
            pass
        try:
            if offscreen is not None:
                offscreen.free()
        except (AttributeError, RuntimeError):
            pass
        overlay_restore_ok = True
        shading_restore_ok = True
        for owner, key in ((overlay, "overlay"), (shading, "shading")):
            values = saved.get(key, {})
            if owner is None:
                continue
            for prop_name, value in values.items():
                try:
                    setattr(owner, prop_name, value)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    if key == "overlay":
                        overlay_restore_ok = False
                    else:
                        shading_restore_ok = False
        result["overlay_restore_verified"] = bool(overlay_restore_ok)
        result["shading_restore_verified"] = bool(shading_restore_ok)
        result["restore_verified"] = bool(overlay_restore_ok and shading_restore_ok)
        if result["ok"] and not result["restore_verified"]:
            result["ok"] = False
            result["reason"] = "viewport-state-restore-failed"
    return result


def _fill_preview_visible_analysis_profile(context):
    """Apply the manually toggled analysis look to the visible View3D.

    This helper is intentionally independent from normal E.  It is a viewing
    utility only: toon_dark is shown together with semi-transparent Face Set
    colors and mesh wire overlay.  No material/datablock is touched; the exact
    SpaceView3D values are retained in the returned token and restoration is
    idempotent.
    """
    space = getattr(context, "space_data", None)
    overlay = getattr(space, "overlay", None)
    shading = getattr(space, "shading", None)
    token = {
        "active": False,
        "applied": False,
        "restore_verified": False,
        "space": space,
        "overlay": overlay,
        "shading": shading,
        "saved": {"overlay": {}, "shading": {}},
        "analysis_shader_name": _FILL_PREVIEW_ANALYSIS_SHADER_PROFILE["name"],
        "analysis_shader_matcap": "",
        "reason": "not-attempted",
    }
    if space is None or (overlay is None and shading is None):
        token["reason"] = "view3d-state-unavailable"
        return token
    matcap_name, _matcap_path, profile_registered = _fill_preview_analysis_shader_path()
    token["analysis_shader_matcap"] = matcap_name
    if not profile_registered:
        token["reason"] = "analysis-shader-not-found"
        return token
    overlay_properties = (
        "show_overlays", "show_sculpt_face_sets",
        "sculpt_mode_face_sets_opacity", "show_sculpt_mask",
        "sculpt_mode_mask_opacity",
        "show_wireframes", "wireframe_threshold",
    )
    shading_properties = (
        "type", "light", "studio_light", "use_world_space_lighting",
        "use_studiolight_view_rotation", "studiolight_rotate_z", "intensity",
        "color_type", "single_color", "show_shadows", "shadow_intensity",
        "show_specular_highlight", "show_cavity", "cavity_type",
        "curvature_ridge_factor", "curvature_valley_factor", "background_type",
        "background_color", "background_strength", "show_xray",
        "show_backface_culling",
    )
    for owner, properties, key in (
        (overlay, overlay_properties, "overlay"),
        (shading, shading_properties, "shading"),
    ):
        if owner is None:
            continue
        for prop_name in properties:
            if not hasattr(owner, prop_name):
                continue
            try:
                token["saved"][key][prop_name] = copy.deepcopy(getattr(owner, prop_name))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
    errors = []
    def apply(owner, name, value):
        if owner is None or not hasattr(owner, name):
            return True
        try:
            setattr(owner, name, value)
            return True
        except (AttributeError, RuntimeError, TypeError, ValueError):
            errors.append(name)
            return False
    # Keep the overlay pass visible for the custom orange/cyan draw handler.
    # Manual analysis deliberately retains Face Set colors at a readable,
    # semi-transparent opacity and enables wireframes when Blender exposes
    # those overlay properties.
    if overlay is not None:
        apply(overlay, "show_overlays", True)
        apply(overlay, "show_sculpt_face_sets", True)
        apply(overlay, "sculpt_mode_face_sets_opacity", 0.45)
        apply(overlay, "show_sculpt_mask", False)
        apply(overlay, "sculpt_mode_mask_opacity", 0.0)
        apply(overlay, "show_wireframes", True)
        apply(overlay, "wireframe_threshold", 0.5)
    if shading is not None:
        apply(shading, "type", "SOLID")
        apply(shading, "light", "MATCAP")
        apply(shading, "studio_light", matcap_name)
        apply(shading, "color_type", "SINGLE")
        apply(shading, "single_color", (0.72, 0.72, 0.72))
        apply(shading, "show_specular_highlight", False)
        apply(shading, "show_shadows", False)
        apply(shading, "show_cavity", False)
    if errors:
        token["reason"] = "analysis-profile-apply-failed"
        _fill_preview_restore_visible_analysis_profile(token)
        return token
    token["active"] = True
    token["applied"] = True
    token["reason"] = "visible-analysis-profile-active"
    try:
        _tag_redraw(getattr(context, "area", None))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return token


def _fill_preview_restore_visible_analysis_profile(token):
    """Restore a visible analysis profile exactly once and report success."""
    if not isinstance(token, dict):
        return False
    if token.get("restore_verified"):
        return True
    saved = token.get("saved", {})
    ok = True
    for owner, key in (
        (token.get("overlay"), "overlay"),
        (token.get("shading"), "shading"),
    ):
        if owner is None:
            continue
        for prop_name, value in saved.get(key, {}).items():
            try:
                setattr(owner, prop_name, value)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                ok = False
    token["active"] = False
    token["restore_verified"] = bool(ok)
    if not ok:
        token["reason"] = "visible-analysis-restore-failed"
    return bool(ok)


def _fill_preview_shadow_view_key(context):
    """Return the current View3D identity without retaining a dead area."""
    try:
        space = getattr(context, "space_data", None)
        return int(space.as_pointer()) if space is not None else 0
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return 0


def _fill_preview_restore_manual_analysis_profiles():
    """Restore and clear all manually toggled View3D analysis profiles."""
    tokens = list(_runtime.shadow_analysis_view_tokens.values())
    _runtime.shadow_analysis_view_tokens.clear()
    for token in tokens:
        try:
            _fill_preview_restore_visible_analysis_profile(token)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            if isinstance(token, dict):
                token["restore_verified"] = False
