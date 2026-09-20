"""Tube Shape component.

Loaded by the package entry point in dependency order. This module owns the
Tube Shape operators and imports shared runtime state explicitly.
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
from .foundation import _fill_preview_shader_get, _next_session_id, _tag_redraw
from .smart_fill import geometry as _smart_fill_geometry
from .guided_ridge.core import _raycast_sculpt_face_set
from .viewport import sculpt_cursor_region_coordinate as _sculpt_cursor_region_coordinate

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
# Curved Face Set tube shaping
# ---------------------------------------------------------------------------

def _tube_unit(value):
    import numpy as np
    value = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(value))
    return value / max(length, 1.0e-12)


def _tube_smoothstep(value):
    import numpy as np
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _tube_frames(points):
    """Parallel-transport frames along a sampled, curved centerline."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("centerline section count is insufficient")
    tangents = np.gradient(points, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1.0e-12)
    e1 = np.empty_like(points)
    e2 = np.empty_like(points)
    seed = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(seed, tangents[0]))) > 0.9:
        seed = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    e1[0] = _tube_unit(seed - tangents[0] * np.dot(seed, tangents[0]))
    e2[0] = _tube_unit(np.cross(tangents[0], e1[0]))
    for index in range(1, len(points)):
        candidate = e1[index - 1] - tangents[index] * np.dot(
            e1[index - 1], tangents[index]
        )
        if float(np.linalg.norm(candidate)) < 1.0e-7:
            candidate = e2[index - 1] - tangents[index] * np.dot(
                e2[index - 1], tangents[index]
            )
        e1[index] = _tube_unit(candidate)
        e2[index] = _tube_unit(np.cross(tangents[index], e1[index]))
    return tangents, e1, e2


def _tube_symmetric_sqrt(matrix, inverse=False):
    """Return a sign-stable symmetric square root of a 2D covariance."""
    import numpy as np

    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    floor = max(float(np.max(values)) * 1.0e-8, 1.0e-14)
    if len(values) != 2 or float(np.max(values)) <= 1.0e-12 or float(np.min(values)) <= floor:
        raise ValueError("断面輪郭の共分散が退化しています")
    factors = 1.0 / np.sqrt(values) if inverse else np.sqrt(values)
    return (vectors * factors[None, :]) @ vectors.T


def _tube_topology(faces, face_sets, seed_face):
    """Build a compact edge incidence view and the seed Face Set component.

    The edge table is made with NumPy sorting.  Python adjacency is created
    only for pairs whose Face Set equals the clicked seed, so a large unrelated
    mesh does not become a long-lived Python graph.
    """
    import numpy as np
    from collections import defaultdict, deque

    faces = np.asarray(faces, dtype=np.int32).reshape((-1, 3))
    face_sets = np.asarray(face_sets, dtype=np.int32).reshape(-1)
    if len(faces) == 0 or len(face_sets) != len(faces):
        raise ValueError("mesh faces or Face Set data is unavailable")
    seed_face = int(seed_face)
    if seed_face < 0 or seed_face >= len(faces):
        raise ValueError("seed face is outside the mesh")
    target_id = int(face_sets[seed_face])
    edge_rows = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edge_rows.sort(axis=1)
    # edge_rows is concatenated as three face-wide blocks, so the matching
    # face ids must use the same order (a block-wise tile, not face-wise
    # repeat).  Keeping this explicit prevents a subtle boundary corruption on
    # every nontrivial triangle mesh.
    edge_face_ids = np.tile(np.arange(len(faces), dtype=np.int32), 3)
    unique_edges, inverse, counts = np.unique(
        edge_rows, axis=0, return_inverse=True, return_counts=True
    )
    edge_order = np.argsort(inverse, kind="stable")
    edge_starts = np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)]
    pair_edge_ids = np.flatnonzero(counts == 2)
    pair_refs = edge_order[np.repeat(edge_starts[pair_edge_ids], 2) + np.tile((0, 1), len(pair_edge_ids))]
    pair_faces = edge_face_ids[pair_refs].reshape((-1, 2))
    pair_faces.sort(axis=1)

    same_id = (
        (face_sets[pair_faces[:, 0]] == target_id)
        & (face_sets[pair_faces[:, 1]] == target_id)
    )
    same_pairs = pair_faces[same_id]
    compact_neighbors = defaultdict(list)
    for first, second in same_pairs:
        first, second = int(first), int(second)
        compact_neighbors[first].append(second)
        compact_neighbors[second].append(first)
    target_faces = {seed_face}
    queue = deque((seed_face,))
    while queue:
        face = queue.popleft()
        for neighbor in compact_neighbors.get(face, ()):
            if neighbor not in target_faces:
                target_faces.add(neighbor)
                queue.append(neighbor)
    target_faces = np.asarray(sorted(target_faces), dtype=np.int32)
    target_face_mask = np.zeros(len(faces), dtype=bool)
    target_face_mask[target_faces] = True
    edge_faces = edge_face_ids[edge_order]

    # Record closure from the target component's actual edge incidence.  A
    # closed terminal cap may contain a small ring or another multi-vertex
    # triangulated patch; its last frontier size is not a topological test.
    target_component_edge_counts = np.add.reduceat(
        target_face_mask[edge_faces].astype(np.int8), edge_starts
    )
    target_component_edge_ids = np.flatnonzero(target_component_edge_counts > 0)
    target_component_boundary_edge_count = int(
        np.count_nonzero(target_component_edge_counts == 1)
    )
    target_component_edge_count = int(len(target_component_edge_ids))
    target_component_vertex_count = int(len(np.unique(faces[target_faces])))
    target_component_euler = int(
        target_component_vertex_count - target_component_edge_count + len(target_faces)
    )

    external_edges = []
    open_edges = []
    nonmanifold_edges = []
    boundary_edge_ids = []
    # Only edges touching a face with the seed ID can belong to the selected
    # component or its external boundary.  This keeps the Python loop compact
    # even when the scene contains a large unrelated mesh.
    target_id_edge_contact = np.add.reduceat(
        (face_sets[edge_faces] == target_id).astype(np.int8), edge_starts
    ) > 0
    candidate_edge_ids = np.flatnonzero(target_id_edge_contact)
    for edge_id in candidate_edge_ids:
        start, incidence = edge_starts[int(edge_id)], counts[int(edge_id)]
        refs = edge_faces[int(start):int(start + incidence)]
        target_refs = refs[target_face_mask[refs]]
        if not len(target_refs):
            continue
        if int(incidence) > 2:
            nonmanifold_edges.append(tuple(int(v) for v in unique_edges[edge_id]))
            continue
        if int(incidence) == 1:
            open_edges.append(tuple(int(v) for v in unique_edges[edge_id]))
            continue
        if len(target_refs) == 1:
            other = refs[~target_face_mask[refs]]
            if len(other) and any(int(face_sets[int(value)]) != target_id for value in other):
                edge = tuple(int(v) for v in unique_edges[edge_id])
                external_edges.append(edge)
                boundary_edge_ids.append(edge_id)

    boundary_graph = defaultdict(set)
    for first, second in external_edges:
        boundary_graph[first].add(second)
        boundary_graph[second].add(first)
    components = []
    unseen = set(boundary_graph)
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        component = {root}
        pending = [root]
        while pending:
            vertex = pending.pop()
            for neighbor in boundary_graph[vertex]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int32))
    components.sort(key=lambda values: int(values[0]) if len(values) else -1)
    boundary_branch_vertices = sorted(
        int(vertex) for vertex, links in boundary_graph.items() if len(links) != 2
    )
    return {
        "faces": faces,
        "face_sets": face_sets,
        "seed_face": seed_face,
        "target_id": target_id,
        "target_faces": target_faces,
        "target_vertices": np.unique(faces[target_faces]),
        "target_component_edge_count": target_component_edge_count,
        "target_component_boundary_edge_count": target_component_boundary_edge_count,
        "target_component_euler": target_component_euler,
        "unique_edges": unique_edges,
        "edge_faces": edge_faces,
        "edge_starts": edge_starts,
        "edge_counts": counts,
        "pair_faces": pair_faces,
        "same_pairs": same_pairs,
        "external_edges": tuple(external_edges),
        "boundary_edge_ids": np.asarray(boundary_edge_ids, dtype=np.int32),
        "boundary_components": tuple(components),
        "open_edges": tuple(open_edges),
        "nonmanifold_edges": tuple(nonmanifold_edges),
        "boundary_branch_vertices": tuple(boundary_branch_vertices),
    }


def _tube_vertex_graph(points, faces):
    """Return target-only weighted vertex adjacency for Dijkstra and ring walk."""
    import numpy as np
    from collections import defaultdict

    points = np.asarray(points, dtype=np.float64)
    graph = defaultdict(dict)
    for face in np.asarray(faces, dtype=np.int32).reshape((-1, 3)):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            first, second = int(first), int(second)
            weight = float(np.linalg.norm(points[first] - points[second]))
            if weight <= 1.0e-12:
                continue
            old = graph[first].get(second, float("inf"))
            if weight < old:
                graph[first][second] = weight
                graph[second][first] = weight
    return graph


def _tube_dijkstra(graph, source_vertices, target_vertices):
    import heapq
    import math

    target = {int(value) for value in target_vertices}
    distances = {vertex: math.inf for vertex in target}
    heap = []
    for value in source_vertices:
        vertex = int(value)
        if vertex in distances:
            distances[vertex] = 0.0
            heapq.heappush(heap, (0.0, vertex))
    while heap:
        distance, vertex = heapq.heappop(heap)
        if distance > distances[vertex] + 1.0e-12:
            continue
        for neighbor, weight in graph.get(vertex, {}).items():
            if neighbor not in distances:
                continue
            candidate = distance + float(weight)
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return distances


def _tube_centerline(points, target_vertices, root_vertices, graph):
    """Infer local sections by walking the actual target mesh connectivity."""
    import numpy as np

    target_vertices = np.asarray(sorted(int(value) for value in target_vertices), dtype=np.int32)
    distances = _tube_dijkstra(graph, root_vertices, target_vertices)
    values = np.asarray([distances[int(value)] for value in target_vertices], dtype=np.float64)
    finite = np.isfinite(values)
    ids = target_vertices[finite]
    values = values[finite]
    if len(ids) < 20:
        raise ValueError("断面不足: reachable target vertices < 20")
    extent = float(np.max(values))
    all_lengths = np.asarray(
        [weight for links in graph.values() for weight in links.values()], dtype=np.float64
    )
    longitudinal_edge = float(np.percentile(all_lengths, 20.0)) if len(all_lengths) else 0.0
    if extent < 1.0e-7 or longitudinal_edge < 1.0e-8:
        raise ValueError("長手方向が不足")
    distance_by_id = {int(vertex): float(distance) for vertex, distance in zip(ids, values)}
    current = {int(vertex) for vertex in root_vertices if int(vertex) in distance_by_id}
    if len(current) < 3:
        current = set(ids[np.argsort(values)[:max(3, min(24, len(ids)))]].tolist())
    groups = []
    levels = []
    seen = set()
    # Do not use a fixed section count as a hidden truncation limit.  The walk
    # is bounded by the compact target vertex count and must either reach the
    # actual terminal ring or return a visible rejection reason.
    max_walk_steps = len(ids) + 2
    walk_steps = 0
    while walk_steps < max_walk_steps:
        walk_steps += 1
        current -= seen
        if len(current) < 3 and len(current) != 1:
            break
        if not current:
            break
        seen.update(current)
        groups.append(np.asarray(sorted(current), dtype=np.int32))
        levels.append(float(np.median([distance_by_id[int(value)] for value in current])))
        candidates = sorted(
            {
                int(neighbor)
                for vertex in current
                for neighbor in graph.get(int(vertex), {})
                if int(neighbor) not in seen
            },
            key=lambda value: distance_by_id.get(int(value), float("inf")),
        )
        if not candidates:
            break
        candidate_distances = np.asarray(
            [distance_by_id[int(value)] for value in candidates], dtype=np.float64
        )
        # Keep every unseen forward neighbor.  A largest-gap crop can strand
        # inner-side vertices on a long bend; local ring statistics below are
        # robust to that small overlap and the full target graph remains
        # represented.  Sections with insufficient or degenerate contours are
        # rejected by the profile guards after the complete walk.
        if len(candidates) < 3 and len(candidates) != 1:
            break
        current = set(candidates)
        # Do not stop on a distance threshold.  The terminal boundary/apex is
        # reached by exhausting the compact target graph below, which keeps
        # long tubes from being silently truncated and then extrapolated.
    if walk_steps >= max_walk_steps and len(seen) < len(ids):
        raise ValueError("長手方向の末端へ到達できません")
    if len(seen) < len(ids):
        raise ValueError("target頂点の一部を断面推定から除外しました")
    if len(groups) < 4:
        raise ValueError("断面不足: adjacency ring walk < 4")
    levels = np.asarray(levels, dtype=np.float64) / extent
    levels[0] = 0.0
    if levels[-1] >= 1.0 - 1.5 * longitudinal_edge / max(extent, 1.0e-12):
        levels[-1] = 1.0
    levels = np.maximum.accumulate(levels)
    for index in range(1, len(levels)):
        if levels[index] <= levels[index - 1]:
            levels[index] = min(1.0, levels[index - 1] + 1.0e-5)
    centers = np.asarray([np.mean(points[group], axis=0) for group in groups], dtype=np.float64)
    if len(centers) >= 5:
        smoothed = centers.copy()
        smoothed[2:-2] = (
            centers[:-4] + 4.0 * centers[1:-3] + 6.0 * centers[2:-2]
            + 4.0 * centers[3:-1] + centers[4:]
        ) / 16.0
        centers = smoothed
    tangents, e1, e2 = _tube_frames(centers)
    section_by_vertex = {
        int(vertex): int(section)
        for section, group in enumerate(groups)
        for vertex in group
    }
    section_ids = np.asarray(
        [section_by_vertex[int(vertex)] for vertex in ids], dtype=np.int32
    )
    return {
        "ids": ids,
        "groups": tuple(groups),
        "levels": levels,
        "centers": centers,
        "tangents": tangents,
        "e1": e1,
        "e2": e2,
        "section_ids": section_ids,
        "distances": values,
        "distance_by_id": distance_by_id,
        "extent": extent,
    }


def _tube_interp_rows(rows, samples, levels):
    import numpy as np
    rows = np.asarray(rows, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    return np.column_stack(
        [np.interp(samples, levels, rows[:, axis]) for axis in range(rows.shape[1])]
    )


def _tube_infer_geometry(points, topology):
    """Compute local centers, frames, radii, and fixed boundary vertices."""
    import numpy as np

    target_faces = topology["target_faces"]
    target_vertices = topology["target_vertices"]
    boundary_components = topology["boundary_components"]
    if not boundary_components:
        raise ValueError("外部 Face Set 境界がありません")
    graph = _tube_vertex_graph(points, topology["faces"][target_faces])
    centerline = _tube_centerline(points, target_vertices, boundary_components[0], graph)
    ids = centerline["ids"]
    u = np.asarray(
        [centerline["distance_by_id"][int(value)] for value in ids], dtype=np.float64
    ) / max(float(centerline["extent"]), 1.0e-12)
    levels = centerline["levels"]
    centers = _tube_interp_rows(centerline["centers"], u, levels)
    tangents = _tube_interp_rows(centerline["tangents"], u, levels)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1.0e-12)
    e1 = _tube_interp_rows(centerline["e1"], u, levels)
    e1 -= tangents * np.sum(e1 * tangents, axis=1, keepdims=True)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1.0e-12)
    e2 = np.cross(tangents, e1)
    offset = points[ids] - centers
    axial = np.sum(offset * tangents, axis=1)
    radial = offset - axial[:, None] * tangents
    radii = np.linalg.norm(radial, axis=1)
    section_radii = []
    circularity = []
    section_ids = np.asarray(centerline["section_ids"], dtype=np.int32)
    for group in centerline["groups"]:
        local = np.flatnonzero(np.isin(ids, group))
        section_radii.append(float(np.median(radii[local])) if len(local) else 0.0)
        if len(local) >= 8:
            plane = np.column_stack(
                (np.sum(radial[local] * e1[local], axis=1),
                 np.sum(radial[local] * e2[local], axis=1))
            )
            covariance = np.cov(plane.T)
            eigenvalues = np.linalg.eigvalsh(covariance)
            circularity.append(float(eigenvalues[-2] / max(eigenvalues[-1], 1.0e-12)))
        else:
            circularity.append(0.0)
    boundary_ids = np.unique(np.concatenate(boundary_components)).astype(np.int32)
    tip_ids = np.asarray(centerline["groups"][-1], dtype=np.int32)
    root_ids = np.asarray(boundary_components[0], dtype=np.int32)
    return {
        **centerline,
        "target_faces": np.asarray(topology["faces"][target_faces], dtype=np.int32),
        "ids": ids,
        "vertex_u": u,
        "vertex_centers": centers,
        "vertex_tangents": tangents,
        "vertex_e1": e1,
        "vertex_e2": e2,
        "radial": radial,
        "vertex_radii": radii,
        "section_radii": np.asarray(section_radii, dtype=np.float64),
        "circularity": np.asarray(circularity, dtype=np.float64),
        "section_ids": section_ids.astype(np.int32),
        "boundary_vertex_ids": boundary_ids,
        "root_vertex_ids": root_ids,
        "tip_vertex_ids": tip_ids,
        "seed_vertex_ids": np.unique(topology["faces"][topology["seed_face"]]).astype(np.int32),
    }


def _tube_reference_profile(points, geometry, seed_hit):
    """Extract one actual, star-shaped cross-section profile near the cursor.

    The profile is represented as periodic angle/radius samples in the
    transported local frame.  It intentionally keeps ellipse axes and modest
    radial irregularity instead of reducing the section to one radius.
    """
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    ids = np.asarray(geometry["ids"], dtype=np.int32)
    groups = tuple(np.asarray(group, dtype=np.int32) for group in geometry["groups"])
    seed_ids = np.asarray(geometry.get("seed_vertex_ids", ()), dtype=np.int32)
    centers = np.asarray(geometry["centers"], dtype=np.float64)
    tangents = np.asarray(geometry["tangents"], dtype=np.float64)
    e1 = np.asarray(geometry["e1"], dtype=np.float64)
    e2 = np.asarray(geometry["e2"], dtype=np.float64)
    if len(points) == 0 or len(ids) == 0 or len(groups) < 2:
        raise ValueError("カーソル参照断面が不足しています")
    seed_hit = np.asarray(seed_hit, dtype=np.float64)
    candidate_sections = []
    if len(seed_ids):
        for section, group in enumerate(groups):
            if np.intersect1d(group, seed_ids).size:
                candidate_sections.append(section)
    if not candidate_sections:
        candidate_sections = list(range(len(groups)))
    reference_section = min(
        candidate_sections,
        key=lambda section: float(np.linalg.norm(centers[section] - seed_hit)),
    )
    ref_ids = np.asarray(groups[reference_section], dtype=np.int32)
    local = np.flatnonzero(np.isin(ids, ref_ids))
    if len(local) < 8:
        raise ValueError("カーソル参照断面の輪郭点が不足しています")
    vertex_ids = ids[local]
    relative = points[vertex_ids] - centers[reference_section]
    axial = np.sum(relative * tangents[reference_section], axis=1, keepdims=True)
    radial = relative - axial * tangents[reference_section]
    q = np.column_stack((
        np.sum(radial * e1[reference_section], axis=1),
        np.sum(radial * e2[reference_section], axis=1),
    ))
    if not np.all(np.isfinite(q)):
        raise ValueError("カーソル参照断面に有限でない輪郭点があります")
    # Center the sampled profile in the transported section plane.  The
    # centerline itself remains the source of the longitudinal center motion.
    q -= np.mean(q, axis=0, keepdims=True)
    radii = np.linalg.norm(q, axis=1)
    if float(np.max(radii)) <= 1.0e-8 or int(np.count_nonzero(radii > 1.0e-8)) < 8:
        raise ValueError("カーソル参照断面の輪郭が退化しています")
    theta = np.mod(np.arctan2(q[:, 1], q[:, 0]), 2.0 * np.pi)
    order = np.argsort(theta, kind="stable")
    theta = theta[order]
    radii = radii[order]
    if len(theta) >= 3:
        gaps = np.diff(np.r_[theta, theta[0] + 2.0 * np.pi])
        if float(np.max(gaps)) > np.pi:
            raise ValueError("カーソル参照断面の輪郭角度が不足しています")
    median_radius = float(np.median(radii))
    if median_radius <= 1.0e-8:
        raise ValueError("カーソル参照断面の半径が退化しています")
    covariance = np.cov(q.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if float(eigenvalues[-1]) <= 1.0e-12 or float(eigenvalues[0]) <= max(
        float(eigenvalues[-1]) * 1.0e-8, 1.0e-14
    ):
        raise ValueError("カーソル参照断面が線状に退化しています")
    inverse_sqrt = _tube_symmetric_sqrt(covariance, inverse=True)
    square_root = _tube_symmetric_sqrt(covariance)
    whitened = q @ inverse_sqrt.T
    whitened_theta = np.mod(np.arctan2(whitened[:, 1], whitened[:, 0]), 2.0 * np.pi)
    whitened_order = np.argsort(whitened_theta, kind="stable")
    whitened_theta = whitened_theta[whitened_order]
    whitened_radii = np.linalg.norm(whitened[whitened_order], axis=1)
    return {
        "section": int(reference_section),
        "vertex_ids": np.asarray(vertex_ids[order], dtype=np.int32),
        "theta": np.asarray(theta, dtype=np.float64),
        "radii": np.asarray(radii, dtype=np.float64),
        "median_radius": median_radius,
        "covariance": np.asarray(covariance, dtype=np.float64),
        "square_root": np.asarray(square_root, dtype=np.float64),
        "inverse_square_root": np.asarray(inverse_sqrt, dtype=np.float64),
        "whitened_theta": np.asarray(whitened_theta, dtype=np.float64),
        "whitened_radii": np.asarray(whitened_radii, dtype=np.float64),
        "axis_ratio": float(np.sqrt(
            max(float(eigenvalues[-1]), 1.0e-12)
            / max(float(eigenvalues[0]), 1.0e-12)
        )),
    }


def _tube_profile_radii(theta, profile, whitened=False):
    """Sample a periodic angle/radius profile at target vertex angles."""
    import numpy as np

    theta = np.mod(np.asarray(theta, dtype=np.float64), 2.0 * np.pi)
    theta_key = "whitened_theta" if whitened else "theta"
    radii_key = "whitened_radii" if whitened else "radii"
    reference_theta = np.asarray(profile[theta_key], dtype=np.float64)
    reference_radii = np.asarray(profile[radii_key], dtype=np.float64)
    if len(reference_theta) < 2 or len(reference_theta) != len(reference_radii):
        raise ValueError("カーソル参照断面の輪郭profileが不足しています")
    extended_theta = np.concatenate(
        (reference_theta[-1:] - 2.0 * np.pi, reference_theta,
         reference_theta[:1] + 2.0 * np.pi)
    )
    extended_radii = np.concatenate(
        (reference_radii[-1:], reference_radii, reference_radii[:1])
    )
    return np.interp(theta, extended_theta, extended_radii)


def _tube_attach_reference_profile(points, geometry, seed_hit):
    """Cache the cursor profile and stable 2D maps for every prepared section."""
    import numpy as np

    enriched = dict(geometry)
    profile = _tube_reference_profile(points, enriched, seed_hit)
    ids = np.asarray(enriched["ids"], dtype=np.int32)
    section_ids = np.asarray(enriched["section_ids"], dtype=np.int32)
    centers = np.asarray(enriched["centers"], dtype=np.float64)
    tangents = np.asarray(enriched["tangents"], dtype=np.float64)
    e1 = np.asarray(enriched["e1"], dtype=np.float64)
    e2 = np.asarray(enriched["e2"], dtype=np.float64)
    section_centers = centers[section_ids]
    section_tangents = tangents[section_ids]
    section_e1 = e1[section_ids]
    section_e2 = e2[section_ids]
    relative = np.asarray(points, dtype=np.float64)[ids] - section_centers
    axial = np.sum(relative * section_tangents, axis=1, keepdims=True)
    source_radial = relative - axial * section_tangents
    mapped_radial = source_radial.copy()
    fixed_ids = set(int(value) for value in np.asarray(
        enriched.get("boundary_vertex_ids", ()), dtype=np.int32
    )) | set(int(value) for value in np.asarray(
        enriched.get("tip_vertex_ids", ()), dtype=np.int32
    ))
    for section in range(len(enriched["levels"])):
        local = np.flatnonzero(section_ids == section)
        if len(local) < 4:
            if any(int(ids[index]) not in fixed_ids for index in local):
                raise ValueError("断面輪郭点が不足しています")
            continue
        # A root boundary or a closed B terminal may itself be a single apex
        # or a degenerate multi-vertex cap.  Those vertices are immutable, so
        # they do not need a profile map and must not make a valid tube fail.
        if all(int(ids[index]) in fixed_ids for index in local):
            continue
        q = np.column_stack((
            np.sum(source_radial[local] * e1[section], axis=1),
            np.sum(source_radial[local] * e2[section], axis=1),
        ))
        if not np.all(np.isfinite(q)):
            raise ValueError("断面輪郭に有限でない点があります")
        q -= np.mean(q, axis=0, keepdims=True)
        covariance = np.cov(q.T)
        inverse_sqrt = _tube_symmetric_sqrt(covariance, inverse=True)
        whitened = q @ inverse_sqrt.T
        theta = np.mod(np.arctan2(whitened[:, 1], whitened[:, 0]), 2.0 * np.pi)
        desired_radii = _tube_profile_radii(theta, profile, whitened=True)
        direction = whitened / np.maximum(
            np.linalg.norm(whitened, axis=1, keepdims=True), 1.0e-12
        )
        desired_q = direction * desired_radii[:, None]
        desired_q = desired_q @ np.asarray(profile["square_root"], dtype=np.float64).T
        mapped_radial[local] = (
            desired_q[:, 0, None] * e1[section]
            + desired_q[:, 1, None] * e2[section]
        )
    enriched["reference_profile"] = profile
    enriched["profile_reference_radial"] = mapped_radial
    return enriched


def _tube_fit_taper(levels, radii):
    import numpy as np
    levels = np.asarray(levels, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    good = (levels > 0.05) & (levels < 0.94) & (radii > max(float(np.max(radii)) * 1.0e-3, 1.0e-8))
    if int(np.count_nonzero(good)) < 5:
        raise ValueError("先細り fit の断面不足")
    x = np.log(np.maximum(1.0 - levels[good], 1.0e-4))
    root = max(float(np.median(radii[:min(3, len(radii))])), 1.0e-8)
    y = np.log(np.maximum(radii[good] / root, 1.0e-4))
    exponent = float(np.clip(np.dot(x, y) / max(np.dot(x, x), 1.0e-12), 0.35, 3.0))
    predicted = root * np.maximum(1.0 - levels, 0.0) ** exponent
    correlation = float(np.corrcoef(radii[good], predicted[good])[0, 1]) if len(x) > 2 else 0.0
    tolerance = max(root * 0.035, 1.0e-4)
    monotone = float(np.mean(np.diff(radii) <= tolerance)) if len(radii) > 1 else 0.0
    return exponent, predicted, monotone, correlation


def _tube_classify(points, faces, face_sets, seed_face=0):
    """Classify a seed Face Set component as uniform A, tapered B, or reject."""
    import numpy as np

    topology = _tube_topology(faces, face_sets, seed_face)
    result = {
        "target_face_count": int(len(topology["target_faces"])),
        "target_vertex_count": int(len(topology["target_vertices"])),
        "external_boundary_count": int(len(topology["boundary_components"])),
        "external_boundary_sizes": [int(len(value)) for value in topology["boundary_components"]],
        "mesh_open_edge_count": int(len(topology["open_edges"])),
        "nonmanifold_edge_count": int(len(topology["nonmanifold_edges"])),
        "boundary_branch_vertex_count": int(len(topology["boundary_branch_vertices"])),
        "mode": None,
        "reason": None,
    }
    if topology["nonmanifold_edges"]:
        result["reason"] = (
            f"target接続edgeが非manifold（incidence>2 が "
            f"{len(topology['nonmanifold_edges'])} 本）"
        )
        return result, None
    if topology["boundary_branch_vertices"]:
        result["reason"] = "外部Face Set境界が分岐しているため断面を一意に推定できません"
        return result, None
    if len(topology["boundary_components"]) not in (1, 2):
        result["reason"] = (
            f"外部Face Set境界が{len(topology['boundary_components'])}個 "
            "（1/2以外は分岐または塗り残しの疑い）"
        )
        return result, None
    if topology["open_edges"]:
        result["reason"] = "targetにmesh穴/開放edgeがあり、閉端の判定不能"
        return result, None
    try:
        geometry = _tube_infer_geometry(np.asarray(points, dtype=np.float64), topology)
    except (IndexError, KeyError, MemoryError, TypeError, ValueError) as error:
        result["reason"] = str(error) or "局所断面を推定できません"
        return result, None
    circular = float(
        np.median(geometry["circularity"][1:-1])
        if len(geometry["circularity"]) > 2
        else 0.0
    )
    radii = geometry["section_radii"]
    result.update({
        "section_count": int(len(geometry["levels"])),
        "circularity_median": circular,
        "radius_cv": float(np.std(radii) / max(np.mean(radii), 1.0e-12)),
    })
    if len(geometry["levels"]) < 12:
        result["reason"] = f"長手方向が短すぎる（推定断面 {len(geometry['levels'])}）"
        return result, None
    if len(topology["boundary_components"]) == 2:
        result["mode"] = "A"
        result["reason"] = "2つの別Face Set境界、局所断面profile、長手方向を確認"
        return result, geometry
    try:
        exponent, predicted, monotone, correlation = _tube_fit_taper(
            geometry["levels"], radii
        )
    except ValueError as error:
        result["reason"] = str(error)
        return result, None
    tip_ids = np.asarray(geometry["tip_vertex_ids"], dtype=np.int32)
    boundary_edge_count = int(topology.get("target_component_boundary_edge_count", 0))
    expected_boundary_edge_count = int(
        sum(len(component) for component in topology["boundary_components"])
    )
    # One external boundary plus Euler characteristic 1 is a target-component
    # disk.  With open/non-manifold/branch edges already rejected, the disk's
    # other end is closed by its own target faces, regardless of whether the
    # terminal patch has one apex or many vertices.
    tip_is_closed = bool(
        len(topology["boundary_components"]) == 1
        and int(topology.get("target_component_euler", 0)) == 1
        and boundary_edge_count == expected_boundary_edge_count
        and not topology["open_edges"]
        and not topology["nonmanifold_edges"]
        and not topology["boundary_branch_vertices"]
        and len(tip_ids)
        and float(np.max(geometry["vertex_u"][np.isin(geometry["ids"], tip_ids)])) > 0.94
    )
    tip_ratio = float(np.median(radii[-min(3, len(radii)):]) / max(np.median(radii[:min(3, len(radii))]), 1.0e-12))
    result.update({
        "taper_p": exponent,
        "taper_tip_root_ratio": tip_ratio,
        "taper_monotone_fraction": monotone,
        "taper_corr": correlation,
        "tip_vertex_count": int(len(tip_ids)),
        "tip_closed": tip_is_closed,
        "terminal_euler": int(topology.get("target_component_euler", 0)),
        "terminal_boundary_edge_count": boundary_edge_count,
        "terminal_closure_evidence": bool(tip_is_closed),
    })
    if not tip_is_closed or tip_ratio > 0.65 or monotone < 0.76:
        result["reason"] = (
            "1境界だが閉じた先端/単調先細りを確認できない "
            f"(closed={tip_is_closed}, tip/root={tip_ratio:.3f}, "
            f"monotone={monotone:.3f}, corr={correlation:.3f})"
        )
        return result, None
    result["mode"] = "B"
    result["reason"] = "1つの別Face Set root境界、閉じたtip、単調先細りを確認"
    return result, geometry


def _tube_apply_shape(points_snapshot, classification, geometry, strength=1.0,
                      radius_scale=1.0, taper_strength=1.0, seed_hit=None,
                      vertex_weights=None):
    """Return shaped coordinates from one immutable snapshot.

    The function is intentionally pure: repeated parameter changes always use
    the same snapshot, and a zero strength call is byte-for-byte unchanged.
    """
    import numpy as np

    points0 = np.asarray(points_snapshot, dtype=np.float64).copy()
    output = points0.copy()
    if geometry is None or not classification or classification.get("mode") not in {"A", "B"}:
        return output
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return output
    ids = np.asarray(geometry["ids"], dtype=np.int32)
    u = np.asarray(geometry["vertex_u"], dtype=np.float64)
    section_ids = np.asarray(geometry["section_ids"], dtype=np.int32)
    section_centers = np.asarray(geometry["centers"], dtype=np.float64)[section_ids]
    section_tangents = np.asarray(geometry["tangents"], dtype=np.float64)[section_ids]
    relative = points0[ids] - section_centers
    axial = np.sum(relative * section_tangents, axis=1, keepdims=True)
    radial = relative - axial * section_tangents
    if seed_hit is None:
        centerline_centers = np.asarray(geometry["centers"], dtype=np.float64)
        seed_hit = centerline_centers[len(centerline_centers) // 2]
    seed_hit = np.asarray(seed_hit, dtype=np.float64)
    # The map is prepared once with the common section frame. It transports
    # each target contour onto the cursor contour in whitened 2D coordinates,
    # preserving long/short axes and mild angular irregularity. Wheel updates
    # therefore do not redo profile extraction or covariance decompositions.
    reference_profile = geometry.get("reference_profile")
    mapped_radial = geometry.get("profile_reference_radial")
    if reference_profile is None or mapped_radial is None:
        geometry = _tube_attach_reference_profile(points0, geometry, seed_hit)
        reference_profile = geometry["reference_profile"]
        mapped_radial = geometry["profile_reference_radial"]
    mapped_radial = np.asarray(mapped_radial, dtype=np.float64)
    if mapped_radial.shape != radial.shape or not np.all(np.isfinite(mapped_radial)):
        raise ValueError("カーソル参照断面profileの写像が不正です")
    mode = classification["mode"]
    if mode == "A":
        desired_radial = mapped_radial * float(np.clip(radius_scale, 0.125, 8.0))
        envelope = _tube_smoothstep(u / 0.13) * _tube_smoothstep((1.0 - u) / 0.13)
    else:
        exponent = float(classification["taper_p"]) * float(np.clip(taper_strength, 0.2, 3.0))
        root_radius = float(np.median(geometry["section_radii"][:min(3, len(geometry["section_radii"]))]))
        tip_floor = max(
            0.012 * root_radius,
            float(np.median(geometry["section_radii"][-min(2, len(geometry["section_radii"])):])) * 0.6,
        )
        taper_radius = root_radius * np.maximum(1.0 - u, 0.0) ** exponent + tip_floor
        shape_unit = mapped_radial / max(float(reference_profile["median_radius"]), 1.0e-12)
        desired_radial = shape_unit * taper_radius[:, None]
        envelope = _tube_smoothstep(u / 0.16)
    corrected = section_centers + (
        axial * section_tangents
        + radial + strength * envelope[:, None] * (desired_radial - radial)
    )
    if vertex_weights is None:
        weights = np.ones(len(points0), dtype=np.float64)
    else:
        weights = np.asarray(vertex_weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(points0):
            raise ValueError("vertex mask data does not match mesh vertex count")
        weights = np.clip(1.0 - weights, 0.0, 1.0)
    boundary = set(int(value) for value in np.asarray(geometry["boundary_vertex_ids"], dtype=np.int32))
    tip = set(int(value) for value in np.asarray(geometry.get("tip_vertex_ids", ()), dtype=np.int32))
    fixed = np.asarray(
        [int(value) in boundary or (mode == "B" and int(value) in tip) for value in ids],
        dtype=bool,
    )
    raw_delta = corrected - points0[ids]
    delta = raw_delta * weights[ids, None]
    for section in range(len(geometry["levels"])):
        local = np.flatnonzero((section_ids == section) & ~fixed & (weights[ids] > 1.0e-8))
        if len(local):
            local_weights = weights[ids[local]]
            # Remove center drift while retaining mask attenuation.  The
            # correction subtracted from each vertex is multiplied by its own
            # weight, so a nearly masked vertex tends continuously to zero
            # displacement instead of receiving an unweighted mean shift.
            denominator = max(float(np.sum(local_weights * local_weights)), 1.0e-12)
            mean = np.sum(delta[local] * local_weights[:, None], axis=0) / denominator
            delta[local] -= mean * local_weights[:, None]
    delta[fixed] = 0.0
    output[ids] = points0[ids] + delta
    return output


def _tube_scatter_local_delta(local_snapshot, world_source, world_target, target_ids, world_matrix):
    """Scatter world-space target deltas while preserving non-target locals."""
    import numpy as np

    local_snapshot = np.asarray(local_snapshot, dtype=np.float64)
    world_source = np.asarray(world_source, dtype=np.float64)
    world_target = np.asarray(world_target, dtype=np.float64)
    target_ids = np.asarray(target_ids, dtype=np.int32).reshape(-1)
    matrix = np.asarray(world_matrix, dtype=np.float64)
    if len(world_source) != len(target_ids) or len(world_target) != len(target_ids):
        raise ValueError("target delta rows do not match target ids")
    inverse = np.linalg.inv(matrix[:3, :3])
    output = local_snapshot.copy()
    output[target_ids] += (world_target - world_source) @ inverse.T
    return output


def _tube_preview_lines(points, geometry, max_edges=2400):
    """Create a small line set from the actual shaped candidate coordinates."""
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    ids = np.asarray(geometry["ids"], dtype=np.int32)
    local_faces = np.asarray(geometry.get("target_faces", ()), dtype=np.int32).reshape((-1, 3))
    if len(local_faces) == 0:
        return np.empty((0, 3), dtype=np.float32)
    edges = np.concatenate(
        (local_faces[:, [0, 1]], local_faces[:, [1, 2]], local_faces[:, [2, 0]]),
        axis=0,
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    # Keep the preview bounded and deterministic.  The sampled lines still
    # use the final candidate coordinates, including mask and fixed-end rules.
    if len(edges) > int(max_edges):
        indices = np.linspace(0, len(edges) - 1, int(max_edges), dtype=np.int64)
        edges = edges[indices]
    return points[edges].reshape((-1, 3)).astype(np.float32, copy=False)


def _tube_prepare_faces(loop_vertices, totals, face_sets):
    import numpy as np
    loop_vertices = np.asarray(loop_vertices, dtype=np.int32)
    totals = np.asarray(totals, dtype=np.int32)
    face_sets = np.asarray(face_sets, dtype=np.int32)
    starts = np.r_[0, np.cumsum(totals[:-1], dtype=np.int64)] if len(totals) else np.empty(0, dtype=np.int64)
    valid = totals >= 3
    if not np.any(valid):
        return (np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
    if np.all(totals[valid] == 3) and np.all(valid):
        starts3 = starts.astype(np.int64)
        faces = np.column_stack((
            loop_vertices[starts3], loop_vertices[starts3 + 1], loop_vertices[starts3 + 2]
        )).astype(np.int32, copy=False)
        return faces, face_sets.astype(np.int32, copy=True), np.arange(len(totals), dtype=np.int32)
    # Vectorized triangle-fan expansion for quads/ngons.  No per-polygon Python
    # loop is needed, which keeps the common large-mesh preparation bounded by
    # NumPy array work.
    face_indices = np.flatnonzero(valid).astype(np.int32)
    valid_totals = totals[valid].astype(np.int64)
    triangle_counts = valid_totals - 2
    repeated_faces = np.repeat(face_indices, triangle_counts)
    repeated_starts = np.repeat(starts[valid], triangle_counts)
    local_offsets = np.arange(int(np.sum(triangle_counts)), dtype=np.int64)
    local_offsets -= np.repeat(np.cumsum(triangle_counts) - triangle_counts, triangle_counts)
    local_offsets += 1
    faces = np.column_stack((
        loop_vertices[repeated_starts],
        loop_vertices[repeated_starts + local_offsets],
        loop_vertices[repeated_starts + local_offsets + 1],
    )).astype(np.int32, copy=False)
    sets = face_sets[repeated_faces].astype(np.int32, copy=False)
    source_ids = repeated_faces.astype(np.int32, copy=False)
    return (
        np.asarray(faces, dtype=np.int32).reshape((-1, 3)),
        np.asarray(sets, dtype=np.int32),
        np.asarray(source_ids, dtype=np.int32),
    )


def _tube_shape_from_arrays(coordinates_world, faces, face_sets, seed_face, seed_hit,
                            strength=1.0, radius_scale=1.0, taper_strength=1.0,
                            vertex_weights=None):
    """Production/test entry point: classify and shape only the seed component."""
    import numpy as np
    classification, geometry = _tube_classify(
        coordinates_world, faces, face_sets, seed_face
    )
    if geometry is not None:
        try:
            geometry = _tube_attach_reference_profile(
                coordinates_world, geometry, seed_hit
            )
            result = _tube_apply_shape(
                coordinates_world, classification, geometry, strength=strength,
                radius_scale=radius_scale, taper_strength=taper_strength,
                seed_hit=seed_hit, vertex_weights=vertex_weights,
            )
        except (IndexError, KeyError, MemoryError, TypeError, ValueError) as error:
            classification = dict(classification)
            classification["mode"] = None
            classification["reason"] = str(error) or "カーソル参照断面を取得できません"
            geometry = None
            result = np.asarray(coordinates_world, dtype=np.float64).copy()
    else:
        result = np.asarray(coordinates_world, dtype=np.float64).copy()
    if geometry is not None:
        result_meta = dict(classification)
        result_meta["preview_lines"] = _tube_preview_lines(result, geometry)
    else:
        result_meta = dict(classification)
    return result_meta, geometry, result


def _tube_safety_reason(obj):
    """Reject mesh states whose Sculpt coordinates cannot be mapped safely."""
    try:
        mesh = obj.data
        if int(getattr(mesh, "users", 1)) > 1:
            return "linked mesh data has multiple users; tube shaping made no changes"
        if getattr(obj, "library", None) is not None or getattr(mesh, "library", None) is not None:
            return "linked library mesh is read-only; tube shaping made no changes"
        if getattr(mesh, "shape_keys", None) is not None:
            return "shape-key coordinates are not a safe Sculpt snapshot target"
        if bool(getattr(obj, "use_dynamic_topology_sculpting", False)):
            return "Dyntopo is active; stable source vertex mapping is unavailable"
        for modifier in getattr(obj, "modifiers", ()):
            if str(getattr(modifier, "type", "")) == "MULTIRES":
                return "Multires modifier is active; stable source vertex mapping is unavailable"
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return "mesh safety state could not be read; tube shaping made no changes"
    return None


def _tube_prepare_steps(state):
    """Read only the arrays needed by the seed component, yielding per array."""
    import numpy as np

    obj = state["obj"]
    mesh = obj.data
    face_count = len(mesh.polygons)
    loop_count = len(mesh.loops)
    loop_vertices = np.empty(loop_count, dtype=np.int32)
    totals = np.empty(face_count, dtype=np.int32)
    face_sets = np.empty(face_count, dtype=np.int32)
    polygon_hidden = np.empty(face_count, dtype=bool)
    vertex_hidden = np.empty(len(mesh.vertices), dtype=bool)
    yield "allocate"
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    yield "loop-vertices"
    mesh.polygons.foreach_get("loop_total", totals)
    yield "polygon-totals"
    mesh.polygons.foreach_get("hide", polygon_hidden)
    yield "polygon-visibility"
    mesh.vertices.foreach_get("hide", vertex_hidden)
    yield "vertex-visibility"
    attribute = mesh.attributes.get(".sculpt_face_set")
    if attribute is None or attribute.domain != "FACE" or len(attribute.data) != face_count:
        raise ValueError(".sculpt_face_set がないため tube を判定できません")
    attribute.data.foreach_get("value", face_sets)
    yield "face-sets"
    mask = np.zeros(len(mesh.vertices), dtype=np.float64)
    mask_attribute = mesh.attributes.get(".sculpt_mask")
    if mask_attribute is not None and mask_attribute.domain == "POINT" and len(mask_attribute.data) == len(mask):
        mask_attribute.data.foreach_get("value", mask)
    yield "sculpt-mask"
    faces, triangle_sets, source_ids = _tube_prepare_faces(loop_vertices, totals, face_sets)
    if len(faces) == 0:
        raise ValueError("有効なmesh faceがありません")
    # Keep all triangles carrying the seed ID (needed to find its connected
    # component) plus only different-ID faces touching those vertices (needed
    # to detect external Face Set boundaries).  Unrelated scene regions are
    # excluded before the edge table is built.
    seed_face_set_id = int(face_sets[int(state["seed_face"])])
    seed_id_mask = triangle_sets == seed_face_set_id
    seed_id_vertices = np.unique(faces[seed_id_mask])
    touching_seed_vertices = np.any(np.isin(faces, seed_id_vertices), axis=1)
    candidate_mask = seed_id_mask | touching_seed_vertices
    faces = faces[candidate_mask]
    triangle_sets = triangle_sets[candidate_mask]
    source_ids = source_ids[candidate_mask]
    seed_triangles = np.flatnonzero(source_ids == int(state["seed_face"]))
    if len(seed_triangles) == 0:
        raise ValueError("seed face の三角化結果がありません")
    # Hidden Sculpt vertices and every vertex touching a hidden polygon are
    # hard protected.  Keep partial mask values for the remaining vertices.
    raw_sculpt_mask = np.clip(mask, 0.0, 1.0).copy()
    hidden_weight = np.zeros(len(mesh.vertices), dtype=np.float64)
    hidden_weight[vertex_hidden] = 1.0
    hidden_triangles = np.flatnonzero(polygon_hidden[source_ids])
    if len(hidden_triangles):
        hidden_weight[np.unique(faces[hidden_triangles])] = 1.0
    mask = np.maximum(np.clip(mask, 0.0, 1.0), hidden_weight)
    state_matrix = np.asarray(state["world_matrix"], dtype=np.float64)
    local = np.asarray(state["coordinates_local"], dtype=np.float64)
    linear = state_matrix[:3, :3]
    translation = state_matrix[:3, 3]
    world = local @ linear.T + translation
    yield "triangles"
    return {
        "coordinates_world": world,
        "faces": faces,
        "face_sets": triangle_sets,
        "source_face_ids": source_ids,
        "seed_triangle": int(seed_triangles[0]),
        "vertex_hidden": vertex_hidden,
        "polygon_hidden": polygon_hidden,
        "loop_vertices": np.array(loop_vertices, dtype=np.int32, copy=True),
        "totals": np.array(totals, dtype=np.int32, copy=True),
        "polygon_face_sets": np.array(face_sets, dtype=np.int32, copy=True),
        "sculpt_mask": np.array(raw_sculpt_mask, dtype=np.float64, copy=True),
        "vertex_weights": mask,
        "signature": state["signature"],
    }


def _tube_make_result(state, prepared):
    import numpy as np
    classification = prepared.get("classification")
    geometry = prepared.get("geometry")
    if classification is None or geometry is None:
        classification, geometry = _tube_classify(
            prepared["coordinates_world"], prepared["faces"],
            prepared["face_sets"], prepared["seed_triangle"]
        )
        if geometry is None:
            raise ValueError(str(classification.get("reason") or "tube shape is ambiguous"))
        prepared["classification"] = classification
        prepared["geometry"] = geometry
    if "reference_profile" not in geometry:
        geometry = _tube_attach_reference_profile(
            prepared["coordinates_world"], geometry, state["seed_hit"]
        )
        prepared["geometry"] = geometry
    shaped = _tube_apply_shape(
        prepared["coordinates_world"], classification, geometry,
        strength=float(state["strength"]),
        radius_scale=float(state["radius_scale"]),
        taper_strength=float(state["taper_strength"]),
        seed_hit=state["seed_hit"],
        vertex_weights=prepared["vertex_weights"],
    )
    if geometry is None:
        raise ValueError("tube shape is ambiguous")
    result_meta = dict(classification)
    preview_lines = _tube_preview_lines(shaped, geometry)
    target_ids = np.asarray(geometry["ids"], dtype=np.int32)
    return {
        "classification": result_meta,
        "geometry": geometry,
        # Keep only the shaped target rows in the wheel cache.  The immutable
        # source world array remains in prepared once; non-target rows are
        # reconstructed from it at confirmation.
        "target_ids": np.array(target_ids, dtype=np.int32, copy=True),
        "target_world": np.asarray(shaped[target_ids], dtype=np.float64).copy(),
        "preview_lines": np.asarray(preview_lines, dtype=np.float32),
        "candidate_count": int(len(geometry.get("ids", ()))),
        "created_generation": int(state["generation"]),
        "gpu_batch": None,
    }


def _tube_preview_tag_redraw(state=None):
    if state is not None:
        _tag_redraw(state.get("area"))


def _tube_preview_stop_draw():
    for handler in (_runtime.tube_preview_draw_handler, _runtime.tube_preview_text_draw_handler):
        if handler is None:
            continue
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handler, "WINDOW")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    _runtime.tube_preview_draw_handler = None
    _runtime.tube_preview_text_draw_handler = None


def _tube_preview_cancel(state=None, reason="cancel"):
    current = _runtime.tube_preview_state
    if state is not None and current is not state:
        return
    if current is None:
        return
    current["active"] = False
    timer = current.get("timer")
    if timer is not None:
        try:
            current["window_manager"].event_timer_remove(timer)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
    current["timer"] = None
    _tube_preview_stop_draw()
    _runtime.tube_preview_state = None
    _tube_preview_tag_redraw(current)


def _tube_preview_request_cancel(state=None, reason="external-change"):
    current = _runtime.tube_preview_state
    if state is not None and current is not state:
        return
    if current is None:
        return
    entry = _lifecycle.modal_request_cancel(
        operator=current.get("operator"),
        reason=reason,
    )
    _tube_preview_cancel(current, reason)
    _lifecycle.modal_schedule_terminal_event(
        operator=current.get("operator"),
        token=entry.get("token") if entry else None,
        state=current,
    )


def _tube_preview_valid(state, context):
    if state is None or not state.get("active"):
        return False
    try:
        window = getattr(context, "window", None)
        window_key = int(state.get("window_key", 0) or 0)
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
            and int(context.area.as_pointer()) == int(state["area_key"])
            and (not window_key or (window is not None and int(window.as_pointer()) == window_key))
            and context.mode == "SCULPT"
            and context.active_object is state["obj"]
            and _smart_fill_geometry._fill_preview_signature(state["obj"]) == state["signature"]
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _tube_confirm_source_unchanged(state):
    """Recheck topology and Sculpt protection layers once before the write."""
    import numpy as np

    obj = state["obj"]
    mesh = obj.data
    prepared = state.get("prepared") or {}
    try:
        current_loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
        current_totals = np.empty(len(mesh.polygons), dtype=np.int32)
        current_face_sets = np.empty(len(mesh.polygons), dtype=np.int32)
        current_polygon_hidden = np.empty(len(mesh.polygons), dtype=bool)
        current_vertex_hidden = np.empty(len(mesh.vertices), dtype=bool)
        mesh.loops.foreach_get("vertex_index", current_loop_vertices)
        mesh.polygons.foreach_get("loop_total", current_totals)
        mesh.polygons.foreach_get("hide", current_polygon_hidden)
        mesh.vertices.foreach_get("hide", current_vertex_hidden)
        attribute = mesh.attributes.get(".sculpt_face_set")
        if attribute is None or attribute.domain != "FACE" or len(attribute.data) != len(current_face_sets):
            return False
        attribute.data.foreach_get("value", current_face_sets)
        if not np.array_equal(current_loop_vertices, prepared.get("loop_vertices")):
            return False
        if not np.array_equal(current_totals, prepared.get("totals")):
            return False
        if not np.array_equal(current_face_sets, prepared.get("polygon_face_sets")):
            return False
        if not np.array_equal(current_polygon_hidden, prepared.get("polygon_hidden")):
            return False
        if not np.array_equal(current_vertex_hidden, prepared.get("vertex_hidden")):
            return False
        current_mask = np.zeros(len(mesh.vertices), dtype=np.float64)
        mask_attribute = mesh.attributes.get(".sculpt_mask")
        if mask_attribute is not None and mask_attribute.domain == "POINT" and len(mask_attribute.data) == len(current_mask):
            mask_attribute.data.foreach_get("value", current_mask)
        if not np.array_equal(
            np.clip(current_mask, 0.0, 1.0),
            np.asarray(prepared.get("sculpt_mask"), dtype=np.float64),
        ):
            return False
    except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _tube_preview_draw():
    state = _runtime.tube_preview_state
    if state is None or not state.get("active"):
        return
    try:
        context = bpy.context
        if context.area is None or int(context.area.as_pointer()) != int(state["area_key"]):
            return
        result = state.get("result")
        if result is None:
            return
        lines = result.get("preview_lines")
        if lines is None or not len(lines):
            return
        shader = _fill_preview_shader_get()
        if shader is None:
            return
        if result.get("gpu_batch") is None:
            result["gpu_batch"] = batch_for_shader(shader, "LINES", {"pos": lines})
        gpu.state.blend_set("ALPHA")
        try:
            # This is a prediction overlay.  Disable depth testing so a
            # contracted candidate remains visible through the source surface.
            try:
                gpu.state.depth_test_set("NONE")
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            shader.bind()
            mode = result.get("classification", {}).get("mode")
            shader.uniform_float("color", (1.0, 0.32, 0.08, 0.92) if mode == "A" else (0.28, 0.85, 1.0, 0.92))
            result["gpu_batch"].draw(shader)
        finally:
            try:
                gpu.state.depth_test_set("NONE")
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            gpu.state.blend_set("NONE")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        try:
            gpu.state.blend_set("NONE")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


def _tube_preview_draw_text():
    state = _runtime.tube_preview_state
    if state is None or not state.get("active"):
        return
    try:
        context = bpy.context
        if context.area is None or int(context.area.as_pointer()) != int(state["area_key"]):
            return
        font_id = 0
        blf.size(font_id, 13)
        width = int(getattr(state.get("region"), "width", 0))
        height = int(getattr(state.get("region"), "height", 0))
        phase = state.get("phase")
        result = state.get("result") or {}
        if phase == "prepare":
            lines = [
                "Tube Shape - reading target mesh",
                "Analyzing Face Set boundary and local sections",
                "Esc: cancel",
            ]
        elif phase == "build":
            lines = ["Tube Shape - checking shape and taper", "Esc: cancel"]
        else:
            classification = result.get("classification", {})
            mode = classification.get("mode", "?")
            mode_name = "Uniform Tube" if mode == "A" else "Tapered Tip"
            lines = [
                f"Tube Shape  {mode_name}",
                f"Radius x {float(state['radius_scale']):.3g}  Taper x {float(state['taper_strength']):.3g}  Strength {float(state['strength']):.2f}",
                "Wheel: mode value  Shift+Wheel: correction  T/Left: apply  Esc: cancel",
            ]
        x, y = max(18, width - 650), max(120, height - 100)
        for index, line in enumerate(lines):
            blf.position(font_id, x, y - index * 18, 0)
            blf.color(font_id, 0.92, 0.96, 1.0, 1.0)
            blf.draw(font_id, line)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass


def _on_tube_preview_depsgraph_update(_scene, depsgraph):
    """Cancel a live prediction when its source object or mesh is rebuilt."""
    state = _runtime.tube_preview_state
    if state is None or not state.get("active"):
        return
    try:
        updated = set()
        for update in depsgraph.updates:
            data = update.id
            updated.add(int(data.as_pointer()))
            original = getattr(data, "original", None)
            if original is not None:
                updated.add(int(original.as_pointer()))
        if int(state["obj"].as_pointer()) in updated or int(state["obj"].data.as_pointer()) in updated:
            _tube_preview_cancel(state, "stale")
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _tube_preview_request_cancel(state, "stale")


class VIEW3D_OT_mesh_focus_tube_shape(bpy.types.Operator):
    """Shape one curved Sculpt Face Set component with a safe preview."""

    bl_idname = TUBE_SHAPE_OPERATOR_ID
    bl_label = "Mesh Focus: Curved Face Set Tube Shape"
    bl_description = "Preview and shape the connected Face Set tube under the cursor"
    bl_options = {"REGISTER", "UNDO"}

    mouse_region_x: IntProperty(options={"SKIP_SAVE"})
    mouse_region_y: IntProperty(options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None and context.area.type == "VIEW_3D"
            and context.region is not None and context.region.type == "WINDOW"
            and context.space_data is not None and context.mode == "SCULPT"
            and context.active_object is not None and context.active_object.type == "MESH"
        )

    def invoke(self, context, event):
        import numpy as np
        if not self.poll(context):
            return {"PASS_THROUGH"}
        if _runtime.fill_preview_state is not None and _runtime.fill_preview_state.get("active"):
            self.report({"WARNING"}, "Tube Shape: 先に Smart Fill の予測を終了してください")
            return {"CANCELLED"}
        if _runtime.tube_preview_state is not None:
            _tube_preview_cancel(reason="replaced")
        obj = context.active_object
        safety_reason = _tube_safety_reason(obj)
        if safety_reason:
            self.report({"WARNING"}, f"Tube Shape: {safety_reason}")
            return {"CANCELLED"}
        coord = _sculpt_cursor_region_coordinate(context, event)
        if coord is None:
            self.report({"WARNING"}, "Tube Shape: cursor is outside the View3D window region")
            return {"CANCELLED"}
        hit = _raycast_sculpt_face_set(context, coord)
        if hit is None:
            self.report({"WARNING"}, "Tube Shape: visible Sculpt Face Set face not found")
            return {"CANCELLED"}
        hit_obj, seed_face, seed_face_set, seed_hit, _screen = hit
        if hit_obj is not obj:
            self.report({"WARNING"}, "Tube Shape: active mesh was not the ray-hit object")
            return {"CANCELLED"}
        try:
            coordinates_local = np.empty((len(obj.data.vertices), 3), dtype=np.float64)
            obj.data.vertices.foreach_get("co", coordinates_local.ravel())
            signature = _smart_fill_geometry._fill_preview_signature(obj)
            matrix = obj.matrix_world.copy()
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError, MemoryError):
            self.report({"WARNING"}, "Tube Shape: source coordinate snapshot failed")
            return {"CANCELLED"}
        if signature is None:
            return {"CANCELLED"}
        self.mouse_region_x = int(round(coord.x))
        self.mouse_region_y = int(round(coord.y))
        self._tube_session_id = _next_session_id()
        state = {
            "active": True, "phase": "prepare", "operator": self,
            "session_id": int(self._tube_session_id),
            "area": context.area, "area_key": int(context.area.as_pointer()),
            "window_key": int(context.window.as_pointer()) if context.window else 0,
            "window_manager": context.window_manager, "region": context.region,
            "obj": obj, "obj_pointer": int(obj.as_pointer()),
            "mesh_pointer": int(obj.data.as_pointer()), "signature": signature,
            "mode": str(context.mode), "seed_face": int(seed_face),
            "seed_face_set": int(seed_face_set),
            "seed_hit": tuple(float(value) for value in seed_hit),
            "coordinates_local": coordinates_local, "world_matrix": matrix,
            "prepare_job": None, "prepare_stage": "queued", "prepared": None,
            "result": None, "results": {}, "generation": 0,
            "strength": 1.0, "radius_scale": 1.0, "taper_strength": 1.0,
            "start_key": str(getattr(event, "type", TUBE_SHAPE_KEY) or TUBE_SHAPE_KEY),
            "start_key_released": False, "timer": None,
            "prepare_seconds": 0.0, "last_tick_seconds": 0.0,
        }
        _runtime.tube_preview_state = state
        try:
            state["timer"] = context.window_manager.event_timer_add(0.01, window=context.window)
            context.window_manager.modal_handler_add(self)
            _lifecycle.modal_register(self, "Tube Shape", state.get("session_id"), state=state)
            _runtime.tube_preview_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                _tube_preview_draw, (), "WINDOW", "POST_VIEW"
            )
            _runtime.tube_preview_text_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                _tube_preview_draw_text, (), "WINDOW", "POST_PIXEL"
            )
            _tube_preview_tag_redraw(state)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _tube_preview_cancel(state, "start")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _process_timer(self, context, state):
        started = time.perf_counter()
        try:
            if not _tube_preview_valid(state, context):
                _tube_preview_cancel(state, "stale")
                return False
            if state["phase"] == "prepare":
                if state["prepare_job"] is None:
                    state["prepare_job"] = _tube_prepare_steps(state)
                try:
                    state["prepare_stage"] = next(state["prepare_job"])
                except StopIteration as complete:
                    state["prepare_job"] = None
                    state["prepared"] = complete.value
                    state["phase"] = "build"
                state["last_tick_seconds"] = time.perf_counter() - started
                state["prepare_seconds"] += state["last_tick_seconds"]
                _tube_preview_tag_redraw(state)
                return True
            if state["phase"] == "build":
                state["generation"] += 1
                result = _tube_make_result(state, state["prepared"])
                state["result"] = result
                state["results"][(round(state["strength"], 6), round(state["radius_scale"], 6), round(state["taper_strength"], 6))] = result
                state["phase"] = "ready"
                state["last_tick_seconds"] = time.perf_counter() - started
                _tube_preview_tag_redraw(state)
                return True
            return True
        except (AttributeError, IndexError, KeyError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
            try:
                self.report({"WARNING"}, f"Tube Shape: {str(error).strip() or '判定に失敗しました'}")
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            _tube_preview_cancel(state, "prepare-error")
            return False

    def _finish_confirm(self, context, state):
        import numpy as np
        if state.get("phase") != "ready" or state.get("result") is None:
            return {"RUNNING_MODAL"}
        if not _tube_preview_valid(state, context):
            _tube_preview_cancel(state, "stale")
            return {"CANCELLED"}
        result = state["result"]
        wrote = False
        try:
            current_local = np.empty_like(state["coordinates_local"])
            state["obj"].data.vertices.foreach_get("co", current_local.ravel())
            if not np.array_equal(current_local, state["coordinates_local"]):
                self.report({"WARNING"}, "Tube Shape: source mesh changed during preview; no changes applied")
                _tube_preview_cancel(state, "source-changed")
                return {"CANCELLED"}
            if not _tube_confirm_source_unchanged(state):
                self.report({"WARNING"}, "Tube Shape: Face Set, topology, hidden, or mask state changed; no changes applied")
                _tube_preview_cancel(state, "source-attributes-changed")
                return {"CANCELLED"}
            target_ids = np.asarray(result["target_ids"], dtype=np.int32)
            target_world = np.asarray(result["target_world"], dtype=np.float64)
            prepared_world = np.asarray(state["prepared"]["coordinates_world"], dtype=np.float64)
            if len(target_ids) != len(target_world) or np.any(target_ids < 0) or np.any(target_ids >= len(prepared_world)):
                raise ValueError("tube candidate rows are invalid")
            # Scatter only the movable target deltas onto the immutable local
            # snapshot.  This preserves every non-target value exactly,
            # including vertices which would otherwise change by a world/local
            # round trip under a non-uniform transform.
            local = _tube_scatter_local_delta(
                state["coordinates_local"], prepared_world[target_ids],
                target_world, target_ids, state["world_matrix"]
            )
            state["obj"].data.vertices.foreach_set("co", local.ravel())
            wrote = True
            state["obj"].data.update()
            state["obj"].update_tag(refresh={"DATA"})
            context.view_layer.update()
        except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError):
            if wrote:
                try:
                    state["obj"].data.vertices.foreach_set(
                        "co", np.asarray(state["coordinates_local"], dtype=np.float64).ravel()
                    )
                    state["obj"].data.update()
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    pass
            _tube_preview_cancel(state, "write-error")
            return {"CANCELLED"}
        candidate_count = int(result.get("candidate_count", 0))
        mode = str(result.get("classification", {}).get("mode", "?"))
        _tube_preview_cancel(state, "confirm")
        self.report({"INFO"}, f"Tube Shape: mode {mode}, {candidate_count} target vertices shaped")
        return {"FINISHED"}

    def modal(self, context, event):
        state = _runtime.tube_preview_state
        entry = _lifecycle.modal_entry(operator=self)
        if entry is not None and (
            entry.get("cancel_requested") or entry.get("teardown_requested")
        ):
            if state is not None:
                _tube_preview_cancel(state, entry.get("cancel_reason") or "external-change")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        if state is None or state.get("operator") is not self:
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        event_type = getattr(event, "type", "")
        event_value = getattr(event, "value", None)
        if event_type == "ESC" and event_value in {None, "PRESS"}:
            _tube_preview_cancel(state, "escape")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        if event_type == state.get("start_key", TUBE_SHAPE_KEY):
            if event_value == "RELEASE":
                state["start_key_released"] = True
            elif event_value == "PRESS" and state.get("start_key_released") and not bool(getattr(event, "is_repeat", False)):
                result = self._finish_confirm(context, state)
                if result != {"RUNNING_MODAL"}:
                    _lifecycle.modal_terminal(
                        operator=self,
                        status="FINISHED" if result == {"FINISHED"} else "CANCELLED",
                    )
                return result
            return {"RUNNING_MODAL"}
        if event_type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            if state.get("phase") != "ready":
                return {"RUNNING_MODAL"}
            factor = 1.2 if event_type == "WHEELUPMOUSE" else 1.0 / 1.2
            if bool(getattr(event, "shift", False)):
                state["strength"] = max(0.0, min(1.0, float(state["strength"]) + (0.1 if factor > 1.0 else -0.1)))
            elif state.get("result", {}).get("classification", {}).get("mode") == "A":
                state["radius_scale"] = max(0.125, min(8.0, float(state["radius_scale"]) * factor))
            else:
                state["taper_strength"] = max(0.2, min(3.0, float(state["taper_strength"]) * factor))
            key = (round(state["strength"], 6), round(state["radius_scale"], 6), round(state["taper_strength"], 6))
            try:
                cached = state["results"].get(key)
                state["generation"] += 1
                if cached is None:
                    cached = _tube_make_result(state, state["prepared"])
                    state["results"][key] = cached
                    while len(state["results"]) > 6:
                        state["results"].pop(next(iter(state["results"])), None)
                cached["created_generation"] = int(state["generation"])
                state["result"] = cached
                _tube_preview_tag_redraw(state)
                return {"RUNNING_MODAL"}
            except (AttributeError, IndexError, KeyError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
                try:
                    self.report({"WARNING"}, f"Tube Shape: {str(error).strip() or 'preview calculation failed'}")
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                _tube_preview_cancel(state, "wheel-error")
                _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                return {"CANCELLED"}
        if event_type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER", "ENTER"}:
            if event_type == "LEFTMOUSE" and event_value not in {None, "PRESS"}:
                return {"RUNNING_MODAL"}
            result = self._finish_confirm(context, state)
            if result != {"RUNNING_MODAL"}:
                _lifecycle.modal_terminal(
                    operator=self,
                    status=(
                        "FINISHED"
                        if result == {"FINISHED"}
                        else "CANCELLED"
                    ),
                )
            return result
        if event_type == "RIGHTMOUSE":
            return {"RUNNING_MODAL"}
        if event_type == "TIMER":
            event_timer = getattr(event, "timer", None)
            if event_timer is not None and event_timer is not state.get("timer"):
                return {"RUNNING_MODAL"}
            if not self._process_timer(context, state):
                _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}
        if not _tube_preview_valid(state, context):
            _tube_preview_cancel(state, "stale")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}
