"""Isolated Blender API checks for the resident MFO WorkSpaceTools.

Run inside Blender with::

    exec(compile(open(path).read(), path, "exec"), {"__name__": "__main__"})

The test does not select a tool, edit scene data, or save the current file.
"""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import time
from types import SimpleNamespace


def _load_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "mesh_focus_orbit.py"
    spec = importlib.util.spec_from_file_location("mfo_toolbar_isolated", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_prepare_job(job):
    """Consume the real preparation generator and retain stage timing."""
    tokens = []
    durations = []
    while True:
        started = time.perf_counter()
        try:
            token = next(job)
        except StopIteration as complete:
            return complete.value, tokens, durations
        tokens.append(token)
        durations.append(time.perf_counter() - started)


def _guided_ridge_snapshot_equivalence(module):
    """Compare sync and cooperative snapshots on a temporary Face Set mesh."""
    import bpy
    import numpy as np
    from mathutils import Matrix

    mesh = bpy.data.meshes.new("_mfo_guided_ridge_equivalence_mesh")
    mesh.from_pydata(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        [],
        [(0, 1, 2), (0, 2, 3)],
    )
    mesh.update()
    attr = mesh.attributes.new(".sculpt_face_set", "INT", "FACE")
    for item in attr.data:
        item.value = 7
    obj = bpy.data.objects.new("_mfo_guided_ridge_equivalence_object", mesh)
    obj.matrix_world = Matrix.Translation((1.25, -2.0, 0.5)) @ Matrix.Rotation(0.37, 4, "Z")
    try:
        baseline, baseline_reason = module._guided_ridge_prepare_snapshot_sync(obj, 0, 7)
        cooperative, cooperative_reason = module._guided_ridge_prepare_snapshot(obj, 0, 7)
        assert baseline_reason is None, baseline_reason
        assert cooperative_reason is None, cooperative_reason
        assert baseline is not None and cooperative is not None
        comparable = (
            "object_pointer", "mesh_pointer", "counts", "face_set_id",
            "face_indices", "face_vertices", "vertex_indices", "triangles",
            "triangle_face_indices", "boundary_vertices", "protected_vertices",
            "boundary_edges", "adjacency", "coords_local", "coords_world",
            "normals_world", "hidden_vertices", "sculpt_mask", "average_edge",
            "signature",
        )
        for key in comparable:
            left, right = baseline[key], cooperative[key]
            if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                assert np.array_equal(np.asarray(left), np.asarray(right)), key
            else:
                assert left == right, key
        final_value, tokens, durations = _run_prepare_job(
            module._guided_ridge_prepare_snapshot_steps(obj, 0, 7)
        )
        assert final_value[1] is None
        assert any(
            isinstance(token, dict) and token.get("stage") == "finalizing snapshot"
            for token in tokens
        )
        max_slice = max(durations) if durations else 0.0
        assert max_slice < 0.1, max_slice
        return {
            "passed": True,
            "faces": len(mesh.polygons),
            "vertices": len(mesh.vertices),
            "non_identity_transform": True,
            "stages": tuple(token.get("stage") for token in tokens if isinstance(token, dict)),
            "max_slice_seconds": max_slice,
        }
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def _guided_ridge_bfs_fixture(module):
    """Exercise a non-256-face connected component and its pop-count yields."""
    import bpy

    width, height = 15, 10
    vertices = [(float(x), float(y), 0.0) for y in range(height + 1) for x in range(width + 1)]
    faces = []
    for y in range(height):
        for x in range(width):
            a = y * (width + 1) + x
            b = a + 1
            c = a + width + 2
            d = a + width + 1
            faces.extend(((a, b, c), (a, c, d)))
    mesh = bpy.data.meshes.new("_mfo_guided_ridge_bfs_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    attr = mesh.attributes.new(".sculpt_face_set", "INT", "FACE")
    for item in attr.data:
        item.value = 7
    obj = bpy.data.objects.new("_mfo_guided_ridge_bfs_object", mesh)
    try:
        baseline, baseline_reason = module._guided_ridge_prepare_snapshot_sync(obj, 0, 7)
        assert baseline_reason is None, baseline_reason
        job = module._guided_ridge_prepare_snapshot_steps(obj, 0, 7)
        tokens = []
        while True:
            try:
                tokens.append(next(job))
            except StopIteration as complete:
                cooperative, reason = complete.value
                break
        assert reason is None, reason
        assert cooperative is not None
        assert cooperative["face_indices"] == baseline["face_indices"]
        assert cooperative["triangles"] == baseline["triangles"]
        component_progress = [
            int(token["done"])
            for token in tokens
            if isinstance(token, dict) and token.get("stage") == "finding connected component"
        ]
        assert 256 in component_progress
        assert component_progress[-1] == len(faces)
        assert max(
            right - left for left, right in zip(component_progress, component_progress[1:])
        ) <= module.GUIDED_RIDGE_PREPARE_WORK_CHUNK
        return {
            "passed": True,
            "faces": len(faces),
            "component_progress": component_progress,
            "max_pop_batch": max(
                right - left for left, right in zip(component_progress, component_progress[1:])
            ),
        }
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def _guided_ridge_surface_fixture(module):
    """Use a temporary cylindrical surface to verify crest-side planing."""
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    axial, radial = 17, 24
    points = np.asarray(
        [
            (np.cos(angle), np.sin(angle), z)
            for z in np.linspace(0.0, 1.0, axial)
            for angle in np.linspace(0.0, 2.0 * np.pi, radial, endpoint=False)
        ],
        dtype=np.float64,
    )
    vertex = lambda z, angle: int(z * radial + (angle % radial))
    triangles = []
    for z in range(axial - 1):
        for angle in range(radial):
            first = vertex(z, angle)
            second = vertex(z, angle + 1)
            third = vertex(z + 1, angle + 1)
            fourth = vertex(z + 1, angle)
            triangles.extend(((first, second, third), (first, third, fourth)))
    triangles = np.asarray(triangles, dtype=np.int64)
    boundary_edges = []
    boundary_vertices = []
    for angle in range(radial):
        boundary_edges.extend(
            ((vertex(0, angle), vertex(0, angle + 1)), (vertex(axial - 1, angle), vertex(axial - 1, angle + 1)))
        )
        boundary_vertices.extend((vertex(0, angle), vertex(axial - 1, angle)))
    adjacency = [set() for _ in points]
    for first, second, third in triangles:
        for left, right in ((first, second), (second, third), (third, first)):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
    snapshot = {
        "coords_world": points,
        "triangles": triangles,
        "triangle_face_indices": list(range(len(triangles))),
        "boundary_edges": np.asarray(boundary_edges, dtype=np.int64),
        "boundary_vertices": np.asarray(sorted(set(boundary_vertices)), dtype=np.int64),
        "vertex_indices": list(range(len(points))),
        "hidden_vertices": np.zeros(len(points), dtype=bool),
        "sculpt_mask": np.zeros(len(points), dtype=np.float64),
        "normals_world": np.asarray(
            [(np.cos(angle), np.sin(angle), 0.0) for z in np.linspace(0.0, 1.0, axial) for angle in np.linspace(0.0, 2.0 * np.pi, radial, endpoint=False)],
            dtype=np.float64,
        ),
        "adjacency": [tuple(sorted(values)) for values in adjacency],
        "average_edge": 0.27,
        "bvh": BVHTree.FromPolygons(points.tolist(), triangles.tolist()),
    }
    controls = [Vector((1.0, 0.0, z)) for z in np.linspace(0.12, 0.88, 5)]
    normals = [Vector((1.0, 0.0, 0.0)) for _ in controls]
    guide = list(controls)
    candidate, info = module._guided_ridge_exact_candidate(snapshot, guide, controls, normals)
    cooperative_job = module._guided_ridge_exact_candidate_steps(snapshot, guide, controls, normals)
    while True:
        try:
            next(cooperative_job)
        except StopIteration as complete:
            cooperative_candidate, cooperative_info = complete.value
            break
    assert np.array_equal(np.asarray(candidate), np.asarray(cooperative_candidate))
    for key in ("crest_side_mask", "protrusion_mask", "anchor_mask", "selected_arc_mask", "delta_h"):
        assert np.array_equal(np.asarray(info[key]), np.asarray(cooperative_info[key])), key
    delta = np.asarray(candidate) - points
    # Independent geometry oracle: this fixture is a unit cylinder with
    # outward radial normals.  A valid Guided Ridge result may only move
    # inward (never add material), must leave the end boundary rings and the
    # known crest samples untouched, and must not move the opposite half.
    radial_normals = np.asarray(
        [(np.cos(angle), np.sin(angle), 0.0) for _z in np.linspace(0.0, 1.0, axial)
         for angle in np.linspace(0.0, 2.0 * np.pi, radial, endpoint=False)],
        dtype=np.float64,
    )
    assert float(np.max(np.sum(delta * radial_normals, axis=1))) <= 1.0e-10
    boundary_rows = np.r_[np.arange(radial), np.arange((axial - 1) * radial, axial * radial)]
    assert float(np.max(np.linalg.norm(delta[boundary_rows], axis=1))) <= 1.0e-10
    guide_rows = [int(round(float(z) * (axial - 1))) * radial for z in np.linspace(0.12, 0.88, 5)]
    assert float(np.max(np.linalg.norm(delta[guide_rows], axis=1))) <= 1.0e-10
    backside = points[:, 0] < -0.25
    assert float(np.max(np.linalg.norm(delta[backside], axis=1))) <= 1.0e-10
    assert info["crest_side_mask"].any()
    assert np.max(np.linalg.norm(delta[~info["crest_side_mask"]], axis=1)) == 0.0
    assert np.max(np.linalg.norm(delta[info["anchor_mask"]], axis=1)) == 0.0
    assert not np.any(np.asarray(info["delta_h"]) * np.asarray(info["outward_component"]) > 1.0e-10)
    final_height = np.asarray(info["h"]) + np.asarray(info["delta_h"])
    protrusions = np.asarray(info["protrusion_mask"], dtype=bool)
    assert np.all(final_height[protrusions] >= np.asarray(info["target_h"])[protrusions] - 1.0e-9)
    integrity = module._guided_ridge_mesh_integrity(snapshot, points, candidate)
    assert integrity["finite"] and not integrity["normal_pair_flips"] and not integrity["new_degenerate_faces"]
    changed_values = np.linalg.norm(delta, axis=1)
    assert int(np.count_nonzero(changed_values > 1.0e-12)) > 0
    return {
        "passed": True,
        "valid_mask_vertices": int(np.count_nonzero(info["crest_side_mask"])),
        "backside_fixed": True,
        "anchors_fixed": True,
        "planing_only": True,
        "changed_vertices": int(np.count_nonzero(changed_values > 1.0e-12)),
        "changed_median": float(np.median(changed_values[changed_values > 1.0e-12])),
        "changed_max": float(np.max(changed_values)),
        "integrity": integrity,
    }


def _guided_ridge_open_surface_fixture(module):
    """Regression fixture for an open V-shaped Face Set strip."""
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    rows = 17
    points = np.asarray(
        [(x, height, z) for z in np.linspace(0.0, 1.0, rows) for x, height in ((-1.0, 0.0), (0.0, 0.8), (1.0, 0.0))],
        dtype=np.float64,
    )
    triangles = []
    for row in range(rows - 1):
        left, crest, right = 3 * row, 3 * row + 1, 3 * row + 2
        next_left, next_crest, next_right = 3 * (row + 1), 3 * (row + 1) + 1, 3 * (row + 1) + 2
        triangles.extend(((left, crest, next_crest), (left, next_crest, next_left), (crest, right, next_right), (crest, next_right, next_crest)))
    triangles = np.asarray(triangles, dtype=np.int64)
    boundary_edges = []
    for row in (0, rows - 1):
        boundary_edges.extend(((3 * row, 3 * row + 1), (3 * row + 1, 3 * row + 2)))
    for column in range(3):
        boundary_edges.extend((3 * row + column, 3 * (row + 1) + column) for row in range(rows - 1))
    adjacency = [set() for _ in points]
    for first, second, third in triangles:
        for left, right in ((first, second), (second, third), (third, first)):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
    global_ids = np.asarray([1000 + 7 * index for index in range(len(points))], dtype=np.int64)
    snapshot = {
        "coords_world": points,
        "triangles": triangles,
        "triangle_face_indices": list(range(len(triangles))),
        "boundary_edges": np.asarray(boundary_edges, dtype=np.int64),
        # Deliberately non-contiguous mesh-global IDs exercise the explicit
        # global->component-local boundary conversion.
        "boundary_vertices": np.asarray(
            sorted({int(global_ids[value]) for edge in boundary_edges for value in edge}),
            dtype=np.int64,
        ),
        "vertex_indices": list(global_ids),
        "hidden_vertices": np.zeros(len(points), dtype=bool),
        "sculpt_mask": np.zeros(len(points), dtype=np.float64),
        "normals_world": np.asarray([(0.0, 1.0, 0.0) for _ in points], dtype=np.float64),
        "adjacency": [tuple(sorted(values)) for values in adjacency],
        "average_edge": 0.2,
        "bvh": BVHTree.FromPolygons(points.tolist(), triangles.tolist()),
    }
    controls = [Vector((0.0, 0.8, z)) for z in np.linspace(0.12, 0.88, 5)]
    candidate, info = module._guided_ridge_exact_candidate(snapshot, controls, controls, [Vector((0.0, 1.0, 0.0))] * 5)
    delta = np.asarray(candidate) - points
    boundary_local = sorted({int(value) for edge in boundary_edges for value in edge})
    assert float(np.max(np.linalg.norm(delta[boundary_local], axis=1))) <= 1.0e-10
    # The fixture normal points +Y, so the independent planing oracle rejects
    # any outward/material-adding displacement regardless of implementation
    # masks.
    assert float(np.max(delta[:, 1])) <= 1.0e-10
    assert info["crest_side_mask"].any()
    return {"passed": True, "open_surface": True, "crest_mask_vertices": int(np.count_nonzero(info["crest_side_mask"]))}


def _guided_ridge_ambiguous_surface_fixture(module):
    """Reject two equally-close surface sheets instead of global-extrema fallback."""
    import numpy as np

    # Two parallel open sheets at x=-1 and x=+1.  A station plane normal to
    # the z guide intersects both at the same distance from the crest, so a
    # deterministic section source cannot be selected safely.
    points = np.asarray(
        [
            (-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (-1.0, 1.0, 1.0), (-1.0, -1.0, 1.0),
            (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0),
        ],
        dtype=np.float64,
    )
    triangles = np.asarray(
        [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)],
        dtype=np.int64,
    )
    boundary_edges = np.asarray(
        [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)],
        dtype=np.int64,
    )
    frame = {
        "guide": np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64),
        "arc": np.asarray((0.0, 1.0), dtype=np.float64),
        "tangent": np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)), dtype=np.float64),
        "normal": np.asarray(((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64),
        "lateral": np.asarray((-1.0, 0.0, 0.0), dtype=np.float64)[None, :].repeat(2, axis=0),
        "length": 1.0,
        "average_edge": 1.0,
    }
    job = module._guided_ridge_surface_section_record_job(
        points,
        triangles,
        boundary_edges,
        np.arange(len(points), dtype=np.int64),
        frame,
        0.5,
    )
    while True:
        try:
            next(job)
        except StopIteration as complete:
            record, reason = complete.value
            break
    assert record is None
    assert "equally-close" in (reason or "") or "ambiguous" in (reason or "")
    return {"passed": True, "rejected_reason": reason}


def _guided_ridge_width_density_fixture(module):
    """The same cylindrical surface at two tessellation densities has one width domain."""
    import numpy as np
    from mathutils import Vector

    def make_snapshot(radial):
        axial = 17
        def radius(angle):
            # Deliberate, known protrusions near the minimum-width rails.  The
            # oracle below can therefore prove that a narrow but valid width
            # still produces visible planing instead of a safety no-op.
            return 1.22 if np.cos(angle) > 0.0 and abs(abs(np.sin(angle)) - 0.18) < 0.08 else 1.0

        points = np.asarray(
            [
                (radius(angle) * np.cos(angle), radius(angle) * np.sin(angle), z)
                for z in np.linspace(0.0, 1.0, axial)
                for angle in np.linspace(0.0, 2.0 * np.pi, radial, endpoint=False)
            ],
            dtype=np.float64,
        )
        vertex = lambda z, angle: int(z * radial + (angle % radial))
        triangles = []
        for z in range(axial - 1):
            for angle in range(radial):
                a, b = vertex(z, angle), vertex(z, angle + 1)
                c, d = vertex(z + 1, angle + 1), vertex(z + 1, angle)
                triangles.extend(((a, b, c), (a, c, d)))
        triangles = np.asarray(triangles, dtype=np.int64)
        boundary_edges = []
        for angle in range(radial):
            boundary_edges.extend(
                ((vertex(0, angle), vertex(0, angle + 1)),
                 (vertex(axial - 1, angle), vertex(axial - 1, angle + 1)))
            )
        boundary_vertices = sorted({value for edge in boundary_edges for value in edge})
        adjacency = [set() for _ in points]
        for first, second, third in triangles:
            for left, right in ((first, second), (second, third), (third, first)):
                adjacency[int(left)].add(int(right))
                adjacency[int(right)].add(int(left))
        edge_lengths = np.concatenate(
            (
                np.linalg.norm(points[triangles[:, 1]] - points[triangles[:, 0]], axis=1),
                np.linalg.norm(points[triangles[:, 2]] - points[triangles[:, 1]], axis=1),
                np.linalg.norm(points[triangles[:, 0]] - points[triangles[:, 2]], axis=1),
            )
        )
        actual_average_edge = float(np.mean(edge_lengths))
        return {
            "coords_world": points,
            "triangles": triangles,
            "triangle_face_indices": list(range(len(triangles))),
            "boundary_edges": np.asarray(boundary_edges, dtype=np.int64),
            "boundary_vertices": np.asarray(boundary_vertices, dtype=np.int64),
            "vertex_indices": list(range(len(points))),
            "hidden_vertices": np.zeros(len(points), dtype=bool),
            "sculpt_mask": np.zeros(len(points), dtype=np.float64),
            "normals_world": np.asarray(
                [
                    tuple(
                        np.asarray((points[index][0], points[index][1], 0.0), dtype=np.float64)
                        / max(float(np.linalg.norm(points[index][:2])), 1.0e-12)
                    )
                    for index in range(len(points))
                ],
                dtype=np.float64,
            ),
            "adjacency": [tuple(sorted(values)) for values in adjacency],
            "average_edge": actual_average_edge,
        }

    controls = [Vector((1.0, 0.0, z)) for z in np.linspace(0.12, 0.88, 5)]
    normals = [Vector((1.0, 0.0, 0.0)) for _ in controls]
    results = []
    for radial in (17, 64):
        snapshot = make_snapshot(radial)
        candidate, info = module._guided_ridge_exact_candidate(
            snapshot, controls, controls, normals, half_width=0.9
        )
        mask = np.asarray(info["crest_side_mask"], dtype=bool)
        changed = np.linalg.norm(np.asarray(candidate) - snapshot["coords_world"], axis=1) > 1.0e-12
        angular = np.mod(np.arctan2(snapshot["coords_world"][:, 1], snapshot["coords_world"][:, 0]), 2.0 * np.pi)
        normal_values = np.linalg.norm(np.asarray(candidate) - snapshot["coords_world"], axis=1)
        normal_threshold = max(float(snapshot["average_edge"]) * 0.02, 1.0e-8)
        assert int(np.count_nonzero(normal_values > 1.0e-12)) > 0
        minimum_width = module._guided_ridge_width_limits(snapshot)[1]
        if radial >= 64:
            minimum_candidate, minimum_info = module._guided_ridge_exact_candidate(
                snapshot, controls, controls, normals, half_width=minimum_width
            )
            minimum_values = np.linalg.norm(np.asarray(minimum_candidate) - snapshot["coords_world"], axis=1)
            minimum_changed = minimum_values > 1.0e-12
            minimum_significant = minimum_changed & (minimum_values > normal_threshold)
            assert int(np.count_nonzero(minimum_changed)) > 0
            assert int(np.count_nonzero(minimum_significant)) > 0
        else:
            minimum_values = np.zeros(len(snapshot["coords_world"]), dtype=np.float64)
            minimum_changed = np.zeros(len(snapshot["coords_world"]), dtype=bool)
            minimum_significant = minimum_changed
        results.append(
            {
                "radial": radial,
                "eligible_fraction": float(np.mean(mask)),
                "changed_fraction": float(np.mean(changed)),
                "changed_angular_extent": float(np.max(angular[changed]) - np.min(angular[changed])) if np.any(changed) else 0.0,
                "average_edge": float(snapshot["average_edge"]),
                "minimum_width": float(minimum_width),
                "normal_changed_vertices": int(np.count_nonzero(changed)),
                "normal_changed_median": float(np.median(normal_values[normal_values > 1.0e-12])),
                "normal_changed_max": float(np.max(normal_values)),
                "minimum_changed_vertices": int(np.count_nonzero(minimum_changed)),
                "minimum_significant_vertices": int(np.count_nonzero(minimum_significant)),
                "minimum_changed_median": float(np.median(minimum_values[minimum_changed])) if np.any(minimum_changed) else 0.0,
                "minimum_changed_max": float(np.max(minimum_values)),
                "minimum_width_tested": bool(radial >= 64),
            }
        )
    assert abs(results[0]["eligible_fraction"] - results[1]["eligible_fraction"]) < 0.20
    assert abs(results[0]["changed_fraction"] - results[1]["changed_fraction"]) < 0.20
    return {"passed": True, "densities": results}


def _guided_ridge_width_ambiguous_fixture(module):
    """Reject two front-facing sheets sharing the same requested width rail."""
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    rows = 17
    points = np.asarray(
        [
            (x, sheet_y, z)
            for sheet_y in (0.5, 1.2)
            for z in np.linspace(0.0, 1.0, rows)
            for x in (-0.2, 0.2)
        ],
        dtype=np.float64,
    )
    triangles = []
    boundary_edges = []
    for sheet in range(2):
        base = sheet * rows * 2
        for row in range(rows - 1):
            left, right = base + row * 2, base + row * 2 + 1
            next_left, next_right = base + (row + 1) * 2, base + (row + 1) * 2 + 1
            # Winding is +Y for both sheets: the two surfaces are equally
            # eligible but are separated in height.
            triangles.extend(((left, next_left, next_right), (left, next_right, right)))
            boundary_edges.extend(((left, next_left), (right, next_right)))
        boundary_edges.extend(
            ((base, base + 1), (base + (rows - 1) * 2, base + (rows - 1) * 2 + 1))
        )
    triangles = np.asarray(triangles, dtype=np.int64)
    boundary_edges = np.asarray(boundary_edges, dtype=np.int64)
    adjacency = [set() for _ in points]
    for first, second, third in triangles:
        for left, right in ((first, second), (second, third), (third, first)):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
    # The layers are connected only at the far end of the strip.  The local
    # station still contains two equally oriented surfaces with a gap smaller
    # than the default requested width, but larger than the actual edge-size
    # conflict tolerance, which must be rejected rather than averaged.
    first_far = (rows - 1) * 2
    second_far = rows * 2 + first_far
    for left, right in ((first_far, second_far), (first_far + 1, second_far + 1)):
        adjacency[left].add(right)
        adjacency[right].add(left)
    edge_lengths = np.concatenate(
        (
            np.linalg.norm(points[triangles[:, 1]] - points[triangles[:, 0]], axis=1),
            np.linalg.norm(points[triangles[:, 2]] - points[triangles[:, 1]], axis=1),
            np.linalg.norm(points[triangles[:, 0]] - points[triangles[:, 2]], axis=1),
        )
    )
    actual_average_edge = float(np.mean(edge_lengths))
    snapshot = {
        "coords_world": points,
        "triangles": triangles,
        "triangle_face_indices": list(range(len(triangles))),
        "boundary_edges": boundary_edges,
        "boundary_vertices": np.arange(len(points), dtype=np.int64),
        "vertex_indices": np.arange(len(points), dtype=np.int64),
        "hidden_vertices": np.zeros(len(points), dtype=bool),
        "sculpt_mask": np.zeros(len(points), dtype=np.float64),
        "normals_world": np.asarray([(0.0, 1.0, 0.0)] * len(points), dtype=np.float64),
        "adjacency": [tuple(sorted(values)) for values in adjacency],
        "average_edge": actual_average_edge,
        "bvh": BVHTree.FromPolygons(points.tolist(), triangles.tolist()),
    }
    controls = [Vector((0.0, 0.5, z)) for z in np.linspace(0.1, 0.9, 5)]
    normals = [Vector((0.0, 1.0, 0.0)) for _ in controls]
    def reject_reason(current_snapshot):
        try:
            module._guided_ridge_exact_candidate(current_snapshot, controls, controls, normals)
        except ValueError as error:
            reason = str(error)
            assert any(token in reason.lower() for token in ("ambiguous", "competing", "layer", "gap"))
            return reason
        raise AssertionError("parallel width sheets were not rejected")

    base_reason = reject_reason(snapshot)
    # Reorder component-local vertices while preserving all topology and
    # adjacency.  The conflict decision must remain identical because height
    # values and vertex indices stay paired through sorting/partitioning.
    permutation = np.roll(np.arange(len(points), dtype=np.int64), 7)
    inverse = np.empty(len(points), dtype=np.int64)
    inverse[permutation] = np.arange(len(points), dtype=np.int64)
    shuffled_adjacency = [
        tuple(sorted(int(inverse[value]) for value in adjacency[int(old_index)]))
        for old_index in permutation
    ]
    shuffled = dict(snapshot)
    shuffled.update(
        {
            "coords_world": points[permutation],
            "triangles": inverse[triangles],
            "boundary_edges": inverse[boundary_edges],
            "boundary_vertices": inverse[snapshot["boundary_vertices"]],
            "vertex_indices": np.asarray(snapshot["vertex_indices"])[permutation],
            "hidden_vertices": snapshot["hidden_vertices"][permutation],
            "sculpt_mask": snapshot["sculpt_mask"][permutation],
            "normals_world": snapshot["normals_world"][permutation],
            "adjacency": shuffled_adjacency,
        }
    )
    shuffled_reason = reject_reason(shuffled)
    assert any(token in base_reason.lower() for token in ("competing surface layers", "local surface gap", "ambiguous local surface layer"))
    assert any(token in shuffled_reason.lower() for token in ("competing surface layers", "local surface gap", "ambiguous local surface layer"))
    return {
        "passed": True,
        "rejected_reason": base_reason,
        "shuffled_rejected_reason": shuffled_reason,
        "average_edge_from_geometry": actual_average_edge,
        "default_width": module._guided_ridge_width_limits(snapshot)[0],
    }


def _guided_ridge_projected_rail_support_fixture(module):
    """Use actual rail q/h endpoints and preserve an edge/interior barycentric hit."""
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    rows = 9
    x_values = np.asarray((-0.75, -0.375, 0.0, 0.375, 0.75), dtype=np.float64)
    height_values = np.asarray((0.20, 0.52, 0.50, 0.52, 0.20), dtype=np.float64)
    points = np.asarray(
        [
            (float(x), float(height), float(z))
            for z in np.linspace(0.0, 1.0, rows)
            for x, height in zip(x_values, height_values)
        ],
        dtype=np.float64,
    )
    triangles = []
    boundary_edges = []
    for row in range(rows - 1):
        for column in range(len(x_values) - 1):
            a = row * len(x_values) + column
            b = a + 1
            c = a + len(x_values) + 1
            d = a + len(x_values)
            triangles.extend(((a, b, c), (a, c, d)))
        boundary_edges.extend(
            (
                (row * len(x_values), (row + 1) * len(x_values)),
                (row * len(x_values) + len(x_values) - 1, (row + 1) * len(x_values) + len(x_values) - 1),
            )
        )
    boundary_edges.extend(
        (column, column + 1) for column in range(len(x_values) - 1)
    )
    boundary_edges.extend(
        ((rows - 1) * len(x_values) + column, (rows - 1) * len(x_values) + column + 1)
        for column in range(len(x_values) - 1)
    )
    triangles = np.asarray(triangles, dtype=np.int64)
    boundary_edges = np.asarray(boundary_edges, dtype=np.int64)
    adjacency = [set() for _ in points]
    for first, second, third in triangles:
        for left, right in ((first, second), (second, third), (third, first)):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
    edge_lengths = np.concatenate(
        (
            np.linalg.norm(points[triangles[:, 1]] - points[triangles[:, 0]], axis=1),
            np.linalg.norm(points[triangles[:, 2]] - points[triangles[:, 1]], axis=1),
            np.linalg.norm(points[triangles[:, 0]] - points[triangles[:, 2]], axis=1),
        )
    )
    snapshot = {
        "coords_world": points,
        "triangles": triangles,
        "triangle_face_indices": list(range(len(triangles))),
        "boundary_edges": boundary_edges,
        "boundary_vertices": np.unique(boundary_edges.reshape(-1)),
        "vertex_indices": np.arange(len(points), dtype=np.int64),
        "hidden_vertices": np.zeros(len(points), dtype=bool),
        "sculpt_mask": np.zeros(len(points), dtype=np.float64),
        "normals_world": np.asarray([(0.0, 1.0, 0.0)] * len(points), dtype=np.float64),
        "adjacency": [tuple(sorted(values)) for values in adjacency],
        "average_edge": float(np.mean(edge_lengths)),
        "bvh": BVHTree.FromPolygons(points.tolist(), triangles.tolist()),
    }
    controls = [Vector((0.0, 0.5, z)) for z in np.linspace(0.12, 0.88, 5)]
    normals = [Vector((0.0, 1.0, 0.0))] * len(controls)
    guide = list(controls)
    width_result = module._guided_ridge_width_frame_result(
        snapshot, guide, controls, normals, half_width=None
    )
    candidate, info = module._guided_ridge_exact_candidate(
        snapshot,
        guide,
        controls,
        normals,
        width_result=width_result,
    )
    assert width_result["rail_hit_ok"]
    assert all(
        hit is not None
        and len(hit["triangle_vertices"]) == 3
        and len(hit["barycentric_weights"]) == 3
        and abs(sum(hit["barycentric_weights"]) - 1.0) < 1.0e-6
        for side in ("left", "right")
        for hit in width_result["rail_hits"][side]
    )
    edge_hits = [
        hit
        for side in ("left", "right")
        for hit in width_result["rail_hits"][side]
        if hit.get("edge_index") is not None
    ]
    assert edge_hits
    rail_q = width_result["rail_anchor_lateral"]
    rail_h = width_result["rail_anchor_height"]
    width = float(width_result["width"])
    assert np.max(np.abs(np.asarray(rail_q["left"]) + width)) > 1.0e-3
    assert np.max(np.abs(np.asarray(rail_q["right"]) - width)) > 1.0e-3
    assert np.all(np.asarray(rail_q["left"]) < 0.0)
    assert np.all(np.asarray(rail_q["right"]) > 0.0)
    # Independent target-plane oracle: q=0,h=0 and each actual projected rail
    # endpoint must lie exactly on its corresponding side plane.
    left_q = np.asarray(info["left_rail_q"], dtype=np.float64)
    right_q = np.asarray(info["right_rail_q"], dtype=np.float64)
    left_h = np.interp(info["station"], width_result["frame"]["arc"], np.asarray(rail_h["left"]))
    right_h = np.interp(info["station"], width_result["frame"]["arc"], np.asarray(rail_h["right"]))
    expected_target = np.where(
        info["q"] <= 0.0,
        info["q"] / left_q * left_h,
        info["q"] / right_q * right_h,
    )
    assert np.allclose(info["target_h"], expected_target, atol=1.0e-10)
    assert float(np.max(np.abs(expected_target - np.where(
        info["q"] <= 0.0,
        info["q"] / -max(width, 1.0e-12) * left_h,
        info["q"] / max(width, 1.0e-12) * right_h,
    )))) > 1.0e-4
    before = np.asarray(snapshot["coords_world"], dtype=np.float64)
    max_support_motion = 0.0
    support_count = 0
    for side in ("left", "right"):
        for hit in width_result["rail_hits"][side]:
            vertices = np.asarray(hit["triangle_vertices"], dtype=np.int64)
            weights = np.asarray(hit["barycentric_weights"], dtype=np.float64)
            pre = np.sum(before[vertices] * weights[:, None], axis=0)
            post = np.sum(np.asarray(candidate)[vertices] * weights[:, None], axis=0)
            max_support_motion = max(max_support_motion, float(np.linalg.norm(post - pre)))
            support_count += len(hit["support_vertex_indices"])
    assert support_count >= 2
    assert max_support_motion <= 1.0e-9
    assert module._guided_ridge_rail_support_unchanged(
        snapshot, before, candidate, info["rail_hits"]
    )
    return {
        "passed": True,
        "width": width,
        "actual_left_q_range": (float(np.min(rail_q["left"])), float(np.max(rail_q["left"]))),
        "actual_right_q_range": (float(np.min(rail_q["right"])), float(np.max(rail_q["right"]))),
        "support_vertex_count": int(support_count),
        "edge_hit_count": len(edge_hits),
        "max_barycentric_support_motion": max_support_motion,
        "changed_vertices": int(np.count_nonzero(np.linalg.norm(np.asarray(candidate) - before, axis=1) > 1.0e-12)),
    }


def _guided_ridge_tapered_rail_fixture(module):
    """Surface availability tapers rails at a narrowing mesh tip."""
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    rows, columns = 33, 21
    z_values = np.linspace(0.0, 1.0, rows)
    scales = np.interp(
        z_values,
        np.linspace(0.0, 1.0, 9),
        np.asarray((0.08, 0.35, 0.75, 1.0, 1.0, 1.0, 0.75, 0.35, 0.08)),
    )
    x_values = np.linspace(-0.9, 0.9, columns)
    heights = 0.2 + 0.3 * (1.0 - (x_values / 0.9) ** 2)
    points = np.asarray(
        [
            (float(x * scale), float(height), float(z))
            for scale, z in zip(scales, z_values)
            for x, height in zip(x_values, heights)
        ],
        dtype=np.float64,
    )
    triangles = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            b = a + 1
            c = a + columns + 1
            d = a + columns
            triangles.extend(((a, b, c), (a, c, d)))
    triangles = np.asarray(triangles, dtype=np.int64)
    boundary_edges = []
    for row in range(rows - 1):
        boundary_edges.extend(
            ((row * columns, (row + 1) * columns),
             (row * columns + columns - 1, (row + 1) * columns + columns - 1))
        )
    for column in range(columns - 1):
        boundary_edges.extend(
            ((column, column + 1),
             ((rows - 1) * columns + column, (rows - 1) * columns + column + 1))
        )
    boundary_edges = np.asarray(boundary_edges, dtype=np.int64)
    adjacency = [set() for _ in points]
    for first, second, third in triangles:
        for left, right in ((first, second), (second, third), (third, first)):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
    edge_lengths = np.concatenate(
        (
            np.linalg.norm(points[triangles[:, 1]] - points[triangles[:, 0]], axis=1),
            np.linalg.norm(points[triangles[:, 2]] - points[triangles[:, 1]], axis=1),
            np.linalg.norm(points[triangles[:, 0]] - points[triangles[:, 2]], axis=1),
        )
    )
    snapshot = {
        "coords_world": points,
        "triangles": triangles,
        "boundary_edges": boundary_edges,
        "boundary_vertices": np.unique(boundary_edges.reshape(-1)),
        "vertex_indices": np.arange(len(points), dtype=np.int64),
        "hidden_vertices": np.zeros(len(points), dtype=bool),
        "sculpt_mask": np.zeros(len(points), dtype=np.float64),
        "normals_world": np.tile((0.0, 1.0, 0.0), (len(points), 1)),
        "adjacency": [tuple(sorted(values)) for values in adjacency],
        "average_edge": float(np.mean(edge_lengths)),
        "bvh": BVHTree.FromPolygons(points.tolist(), triangles.tolist()),
    }
    controls = [Vector((0.0, 0.5, z)) for z in np.linspace(0.12, 0.88, 5)]
    normals = [Vector((0.0, 1.0, 0.0))] * len(controls)
    frame = module._guided_ridge_width_frame(snapshot, controls, controls, normals)
    width_result = module._guided_ridge_width_frame_result(
        snapshot, controls, controls, normals, 0.8, frame=frame
    )
    candidate, info = module._guided_ridge_exact_candidate(
        snapshot, controls, controls, normals, 0.8, width_result=width_result
    )
    left_q = np.asarray(width_result["rail_anchor_lateral"]["left"], dtype=np.float64)
    right_q = np.asarray(width_result["rail_anchor_lateral"]["right"], dtype=np.float64)
    assert np.all(left_q < 0.0) and np.all(right_q > 0.0)
    # The requested max width is wider than the tapered tip, so the first and
    # last projected rails must converge toward the crest while the middle
    # remains wider.  Candidate and support provenance use these same values.
    assert abs(left_q[0]) < abs(left_q[len(left_q) // 2])
    assert abs(right_q[-1]) < abs(right_q[len(right_q) // 2])
    assert module._guided_ridge_rail_support_unchanged(
        snapshot, snapshot["coords_world"], candidate, info["rail_hits"]
    )
    changed = np.linalg.norm(np.asarray(candidate) - snapshot["coords_world"], axis=1) > 1.0e-12
    assert int(np.count_nonzero(changed)) > 0
    return {
        "passed": True,
        "requested_max_width": float(width_result["width"]),
        "left_q": tuple(float(value) for value in left_q),
        "right_q": tuple(float(value) for value in right_q),
        "changed_vertices": int(np.count_nonzero(changed)),
    }


def _guided_ridge_width_controls_fixture(module):
    """Wheel changes only the preview width and preserves modal ownership."""
    import numpy as np
    from mathutils import Vector

    original_state = module._guided_ridge_state
    original_preview = module._guided_ridge_update_width_preview
    try:
        preview_calls = []
        state = {
            "active": True,
            "operator": SimpleNamespace(report=lambda *_args, **_kwargs: None),
            "phase": "ready",
            "start_guard_active": False,
            "controls": [Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))],
            "control_normals": [Vector((0.0, 1.0, 0.0))] * 2,
            "guide": [Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))],
            "half_width": 0.5,
            "width_min": 0.1,
            "width_max": 2.0,
            "snapshot": {"coords_world": np.zeros((2, 3)), "triangles": np.asarray([[0, 1, 1]])},
            "width_rails": {"left": [], "right": []},
        }
        module._guided_ridge_state = state
        module._guided_ridge_update_width_preview = lambda current: preview_calls.append(current["half_width"]) or True
        event = SimpleNamespace(type="WHEELUPMOUSE", value="PRESS", shift=False)
        result = module.VIEW3D_OT_mesh_focus_guided_ridge.modal(state["operator"], SimpleNamespace(), event)
        assert result == {"RUNNING_MODAL"}
        coarse = state["half_width"]
        event.shift = True
        module.VIEW3D_OT_mesh_focus_guided_ridge.modal(state["operator"], SimpleNamespace(), event)
        assert state["half_width"] > coarse and len(preview_calls) == 2
        return {"passed": True, "coarse_width": coarse, "fine_width": state["half_width"], "preview_calls": len(preview_calls)}
    finally:
        module._guided_ridge_update_width_preview = original_preview
        module._guided_ridge_state = original_state


def _guided_ridge_nonuniform_frame_fixture(module):
    """Non-uniform curved guide interpolation uses segment parameters, not a uniform index."""
    import numpy as np

    frame = {
        "guide": np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.2, 0.0, 0.6), (0.5, 0.2, 1.0)), dtype=float),
        "arc": np.asarray((0.0, 0.1, 0.6, 1.271779788), dtype=float),
        "tangent": np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)), dtype=float),
        "lateral": np.asarray(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)), dtype=float),
        "normal": np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=float),
    }
    segment_index = np.asarray((0, 1, 2), dtype=np.int64)
    segment_t = np.asarray((0.5, 0.25, 0.75), dtype=float)
    tangent, lateral, normal = module._guided_ridge_interpolate_vertex_frame(
        frame, segment_index, segment_t
    )
    expected_tangent = module._guided_ridge_unit_array(
        (1.0 - segment_t)[:, None] * frame["tangent"][segment_index]
        + segment_t[:, None] * frame["tangent"][segment_index + 1]
    )
    assert np.allclose(tangent, expected_tangent)
    uniform_index = np.floor(np.asarray((0.5, 0.75, 2.25))).astype(np.int64)
    uniform_t = np.asarray((0.5, 0.75, 0.25))
    uniform, _unused_lateral, _unused_normal = module._guided_ridge_interpolate_vertex_frame(
        frame, uniform_index, uniform_t
    )
    assert float(np.max(np.abs(tangent - uniform))) > 1.0e-3
    return {"passed": True, "nonuniform_diff": float(np.max(np.abs(tangent - uniform)))}


def _guided_ridge_width_cache_fixture(module):
    """Wheel events reuse the stable frame and only refresh rails/width."""
    import numpy as np
    from mathutils import Vector

    original_frame = module._guided_ridge_width_frame
    original_rails = module._guided_ridge_width_rails
    original_layers = module._guided_ridge_surface_layer_steps
    original_state = module._guided_ridge_state
    try:
        frame_calls = []
        rail_calls = []
        layer_calls = []
        frame = {
            "guide": np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=float),
            "arc": np.asarray((0.0, 1.0), dtype=float),
            "tangent": np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)), dtype=float),
            "lateral": np.asarray(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=float),
            "normal": np.asarray(((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)), dtype=float),
        }
        module._guided_ridge_width_frame = lambda *_args, **_kwargs: frame_calls.append(True) or frame
        module._guided_ridge_width_rails = lambda *_args, **_kwargs: rail_calls.append(True) or {"left": [Vector((0, 0, 0))] * 2, "right": [Vector((0, 0, 0))] * 2}
        def layer_job(*_args, **_kwargs):
            layer_calls.append(True)
            yield {"stage": "building width surface layers", "done": 0, "total": 4}
            return {"entries": {"left": [None, None], "right": [None, None]}, "reasons": {}, "available": True}
        module._guided_ridge_surface_layer_steps = layer_job
        state = {
            "active": True,
            "operator": SimpleNamespace(report=lambda *_args, **_kwargs: None),
            "phase": "ready",
            "start_guard_active": False,
            "controls": [Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))],
            "control_normals": [Vector((0.0, 1.0, 0.0))] * 2,
            "guide": [Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))],
            "half_width": 0.5,
            "width_min": 0.1,
            "width_max": 1.0,
            "snapshot": {
                "coords_world": np.asarray(((0, 0, 0), (1, 0, 0), (0, 0, 1)), dtype=float),
                "triangles": np.asarray(((0, 1, 2),), dtype=np.int64),
                "average_edge": 0.5,
                "bvh": object(),
            },
            "width_frame_result": None,
            "width_rails": {"left": [], "right": []},
        }
        module._guided_ridge_state = state
        assert module._guided_ridge_update_width_preview(state)
        durations = []
        for index in range(8):
            event = SimpleNamespace(
                type="WHEELUPMOUSE" if index % 2 == 0 else "WHEELDOWNMOUSE",
                value="PRESS",
                shift=bool(index % 3 == 0),
            )
            started = time.perf_counter()
            assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
                state["operator"], SimpleNamespace(), event
            ) == {"RUNNING_MODAL"}
            durations.append(time.perf_counter() - started)
        assert len(frame_calls) == 1
        assert len(rail_calls) == 9
        assert len(layer_calls) == 1
        return {
            "passed": True,
            "frame_rebuilds": len(frame_calls),
            "rail_refreshes": len(rail_calls),
            "layer_builds": len(layer_calls),
            "wheel_event_count": len(durations),
            "wheel_p95_seconds": float(np.percentile(durations, 95)),
            "wheel_max_seconds": float(max(durations)),
        }
    finally:
        module._guided_ridge_width_frame = original_frame
        module._guided_ridge_width_rails = original_rails
        module._guided_ridge_surface_layer_steps = original_layers
        module._guided_ridge_state = original_state


def _guided_ridge_aligned_prepare_equivalence_fixture(module):
    """Sync/cooperative layer provenance uses the same aligned frame."""
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    points = np.asarray(
        tuple(
            (x, 0.2, z)
            for z in (0.0, 1.0)
            for x in (-1.0, -0.5, 0.5, 1.0)
        ),
        dtype=np.float64,
    )
    triangles = np.asarray(
        ((0, 1, 4), (1, 5, 4), (1, 2, 5), (2, 6, 5), (2, 3, 6), (3, 7, 6)),
        dtype=np.int64,
    )
    adjacency = [set() for _ in points]
    for first, second, third in triangles:
        for left, right in ((first, second), (second, third), (third, first)):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
    snapshot = {
        "coords_world": points,
        "triangles": triangles,
        "adjacency": [tuple(sorted(values)) for values in adjacency],
        "normals_world": np.asarray(((0.0, 1.0, 0.0),) * len(points)),
        "average_edge": float(np.mean(np.linalg.norm(points[triangles[:, 1]] - points[triangles[:, 0]], axis=1))),
        "bvh": BVHTree.FromPolygons(points.tolist(), triangles.tolist()),
    }
    guide = [Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))]
    controls = list(guide)
    control_normals = [Vector((0.0, 1.0, 0.0))] * 2
    base_frame = {
        "guide": np.asarray(tuple(tuple(value) for value in guide), dtype=np.float64),
        "arc": np.asarray((0.0, 1.0), dtype=np.float64),
        "tangent": np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)), dtype=np.float64),
        "lateral": np.asarray(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64),
        # Deliberately reversed: alignment must flip this before layer build.
        "normal": np.asarray(((0.0, -1.0, 0.0), (0.0, -1.0, 0.0)), dtype=np.float64),
    }

    def clone_frame():
        return {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in base_frame.items()
        }

    original_sync = module._guided_ridge_stable_frame
    original_steps = module._guided_ridge_stable_frame_steps
    try:
        module._guided_ridge_stable_frame = lambda *_args, **_kwargs: clone_frame()

        def stable_steps(*_args, **_kwargs):
            yield {"stage": "building guide frame", "done": 0, "total": 1}
            return clone_frame()

        module._guided_ridge_stable_frame_steps = stable_steps
        sync_frame = module._guided_ridge_width_frame(snapshot, guide, controls, control_normals)
        sync_result = module._guided_ridge_width_frame_result(
            snapshot, guide, controls, control_normals, 0.8, frame=sync_frame
        )
        prepare_job = module._guided_ridge_width_prepare_steps(
            snapshot,
            module._guided_ridge_stable_frame_steps(
                points, triangles, guide, controls, control_normals
            ),
            controls,
            control_normals,
        )
        while True:
            try:
                next(prepare_job)
            except StopIteration as complete:
                cooperative_frame, cooperative_layers = complete.value
                break
        cooperative_result = module._guided_ridge_width_frame_result(
            snapshot,
            guide,
            controls,
            control_normals,
            0.8,
            frame=cooperative_frame,
            layer_result=cooperative_layers,
        )
        assert np.allclose(sync_frame["normal"], cooperative_frame["normal"])
        assert np.allclose(sync_frame["lateral"], cooperative_frame["lateral"])
        for side in ("left", "right"):
            sync_entries = sync_result["layer_result"]["entries"][side]
            cooperative_entries = cooperative_layers["entries"][side]
            assert [entry["vertices"] if entry else None for entry in sync_entries] == [
                entry["vertices"] if entry else None for entry in cooperative_entries
            ]
            assert np.allclose(sync_result["rail_anchor_height"][side], cooperative_result["rail_anchor_height"][side])
            assert np.allclose(sync_result["rail_anchor_lateral"][side], cooperative_result["rail_anchor_lateral"][side])
            for left_hit, right_hit in zip(sync_result["rail_hits"][side], cooperative_result["rail_hits"][side]):
                assert left_hit["triangle_index"] == right_hit["triangle_index"]
                assert np.allclose(left_hit["barycentric_weights"], right_hit["barycentric_weights"])
        return {"passed": True, "normal_flipped_before_layers": True, "frame_equal": True, "provenance_equal": True}
    finally:
        module._guided_ridge_stable_frame = original_sync
        module._guided_ridge_stable_frame_steps = original_steps


def _guided_ridge_width_frame_rebuild_fixture(module):
    """Guide edits rebuild the frame cooperatively and Esc clears the job."""
    import numpy as np
    from mathutils import Vector

    original_state = module._guided_ridge_state
    original_steps = module._guided_ridge_stable_frame_steps
    original_matches = module._guided_ridge_context_matches
    original_result = module._guided_ridge_width_frame_result
    original_overlay = module._guided_ridge_overlay_tag
    try:
        frame = {
            "guide": np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=float),
            "arc": np.asarray((0.0, 1.0), dtype=float),
            "tangent": np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)), dtype=float),
            "lateral": np.asarray(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=float),
            "normal": np.asarray(((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)), dtype=float),
        }

        def frame_job(*_args):
            yield {"stage": "building guide frame", "done": 0, "total": 0, "indeterminate": True}
            return frame

        class _WM:
            def __init__(self):
                self.added = 0
                self.removed = []

            def event_timer_add(self, *_args, **_kwargs):
                self.added += 1
                return object()

            def event_timer_remove(self, timer):
                self.removed.append(timer)

        wm = _WM()
        proxy = SimpleNamespace(report=lambda *_args, **_kwargs: None)
        context = SimpleNamespace(window_manager=wm, window=object())
        snapshot = {
            "context_signature": {"marker": 1},
            "coords_world": np.asarray(((0, 0, 0), (1, 0, 0), (0, 0, 1)), dtype=float),
            "triangles": np.asarray(((0, 1, 2),), dtype=np.int64),
            "average_edge": 0.5,
        }
        state = {
            "active": True,
            "operator": proxy,
            "phase": "ready",
            "snapshot": snapshot,
            "guide": [Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))],
            "controls": [Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))],
            "control_normals": [Vector((0.0, 1.0, 0.0))] * 2,
            "half_width": 0.5,
            "timer": None,
            "window_manager": wm,
            "width_frame_result": None,
            "width_frame_job": None,
            "width_rails": {"left": [], "right": []},
        }
        module._guided_ridge_state = state
        module._guided_ridge_stable_frame_steps = frame_job
        module._guided_ridge_context_matches = lambda *_args: True
        module._guided_ridge_overlay_tag = lambda _state: None
        module._guided_ridge_width_frame_result = lambda *_args, **_kwargs: {
            "frame": frame,
            "width": 0.5,
            "width_min": 0.1,
            "width_max": 1.0,
            "rails": {"left": [], "right": []},
            "projection_ok": True,
            "projection_required": False,
            "cache_key": (1,),
        }
        assert module._guided_ridge_begin_width_frame_rebuild(context, state)
        assert state["phase"] == "width_prepare" and state["width_frame_job"] is not None
        assert module._guided_ridge_process_width_frame_timer(context, state)
        assert state["phase"] == "width_prepare"
        for _ in range(8):
            if state["phase"] == "ready":
                break
            assert module._guided_ridge_process_width_frame_timer(context, state)
        assert state["phase"] == "ready" and state["width_frame_job"] is None
        assert wm.removed

        state["phase"] = "ready"
        assert module._guided_ridge_begin_width_frame_rebuild(context, state)
        module._guided_ridge_state = state
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            proxy, context, SimpleNamespace(type="ESC", value="PRESS")
        ) == {"CANCELLED"}
        assert module._guided_ridge_state is None and state.get("width_frame_job") is None
        return {"passed": True, "stage_tick_then_complete": True, "esc_cleanup": True}
    finally:
        module._guided_ridge_stable_frame_steps = original_steps
        module._guided_ridge_context_matches = original_matches
        module._guided_ridge_width_frame_result = original_result
        module._guided_ridge_overlay_tag = original_overlay
        module._guided_ridge_state = original_state


def _guided_ridge_live_snapshot_diagnostic(module):
    """Read-only regression check using the reported Hair/Face Set fixture."""
    import bpy
    import numpy as np
    from mathutils import Vector

    obj = bpy.data.objects.get("Mesh_00.Hair")
    if obj is None or obj.type != "MESH" or obj.data.attributes.get(".sculpt_face_set") is None:
        return {"skipped": "Mesh_00.Hair Face Set fixture unavailable"}
    snapshot, reason = module._guided_ridge_prepare_snapshot_sync(obj, 1573375, 89)
    if snapshot is None:
        return {"skipped": reason or "fixture snapshot unavailable"}
    controls = [
        Vector(value)
        for value in (
            (0.060091856867074966, -0.02921895682811737, 1.3922059535980225),
            (0.05857300013303757, -0.030253775417804718, 1.3919599056243896),
            (0.055876150727272034, -0.0321253277361393, 1.3916118144989014),
            (0.05278816819190979, -0.03375370800495148, 1.3917324542999268),
            (0.050072066485881805, -0.035010263323783875, 1.3920400142669678),
        )
    ]
    normals = [Vector((0.3040010631084442, -0.08999263495206833, -0.9484117031097412))] * len(controls)
    guide, project_reason = module._guided_ridge_project_curve(snapshot, controls)
    assert project_reason is None and guide is not None
    candidate, info = module._guided_ridge_exact_candidate(snapshot, guide, controls, normals)
    before = np.asarray(snapshot["coords_world"], dtype=np.float64)
    integrity = module._guided_ridge_mesh_integrity(snapshot, before, candidate)
    delta = np.linalg.norm(candidate - before, axis=1)
    assert integrity["finite"] and not integrity["normal_pair_flips"] and not integrity["new_degenerate_faces"]
    assert float(np.max(delta[~info["crest_side_mask"]])) == 0.0
    assert float(np.max(delta[info["anchor_mask"]])) == 0.0
    assert not np.any(np.asarray(info["delta_h"]) * np.asarray(info["outward_component"]) > 1.0e-10)
    changed_values = delta[delta > 1.0e-12]
    significant_threshold = max(float(snapshot.get("average_edge", 0.0)) * 0.02, 1.0e-8)
    significant = (delta > significant_threshold) & np.asarray(info["crest_side_mask"], dtype=bool)
    return {
        "passed": True,
        "faces": len(snapshot["face_indices"]),
        "vertices": len(snapshot["vertex_indices"]),
        "triangles": len(snapshot["triangles"]),
        "mask_vertices": int(np.count_nonzero(info["crest_side_mask"])),
        "changed_vertices": int(np.count_nonzero(delta > 1.0e-12)),
        "changed_median": float(np.median(changed_values)) if len(changed_values) else 0.0,
        "changed_max": float(np.max(changed_values)) if len(changed_values) else 0.0,
        "significant_eligible_fraction": float(
            np.count_nonzero(significant) / max(int(np.count_nonzero(info["crest_side_mask"])), 1)
        ),
        "integrity_scale": info["integrity_scale"],
        "backside_fixed": True,
        "anchors_fixed": True,
        "planing_only": True,
    }


def _guided_ridge_compute_lifecycle(module):
    """Verify candidate work is timer-owned and Esc clears runtime state."""
    from mathutils import Vector

    original_job = module._guided_ridge_exact_candidate_steps
    original_matches = module._guided_ridge_context_matches
    original_overlay = module._guided_ridge_overlay_tag
    original_state = module._guided_ridge_state
    removed = []
    events = []
    try:
        def job(*_args, **_kwargs):
            yield {"stage": "stage-a", "stage_index": 1, "stage_count": 4, "done": 0, "total": 1}
            yield {"stage": "stage-b", "stage_index": 2, "stage_count": 4, "done": 0, "total": 1}
            yield {"stage": "stage-c", "stage_index": 3, "stage_count": 4, "done": 0, "total": 1}
            return ([], {"delta_h": [], "outward_component": []})

        class _WM:
            def event_timer_add(self, *_args, **_kwargs):
                return object()

            def event_timer_remove(self, timer):
                removed.append(timer)

        def _commit(_context, _state, **kwargs):
            # Mirror the real successful commit contract: the write path marks
            # the state committed and tears down its active runtime before
            # returning to the modal dispatcher.
            events.append(kwargs)
            _state["committed"] = True
            _state["active"] = False
            return True

        proxy = SimpleNamespace(_commit=_commit, report=lambda *_args, **_kwargs: None)
        context = SimpleNamespace(window_manager=_WM(), window=object())
        state = {
            "active": True,
            "phase": "ready",
            "operator": proxy,
            "snapshot": {"context_signature": {"marker": 1}},
            "guide": [Vector((0.0, 0.0, 0.0))],
            "controls": [Vector((0.0, 0.0, 0.0))],
            "control_normals": [Vector((0.0, 0.0, 1.0))],
            "timer": None,
            "window_manager": context.window_manager,
            "area": None,
            "compute_stage": "idle",
            "compute_max_slice_seconds": 0.0,
            "compute_elapsed_seconds": 0.0,
            "modal_result": None,
            "committed": False,
        }
        module._guided_ridge_exact_candidate_steps = job
        module._guided_ridge_context_matches = lambda *_args: True
        module._guided_ridge_overlay_tag = lambda _state: None
        module._guided_ridge_state = state
        assert module._guided_ridge_begin_compute(context, state)
        assert state["phase"] == "compute" and state["timer"] is not None
        timer_event = lambda: SimpleNamespace(type="TIMER", value="NOTHING", timer=state["timer"])
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(proxy, context, timer_event()) == {"RUNNING_MODAL"}
        assert state["phase"] == "compute" and not events
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(proxy, context, timer_event()) == {"RUNNING_MODAL"}
        assert not events
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(proxy, context, timer_event()) == {"RUNNING_MODAL"}
        assert not events
        # The third timer tick reaches StopIteration, invokes the real modal
        # completion path, and must return FINISHED even though commit cleanup
        # has already set active=False.
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(proxy, context, timer_event()) == {"FINISHED"}
        assert events and events[0].get("_compute_sync") is True
        assert state["committed"] and state["modal_result"] == "FINISHED"
        state["phase"] = "compute"
        state["active"] = True
        state["modal_result"] = None
        state["timer"] = object()
        module._guided_ridge_state = state
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            proxy, context, SimpleNamespace(type="ESC", value="PRESS")
        ) == {"CANCELLED"}
        assert module._guided_ridge_state is None and removed
        return {"passed": True, "timer_removed": True, "esc_cleanup": True}
    finally:
        module._guided_ridge_exact_candidate_steps = original_job
        module._guided_ridge_context_matches = original_matches
        module._guided_ridge_overlay_tag = original_overlay
        module._guided_ridge_state = original_state


def _guided_ridge_commit_fixture(module):
    """Verify the approved modal Enter path writes one bounded candidate."""
    import bpy
    import numpy as np
    from mathutils import Vector

    assert module.GUIDED_RIDGE_UI_PROTOTYPE is False
    assert "UNDO" in module.VIEW3D_OT_mesh_focus_guided_ridge.bl_options
    mesh = bpy.data.meshes.new("_mfo_guided_ridge_commit_mesh")
    mesh.from_pydata(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        [],
        [(0, 1, 2)],
    )
    mesh.update()
    obj = bpy.data.objects.new("_mfo_guided_ridge_commit_object", mesh)
    original_state = module._guided_ridge_state
    original_last = module._guided_ridge_last_guide
    patched = {
        name: getattr(module, name)
        for name in (
            "_guided_ridge_safety_reason",
            "_guided_ridge_context_matches",
            "_guided_ridge_surface_anchors",
            "_guided_ridge_build_last_guide",
            "_guided_ridge_current_signature",
            "_guided_ridge_mesh_integrity",
        )
    }
    try:
        before = np.asarray([tuple(vertex.co) for vertex in mesh.vertices], dtype=np.float64)
        target = before.copy()
        target[1, 0] = 0.5
        proxy = SimpleNamespace(report=lambda *_args, **_kwargs: None)
        context = SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None))
        module._guided_ridge_safety_reason = lambda _obj: None
        module._guided_ridge_context_matches = lambda *_args: True
        module._guided_ridge_surface_anchors = lambda *_args: ({"anchors": True}, None)
        module._guided_ridge_build_last_guide = lambda *_args: ({"half_width": 0.25}, None)
        module._guided_ridge_current_signature = lambda *_args: before.copy()
        module._guided_ridge_mesh_integrity = lambda *_args: {
            "finite": True,
            "normal_pair_flips": 0,
            "new_degenerate_faces": 0,
        }
        state = {
            "active": True,
            "operator": proxy,
            "phase": "ready",
            "controls": [Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))],
            "control_normals": [Vector((0.0, 1.0, 0.0))] * 2,
            "obj": obj,
            "snapshot": {
                "coords_world": before.copy(),
                "coords_local": before.copy(),
                "vertex_indices": [0, 1, 2],
            },
            "timer": None,
            "draw_handler": None,
            "text_draw_handler": None,
            "area": None,
            "committed": False,
            "modal_result": None,
        }
        module._guided_ridge_state = state
        committed = module.VIEW3D_OT_mesh_focus_guided_ridge._commit(
            proxy,
            context,
            state,
            _compute_sync=True,
            _candidate_override=(
                target,
                {
                    "delta_h": np.asarray((0.0, -0.5, 0.0)),
                    "outward_component": np.zeros(3),
                    "anchor_mask": np.zeros(3, dtype=bool),
                    "crest_side_mask": np.ones(3, dtype=bool),
                },
            ),
        )
        after_commit = np.asarray([tuple(vertex.co) for vertex in mesh.vertices], dtype=np.float64)
        assert committed is True
        assert np.allclose(after_commit, target)
        assert module._guided_ridge_state is None
        return {
            "passed": True,
            "commit": "candidate_written",
            "changed_vertices": int(np.count_nonzero(np.linalg.norm(after_commit - before, axis=1) > 1.0e-12)),
            "undo_option": True,
        }
    finally:
        module._guided_ridge_state = original_state
        module._guided_ridge_last_guide = original_last
        for name, value in patched.items():
            setattr(module, name, value)
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def _guided_ridge_changed_only_nonidentity_fixture(module):
    """Write only changed vertices and preserve unchanged local coordinates."""
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector

    mesh = bpy.data.meshes.new("_mfo_guided_ridge_changed_only_mesh")
    mesh.from_pydata([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)], [], [(0, 1, 2)])
    mesh.update()
    obj = bpy.data.objects.new("_mfo_guided_ridge_changed_only_object", mesh)
    obj.matrix_world = Matrix.Translation((1.25, -2.0, 0.5)) @ Matrix.Rotation(0.37, 4, "Z")
    original_state = module._guided_ridge_state
    original_last = module._guided_ridge_last_guide
    names = (
        "_guided_ridge_safety_reason", "_guided_ridge_context_matches",
        "_guided_ridge_surface_anchors", "_guided_ridge_build_last_guide",
        "_guided_ridge_current_signature", "_guided_ridge_mesh_integrity",
    )
    patched = {name: getattr(module, name) for name in names}
    try:
        local_before = np.asarray([tuple(vertex.co) for vertex in mesh.vertices], dtype=np.float64)
        world_before = np.asarray([tuple(obj.matrix_world @ Vector(tuple(vertex.co))) for vertex in mesh.vertices], dtype=np.float64)
        local_target = local_before.copy()
        local_target[1, 0] = 0.5
        world_target = np.asarray([tuple(obj.matrix_world @ Vector(tuple(co))) for co in local_target], dtype=np.float64)
        proxy = SimpleNamespace(report=lambda *_args, **_kwargs: None)
        context = SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None))
        module._guided_ridge_safety_reason = lambda _obj: None
        module._guided_ridge_context_matches = lambda *_args: True
        module._guided_ridge_surface_anchors = lambda *_args: ({"anchors": True}, None)
        module._guided_ridge_build_last_guide = lambda *_args: ({"half_width": 0.25}, None)
        module._guided_ridge_current_signature = lambda *_args: local_before.copy()
        module._guided_ridge_mesh_integrity = lambda *_args: {
            "finite": True, "normal_pair_flips": 0, "new_degenerate_faces": 0,
        }
        state = {
            "active": True, "operator": proxy, "phase": "ready",
            "controls": [Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))],
            "control_normals": [Vector((0.0, 1.0, 0.0))] * 2,
            "obj": obj,
            "snapshot": {"coords_world": world_before, "coords_local": local_before,
                         "vertex_indices": [0, 1, 2]},
            "timer": None, "draw_handler": None, "text_draw_handler": None,
            "area": None, "committed": False, "modal_result": None,
        }
        module._guided_ridge_state = state
        assert module.VIEW3D_OT_mesh_focus_guided_ridge._commit(
            proxy, context, state, _compute_sync=True,
            _candidate_override=(world_target, {
                "delta_h": np.asarray((0.0, -0.5, 0.0)),
                "outward_component": np.zeros(3),
                "anchor_mask": np.zeros(3, dtype=bool),
                "crest_side_mask": np.ones(3, dtype=bool),
            }),
        ) is True
        after_local = np.asarray([tuple(vertex.co) for vertex in mesh.vertices], dtype=np.float64)
        assert np.array_equal(after_local[[0, 2]], local_before[[0, 2]])
        assert np.allclose(after_local[1], local_target[1], atol=2.0e-6)
        return {"passed": True, "changed_vertices": 1, "unchanged_local_exact": True}
    finally:
        module._guided_ridge_state = original_state
        module._guided_ridge_last_guide = original_last
        for name, value in patched.items():
            setattr(module, name, value)
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def _guided_ridge_mfo_navigation_fixture(module):
    """Verify MFO focus coexistence and Blender navigation pass-through."""
    from mathutils import Vector

    guided_class = module.VIEW3D_OT_mesh_focus_guided_ridge
    original = {
        "bpy": module.bpy,
        "active_states": module._active_states,
        "guided_state": module._guided_ridge_state,
        "face_attr": module._guided_ridge_face_set_attribute,
        "coordinate": module._sculpt_cursor_region_coordinate,
        "raycast": module._raycast_sculpt_face_set,
        "safety": module._guided_ridge_safety_reason,
        "signature": module._guided_ridge_context_signature,
        "ray": module._guided_ridge_ray,
        "overlay": module._guided_ridge_overlay_tag,
    }
    try:
        class _Area:
            type = "VIEW_3D"

            def as_pointer(self):
                return 9123

        class _Region:
            type = "WINDOW"
            width = 1200
            height = 800

        class _Window:
            def as_pointer(self):
                return 9124

        class _WM:
            def event_timer_add(self, *_args, **_kwargs):
                return object()

            def event_timer_remove(self, _timer):
                return None

            def modal_handler_add(self, _operator):
                return None

        obj = SimpleNamespace(type="MESH", data=SimpleNamespace())
        context = SimpleNamespace(
            area=_Area(), region=_Region(), window=_Window(), window_manager=_WM(),
            active_object=obj, mode="SCULPT",
            space_data=SimpleNamespace(show_gizmo=True, show_gizmo_navigate=True),
            preferences=SimpleNamespace(system=SimpleNamespace(ui_scale=1.0)),
        )
        # Construct a real MFO state object without touching camera/visibility
        # data.  The conflict check must recognize its class and leave it in
        # the registry after Guided Ridge cleanup.
        focus_state = object.__new__(module._TempOrbitState)
        focus_state.active = True
        focus_state.area_key = 9123
        module._active_states = {9123: focus_state}
        module._guided_ridge_state = None
        module._guided_ridge_face_set_attribute = lambda _obj: object()
        module._sculpt_cursor_region_coordinate = lambda _context, _event: Vector((20.0, 30.0))
        module._raycast_sculpt_face_set = lambda _context, _coord: (
            obj, 0, 7, Vector((0.0, 0.0, 0.0)), None
        )
        module._guided_ridge_safety_reason = lambda _obj: None
        module._guided_ridge_context_signature = lambda _context: {"area_pointer": 9123}
        module._guided_ridge_ray = lambda _context, _coord: (
            Vector((0.0, 0.0, 2.0)), Vector((0.0, 0.0, -1.0))
        )
        module._guided_ridge_overlay_tag = lambda _state: None
        module.bpy = SimpleNamespace(
            types=SimpleNamespace(
                SpaceView3D=SimpleNamespace(
                    draw_handler_add=lambda *_args: object(),
                    draw_handler_remove=lambda *_args: None,
                )
            ),
            context=SimpleNamespace(window_manager=context.window_manager),
        )
        assert module._guided_ridge_conflict_reason(context) is None
        assert guided_class.poll(context)
        proxy = SimpleNamespace(report=lambda *_args, **_kwargs: None, poll=guided_class.poll)
        assert guided_class.invoke(proxy, context, SimpleNamespace(type="LEFTMOUSE", value="PRESS")) == {"RUNNING_MODAL"}
        assert module._active_states[9123] is focus_state
        assert guided_class.modal(proxy, context, SimpleNamespace(type="ESC", value="PRESS")) == {"CANCELLED"}
        assert module._active_states[9123] is focus_state

        nav_state = {"region": _Region(), "navigation_gizmo_active": False, "navigation_mmb_active": False}
        assert module._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="MIDDLEMOUSE", value="PRESS"), nav_state
        )
        assert module._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="MOUSEMOVE", value="NOTHING", mouse_region_x=500, mouse_region_y=300), nav_state
        )
        assert module._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="MIDDLEMOUSE", value="RELEASE"), nav_state
        )
        assert not nav_state["navigation_mmb_active"]
        assert module._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="NDOF_MOTION", value="NOTHING"), nav_state
        )
        assert module._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="NUMPAD_1", value="PRESS"), nav_state
        )
        assert not module._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="NUMPAD_ENTER", value="PRESS"), nav_state
        )
        context.space_data.show_gizmo = False
        assert not module._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=1150, mouse_region_y=750), nav_state
        )
        context.space_data.show_gizmo = True
        # Top-right LMB is the Navigation Gizmo, while the ordinary guide LMB
        # remains owned by this modal.
        assert module._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=1150, mouse_region_y=750), nav_state
        )
        assert nav_state["navigation_gizmo_active"]
        assert module._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="MOUSEMOVE", value="NOTHING", mouse_region_x=1140, mouse_region_y=740), nav_state
        )
        assert module._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="LEFTMOUSE", value="RELEASE", mouse_region_x=1140, mouse_region_y=740), nav_state
        )
        assert not nav_state["navigation_gizmo_active"]
        assert not module._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=500, mouse_region_y=300), nav_state
        )
        return {
            "passed": True,
            "mfo_state_preserved": True,
            "poll_invoke_modal_cleanup": True,
            "navigation_events_pass_through": True,
            "gizmo_hit_zone": True,
        }
    finally:
        module.bpy = original["bpy"]
        module._active_states = original["active_states"]
        module._guided_ridge_state = original["guided_state"]
        module._guided_ridge_face_set_attribute = original["face_attr"]
        module._sculpt_cursor_region_coordinate = original["coordinate"]
        module._raycast_sculpt_face_set = original["raycast"]
        module._guided_ridge_safety_reason = original["safety"]
        module._guided_ridge_context_signature = original["signature"]
        module._guided_ridge_ray = original["ray"]
        module._guided_ridge_overlay_tag = original["overlay"]


def run():
    module = _load_module()
    region = SimpleNamespace(type="WINDOW", x=40, y=30, width=800, height=600)
    context = SimpleNamespace(
        area=SimpleNamespace(type="VIEW_3D"),
        region=region,
        space_data=SimpleNamespace(region_3d=object()),
    )
    event = SimpleNamespace(
        # Deliberately stale region-relative values prove that absolute event
        # coordinates are authoritative.
        mouse_x=270,
        mouse_y=190,
        mouse_region_x=1,
        mouse_region_y=2,
    )
    coordinate = module._tool_event_coordinate(context, event)
    assert tuple(coordinate) == (230.0, 160.0)
    outside = SimpleNamespace(
        mouse_x=20,
        mouse_y=190,
        mouse_region_x=500,
        mouse_region_y=160,
    )
    assert module._tool_event_coordinate(context, outside) is None

    normal_object = next(
        tool for tool in module._TOOL_CLASSES if tool.bl_idname == "mfo.normal_object"
    )
    assert normal_object.bl_keymap[1][0] == module.TOOL_FACE_SET_OPERATOR_ID
    assert normal_object.bl_keymap[1][1]["ctrl"] is True
    assert all(len(item) == 3 for tool in module._TOOL_CLASSES for item in tool.bl_keymap)
    # The resident click operator is intentionally INTERNAL only.  Its ON
    # path must dispatch through the established UNDO-bearing Activation
    # operator, while OFF is a direct runtime teardown without another step.
    assert "UNDO" not in module.VIEW3D_OT_mesh_focus_face_set_tool.bl_options
    assert "UNDO" in module.VIEW3D_OT_mesh_focus_face_set_activate.bl_options
    assert "mesh_focus_face_set_activate" in inspect.getsource(
        module.VIEW3D_OT_mesh_focus_face_set_tool.invoke
    )
    activation_props = {
        "mouse_region_x",
        "mouse_region_y",
        "use_click_coordinate",
    }
    assert activation_props.issubset(module.VIEW3D_OT_mesh_focus_face_set_activate.__annotations__)

    # Dispatch-level observation with a fake operator backend: the click
    # operator forwards the exact region coordinate to Activation on ON, and
    # the next same-entry click tears down the active Face Set state directly.
    dispatch_area = SimpleNamespace(type="VIEW_3D", as_pointer=lambda: 9123)
    dispatch_context = SimpleNamespace(
        area=dispatch_area,
        region=region,
        space_data=SimpleNamespace(region_3d=object()),
        mode="OBJECT",
    )
    dispatch_event = SimpleNamespace(mouse_x=270, mouse_y=190, mouse_region_x=1, mouse_region_y=2)
    original_bpy = module.bpy
    original_states = module._active_states
    original_deactivate = module._deactivate_state
    original_face_set_state = module._FaceSetOrbitState
    original_activate_face_set_session = module._activate_face_set_session
    dispatch_calls = []
    off_calls = []

    class _FakeFaceSetState:
        def __init__(self):
            self.active = True

    class _FakeActivation:
        def __call__(self, *args, **kwargs):
            dispatch_calls.append((args, kwargs))
            return {"FINISHED"}

    try:
        module.bpy = SimpleNamespace(
            ops=SimpleNamespace(
                view3d=SimpleNamespace(mesh_focus_face_set_activate=_FakeActivation())
            )
        )
        module._active_states = {}
        module._FaceSetOrbitState = _FakeFaceSetState
        module._deactivate_state = lambda state: off_calls.append(state)
        self_proxy = SimpleNamespace(
            report=lambda *_args, **_kwargs: None,
            poll=module.VIEW3D_OT_mesh_focus_face_set_tool.poll,
        )
        on_result = module.VIEW3D_OT_mesh_focus_face_set_tool.invoke(
            self_proxy, dispatch_context, dispatch_event
        )
        assert on_result == {"FINISHED"}
        assert dispatch_calls and dispatch_calls[0][0][:2] == ("EXEC_DEFAULT", True)
        assert dispatch_calls[0][1] == {
            "mouse_region_x": 230,
            "mouse_region_y": 160,
            "use_click_coordinate": True,
        }
        module._active_states = {9123: _FakeFaceSetState()}
        off_result = module.VIEW3D_OT_mesh_focus_face_set_tool.invoke(
            self_proxy, dispatch_context, dispatch_event
        )
        assert off_result == {"FINISHED"}
        assert len(off_calls) == 1

        activation_coordinate = []
        module._active_states = {}
        module._activate_face_set_session = lambda _context, coordinate=None: (
            activation_coordinate.append(coordinate) or True
        )
        activation_self = SimpleNamespace(
            use_click_coordinate=True,
            mouse_region_x=230,
            mouse_region_y=160,
            poll=module.VIEW3D_OT_mesh_focus_face_set_activate.poll,
        )
        assert module.VIEW3D_OT_mesh_focus_face_set_activate.execute(
            activation_self, dispatch_context
        ) == {"FINISHED"}
        assert [tuple(value) for value in activation_coordinate] == [(230.0, 160.0)]
    finally:
        module.bpy = original_bpy
        module._active_states = original_states
        module._deactivate_state = original_deactivate
        module._FaceSetOrbitState = original_face_set_state
        module._activate_face_set_session = original_activate_face_set_session

    # Guided Ridge lifecycle: the WorkSpaceTool's initial LEFTMOUSE pair is
    # a start-only event.  It must not add a second point or commit, while the
    # first LMB after release adds a point and Enter is the only commit path.
    from mathutils import Vector

    guided_class = module.VIEW3D_OT_mesh_focus_guided_ridge
    original_guided_state = module._guided_ridge_state
    original_guided_coordinate = module._sculpt_cursor_region_coordinate
    original_guided_hit = module._guided_ridge_snapshot_hit
    original_guided_project = module._guided_ridge_project_curve
    original_guided_overlay = module._guided_ridge_overlay_tag
    guided_commit_calls = []
    try:
        guided_state = {
            "active": True,
            "operator": None,
            "phase": "ready",
            "start_key": "LEFTMOUSE",
            "start_key_released": False,
            "start_guard_active": True,
            "controls": [Vector((0.0, 0.0, 0.0))],
            "control_normals": [Vector((0.0, 0.0, 1.0))],
            "guide": [Vector((0.0, 0.0, 0.0))],
            "snapshot": {},
            "apply_rect": (900, 900, 950, 950),
            "cancel_rect": (960, 900, 990, 950),
            "committed": False,
        }
        module._guided_ridge_state = guided_state
        module._sculpt_cursor_region_coordinate = lambda _context, _event: Vector((10.0, 10.0))
        module._guided_ridge_snapshot_hit = lambda _context, _snapshot, _coord: (
            Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)), 0, 0.0
        )
        module._guided_ridge_project_curve = lambda _snapshot, controls: (list(controls), None)
        module._guided_ridge_overlay_tag = lambda _state: None

        guided_proxy = SimpleNamespace(
            report=lambda *_args, **_kwargs: None,
            _commit=lambda _context, state: (
                guided_commit_calls.append(True), state.__setitem__("committed", True), True
            )[-1],
        )
        guided_state["operator"] = guided_proxy
        guided_context = SimpleNamespace()
        first_press = SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=100, mouse_region_y=100)
        first_release = SimpleNamespace(type="LEFTMOUSE", value="RELEASE", mouse_region_x=100, mouse_region_y=100)
        next_click = SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=120, mouse_region_y=120)
        enter = SimpleNamespace(type="ENTER", value="PRESS", mouse_region_x=120, mouse_region_y=120)
        assert guided_class.modal(guided_proxy, guided_context, first_press) == {"RUNNING_MODAL"}
        assert len(guided_state["controls"]) == 1
        assert not guided_commit_calls
        assert guided_class.modal(guided_proxy, guided_context, first_release) == {"RUNNING_MODAL"}
        assert guided_state["start_key_released"] is True
        assert guided_class.modal(guided_proxy, guided_context, next_click) == {"RUNNING_MODAL"}
        assert len(guided_state["controls"]) == 2
        assert not guided_commit_calls
        assert guided_class.modal(guided_proxy, guided_context, enter) == {"FINISHED"}
        assert len(guided_commit_calls) == 1
        guided_lifecycle = {"passed": True, "first_controls": 1, "after_next_lmb_controls": 2}
    finally:
        module._guided_ridge_state = original_guided_state
        module._sculpt_cursor_region_coordinate = original_guided_coordinate
        module._guided_ridge_snapshot_hit = original_guided_hit
        module._guided_ridge_project_curve = original_guided_project
        module._guided_ridge_overlay_tag = original_guided_overlay

    # Preparation owns LMB/Enter while the timer is active, and Esc removes
    # the timer and generator state without requiring a mesh or scene fixture.
    original_guided_overlay = module._guided_ridge_overlay_tag
    try:
        removed_timers = []

        class _PrepareWM:
            def event_timer_remove(self, timer):
                removed_timers.append(timer)

        prepare_state = {
            "active": True,
            "operator": None,
            "phase": "prepare",
            "start_key": "LEFTMOUSE",
            "start_key_released": False,
            "start_guard_active": False,
            "controls": [],
            "control_normals": [],
            "guide": [],
            "snapshot": None,
            "apply_rect": (900, 900, 950, 950),
            "cancel_rect": (960, 900, 990, 950),
            "timer": object(),
            "window_manager": _PrepareWM(),
            "prepare_job": iter((None,)),
            "draw_handler": None,
            "text_draw_handler": None,
        }
        prepare_proxy = SimpleNamespace(report=lambda *_args, **_kwargs: None)
        prepare_state["operator"] = prepare_proxy
        module._guided_ridge_state = prepare_state
        module._guided_ridge_overlay_tag = lambda _state: None
        lmb = SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=100, mouse_region_y=100)
        enter_prepare = SimpleNamespace(type="ENTER", value="PRESS", mouse_region_x=100, mouse_region_y=100)
        assert guided_class.modal(prepare_proxy, SimpleNamespace(), lmb) == {"RUNNING_MODAL"}
        assert guided_class.modal(prepare_proxy, SimpleNamespace(), enter_prepare) == {"RUNNING_MODAL"}
        assert prepare_state["controls"] == []
        assert guided_class.modal(
            prepare_proxy, SimpleNamespace(), SimpleNamespace(type="ESC", value="PRESS")
        ) == {"CANCELLED"}
        assert removed_timers and module._guided_ridge_state is None
        prepare_cleanup = {"passed": True, "timer_removed": True, "generator_cleared": True}
    finally:
        module._guided_ridge_overlay_tag = original_guided_overlay
        module._guided_ridge_state = original_guided_state

    # A completed timer job transitions to Ready, seeds exactly the first
    # clicked control, removes its timer, and leaves later LMB/Enter handling
    # to the normal lifecycle tested above.
    original_prepare_context_matches = module._guided_ridge_prepare_context_matches
    original_prepare_snapshot_hit_ray = module._guided_ridge_snapshot_hit_ray
    original_prepare_context_signature = module._guided_ridge_context_signature
    original_prepare_overlay = module._guided_ridge_overlay_tag
    try:
        removed_ready_timers = []

        class _ReadyWM:
            def event_timer_remove(self, timer):
                removed_ready_timers.append(timer)

        ready_proxy = SimpleNamespace(report=lambda *_args, **_kwargs: None)
        ready_context = SimpleNamespace(active_object=object(), mode="SCULPT")
        ready_state = {
            "active": True,
            "operator": ready_proxy,
            "phase": "prepare",
            "obj": ready_context.active_object,
            "context_signature": {"marker": 1},
            "prepare_job": iter(
                (
                    {"stage": "reading Face Set data", "stage_index": 1, "stage_count": 13, "done": 1, "total": 1},
                )
            ),
            "start_coordinate": Vector((4.0, 5.0)),
            "start_ray": (Vector((0.0, 0.0, 2.0)), Vector((0.0, 0.0, -1.0))),
            "timer": object(),
            "window_manager": _ReadyWM(),
            "prepare_stage_count": 13,
            "prepare_stage": "queued",
            "prepare_slices": 0,
            "prepare_elapsed_seconds": 0.0,
            "prepare_max_slice_seconds": 0.0,
            "draw_handler": None,
            "text_draw_handler": None,
            "area": None,
        }
        # StopIteration.value is the snapshot/result pair for the generator.
        def _ready_job():
            yield {"stage": "finalizing snapshot", "stage_index": 12, "stage_count": 13, "done": 1, "total": 1}
            return ({"fixture": True}, None)

        ready_state["prepare_job"] = _ready_job()
        module._guided_ridge_state = ready_state
        module._guided_ridge_prepare_context_matches = lambda _context, _state: True
        module._guided_ridge_snapshot_hit_ray = lambda *_args: (
            Vector((4.0, 5.0, 6.0)), Vector((0.0, 0.0, 1.0)), 0, 0.0
        )
        module._guided_ridge_context_signature = lambda _context: {"marker": 2}
        module._guided_ridge_overlay_tag = lambda _state: None
        assert module._guided_ridge_process_prepare_timer(ready_context, ready_state) is True
        assert ready_state["phase"] == "prepare"
        assert module._guided_ridge_process_prepare_timer(ready_context, ready_state) is True
        assert ready_state["phase"] == "ready"
        assert len(ready_state["controls"]) == 1
        assert ready_state["prepare_job"] is None
        assert ready_state["timer"] is None
        assert removed_ready_timers
        guided_ridge_ready = {"passed": True, "timer_removed": True, "seed_controls": 1}
    finally:
        module._guided_ridge_prepare_context_matches = original_prepare_context_matches
        module._guided_ridge_snapshot_hit_ray = original_prepare_snapshot_hit_ray
        module._guided_ridge_context_signature = original_prepare_context_signature
        module._guided_ridge_overlay_tag = original_prepare_overlay
        module._guided_ridge_state = original_guided_state

    # The first timer tick publishes the stage and returns to the event loop
    # before the generator enters the stage's expensive body.  The following
    # tick may perform that work.
    original_prepare_context_matches = module._guided_ridge_prepare_context_matches
    original_prepare_overlay = module._guided_ridge_overlay_tag
    try:
        class _PaintWM:
            def event_timer_remove(self, _timer):
                return None

        heavy_calls = []

        def _first_paint_job():
            yield {"stage": "reading Face Set data", "stage_index": 1, "stage_count": 13, "done": 0, "total": 0}
            heavy_calls.append("foreach_get")
            yield {"stage": "reading hidden-face data", "stage_index": 1, "stage_count": 13, "done": 0, "total": 0}
            return ({"fixture": True}, None)

        paint_proxy = SimpleNamespace(report=lambda *_args, **_kwargs: None)
        paint_context = SimpleNamespace(active_object=object(), mode="SCULPT")
        paint_state = {
            "active": True,
            "operator": paint_proxy,
            "phase": "prepare",
            "obj": paint_context.active_object,
            "context_signature": {"marker": 1},
            "prepare_job": _first_paint_job(),
            "prepare_stage": "queued",
            "prepare_stage_count": 13,
            "prepare_stage_done": 0,
            "prepare_stage_total": 0,
            "prepare_progress_indeterminate": True,
            "prepare_slices": 0,
            "prepare_elapsed_seconds": 0.0,
            "prepare_max_slice_seconds": 0.0,
            "timer": object(),
            "window_manager": _PaintWM(),
            "draw_handler": None,
            "text_draw_handler": None,
            "area": None,
        }
        module._guided_ridge_state = paint_state
        module._guided_ridge_prepare_context_matches = lambda _context, _state: True
        module._guided_ridge_overlay_tag = lambda _state: None
        assert module._guided_ridge_process_prepare_timer(paint_context, paint_state) is True
        assert paint_state["prepare_stage"] == "reading Face Set data"
        assert not heavy_calls
        assert module._guided_ridge_process_prepare_timer(paint_context, paint_state) is True
        assert heavy_calls == ["foreach_get"]
        assert paint_state["prepare_stage"] == "reading hidden-face data"
        guided_ridge_first_paint = {"passed": True, "first_tick_heavy_calls": 0, "second_tick_heavy_calls": 1}
        module._guided_ridge_cancel(paint_state, "test-cleanup")
    finally:
        module._guided_ridge_prepare_context_matches = original_prepare_context_matches
        module._guided_ridge_overlay_tag = original_prepare_overlay
        module._guided_ridge_state = original_guided_state

    # A saved Repeat Last payload must not be touched by the T-tool invoke.
    # This uses only fake context/data and draw/timer callbacks.
    original_bpy_for_guided = module.bpy
    original_guided_state = module._guided_ridge_state
    original_guided_last = module._guided_ridge_last_guide
    saved_guided_functions = {
        name: getattr(module, name)
        for name in (
            "_sculpt_cursor_region_coordinate",
            "_raycast_sculpt_face_set",
            "_guided_ridge_safety_reason",
            "_guided_ridge_context_signature",
            "_guided_ridge_ray",
            "_guided_ridge_snapshot_hit_ray",
        )
    }
    try:
        class _GuidedArea:
            type = "VIEW_3D"

            def as_pointer(self):
                return 1901

        class _GuidedRegion:
            type = "WINDOW"
            width = 1000
            height = 800

        class _GuidedWindow:
            def as_pointer(self):
                return 1902

        class _GuidedWM:
            def event_timer_add(self, *_args, **_kwargs):
                return object()

            def modal_handler_add(self, _operator):
                return None

            def event_timer_remove(self, _timer):
                return None

        fake_obj = SimpleNamespace(type="MESH", data=SimpleNamespace())
        guided_context = SimpleNamespace(
            area=_GuidedArea(), region=_GuidedRegion(), window=_GuidedWindow(),
            window_manager=_GuidedWM(), active_object=fake_obj, mode="SCULPT",
        )
        module.bpy = SimpleNamespace(
            types=SimpleNamespace(
                SpaceView3D=SimpleNamespace(
                    draw_handler_add=lambda *_args: object(),
                    draw_handler_remove=lambda *_args: None,
                )
            ),
            context=SimpleNamespace(window_manager=guided_context.window_manager),
        )
        module._guided_ridge_state = None
        repeat_calls = []
        sentinel = {"saved": True}
        module._guided_ridge_last_guide = sentinel
        module._sculpt_cursor_region_coordinate = lambda _context, _event: Vector((20.0, 30.0))
        module._raycast_sculpt_face_set = lambda _context, _coord: (fake_obj, 7, 11, None, None)
        module._guided_ridge_safety_reason = lambda _obj: None
        module._guided_ridge_context_signature = lambda _context: {"area_pointer": 1901}
        module._guided_ridge_ray = lambda _context, _coord: (
            Vector((0.0, 0.0, 2.0)), Vector((0.0, 0.0, -1.0))
        )
        guided_proxy = SimpleNamespace(
            poll=lambda _context: True,
            report=lambda *_args, **_kwargs: None,
            _execute_repeat=lambda _context, guide: repeat_calls.append(guide) or False,
        )
        result_invoke = guided_class.invoke(
            guided_proxy,
            guided_context,
            SimpleNamespace(type="LEFTMOUSE", value="PRESS"),
        )
        assert result_invoke == {"RUNNING_MODAL"}
        assert not repeat_calls
        assert module._guided_ridge_state.get("phase") == "prepare"
        # An accidental direct EXEC_DEFAULT while the timer owns preparation
        # must not reach _commit (or the saved-guide Repeat Last path).
        assert guided_class.execute(guided_proxy, guided_context) == {"CANCELLED"}
        assert not repeat_calls
        saved_ray = module._guided_ridge_state["start_ray"]
        used_rays = []
        module._guided_ridge_snapshot_hit_ray = lambda _snapshot, ray: (
            used_rays.append(ray)
            or (Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)), 0, 0.0)
        )
        def _saved_ray_job():
            yield {"stage": "finalizing snapshot", "stage_index": 12, "stage_count": 13, "done": 1, "total": 1}
            return ({"fixture": True}, None)
        module._guided_ridge_state["prepare_job"] = _saved_ray_job()
        guided_context.region.width = 1200
        guided_context.region.height = 900
        assert module._guided_ridge_process_prepare_timer(
            guided_context, module._guided_ridge_state
        ) is True
        assert module._guided_ridge_state.get("phase") == "prepare"
        assert module._guided_ridge_process_prepare_timer(
            guided_context, module._guided_ridge_state
        ) is True
        assert used_rays and tuple(used_rays[0][0]) == tuple(saved_ray[0])
        assert tuple(used_rays[0][1]) == tuple(saved_ray[1])
        guided_invoke_repeat = {"passed": True, "repeat_calls": 0}
        module._guided_ridge_cancel(module._guided_ridge_state, "test-cleanup")
        assert module._guided_ridge_state is None
    finally:
        module.bpy = original_bpy_for_guided
        module._guided_ridge_state = original_guided_state
        module._guided_ridge_last_guide = original_guided_last
        for name, value in saved_guided_functions.items():
            setattr(module, name, value)

    # The retained synchronous implementation and the new cooperative
    # generator must produce the same snapshot/signature/BVH inputs.  This is
    # a temporary datablock-only fixture; it is removed in the helper finally.
    guided_ridge_equivalence = None
    try:
        guided_ridge_equivalence = _guided_ridge_snapshot_equivalence(module)
    except ImportError:
        # Source-only harnesses do not have bpy; Blender runs this branch.
        guided_ridge_equivalence = {"skipped": "Blender bpy is unavailable"}
    guided_ridge_bfs = None
    try:
        guided_ridge_bfs = _guided_ridge_bfs_fixture(module)
    except ImportError:
        guided_ridge_bfs = {"skipped": "Blender bpy is unavailable"}
    guided_ridge_surface = _guided_ridge_surface_fixture(module)
    guided_ridge_open_surface = _guided_ridge_open_surface_fixture(module)
    guided_ridge_ambiguous_surface = _guided_ridge_ambiguous_surface_fixture(module)
    guided_ridge_width_density = _guided_ridge_width_density_fixture(module)
    guided_ridge_width_ambiguous = _guided_ridge_width_ambiguous_fixture(module)
    guided_ridge_projected_rail_support = _guided_ridge_projected_rail_support_fixture(module)
    guided_ridge_tapered_rail = _guided_ridge_tapered_rail_fixture(module)
    guided_ridge_width_controls = _guided_ridge_width_controls_fixture(module)
    guided_ridge_nonuniform = _guided_ridge_nonuniform_frame_fixture(module)
    guided_ridge_width_cache = _guided_ridge_width_cache_fixture(module)
    guided_ridge_aligned_prepare = _guided_ridge_aligned_prepare_equivalence_fixture(module)
    guided_ridge_width_rebuild = _guided_ridge_width_frame_rebuild_fixture(module)
    guided_ridge_live = _guided_ridge_live_snapshot_diagnostic(module)
    guided_ridge_compute = _guided_ridge_compute_lifecycle(module)
    guided_ridge_commit = _guided_ridge_commit_fixture(module)
    guided_ridge_changed_only = _guided_ridge_changed_only_nonidentity_fixture(module)
    guided_ridge_mfo_navigation = _guided_ridge_mfo_navigation_fixture(module)

    sfsf_sculpt = next(
        tool for tool in module._TOOL_CLASSES if tool.bl_idname == "mfo.smart_fill_sculpt"
    )
    assert any(item[1].get("ctrl") is True for item in sfsf_sculpt.bl_keymap)

    # WorkSpaceTool icons must resolve to Blender's shipped .dat assets, not
    # regular UI enum names (which silently render as no icon).
    loaded_icon_values = []
    try:
        import bpy

        icon_root = pathlib.Path(bpy.app.binary_path).resolve().parent / "5.2" / "datafiles" / "icons"
        if not icon_root.is_dir():
            icon_root = pathlib.Path(bpy.app.binary_path).resolve().parent / "datafiles" / "icons"
        icon_paths = [
            pathlib.Path(f"{tool.bl_icon}.dat")
            if pathlib.Path(str(tool.bl_icon)).is_absolute()
            else icon_root / f"{tool.bl_icon}.dat"
            for tool in module._TOOL_CLASSES
        ]
        assert icon_root.is_dir(), f"Blender icon directory not found: {icon_root}"
        assert all(path.is_file() for path in icon_paths), icon_paths
        loaded_icon_values = [bpy.app.icons.new_triangles_from_file(str(path)) for path in icon_paths]
        assert all(int(value) > 0 for value in loaded_icon_values)
    except ImportError:
        # This test is normally executed from Blender; the rest remains
        # useful for source-only harnesses.
        icon_paths = []
    finally:
        # Blender 5.2 names the icon-ID destructor ``release``; use ``delete``
        # when a future API exposes that spelling and always release IDs even
        # if a later icon load or assertion fails.
        if "bpy" in locals():
            release_icon = getattr(bpy.app.icons, "delete", None)
            if release_icon is None:
                release_icon = getattr(bpy.app.icons, "release", None)
            if release_icon is not None:
                for icon_id in loaded_icon_values:
                    try:
                        release_icon(icon_id)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass

    # Regression guard: stale-tool cleanup must leave a user-customized
    # keymap intact without touching Blender's live keyconfigs.  All three
    # collections below are fake, so this test has no persistent side effect.
    class _FakeKeymap:
        def __init__(self, name, items=()):
            self.name = name
            self.keymap_items = list(items)

    class _FakeKeymaps(list):
        def remove(self, keymap):
            super().remove(keymap)

    class _FakeKeyconfig:
        def __init__(self, keymaps):
            self.keymaps = _FakeKeymaps(keymaps)

    known_name = "3D View Tool: Object, MFO: Focus Surface"
    unknown_name = "3D View Tool: Object, MFO: Focus Surface Custom"
    fake_user_sentinel = _FakeKeymap(known_name, [SimpleNamespace(idname="wm.search_menu")])
    fake_keyconfigs = SimpleNamespace(
        default=_FakeKeyconfig([_FakeKeymap(known_name), _FakeKeymap(unknown_name)]),
        addon=_FakeKeyconfig([_FakeKeymap(known_name)]),
        user=_FakeKeyconfig([fake_user_sentinel]),
    )
    original_bpy_for_keymaps = module.bpy
    try:
        module.bpy = SimpleNamespace(
            context=SimpleNamespace(
                window_manager=SimpleNamespace(keyconfigs=fake_keyconfigs)
            )
        )
        module._remove_stale_tool_keymaps()
        assert [km.name for km in fake_keyconfigs.default.keymaps] == [unknown_name]
        assert list(fake_keyconfigs.addon.keymaps) == []
        assert fake_keyconfigs.user.keymaps[0] is fake_user_sentinel
        assert fake_user_sentinel.keymap_items[0].idname == "wm.search_menu"
        user_keymap_result = {
            "preserved": True,
            "fake_collections_only": True,
            "unknown_default_preserved": True,
        }
    finally:
        module.bpy = original_bpy_for_keymaps

    poll_modes = {}
    for mode in ("OBJECT", "EDIT_MESH", "SCULPT"):
        mode_context = SimpleNamespace(
            area=context.area,
            region=region,
            space_data=context.space_data,
            mode=mode,
        )
        poll_modes[mode] = {
            "normal": module.VIEW3D_OT_mesh_focus_orbit_tool.poll(mode_context),
            "face_set": module.VIEW3D_OT_mesh_focus_face_set_tool.poll(mode_context),
        }
    assert poll_modes["OBJECT"] == {"normal": True, "face_set": True}
    assert poll_modes["EDIT_MESH"] == {"normal": True, "face_set": True}
    assert poll_modes["SCULPT"] == {"normal": True, "face_set": False}
    return {
        "passed": True,
        "click_region": tuple(coordinate),
        "outside_rejected": True,
        "ctrl_strict_operator": module.TOOL_FACE_SET_OPERATOR_ID,
        "tool_count": len(module._TOOL_CLASSES),
        "poll_modes": poll_modes,
        "face_set_undo_contract": True,
        "icon_dat_count": len(icon_paths),
        "user_keymap": user_keymap_result,
        "sfsf_ctrl_binding": True,
        "dispatch_observed": {"on_coordinate": True, "off_toggle": True},
        "guided_ridge_lifecycle": guided_lifecycle,
        "guided_ridge_prepare_cleanup": prepare_cleanup,
        "guided_ridge_ready_transition": guided_ridge_ready,
        "guided_ridge_first_paint": guided_ridge_first_paint,
        "guided_ridge_invoke_repeat_guard": guided_invoke_repeat,
        "guided_ridge_snapshot_equivalence": guided_ridge_equivalence,
        "guided_ridge_bfs": guided_ridge_bfs,
        "guided_ridge_surface_sections": guided_ridge_surface,
        "guided_ridge_open_surface": guided_ridge_open_surface,
        "guided_ridge_ambiguous_surface": guided_ridge_ambiguous_surface,
        "guided_ridge_width_density": guided_ridge_width_density,
        "guided_ridge_width_ambiguous": guided_ridge_width_ambiguous,
        "guided_ridge_projected_rail_support": guided_ridge_projected_rail_support,
        "guided_ridge_tapered_rail": guided_ridge_tapered_rail,
        "guided_ridge_width_controls": guided_ridge_width_controls,
        "guided_ridge_nonuniform_frame": guided_ridge_nonuniform,
        "guided_ridge_width_cache": guided_ridge_width_cache,
        "guided_ridge_aligned_prepare": guided_ridge_aligned_prepare,
        "guided_ridge_width_rebuild": guided_ridge_width_rebuild,
        "guided_ridge_live_snapshot": guided_ridge_live,
        "guided_ridge_compute_lifecycle": guided_ridge_compute,
        "guided_ridge_commit": guided_ridge_commit,
        "guided_ridge_changed_only": guided_ridge_changed_only,
        "guided_ridge_mfo_navigation": guided_ridge_mfo_navigation,
    }


if __name__ == "__main__":
    print(run())
