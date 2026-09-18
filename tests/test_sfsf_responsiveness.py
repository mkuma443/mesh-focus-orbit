"""Headless heartbeat checks for Smart Face Set Fill preparation.

The test exercises the real generators without entering a modal operator or
changing the open scene.  It verifies that the long Python face-to-vertex
conversion is sliced and that the HUD tokens expose indeterminate boundaries
around native NumPy/RNA calls.
"""

from __future__ import annotations

import importlib.util
import pathlib
import time


def _load_module():
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "mesh_focus_orbit.py"
    spec = importlib.util.spec_from_file_location("mfo_sfsf_responsiveness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(job):
    slices = []
    while True:
        started = time.perf_counter()
        try:
            token = next(job)
        except StopIteration as done:
            return done.value, slices
        slices.append((token, time.perf_counter() - started))


def _assert_same_value(left, right):
    import numpy as np

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    elif isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same_value(a, b)
    else:
        assert left == right


def _assert_builder_equivalence(module, obj):
    """Compare the active cooperative graph with the retained 3.3.7 builder."""

    import numpy as np

    common = (
        "count", "face_edge_counts", "centers", "normals", "hidden",
        "world_vertices", "offsets", "neighbors", "neighbor_lengths",
        "edge_indices", "edge_v0", "edge_v1", "first", "second",
        "pair_lengths", "pair_v0", "pair_v1", "pair_edge_points",
        "face_ids", "seam_count", "topology_fingerprint",
        "topology_first", "topology_second", "topology_edge_v0", "topology_edge_v1",
    )
    module._fill_preview_adjacency_cache.clear()
    baseline = module._fill_preview_build_adjacency(obj)
    module._fill_preview_adjacency_cache.clear()
    raw, _ = _run(module._fill_preview_adjacency_steps(obj))
    candidate, _ = _run(module._fill_preview_build_adjacency_cooperative(obj, prepared=raw))
    for key in common:
        assert key in baseline and key in candidate, key
        _assert_same_value(baseline[key], candidate[key])
    baseline_faces = baseline["face_vertex_ids"]
    candidate_faces = module._fill_preview_face_vertex_sequence(candidate)
    assert candidate_faces is not None
    assert len(baseline_faces) == len(candidate_faces)
    for index in range(len(baseline_faces)):
        _assert_same_value(baseline_faces[index], candidate_faces[index])
    baseline_nbytes = sum(
        int(value.nbytes) for value in baseline.values() if isinstance(value, np.ndarray)
    )
    candidate_nbytes = sum(
        int(value.nbytes) for value in candidate.values() if isinstance(value, np.ndarray)
    )
    assert candidate_nbytes > 0
    return {
        "baseline_nbytes": baseline_nbytes,
        "candidate_nbytes": candidate_nbytes,
        "candidate_cache_keys": tuple(sorted(candidate)),
    }


def _fixture_equivalence(module, name, vertices, faces, hidden_index=None):
    import bpy

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name + "_object", mesh)
    try:
        if hidden_index is not None:
            mesh.polygons[int(hidden_index)].hide = True
            mesh.update()
        return _assert_builder_equivalence(module, obj)
    finally:
        module._fill_preview_adjacency_cache.clear()
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def run():
    import bpy

    module = _load_module()
    size = 80
    vertices = [(x, y, 0.0) for y in range(size + 1) for x in range(size + 1)]
    faces = []
    for y in range(size):
        for x in range(size):
            a = y * (size + 1) + x
            b = a + 1
            c = a + size + 2
            d = a + size + 1
            faces.extend(((a, b, c), (a, c, d)))
    mesh = bpy.data.meshes.new("_sfsf_responsiveness_test_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("_sfsf_responsiveness_test_object", mesh)
    try:
        equivalence = _assert_builder_equivalence(module, obj)
        module._fill_preview_adjacency_cache.clear()
        raw, raw_slices = _run(module._fill_preview_adjacency_steps(obj))
        cache, build_slices = _run(
            module._fill_preview_build_adjacency_cooperative(obj, prepared=raw)
        )
        assert cache["count"] == len(faces)
        # The active cooperative path must retain only the flat schema; the
        # compatibility accessor is for the old builder and compact patches.
        assert "face_vertex_ids" not in cache
        assert "face_vertex_flat" in cache
        assert "face_vertex_offsets" in cache
        stages = [token.get("stage") for token, _ in build_slices if isinstance(token, dict)]
        assert "face-vertex-schema" in stages
        assert "cache-build" in stages
        # The test mesh is intentionally modest; the production 2.46M-face
        # measurement is reported separately because native foreach_get and
        # NumPy sorts are indivisible on Blender's main thread.
        max_slice = max(seconds for _, seconds in raw_slices + build_slices)
        assert max_slice < 0.1, max_slice
        fixture_results = {
            "all_visible": equivalence,
            "hidden_face": _fixture_equivalence(
                module,
                "_sfsf_hidden_fixture",
                [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                [(0, 1, 2), (0, 2, 3)],
                hidden_index=1,
            ),
            "non_manifold": _fixture_equivalence(
                module,
                "_sfsf_nonmanifold_fixture",
                [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0.5, 0, 1)],
                [(0, 1, 2), (1, 0, 3), (0, 1, 4)],
            ),
            "unwelded_seam": _fixture_equivalence(
                module,
                "_sfsf_seam_fixture",
                [(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (1, 0, 0), (0, 0, 0), (0.5, -1, 0)],
                [(0, 1, 2), (3, 4, 5)],
            ),
            "disconnected": _fixture_equivalence(
                module,
                "_sfsf_disconnected_fixture",
                [(0, 0, 0), (1, 0, 0), (0, 1, 0), (10, 0, 0), (11, 0, 0), (10, 1, 0)],
                [(0, 1, 2), (3, 4, 5)],
            ),
        }
        return {
            "passed": True,
            "faces": len(faces),
            "raw_slices": len(raw_slices),
            "build_slices": len(build_slices),
            "max_slice_seconds": max_slice,
            "equivalence": equivalence,
            "fixtures": fixture_results,
        }
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


if __name__ == "__main__":
    print(run())
