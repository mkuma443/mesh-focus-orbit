"""Focused in-memory regression tests for the Smart Face Set Fill cache layers.

Run from Blender (the add-on imports ``bpy`` at module scope)::

    exec(compile(open(path).read(), path, "exec"), {"__name__": "__main__"})

The fake mesh deliberately exposes only the foreach_get surface used by the
cache refresh code.  The production-builder check creates only a temporary
unlinked mesh and removes it before returning; no scene is changed or saved.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np


class _ArrayCollection:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def foreach_get(self, prop, output):
        if prop == "co":
            np.copyto(output, np.asarray(self.values, dtype=np.float32).ravel())
        elif prop == "center":
            np.copyto(output, np.asarray(self.values, dtype=np.float32).ravel())
        elif prop == "normal":
            np.copyto(output, np.asarray(self.values, dtype=np.float32).ravel())
        elif prop in {"edge_index", "vertex_index", "loop_total", "hide"}:
            np.copyto(output, np.asarray(self.values, dtype=output.dtype))
        else:
            raise AssertionError(prop)


class _FaceSetData:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def foreach_get(self, prop, output):
        assert prop == "value"
        np.copyto(output, np.asarray(self.values, dtype=output.dtype))

    def foreach_set(self, prop, values):
        assert prop == "value"
        self.values = [int(value) for value in values]


class _FaceSetAttribute:
    domain = "FACE"

    def __init__(self, values):
        self.data = _FaceSetData(values)


class _Attributes:
    def __init__(self, values):
        self.face_set = _FaceSetAttribute(values)

    def get(self, name):
        return self.face_set if name == ".sculpt_face_set" else None


class _Mesh:
    def __init__(self):
        self.vertices = _ArrayCollection(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        )
        self.loops = _ArrayCollection(
            [0, 1, 2, 2, 3, 4],  # shared edge 2 joins the two triangles
        )
        self.loops.values = [0, 1, 2, 2, 3, 4]
        self.loop_vertices = [0, 1, 2, 0, 2, 3]
        self.polygons = _ArrayCollection(
            [
                (0.6666667, 0.3333333, 0),
                (0.3333333, 0.6666667, 0),
            ]
        )
        self.polygons.loop_total = [3, 3]
        self.polygons.normals = [(0, 0, 1), (0, 0, 1)]
        self.polygons.hidden = [False, False]
        self.edges = [None] * 5
        self.attributes = _Attributes([1, 2])

    def as_pointer(self):
        return 200

    def update(self):
        return None


class _Object:
    def __init__(self):
        self.data = _Mesh()
        self.matrix_world = np.eye(4, dtype=np.float64)

    def as_pointer(self):
        return 100


def _patch_foreach_get(mesh):
    """Install property-aware collection readers on the tiny fake mesh."""

    original_loops = mesh.loops.foreach_get
    original_polygons = mesh.polygons.foreach_get

    def loops(prop, output):
        if prop == "vertex_index":
            np.copyto(output, np.asarray(mesh.loop_vertices, dtype=output.dtype))
        else:
            original_loops(prop, output)

    def polygons(prop, output):
        if prop == "loop_total":
            np.copyto(output, np.asarray(mesh.polygons.loop_total, dtype=output.dtype))
        elif prop == "normal":
            np.copyto(output, np.asarray(mesh.polygons.normals, dtype=output.dtype).ravel())
        elif prop == "hide":
            np.copyto(output, np.asarray(mesh.polygons.hidden, dtype=output.dtype))
        else:
            original_polygons(prop, output)

    mesh.loops.foreach_get = loops
    mesh.polygons.foreach_get = polygons


def _load_module():
    source = pathlib.Path(__file__).resolve().parents[1] / "mesh_focus_orbit.py"
    name = "_sfsf_cache_test_module"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_cache(module, obj):
    mesh = obj.data
    edges = np.asarray(mesh.loops.values, dtype=np.int32)
    vertices = np.asarray(mesh.loop_vertices, dtype=np.int32)
    totals = np.asarray(mesh.polygons.loop_total, dtype=np.int32)
    topology = module._fill_preview_topology_fingerprint(edges, vertices, totals)
    coords = np.asarray(mesh.vertices.values, dtype=np.float32)
    hidden = np.asarray(mesh.polygons.hidden, dtype=bool)
    centers = np.asarray(mesh.polygons.values, dtype=np.float64)
    normals = np.asarray(mesh.polygons.normals, dtype=np.float64)
    face_set = np.asarray(mesh.attributes.face_set.data.values, dtype=np.int32)
    return {
        "signature": module._fill_preview_signature(obj),
        "count": 2,
        "hidden": hidden.copy(),
        "topology_fingerprint": topology,
        "topology_first": np.asarray([0], dtype=np.int32),
        "topology_second": np.asarray([1], dtype=np.int32),
        "topology_edge_v0": np.asarray([0], dtype=np.int32),
        "topology_edge_v1": np.asarray([2], dtype=np.int32),
        "coordinate_fingerprint": module._fill_preview_array_fingerprint(coords),
        "face_set_fingerprint": module._fill_preview_array_fingerprint(face_set),
        "centers": centers.copy(),
        "normals": normals.copy(),
        "world_vertices": coords.astype(np.float64).copy(),
        "face_edge_counts": totals.copy(),
        "offsets": np.asarray([0, 1, 2], dtype=np.int64),
        "neighbors": np.asarray([1, 0], dtype=np.int32),
        "neighbor_lengths": np.asarray([1, 1], dtype=np.float64),
        "edge_indices": np.asarray([0, 0], dtype=np.int32),
        "edge_v0": np.asarray([0, 0], dtype=np.int32),
        "edge_v1": np.asarray([2, 2], dtype=np.int32),
        "first": np.asarray([0], dtype=np.int32),
        "second": np.asarray([1], dtype=np.int32),
        "pair_lengths": np.asarray([1], dtype=np.float64),
        "pair_v0": np.asarray([0], dtype=np.int32),
        "pair_v1": np.asarray([2], dtype=np.int32),
        "pair_edge_points": np.asarray([[[0, 0, 0], [1, 1, 0]]], dtype=np.float64),
        "geometry_dirty": False,
    }


def _test_real_builder(module):
    """Exercise the production cooperative builder on a temporary mesh."""

    import bpy

    mesh = bpy.data.meshes.new("_sfsf_cache_layer_test_mesh")
    mesh.from_pydata(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        [],
        [(0, 1, 2), (0, 2, 3)],
    )
    mesh.update()
    obj = bpy.data.objects.new("_sfsf_cache_layer_test_object", mesh)
    try:
        job = module._fill_preview_build_adjacency_cooperative(obj)
        while True:
            try:
                next(job)
            except StopIteration as done:
                cache = done.value
                break
        assert cache["count"] == 2
        assert len(cache["first"]) == 1
        assert "topology_fingerprint" in cache
        assert "topology_first" in cache
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def run():
    module = _load_module()
    _test_real_builder(module)
    obj = _Object()
    _patch_foreach_get(obj.data)
    cache = _make_cache(module, obj)
    signature = cache["signature"]
    module._fill_preview_adjacency_cache.clear()
    module._fill_preview_adjacency_cache[signature] = cache
    cache_identity = id(cache)
    adjacency_before = cache["first"].copy()
    topology_before = cache["topology_first"].copy()

    # External Face Set Paint: only the volatile Face Set values change.
    obj.data.attributes.face_set.data.values = [7, 7]
    module._fill_preview_mark_cache_pointers_dirty({200})
    assert cache["geometry_dirty"]
    assert module._fill_preview_refresh_cached_adjacency(obj, cache)
    assert id(module._fill_preview_adjacency_cache[signature]) == cache_identity
    assert cache["geometry_refresh_reason"] == "face-set-only"
    assert np.array_equal(cache["first"], adjacency_before)
    assert np.array_equal(cache["topology_first"], topology_before)

    # SFSF confirmation uses the same ordinary attribute write path.
    obj.data.attributes.face_set.data.foreach_set("value", [9, 9])
    obj.data.update()
    module._fill_preview_mark_cache_pointers_dirty({100, 200})
    assert module._fill_preview_refresh_cached_adjacency(obj, cache)
    assert cache["geometry_refresh_reason"] == "face-set-only"
    assert id(module._fill_preview_adjacency_cache[signature]) == cache_identity

    # A coordinate edit refreshes geometry in place, without changing the graph.
    obj.data.vertices.values[2] = (1, 1, 0.25)
    obj.data.polygons.values[0] = (0.6666667, 0.3333333, 0.0833333)
    module._fill_preview_mark_cache_pointers_dirty({200})
    assert module._fill_preview_refresh_cached_adjacency(obj, cache)
    assert cache["geometry_refresh_reason"] == "depsgraph-dirty"
    assert id(module._fill_preview_adjacency_cache[signature]) == cache_identity
    assert cache["world_vertices"][2, 2] == 0.25

    # Visibility is refreshed from the retained raw pair list.
    obj.data.polygons.hidden[1] = True
    module._fill_preview_mark_cache_pointers_dirty({200})
    assert module._fill_preview_refresh_cached_adjacency(obj, cache)
    assert cache["geometry_refresh_reason"] == "depsgraph-dirty"
    assert len(cache["first"]) == 0

    # Same-count rewiring fails the topology guard and must be rebuilt by caller.
    obj.data.polygons.hidden[1] = False
    obj.data.loops.values[2] = 5
    module._fill_preview_mark_cache_pointers_dirty({200})
    assert not module._fill_preview_refresh_cached_adjacency(obj, cache)

    return {
        "passed": True,
        "cache_identity_preserved": True,
        "face_set_refresh": "face-set-only",
        "geometry_refresh": "depsgraph-dirty",
        "visibility_refresh": "retained-topology",
        "same_count_rewire": "rebuild-required",
    }


if __name__ == "__main__":
    print(run())
