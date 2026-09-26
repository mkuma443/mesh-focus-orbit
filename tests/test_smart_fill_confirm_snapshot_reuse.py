"""Focused regression fixtures for the normal-E confirm snapshot cache.

Run inside Blender so the add-on's bpy imports are available::

    exec(compile(open(path).read(), path, "exec"), {"__name__": "__main__"})

The fixtures use only NumPy arrays and a temporary package import.  They do not
create or edit Blender data, scenes, meshes, or Face Sets.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np


def _load_module():
    package_dir = pathlib.Path(__file__).resolve().parents[1] / "mesh_focus_orbit"
    source = package_dir / "__init__.py"
    name = "_sfsf_confirm_snapshot_test_module"
    spec = importlib.util.spec_from_file_location(
        name,
        source,
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_invariants(preview):
    # Use the exact module object imported by preview so provenance sentinels
    # share their identity across the fixture and production path.
    return sys.modules[f"{preview.__package__}.invariants"]


def _geometry():
    count = 5
    return {
        "count": count,
        "face_ids": np.asarray([10, 11, 12, 13, 14], dtype=np.int32),
        "hidden": np.asarray([False, False, True, False, False], dtype=bool),
        "offsets": np.asarray([0, 1, 3, 5, 7, 8], dtype=np.int64),
        "neighbors": np.asarray([1, 0, 2, 1, 3, 2, 4, 3], dtype=np.int32),
        "geometry_dirty": False,
        "geometry_refresh_count": 0,
    }


def _state(geometry, **overrides):
    state = {
        "session_id": 31415,
        "adjacency": geometry,
        "signature": ("fixture-mesh", 101, 202),
        "seed_face": 10,
        "seed_local": 0,
        "strict_mode": False,
        "expand_only": False,
    }
    state.update(overrides)
    return state


def _assert_result_equal(left, right):
    left_snapshot, left_domain = left
    right_snapshot, right_domain = right
    assert (left_snapshot is None) == (right_snapshot is None)
    assert np.array_equal(left_domain, right_domain)
    if left_snapshot is None:
        return
    assert set(left_snapshot) == set(right_snapshot)
    for name in left_snapshot:
        assert np.array_equal(left_snapshot[name], right_snapshot[name]), name


def _test_radius_series(preview):
    geometry = _geometry()
    distances = np.asarray([0.0, 0.6, 1.0, 1.0, 2.0], dtype=np.float64)
    state = _state(geometry)
    radii = (0.75, 0.9375, 1.171875, 0.9375, 1.171875)
    expected = [
        preview._fill_preview_confirm_graph_snapshot(geometry, distances, radius)
        for radius in radii
    ]

    original = preview._fill_preview_confirm_graph_snapshot
    calls = []

    def counted(*args, **kwargs):
        calls.append(float(args[2]))
        return original(*args, **kwargs)

    preview._fill_preview_confirm_graph_snapshot = counted
    try:
        actual = [
            preview._fill_preview_confirm_graph_snapshot_for_session(
                state, geometry, distances, radius
            )
            for radius in radii
        ]
    finally:
        preview._fill_preview_confirm_graph_snapshot = original

    assert len(calls) == 1, calls
    first_snapshot = actual[0][0]
    assert first_snapshot is not None
    array_names = ("face_ids", "hidden", "offsets", "neighbors")
    for snapshot, _domain in actual:
        assert snapshot is not None
        for name in array_names:
            assert snapshot[name] is first_snapshot[name]
            assert not snapshot[name].flags.writeable
    for got, want in zip(actual, expected):
        _assert_result_equal(got, want)
    for index in range(1, len(actual)):
        assert actual[index][1] is not actual[index - 1][1]
    assert not np.array_equal(actual[1][1], actual[2][1])
    assert actual[2][1].tolist() == [0, 1, 3]
    return {"radius_stages": len(radii), "snapshot_builds": len(calls), "domains_recomputed": True}


def _test_key_misses(preview):
    distances = np.arange(5, dtype=np.float64) * 0.25

    def assert_miss(mutator):
        geometry = _geometry()
        state = _state(geometry)
        before, _domain = preview._fill_preview_confirm_graph_snapshot_for_session(
            state, geometry, distances, 0.5
        )
        assert before is not None
        mutator(state, geometry)
        call_geometry = state["adjacency"]
        after, _domain = preview._fill_preview_confirm_graph_snapshot_for_session(
            state, call_geometry, distances, 0.5
        )
        assert after is not None
        for name in ("face_ids", "hidden", "offsets", "neighbors"):
            assert after[name] is not before[name], name

    assert_miss(lambda state, geometry: state.__setitem__("signature", ("changed",)))
    assert_miss(lambda state, geometry: state.__setitem__("seed_face", 11))
    assert_miss(
        lambda state, geometry: geometry.__setitem__(
            "geometry_refresh_count", geometry["geometry_refresh_count"] + 1
        )
    )
    assert_miss(
        lambda state, geometry: state.__setitem__("adjacency", _geometry())
    )
    assert_miss(
        lambda state, geometry: geometry.__setitem__(
            "neighbors", geometry["neighbors"].copy()
        )
    )

    # Same ndarray identity and shape, changed dtype only: dtype belongs to the
    # cache key, so this must rebuild rather than reuse the old snapshot.
    geometry = _geometry()
    state = _state(geometry)
    before, _ = preview._fill_preview_confirm_graph_snapshot_for_session(
        state, geometry, distances, 0.5
    )
    neighbors = geometry["neighbors"]
    neighbors_id, neighbors_shape = id(neighbors), neighbors.shape
    neighbors.dtype = np.dtype(np.uint32)
    assert id(neighbors) == neighbors_id and neighbors.shape == neighbors_shape
    expected = preview._fill_preview_confirm_graph_snapshot(geometry, distances, 0.5)
    after = preview._fill_preview_confirm_graph_snapshot_for_session(
        state, geometry, distances, 0.5
    )
    _assert_result_equal(after, expected)
    assert all(after[0][name] is not before[name] for name in ("face_ids", "hidden", "offsets", "neighbors"))

    # Same identity and dtype, malformed rank/shape: preserve the original
    # helper's returned shape and do not reuse the prior snapshot.
    geometry = _geometry()
    state = _state(geometry)
    before, _ = preview._fill_preview_confirm_graph_snapshot_for_session(
        state, geometry, distances, 0.5
    )
    neighbors = geometry["neighbors"]
    neighbors.shape = (2, 4)
    expected = preview._fill_preview_confirm_graph_snapshot(geometry, distances, 0.5)
    after = preview._fill_preview_confirm_graph_snapshot_for_session(
        state, geometry, distances, 0.5
    )
    _assert_result_equal(after, expected)
    assert after[0]["neighbors"].shape == (2, 4)
    assert after[0]["neighbors"] is not before["neighbors"]
    return {"signature_seed_geometry_refresh_array_identity_dtype_shape": "miss"}


def _test_invalid_schema_fallback(preview):
    distances = np.arange(5, dtype=np.float64) * 0.25

    def check(geometry, values):
        state = _state(geometry)
        result = preview._fill_preview_confirm_graph_snapshot_for_session(
            state, geometry, values, 0.5
        )
        expected = preview._fill_preview_confirm_graph_snapshot(geometry, values, 0.5)
        _assert_result_equal(result, expected)
        assert preview._FILL_CONFIRM_GRAPH_CACHE_KEY not in state

    geometry = _geometry()
    geometry["count"] = 0
    check(geometry, distances)

    geometry = _geometry()
    check(geometry, distances[:-1])

    geometry = _geometry()
    geometry["hidden"] = geometry["hidden"][:-1]
    check(geometry, distances)

    geometry = _geometry()
    geometry["face_ids"] = geometry["face_ids"][:-1]
    check(geometry, distances)

    # Same identity and shape, but unsupported dtype is rejected and follows
    # the exact original cast/fallback behavior.
    geometry = _geometry()
    geometry["face_ids"].dtype = np.dtype(np.float32)
    check(geometry, distances)
    return {"count_distances_hidden_face_ids_dtype": "fallback_matches_original"}


def _test_identity_provenance_copy_and_invalidation(preview):
    invariants = _load_invariants(preview)
    geometry = _geometry()
    face_ids, provenance = invariants.make_mesh_global_identity_face_ids(
        geometry["count"]
    )
    geometry["face_ids"] = face_ids
    geometry[invariants._FACE_IDS_IDENTITY_PROVENANCE_KEY] = provenance
    state = _state(geometry, seed_face=0, seed_local=0)
    distances = np.arange(5, dtype=np.float64) * 0.25

    snapshot, _domain = preview._fill_preview_confirm_graph_snapshot_for_session(
        state, geometry, distances, 0.5
    )
    assert invariants.has_mesh_global_identity_face_ids(snapshot)
    assert snapshot["face_ids"] is not face_ids
    assert not snapshot["face_ids"].flags.writeable
    assert np.array_equal(
        invariants.accepted_graph_rows_from_result(
            {"faces": [4, 0, 4], "confirm_geometry": snapshot}
        ),
        np.asarray([0, 4], dtype=np.int32),
    )

    # Replacing the graph's IDs invalidates the source token and forces a new
    # legacy snapshot without identity provenance.
    geometry["face_ids"] = np.asarray([20, 10, 30, 40, 50], dtype=np.int32)
    replaced, _domain = preview._fill_preview_confirm_graph_snapshot_for_session(
        state, geometry, distances, 0.5
    )
    assert not invariants.has_mesh_global_identity_face_ids(geometry)
    assert not invariants.has_mesh_global_identity_face_ids(replaced)

    dirty_geometry = _geometry()
    dirty_ids, dirty_provenance = invariants.make_mesh_global_identity_face_ids(5)
    dirty_geometry["face_ids"] = dirty_ids
    dirty_geometry[invariants._FACE_IDS_IDENTITY_PROVENANCE_KEY] = dirty_provenance
    dirty_geometry["geometry_dirty"] = True
    dirty_state = _state(dirty_geometry, seed_face=0, seed_local=0)
    dirty_snapshot, _domain = preview._fill_preview_confirm_graph_snapshot_for_session(
        dirty_state, dirty_geometry, distances, 0.5
    )
    assert not invariants.has_mesh_global_identity_face_ids(dirty_snapshot)
    return {"snapshot_copy": "rebound-readonly", "replacement": "fallback", "dirty": "fallback"}


def _test_dirty_and_lifetime(preview, runtime):
    geometry = _geometry()
    distances = np.arange(5, dtype=np.float64) * 0.25
    state = _state(geometry)
    first, _ = preview._fill_preview_confirm_graph_snapshot_for_session(
        state, geometry, distances, 0.5
    )
    geometry["geometry_dirty"] = True
    dirty, expected = preview._fill_preview_confirm_graph_snapshot_for_session(
        state, geometry, distances, 0.75
    )
    _assert_result_equal((dirty, expected), preview._fill_preview_confirm_graph_snapshot(geometry, distances, 0.75))
    assert preview._FILL_CONFIRM_GRAPH_CACHE_KEY not in state
    geometry["geometry_dirty"] = False
    rebuilt, _ = preview._fill_preview_confirm_graph_snapshot_for_session(
        state, geometry, distances, 0.5
    )
    assert all(rebuilt[name] is not first[name] for name in ("face_ids", "hidden", "offsets", "neighbors"))

    new_session = _state(geometry)
    new_session_snapshot, _ = preview._fill_preview_confirm_graph_snapshot_for_session(
        new_session, geometry, distances, 0.5
    )
    assert all(new_session_snapshot[name] is not rebuilt[name] for name in ("face_ids", "hidden", "offsets", "neighbors"))

    stop_draw, tag_redraw = preview._fill_preview_stop_draw, preview._fill_preview_tag_redraw
    previous_state = runtime.fill_preview_state
    preview._fill_preview_stop_draw = lambda: None
    preview._fill_preview_tag_redraw = lambda _state: None
    try:
        state["metrics"] = {}
        state["timer"] = None
        state["result"] = None
        runtime.fill_preview_state = state
        preview._fill_preview_cancel(state, "fixture-cancel")
        assert runtime.fill_preview_state is None
        assert preview._FILL_CONFIRM_GRAPH_CACHE_KEY not in state
    finally:
        runtime.fill_preview_state = previous_state
        preview._fill_preview_stop_draw = stop_draw
        preview._fill_preview_tag_redraw = tag_redraw

    post_cancel_session = _state(geometry)
    post_cancel, _ = preview._fill_preview_confirm_graph_snapshot_for_session(
        post_cancel_session, geometry, distances, 0.5
    )
    assert all(post_cancel[name] is not rebuilt[name] for name in ("face_ids", "hidden", "offsets", "neighbors"))
    return {"dirty": "cleared", "new_session": "isolated", "cancel": "cleared"}


def run():
    module = _load_module()
    preview = module.smart_fill_preview
    results = {
        "radius_series": _test_radius_series(preview),
        "key_misses": _test_key_misses(preview),
        "invalid_schema": _test_invalid_schema_fallback(preview),
        "identity_provenance": _test_identity_provenance_copy_and_invalidation(preview),
        "lifetime": _test_dirty_and_lifetime(preview, module.runtime),
    }
    results["passed"] = True
    return results


if __name__ == "__main__":
    print(run())
