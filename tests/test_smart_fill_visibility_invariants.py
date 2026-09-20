"""Pure Smart Fill visibility/monotonicity regression tests.

This file loads the no-Blender invariant module directly, so it can run after
an interrupted Blender test without importing bpy or touching a live process.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _load_invariants():
    path = (
        Path(__file__).resolve().parents[1]
        / "mesh_focus_orbit"
        / "smart_fill"
        / "invariants.py"
    )
    spec = spec_from_file_location("_mfo_smart_fill_invariants_test", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _csr(edges, count):
    rows = [[] for _ in range(count)]
    for left, right in edges:
        rows[left].append(right)
        rows[right].append(left)
    offsets = [0]
    neighbors = []
    for row in rows:
        neighbors.extend(sorted(row))
        offsets.append(len(neighbors))
    return np.asarray(offsets, dtype=np.int64), np.asarray(neighbors, dtype=np.int32)


def test_hidden_strip_is_a_stable_barrier_and_seed_component_is_fixed():
    module = _load_invariants()
    offsets, neighbors = _csr([(0, 1), (1, 2), (2, 3), (3, 4)], 5)
    visible, component, reason = module.visible_seed_component(
        5, offsets, neighbors, np.asarray([False, False, True, False, False]), 0
    )
    assert reason == "ok"
    assert np.array_equal(np.flatnonzero(visible), np.asarray([0, 1, 3, 4]))
    assert np.array_equal(np.flatnonzero(component), np.asarray([0, 1]))


def test_disconnected_visible_island_never_enters_progressive_selection():
    module = _load_invariants()
    offsets, neighbors = _csr([(0, 1), (1, 2), (3, 4)], 5)
    visible, component, reason = module.visible_seed_component(
        5, offsets, neighbors, np.zeros(5, dtype=bool), 0
    )
    assert reason == "ok"
    merged, reason = module.monotonic_visible_selection(
        [0, 1], [0, 1, 2, 3, 4], visible, component, 0
    )
    assert reason == "ok"
    assert np.array_equal(merged, np.asarray([0, 1, 2]))
    assert module.selection_monotonic([0, 1], merged)


def test_seed_is_pinned_and_hidden_faces_are_never_reintroduced():
    module = _load_invariants()
    offsets, neighbors = _csr([(0, 1), (1, 2), (2, 3)], 4)
    visible, component, reason = module.visible_seed_component(
        4, offsets, neighbors, np.asarray([False, False, True, False]), 0
    )
    assert reason == "ok"
    merged, reason = module.monotonic_visible_selection(
        [0, 1], [1, 2, 3], visible, component, 0
    )
    assert reason == "ok"
    assert np.array_equal(merged, np.asarray([0, 1]))
    assert 2 not in set(merged.tolist())
    assert 3 not in set(merged.tolist())


def test_nearby_hidden_front_surface_cannot_leak_into_exposed_back_surface():
    module = _load_invariants()
    # Two spatially close layers intentionally have no topological adjacency:
    # the front layer is hidden, while the exposed back layer is the seed
    # component.  Geometry/normal proximity must not create a bridge.
    offsets, neighbors = _csr(
        [(0, 1), (1, 2), (3, 4), (4, 5)],
        6,
    )
    hidden = np.asarray([True, True, True, False, False, False])
    visible, component, reason = module.visible_seed_component(
        6, offsets, neighbors, hidden, 3
    )
    assert reason == "ok"
    assert np.array_equal(np.flatnonzero(visible), np.asarray([3, 4, 5]))
    assert np.array_equal(np.flatnonzero(component), np.asarray([3, 4, 5]))
    grown, reason = module.monotonic_visible_selection(
        [3], [3, 4, 5, 0, 1, 2], visible, component, 3
    )
    assert reason == "ok"
    assert np.array_equal(grown, np.asarray([3, 4, 5]))


def test_onion_shells_keep_the_exposed_seed_layer_as_the_only_component():
    module = _load_invariants()
    # Three concentric shells are deliberately close in 3D but have no
    # topological links.  The outer shell is hidden and the inner shell is
    # another nearby surface; only the exposed middle shell may be analysed.
    offsets, neighbors = _csr(
        [
            (0, 1), (1, 2), (2, 0),
            (3, 4), (4, 5), (5, 3),
            (6, 7), (7, 8), (8, 6),
        ],
        9,
    )
    hidden = np.asarray(
        [True, True, True, False, False, False, True, True, True]
    )
    visible, component, reason = module.visible_seed_component(
        9, offsets, neighbors, hidden, 4
    )
    assert reason == "ok"
    assert np.array_equal(np.flatnonzero(visible), np.asarray([3, 4, 5]))
    assert np.array_equal(np.flatnonzero(component), np.asarray([3, 4, 5]))


def test_concave_cuff_occluder_cannot_replace_the_seed_component():
    module = _load_invariants()
    # A self-occluding cuff has a hidden front rim near a visible recessed
    # surface.  The two strips share no mesh edge; screen/depth proximity must
    # not replace the recessed seed component on a later radius stage.
    offsets, neighbors = _csr(
        [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)],
        8,
    )
    hidden = np.asarray([True, True, True, True, False, False, False, False])
    visible, component, reason = module.visible_seed_component(
        8, offsets, neighbors, hidden, 5
    )
    assert reason == "ok"
    assert np.array_equal(np.flatnonzero(component), np.asarray([4, 5, 6, 7]))
    stage_a, reason = module.monotonic_visible_selection(
        [5], [4, 5, 6], visible, component, 5
    )
    assert reason == "ok"
    stage_b, reason = module.monotonic_visible_selection(
        stage_a, [0, 1, 2, 5, 6, 7], visible, component, 5
    )
    assert reason == "ok"
    assert np.array_equal(stage_b, np.asarray([4, 5, 6, 7]))
    assert module.selection_monotonic(stage_a, stage_b)


def test_grow_is_superset_from_the_immediately_previous_stage():
    module = _load_invariants()
    offsets, neighbors = _csr([(0, 1), (1, 2), (2, 3)], 4)
    visible, component, reason = module.visible_seed_component(
        4, offsets, neighbors, np.zeros(4, dtype=bool), 0
    )
    assert reason == "ok"
    stage0 = np.asarray([0], dtype=np.int32)
    stage1, reason = module.monotonic_visible_selection(
        stage0, [0, 1], visible, component, 0
    )
    assert reason == "ok"
    stage2, reason = module.monotonic_visible_selection(
        stage1, [0, 1, 2], visible, component, 0
    )
    assert reason == "ok"
    assert module.selection_monotonic(stage0, stage1)
    assert module.selection_monotonic(stage1, stage2)
    # A later shrink is allowed to start a fresh computation. The invariant
    # required here is the grow-side superset of the immediately preceding
    # displayed stage.
    assert module.selection_monotonic(stage1, stage2)


def test_high_global_face_ids_map_to_full_graph_rows_for_floor_and_confirm():
    module = _load_invariants()
    # The analysis patch and immutable confirm graph have different row
    # spaces. Production must map accepted global ids into the full graph,
    # never union analysis-patch row numbers into the floor.
    patch_global = np.asarray([12003, 1007, 4096], dtype=np.int32)
    full_graph_global = np.asarray(
        [1007, 4096, 12003, 50000, 90000], dtype=np.int32
    )
    accepted_global = np.asarray([1007, 12003], dtype=np.int32)
    patch_rows = module.map_global_faces_to_local(patch_global, accepted_global)
    full_rows = module.map_global_faces_to_local(full_graph_global, accepted_global)
    assert np.array_equal(patch_rows, np.asarray([0, 1], dtype=np.int32))
    assert np.array_equal(full_rows, np.asarray([0, 2], dtype=np.int32))
    visible = np.ones(len(full_graph_global), dtype=bool)
    component = np.ones(len(full_graph_global), dtype=bool)
    merged, reason = module.monotonic_visible_selection(
        [0], full_rows, visible, component, 0
    )
    assert reason == "ok"
    assert np.array_equal(merged, np.asarray([0, 2], dtype=np.int32))
    assert 1 not in set(merged.tolist())
    result = {
        "faces": accepted_global,
        "confirm_geometry": {
            "face_ids": full_graph_global,
            "count": len(full_graph_global),
        },
    }
    assert np.array_equal(
        module.accepted_graph_rows_from_result(result), full_rows
    )


def test_normal_cache_hit_requires_current_floor_and_bypasses_shrink():
    module = _load_invariants()
    full_graph = np.asarray([1007, 4096, 12003], dtype=np.int32)
    cached_a = {
        "faces": np.asarray([1007], dtype=np.int32),
        "confirm_geometry": {"face_ids": full_graph, "count": len(full_graph)},
    }
    # A at radius 1.0 is a valid hit for its own immediately preceding floor.
    assert module.normal_cache_hit_allowed(cached_a, 1.0, 1.0, [0])
    # A -> shrink B changes the displayed floor.  Reusing the old radius-only
    # A entry would drop B's accepted face and is therefore forbidden.
    assert not module.normal_cache_hit_allowed(cached_a, 0.8, 1.0, [0, 2])
    # Shrink itself always bypasses the cache, even if the entry contains the
    # current floor by coincidence.
    assert not module.normal_cache_hit_allowed(cached_a, 1.0, 0.8, [0])
    cached_b = {
        "faces": np.asarray([1007, 12003], dtype=np.int32),
        "confirm_geometry": {"face_ids": full_graph, "count": len(full_graph)},
    }
    assert module.normal_cache_hit_allowed(cached_b, 0.8, 1.0, [0, 2])
    # The helper validates the same full candidate provenance used by the
    # renderer/confirm path; it never repairs a cache hit by faces-only union.
    assert np.array_equal(
        module.accepted_graph_rows_from_result(cached_b),
        np.asarray([0, 2], dtype=np.int32),
    )


def test_terminal_radius_publications_keep_bounded_render_cache():
    module = _load_invariants()
    cache = {}
    for radius in (1.0, 1.25, 1.5625, 1.953125, 2.44140625):
        module.publish_bounded_result(
            cache,
            ("progressive-range", radius),
            {"radius": radius, "confirm_geometry": {"face_ids": [0], "count": 1}},
            limit=3,
        )
        assert len(cache) <= 3
    assert [key[1] for key in cache] == [1.5625, 1.953125, 2.44140625]
    # Re-publishing a previously evicted/revisited terminal radius remains
    # bounded and moves only that entry to the newest position.
    module.publish_bounded_result(
        cache,
        ("progressive-range", 1.953125),
        {"radius": 1.953125, "confirm_geometry": {"face_ids": [0], "count": 1}},
        limit=3,
    )
    assert len(cache) == 3
    assert [key[1] for key in cache] == [1.5625, 2.44140625, 1.953125]


def test_reached_component_avoids_full_mesh_bfs_on_large_sparse_graph():
    module = _load_invariants()
    count = 2_000_000
    hidden = np.zeros(count, dtype=bool)
    hidden[1_500_000] = True
    visible, component, reason = module.visible_seed_reached_component(
        count, hidden, 7, np.asarray([7, 100_000, 1_500_000], dtype=np.int32)
    )
    assert reason == "ok"
    assert visible[7] and visible[100_000]
    assert component[7] and component[100_000]
    assert not component[1_500_000]
    assert int(np.count_nonzero(component)) == 2


def test_out_of_range_floor_ids_are_filtered_before_mask_indexing():
    module = _load_invariants()
    visible = np.ones(4, dtype=bool)
    component = np.ones(4, dtype=bool)
    merged, reason = module.monotonic_visible_selection(
        np.asarray([-9, 9000], dtype=np.int32),
        np.asarray([0, 3], dtype=np.int32),
        visible,
        component,
        0,
    )
    assert reason == "ok"
    assert np.array_equal(merged, np.asarray([0, 3], dtype=np.int32))


def test_strict_crop_growth_does_not_apply_normal_visible_floor():
    module = _load_invariants()
    # Ctrl+E may expand a compact cursor crop, changing local row ids.  Only
    # Normal E applies the immutable visible-component floor to those rows.
    assert module.visible_floor_enabled(False)
    assert not module.visible_floor_enabled(True)
    previous_crop = np.asarray([0, 1], dtype=np.int32)
    expanded_crop = np.asarray([4, 5, 6], dtype=np.int32)
    assert module.selection_monotonic(previous_crop, expanded_crop) is False


def run():
    tests = [
        test_hidden_strip_is_a_stable_barrier_and_seed_component_is_fixed,
        test_disconnected_visible_island_never_enters_progressive_selection,
        test_seed_is_pinned_and_hidden_faces_are_never_reintroduced,
        test_nearby_hidden_front_surface_cannot_leak_into_exposed_back_surface,
        test_onion_shells_keep_the_exposed_seed_layer_as_the_only_component,
        test_concave_cuff_occluder_cannot_replace_the_seed_component,
        test_grow_is_superset_from_the_immediately_previous_stage,
        test_high_global_face_ids_map_to_full_graph_rows_for_floor_and_confirm,
        test_normal_cache_hit_requires_current_floor_and_bypasses_shrink,
        test_terminal_radius_publications_keep_bounded_render_cache,
        test_strict_crop_growth_does_not_apply_normal_visible_floor,
        test_reached_component_avoids_full_mesh_bfs_on_large_sparse_graph,
        test_out_of_range_floor_ids_are_filtered_before_mask_indexing,
    ]
    for test in tests:
        test()
    return {"passed": len(tests), "blender_imported": False}


if __name__ == "__main__":
    print(run())
