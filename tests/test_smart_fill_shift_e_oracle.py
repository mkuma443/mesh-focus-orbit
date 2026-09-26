"""Independent legacy oracle and fixtures for Smart Fill preview work.

This module intentionally does not import the add-on.  The legacy oracle copies
the old full-count selection and global-pair crossing rules so production
localization can be compared against frozen observable output.
"""

import numpy as np
import ast
from pathlib import Path


def _shift_fixture():
    count = 10
    # Compact/global mapping is deliberately nonidentity.  Isolated hidden,
    # disconnected, mesh-border, and nonmanifold faces remain represented.
    face_ids = np.asarray([40, 42, 43, 50, 70, 71, 90, 91, 105, 120], dtype=np.int32)
    radius = 1.0
    tolerance = max(radius * 1.0e-8, 1.0e-9)
    distances = np.asarray(
        [
            0.0,
            0.4,
            radius + tolerance,       # included exactly at the limit
            radius + tolerance + 1.e-5,
            1.2,
            0.1,                      # hidden, so excluded
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ],
        dtype=np.float64,
    )
    hidden = np.asarray(
        [False, False, False, False, False, True, False, False, False, False],
        dtype=bool,
    )
    # Rows 2 and 3 have identical endpoints but are distinct pair rows and
    # therefore must both survive. Row 4 is a synthetic seam pair.
    first = np.asarray([0, 1, 2, 2, 2, 6], dtype=np.int32)
    second = np.asarray([1, 2, 3, 3, 4, 7], dtype=np.int32)
    pair_v0 = np.asarray([0, 1, 2, 2, 3, 6], dtype=np.int32)
    pair_v1 = np.asarray([1, 2, 3, 3, 4, 7], dtype=np.int32)
    world_vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )
    pair_kind = ("manifold", "manifold", "manifold", "manifold", "seam", "manifold")
    # These raw mesh conditions intentionally have no graph pair row in the
    # prepared visible graph: hidden pair, an open border, and a nonmanifold
    # edge. The oracle must preserve that existing graph-derived output.
    omitted_mesh_edges = (
        (4, 5, "hidden"),
        (8, 9, "nonmanifold"),
        (9, -1, "mesh-border"),
    )
    return {
        "count": count,
        "radius": radius,
        "tolerance": tolerance,
        "seed_mesh_face": int(face_ids[0]),
        "seed_local": 0,
        "face_ids": face_ids,
        "distances": distances,
        "hidden": hidden,
        "first": first,
        "second": second,
        "pair_v0": pair_v0,
        "pair_v1": pair_v1,
        "world_vertices": world_vertices,
        "pair_kind": pair_kind,
        "omitted_mesh_edges": omitted_mesh_edges,
        "csr_edge_indices_space": "pair-row",
    }


def _legacy_shift_oracle(graph):
    """Frozen equivalent of the old full-mask/global-pair Shift+E path."""
    count = int(graph["count"])
    distances = np.asarray(graph["distances"], dtype=np.float64).reshape(-1)
    hidden = np.asarray(graph["hidden"], dtype=bool).reshape(-1)
    radius = float(graph["radius"])
    tolerance = max(radius * 1.0e-8, 1.0e-9)
    selected_mask = (
        np.isfinite(distances[:count])
        & (distances[:count] <= radius + tolerance)
        & ~hidden[:count]
    )
    selected_ids = np.flatnonzero(selected_mask).astype(np.int32)
    first = np.asarray(graph["first"], dtype=np.int32).reshape(-1)
    second = np.asarray(graph["second"], dtype=np.int32).reshape(-1)
    crossing_pair_rows = np.flatnonzero(
        selected_mask[first] != selected_mask[second]
    ).astype(np.int32)
    face_ids = np.asarray(graph["face_ids"], dtype=np.int32).reshape(-1)
    pair_v0 = np.asarray(graph["pair_v0"], dtype=np.int32).reshape(-1)
    pair_v1 = np.asarray(graph["pair_v1"], dtype=np.int32).reshape(-1)
    vertices = np.asarray(graph["world_vertices"], dtype=np.float64).reshape(-1, 3)
    segments = tuple(
        (
            tuple(float(value) for value in vertices[int(pair_v0[row])]),
            tuple(float(value) for value in vertices[int(pair_v1[row])]),
        )
        for row in crossing_pair_rows
    )
    records = tuple(
        {
            "geometry_face_a": int(first[row]),
            "geometry_face_b": int(second[row]),
            "mesh_face_a": int(face_ids[int(first[row])]),
            "mesh_face_b": int(face_ids[int(second[row])]),
            "mesh_edge": -1,
            "shape": False,
        }
        for row in crossing_pair_rows
    )
    domain_ids = np.flatnonzero(
        np.isfinite(distances[:count])
        & (distances[:count] <= radius + tolerance)
        & ~hidden[:count]
    ).astype(np.int32)
    face_seed_rows = np.flatnonzero(
        face_ids == int(graph["seed_mesh_face"])
    ).astype(np.int32)
    return {
        "selected_ids": selected_ids,
        "selected_mask": selected_mask,
        "crossing_pair_rows": crossing_pair_rows,
        "distance_segments": segments,
        "shape_segments": (),
        "boundary_records": records,
        "confirm_domain_ids": domain_ids,
        "confirm_seed_rows": face_seed_rows,
    }


def _csr_pair_rows(graph):
    """Build deterministic CSR rows with an explicit pair-row mapping."""
    count = int(graph["count"])
    first = np.asarray(graph["first"], dtype=np.int32).reshape(-1)
    second = np.asarray(graph["second"], dtype=np.int32).reshape(-1)
    pair_rows = np.arange(len(first), dtype=np.int32)
    sources = np.concatenate((first, second))
    destinations = np.concatenate((second, first))
    directed_pair_rows = np.concatenate((pair_rows, pair_rows))
    order = np.argsort(sources, kind="stable")
    degree = np.bincount(sources, minlength=count)
    offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(degree, dtype=np.int64))
    )
    return {
        "offsets": offsets,
        "neighbors": destinations[order],
        "edge_indices": directed_pair_rows[order],
        "csr_edge_indices_space": "pair-row",
    }


def _candidate_local_pair_rows(graph, selected_ids, csr):
    """Independent model of CSR incident-row collection for the fixture."""
    selected_ids = np.asarray(selected_ids, dtype=np.int32).reshape(-1)
    selected_set = set(int(value) for value in selected_ids)
    offsets = np.asarray(csr["offsets"], dtype=np.int64).reshape(-1)
    neighbors = np.asarray(csr["neighbors"], dtype=np.int32).reshape(-1)
    edge_indices = np.asarray(csr["edge_indices"], dtype=np.int32).reshape(-1)
    first = np.asarray(graph["first"], dtype=np.int32).reshape(-1)
    second = np.asarray(graph["second"], dtype=np.int32).reshape(-1)
    incident_pair_rows = set()
    for face in selected_ids:
        start, end = int(offsets[int(face)]), int(offsets[int(face) + 1])
        for slot in range(start, end):
            incident_pair_rows.add(int(edge_indices[slot]))
    crossing = [
        row
        for row in sorted(incident_pair_rows)
        if (int(first[row]) in selected_set) != (int(second[row]) in selected_set)
    ]
    return np.asarray(crossing, dtype=np.int32)


def test_legacy_oracle_contract():
    graph = _shift_fixture()
    oracle = _legacy_shift_oracle(graph)
    assert oracle["selected_ids"].tolist() == [0, 1, 2]
    assert oracle["crossing_pair_rows"].tolist() == [2, 3, 4]
    assert len(oracle["distance_segments"]) == 3
    assert oracle["shape_segments"] == ()
    assert [record["mesh_edge"] for record in oracle["boundary_records"]] == [-1, -1, -1]
    assert [record["shape"] for record in oracle["boundary_records"]] == [False, False, False]
    assert oracle["confirm_domain_ids"].tolist() == [0, 1, 2]
    assert oracle["confirm_seed_rows"].tolist() == [0]
    assert graph["pair_kind"][4] == "seam"
    assert graph["omitted_mesh_edges"] == (
        (4, 5, "hidden"),
        (8, 9, "nonmanifold"),
        (9, -1, "mesh-border"),
    )


def test_pair_row_localization_matches_legacy_oracle():
    graph = _shift_fixture()
    csr = _csr_pair_rows(graph)
    oracle = _legacy_shift_oracle(graph)
    namespace = _load_snapshot_production_functions()
    local_graph = {
        "count": graph["count"],
        "offsets": csr["offsets"],
        "edge_indices": csr["edge_indices"],
        "edge_indices_space": "pair-row",
    }
    local_rows = namespace[
        "_fill_preview_expand_only_local_boundary_rows"
    ](local_graph, oracle["selected_ids"], graph["first"], graph["second"])
    assert csr["csr_edge_indices_space"] == "pair-row"
    assert local_rows.tolist() == oracle["crossing_pair_rows"].tolist()
    assert len(np.unique(local_rows)) == len(local_rows)
    # Old cache schemas and cursor-local edge-id namespaces are deliberately
    # rejected so the production caller can retain the global-pair fallback.
    assert namespace["_fill_preview_expand_only_local_boundary_rows"](
        {**local_graph, "edge_indices_space": "mesh-edge-id"},
        oracle["selected_ids"],
        graph["first"],
        graph["second"],
    ) is None
    assert namespace["_fill_preview_expand_only_local_boundary_rows"](
        {key: value for key, value in local_graph.items() if key != "edge_indices_space"},
        oracle["selected_ids"],
        graph["first"],
        graph["second"],
    ) is None
    # Pair rows 2 and 3 intentionally share endpoints and remain distinct.
    assert graph["pair_v0"][2] == graph["pair_v0"][3]
    assert graph["pair_v1"][2] == graph["pair_v1"][3]
    shape, distance, records = namespace[
        "_fill_preview_materialize_boundary_rows"
    ](
        local_rows,
        graph["first"],
        graph["second"],
        graph["pair_v0"],
        graph["pair_v1"],
        graph["world_vertices"],
        graph["face_ids"],
    )
    assert shape == []
    assert tuple(distance) == oracle["distance_segments"]
    assert tuple(records) == oracle["boundary_records"]


def _legacy_normal_materializer(
    rows, first, second, pair_v0, pair_v1, vertices, face_ids,
    draw_full_geometry, analysis_ids, shape_mask,
):
    """Independent copy of the pre-refactor normal/strict row loop."""
    shape_segments = []
    distance_segments = []
    records = []
    for edge in rows:
        edge_index = -1
        if (
            edge >= len(pair_v0)
            or edge >= len(pair_v1)
            or int(pair_v0[edge]) < 0
            or int(pair_v1[edge]) < 0
            or int(pair_v0[edge]) >= len(vertices)
            or int(pair_v1[edge]) >= len(vertices)
        ):
            continue
        segment = (
            tuple(float(value) for value in vertices[int(pair_v0[edge])]),
            tuple(float(value) for value in vertices[int(pair_v1[edge])]),
        )
        geometry_face_a = int(
            first[edge] if draw_full_geometry else analysis_ids[int(first[edge])]
        )
        geometry_face_b = int(
            second[edge] if draw_full_geometry else analysis_ids[int(second[edge])]
        )
        shape_boundary = bool(shape_mask[edge])
        records.append(
            {
                "geometry_face_a": geometry_face_a,
                "geometry_face_b": geometry_face_b,
                "mesh_face_a": int(face_ids[geometry_face_a])
                if face_ids is not None
                else -1,
                "mesh_face_b": int(face_ids[geometry_face_b])
                if face_ids is not None
                else -1,
                "mesh_edge": int(edge_index),
                "shape": bool(shape_boundary),
            }
        )
        (shape_segments if shape_boundary else distance_segments).append(segment)
    return shape_segments, distance_segments, records


def test_shared_materializer_preserves_normal_and_strict_records():
    namespace = _load_snapshot_production_functions()
    materialize = namespace["_fill_preview_materialize_boundary_rows"]
    rows = np.asarray([0, 1, 2, 4], dtype=np.int32)
    first = np.asarray([0, 1, 2, 4, 3], dtype=np.int32)
    second = np.asarray([1, 2, 4, 0, 2], dtype=np.int32)
    pair_v0 = np.asarray([0, 1, -1, 2, 3], dtype=np.int32)
    pair_v1 = np.asarray([1, 2, 3, 3, 9], dtype=np.int32)
    vertices = np.asarray(
        [[0., 0., 0.], [1., 0., 0.], [2., 1., 0.], [3., 1., 0.]],
        dtype=np.float64,
    )
    face_ids = np.asarray([60, 62, 70, 74, 80, 90, 95, 101], dtype=np.int32)
    shape_mask = np.asarray([False, True, True, False, True], dtype=bool)
    analysis_ids = np.asarray([4, 1, 6, 3, 5], dtype=np.int32)
    for draw_full, mapping in ((True, None), (False, analysis_ids)):
        expected = _legacy_normal_materializer(
            rows, first, second, pair_v0, pair_v1, vertices, face_ids,
            draw_full, mapping, shape_mask,
        )
        actual = materialize(
            rows, first, second, pair_v0, pair_v1, vertices, face_ids,
            draw_full_geometry=draw_full,
            analysis_ids=mapping,
            shape_boundary_mask=shape_mask,
        )
        assert actual == expected


def test_pair_row_tag_is_only_emitted_by_full_graph_builder_and_refresh():
    root = Path(__file__).resolve().parents[1]
    source = (root / "mesh_focus_orbit" / "smart_fill" / "geometry.py").read_text(
        encoding="utf-8"
    )
    assert source.count('"edge_indices_space": "pair-row"') == 2


def _load_snapshot_production_functions():
    """Load the small pure snapshot functions without importing Blender."""
    root = Path(__file__).resolve().parents[1]
    invariant_tree = ast.parse(
        (root / "mesh_focus_orbit" / "smart_fill" / "invariants.py").read_text(
            encoding="utf-8"
        )
    )
    preview_tree = ast.parse(
        (root / "mesh_focus_orbit" / "smart_fill" / "preview.py").read_text(
            encoding="utf-8"
        )
    )
    wanted_invariant_functions = {
        "make_mesh_global_identity_face_ids",
        "has_mesh_global_identity_face_ids",
        "copy_mesh_global_identity_face_ids_provenance",
    }
    wanted_invariant_assignments = {
        "_FACE_IDS_IDENTITY_PROVENANCE_KEY",
        "_FACE_IDS_IDENTITY_SENTINEL",
    }
    wanted_preview_functions = {
        "_fill_preview_expand_only_local_boundary_rows",
        "_fill_preview_materialize_boundary_rows",
        "_fill_preview_confirm_domain_ids_from_reached",
        "_fill_preview_confirm_graph_snapshot",
        "_fill_preview_confirm_graph_snapshot_for_session",
    }
    nodes = []
    for node in invariant_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_invariant_functions:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in wanted_invariant_assignments
            for target in node.targets
        ):
            nodes.append(node)
    for node in preview_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_preview_functions:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "_FILL_CONFIRM_GRAPH_CACHE_KEY"
            for target in node.targets
        ):
            nodes.append(node)
    namespace = {"np": np}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, "smart_fill_snapshot_test_extract.py", "exec"), namespace)
    return namespace


def _snapshot_fixture(namespace):
    count = 8
    face_ids, provenance = namespace["make_mesh_global_identity_face_ids"](count)
    hidden = np.asarray([False, False, False, True, False, False, False, False])
    offsets = np.zeros(count + 1, dtype=np.int64)
    neighbors = np.empty(0, dtype=np.int32)
    graph = {
        "count": count,
        "face_ids": face_ids,
        "hidden": hidden,
        "offsets": offsets,
        "neighbors": neighbors,
        "geometry_refresh_count": 0,
        "geometry_dirty": False,
        namespace["_FACE_IDS_IDENTITY_PROVENANCE_KEY"]: provenance,
    }
    distances = np.asarray(
        [0.0, 0.25, 1.0, 0.1, 1.0001, 1.75, np.inf, np.inf],
        dtype=np.float64,
    )
    reached_ids = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32)
    state = {
        "session_id": 19,
        "signature": ("fixture", 8),
        "adjacency": graph,
        "seed_face": 0,
        "seed_local": 0,
        "strict_mode": False,
        "expand_only": False,
    }
    return graph, distances, reached_ids, state


def _legacy_confirm_domain(distances, hidden, radius):
    tolerance = max(float(radius) * 1.0e-8, 1.0e-9)
    return np.flatnonzero(
        np.isfinite(distances)
        & (distances <= float(radius) + tolerance)
        & ~hidden
    ).astype(np.int32)


def test_progressive_domain_matches_legacy_and_requires_complete_bound():
    namespace = _load_snapshot_production_functions()
    graph, distances, reached_ids, _state = _snapshot_fixture(namespace)
    local = namespace["_fill_preview_confirm_domain_ids_from_reached"](
        distances, graph["hidden"], graph["count"], 1.0, reached_ids, 2.0
    )
    assert local.tolist() == _legacy_confirm_domain(
        distances, graph["hidden"], 1.0
    ).tolist()
    # A bound that does not cover radius+tolerance and an old list schema both
    # request the legacy full-count fallback.
    assert namespace["_fill_preview_confirm_domain_ids_from_reached"](
        distances, graph["hidden"], graph["count"], 1.0, reached_ids, 1.0
    ) is None
    assert namespace["_fill_preview_confirm_domain_ids_from_reached"](
        distances, graph["hidden"], graph["count"], 1.0, reached_ids.tolist(), 2.0
    ) is None


def test_session_snapshot_reuses_across_modes_and_radius_without_changing_domain():
    namespace = _load_snapshot_production_functions()
    graph, distances, reached_ids, state = _snapshot_fixture(namespace)
    builder = namespace["_fill_preview_confirm_graph_snapshot_for_session"]
    snapshot, domain = builder(
        state,
        graph,
        distances,
        1.0,
        reached_ids=reached_ids,
        reached_complete_through=2.0,
    )
    assert domain.tolist() == _legacy_confirm_domain(
        distances, graph["hidden"], 1.0
    ).tolist()
    assert namespace["has_mesh_global_identity_face_ids"](snapshot, graph["count"])
    assert all(
        not snapshot[name].flags.writeable
        for name in ("face_ids", "hidden", "offsets", "neighbors")
    )
    cached_face_ids = snapshot["face_ids"]
    smaller, smaller_domain = builder(
        state,
        graph,
        distances,
        0.5,
        reached_ids=reached_ids,
        reached_complete_through=2.0,
    )
    assert smaller["face_ids"] is cached_face_ids
    assert smaller_domain.tolist() == _legacy_confirm_domain(
        distances, graph["hidden"], 0.5
    ).tolist()
    assert state[namespace["_FILL_CONFIRM_GRAPH_CACHE_KEY"]]["mode"] == "normal"

    # Mode is part of the cache key; Ctrl+E and Shift+E own their own cache
    # entries without inheriting a previous strategy's snapshot identity.
    state["strict_mode"] = True
    strict, _ = builder(
        state,
        graph,
        distances,
        1.0,
        reached_ids=reached_ids,
        reached_complete_through=2.0,
    )
    assert strict["face_ids"] is not cached_face_ids
    assert state[namespace["_FILL_CONFIRM_GRAPH_CACHE_KEY"]]["mode"] == "strict"
    state["strict_mode"] = False
    state["expand_only"] = True
    expand, _ = builder(
        state,
        graph,
        distances,
        1.0,
        reached_ids=reached_ids,
        reached_complete_through=2.0,
    )
    assert expand["face_ids"] is not strict["face_ids"]
    assert state[namespace["_FILL_CONFIRM_GRAPH_CACHE_KEY"]]["mode"] == "expand-only"


def test_session_snapshot_invalidates_graph_seed_session_and_dirty_state():
    namespace = _load_snapshot_production_functions()
    graph, distances, reached_ids, state = _snapshot_fixture(namespace)
    builder = namespace["_fill_preview_confirm_graph_snapshot_for_session"]
    key = namespace["_FILL_CONFIRM_GRAPH_CACHE_KEY"]

    def build():
        return builder(
            state,
            state["adjacency"],
            distances,
            1.0,
            reached_ids=reached_ids,
            reached_complete_through=2.0,
        )[0]

    first = build()
    first_ids = first["face_ids"]
    state["seed_local"] = 1
    assert build()["face_ids"] is not first_ids
    seed_ids = state[key]["snapshot"]["face_ids"]
    state["session_id"] += 1
    assert build()["face_ids"] is not seed_ids
    session_ids = state[key]["snapshot"]["face_ids"]
    graph["geometry_refresh_count"] += 1
    assert build()["face_ids"] is not session_ids
    refreshed_ids = state[key]["snapshot"]["face_ids"]

    replacement = dict(graph)
    replacement["offsets"] = np.array(graph["offsets"], copy=True)
    state["adjacency"] = replacement
    assert build()["face_ids"] is not refreshed_ids
    state["adjacency"]["geometry_dirty"] = True
    dirty_snapshot, dirty_domain = build(), None
    assert key not in state
    assert dirty_snapshot["face_ids"].shape == (graph["count"],)


def run():
    tests = [
        test_legacy_oracle_contract,
        test_pair_row_localization_matches_legacy_oracle,
        test_shared_materializer_preserves_normal_and_strict_records,
        test_pair_row_tag_is_only_emitted_by_full_graph_builder_and_refresh,
        test_progressive_domain_matches_legacy_and_requires_complete_bound,
        test_session_snapshot_reuses_across_modes_and_radius_without_changing_domain,
        test_session_snapshot_invalidates_graph_seed_session_and_dirty_state,
    ]
    for test in tests:
        test()
    return {"passed": len(tests), "production_imported": False}


if __name__ == "__main__":
    print(run())
