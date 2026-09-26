"""Blender-free equivalence checks for the Smart Fill region reuse path."""

import ast
from collections import deque
import math
from pathlib import Path
import statistics
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "mesh_focus_orbit" / "smart_fill" / "preview.py"
GEOMETRY = ROOT / "mesh_focus_orbit" / "smart_fill" / "geometry.py"


def _function(tree, name):
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_region_helpers():
    preview_source = PREVIEW.read_text(encoding="utf-8")
    geometry_source = GEOMETRY.read_text(encoding="utf-8")
    preview_tree = ast.parse(preview_source, filename=str(PREVIEW))
    geometry_tree = ast.parse(geometry_source, filename=str(GEOMETRY))
    namespace = {"np": np, "math": math, "deque": deque}
    nodes = (
        _function(preview_tree, "_fill_partition"),
        _function(geometry_tree, "_fill_preview_region"),
        _function(preview_tree, "_fill_preview_compute_geometry_regions"),
    )
    exec(
        compile(ast.Module(body=list(nodes), type_ignores=[]), str(PREVIEW), "exec"),
        namespace,
    )
    return namespace


def _ring_local(count, hidden_face=None, face_set_schema="absent"):
    face = np.arange(count, dtype=np.int32)
    neighbor = np.empty(count * 2, dtype=np.int32)
    neighbor[0::2] = (face - 1) % count
    neighbor[1::2] = (face + 1) % count
    hidden = np.zeros(count, dtype=bool)
    if hidden_face is not None:
        hidden[int(hidden_face)] = True
    local = {
        "count": int(count),
        "first": face.copy(),
        "second": np.roll(face, -1),
        "offsets": np.arange(count + 1, dtype=np.int64) * 2,
        "neighbors": neighbor,
        "neighbor_lengths": np.ones(count * 2, dtype=np.float64),
        "centers": np.zeros((count, 3), dtype=np.float64),
        "normals": np.tile(np.asarray([[0.0, 0.0, 1.0]]), (count, 1)),
        "hidden": hidden,
        "scale": np.ones(count, dtype=np.float64),
        "contrast": np.zeros(count, dtype=np.float64),
        "concavity": np.zeros(count, dtype=np.float64),
        "directional_valley": np.zeros(count, dtype=np.float64),
        "crease": np.zeros(count, dtype=np.float64),
        "contour_cost": np.ones(count * 2, dtype=np.float64),
        "face_edge_counts": np.full(count, 2, dtype=np.int32),
        "source_hard": np.zeros(count, dtype=bool),
        "partitions": {},
    }
    if face_set_schema == "valid":
        labels = np.where(face < 6, 4, 5).astype(np.int32)
        local["face_set_values"] = labels
        local["face_set_seed_id"] = 4
        local["face_set_prior_enabled"] = True
    elif face_set_schema == "missing-values":
        local["face_set_seed_id"] = 4
        local["face_set_prior_enabled"] = True
    elif face_set_schema == "wrong-length":
        local["face_set_values"] = np.asarray([4, 4], dtype=np.int32)
        local["face_set_seed_id"] = 4
        local["face_set_prior_enabled"] = True
    elif face_set_schema == "disabled-with-keys":
        local["face_set_values"] = np.full(count, 4, dtype=np.int32)
        local["face_set_seed_id"] = 4
        local["face_set_prior_enabled"] = False
    elif face_set_schema != "absent":
        raise ValueError(face_set_schema)
    return local


def _geometry_local(local):
    geometry_local = dict(local)
    for key in ("face_set_values", "face_set_seed_id", "face_set_prior_enabled"):
        geometry_local.pop(key, None)
    geometry_local["partitions"] = {}
    return geometry_local


def _legacy_region_stage(namespace, geometry_local, local, seed, strict):
    geometry_region, geometry_partition = namespace["_fill_preview_region"](
        geometry_local, seed, strict
    )
    assisted_region, assisted_partition = namespace["_fill_preview_region"](
        local, seed, strict
    )
    local_region = np.unique(
        np.r_[geometry_region, assisted_region]
    ).astype(np.int32, copy=False)
    return (
        geometry_region,
        geometry_partition,
        assisted_region,
        assisted_partition,
        local_region,
    )


def _assert_equal(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
        assert left.dtype == right.dtype
        assert np.array_equal(left, right)
    elif isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_equal(left_item, right_item)
    else:
        assert left == right


def _downstream_projection(stage):
    """Exercise selection, hole additions, physical boundary and Face Set metrics."""
    geometry_region, geometry_partition, assisted_region, assisted_partition, rows = stage
    count = 18
    distances = np.arange(count, dtype=np.float64)
    hidden = np.zeros(count, dtype=bool)
    hidden[8] = True
    base = rows[
        (distances[rows] <= 5.0) & ~hidden[rows]
    ].astype(np.int32, copy=False)
    base = np.unique(base).astype(np.int32, copy=False)
    # Stand-ins for independent hole/band resolver results. Their input is the
    # base candidate, which must be identical for both region paths.
    enclosed = (
        np.asarray([6], dtype=np.int32)
        if {0, 1, 2}.issubset(set(base))
        else np.empty(0, dtype=np.int32)
    )
    sandwiched = (
        np.asarray([7], dtype=np.int32)
        if {3, 4, 5}.issubset(set(base))
        else np.empty(0, dtype=np.int32)
    )
    preview = np.unique(np.concatenate((base, enclosed, sandwiched))).astype(np.int32)
    first = np.arange(count, dtype=np.int32)
    second = np.roll(first, -1)
    selected = np.zeros(count, dtype=bool)
    selected[preview] = True
    boundary = selected[first] != selected[second]
    shape_edge_ids = {5, 7}
    records = tuple(
        (
            int(edge),
            int(first[edge]),
            int(second[edge]),
            int(edge in shape_edge_ids),
        )
        for edge in np.flatnonzero(boundary)
    )
    bonus_count = len(
        np.setdiff1d(assisted_region, geometry_region, assume_unique=False)
    )
    face_set_metrics = {
        "relaxed_pairs": int(assisted_partition.get("face_set_relaxed_pair_count", 0)),
        "same_set_reached": int(assisted_partition.get("face_set_reached_same_set_faces", 0)),
        "geometry_baseline": int(len(geometry_region)),
        "same_id_bonus": int(bonus_count),
        "different_id_baseline_preserved": int(len(geometry_region)),
    }
    return {
        "selected": preview,
        "base_candidate": base,
        "enclosed_holes": enclosed,
        "sandwiched_bands": sandwiched,
        "boundary_records": records,
        "shape_boundary_count": sum(record[3] for record in records),
        "distance_boundary_count": sum(not record[3] for record in records),
        "face_set_metrics": face_set_metrics,
        "geometry_partition": geometry_partition,
    }


def _run_case(namespace, schema, strict=False):
    local = _ring_local(18, hidden_face=8, face_set_schema=schema)
    legacy = _legacy_region_stage(namespace, _geometry_local(local), local, 0, strict)
    optimized = namespace["_fill_preview_compute_geometry_regions"](
        _geometry_local(local),
        local,
        0,
        strict,
        bool(local.get("face_set_prior_enabled", False)),
    )
    _assert_equal(legacy, optimized)
    _assert_equal(_downstream_projection(legacy), _downstream_projection(optimized))
    return legacy, optimized


def test_no_prior_reuses_one_region_and_preserves_all_downstream_outputs():
    namespace = _load_region_helpers()
    calls = []
    original = namespace["_fill_preview_region"]

    def counted(local, seed, strict):
        calls.append(local)
        return original(local, seed, strict)

    namespace["_fill_preview_region"] = counted
    local = _ring_local(18, hidden_face=8)
    legacy = _legacy_region_stage(namespace, _geometry_local(local), local, 0, False)
    assert len(calls) == 2
    calls.clear()
    optimized = namespace["_fill_preview_compute_geometry_regions"](
        _geometry_local(local), local, 0, False, False
    )
    assert len(calls) == 1
    _assert_equal(legacy, optimized)
    assert optimized[0] is optimized[2]
    assert optimized[1] is optimized[3]
    assert optimized[0] is optimized[4]
    _assert_equal(_downstream_projection(legacy), _downstream_projection(optimized))


def test_face_set_and_legacy_schema_paths_keep_two_call_semantics():
    namespace = _load_region_helpers()
    for schema in ("valid", "missing-values", "wrong-length"):
        local = _ring_local(18, hidden_face=8, face_set_schema=schema)
        calls = []
        original = namespace["_fill_preview_region"]

        def counted(local_arg, seed, strict):
            calls.append(local_arg)
            return original(local_arg, seed, strict)

        namespace["_fill_preview_region"] = counted
        legacy = _legacy_region_stage(
            namespace, _geometry_local(local), local, 0, False
        )
        assert len(calls) == 2
        calls.clear()
        optimized = namespace["_fill_preview_compute_geometry_regions"](
            _geometry_local(local), local, 0, False, True
        )
        assert len(calls) == 2
        _assert_equal(legacy, optimized)
        _assert_equal(_downstream_projection(legacy), _downstream_projection(optimized))
        namespace["_fill_preview_region"] = original


def test_missing_or_disabled_prior_key_uses_geometry_only_result():
    namespace = _load_region_helpers()
    for schema in ("absent", "disabled-with-keys"):
        legacy, optimized = _run_case(namespace, schema, strict=True)
        _assert_equal(legacy, optimized)
        assert optimized[0] is optimized[2]
        assert optimized[1] is optimized[3]


def test_shared_partition_is_only_read_after_reuse():
    source = PREVIEW.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PREVIEW))
    make_result = _function(tree, "_fill_preview_make_result")
    helper_call = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_fill_preview_compute_geometry_regions"
        for node in ast.walk(make_result)
    )
    assert helper_call
    for node in ast.walk(make_result):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            assert not (
                isinstance(node.value, ast.Name)
                and node.value.id in {"assisted_partition", "geometry_partition"}
            )
    shared_partition_writes = [
        node
        for node in ast.walk(make_result)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "assisted_partition"
        and node.func.attr in {"update", "pop", "setdefault", "clear"}
    ]
    assert not shared_partition_writes


def _legacy_region_stage_for_bench(namespace, geometry_local, local, seed, strict):
    first = namespace["_fill_preview_region"](geometry_local, seed, strict)
    second = namespace["_fill_preview_region"](local, seed, strict)
    return (
        first[0],
        first[1],
        second[0],
        second[1],
        np.unique(np.r_[first[0], second[0]]).astype(np.int32, copy=False),
    )


def bench_190k_region_reuse():
    """Isolated actual-region benchmark; not a Blender UI measurement."""
    namespace = _load_region_helpers()
    count = 190_609
    local = _ring_local(count)
    samples = {"legacy_two_calls": [], "optimized_one_call": []}
    for _ in range(3):
        geometry_local = _geometry_local(local)
        local["partitions"] = {}
        started = time.perf_counter()
        old = _legacy_region_stage_for_bench(
            namespace, geometry_local, local, 0, False
        )
        samples["legacy_two_calls"].append(time.perf_counter() - started)
        assert len(old[0]) == count
        geometry_local = _geometry_local(local)
        local["partitions"] = {}
        started = time.perf_counter()
        new = namespace["_fill_preview_compute_geometry_regions"](
            geometry_local, local, 0, False, False
        )
        samples["optimized_one_call"].append(time.perf_counter() - started)
        _assert_equal(old, new)
    old_median = statistics.median(samples["legacy_two_calls"])
    new_median = statistics.median(samples["optimized_one_call"])
    return {
        "benchmark": "isolated-actual-region-ring",
        "local_faces": count,
        "samples_seconds": samples,
        "legacy_median_seconds": old_median,
        "optimized_median_seconds": new_median,
        "speedup": old_median / max(new_median, 1.0e-12),
        "results_equal": True,
        "blender_ui_measurement": False,
    }


def run():
    tests = (
        test_no_prior_reuses_one_region_and_preserves_all_downstream_outputs,
        test_face_set_and_legacy_schema_paths_keep_two_call_semantics,
        test_missing_or_disabled_prior_key_uses_geometry_only_result,
        test_shared_partition_is_only_read_after_reuse,
    )
    for test in tests:
        test()
    return {"passed": len(tests), "blender_imported": False}


if __name__ == "__main__":
    print(run())
    print(bench_190k_region_reuse())
