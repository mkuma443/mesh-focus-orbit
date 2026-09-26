"""Focused pure fixtures for Smart Fill accepted-face row mapping.

These tests load invariants.py directly and do not import Blender or alter a scene.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import time

import numpy as np


def _load_invariants():
    path = (
        Path(__file__).resolve().parents[1]
        / "mesh_focus_orbit"
        / "smart_fill"
        / "invariants.py"
    )
    spec = spec_from_file_location("_mfo_smart_fill_identity_mapping_test", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _identity_result(module, count, faces):
    face_ids, provenance = module.make_mesh_global_identity_face_ids(count)
    confirm = {
        "face_ids": face_ids,
        "count": count,
        module._FACE_IDS_IDENTITY_PROVENANCE_KEY: provenance,
    }
    return {"faces": faces, "confirm_geometry": confirm}


def test_identity_fast_path_only_uniques_candidate_ids():
    module = _load_invariants()
    result = _identity_result(
        module,
        20_000,
        np.asarray([7, 2, 7, 0], dtype=np.int32),
    )
    original_unique = module.np.unique
    observed_sizes = []

    def counted_unique(values, *args, **kwargs):
        observed_sizes.append(np.asarray(values).size)
        return original_unique(values, *args, **kwargs)

    module.np.unique = counted_unique
    try:
        rows = module.accepted_graph_rows_from_result(result)
    finally:
        module.np.unique = original_unique

    assert rows.dtype == np.dtype(np.int32)
    assert rows.tolist() == [0, 2, 7]
    assert observed_sizes == [4], observed_sizes


def test_nonidentity_partial_and_legacy_graphs_use_safe_fallback():
    module = _load_invariants()
    cases = (
        (
            {"face_ids": np.asarray([40, 10, 90], dtype=np.int32), "count": 3},
            [90, 40, 90],
            [0, 2],
        ),
        (
            {"face_ids": np.asarray([12003, 1007, 4096], dtype=np.int32), "count": 3},
            [1007, 12003],
            [0, 1],
        ),
        (
            {"face_ids": [1007, 4096, 12003], "count": 3},
            [12003, 1007, 12003],
            [0, 2],
        ),
    )
    for confirm, faces, expected in cases:
        result = {"faces": faces, "confirm_geometry": confirm}
        assert module.accepted_graph_rows_from_result(result).tolist() == expected


def test_replaced_identity_array_invalidates_provenance_and_falls_back():
    module = _load_invariants()
    result = _identity_result(module, 3, [90, 10])
    confirm = result["confirm_geometry"]
    confirm["face_ids"] = np.asarray([90, 10, 70], dtype=np.int32)
    assert not module.has_mesh_global_identity_face_ids(confirm)
    assert module.accepted_graph_rows_from_result(result).tolist() == [0, 1]


def test_identity_and_fallback_paths_reject_out_of_range_candidates():
    module = _load_invariants()
    identity = _identity_result(module, 4, [-1, 2])
    try:
        module.accepted_graph_rows_from_result(identity)
    except ValueError:
        pass
    else:
        raise AssertionError("identity mapping accepted an out-of-range face id")

    nonidentity = {
        "faces": [500],
        "confirm_geometry": {
            "face_ids": np.asarray([100, 200, 300], dtype=np.int32),
            "count": 3,
        },
    }
    try:
        module.accepted_graph_rows_from_result(nonidentity)
    except ValueError:
        pass
    else:
        raise AssertionError("fallback mapping accepted an absent face id")


def bench_six_million_mapping():
    """Compare the former full-map operations with the candidate-only path."""
    module = _load_invariants()
    face_count = 6_040_072
    result = _identity_result(
        module,
        face_count,
        np.tile(np.arange(0, 400_000, 4, dtype=np.int32), 2),
    )
    face_ids = result["confirm_geometry"]["face_ids"]
    accepted_faces = np.asarray(result["faces"], dtype=np.int64).reshape(-1)

    started = time.perf_counter()
    legacy_ids = np.asarray(face_ids, dtype=np.int64).reshape(-1)
    if len(legacy_ids) == 0 or len(np.unique(legacy_ids)) != len(legacy_ids):
        raise AssertionError("legacy full-graph mapping validation failed")
    accepted_global = np.unique(accepted_faces)
    legacy_rows = module.map_global_faces_to_local(legacy_ids, accepted_global)
    legacy_seconds = time.perf_counter() - started

    samples = []
    for _ in range(5):
        started = time.perf_counter()
        rows = module.accepted_graph_rows_from_result(result)
        samples.append(time.perf_counter() - started)
    if not np.array_equal(rows, legacy_rows):
        raise AssertionError("candidate-only mapping changed row results")
    return {
        "mesh_faces": face_count,
        "candidate_ids": len(accepted_faces),
        "unique_candidate_ids": len(rows),
        "legacy_seconds": legacy_seconds,
        "fast_seconds_median": float(np.median(samples)),
        "speedup": legacy_seconds / max(float(np.median(samples)), 1.0e-12),
        "row_results_equal": True,
    }


def run():
    tests = [
        test_identity_fast_path_only_uniques_candidate_ids,
        test_nonidentity_partial_and_legacy_graphs_use_safe_fallback,
        test_replaced_identity_array_invalidates_provenance_and_falls_back,
        test_identity_and_fallback_paths_reject_out_of_range_candidates,
    ]
    for test in tests:
        test()
    return {"passed": len(tests), "blender_imported": False}


if __name__ == "__main__":
    print(run())
