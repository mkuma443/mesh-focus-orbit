"""Small Blender fixtures for the translucent Smart Fill geometry preview."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np

_PREVIEW_MODULE = None


def _load_preview_module():
    global _PREVIEW_MODULE
    if _PREVIEW_MODULE is not None:
        return _PREVIEW_MODULE
    root = pathlib.Path(__file__).resolve().parents[1]
    package_dir = root / "mesh_focus_orbit"
    package_name = "mfo_fill_preview_fixture"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    _PREVIEW_MODULE = sys.modules[f"{package_name}.smart_fill.preview"]
    return _PREVIEW_MODULE


def _fixture_graph():
    faces = (
        ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((4, 0, 0), (5, 0, 0), (5, 1, 0), (4, 1, 0)),
        ((8, 0, 0), (10, 0, 0), (10, 2, 0), (9, 1, 0), (8, 2, 0)),
        ((12, 0, 0), (13, 0, 0), (13, 1, 0), (12, 1, 0)),
        ((16, 0, 0), (18, 0, 0), (16.7, 0.5, 0), (16, 2, 0)),
    )
    flat = np.concatenate(
        [np.arange(sum(len(face) for face in faces[:i]),
                   sum(len(face) for face in faces[:i + 1]), dtype=np.int32)
         for i in range(len(faces))]
    )
    counts = np.asarray([len(face) for face in faces], dtype=np.int32)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    vertices = np.asarray(
        [point for face in faces for point in face], dtype=np.float64
    )
    return {
        "count": len(faces),
        "face_ids": np.asarray([501, 502, 900, 1200, 1300], dtype=np.int32),
        "face_vertex_flat": flat,
        "face_vertex_offsets": offsets,
        "face_vertex_counts": counts,
        "face_vertex_ids": tuple(
            np.arange(offsets[i], offsets[i + 1], dtype=np.int32)
            for i in range(len(faces))
        ),
        "world_vertices": vertices,
        "normals": np.tile((0.0, 0.0, 1.0), (len(faces), 1)),
        "hidden": np.asarray([False, False, False, True, False]),
        "signature": ("fixture", 17),
        "vertex_id_space": "mesh-global",
    }


def _triangle_area_sum(positions):
    triangles = np.asarray(positions, dtype=np.float64).reshape((-1, 3, 3))
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    return float(np.sum(np.linalg.norm(cross, axis=1) * 0.5))


def test_selected_faces_tessellate_and_hidden_faces_are_omitted():
    preview = _load_preview_module()
    graph = _fixture_graph()
    result = {
        "faces": np.asarray([501, 900, 1300, 1200], dtype=np.int32),
        "fill_draw_source_geometry": graph,
        # These are graph row IDs, not the non-identity mesh-global face IDs.
        "fill_draw_source_face_ids": np.asarray([0, 2, 4, 3], dtype=np.int32),
    }

    positions = preview._fill_preview_build_triangle_positions(result)
    triangles = positions.reshape((-1, 3, 3))

    assert len(triangles) == 6  # tri: 1, concave ngon: 3, concave quad: 2
    assert np.isclose(_triangle_area_sum(positions), 4.7, atol=1.0e-6)
    assert np.all(np.einsum(
        "ij,j->i",
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]),
        np.asarray((0.0, 0.0, 1.0)),
    ) > 0.0)
    assert not np.any((positions[:, 0] >= 4.0) & (positions[:, 0] <= 5.0))
    assert not np.any((positions[:, 0] >= 12.0) & (positions[:, 0] <= 13.0))


def test_compact_face_vertex_sequence_uses_selected_rows_only():
    preview = _load_preview_module()
    graph = _fixture_graph()
    graph.pop("face_vertex_flat")
    graph.pop("face_vertex_offsets")
    graph.pop("face_vertex_counts")
    result = {
        "faces": np.asarray([501, 900, 1300, 1200], dtype=np.int32),
        "fill_draw_source_geometry": graph,
        "fill_draw_source_face_ids": np.asarray([0, 2, 4, 3], dtype=np.int32),
    }

    positions = preview._fill_preview_build_triangle_positions(result)

    assert len(positions) == 18
    assert np.isclose(_triangle_area_sum(positions), 4.7, atol=1.0e-6)
    assert not np.any((positions[:, 0] >= 4.0) & (positions[:, 0] <= 5.0))
    assert not np.any((positions[:, 0] >= 12.0) & (positions[:, 0] <= 13.0))


def test_existing_face_vertex_sequence_offset_schema_is_accepted():
    preview = _load_preview_module()
    graph = _fixture_graph()
    graph.pop("face_vertex_ids")
    counts = graph.pop("face_vertex_counts")
    offsets = graph.pop("face_vertex_offsets")
    graph["face_vertex_offsets"] = offsets[:-1]
    graph["face_edge_counts"] = counts

    sequence = preview._fill_preview_face_vertex_sequence(graph)
    assert sequence is not None
    assert len(sequence) == graph["count"]
    result = {
        "faces": np.asarray([501, 900, 1300, 1200], dtype=np.int32),
        "fill_draw_source_geometry": graph,
        "fill_draw_source_face_ids": np.asarray([0, 2, 4, 3], dtype=np.int32),
    }

    positions = preview._fill_preview_build_triangle_positions(result)

    assert len(positions) == 18
    assert np.isclose(_triangle_area_sum(positions), 4.7, atol=1.0e-6)


def test_nonplanar_quad_falls_back_to_blender_tessellator_in_both_schemas():
    preview = _load_preview_module()
    points = np.asarray(
        ((0, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1)),
        dtype=np.float64,
    )
    graph = {
        "count": 1,
        "face_ids": np.asarray([77], dtype=np.int32),
        "face_vertex_flat": np.asarray([0, 1, 2, 3], dtype=np.int32),
        "face_vertex_offsets": np.asarray([0, 4], dtype=np.int64),
        "face_vertex_counts": np.asarray([4], dtype=np.int32),
        "face_vertex_ids": ((0, 1, 2, 3),),
        "world_vertices": points,
        # This approximate plane would project the quad as convex, but is not
        # trusted for the planarity decision.
        "normals": np.asarray(((0.0, 0.0, 1.0),), dtype=np.float64),
        "hidden": np.asarray([False]),
        "signature": ("nonplanar-quad", 1),
    }
    expected = preview._fill_preview_tessellate_draw_face(points)
    fixed_diagonal_fan = points[np.asarray(((0, 1, 2), (0, 2, 3)))]
    assert not np.array_equal(expected, fixed_diagonal_fan.reshape((-1, 3)))

    for schema in ("flat", "tuple"):
        source = dict(graph)
        if schema == "tuple":
            source.pop("face_vertex_flat")
            source.pop("face_vertex_offsets")
            source.pop("face_vertex_counts")
        result = {
            "faces": np.asarray([77], dtype=np.int32),
            "fill_draw_source_geometry": source,
            "fill_draw_source_face_ids": np.asarray([0], dtype=np.int32),
        }
        original_tessellator = preview._fill_preview_tessellate_draw_face
        with mock.patch.object(
            preview,
            "_fill_preview_tessellate_draw_face",
            wraps=original_tessellator,
        ) as tessellate:
            positions = preview._fill_preview_build_triangle_positions(result)

        assert tessellate.call_count == 1
        assert np.array_equal(positions, expected)


def test_quad_fast_path_guard_rejects_invalid_normals_and_nonfinite_points():
    preview = _load_preview_module()
    planar = np.asarray(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
        dtype=np.float64,
    )
    nonplanar = np.asarray(
        ((0, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1)),
        dtype=np.float64,
    )
    nonfinite = planar.copy()
    nonfinite[2, 1] = np.nan
    degenerate = np.asarray(
        ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)),
        dtype=np.float64,
    )

    mask = preview._fill_preview_planar_convex_quad_mask(
        np.stack((planar, nonplanar, nonfinite, degenerate))
    )

    assert np.array_equal(mask, [True, False, False, False])


def test_draw_batches_are_reused_for_same_result_generation():
    preview = _load_preview_module()
    graph = _fixture_graph()
    result = {
        "faces": np.asarray([501, 900, 1300], dtype=np.int32),
        "shape_segments": [],
        "distance_segments": [],
        "created_generation": 4,
        "fill_draw_source_generation": 4,
        "fill_draw_source_signature": graph["signature"],
        "fill_draw_source_geometry": graph,
        "fill_draw_source_face_ids": np.asarray([0, 2, 4], dtype=np.int32),
    }
    state = {"generation": 4, "signature": graph["signature"]}

    with mock.patch.object(preview, "_fill_preview_shader_get", return_value=object()):
        with mock.patch.object(preview, "batch_for_shader", return_value=object()) as build:
            assert preview._fill_preview_build_draw_batches(state, result)
            first_cpu_seconds = result["fill_triangle_build_seconds"]
            assert preview._fill_preview_build_draw_batches(state, result)

    assert build.call_count == 1
    assert result["fill_gpu_batch_build_count"] == 1
    assert result["fill_triangle_vertex_count"] == 18
    assert result["fill_triangle_build_seconds"] == first_cpu_seconds


def test_changed_selected_rows_rebuild_the_fill_batch():
    preview = _load_preview_module()
    graph = _fixture_graph()
    result = {
        "faces": np.asarray([501], dtype=np.int32),
        "shape_segments": [],
        "distance_segments": [],
        "created_generation": 4,
        "fill_draw_source_generation": 4,
        "fill_draw_source_signature": graph["signature"],
        "fill_draw_source_geometry": graph,
        "fill_draw_source_face_ids": np.asarray([0], dtype=np.int32),
    }
    state = {"generation": 4, "signature": graph["signature"]}

    with mock.patch.object(preview, "_fill_preview_shader_get", return_value=object()):
        with mock.patch.object(preview, "batch_for_shader", return_value=object()) as build:
            assert preview._fill_preview_build_draw_batches(state, result)
            assert result["fill_triangle_vertex_count"] == 3
            result["fill_draw_source_face_ids"] = np.asarray([0, 2], dtype=np.int32)
            result["faces"] = np.asarray([501, 900], dtype=np.int32)
            assert preview._fill_preview_build_draw_batches(state, result)

    assert build.call_count == 2
    assert result["fill_gpu_batch_build_count"] == 2
    assert result["fill_triangle_vertex_count"] == 12


def test_signature_mismatch_does_not_build_or_draw_batches():
    preview = _load_preview_module()
    graph = _fixture_graph()
    result = {
        "faces": np.asarray([501], dtype=np.int32),
        "shape_segments": [],
        "distance_segments": [],
        "created_generation": 4,
        "fill_draw_source_generation": 4,
        "fill_draw_source_signature": ("fixture", 16),
        "fill_draw_source_geometry": graph,
        "fill_draw_source_face_ids": np.asarray([0], dtype=np.int32),
    }

    assert not preview._fill_preview_build_draw_batches(
        {"generation": 4, "signature": graph["signature"]}, result
    )
    assert result.get("gpu_batches") is None


def test_green_boundary_uses_matching_translucent_fill_color():
    preview = _load_preview_module()

    assert preview._fill_preview_fill_color({"boundary_all_orange": True}) == (
        0.16, 1.0, 0.34, 0.20
    )
    assert preview._fill_preview_fill_color({"boundary_all_orange": False}) == (
        0.16, 0.72, 0.96, 0.20
    )


def test_green_terminal_wheel_up_preserves_current_preview_exactly():
    preview = _load_preview_module()
    result = {
        "boundary_all_orange": True,
        "gpu_batches": {},
    }
    batch = object()
    result["gpu_batches"]["triangles"] = batch
    draw_cache = {"triangles": object()}
    state = {
        "phase": "ready",
        "pending": False,
        "generation": 23,
        "drawn_generation": 23,
        "result": result,
        "expand_only": True,
        "strict_mode": False,
        "initial_radius": 2.0,
        "desired_radius": 32.0,
        "wheel_armed": True,
        "wheel_gate": False,
        "wheel_drain_until": 0.0,
        "draw_batches": draw_cache,
    }
    before = dict(state)

    assert preview._fill_preview_green_wheel_up_is_noop(
        state, SimpleNamespace(type="WHEELUPMOUSE")
    )

    with mock.patch.object(
        preview, "_fill_preview_release_state_draw_caches"
    ) as release_caches:
        with mock.patch.object(preview, "_fill_preview_tag_redraw") as redraw:
            assert preview._fill_preview_accept_normal_wheel(
                state, SimpleNamespace(type="WHEELUPMOUSE")
            )

    assert state == before
    assert state["result"] is result
    assert state["draw_batches"] is draw_cache
    assert result["gpu_batches"]["triangles"] is batch
    release_caches.assert_not_called()
    redraw.assert_not_called()


def test_green_terminal_wheel_down_still_uses_normal_step():
    preview = _load_preview_module()
    state = {
        "phase": "ready",
        "pending": False,
        "generation": 23,
        "drawn_generation": 23,
        "result": {"boundary_all_orange": True},
        "initial_radius": 2.0,
        "desired_radius": 8.0,
        "wheel_armed": True,
        "wheel_gate": False,
        "wheel_drain_until": 0.0,
    }

    with mock.patch.object(preview, "_fill_preview_release_state_draw_caches"):
        with mock.patch.object(preview, "_fill_preview_tag_redraw"):
            assert preview._fill_preview_accept_normal_wheel(
                state, SimpleNamespace(type="WHEELDOWNMOUSE")
            )

    assert state["desired_radius"] == 8.0 / 1.25
    assert state["phase"] == "compute"
    assert state["pending"] is True
    assert state["result"] is None
    assert state["generation"] == 23
    assert state["drawn_generation"] == 23


def test_green_wheel_up_predicate_covers_strict_and_expand_only_modes():
    preview = _load_preview_module()
    for strict_mode, expand_only in ((False, False), (True, False), (False, True)):
        state = {
            "phase": "ready",
            "pending": False,
            "generation": 8,
            "drawn_generation": 8,
            "result": {"boundary_all_orange": True},
            "strict_mode": strict_mode,
            "expand_only": expand_only,
        }
        assert preview._fill_preview_green_wheel_up_is_noop(
            state, SimpleNamespace(type="WHEELUPMOUSE")
        )
        assert not preview._fill_preview_green_wheel_up_is_noop(
            state, SimpleNamespace(type="WHEELDOWNMOUSE")
        )


def test_full_terminal_frontier_result_reaches_draw_with_current_source_rows():
    preview = _load_preview_module()
    graph = _fixture_graph()
    graph.update({
        "offsets": np.asarray([0, 1, 2, 2, 2, 2], dtype=np.int64),
        "neighbors": np.asarray([1, 0], dtype=np.int32),
        "neighbor_lengths": np.asarray([1.0, 1.0], dtype=np.float64),
        "first": np.asarray([1], dtype=np.int32),
        "second": np.asarray([2], dtype=np.int32),
        "pair_v0": np.asarray([3], dtype=np.int32),
        "pair_v1": np.asarray([4], dtype=np.int32),
    })
    signature = graph["signature"]
    stale_base_result = {
        "faces": np.asarray([501], dtype=np.int32),
        "created_generation": 6,
        "fill_draw_source_generation": 6,
        "fill_draw_source_signature": signature,
        "fill_draw_source_geometry": graph,
        "fill_draw_source_face_ids": np.asarray([0], dtype=np.int32),
        "shape_segments": [],
        "distance_segments": [],
    }
    state = {
        "active": True,
        "phase": "ready",
        "area_key": 123,
        "drawn_generation": 6,
        "signature": signature,
        "generation": 7,
        "seed_face": 501,
        "strict_mode": False,
        "terminal_base_signature": signature,
        "terminal_base_radius": 1.0,
        "terminal_base": {
            "signature": signature,
            "face_ids": graph["face_ids"],
            "result": stale_base_result,
        },
        "terminal_frontier": {
            "geometry": graph,
            "distances": np.asarray([0.0, 1.1, np.inf, np.inf, np.inf]),
            "heap": [(1.1, 1)],
            "selected": {0},
            "barriers": frozenset(),
            "last_radius": 1.0,
            "popped": 0,
        },
    }

    with mock.patch.object(
        preview, "_fill_preview_terminal_coverage_certified", return_value=True
    ):
        with mock.patch.object(
            preview,
            "_fill_preview_confirm_graph_snapshot",
            return_value=(graph, np.asarray([0, 1], dtype=np.int32)),
        ):
            result = preview._fill_preview_terminal_result(state, 1.5)

    assert result is not None
    assert np.array_equal(result["fill_draw_source_face_ids"], [0, 1]), (
        result["fill_draw_source_face_ids"].tolist(), result["faces"].tolist()
    )
    assert result["fill_draw_source_geometry"] is graph
    assert result["created_generation"] == 7
    assert result["fill_draw_source_generation"] == 7
    assert result["fill_draw_source_signature"] == signature
    # Exercise the production color selection on the actual draw callback.
    result["boundary_all_orange"] = True
    result["shape_segments"] = [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
    state["result"] = result
    area = mock.Mock()
    area.as_pointer.return_value = 123
    shader = mock.Mock()
    gpu_state = mock.Mock()
    runtime = preview._runtime
    previous_state = runtime.fill_preview_state
    previous_shader = runtime.fill_preview_shader
    runtime.fill_preview_state = state
    runtime.fill_preview_shader = shader
    try:
        with mock.patch.object(
            preview, "bpy", SimpleNamespace(context=SimpleNamespace(area=area))
        ):
            with mock.patch.object(
                preview, "gpu", SimpleNamespace(state=gpu_state)
            ):
                with mock.patch.object(
                    preview, "_fill_preview_shader_get", return_value=shader
                ):
                    with mock.patch.object(
                        preview, "batch_for_shader", return_value=mock.Mock()
                    ):
                        preview._fill_preview_draw()
    finally:
        runtime.fill_preview_state = previous_state
        runtime.fill_preview_shader = previous_shader

    assert state["drawn_generation"] == 7
    assert result["fill_triangle_vertex_count"] == 9
    assert "triangles" in result["gpu_batches"]
    drawn_colors = [
        args[0][1]
        for args in shader.uniform_float.call_args_list
        if args and args[0] and args[0][0] == "color"
    ]
    assert (0.16, 1.0, 0.34, 0.20) in drawn_colors
    assert (0.16, 1.0, 0.34, 0.95) in drawn_colors


def test_cached_radius_revisit_draws_and_releases_current_generation_gates():
    preview = _load_preview_module()
    package_name = preview.__package__.split(".")[0]
    registration = sys.modules[f"{package_name}.registration"]
    graph = _fixture_graph()
    signature = graph["signature"]

    class _Area:
        def as_pointer(self):
            return 123

    area = _Area()
    runtime = preview._runtime
    previous_state = runtime.fill_preview_state
    previous_shader = runtime.fill_preview_shader
    shader = mock.Mock()
    batch = mock.Mock()
    proxy = SimpleNamespace(
        report=lambda *_args, **_kwargs: None,
    )
    modes = (
        ("normal", False, False, ("progressive-range", 1.25)),
        ("shift", False, True, ("expand-only", 1.25)),
        ("ctrl", True, False, ("geometry-strict", 1.25)),
    )

    try:
        with mock.patch.object(registration, "_fill_preview_valid", return_value=True):
            with mock.patch.object(registration, "_fill_preview_tag_redraw"):
                with mock.patch.object(
                    registration,
                    "_fill_preview_make_result",
                    side_effect=AssertionError("a valid radius hit must not recompute"),
                ):
                    with mock.patch.object(
                        registration, "_fill_preview_write_vertex_paint",
                        return_value=((1, 1, "POINT"), None),
                    ):
                        with mock.patch.object(registration, "_fill_preview_cancel"):
                            for _name, strict_mode, expand_only, cache_key in modes:
                                cached = {
                                    "radius": 1.25,
                                    "faces": np.asarray([501], dtype=np.int32),
                                    "candidate_count": 1,
                                    "shape_segments": [],
                                    "distance_segments": [],
                                    "created_generation": 1,
                                    "fill_draw_source_generation": 1,
                                    "fill_draw_source_signature": signature,
                                    "fill_draw_source_geometry": graph,
                                    "fill_draw_source_face_ids": np.asarray(
                                        [0], dtype=np.int32
                                    ),
                                    "confirm_geometry": graph,
                                    "confirm_seed_local": 0,
                                    "confirm_domain_ids": np.asarray(
                                        [0], dtype=np.int32
                                    ),
                                    "confirm_signature": signature,
                                    "compute_seconds": 0.001,
                                }
                                state = {
                                    "active": True,
                                    "phase": "compute",
                                    "pending": True,
                                    "generation": 2,
                                    "drawn_generation": 1,
                                    "signature": signature,
                                    "strict_mode": strict_mode,
                                    "expand_only": expand_only,
                                    "initial_radius": 1.0,
                                    "desired_radius": 1.25,
                                    "processed_radius": 1.0,
                                    "accepted_visible_ids": np.empty(0, dtype=np.int32),
                                    "seed_face": 501,
                                    "results": {cache_key: cached},
                                    "result": {"radius": 1.0, "created_generation": 2},
                                    "backend": "PAINT_VERTEX",
                                    "obj": SimpleNamespace(),
                                    "metrics": {"make_result_seconds": 0.0},
                                    "last_tick_seconds": 0.0,
                                    "max_tick_seconds": 0.0,
                                    "area_key": 123,
                                    "area": area,
                                }

                                assert registration.VIEW3D_OT_mesh_focus_local_face_set_grow._process_timer(
                                    proxy, SimpleNamespace(), state
                                )
                                result = state["result"]
                                assert result is not cached
                                assert result["created_generation"] == 3
                                assert result["fill_draw_source_generation"] == 1
                                assert result["fill_draw_source_geometry"] is graph
                                assert result["fill_draw_source_face_ids"] is cached[
                                    "fill_draw_source_face_ids"
                                ]
                                assert result["progressive_range_cache_hit"] is True
                                assert state["phase"] == "ready"
                                assert state["pending"] is False

                                if not strict_mode:
                                    assert registration.VIEW3D_OT_mesh_focus_local_face_set_grow._finish_confirm(
                                        proxy, SimpleNamespace(), state
                                    ) == {"RUNNING_MODAL"}

                                runtime.fill_preview_state = state
                                runtime.fill_preview_shader = shader
                                batch.reset_mock()
                                with mock.patch.object(
                                    preview,
                                    "bpy",
                                    SimpleNamespace(
                                        context=SimpleNamespace(area=area)
                                    ),
                                ):
                                    with mock.patch.object(
                                        preview,
                                        "gpu",
                                        SimpleNamespace(state=mock.Mock()),
                                    ):
                                        with mock.patch.object(
                                            preview,
                                            "_fill_preview_shader_get",
                                            return_value=shader,
                                        ):
                                            with mock.patch.object(
                                                preview,
                                                "batch_for_shader",
                                                return_value=batch,
                                            ) as build_batch:
                                                preview._fill_preview_draw()

                                assert state["drawn_generation"] == 3
                                assert build_batch.call_count == 1
                                assert batch.draw.call_count == 1
                                assert registration.VIEW3D_OT_mesh_focus_local_face_set_grow._finish_confirm(
                                    proxy, SimpleNamespace(), state
                                ) == {"FINISHED"}

                                wheel_state = dict(state)
                                wheel_state.update({
                                    "wheel_armed": True,
                                    "wheel_gate": False,
                                    "wheel_drain_until": 0.0,
                                    "pending": False,
                                    "phase": "ready",
                                })
                                with mock.patch.object(
                                    preview, "_fill_preview_release_state_draw_caches"
                                ):
                                    with mock.patch.object(
                                        preview, "_fill_preview_tag_redraw"
                                    ):
                                        assert preview._fill_preview_accept_normal_wheel(
                                            wheel_state,
                                            SimpleNamespace(type="WHEELUPMOUSE"),
                                        )
                                assert wheel_state["phase"] == "compute"
                                assert wheel_state["pending"] is True
                                assert wheel_state["desired_radius"] == 1.25 * 1.25
    finally:
        runtime.fill_preview_state = previous_state
        runtime.fill_preview_shader = previous_shader


def test_stale_generation_does_not_build_or_draw_batches():
    preview = _load_preview_module()
    graph = _fixture_graph()
    result = {
        "faces": np.asarray([501], dtype=np.int32),
        "shape_segments": [],
        "distance_segments": [],
        "created_generation": 3,
        "fill_draw_source_generation": 3,
        "fill_draw_source_signature": graph["signature"],
        "fill_draw_source_geometry": graph,
        "fill_draw_source_face_ids": np.asarray([0], dtype=np.int32),
    }

    assert not preview._fill_preview_build_draw_batches(
        {"generation": 4, "signature": graph["signature"]}, result
    )
    assert result.get("gpu_batches") is None


def run():
    test_selected_faces_tessellate_and_hidden_faces_are_omitted()
    test_compact_face_vertex_sequence_uses_selected_rows_only()
    test_existing_face_vertex_sequence_offset_schema_is_accepted()
    test_nonplanar_quad_falls_back_to_blender_tessellator_in_both_schemas()
    test_quad_fast_path_guard_rejects_invalid_normals_and_nonfinite_points()
    test_draw_batches_are_reused_for_same_result_generation()
    test_changed_selected_rows_rebuild_the_fill_batch()
    test_signature_mismatch_does_not_build_or_draw_batches()
    test_green_boundary_uses_matching_translucent_fill_color()
    test_green_terminal_wheel_up_preserves_current_preview_exactly()
    test_green_terminal_wheel_down_still_uses_normal_step()
    test_green_wheel_up_predicate_covers_strict_and_expand_only_modes()
    test_full_terminal_frontier_result_reaches_draw_with_current_source_rows()
    test_cached_radius_revisit_draws_and_releases_current_generation_gates()
    test_stale_generation_does_not_build_or_draw_batches()
    return {"passed": True, "fixtures": 15}


if __name__ == "__main__":
    print(run())
