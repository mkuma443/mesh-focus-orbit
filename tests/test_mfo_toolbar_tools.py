"""Isolated Blender API checks for the resident MFO WorkSpaceTools.

Run inside Blender with::

    exec(compile(open(path).read(), path, "exec"), {"__name__": "__main__"})

The test does not select a tool, edit scene data, or save the current file.
"""

from __future__ import annotations

import importlib
import importlib.util
import ast
import copy
import inspect
import math
import pathlib
import sys
import time
from types import SimpleNamespace


def _package_reload_contract(inject_failure=False):
    """Reload the checkout package while restoring the live addon in finally."""
    import bpy

    root = pathlib.Path(__file__).resolve().parents[1]
    source_root = str(root)
    package_dir = root / "mesh_focus_orbit"
    path = package_dir / "__init__.py"
    package_name = "mesh_focus_orbit"
    handler_lists = (
        bpy.app.handlers.load_pre,
        bpy.app.handlers.load_post,
        bpy.app.handlers.undo_post,
        bpy.app.handlers.redo_post,
        bpy.app.handlers.depsgraph_update_post,
    )
    original_handlers = tuple(tuple(values) for values in handler_lists)
    original_sys_path = list(sys.path)
    original_modules = {
        name: sys.modules[name]
        for name in tuple(sys.modules)
        if name == package_name or name.startswith(package_name + ".")
    }
    original_module = original_modules.get(package_name)
    original_registered = bool(
        original_module is not None
        and getattr(getattr(original_module, "runtime", None), "is_registered", False)
    )
    original_timer_callbacks = tuple(
        getattr(getattr(original_module, "runtime", None), name, None)
        for name in ("deferred_orphan_cleanup", "deferred_undo_orphan_cleanup")
    )
    prefs_before = None
    prefs_schema = {}
    addon_entry = bpy.context.preferences.addons.get(package_name)
    if addon_entry is not None:
        prefs = addon_entry.preferences
        for prop in prefs.bl_rna.properties:
            if prop.identifier == "rna_type":
                continue
            try:
                value = getattr(prefs, prop.identifier)
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                continue
            prefs_schema[prop.identifier] = (prop.type, int(getattr(prop, "array_length", 0) or 0))
            try:
                value = copy.deepcopy(value)
            except (TypeError, ValueError):
                pass
            prefs_before = prefs_before or {}
            prefs_before[prop.identifier] = value
    module = None
    try:
        if original_registered:
            original_module.unregister()
        timer_unregistered_before_import = tuple(
            not callback or not bpy.app.timers.is_registered(callback)
            for callback in original_timer_callbacks
        )
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
        sys.path[:] = [entry for entry in sys.path if str(entry) != source_root]
        sys.path.insert(0, source_root)
        module = importlib.import_module(package_name)
        assert str(pathlib.Path(module.__file__).resolve()).startswith(str(package_dir.resolve()))
        baseline_ids = tuple(tuple(id(callback) for callback in values) for values in handler_lists)
        module.register()
        registered_once = bool(module.runtime.is_registered)
        timers_after_register = tuple(
            bool(callback and bpy.app.timers.is_registered(callback))
            for callback in (
                module.runtime.deferred_orphan_cleanup,
                module.runtime.deferred_undo_orphan_cleanup,
            )
        )
        if inject_failure:
            raise RuntimeError("injected reload cleanup failure")
        counts_once = tuple(
            sum(
                1
                for callback in values
                if getattr(callback, "__module__", "").startswith(package_name)
            )
            for values in handler_lists
        )
        module = importlib.reload(module)
        reload_unregistered = not bool(module.runtime.is_registered)
        timers_after_reload = tuple(
            bool(callback and bpy.app.timers.is_registered(callback))
            for callback in (
                module.runtime.deferred_orphan_cleanup,
                module.runtime.deferred_undo_orphan_cleanup,
            )
        )
        counts_after_reload = tuple(
            sum(
                1
                for callback in values
                if getattr(callback, "__module__", "").startswith(package_name)
            )
            for values in handler_lists
        )
        module.register()
        module = importlib.reload(module)
        module.register()
        counts_second_cycle = tuple(
            sum(
                1
                for callback in values
                if getattr(callback, "__module__", "").startswith(package_name)
            )
            for values in handler_lists
        )
        origins = sorted(
            {getattr(cls, "__module__", "") for cls in module.registration.CLASSES}
        )
        module.unregister()
        after_baseline_ids = tuple(
            tuple(id(callback) for callback in values) for values in handler_lists
        )
        assert registered_once and reload_unregistered
        assert counts_once == counts_second_cycle
        assert after_baseline_ids == baseline_ids
        assert all(timer_unregistered_before_import)
        assert not any(timers_after_reload)
        return {
            "passed": True,
            "registered_once": registered_once,
            "reload_unregistered": reload_unregistered,
            "counts_once": counts_once,
            "counts_after_reload": counts_after_reload,
            "counts_second_cycle": counts_second_cycle,
            "handler_identity_restored": True,
            "origins": origins,
            "production_module_restored": original_module is not None,
            "prefs_snapshot_scope": "all compatible RNA properties before unregister",
            "prefs_schema": prefs_schema,
            "checkout_path_index0": sys.path[0] == source_root,
            "timer_identity_cleanup": {
                "before_import": timer_unregistered_before_import,
                "after_register": timers_after_register,
                "after_reload": timers_after_reload,
                "old_callbacks_unregistered": all(timer_unregistered_before_import),
                "reload_has_no_stale_callbacks": not any(timers_after_reload),
            },
        }
    finally:
        try:
            if module is not None and bool(
                getattr(getattr(module, "runtime", None), "is_registered", False)
            ):
                module.unregister()
        finally:
            for target, original in zip(handler_lists, original_handlers):
                target[:] = original
            sys.path[:] = original_sys_path
            for name in tuple(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    sys.modules.pop(name, None)
            sys.modules.update(original_modules)
            if original_registered and original_module is not None:
                original_module.register()
            if prefs_before is not None:
                restored_entry = bpy.context.preferences.addons.get(package_name)
                assert restored_entry is not None
                restored = restored_entry.preferences
                for name, value in prefs_before.items():
                    prop = restored.bl_rna.properties.get(name)
                    if prop is None or prefs_schema.get(name, (None,))[0] != prop.type:
                        continue
                    try:
                        setattr(restored, name, value)
                    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                        continue
                for name, value in prefs_before.items():
                    if hasattr(restored, name):
                        assert getattr(restored, name) == value


def _load_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    package_dir = root / "mesh_focus_orbit"
    path = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "mfo_toolbar_isolated",
        path,
        submodule_search_locations=[str(package_dir)],
    )
    for name in tuple(sys.modules):
        if name == spec.name or name.startswith(spec.name + "."):
            sys.modules.pop(name, None)
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
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ],
        [],
        [(0, 1, 2), (0, 2, 3), (0, 3, 4)],
    )
    mesh.update()
    attr = mesh.attributes.new(".sculpt_face_set", "INT", "FACE")
    for item in attr.data:
        item.value = 7
    obj = bpy.data.objects.new("_mfo_guided_ridge_equivalence_object", mesh)
    obj.matrix_world = Matrix.Translation((1.25, -2.0, 0.5)) @ Matrix.Rotation(0.37, 4, "Z")
    try:
        baseline, baseline_reason = module.guided_ridge._guided_ridge_prepare_snapshot_sync(obj, 0, 7)
        cooperative, cooperative_reason = module.guided_ridge._guided_ridge_prepare_snapshot(obj, 0, 7)
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
            module.guided_ridge._guided_ridge_prepare_snapshot_steps(obj, 0, 7)
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
        baseline, baseline_reason = module.guided_ridge._guided_ridge_prepare_snapshot_sync(obj, 0, 7)
        assert baseline_reason is None, baseline_reason
        job = module.guided_ridge._guided_ridge_prepare_snapshot_steps(obj, 0, 7)
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
    candidate, info = module.guided_ridge._guided_ridge_exact_candidate(snapshot, guide, controls, normals)
    cooperative_job = module.guided_ridge._guided_ridge_exact_candidate_steps(snapshot, guide, controls, normals)
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
    integrity = module.guided_ridge._guided_ridge_mesh_integrity(snapshot, points, candidate)
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
    candidate, info = module.guided_ridge._guided_ridge_exact_candidate(snapshot, controls, controls, [Vector((0.0, 1.0, 0.0))] * 5)
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
    job = module.guided_ridge._guided_ridge_surface_section_record_job(
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
        candidate, info = module.guided_ridge._guided_ridge_exact_candidate(
            snapshot, controls, controls, normals, half_width=0.9
        )
        mask = np.asarray(info["crest_side_mask"], dtype=bool)
        changed = np.linalg.norm(np.asarray(candidate) - snapshot["coords_world"], axis=1) > 1.0e-12
        angular = np.mod(np.arctan2(snapshot["coords_world"][:, 1], snapshot["coords_world"][:, 0]), 2.0 * np.pi)
        normal_values = np.linalg.norm(np.asarray(candidate) - snapshot["coords_world"], axis=1)
        normal_threshold = max(float(snapshot["average_edge"]) * 0.02, 1.0e-8)
        assert int(np.count_nonzero(normal_values > 1.0e-12)) > 0
        minimum_width = module.guided_ridge._guided_ridge_width_limits(snapshot)[1]
        if radial >= 64:
            minimum_candidate, minimum_info = module.guided_ridge._guided_ridge_exact_candidate(
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
            module.guided_ridge._guided_ridge_exact_candidate(current_snapshot, controls, controls, normals)
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
        "default_width": module.guided_ridge._guided_ridge_width_limits(snapshot)[0],
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
    width_result = module.guided_ridge._guided_ridge_width_frame_result(
        snapshot, guide, controls, normals, half_width=None
    )
    candidate, info = module.guided_ridge._guided_ridge_exact_candidate(
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
    assert module.guided_ridge._guided_ridge_rail_support_unchanged(
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
    frame = module.guided_ridge._guided_ridge_width_frame(snapshot, controls, controls, normals)
    width_result = module.guided_ridge._guided_ridge_width_frame_result(
        snapshot, controls, controls, normals, 0.8, frame=frame
    )
    candidate, info = module.guided_ridge._guided_ridge_exact_candidate(
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
    assert module.guided_ridge._guided_ridge_rail_support_unchanged(
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

    original_state = module.runtime.guided_ridge_state
    original_preview = module.guided_ridge._guided_ridge_update_width_preview
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
        module.runtime.guided_ridge_state = state
        module.guided_ridge._guided_ridge_update_width_preview = lambda current: preview_calls.append(current["half_width"]) or True
        event = SimpleNamespace(type="WHEELUPMOUSE", value="PRESS", shift=False)
        result = module.VIEW3D_OT_mesh_focus_guided_ridge.modal(state["operator"], SimpleNamespace(), event)
        assert result == {"RUNNING_MODAL"}
        coarse = state["half_width"]
        event.shift = True
        module.VIEW3D_OT_mesh_focus_guided_ridge.modal(state["operator"], SimpleNamespace(), event)
        assert state["half_width"] > coarse and len(preview_calls) == 2
        return {"passed": True, "coarse_width": coarse, "fine_width": state["half_width"], "preview_calls": len(preview_calls)}
    finally:
        module.guided_ridge._guided_ridge_update_width_preview = original_preview
        module.runtime.guided_ridge_state = original_state


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
    tangent, lateral, normal = module.guided_ridge._guided_ridge_interpolate_vertex_frame(
        frame, segment_index, segment_t
    )
    expected_tangent = module.guided_ridge._guided_ridge_unit_array(
        (1.0 - segment_t)[:, None] * frame["tangent"][segment_index]
        + segment_t[:, None] * frame["tangent"][segment_index + 1]
    )
    assert np.allclose(tangent, expected_tangent)
    uniform_index = np.floor(np.asarray((0.5, 0.75, 2.25))).astype(np.int64)
    uniform_t = np.asarray((0.5, 0.75, 0.25))
    uniform, _unused_lateral, _unused_normal = module.guided_ridge._guided_ridge_interpolate_vertex_frame(
        frame, uniform_index, uniform_t
    )
    assert float(np.max(np.abs(tangent - uniform))) > 1.0e-3
    return {"passed": True, "nonuniform_diff": float(np.max(np.abs(tangent - uniform)))}


def _guided_ridge_width_cache_fixture(module):
    """Wheel events reuse the stable frame and only refresh rails/width."""
    import numpy as np
    from mathutils import Vector

    original_frame = module.guided_ridge._guided_ridge_width_frame
    original_rails = module.guided_ridge._guided_ridge_width_rails
    original_layers = module.guided_ridge._guided_ridge_surface_layer_steps
    original_state = module.runtime.guided_ridge_state
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
        module.guided_ridge._guided_ridge_width_frame = lambda *_args, **_kwargs: frame_calls.append(True) or frame
        module.guided_ridge._guided_ridge_width_rails = lambda *_args, **_kwargs: rail_calls.append(True) or {"left": [Vector((0, 0, 0))] * 2, "right": [Vector((0, 0, 0))] * 2}
        def layer_job(*_args, **_kwargs):
            layer_calls.append(True)
            yield {"stage": "building width surface layers", "done": 0, "total": 4}
            return {"entries": {"left": [None, None], "right": [None, None]}, "reasons": {}, "available": True}
        module.guided_ridge._guided_ridge_surface_layer_steps = layer_job
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
        module.runtime.guided_ridge_state = state
        assert module.guided_ridge._guided_ridge_update_width_preview(state)
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
        module.guided_ridge._guided_ridge_width_frame = original_frame
        module.guided_ridge._guided_ridge_width_rails = original_rails
        module.guided_ridge._guided_ridge_surface_layer_steps = original_layers
        module.runtime.guided_ridge_state = original_state


def _smart_fill_wheel_gated_fixture(module):
    """Normal-E accepts one wheel step only after a successful visible draw."""
    import numpy as np

    operator_class = module.VIEW3D_OT_mesh_focus_local_face_set_grow
    original_state = module.runtime.fill_preview_state
    original_valid = module.smart_fill_preview._fill_preview_valid
    original_make_result = module.smart_fill_preview._fill_preview_make_result
    original_tag_redraw = module.smart_fill_preview._fill_preview_tag_redraw
    original_writer = module.smart_fill_preview._fill_preview_write_vertex_paint
    original_cancel = module.smart_fill_preview._fill_preview_cancel
    original_reg_fill_valid = module.registration._fill_preview_valid
    original_reg_make_result = module.registration._fill_preview_make_result
    original_reg_tag_redraw = module.registration._fill_preview_tag_redraw
    original_reg_writer = module.registration._fill_preview_write_vertex_paint
    original_reg_cancel = module.registration._fill_preview_cancel
    calls = []

    class _Proxy:
        def report(self, *_args, **_kwargs):
            return None

        def _process_timer(self, context, state):
            return operator_class._process_timer(self, context, state)

        def _finish_confirm(self, context, state):
            return operator_class._finish_confirm(self, context, state)

    proxy = _Proxy()

    class _Ptr:
        def __init__(self, value):
            self.value = int(value)

        def as_pointer(self):
            return self.value

    _area = _Ptr(1201)
    _region = _Ptr(1202)
    _window = _Ptr(1203)
    _mesh = _Ptr(1205)
    _obj = _Ptr(1204)
    _obj.data = _mesh
    _area.tag_redraw = lambda: None
    context = SimpleNamespace(
        area=_area,
        region=_region,
        window=_window,
        active_object=_obj,
        mode="SCULPT",
    )

    def make_result(state, radius):
        calls.append(float(radius))
        return {
            "radius": float(radius),
            "compute_seconds": 0.001,
            "created_generation": int(state["generation"]),
        }

    def new_state(backend="SCULPT"):
        return {
            "active": True,
            "operator": proxy,
            "area": _area,
            "area_key": 1201,
            "region_key": 1202,
            "window_key": 1203,
            "obj": _obj,
            "obj_pointer": 1204,
            "mesh_pointer": 1205,
            "mode": "SCULPT",
            "strict_mode": False,
            "phase": "ready",
            "backend": backend,
            "initial_radius": 1.0,
            "desired_radius": 1.0,
            "processed_radius": 1.0,
            "wheel_armed": True,
            "wheel_gate": False,
            "pending": False,
            "result": {"radius": 1.0, "created_generation": 0},
            "results": {},
            "generation": 0,
            "drawn_generation": 0,
            "wheel_drain_until": 0.0,
            "timer": object(),
            "last_timer_dispatch": 0.0,
            "dropped_wheel_events": 0,
            "adjacency": {"hidden": np.asarray([False])},
            "seed_face": 0,
            "metrics": {"make_result_seconds": 0.0},
            "last_tick_seconds": 0.0,
            "max_tick_seconds": 0.0,
        }

    try:
        module.smart_fill_preview._fill_preview_valid = lambda _state, _context: True
        module.smart_fill_preview._fill_preview_make_result = make_result
        module.smart_fill_preview._fill_preview_tag_redraw = lambda _state=None: None
        module.smart_fill_preview._fill_preview_cancel = lambda _state=None, _reason="cancel": None
        module.registration._fill_preview_valid = module.smart_fill_preview._fill_preview_valid
        module.registration._fill_preview_make_result = module.smart_fill_preview._fill_preview_make_result
        module.registration._fill_preview_tag_redraw = module.smart_fill_preview._fill_preview_tag_redraw
        module.registration._fill_preview_write_vertex_paint = module.smart_fill_preview._fill_preview_write_vertex_paint
        module.registration._fill_preview_cancel = module.smart_fill_preview._fill_preview_cancel

        state = new_state("SCULPT")
        module.runtime.fill_preview_state = state
        wheel_up = SimpleNamespace(type="WHEELUPMOUSE", value="PRESS", shift=False)
        wheel_down = SimpleNamespace(type="WHEELDOWNMOUSE", value="PRESS", shift=False)
        # A burst while the first step is in flight accepts only the first
        # wheel.  The remaining events are discarded, not replayed later.
        assert operator_class.modal(proxy, context, wheel_up) == {"RUNNING_MODAL"}
        for _index in range(9):
            assert operator_class.modal(proxy, context, wheel_up) == {"RUNNING_MODAL"}
        assert not calls
        assert state["dropped_wheel_events"] == 9
        expected = 1.25
        assert np.isclose(state["desired_radius"], expected)
        assert state["phase"] == "compute" and state["pending"] is True
        # Confirm is blocked while the compute is pending.
        assert operator_class.modal(
            proxy, context, SimpleNamespace(type="ENTER", value="PRESS")
        ) == {"RUNNING_MODAL"}
        assert operator_class.modal(
            proxy, context, SimpleNamespace(type="TIMER", value="NOTHING")
        ) == {"RUNNING_MODAL"}
        assert len(calls) == 1 and np.isclose(calls[-1], expected)
        assert state["phase"] == "ready" and state["pending"] is False
        assert state["wheel_gate"] is True and state["wheel_armed"] is False

        # Before a successful draw/rearm, even a ready-state burst is ignored.
        for event in (wheel_up, wheel_down, wheel_up):
            assert operator_class.modal(proxy, context, event) == {"RUNNING_MODAL"}
        assert len(calls) == 1 and np.isclose(state["desired_radius"], expected)
        assert state["dropped_wheel_events"] == 12
        # Simulate the draw callback and its one-shot post-draw rearm timer.
        state["drawn_generation"] = state["generation"]
        state["wheel_drain_until"] = time.perf_counter() - 1.0
        state["last_timer_dispatch"] = 0.0
        assert operator_class.modal(
            proxy, context, SimpleNamespace(type="TIMER", value="NOTHING")
        ) == {"RUNNING_MODAL"}
        assert state["wheel_gate"] is False and state["wheel_armed"] is True
        assert operator_class.modal(proxy, context, wheel_up) == {"RUNNING_MODAL"}
        assert np.isclose(state["desired_radius"], expected * 1.25)
        assert len(calls) == 1
        assert state["phase"] == "compute" and state["pending"] is True
        # This step is not accepted again until its result is computed and
        # drawn, proving one accepted event maps to one radius step.
        assert operator_class.modal(proxy, context, wheel_down) == {"RUNNING_MODAL"}
        assert np.isclose(state["desired_radius"], expected * 1.25)
        assert state["dropped_wheel_events"] == 13

        # The normal path retains multiplicative clamp semantics once each
        # step is rearmed; no burst can jump directly to the limit.
        state["phase"] = "ready"
        state["pending"] = False
        state["generation"] += 1
        state["result"] = {"radius": 16.0, "created_generation": state["generation"]}
        state["drawn_generation"] = state["generation"]
        state["wheel_gate"] = False
        state["wheel_armed"] = True
        state["desired_radius"] = 16.0
        assert operator_class.modal(proxy, context, wheel_up) == {"RUNNING_MODAL"}
        assert np.isclose(state["desired_radius"], 16.0)
        state["phase"] = "ready"
        state["pending"] = False
        state["generation"] += 1
        state["result"] = {"radius": 0.125, "created_generation": state["generation"]}
        state["drawn_generation"] = state["generation"]
        state["wheel_gate"] = False
        state["wheel_armed"] = True
        state["desired_radius"] = 0.125
        assert operator_class.modal(proxy, context, wheel_down) == {"RUNNING_MODAL"}
        assert np.isclose(state["desired_radius"], 0.125)

        def accepted_sequence():
            sequence_state = new_state("SCULPT")
            module.runtime.fill_preview_state = sequence_state
            values = []
            for _index in range(4):
                sequence_state["drawn_generation"] = sequence_state["generation"]
                sequence_state["wheel_gate"] = True
                sequence_state["wheel_armed"] = False
                sequence_state["wheel_drain_until"] = time.perf_counter() - 1.0
                sequence_state["last_timer_dispatch"] = 0.0
                assert operator_class.modal(
                    proxy, context, SimpleNamespace(type="TIMER", value="NOTHING")
                ) == {"RUNNING_MODAL"}
                assert operator_class.modal(proxy, context, wheel_up) == {"RUNNING_MODAL"}
                values.append(float(sequence_state["desired_radius"]))
                assert operator_class.modal(
                    proxy, context, SimpleNamespace(type="TIMER", value="NOTHING")
                ) == {"RUNNING_MODAL"}
                sequence_state["drawn_generation"] = sequence_state["generation"]
                sequence_state["wheel_drain_until"] = time.perf_counter() - 1.0
            return values

        rapid_ready = accepted_sequence()
        slow_ready = accepted_sequence()
        assert np.allclose(rapid_ready, slow_ready)

        # Latest result becomes confirmable only after its draw generation is
        # published; this also exercises the shared normal path for Vertex Paint.
        state["backend"] = "PAINT_VERTEX"
        module.smart_fill_preview._fill_preview_write_vertex_paint = lambda _state, _result: (
            (1, 1, "POINT"), None
        )
        module.registration._fill_preview_write_vertex_paint = module.smart_fill_preview._fill_preview_write_vertex_paint
        state["pending"] = False
        state["phase"] = "ready"
        state["result"] = {"created_generation": state["generation"]}
        state["drawn_generation"] = state["generation"] - 1
        assert operator_class._finish_confirm(proxy, context, state) == {"RUNNING_MODAL"}
        state["drawn_generation"] = state["generation"]
        state["obj"] = SimpleNamespace()
        state["signature"] = None
        assert operator_class._finish_confirm(proxy, context, state) == {"FINISHED"}

        # Ctrl+E retains the historical immediate pending/compute transition
        # and never enters the normal-mode draw gate.
        strict_state = new_state("SCULPT")
        strict_state.update(
            {
                "strict_mode": True,
                "wheel_armed": True,
                "result": {"radius": 1.0, "created_generation": 0},
            }
        )
        module.runtime.fill_preview_state = strict_state
        assert operator_class.modal(proxy, context, wheel_up) == {"RUNNING_MODAL"}
        assert strict_state["phase"] == "compute"
        assert strict_state["pending"] is True
        assert strict_state["result"] is None
        return {
            "passed": True,
            "normal_burst_events": 10,
            "normal_make_result_calls": 1,
            "dropped_inflight_events": 13,
            "one_step_per_accept": True,
            "clamp_min": 0.125,
            "rapid_slow_sequence_equal": True,
            "confirm_blocked_until_draw": True,
            "sculpt_and_vertex_backends": True,
            "strict_unchanged": True,
        }
    finally:
        module.runtime.fill_preview_state = original_state
        module.registration._fill_preview_valid = original_reg_fill_valid
        module.registration._fill_preview_make_result = original_reg_make_result
        module.registration._fill_preview_tag_redraw = original_reg_tag_redraw
        module.registration._fill_preview_write_vertex_paint = original_reg_writer
        module.registration._fill_preview_cancel = original_reg_cancel
        module.smart_fill_preview._fill_preview_valid = original_valid
        module.smart_fill_preview._fill_preview_make_result = original_make_result
        module.smart_fill_preview._fill_preview_tag_redraw = original_tag_redraw
        module.smart_fill_preview._fill_preview_write_vertex_paint = original_writer
        module.smart_fill_preview._fill_preview_cancel = original_cancel


def _smart_fill_draw_generation_fixture(module):
    """A failed GPU draw must not open the confirm generation gate."""
    import numpy as np

    original_state = module.runtime.fill_preview_state
    original_bpy = module.smart_fill_preview.bpy
    original_gpu = module.smart_fill_preview.gpu
    original_shader = module.runtime.fill_preview_shader
    original_batches = module.smart_fill_preview._fill_preview_build_draw_batches
    original_draw = module.smart_fill_preview._fill_preview_draw
    original_valid = module.smart_fill_preview._fill_preview_valid
    original_writer = module.smart_fill_preview._fill_preview_write_vertex_paint
    original_cancel = module.smart_fill_preview._fill_preview_cancel
    original_reg_valid = module.registration._fill_preview_valid
    original_reg_writer = module.registration._fill_preview_write_vertex_paint
    original_reg_cancel = module.registration._fill_preview_cancel

    class _Area:
        def as_pointer(self):
            return 941

    class _Shader:
        def bind(self):
            return None

        def uniform_float(self, *_args):
            return None

    class _Batch:
        def __init__(self):
            self.fail = True

        def draw(self, _shader):
            if self.fail:
                raise RuntimeError("forced preview draw failure")

    class _GPUState:
        def blend_set(self, _value):
            return None

        def depth_test_set(self, _value):
            return None

        def depth_mask_set(self, _value):
            return None

        def line_width_set(self, _value):
            return None

        def point_size_set(self, _value):
            return None

    fake_gpu = SimpleNamespace(state=_GPUState())
    fake_area = _Area()
    fake_bpy = SimpleNamespace(context=SimpleNamespace(area=fake_area))
    batch = _Batch()
    result = {
        "triangles_np": np.empty((0, 3), dtype=np.float32),
        "distance_lines_np": np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float32),
        "shape_lines_np": np.empty((0, 3), dtype=np.float32),
        "gpu_batches": {"distance_lines": batch},
        "boundary_all_orange": False,
        "created_generation": 4,
    }
    state = {
        "active": True,
        "area_key": 941,
        "area": fake_area,
        "phase": "ready",
        "result": result,
        "generation": 4,
        "drawn_generation": None,
        "wheel_drain_until": 0.0,
        "strict_mode": False,
        "pending": False,
        "backend": "PAINT_VERTEX",
        "obj": SimpleNamespace(),
        "signature": None,
    }
    try:
        module.runtime.fill_preview_state = state
        module.smart_fill_preview.bpy = fake_bpy
        module.smart_fill_preview.gpu = fake_gpu
        module.runtime.fill_preview_shader = _Shader()
        module.smart_fill_preview._fill_preview_build_draw_batches = lambda _state, _result: None
        module.smart_fill_preview._fill_preview_valid = lambda _state, _context: True
        module.smart_fill_preview._fill_preview_write_vertex_paint = lambda _state, _result: ((1, 1, "POINT"), None)
        module.smart_fill_preview._fill_preview_cancel = lambda _state=None, _reason="cancel": None
        module.registration._fill_preview_valid = module.smart_fill_preview._fill_preview_valid
        module.registration._fill_preview_write_vertex_paint = module.smart_fill_preview._fill_preview_write_vertex_paint
        module.registration._fill_preview_cancel = module.smart_fill_preview._fill_preview_cancel
        module.smart_fill_preview._fill_preview_draw = original_draw

        # The real draw body swallows the batch exception; the generation must
        # nevertheless remain stale and confirmation must remain RUNNING.
        module.smart_fill_preview._fill_preview_draw()
        assert state["drawn_generation"] is None
        operator_class = module.VIEW3D_OT_mesh_focus_local_face_set_grow
        proxy = SimpleNamespace(report=lambda *_args, **_kwargs: None)
        assert operator_class._finish_confirm(proxy, SimpleNamespace(), state) == {"RUNNING_MODAL"}
        wheel_proxy = SimpleNamespace(
            report=lambda *_args, **_kwargs: None,
            _process_timer=lambda _context, _state: True,
        )
        state.update({
            "operator": wheel_proxy,
            "initial_radius": 1.0,
            "desired_radius": 1.0,
            "phase": "ready",
            "pending": False,
            "wheel_armed": False,
            "wheel_gate": True,
            "dropped_wheel_events": 0,
        })
        assert operator_class.modal(
            wheel_proxy,
            SimpleNamespace(),
            SimpleNamespace(type="WHEELUPMOUSE", value="PRESS"),
        ) == {"RUNNING_MODAL"}
        assert state["dropped_wheel_events"] == 1

        # A later redraw succeeds and only then opens confirmation.
        batch.fail = False
        module.smart_fill_preview._fill_preview_draw()
        assert state["drawn_generation"] == 4
        first_drain = float(state["wheel_drain_until"])
        state["strict_mode"] = True
        module.smart_fill_preview._fill_preview_draw()
        module.smart_fill_preview._fill_preview_draw()
        assert float(state["wheel_drain_until"]) == first_drain

        class _ModalProxy:
            def _process_timer(self, _context, _state):
                return True

            def report(self, *_args, **_kwargs):
                return None

        modal_proxy = _ModalProxy()
        state["operator"] = modal_proxy
        state.update({
            "wheel_gate": True,
            "wheel_armed": False,
            "wheel_drain_until": time.perf_counter() - 1.0,
            "last_timer_dispatch": 0.0,
            "timer": None,
        })
        assert operator_class.modal(
            modal_proxy,
            SimpleNamespace(),
            SimpleNamespace(type="TIMER", value="NOTHING", timer=None),
        ) == {"RUNNING_MODAL"}
        assert state["wheel_armed"] is True and state["wheel_gate"] is False
        state["strict_mode"] = False
        assert operator_class._finish_confirm(proxy, SimpleNamespace(), state) == {"FINISHED"}
        return {
            "passed": True,
            "forced_draw_blocked": True,
            "successful_redraw_generation": int(state["drawn_generation"]),
            "strict_redraw_drain_stable": True,
            "strict_redraw_rearmed": True,
            "draw_failure_wheel_blocked": True,
        }
    finally:
        module.runtime.fill_preview_state = original_state
        module.smart_fill_preview.bpy = original_bpy
        module.smart_fill_preview.gpu = original_gpu
        module.runtime.fill_preview_shader = original_shader
        module.smart_fill_preview._fill_preview_build_draw_batches = original_batches
        module.smart_fill_preview._fill_preview_draw = original_draw
        module.smart_fill_preview._fill_preview_valid = original_valid
        module.smart_fill_preview._fill_preview_write_vertex_paint = original_writer
        module.smart_fill_preview._fill_preview_cancel = original_cancel
        module.registration._fill_preview_valid = original_reg_valid
        module.registration._fill_preview_write_vertex_paint = original_reg_writer
        module.registration._fill_preview_cancel = original_reg_cancel


def _smart_fill_terminal_expansion_fixture(module):
    """The terminal normal range uses one retained frontier and barriers."""
    import numpy as np

    def geometry():
        # A linear six-face graph with deliberately non-wheel-sized distances.
        count = 6
        first = np.arange(0, 5, dtype=np.int32)
        second = np.arange(1, 6, dtype=np.int32)
        neighbors = np.asarray((1, 0, 2, 1, 3, 2, 4, 3, 5, 4), dtype=np.int32)
        offsets = np.asarray((0, 1, 3, 5, 7, 9, 10), dtype=np.int64)
        pair_v0 = np.arange(0, 5, dtype=np.int32)
        pair_v1 = np.arange(1, 6, dtype=np.int32)
        return {
            "count": count,
            "face_ids": np.arange(count, dtype=np.int32),
            "hidden": np.zeros(count, dtype=bool),
            "offsets": offsets,
            "neighbors": neighbors,
            "neighbor_lengths": np.asarray((1.0, 1.0, 9.0, 9.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0), dtype=np.float64),
            "first": first,
            "second": second,
            "pair_v0": pair_v0,
            "pair_v1": pair_v1,
            "world_vertices": np.asarray([(float(i), 0.0, 0.0) for i in range(6)], dtype=np.float64),
            "terminal_coverage_certified": True,
        }

    def state_for(barrier=False):
        graph = geometry()
        base = {
            "faces": np.asarray((0, 1), dtype=np.int32),
            "boundary_records": (
                {
                    "geometry_face_a": 1,
                    "geometry_face_b": 2,
                    "shape": bool(barrier),
                },
            ) if barrier else (),
            "shape_segments": [((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))] if barrier else [],
            "distance_segments": [],
            "signature": ("terminal", int(barrier)),
        }
        state = {
            "strict_mode": False,
            "backend": "SCULPT",
            "adjacency": graph,
            "signature": base["signature"],
            "seed_face": 0,
            "initial_radius": 1.0,
            "generation": 8,
            "terminal_base": None,
            "terminal_base_radius": None,
            "terminal_base_signature": None,
            "terminal_frontier": None,
            "terminal_expansion_count": 0,
            "distance_state": {
                "distances": np.asarray((0.0, 1.0, 2.0, 3.0, 4.0, 5.0), dtype=np.float64),
            },
        }
        return state, base

    state, base = state_for(False)
    assert module.smart_fill_preview._fill_preview_terminal_store(state, base, 8.192)
    full_calls = {"count": 0}
    original_enclosed = module.smart_fill_preview._fill_preview_resolve_enclosed_components
    original_sandwiched = module.smart_fill_preview._fill_preview_resolve_sandwiched_bands
    module.smart_fill_preview._fill_preview_resolve_enclosed_components = lambda *args, **kwargs: full_calls.__setitem__("count", full_calls["count"] + 1)
    module.smart_fill_preview._fill_preview_resolve_sandwiched_bands = lambda *args, **kwargs: full_calls.__setitem__("count", full_calls["count"] + 1)
    try:
        radii = (10.24, 12.8, 16.0)
        results = [module.smart_fill_preview._fill_preview_terminal_result(state, radius) for radius in radii]
        assert all(result is not None for result in results)
        sizes = [len(result["faces"]) for result in results]
        assert sizes == [3, 4, 6], sizes
        for result in results:
            assert np.array_equal(
                np.sort(np.asarray(result["faces"])),
                np.sort(np.asarray(result["confirm_geometry"]["face_ids"])[np.asarray(result["confirm_domain_ids"])])
            )
        barrier_state, barrier_base = state_for(True)
        assert module.smart_fill_preview._fill_preview_terminal_store(barrier_state, barrier_base, 8.192)
        blocked = module.smart_fill_preview._fill_preview_terminal_result(barrier_state, 16.0)
        assert blocked is not None and np.array_equal(blocked["faces"], np.asarray((0, 1), dtype=np.int32))
        assert full_calls["count"] == 0
        stale_state, stale_base = state_for(False)
        assert module.smart_fill_preview._fill_preview_terminal_store(stale_state, stale_base, 8.192)
        stale_state["signature"] = ("stale",)
        assert module.smart_fill_preview._fill_preview_terminal_result(stale_state, 16.0) is None
        assert stale_state["terminal_base"] is None
        strict_state, strict_base = state_for(False)
        strict_state["strict_mode"] = True
        assert not module.smart_fill_preview._fill_preview_terminal_store(strict_state, strict_base, 8.192)
        paint_state, paint_base = state_for(False)
        paint_state["backend"] = "PAINT_VERTEX"
        assert module.smart_fill_preview._fill_preview_terminal_store(paint_state, paint_base, 8.192)
        paint_result = module.smart_fill_preview._fill_preview_terminal_result(paint_state, 10.24)
        assert paint_result is not None
        assert np.array_equal(paint_result["faces"], results[0]["faces"])
        # Once the frontier has reached 16, an uncached shrink must invalidate
        # the additive set.  A fresh full result at 13.5 is the only authority.
        shrink_state, shrink_base = state_for(False)
        assert module.smart_fill_preview._fill_preview_terminal_store(shrink_state, shrink_base, 8.192)
        assert module.smart_fill_preview._fill_preview_terminal_result(shrink_state, 16.0) is not None
        assert module.smart_fill_preview._fill_preview_terminal_result(shrink_state, 13.5) is None
        assert shrink_state["terminal_base"] is None
        shrink_distances = np.asarray((0.0, 1.0, 10.0, 12.0, np.inf, np.inf), dtype=np.float64)
        shrink_snapshot, shrink_domain = module.smart_fill_preview._fill_preview_confirm_graph_snapshot(
            shrink_state["adjacency"], shrink_distances, 13.5
        )
        shrink_display = np.asarray((0, 1, 2, 3), dtype=np.int32)
        assert np.array_equal(
            np.sort(shrink_display),
            np.sort(np.asarray(shrink_snapshot["face_ids"])[shrink_domain]),
        )

        # Exercise the actual timer branch at just below/at threshold and the
        # following terminal stage rather than only calling the helper.
        original_make = module.smart_fill_preview._fill_preview_make_result
        original_valid = module.smart_fill_preview._fill_preview_valid
        original_tag = module.smart_fill_preview._fill_preview_tag_redraw
        original_reg_make = module.registration._fill_preview_make_result
        original_reg_valid = module.registration._fill_preview_valid
        original_reg_tag = module.registration._fill_preview_tag_redraw
        timer_calls = []
        timer_state, timer_base = state_for(False)
        timer_state.update({
            "phase": "compute", "pending": True, "results": {},
            "result": None, "generation": 0, "desired_radius": 8.1,
            "processed_radius": None, "wheel_armed": False, "wheel_gate": False,
            "active_result_cache_hit": False, "metrics": {"make_result_seconds": 0.0},
            "last_tick_seconds": 0.0, "max_tick_seconds": 0.0,
        })
        def timer_make_result(current_state, radius):
            timer_calls.append(float(radius))
            result = dict(timer_base)
            result.update({
                "radius": float(radius),
                "created_generation": int(current_state["generation"]),
                "compute_seconds": 0.0,
            })
            return result
        timer_proxy = SimpleNamespace(report=lambda *_args, **_kwargs: None)
        module.smart_fill_preview._fill_preview_make_result = timer_make_result
        module.smart_fill_preview._fill_preview_valid = lambda _state, _context: True
        module.smart_fill_preview._fill_preview_tag_redraw = lambda _state=None: None
        module.registration._fill_preview_make_result = module.smart_fill_preview._fill_preview_make_result
        module.registration._fill_preview_valid = module.smart_fill_preview._fill_preview_valid
        module.registration._fill_preview_tag_redraw = module.smart_fill_preview._fill_preview_tag_redraw
        try:
            assert module.VIEW3D_OT_mesh_focus_local_face_set_grow._process_timer(
                timer_proxy, SimpleNamespace(), timer_state
            ) is True
            assert timer_calls == [8.1]
            assert timer_state["terminal_base"] is None
            timer_state.update({"phase": "compute", "pending": True, "desired_radius": 8.192})
            assert module.VIEW3D_OT_mesh_focus_local_face_set_grow._process_timer(
                timer_proxy, SimpleNamespace(), timer_state
            ) is True
            assert timer_calls == [8.1, 8.192]
            assert timer_state["terminal_base_radius"] == 8.192
            timer_state.update({"phase": "compute", "pending": True, "desired_radius": 10.24})
            assert module.VIEW3D_OT_mesh_focus_local_face_set_grow._process_timer(
                timer_proxy, SimpleNamespace(), timer_state
            ) is True
            assert timer_calls == [8.1, 8.192]
        finally:
            module.smart_fill_preview._fill_preview_make_result = original_make
            module.smart_fill_preview._fill_preview_valid = original_valid
            module.smart_fill_preview._fill_preview_tag_redraw = original_tag
            module.registration._fill_preview_make_result = original_reg_make
            module.registration._fill_preview_valid = original_reg_valid
            module.registration._fill_preview_tag_redraw = original_reg_tag
            module.registration._fill_preview_make_result = original_reg_make
            module.registration._fill_preview_valid = original_reg_valid
            module.registration._fill_preview_tag_redraw = original_reg_tag
        cropped_state, cropped_base = state_for(False)
        cropped_state["adjacency"]["terminal_coverage_certified"] = False
        assert not module.smart_fill_preview._fill_preview_terminal_store(cropped_state, cropped_base, 8.192)
        assert cropped_state["terminal_base"] is None
        return {
            "passed": True,
            "terminal_base_radius": 8.192,
            "expansion_stages": 3,
            "shape_barrier_preserved": True,
            "expensive_pass_calls": int(full_calls["count"]),
            "max_frontier_pop": max(int(state["terminal_frontier"]["popped"]), 0),
            "stale_and_strict_fallbacks": True,
            "sculpt_and_vertex_backends": True,
            "shrink_full_result_only": True,
            "threshold_timer_branch": True,
            "cropped_graph_fallback": True,
        }
    finally:
        module.smart_fill_preview._fill_preview_resolve_enclosed_components = original_enclosed
        module.smart_fill_preview._fill_preview_resolve_sandwiched_bands = original_sandwiched


def _smart_fill_large_region_delta_fixture(module):
    """Exercise count-triggered frontier-delta maintenance before 8.192x."""
    import time
    import numpy as np

    pairs = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7))
    count = 8
    first = np.asarray([pair[0] for pair in pairs], dtype=np.int32)
    second = np.asarray([pair[1] for pair in pairs], dtype=np.int32)
    adjacency = [[] for _ in range(count)]
    for edge, (left, right) in enumerate(pairs):
        adjacency[left].append((right, edge))
        adjacency[right].append((left, edge))
    offsets = [0]
    neighbors = []
    lengths = []
    for values in adjacency:
        for neighbor, _edge in values:
            neighbors.append(neighbor)
            lengths.append(1.0)
        offsets.append(len(neighbors))
    graph = {
        "count": count,
        "face_ids": np.arange(count, dtype=np.int32),
        "hidden": np.zeros(count, dtype=bool),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "neighbors": np.asarray(neighbors, dtype=np.int32),
        "neighbor_lengths": np.asarray(lengths, dtype=np.float64),
        "first": first,
        "second": second,
        "pair_v0": np.arange(len(pairs), dtype=np.int32),
        "pair_v1": np.arange(1, len(pairs) + 1, dtype=np.int32),
        "world_vertices": np.asarray(
            [(float(index), 0.0, 0.0) for index in range(len(pairs) + 1)],
            dtype=np.float64,
        ),
        "terminal_coverage_certified": True,
    }

    def make_state(backend):
        base_geometry, base_domain = module.smart_fill_preview._fill_preview_confirm_graph_snapshot(
            graph, np.asarray((0, 1, 2, 3, 4, np.inf, 3, 4), dtype=np.float64), 2.0
        )
        base_domain = np.asarray((0, 1, 2, 3, 4), dtype=np.int32)
        # Keep an orange barrier on 4-5 while branch 2-6-7 remains reachable.
        base = {
            "faces": np.asarray((0, 1, 2, 3, 4), dtype=np.int32),
            "boundary_records": ({
                "geometry_face_a": 4,
                "geometry_face_b": 5,
                "mesh_face_a": 4,
                "mesh_face_b": 5,
                "mesh_edge": -1,
                "shape": True,
            },),
            "shape_segments": [((4.0, 0.0, 0.0), (5.0, 0.0, 0.0))],
            "distance_segments": [],
            "confirm_geometry": base_geometry,
            "confirm_domain_ids": base_domain,
            "signature": ("large-delta", backend),
        }
        state = {
            "strict_mode": False,
            "backend": backend,
            "adjacency": graph,
            "signature": base["signature"],
            "seed_face": 0,
            "initial_radius": 1.0,
            "generation": 3,
            "terminal_base": None,
            "terminal_base_radius": None,
            "terminal_base_signature": None,
            "terminal_frontier": None,
            "terminal_expansion_count": 0,
            "distance_state": {
                "distances": np.asarray(
                    (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0),
                    dtype=np.float64,
                ),
            },
        }
        return state, base

    original_threshold = module.smart_fill_preview.FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD
    original_threshold_private = module.smart_fill_preview._FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD
    original_cache_builder = module.smart_fill_preview._fill_preview_terminal_boundary_cache
    cache_calls = {"count": 0}
    try:
        # An injectable low threshold keeps this regression small while
        # exercising the exact >=100k branch used by production (100,000).
        module.smart_fill_preview.FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = 4
        module.smart_fill_preview._FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = 4
        for backend in ("SCULPT", "PAINT_VERTEX"):
            state, base = make_state(backend)
            assert module.smart_fill_preview._fill_preview_terminal_store(state, base, 2.0)
            # A support-domain face may already be present before it becomes
            # selected.  The delta result must union it exactly once.
            state["terminal_frontier"]["confirm_domain"].add(5)
            assert state["terminal_base_radius"] < (
                state["initial_radius"] * module.smart_fill_preview.FILL_PREVIEW_TERMINAL_THRESHOLD_FACTOR
            )

            def counted_builder(*args, **kwargs):
                cache_calls["count"] += 1
                return original_cache_builder(*args, **kwargs)

            module.smart_fill_preview._fill_preview_terminal_boundary_cache = counted_builder
            cache_calls_before_delta = int(cache_calls["count"])
            started = time.perf_counter()
            delta = module.smart_fill_preview._fill_preview_terminal_result(state, 4.5)
            elapsed = time.perf_counter() - started
            assert delta is not None
            assert delta["progressive_range_terminal_mode"] == "frontier-delta"
            assert delta["progressive_range_full_boundary_scan"] is False
            assert delta["progressive_range_confirm_geometry_reused"] is True
            confirm_ids = np.asarray(
                delta["confirm_geometry"]["face_ids"]
            )[delta["confirm_domain_ids"]]
            assert set(int(value) for value in delta["faces"]).issubset(
                set(int(value) for value in confirm_ids)
            )
            assert len(delta["confirm_domain_ids"]) == len(np.unique(delta["confirm_domain_ids"]))
            assert 5 in set(int(value) for value in delta["confirm_domain_ids"])
            assert delta["progressive_range_confirm_sort_count"] == 1
            # The only boundary is the retained orange barrier; the branch
            # edge 2-6 is internal after growth and is removed incrementally.
            actual_pairs = {
                tuple(sorted((int(row["geometry_face_a"]), int(row["geometry_face_b"]))))
                for row in delta["boundary_records"]
            }
            expected_pairs = {
                tuple(sorted(pair))
                for pair in pairs
                if ((pair[0] in set(int(value) for value in delta["faces"]))
                    != (pair[1] in set(int(value) for value in delta["faces"])))
            }
            assert actual_pairs == expected_pairs, (actual_pairs, expected_pairs)
            assert any(bool(row["shape"]) for row in delta["boundary_records"])
            assert cache_calls["count"] == cache_calls_before_delta, {
                "cache_calls": cache_calls["count"],
                "mode": delta.get("progressive_range_terminal_mode"),
                "full_scan": delta.get("progressive_range_full_boundary_scan"),
            }
            same_faces = module.smart_fill_preview._fill_preview_terminal_result(state, 4.75)
            assert same_faces is not None
            assert same_faces["progressive_range_newly_processed_faces"] == 0
            assert same_faces["faces"] is delta["faces"]
            assert cache_calls["count"] == cache_calls_before_delta
            assert elapsed < 0.5

        # Below the production threshold retains the old full terminal path.
        module.smart_fill_preview.FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = 100_000
        module.smart_fill_preview._FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = 100_000
        state, base = make_state("SCULPT")
        assert module.smart_fill_preview._fill_preview_terminal_store(state, base, 2.0)
        full = module.smart_fill_preview._fill_preview_terminal_result(state, 4.5)
        assert full is not None
        assert full["progressive_range_terminal_mode"] == "frontier"
        assert full["progressive_range_full_boundary_scan"] is True

        # Crossing the count threshold must refresh the boundary cache from
        # the immediately preceding full result before the first delta tick.
        module.smart_fill_preview._fill_preview_terminal_boundary_cache = original_cache_builder
        module.smart_fill_preview.FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = 3
        module.smart_fill_preview._FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = 3
        transition_geometry, transition_domain = module.smart_fill_preview._fill_preview_confirm_graph_snapshot(
            graph, np.asarray((0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0)), 1.0
        )
        transition_base = {
            "faces": np.asarray((0, 1), dtype=np.int32),
            "boundary_records": (
                {"geometry_face_a": 1, "geometry_face_b": 2, "shape": False},
                {"geometry_face_a": 2, "geometry_face_b": 6, "shape": True},
            ),
            "shape_segments": [((2.0, 0.0, 0.0), (6.0, 0.0, 0.0))],
            "distance_segments": [],
            "confirm_geometry": transition_geometry,
            "confirm_domain_ids": np.asarray((0, 1), dtype=np.int32),
        }
        transition_state = {
            "strict_mode": False, "backend": "SCULPT", "adjacency": graph,
            "signature": ("transition",), "seed_face": 0, "initial_radius": 1.0,
            "generation": 1, "terminal_base": None, "terminal_base_radius": None,
            "terminal_base_signature": None, "terminal_frontier": None,
            "terminal_expansion_count": 0,
            "distance_state": {"distances": np.asarray((0., 1., 2., 3., 4., 5., 3., 4.))},
        }
        assert module.smart_fill_preview._fill_preview_terminal_store(transition_state, transition_base, 1.0)
        subthreshold = module.smart_fill_preview._fill_preview_terminal_result(transition_state, 2.0)
        assert subthreshold is not None and subthreshold["progressive_range_terminal_mode"] == "frontier"
        assert np.array_equal(subthreshold["faces"], np.asarray((0, 1, 2), dtype=np.int32))
        assert transition_state["terminal_frontier"]["selected"] == {0, 1, 2}
        crossing = module.smart_fill_preview._fill_preview_terminal_result(transition_state, 3.5)
        assert crossing is not None and crossing["progressive_range_terminal_mode"] == "frontier-delta"
        crossing_pairs = {
            tuple(sorted((int(row["geometry_face_a"]), int(row["geometry_face_b"]))))
            for row in crossing["boundary_records"]
        }
        assert (1, 2) not in crossing_pairs and (2, 3) not in crossing_pairs
        assert (3, 4) in crossing_pairs and (2, 6) in crossing_pairs

        # Inject a provenance failure after the walk has reached a new face.
        # The mutated terminal state must be discarded and the timer must run
        # the ordinary full resolver instead.
        module.smart_fill_preview.FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = 4
        module.smart_fill_preview._FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = 4
        module.smart_fill_preview._fill_preview_terminal_boundary_cache = original_cache_builder
        failure_state, failure_base = make_state("SCULPT")
        assert module.smart_fill_preview._fill_preview_terminal_store(failure_state, failure_base, 2.0)
        original_delta = module.smart_fill_preview._fill_preview_terminal_delta_result
        original_make = module.smart_fill_preview._fill_preview_make_result
        original_valid = module.smart_fill_preview._fill_preview_valid
        original_tag = module.smart_fill_preview._fill_preview_tag_redraw
        original_reg_make = module.registration._fill_preview_make_result
        original_reg_valid = module.registration._fill_preview_valid
        original_reg_tag = module.registration._fill_preview_tag_redraw
        failure_seen = {"called": False}
        fallback_calls = []
        def failing_delta(current_state, base_value, frontier_value, *args):
            failure_seen["called"] = True
            frontier_value["boundary_edges"][999999] = True
            frontier_value["confirm_domain"].add(999999)
            return None
        def fallback_make(current_state, radius):
            fallback_calls.append(float(radius))
            value = dict(failure_base)
            value.update({"radius": float(radius), "compute_seconds": 0.0, "created_generation": int(current_state["generation"])})
            return value
        failure_state.update({
            "phase": "compute", "pending": True, "results": {}, "result": None,
            "generation": 0, "desired_radius": 4.5, "processed_radius": None,
            "wheel_armed": False, "wheel_gate": False, "active_result_cache_hit": False,
            "metrics": {"make_result_seconds": 0.0}, "last_tick_seconds": 0.0,
            "max_tick_seconds": 0.0,
        })
        module.smart_fill_preview._fill_preview_terminal_delta_result = failing_delta
        module.smart_fill_preview._fill_preview_make_result = fallback_make
        module.smart_fill_preview._fill_preview_valid = lambda _state, _context: True
        module.smart_fill_preview._fill_preview_tag_redraw = lambda _state=None: None
        module.registration._fill_preview_make_result = fallback_make
        module.registration._fill_preview_valid = lambda _state, _context: True
        module.registration._fill_preview_tag_redraw = lambda _state=None: None
        try:
            proxy = SimpleNamespace(report=lambda *_args, **_kwargs: None)
            assert module.VIEW3D_OT_mesh_focus_local_face_set_grow._process_timer(
                proxy, SimpleNamespace(), failure_state
            ) is True
            assert failure_seen["called"] and fallback_calls == [4.5]
            assert failure_state["result"]["radius"] == 4.5
        finally:
            module.smart_fill_preview._fill_preview_terminal_delta_result = original_delta
            module.smart_fill_preview._fill_preview_make_result = original_make
            module.smart_fill_preview._fill_preview_valid = original_valid
            module.smart_fill_preview._fill_preview_tag_redraw = original_tag
        return {
            "passed": True,
            "backends": ["SCULPT", "PAINT_VERTEX"],
            "count_trigger_before_radius_threshold": True,
            "delta_mode": True,
            "boundary_equivalent": True,
            "orange_barrier_preserved": True,
            "confirm_display_equal": True,
            "terminal_base_cache_build_calls": int(cache_calls["count"]),
            "delta_cache_builder_calls_after_base": 0,
            "delta_seconds": float(elapsed),
            "below_threshold_full_path": True,
            "crossing_cache_refresh": True,
            "delta_failure_discards_frontier": True,
        }
    finally:
        module.smart_fill_preview.FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = original_threshold
        module.smart_fill_preview._FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = original_threshold_private
        module.smart_fill_preview._fill_preview_terminal_boundary_cache = original_cache_builder


def _smart_fill_large_region_benchmark_fixture(module):
    """Measure the retained-frontier path on a stable 200k-face chain."""
    import time
    import numpy as np

    count = 200_000
    base_count = 120_000
    delta_count = 20_000
    target_count = base_count + delta_count
    first = np.arange(count - 1, dtype=np.int32)
    second = first + 1
    # Neighbor rows are [right] for face 0, [left,right] for interiors,
    # [left] for the last face.
    neighbors = np.empty(2 * count - 2, dtype=np.int32)
    neighbors[0] = 1
    cursor = 1
    for face in range(1, count - 1):
        neighbors[cursor:cursor + 2] = (face - 1, face + 1)
        cursor += 2
    neighbors[cursor] = count - 2
    offsets = np.empty(count + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1] = 1
    offsets[2:count] = 3 + 2 * np.arange(count - 2, dtype=np.int64)
    offsets[count] = len(neighbors)
    graph = {
        "count": count,
        "face_ids": np.arange(count, dtype=np.int32),
        "hidden": np.zeros(count, dtype=bool),
        "offsets": offsets,
        "neighbors": neighbors,
        "neighbor_lengths": np.ones(len(neighbors), dtype=np.float64),
        "first": first,
        "second": second,
        "pair_v0": first.copy(),
        "pair_v1": second.copy(),
        "world_vertices": np.column_stack((
            np.arange(count, dtype=np.float64),
            np.zeros((count, 2), dtype=np.float64),
        )),
        "terminal_coverage_certified": True,
    }
    confirm_geometry = {
        "count": count,
        "face_ids": np.arange(count, dtype=np.int32),
        "hidden": np.zeros(count, dtype=bool),
        "offsets": offsets.copy(),
        "neighbors": neighbors.copy(),
    }
    base = {
        "faces": np.arange(base_count, dtype=np.int32),
        "boundary_records": ({
            "geometry_face_a": base_count - 1,
            "geometry_face_b": base_count,
            "shape": False,
        },),
        "shape_segments": [],
        "distance_segments": [
            ((float(base_count - 1), 0.0, 0.0), (float(base_count), 0.0, 0.0))
        ],
        "confirm_geometry": confirm_geometry,
        "confirm_domain_ids": np.arange(base_count, dtype=np.int32),
        "signature": ("200k-benchmark",),
    }
    state = {
        "strict_mode": False,
        "backend": "SCULPT",
        "adjacency": graph,
        "signature": base["signature"],
        "seed_face": 0,
        "initial_radius": 1.0,
        "generation": 4,
        "terminal_base": None,
        "terminal_base_radius": None,
        "terminal_base_signature": None,
        "terminal_frontier": None,
        "terminal_expansion_count": 0,
        "distance_state": {"distances": np.arange(count, dtype=np.float64)},
    }
    original_threshold = module._FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD
    try:
        module._FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = 100_000
        assert module.smart_fill_preview._fill_preview_terminal_store(state, base, float(base_count - 1))
        assert state["terminal_frontier"]["frontier_seed_count"] == 1
        # Reference edge ownership is the one chain edge crossing the final
        # selected face.  The delta path must produce the same result.
        reference_started = time.perf_counter()
        reference_boundary = {
            int(edge)
            for edge in range(count - 1)
            if (edge < target_count) != (edge + 1 < target_count)
        }
        reference_seconds = time.perf_counter() - reference_started
        before_scans = int(state.get("terminal_coverage_scan_count", 0))
        started = time.perf_counter()
        result = module.smart_fill_preview._fill_preview_terminal_result(state, float(target_count - 1))
        delta_seconds = time.perf_counter() - started
        assert result is not None
        assert result["progressive_range_terminal_mode"] == "frontier-delta"
        selected = set(int(value) for value in result["faces"])
        actual_boundary = {
            min(int(record["geometry_face_a"]), int(record["geometry_face_b"]))
            for record in result["boundary_records"]
            if (
                (int(record["geometry_face_a"]) in selected)
                != (int(record["geometry_face_b"]) in selected)
            )
        }
        # Boundary records are face pairs, so compare against the equivalent
        # full-reference edge ownership without scanning the selected set in
        # the production delta implementation.
        assert len(selected) == target_count
        assert actual_boundary == {
            target_count - 1
        }
        assert reference_boundary == {target_count - 1}
        assert result["progressive_range_full_boundary_scan"] is False
        assert result["progressive_range_confirm_geometry_reused"] is True
        assert result["progressive_range_new_face_dedupe_mode"] == "set"
        assert result["progressive_range_new_face_unique_count"] == delta_count
        assert result["progressive_range_selected_sort_count"] == 1
        assert result["progressive_range_confirm_sort_count"] == 1
        assert int(state.get("terminal_coverage_scan_count", 0)) == before_scans
        assert state["terminal_frontier"].get("delta_face_materializations") == 1
        return {
            "passed": True,
            "faces": count,
            "base_selected": base_count,
            "delta_faces_added": delta_count,
            "delta_selected": len(selected),
            "full_reference_seconds": float(reference_seconds),
            "delta_seconds": float(delta_seconds),
            "coverage_scan_reused": True,
            "full_boundary_scan": False,
            "frontier_seed_count": int(state["terminal_frontier"]["frontier_seed_count"]),
            "selected_sort_count": int(result.get("progressive_range_selected_sort_count", 0)),
            "confirm_sort_count": int(result.get("progressive_range_confirm_sort_count", 0)),
            "new_face_dedupe_mode": str(result.get("progressive_range_new_face_dedupe_mode", "")),
            "new_face_unique_count": int(result.get("progressive_range_new_face_unique_count", 0)),
            "confirm_materializations": int(state["terminal_frontier"].get("delta_face_materializations", 0)),
        }
    finally:
        module._FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = original_threshold


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

    original_sync = module.guided_ridge._guided_ridge_stable_frame
    original_steps = module.guided_ridge._guided_ridge_stable_frame_steps
    try:
        module.guided_ridge._guided_ridge_stable_frame = lambda *_args, **_kwargs: clone_frame()

        def stable_steps(*_args, **_kwargs):
            yield {"stage": "building guide frame", "done": 0, "total": 1}
            return clone_frame()

        module.guided_ridge._guided_ridge_stable_frame_steps = stable_steps
        sync_frame = module.guided_ridge._guided_ridge_width_frame(snapshot, guide, controls, control_normals)
        sync_result = module.guided_ridge._guided_ridge_width_frame_result(
            snapshot, guide, controls, control_normals, 0.8, frame=sync_frame
        )
        prepare_job = module.guided_ridge._guided_ridge_width_prepare_steps(
            snapshot,
            module.guided_ridge._guided_ridge_stable_frame_steps(
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
        cooperative_result = module.guided_ridge._guided_ridge_width_frame_result(
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
        module.guided_ridge._guided_ridge_stable_frame = original_sync
        module.guided_ridge._guided_ridge_stable_frame_steps = original_steps


def _guided_ridge_width_frame_rebuild_fixture(module):
    """Guide edits rebuild the frame cooperatively and Esc clears the job."""
    import numpy as np
    from mathutils import Vector

    original_state = module.runtime.guided_ridge_state
    original_steps = module.guided_ridge._guided_ridge_stable_frame_steps
    original_matches = module.guided_ridge._guided_ridge_context_matches
    original_result = module.guided_ridge._guided_ridge_width_frame_result
    original_overlay = module.guided_ridge._guided_ridge_overlay_tag
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
        module.runtime.guided_ridge_state = state
        module.guided_ridge._guided_ridge_stable_frame_steps = frame_job
        module.guided_ridge._guided_ridge_context_matches = lambda *_args: True
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        module.guided_ridge._guided_ridge_width_frame_result = lambda *_args, **_kwargs: {
            "frame": frame,
            "width": 0.5,
            "width_min": 0.1,
            "width_max": 1.0,
            "rails": {"left": [], "right": []},
            "projection_ok": True,
            "projection_required": False,
            "cache_key": (1,),
        }
        assert module.guided_ridge._guided_ridge_begin_width_frame_rebuild(context, state)
        assert state["phase"] == "width_prepare" and state["width_frame_job"] is not None
        assert module.guided_ridge._guided_ridge_process_width_frame_timer(context, state)
        assert state["phase"] == "width_prepare"
        for _ in range(8):
            if state["phase"] == "ready":
                break
            assert module.guided_ridge._guided_ridge_process_width_frame_timer(context, state)
        assert state["phase"] == "ready" and state["width_frame_job"] is None
        assert wm.removed

        state["phase"] = "ready"
        assert module.guided_ridge._guided_ridge_begin_width_frame_rebuild(context, state)
        module.runtime.guided_ridge_state = state
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            proxy, context, SimpleNamespace(type="ESC", value="PRESS")
        ) == {"CANCELLED"}
        assert module.runtime.guided_ridge_state is None and state.get("width_frame_job") is None
        return {"passed": True, "stage_tick_then_complete": True, "esc_cleanup": True}
    finally:
        module.guided_ridge._guided_ridge_stable_frame_steps = original_steps
        module.guided_ridge._guided_ridge_context_matches = original_matches
        module.guided_ridge._guided_ridge_width_frame_result = original_result
        module.guided_ridge._guided_ridge_overlay_tag = original_overlay
        module.runtime.guided_ridge_state = original_state


def _guided_ridge_live_snapshot_diagnostic(module):
    """Read-only regression check using the reported Hair/Face Set fixture."""
    import bpy
    import numpy as np
    from mathutils import Vector

    obj = bpy.data.objects.get("Mesh_00.Hair")
    if obj is None or obj.type != "MESH" or obj.data.attributes.get(".sculpt_face_set") is None:
        return {"skipped": "Mesh_00.Hair Face Set fixture unavailable"}
    snapshot, reason = module.guided_ridge._guided_ridge_prepare_snapshot_sync(obj, 1573375, 89)
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
    guide, project_reason = module.guided_ridge._guided_ridge_project_curve(snapshot, controls)
    assert project_reason is None and guide is not None
    candidate, info = module.guided_ridge._guided_ridge_exact_candidate(snapshot, guide, controls, normals)
    before = np.asarray(snapshot["coords_world"], dtype=np.float64)
    integrity = module.guided_ridge._guided_ridge_mesh_integrity(snapshot, before, candidate)
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

    original_job = module.guided_ridge._guided_ridge_exact_candidate_steps
    original_matches = module.guided_ridge._guided_ridge_context_matches
    original_overlay = module.guided_ridge._guided_ridge_overlay_tag
    original_state = module.runtime.guided_ridge_state
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
        module.guided_ridge._guided_ridge_exact_candidate_steps = job
        module.guided_ridge._guided_ridge_context_matches = lambda *_args: True
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        module.runtime.guided_ridge_state = state
        assert module.guided_ridge._guided_ridge_begin_compute(context, state)
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
        module.runtime.guided_ridge_state = state
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            proxy, context, SimpleNamespace(type="ESC", value="PRESS")
        ) == {"CANCELLED"}
        assert module.runtime.guided_ridge_state is None and removed
        return {"passed": True, "timer_removed": True, "esc_cleanup": True}
    finally:
        module.guided_ridge._guided_ridge_exact_candidate_steps = original_job
        module.guided_ridge._guided_ridge_context_matches = original_matches
        module.guided_ridge._guided_ridge_overlay_tag = original_overlay
        module.runtime.guided_ridge_state = original_state


def _guided_ridge_commit_fixture(module):
    """Verify the approved modal Enter path writes one bounded candidate."""
    import bpy
    import numpy as np
    from mathutils import Vector

    # The registered interactive operator is the Step 1 preview gate.  The
    # fixture below calls the retained legacy writer directly with a state
    # that is not an interactive preview session, so that internal reference
    # coverage remains available without exposing a UI write path.
    assert module.GUIDED_RIDGE_UI_PROTOTYPE is True
    assert module.GUIDED_RIDGE_CURVE_SCULPT_STEP1 is True
    assert "UNDO" not in module.VIEW3D_OT_mesh_focus_guided_ridge.bl_options
    mesh = bpy.data.meshes.new("_mfo_guided_ridge_commit_mesh")
    mesh.from_pydata(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        [],
        [(0, 1, 2)],
    )
    mesh.update()
    obj = bpy.data.objects.new("_mfo_guided_ridge_commit_object", mesh)
    original_state = module.runtime.guided_ridge_state
    original_last = module.runtime.guided_ridge_last_guide
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
        module.guided_ridge._guided_ridge_safety_reason = lambda _obj: None
        module.guided_ridge._guided_ridge_context_matches = lambda *_args: True
        module.guided_ridge._guided_ridge_surface_anchors = lambda *_args: ({"anchors": True}, None)
        module.guided_ridge._guided_ridge_build_last_guide = lambda *_args: ({"half_width": 0.25}, None)
        module.guided_ridge._guided_ridge_current_signature = lambda *_args: before.copy()
        module.guided_ridge._guided_ridge_mesh_integrity = lambda *_args: {
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
        module.runtime.guided_ridge_state = state
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
        assert module.runtime.guided_ridge_state is None
        return {
            "passed": True,
            "commit": "candidate_written",
            "changed_vertices": int(np.count_nonzero(np.linalg.norm(after_commit - before, axis=1) > 1.0e-12)),
            "undo_option": True,
        }
    finally:
        module.runtime.guided_ridge_state = original_state
        module.runtime.guided_ridge_last_guide = original_last
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
    original_state = module.runtime.guided_ridge_state
    original_last = module.runtime.guided_ridge_last_guide
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
        module.guided_ridge._guided_ridge_safety_reason = lambda _obj: None
        module.guided_ridge._guided_ridge_context_matches = lambda *_args: True
        module.guided_ridge._guided_ridge_surface_anchors = lambda *_args: ({"anchors": True}, None)
        module.guided_ridge._guided_ridge_build_last_guide = lambda *_args: ({"half_width": 0.25}, None)
        module.guided_ridge._guided_ridge_current_signature = lambda *_args: local_before.copy()
        module.guided_ridge._guided_ridge_mesh_integrity = lambda *_args: {
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
        module.runtime.guided_ridge_state = state
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
        module.runtime.guided_ridge_state = original_state
        module.runtime.guided_ridge_last_guide = original_last
        for name, value in patched.items():
            setattr(module, name, value)
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def _guided_ridge_mfo_navigation_fixture(module):
    """Verify MFO focus coexistence and Blender navigation pass-through."""
    from mathutils import Vector

    guided_class = module.VIEW3D_OT_mesh_focus_guided_ridge
    original = {
        "bpy": module.guided_ridge.bpy,
        "active_states": module.runtime.active_states,
        "guided_state": module.runtime.guided_ridge_state,
        "face_attr": module.guided_ridge._guided_ridge_face_set_attribute,
        "coordinate": module.guided_ridge._sculpt_cursor_region_coordinate,
        "raycast": module.guided_ridge._raycast_sculpt_face_set,
        "safety": module.guided_ridge._guided_ridge_safety_reason,
        "signature": module.guided_ridge._guided_ridge_context_signature,
        "ray": module.guided_ridge._guided_ridge_ray,
        "overlay": module.guided_ridge._guided_ridge_overlay_tag,
        "curve_step1": module.guided_ridge.GUIDED_RIDGE_CURVE_SCULPT_STEP1,
    }
    try:
        # This fixture targets MFO coexistence/navigation on the legacy
        # snapshot path; Step 1's surface-only entry is tested separately.
        module.guided_ridge.GUIDED_RIDGE_CURVE_SCULPT_STEP1 = False
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
        module.runtime.active_states = {9123: focus_state}
        module.runtime.guided_ridge_state = None
        module.guided_ridge._guided_ridge_face_set_attribute = lambda _obj: object()
        module.guided_ridge._sculpt_cursor_region_coordinate = lambda _context, _event: Vector((20.0, 30.0))
        module.guided_ridge._raycast_sculpt_face_set = lambda _context, _coord: (
            obj, 0, 7, Vector((0.0, 0.0, 0.0)), None
        )
        module.guided_ridge._guided_ridge_safety_reason = lambda _obj: None
        module.guided_ridge._guided_ridge_context_signature = lambda _context: {"area_pointer": 9123}
        module.guided_ridge._guided_ridge_ray = lambda _context, _coord: (
            Vector((0.0, 0.0, 2.0)), Vector((0.0, 0.0, -1.0))
        )
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        module.guided_ridge.bpy = SimpleNamespace(
            types=SimpleNamespace(
                SpaceView3D=SimpleNamespace(
                    draw_handler_add=lambda *_args: object(),
                    draw_handler_remove=lambda *_args: None,
                )
            ),
            context=SimpleNamespace(window_manager=context.window_manager),
        )
        assert module.guided_ridge._guided_ridge_conflict_reason(context) is None
        assert guided_class.poll(context)
        proxy = SimpleNamespace(report=lambda *_args, **_kwargs: None, poll=guided_class.poll)
        assert guided_class.invoke(proxy, context, SimpleNamespace(type="LEFTMOUSE", value="PRESS")) == {"RUNNING_MODAL"}
        assert module.runtime.active_states[9123] is focus_state
        assert guided_class.modal(proxy, context, SimpleNamespace(type="ESC", value="PRESS")) == {"CANCELLED"}
        assert module.runtime.active_states[9123] is focus_state

        nav_state = {"region": _Region(), "navigation_gizmo_active": False, "navigation_mmb_active": False}
        assert module.guided_ridge._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="MIDDLEMOUSE", value="PRESS"), nav_state
        )
        assert module.guided_ridge._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="MOUSEMOVE", value="NOTHING", mouse_region_x=500, mouse_region_y=300), nav_state
        )
        assert module.guided_ridge._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="MIDDLEMOUSE", value="RELEASE"), nav_state
        )
        assert not nav_state["navigation_mmb_active"]
        assert module.guided_ridge._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="NDOF_MOTION", value="NOTHING"), nav_state
        )
        assert module.guided_ridge._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="NUMPAD_1", value="PRESS"), nav_state
        )
        assert not module.guided_ridge._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="NUMPAD_ENTER", value="PRESS"), nav_state
        )
        context.space_data.show_gizmo = False
        assert not module.guided_ridge._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=1150, mouse_region_y=750), nav_state
        )
        context.space_data.show_gizmo = True
        # Top-right LMB is the Navigation Gizmo, while the ordinary guide LMB
        # remains owned by this modal.
        assert module.guided_ridge._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=1150, mouse_region_y=750), nav_state
        )
        assert nav_state["navigation_gizmo_active"]
        assert module.guided_ridge._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="MOUSEMOVE", value="NOTHING", mouse_region_x=1140, mouse_region_y=740), nav_state
        )
        assert module.guided_ridge._guided_ridge_navigation_passthrough(
            context, SimpleNamespace(type="LEFTMOUSE", value="RELEASE", mouse_region_x=1140, mouse_region_y=740), nav_state
        )
        assert not nav_state["navigation_gizmo_active"]
        assert not module.guided_ridge._guided_ridge_navigation_passthrough(
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
        module.guided_ridge.bpy = original["bpy"]
        module.runtime.active_states = original["active_states"]
        module.runtime.guided_ridge_state = original["guided_state"]
        module.guided_ridge._guided_ridge_face_set_attribute = original["face_attr"]
        module.guided_ridge._sculpt_cursor_region_coordinate = original["coordinate"]
        module.guided_ridge._raycast_sculpt_face_set = original["raycast"]
        module.guided_ridge._guided_ridge_safety_reason = original["safety"]
        module.guided_ridge._guided_ridge_context_signature = original["signature"]
        module.guided_ridge._guided_ridge_ray = original["ray"]
        module.guided_ridge._guided_ridge_overlay_tag = original["overlay"]
        module.guided_ridge.GUIDED_RIDGE_CURVE_SCULPT_STEP1 = original["curve_step1"]


def _smart_fill_vertex_paint_fixture(module):
    """Confirm Smart Fill writes only the sampled Vertex Paint region."""
    import bpy
    import numpy as np
    from mathutils import Vector

    mesh = bpy.data.meshes.new("_mfo_smart_fill_vertex_mesh")
    mesh.from_pydata(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ],
        [],
        [(0, 1, 2), (0, 2, 3), (0, 3, 4)],
    )
    mesh.update()
    obj = bpy.data.objects.new("_mfo_smart_fill_vertex_object", mesh)
    original_flood = module.smart_fill_preview._fill_preview_confirm_flood
    original_valid = module.smart_fill_preview._fill_preview_valid
    original_cancel = module.smart_fill_preview._fill_preview_cancel
    original_sample = module.guided_ridge._vertex_paint_sample_color
    try:
        corner = mesh.color_attributes.new(
            name="mfo_corner_color", type="BYTE_COLOR", domain="CORNER"
        )
        point = mesh.color_attributes.new(
            name="mfo_point_color", type="FLOAT_COLOR", domain="POINT"
        )
        # Make the clicked face red, the adjacent target face blue, and the
        # outside face green.  The two target faces share vertices, so the
        # POINT case also documents Blender's native shared-vertex semantics.
        red = (0.9, 0.1, 0.05, 1.0)
        blue = (0.05, 0.1, 0.9, 1.0)
        green = (0.1, 0.8, 0.2, 1.0)
        for index, item in enumerate(corner.data):
            item.color = red if index in {0, 1, 2} else blue if index in {3, 4, 5} else green
        for item in point.data:
            item.color = blue
        mesh.color_attributes.active_color_index = 0
        active, reason = module.guided_ridge._vertex_paint_active_color_attribute(obj)
        assert active is not None and active.name == corner.name and reason is None
        no_active, no_active_reason = module.guided_ridge._vertex_paint_active_color_attribute(
            SimpleNamespace(data=SimpleNamespace(color_attributes=SimpleNamespace(active_color=None)))
        )
        assert no_active is None and "active" in no_active_reason.lower()
        seed_location = Vector((0.2, 0.2, 0.0))
        seed = module.guided_ridge._vertex_paint_sample_color(obj, 0, seed_location, corner)
        assert seed is not None and np.allclose(seed, red, atol=1.0e-2)
        module.smart_fill_preview._fill_preview_confirm_flood = lambda _state, _result: np.asarray(
            [0, 1], dtype=np.int32
        )
        state = {
            "backend": "PAINT_VERTEX",
            "obj": obj,
            "seed_face": 0,
            "seed_local_location": tuple(seed_location),
            "seed_color": tuple(seed),
            "color_attribute_signature": module.guided_ridge._vertex_paint_color_signature(obj, corner),
        }
        written, reason = module.smart_fill_preview._fill_preview_write_vertex_paint(state, {})
        assert reason is None and written[0] == 3 and written[2] == "CORNER"
        corner_values = np.empty((len(corner.data), 4), dtype=np.float32)
        corner.data.foreach_get("color", corner_values.ravel())
        assert np.allclose(corner_values[:3], seed, atol=2.0e-5)
        assert np.allclose(corner_values[3:6], seed, atol=2.0e-5)
        outside_corner_before = np.array(corner_values[6:], copy=True)
        assert np.allclose(corner_values[6], green, atol=1.0e-2)
        assert np.array_equal(corner_values[6:], outside_corner_before)

        mesh.color_attributes.active_color_index = 1
        active, reason = module.guided_ridge._vertex_paint_active_color_attribute(obj)
        assert active is not None and active.name == point.name and reason is None
        for index, item in enumerate(point.data):
            item.color = red if index == 0 else blue
        point_seed = module.guided_ridge._vertex_paint_sample_color(
            obj, 0, seed_location, point
        )
        assert point_seed is not None
        point_state = dict(state)
        point_state.update(
            {
                "seed_color": point_seed,
                "color_attribute_signature": module.guided_ridge._vertex_paint_color_signature(obj, point),
            }
        )
        point_values = np.empty((len(point.data), 4), dtype=np.float32)
        point.data.foreach_get("color", point_values.ravel())
        outside_point_before = np.array(point_values[4], copy=True)
        point_written, reason = module.smart_fill_preview._fill_preview_write_vertex_paint(point_state, {})
        assert reason is None and point_written[0] > 0 and point_written[2] == "POINT"
        point_values = np.empty((len(point.data), 4), dtype=np.float32)
        point.data.foreach_get("color", point_values.ravel())
        assert np.allclose(point_values[:4], point_seed, atol=2.0e-5)
        assert np.allclose(point_values[4], blue, atol=2.0e-5)
        assert np.array_equal(point_values[4], outside_point_before)

        stale = dict(point_state)
        stale["color_attribute_signature"] = (0, "stale", "POINT", "FLOAT_COLOR", 4)
        stale_written, stale_reason = module.smart_fill_preview._fill_preview_write_vertex_paint(stale, {})
        assert stale_written is None and "changed" in stale_reason
        # Exercise the real modal confirm branch: Vertex Paint must not look
        # up Sculpt's .sculpt_face_set attribute before dispatching its writer.
        module.smart_fill_preview._fill_preview_valid = lambda _state, _context: True
        module.smart_fill_preview._fill_preview_cancel = lambda current, _reason: current.__setitem__("active", False)
        confirm_state = dict(point_state)
        confirm_state.update(
            {
                "active": True,
                "phase": "ready",
                "pending": False,
                "result": {"created_generation": 0},
                "generation": 0,
                "drawn_generation": 0,
            }
        )
        confirm_reports = []
        confirm_operator = SimpleNamespace(
            report=lambda level, message: confirm_reports.append((level, message))
        )
        confirm_result = module.VIEW3D_OT_mesh_focus_local_face_set_grow._finish_confirm(
            confirm_operator, SimpleNamespace(), confirm_state
        )
        assert confirm_result == {"FINISHED"}
        assert confirm_reports and "point" in confirm_reports[-1][1].lower()

        # Unmodified source geometry is supported, while a viewport modifier
        # would make the evaluated hit triangle diverge from source colors.
        compatible, compatibility_reason = module.guided_ridge._vertex_paint_geometry_compatibility(obj)
        assert compatible and compatibility_reason is None
        modifier = obj.modifiers.new("mfo_vertex_paint_test_subsurf", "SUBSURF")
        try:
            compatible, compatibility_reason = module.guided_ridge._vertex_paint_geometry_compatibility(obj)
            assert not compatible and "modifier" in compatibility_reason.lower()
        finally:
            obj.modifiers.remove(modifier)
        shape_key_obj = SimpleNamespace(
            modifiers=[],
            data=SimpleNamespace(
                shape_keys=SimpleNamespace(key_blocks=[object(), object()])
            ),
        )
        compatible, compatibility_reason = module.guided_ridge._vertex_paint_geometry_compatibility(
            shape_key_obj
        )
        assert not compatible and "shape" in compatibility_reason.lower()

        # Isolated invoke observation: an evaluated/source mismatch is
        # rejected before raycast, preview allocation, or any color write;
        # Ctrl remains visible to the strict-mode flag on this path.
        original_cursor_coordinate = module.guided_ridge._sculpt_cursor_region_coordinate
        original_geometry_compatibility = module.guided_ridge._vertex_paint_geometry_compatibility
        original_reg_cursor_coordinate = module.registration._sculpt_cursor_region_coordinate
        original_reg_geometry_compatibility = module.registration._vertex_paint_geometry_compatibility
        try:
            fake_cursor_coordinate = lambda _context, _event: Vector((2.0, 3.0))
            fake_geometry_compatibility = lambda _obj: (
                False, "injected evaluated/source mismatch"
            )
            module.guided_ridge._sculpt_cursor_region_coordinate = fake_cursor_coordinate
            module.guided_ridge._vertex_paint_geometry_compatibility = fake_geometry_compatibility
            module.registration._sculpt_cursor_region_coordinate = fake_cursor_coordinate
            module.registration._vertex_paint_geometry_compatibility = fake_geometry_compatibility
            invoke_reports = []
            invoke_proxy = SimpleNamespace(
                poll=module.VIEW3D_OT_mesh_focus_local_face_set_grow.poll,
                report=lambda level, message: invoke_reports.append((level, message)),
                strict_mode=False,
            )
            invoke_context = SimpleNamespace(
                area=SimpleNamespace(type="VIEW_3D"),
                region=SimpleNamespace(type="WINDOW"),
                space_data=SimpleNamespace(),
                mode="PAINT_VERTEX",
                active_object=SimpleNamespace(type="MESH"),
            )
            invoke_result = module.VIEW3D_OT_mesh_focus_local_face_set_grow.invoke(
                invoke_proxy,
                invoke_context,
                SimpleNamespace(type="E", value="PRESS", ctrl=True),
            )
            assert invoke_result == {"CANCELLED"}
            assert invoke_proxy.strict_mode is True
            assert invoke_reports and "mismatch" in invoke_reports[-1][1]
        finally:
            module.guided_ridge._sculpt_cursor_region_coordinate = original_cursor_coordinate
            module.guided_ridge._vertex_paint_geometry_compatibility = original_geometry_compatibility

        # Exercise the n-gon fallback path: a point outside all loop triangles
        # samples the deterministic average of the polygon's corner colors.
        ngon_mesh = bpy.data.meshes.new("_mfo_smart_fill_vertex_ngon_mesh")
        ngon_obj = None
        try:
            ngon_mesh.from_pydata(
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.8, 0.0),
                 (0.5, 1.3, 0.0), (-0.2, 0.7, 0.0)],
                [], [(0, 1, 2, 3, 4)],
            )
            ngon_mesh.update()
            ngon_obj = bpy.data.objects.new("_mfo_smart_fill_vertex_ngon_object", ngon_mesh)
            ngon = ngon_mesh.color_attributes.new(
                name="mfo_ngon_color", type="FLOAT_COLOR", domain="CORNER"
            )
            ngon_colors = np.asarray(
                [(0.9, 0.1, 0.05, 1.0), (0.1, 0.8, 0.2, 1.0),
                 (0.2, 0.3, 0.9, 1.0), (0.8, 0.4, 0.1, 1.0),
                 (0.4, 0.1, 0.7, 1.0)],
                dtype=np.float32,
            )
            ngon.data.foreach_set("color", ngon_colors.ravel())
            ngon_seed = module.guided_ridge._vertex_paint_sample_color(
                ngon_obj, 0, Vector((10.0, 10.0, 0.0)), ngon
            )
            assert ngon_seed is not None
            assert np.allclose(ngon_seed, np.mean(ngon_colors, axis=0), atol=1.0e-5)
        finally:
            if ngon_obj is not None:
                bpy.data.objects.remove(ngon_obj, do_unlink=True)
            bpy.data.meshes.remove(ngon_mesh)

        # Inject a failure after the first foreach_set (the simulated write)
        # and verify that the writer restores the complete original array.
        class _FailingColorData:
            def __init__(self, values):
                self.values = np.array(values, dtype=np.float32, copy=True)
                self.calls = 0

            def __len__(self):
                return len(self.values)

            def foreach_get(self, _name, target):
                target[:] = self.values.ravel()

            def foreach_set(self, _name, source):
                self.calls += 1
                self.values = np.array(source, dtype=np.float32, copy=True).reshape((-1, 4))
                if self.calls == 1:
                    raise RuntimeError("injected foreach_set failure")

        original_failure_values = np.asarray(
            [(0.2, 0.3, 0.4, 1.0), (0.7, 0.6, 0.5, 1.0), (0.1, 0.2, 0.3, 1.0)],
            dtype=np.float32,
        )
        failing_data = _FailingColorData(original_failure_values)
        failing_attribute = SimpleNamespace(
            name="mfo_failure_color",
            domain="POINT",
            data_type="FLOAT_COLOR",
            data=failing_data,
        )
        failing_attrs = SimpleNamespace(active_color=failing_attribute)
        failing_mesh = SimpleNamespace(
            color_attributes=failing_attrs,
            polygons=[SimpleNamespace(loop_indices=[0, 1, 2])],
            loops=[SimpleNamespace(vertex_index=0), SimpleNamespace(vertex_index=1),
                   SimpleNamespace(vertex_index=2)],
            as_pointer=lambda: 9137,
            update=lambda: None,
        )
        failing_obj = SimpleNamespace(data=failing_mesh, modifiers=[])
        original_preview_sample_color = module.smart_fill_preview._vertex_paint_sample_color
        fake_failure_sample_color = lambda *_args: tuple(original_failure_values[0])
        module.guided_ridge._vertex_paint_sample_color = fake_failure_sample_color
        module.smart_fill_preview._vertex_paint_sample_color = fake_failure_sample_color
        failure_state = {
            "obj": failing_obj,
            "seed_face": 0,
            "seed_local_location": (0.0, 0.0, 0.0),
            "seed_color": tuple(original_failure_values[0]),
            "color_attribute_signature": module.guided_ridge._vertex_paint_color_signature(
                failing_obj, failing_attribute
            ),
        }
        module.smart_fill_preview._fill_preview_confirm_flood = lambda _state, _result: np.asarray(
            [0], dtype=np.int32
        )
        failure_written, failure_reason = module.smart_fill_preview._fill_preview_write_vertex_paint(
            failure_state, {}
        )
        assert failure_written is None and "restored" in failure_reason
        assert failing_data.calls == 2
        assert np.array_equal(failing_data.values, original_failure_values)
        return {
            "passed": True,
            "corner_changed": int(written[0]),
            "point_changed": int(point_written[0]),
            "outside_corner_exact": True,
            "outside_point_exact": True,
            "shared_point_semantics": True,
            "stale_attribute_rejected": True,
            "modifier_rejected": True,
            "shape_key_rejected": True,
            "no_active_attribute_rejected": True,
            "invoke_modifier_guard": True,
            "ngon_sampling": True,
            "modal_confirm_finished": True,
        }
    finally:
        module.smart_fill_preview._fill_preview_confirm_flood = original_flood
        module.smart_fill_preview._fill_preview_valid = original_valid
        module.smart_fill_preview._fill_preview_cancel = original_cancel
        module.guided_ridge._vertex_paint_sample_color = original_sample
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def _guided_ridge_curve_preview_fixture(module):
    """Step 1 route/curve lifecycle stays entirely outside mesh writes."""
    from mathutils import Vector

    original_state = module.runtime.guided_ridge_state
    original_signature = module.guided_ridge._guided_ridge_context_signature
    original_navigation = module.guided_ridge._guided_ridge_navigation_passthrough
    original_overlay = module.guided_ridge._guided_ridge_overlay_tag
    original_width_preview = module.guided_ridge._guided_ridge_update_width_preview
    original_projection = module.guided_ridge._guided_ridge_project_3d_to_region_2d
    original_smoothing = module.runtime.guided_ridge_curve_last_smoothing
    reports = []
    before_route = [
        Vector((0.0, 0.0, 0.0)),
        Vector((1.0e-8, 0.0, 0.0)),  # short duplicate-like segment
        Vector((0.8, 0.12, 0.0)),
        Vector((1.6, 0.0, 0.0)),
    ]
    context_signature = {
        "area_pointer": 1, "region_pointer": 2, "window_pointer": 3,
        "space_pointer": 4, "scene_pointer": 5, "active_object_pointer": 6,
        "mesh_pointer": 7, "mode": "SCULPT", "space_type": "VIEW_3D",
    }

    class _WM:
        def event_timer_add(self, *_args, **_kwargs):
            return object()

        def event_timer_remove(self, _timer):
            return None

    context = SimpleNamespace(
        window_manager=_WM(), window=object(),
        region=SimpleNamespace(width=100, height=100),
        space_data=SimpleNamespace(region_3d=object()),
    )
    try:
        module.guided_ridge._guided_ridge_context_signature = lambda _context: dict(context_signature)
        module.guided_ridge._guided_ridge_navigation_passthrough = lambda *_args: False
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        rebuild_calls = []
        module.guided_ridge._guided_ridge_update_width_preview = lambda _state: rebuild_calls.append(True)
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = (
            lambda _region, _region_3d, point: SimpleNamespace(
                x=50.0 + float(point.x) * 20.0,
                y=50.0 + float(point.y) * 20.0,
            )
        )
        original_guide = [Vector((0.0, 0.0, 0.0)), Vector((0.7, 0.2, 0.0)), Vector((1.6, 0.0, 0.0))]
        original_frame = object()
        original_width_result = object()
        original_rails = {"left": [Vector((0, 1, 0))], "right": [Vector((0, -1, 0))]}
        original_cache = object()
        state = {
            "active": True,
            "phase": "ready",
            "operator": SimpleNamespace(report=lambda _kind, message: reports.append(message)),
            "apply_rect": (10000.0, 10000.0, 10001.0, 10001.0),
            "cancel_rect": (10000.0, 10000.0, 10001.0, 10001.0),
            "controls": [point.copy() for point in before_route],
            "guide": original_guide,
            "width_frame": original_frame,
            "width_frame_result": original_width_result,
            "width_rails": original_rails,
            "width_cache_serial": 17,
            "width_frame_job": None,
            "cache": original_cache,
            "context_signature": dict(context_signature),
            "curve_preview_session": True,
            "curve_smoothing": 0.0,
            "curve_preview_generation": 0,
            "timer": None,
            "window_manager": context.window_manager,
            "snapshot": {},
        }
        module.runtime.guided_ridge_state = state
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            state["operator"], context, SimpleNamespace(type="ENTER", value="PRESS")
        ) == {"RUNNING_MODAL"}
        assert state["phase"] == "curve_preview"
        assert len(state["curve_route"]) == 3
        assert state["curve_preview"][0] == before_route[0]
        assert state["curve_preview"][-1] == before_route[-1]
        assert len(state.get("curve_screen_raw") or ()) >= 2
        assert len(state.get("curve_screen_preview") or ()) >= 2
        assert state.get("curve_screen_signature") is not None
        generated_at_zero = [point.copy() for point in state["curve_preview"]]
        assert state["curve_smoothing"] == module.GUIDED_RIDGE_CURVE_DEFAULT_SMOOTHING
        smoothing_before_wheel = state["curve_smoothing"]
        coordinates_before = [tuple(point) for point in state["controls"]]
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            state["operator"], context, SimpleNamespace(type="WHEELUPMOUSE", value="PRESS", shift=False)
        ) == {"RUNNING_MODAL"}
        generated_at_ten = [point.copy() for point in state["curve_preview"]]
        assert state["curve_smoothing"] == smoothing_before_wheel + 10.0
        assert any((a - b).length > 1.0e-9 for a, b in zip(generated_at_zero, generated_at_ten))
        assert [tuple(point) for point in state["controls"]] == coordinates_before
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            state["operator"], context, SimpleNamespace(type="ENTER", value="PRESS")
        ) == {"RUNNING_MODAL"}
        assert state["phase"] == "curve_preview"
        assert [tuple(point) for point in state["controls"]] == coordinates_before
        control_count_before_preview_lmb = len(state["controls"])
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            state["operator"], context, SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_x=50, mouse_y=50)
        ) == {"RUNNING_MODAL"}
        assert len(state["controls"]) == control_count_before_preview_lmb
        state["phase"] = "curve_sculpt"
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            state["operator"], context, SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_x=50, mouse_y=50)
        ) == {"RUNNING_MODAL"}
        assert len(state["controls"]) == control_count_before_preview_lmb
        # A projection failure is a warning-only preview state, never a write.
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = lambda *_args: SimpleNamespace(x=150.0, y=50.0)
        # In Blender this corresponds to a changed projection/view signature;
        # the lightweight fake must make that change explicit so the cache is
        # not intentionally reused.
        context_signature["view_marker"] = 1
        assert module.guided_ridge._guided_ridge_curve_refresh_projection(context, state)
        assert state["curve_projection_warning"]
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = lambda *_args: object()
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            state["operator"], context, SimpleNamespace(type="BACK_SPACE", value="PRESS")
        ) == {"RUNNING_MODAL"}
        assert state["phase"] == "ready"
        assert [tuple(point) for point in state["controls"]] == coordinates_before
        assert state["guide"] is original_guide
        assert state["width_frame"] is original_frame
        assert state["width_frame_result"] is original_width_result
        assert state["width_rails"] is original_rails
        assert state["cache"] is original_cache
        assert state["width_cache_serial"] == 17
        assert not rebuild_calls
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            state["operator"], context, SimpleNamespace(type="ESC", value="PRESS")
        ) == {"CANCELLED"}
        return {
            "passed": True,
            "phase_roundtrip": True,
            "raw_curve_distinct": True,
            "smoothing_step": 10.0,
            "endpoints_preserved": True,
            "mesh_unchanged": True,
            "projection_warning": True,
            "cleanup": module.runtime.guided_ridge_state is None,
        }
    finally:
        module.runtime.guided_ridge_state = original_state
        module.guided_ridge._guided_ridge_context_signature = original_signature
        module.guided_ridge._guided_ridge_navigation_passthrough = original_navigation
        module.guided_ridge._guided_ridge_overlay_tag = original_overlay
        module.guided_ridge._guided_ridge_update_width_preview = original_width_preview
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = original_projection
        module.runtime.guided_ridge_curve_last_smoothing = original_smoothing


def _smart_fill_modal_safety_fixture(module):
    """Exercise modal termination without touching the current scene.

    Native mode switches and sculpt/paint operators belong to the isolated
    Blender runner only.  This contract uses the real Smart Fill modal method
    with a disposable context/state to prove that a context change returns a
    terminal result and synchronously clears the canonical runtime handles.
    """
    import bpy

    runtime = module.runtime
    registration = module.registration
    operator = SimpleNamespace()

    class _Ptr:
        def __init__(self, value):
            self.value = int(value)

        def as_pointer(self):
            return self.value

    area = _Ptr(901)
    area.tag_redraw = lambda: None
    region = _Ptr(902)
    window = _Ptr(903)
    mesh = _Ptr(905)
    obj = _Ptr(904)
    obj.data = mesh
    context = SimpleNamespace(
        area=area,
        region=region,
        window=window,
        active_object=obj,
        mode="PAINT_VERTEX",
    )
    state = {
        "active": True,
        "operator": operator,
        "session_id": 987654,
        "area": area,
        "area_key": 901,
        "region_key": 902,
        "window_key": 903,
        "obj": obj,
        "obj_pointer": 904,
        "mesh_pointer": 905,
        "mode": "PAINT_VERTEX",
        "timer": None,
        "metrics": {},
    }
    old_state = runtime.fill_preview_state
    old_owner = (
        runtime.fill_preview_modal_operator,
        runtime.fill_preview_modal_session,
        runtime.fill_preview_modal_context,
        runtime.fill_preview_modal_handler_live,
        runtime.fill_preview_modal_cancel_requested,
        runtime.fill_preview_modal_cancel_reason,
    )
    old_registered = runtime.is_registered
    old_reload_blocked = runtime.fill_preview_reload_blocked
    old_reload_warning = runtime.fill_preview_reload_warning
    runtime.fill_preview_state = state
    runtime.fill_preview_modal_operator = operator
    runtime.fill_preview_modal_session = 987654
    runtime.fill_preview_modal_context = (903, 901, 902, 904, 905)
    runtime.fill_preview_modal_handler_live = True
    runtime.fill_preview_modal_cancel_requested = False
    runtime.fill_preview_modal_cancel_reason = ""
    try:
        context.mode = "OBJECT"
        result = registration._sfsf_modal_uninstrumented(
            operator,
            context,
            SimpleNamespace(type="MOUSEMOVE", value="NOTHING"),
        )
        assert result == {"CANCELLED"}
        assert runtime.fill_preview_state is None
        assert runtime.fill_preview_modal_operator is None
        assert runtime.fill_preview_modal_session is None
        assert runtime.fill_preview_modal_context is None

        # An external update cleans the visible state but leaves the modal
        # owner live until its next modal event returns CANCELLED.
        runtime.fill_preview_state = dict(state, active=True)
        runtime.fill_preview_modal_operator = operator
        runtime.fill_preview_modal_session = 987654
        runtime.fill_preview_modal_handler_live = True
        registration._fill_preview_request_cancel(runtime.fill_preview_state, "depsgraph")
        assert runtime.fill_preview_state is None
        assert runtime.fill_preview_modal_operator is operator
        assert runtime.fill_preview_modal_handler_live is True
        terminal = registration._sfsf_modal_uninstrumented(
            operator,
            context,
            SimpleNamespace(type="MOUSEMOVE", value="NOTHING"),
        )
        assert terminal == {"CANCELLED"}
        assert runtime.fill_preview_modal_operator is None
        assert runtime.fill_preview_modal_handler_live is False

        # Normal unregister follows Blender's teardown contract.  A live
        # Python modal owner is requested to terminate, visible state is
        # cleaned, and the registration flag is cleared; root hot reload
        # preflight is tested separately and refuses before child reload.
        runtime.fill_preview_state = dict(state, active=True)
        runtime.fill_preview_modal_operator = operator
        runtime.fill_preview_modal_session = 987654
        runtime.fill_preview_modal_handler_live = True
        runtime.is_registered = True
        result = registration.unregister()
        assert result is None
        assert runtime.fill_preview_state is None
        assert runtime.fill_preview_reload_blocked is False
        assert runtime.fill_preview_reload_warning == ""
        assert runtime.is_registered is False
        return {
            "passed": True,
            "context_change_terminal": True,
            "unregister_teardown": True,
            "state_cleanup": True,
            "modal_handler_remove_available": bool(
                hasattr(getattr(bpy, "types", None), "WindowManager")
                and hasattr(bpy.types.WindowManager, "modal_handler_remove")
            ),
        }
    finally:
        runtime.fill_preview_state = old_state
        (
            runtime.fill_preview_modal_operator,
            runtime.fill_preview_modal_session,
            runtime.fill_preview_modal_context,
            runtime.fill_preview_modal_handler_live,
            runtime.fill_preview_modal_cancel_requested,
            runtime.fill_preview_modal_cancel_reason,
        ) = old_owner
        runtime.fill_preview_reload_blocked = old_reload_blocked
        runtime.fill_preview_reload_warning = old_reload_warning
        runtime.is_registered = old_registered


def _guided_ridge_native_asset_fixture(module):
    """Exercise Blender 5.2 asset activation and PaintCurve RNA on temp data."""
    import bpy

    service = module.guided_ridge_curve_sculpt
    scene = bpy.context.scene
    original_object = bpy.context.active_object
    original_mode = getattr(original_object, "mode", None) if original_object else None
    original_selected = tuple(bpy.context.selected_objects)
    sculpt = scene.tool_settings.sculpt
    reference = sculpt.brush_asset_reference
    original_reference = {
        key: getattr(reference, key, "")
        for key in ("asset_library_type", "asset_library_identifier", "relative_asset_identifier")
    }
    mesh = bpy.data.meshes.new("MFO Native Curve Test Mesh")
    mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
    obj = bpy.data.objects.new("MFO Native Curve Test Object", mesh)
    scene.collection.objects.link(obj)
    paint_curve = None
    state = {
        "active": True,
        "phase": "curve_preview",
        "obj": obj,
        "preview_context": SimpleNamespace(scene=scene),
        "curve_sculpt_session_token": "native-test-token",
        "curve_sculpt_owned_brushes": {},
        "curve_sculpt_owned_curves": {},
        "curve_sculpt_managed_brushes": {},
        "curve_sculpt_managed_brush_settings": {},
        "curve_sculpt_managed_preexisting_curves": {},
        "curve_sculpt_brushes_by_mode": {},
        "curve_sculpt_created_brushes": [],
        "curve_sculpt_created_curves": [],
        "curve_sculpt_restore": {
            "brush": sculpt.brush,
            "stroke_method": getattr(sculpt.brush, "stroke_method", None),
            "paint_curve": getattr(sculpt.brush, "paint_curve", None),
            "strength": getattr(sculpt.brush, "strength", None),
            "size": getattr(sculpt.brush, "size", None),
            "sculpt_tool": getattr(sculpt.brush, "sculpt_tool", None),
        },
        "curve_sculpt_restore_asset_reference": original_reference,
        "operator": SimpleNamespace(report=lambda *_args: None),
    }
    try:
        if original_object is not None and original_object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for item in bpy.context.selected_objects:
            item.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="SCULPT")
        brush = service._activate_mfo_asset(bpy.context, state, "RIDGE")
        assert brush.get("mfo_local_feature_marker") == service._LOCAL_ASSET_MARKER
        assert brush.sculpt_brush_type == "PINCH"
        assert sculpt.brush_asset_reference.asset_library_type == "CUSTOM"
        assert sculpt.brush_asset_reference.asset_library_identifier == service._LOCAL_ASSET_LIBRARY
        result = bpy.ops.paintcurve.new()
        assert "FINISHED" in result
        paint_curve = brush.paint_curve
        assert paint_curve is not None
        native_points = ((100, 100), (140, 120), (180, 100))
        for point in native_points:
            assert "FINISHED" in bpy.ops.paintcurve.add_point(location=point)
        state["curve_sculpt_native_points"] = native_points
        draw_result = bpy.ops.paintcurve.draw()
        assert set(draw_result).intersection({"FINISHED", "RUNNING_MODAL", "PASS_THROUGH"})
        state["curve_sculpt_paint_curve"] = paint_curve
        service._mark_owned(paint_curve, state, "curve")
        assert state["curve_sculpt_native_points"] == native_points
        return {
            "passed": True,
            "asset_activate": True,
            "mfo_marker": True,
            "pinch_tool": True,
            "paintcurve_new_add_draw": True,
            "native_points_integer_region": True,
            "mesh_write": False,
        }
    finally:
        try:
            service.cleanup(state)
        finally:
            if obj.name in bpy.data.objects:
                if obj.mode != "OBJECT":
                    bpy.ops.object.mode_set(mode="OBJECT")
                bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            for item in bpy.context.selected_objects:
                item.select_set(False)
            if original_object is not None and original_object.name in bpy.data.objects:
                original_object.select_set(True)
                bpy.context.view_layer.objects.active = original_object
                if original_mode and original_object.mode != original_mode:
                    bpy.ops.object.mode_set(mode=original_mode)
            for item in original_selected:
                if item.name in bpy.data.objects:
                    item.select_set(True)


def _guided_ridge_native_apply_transaction_fixture(module):
    """Run the native apply bridge in a temporary real View3D context.

    The MCP test runner has no implicit area/region, so this fixture explicitly
    overrides a real VIEW_3D WINDOW region.  It verifies public asset activation,
    that the readonly Sculpt.brush pointer is not assigned, and exact asset
    reference restoration after the native Paint Curve bridge is cleaned up.
    Mesh deformation itself remains event/modal-owned by Blender and is reported
    separately by the live smoke test when the draw operator is PASS_THROUGH.
    """
    import bpy

    service = module.guided_ridge_curve_sculpt
    windows = list(bpy.context.window_manager.windows)
    area = next(
        (candidate for window in windows for candidate in window.screen.areas if candidate.type == "VIEW_3D"),
        None,
    )
    if area is None:
        return {"skipped": "no VIEW_3D area available"}
    window = next(
        window
        for window in windows
        if any(candidate.as_pointer() == area.as_pointer() for candidate in window.screen.areas)
    )
    region = next((candidate for candidate in area.regions if candidate.type == "WINDOW"), None)
    if region is None:
        return {"skipped": "no VIEW_3D WINDOW region available"}

    scene = bpy.context.scene
    original_object = bpy.context.view_layer.objects.active
    original_selected = tuple(bpy.context.selected_objects)
    original_mode = bpy.context.mode
    temp_object = None
    temp_mesh = None
    state = None
    reports = []
    previous_runtime_state = None
    signature_equal = False
    delayed_callback_seen = False
    external_update_cancelled = False
    depsgraph_handler_events = []
    guided_core = None

    def _native_depsgraph_handler(scene_arg, depsgraph):
        """Observe the real post-update callback, preserving the native graph."""
        depsgraph_handler_events.append(len(tuple(depsgraph.updates)))
        guided_core._on_guided_ridge_depsgraph_update(scene_arg, depsgraph)

    def asset_ref(sculpt):
        reference = getattr(sculpt, "brush_asset_reference", None)
        if reference is None:
            return None
        return tuple(
            getattr(reference, key, "")
            for key in ("asset_library_type", "asset_library_identifier", "relative_asset_identifier")
        )

    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            if bpy.context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            temp_mesh = bpy.data.meshes.new("MFO Native Apply Transaction Mesh")
            temp_mesh.from_pydata(
                [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)],
                [],
                [(0, 1, 2, 3)],
            )
            temp_mesh.update()
            temp_object = bpy.data.objects.new("MFO Native Apply Transaction Object", temp_mesh)
            scene.collection.objects.link(temp_object)
            bpy.ops.object.select_all(action="DESELECT")
            temp_object.select_set(True)
            bpy.context.view_layer.objects.active = temp_object
            bpy.ops.object.mode_set(mode="SCULPT")
            sculpt = scene.tool_settings.sculpt
            before_reference = asset_ref(sculpt)
            state = {
                "active": True,
                "phase": "curve_preview",
                "obj": temp_object,
                "preview_context": bpy.context,
                "operator": SimpleNamespace(report=lambda _kind, message: reports.append(message)),
                "curve_screen_preview": [
                    (region.width * 0.35, region.height * 0.45),
                    (region.width * 0.50, region.height * 0.55),
                    (region.width * 0.65, region.height * 0.45),
                ],
                "curve_screen_cache_status": "valid",
                "curve_screen_signature": ("native-transaction", id(temp_object)),
            }
            assert service.apply(bpy.context, state, "RIDGE")
            activated_reference = asset_ref(sculpt)
            assert activated_reference is not None
            assert activated_reference[0] == "CUSTOM"
            assert activated_reference[1] == service._LOCAL_ASSET_LIBRARY
            assert state.get("curve_sculpt_native_asset") is True
            assert state.get("phase") == "curve_sculpt"
            previous_runtime_state = module.runtime.guided_ridge_state
            module.runtime.guided_ridge_state = state
            signature_equal = (
                state.get("curve_sculpt_expected_mesh_signature")
                == service.mesh_revision_signature(state)
            )
            guided_core = importlib.import_module(
                f"{module.__name__}.guided_ridge.core"
            )
            # Register the actual persistent-style callback before producing
            # updates.  Calling the callback with evaluated_depsgraph_get()
            # directly is insufficient in Blender 5.2 because its consumed
            # ``updates`` collection is empty outside the post-update hook.
            bpy.app.handlers.depsgraph_update_post.append(_native_depsgraph_handler)
            # First update: object-only, signature unchanged.  This is the
            # delayed self-update consumed from the finite generation budget.
            temp_object.location.x += 0.125
            bpy.context.view_layer.update()
            # Second update: mesh revision changes, so the same native callback
            # must cancel instead of swallowing an external edit.
            temp_mesh.vertices[0].co.x += 0.125
            temp_mesh.update()
            bpy.context.view_layer.update()
            delayed_callback_seen = bool(depsgraph_handler_events)
            external_update_cancelled = not state.get("active")
            assert len(depsgraph_handler_events) >= 2
            service.cleanup(state)
            restored_reference = asset_ref(sculpt)
            assert restored_reference == before_reference
            assert signature_equal
            assert delayed_callback_seen
            assert external_update_cancelled
            return {
                "passed": True,
                "view3d_override": True,
                "asset_activated": activated_reference,
                "asset_restored": restored_reference,
                "native_asset_path": True,
                "mesh_write": False,
                "canonical_signature_equal": signature_equal,
                "delayed_depsgraph_callback": delayed_callback_seen,
                "depsgraph_handler_events": tuple(depsgraph_handler_events),
                "external_update_cancelled": external_update_cancelled,
                "reports": reports,
            }
    finally:
        try:
            if _native_depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
                bpy.app.handlers.depsgraph_update_post.remove(_native_depsgraph_handler)
        finally:
            try:
                if state is not None and state.get("active"):
                    service.cleanup(state)
            finally:
                module.runtime.guided_ridge_state = previous_runtime_state
                with bpy.context.temp_override(window=window, area=area, region=region):
                    if temp_object is not None and bpy.context.mode != "OBJECT":
                        bpy.ops.object.mode_set(mode="OBJECT")
                    if temp_object is not None and temp_object.name in bpy.data.objects:
                        bpy.data.objects.remove(temp_object, do_unlink=True)
                    if temp_mesh is not None and temp_mesh.users == 0:
                        bpy.data.meshes.remove(temp_mesh)
                    bpy.ops.object.select_all(action="DESELECT")
                    for item in original_selected:
                        if item.name in bpy.data.objects:
                            item.select_set(True)
                    bpy.context.view_layer.objects.active = original_object
                    if original_mode != "OBJECT" and original_object is not None:
                        bpy.ops.object.mode_set(mode=original_mode)


def _guided_ridge_curve_sculpt_legacy_fixture(module):
    """Native Curve Sculpt bridge is explicit, repeatable, and transactional."""
    from contextlib import nullcontext

    service = module.guided_ridge_curve_sculpt
    original_bpy = service.bpy
    calls = []
    removed = []
    draw_failure = {"value": False}

    class _FakeOp:
        def __init__(self, name):
            self.name = name

        def poll(self):
            return True

        def __call__(self, **kwargs):
            calls.append((self.name, kwargs))
            if self.name == "new":
                fake_sculpt.brush.paint_curve = fake_paint_curve
            if self.name == "draw" and draw_failure["value"]:
                raise RuntimeError("injected native draw failure")
            return {"FINISHED"}

    class _FakePaintCurveOps:
        new = _FakeOp("new")
        add_point = _FakeOp("add_point")
        draw = _FakeOp("draw")

    class _FakeBrush:
        def __init__(self, name):
            self.name = name
            self.sculpt_tool = None
            self.stroke_method = "DOTS"
            self.paint_curve = None
            self.strength = 0.2
            self.size = 20

    brushes = {}

    class _Brushes:
        def get(self, name):
            return brushes.get(name)

        def new(self, name, mode="SCULPT"):
            brush = _FakeBrush(name)
            brushes[name] = brush
            return brush

    fake_paint_curve = SimpleNamespace(users=0)
    original_brush = _FakeBrush("User Brush")
    original_brush.stroke_method = "LINE"
    # A same-name user asset must never be selected or mutated.  The service
    # creates a session-owned unique brush instead.
    user_named_curve = SimpleNamespace(users=1, marker="user-curve")
    user_named_brush = _FakeBrush(service._RIDGE_BRUSH_NAME)
    user_named_brush.paint_curve = user_named_curve
    user_named_brush.stroke_method = "AIRBRUSH"
    brushes[user_named_brush.name] = user_named_brush
    fake_sculpt = SimpleNamespace(brush=original_brush)
    context = SimpleNamespace(
        scene=SimpleNamespace(tool_settings=SimpleNamespace(sculpt=fake_sculpt)),
        window=object(), area=object(), region=object(),
        temp_override=lambda **_kwargs: nullcontext(),
    )
    fake_bpy = SimpleNamespace(
        data=SimpleNamespace(
            brushes=_Brushes(),
            batch_remove=lambda ids: removed.extend(ids),
        ),
        ops=SimpleNamespace(paintcurve=_FakePaintCurveOps()),
        context=SimpleNamespace(preferences=SimpleNamespace(addons={})),
    )
    reports = []
    state = {
        "active": True,
        "phase": "curve_preview",
        "operator": SimpleNamespace(report=lambda _kind, message: reports.append(message)),
        "curve_screen_preview": [(10.0, 10.0), (20.0, 14.0), (30.0, 10.0)],
        "curve_screen_cache_status": "valid",
        "curve_screen_signature": (1, 2, 3),
        "preview_context": context,
    }
    try:
        service.bpy = fake_bpy
        assert service.apply(context, state, "RIDGE")
        assert state["phase"] == "curve_sculpt"
        assert state["curve_sculpt_ridge_applications"] == 1
        assert any(name == "new" for name, _kwargs in calls)
        assert any(name == "draw" for name, _kwargs in calls)
        first_call_count = len(calls)
        assert service.apply(context, state, "GROOVE")
        assert state["curve_sculpt_groove_applications"] == 1
        assert len([name for name, _kwargs in calls if name == "new"]) == 1
        assert len(calls) > first_call_count
        service.cleanup(state)
        assert fake_sculpt.brush is original_brush
        assert original_brush.stroke_method == "LINE"
        assert fake_paint_curve in removed
        assert user_named_brush.paint_curve is user_named_curve
        assert user_named_brush.stroke_method == "AIRBRUSH"
        assert user_named_curve not in removed
        # A failure after paintcurve.new must detach and remove the implicitly
        # attached curve as well; no failed apply may leave a temporary
        # datablock or an active Curve Sculpt session behind.
        failed_state = {
            "active": True,
            "phase": "curve_preview",
            "operator": SimpleNamespace(report=lambda _kind, message: reports.append(message)),
            "curve_screen_preview": [(10.0, 10.0), (20.0, 14.0), (30.0, 10.0)],
            "curve_screen_cache_status": "valid",
            "curve_screen_signature": (4, 5, 6),
            "preview_context": context,
        }
        removed_before_failure = len(removed)
        draw_failure["value"] = True
        assert not service.apply(context, failed_state, "RIDGE")
        assert failed_state.get("curve_sculpt_active") is False
        assert failed_state.get("curve_sculpt_paint_curve") is None
        assert len(removed) > removed_before_failure
        assert fake_sculpt.brush is original_brush
        assert all(
            getattr(brush, "paint_curve", None) is None
            for brush in brushes.values()
            if brush is not user_named_brush
        )
        assert user_named_brush.paint_curve is user_named_curve
        return {
            "passed": True,
            "ridge_then_groove": True,
            "repeat_reuses_paint_curve": True,
            "brush_restored": True,
            "temporary_curve_removed": True,
            "failed_draw_cleanup": True,
            "mesh_write_owned_by_native_operator": True,
        }
    finally:
        service.bpy = original_bpy


def _guided_ridge_curve_sculpt_transaction_fixture(module):
    """Native apply notifications are ignored only for one bounded generation."""
    original_state = module.runtime.guided_ridge_state
    original_cancel = module.guided_ridge._guided_ridge_cancel
    cancelled = []
    data = SimpleNamespace(as_pointer=lambda: 22, vertices=(), polygons=())
    obj = SimpleNamespace(as_pointer=lambda: 11, data=data)
    state = {
        "active": True,
        "obj": obj,
        "curve_sculpt_apply_generation": 3,
        "curve_sculpt_apply_active": True,
        "curve_sculpt_expected_update_generation": None,
        "curve_sculpt_expected_update_budget": 0,
        "curve_sculpt_expected_mesh_signature": None,
    }
    update = SimpleNamespace(id=obj)
    depsgraph = SimpleNamespace(updates=[update])
    try:
        module.guided_ridge._guided_ridge_cancel = lambda _state, reason: cancelled.append(reason)
        module.runtime.guided_ridge_state = state
        module.guided_ridge._on_guided_ridge_depsgraph_update(None, depsgraph)
        assert not cancelled
        assert state["curve_sculpt_expected_update_generation"] == 3
        assert state["curve_sculpt_expected_update_budget"] >= 1
        state["curve_sculpt_apply_active"] = False
        state["curve_sculpt_expected_update_budget"] = 1
        module.guided_ridge._on_guided_ridge_depsgraph_update(None, depsgraph)
        assert not cancelled
        assert state["curve_sculpt_expected_update_generation"] is None
        module.guided_ridge._on_guided_ridge_depsgraph_update(None, depsgraph)
        assert cancelled == ["stale"]
        return {
            "passed": True,
            "generation_bounded": True,
            "external_update_resumes_cancel": True,
        }
    finally:
        module.guided_ridge._guided_ridge_cancel = original_cancel
        module.runtime.guided_ridge_state = original_state


def _guided_ridge_curve_default_inscribed_fixture(module):
    """A fresh route starts as a tight, clean inscribed C1 curve."""
    from mathutils import Vector

    route = [
        Vector((0.0, 0.0)),
        Vector((1.0, 0.55)),
        Vector((2.0, 1.05)),
        Vector((3.0, 0.72)),
        Vector((4.0, 0.0)),
    ]
    default_shape = float(module.GUIDED_RIDGE_CURVE_DEFAULT_SMOOTHING)
    assert -100.0 < default_shape < 0.0
    raw_data = module.guided_ridge._guided_ridge_curve_2d_preview_points(route, 0.0)
    default_data = module.guided_ridge._guided_ridge_curve_2d_preview_points(route, default_shape)
    chord_data = module.guided_ridge._guided_ridge_curve_2d_preview_points(route, -100.0)
    raw = raw_data["preview"]
    default = default_data["preview"]
    chord = chord_data["preview"]
    assert len(default) >= 8
    assert default[0] == route[0] and default[-1] == route[-1]
    assert chord[0] == route[0] and chord[-1] == route[-1]

    def _distances(points):
        first, last = points[0], points[-1]
        delta = last - first
        normal = Vector((-delta.y, delta.x)).normalized()
        return [
            (point - first.lerp(last, index / float(len(points) - 1))).dot(normal)
            for index, point in enumerate(points)
        ]

    raw_wave = _distances(raw)
    default_wave = _distances(default)
    chord_wave = _distances(chord)
    raw_amplitude = max(abs(value) for value in raw_wave[1:-1])
    default_amplitude = max(abs(value) for value in default_wave[1:-1])
    chord_amplitude = max(abs(value) for value in chord_wave[1:-1])
    assert chord_amplitude < default_amplitude < raw_amplitude
    raw_parameters = [
        index / float(max(len(raw) - 1, 1)) for index in range(len(raw))
    ]
    default_at_raw = module.guided_ridge._guided_ridge_curve_2d_bezier_samples(
        default_data["preview_bezier_segments"], raw_parameters
    )
    chord_at_raw = module.guided_ridge._guided_ridge_curve_2d_bezier_samples(
        chord_data["preview_bezier_segments"], raw_parameters
    )
    raw_distance = sum(
        (raw[index] - default_at_raw[index]).length for index in range(len(raw))
    )
    chord_distance = sum(
        (raw[index] - chord_at_raw[index]).length for index in range(len(raw))
    )
    assert raw_distance < chord_distance
    assert max(abs(value) for value in default_wave[1:-1]) > 1.0e-6

    # The historical inscribed construction is the restored pre-outward
    # baseline.  Its compact fit may flatten a small opposite lobe to the
    # endpoint chord; it must not invert that lobe or amplify it.
    s_route = [
        Vector((0.0, 0.0)),
        Vector((1.0, 0.85)),
        Vector((2.0, -0.62)),
        Vector((3.0, 0.78)),
        Vector((4.0, 0.0)),
    ]
    s_raw = module.guided_ridge._guided_ridge_curve_2d_preview_points(s_route, 0.0)["preview"]
    s_default = module.guided_ridge._guided_ridge_curve_2d_preview_points(s_route, default_shape)["preview"]
    s_raw_wave = _distances(s_raw)
    s_default_wave = _distances(s_default)
    positive_raw = max(s_raw_wave)
    negative_raw = min(s_raw_wave)
    positive_default = max(s_default_wave)
    negative_default = min(s_default_wave)
    assert positive_raw > 1.0e-6 and negative_raw < -1.0e-6
    assert 0.0 < positive_default < positive_raw
    assert negative_raw <= negative_default <= 0.0
    return {
        "passed": True,
        "default_shape": default_shape,
        "raw_amplitude": raw_amplitude,
        "default_amplitude": default_amplitude,
        "chord_amplitude": chord_amplitude,
        "raw_closeness_better_than_chord": True,
        "s_positive_raw": positive_raw,
        "s_positive_default": positive_default,
        "s_negative_raw": negative_raw,
        "s_negative_default": negative_default,
    }


def _guided_ridge_curve_signed_shape_fixture(module):
    """Signed Curve Shape mirrors filtered displacement without mesh writes."""
    from mathutils import Vector

    original_state = module.runtime.guided_ridge_state
    original_overlay = module.guided_ridge._guided_ridge_overlay_tag
    original_smoothing = module.runtime.guided_ridge_curve_last_smoothing
    route = [
        Vector((0.0, 0.0, 0.0)),
        Vector((1.0, 0.55, 0.0)),
        Vector((2.0, 1.20, 0.0)),
        Vector((3.0, 0.85, 0.0)),
        Vector((4.0, 1.90, 0.0)),
        Vector((5.0, 0.75, 0.0)),
        Vector((6.0, 1.35, 0.0)),
        Vector((7.0, 0.40, 0.0)),
        Vector((8.0, 0.0, 0.0)),
    ]

    def high_frequency_energy(points):
        return sum(
            (points[index + 1] - points[index] * 2.0 + points[index - 1]).length
            for index in range(1, len(points) - 1)
        )

    try:
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        raw = module.guided_ridge._guided_ridge_curve_preview_points(route, 0.0)
        positive = module.guided_ridge._guided_ridge_curve_preview_points(route, 100.0)
        negative = module.guided_ridge._guided_ridge_curve_preview_points(route, -100.0)
        assert raw and len(raw) == len(positive) == len(negative)
        assert negative[0] == route[0] and negative[-1] == route[-1]
        assert positive[0] == route[0] and positive[-1] == route[-1]
        opposite_dot = sum(
            (positive[index] - raw[index]).dot(negative[index] - raw[index])
            for index in range(1, len(raw) - 1)
        )
        assert opposite_dot < 0.0
        minus_one = module.guided_ridge._guided_ridge_curve_preview_points(route, -1.0)
        plus_one = module.guided_ridge._guided_ridge_curve_preview_points(route, 1.0)
        assert max((minus_one[i] - raw[i]).length for i in range(len(raw))) > 1.0e-8
        assert max((plus_one[i] - raw[i]).length for i in range(len(raw))) > 1.0e-8
        assert sum(
            (plus_one[i] - raw[i]).dot(minus_one[i] - raw[i])
            for i in range(1, len(raw) - 1)
        ) < 0.0
        raw_displacement = [raw[i] - positive[i] for i in range(len(raw))]
        negative_displacement = [negative[i] - raw[i] for i in range(len(raw))]
        assert high_frequency_energy(negative_displacement) < high_frequency_energy(raw_displacement)
        # Fractional historical passes keep every negative wheel step
        # observable; there must be no old five-bucket plateaus or jumps.
        negative_values = (-1.0, -2.0, -11.0, -12.0, -13.0,
                           -37.0, -38.0, -62.0, -63.0, -87.0,
                           -88.0, -99.0, -100.0)
        negative_results = [
            module.guided_ridge._guided_ridge_curve_2d_preview_points(route, value)
            for value in negative_values
        ]
        negative_distance = [
            sum(
                (item["preview"][index] - item["raw"][index]).length
                for index in range(len(item["raw"]))
            )
            for item in negative_results
        ]
        assert all(
            negative_distance[index] > negative_distance[index + 1] + 1.0e-7
            for index in range(len(negative_distance) - 1)
        )
        assert all(
            negative_results[index]["historical_inscribed_passes"]
            < negative_results[index + 1]["historical_inscribed_passes"]
            for index in range(len(negative_results) - 1)
        )
        original_wave_family = module.guided_ridge._guided_ridge_curve_2d_wave_family
        try:
            def _positive_failure(*_args, **_kwargs):
                raise RuntimeError("injected positive waveform failure")
            module.guided_ridge._guided_ridge_curve_2d_wave_family = _positive_failure
            isolated_negative = module.guided_ridge._guided_ridge_curve_2d_preview_points(route, -12.0)
            assert isolated_negative["opposite_status"] == "historical-inscribed"
            assert isolated_negative["preview"] == negative_results[3]["preview"]
        finally:
            module.guided_ridge._guided_ridge_curve_2d_wave_family = original_wave_family
        for points in (raw, positive, negative):
            for point in points:
                assert all(math.isfinite(float(value)) for value in point)
            assert min((points[i + 1] - points[i]).length for i in range(len(points) - 1)) > 1.0e-8

        state = {
            "active": True,
            "operator": SimpleNamespace(report=lambda *_args: None),
            "phase": "curve_preview",
            "curve_route": [point.copy() for point in route],
            "controls": [point.copy() for point in route],
            "curve_smoothing": 0.0,
            "curve_preview": [],
            "curve_preview_generation": 0,
            "timer": None,
            "draw_handler": None,
            "text_draw_handler": None,
        }
        module.runtime.guided_ridge_state = state
        context = SimpleNamespace()
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            state["operator"], context, SimpleNamespace(type="WHEELDOWNMOUSE", value="PRESS", shift=False)
        ) == {"RUNNING_MODAL"}
        assert state["curve_smoothing"] == -10.0
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            state["operator"], context, SimpleNamespace(type="WHEELUPMOUSE", value="PRESS", shift=True)
        ) == {"RUNNING_MODAL"}
        assert state["curve_smoothing"] == -8.0
        assert module.runtime.guided_ridge_curve_last_smoothing == -8.0
        module.guided_ridge._guided_ridge_curve_set_smoothing(state, -150.0)
        assert state["curve_smoothing"] == -100.0
        module.guided_ridge._guided_ridge_curve_set_smoothing(state, 150.0)
        assert state["curve_smoothing"] == 100.0
        module.guided_ridge._guided_ridge_cancel(state, "signed-shape-test")
        assert module.runtime.guided_ridge_state is None

        # Every original L-shaped knot is present in the common parameterization;
        # Shape=0 therefore follows the exact raw polyline instead of cutting
        # across its corner.  The fixed target makes all integer values smooth.
        l_route = [
            Vector((0.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)),
            Vector((1.0, 1.0, 0.0)), Vector((2.0, 1.0, 0.0)),
        ]
        l_raw = module.guided_ridge._guided_ridge_curve_preview_points(l_route, 0.0)
        assert all(any((sample - knot).length <= 1.0e-9 for sample in l_raw) for knot in l_route)

        # An adversarial but exactly collinear route must remain geometrically
        # straight for every signed Shape value, even when its knot spacing is
        # highly non-uniform.  This catches density-dependent smoothing and
        # tangential reparameterization/foldback.
        adversarial_line = [
            Vector((0.0, 0.0, 0.0)),
            Vector((15.0 / 11.0 + 0.0001, 0.0, 0.0)),
            Vector((3.0, 0.0, 0.0)),
        ]
        for value in range(-100, 101):
            line = module.guided_ridge._guided_ridge_curve_preview_points(adversarial_line, value)
            assert line[0] == adversarial_line[0] and line[-1] == adversarial_line[-1]
            for index, point in enumerate(line):
                assert abs(float(point.y)) <= 1.0e-8 and abs(float(point.z)) <= 1.0e-8
                if index:
                    assert point.x >= line[index - 1].x - 1.0e-8
                    assert (point - line[index - 1]).length > 1.0e-8

        # Uniformly scaled copies, including a close interior knot, must keep
        # the same normalized Shape response.  This guards against fixed
        # world-space dot/length floors in the progression limiter.
        scaled_route_base = [
            Vector((0.0, 0.0, 0.0)), Vector((0.0001, 0.00005, 0.0)),
        ] + route[1:]
        normalized_responses = {}
        for scale in (1.0e-6, 1.0, 1.0e6):
            scaled_route = [point * scale for point in scaled_route_base]
            scaled_raw = module.guided_ridge._guided_ridge_curve_preview_points(scaled_route, 0.0)
            scaled_plus = module.guided_ridge._guided_ridge_curve_preview_points(scaled_route, 1.0)
            scaled_minus = module.guided_ridge._guided_ridge_curve_preview_points(scaled_route, -1.0)
            scaled_plus_max = module.guided_ridge._guided_ridge_curve_preview_points(scaled_route, 100.0)
            scaled_minus_max = module.guided_ridge._guided_ridge_curve_preview_points(scaled_route, -100.0)
            assert len(scaled_raw) == len(scaled_plus) == len(scaled_minus)
            plus_response = sum(
                (scaled_plus[index] - scaled_raw[index]).length
                for index in range(len(scaled_raw))
            ) / scale
            minus_response = sum(
                (scaled_minus[index] - scaled_raw[index]).length
                for index in range(len(scaled_raw))
            ) / scale
            plus_max_response = sum(
                (scaled_plus_max[index] - scaled_raw[index]).length
                for index in range(len(scaled_raw))
            ) / scale
            minus_max_response = sum(
                (scaled_minus_max[index] - scaled_raw[index]).length
                for index in range(len(scaled_raw))
            ) / scale
            assert plus_response > 1.0e-8 and minus_response > 1.0e-8
            assert plus_max_response > 1.0e-6 and minus_max_response > 1.0e-6
            for index in range(len(scaled_raw) - 1):
                raw_step = scaled_raw[index + 1] - scaled_raw[index]
                for shaped in (scaled_plus, scaled_minus, scaled_plus_max, scaled_minus_max):
                    assert raw_step.dot(shaped[index + 1] - shaped[index]) >= -1.0e-14 * scale * scale
            normalized_responses[scale] = (plus_response, minus_response)
        reference_plus, reference_minus = normalized_responses[1.0]
        for plus_response, minus_response in normalized_responses.values():
            assert math.isclose(plus_response, reference_plus, rel_tol=0.02, abs_tol=1.0e-8)
            assert math.isclose(minus_response, reference_minus, rel_tol=0.02, abs_tol=1.0e-8)

        near_coincident = [
            Vector((0.0, 0.0, 0.0)), Vector((1.0e-8, 0.0, 0.0)),
            Vector((0.0002, 0.0, 0.0)), Vector((0.00021, 0.3, 0.0)),
            Vector((7.0, 0.3, 0.0)), Vector((7.001, 0.31, 0.0)),
            Vector((12.0, 0.0, 0.0)),
        ]
        near_raw = module.guided_ridge._guided_ridge_curve_preview_points(near_coincident, 0.0)
        near_positive = module.guided_ridge._guided_ridge_curve_preview_points(near_coincident, 100.0)
        assert near_raw[0] == near_coincident[0] and near_raw[-1] == near_coincident[-1]
        assert all(all(math.isfinite(float(value)) for value in point) for point in near_positive)

        # Compare every generated segment with the corresponding raw arc
        # segment.  Positive dot products are a practical planar progression
        # check; full 3D self-intersection is intentionally outside this
        # preview fixture's guarantee.
        for shaped in (positive, negative):
            for index in range(len(raw) - 1):
                raw_step = raw[index + 1] - raw[index]
                shaped_step = shaped[index + 1] - shaped[index]
                assert raw_step.dot(shaped_step) > -1.0e-8
        sweep = [module.guided_ridge._guided_ridge_curve_preview_points(route, value) for value in range(-100, 101)]
        frame_delta = max(
            (sweep[index + 1][point] - sweep[index][point]).length
            for index in range(len(sweep) - 1)
            for point in range(len(sweep[index]))
        )
        assert frame_delta < 0.05
        for value in (9.0, 10.0, 11.0, 12.0):
            current = module.guided_ridge._guided_ridge_curve_preview_points(route, value)
            previous = module.guided_ridge._guided_ridge_curve_preview_points(route, value - 1.0)
            assert max((current[i] - previous[i]).length for i in range(len(current))) < 0.05
        return {
            "passed": True,
            "bounds": True,
            "wheel_direction": True,
            "persistence": True,
            "endpoints": True,
            "continuous_zero": True,
            "raw_knots_preserved": True,
            "integer_sweep_continuous": True,
            "opposite_displacement": True,
            "filtered_displacement": True,
            "progression_no_foldback": True,
            "planar_segment_progression": True,
            "scale_invariant_progression": True,
            "mesh_unchanged": True,
        }
    finally:
        module.runtime.guided_ridge_state = original_state
        module.guided_ridge._guided_ridge_overlay_tag = original_overlay
        module.runtime.guided_ridge_curve_last_smoothing = original_smoothing


def _guided_ridge_curve_2d_quality_fixture(module):
    """Current-view 2D fit is dense, C1, and soft to manual outliers."""
    from mathutils import Vector

    def angle(first, second):
        if first.length <= 1.0e-9 or second.length <= 1.0e-9:
            return 0.0
        return math.acos(max(-1.0, min(1.0, first.normalized().dot(second.normalized()))))

    # Uneven route spacing plus one intentionally misplaced hand click.  The
    # interior click is an observation only; it must not become a Bezier knot.
    route = [
        Vector((40.0, 300.0)), Vector((66.0, 286.0)), Vector((104.0, 278.0)),
        Vector((146.0, 256.0)), Vector((210.0, 386.0)), Vector((250.0, 222.0)),
        Vector((306.0, 208.0)), Vector((352.0, 190.0)), Vector((390.0, 176.0)),
    ]
    result = module.guided_ridge._guided_ridge_curve_2d_preview_points(route, 100.0)
    raw_result = module.guided_ridge._guided_ridge_curve_2d_preview_points(route, 0.0)
    negative = module.guided_ridge._guided_ridge_curve_2d_preview_points(route, -100.0)["preview"]
    preview = result["preview"]
    smooth = result["smooth"]
    raw = raw_result["raw"]
    assert len(preview) >= 96
    assert result["fit_segment_count"] <= len(route)
    assert preview[0] == route[0] and preview[-1] == route[-1]
    assert raw[0] == route[0] and raw[-1] == route[-1]
    for knot in route:
        assert min((sample - knot).length for sample in raw) <= 1.0e-6
    tangents = result["bezier_segments"]
    join_tolerance = max(
        1.0e-6,
        sum((route[index + 1] - route[index]).length for index in range(len(route) - 1))
        * 1.0e-6,
    )
    for first, second in zip(tangents, tangents[1:]):
        outgoing = first[3] - first[2]
        incoming = second[1] - second[0]
        assert (outgoing - incoming).length <= join_tolerance
    angles = [
        angle(preview[index] - preview[index - 1], preview[index + 1] - preview[index])
        for index in range(1, len(preview) - 1)
    ]
    assert max(angles) < 0.80
    assert min((preview[index + 1] - preview[index]).length for index in range(len(preview) - 1)) > 1.0e-5
    assert max((preview[index + 1] - preview[index]).length for index in range(len(preview) - 1)) < 12.0
    assert min((negative[ index + 1] - negative[index]).length for index in range(len(negative) - 1)) > 1.0e-5

    # The outlier must not be interpolated, and changing it must alter the
    # low-frequency result globally rather than introduce a local cusp.
    outlier = route[4]
    assert min((point - outlier).length for point in preview) > 18.0
    regular = list(route)
    regular[4] = Vector((210.0, 242.0))
    regular_preview = module.guided_ridge._guided_ridge_curve_2d_preview_points(regular, 100.0)["preview"]
    comparison_count = min(len(preview), len(regular_preview))
    assert max(
        (preview[index] - regular_preview[index]).length
        for index in range(comparison_count)
    ) < 240.0
    assert max(
        angle(
            regular_preview[index] - regular_preview[index - 1],
            regular_preview[index + 1] - regular_preview[index],
        )
        for index in range(1, len(regular_preview) - 1)
    ) < 0.80

    # A perspective-like projected route with highly uneven depth-induced
    # spacing remains smooth because all fitting is performed in this 2D list.
    perspective = [
        Vector((18.0, 420.0)), Vector((94.0, 365.0)), Vector((134.0, 352.0)),
        Vector((212.0, 280.0)), Vector((266.0, 268.0)), Vector((338.0, 190.0)),
        Vector((430.0, 176.0)),
    ]
    perspective_result = module.guided_ridge._guided_ridge_curve_2d_preview_points(perspective, 100.0)
    assert max(
        angle(
            perspective_result["preview"][index] - perspective_result["preview"][index - 1],
            perspective_result["preview"][index + 1] - perspective_result["preview"][index],
        )
        for index in range(1, len(perspective_result["preview"]) - 1)
    ) < 0.80
    return {
        "passed": True,
        "projected_fit": True,
        "sample_count": len(preview),
        "fit_segments": result["fit_segment_count"],
        "max_angle_radians": max(angles),
        "max_chord_px": max((preview[index + 1] - preview[index]).length for index in range(len(preview) - 1)),
        "outlier_clearance_px": min((point - outlier).length for point in preview),
        "c1_continuity": True,
        "interior_knots_soft": True,
        "shape_zero_raw_knots": True,
    }


def _guided_ridge_curve_screen_display_fixture(module):
    """Verify the actual screen-space draw input at Shape=+100."""
    from mathutils import Vector

    original_projection = module.guided_ridge._guided_ridge_project_3d_to_region_2d
    original_overlay = module.guided_ridge._guided_ridge_overlay_tag
    original_state = module.runtime.guided_ridge_state
    original_opposite = module.guided_ridge._guided_ridge_curve_2d_opposite_bezier
    original_wave = module.guided_ridge._guided_ridge_curve_2d_wave_family
    route = [
        Vector((0.0, 0.0, 0.0)), Vector((18.0, -28.0, 0.0)),
        Vector((35.0, -60.0, 0.0)), Vector((58.0, -26.0, 0.0)),
        Vector((80.0, 0.0, 0.0)),
    ]
    negative_route = [
        Vector((0.0, 0.0, 0.0)), Vector((18.0, -28.0, 0.0)),
        Vector((35.0, -60.0, 0.0)), Vector((58.0, -26.0, 0.0)),
        Vector((80.0, 0.0, 0.0)),
    ]
    try:
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = (
            lambda _region, _region_3d, point: Vector((point.x + 20.0, point.y + 130.0))
        )
        state = {
            "phase": "curve_preview",
            "curve_route": [point.copy() for point in route],
            "curve_smoothing": 100.0,
            "curve_screen_generation": 0,
        }
        context = SimpleNamespace(
            region=SimpleNamespace(width=220, height=320),
            space_data=SimpleNamespace(region_3d=SimpleNamespace()),
            preferences=SimpleNamespace(system=SimpleNamespace(ui_scale=1.0)),
        )

        def _assert_drawn_bezier_matches(route_points, screen_points, segments, parameters=None):
            assert segments
            clean, cumulative, total = module.guided_ridge._guided_ridge_curve_2d_arc_data(route_points)
            if parameters is None:
                distances = module.guided_ridge._guided_ridge_curve_output_parameters(
                    total, cumulative, len(screen_points)
                )
                parameters = [distance / total for distance in distances]
            tessellated = module.guided_ridge._guided_ridge_curve_2d_bezier_samples(
                segments, parameters
            )
            assert len(tessellated) == len(screen_points)
            assert max(
                (tessellated[index] - screen_points[index]).length
                for index in range(len(screen_points))
            ) <= 1.0e-5
            scale = max(total, 1.0e-12)
            join_angles = []
            join_errors = []
            for index in range(len(segments) - 1):
                outgoing = segments[index][3] - segments[index][2]
                incoming = segments[index + 1][1] - segments[index + 1][0]
                assert outgoing.length > scale * 1.0e-9
                assert incoming.length > scale * 1.0e-9
                join_errors.append((outgoing - incoming).length)
                join_angles.append(
                    math.acos(max(-1.0, min(1.0, outgoing.normalized().dot(incoming.normalized()))))
                )
            assert max(join_errors or [0.0]) <= scale * 1.0e-5
            assert max(join_angles or [0.0]) <= 0.01
            return {
                "segment_count": len(segments),
                "max_join_error": max(join_errors or [0.0]),
                "max_join_angle": max(join_angles or [0.0]),
            }

        assert module.guided_ridge._guided_ridge_curve_update_screen_preview(context, state)
        projected = [Vector((point.x + 20.0, point.y + 130.0)) for point in route]
        expected = module.guided_ridge._guided_ridge_curve_2d_preview_points(projected, 100.0)
        displayed = state["curve_screen_preview"]
        assert displayed and len(displayed) == len(expected["preview"])
        assert all(
            (displayed[index] - expected["preview"][index]).length <= 1.0e-7
            for index in range(len(displayed))
        )
        initial_draw_metrics = _assert_drawn_bezier_matches(
            projected, displayed, state["curve_screen_bezier_segments"],
            expected["parameters"],
        )
        old_scale = module.guided_ridge._guided_ridge_curve_progression_scale(
            expected["raw"],
            [expected["smooth"][index] - expected["raw"][index] for index in range(len(displayed))],
        )
        assert 0.0 <= old_scale <= 1.0
        before_100 = module.guided_ridge._guided_ridge_curve_2d_preview_points(projected, 99.0)["preview"]
        assert max(
            (displayed[index] - before_100[index]).length
            for index in range(len(displayed))
        ) < 1.0
        assert displayed[0] == projected[0] and displayed[-1] == projected[-1]
        outlier = projected[2]
        # The amplified route may pass near a broad manual crest by design;
        # it must remain a fitted curve rather than interpolate the knot.
        assert min((point - outlier).length for point in displayed) > 1.0
        def _turning_angle(first, second):
            if first.length <= 1.0e-9 or second.length <= 1.0e-9:
                return 0.0
            return math.acos(max(-1.0, min(1.0, first.normalized().dot(second.normalized()))))
        max_angle = max(
            _turning_angle(displayed[index] - displayed[index - 1], displayed[index + 1] - displayed[index])
            for index in range(1, len(displayed) - 1)
        )
        # Raw-relative amplification intentionally increases broad-lobe
        # curvature; keep a meaningful no-cusp bound without requiring the
        # old sub-raw envelope.
        assert max_angle < 0.80
        # The screen draw path must use the independently fitted emphasized
        # raw-outer Bezier at the positive endpoint.
        negative_state = {
            "phase": "curve_preview",
            "curve_route": [point.copy() for point in negative_route],
            "curve_smoothing": 100.0,
            "curve_screen_generation": 0,
        }
        assert module.guided_ridge._guided_ridge_curve_update_screen_preview(context, negative_state)
        negative_projected = [Vector((point.x + 20.0, point.y + 130.0)) for point in negative_route]
        negative_expected = module.guided_ridge._guided_ridge_curve_2d_preview_points(
            negative_projected, 100.0
        )
        negative_displayed = negative_state["curve_screen_preview"]
        assert negative_displayed == negative_expected["preview"]
        negative_draw_metrics = _assert_drawn_bezier_matches(
            negative_projected,
            negative_displayed,
            negative_state["curve_screen_bezier_segments"],
            negative_expected["parameters"],
        )
        assert negative_expected["opposite"]
        negative_displacement = sum(
            (negative_displayed[index] - negative_expected["raw"][index]).length
            for index in range(len(negative_displayed))
        )
        positive_displacement = sum(
            (displayed[index] - expected["raw"][index]).length
            for index in range(len(displayed))
        )
        assert negative_displacement > 1.0e-3
        # Positive is the local-wave Amplified family: its displacement is
        # derived from shared Bezier targets, never local manual-knot
        # attraction or a global chord-side classification.
        chord = negative_expected["raw"][-1] - negative_expected["raw"][0]
        chord_normal = Vector((-chord.y, chord.x)).normalized()
        def _signed_bulge(points):
            values = [
                (points[index] - points[0]).dot(chord_normal)
                for index in range(1, len(points) - 1)
            ]
            values = sorted(values, key=abs)
            return values[int((len(values) - 1) * 0.75)]
        raw_bulge = _signed_bulge(negative_expected["raw"])
        positive_bulge = _signed_bulge(negative_displayed)
        inscribed_expected = module.guided_ridge._guided_ridge_curve_2d_preview_points(
            negative_projected, -100.0
        )
        negative_bulge = _signed_bulge(inscribed_expected["preview"])
        assert raw_bulge < 0.0 and positive_bulge < 0.0
        assert inscribed_expected["opposite_status"] == "historical-inscribed"
        assert abs(negative_bulge) < abs(raw_bulge)
        assert abs(positive_bulge) > abs(raw_bulge) * 1.15
        assert abs(negative_bulge) < abs(raw_bulge) * 0.85
        assert abs(positive_bulge) > abs(negative_bulge) * 1.8
        assert negative_expected["opposite_status"] == "valid"
        moved_route = [
            negative_projected[0], Vector((38.0, 102.0)), Vector((55.0, 78.0)),
            Vector((75.0, 104.0)), negative_projected[-1],
        ]
        moved_negative = module.guided_ridge._guided_ridge_curve_2d_preview_points(moved_route, 100.0)
        assert moved_negative["opposite_status"] == "valid"
        assert moved_negative["preview"][0] == negative_projected[0]
        assert moved_negative["preview"][-1] == negative_projected[-1]
        def _arch_angle(first, second):
            if first.length <= 1.0e-9 or second.length <= 1.0e-9:
                return 0.0
            return math.acos(max(-1.0, min(1.0, first.normalized().dot(second.normalized()))))
        assert max(
            _arch_angle(
                moved_negative["preview"][index] - moved_negative["preview"][index - 1],
                moved_negative["preview"][index + 1] - moved_negative["preview"][index],
            )
            for index in range(1, len(moved_negative["preview"]) - 1)
        ) < 0.80
        # Inject a complete opposite-fit failure on a non-straight route.  The
        # pure helper is explicit, while the screen state retains the exact
        # last valid display rather than substituting a new smooth curve.
        straight = module.guided_ridge._guided_ridge_curve_2d_preview_points(
            [Vector((0.0, 0.0)), Vector((40.0, 0.0)), Vector((80.0, 0.0))], -100.0
        )
        assert straight["opposite_status"] == "historical-inscribed"
        assert straight["opposite_warning"] is None
        assert all(abs(point.y) <= 1.0e-7 for point in straight["preview"])
        assert max(
            (straight["preview"][index] - straight["raw"][index]).length
            for index in range(len(straight["preview"]))
        ) <= 1.0e-4
        module.guided_ridge._guided_ridge_curve_2d_wave_family = (
            lambda *_args, **_kwargs: (None, "failed")
        )
        failed = module.guided_ridge._guided_ridge_curve_2d_preview_points(negative_projected, 100.0)
        assert failed["opposite_status"] == "failed"
        assert failed["opposite_warning"]
        assert failed["preview"] == failed["smooth"]
        assert failed["preview"] != failed["raw"]
        retained_state = {
            "phase": "curve_preview",
            "curve_route": [point.copy() for point in negative_route],
            "curve_smoothing": -100.0,
            "curve_screen_generation": 0,
        }
        assert module.guided_ridge._guided_ridge_curve_update_screen_preview(context, retained_state)
        prior_display = [point.copy() for point in retained_state["curve_screen_preview"]]
        retained_state["curve_smoothing"] = 100.0
        assert module.guided_ridge._guided_ridge_curve_update_screen_preview(context, retained_state)
        assert retained_state["curve_shape_status"] == "retained"
        assert retained_state["curve_effective_smoothing"] == -100.0
        assert retained_state["curve_shape_warning"]
        assert retained_state["curve_screen_preview"] == prior_display
        assert "Emphasized" not in retained_state["curve_shape_warning"]

        module.guided_ridge._guided_ridge_curve_2d_opposite_bezier = original_opposite
        module.guided_ridge._guided_ridge_curve_2d_wave_family = original_wave
        mirrored_route = [
            Vector((0.0, 0.0)), Vector((18.0, 28.0)), Vector((35.0, 60.0)),
            Vector((58.0, 26.0)), Vector((80.0, 0.0)),
        ]
        mirrored = module.guided_ridge._guided_ridge_curve_2d_preview_points(mirrored_route, 100.0)
        assert mirrored["opposite_status"] == "valid"
        assert mirrored["preview"][len(mirrored["preview"]) // 2].y > 0.0
        ambiguous = module.guided_ridge._guided_ridge_curve_2d_preview_points(
            [Vector((0.0, 0.0)), Vector((20.0, 45.0)), Vector((40.0, -45.0)), Vector((60.0, 45.0)), Vector((80.0, 0.0))],
            100.0,
        )
        assert ambiguous["opposite_status"] == "valid"
        if ambiguous["opposite_warning"]:
            assert "reduced" in ambiguous["opposite_warning"]
            assert ambiguous["effective_smoothing"] < 100.0
        ambiguous_inscribed = module.guided_ridge._guided_ridge_curve_2d_preview_points(
            [Vector((0.0, 0.0)), Vector((20.0, 45.0)), Vector((40.0, -45.0)), Vector((60.0, 45.0)), Vector((80.0, 0.0))],
            -100.0,
        )
        assert ambiguous_inscribed["opposite_status"] == "historical-inscribed"
        assert ambiguous_inscribed["opposite_warning"] is None

        # Exercise the real screen-preview state, not only the pure helper,
        # across both signs for simple, mirrored, S-shaped, and multi-wave
        # routes.  The stored draw segments must tessellate to the exact final
        # draw input and retain every lobe's local waveform orientation.
        screen_routes = (
            negative_route,
            [Vector((0.0, 0.0)), Vector((18.0, 28.0)), Vector((35.0, 60.0)), Vector((58.0, 26.0)), Vector((80.0, 0.0))],
            [Vector((0.0, 0.0)), Vector((18.0, 38.0)), Vector((36.0, -34.0)), Vector((54.0, 42.0)), Vector((72.0, -30.0)), Vector((90.0, 0.0))],
            [Vector((0.0, 0.0)), Vector((14.0, 20.0)), Vector((28.0, -17.0)), Vector((42.0, 16.0)), Vector((56.0, -13.0)), Vector((70.0, 12.0)), Vector((84.0, 0.0))],
        )
        def _chord_lobe_amplitudes(values):
            start, end = values[0], values[-1]
            chord = end - start
            if chord.length <= 1.0e-9:
                return {-1.0: 0.0, 1.0: 0.0}
            normal = Vector((-chord.y, chord.x)).normalized()
            offsets = [
                (point - start.lerp(end, index / float(len(values) - 1))).dot(normal)
                for index, point in enumerate(values[1:-1], 1)
            ]
            noise = max(
                sum((values[index + 1] - values[index]).length for index in range(len(values) - 1)) * 1.0e-5,
                1.0e-8,
            )
            return {
                side: max(
                    (abs(value) for value in offsets if value * side > noise),
                    default=0.0,
                )
                for side in (-1.0, 1.0)
            }
        def _raw_oracle_lobe_amplitudes(expected):
            oracle = expected.get("raw_chord_oracle") or {}
            offsets = tuple(oracle.get("offsets", ()))
            noise = float(oracle.get("noise", 0.0) or 0.0)
            return {
                side: max(
                    (abs(value) for value in offsets if value * side > noise),
                    default=0.0,
                )
                for side in (-1.0, 1.0)
            }
        def _per_lobe_oracle_amplitudes(expected, values):
            """Compare each meaningful raw-knot lobe on its own arc interval."""
            oracle = expected.get("raw_chord_oracle") or {}
            raw_parameters = tuple(oracle.get("parameters", ()))
            raw_offsets = tuple(oracle.get("offsets", ()))
            sample_parameters = tuple(expected.get("parameters", ()))
            if not raw_parameters or len(values) != len(sample_parameters):
                return []
            raw = expected["raw"]
            chord = raw[-1] - raw[0]
            if chord.length <= 1.0e-9:
                return []
            normal = Vector((-chord.y, chord.x)).normalized()
            sample_offsets = [
                (point - raw[0].lerp(raw[-1], float(parameter))).dot(normal)
                for point, parameter in zip(values, sample_parameters)
            ]
            noise = float(oracle.get("noise", 0.0) or 0.0)
            result = []
            for interval in expected.get("lobe_intervals") or ():
                side = float(interval.get("side", 0.0))
                start = float(interval.get("start", 0.0))
                end = float(interval.get("end", 1.0))
                raw_amplitude = max(
                    (
                        abs(value)
                        for parameter, value in zip(raw_parameters, raw_offsets)
                        if start - 1.0e-9 <= parameter <= end + 1.0e-9
                        and value * side > noise
                    ),
                    default=0.0,
                )
                candidate_amplitude = max(
                    (
                        abs(value)
                        for parameter, value in zip(sample_parameters, sample_offsets)
                        if start - 1.0e-9 <= float(parameter) <= end + 1.0e-9
                        and value * side > noise
                    ),
                    default=0.0,
                )
                if raw_amplitude > noise:
                    result.append((side, raw_amplitude, candidate_amplitude))
            return result
        screen_shape_checks = 0
        for route_points in screen_routes:
            projected_route = [Vector((point.x + 20.0, point.y + 130.0)) for point in route_points]
            for shape_value in (-100.0, -50.0, -10.0, -1.0, 1.0, 10.0, 50.0, 100.0):
                live_state = {
                    "phase": "curve_preview",
                    "curve_route": [Vector((point.x, point.y, 0.0)) for point in route_points],
                    "curve_smoothing": shape_value,
                    "curve_screen_generation": 0,
                }
                assert module.guided_ridge._guided_ridge_curve_update_screen_preview(context, live_state)
                assert live_state["curve_shape_status"] in {
                    "valid", "historical-inscribed"
                }
                assert live_state["curve_screen_preview"] != live_state["curve_screen_raw"]
                live_expected = module.guided_ridge._guided_ridge_curve_2d_preview_points(
                    projected_route, shape_value
                )
                _assert_drawn_bezier_matches(
                    projected_route,
                    live_state["curve_screen_preview"],
                    live_state["curve_screen_bezier_segments"],
                    live_expected["parameters"],
                )
                if shape_value < 0.0:
                    # Negative/inscribed values use the restored pre-outward
                    # C1 construction, independently of the positive
                    # waveform gain validator.
                    assert live_expected["opposite_status"] == "historical-inscribed"
                    assert live_state["curve_screen_preview"] == live_expected["preview"]
                    assert live_expected["historical_inscribed_passes"] >= 1
                    screen_shape_checks += 1
                    continue
                # The waveform family is now measured from the endpoint chord.
                # The old multi-pass smooth fit absorbs a gentle arc and is
                # retained only as a compatibility diagnostic, not as the
                # amplitude baseline.
                base_points = live_expected["chord"]
                detail_points = live_expected["inscribed"]
                final_points = live_state["curve_screen_preview"]
                expected_gain = 1.0 + float(
                    live_expected.get("effective_smoothing", shape_value)
                ) / 100.0
                route_scale = max(
                    sum((point_b - point_a).length for point_a, point_b in zip(projected_route, projected_route[1:])),
                    1.0e-12,
                )
                lobe_checks = 0
                for index in range(1, len(final_points) - 1):
                    tangent = base_points[index + 1] - base_points[index - 1]
                    if tangent.length <= route_scale * 1.0e-8:
                        continue
                    normal = Vector((-tangent.y, tangent.x)).normalized()
                    wave = (detail_points[index] - base_points[index]).dot(normal)
                    delta = (final_points[index] - base_points[index]).dot(normal)
                    if abs(wave) <= route_scale * 1.0e-8:
                        continue
                    ratio = delta / wave
                    assert wave * delta >= -(route_scale * 1.0e-8) ** 2
                    if shape_value > 0.0:
                        assert ratio >= expected_gain - 0.02
                    else:
                        assert -0.02 <= ratio <= expected_gain + 0.02
                    lobe_checks += 1
                assert lobe_checks > 0
                raw_lobes = _raw_oracle_lobe_amplitudes(live_expected)
                final_lobes = _chord_lobe_amplitudes(final_points)
                for side in (-1.0, 1.0):
                    if raw_lobes[side] <= route_scale * 1.0e-5:
                        continue
                    margin = max(raw_lobes[side] * 1.0e-4, route_scale * 1.0e-5)
                    if shape_value > 0.0:
                        assert final_lobes[side] > raw_lobes[side] + margin
                    else:
                        assert final_lobes[side] < raw_lobes[side] - margin
                # The global side check above is not sufficient for an S or
                # multi-wave route.  Use the production lobe intervals and
                # all-knot oracle independently to ensure a short secondary
                # lobe cannot disappear behind the tallest one.
                for _side, raw_lobe, candidate_lobe in _per_lobe_oracle_amplitudes(
                    live_expected, final_points
                ):
                    margin = max(raw_lobe * 1.0e-4, route_scale * 1.0e-5)
                    if shape_value > 0.0:
                        assert candidate_lobe > raw_lobe + margin
                    else:
                        assert candidate_lobe < raw_lobe - margin
                screen_shape_checks += 1

        # A narrow/tall arch used to be clipped by the fixed total*0.20 fit
        # envelope. Exercise the actual screen-state path and compare both
        # robust maximum and upper-quartile chord amplitudes. The raw-relative
        # contract also applies to small positive steps, not only +100.
        narrow_route = [
            Vector((0.0, 0.0, 0.0)), Vector((1.0, 100.0, 0.0)),
            Vector((2.0, 0.0, 0.0)),
        ]
        narrow_state = {
            "phase": "curve_preview",
            "curve_route": [point.copy() for point in narrow_route],
            "curve_smoothing": 100.0,
            "curve_screen_generation": 0,
        }
        assert module.guided_ridge._guided_ridge_curve_update_screen_preview(context, narrow_state)
        narrow_projected = [Vector((point.x + 20.0, point.y + 130.0)) for point in narrow_route]
        narrow_expected = module.guided_ridge._guided_ridge_curve_2d_preview_points(narrow_projected, 100.0)
        assert narrow_expected["opposite_status"] == "valid"
        assert narrow_state["curve_screen_preview"] == narrow_expected["preview"]
        def _chord_amplitudes(values):
            start, end = values[0], values[-1]
            chord = end - start
            normal = Vector((-chord.y, chord.x)).normalized()
            offsets = [
                abs((point - start.lerp(end, index / float(len(values) - 1))).dot(normal))
                for index, point in enumerate(values)
            ]
            ordered = sorted(offsets)
            return ordered[-1], ordered[int((len(ordered) - 1) * 0.75)]
        narrow_oracle_offsets = [
            abs(value)
            for value in (narrow_expected.get("raw_chord_oracle") or {}).get("offsets", ())
        ]
        narrow_oracle_offsets.sort()
        narrow_raw_max = narrow_oracle_offsets[-1]
        # The all-knot oracle supplies the exact peak.  For a three-knot route
        # its 75th percentile is necessarily an endpoint zero, so use the
        # bounded screen-space raw samples for the distribution metric while
        # retaining the oracle peak for the strict +1 check.
        _narrow_raw_sample_max, narrow_raw_q75 = _chord_amplitudes(
            narrow_expected["raw"]
        )
        narrow_plus_max, narrow_plus_q75 = _chord_amplitudes(narrow_expected["preview"])
        narrow_plus_one = module.guided_ridge._guided_ridge_curve_2d_preview_points(
            narrow_projected, 1.0
        )
        narrow_plus_one_max, _narrow_plus_one_q75 = _chord_amplitudes(
            narrow_plus_one["preview"]
        )
        assert narrow_plus_one["opposite_status"] == "valid"
        assert narrow_plus_one_max > narrow_raw_max
        narrow_minus = module.guided_ridge._guided_ridge_curve_2d_preview_points(narrow_projected, -100.0)
        narrow_minus_max, narrow_minus_q75 = _chord_amplitudes(narrow_minus["preview"])
        assert narrow_plus_max > narrow_raw_max * 1.05
        assert narrow_plus_q75 > narrow_raw_q75 * 1.02
        assert narrow_minus_max < narrow_raw_max * 0.85
        assert narrow_minus_q75 < narrow_raw_q75 * 0.85
        small_shape = module.guided_ridge._guided_ridge_curve_2d_preview_points(narrow_projected, 1.0)
        half_shape = module.guided_ridge._guided_ridge_curve_2d_preview_points(narrow_projected, 50.0)
        small_max, small_q75 = _chord_amplitudes(small_shape["preview"])
        half_max, half_q75 = _chord_amplitudes(half_shape["preview"])
        assert small_shape["opposite_status"] == "valid"
        assert half_shape["opposite_status"] == "valid"
        assert small_max > narrow_raw_max * 1.001
        assert small_q75 > narrow_raw_q75 * 1.001
        assert half_max > small_max
        assert half_q75 > small_q75

        # Lobe intervals are an independent raw-knot construction.  The
        # cleaned fit is intentionally given a single broad observation here;
        # production detection must still retain both unequal same-sign hills.
        two_hill_route = [
            Vector((0.0, 0.0)), Vector((20.0, 36.0)), Vector((40.0, 10.0)),
            Vector((60.0, 28.0)), Vector((80.0, 0.0)),
        ]
        two_clean, two_cumulative, two_total = module.guided_ridge._guided_ridge_curve_2d_arc_data(
            two_hill_route
        )
        two_oracle = module.guided_ridge._guided_ridge_curve_2d_raw_chord_oracle(
            two_clean, two_cumulative, two_total
        )
        two_raw_parameters = tuple(two_oracle["parameters"])
        two_raw_offsets = tuple(two_oracle["offsets"])
        two_intervals = module.guided_ridge._guided_ridge_curve_2d_detect_lobes(
            [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], two_oracle
        )
        assert len(two_intervals) == 2
        assert all(item["side"] > 0.0 for item in two_intervals)
        # These expected ranges are derived from the known raw knot indices,
        # not from the production interval list.
        assert (
            two_intervals[0]["start"] < two_raw_parameters[1]
            < two_intervals[0]["end"] <= two_raw_parameters[2] + 1.0e-9
        )
        assert (
            two_intervals[1]["start"] <= two_raw_parameters[2] + 1.0e-9
            < two_raw_parameters[3] < two_intervals[1]["end"]
        )
        assert two_intervals[0]["raw_peak_indices"] == (1,)
        assert two_intervals[1]["raw_peak_indices"] == (3,)
        two_expected = module.guided_ridge._guided_ridge_curve_2d_preview_points(
            two_hill_route, 100.0
        )
        assert two_expected["opposite_status"] == "valid"
        assert len(two_expected["lobe_intervals"]) == 2

        # A post-C1 shortage in the first lobe must affect the controls in that
        # interval only.  This catches the stale-parameter bug where every
        # control reused the final loop parameter and no interval matched.
        dense_two_hill = [
            Vector((0.0, 0.0)), Vector((10.0, 20.0)), Vector((20.0, 36.0)),
            Vector((30.0, 18.0)), Vector((40.0, 10.0)), Vector((50.0, 16.0)),
            Vector((60.0, 28.0)), Vector((70.0, 14.0)), Vector((80.0, 0.0)),
        ]
        dense_clean, dense_cumulative, dense_total = module.guided_ridge._guided_ridge_curve_2d_arc_data(
            dense_two_hill
        )
        dense_oracle = module.guided_ridge._guided_ridge_curve_2d_raw_chord_oracle(
            dense_clean, dense_cumulative, dense_total
        )
        dense_intervals = module.guided_ridge._guided_ridge_curve_2d_detect_lobes(
            [0.0] * 9,
            [value / dense_total for value in dense_cumulative],
            dense_oracle,
        )
        dense_parameters = [value / dense_total for value in dense_cumulative]
        dense_baseline = module.guided_ridge._guided_ridge_curve_2d_endpoint_chord_segments(
            dense_clean, 2
        )
        dense_detail = []
        for segment_index, segment in enumerate(dense_baseline):
            lift = 8.0 if segment_index == 0 else 2.0
            dense_detail.append(tuple(
                point + Vector((0.0, lift if control_index in (1, 2) else 0.0))
                for control_index, point in enumerate(segment)
            ))
        normalized_with_lobes, _ = module.guided_ridge._guided_ridge_curve_2d_normalize_detail_segments(
            dense_baseline,
            dense_detail,
            dense_clean,
            module.guided_ridge._guided_ridge_curve_2d_bezier_samples(
                dense_detail, dense_parameters
            ),
            dense_parameters,
            raw_oracle=dense_oracle,
            lobe_intervals=dense_intervals,
        )
        normalized_without_lobes, _ = module.guided_ridge._guided_ridge_curve_2d_normalize_detail_segments(
            dense_baseline,
            dense_detail,
            dense_clean,
            module.guided_ridge._guided_ridge_curve_2d_bezier_samples(
                dense_detail, dense_parameters
            ),
            dense_parameters,
            raw_oracle=dense_oracle,
            lobe_intervals=[],
        )
        first_interval_delta = sum(
            (normalized_with_lobes[0][index] - normalized_without_lobes[0][index]).length
            for index in (1, 2)
        )
        second_interval_delta = sum(
            (normalized_with_lobes[1][index] - normalized_without_lobes[1][index]).length
            for index in (1, 2)
        )
        assert first_interval_delta > 1.0e-6
        assert second_interval_delta >= 0.0
        assert abs(first_interval_delta - second_interval_delta) > 1.0e-6

        # Hidden Bezier peaks must not pass the final validator when the
        # displayed bounded polyline itself remains on the chord.
        hidden_raw = [
            Vector((0.0, 0.0)), Vector((50.0, 25.0)), Vector((100.0, 0.0))
        ]
        hidden_clean, hidden_cumulative, hidden_total = module.guided_ridge._guided_ridge_curve_2d_arc_data(
            hidden_raw
        )
        hidden_oracle = module.guided_ridge._guided_ridge_curve_2d_raw_chord_oracle(
            hidden_clean, hidden_cumulative, hidden_total
        )
        hidden_parameters = [0.0, 0.5, 1.0]
        hidden_intervals = module.guided_ridge._guided_ridge_curve_2d_detect_lobes(
            [0.0, 0.0, 0.0], hidden_parameters, hidden_oracle
        )
        hidden_baseline_segments = module.guided_ridge._guided_ridge_curve_2d_endpoint_chord_segments(
            hidden_clean, 1
        )
        hidden_baseline = module.guided_ridge._guided_ridge_curve_2d_bezier_samples(
            hidden_baseline_segments, hidden_parameters
        )
        hidden_segments = [(
            hidden_baseline_segments[0][0],
            hidden_baseline_segments[0][0] + Vector((0.0, 120.0)),
            hidden_baseline_segments[0][3] + Vector((0.0, 120.0)),
            hidden_baseline_segments[0][3],
        )]
        hidden_validation = module.guided_ridge._guided_ridge_curve_2d_wave_validate(
            hidden_clean,
            hidden_baseline,
            hidden_baseline,
            hidden_baseline,
            hidden_segments,
            1.0,
            raw_oracle=hidden_oracle,
            lobe_intervals=hidden_intervals,
            parameters=hidden_parameters,
        )
        assert not hidden_validation["valid"]
        assert "amplified" in hidden_validation["reason"]

        # The signed family is local waveform gain, not a global chord-side
        # classifier.  Both lobes of this S route remain valid, and the
        # displacement from the shared low-frequency baseline grows
        # monotonically for -100..+100 without replaying raw knots.
        wave_route = [
            Vector((0.0, 0.0)), Vector((18.0, 38.0)), Vector((36.0, -34.0)),
            Vector((54.0, 42.0)), Vector((72.0, -30.0)), Vector((90.0, 0.0)),
        ]
        wave_base = module.guided_ridge._guided_ridge_curve_2d_preview_points(wave_route, -100.0)
        wave_amounts = []
        for shape in (-100.0, -50.0, -1.0, 1.0, 50.0, 100.0):
            wave_result = module.guided_ridge._guided_ridge_curve_2d_preview_points(wave_route, shape)
            assert wave_result["opposite_status"] in {"valid", "historical-inscribed"}
            assert wave_result["opposite_warning"] is None
            wave_amounts.append(
                sum(
                    (wave_result["preview"][index] - wave_base["chord"][index]).length
                    for index in range(1, len(wave_result["preview"]) - 1)
                )
            )
        assert wave_amounts[0] < wave_amounts[1] < wave_amounts[2]
        assert wave_amounts[2] < wave_amounts[3] < wave_amounts[4] < wave_amounts[5]

        # Only Shape=0 may expose the angular/manual polyline.  Every other
        # signed wheel value stays in a Bezier-generated smooth family.
        nonzero_shape_metrics = {}
        for shape_value in range(-100, 101):
            if shape_value == 0:
                continue
            candidate = module.guided_ridge._guided_ridge_curve_2d_preview_points(
                negative_projected, float(shape_value)
            )
            candidate_points = candidate["preview"]
            assert candidate_points != candidate["raw"]
            turns = [
                _turning_angle(
                    candidate_points[index] - candidate_points[index - 1],
                    candidate_points[index + 1] - candidate_points[index],
                )
                for index in range(1, len(candidate_points) - 1)
            ]
            assert max(turns) < 0.80
            nonzero_shape_metrics[shape_value] = max(turns)

        def _perpendicular_stats(values, raw_values):
            components = []
            for index in range(1, len(raw_values) - 1):
                tangent = raw_values[index + 1] - raw_values[index - 1]
                normal = Vector((-tangent.y, tangent.x)).normalized()
                components.append(abs((values[index] - raw_values[index]).dot(normal)))
            return sum(components) / len(components), max(components)

        negative_mean_perp, negative_max_perp = _perpendicular_stats(
            negative_displayed, negative_expected["raw"]
        )
        positive_mean_perp, positive_max_perp = _perpendicular_stats(
            displayed, expected["raw"]
        )
        attenuated_mean_perp, attenuated_max_perp = _perpendicular_stats(
            inscribed_expected["preview"], negative_expected["raw"]
        )
        clean, cumulative, total = module.guided_ridge._guided_ridge_curve_2d_arc_data(negative_projected)
        base_count = max(64, min(512, int(total / 3.0) + len(clean) * 2))
        parameters = [
            distance / total
            for distance in module.guided_ridge._guided_ridge_curve_output_parameters(
                total, cumulative, base_count
            )
        ]
        reevaluated = negative_expected["preview"]
        negative_direct_error = max(
            (negative_displayed[index] - reevaluated[index]).length
            for index in range(len(negative_displayed))
        )
        joins = [negative_draw_metrics["max_join_error"]]
        negative_max_turn = max(
            _turning_angle(
                negative_displayed[index] - negative_displayed[index - 1],
                negative_displayed[index + 1] - negative_displayed[index],
            )
            for index in range(1, len(negative_displayed) - 1)
        )
        # The raw-envelope-aware gain intentionally gives the emphasized
        # screen curve a larger broad lobe than the previous subtle fit.  Keep
        # the same no-cusp bound used by the actual nonzero sweep.
        assert negative_max_turn < 0.80
        negative_turn_records = [
            (
                _turning_angle(
                    negative_displayed[index] - negative_displayed[index - 1],
                    negative_displayed[index + 1] - negative_displayed[index],
                ),
                index,
            )
            for index in range(1, len(negative_displayed) - 1)
        ]
        negative_turn_value, negative_turn_index = max(negative_turn_records)
        negative_turn_parameter = negative_turn_index / max(len(negative_displayed) - 1, 1)
        negative_turn_lengths = (
            (negative_displayed[negative_turn_index] - negative_displayed[negative_turn_index - 1]).length,
            (negative_displayed[negative_turn_index + 1] - negative_displayed[negative_turn_index]).length,
        )
        negative_min_tangent = min(
            min(
                (negative_expected["preview"][1]
                 - negative_expected["preview"][0]).length,
                (negative_expected["preview"][-1]
                 - negative_expected["preview"][-2]).length,
            ),
            min(
                (negative_displayed[index + 1] - negative_displayed[index]).length
                for index in range(len(negative_displayed) - 1)
            ),
        )
        knot_clearance = min(
            min((point - knot).length for point in negative_displayed)
            for knot in projected[1:-1]
        )
        return {
            "passed": True,
            "draw_input_is_bezier": True,
            "old_progression_scale": old_scale,
            "max_turning_angle_radians": max_angle,
            "outlier_clearance_px": min((point - outlier).length for point in displayed),
            "plus99_to_plus100_max_delta_px": max(
                (displayed[index] - before_100[index]).length
                for index in range(len(displayed))
            ),
            "negative_bezier_distinct": True,
            "negative_displacement_sum": negative_displacement,
            "positive_displacement_sum": positive_displacement,
            "negative_failure_is_explicit": True,
            "straight_neutral": True,
            "negative_sample_count": len(negative_displayed),
            "negative_mean_perpendicular_px": negative_mean_perp,
            "negative_max_perpendicular_px": negative_max_perp,
            "positive_mean_perpendicular_px": positive_mean_perp,
            "positive_max_perpendicular_px": positive_max_perp,
            "amplified_mean_perpendicular_px": positive_mean_perp,
            "amplified_max_perpendicular_px": positive_max_perp,
            "attenuated_mean_perpendicular_px": attenuated_mean_perp,
            "attenuated_max_perpendicular_px": attenuated_max_perp,
            "amplified_attenuated_mean_ratio": positive_mean_perp / max(attenuated_mean_perp, 1.0e-12),
            "negative_positive_mean_perp_ratio": negative_mean_perp / max(positive_mean_perp, 1.0e-12),
            "negative_positive_max_perp_ratio": negative_max_perp / max(positive_max_perp, 1.0e-12),
            "raw_chord_bulge_px": abs(raw_bulge),
            "amplified_chord_bulge_px": abs(positive_bulge),
            "attenuated_chord_bulge_px": abs(negative_bulge),
            "amplified_raw_bulge_ratio": abs(positive_bulge) / max(abs(raw_bulge), 1.0e-12),
            "attenuated_raw_bulge_ratio": abs(negative_bulge) / max(abs(raw_bulge), 1.0e-12),
            "negative_knot_clearance_px": knot_clearance,
            "negative_max_turning_angle_radians": negative_max_turn,
            "negative_turn_argmax_index": negative_turn_index,
            "negative_turn_normalized_parameter": negative_turn_parameter,
            "negative_turn_neighbor_lengths_px": negative_turn_lengths,
            "negative_turn_bezier_segment_index": 0,
            "negative_turn_bezier_local_t": negative_turn_parameter,
            "negative_min_tangent_length_px": negative_min_tangent,
            "negative_turn_argmax_value": negative_turn_value,
            "negative_direct_bezier_error_px": negative_direct_error,
            "negative_c1_join_error_px": max(joins),
            "negative_c1_join_angle_radians": negative_draw_metrics["max_join_angle"],
            "amplified_c1_join_error_px": initial_draw_metrics["max_join_error"],
            "amplified_c1_join_angle_radians": initial_draw_metrics["max_join_angle"],
            "drawn_bezier_segment_count": negative_draw_metrics["segment_count"],
            "negative_endpoint_error_px": max(
                (negative_displayed[0] - projected[0]).length,
                (negative_displayed[-1] - projected[-1]).length,
            ),
            # The fixture predates the final sign naming; these aliases make
            # Keep the legacy metric aliases explicit after Shape+ became
            # Amplified; the fixture still reports both names for consumers.
            "emphasized_sample_count": len(negative_displayed),
            "emphasized_mean_perpendicular_px": negative_mean_perp,
            "emphasized_max_perpendicular_px": negative_max_perp,
            "emphasized_max_turning_angle_radians": negative_max_turn,
            "emphasized_direct_bezier_error_px": negative_direct_error,
            "nonzero_shape_max_turn_radians": max(nonzero_shape_metrics.values()),
            "nonzero_shape_count": len(nonzero_shape_metrics),
            "actual_screen_shape_checks": screen_shape_checks,
            "actual_drawn_bezier_verified": True,
            "endpoints_exact": True,
            "narrow_arch_raw_max_px": narrow_raw_max,
            "narrow_arch_plus_max_px": narrow_plus_max,
            "narrow_arch_minus_max_px": narrow_minus_max,
            "narrow_arch_raw_q75_px": narrow_raw_q75,
            "narrow_arch_plus_q75_px": narrow_plus_q75,
            "narrow_arch_minus_q75_px": narrow_minus_q75,
        }
    finally:
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = original_projection
        module.guided_ridge._guided_ridge_overlay_tag = original_overlay
        module.runtime.guided_ridge_state = original_state
        module.guided_ridge._guided_ridge_curve_2d_opposite_bezier = original_opposite
        module.guided_ridge._guided_ridge_curve_2d_wave_family = original_wave


def _guided_ridge_prototype_face_set_limit_fixture(module):
    """Step 1 starts on large/no-Face-Set meshes without legacy preprocessing."""
    import bpy
    from mathutils import Vector

    guided_class = module.VIEW3D_OT_mesh_focus_guided_ridge
    original_state = module.runtime.guided_ridge_state
    original_hit = module.guided_ridge._guided_ridge_prototype_surface_hit
    original_coord = module.guided_ridge._sculpt_cursor_region_coordinate
    original_ray = module.guided_ridge._guided_ridge_ray
    original_signature = module.guided_ridge._guided_ridge_context_signature
    original_conflict = module.guided_ridge._guided_ridge_conflict_reason
    original_overlay = module.guided_ridge._guided_ridge_overlay_tag
    original_prepare = module.guided_ridge._guided_ridge_prepare_snapshot_steps
    original_add = bpy.types.SpaceView3D.draw_handler_add
    original_remove = bpy.types.SpaceView3D.draw_handler_remove
    original_limit = module.guided_ridge.GUIDED_RIDGE_MAX_CANDIDATE_FACES
    mesh = bpy.data.meshes.new("_mfo_guided_ridge_prototype_mesh")
    mesh.from_pydata(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.update()
    obj = bpy.data.objects.new("_mfo_guided_ridge_prototype_object", mesh)
    pointers = iter(range(100, 200))

    class _Pointer:
        def __init__(self, **values):
            self.__dict__.update(values)
            self._pointer = next(pointers)

        def as_pointer(self):
            return self._pointer

    class _WM:
        def event_timer_add(self, *_args, **_kwargs):
            return object()

        def event_timer_remove(self, _timer):
            return None

        def modal_handler_add(self, _operator):
            return None

    area = _Pointer(type="VIEW_3D")
    region = _Pointer(type="WINDOW", x=0, y=0, width=200, height=200)
    space = _Pointer(type="VIEW_3D", region_3d=object())
    window = _Pointer()
    context = _Pointer(
        area=area,
        region=region,
        space_data=space,
        window=window,
        window_manager=_WM(),
        scene=_Pointer(),
        active_object=obj,
        mode="SCULPT",
    )
    context_signature = {
        "area_pointer": area.as_pointer(),
        "region_pointer": region.as_pointer(),
        "window_pointer": window.as_pointer(),
        "space_pointer": space.as_pointer(),
        "scene_pointer": context.scene.as_pointer(),
        "active_object_pointer": int(obj.as_pointer()),
        "mesh_pointer": int(mesh.as_pointer()),
        "mode": "SCULPT",
        "space_type": "VIEW_3D",
    }
    hit_points = [Vector((0.2, 0.2, 0.0)), Vector((0.7, 0.2, 0.0))]
    hit_index = [0]

    def prototype_hit(_context, _coord):
        point = hit_points[min(hit_index[0], len(hit_points) - 1)]
        hit_index[0] += 1
        return obj, 0, -1, point.copy(), Vector((0.0, 0.0, 1.0)), Vector((50.0, 50.0))

    prepare_calls = []

    def fail_if_prepared(*_args, **_kwargs):
        prepare_calls.append(True)
        raise AssertionError("prototype route must not build the Face Set snapshot")

    try:
        # No .sculpt_face_set attribute is present.  poll must still accept
        # the surface-only Step 1 route.
        assert module.guided_ridge._guided_ridge_face_set_attribute(obj) is None
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.poll(context)
        module.guided_ridge._guided_ridge_prototype_surface_hit = prototype_hit
        module.guided_ridge._sculpt_cursor_region_coordinate = lambda *_args: Vector((50.0, 50.0))
        module.guided_ridge._guided_ridge_ray = lambda *_args: (Vector((0.0, 0.0, 1.0)), Vector((0.0, 0.0, -1.0)))
        module.guided_ridge._guided_ridge_context_signature = lambda _context: dict(context_signature)
        module.guided_ridge._guided_ridge_conflict_reason = lambda _context: None
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        module.guided_ridge._guided_ridge_prepare_snapshot_steps = fail_if_prepared
        bpy.types.SpaceView3D.draw_handler_add = staticmethod(lambda *_args, **_kwargs: object())
        bpy.types.SpaceView3D.draw_handler_remove = staticmethod(lambda *_args, **_kwargs: None)

        reports = []
        operator = SimpleNamespace(
            poll=guided_class.poll,
            report=lambda _kind, message: reports.append(message),
        )
        event = SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=50, mouse_region_y=50)
        launch_started = time.perf_counter()
        result = module.VIEW3D_OT_mesh_focus_guided_ridge.invoke(operator, context, event)
        launch_seconds = time.perf_counter() - launch_started
        assert result == {"RUNNING_MODAL"}
        state = module.runtime.guided_ridge_state
        assert state is not None and state["phase"] == "ready"
        assert state["prototype_route_only"] is True
        assert state["snapshot"] is None
        assert len(state["controls"]) == 1
        assert state["prepare_job"] is None

        # The initial press/release is guarded, then a second visible-mesh hit
        # adds a point and Enter reaches the non-mutating curve preview.
        proxy_context = context
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            operator, proxy_context, SimpleNamespace(type="LEFTMOUSE", value="RELEASE", mouse_region_x=50, mouse_region_y=50)
        ) == {"RUNNING_MODAL"}
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            operator, proxy_context, SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=100, mouse_region_y=100)
        ) == {"RUNNING_MODAL"}
        assert len(state["controls"]) == 2
        before_coordinates = [tuple(point) for point in state["controls"]]
        assert module.VIEW3D_OT_mesh_focus_guided_ridge.modal(
            operator, proxy_context, SimpleNamespace(type="ENTER", value="PRESS", mouse_region_x=100, mouse_region_y=100)
        ) == {"RUNNING_MODAL"}
        assert state["phase"] == "curve_preview"
        assert [tuple(point) for point in state["controls"]] == before_coordinates
        assert state["snapshot"] is None

        # A Face Set that would be over the legacy limit must take the same
        # prototype route-only path rather than constructing that snapshot.
        attr = mesh.attributes.new(".sculpt_face_set", "INT", "FACE")
        attr.data[0].value = 7
        module.guided_ridge.GUIDED_RIDGE_MAX_CANDIDATE_FACES = 0
        module.guided_ridge._guided_ridge_cancel(state, "test-prototype-cleanup")
        hit_index[0] = 0
        second_invoke = module.VIEW3D_OT_mesh_focus_guided_ridge.invoke(operator, context, event)
        assert second_invoke == {"RUNNING_MODAL"}
        oversized_state = module.runtime.guided_ridge_state
        assert oversized_state is not None and oversized_state["phase"] == "ready"
        assert oversized_state["prototype_route_only"] is True
        assert oversized_state["snapshot"] is None
        module.guided_ridge._guided_ridge_cancel(oversized_state, "test-oversized-cleanup")

        # The legacy snapshot guard remains intact when directly exercised.
        legacy_snapshot, legacy_reason = module.guided_ridge._guided_ridge_prepare_snapshot_sync(obj, 0, 7)
        assert legacy_snapshot is None
        assert "50,000" in (legacy_reason or "")

        return {
            "passed": True,
            "poll_without_face_set": True,
            "oversized_guard_bypassed_for_step1": True,
            "oversized_face_set_route_ready": True,
            "route_ready_without_snapshot": True,
            "preview_reached": True,
            "legacy_guard_preserved": True,
            "mesh_unchanged": True,
            "prepare_generator_calls": len(prepare_calls),
            "launch_seconds": float(launch_seconds),
            "reports": tuple(reports),
        }
    finally:
        module.guided_ridge.GUIDED_RIDGE_MAX_CANDIDATE_FACES = original_limit
        module.guided_ridge._guided_ridge_prepare_snapshot_steps = original_prepare
        module.guided_ridge._guided_ridge_prototype_surface_hit = original_hit
        module.guided_ridge._sculpt_cursor_region_coordinate = original_coord
        module.guided_ridge._guided_ridge_ray = original_ray
        module.guided_ridge._guided_ridge_context_signature = original_signature
        module.guided_ridge._guided_ridge_conflict_reason = original_conflict
        module.guided_ridge._guided_ridge_overlay_tag = original_overlay
        module.runtime.guided_ridge_state = original_state
        bpy.types.SpaceView3D.draw_handler_add = original_add
        bpy.types.SpaceView3D.draw_handler_remove = original_remove
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def _guided_ridge_prototype_context_fixture(module):
    """Step 1 cancels rather than sampling a changed object or View3D."""
    import bpy
    from mathutils import Vector

    guided_class = module.VIEW3D_OT_mesh_focus_guided_ridge
    original_state = module.runtime.guided_ridge_state
    original_hit = module.guided_ridge._guided_ridge_prototype_surface_hit
    original_coord = module.guided_ridge._sculpt_cursor_region_coordinate
    original_ray = module.guided_ridge._guided_ridge_ray
    original_signature = module.guided_ridge._guided_ridge_context_signature
    original_conflict = module.guided_ridge._guided_ridge_conflict_reason
    original_overlay = module.guided_ridge._guided_ridge_overlay_tag
    original_add = bpy.types.SpaceView3D.draw_handler_add
    original_remove = bpy.types.SpaceView3D.draw_handler_remove
    mesh = bpy.data.meshes.new("_mfo_guided_ridge_context_mesh")
    mesh.from_pydata(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (), ((0, 1, 2),)
    )
    mesh.update()
    replacement_mesh = bpy.data.meshes.new("_mfo_guided_ridge_context_replacement")
    replacement_mesh.from_pydata(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)), (), ((0, 1, 2),)
    )
    replacement_mesh.update()
    obj = bpy.data.objects.new("_mfo_guided_ridge_context_object", mesh)
    other_mesh = bpy.data.meshes.new("_mfo_guided_ridge_other_mesh")
    other_mesh.from_pydata(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (), ((0, 1, 2),)
    )
    other_mesh.update()
    other_obj = bpy.data.objects.new("_mfo_guided_ridge_other_object", other_mesh)
    pointers = iter(range(1000, 1100))
    active_handlers = set()
    removed_timers = []

    class _Pointer:
        def __init__(self, **values):
            self.__dict__.update(values)
            self._pointer = next(pointers)

        def as_pointer(self):
            return self._pointer

    class _WM:
        def event_timer_add(self, *_args, **_kwargs):
            token = object()
            self.timer = token
            return token

        def event_timer_remove(self, token):
            removed_timers.append(token)

        def modal_handler_add(self, _operator):
            return None

    area = _Pointer(type="VIEW_3D")
    region = _Pointer(type="WINDOW", x=0, y=0, width=200, height=200)
    space = _Pointer(type="VIEW_3D", region_3d=object())
    window = _Pointer()
    scene = _Pointer()
    context = _Pointer(
        area=area,
        region=region,
        space_data=space,
        window=window,
        window_manager=_WM(),
        scene=scene,
        active_object=obj,
        mode="SCULPT",
    )
    signature = {
        "area_pointer": area.as_pointer(),
        "region_pointer": region.as_pointer(),
        "window_pointer": window.as_pointer(),
        "space_pointer": space.as_pointer(),
        "scene_pointer": scene.as_pointer(),
        "active_object_pointer": int(obj.as_pointer()),
        "mesh_pointer": int(mesh.as_pointer()),
        "mode": "SCULPT",
        "space_type": "VIEW_3D",
    }
    hit_points = [Vector((0.2, 0.2, 0.0)), Vector((0.7, 0.2, 0.0))]
    hit_index = [0]
    hit_override = [None]

    def prototype_hit(_context, _coord):
        if hit_override[0] is not None:
            return hit_override[0], 0, -1, hit_points[0].copy(), Vector((0.0, 0.0, 1.0)), Vector((50.0, 50.0))
        point = hit_points[min(hit_index[0], len(hit_points) - 1)]
        hit_index[0] += 1
        return obj, 0, -1, point.copy(), Vector((0.0, 0.0, 1.0)), Vector((50.0, 50.0))

    def add_handler(*_args, **_kwargs):
        token = object()
        active_handlers.add(token)
        return token

    def remove_handler(token, *_args, **_kwargs):
        active_handlers.discard(token)

    try:
        module.guided_ridge._guided_ridge_prototype_surface_hit = prototype_hit
        module.guided_ridge._sculpt_cursor_region_coordinate = lambda *_args: Vector((50.0, 50.0))
        module.guided_ridge._guided_ridge_ray = lambda *_args: (Vector((0.0, 0.0, 1.0)), Vector((0.0, 0.0, -1.0)))
        module.guided_ridge._guided_ridge_context_signature = lambda _context: dict(signature)
        module.guided_ridge._guided_ridge_conflict_reason = lambda _context: None
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        bpy.types.SpaceView3D.draw_handler_add = staticmethod(add_handler)
        bpy.types.SpaceView3D.draw_handler_remove = staticmethod(remove_handler)
        reports = []
        operator = SimpleNamespace(
            poll=guided_class.poll,
            report=lambda _kind, message: reports.append(message),
        )
        launch = SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=50, mouse_region_y=50)
        release = SimpleNamespace(type="LEFTMOUSE", value="RELEASE", mouse_region_x=50, mouse_region_y=50)
        click = SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=100, mouse_region_y=100)
        timer_event = lambda token: SimpleNamespace(type="TIMER", value=None, timer=token)
        baseline = [tuple(vertex.co) for vertex in mesh.vertices]

        def invoke_again():
            hit_index[0] = 0
            assert guided_class.invoke(operator, context, launch) == {"RUNNING_MODAL"}
            active = module.runtime.guided_ridge_state
            assert active is not None and active.get("timer") is not None
            guided_class.modal(operator, context, release)
            return active

        state = invoke_again()
        context.active_object = other_obj
        assert guided_class.modal(operator, context, click) == {"CANCELLED"}
        assert module.runtime.guided_ridge_state is None
        assert len(state["controls"]) == 1
        assert not active_handlers

        context.active_object = obj
        state = invoke_again()
        hit_override[0] = other_obj
        assert guided_class.modal(operator, context, click) == {"CANCELLED"}
        assert len(state["controls"]) == 1
        assert module.runtime.guided_ridge_state is None and not active_handlers
        hit_override[0] = None

        state = invoke_again()
        context.mode = "OBJECT"
        assert guided_class.modal(operator, context, timer_event(state["timer"])) == {"CANCELLED"}
        assert module.runtime.guided_ridge_state is None and not active_handlers
        context.mode = "SCULPT"

        state = invoke_again()
        original_data = obj.data
        obj.data = replacement_mesh
        assert guided_class.modal(operator, context, timer_event(state["timer"])) == {"CANCELLED"}
        assert module.runtime.guided_ridge_state is None and not active_handlers
        obj.data = original_data

        state = invoke_again()
        original_region_type = region.type
        region.type = "UI"
        assert guided_class.modal(operator, context, timer_event(state["timer"])) == {"CANCELLED"}
        assert module.runtime.guided_ridge_state is None and not active_handlers
        region.type = original_region_type

        state = invoke_again()
        assert guided_class.modal(operator, context, timer_event(state["timer"])) == {"RUNNING_MODAL"}
        assert guided_class.modal(operator, context, click) == {"RUNNING_MODAL"}
        assert len(state["controls"]) == 2
        assert [tuple(vertex.co) for vertex in mesh.vertices] == baseline
        assert guided_class.modal(operator, context, SimpleNamespace(type="ESC", value="PRESS")) == {"CANCELLED"}
        assert module.runtime.guided_ridge_state is None and not active_handlers
        return {
            "passed": True,
            "context_identity_guard": True,
            "other_object_hit_rejected": True,
            "mode_mesh_area_guard": True,
            "monitor_timer_cleanup": not removed_timers or module.runtime.guided_ridge_state is None,
            "normal_route_unchanged": True,
            "mesh_unchanged": True,
            "reports": tuple(reports),
        }
    finally:
        module.runtime.guided_ridge_state = original_state
        module.guided_ridge._guided_ridge_prototype_surface_hit = original_hit
        module.guided_ridge._sculpt_cursor_region_coordinate = original_coord
        module.guided_ridge._guided_ridge_ray = original_ray
        module.guided_ridge._guided_ridge_context_signature = original_signature
        module.guided_ridge._guided_ridge_conflict_reason = original_conflict
        module.guided_ridge._guided_ridge_overlay_tag = original_overlay
        bpy.types.SpaceView3D.draw_handler_add = original_add
        bpy.types.SpaceView3D.draw_handler_remove = original_remove
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.objects.remove(other_obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        bpy.data.meshes.remove(replacement_mesh)
        bpy.data.meshes.remove(other_mesh)


def _guided_ridge_curve_cache_fixture(module):
    """Screen cache skips idle fitting and never leaks an old-view curve."""
    from mathutils import Vector

    original_projection = module.guided_ridge._guided_ridge_project_3d_to_region_2d
    original_signature = module.guided_ridge._guided_ridge_context_signature
    original_overlay = module.guided_ridge._guided_ridge_overlay_tag
    original_helper = module.guided_ridge._guided_ridge_curve_2d_preview_points
    original_state = module.runtime.guided_ridge_state
    marker = [0]
    fit_calls = [0]
    try:
        route = [Vector((float(index), 18.0 * math.sin(index * 0.035), 0.0)) for index in range(520)]
        context = SimpleNamespace(
            region=SimpleNamespace(width=800, height=800),
            space_data=SimpleNamespace(region_3d=SimpleNamespace()),
            preferences=SimpleNamespace(system=SimpleNamespace(ui_scale=1.0)),
        )
        base_signature = {
            "area_pointer": 1, "region_pointer": 2, "window_pointer": 3,
            "space_pointer": 4, "scene_pointer": 5,
            "active_object_pointer": 6, "mesh_pointer": 7,
            "mode": "SCULPT", "space_type": "VIEW_3D",
        }
        module.guided_ridge._guided_ridge_context_signature = lambda _context: dict(
            base_signature, view_marker=marker[0]
        )
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = (
            lambda _region, _region_3d, point: Vector((point.x + 20.0, point.y + 300.0))
        )

        def counted_preview(points, smoothing):
            fit_calls[0] += 1
            return original_helper(points, smoothing)

        module.guided_ridge._guided_ridge_curve_2d_preview_points = counted_preview
        state = {
            "phase": "curve_preview",
            "curve_route": [point.copy() for point in route],
            "curve_smoothing": 100.0,
            "curve_screen_generation": 0,
            "context_signature": dict(base_signature, view_marker=0),
            "curve_route_revision": 1,
        }
        module.runtime.guided_ridge_state = state
        first_started = time.perf_counter()
        assert module.guided_ridge._guided_ridge_curve_update_screen_preview(context, state)
        first_refresh_seconds = time.perf_counter() - first_started
        first_fit_calls = fit_calls[0]
        assert first_fit_calls == 1
        assert state["curve_screen_sample_count"] <= module.GUIDED_RIDGE_CURVE_NONZERO_MAX_SAMPLES
        assert state["curve_screen_fit_segment_count"] > 0
        nonzero_sample_count = int(state["curve_screen_sample_count"])
        idle_started = time.perf_counter()
        for _idle_tick in range(100):
            assert module.guided_ridge._guided_ridge_curve_refresh_projection(context, state)
        idle_100_seconds = time.perf_counter() - idle_started
        assert fit_calls[0] == first_fit_calls
        assert state["curve_idle_cache_hits"] >= 100

        # A same-view candidate failure retains the exact prior generated
        # display, while a changed projection may show only its current raw
        # route and must clear the old Bezier data.
        prior_display = [point.copy() for point in state["curve_screen_preview"]]
        state["curve_smoothing"] = 80.0
        module.guided_ridge._guided_ridge_curve_2d_preview_points = (
            lambda _points, _smoothing: {
                "raw": [point.copy() for point in prior_display],
                "preview": [point.copy() for point in prior_display],
                "smooth": [point.copy() for point in prior_display],
                "bezier_segments": [], "preview_bezier_segments": [],
                "opposite_status": "failed", "opposite_warning": "forced failure",
                "sample_count": len(prior_display), "fit_segment_count": 0,
            }
        )
        assert module.guided_ridge._guided_ridge_curve_update_screen_preview(context, state)
        assert state["curve_shape_status"] == "retained"
        assert state["curve_screen_preview"] == prior_display
        marker[0] = 1
        assert module.guided_ridge._guided_ridge_curve_refresh_projection(context, state)
        assert state["curve_shape_status"] == "failed"
        assert not state["curve_screen_bezier_segments"]
        assert "current view" in state["curve_shape_warning"]

        module.guided_ridge._guided_ridge_curve_2d_preview_points = counted_preview
        state["curve_smoothing"] = 0.0
        assert module.guided_ridge._guided_ridge_curve_refresh_projection(context, state)
        assert state["curve_screen_cache_status"] == "valid"
        assert fit_calls[0] >= 2

        # A complete projection with one or more points outside the current
        # region is still a useful raw route.  The preview must not go blank;
        # fitting is withheld until the route is fully visible.
        outside_state = {
            "phase": "curve_preview",
            "curve_route": [Vector((0.0, 0.0, 0.0)), Vector((10.0, 6.0, 0.0)), Vector((20.0, 0.0, 0.0))],
            "curve_smoothing": 100.0,
            "curve_screen_generation": 0,
            "context_signature": dict(base_signature, view_marker=marker[0]),
            "curve_route_revision": 2,
        }
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = (
            lambda _region, _region_3d, point: Vector((900.0 + point.x, 300.0 + point.y))
        )
        assert not module.guided_ridge._guided_ridge_curve_update_screen_preview(context, outside_state)
        assert len(outside_state["curve_screen_raw"]) == 3
        assert outside_state["curve_screen_preview"] == outside_state["curve_screen_raw"]
        assert not outside_state["curve_screen_bezier_segments"]
        assert "projected raw route" in outside_state["curve_shape_warning"]

        collapsed_state = {
            "phase": "curve_preview",
            "curve_route": [Vector((0.0, 0.0, 0.0)), Vector((10.0, 6.0, 0.0)), Vector((20.0, 0.0, 0.0))],
            "curve_smoothing": 50.0,
            "curve_screen_generation": 0,
            "context_signature": dict(base_signature, view_marker=marker[0]),
            "curve_route_revision": 3,
        }
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = (
            lambda _region, _region_3d, _point: Vector((50.0, 50.0))
        )
        assert not module.guided_ridge._guided_ridge_curve_update_screen_preview(context, collapsed_state)
        assert len(collapsed_state["curve_screen_raw"]) == 3
        assert collapsed_state["curve_screen_preview"] == collapsed_state["curve_screen_raw"]
        assert not collapsed_state["curve_screen_bezier_segments"]
        assert collapsed_state["curve_projection_unique_count"] == 1
        assert "collapsed" in collapsed_state["curve_shape_warning"]

        # An accidentally initialized signature must not turn an empty screen
        # state into a cache hit.  Clearing the arrays forces exactly one new
        # fit even though the signature itself is unchanged.
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = (
            lambda _region, _region_3d, point: Vector((point.x + 20.0, point.y + 300.0))
        )
        state["curve_screen_raw"] = []
        state["curve_screen_preview"] = []
        state["curve_screen_bezier_segments"] = []
        state["curve_screen_signature"] = module.guided_ridge._guided_ridge_curve_preview_signature(
            context, state
        )["full"]
        fit_before_empty_recovery = fit_calls[0]
        assert module.guided_ridge._guided_ridge_curve_refresh_projection(context, state)
        assert fit_calls[0] == fit_before_empty_recovery + 1
        assert len(state["curve_screen_raw"]) >= 2
        assert len(state["curve_screen_preview"]) >= 2
        return {
            "passed": True,
            "nonzero_sample_cap": nonzero_sample_count,
            "idle_fit_calls_unchanged": True,
            "idle_cache_hits": int(state["curve_idle_cache_hits"]),
            "same_view_last_valid_retained": True,
            "changed_view_old_curve_cleared": True,
            "view_recovery_refit": True,
            "outside_projection_raw_visible": True,
            "collapsed_projection_diagnosed": True,
            "empty_signature_recomputed": True,
            "route_knots": len(route),
            "first_refresh_seconds": first_refresh_seconds,
            "idle_100_seconds": idle_100_seconds,
        }
    finally:
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = original_projection
        module.guided_ridge._guided_ridge_context_signature = original_signature
        module.guided_ridge._guided_ridge_overlay_tag = original_overlay
        module.guided_ridge._guided_ridge_curve_2d_preview_points = original_helper
        module.runtime.guided_ridge_state = original_state


def _guided_ridge_curve_draw_callback_fixture(module):
    """Capture the actual POST_PIXEL callback's raw/generated draw batches."""
    from mathutils import Vector

    original_bpy = module.guided_ridge.bpy
    original_gpu = module.guided_ridge.gpu
    original_batch = module.guided_ridge.batch_for_shader
    original_shader_get = module.guided_ridge._guided_ridge_curve_2d_shader_get
    original_state = module.runtime.guided_ridge_state
    draw_calls = []

    class _Area:
        def as_pointer(self):
            return 901

    class _Context:
        area = _Area()

    class _Shader:
        def bind(self):
            draw_calls.append("bind")

        def uniform_float(self, _name, _value):
            return None

    class _Batch:
        def __init__(self, mode, data):
            self.mode = mode
            self.data = data

        def draw(self, _shader):
            draw_calls.append((self.mode, tuple(self.data["pos"])))

    class _GPUState:
        def blend_set(self, value):
            draw_calls.append(("blend", value))

        def line_width_set(self, value):
            draw_calls.append(("width", value))

    try:
        shader = _Shader()
        module.guided_ridge.bpy = SimpleNamespace(context=_Context())
        module.guided_ridge.gpu = SimpleNamespace(state=_GPUState())
        module.guided_ridge.batch_for_shader = lambda _shader, mode, data: _Batch(mode, data)
        module.guided_ridge._guided_ridge_curve_2d_shader_get = lambda: shader
        module.runtime.guided_ridge_state = {
            "active": True,
            "phase": "curve_preview",
            "area_key": 901,
            "curve_screen_raw": [Vector((10.0, 20.0)), Vector((40.0, 45.0))],
            "curve_screen_preview": [Vector((10.0, 20.0)), Vector((40.0, 60.0))],
            "curve_projection_warning": None,
        }
        module.guided_ridge._guided_ridge_draw_curve_screen()
        batches = [entry for entry in draw_calls if isinstance(entry, tuple) and entry[0] == "LINE_STRIP"]
        assert len(batches) == 2
        return {
            "passed": True,
            "post_pixel_raw_batch": True,
            "post_pixel_generated_batch": True,
            "batch_count": len(batches),
            "state_keys_match_callback": True,
        }
    finally:
        module.guided_ridge.bpy = original_bpy
        module.guided_ridge.gpu = original_gpu
        module.guided_ridge.batch_for_shader = original_batch
        module.guided_ridge._guided_ridge_curve_2d_shader_get = original_shader_get
        module.runtime.guided_ridge_state = original_state


def _guided_ridge_projection_dependency_fixture(module, shared_projection_identities):
    """Exercise only the add-on-local projection seam.

    The Blender ``bpy_extras.view3d_utils`` module is process-global.  A
    deliberately failing dependency is injected into the add-on-local
    reference and restored in ``finally``; the shared helper identities must
    remain byte-for-byte the same throughout the simulation.
    """
    original_local = module.guided_ridge._guided_ridge_project_3d_to_region_2d
    injected_calls = []

    def _injected_failure(*_args, **_kwargs):
        injected_calls.append(True)
        raise RuntimeError("injected local projection failure")

    try:
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = _injected_failure
        try:
            module.guided_ridge._guided_ridge_project_3d_to_region_2d(None, None, None)
        except RuntimeError as error:
            assert str(error) == "injected local projection failure"
    finally:
        module.guided_ridge._guided_ridge_project_3d_to_region_2d = original_local

    assert injected_calls == [True]
    assert module.guided_ridge._guided_ridge_project_3d_to_region_2d is original_local
    for name, identity in shared_projection_identities.items():
        import bpy_extras.view3d_utils as shared_view3d_utils

        assert getattr(shared_view3d_utils, name) is identity
    return {
        "passed": True,
        "injected_exception_observed": True,
        "local_dependency_restored": True,
        "shared_identities_unchanged": True,
    }


def _guided_ridge_native_asset_fixture(module):
    import bpy

    service = module.guided_ridge_curve_sculpt
    context = bpy.context
    if context.mode != "SCULPT":
        return {"skipped": "active Sculpt View3D context required"}
    window = next(iter(bpy.context.window_manager.windows), None)
    area = next((candidate for candidate in window.screen.areas if candidate.type == "VIEW_3D"), None) if window else None
    region = next((candidate for candidate in area.regions if candidate.type == "WINDOW"), None) if area else None
    if window is None or area is None or region is None:
        return {"skipped": "active Sculpt View3D context required"}
    native_context = SimpleNamespace(
        scene=bpy.context.scene, window=window, area=area, region=region,
        region_data=area.spaces.active.region_3d, view_layer=bpy.context.view_layer,
        temp_override=lambda **kwargs: bpy.context.temp_override(**kwargs),
    )
    sculpt = native_context.scene.tool_settings.sculpt
    before = service._asset_reference(sculpt)
    state = {"operator": SimpleNamespace(report=lambda *_args, **_kwargs: None)}
    try:
        ridge = service._activate_builtin_asset(native_context, sculpt, state, "RIDGE")
        ridge_ref = service._asset_reference(sculpt)
        assert ridge_ref["relative_asset_identifier"].endswith("/Brush/Pinch/Magnify")
        assert ridge.sculpt_brush_type == "PINCH"
        groove = service._activate_builtin_asset(native_context, sculpt, state, "GROOVE")
        groove_ref = service._asset_reference(sculpt)
        assert groove_ref["relative_asset_identifier"].endswith("/Brush/Crease Sharp")
        assert groove.sculpt_brush_type == "CREASE"
        return {"passed": True, "asset_activate": True, "ridge_asset": ridge_ref["relative_asset_identifier"], "groove_asset": groove_ref["relative_asset_identifier"], "no_custom_asset": True}
    finally:
        if before:
            assert service._restore_active_tool(native_context, {"asset_reference": before, "tool_id": None})


def _guided_ridge_native_apply_transaction_fixture(module):
    import bpy

    service = module.guided_ridge_curve_sculpt
    context = bpy.context
    window = next(iter(bpy.context.window_manager.windows), None)
    area = next((candidate for candidate in window.screen.areas if candidate.type == "VIEW_3D"), None) if window else None
    region = next((candidate for candidate in area.regions if candidate.type == "WINDOW"), None) if area else None
    if window is None or area is None or region is None:
        return {"skipped": "a View3D area is required"}
    if region is None or region.width < 160 or region.height < 160:
        return {"skipped": "View3D region too small"}
    original_object = context.view_layer.objects.active
    original_selected = tuple(context.selected_objects)
    original_mode = context.mode
    temp_mesh = bpy.data.meshes.new("MFO isolated native stroke mesh")
    temp_object = bpy.data.objects.new("MFO isolated native stroke object", temp_mesh)
    context.scene.collection.objects.link(temp_object)
    try:
        from bpy_extras import view3d_utils
        from mathutils import Vector

        size = 24
        rv3d = area.spaces.active.region_3d
        center_mouse = (float(region.width) * 0.5, float(region.height) * 0.5)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, center_mouse)
        ray_direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, center_mouse).normalized()
        plane_center = ray_origin + ray_direction * 3.0
        plane_right = (rv3d.view_rotation @ Vector((1.0, 0.0, 0.0))).normalized()
        plane_up = (rv3d.view_rotation @ Vector((0.0, 1.0, 0.0))).normalized()
        vertices = [
            tuple(plane_center + plane_right * (-0.75 + 1.5 * column / size) + plane_up * (-0.75 + 1.5 * row / size))
            for row in range(size + 1) for column in range(size + 1)
        ]
        faces = []
        for row in range(size):
            for column in range(size):
                base = row * (size + 1) + column
                faces.append((base, base + 1, base + size + 2, base + size + 1))
        temp_mesh.from_pydata(vertices, [], faces)
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for item in context.selected_objects:
            item.select_set(False)
        temp_object.select_set(True)
        context.view_layer.objects.active = temp_object
        bpy.ops.object.mode_set(mode="SCULPT")
        native_context = SimpleNamespace(
            scene=bpy.context.scene, window=window, area=area, region=region,
            region_data=area.spaces.active.region_3d, view_layer=bpy.context.view_layer,
            temp_override=lambda **kwargs: bpy.context.temp_override(**kwargs),
        )
        sculpt = native_context.scene.tool_settings.sculpt
        before_asset = service._asset_reference(sculpt)
        before = tuple(tuple(float(value) for value in vertex.co) for vertex in temp_mesh.vertices)
        cx, cy = float(region.width) * 0.5, float(region.height) * 0.5
        points = ((cx - 70.0, cy), (cx - 35.0, cy + 18.0), (cx, cy), (cx + 35.0, cy - 18.0), (cx + 70.0, cy))
        state = {
            "active": True, "phase": "curve_preview", "obj": temp_object,
            "preview_context": context, "curve_screen_preview": points,
            "curve_screen_cache_status": "valid", "curve_screen_signature": ("native", region.width, region.height),
            "guide": (tuple(plane_center),), "operator": SimpleNamespace(report=lambda *_args, **_kwargs: None),
        }
        ridge_ok = service.apply(native_context, state, "RIDGE")
        after_ridge = tuple(tuple(float(value) for value in vertex.co) for vertex in temp_mesh.vertices)
        ridge_changed = any(before[index] != after_ridge[index] for index in range(len(before)))
        groove_ok = service.apply(native_context, state, "GROOVE")
        after_groove = tuple(tuple(float(value) for value in vertex.co) for vertex in temp_mesh.vertices)
        groove_changed = any(after_ridge[index] != after_groove[index] for index in range(len(before)))
        assert service._asset_reference(sculpt) == before_asset
        service.cleanup(state)
        assert state.get("curve_sculpt_apply_active") is False
        return {"passed": bool(ridge_ok and groove_ok and ridge_changed and groove_changed), "ridge_finished": ridge_ok, "groove_finished": groove_ok, "ridge_mesh_changed": ridge_changed, "groove_mesh_changed": groove_changed, "stroke_float_points": True, "synchronous": True, "isolated_temp_mesh": True}
    finally:
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if temp_object.name in bpy.data.objects:
            bpy.data.objects.remove(temp_object, do_unlink=True)
        if temp_mesh.users == 0:
            bpy.data.meshes.remove(temp_mesh)
        for item in context.selected_objects:
            item.select_set(False)
        if original_object is not None and original_object.name in bpy.data.objects:
            original_object.select_set(True)
            context.view_layer.objects.active = original_object
            if original_mode != "OBJECT" and original_object.mode != original_mode:
                bpy.ops.object.mode_set(mode=original_mode)
        for item in original_selected:
            if item.name in bpy.data.objects:
                item.select_set(True)


def _guided_ridge_curve_sculpt_apply_fixture(module):
    """Exercise the real synchronous stroke seam, including Blender's flag."""
    service = module.guided_ridge_curve_sculpt
    original_bpy = service.bpy
    calls = []

    def _brush_stroke(**kwargs):
        calls.append(kwargs)
        return {"FINISHED"}

    fake_bpy = SimpleNamespace(
        ops=SimpleNamespace(sculpt=SimpleNamespace(brush_stroke=_brush_stroke)),
        context=SimpleNamespace(
            preferences=SimpleNamespace(addons={}),
        ),
    )
    context = SimpleNamespace(region=None, region_data=None)
    state = {"guide": [(0.0, 0.0, 0.0)]}
    try:
        service.bpy = fake_bpy
        result = service._apply_native_stroke(
            context,
            state,
            ((12.5, 20.25), (32.75, 48.5)),
        )
        assert result == {"FINISHED"}
        assert len(calls) == 1
        assert calls[0]["override_location"] is True
        assert all(
            isinstance(item["mouse"][0], float) and isinstance(item["mouse"][1], float)
            for item in calls[0]["stroke"]
        )
        return {
            "passed": True,
            "synchronous_brush_stroke": True,
            "override_location": True,
            "float_mouse_coordinates": True,
        }
    finally:
        service.bpy = original_bpy


def _guided_ridge_restore_retry_fixture(module):
    service = module.guided_ridge_curve_sculpt
    original_restore = service._restore_active_tool
    attempts = []
    state = {
        "curve_sculpt_restore": {
            "asset_reference": {"relative_asset_identifier": "Brush/Original"},
            "tool_id": "builtin.select_box",
        },
        "curve_sculpt_restore_pending": True,
    }

    def _restore(_context, _record):
        attempts.append(True)
        return len(attempts) > 1

    try:
        service._restore_active_tool = _restore
        assert not service.retry_restore(None, state)
        assert "curve_sculpt_restore" in state
        assert state["curve_sculpt_restore_pending"] is True
        assert service.retry_restore(None, state)
        assert "curve_sculpt_restore" not in state
        assert "curve_sculpt_restore_pending" not in state
        assert len(attempts) == 2
        return {"passed": True, "record_retained_on_failure": True, "retry_succeeded": True}
    finally:
        service._restore_active_tool = original_restore


def _durable_restore_fixture(module):
    """The restore record survives runtime teardown and is re-hydratable."""
    import bpy
    import copy
    lifecycle = module.lifecycle
    runtime = module.runtime
    key_a = "guided-ridge-durable-test-a"
    key_b = "guided-ridge-durable-test-b"
    namespace = bpy.app.driver_namespace
    original_root = getattr(lifecycle, "_PACKAGE_ROOT", "mesh_focus_orbit")
    production_key = "mesh_focus_orbit.guided_ridge.pending_restore"
    isolated_key = "mfo_toolbar_isolated.guided_ridge.pending_restore"
    sentinel = {"production-sentinel": {"value": 17, "nested": ["keep", 3]}}
    original_production = namespace.get(production_key)
    original_isolated = namespace.get(isolated_key)
    original_records = dict(runtime.pending_restores)
    original_timer = runtime.pending_restore_timer
    try:
        assert lifecycle.runtime is runtime, (id(lifecycle.runtime), id(runtime))
        runtime.modal_registry.clear()
        namespace[production_key] = copy.deepcopy(sentinel)
        lifecycle._PACKAGE_ROOT = "mfo_toolbar_isolated"
        callback = module.guided_ridge_curve_sculpt._restore_active_tool
        lifecycle.pending_restore_register(
            key_a,
            SimpleNamespace(),
            {"tool_id": "builtin.select_box", "workspace_name": "MFO durable test"},
            callback,
        )
        lifecycle.pending_restore_register(
            key_b,
            SimpleNamespace(),
            {"tool_id": "builtin.select_box", "workspace_name": "MFO durable test"},
            callback,
        )
        persisted = namespace.get(isolated_key, {})
        assert key_a in persisted and key_b in persisted
        assert namespace[production_key] == sentinel
        lifecycle.pending_restore_remove(key_a)
        assert key_a not in namespace.get(isolated_key, {})
        assert key_b in namespace.get(isolated_key, {})
        lifecycle.pending_restore_clear()
        assert key_b in namespace.get(isolated_key, {})
        runtime.pending_restores.clear()
        lifecycle.pending_restore_resume()
        assert key_b in runtime.pending_restores and key_a not in runtime.pending_restores
        lifecycle.pending_restore_remove(key_b)
        assert key_b not in namespace.get(isolated_key, {})
        return {
            "passed": True,
            "driver_namespace_persisted": True,
            "rehydrated": True,
            "removal_sync": True,
            "package_namespace_isolated": True,
            "production_sentinel_unchanged": namespace[production_key] == sentinel,
        }
    finally:
        lifecycle.pending_restore_remove(key_a)
        lifecycle.pending_restore_remove(key_b)
        runtime.pending_restores.clear()
        runtime.pending_restores.update(original_records)
        runtime.pending_restore_timer = original_timer
        lifecycle._PACKAGE_ROOT = original_root
        if original_production is None:
            namespace.pop(production_key, None)
        else:
            namespace[production_key] = original_production
        if original_isolated is None:
            namespace.pop(isolated_key, None)
        else:
            namespace[isolated_key] = original_isolated


def _smart_fill_terminal_failure_fixture(module):
    """Statically verify the real Smart Fill invoke failure cleanup contract.

    Native INVOKE_DEFAULT failure injection belongs to the disposable GUI
    runner. The background suite must not fabricate an operator or manually
    call the cleanup helper; it instead checks that the production invoke
    exception branch calls both lifecycle terminalization and the complete
    Smart Fill owner terminal helper exactly once.
    """
    source_path = pathlib.Path(module.registration.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    operator_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VIEW3D_OT_mesh_focus_local_face_set_grow"
    )
    invoke = next(
        node
        for node in operator_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "invoke"
    )
    helper_calls = [
        node
        for node in ast.walk(invoke)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_fill_preview_modal_owner_terminal"
    ]
    lifecycle_calls = [
        node
        for node in ast.walk(invoke)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "modal_terminal"
    ]
    assert len(helper_calls) == 1, (
        "Smart Fill invoke must have one idempotent owner-terminal call; "
        f"found {len(helper_calls)}"
    )
    assert len(lifecycle_calls) == 1, (
        "Smart Fill invoke failure must have one lifecycle terminal call; "
        f"found {len(lifecycle_calls)}"
    )
    return {
        "passed": True,
        "native_invoke": False,
        "native_invoke_note": "deferred to disabled isolated GUI path after Blender native crash",
        "owner_terminal_calls": len(helper_calls),
        "lifecycle_terminal_calls": len(lifecycle_calls),
    }


def _hidden_gui_entry_guard_fixture():
    """Prove the native GUI runner stops before any candidate side effect."""
    runner_path = pathlib.Path(__file__).with_name("test_smart_fill_modal_hidden_gui.py")
    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(runner_path))
    run_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    body_text = "\n".join(source.splitlines()[run_node.lineno - 1 : run_node.end_lineno])
    guard_pos = body_text.find("_NATIVE_GUI_OPT_IN")
    assert guard_pos >= 0, "native GUI runner has no common opt-in guard"
    for expression in (
        "importlib.import_module",
        "module.register()",
        "bpy.app.timers.register",
        "bpy.ops.object.mode_set",
    ):
        position = body_text.find(expression)
        if position >= 0:
            assert guard_pos < position, (
                f"native GUI opt-in guard must dominate {expression}"
            )
    return {
        "passed": True,
        "guard": "_NATIVE_GUI_OPT_IN",
        "side_effects_after_guard_only": True,
    }


def _modal_terminal_return_contract(module):
    """Statically enumerate terminal modal branches and require one owner path."""
    root = pathlib.Path(module.__file__).resolve().parent
    terminal_branches = []
    missing = []
    for source_path in sorted(root.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in {"modal", "invoke"}:
                continue
            returns_terminal = []
            has_modal_terminal = False
            has_terminal_helper = False
            has_modal_register = False
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    callee = child.func.attr if isinstance(child.func, ast.Attribute) else getattr(child.func, "id", "")
                    has_modal_terminal |= callee == "modal_terminal"
                    has_terminal_helper |= callee in {"_finish", "_finish_confirm"}
                    has_modal_register |= callee == "modal_register"
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Set):
                    values = {item.value for item in child.value.elts if isinstance(item, ast.Constant)}
                    values &= {"FINISHED", "CANCELLED"}
                    if values:
                        returns_terminal.append(tuple(sorted(values)))
            if returns_terminal:
                item = (str(source_path.relative_to(root)), node.name, tuple(returns_terminal))
                terminal_branches.append(item)
                if node.name == "invoke" and not has_modal_register:
                    continue
                if not (has_modal_terminal or has_terminal_helper):
                    missing.append(item)
    assert not missing, f"terminal branches without common owner cleanup: {missing!r}"
    return {"passed": True, "branch_count": len(terminal_branches), "missing": ()}


def _modal_owner_preflight_fixture(module):
    import bpy
    runtime = module.runtime
    lifecycle = module.lifecycle
    original = (
        runtime.guided_ridge_state,
        runtime.tube_preview_state,
        runtime.local_feature_brush_stroke_operator,
        runtime.local_feature_brush_pending_stroke,
    )
    original_registry = dict(runtime.modal_registry)
    original_serial = runtime.modal_registry_serial
    original_quiescence = dict(runtime.modal_quiescence_pending)
    original_quiescence_generation = runtime.modal_quiescence_generation
    original_quiescence_timer = runtime.modal_quiescence_timer
    try:
        modal_operators = [object() for _ in range(5)]
        modal_kinds = (
            "Smart Fill",
            "Guided Ridge",
            "Tube Shape",
            "Local Feature",
            "MFO Watcher",
        )
        tokens = [
            lifecycle.modal_register(operator, kind, session=index)
            for index, (operator, kind) in enumerate(zip(modal_operators, modal_kinds))
        ]
        assert set(lifecycle.modal_registry_owners()) == set(modal_kinds)
        assert module.registration._lifecycle_modal_preflight()["safe"] is False
        lifecycle.modal_request_cancel(operator=modal_operators[0], reason="test")
        assert lifecycle.modal_entry(operator=modal_operators[0]) is not None
        lifecycle.modal_terminal(operator=modal_operators[0])
        assert lifecycle.modal_entry(operator=modal_operators[0]) is None
        assert "Guided Ridge" in lifecycle.modal_registry_owners()
        for operator in modal_operators[1:]:
            lifecycle.modal_terminal(operator=operator)
        assert set(runtime.modal_registry) == set(original_registry)
        retired_operator = object()
        lifecycle.modal_register(retired_operator, "Smart Fill", session="grace")
        lifecycle.modal_terminal(
            operator=retired_operator,
            status="CANCELLED",
        )
        assert not lifecycle.modal_quiescence_is_safe()
        assert module.registration._lifecycle_modal_preflight()["safe"] is False
        lifecycle._modal_quiescence_tick()
        assert not lifecycle.modal_quiescence_is_safe()
        lifecycle._modal_quiescence_tick()
        assert lifecycle.modal_quiescence_is_safe()
        assert module.registration._lifecycle_modal_preflight()["safe"] is True
        runtime.guided_ridge_state = {"active": True}
        assert "Guided Ridge" in module.registration._active_modal_owners()
        runtime.guided_ridge_state = None
        runtime.tube_preview_state = {"active": True}
        assert "Tube Shape" in module.registration._active_modal_owners()
        runtime.tube_preview_state = None
        runtime.local_feature_brush_stroke_operator = object()
        assert "Local Feature" in module.registration._active_modal_owners()
        assert module.registration._lifecycle_modal_preflight()["safe"] is False
        return {
            "passed": True,
            "guided_ridge": True,
            "tube_shape": True,
            "local_feature": True,
            "all_modal_registry_owners": modal_kinds,
            "cancel_retains_until_terminal": True,
        }
    finally:
        (
            runtime.guided_ridge_state,
            runtime.tube_preview_state,
            runtime.local_feature_brush_stroke_operator,
            runtime.local_feature_brush_pending_stroke,
        ) = original
        runtime.modal_registry.clear()
        runtime.modal_registry.update(original_registry)
        runtime.modal_registry_serial = original_serial
        timer = runtime.modal_quiescence_timer
        if timer is not None:
            try:
                if bpy.app.timers.is_registered(lifecycle._modal_quiescence_tick):
                    bpy.app.timers.unregister(lifecycle._modal_quiescence_tick)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        runtime.modal_quiescence_pending.clear()
        runtime.modal_quiescence_pending.update(original_quiescence)
        runtime.modal_quiescence_generation = original_quiescence_generation
        runtime.modal_quiescence_timer = original_quiescence_timer


def _guided_ridge_curve_sculpt_transaction_fixture(module):
    guided_core = module.guided_ridge
    original_state = module.runtime.guided_ridge_state
    original_cancel = guided_core._guided_ridge_cancel
    cancelled = []
    data = SimpleNamespace(as_pointer=lambda: 22, vertices=(), polygons=())
    obj = SimpleNamespace(as_pointer=lambda: 11, data=data)
    state = {
        "active": True,
        "obj": obj,
        "curve_sculpt_apply_generation": 3,
        "curve_sculpt_apply_active": True,
        "curve_sculpt_expected_update_generation": 3,
    }
    depsgraph = SimpleNamespace(updates=[SimpleNamespace(id=obj)])
    try:
        module.runtime.guided_ridge_state = state
        guided_core._guided_ridge_cancel = lambda *_args, **_kwargs: cancelled.append(True)
        guided_core._on_guided_ridge_depsgraph_update(None, depsgraph)
        assert not cancelled
        state["curve_sculpt_apply_active"] = False
        guided_core._on_guided_ridge_depsgraph_update(None, depsgraph)
        assert cancelled
        return {"passed": True, "self_update_ignored_during_generation": True, "external_update_cancelled": True}
    finally:
        guided_core._guided_ridge_cancel = original_cancel
        module.runtime.guided_ridge_state = original_state


def run():
    module = _load_module()
    package_path = pathlib.Path(module.__file__).resolve()
    assert package_path.name == "__init__.py"
    assert package_path.parent.name == "mesh_focus_orbit"
    assert module.__package__ == "mfo_toolbar_isolated"
    assert callable(module.register) and callable(module.unregister)
    package_files = tuple(package_path.parent.rglob("*"))
    package_allowlist_extensions = {".py", ".dat", ".png"}
    distributable_files = tuple(
        item for item in package_files if item.is_file() and item.suffix in package_allowlist_extensions
    )
    forbidden_package_paths = tuple(
        str(item)
        for item in distributable_files
        if (
            "__pycache__" in item.parts
            or "diagnostics" in item.parts
            or item.name in {"asset_metadata_probe_results.json", "asset_preview_results.json"}
            or item.suffix == ".cache"
        )
    )
    assert not forbidden_package_paths
    runtime_owner = module.runtime
    for component in (
        module.foundation,
        module.guided_ridge,
        module.local_feature,
        module.tube_shape,
        module.smart_fill_geometry,
        module.smart_fill_preview,
        module.registration,
    ):
        assert getattr(component, "_runtime", None) is runtime_owner
    _runtime_sentinel = object()
    _runtime_previous = runtime_owner.fill_preview_state
    runtime_owner.fill_preview_state = _runtime_sentinel
    try:
        assert module.smart_fill_preview._runtime.fill_preview_state is _runtime_sentinel
    finally:
        runtime_owner.fill_preview_state = _runtime_previous
    runtime_ownership = {
        "passed": True,
        "canonical_owner": "mfo_toolbar_isolated.runtime",
        "components_checked": 7,
        "scalar_visibility": True,
    }
    assert getattr(module, "_MFO_TOOL_ICON_DIR", "")
    # The package is a real component split, not an exec-based compatibility
    # wrapper. Each child has its own importable namespace; the facade only
    # forwards deliberate legacy attributes.
    expected_components = (
        "foundation.py",
        "guided_ridge/curve_sculpt.py",
        "guided_ridge/core.py",
        "local_feature.py",
        "tube_shape.py",
        "smart_fill/geometry.py",
        "smart_fill/preview.py",
        "registration.py",
    )
    assert tuple(module._MFO_COMPONENT_FILES) == expected_components
    assert set(module._MFO_COMPONENT_ORIGINS) == set(expected_components)
    for relative_path in expected_components:
        component_origin = pathlib.Path(module._MFO_COMPONENT_ORIGINS[relative_path])
        assert component_origin.is_file(), relative_path
        assert component_origin.parent.name in {"mesh_focus_orbit", "guided_ridge", "smart_fill"}
    expected_support = ("config.py", "runtime.py", "viewport.py", "lifecycle.py")
    assert tuple(module._MFO_SUPPORT_FILES) == expected_support
    for relative_path in expected_support:
        assert pathlib.Path(module._MFO_SUPPORT_ORIGINS[relative_path]).is_file()
    assert module.runtime.active_states is module.foundation._active_states
    assert module.runtime.session_cleanup_areas is module.foundation._session_cleanup_areas
    assert module.lifecycle.remove_handler_identity.__module__ == module.lifecycle.__name__
    import bpy_extras.view3d_utils as shared_view3d_utils

    # Capture the canonical public projection helpers before any fixture runs.
    # No fixture is allowed to reload or monkeypatch this shared module.
    shared_projection_identities = {
        name: getattr(shared_view3d_utils, name)
        for name in (
            "location_3d_to_region_2d",
            "region_2d_to_origin_3d",
            "region_2d_to_vector_3d",
            "region_2d_to_location_3d",
        )
    }
    assert module.guided_ridge._guided_ridge_project_3d_to_region_2d is (
        shared_projection_identities["location_3d_to_region_2d"]
    )
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
    original_bpy = module.guided_ridge.bpy
    original_states = module.runtime.active_states
    original_deactivate = module.guided_ridge._deactivate_state
    original_face_set_state = module.guided_ridge._FaceSetOrbitState
    original_activate_face_set_session = module.guided_ridge._activate_face_set_session
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
        module.guided_ridge.bpy = SimpleNamespace(
            ops=SimpleNamespace(
                view3d=SimpleNamespace(mesh_focus_face_set_activate=_FakeActivation())
            )
        )
        module.runtime.active_states = {}
        module.guided_ridge._FaceSetOrbitState = _FakeFaceSetState
        module.guided_ridge._deactivate_state = lambda state: off_calls.append(state)
        self_proxy = SimpleNamespace(
            report=lambda *_args, **_kwargs: None,
            poll=module.guided_ridge.VIEW3D_OT_mesh_focus_face_set_tool.poll,
        )
        on_result = module.guided_ridge.VIEW3D_OT_mesh_focus_face_set_tool.invoke(
            self_proxy, dispatch_context, dispatch_event
        )
        assert on_result == {"FINISHED"}
        assert dispatch_calls and dispatch_calls[0][0][:2] == ("EXEC_DEFAULT", True)
        assert dispatch_calls[0][1] == {
            "mouse_region_x": 230,
            "mouse_region_y": 160,
            "use_click_coordinate": True,
        }
        module.runtime.active_states = {9123: _FakeFaceSetState()}
        off_result = module.guided_ridge.VIEW3D_OT_mesh_focus_face_set_tool.invoke(
            self_proxy, dispatch_context, dispatch_event
        )
        assert off_result == {"FINISHED"}
        assert len(off_calls) == 1

        activation_coordinate = []
        module.runtime.active_states = {}
        module.guided_ridge._activate_face_set_session = lambda _context, coordinate=None: (
            activation_coordinate.append(coordinate) or True
        )
        activation_self = SimpleNamespace(
            use_click_coordinate=True,
            mouse_region_x=230,
            mouse_region_y=160,
            poll=module.guided_ridge.VIEW3D_OT_mesh_focus_face_set_activate.poll,
        )
        assert module.guided_ridge.VIEW3D_OT_mesh_focus_face_set_activate.execute(
            activation_self, dispatch_context
        ) == {"FINISHED"}
        assert [tuple(value) for value in activation_coordinate] == [(230.0, 160.0)]
    finally:
        module.guided_ridge.bpy = original_bpy
        module.runtime.active_states = original_states
        module.guided_ridge._deactivate_state = original_deactivate
        module.guided_ridge._FaceSetOrbitState = original_face_set_state
        module.guided_ridge._activate_face_set_session = original_activate_face_set_session

    # Guided Ridge lifecycle: the WorkSpaceTool's initial LEFTMOUSE pair is
    # a start-only event.  It must not add a second point or commit, while the
    # first LMB after release adds a point and Enter enters Curve Preview.
    from mathutils import Vector

    guided_class = module.VIEW3D_OT_mesh_focus_guided_ridge
    original_guided_state = module.runtime.guided_ridge_state
    original_guided_coordinate = module.guided_ridge._sculpt_cursor_region_coordinate
    original_guided_hit = module.guided_ridge._guided_ridge_snapshot_hit
    original_guided_project = module.guided_ridge._guided_ridge_project_curve
    original_guided_overlay = module.guided_ridge._guided_ridge_overlay_tag
    guided_commit_calls = []
    try:
        guided_state = {
            "active": True,
            "operator": None,
            "phase": "ready",
            "curve_preview_session": True,
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
        module.runtime.guided_ridge_state = guided_state
        module.guided_ridge._sculpt_cursor_region_coordinate = lambda _context, _event: Vector((10.0, 10.0))
        module.guided_ridge._guided_ridge_snapshot_hit = lambda _context, _snapshot, _coord: (
            Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)), 0, 0.0
        )
        module.guided_ridge._guided_ridge_project_curve = lambda _snapshot, controls: (list(controls), None)
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None

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
        assert guided_class.modal(guided_proxy, guided_context, enter) == {"RUNNING_MODAL"}
        assert guided_state["phase"] == "curve_preview"
        assert not guided_commit_calls
        guided_lifecycle = {
            "passed": True,
            "first_controls": 1,
            "after_next_lmb_controls": 2,
            "enter_preview_only": True,
        }
    finally:
        module.runtime.guided_ridge_state = original_guided_state
        module.guided_ridge._sculpt_cursor_region_coordinate = original_guided_coordinate
        module.guided_ridge._guided_ridge_snapshot_hit = original_guided_hit
        module.guided_ridge._guided_ridge_project_curve = original_guided_project
        module.guided_ridge._guided_ridge_overlay_tag = original_guided_overlay

    # Preparation owns LMB/Enter while the timer is active, and Esc removes
    # the timer and generator state without requiring a mesh or scene fixture.
    original_guided_overlay = module.guided_ridge._guided_ridge_overlay_tag
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
        module.runtime.guided_ridge_state = prepare_state
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        lmb = SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_region_x=100, mouse_region_y=100)
        enter_prepare = SimpleNamespace(type="ENTER", value="PRESS", mouse_region_x=100, mouse_region_y=100)
        assert guided_class.modal(prepare_proxy, SimpleNamespace(), lmb) == {"RUNNING_MODAL"}
        assert guided_class.modal(prepare_proxy, SimpleNamespace(), enter_prepare) == {"RUNNING_MODAL"}
        assert prepare_state["controls"] == []
        assert guided_class.modal(
            prepare_proxy, SimpleNamespace(), SimpleNamespace(type="ESC", value="PRESS")
        ) == {"CANCELLED"}
        assert removed_timers and module.runtime.guided_ridge_state is None
        prepare_cleanup = {"passed": True, "timer_removed": True, "generator_cleared": True}
    finally:
        module.guided_ridge._guided_ridge_overlay_tag = original_guided_overlay
        module.runtime.guided_ridge_state = original_guided_state

    # A completed timer job transitions to Ready, seeds exactly the first
    # clicked control, removes its timer, and leaves later LMB/Enter handling
    # to the normal lifecycle tested above.
    original_prepare_context_matches = module.guided_ridge._guided_ridge_prepare_context_matches
    original_prepare_snapshot_hit_ray = module.guided_ridge._guided_ridge_snapshot_hit_ray
    original_prepare_context_signature = module.guided_ridge._guided_ridge_context_signature
    original_prepare_overlay = module.guided_ridge._guided_ridge_overlay_tag
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
        module.runtime.guided_ridge_state = ready_state
        module.guided_ridge._guided_ridge_prepare_context_matches = lambda _context, _state: True
        module.guided_ridge._guided_ridge_snapshot_hit_ray = lambda *_args: (
            Vector((4.0, 5.0, 6.0)), Vector((0.0, 0.0, 1.0)), 0, 0.0
        )
        module.guided_ridge._guided_ridge_context_signature = lambda _context: {"marker": 2}
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        assert module.guided_ridge._guided_ridge_process_prepare_timer(ready_context, ready_state) is True
        assert ready_state["phase"] == "prepare"
        assert module.guided_ridge._guided_ridge_process_prepare_timer(ready_context, ready_state) is True
        assert ready_state["phase"] == "ready"
        assert len(ready_state["controls"]) == 1
        assert ready_state["prepare_job"] is None
        assert ready_state["timer"] is None
        assert removed_ready_timers
        guided_ridge_ready = {"passed": True, "timer_removed": True, "seed_controls": 1}
    finally:
        module.guided_ridge._guided_ridge_prepare_context_matches = original_prepare_context_matches
        module.guided_ridge._guided_ridge_snapshot_hit_ray = original_prepare_snapshot_hit_ray
        module.guided_ridge._guided_ridge_context_signature = original_prepare_context_signature
        module.guided_ridge._guided_ridge_overlay_tag = original_prepare_overlay
        module.runtime.guided_ridge_state = original_guided_state

    # The first timer tick publishes the stage and returns to the event loop
    # before the generator enters the stage's expensive body.  The following
    # tick may perform that work.
    original_prepare_context_matches = module.guided_ridge._guided_ridge_prepare_context_matches
    original_prepare_overlay = module.guided_ridge._guided_ridge_overlay_tag
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
        module.runtime.guided_ridge_state = paint_state
        module.guided_ridge._guided_ridge_prepare_context_matches = lambda _context, _state: True
        module.guided_ridge._guided_ridge_overlay_tag = lambda _state: None
        assert module.guided_ridge._guided_ridge_process_prepare_timer(paint_context, paint_state) is True
        assert paint_state["prepare_stage"] == "reading Face Set data"
        assert not heavy_calls
        assert module.guided_ridge._guided_ridge_process_prepare_timer(paint_context, paint_state) is True
        assert heavy_calls == ["foreach_get"]
        assert paint_state["prepare_stage"] == "reading hidden-face data"
        guided_ridge_first_paint = {"passed": True, "first_tick_heavy_calls": 0, "second_tick_heavy_calls": 1}
        module.guided_ridge._guided_ridge_cancel(paint_state, "test-cleanup")
    finally:
        module.guided_ridge._guided_ridge_prepare_context_matches = original_prepare_context_matches
        module.guided_ridge._guided_ridge_overlay_tag = original_prepare_overlay
        module.runtime.guided_ridge_state = original_guided_state

    # A saved Repeat Last payload must not be touched by the T-tool invoke.
    # This uses only fake context/data and draw/timer callbacks.
    original_bpy_for_guided = module.guided_ridge.bpy
    original_guided_state = module.runtime.guided_ridge_state
    original_guided_last = module.runtime.guided_ridge_last_guide
    original_curve_step1 = module.guided_ridge.GUIDED_RIDGE_CURVE_SCULPT_STEP1
    saved_guided_functions = {
        name: getattr(module.guided_ridge, name)
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
        # This fixture exercises the legacy preparation/Repeat guard in
        # isolation; the production Step 1 route-only path is covered below.
        module.guided_ridge.GUIDED_RIDGE_CURVE_SCULPT_STEP1 = False
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
        module.guided_ridge.bpy = SimpleNamespace(
            types=SimpleNamespace(
                SpaceView3D=SimpleNamespace(
                    draw_handler_add=lambda *_args: object(),
                    draw_handler_remove=lambda *_args: None,
                )
            ),
            context=SimpleNamespace(window_manager=guided_context.window_manager),
        )
        module.runtime.guided_ridge_state = None
        repeat_calls = []
        sentinel = {"saved": True}
        module.runtime.guided_ridge_last_guide = sentinel
        module.guided_ridge._sculpt_cursor_region_coordinate = lambda _context, _event: Vector((20.0, 30.0))
        module.guided_ridge._raycast_sculpt_face_set = lambda _context, _coord: (fake_obj, 7, 11, None, None)
        module.guided_ridge._guided_ridge_safety_reason = lambda _obj: None
        module.guided_ridge._guided_ridge_context_signature = lambda _context: {"area_pointer": 1901}
        module.guided_ridge._guided_ridge_ray = lambda _context, _coord: (
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
        assert module.runtime.guided_ridge_state.get("phase") == "prepare"
        # An accidental direct EXEC_DEFAULT while the timer owns preparation
        # must not reach _commit (or the saved-guide Repeat Last path).
        assert guided_class.execute(guided_proxy, guided_context) == {"CANCELLED"}
        assert not repeat_calls
        saved_ray = module.runtime.guided_ridge_state["start_ray"]
        used_rays = []
        module.guided_ridge._guided_ridge_snapshot_hit_ray = lambda _snapshot, ray: (
            used_rays.append(ray)
            or (Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)), 0, 0.0)
        )
        def _saved_ray_job():
            yield {"stage": "finalizing snapshot", "stage_index": 12, "stage_count": 13, "done": 1, "total": 1}
            return ({"fixture": True}, None)
        module.runtime.guided_ridge_state["prepare_job"] = _saved_ray_job()
        guided_context.region.width = 1200
        guided_context.region.height = 900
        assert module.guided_ridge._guided_ridge_process_prepare_timer(
            guided_context, module.runtime.guided_ridge_state
        ) is True
        assert module.runtime.guided_ridge_state.get("phase") == "prepare"
        assert module.guided_ridge._guided_ridge_process_prepare_timer(
            guided_context, module.runtime.guided_ridge_state
        ) is True
        assert used_rays and tuple(used_rays[0][0]) == tuple(saved_ray[0])
        assert tuple(used_rays[0][1]) == tuple(saved_ray[1])
        guided_invoke_repeat = {"passed": True, "repeat_calls": 0}
        module.guided_ridge._guided_ridge_cancel(module.runtime.guided_ridge_state, "test-cleanup")
        assert module.runtime.guided_ridge_state is None
    finally:
        module.guided_ridge.GUIDED_RIDGE_CURVE_SCULPT_STEP1 = original_curve_step1
        module.guided_ridge.bpy = original_bpy_for_guided
        module.runtime.guided_ridge_state = original_guided_state
        module.runtime.guided_ridge_last_guide = original_guided_last
        for name, value in saved_guided_functions.items():
            setattr(module.guided_ridge, name, value)

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
    guided_ridge_curve_preview = _guided_ridge_curve_preview_fixture(module)
    guided_ridge_curve_sculpt_apply = _guided_ridge_curve_sculpt_apply_fixture(module)
    guided_ridge_restore_retry = _guided_ridge_restore_retry_fixture(module)
    durable_restore = _durable_restore_fixture(module)
    smart_fill_terminal_failure = _smart_fill_terminal_failure_fixture(module)
    hidden_gui_entry_guard = _hidden_gui_entry_guard_fixture()
    modal_return_contract = _modal_terminal_return_contract(module)
    modal_owner_preflight = _modal_owner_preflight_fixture(module)
    guided_ridge_curve_sculpt_transaction = _guided_ridge_curve_sculpt_transaction_fixture(module)
    smart_fill_modal_safety = _smart_fill_modal_safety_fixture(module)
    # Native asset activation, brush strokes, and mode switches are never run
    # in the user's connected Blender.  They are exercised by the dedicated
    # isolated/headless native runner instead.
    guided_ridge_native_asset = {
        "skipped": "isolated Blender native runner only",
        "mode_switches": 0,
        "brush_strokes": 0,
    }
    guided_ridge_native_apply = {
        "skipped": "isolated Blender native runner only",
        "mode_switches": 0,
        "brush_strokes": 0,
    }
    guided_ridge_curve_default_inscribed = _guided_ridge_curve_default_inscribed_fixture(module)
    guided_ridge_curve_signed_shape = _guided_ridge_curve_signed_shape_fixture(module)
    guided_ridge_curve_2d_quality = _guided_ridge_curve_2d_quality_fixture(module)
    guided_ridge_curve_screen_display = _guided_ridge_curve_screen_display_fixture(module)
    guided_ridge_curve_cache = _guided_ridge_curve_cache_fixture(module)
    guided_ridge_curve_draw_callback = _guided_ridge_curve_draw_callback_fixture(module)
    guided_ridge_projection_dependency = _guided_ridge_projection_dependency_fixture(
        module, shared_projection_identities
    )
    guided_ridge_prototype_limits = _guided_ridge_prototype_face_set_limit_fixture(module)
    guided_ridge_prototype_context = _guided_ridge_prototype_context_fixture(module)
    smart_fill_wheel = _smart_fill_wheel_gated_fixture(module)
    smart_fill_draw_generation = _smart_fill_draw_generation_fixture(module)
    smart_fill_terminal_expansion = _smart_fill_terminal_expansion_fixture(module)
    smart_fill_large_region_delta = _smart_fill_large_region_delta_fixture(module)
    smart_fill_large_region_benchmark = _smart_fill_large_region_benchmark_fixture(module)

    sfsf_sculpt = next(
        tool for tool in module._TOOL_CLASSES if tool.bl_idname == "mfo.smart_fill_sculpt"
    )
    assert any(item[1].get("ctrl") is True for item in sfsf_sculpt.bl_keymap)
    sfsf_vertex = next(
        tool for tool in module._TOOL_CLASSES if tool.bl_idname == "mfo.smart_fill_vertex"
    )
    assert sfsf_vertex.bl_context_mode == "PAINT_VERTEX"
    assert sfsf_vertex.bl_icon == sfsf_sculpt.bl_icon
    assert any(item[1].get("ctrl") is True for item in sfsf_vertex.bl_keymap)
    vertex_poll_context = SimpleNamespace(
        area=context.area,
        region=region,
        space_data=context.space_data,
        mode="PAINT_VERTEX",
        active_object=SimpleNamespace(type="MESH"),
    )
    assert module.VIEW3D_OT_mesh_focus_local_face_set_grow.poll(vertex_poll_context)
    smart_fill_vertex = _smart_fill_vertex_paint_fixture(module)

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
        loaded_icon_values = []
        try:
            loaded_icon_values = [bpy.app.icons.new_triangles_from_file(str(path)) for path in icon_paths]
            assert all(int(value) > 0 for value in loaded_icon_values)
        finally:
            for icon_value in loaded_icon_values:
                try:
                    bpy.app.icons.release(icon_value)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
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
    known_vertex_name = "3D View Tool: Paint Vertex, MFO: Smart Fill"
    unknown_name = "3D View Tool: Object, MFO: Focus Surface Custom"
    fake_user_sentinel = _FakeKeymap(known_name, [SimpleNamespace(idname="wm.search_menu")])
    fake_keyconfigs = SimpleNamespace(
        default=_FakeKeyconfig([
            _FakeKeymap(known_name), _FakeKeymap(known_vertex_name), _FakeKeymap(unknown_name)
        ]),
        addon=_FakeKeyconfig([_FakeKeymap(known_name), _FakeKeymap(known_vertex_name)]),
        user=_FakeKeyconfig([fake_user_sentinel]),
    )
    original_bpy_for_keymaps = module.registration.bpy
    try:
        module.registration.bpy = SimpleNamespace(
            context=SimpleNamespace(
                window_manager=SimpleNamespace(keyconfigs=fake_keyconfigs)
            )
        )
        module.registration._remove_stale_tool_keymaps()
        assert [km.name for km in fake_keyconfigs.default.keymaps] == [unknown_name]
        assert list(fake_keyconfigs.addon.keymaps) == []
        assert known_vertex_name in module._MFO_TOOL_KEYMAP_NAMES
        assert fake_keyconfigs.user.keymaps[0] is fake_user_sentinel
        assert fake_user_sentinel.keymap_items[0].idname == "wm.search_menu"
        user_keymap_result = {
            "preserved": True,
            "fake_collections_only": True,
            "unknown_default_preserved": True,
        }
    finally:
        module.registration.bpy = original_bpy_for_keymaps

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
    import bpy
    topology_context = SimpleNamespace(
        area=context.area,
        active_object=bpy.context.active_object,
        edit_object=bpy.context.active_object,
        mode="EDIT_MESH",
    )
    topology_service = module.registration._topology_color_object
    assert topology_service is module.foundation._topology_color_object
    topology_operator_poll = module.registration.VIEW3D_OT_mesh_focus_topology_color_assign.poll(
        topology_context
    )
    topology_panel_poll = module.registration.VIEW3D_PT_mesh_focus_topology_colors.poll(
        topology_context
    )
    assert topology_operator_poll and topology_panel_poll
    assert not hasattr(module.runtime, "topology_color_object")
    assert (
        module.tube_shape._smart_fill_geometry._fill_preview_signature
        is module.smart_fill_geometry._fill_preview_signature
    )
    for name, identity in shared_projection_identities.items():
        assert getattr(shared_view3d_utils, name) is identity
    reload_contract = _package_reload_contract()
    try:
        _package_reload_contract(inject_failure=True)
    except RuntimeError as error:
        assert str(error) == "injected reload cleanup failure"
        reload_failure_restored = True
    else:
        raise AssertionError("injected reload failure did not propagate")
    # Never leave a Smart Fill modal/timer/draw handle behind for the caller.
    # Native mode-switch and brush tests are isolated elsewhere, so this final
    # cleanup is safe and keeps the connected suite lifecycle-neutral.
    if module.runtime.fill_preview_state is not None:
        module.smart_fill_preview._fill_preview_cancel(
            module.runtime.fill_preview_state, "test-final-cleanup"
        )
    assert module.runtime.fill_preview_state is None
    assert module.runtime.fill_preview_modal_session is None
    return {
        "passed": True,
        "package_entrypoint": str(package_path),
        "package_allowlist": {
            "passed": not forbidden_package_paths,
            "forbidden_paths": forbidden_package_paths,
        },
        "package_import_contract": True,
        "runtime_ownership": runtime_ownership,
        "component_split_contract": {
            "passed": True,
            "component_count": len(expected_components),
            "origins": tuple(expected_components),
            "shared_compatibility_namespace": True,
        },
        "click_region": tuple(coordinate),
        "outside_rejected": True,
        "ctrl_strict_operator": module.TOOL_FACE_SET_OPERATOR_ID,
        "tool_count": len(module._TOOL_CLASSES),
        "poll_modes": poll_modes,
        "topology_color_poll": {
            "operator": topology_operator_poll,
            "panel": topology_panel_poll,
            "canonical_service": True,
        },
        "tube_geometry_reload_dependency": True,
        "topology_color_poll": {
            "operator": topology_operator_poll,
            "panel": topology_panel_poll,
            "canonical_service": True,
        },
        "tube_geometry_reload_dependency": True,
        "face_set_undo_contract": True,
        "icon_dat_count": len(icon_paths),
        "user_keymap": user_keymap_result,
        "sfsf_ctrl_binding": True,
        "smart_fill_vertex": smart_fill_vertex,
        "smart_fill_modal_safety": smart_fill_modal_safety,
        "smart_fill_wheel_gated": smart_fill_wheel,
        "smart_fill_draw_generation": smart_fill_draw_generation,
        "smart_fill_terminal_expansion": smart_fill_terminal_expansion,
        "smart_fill_large_region_delta": smart_fill_large_region_delta,
        "smart_fill_large_region_benchmark": smart_fill_large_region_benchmark,
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
        "guided_ridge_curve_preview": guided_ridge_curve_preview,
        "guided_ridge_curve_sculpt_apply": guided_ridge_curve_sculpt_apply,
        "guided_ridge_restore_retry": guided_ridge_restore_retry,
        "guided_ridge_durable_restore": durable_restore,
        "smart_fill_terminal_failure": smart_fill_terminal_failure,
        "hidden_gui_entry_guard": hidden_gui_entry_guard,
        "modal_return_contract": modal_return_contract,
        "modal_owner_preflight": modal_owner_preflight,
        "guided_ridge_curve_sculpt_transaction": guided_ridge_curve_sculpt_transaction,
        "guided_ridge_native_asset": guided_ridge_native_asset,
        "guided_ridge_native_apply": guided_ridge_native_apply,
        "guided_ridge_curve_default_inscribed": guided_ridge_curve_default_inscribed,
        "guided_ridge_curve_signed_shape": guided_ridge_curve_signed_shape,
        "guided_ridge_curve_2d_quality": guided_ridge_curve_2d_quality,
        "guided_ridge_curve_screen_display": guided_ridge_curve_screen_display,
        "guided_ridge_curve_cache": guided_ridge_curve_cache,
        "guided_ridge_curve_draw_callback": guided_ridge_curve_draw_callback,
        "guided_ridge_projection_dependency": guided_ridge_projection_dependency,
        "shared_projection_identities_unchanged": True,
        "guided_ridge_prototype_limits": guided_ridge_prototype_limits,
        "guided_ridge_prototype_context": guided_ridge_prototype_context,
        "package_reload_contract": reload_contract,
        "package_reload_failure_restored": reload_failure_restored,
    }


if __name__ == "__main__":
    print(run())
