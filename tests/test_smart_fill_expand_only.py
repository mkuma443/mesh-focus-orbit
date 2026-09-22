"""Static and pure regression checks for Smart Fill Shift+E.

The add-on imports Blender (and therefore cannot be imported by the ordinary
Python test interpreter).  These checks parse the changed source, then exercise
the small topology-stage model used by the mode.  They are deliberately
Blender-free so they can run before a live install.
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "mesh_focus_orbit" / "registration.py"
PREVIEW = ROOT / "mesh_focus_orbit" / "smart_fill" / "preview.py"
INIT = ROOT / "mesh_focus_orbit" / "__init__.py"


def _read(path):
    return path.read_text(encoding="utf-8")


def _tree(path):
    source = _read(path)
    return source, ast.parse(source, filename=str(path))


def _class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _function(root, name):
    for node in ast.walk(root):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _segment(source, node):
    result = ast.get_source_segment(source, node)
    assert result is not None
    return result


def _call_names(node):
    names = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        target = call.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def test_patch_version_is_bumped():
    source, tree = _tree(INIT)
    del source
    info = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "bl_info" for target in node.targets)
    )
    version = next(
        value
        for key, value in zip(info.keys, info.values)
        if isinstance(key, ast.Constant) and key.value == "version"
    )
    assert isinstance(version, ast.Tuple)
    assert tuple(element.value for element in version.elts) == (3, 4, 2)


def test_shift_e_dispatch_is_distinct_from_normal_and_strict_modes():
    source, tree = _tree(REGISTRATION)
    operator = _class(tree, "VIEW3D_OT_mesh_focus_local_face_set_grow")
    invoke = _segment(source, _function(operator, "invoke"))
    process = _segment(source, _function(operator, "_process_timer"))

    assert "expand_only: BoolProperty" in source
    assert 'getattr(event, "shift", False)' in invoke
    assert 'not getattr(event, "ctrl", False)' in invoke
    assert 'not getattr(event, "alt", False)' in invoke
    assert '_fill_preview_make_expand_only_result(state, radius)' in process
    assert '_fill_preview_make_result(state, radius)' in process
    assert '("expand-only", round(radius, 10))' in process
    assert '"progressive-range"' in process
    assert '"geometry-strict"' in process


def test_enter_and_e_confirmation_share_single_modal_owner_cleanup():
    source, tree = _tree(REGISTRATION)
    operator = _class(tree, "VIEW3D_OT_mesh_focus_local_face_set_grow")
    modal = _segment(source, _function(operator, "modal"))
    terminal = _segment(source, _function(operator, "_finish_confirm_terminal"))

    assert modal.count("return self._finish_confirm_terminal(context, state)") == 2
    assert "return self._finish_confirm(context, state)" not in modal
    assert "_lifecycle.modal_terminal(" in terminal
    assert "_fill_preview_modal_owner_terminal(self)" in terminal
    assert terminal.count("_lifecycle.modal_terminal(") == 1
    assert terminal.count("_fill_preview_modal_owner_terminal(self)") == 1


def test_shift_e_keymap_does_not_conflict_with_e_ctrl_e_or_shift_alt_e():
    source, tree = _tree(REGISTRATION)
    rebuild = _segment(source, _function(tree, "_rebuild_keymaps"))
    assert "expand_only_item = keymap.keymap_items.new(" in rebuild
    assert "expand_only_item.properties.expand_only = True" in rebuild
    assert "shift=True,\n            ctrl=False,\n            alt=False," in rebuild
    assert "strict_grow_item.properties.strict_mode = True" in rebuild
    assert '"view3d.mesh_focus_shadow_analysis_toggle"' in rebuild
    assert "alt=True,\n            ctrl=False,\n            shift=True," in rebuild

    # The plain E binding is intentionally still the unmodified item.
    normal_start = rebuild.index("local_grow_item = keymap.keymap_items.new(")
    normal_end = rebuild.index("_runtime.addon_keymaps.append((keymap, local_grow_item))", normal_start)
    normal = rebuild[normal_start:normal_end]
    assert 'LOCAL_FACE_SET_GROW_KEY,\n            "PRESS",\n        )' in normal


def test_expand_only_uses_only_lightweight_dihedral_surface_cost():
    source, tree = _tree(PREVIEW)
    helper = _function(tree, "_fill_preview_make_expand_only_result")
    helper_source = _segment(source, helper)
    calls = _call_names(helper)
    geometry_source, geometry_tree = _tree(
        ROOT / "mesh_focus_orbit" / "smart_fill" / "geometry.py"
    )
    progressive_source = _segment(
        geometry_source,
        _function(geometry_tree, "_fill_preview_progressive_range_step"),
    )
    assert "_fill_preview_progressive_range_step" in calls
    forbidden_calls = {
        "_fill_preview_local_geometry",
        "_fill_partition",
        "_fill_preview_refine_shape_boundary",
        "_fill_preview_valley_boundary",
        "_fill_preview_ridge_boundary",
        "_fill_preview_contour_cost",
    }
    assert not (calls & forbidden_calls)
    assert '"shape_segments": []' in helper_source
    assert '"expand_only_mode": True' in helper_source
    assert '"surface_evaluation_bypassed": False' in helper_source
    assert '"full_surface_analysis_bypassed": True' in helper_source
    assert '"surface_analysis": "simple-dihedral-cost"' in helper_source
    assert '"Smart Fill Preview - Simple Ridge/Valley Cost"' in source
    assert "_fill_preview_simple_feature_neighbor_lengths(geometry)" in progressive_source


def test_simple_feature_cost_doubles_ridge_and_valley_crossings():
    geometry_source, geometry_tree = _tree(
        ROOT / "mesh_focus_orbit" / "smart_fill" / "geometry.py"
    )
    node = _function(geometry_tree, "_fill_preview_simple_feature_neighbor_lengths")
    namespace = {"math": __import__("math")}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(PREVIEW), "exec"),
        namespace,
    )
    import numpy as np

    geometry = {
        "count": 3,
        "normals": np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))),
        "first": np.asarray((0, 1), dtype=np.int32),
        "second": np.asarray((1, 2), dtype=np.int32),
        "neighbor_lengths": np.ones(4, dtype=np.float64),
        "edge_indices": np.asarray((0, 1, 0, 1), dtype=np.int32),
    }
    weighted, feature_pairs = namespace[
        "_fill_preview_simple_feature_neighbor_lengths"
    ](geometry)
    assert feature_pairs.tolist() == [False, True]
    assert weighted.tolist() == [1.0, 2.0, 1.0, 2.0]


def test_actual_expand_only_selection_helper_respects_hidden_barrier():
    source, tree = _tree(PREVIEW)
    node = _function(tree, "_fill_preview_expand_only_selection")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PREVIEW), "exec"), namespace)
    selection = namespace["_fill_preview_expand_only_selection"](
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [False, False, True, False, False],
        4.0,
    )
    assert selection.tolist() == [True, True, False, True, True]
    assert namespace["_fill_preview_expand_only_selection"](
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [False, False, True, False, False],
        1.0,
    ).tolist() == [True, True, False, False, False]


def _topology_stage(distances, hidden, radius):
    return frozenset(
        index
        for index, distance in enumerate(distances)
        if distance <= radius and not hidden[index]
    )


def test_outward_wheel_stages_are_monotonic_visible_supersets():
    distances = (0.0, 1.0, 2.0, 3.0, 4.0)
    hidden = (False, False, False, True, False)
    stages = [_topology_stage(distances, hidden, radius) for radius in (0.0, 1.0, 2.0, 4.0)]
    assert all(stages[index] <= stages[index + 1] for index in range(len(stages) - 1))
    assert 3 not in stages[-1]
    assert stages[-1] == frozenset({0, 1, 2, 4})


def test_expand_only_shrink_reuses_the_exact_previous_cached_stage():
    source, tree = _tree(REGISTRATION)
    process = _segment(source, _function(_class(tree, "VIEW3D_OT_mesh_focus_local_face_set_grow"), "_process_timer"))
    expand_branch_start = process.index('if bool(state.get("expand_only", False))')
    expand_branch_end = process.index('strict_mode = bool(state.get("strict_mode", False))', expand_branch_start)
    expand_branch = process[expand_branch_start:expand_branch_end]
    assert '("expand-only", round(radius, 10))' in expand_branch
    assert "cached = cached_candidate" in process

    # A shrink is a stage lookup, not a union with the larger stage.
    cache = {1.0: frozenset({0}), 1.25: frozenset({0, 1}), 1.5625: frozenset({0, 1, 2})}
    assert cache[1.25] == frozenset({0, 1})
    assert cache[1.25] < cache[1.5625]


def test_expand_only_keeps_confirmation_parity_for_face_sets_and_vertex_paint():
    preview_source, preview_tree = _tree(PREVIEW)
    helper_source = _segment(preview_source, _function(preview_tree, "_fill_preview_make_expand_only_result"))
    assert "_fill_preview_confirm_graph_snapshot" in helper_source
    assert '"confirm_geometry"' in helper_source
    assert '"confirm_domain_ids"' in helper_source
    assert '"faces"' in helper_source

    registration_source, registration_tree = _tree(REGISTRATION)
    finish = _segment(
        registration_source,
        _function(_class(registration_tree, "VIEW3D_OT_mesh_focus_local_face_set_grow"), "_finish_confirm"),
    )
    assert "_fill_preview_confirm_flood(state, result)" in finish
    assert "_fill_preview_write_vertex_paint(state, result)" in finish
    assert '".sculpt_face_set"' in finish
    assert 'foreach_set("value", values)' in finish


def run():
    tests = [
        test_patch_version_is_bumped,
        test_shift_e_dispatch_is_distinct_from_normal_and_strict_modes,
        test_enter_and_e_confirmation_share_single_modal_owner_cleanup,
        test_shift_e_keymap_does_not_conflict_with_e_ctrl_e_or_shift_alt_e,
        test_expand_only_uses_only_lightweight_dihedral_surface_cost,
        test_simple_feature_cost_doubles_ridge_and_valley_crossings,
        test_actual_expand_only_selection_helper_respects_hidden_barrier,
        test_outward_wheel_stages_are_monotonic_visible_supersets,
        test_expand_only_shrink_reuses_the_exact_previous_cached_stage,
        test_expand_only_keeps_confirmation_parity_for_face_sets_and_vertex_paint,
    ]
    for test in tests:
        test()
    return {"passed": len(tests), "blender_imported": False}


if __name__ == "__main__":
    print(run())
