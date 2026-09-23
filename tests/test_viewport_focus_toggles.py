"""Focused regressions for the View3D wire and far-clip shortcuts."""

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "mesh_focus_orbit" / "smart_fill" / "preview.py"
REGISTRATION = ROOT / "mesh_focus_orbit" / "registration.py"
INIT = ROOT / "mesh_focus_orbit" / "__init__.py"


class _OperatorBase:
    def report(self, _levels, _message):
        pass


def _source_tree(path):
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _class_node(tree, class_name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _operator(class_name):
    source, tree = _source_tree(PREVIEW)
    node = _class_node(tree, class_name)
    namespace = {
        "bpy": SimpleNamespace(types=SimpleNamespace(Operator=_OperatorBase)),
    }
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(PREVIEW), "exec"), namespace)
    return namespace[class_name], source, ast.get_source_segment(source, node)


def _space(
    *,
    show_overlays=True,
    show_wireframes=False,
    wireframe_threshold=0.5,
    wireframe_opacity=0.8,
    clip_end=1000.0,
    clip_start=0.001,
    view_distance=0.374,
):
    return SimpleNamespace(
        overlay=SimpleNamespace(
            show_overlays=show_overlays,
            show_wireframes=show_wireframes,
            wireframe_threshold=wireframe_threshold,
            wireframe_opacity=wireframe_opacity,
        ),
        clip_end=clip_end,
        clip_start=clip_start,
        region_3d=SimpleNamespace(view_distance=view_distance),
    )


def _run(operator_class, space, mode="OBJECT"):
    area = SimpleNamespace(type="VIEW_3D", tag_redraw=lambda: None)
    context = SimpleNamespace(area=area, space_data=space, mode=mode)
    result = operator_class().execute(context)
    assert result == {"FINISHED"}
    return context


def test_wire_toggle_uses_current_value_across_recreated_operators():
    operator, _source, operator_source = _operator(
        "VIEW3D_OT_mesh_focus_shadow_analysis_toggle"
    )
    assert "shadow_analysis_view_tokens" not in operator_source
    space = _space(show_overlays=True, wireframe_threshold=1.0, wireframe_opacity=0.22)

    _run(operator, space, mode="OBJECT")
    assert space.overlay.show_wireframes is True
    _run(operator, space, mode="EDIT_MESH")
    assert space.overlay.show_wireframes is False
    _run(operator, space, mode="SCULPT")
    assert space.overlay.show_wireframes is True
    _run(operator, space, mode="OBJECT")
    assert space.overlay.show_wireframes is False


def test_wire_off_changes_only_wire_flag_and_initially_on_turns_off():
    operator, _source, _operator_source = _operator(
        "VIEW3D_OT_mesh_focus_shadow_analysis_toggle"
    )
    space = _space(
        show_overlays=False,
        show_wireframes=True,
        wireframe_threshold=0.43,
        wireframe_opacity=0.71,
    )
    _run(operator, space)
    assert space.overlay.show_wireframes is False
    assert space.overlay.show_overlays is False
    assert space.overlay.wireframe_threshold == 0.43
    assert space.overlay.wireframe_opacity == 0.71


def test_wire_on_sets_faint_profile_only_for_the_target_view():
    operator, _source, _operator_source = _operator(
        "VIEW3D_OT_mesh_focus_shadow_analysis_toggle"
    )
    first = _space(
        show_overlays=False,
        wireframe_threshold=0.5,
        wireframe_opacity=0.8,
    )
    other = _space(
        show_overlays=True,
        show_wireframes=False,
        wireframe_threshold=0.31,
        wireframe_opacity=0.63,
    )

    _run(operator, first, mode="EDIT_MESH")
    assert first.overlay.show_wireframes is True
    assert first.overlay.show_overlays is True
    assert first.overlay.wireframe_threshold == 1.0
    assert abs(first.overlay.wireframe_opacity - 0.22) < 1.0e-7
    assert other.overlay.show_overlays is True
    assert other.overlay.show_wireframes is False
    assert other.overlay.wireframe_threshold == 0.31
    assert other.overlay.wireframe_opacity == 0.63


def test_display_distance_toggles_per_view_and_preserves_other_view_values():
    operator, _source, _operator_source = _operator(
        "VIEW3D_OT_mesh_focus_display_distance_toggle"
    )
    first = _space(clip_end=0.13, clip_start=0.001, view_distance=0.374)
    other = _space(clip_end=23.0, clip_start=0.02, view_distance=4.0)
    first.overlay.show_wireframes = True

    _run(operator, first, mode="EDIT_MESH")
    assert first.clip_end == 1000.0
    assert first.clip_start == 0.001
    assert first.region_3d.view_distance == 0.374
    assert first.overlay.show_wireframes is True
    assert other.clip_end == 23.0

    _run(operator, first, mode="OBJECT")
    assert first.clip_end == 0.13
    _run(operator, first, mode="SCULPT")
    assert first.clip_end == 1000.0

    _run(operator, other)
    assert other.clip_end == 0.13
    assert other.clip_start == 0.02
    assert other.region_3d.view_distance == 4.0


def test_keymap_registers_shift_alt_v_and_retains_shift_alt_e():
    source, tree = _source_tree(REGISTRATION)
    rebuild = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_rebuild_keymaps"
    )
    keymap_items = [
        node
        for node in ast.walk(rebuild)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "new"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "view3d.mesh_focus_display_distance_toggle"
    ]
    assert len(keymap_items) == 1
    item = keymap_items[0]
    assert isinstance(item.args[1], ast.Constant) and item.args[1].value == "V"
    modifiers = {keyword.arg: keyword.value for keyword in item.keywords}
    assert isinstance(modifiers["shift"], ast.Constant) and modifiers["shift"].value is True
    assert isinstance(modifiers["alt"], ast.Constant) and modifiers["alt"].value is True
    assert isinstance(modifiers["ctrl"], ast.Constant) and modifiers["ctrl"].value is False

    assert "OPEN_BOUNDARY_LOOP_OPERATOR_ID" in source
    assert '"view3d.mesh_focus_shadow_analysis_toggle"' in source
    assert "VIEW3D_OT_mesh_focus_display_distance_toggle" in source


def test_bl_info_version_matches_current_patch():
    _source, tree = _source_tree(INIT)
    bl_info = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "bl_info"
            for target in node.targets
        )
    )
    version = next(
        value
        for key, value in zip(bl_info.keys, bl_info.values)
        if isinstance(key, ast.Constant) and key.value == "version"
    )
    assert tuple(item.value for item in version.elts) == (3, 4, 7)


def run():
    tests = [
        test_wire_toggle_uses_current_value_across_recreated_operators,
        test_wire_off_changes_only_wire_flag_and_initially_on_turns_off,
        test_wire_on_sets_faint_profile_only_for_the_target_view,
        test_display_distance_toggles_per_view_and_preserves_other_view_values,
        test_keymap_registers_shift_alt_v_and_retains_shift_alt_e,
        test_bl_info_version_matches_current_patch,
    ]
    for test in tests:
        test()
    return {"passed": len(tests), "blender_imported": False}


if __name__ == "__main__":
    print(run())
