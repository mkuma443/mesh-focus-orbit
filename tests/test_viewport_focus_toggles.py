"""Focused regressions for the View3D wire and far-clip shortcuts."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "mesh_focus_orbit" / "smart_fill" / "preview.py"
REGISTRATION = ROOT / "mesh_focus_orbit" / "registration.py"
INIT = ROOT / "mesh_focus_orbit" / "__init__.py"


class _OperatorBase:
    def report(self, _levels, _message):
        pass


class _LifecycleProbe:
    def __init__(self):
        self.serial = 0
        self.entries = {}
        self.terminals = []

    def modal_register(self, operator, kind, session=None, *, state=None):
        self.serial += 1
        token = f"{kind}:{self.serial}:{id(operator):x}"
        entry = {"operator": operator, "token": token, "state": state}
        self.entries[token] = entry
        if isinstance(state, dict):
            state["modal_token"] = token
        return token

    def modal_terminal(self, operator=None, token=None, status=None):
        entry = self.entries.get(token)
        if entry is None:
            entry = next(
                (value for value in self.entries.values()
                 if value["operator"] is operator),
                None,
            )
        if entry is None:
            return False
        self.terminals.append((entry["token"], status))
        self.entries.pop(entry["token"], None)
        return True

    def modal_request_cancel(self, operator=None, token=None, reason=""):
        entry = self.entries.get(token)
        if entry is not None:
            entry["cancel_reason"] = reason
        return entry

    def modal_schedule_terminal_event(self, operator=None, token=None, state=None):
        entry = self.entries.get(token)
        if entry is not None:
            entry["terminal_scheduled"] = True
        return entry


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


def _display_distance_operator():
    source, tree = _source_tree(PREVIEW)
    helper_names = {
        "_display_distance_pointer",
        "_display_distance_ensure_runtime",
        "_display_distance_clamp",
        "_display_distance_write",
        "_display_distance_area_is_live",
        "_display_distance_unlink",
        "_display_distance_end_session",
        "_display_distance_prune_sessions",
        "_display_distance_cancel_all",
        "_display_distance_save_pre",
        "_display_distance_restore_after_save",
        "_display_distance_save_post",
        "_display_distance_load_pre",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    nodes.append(_class_node(tree, "VIEW3D_OT_mesh_focus_display_distance_toggle"))
    lifecycle = _LifecycleProbe()
    namespace = {
        "bpy": SimpleNamespace(types=SimpleNamespace(Operator=_OperatorBase)),
        "math": math,
        "persistent": lambda function: function,
        "_runtime": SimpleNamespace(display_distance_sessions={}),
        "_lifecycle": lifecycle,
        "_DISPLAY_DISTANCE_INITIAL": 0.13,
        "_DISPLAY_DISTANCE_RESET": 1000.0,
        "_DISPLAY_DISTANCE_MIN": 0.001,
        "_DISPLAY_DISTANCE_MAX": 1000.0,
        "_DISPLAY_DISTANCE_STEP": 1.15,
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(PREVIEW), "exec"), namespace)
    return namespace["VIEW3D_OT_mesh_focus_display_distance_toggle"], namespace, lifecycle


def _root_loader_probe(runtime):
    """Run the production root loader against retained fake child modules."""
    _source, tree = _source_tree(INIT)
    loader_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_components"
    )
    module_names = ast.literal_eval(
        next(
            node.value
            for node in ast.walk(loader_node)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "module_names"
                for target in node.targets
            )
        )
    )
    events = []
    teardown_sessions = []

    def old_unregister():
        events.append(("old_unregister", None))
        sessions = runtime.display_distance_sessions
        tuple(sessions.values())
        teardown_sessions.append(sessions)
        runtime.is_registered = False

    registration = SimpleNamespace(
        __name__="mesh_focus_orbit.registration",
        _runtime=runtime,
        _lifecycle=SimpleNamespace(),
        _lifecycle_modal_preflight=lambda: {"safe": True, "owners": []},
        unregister=old_unregister,
    )
    modules = {}
    for module_name in module_names:
        qualified = "mesh_focus_orbit." + module_name
        modules[qualified] = (
            runtime
            if module_name == "runtime"
            else registration
            if module_name == "registration"
            else SimpleNamespace(__name__=qualified)
        )

    def reload_module(module):
        events.append(("reload", module.__name__))
        return module

    def import_module(name):
        events.append(("import", name))
        return modules[name]

    namespace = {
        "__name__": "mesh_focus_orbit",
        "sys": SimpleNamespace(modules=modules),
        "importlib": SimpleNamespace(
            reload=reload_module,
            import_module=import_module,
        ),
    }
    exec(
        compile(ast.Module(body=[loader_node], type_ignores=[]), str(INIT), "exec"),
        namespace,
    )
    return namespace["_load_components"], events, teardown_sessions


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
        type="VIEW_3D",
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


class _WindowManager:
    def __init__(self):
        self.modal_operators = []

    def modal_handler_add(self, operator):
        self.modal_operators.append(operator)


def _view_context(space, other_spaces=()):
    area = SimpleNamespace(type="VIEW_3D", spaces=[space], tag_redraw=lambda: None)
    areas = [area]
    for other in other_spaces:
        areas.append(
            SimpleNamespace(type="VIEW_3D", spaces=[other], tag_redraw=lambda: None)
        )
    screen = SimpleNamespace(areas=areas)
    return SimpleNamespace(
        area=area,
        screen=screen,
        space_data=space,
        window_manager=_WindowManager(),
        window=SimpleNamespace(),
        mode="OBJECT",
    )


def _event(event_type, value="PRESS", *, shift=False, alt=False, ctrl=False):
    return SimpleNamespace(
        type=event_type,
        value=value,
        shift=shift,
        alt=alt,
        ctrl=ctrl,
    )


def _start_distance_adjustment(operator_class, context, operator=None):
    operator = operator or operator_class()
    assert operator.invoke(
        context, _event("V", shift=True, alt=True)
    ) == {"RUNNING_MODAL"}
    return operator


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


def test_display_distance_adjust_then_keep_ends_modal_and_later_press_restores():
    operator_class, namespace, lifecycle = _display_distance_operator()
    first = _space(clip_end=23.0, clip_start=0.001, view_distance=0.374)
    other = _space(clip_end=47.0, clip_start=0.02, view_distance=4.0)
    first.overlay.show_wireframes = True
    context = _view_context(first, [other])
    operator = _start_distance_adjustment(operator_class, context)

    assert first.clip_end == 0.13
    assert operator.modal(context, _event("WHEELUPMOUSE")) == {"RUNNING_MODAL"}
    assert abs(first.clip_end - 0.1495) < 1.0e-9
    assert operator.modal(context, _event("WHEELDOWNMOUSE")) == {"RUNNING_MODAL"}
    assert abs(first.clip_end - 0.13) < 1.0e-9

    # Non-wheel view edits pass through; an event in another viewport cannot
    # change the active target's distance.
    assert operator.modal(context, _event("MIDDLEMOUSE")) == {"PASS_THROUGH"}
    assert abs(first.clip_end - 0.13) < 1.0e-9
    other_context = SimpleNamespace(space_data=other)
    assert operator.modal(other_context, _event("WHEELUPMOUSE")) == {"PASS_THROUGH"}
    assert other.clip_end == 47.0

    assert operator.modal(context, _event("WHEELUPMOUSE")) == {"RUNNING_MODAL"}
    kept = first.clip_end
    assert operator.modal(context, _event("LEFTMOUSE")) == {"FINISHED"}
    session = namespace["_runtime"].display_distance_sessions[id(first)]
    assert session["phase"] == "working"
    assert first.clip_end == kept
    assert not lifecycle.entries
    assert lifecycle.terminals[-1][1] == "FINISHED"

    # The accepted modal has returned; a wheel in the next operation has no
    # active distance modal to capture it. The later shortcut is a fresh toggle.
    assert operator._display_distance_session is None
    assert context.window_manager.modal_operators == [operator]
    restore = operator_class()
    assert restore.invoke(context, _event("V", shift=True, alt=True)) == {"FINISHED"}
    assert first.clip_end == 1000.0
    assert not namespace["_runtime"].display_distance_sessions

    assert first.clip_start == 0.001
    assert first.region_3d.view_distance == 0.374
    assert first.overlay.show_wireframes is True
    assert other.clip_end == 47.0
    assert other.clip_start == 0.02
    assert other.region_3d.view_distance == 4.0


def test_runtime_singleton_reload_backfills_display_distance_state():
    source, tree = _source_tree(PREVIEW)
    node = next(
        value
        for value in tree.body
        if isinstance(value, ast.FunctionDef)
        and value.name == "_display_distance_ensure_runtime"
    )
    runtime = SimpleNamespace()
    namespace = {"_runtime": runtime}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(PREVIEW), "exec"), namespace)
    sessions = namespace["_display_distance_ensure_runtime"]()
    assert sessions == {}
    assert runtime.display_distance_sessions is sessions

    runtime.display_distance_sessions = {123: {"phase": "working"}}
    assert namespace["_display_distance_ensure_runtime"]() is runtime.display_distance_sessions
    assert runtime.display_distance_sessions[123]["phase"] == "working"

    malformed = []
    runtime.display_distance_sessions = malformed
    try:
        namespace["_display_distance_ensure_runtime"]()
    except TypeError:
        pass
    else:
        raise AssertionError("malformed runtime session state must fail closed")
    assert runtime.display_distance_sessions is malformed


def test_root_reload_backfills_missing_legacy_state_before_old_teardown_and_child_import():
    runtime = SimpleNamespace(is_registered=True, modal_registry={})
    load_components, events, teardown_sessions = _root_loader_probe(runtime)

    modules = load_components()

    assert isinstance(runtime.display_distance_sessions, dict)
    assert runtime.display_distance_sessions == {}
    assert teardown_sessions == [runtime.display_distance_sessions]
    assert events[0] == ("old_unregister", None)
    first_child_import = next(
        index
        for index, event in enumerate(events)
        if event[0] in {"import", "reload"}
    )
    assert events.index(("old_unregister", None)) < first_child_import
    assert not any(
        event == ("reload", "mesh_focus_orbit.runtime") for event in events
    )
    assert modules["runtime"] is runtime
    assert runtime.is_registered is False


def test_root_reload_preserves_existing_state_and_rejects_malformed_state():
    retained = {123: {"phase": "working", "value": 0.1495}}
    runtime = SimpleNamespace(
        is_registered=True,
        modal_registry={},
        display_distance_sessions=retained,
    )
    load_components, _events, teardown_sessions = _root_loader_probe(runtime)
    load_components()
    assert runtime.display_distance_sessions is retained
    assert runtime.display_distance_sessions[123]["value"] == 0.1495
    assert teardown_sessions == [retained]

    malformed = []
    broken_runtime = SimpleNamespace(
        is_registered=True,
        modal_registry={},
        display_distance_sessions=malformed,
    )
    broken_loader, broken_events, _teardown_sessions = _root_loader_probe(
        broken_runtime
    )
    try:
        broken_loader()
    except TypeError as error:
        assert "display_distance_sessions" in str(error)
    else:
        raise AssertionError("root reload must reject malformed retained state")
    assert broken_runtime.display_distance_sessions is malformed
    assert not broken_events
    assert broken_runtime.is_registered is True


def test_display_distance_escape_right_click_and_repress_restore_1000():
    operator_class, namespace, lifecycle = _display_distance_operator()
    for event_type, modifiers, terminal in (
        ("ESC", {}, "CANCELLED"),
        ("RIGHTMOUSE", {}, "CANCELLED"),
        ("V", {"shift": True, "alt": True}, "FINISHED"),
    ):
        space = _space(clip_end=73.0)
        context = _view_context(space)
        operator = _start_distance_adjustment(operator_class, context)
        assert space.clip_end == 0.13
        event = _event(event_type, **modifiers)
        result = operator.modal(context, event)
        assert result == {terminal}
        assert space.clip_end == 1000.0
        assert not namespace["_runtime"].display_distance_sessions
        assert lifecycle.terminals[-1][1] == terminal


def test_display_distance_enter_variants_keep_value_and_end_modal():
    operator_class, namespace, lifecycle = _display_distance_operator()
    for event_type in ("RET", "NUMPAD_ENTER"):
        space = _space(clip_end=81.0)
        context = _view_context(space)
        operator = _start_distance_adjustment(operator_class, context)
        assert operator.modal(context, _event("WHEELDOWNMOUSE")) == {"RUNNING_MODAL"}
        kept = space.clip_end
        assert operator.modal(context, _event(event_type)) == {"FINISHED"}
        assert space.clip_end == kept
        assert namespace["_runtime"].display_distance_sessions[id(space)]["phase"] == "working"
        assert lifecycle.terminals[-1][1] == "FINISHED"
        namespace["_display_distance_cancel_all"]()
        assert space.clip_end == 1000.0
        assert not namespace["_runtime"].display_distance_sessions


def test_display_distance_bounds_and_rejects_nonfinite_values():
    operator_class, namespace, _lifecycle_probe = _display_distance_operator()
    clamp = namespace["_display_distance_clamp"]
    assert clamp(float("nan")) == 0.13
    assert clamp(float("inf")) == 0.13
    assert clamp(-3.0) == 0.13
    assert clamp(0.0) == 0.13

    space = _space(clip_end=12.0)
    context = _view_context(space)
    operator = _start_distance_adjustment(operator_class, context)
    for _index in range(250):
        assert operator.modal(context, _event("WHEELDOWNMOUSE")) == {"RUNNING_MODAL"}
        assert math.isfinite(space.clip_end)
        assert space.clip_end >= 0.001
    assert space.clip_end == 0.001
    for _index in range(250):
        assert operator.modal(context, _event("WHEELUPMOUSE")) == {"RUNNING_MODAL"}
    assert space.clip_end == 1000.0
    namespace["_display_distance_cancel_all"]()


def test_display_distance_save_handlers_restore_live_value_and_load_cleans_up():
    operator_class, namespace, lifecycle = _display_distance_operator()
    space = _space(clip_end=29.0)
    context = _view_context(space)
    operator = _start_distance_adjustment(operator_class, context)
    assert operator.modal(context, _event("WHEELUPMOUSE")) == {"RUNNING_MODAL"}
    preview_value = space.clip_end

    namespace["_display_distance_save_pre"]()
    assert space.clip_end == 1000.0
    session = namespace["_runtime"].display_distance_sessions[id(space)]
    assert session["save_pending"]
    assert session["save_value"] == preview_value
    namespace["_display_distance_save_pre"]()
    assert session["save_value"] == preview_value
    assert space.clip_end == 1000.0
    # save_post_fail uses the same idempotent display restoration callback.
    namespace["_display_distance_save_post"]()
    assert abs(space.clip_end - preview_value) < 1.0e-9

    namespace["_display_distance_load_pre"]()
    assert space.clip_end == 1000.0
    assert not namespace["_runtime"].display_distance_sessions
    assert lifecycle.entries
    assert next(iter(lifecycle.entries.values())).get("terminal_scheduled") is True
    assert operator.modal(context, _event("TIMER")) == {"CANCELLED"}
    assert not lifecycle.entries


def test_display_distance_area_prune_and_unregister_hooks():
    operator_class, namespace, lifecycle = _display_distance_operator()
    space = _space(clip_end=34.0)
    context = _view_context(space)
    operator = _start_distance_adjustment(operator_class, context)
    context.screen.areas.clear()
    namespace["_display_distance_prune_sessions"]()
    assert space.clip_end == 1000.0
    assert not namespace["_runtime"].display_distance_sessions
    assert next(iter(lifecycle.entries.values())).get("terminal_scheduled") is True
    assert operator.modal(context, _event("TIMER")) == {"CANCELLED"}

    source, tree = _source_tree(REGISTRATION)
    unregister = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "unregister"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_display_distance_cancel_all"
        for node in ast.walk(unregister)
    )
    assert "_display_distance_load_pre" in source
    assert "_display_distance_save_pre" in source
    assert "_display_distance_save_post" in source
    handlers = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_remove_registered_handlers"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "save_post_fail"
        for node in ast.walk(handlers)
    )
    assert any(
        isinstance(node, ast.Name) and node.id == "_display_distance_save_post"
        for node in ast.walk(handlers)
    )

    register = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "register"
    )
    save_fail_guarded = any(
        isinstance(node, ast.If)
        and any(
            isinstance(part, ast.Compare)
            and any(
                isinstance(operator, ast.NotIn)
                for operator in part.ops
            )
            for part in ast.walk(node.test)
        )
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "append"
            and call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "_display_distance_save_post"
            for call in ast.walk(node)
        )
        for node in ast.walk(register)
    )
    assert save_fail_guarded

    preview_source, preview_tree = _source_tree(PREVIEW)
    persistent_save_callbacks = {
        node.name
        for node in preview_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_display_distance_save_pre", "_display_distance_save_post"}
        and any(
            isinstance(decorator, ast.Name) and decorator.id == "persistent"
            for decorator in node.decorator_list
        )
    }
    assert persistent_save_callbacks == {
        "_display_distance_save_pre",
        "_display_distance_save_post",
    }


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
    assert tuple(item.value for item in version.elts) == (3, 4, 12)


def run():
    tests = [
        test_wire_toggle_uses_current_value_across_recreated_operators,
        test_wire_off_changes_only_wire_flag_and_initially_on_turns_off,
        test_wire_on_sets_faint_profile_only_for_the_target_view,
        test_display_distance_adjust_then_keep_ends_modal_and_later_press_restores,
        test_runtime_singleton_reload_backfills_display_distance_state,
        test_root_reload_backfills_missing_legacy_state_before_old_teardown_and_child_import,
        test_root_reload_preserves_existing_state_and_rejects_malformed_state,
        test_display_distance_escape_right_click_and_repress_restore_1000,
        test_display_distance_enter_variants_keep_value_and_end_modal,
        test_display_distance_bounds_and_rejects_nonfinite_values,
        test_display_distance_save_handlers_restore_live_value_and_load_cleans_up,
        test_display_distance_area_prune_and_unregister_hooks,
        test_keymap_registers_shift_alt_v_and_retains_shift_alt_e,
        test_bl_info_version_matches_current_patch,
    ]
    for test in tests:
        test()
    return {"passed": len(tests), "blender_imported": False}


if __name__ == "__main__":
    print(run())
