"""Disposable GUI Blender modal-boundary runner.

Run only as a separate factory-startup GUI process.  It never targets the
connected user Blender.  The runner attempts a real INVOKE_DEFAULT Smart Fill
operator in a View3D, changes mode on a temporary object, and waits for the
real modal timer/depsgraph boundary to return terminal.  Guided Ridge native
brush tests remain a separate opt-in runner because they can invoke Blender's
native sculpt stroke.
"""
import bpy
import addon_utils
import importlib
import json
import os
import pathlib
import sys
import traceback
import hashlib

SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULT = {
    "passed": False,
    "invoke_attempted": False,
    "invoke_result": None,
    "modal_terminal": False,
    "mode_switch_isolated": False,
    "runtime_clean": False,
    "native_invoke_supported": False,
    "disable_attempted": False,
    "disable_check_while_active": None,
    "loader_mode": None,
    "loader_before": None,
    "unregister_completed": False,
    "classes_handlers_absent": False,
    "lifecycle_preflight_before_disable": None,
    "owner_before_cancel": None,
    "terminal_event_before_disable": None,
    "candidate_module_file": None,
    "candidate_version": None,
    "package_manifest_matches": False,
    "candidate_root_expected": False,
    "actual_addon_keymaps_absent": False,
    "actual_handler_module_absent": False,
    "smart_fill_invoke_failure_matrix": None,
    "isolation": None,
    "native_mode_switch_test_enabled": False,
    "native_mode_switch_test_disabled_reason": "",
    "quiescence_safe_before_disable": False,
    "quiescence_ticks": 0,
    "errors": [],
}


def _view3d_context():
    window = bpy.context.window
    if window is None:
        return None
    area = next((item for item in window.screen.areas if item.type == "VIEW_3D"), None)
    if area is None:
        return None
    region = next((item for item in area.regions if item.type == "WINDOW"), None)
    return window, area, region


def _isolation_preflight():
    """Refuse the native run unless only the disposable candidate is loaded."""
    expected_config = os.environ.get("MFO_EXPECTED_CONFIG", "")
    expected_scripts = os.environ.get("MFO_EXPECTED_SCRIPTS", "")
    actual_config = ""
    actual_scripts = ""
    try:
        actual_config = str(pathlib.Path(bpy.utils.user_resource("CONFIG")).resolve())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        actual_config = ""
    try:
        actual_scripts = str(pathlib.Path(bpy.utils.script_path_user()).resolve())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        actual_scripts = ""
    enabled_addons = tuple(
        str(name) for name in getattr(bpy.context.preferences, "addons", {}).keys()
    )
    try:
        modules = tuple(addon_utils.modules(refresh=True))
    except Exception:
        modules = ()
    module_names = tuple(getattr(item, "__name__", "") for item in modules)
    forbidden = tuple(
        name for name in module_names
        if any(token in name.lower() for token in ("retopoflow", "autosave"))
    )
    module_by_name = {
        str(getattr(item, "__name__", "")): item for item in modules
    }
    system_roots = []
    for resource_kind in ("LOCAL", "SYSTEM"):
        try:
            system_roots.append(pathlib.Path(
                bpy.utils.resource_path(resource_kind)
            ).resolve())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    enabled_paths = {}
    unexpected_enabled = []
    for name in enabled_addons:
        module = module_by_name.get(name)
        module_path = getattr(module, "__file__", "") if module is not None else ""
        enabled_paths[name] = str(module_path)
        if name == "mesh_focus_orbit" or name.startswith("bl_ext"):
            continue
        try:
            builtin = any(
                pathlib.Path(module_path).resolve().is_relative_to(root)
                for root in system_roots
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            builtin = False
        if not builtin:
            unexpected_enabled.append(name)
    config_ok = bool(expected_config) and actual_config == str(
        pathlib.Path(expected_config).resolve()
    )
    scripts_ok = bool(expected_scripts) and actual_scripts == str(
        pathlib.Path(expected_scripts).resolve()
    )
    result = {
        "expected_config": expected_config,
        "actual_config": actual_config,
        "expected_scripts": expected_scripts,
        "actual_scripts": actual_scripts,
        "config_ok": config_ok,
        "scripts_ok": scripts_ok,
        "enabled_addons": enabled_addons,
        "forbidden_modules": forbidden,
        "unexpected_enabled": unexpected_enabled,
        "enabled_paths": enabled_paths,
        "system_roots": tuple(str(root) for root in system_roots),
        "passed": bool(config_ok and scripts_ok and not forbidden and not unexpected_enabled),
    }
    RESULT["isolation"] = result
    if not result["passed"]:
        RESULT["errors"].append("GUI isolation preflight failed")
    return bool(result["passed"])


def _report_and_quit():
    try:
        module = sys.modules.get("mesh_focus_orbit")
        runtime = getattr(module, "runtime", None)
        source_candidates = (
            pathlib.Path(SOURCE_ROOT) / "mesh_focus_orbit" / "__init__.py",
            pathlib.Path(SOURCE_ROOT) / "scripts" / "addons" / "mesh_focus_orbit" / "__init__.py",
            pathlib.Path(__file__).resolve().parent / "mesh_focus_orbit" / "__init__.py",
            pathlib.Path(__file__).resolve().parent / "scripts" / "addons" / "mesh_focus_orbit" / "__init__.py",
        )
        source_file = next(
            (candidate for candidate in source_candidates if candidate.is_file()),
            source_candidates[0],
        )
        expected_hash = None
        try:
            expected_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
        except OSError:
            pass
        RESULT["candidate_hash"] = (
            hashlib.sha256(pathlib.Path(module.__file__).resolve().read_bytes()).hexdigest()
            if module is not None else None
        )
        package_root = pathlib.Path(module.__file__).resolve().parent if module is not None else source_file.parent
        manifest = {}
        for candidate in sorted(package_root.rglob("*")):
            if not candidate.is_file() or "__pycache__" in candidate.parts:
                continue
            manifest[str(candidate.relative_to(package_root)).replace("\\", "/")] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        RESULT["package_manifest"] = manifest
        RESULT["package_manifest_count"] = len(manifest)
        expected_manifest = None
        expected_path = os.environ.get("MFO_EXPECTED_MANIFEST", "")
        if expected_path:
            try:
                expected_manifest = json.loads(pathlib.Path(expected_path).read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                expected_manifest = None
        if isinstance(expected_manifest, dict):
            RESULT["package_manifest_matches"] = manifest == expected_manifest
        expected_root = os.environ.get("MFO_EXPECTED_ROOT", "")
        if expected_root:
            try:
                RESULT["candidate_root_expected"] = (
                    pathlib.Path(package_root).resolve().is_relative_to(pathlib.Path(expected_root).resolve())
                )
            except (AttributeError, OSError, RuntimeError, ValueError):
                RESULT["candidate_root_expected"] = False
        RESULT["runtime_clean"] = bool(
            runtime is None
            or (
                runtime.fill_preview_state is None
                and runtime.fill_preview_modal_operator is None
                and not runtime.fill_preview_modal_handler_live
                and not runtime.modal_registry
            )
        )
        RESULT["final_loader_check"] = addon_utils.check("mesh_focus_orbit")
        RESULT["passed"] = bool(
            RESULT["invoke_attempted"]
            and RESULT["modal_terminal"]
            and RESULT["mode_switch_isolated"]
            and RESULT["runtime_clean"]
            and RESULT["disable_attempted"]
            and RESULT["unregister_completed"]
            and RESULT["classes_handlers_absent"]
            and RESULT["loader_mode"] == "addon-utils"
            and RESULT["owner_before_cancel"]
            and RESULT["terminal_event_before_disable"] is not None
            and RESULT["terminal_event_before_disable"].get("status") == "CANCELLED"
            and RESULT["candidate_version"] == (3, 3, 96)
            and RESULT.get("smart_fill_invoke_failure_matrix", {}).get("passed")
            and RESULT["native_invoke_supported"]
            and RESULT.get("package_manifest_count", 0) > 0
            and RESULT["package_manifest_matches"]
            and RESULT["candidate_root_expected"]
            and RESULT["actual_addon_keymaps_absent"]
            and RESULT["actual_handler_module_absent"]
            and RESULT["candidate_module_file"]
            and expected_hash == RESULT["candidate_hash"]
            and RESULT.get("isolation", {}).get("passed")
            and RESULT["quiescence_safe_before_disable"]
            and not RESULT.get("errors")
        )
    except Exception as error:
        RESULT["errors"].append(f"finalize: {error!r}")
    print(json.dumps(RESULT, sort_keys=True))
    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        pass
    return None


def _finish_check():
    module = sys.modules.get("mesh_focus_orbit")
    runtime = getattr(module, "runtime", None)
    if runtime is not None:
        events = tuple(getattr(runtime, "modal_terminal_events", ()))
        token = RESULT.get("modal_token")
        matching = [
            event for event in events
            if token is not None and event.get("token") == token
            and event.get("status") in {"FINISHED", "CANCELLED"}
        ]
        RESULT["terminal_event_before_disable"] = matching[-1] if matching else None
        RESULT["modal_terminal"] = bool(matching)
        RESULT["runtime_owner_after_terminal"] = tuple(runtime.modal_registry.keys())
    lifecycle = getattr(sys.modules.get("mesh_focus_orbit"), "lifecycle", None)
    if RESULT["modal_terminal"] and RESULT["mode_switch_isolated"]:
        if lifecycle is not None and not lifecycle.modal_quiescence_is_safe():
            RESULT["quiescence_ticks"] = int(RESULT.get("quiescence_ticks", 0)) + 1
            if RESULT["quiescence_ticks"] < 50:
                return 0.05
            RESULT["errors"].append("modal quiescence did not become safe")
            return _report_and_quit()
        RESULT["quiescence_safe_before_disable"] = True
        bpy.app.timers.register(_attempt_disable_active, first_interval=0.01)
        return None
    # A real modal handler gets TIMER events from its own event timer.  Give it
    # several GUI turns after the depsgraph-triggered cancellation.
    if int(RESULT.get("_polls", 0)) < 20:
        RESULT["_polls"] = int(RESULT.get("_polls", 0)) + 1
        return 0.10
    RESULT["errors"].append("modal handler did not return terminal within GUI event budget")
    return _report_and_quit()


def _switch_mode():
    try:
        # The prior timer-driven mode switch was removed from normal automated
        # runs after Blender 5.2 showed a native rna_operator_modal_cb crash at
        # this boundary. Keep this path opt-in for a future manually supervised
        # native investigation; never invoke bpy.ops.object.mode_set from the
        # default timer helper while a modal owner is live.
        if os.environ.get("MFO_ENABLE_UNSAFE_MODE_SWITCH_TEST") != "1":
            RESULT["native_mode_switch_test_disabled_reason"] = "timer-driven mode switch disabled after native crash"
            return _report_and_quit()
        lifecycle = getattr(sys.modules.get("mesh_focus_orbit"), "lifecycle", None)
        runtime = getattr(sys.modules.get("mesh_focus_orbit"), "runtime", None)
        if runtime is not None and runtime.modal_registry:
            RESULT["errors"].append("mode switch blocked while modal owner is live")
            return _report_and_quit()
        if lifecycle is not None and not lifecycle.modal_quiescence_is_safe():
            RESULT["quiescence_ticks"] = int(RESULT.get("quiescence_ticks", 0)) + 1
            if RESULT["quiescence_ticks"] < 50:
                return 0.05
            RESULT["errors"].append("modal quiescence did not become safe before mode switch")
            return _report_and_quit()
        ctx = _view3d_context()
        if ctx is None:
            RESULT["errors"].append("no View3D after invoke")
            return _report_and_quit()
        window, area, region = ctx
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="OBJECT")
        RESULT["mode_switch_isolated"] = bpy.context.mode == "OBJECT"
    except Exception as error:
        RESULT["errors"].append(f"mode switch: {error!r}")
    bpy.app.timers.register(_finish_check, first_interval=0.20)
    return None


def _smart_fill_invoke_failure_matrix(addon, window, area, region):
    """Exercise real INVOKE_DEFAULT failures at every startup checkpoint."""
    registration = addon.registration
    runtime = addon.runtime
    original_owner_terminal = registration._fill_preview_modal_owner_terminal
    calls = []
    records = []

    def counting_owner_terminal(operator):
        calls.append(id(operator))
        return original_owner_terminal(operator)

    registration._fill_preview_modal_owner_terminal = counting_owner_terminal
    try:
        for stage in ("timer", "handler", "draw", "text-draw", "redraw"):
            runtime._test_invoke_failure_stage = stage
            before_calls = len(calls)
            with bpy.context.temp_override(
                window=window,
                area=area,
                region=region,
                space_data=area.spaces.active,
                region_data=area.spaces.active.region_3d,
            ):
                invoke_result = bpy.ops.view3d.mesh_focus_local_face_set_grow_v2(
                    "INVOKE_DEFAULT",
                    mouse_region_x=int(region.width * 0.5),
                    mouse_region_y=int(region.height * 0.5),
                )
            records.append({
                "stage": stage,
                "invoke_result": sorted(str(item) for item in invoke_result),
                "owner_terminal_calls": len(calls) - before_calls,
                "registry_empty": not bool(runtime.modal_registry),
                "operator_cleared": runtime.fill_preview_modal_operator is None,
                "session_cleared": runtime.fill_preview_modal_session is None,
                "handler_live_cleared": not runtime.fill_preview_modal_handler_live,
                "state_cleared": runtime.fill_preview_state is None,
                "draw_handles_cleared": (
                    runtime.fill_preview_draw_handler is None
                    and runtime.fill_preview_text_draw_handler is None
                ),
                "test_hook_consumed": getattr(runtime, "_test_invoke_failure_stage", None) is None,
            })
        passed = all(
            record["invoke_result"] == ["CANCELLED"]
            and record["owner_terminal_calls"] == 1
            and record["registry_empty"]
            and record["operator_cleared"]
            and record["session_cleared"]
            and record["handler_live_cleared"]
            and record["state_cleared"]
            and record["draw_handles_cleared"]
            and record["test_hook_consumed"]
            for record in records
        )
        return {
            "passed": passed,
            "stages": tuple(record["stage"] for record in records),
            "records": tuple(records),
            "next_invoke_allowed": passed and not runtime.modal_registry,
        }
    finally:
        registration._fill_preview_modal_owner_terminal = original_owner_terminal
        if hasattr(runtime, "_test_invoke_failure_stage"):
            runtime._test_invoke_failure_stage = None


def _attempt_disable_active():
    try:
        RESULT["disable_attempted"] = True
        module = sys.modules.get("mesh_focus_orbit")
        registration = getattr(module, "registration", None)
        preflight = getattr(registration, "_lifecycle_modal_preflight", None)
        if callable(preflight):
            RESULT["lifecycle_preflight_before_disable"] = preflight()
        addon_utils.disable("mesh_focus_orbit", default_set=False)
        RESULT["disable_check_while_active"] = addon_utils.check("mesh_focus_orbit")
        RESULT["unregister_completed"] = RESULT["disable_check_while_active"][1] is False
        module = sys.modules.get("mesh_focus_orbit")
        operator_rna_present = False
        try:
            view3d_rna = bpy.ops.view3d.get_rna_type()
            operator_rna_present = any(
                getattr(item, "identifier", "") == "mesh_focus_local_face_set_grow_v2"
                for item in getattr(view3d_rna, "functions", ())
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            operator_rna_present = bool(
                hasattr(bpy.ops.view3d, "mesh_focus_local_face_set_grow_v2")
            )
        operator_class_present = hasattr(
            bpy.types, "VIEW3D_OT_mesh_focus_local_face_set_grow"
        )
        operator_absent = not operator_class_present
        runtime = getattr(module, "runtime", None)
        handlers_absent = runtime is None or not getattr(runtime, "is_registered", False)
        declared_classes = tuple(getattr(getattr(module, "registration", None), "CLASSES", ())) if module is not None else ()
        classes_absent = all(not hasattr(bpy.types, getattr(cls, "__name__", "")) for cls in declared_classes)
        handler_names = (
            "_on_load_pre", "_on_load_post", "_on_undo_post", "_on_fill_preview_redo_post",
            "_on_topology_color_depsgraph_update", "_on_topology_color_undo_post",
            "_on_topology_color_redo_post", "_on_topology_color_load_pre", "_on_topology_color_load_post",
            "_on_fill_preview_depsgraph_update", "_on_guided_ridge_depsgraph_update",
            "_on_guided_ridge_load_pre", "_on_guided_ridge_load_post",
            "_on_tube_preview_depsgraph_update", "_on_local_feature_brush_depsgraph_update",
        )
        callbacks = tuple(getattr(getattr(module, "registration", None), name, None) for name in handler_names) if module is not None else ()
        handler_lists = (
            bpy.app.handlers.load_pre, bpy.app.handlers.load_post, bpy.app.handlers.undo_post,
            bpy.app.handlers.redo_post, bpy.app.handlers.depsgraph_update_post,
        )
        exact_handlers_absent = not any(
            callback is not None and any(existing is callback for existing in handler_list)
            for callback in callbacks for handler_list in handler_lists
        )
        actual_keymap_hits = []
        try:
            addon_keyconfig = bpy.context.window_manager.keyconfigs.addon
            for keymap in tuple(getattr(addon_keyconfig, "keymaps", ()) or ()):
                for item in tuple(getattr(keymap, "keymap_items", ()) or ()):
                    idname = str(getattr(item, "idname", ""))
                    keymap_name = str(getattr(keymap, "name", ""))
                    if (
                        idname.startswith(("view3d.mesh_focus_", "sculpt.mesh_focus_", "object.mesh_focus_", "paint.mesh_focus_"))
                        or "MFO" in keymap_name
                    ):
                        actual_keymap_hits.append((keymap_name, idname))
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            actual_keymap_hits = [("<inspection-error>", "<inspection-error>")]
        RESULT["actual_addon_keymaps"] = tuple(actual_keymap_hits)
        RESULT["actual_addon_keymaps_absent"] = not actual_keymap_hits
        addon_keymaps_absent = RESULT["actual_addon_keymaps_absent"]
        handler_module_hits = []
        try:
            handler_lists = tuple(
                getattr(bpy.app.handlers, name)
                for name in dir(bpy.app.handlers)
                if not name.startswith("_")
                and isinstance(getattr(bpy.app.handlers, name), (list, tuple))
            )
            for handler_list in handler_lists:
                for callback in tuple(handler_list):
                    module_name = str(getattr(callback, "__module__", ""))
                    if module_name == "mesh_focus_orbit" or module_name.startswith("mesh_focus_orbit."):
                        handler_module_hits.append((module_name, getattr(callback, "__qualname__", "")))
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            handler_module_hits = [("<inspection-error>", "<inspection-error>")]
        RESULT["actual_handler_module_hits"] = tuple(handler_module_hits)
        RESULT["actual_handler_module_absent"] = not handler_module_hits
        RESULT["runtime_after_disable"] = {
            "is_registered": getattr(runtime, "is_registered", None) if runtime else None,
            "handler_live": getattr(runtime, "fill_preview_modal_handler_live", None) if runtime else None,
            "operator": bool(getattr(runtime, "fill_preview_modal_operator", None)) if runtime else None,
            "registry_owners": tuple(runtime.modal_registry.keys()) if runtime else (),
            "orphan_timer": bool(getattr(runtime, "modal_orphan_cleanup_timer", None)) if runtime else None,
            "orphan_timer_registered": bool(
                bpy.app.timers.is_registered(
                    getattr(sys.modules.get("mesh_focus_orbit"), "lifecycle", None)._modal_orphan_cleanup_tick
                )
            ) if runtime is not None and getattr(runtime, "modal_orphan_cleanup_timer", None) is not None else None,
            "runtime_id": id(runtime) if runtime is not None else None,
            "registration_runtime_id": id(getattr(getattr(module, "registration", None), "_runtime", None)) if module is not None else None,
            "registration_registry_owners": tuple(
                getattr(getattr(getattr(module, "registration", None), "_runtime", None), "modal_registry", {}).keys()
            ) if module is not None else (),
            "operator_rna_present": operator_rna_present,
            "operator_class_present": operator_class_present,
        }
        RESULT["classes_handlers_absent"] = bool(operator_absent and handlers_absent and classes_absent and exact_handlers_absent and addon_keymaps_absent and RESULT["actual_handler_module_absent"])
        RESULT["declared_class_count"] = len(declared_classes)
        RESULT["exact_handlers_absent"] = exact_handlers_absent
        RESULT["addon_keymaps_absent"] = addon_keymaps_absent
    except Exception as error:
        RESULT["errors"].append(f"disable while active: {error!r}")
    return _report_and_quit()


def _invoke():
    try:
        ctx = _view3d_context()
        if ctx is None:
            RESULT["errors"].append("factory startup did not expose View3D")
            return _report_and_quit()
        window, area, region = ctx
        mesh = bpy.data.meshes.new("MFO isolated GUI Smart Fill mesh")
        mesh.from_pydata(
            ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
             (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
             (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
             (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)),
            (), (
                (0, 1, 2, 3), (4, 7, 6, 5),
                (0, 4, 5, 1), (1, 5, 6, 2),
                (2, 6, 7, 3), (4, 0, 3, 7),
            ),
        )
        obj = bpy.data.objects.new("MFO isolated GUI Smart Fill object", mesh)
        bpy.context.scene.collection.objects.link(obj)
        try:
            face_set_attribute = mesh.attributes.get(".sculpt_face_set")
            if face_set_attribute is None:
                face_set_attribute = mesh.attributes.new(
                    name=".sculpt_face_set",
                    type="INT",
                    domain="FACE",
                )
            for item in face_set_attribute.data:
                item.value = 1
            RESULT["face_set_attribute"] = True
        except Exception as error:
            RESULT["face_set_attribute"] = repr(error)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        space_data = area.spaces.active
        region_data = space_data.region_3d
        with bpy.context.temp_override(
            window=window,
            area=area,
            region=region,
            space_data=space_data,
            region_data=region_data,
        ):
            bpy.context.view_layer.update()
            # bpy.ops INVOKE_DEFAULT obtains the real mouse event from the
            # window, which a hidden process cannot position.  Patch only the
            # add-on-local coordinate seam in this disposable process; the
            # operator, poll, raycast, modal_handler_add, timer and mode switch
            # remain native.
            addon = sys.modules["mesh_focus_orbit"]
            from mathutils import Vector
            addon.registration._sculpt_cursor_region_coordinate = (
                lambda _context, _event: Vector((region.width * 0.5, region.height * 0.5))
            )
            bpy.ops.object.mode_set(mode="SCULPT")
            try:
                face_set_result = bpy.ops.sculpt.face_sets_init(mode="LOOSE_PARTS")
                RESULT["face_set_init"] = sorted(str(item) for item in face_set_result)
            except Exception as error:
                RESULT["face_set_init"] = [f"FAILED: {error!r}"]
            # Frame the throwaway object, then let the actual operator invoke.
            try:
                bpy.ops.view3d.view_axis(type="FRONT", align_active=False)
                bpy.ops.view3d.view_selected(use_all_regions=False)
                bpy.ops.view3d.view_all(center=False)
            except Exception:
                pass
            try:
                import mathutils
                addon = sys.modules["mesh_focus_orbit"]
                hit = addon.guided_ridge._raycast_sculpt_face_set(
                    bpy.context,
                    mathutils.Vector((region.width * 0.5, region.height * 0.5)),
                )
                RESULT["addon_raycast_hit"] = hit is not None
            except Exception as error:
                RESULT["addon_raycast_hit"] = repr(error)
            RESULT["smart_fill_invoke_failure_matrix"] = _smart_fill_invoke_failure_matrix(
                addon,
                window,
                area,
                region,
            )
            if not RESULT["smart_fill_invoke_failure_matrix"].get("passed"):
                RESULT["errors"].append("native Smart Fill invoke-failure matrix failed")
                return _report_and_quit()
            RESULT["invoke_attempted"] = True
            result = bpy.ops.view3d.mesh_focus_local_face_set_grow_v2(
                "INVOKE_DEFAULT",
                mouse_region_x=int(region.width * 0.5),
                mouse_region_y=int(region.height * 0.5),
            )
        RESULT["invoke_result"] = sorted(str(item) for item in result)
        runtime = sys.modules["mesh_focus_orbit"].runtime
        RESULT["owner_before_cancel"] = tuple(runtime.modal_registry.keys())
        RESULT["modal_token"] = (
            next(iter(runtime.modal_registry), None)
        )
        RESULT["candidate_module_file"] = str(
            pathlib.Path(sys.modules["mesh_focus_orbit"].__file__).resolve()
        )
        RESULT["candidate_version"] = tuple(
            sys.modules["mesh_focus_orbit"].bl_info["version"]
        )
        RESULT["native_invoke_supported"] = "RUNNING_MODAL" in (RESULT["invoke_result"] or ())
        bpy.app.timers.register(_switch_mode, first_interval=0.10)
    except Exception as error:
        RESULT["errors"].append(f"invoke: {error!r}")
        RESULT["traceback"] = traceback.format_exc()
        return _report_and_quit()
    return None


_NATIVE_GUI_OPT_IN = "MFO_ENABLE_UNSAFE_MODE_SWITCH_TEST"


def run():
    # Hard stop before candidate discovery/import, registration, timers, or
    # mode operators. No default launcher sets this opt-in after the native
    # Blender 5.2 crash at the modal mode-switch boundary.
    if os.environ.get(_NATIVE_GUI_OPT_IN) != "1":
        RESULT["native_mode_switch_test_disabled_reason"] = (
            "native GUI modal/mode-switch runner disabled; "
            "set MFO_ENABLE_UNSAFE_MODE_SWITCH_TEST=1 only for supervised research"
        )
        RESULT["loader_mode"] = "disabled"
        return _report_and_quit()
    if os.environ.get("MFO_HIDDEN_USE_INSTALLED") == "1":
        RESULT["loader_mode"] = "addon-utils"
        RESULT["native_mode_switch_test_enabled"] = True
        if not _isolation_preflight():
            return _report_and_quit()
        # Factory-startup GUI processes do not necessarily refresh the user
        # scripts directory before this runner asks addon_utils to enable the
        # installed package. Refresh discovery explicitly; this does not alter
        # the connected live Blender process.
        addon_utils.modules_refresh()
        RESULT["addon_paths"] = tuple(addon_utils.paths())
        try:
            RESULT["discovered_addons"] = tuple(
                getattr(item, "__name__", "")
                for item in addon_utils.modules(refresh=False)
                if "mesh_focus_orbit" in getattr(item, "__name__", "")
            )
        except Exception as error:
            RESULT["discovery_error"] = repr(error)
        RESULT["loader_before"] = addon_utils.check("mesh_focus_orbit")
        addon_utils.enable("mesh_focus_orbit", default_set=False, persistent=False)
        RESULT["loader_enabled_check"] = addon_utils.check("mesh_focus_orbit")
        if not addon_utils.check("mesh_focus_orbit")[1]:
            RESULT["enable_check"] = addon_utils.check("mesh_focus_orbit")
            RESULT["errors"].append("addon_utils.enable did not load mesh_focus_orbit")
            return _report_and_quit()
    else:
        RESULT["loader_mode"] = "source-direct"
        source = str(SOURCE_ROOT)
        if source not in sys.path:
            sys.path.insert(0, source)
        for name in tuple(sys.modules):
            if name == "mesh_focus_orbit" or name.startswith("mesh_focus_orbit."):
                sys.modules.pop(name, None)
        module = importlib.import_module("mesh_focus_orbit")
        module.register()
    RESULT["loader_before"] = RESULT.get("loader_before") or addon_utils.check("mesh_focus_orbit")
    bpy.app.timers.register(_invoke, first_interval=0.20)
    return None


if __name__ == "__main__":
    run()
