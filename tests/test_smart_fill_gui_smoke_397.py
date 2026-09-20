"""One-shot isolated Blender GUI smoke for Smart Fill visibility invariants.

This runner is deliberately opt-in and never changes mode while Smart Fill is
modal.  It is launched only by the supervised monitor-2 wrapper.  The normal
test suite does not execute this file.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
import time
import traceback
from types import SimpleNamespace


_OPT_IN = "MFO_ENABLE_SAFE_SMART_FILL_GUI"
_RESULT_PATH = os.environ.get("MFO_GUI_RESULT", "")
_SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_RESULT = {
    "runner": "test_smart_fill_gui_smoke_397",
    "errors": [],
    "warnings": [],
    "stage_results": [],
    "cleanup": {},
}


def _write_result():
    payload = json.dumps(_RESULT, sort_keys=True, default=str)
    print("MFO_GUI_RESULT=" + payload, flush=True)
    if _RESULT_PATH:
        pathlib.Path(_RESULT_PATH).write_text(payload, encoding="utf-8")


def _quit_later():
    try:
        import bpy

        bpy.ops.wm.quit_blender()
    except Exception as error:  # pragma: no cover - final process guard
        _RESULT["errors"].append("quit: " + repr(error))
    return None


def _context_parts():
    import bpy

    window = bpy.context.window
    if window is None:
        raise RuntimeError("no active Blender window")
    area = next((item for item in window.screen.areas if item.type == "VIEW_3D"), None)
    if area is None:
        raise RuntimeError("no VIEW_3D area in test window")
    region = next((item for item in area.regions if item.type == "WINDOW"), None)
    if region is None:
        raise RuntimeError("no WINDOW region in test View3D")
    return window, area, region


def _event(kind, value="PRESS", *, ctrl=False, shift=False):
    return SimpleNamespace(
        type=kind,
        value=value,
        ctrl=ctrl,
        shift=shift,
        is_repeat=False,
        mouse_region_x=0,
        mouse_region_y=0,
        timer=None,
    )


def _as_ids(value):
    if value is None:
        return []
    try:
        return sorted({int(item) for item in value})
    except (TypeError, ValueError):
        return []


def _result_snapshot(state):
    result = state.get("result") or {}
    faces = _as_ids(result.get("faces"))
    confirm = result.get("confirm_geometry")
    confirm_ids = _as_ids(confirm.get("face_ids")) if isinstance(confirm, dict) else []
    boundary = result.get("boundary_records")
    candidate = result.get("candidate_faces")
    return {
        "phase": str(state.get("phase")),
        "generation": int(state.get("generation", -1)),
        "radius": float(state.get("processed_radius") or 0.0),
        "faces": faces,
        "count": len(faces),
        "seed": int(state.get("seed_face", -1)),
        "boundary_count": len(boundary) if boundary is not None else None,
        "candidate_count": len(candidate) if candidate is not None else None,
        "confirm_count": len(confirm_ids),
        "confirm_contains_display": bool(not faces or set(faces).issubset(confirm_ids)),
        "result_keys": sorted(str(key) for key in result.keys()),
    }


def _record_stage(runtime, label):
    state = runtime.fill_preview_state
    if state is None:
        raise RuntimeError("Smart Fill state disappeared at " + label)
    snapshot = _result_snapshot(state)
    snapshot["label"] = label
    _RESULT["stage_results"].append(snapshot)
    return snapshot


def _wait_ready(runtime, timeout=45.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = runtime.fill_preview_state
        if state is not None and state.get("phase") == "ready" and state.get("result") is not None:
            return state
        time.sleep(0.05)
    state = runtime.fill_preview_state
    raise RuntimeError(
        "Smart Fill did not reach ready: "
        + repr({key: state.get(key) for key in ("phase", "prepare_stage", "generation")})
        if state is not None
        else "Smart Fill state missing"
    )


def _invoke_modal_event(runtime, window, area, region, operator, event):
    import bpy

    with bpy.context.temp_override(window=window, area=area, region=region):
        return operator.modal(bpy.context, event)


def _make_onion_mesh():
    import bpy

    verts = []
    faces = []
    grid = 4
    for layer_z in (0.0, 0.06):
        base = len(verts)
        for row in range(grid):
            for col in range(grid):
                verts.append((float(col) - 1.5, float(row) - 1.5, layer_z))
        for row in range(grid - 1):
            for col in range(grid - 1):
                i = base + row * grid + col
                faces.append((i, i + 1, i + grid + 1, i + grid))
    mesh = bpy.data.meshes.new("MFO_GUI_OnionShell_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("MFO_GUI_OnionShell", mesh)
    bpy.context.collection.objects.link(obj)
    attr = mesh.attributes.new(name=".sculpt_face_set", type="INT", domain="FACE")
    values = [10] * 9 + [20] * 9
    for index, value in enumerate(values):
        attr.data[index].value = value
    # The outer shell is hidden before entering Sculpt; the inner shell remains
    # the only visible seed component despite being spatially close.
    for poly in mesh.polygons[:9]:
        poly.hide = True
    mesh.update()
    return obj, 9 + 4


def _prepare_scene():
    import bpy

    window, area, region = _context_parts()
    obj, seed_face = _make_onion_mesh()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.object.mode_set(mode="SCULPT")
        bpy.ops.view3d.view_axis(type="TOP", align_active=False)
        bpy.ops.view3d.view_selected(use_all_regions=False)
    _RESULT["scene"] = {
        "object": obj.name,
        "faces": len(obj.data.polygons),
        "hidden_faces": sum(1 for poly in obj.data.polygons if poly.hide),
        "seed_face": seed_face,
        "mode": bpy.context.mode,
        "area": area.type,
        "region": region.type,
        "region_size": (int(region.width), int(region.height)),
        "region_origin": (int(region.x), int(region.y)),
    }
    return window, area, region, obj, seed_face


_GUI = {
    "step": "setup",
    "deadline": 0.0,
    "module": None,
    "runtime": None,
    "window": None,
    "area": None,
    "region": None,
    "operator": None,
    "seed_face": None,
    "grow_index": 0,
    "expected_generation": None,
    "before_faces": set(),
    "shrink": None,
    "done": False,
}


def _finish_gui(runtime, passed=False, error=None):
    import bpy

    if _GUI["done"]:
        return
    _GUI["done"] = True
    if error is not None:
        _RESULT["errors"].append(repr(error))
        _RESULT["traceback"] = traceback.format_exc()
        try:
            operator = runtime.fill_preview_modal_operator if runtime is not None else None
            if operator is not None:
                _invoke_modal_event(
                    runtime,
                    _GUI["window"],
                    _GUI["area"],
                    _GUI["region"],
                    operator,
                    _event("ESC"),
                )
        except Exception as cleanup_error:
            _RESULT["errors"].append("cleanup: " + repr(cleanup_error))
    _RESULT["cleanup"] = {
        "state_none": bool(runtime is not None and runtime.fill_preview_state is None),
        "owners": (
            sorted(str(key) for key in runtime.modal_registry)
            if runtime is not None
            else []
        ),
        "handler_live": bool(
            runtime is not None and runtime.fill_preview_modal_handler_live
        ),
        "timer": (
            runtime.fill_preview_state.get("timer")
            if runtime is not None and runtime.fill_preview_state
            else None
        ),
    }
    if passed:
        _RESULT["passed"] = True
    _write_result()
    bpy.app.timers.register(_quit_later, first_interval=0.25)


def _run():
    """Non-blocking state machine; each invocation returns to Blender's UI."""
    import addon_utils
    import bpy

    if _GUI["done"]:
        return None
    if os.environ.get(_OPT_IN) != "1":
        _RESULT["disabled"] = "safe GUI runner requires explicit opt-in"
        _write_result()
        _GUI["done"] = True
        bpy.app.timers.register(_quit_later, first_interval=0.1)
        return None
    runtime = _GUI.get("runtime")
    try:
        now = time.monotonic()
        if _GUI["step"] == "setup":
            if str(_SOURCE_ROOT) not in sys.path:
                sys.path.insert(0, str(_SOURCE_ROOT))
            for name in tuple(sys.modules):
                if name == "mesh_focus_orbit" or name.startswith("mesh_focus_orbit."):
                    sys.modules.pop(name, None)
            module = importlib.import_module("mesh_focus_orbit")
            module.register()
            runtime = module.runtime
            _GUI["module"] = module
            _GUI["runtime"] = runtime
            _RESULT["module_file"] = str(pathlib.Path(module.__file__).resolve())
            _RESULT["version"] = tuple(module.bl_info.get("version", ()))
            _RESULT["addon_check"] = tuple(addon_utils.check("mesh_focus_orbit"))
            window, area, region, _obj, seed_face = _prepare_scene()
            _GUI.update(
                window=window,
                area=area,
                region=region,
                seed_face=seed_face,
            )
            cx, cy = int(region.width * 0.5), int(region.height * 0.5)
            with bpy.context.temp_override(window=window, area=area, region=region):
                try:
                    window.cursor_warp(int(region.x + cx), int(region.y + cy))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                invoke_result = bpy.ops.view3d.mesh_focus_local_face_set_grow_v2(
                    "INVOKE_DEFAULT",
                    mouse_region_x=cx,
                    mouse_region_y=cy,
                )
            _RESULT["invoke_result"] = sorted(str(item) for item in invoke_result)
            _RESULT["owner_before_ready"] = sorted(str(key) for key in runtime.modal_registry)
            if "RUNNING_MODAL" not in invoke_result:
                raise RuntimeError("Smart Fill invoke did not return RUNNING_MODAL")
            if not runtime.modal_registry:
                raise RuntimeError("modal owner registry empty after native invoke")
            _GUI["operator"] = runtime.fill_preview_modal_operator
            if _GUI["operator"] is None:
                raise RuntimeError("runtime Smart Fill operator missing")
            _GUI["deadline"] = now + 45.0
            _GUI["step"] = "wait_initial"
            return 0.05

        state = runtime.fill_preview_state
        if _GUI["step"] == "wait_initial":
            if state is not None and state.get("phase") == "ready" and state.get("result") is not None:
                first = _record_stage(runtime, "initial")
                if _GUI["seed_face"] not in first["faces"]:
                    raise RuntimeError("seed face was not retained in initial preview")
                if not first["confirm_contains_display"]:
                    raise RuntimeError("initial display/confirm mapping mismatch")
                _GUI["grow_index"] = 0
                _GUI["step"] = "grow_arm"
                return 0.05
            if now > _GUI["deadline"]:
                raise RuntimeError(
                    "Smart Fill did not reach ready: "
                    + repr({key: state.get(key) for key in ("phase", "prepare_stage", "generation")})
                    if state is not None
                    else "Smart Fill state missing"
                )
            return 0.05

        if _GUI["step"] == "grow_arm":
            if state is None:
                raise RuntimeError("Smart Fill state disappeared before grow")
            if state.get("wheel_armed"):
                _GUI["before_faces"] = set(_as_ids((state.get("result") or {}).get("faces")))
                _GUI["expected_generation"] = int(state.get("generation", -1))
                dispatch = _invoke_modal_event(
                    runtime, _GUI["window"], _GUI["area"], _GUI["region"],
                    _GUI["operator"], _event("WHEELUPMOUSE"),
                )
                _RESULT.setdefault("wheel_returns", []).append(
                    sorted(str(item) for item in dispatch)
                )
                _GUI["step"] = "grow_wait"
                _GUI["deadline"] = now + 20.0
                return 0.05
            if now > _GUI["deadline"]:
                _GUI["deadline"] = now + 20.0
                raise RuntimeError("wheel did not re-arm before grow stage")
            return 0.05

        if _GUI["step"] == "grow_wait":
            if state is not None and state.get("phase") == "ready" and state.get("result") is not None and int(state.get("generation", -1)) > _GUI["expected_generation"]:
                current = _record_stage(runtime, "grow-%d" % (_GUI["grow_index"] + 1))
                current_ids = set(current["faces"])
                if not _GUI["before_faces"].issubset(current_ids):
                    raise RuntimeError("grow stage lost previously selected faces")
                if _GUI["seed_face"] not in current_ids:
                    raise RuntimeError("grow stage lost seed face")
                if not current["confirm_contains_display"]:
                    raise RuntimeError("grow stage display/confirm mapping mismatch")
                _GUI["grow_index"] += 1
                _GUI["step"] = "grow_arm" if _GUI["grow_index"] < 3 else "shrink_arm"
                _GUI["deadline"] = now + 20.0
                return 0.05
            if now > _GUI["deadline"]:
                raise RuntimeError("grow stage did not reach a new ready result")
            return 0.05

        if _GUI["step"] == "shrink_arm":
            if state is None:
                raise RuntimeError("Smart Fill state disappeared before shrink")
            if state.get("wheel_armed"):
                _GUI["expected_generation"] = int(state.get("generation", -1))
                dispatch = _invoke_modal_event(
                    runtime, _GUI["window"], _GUI["area"], _GUI["region"],
                    _GUI["operator"], _event("WHEELDOWNMOUSE"),
                )
                _RESULT["shrink_return"] = sorted(str(item) for item in dispatch)
                _GUI["step"] = "shrink_wait"
                _GUI["deadline"] = now + 20.0
                return 0.05
            if now > _GUI["deadline"]:
                raise RuntimeError("wheel did not re-arm before shrink")
            return 0.05

        if _GUI["step"] == "shrink_wait":
            if state is not None and state.get("phase") == "ready" and state.get("result") is not None and int(state.get("generation", -1)) > _GUI["expected_generation"]:
                _GUI["shrink"] = _record_stage(runtime, "shrink")
                _GUI["step"] = "regrow_arm"
                _GUI["deadline"] = now + 20.0
                return 0.05
            if now > _GUI["deadline"]:
                raise RuntimeError("shrink did not reach a new ready result")
            return 0.05

        if _GUI["step"] == "regrow_arm":
            if state is None:
                raise RuntimeError("Smart Fill state disappeared before regrow")
            if state.get("wheel_armed"):
                _GUI["expected_generation"] = int(state.get("generation", -1))
                dispatch = _invoke_modal_event(
                    runtime, _GUI["window"], _GUI["area"], _GUI["region"],
                    _GUI["operator"], _event("WHEELUPMOUSE"),
                )
                _RESULT["regrow_return"] = sorted(str(item) for item in dispatch)
                _GUI["step"] = "regrow_wait"
                _GUI["deadline"] = now + 20.0
                return 0.05
            if now > _GUI["deadline"]:
                raise RuntimeError("wheel did not re-arm before regrow")
            return 0.05

        if _GUI["step"] == "regrow_wait":
            if state is not None and state.get("phase") == "ready" and state.get("result") is not None and int(state.get("generation", -1)) > _GUI["expected_generation"]:
                regrow = _record_stage(runtime, "regrow")
                if not set(_GUI["shrink"]["faces"]).issubset(set(regrow["faces"])):
                    raise RuntimeError("regrow lost faces from immediately previous shrink")
                dispatch = _invoke_modal_event(
                    runtime, _GUI["window"], _GUI["area"], _GUI["region"],
                    _GUI["operator"], _event("ESC"),
                )
                _RESULT["cancel_return"] = sorted(str(item) for item in dispatch)
                _GUI["step"] = "cancel_wait"
                _GUI["deadline"] = now + 5.0
                return 0.05
            if now > _GUI["deadline"]:
                raise RuntimeError("regrow did not reach a new ready result")
            return 0.05

        if _GUI["step"] == "cancel_wait":
            if runtime.fill_preview_state is None and not runtime.modal_registry:
                if _RESULT.get("cancel_return") != ["CANCELLED"]:
                    raise RuntimeError("ESC did not return CANCELLED")
                if runtime.fill_preview_modal_handler_live:
                    raise RuntimeError("Smart Fill handler remained live after ESC")
                _finish_gui(runtime, passed=True)
                return None
            if now > _GUI["deadline"]:
                raise RuntimeError("Smart Fill cleanup did not reach terminal state")
            return 0.05
        return 0.05
    except Exception as error:
        _finish_gui(runtime, passed=False, error=error)
        return None


if __name__ == "__main__":
    import bpy

    bpy.app.timers.register(_run, first_interval=0.5)
