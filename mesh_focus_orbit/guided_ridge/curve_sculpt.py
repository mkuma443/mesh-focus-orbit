"""Guided Ridge Curve Sculpt application service.

This module owns the small Step 2 bridge from the accepted screen-space curve
preview to Blender's native Sculpt Paint Curve operator.  It deliberately
does not know how the route is generated: the core session passes the already
validated, current-view screen polyline.  Keeping that boundary explicit
prevents the historical curve preview construction from being changed by the
application path.
"""

import bpy

from .. import lifecycle as _lifecycle
from .. import runtime as _runtime


_ESSENTIALS_LIBRARY = "ESSENTIALS"
_ESSENTIALS_FILE = "brushes/essentials_brushes-mesh_sculpt.blend"
_ESSENTIALS_ASSETS = {
    "RIDGE": "Pinch/Magnify",
    "GROOVE": "Crease Polish",
}

# Custom ownership marker for any temporary IDs created by the legacy
# compatibility path.  Built-in ESSENTIALS assets are never tagged or removed.
_OWNER_PROP = "_mfo_guided_ridge_owner"


def _report(operator, level, message):
    if operator is not None:
        try:
            operator.report({level}, message)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


def _remove_owned_datablock(collection, datablock):
    """Remove only the exact temporary datablock recorded by this session."""
    if datablock is None:
        return False
    try:
        remove = getattr(collection, "remove", None)
        if remove is not None:
            remove(datablock)
            return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    return False


def _remove_paint_curve(paint_curve):
    if paint_curve is None:
        return True
    try:
        if int(getattr(paint_curve, "users", 0) or 0) > 0:
            return False
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        batch_remove = getattr(bpy.data, "batch_remove", None)
        if batch_remove is not None:
            batch_remove(ids=[paint_curve])
            return True
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    return False


def _session_token(state):
    token = state.get("curve_sculpt_session_token")
    if token:
        return str(token)
    token = f"guided-ridge-{state.get('session_id', id(state))}-{id(state):x}"
    state["curve_sculpt_session_token"] = token
    return token


def _pending_restore_key(state):
    return f"guided-ridge-restore:{_session_token(state)}"


def _mark_restore_pending(context, state, restore):
    key = _pending_restore_key(state)
    state["curve_sculpt_restore_pending"] = True
    state["curve_sculpt_restore_key"] = key
    _lifecycle.pending_restore_register(
        key,
        context,
        restore,
        _restore_active_tool,
        operator=state.get("operator"),
    )
    return key


def _clear_restore_pending(state):
    if not state:
        return
    key = state.pop("curve_sculpt_restore_key", None)
    if key is not None:
        _lifecycle.pending_restore_remove(key)
    state.pop("curve_sculpt_restore_pending", None)


def _mark_owned(datablock, state, kind):
    """Tag a Blender ID when possible and always retain its identity in state."""
    token = _session_token(state)
    try:
        datablock[_OWNER_PROP] = token
    except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    key = "curve_sculpt_owned_brushes" if kind == "brush" else "curve_sculpt_owned_curves"
    owned = state.setdefault(key, {})
    owned[id(datablock)] = datablock
    created_key = "curve_sculpt_created_brushes" if kind == "brush" else "curve_sculpt_created_curves"
    created = state.setdefault(created_key, [])
    if all(candidate is not datablock for candidate in created):
        created.append(datablock)
    return datablock


def _owned_items(state, key):
    values = state.get(key) or {}
    if isinstance(values, dict):
        return tuple(values.values())
    return tuple(values)


def mesh_revision_signature(state):
    "Canonical bounded mesh signature shared by apply and depsgraph code."
    obj = state.get("obj") if state else None
    data = getattr(obj, "data", None)
    try:
        vertices = getattr(data, "vertices", ())
        count = len(vertices)
        samples = []
        for index in (0, count // 2, count - 1):
            if 0 <= index < count:
                co = vertices[index].co
                samples.append((round(float(co.x), 9), round(float(co.y), 9), round(float(co.z), 9)))
        return (
            int(obj.as_pointer()), int(data.as_pointer()), int(count),
            int(len(data.polygons)), tuple(samples),
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _transaction(state):
    tx = state.setdefault(
        "curve_sculpt_transaction",
        {
            "history": [],
            "cursor": 0,
            "pending_history": None,
            "baseline_signature": None,
            "external_change": False,
            "history_pre_direction": None,
            "history_post_direction": None,
            "history_wait_ticks": 0,
        },
    )
    tx.setdefault("history", [])
    tx.setdefault("cursor", 0)
    tx.setdefault("pending_history", None)
    tx.setdefault("history_pre_direction", None)
    tx.setdefault("history_post_direction", None)
    tx.setdefault("history_wait_ticks", 0)
    return tx


def record_transaction_apply(state, pre_signature, post_signature, mode):
    """Record one completed native stroke without taking an Undo step in Python.

    Blender's native Sculpt operator owns the actual Undo record.  The editor
    keeps only bounded signatures so Ctrl+Z/Ctrl+Shift+Z can validate the
    history change and refresh/cancel the guide safely.
    """
    if pre_signature is None or post_signature is None:
        return False
    tx = _transaction(state)
    history = tx["history"]
    cursor = max(0, min(int(tx.get("cursor", 0)), len(history)))
    if cursor < len(history):
        del history[cursor:]
    history.append({"mode": str(mode), "pre": pre_signature, "post": post_signature})
    tx["cursor"] = len(history)
    if tx.get("baseline_signature") is None:
        tx["baseline_signature"] = pre_signature
    tx["pending_history"] = None
    return True


def request_history_change(state, direction):
    """Arm standard Blender Undo/Redo pass-through for the active session."""
    if not state or not state.get("active"):
        return False
    tx = _transaction(state)
    history = tx["history"]
    cursor = int(tx.get("cursor", 0))
    direction = "redo" if str(direction).lower() == "redo" else "undo"
    if direction == "undo":
        if cursor <= 0:
            return False
        expected = history[cursor - 1]["pre"]
        index = cursor - 1
    else:
        if cursor >= len(history):
            return False
        expected = history[cursor]["post"]
        index = cursor
    tx["pending_history"] = {"direction": direction, "index": index, "expected": expected}
    tx["history_pre_direction"] = None
    tx["history_post_direction"] = None
    tx["history_last_phase"] = None
    tx["history_wait_ticks"] = 0
    state["curve_sculpt_history_reacquire_pending"] = False
    return True


def observe_history_change(state, current_signature, *, advance_wait=True):
    """Consume a validated standard Undo/Redo notification."""
    if not state or current_signature is None:
        return "unavailable"
    tx = _transaction(state)
    pending = tx.get("pending_history")
    if pending is None:
        return None
    direction = str(pending.get("direction"))
    if tx.get("history_post_direction") != direction:
        if advance_wait:
            tx["history_wait_ticks"] = int(tx.get("history_wait_ticks", 0) or 0) + 1
        if tx["history_wait_ticks"] >= 20:
            tx["pending_history"] = None
            tx["external_change"] = True
            return "external"
        return "pending"
    expected = pending.get("expected")
    signature_shape_matches = (
        current_signature == expected
        or (
            isinstance(current_signature, tuple)
            and isinstance(expected, tuple)
            and len(current_signature) >= 3
            and len(expected) >= 3
            and current_signature[2:] == expected[2:]
        )
    )
    if not signature_shape_matches:
        tx["pending_history"] = None
        tx["external_change"] = True
        return "external"
    if direction == "undo":
        tx["cursor"] = int(pending["index"])
    else:
        tx["cursor"] = int(pending["index"]) + 1
    tx["history_observed_direction"] = direction
    tx["history_wait_ticks"] = 0
    tx["history_pre_direction"] = None
    tx["history_post_direction"] = None
    tx["pending_history"] = None
    state["curve_sculpt_history_reacquire_pending"] = False
    return direction


def prepare_rollback(context, state):
    """Arm post-modal standard Undo rollback without invoking Undo here."""
    if not state or not state.get("active"):
        return False
    tx = _transaction(state)
    count = int(tx.get("cursor", 0))
    if count <= 0 or tx.get("pending_history") is not None:
        return False
    _runtime.guided_ridge_native_transaction = {
        "kind": "rollback",
        "state": state,
        "context": context or state.get("preview_context"),
        "remaining": count,
        "history": tuple(tx.get("history", ())[:count]),
        "baseline_signature": tx.get("baseline_signature"),
        "token": state.get("curve_sculpt_session_token"),
        "status": "armed",
    }
    return True


def rollback_step():
    """Run one standard Undo after the editor modal has returned."""
    tx = _runtime.guided_ridge_native_transaction
    if not tx or tx.get("kind") != "rollback":
        return None
    context = tx.get("context")
    try:
        override = _temporary_override(context)
        if override is None:
            result = bpy.ops.ed.undo()
        else:
            with override:
                result = bpy.ops.ed.undo()
        if "FINISHED" not in result:
            tx["status"] = "undo-unavailable"
            _report(None, "WARNING", "Guided Ridge: standard Undo could not restore the pre-session geometry")
            _runtime.guided_ridge_native_transaction = None
            return None
        tx["remaining"] = int(tx.get("remaining", 0)) - 1
        if tx["remaining"] <= 0:
            tx["status"] = "restored"
            _runtime.guided_ridge_native_transaction = None
            return None
        return 0.0
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        tx["status"] = "undo-error"
        _runtime.guided_ridge_native_transaction = None
        return None


def _mesh_revision_signature(context, state):
    return mesh_revision_signature(state)


def _asset_reference(sculpt):
    reference = getattr(sculpt, "brush_asset_reference", None)
    if reference is None:
        return None
    return {
        key: getattr(reference, key, "")
        for key in (
            "asset_library_type",
            "asset_library_identifier",
            "relative_asset_identifier",
        )
    }


def _active_tool_id(context):
    try:
        tool = context.workspace.tools.from_space_view3d_mode(mode=context.mode)
        return getattr(tool, "idname", None)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _pointer(value):
    try:
        return int(value.as_pointer()) if value is not None else None
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _capture_restore(context, sculpt, state):
    if state.get("curve_sculpt_restore") is not None:
        return state["curve_sculpt_restore"]
    restore = {
        "asset_reference": _asset_reference(sculpt),
        "tool_id": _active_tool_id(context),
        "window": getattr(context, "window", None),
        "workspace": getattr(context, "workspace", None),
        "area": getattr(context, "area", None),
        "region": getattr(context, "region", None),
        "window_pointer": _pointer(getattr(context, "window", None)),
        "workspace_pointer": _pointer(getattr(context, "workspace", None)),
        "workspace_name": getattr(getattr(context, "workspace", None), "name", ""),
        "area_pointer": _pointer(getattr(context, "area", None)),
        "area_type": getattr(getattr(context, "area", None), "type", ""),
        "region_pointer": _pointer(getattr(context, "region", None)),
        "region_type": getattr(getattr(context, "region", None), "type", ""),
        "scene_name": getattr(getattr(context, "scene", None), "name", ""),
        "scene_pointer": _pointer(getattr(context, "scene", None)),
        "object_pointer": _pointer(state.get("obj")),
        "object_name": getattr(state.get("obj"), "name", ""),
        "mode": getattr(context, "mode", ""),
    }
    state["curve_sculpt_restore"] = restore
    return restore


def _resolve_restore_context(fallback_context, restore):
    """Resolve the exact captured Blender context, never an arbitrary current one."""
    if not restore:
        return fallback_context
    # Older unit fixtures intentionally omit identity fields.  Production
    # records created by _capture_restore always contain them and take the
    # strict path below.
    identity_keys = (
        "window_pointer", "workspace_pointer", "area_pointer", "region_pointer",
        "scene_pointer", "object_pointer",
    )
    if not any(restore.get(key) is not None for key in identity_keys):
        return fallback_context
    try:
        wm = getattr(getattr(bpy, "context", None), "window_manager", None)
        windows = tuple(getattr(wm, "windows", ()) or ())
        expected_window = restore.get("window_pointer")
        window = next(
            (item for item in windows if _pointer(item) == expected_window),
            None,
        )
        if window is None:
            return None
        workspace = getattr(window, "workspace", None)
        if restore.get("workspace_pointer") is not None and _pointer(workspace) != restore.get("workspace_pointer"):
            return None
        if restore.get("workspace_name") and getattr(workspace, "name", "") != restore.get("workspace_name"):
            return None
        screen = getattr(window, "screen", None)
        area = next(
            (item for item in tuple(getattr(screen, "areas", ()) or ())
             if _pointer(item) == restore.get("area_pointer")
             and getattr(item, "type", "") == restore.get("area_type", "")),
            None,
        )
        if area is None:
            return None
        region = next(
            (item for item in tuple(getattr(area, "regions", ()) or ())
             if _pointer(item) == restore.get("region_pointer")
             and getattr(item, "type", "") == restore.get("region_type", "")),
            None,
        )
        if region is None:
            return None
        scene = getattr(window, "scene", None)
        if restore.get("scene_pointer") is not None and _pointer(scene) != restore.get("scene_pointer"):
            return None
        if restore.get("scene_name") and getattr(scene, "name", "") != restore.get("scene_name"):
            return None
        object_name = restore.get("object_name")
        obj = None
        if object_name:
            obj = next(
                (candidate for candidate in tuple(getattr(scene, "objects", ()) or ())
                 if getattr(candidate, "name", "") == object_name),
                None,
            )
            if obj is None or (_pointer(obj) != restore.get("object_pointer")):
                return None
        view_layer = getattr(window, "view_layer", None)
        active = getattr(getattr(view_layer, "objects", None), "active", None)
        if obj is not None and (active is None or _pointer(active) != _pointer(obj)):
            return None
        if obj is not None and restore.get("mode") and getattr(obj, "mode", "") != restore.get("mode"):
            return None
        base = getattr(bpy, "context", None)
        if base is None or not hasattr(base, "temp_override"):
            return None
        return base, window, workspace, scene, area, region, obj
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _restore_active_tool(context, restore):
    if not restore:
        return True
    resolved = _resolve_restore_context(context, restore)
    if resolved is None:
        return False
    if isinstance(resolved, tuple):
        base_context, window, workspace, scene, area, region, obj = resolved
        try:
            override = base_context.temp_override(
                window=window, area=area, region=region,
                workspace=workspace, scene=scene,
            )
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            return False
    else:
        base_context = resolved
        override = _temporary_override(base_context)
    try:
        reference = restore.get("asset_reference")
        if reference and reference.get("relative_asset_identifier"):
            kwargs = {
                "asset_library_type": reference.get("asset_library_type", "ESSENTIALS"),
                "asset_library_identifier": reference.get("asset_library_identifier", ""),
                "relative_asset_identifier": reference["relative_asset_identifier"],
                "use_toggle": False,
            }
            if override is None:
                result = bpy.ops.brush.asset_activate(**kwargs)
            else:
                with override:
                    result = bpy.ops.brush.asset_activate(**kwargs)
            if "FINISHED" not in result:
                return False
        tool_id = restore.get("tool_id")
        if tool_id:
            if override is None:
                result = bpy.ops.wm.tool_set_by_id(name=tool_id)
            else:
                with override:
                    result = bpy.ops.wm.tool_set_by_id(name=tool_id)
            if result and "FINISHED" not in result:
                return False
        return True
    except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _activate_builtin_asset(context, sculpt, state, mode):
    asset_name = _ESSENTIALS_ASSETS["GROOVE" if mode == "GROOVE" else "RIDGE"]
    kwargs = {
        "asset_library_type": _ESSENTIALS_LIBRARY,
        "asset_library_identifier": "",
        "relative_asset_identifier": f"{_ESSENTIALS_FILE}/Brush/{asset_name}",
        "use_toggle": False,
    }
    override = _temporary_override(context)
    if override is None:
        result = bpy.ops.brush.asset_activate(**kwargs)
    else:
        with override:
            result = bpy.ops.brush.asset_activate(**kwargs)
    if "FINISHED" not in result:
        raise RuntimeError("Blender ESSENTIALS brush activation was cancelled")
    reference = _asset_reference(sculpt)
    if not reference or reference.get("relative_asset_identifier") != kwargs["relative_asset_identifier"]:
        raise RuntimeError("Blender did not activate the requested ESSENTIALS brush")
    brush = getattr(sculpt, "brush", None)
    expected = "CREASE" if mode == "GROOVE" else "PINCH"
    if brush is None or getattr(brush, "sculpt_brush_type", None) != expected:
        raise RuntimeError("activated ESSENTIALS brush has an unexpected sculpt type")
    state["curve_sculpt_native_asset"] = True
    state["curve_sculpt_native_asset_reference"] = reference
    return brush


def _stroke_points(context, state, points):
    from bpy_extras import view3d_utils
    from mathutils import Vector

    region = getattr(context, "region", None)
    rv3d = getattr(context, "region_data", None)
    guide = state.get("guide") or state.get("controls") or ()
    depth = Vector(guide[len(guide) // 2]) if guide else Vector((0.0, 0.0, 0.0))
    stroke = []
    prefs = None
    try:
        prefs = bpy.context.preferences.addons[__package__.split(".")[0]].preferences
    except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    pressure = float(getattr(prefs, "guided_ridge_ridge_strength", 0.35) if prefs else 0.35)
    size = float(getattr(prefs, "guided_ridge_ridge_radius", 40.0) if prefs else 40.0)
    for index, point in enumerate(points):
        mouse = (float(point[0]), float(point[1]))
        if region is not None and rv3d is not None:
            location = view3d_utils.region_2d_to_location_3d(region, rv3d, mouse, depth)
        else:
            location = depth
        stroke.append({
            "name": "",
            "location": tuple(float(value) for value in location),
            "mouse": mouse,
            "mouse_event": mouse,
            "pressure": pressure,
            "size": size,
            "x_tilt": 0.0,
            "y_tilt": 0.0,
            "time": float(index) * 0.01,
            "is_start": index == 0,
        })
    return stroke


def _apply_native_stroke(context, state, points):
    if len(points) < 2:
        raise RuntimeError("at least two curve samples are required")
    override = _temporary_override(context)
    stroke = _stroke_points(context, state, points)
    state["curve_sculpt_native_points"] = tuple(
        (float(point[0]), float(point[1])) for point in points
    )
    if override is None:
        result = bpy.ops.sculpt.brush_stroke(stroke=stroke, override_location=True)
    else:
        with override:
            result = bpy.ops.sculpt.brush_stroke(stroke=stroke, override_location=True)
    if "RUNNING_MODAL" in result or "PASS_THROUGH" in result or "FINISHED" not in result:
        raise RuntimeError(f"native sculpt stroke did not finish synchronously: {result}")
    try:
        context.view_layer.update()
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    return result


def retry_restore(context, state):
    """Retry a failed brush/tool restore without discarding its exact record."""
    if not state:
        return True
    restore = state.get("curve_sculpt_restore")
    if not restore:
        _clear_restore_pending(state)
        return True
    context = context or state.get("preview_context")
    if not _restore_active_tool(context, restore):
        _mark_restore_pending(context, state, restore)
        return False
    state.pop("curve_sculpt_restore", None)
    _clear_restore_pending(state)
    state["curve_sculpt_native_asset"] = False
    return True


def cleanup(state):
    """Restore the user's sculpt brush and release our temporary Paint Curve."""
    if not state:
        return
    restore = state.get("curve_sculpt_restore")
    if restore and "asset_reference" in restore:
        context = state.get("preview_context")
        if not retry_restore(context, state):
            _report(state.get("operator"), "ERROR", "Guided Ridge: previous brush/tool restoration is pending")
            return False
        for key in (
            "curve_sculpt_apply_active",
            "curve_sculpt_native_asset",
            "curve_sculpt_active",
        ):
            state[key] = False
        state["curve_sculpt_expected_update_generation"] = None
        state["curve_sculpt_expected_mesh_signature"] = None
        state["curve_sculpt_mode"] = None
        state["curve_sculpt_last_apply"] = None
        return True
    restore = state.pop("curve_sculpt_restore", None)
    restore_asset = state.pop("curve_sculpt_restore_asset_reference", None)
    paint_curve = state.pop("curve_sculpt_paint_curve", None)
    sculpt = None
    context = state.get("preview_context")
    try:
        scene = getattr(context, "scene", None)
        tool_settings = getattr(scene, "tool_settings", None)
        sculpt = getattr(tool_settings, "sculpt", None)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        sculpt = None
    owned_curves = list(_owned_items(state, "curve_sculpt_owned_curves"))
    # paintcurve.new attaches the ID before _apply_native_curve returns.  Keep
    # that exact attachment in the owned set so a draw exception cannot leak
    # the partially created curve.
    if paint_curve is not None and all(candidate is not paint_curve for candidate in owned_curves):
        owned_curves.append(paint_curve)
    owned_brushes = _owned_items(state, "curve_sculpt_owned_brushes")
    managed_brushes = _owned_items(state, "curve_sculpt_managed_brushes")
    managed_settings = state.pop("curve_sculpt_managed_brush_settings", {}) or {}
    if sculpt is not None and paint_curve is not None:
        try:
            current_brush = getattr(sculpt, "brush", None)
            if current_brush is not None and getattr(current_brush, "paint_curve", None) is paint_curve:
                current_brush.paint_curve = None
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
    if restore and sculpt is not None:
        try:
            sculpt.brush = restore.get("brush")
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        brush = restore.get("brush")
        if brush is not None:
            for key in ("stroke_method", "paint_curve", "strength", "size", "sculpt_tool"):
                if key not in restore:
                    continue
                try:
                    setattr(brush, key, restore[key])
                except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                    pass
    if restore_asset and sculpt is not None:
        try:
            result = bpy.ops.brush.asset_activate(
                asset_library_type=restore_asset["asset_library_type"],
                asset_library_identifier=restore_asset.get("asset_library_identifier", ""),
                relative_asset_identifier=restore_asset["relative_asset_identifier"],
                use_toggle=False,
            )
            if "FINISHED" not in result:
                raise RuntimeError("original brush asset activation was cancelled")
        except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
            try:
                sculpt.brush = restore.get("brush")
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                pass
    for brush in managed_brushes:
        settings = managed_settings.get(id(brush), {})
        try:
            if getattr(brush, "paint_curve", None) in owned_curves:
                brush.paint_curve = None
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        for key, value in settings.items():
            try:
                setattr(brush, key, value)
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                pass
    for brush in owned_brushes:
        try:
            if getattr(brush, "paint_curve", None) in owned_curves:
                brush.paint_curve = None
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        _remove_owned_datablock(getattr(bpy.data, "brushes", None), brush)
    for owned_curve in owned_curves:
        _remove_paint_curve(owned_curve)
    state["curve_sculpt_owned_brushes"] = {}
    state["curve_sculpt_managed_brushes"] = {}
    state["curve_sculpt_managed_brush_settings"] = {}
    state["curve_sculpt_managed_preexisting_curves"] = {}
    state["curve_sculpt_owned_curves"] = {}
    state["curve_sculpt_created_brushes"] = []
    state["curve_sculpt_created_curves"] = []
    state["curve_sculpt_brushes_by_mode"] = {}
    state["curve_sculpt_native_points"] = ()
    state["curve_sculpt_apply_active"] = False
    state["curve_sculpt_expected_update_generation"] = None
    state["curve_sculpt_expected_update_budget"] = 0
    state["curve_sculpt_expected_mesh_signature"] = None
    state["curve_sculpt_target_object_pointer"] = None
    state["curve_sculpt_target_mesh_pointer"] = None
    state["curve_sculpt_session_token"] = None
    state["curve_sculpt_active"] = False
    state["curve_sculpt_mode"] = None
    state["curve_sculpt_last_apply"] = None


def _temporary_override(context):
    if context is None or not hasattr(context, "temp_override"):
        return None
    window = getattr(context, "window", None)
    area = getattr(context, "area", None)
    region = getattr(context, "region", None)
    if window is None or area is None or region is None:
        return None
    return context.temp_override(window=window, area=area, region=region)


def _legacy_paintcurve_apply(context, state, mode="RIDGE"):
    # Retained only as an explicit tombstone for pre-3.3.83 callers.
    raise RuntimeError("the modal Paint Curve path was removed; use apply()")


def apply(context, state, mode="RIDGE"):
    """Apply the displayed route synchronously with a Blender ESSENTIALS brush.

    Dyntopo is deliberately a hard safety boundary here.  ``sculpt.brush_stroke``
    owns Blender's native paint/BMLog transaction, while this service is called
    from the long-lived Guided Ridge modal.  Starting that native stroke from
    an active Dyntopo modal can leave a vpaint modal callback alive after Esc and
    corrupt the subsequent ``BM_log_undo`` restore.  There is no supported
    Python API to split that native transaction safely, so refuse before brush
    activation and leave the mesh, undo stack, and session untouched.
    """
    if state is None or not state.get("active"):
        return False
    if state.get("phase") not in {"curve_preview", "curve_sculpt"}:
        return False
    mode = "GROOVE" if str(mode).upper() == "GROOVE" else "RIDGE"
    points = tuple(state.get("curve_screen_preview") or ())
    if len(points) < 2 or state.get("curve_screen_cache_status") != "valid":
        _report(state.get("operator"), "WARNING", "Guided Ridge: current-view curve is not ready")
        return False
    obj = state.get("obj")
    try:
        if bool(getattr(obj, "use_dynamic_topology_sculpting", False)):
            reason = "Guided Ridge: Dyntopo is active; curve stroke is disabled to protect Blender's native Undo"
            state["curve_sculpt_apply_rejected_reason"] = "dyntopo-native-undo-boundary"
            _report(state.get("operator"), "WARNING", reason)
            return False
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        state["curve_sculpt_apply_rejected_reason"] = "dyntopo-state-unavailable"
        _report(state.get("operator"), "WARNING", "Guided Ridge: Dyntopo state is unavailable; curve stroke was not started")
        return False
    context = context or state.get("preview_context")
    sculpt = getattr(getattr(getattr(context, "scene", None), "tool_settings", None), "sculpt", None)
    if sculpt is None:
        _report(state.get("operator"), "WARNING", "Guided Ridge: Sculpt settings are unavailable")
        return False
    restore = _capture_restore(context, sculpt, state)
    generation = int(state.get("curve_sculpt_apply_generation", 0) or 0) + 1
    state["curve_sculpt_apply_generation"] = generation
    state["curve_sculpt_expected_update_generation"] = generation
    state["curve_sculpt_apply_active"] = True
    try:
        _activate_builtin_asset(context, sculpt, state, mode)
        pre_signature = mesh_revision_signature(state)
        state["curve_sculpt_target_object_pointer"] = pre_signature[0] if pre_signature else None
        state["curve_sculpt_target_mesh_pointer"] = pre_signature[1] if pre_signature else None
        _apply_native_stroke(context, state, points)
        post_signature = mesh_revision_signature(state)
        if post_signature is None:
            raise RuntimeError("mesh revision could not be captured after synchronous stroke")
        if not record_transaction_apply(state, pre_signature, post_signature, mode):
            raise RuntimeError("Guided Ridge transaction signature could not be recorded")
        state["curve_sculpt_expected_mesh_signature"] = post_signature
        state["curve_sculpt_apply_active"] = False
        state["curve_sculpt_expected_update_generation"] = None
        state["curve_sculpt_expected_update_budget"] = 0
        state["curve_sculpt_active"] = True
        state["curve_sculpt_mode"] = mode
        state["curve_sculpt_last_apply"] = mode
        state["curve_sculpt_ridge_applications"] = int(state.get("curve_sculpt_ridge_applications", 0) or 0) + (mode == "RIDGE")
        state["curve_sculpt_groove_applications"] = int(state.get("curve_sculpt_groove_applications", 0) or 0) + (mode == "GROOVE")
        state["phase"] = "curve_sculpt"
        if not _restore_active_tool(context, restore):
            _mark_restore_pending(context, state, restore)
            _report(state.get("operator"), "ERROR", "Guided Ridge: previous brush/tool restoration is pending")
            return False
        state.pop("curve_sculpt_restore", None)
        _clear_restore_pending(state)
        return True
    except (AttributeError, IndexError, MemoryError, ReferenceError, RuntimeError, TypeError, ValueError) as error:
        state["curve_sculpt_apply_active"] = False
        state["curve_sculpt_expected_update_generation"] = None
        state["curve_sculpt_expected_update_budget"] = 0
        restored = _restore_active_tool(context, restore)
        if not restored:
            _mark_restore_pending(context, state, restore)
            _report(state.get("operator"), "ERROR", "Guided Ridge: previous brush/tool restoration is pending")
        else:
            state.pop("curve_sculpt_restore", None)
            _clear_restore_pending(state)
        state["curve_sculpt_native_asset"] = False
        _report(state.get("operator"), "WARNING", f"Guided Ridge: synchronous Sculpt stroke unavailable ({error})")
        return False


__all__ = (
    "apply",
    "cleanup",
    "retry_restore",
    "mesh_revision_signature",
    "record_transaction_apply",
    "request_history_change",
    "observe_history_change",
    "prepare_rollback",
    "rollback_step",
)
