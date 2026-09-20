"""Isolated Blender crash-boundary checks for Smart Fill.

This file is intentionally not imported by the connected/live toolbar suite.
Run it only in a disposable Blender background process. Background Blender
has no View3D window, so the invoke/raycast half of Smart Fill cannot be
started natively there; the real registered operator modal callback and an
isolated temporary mesh are still used to exercise the terminal context-change
boundary before a native mode switch.
"""

import importlib
import pathlib
import sys
from types import SimpleNamespace


def run():
    import bpy
    # Prefer the checkout over the user's installed copy in this disposable
    # process.  The live Blender session is never part of this test.
    source_root = pathlib.Path(__file__).resolve().parents[1]
    for name in tuple(sys.modules):
        if name == "mesh_focus_orbit" or name.startswith("mesh_focus_orbit."):
            sys.modules.pop(name, None)
    sys.path.insert(0, str(source_root))
    module = importlib.import_module("mesh_focus_orbit")

    # RetopoFlow's GPU shader bootstrap is unavailable in background Blender.
    # The add-on's own RNA/modal registration remains real; only the optional
    # integration hook is disabled for this isolated process.
    restore_hooks = (
        module.registration._restore_retopoflow_hooks,
        module.foundation._restore_retopoflow_hooks,
        module.foundation._release_retopoflow_hooks,
    )
    module.registration._restore_retopoflow_hooks = lambda: None
    module.foundation._restore_retopoflow_hooks = lambda: None
    module.foundation._release_retopoflow_hooks = lambda: None

    registered_here = not module.runtime.is_registered
    if registered_here:
        module.register()
    original_mode = bpy.context.mode
    original_active = bpy.context.view_layer.objects.active
    mesh = bpy.data.meshes.new("MFO isolated Smart Fill crash mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        (),
        ((0, 1, 2, 3),),
    )
    obj = bpy.data.objects.new("MFO isolated Smart Fill crash object", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    class _Ptr:
        def __init__(self, value):
            self.value = int(value)

        def as_pointer(self):
            return self.value

    area = _Ptr(7001)
    area.tag_redraw = lambda: None
    region = _Ptr(7002)
    window = _Ptr(7003)
    obj_ptr = _Ptr(7004)
    obj_ptr.data = _Ptr(7005)
    operator = SimpleNamespace()
    state = {
        "active": True,
        "operator": operator,
        "session_id": 7006,
        "area": area,
        "area_key": 7001,
        "region_key": 7002,
        "window_key": 7003,
        "obj": obj_ptr,
        "obj_pointer": 7004,
        "mesh_pointer": 7005,
        "mode": "PAINT_VERTEX",
        "timer": None,
        "metrics": {},
    }
    runtime = module.runtime
    old_state = runtime.fill_preview_state
    old_owner = (
        runtime.fill_preview_modal_operator,
        runtime.fill_preview_modal_session,
        runtime.fill_preview_modal_context,
        runtime.fill_preview_modal_handler_live,
        runtime.fill_preview_modal_cancel_requested,
        runtime.fill_preview_modal_cancel_reason,
    )
    runtime.fill_preview_state = state
    runtime.fill_preview_modal_operator = operator
    runtime.fill_preview_modal_session = 7006
    runtime.fill_preview_modal_context = (7003, 7001, 7002, 7004, 7005)
    runtime.fill_preview_modal_handler_live = True
    runtime.fill_preview_modal_cancel_requested = False
    runtime.fill_preview_modal_cancel_reason = ""
    try:
        context = SimpleNamespace(
            area=area,
            region=region,
            window=window,
            active_object=obj_ptr,
            mode="OBJECT",
        )
        result = module.registration._sfsf_modal_uninstrumented(
            operator, context, SimpleNamespace(type="MOUSEMOVE", value="NOTHING")
        )
        assert result == {"CANCELLED"}
        assert runtime.fill_preview_state is None
        assert runtime.fill_preview_modal_operator is None
        assert runtime.fill_preview_modal_session is None
        assert runtime.fill_preview_modal_handler_live is False

        # This is the native mode boundary, but on a disposable object/process.
        bpy.ops.object.mode_set(mode="SCULPT")
        assert bpy.context.mode == "SCULPT"
        bpy.ops.object.mode_set(mode="OBJECT")
        return {
            "passed": True,
            "modal_terminal_before_mode_switch": True,
            "runtime_clean": True,
            "native_mode_switch_isolated": True,
            "native_invoke_limitation": "background Blender has no View3D raycast context",
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
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.name in bpy.data.meshes:
            bpy.data.meshes.remove(mesh)
        if original_active is not None and original_active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = original_active
        if bpy.context.mode != original_mode and original_active is not None:
            try:
                bpy.ops.object.mode_set(mode=original_mode)
            except RuntimeError:
                pass
        if registered_here and module.runtime.is_registered:
            module.unregister()
        (
            module.registration._restore_retopoflow_hooks,
            module.foundation._restore_retopoflow_hooks,
            module.foundation._release_retopoflow_hooks,
        ) = restore_hooks


if __name__ == "__main__":
    print(run())
