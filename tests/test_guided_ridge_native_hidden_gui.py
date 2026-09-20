"""Disposable GUI Blender native Guided Ridge stroke proof.

This runner is independent of the connected Blender.  It creates and removes a
temporary non-identity-transform Sculpt object, applies the official ESSENTIALS
Pinch and Crease brushes through the add-on's synchronous service, and restores
the prior brush asset reference after each apply.
"""
import bpy
import importlib
import json
import pathlib
import sys
import traceback

SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULT = {"passed": False, "ridge_finished": False, "ridge_repeat": False, "groove_finished": False,
          "ridge_changed": False, "groove_changed": False,
          "asset_restored": False, "undo_attempted": False, "errors": []}


def _context():
    window = bpy.context.window
    if window is None:
        return None
    area = next((item for item in window.screen.areas if item.type == "VIEW_3D"), None)
    region = next((item for item in area.regions if item.type == "WINDOW"), None) if area else None
    return (window, area, region) if region else None


def _finish():
    RESULT["passed"] = all(
        RESULT[key] for key in (
            "ridge_finished", "ridge_repeat", "groove_finished", "ridge_changed",
            "groove_changed", "asset_restored",
        )
    )
    print(json.dumps(RESULT, sort_keys=True))
    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        pass
    return None


def _run():
    try:
        addon = sys.modules["mesh_focus_orbit"]
        service = addon.guided_ridge_curve_sculpt
        ctx = _context()
        if ctx is None:
            RESULT["errors"].append("no View3D")
            return _finish()
        window, area, region = ctx
        space_data = area.spaces.active
        region_data = space_data.region_3d
        original_object = bpy.context.view_layer.objects.active
        original_selected = tuple(bpy.context.selected_objects)
        original_mode = bpy.context.mode

        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        with bpy.context.temp_override(
            window=window, area=area, region=region,
            space_data=space_data, region_data=region_data,
        ):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=20, radius=1.0)
        obj = bpy.context.view_layer.objects.active
        obj.name = "MFO isolated GUI Guided Ridge object"
        mesh = obj.data
        obj.matrix_world = (
            __import__("mathutils").Matrix.Translation((0.35, -0.2, 0.4))
            @ __import__("mathutils").Matrix.Rotation(0.23, 4, "Z")
        )
        for vertex in mesh.vertices:
            if vertex.co.x > 0.35 and vertex.co.z > 0.15:
                vertex.co.y -= 0.25
        mesh.update()
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        with bpy.context.temp_override(
            window=window, area=area, region=region,
            space_data=space_data, region_data=region_data,
        ):
            bpy.ops.object.mode_set(mode="SCULPT")
            bpy.ops.view3d.view_axis(type="FRONT", align_active=False)
            bpy.ops.view3d.view_selected(use_all_regions=False)
            bpy.context.view_layer.update()
            sculpt = bpy.context.scene.tool_settings.sculpt
            before_asset = service._asset_reference(sculpt)
            prefs = bpy.context.preferences.addons.get("mesh_focus_orbit")
            if prefs is not None:
                prefs = prefs.preferences
                prefs.guided_ridge_ridge_strength = 1.0
                prefs.guided_ridge_ridge_radius = 120.0
            probe_state = {"operator": None}
            probe_brush = service._activate_builtin_asset(bpy.context, sculpt, probe_state, "RIDGE")
            RESULT["activated_ridge_type"] = getattr(probe_brush, "sculpt_brush_type", None)
            RESULT["activated_ridge_tool"] = getattr(probe_brush, "sculpt_tool", None)
            assert service._restore_active_tool(
                bpy.context,
                {"asset_reference": before_asset, "tool_id": None},
            )
            before_asset = service._asset_reference(sculpt)
            points = (
                (float(region.width) * 0.38, float(region.height) * 0.50),
                (float(region.width) * 0.45, float(region.height) * 0.52),
                (float(region.width) * 0.52, float(region.height) * 0.50),
                (float(region.width) * 0.59, float(region.height) * 0.48),
            )
            try:
                from bpy_extras import view3d_utils
                projected_center = view3d_utils.location_3d_to_region_2d(
                    region, region_data, obj.matrix_world.translation
                )
                RESULT["projected_center"] = tuple(float(value) for value in projected_center)
                if projected_center is not None:
                    cx, cy = float(projected_center.x), float(projected_center.y)
                    points = (
                        (cx - 55.0, cy),
                        (cx - 18.0, cy + 18.0),
                        (cx + 18.0, cy - 12.0),
                        (cx + 55.0, cy),
                    )
            except Exception as error:
                RESULT["projected_center"] = repr(error)
            state = {
                "active": True, "phase": "curve_preview", "obj": obj,
                "preview_context": bpy.context, "curve_screen_preview": points,
                "curve_screen_cache_status": "valid",
                "curve_screen_signature": ("native-hidden", region.width, region.height),
                "guide": (
                    (0.35, -0.9, 0.4),
                    (0.55, -0.9, 0.4),
                    (0.75, -0.9, 0.4),
                ),
                "operator": None,
            }
            before = tuple(tuple(float(v) for v in vertex.co) for vertex in mesh.vertices)
            first_ridge = bool(service.apply(bpy.context, state, "RIDGE"))
            active_after_ridge = getattr(sculpt, "brush", None)
            RESULT["ridge_brush_type"] = getattr(active_after_ridge, "sculpt_brush_type", None)
            RESULT["ridge_repeat"] = bool(service.apply(bpy.context, state, "RIDGE"))
            RESULT["ridge_finished"] = bool(first_ridge and RESULT["ridge_repeat"])
            after_ridge = tuple(tuple(float(v) for v in vertex.co) for vertex in mesh.vertices)
            RESULT["ridge_changed"] = before != after_ridge
            RESULT["ridge_max_delta"] = max(
                (
                    sum((after_ridge[index][axis] - before[index][axis]) ** 2 for axis in range(3))
                    ** 0.5
                    for index in range(len(before))
                ),
                default=0.0,
            )
            RESULT["asset_restored"] = service._asset_reference(sculpt) == before_asset
            RESULT["groove_finished"] = bool(service.apply(bpy.context, state, "GROOVE"))
            after_groove = tuple(tuple(float(v) for v in vertex.co) for vertex in mesh.vertices)
            RESULT["groove_changed"] = after_ridge != after_groove
            RESULT["asset_restored"] = RESULT["asset_restored"] and service._asset_reference(sculpt) == before_asset
            try:
                RESULT["undo_attempted"] = "FINISHED" in bpy.ops.ed.undo()
            except Exception as error:
                RESULT["errors"].append(f"undo: {error!r}")
            service.cleanup(state)
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        bpy.context.view_layer.objects.active = original_object
        for item in bpy.context.selected_objects:
            item.select_set(False)
        for item in original_selected:
            if item.name in bpy.data.objects:
                item.select_set(True)
        if original_object is not None and original_mode != "OBJECT":
            bpy.ops.object.mode_set(mode=original_mode)
    except Exception as error:
        RESULT["errors"].append(f"native: {error!r}")
        RESULT["traceback"] = traceback.format_exc()
    return _finish()


def run():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    for name in tuple(sys.modules):
        if name == "mesh_focus_orbit" or name.startswith("mesh_focus_orbit."):
            sys.modules.pop(name, None)
    addon = importlib.import_module("mesh_focus_orbit")
    addon.register()
    bpy.app.timers.register(_run, first_interval=0.30)


if __name__ == "__main__":
    run()
