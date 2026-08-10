"""Mesh Focus Orbit.

Double-tap the configured key in a 3D Viewport to make the first visible mesh
surface under the viewport center the temporary orbit target.  The built-in
viewport navigation, including the Navigation Gizmo, remains responsible for
rotating the view.
"""

bl_info = {
    "name": "Mesh Focus Orbit",
    "author": "OpenAI",
    "version": (1, 4, 0),
    "blender": (5, 2, 0),
    "location": "3D View",
    "description": "Temporarily orbit around the visible mesh surface at the viewport center",
    "category": "3D View",
}

import bpy
import bmesh
import gpu
import time

from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.geometry import intersect_ray_tri, tessellate_polygon


OPERATOR_ID = "view3d.mesh_focus_orbit"

_addon_keymaps = []
_active_states = {}
_modal_operator_areas = {}
_last_tap_times = {}
_is_registered = False

DEFAULT_DOUBLE_TAP_WINDOW = 0.28


ACTIVATION_ITEMS = (
    ("LEFT_CTRL", "Left Ctrl", "Double-tap Left Ctrl to toggle temporary orbit"),
    ("RIGHT_CTRL", "Right Ctrl", "Double-tap Right Ctrl to toggle temporary orbit"),
    ("LEFT_SHIFT", "Left Shift", "Double-tap Left Shift to toggle temporary orbit"),
    ("RIGHT_SHIFT", "Right Shift", "Double-tap Right Shift to toggle temporary orbit"),
    ("LEFT_ALT", "Left Alt", "Double-tap Left Alt to toggle temporary orbit"),
    ("RIGHT_ALT", "Right Alt", "Double-tap Right Alt to toggle temporary orbit"),
)


FOCUS_LOSS_ITEMS = (
    (
        "KEEP",
        "Keep Mode",
        "Keep Mesh Focus Orbit active when the Blender window loses focus",
    ),
    (
        "EXIT",
        "Exit Mode",
        "Exit Mesh Focus Orbit when the Blender window loses focus",
    ),
)


def _addon_preferences():
    """Return this add-on's preferences when Blender has created them."""
    addon_id = __package__ or __name__
    addon = bpy.context.preferences.addons.get(addon_id)
    return addon.preferences if addon else None


def _tag_redraw(area):
    try:
        if area and area.type == "VIEW_3D":
            area.tag_redraw()
    except (ReferenceError, AttributeError, TypeError):
        pass


class _TempOrbitState:
    """State for one active temporary orbit in one viewport."""

    __slots__ = (
        "area",
        "region",
        "rv3d",
        "original_view_location",
        "original_view_distance",
        "original_view_rotation",
        "original_view_perspective",
        "hit_location",
        "hit_distance",
        "temporary_start_distance",
        "is_perspective",
        "activation_key",
        "draw_handler",
        "active",
    )

    def __init__(self, context, hit_location, hit_distance, activation_key):
        self.area = context.area
        self.region = context.region
        self.rv3d = context.space_data.region_3d
        self.original_view_location = self.rv3d.view_location.copy()
        self.original_view_distance = self.rv3d.view_distance
        self.original_view_rotation = self.rv3d.view_rotation.copy()
        self.original_view_perspective = self.rv3d.view_perspective
        self.hit_location = hit_location.copy()
        self.hit_distance = hit_distance
        self.temporary_start_distance = (
            hit_distance if self.rv3d.view_perspective in {"PERSP", "CAMERA"}
            else self.original_view_distance
        )
        self.is_perspective = self.rv3d.view_perspective in {"PERSP", "CAMERA"}
        self.activation_key = activation_key
        self.draw_handler = None
        self.active = True


def _world_ray_from_view_center(context):
    """Return (origin, direction) for the exact center of the 3D View region."""
    region = context.region
    rv3d = context.space_data.region_3d
    center_2d = Vector((region.width * 0.5, region.height * 0.5))

    try:
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, center_2d)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, center_2d)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None

    if direction.length_squared == 0.0:
        return None
    return origin, direction.normalized()


def _object_is_visible(context, obj):
    """Filter objects using viewport visibility, not selection state."""
    try:
        if obj.hide_get(view_layer=context.view_layer):
            return False
    except (AttributeError, RuntimeError, TypeError):
        try:
            if obj.hide_get():
                return False
        except (AttributeError, RuntimeError, TypeError):
            pass

    try:
        if not obj.visible_get(
            view_layer=context.view_layer,
            viewport=context.space_data,
        ):
            return False
    except (AttributeError, RuntimeError, TypeError):
        try:
            if not obj.visible_get():
                return False
        except (AttributeError, RuntimeError, TypeError):
            pass

    # Wire and bounds display has no visible surface to use as an orbit target.
    return obj.display_type not in {"WIRE", "BOUNDS"}


def _distance_along_ray(origin, direction, point):
    distance = (point - origin).dot(direction)
    return distance if distance > 1.0e-7 else None


def _raycast_object(obj, depsgraph, origin, direction):
    """Ray cast an evaluated object and return (world_point, distance)."""
    try:
        evaluated = obj.evaluated_get(depsgraph)
        matrix_world = evaluated.matrix_world
        inverse = matrix_world.inverted_safe()
        local_origin = inverse @ origin
        local_direction = (inverse.to_3x3() @ direction)
        if local_direction.length_squared == 0.0:
            return None
        local_direction.normalize()

        hit, local_location, _normal, _face_index = evaluated.ray_cast(
            local_origin,
            local_direction,
        )
        if not hit:
            return None

        world_location = matrix_world @ local_location
        distance = _distance_along_ray(origin, direction, world_location)
        return (world_location, distance) if distance is not None else None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _raycast_edit_object(obj, origin, direction):
    """Ray cast the live BMesh so unsaved Edit Mode changes are included."""
    try:
        bm = bmesh.from_edit_mesh(obj.data)
    except (AttributeError, RuntimeError, TypeError):
        return None

    matrix_world = obj.matrix_world
    best = None

    for face in bm.faces:
        if not face.is_valid or face.hide:
            continue

        local_vertices = [loop.vert.co.copy() for loop in face.loops]
        if len(local_vertices) < 3 or any(loop.vert.hide for loop in face.loops):
            continue

        if len(local_vertices) == 3:
            triangles = (local_vertices,)
        else:
            try:
                triangle_indices = tessellate_polygon([local_vertices])
                triangles = tuple(
                    tuple(local_vertices[index] for index in triangle)
                    for triangle in triangle_indices
                )
            except (RuntimeError, TypeError, ValueError):
                # A fan is a useful fallback for unusual or temporarily
                # invalid n-gons while the user is editing.
                triangles = tuple(
                    (local_vertices[0], local_vertices[index], local_vertices[index + 1])
                    for index in range(1, len(local_vertices) - 1)
                )

        for triangle in triangles:
            world_triangle = tuple(matrix_world @ vertex for vertex in triangle)
            hit_location = intersect_ray_tri(
                world_triangle[0],
                world_triangle[1],
                world_triangle[2],
                direction,
                origin,
                True,
            )
            if hit_location is None:
                continue

            distance = _distance_along_ray(origin, direction, hit_location)
            if distance is not None and (best is None or distance < best[1]):
                best = (hit_location.copy(), distance)

    return best


def _find_center_hit(context):
    """Find the nearest visible mesh surface under the viewport center."""
    ray = _world_ray_from_view_center(context)
    if ray is None:
        return None
    origin, direction = ray
    depsgraph = context.evaluated_depsgraph_get()
    best = None

    for obj in context.scene.objects:
        if obj.type != "MESH" or not _object_is_visible(context, obj):
            continue

        if context.mode == "EDIT_MESH" and obj.mode == "EDIT":
            candidate = _raycast_edit_object(obj, origin, direction)
        else:
            candidate = _raycast_object(obj, depsgraph, origin, direction)

        if candidate is not None and (best is None or candidate[1] < best[1]):
            best = candidate

    return best


def _draw_debug_point(state):
    if not state.active:
        return

    try:
        shader = gpu.shader.from_builtin("POINT_UNIFORM_COLOR")
        batch = batch_for_shader(
            shader,
            "POINTS",
            {"pos": [state.hit_location]},
        )
        shader.bind()
        shader.uniform_float("color", (0.05, 0.55, 1.0, 1.0))
        shader.uniform_float("size", 10.0)
        batch.draw(shader)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # Debug drawing is optional and must never affect viewport navigation.
        pass


def _start_debug_draw(state):
    prefs = _addon_preferences()
    if not prefs or not prefs.debug_display:
        return
    try:
        state.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_debug_point,
            (state,),
            "WINDOW",
            "POST_VIEW",
        )
    except (AttributeError, RuntimeError, TypeError):
        state.draw_handler = None


def _stop_debug_draw(state):
    if state.draw_handler is None:
        return
    try:
        bpy.types.SpaceView3D.draw_handler_remove(state.draw_handler, "WINDOW")
    except (AttributeError, RuntimeError, TypeError):
        pass
    state.draw_handler = None


def _activate_state(state):
    """Move only the orbit target, keeping the camera fixed at activation."""
    rv3d = state.rv3d
    if state.is_perspective and state.hit_distance is not None:
        # Changing view_location alone would move the camera.  Match the
        # camera-to-hit distance so Ctrl activation itself does not jump.
        rv3d.view_distance = max(state.hit_distance, 1.0e-6)

    rv3d.view_location = state.hit_location
    prefs = _addon_preferences()
    try:
        state.area.header_text_set(
            "MESH FOCUS ORBIT ON" if not prefs or prefs.show_indicator else None
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        pass
    _start_debug_draw(state)
    _tag_redraw(state.area)


def _restore_original_view(state):
    """Restore the complete viewport transform saved at activation."""
    if not state:
        return
    try:
        rv3d = state.rv3d
        rv3d.view_perspective = state.original_view_perspective
        rv3d.view_location = state.original_view_location
        rv3d.view_distance = state.original_view_distance
        rv3d.view_rotation = state.original_view_rotation
        rv3d.update()
    except (ReferenceError, RuntimeError, AttributeError, TypeError, ValueError):
        pass


def _deactivate_state(state):
    """Exit temporary mode and restore the complete saved viewport view."""
    if not state or not state.active:
        return
    area_key = state.area.as_pointer()
    _finish_state(state)
    _active_states.pop(area_key, None)
    for operator_key, mapped_area_key in list(_modal_operator_areas.items()):
        if mapped_area_key == area_key:
            _modal_operator_areas.pop(operator_key, None)


def _finish_state(state):
    """End the modal watcher and restore the saved viewport transform."""
    if not state or not state.active:
        return
    _restore_original_view(state)
    state.active = False
    try:
        state.area.header_text_set(None)
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        pass
    _stop_debug_draw(state)
    _tag_redraw(state.area)


def _finish_all_states():
    for key, state in list(_active_states.items()):
        _finish_state(state)
        _active_states.pop(key, None)
    _modal_operator_areas.clear()
    _last_tap_times.clear()


@persistent
def _on_load_pre(_dummy):
    """Clear viewport-bound state before Blender replaces the current file."""
    _finish_all_states()


def _operator_key(operator):
    """Get a stable key without storing Python attributes on bpy.types.Operator."""
    try:
        return operator.as_pointer()
    except (AttributeError, ReferenceError, RuntimeError):
        return id(operator)


def _finish_operator(operator):
    operator_key = _operator_key(operator)
    area_key = _modal_operator_areas.pop(operator_key, None)
    if area_key is None:
        return
    state = _active_states.pop(area_key, None)
    _finish_state(state)


class VIEW3D_OT_mesh_focus_orbit(bpy.types.Operator):
    """Double-tap the configured key to toggle the viewport-center orbit target."""

    bl_idname = OPERATOR_ID
    bl_label = "Mesh Focus Orbit"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
            and context.space_data is not None
            and context.mode in {"OBJECT", "EDIT_MESH"}
        )

    def invoke(self, context, event):
        if not self.poll(context):
            return {"PASS_THROUGH"}

        area_key = context.area.as_pointer()
        prefs = _addon_preferences()
        activation_key = prefs.activation_key if prefs else "RIGHT_SHIFT"
        double_tap_window = (
            prefs.double_tap_window if prefs else DEFAULT_DOUBLE_TAP_WINDOW
        )

        if event.type != activation_key or event.value != "PRESS":
            return {"PASS_THROUGH"}

        now = time.monotonic()
        previous_tap = _last_tap_times.get(area_key)
        if previous_tap is None or now - previous_tap > double_tap_window:
            _last_tap_times[area_key] = now
            # A single tap must remain a normal Blender modifier event.
            return {"PASS_THROUGH"}

        _last_tap_times.pop(area_key, None)
        existing_state = _active_states.get(area_key)
        if existing_state and existing_state.active:
            _deactivate_state(existing_state)
            return {"FINISHED"}

        hit = _find_center_hit(context)
        if hit is None:
            # No surface at the center: do nothing and do not report an error.
            return {"PASS_THROUGH"}

        state = _TempOrbitState(context, hit[0], hit[1], activation_key)
        operator_key = _operator_key(self)
        _active_states[area_key] = state
        _modal_operator_areas[operator_key] = area_key

        try:
            _activate_state(state)
        except (ReferenceError, RuntimeError, AttributeError, TypeError, ValueError):
            _active_states.pop(area_key, None)
            _modal_operator_areas.pop(operator_key, None)
            state.active = False
            return {"PASS_THROUGH"}

        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, _context, event):
        area_key = _modal_operator_areas.get(_operator_key(self))
        state = _active_states.get(area_key)
        if state is None or not state.active:
            return {"CANCELLED"}

        if event.type == "WINDOW_DEACTIVATE":
            prefs = _addon_preferences()
            if prefs and prefs.focus_loss_behavior == "EXIT":
                _finish_operator(self)
                return {"CANCELLED"}
            return {"PASS_THROUGH"}

        # Passing all events through is what lets Blender's native Navigation
        # Gizmo and MMB orbit continue to handle the gesture.
        return {"PASS_THROUGH"}

    def __del__(self):
        # Covers cancellation during workspace/window teardown.
        try:
            _finish_operator(self)
        except (ReferenceError, RuntimeError, AttributeError):
            pass


def _remove_keymaps():
    for keymap, keymap_item in _addon_keymaps:
        try:
            keymap.keymap_items.remove(keymap_item)
        except (ReferenceError, RuntimeError, ValueError):
            pass
    _addon_keymaps.clear()


def _rebuild_keymaps():
    if not _is_registered:
        return

    _remove_keymaps()
    prefs = _addon_preferences()
    if prefs is not None and not prefs.enabled:
        return

    try:
        keyconfig = bpy.context.window_manager.keyconfigs.addon
        if keyconfig is None:
            return
        keymap = keyconfig.keymaps.new(
            name="3D View",
            space_type="VIEW_3D",
            region_type="WINDOW",
        )
        activation_key = prefs.activation_key if prefs else "RIGHT_SHIFT"
        keymap_item = keymap.keymap_items.new(OPERATOR_ID, activation_key, "PRESS")
        # The configured event itself is a modifier key.  Ignore any other
        # modifier state so Ctrl+... navigation does not disable activation.
        keymap_item.any = True
        _addon_keymaps.append((keymap, keymap_item))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        _remove_keymaps()


def _preferences_changed(_self, _context):
    if _is_registered:
        _rebuild_keymaps()


class MESH_FOCUS_ORBIT_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__ or __name__

    enabled: BoolProperty(
        name="Enable",
        description="Enable the temporary orbit modifier",
        default=True,
        update=_preferences_changed,
    )
    activation_key: EnumProperty(
        name="Activation Key",
        description="Key to double-tap for entering or leaving temporary orbit",
        items=ACTIVATION_ITEMS,
        default="RIGHT_SHIFT",
        update=_preferences_changed,
    )
    focus_loss_behavior: EnumProperty(
        name="Focus Loss Behavior",
        description="Choose what happens when the Blender window loses focus",
        items=FOCUS_LOSS_ITEMS,
        default="KEEP",
    )
    double_tap_window: FloatProperty(
        name="Double-tap Window",
        description="Maximum time in seconds between the two taps",
        default=DEFAULT_DOUBLE_TAP_WINDOW,
        min=0.15,
        max=0.60,
        precision=2,
    )
    debug_display: BoolProperty(
        name="Debug Display",
        description="Draw a point at the current temporary orbit center",
        default=False,
    )
    show_indicator: BoolProperty(
        name="Show Mode Indicator",
        description="Show MESH FOCUS ORBIT ON in the 3D Viewport",
        default=True,
    )

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "enabled")
        layout.prop(self, "activation_key")
        layout.prop(self, "focus_loss_behavior")
        layout.prop(self, "debug_display")
        layout.prop(self, "show_indicator")
        layout.prop(self, "double_tap_window")
        layout.separator()
        layout.label(text="Double-tap the activation key in a 3D Viewport.")
        layout.label(text="The viewport center is ray-cast once on activation.")


CLASSES = (
    VIEW3D_OT_mesh_focus_orbit,
    MESH_FOCUS_ORBIT_AddonPreferences,
)


def register():
    global _is_registered
    if _is_registered:
        return
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    _is_registered = True
    if _on_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_load_pre)
    _rebuild_keymaps()


def unregister():
    global _is_registered
    if not _is_registered:
        return
    _finish_all_states()
    if _on_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_on_load_pre)
    _remove_keymaps()
    _is_registered = False
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
