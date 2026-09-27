"""Select local exposed back faces from a nearby edit-mesh click."""

from collections import deque
from math import cos, sin, tau

import bmesh
import bpy
from bpy.props import IntProperty
from bpy_extras import view3d_utils
from mathutils import Vector

from .config import EXPOSED_BACKFACE_SELECT_OPERATOR_ID
from .guided_ridge.core import _sculpt_cursor_region_coordinate as _cursor_region_coordinate


ROOT_RINGS = 3
LOCAL_SEARCH_STEPS = 20
MAX_LOCAL_FACES = 5_000
PROBE_RADII_PX = (24, 56, 104, 160)
PROBE_DIRECTIONS = 8
PROBES_PER_TICK = 3
TRACE_FACES_PER_TICK = 256


def _probe_points(coordinate, region):
    """Try the click first, then nearby pixels from the center outwards."""
    points = []
    seen = set()
    for radius in (0, *PROBE_RADII_PX):
        count = 1 if radius == 0 else PROBE_DIRECTIONS
        for step in range(count):
            angle = tau * step / count
            point = (
                round(coordinate.x + radius * cos(angle)),
                round(coordinate.y + radius * sin(angle)),
            )
            if (
                point not in seen
                and 0 <= point[0] < region.width
                and 0 <= point[1] < region.height
            ):
                points.append(point)
                seen.add(point)
    return points


def _include_root_faces(patch, *, rings=ROOT_RINGS):
    """Expand the local back faces like three Select More operations."""
    selected = set(patch)
    frontier = set(patch)
    for _ in range(rings):
        next_frontier = set()
        for face in frontier:
            for edge in face.edges:
                if len(edge.link_faces) != 2:
                    continue
                neighbor = edge.link_faces[0] if edge.link_faces[1] is face else edge.link_faces[1]
                if not neighbor.hide and neighbor not in selected:
                    selected.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return selected


class VIEW3D_OT_mesh_focus_select_exposed_backface(bpy.types.Operator):
    """Select nearby inverted faces by clicking the damaged surface or its rim."""

    bl_idname = EXPOSED_BACKFACE_SELECT_OPERATOR_ID
    bl_label = "Select Exposed Back Faces"
    bl_description = "Select local back-facing faces and roots from a nearby surface click"
    bl_options = {"REGISTER", "UNDO"}

    region_x: IntProperty(default=-1, options={"HIDDEN", "SKIP_SAVE"})
    region_y: IntProperty(default=-1, options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "active_object", None)
        return bool(
            getattr(context, "area", None) is not None
            and context.area.type == "VIEW_3D"
            and getattr(context, "region", None) is not None
            and context.region.type == "WINDOW"
            and getattr(context, "mode", None) == "EDIT_MESH"
            and obj is not None
            and obj.type == "MESH"
        )

    def invoke(self, context, event):
        if not self.poll(context):
            return {"CANCELLED"}
        coordinate = _cursor_region_coordinate(context, event)
        if coordinate is None:
            return {"CANCELLED"}
        return self._start(context, coordinate)

    def execute(self, context):
        if not self.poll(context) or self.region_x < 0 or self.region_y < 0:
            return {"CANCELLED"}
        return self._start(context, (self.region_x, self.region_y))

    def _start(self, context, coordinate):
        coordinate = Vector(coordinate)
        if not (
            0 <= coordinate.x < context.region.width
            and 0 <= coordinate.y < context.region.height
        ):
            return {"CANCELLED"}
        self._obj = context.active_object
        self._area = context.area
        self._region = context.region
        self._window = context.window
        self._workspace = context.workspace
        self._wm = context.window_manager
        self._coordinate = coordinate
        self._bm = None
        self._original_faces = set()
        self._original_history = ()
        self._added_faces = set()
        self._selection_started = False
        self._phase = "PREPARE"
        self._timer = None
        self._progress_started = False
        try:
            self._wm.progress_begin(0, 100)
            self._progress_started = True
            self._workspace.status_text_set("MFO: preparing local selection 0% (Esc to cancel)")
            self._timer = self._wm.event_timer_add(0.03, window=self._window)
            self._wm.modal_handler_add(self)
        except Exception as error:
            return self._finish(False, f"MFO: cannot start selection: {error}")
        return {"RUNNING_MODAL"}

    def _prepare_step(self, context):
        with context.temp_override(area=self._area, region=self._region):
            bpy.ops.mesh.select_mode(type="FACE")
        self._bm = bmesh.from_edit_mesh(self._obj.data)
        self._original_faces = {face for face in self._bm.faces if face.select}
        self._original_history = tuple(self._bm.select_history)
        self._points = _probe_points(self._coordinate, self._region)
        self._probe_index = 0
        self._first_hit = None
        self._phase = "PROBE"
        self._wm.progress_update(5)
        self._workspace.status_text_set("MFO: scanning nearby faces 5% (Esc to cancel)")
        return {"RUNNING_MODAL"}

    def _local_view_direction(self, point):
        direction = view3d_utils.region_2d_to_vector_3d(
            self._region, self._area.spaces.active.region_3d, point
        )
        local = self._obj.matrix_world.inverted_safe().to_3x3() @ direction
        if local.length_squared <= 1.0e-20:
            return None
        local.normalize()
        return local

    def _begin_trace(self, face, point):
        local_direction = self._local_view_direction(point)
        if local_direction is None:
            return False
        self._anchor = face
        self._local_direction = local_direction
        self._queue = deque([(face, 0)])
        self._visited = {face}
        self._backfaces = set()
        self._max_depth = 0
        self._truncated = False
        self._phase = "TRACE"
        self._wm.progress_update(50)
        self._workspace.status_text_set("MFO: tracing nearby faces 50% (Esc to cancel)")
        return True

    def _probe_step(self, context):
        for _ in range(PROBES_PER_TICK):
            if self._probe_index >= len(self._points):
                break
            point = self._points[self._probe_index]
            self._probe_index += 1
            self._bm.select_history.clear()
            with context.temp_override(area=self._area, region=self._region):
                bpy.ops.view3d.select(extend=True, deselect_all=False, location=point)
            face = self._bm.select_history.active
            if isinstance(face, bmesh.types.BMFace) and face.is_valid:
                if face not in self._original_faces:
                    self._added_faces.add(face)
                if self._first_hit is None:
                    self._first_hit = (face, point)
                local_direction = self._local_view_direction(point)
                if local_direction is not None and face.normal.dot(local_direction) <= 0.0:
                    if self._begin_trace(face, point):
                        return {"RUNNING_MODAL"}
            progress = 5 + int(40 * self._probe_index / len(self._points))
            self._wm.progress_update(progress)
            self._workspace.status_text_set(
                f"MFO: scanning nearby faces {self._probe_index}/{len(self._points)}"
                f" ({progress}%)"
                " (Esc to cancel)"
            )
        if self._probe_index >= len(self._points):
            if self._first_hit is None:
                return self._finish(False, "MFO: no nearby mesh face")
            if not self._begin_trace(*self._first_hit):
                return self._finish(False, "MFO: cannot determine view direction")
        return {"RUNNING_MODAL"}

    def _trace_step(self):
        processed = 0
        while self._queue and processed < TRACE_FACES_PER_TICK:
            face, depth = self._queue.popleft()
            processed += 1
            self._max_depth = max(self._max_depth, depth)
            if face.normal.dot(self._local_direction) > 0.0:
                self._backfaces.add(face)
            if depth >= LOCAL_SEARCH_STEPS:
                continue
            for edge in face.edges:
                if len(edge.link_faces) != 2:
                    continue
                neighbor = edge.link_faces[0] if edge.link_faces[1] is face else edge.link_faces[1]
                if neighbor.hide or neighbor in self._visited:
                    continue
                if len(self._visited) >= MAX_LOCAL_FACES:
                    self._truncated = True
                    continue
                self._visited.add(neighbor)
                self._queue.append((neighbor, depth + 1))
        if self._queue:
            progress = min(89, 50 + int(39 * self._max_depth / LOCAL_SEARCH_STEPS))
            self._wm.progress_update(progress)
            self._workspace.status_text_set(
                f"MFO: tracing local surface {len(self._visited)} faces"
                f" ({progress}%)"
                " (Esc to cancel)"
            )
            return {"RUNNING_MODAL"}
        self._phase = "SELECT"
        self._wm.progress_update(90)
        self._workspace.status_text_set("MFO: selecting nearby faces 90%")
        return {"RUNNING_MODAL"}

    def _select_step(self, context):
        patch = self._backfaces or {self._anchor}
        selected = _include_root_faces(patch)
        self._selection_started = True
        with context.temp_override(area=self._area, region=self._region):
            bpy.ops.mesh.select_all(action="DESELECT")
        for face in selected:
            face.select_set(True)
        self._bm.select_flush_mode()
        bmesh.update_edit_mesh(self._obj.data, loop_triangles=False, destructive=False)
        self._wm.progress_update(100)
        if self._backfaces:
            message = (
                f"MFO: selected {len(selected)} faces around "
                f"{len(self._backfaces)} local back faces"
            )
        else:
            message = f"MFO: no local back faces; selected {len(selected)} nearby faces"
        if self._truncated:
            message += " (local search limit reached)"
        return self._finish(True, message)

    def _finish(self, success, message):
        if not success and self._bm is not None:
            try:
                if self._obj.mode == "EDIT":
                    if self._selection_started:
                        with bpy.context.temp_override(area=self._area, region=self._region):
                            bpy.ops.mesh.select_all(action="DESELECT")
                        for face in self._original_faces:
                            if face.is_valid:
                                face.select_set(True)
                    else:
                        for face in self._added_faces:
                            if face.is_valid and face not in self._original_faces:
                                face.select_set(False)
                    self._bm.select_history.clear()
                    for element in self._original_history:
                        if element.is_valid and element.select:
                            self._bm.select_history.add(element)
                    self._bm.select_flush_mode()
                    bmesh.update_edit_mesh(
                        self._obj.data, loop_triangles=False, destructive=False
                    )
            except (AttributeError, ReferenceError, RuntimeError, ValueError):
                pass
        if self._timer is not None:
            self._wm.event_timer_remove(self._timer)
            self._timer = None
        if self._progress_started:
            self._wm.progress_end()
            self._progress_started = False
        self._workspace.status_text_set(None)
        self.report({"INFO"} if success else {"WARNING"}, message)
        return {"FINISHED"} if success else {"CANCELLED"}

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            return self._finish(False, "MFO: selection cancelled")
        if context.window != self._window or self._obj.mode != "EDIT":
            return self._finish(False, "MFO: edit session changed")
        if event.type != "TIMER":
            return {"RUNNING_MODAL"}
        try:
            if self._phase == "PREPARE":
                return self._prepare_step(context)
            if self._phase == "PROBE":
                return self._probe_step(context)
            if self._phase == "TRACE":
                return self._trace_step()
            return self._select_step(context)
        except Exception as error:
            return self._finish(False, f"MFO: selection failed: {error}")
