"""Registration and UI component.

Loaded by the package entry point in dependency order. This module owns
registration, tools, and UI; cross-domain state is imported explicitly from
its owning module.
"""

import bpy
import bmesh
import blf
import copy
import gpu
import hashlib
import json
import math
import os
import statistics
import struct
import tempfile
import time
import numpy as np
from array import array
from collections import deque
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
from mathutils.bvhtree import BVHTree

from .config import (
    FACE_SET_ACTIVATION_OPERATOR_ID,
    EXPOSED_BACKFACE_SELECT_OPERATOR_ID,
    FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD,
    FILL_PREVIEW_TERMINAL_GROWTH_STAGES,
    FILL_PREVIEW_TERMINAL_MAX_RADIUS_FACTOR,
    FILL_PREVIEW_TERMINAL_THRESHOLD_FACTOR,
    FILL_PREVIEW_WHEEL_DRAIN_SECONDS,
    GUIDED_RIDGE_CURVE_DEFAULT_SMOOTHING,
    GUIDED_RIDGE_CURVE_SCULPT_STEP1,
    GUIDED_RIDGE_DISTANCE_MAX_PAIRS,
    GUIDED_RIDGE_KEY,
    GUIDED_RIDGE_MAX_CANDIDATE_FACES,
    GUIDED_RIDGE_MAX_CONTROLS,
    GUIDED_RIDGE_PREPARE_WORK_CHUNK,
    GUIDED_RIDGE_UI_PROTOTYPE,
    GUIDED_RIDGE_WIDTH_ANCHOR_BAND_FRACTION,
    GUIDED_RIDGE_WIDTH_EDGE_MULTIPLIER,
    GUIDED_RIDGE_WIDTH_GUIDE_BAND_FRACTION,
    GUIDED_RIDGE_WIDTH_MAX_EXTENT_FRACTION,
    GUIDED_RIDGE_WIDTH_MIN_EDGE_MULTIPLIER,
    GUIDED_RIDGE_WIDTH_STEP_FACTOR,
    GUIDED_RIDGE_OPERATOR_ID,
    LOCAL_FACE_SET_GROW_KEY,
    LOCAL_FACE_SET_GROW_OPERATOR_ID,
    LOCAL_FEATURE_BRUSH_KEY,
    LOCAL_FEATURE_BRUSH_OPERATOR_ID,
    LOCAL_FEATURE_BRUSH_MARKER_CATALOG_ID,
    LOCAL_FEATURE_BRUSH_MARKER_PROPERTY,
    LOCAL_FEATURE_BRUSH_MARKER_VALUE,
    OPERATOR_ID,
    OPEN_BOUNDARY_LOOP_OPERATOR_ID,
    RECOVER_FACE_SET_STATE_OPERATOR_ID,
    TOOL_FACE_SET_OPERATOR_ID,
    TOOL_NORMAL_OPERATOR_ID,
    TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID,
    TOPOLOGY_COLOR_ATTRIBUTE_NAME,
    TOPOLOGY_COLOR_PALETTE,
    TOPOLOGY_COLOR_PANEL_CATEGORY,
    TUBE_SHAPE_KEY,
    TUBE_SHAPE_OPERATOR_ID,
    WATCHER_OPERATOR_ID,
)

# Keep the timer gate name local to registration while the value remains owned
# by config; this avoids an implicit facade lookup during Smart Fill dispatch.
_FILL_PREVIEW_WHEEL_DRAIN_SECONDS = FILL_PREVIEW_WHEEL_DRAIN_SECONDS
from .smart_fill.invariants import make_mesh_global_identity_face_ids
from . import lifecycle as _lifecycle
from . import runtime as _runtime
from .local_remesh import (
    MESH_OT_mesh_focus_local_remesh,
    local_remesh_request_cancel as _local_remesh_request_cancel,
)
from .smart_fill.invariants import (
    accepted_graph_rows_from_result,
    normal_cache_hit_allowed,
    publish_bounded_result,
)

# Registration ownership belongs to the process runtime singleton.  Do not
# reset it at import time: importlib.reload() must see an active registration
# and unregister the old RNA/keymap/handler objects before child reloads.

# Explicit cross-component dependencies.  The previous monolithic loader made
# these names appear through a shared globals dictionary; keep the same public
# identifiers while making every dependency visible to the Python import
# system.  Mutable state is imported from foundation's singleton owner.
from .foundation import (
    ACTIVATION_ITEMS,
    DEFAULT_DOUBLE_TAP_WINDOW,
    _FaceSetOrbitState,
    _addon_preferences,
    _cancel_orphan_cleanup,
    _cancel_undo_orphan_cleanup,
    _cleanup_orphan_face_set_proxies,
    _finish_all_states,
    FOCUS_LOSS_ITEMS,
    _invalidate_topology_color_cache,
    _next_session_id,
    _on_fill_preview_redo_post,
    _on_load_post,
    _on_load_pre,
    _on_topology_color_depsgraph_update,
    _on_topology_color_load_post,
    _on_topology_color_load_pre,
    _on_topology_color_redo_post,
    _on_topology_color_undo_post,
    _on_undo_post,
    _restore_retopoflow_hooks,
    _schedule_orphan_cleanup,
    _start_topology_color_draw,
    _stop_topology_color_draw,
    _topology_color_object,
)
from .foundation import (
    _reference_object_poll,
)
from .guided_ridge.core import (
    VIEW3D_OT_mesh_focus_face_set_activate,
    VIEW3D_OT_mesh_focus_face_set_tool,
    VIEW3D_OT_mesh_focus_guided_ridge,
    VIEW3D_OT_mesh_focus_orbit,
    VIEW3D_OT_mesh_focus_orbit_recover_face_set_state,
    VIEW3D_OT_mesh_focus_orbit_tool,
    VIEW3D_OT_mesh_focus_orbit_watcher,
    _guided_ridge_cancel,
    _guided_ridge_request_cancel,
    _guided_ridge_face_set_attribute,
    _guided_ridge_prototype_route_enabled,
    _on_guided_ridge_depsgraph_update,
    _on_guided_ridge_load_post,
    _on_guided_ridge_load_pre,
    _on_guided_ridge_redo_post,
    _on_guided_ridge_redo_pre,
    _on_guided_ridge_undo_post,
    _on_guided_ridge_undo_pre,
    _raycast_sculpt_face_set,
    _raycast_visible_mesh_face,
    _sculpt_cursor_region_coordinate,
    _vertex_paint_active_color_attribute,
    _vertex_paint_color_signature,
    _vertex_paint_geometry_compatibility,
    _vertex_paint_sample_color,
)
from .backface_select import VIEW3D_OT_mesh_focus_select_exposed_backface
from .local_feature import (
    VIEW3D_OT_mesh_focus_local_feature_brush,
    VIEW3D_OT_mesh_focus_local_feature_brush_stroke,
    _local_feature_cancel_all,
    _local_feature_request_cancel_all,
    _on_local_feature_brush_depsgraph_update,
    _on_local_feature_brush_load_post,
    _on_local_feature_brush_load_pre,
    _on_local_feature_brush_redo_post,
    _on_local_feature_brush_undo_post,
)
from .tube_shape import (
    VIEW3D_OT_mesh_focus_tube_shape,
    _on_tube_preview_depsgraph_update,
    _tube_preview_cancel,
    _tube_preview_request_cancel,
)
from .smart_fill.preview import (
    _display_distance_cancel_all,
    _display_distance_load_pre,
    _display_distance_save_post,
    _display_distance_save_pre,
    VIEW3D_OT_mesh_focus_display_distance_toggle,
    VIEW3D_OT_mesh_focus_shadow_analysis_toggle,
)
from .smart_fill.geometry import (
    _fill_preview_adjacency_steps,
    _fill_preview_array_fingerprint,
    _fill_preview_cached_visibility_check,
    _fill_preview_cursor_prepare_steps,
    _fill_preview_prepare_token,
    _fill_preview_refresh_cached_adjacency,
    _fill_preview_restore_manual_analysis_profiles,
    _fill_preview_shadow_view_key,
    _fill_preview_signature,
    _fill_preview_topology_fingerprint,
)
from .smart_fill.preview import (
    _fill_preview_accept_normal_wheel,
    _fill_preview_green_wheel_up_is_noop,
    _fill_preview_cancel,
    _fill_preview_request_cancel,
    _fill_preview_confirm_flood,
    _fill_preview_modal_context_matches,
    _fill_preview_draw,
    _fill_preview_draw_text_handler,
    _on_fill_preview_depsgraph_update,
    _fill_preview_make_result,
    _fill_preview_make_expand_only_result,
    _fill_preview_initial_radius,
    _fill_preview_shader_get,
    _fill_preview_tag_redraw,
    _fill_preview_terminal_reset,
    _fill_preview_terminal_result,
    _fill_preview_terminal_store,
    _fill_preview_update_prepare_progress,
    _fill_preview_valid,
    _fill_preview_write_vertex_paint,
)

_FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD = FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD
_FILL_PREVIEW_TERMINAL_THRESHOLD_FACTOR = FILL_PREVIEW_TERMINAL_THRESHOLD_FACTOR

class VIEW3D_OT_mesh_focus_local_face_set_grow(bpy.types.Operator):
    """Preview and apply one geometry-aware local Smart Fill region."""

    bl_idname = LOCAL_FACE_SET_GROW_OPERATOR_ID
    bl_label = "Mesh Focus: Smart Fill"
    bl_description = (
        "Fill the connected smooth region under the cursor with the seed Face Set or color; "
        "Shift+E expands with a lightweight ridge/valley crossing cost"
    )
    bl_options = {"REGISTER", "UNDO"}

    mouse_region_x: IntProperty(options={"SKIP_SAVE"})
    mouse_region_y: IntProperty(options={"SKIP_SAVE"})
    strict_mode: BoolProperty(options={"SKIP_SAVE"}, default=False)
    expand_only: BoolProperty(options={"SKIP_SAVE"}, default=False)

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.region is not None
            and context.region.type == "WINDOW"
            and context.space_data is not None
            and context.mode in {"SCULPT", "PAINT_VERTEX"}
            and context.active_object is not None
            and context.active_object.type == "MESH"
        )

    def invoke(self, context, event):
        import numpy as np
        if not self.poll(context):
            return {"PASS_THROUGH"}
        if context.mode == "SCULPT" and _runtime.tube_preview_state is not None and _runtime.tube_preview_state.get("active"):
            self.report({"WARNING"}, "Smart Fill: 先に Tube Shape の予測を終了してください")
            return {"CANCELLED"}

        coord = _sculpt_cursor_region_coordinate(context, event)
        if coord is None:
            self.report({"WARNING"}, "Smart Fill: cursor is outside the View3D window region")
            return {"CANCELLED"}

        self.mouse_region_x = int(round(coord.x))
        self.mouse_region_y = int(round(coord.y))
        self.strict_mode = bool(
            self.strict_mode or getattr(event, "ctrl", False)
        )
        self.expand_only = bool(
            self.expand_only
            or (
                getattr(event, "shift", False)
                and not getattr(event, "ctrl", False)
                and not getattr(event, "alt", False)
            )
        )
        backend = "SCULPT" if context.mode == "SCULPT" else "PAINT_VERTEX"
        color_attribute = None
        color_attribute_reason = None
        seed_color = None
        seed_local_location = None
        if backend == "SCULPT":
            hit = _raycast_sculpt_face_set(context, coord)
            if hit is None:
                self.report({"WARNING"}, "Smart Fill: no visible Face Set face under cursor")
                return {"CANCELLED"}
        else:
            geometry_ok, geometry_reason = _vertex_paint_geometry_compatibility(
                context.active_object
            )
            if not geometry_ok:
                self.report(
                    {"WARNING"},
                    f"Smart Fill: {geometry_reason or 'source/evaluated geometry is incompatible'}",
                )
                return {"CANCELLED"}
            color_attribute, color_attribute_reason = _vertex_paint_active_color_attribute(
                context.active_object
            )
            if color_attribute is None:
                self.report(
                    {"WARNING"},
                    f"Smart Fill: {color_attribute_reason or 'active color attribute is unavailable'}",
                )
                return {"CANCELLED"}
            hit = _raycast_visible_mesh_face(context, coord)
            if hit is None:
                self.report({"WARNING"}, "Smart Fill: no visible mesh face under cursor")
                return {"CANCELLED"}
            hit_obj, hit_face, _hit_world, seed_local_location, _hit_screen = hit
            seed_color = _vertex_paint_sample_color(
                hit_obj, hit_face, seed_local_location, color_attribute
            )
            if seed_color is None:
                self.report({"WARNING"}, "Smart Fill: could not sample the active color")
                return {"CANCELLED"}
        if (
            _runtime.fill_preview_state is not None
            or _runtime.fill_preview_modal_handler_live
        ):
            if _runtime.fill_preview_state is not None:
                _fill_preview_request_cancel(_runtime.fill_preview_state, "replaced")
            self.report(
                {"WARNING"},
                "Smart Fill: previous modal is cancelling; try again after its terminal event",
            )
            return {"CANCELLED"}
        if backend == "SCULPT":
            obj, seed_face, seed_face_set, _location, _screen = hit
        else:
            obj, seed_face, _location, seed_local_location, _screen = hit
            seed_face_set = -1
        try:
            seed_screen = (float(_screen.x), float(_screen.y))
        except (AttributeError, TypeError, ValueError):
            seed_screen = (float(self.mouse_region_x), float(self.mouse_region_y))
        manual_profile_key = _fill_preview_shadow_view_key(context)
        manual_profile = (
            _runtime.shadow_analysis_view_tokens.get(manual_profile_key)
            if manual_profile_key
            else None
        )
        if not isinstance(manual_profile, dict) or not manual_profile.get("active"):
            manual_profile = None
        # Ordinary E is intentionally the fast progressive geometry-range
        # path.  It must not capture the viewport, allocate an ROI, or invoke
        # the screen morphology/sharpening helpers.  A manually-owned
        # Shift+Alt+E view is independent and is left untouched throughout
        # the modal session.  Ctrl+E keeps the same geometry path with its
        # strict boundary policy.
        temporary_profile = None
        shadow_capture = None
        signature = _fill_preview_signature(obj)
        if signature is None:
            return {"CANCELLED"}
        self._preview_id = _next_session_id()
        state = {
            "active": True,
            "phase": "prepare",
            "operator": self,
            "session_id": int(self._preview_id),
            "area": context.area,
            "area_key": int(context.area.as_pointer()),
            "window_key": int(context.window.as_pointer()) if context.window else 0,
            "region": context.region,
            "region_key": int(context.region.as_pointer()),
            "obj": obj,
            "obj_pointer": int(obj.as_pointer()),
            "mesh_pointer": int(obj.data.as_pointer()),
            "signature": signature,
            "mode": str(context.mode),
            "backend": backend,
            "seed_face": int(seed_face),
            "seed_face_set": int(seed_face_set),
            "seed_color": tuple(seed_color) if seed_color is not None else None,
            "seed_local_location": tuple(seed_local_location) if seed_local_location is not None else None,
            "color_attribute_signature": (
                _vertex_paint_color_signature(obj, color_attribute)
                if color_attribute is not None
                else None
            ),
            "color_attribute_name": (
                str(color_attribute.name) if color_attribute is not None else None
            ),
            "color_attribute_domain": (
                str(color_attribute.domain) if color_attribute is not None else None
            ),
            "color_attribute_type": (
                str(color_attribute.data_type) if color_attribute is not None else None
            ),
            "seed_screen": seed_screen,
            # Screen ROI/capture is dormant on the active progressive path.
            "shadow_screen_roi_radius": 0,
            "strict_mode": bool(self.strict_mode),
            "expand_only": bool(self.expand_only and not self.strict_mode),
            "shadow_capture": shadow_capture,
            "shadow_capture_generation": int(time.time_ns()),
            "visible_analysis_profile": None,
            "visible_analysis_profile_manual": bool(
                manual_profile is not None and not bool(self.strict_mode)
            ),
            # Strict mode keeps the compact cursor preparation used by the
            # established path.  Normal E prepares the reusable full graph
            # once, then limits actual work to its small Dijkstra range; this
            # keeps the edge index stable so expansions remain incremental.
            "cursor_prepare": bool(self.strict_mode),
            "cursor_local": bool(self.strict_mode),
            "seed_local": None,
            "start_key": str(getattr(event, "type", "E") or "E"),
            "start_key_released": False,
            "adjacency": None,
            "prepare_job": None,
            "prepare_finalize_job": None,
            "prepare_raw": None,
            "prepare_stage": "queued",
            "prepare_stage_index": 0,
            "prepare_stage_count": 8,
            "prepare_stage_done": 0,
            "prepare_stage_total": 0,
            "prepare_progress_fraction": None,
            "prepare_progress_indeterminate": True,
            "prepare_progress_phase": "prepare",
            "prepare_eta_seconds": None,
            "prepare_stage_started_at": time.perf_counter(),
            "prepare_tick_times": [],
            "distance_state": None,
            "initial_radius": None,
            "desired_radius": None,
            # Kept for compatibility with older in-memory sessions.  Normal
            # E no longer uses screen/luminance stages; wheel state is the
            # physical progressive range below.
            "luminance_step": 0,
            "shadow_luminance_initial_ids": np.empty(0, dtype=np.int32),
            "shadow_luminance_previous_ids": np.empty(0, dtype=np.int32),
            "shadow_luminance_previous_step": 0,
            "shadow_screen_initial_ids": np.empty(0, dtype=np.int32),
            "shadow_screen_previous_ids": np.empty(0, dtype=np.int32),
            "shadow_screen_previous_step": 0,
            "shadow_screen_classifier_cache": None,
            "shadow_screen_buffers": None,
            "shadow_screen_overlay_batches": None,
            "shadow_screen_overlay_cache_key": None,
            "shadow_screen_overlay_alpha": 0.30,
            "cancel_requested": False,
            "processed_radius": None,
            "pending": True,
            "wheel_armed": False,
            "wheel_gate": False,
            "dropped_wheel_events": 0,
            "drawn_generation": None,
            "wheel_drain_until": 0.0,
            # Normal-mode terminal range expansion is enabled only after an
            # exact result at initial_radius*16/(1.25**3).  Strict Ctrl+E
            # never enters this cache.
            "terminal_base": None,
            "terminal_base_radius": None,
            "terminal_base_signature": None,
            "terminal_frontier": None,
            "terminal_expansion_count": 0,
            "result": None,
            "results": {},
            "visibility_cache_checked": False,
            "visibility_cache_invalidated": False,
            "visibility_cache_hidden_count": 0,
            "visibility_cache_reason": "not-checked",
            # Stable mesh polygon ids accepted by the initial Face Set prior;
            # subsequent wheel stages project this cache into their current
            # cursor-local geometry and union it as an immutable floor.
            "initial_accepted_ids": np.empty(0, dtype=np.int32),
            "initial_base_count": 0,
            # Immutable visibility snapshot and the last accepted visible
            # stage.  Radius changes may grow this set, but hidden faces and
            # disconnected visible islands can never enter it.
            "visible_snapshot_ready": False,
            "visible_universe": np.empty(0, dtype=bool),
            "visible_component": np.empty(0, dtype=bool),
            "visible_face_ids": np.empty(0, dtype=np.int32),
            "visible_component_face_ids": np.empty(0, dtype=np.int32),
            "visible_hidden_snapshot": np.empty(0, dtype=bool),
            "visible_topology_fingerprint": None,
            "visible_seed_local": None,
            "accepted_visible_ids": np.empty(0, dtype=np.int32),
            "generation": 0,
            "prepare_seconds": 0.0,
            "last_tick_seconds": 0.0,
            "max_tick_seconds": 0.0,
            "timer": None,
            "last_timer_dispatch": 0.0,
            "metrics": {
                "modal_entries": 0,
                "modal_max_seconds": 0.0,
                "draw_count": 0,
                "draw_max_seconds": 0.0,
                "draw_max_interval": 0.0,
                "cancel_requested_at": None,
                "cancel_finished_at": None,
                "cancel_finish_seconds": None,
                "prepare_finalize_max_seconds": 0.0,
                "cache_adoption_seconds": 0.0,
                "make_result_seconds": 0.0,
                "cleanup_seconds": 0.0,
            },
        }
        if not bool(self.strict_mode):
            # A manual analysis-view owner remains visible by design.  There
            # is no automatic profile in progressive-range mode, so a normal
            # viewport is already restored before the preview becomes ready.
            state["visible_analysis_profile"] = manual_profile
            state["analysis_profile_restored_before_ready"] = False
        _runtime.fill_preview_state = state
        _runtime.fill_preview_modal_operator = self
        _runtime.fill_preview_modal_session = int(self._preview_id)
        _runtime.fill_preview_modal_context = (
            int(state["window_key"]),
            int(state["area_key"]),
            int(state["region_key"]),
            int(state["obj_pointer"]),
            int(state["mesh_pointer"]),
        )
        _runtime.fill_preview_reload_blocked = False
        _runtime.fill_preview_reload_warning = ""
        _runtime.fill_preview_modal_handler_live = False
        _runtime.fill_preview_modal_cancel_requested = False
        _runtime.fill_preview_modal_cancel_reason = ""
        shader = _fill_preview_shader_get()
        if shader is None:
            _fill_preview_cancel(state, "shader")
            return {"CANCELLED"}
        try:
            state["timer"] = context.window_manager.event_timer_add(
                0.01, window=context.window
            )
            _fill_preview_invoke_failure_checkpoint("timer")
            context.window_manager.modal_handler_add(self)
            _fill_preview_invoke_failure_checkpoint("handler")
            _runtime.fill_preview_modal_handler_live = True
            _lifecycle.modal_register(
                self,
                "Smart Fill",
                int(self._preview_id),
                state=state,
            )
            _runtime.fill_preview_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                _fill_preview_draw, (), "WINDOW", "POST_VIEW"
            )
            _fill_preview_invoke_failure_checkpoint("draw")
            _runtime.fill_preview_text_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                _fill_preview_draw_text_handler, (), "WINDOW", "POST_PIXEL"
            )
            _fill_preview_invoke_failure_checkpoint("text-draw")
            _fill_preview_invoke_failure_checkpoint("redraw")
            _fill_preview_tag_redraw(state)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _fill_preview_cancel(state, "start")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            # Invoke failures after ownership registration must retire the
            # exact Smart Fill operator as part of the same terminal path.
            # The helper is idempotent for failures before registration and
            # therefore keeps timer/draw cleanup and registry teardown aligned.
            _fill_preview_modal_owner_terminal(self)
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _process_timer(self, context, state):
        import numpy as np

        if not _fill_preview_valid(state, context):
            _fill_preview_cancel(state, "stale")
            return False
        try:
            if state["phase"] == "prepare":
                started = time.perf_counter()
                if state["prepare_job"] is None and state["prepare_raw"] is None:
                    signature = state["signature"]
                    cached = None if state.get("cursor_prepare") else _runtime.fill_preview_adjacency_cache.get(signature)
                    refreshed = False
                    if cached is not None:
                        if cached.get("geometry_dirty"):
                            refreshed = _fill_preview_refresh_cached_adjacency(
                                state["obj"], cached
                            )
                            if not refreshed:
                                _runtime.fill_preview_adjacency_cache.pop(signature, None)
                                cached = None
                                state["visibility_cache_invalidated"] = True
                                state["visibility_cache_reason"] = "topology-changed"
                    if cached is not None:
                        cache_started = time.perf_counter()
                        if refreshed and cached.pop("visibility_refresh_verified", False):
                            visibility = {
                                "matches": True,
                                "reason": "match",
                                "hidden_count": int(
                                    cached.pop("visibility_refresh_hidden_count", 0)
                                ),
                            }
                        else:
                            visibility = _fill_preview_cached_visibility_check(
                                state["obj"], cached
                            )
                        state["visibility_cache_checked"] = True
                        state["visibility_cache_hidden_count"] = int(
                            visibility.get("hidden_count", 0)
                        )
                        state["visibility_cache_reason"] = str(
                            visibility.get("reason", "unknown")
                        )
                        if not visibility.get("matches"):
                            _runtime.fill_preview_adjacency_cache.pop(signature, None)
                            state["visibility_cache_invalidated"] = True
                            cached = None
                        state["metrics"]["cache_adoption_seconds"] += (
                            time.perf_counter() - cache_started
                        )
                    if cached is not None:
                        state["adjacency"] = cached
                        state["initial_radius"] = _fill_preview_initial_radius(
                            cached, state["seed_face"]
                        )
                        state["desired_radius"] = state["initial_radius"]
                        state["phase"] = "compute"
                        state["pending"] = True
                        state["wheel_armed"] = False
                        state["wheel_gate"] = False
                        state["prepare_stage"] = "cache"
                        return True
                    if state.get("cursor_prepare"):
                        state["prepare_job"] = _fill_preview_cursor_prepare_steps(
                            state["obj"], state["seed_face"]
                        )
                    else:
                        state["prepare_job"] = _fill_preview_adjacency_steps(state["obj"])
                try:
                    prepare_token = next(state["prepare_job"])
                    _fill_preview_update_prepare_progress(state, prepare_token)
                    state["prepare_job_tick"] = time.perf_counter() - started
                    state["prepare_tick_times"].append(
                        {"stage": state["prepare_stage"], "seconds": state["prepare_job_tick"]}
                    )
                except StopIteration as complete:
                    state["prepare_job"] = None
                    if state.get("cursor_prepare"):
                        adjacency = complete.value
                        state["adjacency"] = adjacency
                        state["initial_radius"] = float(adjacency["initial_radius"])
                        state["desired_radius"] = state["initial_radius"]
                        state["processed_radius"] = None
                        state["seed_local"] = int(
                            np.flatnonzero(
                                adjacency["face_ids"] == int(state["seed_face"])
                            )[0]
                        )
                        state["prepare_stage"] = "cursor-ready"
                        state["prepare_progress_indeterminate"] = False
                        state["prepare_progress_fraction"] = 1.0
                        state["prepare_progress_phase"] = "ready"
                        state["phase"] = "compute"
                        state["pending"] = True
                        state["wheel_armed"] = False
                        state["wheel_gate"] = False
                    else:
                        state["prepare_raw"] = complete.value
                        state["prepare_stage"] = "arrays-ready"
                        state["prepare_progress_phase"] = "prepare-finalize"
                        state["prepare_progress_indeterminate"] = True
                        state["prepare_progress_fraction"] = None
                        state["phase"] = "prepare_finalize"
                elapsed = time.perf_counter() - started
                state["last_tick_seconds"] = elapsed
                state["max_tick_seconds"] = max(
                    float(state["max_tick_seconds"]), elapsed
                )
                state["prepare_seconds"] += elapsed
                _fill_preview_tag_redraw(state)
                return True
            if state["phase"] == "prepare_finalize":
                started = time.perf_counter()
                if state["prepare_finalize_job"] is None:
                    state["prepare_stage_count"] = 20
                    state["prepare_stage_index"] = 8
                    state["prepare_stage_started_at"] = time.perf_counter()
                    state["prepare_finalize_job"] = _fill_preview_build_adjacency_cooperative(
                        state["obj"], prepared=state["prepare_raw"]
                    )
                try:
                    prepare_token = next(state["prepare_finalize_job"])
                    _fill_preview_update_prepare_progress(state, prepare_token)
                    state["prepare_finalize_tick"] = time.perf_counter() - started
                    state["metrics"]["prepare_finalize_max_seconds"] = max(
                        float(state["metrics"].get("prepare_finalize_max_seconds", 0.0)),
                        float(state["prepare_finalize_tick"]),
                    )
                except StopIteration as complete:
                    state["prepare_finalize_job"] = None
                    adjacency = complete.value
                    state["adjacency"] = adjacency
                    state["prepare_raw"] = None
                    state["initial_radius"] = _fill_preview_initial_radius(
                        adjacency, state["seed_face"]
                    )
                    state["desired_radius"] = state["initial_radius"]
                    state["prepare_stage"] = "adjacency-ready"
                    state["prepare_progress_indeterminate"] = False
                    state["prepare_progress_fraction"] = 1.0
                    state["prepare_progress_phase"] = "ready"
                elapsed = time.perf_counter() - started
                state["prepare_tick_times"].append(
                    {"stage": state["prepare_stage"], "seconds": elapsed}
                )
                state["last_tick_seconds"] = elapsed
                state["max_tick_seconds"] = max(
                    float(state["max_tick_seconds"]), elapsed
                )
                state["metrics"]["prepare_finalize_max_seconds"] = max(
                    float(state["metrics"].get("prepare_finalize_max_seconds", 0.0)),
                    float(elapsed),
                )
                state["prepare_seconds"] += elapsed
                if state["prepare_finalize_job"] is not None:
                    _fill_preview_tag_redraw(state)
                    return True
                state["phase"] = "compute"
                state["pending"] = True
                state["wheel_armed"] = False
                state["wheel_gate"] = False
                _fill_preview_tag_redraw(state)
                return True
            if state["phase"] == "compute" and state["pending"]:
                radius = float(state["desired_radius"])
                state["result"] = None
                state["generation"] += 1
                # Both normal E and Ctrl+E use the incremental physical-range
                # cache. Normal growth unions only the immediately preceding
                # accepted full-graph rows. A shrink is always recomputed from
                # scratch; it never reuses an additive frontier or a radius-
                # only cache entry from an earlier, larger selection.
                if bool(state.get("expand_only", False)):
                    key = ("expand-only", round(radius, 10))
                else:
                    key = (
                        "progressive-range"
                        if not bool(state.get("strict_mode", False))
                        else "geometry-strict"
                    ), round(radius, 10)
                strict_mode = bool(state.get("strict_mode", False))
                previous_radius = state.get("processed_radius")
                growing = (
                    previous_radius is not None
                    and radius >= float(previous_radius)
                )
                cached_candidate = state["results"].get(key)
                cached = None
                if bool(state.get("expand_only", False)):
                    cached = cached_candidate
                elif strict_mode:
                    # Strict retains its established radius-cache semantics.
                    cached = cached_candidate
                elif growing and normal_cache_hit_allowed(
                    cached_candidate,
                    previous_radius,
                    radius,
                    state.get("accepted_visible_ids", ()),
                ):
                    # A Normal cache hit is valid only if the immutable result
                    # itself contains the immediately preceding accepted floor.
                    # Otherwise a radius-only hit would skip the floor merge and
                    # could make A -> shrink B -> regrow A drop faces from B.
                    cached = cached_candidate
                state["active_result_cache_hit"] = cached is not None
                terminal_expansion = False
                if (
                    not bool(state.get("expand_only", False))
                    and cached is None
                    and not bool(state.get("strict_mode", False))
                    and state.get("terminal_base_radius") is not None
                    and radius > float(
                        (state.get("terminal_frontier") or {}).get(
                            "last_radius",
                            state.get("terminal_base_radius", 0.0),
                        )
                    )
                ):
                    # The terminal frontier is an exact, displayed result
                    # cache.  It is attempted before the full resolver so the
                    # final three monotonic growth stages do not repeat
                    # enclosed-component, wraparound, or shape-refinement
                    # passes.  Any missing/stale provenance returns None and
                    # falls through to the normal safe full computation.
                    cached = _fill_preview_terminal_result(state, radius)
                    if cached is not None and not normal_cache_hit_allowed(
                        cached,
                        previous_radius,
                        radius,
                        state.get("accepted_visible_ids", ()),
                    ):
                        # The frontier is only a valid incremental result when
                        # its complete displayed/confirm candidate contains the
                        # immediately preceding floor.  Discard stale frontier
                        # state rather than repairing faces while leaving its
                        # boundary records or confirm domain inconsistent.
                        _fill_preview_terminal_reset(state)
                        cached = None
                    terminal_expansion = cached is not None
                if cached is None:
                    if (
                        not bool(state.get("expand_only", False))
                        and not bool(state.get("strict_mode", False))
                        and state.get("terminal_base_radius") is not None
                        and radius < float(
                            (state.get("terminal_frontier") or {}).get(
                                "last_radius",
                                state.get("terminal_base_radius", 0.0),
                            )
                        )
                    ):
                        # A shrink without an exact result cannot be derived
                        # from the outward-only frontier.  Reset it before a
                        # full computation; an exact result cache remains
                        # reusable through the ordinary branch.
                        _fill_preview_terminal_reset(state)
                    make_result_started = time.perf_counter()
                    cached = (
                        _fill_preview_make_expand_only_result(state, radius)
                        if bool(state.get("expand_only", False))
                        else _fill_preview_make_result(state, radius)
                    )
                    state["metrics"]["make_result_seconds"] = max(
                        float(state["metrics"].get("make_result_seconds", 0.0)),
                        time.perf_counter() - make_result_started,
                    )
                    terminal_count_eligible = False
                    try:
                        terminal_count_eligible = (
                            len(np.asarray(cached.get("faces", ())).reshape(-1))
                            >= _FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD
                        )
                    except (AttributeError, TypeError, ValueError):
                        terminal_count_eligible = False
                    if (
                        not bool(state.get("expand_only", False))
                        and not bool(state.get("strict_mode", False))
                        and (
                            float(radius)
                            >= float(state.get("initial_radius") or radius)
                            * _FILL_PREVIEW_TERMINAL_THRESHOLD_FACTOR
                            or terminal_count_eligible
                        )
                    ):
                        _fill_preview_terminal_store(state, cached, radius)
                # Terminal frontier expansion also enters through the cache
                # publish branch above.  Apply the same bounded-cache policy to
                # every newly published radius, not only full computations.
                publish_bounded_result(state["results"], key, cached, limit=3)
                # An exact result may have come from the short-lived result
                # cache rather than this dispatch's full resolver.  It is
                # still a valid count-triggered terminal base for the next
                # monotonic wheel growth, even before the radius threshold.
                if (
                    cached is not None
                    and not bool(state.get("expand_only", False))
                    and not bool(state.get("strict_mode", False))
                    and state.get("terminal_base_radius") is None
                ):
                    try:
                        cached_count = len(np.asarray(cached.get("faces", ())).reshape(-1))
                    except (AttributeError, TypeError, ValueError):
                        cached_count = 0
                    if cached_count >= _FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD:
                        _fill_preview_terminal_store(state, cached, radius)
                # Cache bookkeeping is per dispatch, not part of the
                # immutable geometry result.  Publish a shallow copy so the
                # diagnostics can distinguish a reused wheel stage without
                # changing the cached candidate itself.
                cached = dict(cached)
                cached["progressive_range_cache_hit"] = bool(
                    state.get("active_result_cache_hit", False)
                )
                cached["progressive_range_terminal_expansion"] = bool(
                    terminal_expansion
                )
                if int(cached.get("created_generation", -1)) != int(state["generation"]):
                    cached["created_generation"] = int(state["generation"])
                state["last_tick_seconds"] = float(cached.get("compute_seconds", 0.0))
                state["max_tick_seconds"] = max(
                    float(state["max_tick_seconds"]),
                    float(state["last_tick_seconds"]),
                )
                state["result"] = cached
                # Keep the exact visible result that was actually accepted at
                # this stage in the geometry-local index space used by the
                # immutable visible-component snapshot.  The public result
                # stores mesh face ids, so map them explicitly before the next
                # increasing wheel stage; never index a local boolean mask with
                # a mesh-global id.
                try:
                    # ``geometry`` is the compact analysis patch and its
                    # ``_global_face_ids`` are analysis-local provenance. The
                    # Normal floor contract needs full prepared-graph rows,
                    # so map against the immutable confirm snapshot instead.
                    accepted_geometry = cached.get("confirm_geometry")
                    if not isinstance(accepted_geometry, dict):
                        raise RuntimeError(
                            "Smart Fill accepted result has no confirm mapping"
                        )
                    accepted_face_ids = accepted_geometry.get("face_ids")
                    if accepted_face_ids is None:
                        raise RuntimeError(
                            "Smart Fill confirm mapping has no explicit global face ids"
                        )
                    accepted_face_ids = np.asarray(
                        accepted_face_ids,
                        dtype=np.int32,
                    ).reshape(-1)
                    expected_count = int(
                        accepted_geometry.get("count", len(accepted_face_ids))
                    )
                    if len(accepted_face_ids) != expected_count:
                        raise RuntimeError(
                            "Smart Fill accepted result global/local mapping length mismatch"
                        )
                    state["accepted_visible_ids"] = accepted_graph_rows_from_result(
                        cached
                    )
                except (AttributeError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        "Smart Fill accepted result global/local mapping is invalid"
                    ) from error
                state["processed_radius"] = radius
                state["pending"] = False
                state["phase"] = "ready"
                state["wheel_armed"] = False
                state["wheel_gate"] = True
                state["drawn_generation"] = None
                state["wheel_drain_until"] = 0.0
                _fill_preview_tag_redraw(state)
                return True
        except (
            AttributeError,
            IndexError,
            MemoryError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            try:
                message = str(error).strip() or "preview calculation failed"
                self.report({"WARNING"}, f"Smart Fill: {message}")
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            _fill_preview_cancel(state, "compute-error")
            return False
        return True

    def _finish_confirm(self, context, state):
        import numpy as np

        result = state.get("result")
        if state.get("phase") != "ready" or result is None or state.get("pending"):
            return {"RUNNING_MODAL"}
        if (
            not bool(state.get("strict_mode", False))
            and state.get("drawn_generation") != state.get("generation")
        ):
            # Do not apply a result that has not had a draw callback yet.
            return {"RUNNING_MODAL"}
        if not _fill_preview_valid(state, context):
            _fill_preview_cancel(state, "stale")
            return {"CANCELLED"}
        if int(result.get("created_generation", -1)) != int(state["generation"]):
            _fill_preview_cancel(state, "stale-result")
            return {"CANCELLED"}
        obj = state["obj"]
        # Vertex Paint uses the same geometry/preview graph, but its confirm
        # writer is an active color attribute rather than Sculpt's Face Set
        # integer layer.  Keep this branch before the Sculpt-only attribute
        # lookup so PAINT_VERTEX never requires a .sculpt_face_set layer.
        if state.get("backend") == "PAINT_VERTEX":
            try:
                written, reason = _fill_preview_write_vertex_paint(state, result)
            except (
                AttributeError,
                IndexError,
                MemoryError,
                ReferenceError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                written, reason = None, str(error) or "color write failed"
            if written is None:
                _fill_preview_cancel(state, "vertex-color-write")
                self.report({"WARNING"}, f"Smart Fill: {reason or 'color write failed'}")
                return {"CANCELLED"}
            changed, face_count, domain = written
            _fill_preview_cancel(state, "confirm")
            self.report(
                {"INFO"},
                f"Smart Fill: {changed} {str(domain).lower()} color elements changed ({face_count} faces)",
            )
            return {"FINISHED"}
        attr = obj.data.attributes.get(".sculpt_face_set")
        if attr is None or attr.domain != "FACE":
            _fill_preview_cancel(state, "face-set-layer")
            return {"CANCELLED"}
        try:
            # Resolve the generation's immutable candidate snapshot only after
            # the second E/Enter.  The ready result's mesh-global face array is
            # the write source; validation aborts before attr mutation.
            faces = _fill_preview_confirm_flood(state, result)
            if faces is None or len(faces) == 0:
                _fill_preview_cancel(state, "confirm-graph")
                return {"CANCELLED"}
            values = np.empty(len(attr.data), dtype=np.int32)
            attr.data.foreach_get("value", values)
            changed = int(np.count_nonzero(values[faces] != int(state["seed_face_set"])))
            if changed:
                values[faces] = int(state["seed_face_set"])
                # Face Set IDs are intentionally outside the reusable graph.
                # Let the ordinary depsgraph handler mark its volatile layer
                # dirty; no confirm-only exemption or delayed callback drain is
                # needed, and the topology cache object remains in place.
                attr.data.foreach_set("value", values)
                obj.data.update()
                result["self_face_set_cache_preserved"] = bool(
                    state["signature"] in _runtime.fill_preview_adjacency_cache
                    or state["signature"] in _runtime.fill_preview_cursor_cache
                )
                result["self_face_set_update_drain_verified"] = False
        except (
            AttributeError,
            IndexError,
            MemoryError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            _fill_preview_cancel(state, "write-error")
            return {"CANCELLED"}
        candidate_count = int(result.get("candidate_count", len(faces)))
        _fill_preview_cancel(state, "confirm")
        self.report(
            {"INFO"},
            f"Smart Fill: {changed} faces changed ({candidate_count} candidates)",
        )
        return {"FINISHED"}

    def _finish_confirm_terminal(self, context, state):
        """Confirm the ready result and release the modal owner exactly once."""
        result = self._finish_confirm(context, state)
        if result != {"RUNNING_MODAL"}:
            _lifecycle.modal_terminal(
                operator=self,
                status=("FINISHED" if result == {"FINISHED"} else "CANCELLED"),
            )
            _fill_preview_modal_owner_terminal(self)
        return result

    def modal(self, context, event):
        state = _runtime.fill_preview_state
        entry = _lifecycle.modal_entry(operator=self)
        if entry is not None and (
            entry.get("cancel_requested") or entry.get("teardown_requested")
        ):
            if state is not None:
                _fill_preview_cancel(state, entry.get("cancel_reason") or "external-change", terminal=False)
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            _fill_preview_modal_owner_terminal(self)
            return {"CANCELLED"}
        if state is None:
            if entry is not None:
                _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                _fill_preview_modal_owner_terminal(self)
                return {"CANCELLED"}
            if (
                _runtime.fill_preview_modal_handler_live
                and _runtime.fill_preview_modal_operator is self
            ):
                _runtime.fill_preview_modal_operator = None
                _runtime.fill_preview_modal_session = None
                _runtime.fill_preview_modal_context = None
                _runtime.fill_preview_modal_handler_live = False
                _runtime.fill_preview_modal_cancel_requested = False
                _runtime.fill_preview_modal_cancel_reason = ""
            return {"CANCELLED"}
        if state.get("operator") is not self:
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            _fill_preview_modal_owner_terminal(self)
            return {"CANCELLED"}
        event_type = getattr(event, "type", "")
        event_value = getattr(event, "value", None)
        if not _fill_preview_modal_context_matches(state, context):
            _fill_preview_request_cancel(state, "context-changed")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            _fill_preview_modal_owner_terminal(self)
            return {"CANCELLED"}
        # TAB and explicit mode tokens are identifiable before Blender updates
        # context.mode; all other mode/object/area changes are caught above.
        if (
            event_value in {None, "PRESS"}
            and event_type in {"TAB", "SCULPT", "PAINT_VERTEX", "OBJECT", "EDIT_MESH"}
        ):
            _fill_preview_request_cancel(state, "mode-switch-event")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            _fill_preview_modal_owner_terminal(self)
            return {"CANCELLED"}
        if event_type == "ESC" and event_value in {None, "PRESS"}:
            state["cancel_requested"] = True
            _fill_preview_cancel(state, "escape")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            _fill_preview_modal_owner_terminal(self)
            return {"CANCELLED"}
        if event_type == state.get("start_key", "E"):
            if event_value == "RELEASE":
                state["start_key_released"] = True
                return {"RUNNING_MODAL"}
            if (
                event_value == "PRESS"
                and state.get("start_key_released")
                and not bool(getattr(event, "is_repeat", False))
            ):
                return self._finish_confirm_terminal(context, state)
            # The initial press, auto-repeat, and a press held through the
            # preparation phase must never confirm the old/partial result.
            return {"RUNNING_MODAL"}
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            if _fill_preview_green_wheel_up_is_noop(state, event):
                # Green denotes the terminal zoom limit. Preserve this exact
                # visible result instead of clearing it for a futile recompute.
                return {"RUNNING_MODAL"}
            if not bool(state.get("strict_mode", False)):
                # Ordinary E accepts one step only after the current result
                # has been drawn.  Events during compute/draw wait
                # are intentionally discarded rather than queued.
                _fill_preview_accept_normal_wheel(state, event)
                return {"RUNNING_MODAL"}
            # Wheel events cannot carry an input timestamp through Blender's
            # Python modal API.  Strict Ctrl+E retains its historical gate:
            # drop every event until the latest generation has drawn and its
            # short queue-drain interval has elapsed.
            if (
                state.get("initial_radius") is None
                or state.get("phase") != "ready"
                or state.get("pending")
                or not state.get("wheel_armed")
            ):
                state["dropped_wheel_events"] = int(
                    state.get("dropped_wheel_events", 0)
                ) + 1
                if (
                    state.get("wheel_gate")
                    and state.get("drawn_generation") == state.get("generation")
                ):
                    state["wheel_drain_until"] = (
                        time.perf_counter() + _FILL_PREVIEW_WHEEL_DRAIN_SECONDS
                    )
                return {"RUNNING_MODAL"}
            factor = 1.25 if event.type == "WHEELUPMOUSE" else 1.0 / 1.25
            # Normal E expands and shrinks the same physical range as strict
            # mode.  The geometry resolver itself remains less strict for the
            # normal path; no luminance threshold or screen cache is involved.
            base = float(state["desired_radius"] or state["initial_radius"])
            state["desired_radius"] = max(
                float(state["initial_radius"]) * 0.125,
                min(float(state["initial_radius"]) * 16.0, base * factor),
            )
            state["wheel_armed"] = False
            state["wheel_gate"] = False
            state["pending"] = True
            state["phase"] = "compute"
            state["result"] = None
            _fill_preview_tag_redraw(state)
            return {"RUNNING_MODAL"}
        if event.type in {"RET", "NUMPAD_ENTER", "ENTER"}:
            return self._finish_confirm_terminal(context, state)
        if event_type in {"LEFTMOUSE", "RIGHTMOUSE"}:
            # Consume selection/stroke clicks while the preview owns the
            # modal handler.  Neither button confirms nor cancels this tool.
            return {"RUNNING_MODAL"}
        if event.type == "TIMER":
            # Blender 5.2.1 emits Event(type='TIMER', value='NOTHING') without
            # an Event.timer property.  Newer builds may expose the timer and
            # must still be identity-checked.  For the timer-less event, the
            # modal's owning area/window and a short cadence guard provide the
            # available ownership boundary and avoid duplicate foreign ticks.
            event_timer = getattr(event, "timer", None)
            expected_timer = state.get("timer")
            if event_timer is not None and event_timer is not expected_timer:
                return {"RUNNING_MODAL"}
            now = time.perf_counter()
            if (
                event_timer is None
                and now - float(state.get("last_timer_dispatch", 0.0)) < 0.005
            ):
                return {"RUNNING_MODAL"}
            state["last_timer_dispatch"] = now
            if (
                state.get("phase") == "ready"
                and state.get("wheel_gate")
                and state.get("drawn_generation") == state.get("generation")
                and now >= float(state.get("wheel_drain_until", 0.0))
            ):
                # This timer is the first modal boundary after the real draw
                # and the short drain interval; the next wheel is new input.
                state["wheel_gate"] = False
                state["wheel_armed"] = True
            if not self._process_timer(context, state):
                _lifecycle.modal_terminal(operator=self, status="CANCELLED")
                _fill_preview_modal_owner_terminal(self)
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}
        if not _fill_preview_valid(state, context):
            _fill_preview_cancel(state, "stale")
            _lifecycle.modal_terminal(operator=self, status="CANCELLED")
            _fill_preview_modal_owner_terminal(self)
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def execute(self, context):
        # Blender can call execute from a scripted invocation. Route it through
        # the same non-destructive modal entry point using the current cursor.
        if not self.poll(context):
            return {"CANCELLED"}
        return self.invoke(
            context,
            type(
                "_PreviewEvent",
                (),
                {
                    "mouse_region_x": self.mouse_region_x,
                    "mouse_region_y": self.mouse_region_y,
                    "ctrl": bool(self.strict_mode),
                },
            )(),
        )


# Smart Fill responsiveness diagnostics are deliberately installed at the function
# boundary.  This keeps the hot-path selection semantics untouched while
# measuring the full callback, including GPU batch construction and cleanup.
_sfsf_draw_uninstrumented = _fill_preview_draw
def _sfsf_draw_instrumented():
    state = _runtime.fill_preview_state
    started = time.perf_counter()
    previous = float(state.get("last_actual_draw_at", 0.0)) if state else 0.0
    try:
        return _sfsf_draw_uninstrumented()
    finally:
        current = state if state is not None else _runtime.fill_preview_state
        if current is not None:
            now = time.perf_counter()
            metrics = current.setdefault("metrics", {})
            metrics["draw_count"] = int(metrics.get("draw_count", 0)) + 1
            metrics["draw_max_seconds"] = max(
                float(metrics.get("draw_max_seconds", 0.0)), now - started
            )
            if previous > 0.0:
                metrics["draw_max_interval"] = max(
                    float(metrics.get("draw_max_interval", 0.0)), now - previous
                )
            current["last_actual_draw_at"] = now

_fill_preview_draw = _sfsf_draw_instrumented


def _open_boundary_loop_from_seed(seed_edge, *, max_edges=75_000, time_limit=0.35):
    """Select the seed's simple cycle in the visible-face boundary graph.

    Hidden faces remain in BMesh. Count visible incident faces, then split
    boundaries at articulation vertices so point-touching openings stay
    separate. A block containing multiple possible cycles remains ambiguous.
    """
    def is_boundary(edge):
        faces = getattr(edge, "link_faces", ())
        return (
            not bool(getattr(edge, "hide", False))
            and len(faces) in (1, 2)
            and sum(not bool(getattr(face, "hide", False)) for face in faces) == 1
        )

    if seed_edge is None or not is_boundary(seed_edge):
        return None, "Select one visible opening edge (beside a hidden or missing face)"

    started = time.perf_counter()
    boundary_edges = set()
    adjacency = {}
    pending = [seed_edge]
    while pending:
        if time.perf_counter() - started > time_limit:
            return None, "Opening boundary search exceeded its time limit"
        edge = pending.pop()
        if edge in boundary_edges:
            continue
        if len(boundary_edges) >= max_edges:
            return None, "Opening boundary search exceeded its edge limit"
        boundary_edges.add(edge)
        for vert in edge.verts:
            if vert in adjacency:
                continue
            connected = [candidate for candidate in vert.link_edges
                         if is_boundary(candidate)]
            if len(connected) < 2:
                return None, "Boundary is branched or not a closed loop"
            adjacency[vert] = connected
            pending.extend(candidate for candidate in connected
                           if candidate not in boundary_edges)

    # Iterative Tarjan traversal: each popped edge block is biconnected.
    # Unlike a shortest path, this identifies the unique cycle containing the
    # seed and cannot switch to a second hole that only shares one vertex.
    root = seed_edge.verts[0]
    order = {root: 0}
    low = {root: 0}
    parent_edges = {}
    edge_stack = []
    frames = [(root, iter(adjacency[root]))]
    while frames:
        if time.perf_counter() - started > time_limit:
            return None, "Opening boundary search exceeded its time limit"
        vert, neighbors = frames[-1]
        edge = next(neighbors, None)
        if edge is not None:
            if edge is parent_edges.get(vert):
                continue
            other = edge.verts[1] if edge.verts[0] is vert else edge.verts[0]
            if other not in order:
                edge_stack.append(edge)
                parent_edges[other] = edge
                order[other] = low[other] = len(order)
                frames.append((other, iter(adjacency[other])))
            elif order[other] < order[vert]:
                edge_stack.append(edge)
                low[vert] = min(low[vert], order[other])
            continue

        frames.pop()
        parent_edge = parent_edges.get(vert)
        if parent_edge is None:
            continue
        parent = (parent_edge.verts[1]
                  if parent_edge.verts[0] is vert else parent_edge.verts[0])
        low[parent] = min(low[parent], low[vert])
        if low[vert] < order[parent]:
            continue
        block = set()
        while edge_stack:
            member = edge_stack.pop()
            block.add(member)
            if member is parent_edge:
                break
        if seed_edge not in block:
            continue
        if len(block) < 3:
            return None, "Boundary is too short to form a closed loop"
        degree = {}
        for member in block:
            for vertex in member.verts:
                degree[vertex] = degree.get(vertex, 0) + 1
        if any(count != 2 for count in degree.values()):
            return None, "Boundary is branched with multiple possible loops"
        return block, ""

    return None, "No closed opening boundary contains the selected edge"


_OPEN_BOUNDARY_LOOP_REPEAT_TOKEN = None


def _open_boundary_loop_clear_repeat_token(*_args):
    """Forget the process-local arc toggle after state-restoring operations."""
    global _OPEN_BOUNDARY_LOOP_REPEAT_TOKEN
    _OPEN_BOUNDARY_LOOP_REPEAT_TOKEN = None


def _open_boundary_loop_edge_order_key(edge):
    """Return a stable tie-break key for one live BMesh edge."""
    try:
        index = int(edge.index)
    except (AttributeError, ReferenceError, TypeError, ValueError):
        index = -1
    if index >= 0:
        return (0, index)
    try:
        pointer = int(edge.as_pointer())
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pointer = id(edge)
    return (1, pointer)


def _open_boundary_loop_ordered_cycle(loop_edges, first_edge):
    """Order a proven simple cycle without revisiting the boundary graph."""
    if first_edge not in loop_edges or len(loop_edges) < 3:
        return None
    incident = {}
    for edge in loop_edges:
        if len(edge.verts) != 2:
            return None
        for vert in edge.verts:
            incident.setdefault(vert, []).append(edge)
    if any(len(edges) != 2 for edges in incident.values()):
        return None
    start = first_edge.verts[0]
    current = first_edge.verts[1]
    ordered = [first_edge]
    visited_edges = {first_edge}
    previous = first_edge
    while current is not start and len(ordered) <= len(loop_edges):
        next_edges = [edge for edge in incident.get(current, ()) if edge is not previous]
        if len(next_edges) != 1:
            return None
        edge = next_edges[0]
        if edge in visited_edges:
            return None
        ordered.append(edge)
        visited_edges.add(edge)
        current = edge.verts[1] if edge.verts[0] is current else edge.verts[0]
        previous = edge
    if current is not start or len(ordered) != len(loop_edges):
        return None
    if visited_edges != loop_edges:
        return None
    return tuple(ordered)


def _open_boundary_loop_arc_options(loop_edges, anchor_edges, edge_length):
    """Return the shorter and longer inclusive arcs between two rim edges."""
    if len(anchor_edges) != 2:
        return None, None, None, "Select exactly two distinct opening edges"
    first_edge, second_edge = sorted(anchor_edges, key=_open_boundary_loop_edge_order_key)
    if first_edge not in loop_edges or second_edge not in loop_edges:
        return None, None, None, "Selected edges are not on the same unique opening loop"
    ordered = _open_boundary_loop_ordered_cycle(loop_edges, first_edge)
    if ordered is None:
        return None, None, None, "Opening boundary is not one continuous closed loop"
    try:
        second_index = ordered.index(second_edge)
    except ValueError:
        return None, None, None, "Selected edges are not on the same unique opening loop"
    if second_index <= 0:
        return None, None, None, "Select two distinct opening edges"

    first_arc = tuple(ordered[:second_index + 1])
    second_arc = (ordered[0],) + tuple(ordered[second_index:])
    try:
        first_length = sum(float(edge_length(edge)) for edge in first_arc)
        second_length = sum(float(edge_length(edge)) for edge in second_arc)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError, OverflowError):
        return None, None, None, "Could not measure the opening boundary edges"
    if not math.isfinite(first_length) or not math.isfinite(second_length):
        return None, None, None, "Could not measure the opening boundary edges"

    lengths_equal = math.isclose(
        first_length, second_length, rel_tol=1.0e-9, abs_tol=1.0e-12
    )
    if first_length < second_length and not lengths_equal:
        short_arc, long_arc = first_arc, second_arc
    elif second_length < first_length and not lengths_equal:
        short_arc, long_arc = second_arc, first_arc
    else:
        first_key = tuple(sorted(_open_boundary_loop_edge_order_key(edge)
                                 for edge in first_arc))
        second_key = tuple(sorted(_open_boundary_loop_edge_order_key(edge)
                                  for edge in second_arc))
        if first_key <= second_key:
            short_arc, long_arc = first_arc, second_arc
        else:
            short_arc, long_arc = second_arc, first_arc
    return short_arc, long_arc, ordered, ""


def _open_boundary_loop_identity(value):
    try:
        return int(value.as_pointer())
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return id(value)


def _open_boundary_loop_state_signature(bm, obj, ordered_edges):
    """Fingerprint only the chosen ring, its local visibility and its owner."""
    matrix = getattr(obj, "matrix_world", ())
    try:
        matrix_signature = tuple(
            tuple(float(component) for component in row) for row in matrix
        )
    except (TypeError, ValueError):
        matrix_signature = ()
    ring_signature = tuple(
        (
            edge,
            bool(edge.hide),
            tuple(
                (vert, tuple(float(component) for component in getattr(vert, "co", ())))
                for vert in edge.verts
            ),
            tuple(
                (face, bool(face.hide), tuple(getattr(face, "verts", ())))
                for face in edge.link_faces
            ),
        )
        for edge in ordered_edges
    )
    mesh = obj.data
    counts = (len(bm.verts), len(bm.edges), len(bm.faces))
    return (
        _open_boundary_loop_identity(obj),
        _open_boundary_loop_identity(mesh),
        counts,
        matrix_signature,
        ring_signature,
    )


def _open_boundary_loop_repeat_edges(token, bm, obj, selected_edges):
    """Use a repeat token only while its exact output and ring remain live."""
    if not isinstance(token, dict):
        return None
    try:
        if frozenset(selected_edges) != token["output_edges"]:
            return None
        if frozenset(token["anchors"]) - frozenset(token["loop_edges"]):
            return None
        if frozenset(token["alternate_edges"]) - frozenset(token["loop_edges"]):
            return None
        signature = _open_boundary_loop_state_signature(bm, obj, token["loop_edges"])
        if signature != token["state_signature"]:
            return None
        anchors = tuple(sorted(
            token["anchors"], key=_open_boundary_loop_edge_order_key
        ))
        if len(anchors) != 2:
            return None
        current_loop, reason = _open_boundary_loop_from_seed(anchors[0])
        if current_loop is None or frozenset(current_loop) != frozenset(token["loop_edges"]):
            return None
        return frozenset(token["alternate_edges"])
    except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _open_boundary_loop_resolve_selection(bm, obj, selected_edges, token, edge_length):
    """Resolve a fresh one/two-edge input or alternate a valid two-edge token."""
    selected_edges = tuple(selected_edges)
    repeated_edges = _open_boundary_loop_repeat_edges(token, bm, obj, selected_edges)
    if repeated_edges is not None:
        next_token = dict(token)
        next_token["output_edges"] = repeated_edges
        next_token["alternate_edges"] = frozenset(token["output_edges"])
        return repeated_edges, next_token, ""

    if len(selected_edges) == 1:
        loop_edges, reason = _open_boundary_loop_from_seed(selected_edges[0])
        if loop_edges is None:
            return None, None, reason
        return frozenset(loop_edges), None, ""
    if len(selected_edges) != 2 or selected_edges[0] is selected_edges[1]:
        return None, None, "Select one or two visible opening edges"

    anchors = tuple(sorted(selected_edges, key=_open_boundary_loop_edge_order_key))
    loop_edges, reason = _open_boundary_loop_from_seed(anchors[0])
    if loop_edges is None:
        return None, None, reason
    short_arc, long_arc, ordered, reason = _open_boundary_loop_arc_options(
        loop_edges, anchors, edge_length
    )
    if short_arc is None:
        return None, None, reason
    next_token = {
        "anchors": frozenset(anchors),
        "loop_edges": ordered,
        "output_edges": frozenset(short_arc),
        "alternate_edges": frozenset(long_arc),
        "state_signature": _open_boundary_loop_state_signature(bm, obj, ordered),
    }
    return frozenset(short_arc), next_token, ""


@persistent
def _open_boundary_loop_invalidate_repeat_token(*_args):
    _open_boundary_loop_clear_repeat_token()


class MESH_OT_mesh_focus_select_open_boundary_loop(bpy.types.Operator):
    """Select one opening loop or the arc between two selected rim edges."""

    bl_idname = OPEN_BOUNDARY_LOOP_OPERATOR_ID
    bl_label = "Select Opening Boundary Loop"
    bl_description = (
        "Select an opening loop, or the shorter arc between two selected rim edges"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "edit_object", None)
        return (
            context.mode == "EDIT_MESH"
            and obj is not None
            and getattr(obj, "type", None) == "MESH"
        )

    def execute(self, context):
        if not self.poll(context):
            _open_boundary_loop_clear_repeat_token()
            return {"CANCELLED"}
        global _OPEN_BOUNDARY_LOOP_REPEAT_TOKEN
        obj = context.edit_object
        try:
            bm = bmesh.from_edit_mesh(obj.data)
            selected_opening_edges = tuple(
                edge for edge in bm.edges if edge.select and not edge.hide
            )

            def world_edge_length(edge):
                matrix = obj.matrix_world
                first = matrix @ edge.verts[0].co
                second = matrix @ edge.verts[1].co
                return (first - second).length

            result_edges, next_token, reason = _open_boundary_loop_resolve_selection(
                bm,
                obj,
                selected_opening_edges,
                _OPEN_BOUNDARY_LOOP_REPEAT_TOKEN,
                world_edge_length,
            )
            if result_edges is None:
                _open_boundary_loop_clear_repeat_token()
                self.report({"WARNING"}, reason)
                return {"CANCELLED"}

            for face in bm.faces:
                if face.select:
                    face.select_set(False)
            for edge in bm.edges:
                if edge.select:
                    edge.select_set(False)
            for vert in bm.verts:
                if vert.select:
                    vert.select_set(False)
            for edge in result_edges:
                edge.select_set(True)
            bm.select_history.clear()
            anchor = next(
                (edge for edge in selected_opening_edges if edge in result_edges),
                next(iter(result_edges)),
            )
            bm.select_history.add(anchor)
            context.tool_settings.mesh_select_mode = (False, True, False)
            bmesh.update_edit_mesh(
                obj.data,
                loop_triangles=False,
                destructive=False,
            )
            _OPEN_BOUNDARY_LOOP_REPEAT_TOKEN = next_token
            self.report(
                {"INFO"},
                f"Selected opening boundary segment: {len(result_edges)} edges",
            )
            return {"FINISHED"}
        except (
            AttributeError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            _open_boundary_loop_clear_repeat_token()
            self.report({"WARNING"}, f"Opening boundary selection failed: {error}")
            return {"CANCELLED"}


class VIEW3D_OT_mesh_focus_topology_color_assign(bpy.types.Operator):
    """Assign or clear one topology guide color on selected visible faces."""

    bl_idname = TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID
    bl_label = "Mesh Focus: Assign Topology Color"
    bl_description = (
        "Assign the selected visible faces a topology guide color; zero clears it"
    )
    bl_options = {"REGISTER", "UNDO"}

    color_index: IntProperty(
        name="Color",
        description="Stored topology guide color number (0 clears the color)",
        min=0,
        max=6,
        default=0,
        options={"SKIP_SAVE"},
    )

    @classmethod
    def poll(cls, context):
        obj = _topology_color_object(context)
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and obj is not None
        )

    def execute(self, context):
        if not self.poll(context):
            return {"CANCELLED"}
        obj = _topology_color_object(context)
        try:
            bm = bmesh.from_edit_mesh(obj.data)
            selected_faces = [
                face
                for face in bm.faces
                if face.select and not face.hide
            ]
            # A no-selection keypress is intentionally a no-op and does not
            # create the attribute or an unnecessary Undo step.
            if not selected_faces:
                return {"FINISHED"}
            layer = bm.faces.layers.int.get(TOPOLOGY_COLOR_ATTRIBUTE_NAME)
            if layer is None:
                if int(self.color_index) == 0:
                    return {"FINISHED"}
                layer = bm.faces.layers.int.new(TOPOLOGY_COLOR_ATTRIBUTE_NAME)
                # Creating the first custom-data layer can rebuild the
                # BMFace wrappers.  Re-read the selected faces so the write
                # never retains invalid elements across that boundary.
                selected_faces = [
                    face
                    for face in bm.faces
                    if face.select and not face.hide
                ]
            value = int(self.color_index)
            changed = 0
            for face in selected_faces:
                if int(face[layer]) != value:
                    face[layer] = value
                    changed += 1
            if changed:
                bmesh.update_edit_mesh(
                    obj.data,
                    loop_triangles=False,
                    destructive=False,
                )
                obj.data.update_tag()
                _invalidate_topology_color_cache(obj)
            return {"FINISHED"}
        except (
            AttributeError,
            ReferenceError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return {"CANCELLED"}


class VIEW3D_PT_mesh_focus_topology_colors(bpy.types.Panel):
    """N-panel controls for the six stored topology guide colors."""

    bl_idname = "VIEW3D_PT_mesh_focus_topology_colors"
    bl_label = "Topology Colors"
    bl_category = TOPOLOGY_COLOR_PANEL_CATEGORY
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context):
        return _topology_color_object(context) is not None

    def draw(self, context):
        layout = self.layout
        layout.operator(
            OPEN_BOUNDARY_LOOP_OPERATOR_ID,
            text="開口部を一周選択",
            icon="EDGESEL",
        )
        layout.label(text="境界辺を1本選択 → Shift+Alt+L")
        layout.separator()
        prefs = _addon_preferences()
        if prefs is not None:
            layout.prop(prefs, "topology_colors_enabled", text="表示")
            layout.prop(prefs, "topology_color_opacity", text="透明度")
        layout.label(text="選択面へ割り当て")
        color_names = ("赤", "オレンジ", "黄", "緑", "青", "紫")
        for row_start in (1, 4):
            row = layout.row(align=True)
            for color_index in range(row_start, row_start + 3):
                operator = row.operator(
                    TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID,
                    text=color_names[color_index - 1],
                )
                operator.color_index = color_index
        clear_operator = layout.operator(
            TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID,
            text="0 解除",
        )
        clear_operator.color_index = 0


class VIEW3D_PT_mesh_focus_local_feature_brush(bpy.types.Panel):
    """Sculpt N-panel entry and compact controls for Local Feature Brush."""

    bl_idname = "VIEW3D_PT_mesh_focus_local_feature_brush"
    bl_label = "Local Feature Brush"
    bl_category = TOPOLOGY_COLOR_PANEL_CATEGORY
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "active_object", None)
        return (
            getattr(context, "mode", None) == "SCULPT"
            and obj is not None
            and obj.type == "MESH"
        )

    def draw(self, context):
        layout = self.layout
        prefs = _addon_preferences()
        if prefs is not None:
            layout.prop(prefs, "local_feature_strength")
            layout.prop(prefs, "local_feature_scale")
            layout.prop(prefs, "local_feature_radius")
        layout.separator()
        layout.label(text="Select the MFO brush from the Sculpt Asset Shelf")
        layout.label(text="LMB applies; Shift+LMB is native Smooth")


class VIEW3D_PT_mesh_focus_guided_ridge(bpy.types.Panel):
    """Compact Sculpt entry for the Guided Ridge modal."""

    bl_idname = "VIEW3D_PT_mesh_focus_guided_ridge"
    bl_label = "Guided Ridge"
    bl_category = TOPOLOGY_COLOR_PANEL_CATEGORY
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "active_object", None)
        return bool(
            getattr(context, "mode", None) == "SCULPT"
            and obj is not None
            and obj.type == "MESH"
            and (
                _guided_ridge_prototype_route_enabled()
                or _guided_ridge_face_set_attribute(obj) is not None
            )
        )

    def draw(self, context):
        layout = self.layout
        prefs = _addon_preferences()
        if prefs is not None:
            layout.label(text="Curve Sculpt")
            layout.prop(prefs, "guided_ridge_curve_smoothing")
            layout.prop(prefs, "guided_ridge_ridge_strength")
            layout.prop(prefs, "guided_ridge_ridge_radius")
            layout.separator()
        layout.label(text="Guided Ridge Curve Sculpt: Ctrl + G")
        layout.label(text="LMB: 点追加 / Enter: Curve Preview")
        layout.label(text="Wheel: Curve Shape -100..100 (Attenuated/Raw/Amplified)")
        layout.label(text="Backspace: 編集へ戻る")
        layout.label(text="Enter: Pinch Ridge / Ctrl+Enter: Crease Polish Valley")


class VIEW3D_PT_mesh_focus_orbit_tools(bpy.types.Panel):
    """Daily MFO settings kept beside the resident T-toolbar tools."""

    bl_idname = "VIEW3D_PT_mesh_focus_orbit_tools"
    bl_label = "Mesh Focus Orbit"
    bl_category = TOPOLOGY_COLOR_PANEL_CATEGORY
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.mode in {"OBJECT", "EDIT_MESH", "SCULPT"}
        )

    def draw(self, context):
        layout = self.layout
        scene = getattr(context, "scene", None)
        prefs = _addon_preferences()
        if scene is not None and hasattr(scene, "mfo_reference_object"):
            layout.prop(scene, "mfo_reference_object", text="Reference Object")
        if prefs is not None:
            layout.prop(prefs, "activation_key", text="Activation Key")
            layout.prop(prefs, "double_tap_window", text="Double-tap Window")
            layout.prop(prefs, "show_indicator", text="Show Indicator")
            layout.prop(
                prefs,
                "retopoflow_target_island_filter",
                text="RetopoFlow Island Filter",
            )
        try:
            state = _runtime.active_states.get(int(context.area.as_pointer()))
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            state = None
        if state is not None and getattr(state, "active", False):
            label = "Face Set MFO: ON" if isinstance(state, _FaceSetOrbitState) else "MFO: ON"
            layout.label(text=label, icon="CHECKMARK")
        else:
            layout.label(text="MFO: OFF", icon="X")
        layout.separator()
        layout.label(text="Tツール: クリックで対象面を指定")
        if context.mode == "EDIT_MESH":
            previous_operator_context = layout.operator_context
            try:
                layout.operator_context = "INVOKE_REGION_WIN"
                remesh_operator = layout.operator(
                    "mesh.mesh_focus_local_remesh",
                    text="Local Remesh",
                )
                remesh_operator.target_edge_length = 0.0
            finally:
                layout.operator_context = previous_operator_context
        if context.mode in {"OBJECT", "EDIT_MESH"}:
            layout.label(text="Ctrl + クリック: Face Set (厳密)")
        elif context.mode == "SCULPT":
            layout.label(text="Sculpt: Smart Fill / Guided Ridge は専用ツール")


_MFO_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_MFO_TOOL_ICON_DIR = next(
    (
        candidate
        for candidate in (
            os.path.join(_MFO_PACKAGE_DIR, "assets", "mfo-toolbar-icons-astra"),
            os.path.join(os.path.dirname(_MFO_PACKAGE_DIR), "assets", "mfo-toolbar-icons-astra"),
        )
        if os.path.isdir(candidate)
    ),
    os.path.join(_MFO_PACKAGE_DIR, "assets", "mfo-toolbar-icons-astra"),
)


def _mfo_toolbar_icon(stem, fallback):
    """Return an installed VCO stem, or a shipped Blender icon fallback."""
    custom_stem = os.path.join(_MFO_TOOL_ICON_DIR, str(stem))
    return custom_stem if os.path.isfile(custom_stem + ".dat") else fallback


_MFO_ICON_FOCUS = _mfo_toolbar_icon("mfo-focus-surface", "ops.generic.cursor")
_MFO_ICON_FACE_SET = _mfo_toolbar_icon("mfo-face-set", "ops.sculpt.border_face_set")
_MFO_ICON_SMART_FILL = _mfo_toolbar_icon(
    "mfo-smart-face-set-fill", "ops.sculpt.border_face_set"
)
_MFO_ICON_GUIDED_RIDGE = _mfo_toolbar_icon(
    "mfo-guided-ridge", "ops.sculpt.border_mask"
)
_MFO_ICON_TUBE = _mfo_toolbar_icon("mfo-tube-shape", "ops.sculpt.box_trim")
_MFO_ICON_BACKFACE_SELECT = _mfo_toolbar_icon(
    "mfo-select-back-faces", "ops.generic.select_box"
)


class VIEW3D_WST_mesh_focus_normal_object(bpy.types.WorkSpaceTool):
    bl_idname = "mfo.normal_object"
    bl_label = "MFO: Focus Surface"
    bl_description = "Click a mesh surface to set the temporary orbit target"
    bl_space_type = "VIEW_3D"
    bl_context_mode = "OBJECT"
    bl_icon = _MFO_ICON_FOCUS
    bl_keymap = (
        (TOOL_NORMAL_OPERATOR_ID, {"type": "LEFTMOUSE", "value": "PRESS"}, {}),
        (TOOL_FACE_SET_OPERATOR_ID, {"type": "LEFTMOUSE", "value": "PRESS", "ctrl": True}, {}),
    )


class VIEW3D_WST_mesh_focus_normal_edit(bpy.types.WorkSpaceTool):
    bl_idname = "mfo.normal_edit"
    bl_label = "MFO: Focus Surface"
    bl_description = "Click an edit-mode mesh surface to set the orbit target"
    bl_space_type = "VIEW_3D"
    bl_context_mode = "EDIT_MESH"
    bl_icon = _MFO_ICON_FOCUS
    bl_keymap = VIEW3D_WST_mesh_focus_normal_object.bl_keymap


class VIEW3D_WST_mesh_focus_normal_sculpt(bpy.types.WorkSpaceTool):
    bl_idname = "mfo.normal_sculpt"
    bl_label = "MFO: Focus Surface"
    bl_description = "Click a Sculpt mesh surface to set the temporary orbit target"
    bl_space_type = "VIEW_3D"
    bl_context_mode = "SCULPT"
    bl_icon = _MFO_ICON_FOCUS
    bl_keymap = ((TOOL_NORMAL_OPERATOR_ID, {"type": "LEFTMOUSE", "value": "PRESS"}, {}),)


class VIEW3D_WST_mesh_focus_face_set_object(bpy.types.WorkSpaceTool):
    bl_idname = "mfo.face_set_object"
    bl_label = "MFO: Face Set Focus"
    bl_description = "Click a configured Reference Object Face Set"
    bl_space_type = "VIEW_3D"
    bl_context_mode = "OBJECT"
    bl_icon = _MFO_ICON_FACE_SET
    bl_keymap = ((TOOL_FACE_SET_OPERATOR_ID, {"type": "LEFTMOUSE", "value": "PRESS"}, {}),)


class VIEW3D_WST_mesh_focus_face_set_edit(bpy.types.WorkSpaceTool):
    bl_idname = "mfo.face_set_edit"
    bl_label = "MFO: Face Set Focus"
    bl_description = "Click a configured Reference Object Face Set"
    bl_space_type = "VIEW_3D"
    bl_context_mode = "EDIT_MESH"
    bl_icon = _MFO_ICON_FACE_SET
    bl_keymap = VIEW3D_WST_mesh_focus_face_set_object.bl_keymap


class VIEW3D_WST_mesh_focus_smart_fill_sculpt(bpy.types.WorkSpaceTool):
    bl_idname = "mfo.smart_fill_sculpt"
    bl_label = "MFO: Smart Fill"
    bl_description = "Click a Sculpt surface to preview Smart Fill"
    bl_space_type = "VIEW_3D"
    bl_context_mode = "SCULPT"
    bl_icon = _MFO_ICON_SMART_FILL
    bl_keymap = (
        (LOCAL_FACE_SET_GROW_OPERATOR_ID, {"type": "LEFTMOUSE", "value": "PRESS"}, {}),
        (LOCAL_FACE_SET_GROW_OPERATOR_ID, {"type": "LEFTMOUSE", "value": "PRESS", "ctrl": True}, {}),
    )


class VIEW3D_WST_mesh_focus_smart_fill_vertex(bpy.types.WorkSpaceTool):
    bl_idname = "mfo.smart_fill_vertex"
    bl_label = "MFO: Smart Fill"
    bl_description = "Click a Vertex Paint surface to preview Smart Fill"
    bl_space_type = "VIEW_3D"
    bl_context_mode = "PAINT_VERTEX"
    bl_icon = _MFO_ICON_SMART_FILL
    bl_keymap = VIEW3D_WST_mesh_focus_smart_fill_sculpt.bl_keymap


class VIEW3D_WST_mesh_focus_guided_ridge_sculpt(bpy.types.WorkSpaceTool):
    bl_idname = "mfo.guided_ridge_sculpt"
    bl_label = "MFO: Guided Ridge"
    bl_description = "Click a Sculpt Face Set to start a Guided Ridge guide"
    bl_space_type = "VIEW_3D"
    bl_context_mode = "SCULPT"
    bl_icon = _MFO_ICON_GUIDED_RIDGE
    bl_keymap = ((GUIDED_RIDGE_OPERATOR_ID, {"type": "LEFTMOUSE", "value": "PRESS"}, {}),)


class VIEW3D_WST_mesh_focus_tube_sculpt(bpy.types.WorkSpaceTool):
    bl_idname = "mfo.tube_shape_sculpt"
    bl_label = "MFO: Tube Shape"
    bl_description = "Click a Sculpt Face Set tube to start Tube Shape"
    bl_space_type = "VIEW_3D"
    bl_context_mode = "SCULPT"
    bl_icon = _MFO_ICON_TUBE
    bl_keymap = ((TUBE_SHAPE_OPERATOR_ID, {"type": "LEFTMOUSE", "value": "PRESS"}, {}),)


class VIEW3D_WST_mesh_focus_backface_select_edit(bpy.types.WorkSpaceTool):
    bl_idname = "mfo.backface_select_edit"
    bl_label = "MFO: Select Back Faces"
    bl_description = "Click a damaged surface or nearby rim to select local back faces"
    bl_space_type = "VIEW_3D"
    bl_context_mode = "EDIT_MESH"
    bl_icon = _MFO_ICON_BACKFACE_SELECT
    bl_keymap = ((EXPOSED_BACKFACE_SELECT_OPERATOR_ID, {"type": "LEFTMOUSE", "value": "PRESS"}, {}),)


_TOOL_CLASSES = (
    VIEW3D_WST_mesh_focus_normal_object,
    VIEW3D_WST_mesh_focus_normal_edit,
    VIEW3D_WST_mesh_focus_normal_sculpt,
    VIEW3D_WST_mesh_focus_face_set_object,
    VIEW3D_WST_mesh_focus_face_set_edit,
    VIEW3D_WST_mesh_focus_smart_fill_sculpt,
    VIEW3D_WST_mesh_focus_smart_fill_vertex,
    VIEW3D_WST_mesh_focus_guided_ridge_sculpt,
    VIEW3D_WST_mesh_focus_tube_sculpt,
    VIEW3D_WST_mesh_focus_backface_select_edit,
)
_MFO_TOOL_CONTEXT_LABELS = {
    "OBJECT": "Object",
    "EDIT_MESH": "Edit Mesh",
    "SCULPT": "Sculpt",
    "PAINT_VERTEX": "Paint Vertex",
}
_MFO_TOOL_KEYMAP_NAMES = frozenset(
    "3D View Tool: {mode}, {label}".format(
        mode=_MFO_TOOL_CONTEXT_LABELS.get(
            str(tool_cls.bl_context_mode),
            str(tool_cls.bl_context_mode),
        ),
        label=str(tool_cls.bl_label),
    )
    for tool_cls in _TOOL_CLASSES
) | {"3D View Tool: Sculpt, MFO: Select Back Faces"}


def _remove_stale_tool_keymaps():
    """Remove MFO tool keymaps left by a failed/reloaded registration.

    Blender's ``unregister_tool`` can only remove keymaps through the exact
    old Python class object.  After ``importlib.reload`` that object is gone,
    so empty or partial ``3D View Tool: ..., MFO: ...`` keymaps can survive.
    They are owned exclusively by this add-on and are safe to rebuild before
    registering the current tool classes.  Match only the exact names Blender
    generates from the current mode and tool labels, plus the exact retired
    Sculpt binding for Select Back Faces.
    """
    try:
        keyconfigs = bpy.context.window_manager.keyconfigs
        # The user's keyconfig may contain hand-customized generated tool
        # bindings.  Never remove it during registration/reload; only clean
        # the add-on/default owners that MFO can safely regenerate.
        for keyconfig_name in ("default", "addon"):
            keyconfig = getattr(keyconfigs, keyconfig_name, None)
            if keyconfig is None:
                continue
            for keymap in list(keyconfig.keymaps):
                keymap_name = str(getattr(keymap, "name", ""))
                if keymap_name in _MFO_TOOL_KEYMAP_NAMES:
                    keyconfig.keymaps.remove(keymap)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _register_tools():
    registered = []
    _remove_stale_tool_keymaps()
    for tool_cls in _TOOL_CLASSES:
        try:
            bpy.utils.register_tool(tool_cls)
            registered.append(tool_cls)
        except Exception as error:
            # importlib.reload can replace the Python class object while
            # Blender still owns the old WorkSpaceTool registration.  The
            # existing id is already the one visible in the toolbar; keeping
            # it avoids a duplicate and lets addon_utils disable/enable
            # finish without treating a stale class identity as fatal.
            if "already exists" in str(error).lower():
                continue
            for registered_cls in reversed(registered):
                try:
                    bpy.utils.unregister_tool(registered_cls)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            raise


def _unregister_tools():
    for tool_cls in reversed(_TOOL_CLASSES):
        try:
            bpy.utils.unregister_tool(tool_cls)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


def _remove_keymaps():
    for keymap, keymap_item in _runtime.addon_keymaps:
        try:
            keymap.keymap_items.remove(keymap_item)
        except (ReferenceError, RuntimeError, ValueError):
            pass
    _runtime.addon_keymaps.clear()

    # A module reload replaces this Python list, while Blender keeps the old
    # Addon KeyMapItems.  Remove every stale item owned by this add-on so one
    # physical tap cannot invoke the operator twice and look like a double tap.
    try:
        keyconfig = bpy.context.window_manager.keyconfigs.addon
        if keyconfig is None:
            return
        owned_operator_ids = {
            OPERATOR_ID,
            FACE_SET_ACTIVATION_OPERATOR_ID,
            LOCAL_FACE_SET_GROW_OPERATOR_ID,
            TUBE_SHAPE_OPERATOR_ID,
            LOCAL_FEATURE_BRUSH_OPERATOR_ID,
            GUIDED_RIDGE_OPERATOR_ID,
            TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID,
            OPEN_BOUNDARY_LOOP_OPERATOR_ID,
            "view3d.mesh_focus_shadow_analysis_toggle",
            "view3d.mesh_focus_display_distance_toggle",
        }
        for keymap in keyconfig.keymaps:
            # WorkSpaceTool keymaps are owned by ``register_tool`` and may
            # intentionally reuse the same operator IDs as legacy hotkeys.
            # Never remove their click bindings during the regular keymap
            # rebuild; ``_unregister_tools`` handles those as a unit.
            if str(getattr(keymap, "name", "")).startswith("3D View Tool:"):
                continue
            for keymap_item in list(keymap.keymap_items):
                if keymap_item.idname not in owned_operator_ids:
                    continue
                try:
                    keymap.keymap_items.remove(keymap_item)
                except (ReferenceError, RuntimeError, ValueError):
                    pass
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _rebuild_keymaps():
    if not _runtime.is_registered:
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
        # Keep normal MFO and Face Set MFO in separate keymap items.  The
        # normal item must not use ``any=True``: otherwise Ctrl+activation_key
        # also reaches this operator and competes with the direct FSMFO item.
        keymap_item = keymap.keymap_items.new(
            OPERATOR_ID,
            activation_key,
            "PRESS",
            any=False,
            ctrl=False,
        )
        _runtime.addon_keymaps.append((keymap, keymap_item))

        face_set_item = keymap.keymap_items.new(
            FACE_SET_ACTIVATION_OPERATOR_ID,
            activation_key,
            "PRESS",
            any=False,
            ctrl=True,
        )
        _runtime.addon_keymaps.append((keymap, face_set_item))

        local_grow_item = keymap.keymap_items.new(
            LOCAL_FACE_SET_GROW_OPERATOR_ID,
            LOCAL_FACE_SET_GROW_KEY,
            "PRESS",
        )
        _runtime.addon_keymaps.append((keymap, local_grow_item))

        expand_only_item = keymap.keymap_items.new(
            LOCAL_FACE_SET_GROW_OPERATOR_ID,
            LOCAL_FACE_SET_GROW_KEY,
            "PRESS",
            any=False,
            shift=True,
            ctrl=False,
            alt=False,
        )
        expand_only_item.properties.expand_only = True
        _runtime.addon_keymaps.append((keymap, expand_only_item))

        strict_grow_item = keymap.keymap_items.new(
            LOCAL_FACE_SET_GROW_OPERATOR_ID,
            LOCAL_FACE_SET_GROW_KEY,
            "PRESS",
            ctrl=True,
        )
        strict_grow_item.properties.strict_mode = True
        _runtime.addon_keymaps.append((keymap, strict_grow_item))

        shadow_analysis_item = keymap.keymap_items.new(
            "view3d.mesh_focus_shadow_analysis_toggle",
            "E",
            "PRESS",
            any=False,
            alt=True,
            ctrl=False,
            shift=True,
        )
        _runtime.addon_keymaps.append((keymap, shadow_analysis_item))

        display_distance_item = keymap.keymap_items.new(
            "view3d.mesh_focus_display_distance_toggle",
            "V",
            "PRESS",
            any=False,
            alt=True,
            ctrl=False,
            shift=True,
        )
        _runtime.addon_keymaps.append((keymap, display_distance_item))

        mesh_keymap = keyconfig.keymaps.new(
            name="Mesh",
            space_type="EMPTY",
            region_type="WINDOW",
        )
        open_boundary_item = mesh_keymap.keymap_items.new(
            OPEN_BOUNDARY_LOOP_OPERATOR_ID,
            "L",
            "PRESS",
            any=False,
            shift=True,
            ctrl=False,
            alt=True,
        )
        _runtime.addon_keymaps.append((mesh_keymap, open_boundary_item))

        tube_shape_item = keymap.keymap_items.new(
            TUBE_SHAPE_OPERATOR_ID,
            TUBE_SHAPE_KEY,
            "PRESS",
            any=False,
            ctrl=True,
            alt=True,
        )
        _runtime.addon_keymaps.append((keymap, tube_shape_item))

        # The local feature brush is a normal Brush Asset selection.  Its
        # dispatcher lives in the Sculpt keymap and is poll-gated by the
        # stable asset marker, so ordinary assets and Shift Smooth fall back
        # to Blender's native items without a resident modal selector.
        sculpt_keymap = keyconfig.keymaps.new(
            name="Sculpt",
            space_type="EMPTY",
            region_type="WINDOW",
        )
        local_feature_item = sculpt_keymap.keymap_items.new(
            LOCAL_FEATURE_BRUSH_OPERATOR_ID,
            "LEFTMOUSE",
            "PRESS",
            any=False,
            shift=False,
            ctrl=False,
            alt=False,
            head=True,
        )
        local_feature_item.active = True
        _runtime.addon_keymaps.append((sculpt_keymap, local_feature_item))

        guided_ridge_item = sculpt_keymap.keymap_items.new(
            GUIDED_RIDGE_OPERATOR_ID,
            GUIDED_RIDGE_KEY,
            "PRESS",
            any=False,
            shift=False,
            ctrl=True,
            alt=False,
            head=True,
        )
        guided_ridge_item.active = True
        _runtime.addon_keymaps.append((sculpt_keymap, guided_ridge_item))

        for color_index, key in enumerate(
            ("ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "ZERO"),
            start=1,
        ):
            color_item = keymap.keymap_items.new(
                TOPOLOGY_COLOR_ASSIGN_OPERATOR_ID,
                key,
                "PRESS",
                any=False,
                ctrl=True,
                alt=True,
            )
            color_item.properties.color_index = 0 if key == "ZERO" else color_index
            _runtime.addon_keymaps.append((keymap, color_item))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        _remove_keymaps()


def _preferences_changed(_self, _context):
    if _runtime.is_registered:
        _rebuild_keymaps()
        _invalidate_topology_color_cache()


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
    retopoflow_target_island_filter: BoolProperty(
        name="RetopoFlow Focus-Island Snap/Weld Filter",
        description=(
            "During Face Set MFO, restrict PolyPen and Auto Merge targets to "
            "the recorded target Retopo island"
        ),
        default=False,
    )
    backface_auto_relax: BoolProperty(
        name="Auto Relax with LoopTools",
        description=(
            "Run LoopTools Relax on the selected faces after Select Back Faces; "
            "requires the optional LoopTools extension"
        ),
        default=False,
    )
    topology_colors_enabled: BoolProperty(
        name="Topology Colors",
        description="Show the stored six-color topology guide overlay",
        default=True,
        update=_preferences_changed,
    )
    topology_color_opacity: FloatProperty(
        name="Topology Color Opacity",
        description="Opacity of the topology guide face overlay",
        default=0.35,
        min=0.05,
        max=1.0,
        precision=2,
        subtype="FACTOR",
        update=_preferences_changed,
    )
    local_feature_strength: FloatProperty(
        name="Local Feature Strength",
        description="Maximum strength of the existing ridge/valley enhancement dab",
        default=0.35,
        min=0.0,
        max=1.0,
        precision=3,
        subtype="FACTOR",
    )
    local_feature_scale: FloatProperty(
        name="Target Feature Scale",
        description="Low-pass scale for broad hair-bundle features",
        default=0.35,
        min=0.10,
        max=1.0,
        precision=2,
        subtype="FACTOR",
    )
    local_feature_radius: FloatProperty(
        name="Local Feature Radius",
        description="Radius multiplier relative to the selected Sculpt brush",
        default=1.0,
        min=0.10,
        max=4.0,
        precision=2,
    )
    guided_ridge_curve_smoothing: FloatProperty(
        name="Guided Ridge Curve Shape",
        description="Remembered Curve Shape used for the Guided Ridge preview (-100..100)",
        default=-12.0,
        min=-100.0,
        max=100.0,
        precision=0,
    )
    guided_ridge_ridge_strength: FloatProperty(
        name="Guided Ridge Strength",
        description="Strength of the dedicated Pinch/Crease Curve Sculpt brush",
        default=0.35,
        min=0.0,
        max=1.0,
        precision=2,
        subtype="FACTOR",
    )
    guided_ridge_ridge_radius: FloatProperty(
        name="Guided Ridge Radius",
        description="Screen-space radius of the dedicated Curve Sculpt brush",
        default=40.0,
        min=1.0,
        max=500.0,
        precision=0,
        subtype="PIXEL",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "enabled")
        layout.prop(self, "activation_key")
        layout.prop(context.scene, "mfo_reference_object", text="Reference Object")
        layout.prop(self, "focus_loss_behavior")
        layout.prop(self, "debug_display")
        layout.prop(self, "show_indicator")
        layout.prop(self, "retopoflow_target_island_filter")
        layout.prop(self, "backface_auto_relax")
        layout.prop(self, "topology_colors_enabled")
        layout.prop(self, "topology_color_opacity")
        layout.separator()
        layout.label(text="Local Feature Brush (Sculpt only)")
        layout.prop(self, "local_feature_strength")
        layout.prop(self, "local_feature_scale")
        layout.prop(self, "local_feature_radius")
        layout.separator()
        layout.label(text="Guided Ridge Curve Sculpt")
        layout.prop(self, "guided_ridge_curve_smoothing")
        layout.prop(self, "guided_ridge_ridge_strength")
        layout.prop(self, "guided_ridge_ridge_radius")
        layout.prop(self, "double_tap_window")
        layout.separator()
        layout.label(text="Double-tap the activation key in a 3D Viewport.")
        layout.label(text="Ctrl + double-tap uses Face Set MFO on the Reference Object.")
        layout.label(text="The viewport center is ray-cast once on activation.")
        layout.label(text="Ctrl + Alt + 1..6 assigns selected faces; 0 clears.")


CLASSES = (
    VIEW3D_OT_mesh_focus_select_exposed_backface,
    VIEW3D_OT_mesh_focus_orbit_recover_face_set_state,
    VIEW3D_OT_mesh_focus_orbit_watcher,
    VIEW3D_OT_mesh_focus_orbit_tool,
    VIEW3D_OT_mesh_focus_face_set_tool,
    VIEW3D_OT_mesh_focus_face_set_activate,
    VIEW3D_OT_mesh_focus_orbit,
    VIEW3D_OT_mesh_focus_guided_ridge,
    VIEW3D_OT_mesh_focus_shadow_analysis_toggle,
    VIEW3D_OT_mesh_focus_display_distance_toggle,
    VIEW3D_OT_mesh_focus_local_face_set_grow,
    VIEW3D_OT_mesh_focus_tube_shape,
    VIEW3D_OT_mesh_focus_local_feature_brush,
    VIEW3D_OT_mesh_focus_local_feature_brush_stroke,
    MESH_OT_mesh_focus_select_open_boundary_loop,
    MESH_OT_mesh_focus_local_remesh,
    VIEW3D_OT_mesh_focus_topology_color_assign,
    VIEW3D_PT_mesh_focus_local_feature_brush,
    VIEW3D_PT_mesh_focus_guided_ridge,
    VIEW3D_PT_mesh_focus_orbit_tools,
    VIEW3D_PT_mesh_focus_topology_colors,
    MESH_FOCUS_ORBIT_AddonPreferences,
)


@persistent
def _on_local_remesh_load_pre(_dummy):
    _local_remesh_request_cancel("load")


@persistent
def _on_local_remesh_undo_pre(_dummy):
    _local_remesh_request_cancel("undo")


@persistent
def _on_local_remesh_redo_pre(_dummy):
    _local_remesh_request_cancel("redo")


def _remove_registered_handlers():
    """Remove only this module's exact callback objects during teardown."""
    specs = (
        (bpy.app.handlers.load_pre, _on_local_remesh_load_pre),
        (bpy.app.handlers.undo_pre, _on_local_remesh_undo_pre),
        (bpy.app.handlers.redo_pre, _on_local_remesh_redo_pre),
        (bpy.app.handlers.load_pre, _on_load_pre),
        (bpy.app.handlers.load_post, _on_load_post),
        (bpy.app.handlers.load_pre, _display_distance_load_pre),
        (bpy.app.handlers.save_pre, _display_distance_save_pre),
        (bpy.app.handlers.save_post, _display_distance_save_post),
        (bpy.app.handlers.load_pre, _open_boundary_loop_invalidate_repeat_token),
        (bpy.app.handlers.undo_post, _open_boundary_loop_invalidate_repeat_token),
        (bpy.app.handlers.redo_post, _open_boundary_loop_invalidate_repeat_token),
        (bpy.app.handlers.undo_post, _on_undo_post),
        (bpy.app.handlers.redo_post, _on_fill_preview_redo_post),
        (bpy.app.handlers.depsgraph_update_post, _on_topology_color_depsgraph_update),
        (bpy.app.handlers.undo_post, _on_topology_color_undo_post),
        (bpy.app.handlers.redo_post, _on_topology_color_redo_post),
        (bpy.app.handlers.load_pre, _on_topology_color_load_pre),
        (bpy.app.handlers.load_post, _on_topology_color_load_post),
        (bpy.app.handlers.depsgraph_update_post, _on_fill_preview_depsgraph_update),
        (bpy.app.handlers.depsgraph_update_post, _on_guided_ridge_depsgraph_update),
        (bpy.app.handlers.load_pre, _on_guided_ridge_load_pre),
        (bpy.app.handlers.load_post, _on_guided_ridge_load_post),
        (bpy.app.handlers.undo_pre, _on_guided_ridge_undo_pre),
        (bpy.app.handlers.undo_post, _on_guided_ridge_undo_post),
        (bpy.app.handlers.redo_pre, _on_guided_ridge_redo_pre),
        (bpy.app.handlers.redo_post, _on_guided_ridge_redo_post),
        (bpy.app.handlers.depsgraph_update_post, _on_tube_preview_depsgraph_update),
        (bpy.app.handlers.depsgraph_update_post, _on_local_feature_brush_depsgraph_update),
        (bpy.app.handlers.undo_post, _on_local_feature_brush_undo_post),
        (bpy.app.handlers.redo_post, _on_local_feature_brush_redo_post),
        (bpy.app.handlers.load_pre, _on_local_feature_brush_load_pre),
        (bpy.app.handlers.load_post, _on_local_feature_brush_load_post),
    )
    save_post_fail_handlers = getattr(bpy.app.handlers, "save_post_fail", None)
    if save_post_fail_handlers is not None:
        specs += ((save_post_fail_handlers, _display_distance_save_post),)
    return _lifecycle.remove_registered_handlers(specs)


def register():
    if _runtime.is_registered:
        return
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    if not hasattr(bpy.types.Scene, "mfo_reference_object"):
        bpy.types.Scene.mfo_reference_object = PointerProperty(
            name="MFO Reference Object",
            description=(
                "Reference mesh used by Face Set MFO for center ray casting"
            ),
            type=bpy.types.Object,
            poll=_reference_object_poll,
        )
    _runtime.is_registered = True
    # Re-arm any retired-modal grace callback before normal registration work.
    # Failure remains fail-closed in the previous lifecycle preflight.
    _lifecycle.modal_quiescence_ensure()
    _lifecycle.pending_restore_resume()
    _register_tools()
    # A script reload can leave an old MFO wrapper on RetopoFlow's class even
    # though the previous module no longer has Python state for it.
    _restore_retopoflow_hooks()
    _schedule_orphan_cleanup()
    _start_topology_color_draw()
    if _on_local_remesh_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_local_remesh_load_pre)
    if _on_local_remesh_undo_pre not in bpy.app.handlers.undo_pre:
        bpy.app.handlers.undo_pre.append(_on_local_remesh_undo_pre)
    if _on_local_remesh_redo_pre not in bpy.app.handlers.redo_pre:
        bpy.app.handlers.redo_pre.append(_on_local_remesh_redo_pre)
    if _on_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_load_pre)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    for handlers in (
        bpy.app.handlers.load_pre,
        bpy.app.handlers.undo_post,
        bpy.app.handlers.redo_post,
    ):
        if _open_boundary_loop_invalidate_repeat_token not in handlers:
            handlers.append(_open_boundary_loop_invalidate_repeat_token)
    if _on_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_on_undo_post)
    if _on_fill_preview_redo_post not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(_on_fill_preview_redo_post)
    if _on_topology_color_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(
            _on_topology_color_depsgraph_update
        )
    if _on_topology_color_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_on_topology_color_undo_post)
    if _on_topology_color_redo_post not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(_on_topology_color_redo_post)
    if _on_topology_color_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_topology_color_load_pre)
    if _on_topology_color_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_topology_color_load_post)
    if _on_fill_preview_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_fill_preview_depsgraph_update)
    if _on_guided_ridge_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_guided_ridge_depsgraph_update)
    if _on_guided_ridge_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_guided_ridge_load_pre)
    if _on_guided_ridge_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_guided_ridge_load_post)
    if _on_guided_ridge_undo_pre not in bpy.app.handlers.undo_pre:
        bpy.app.handlers.undo_pre.append(_on_guided_ridge_undo_pre)
    if _on_guided_ridge_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_on_guided_ridge_undo_post)
    if _on_guided_ridge_redo_pre not in bpy.app.handlers.redo_pre:
        bpy.app.handlers.redo_pre.append(_on_guided_ridge_redo_pre)
    if _on_guided_ridge_redo_post not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(_on_guided_ridge_redo_post)
    if _on_tube_preview_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_tube_preview_depsgraph_update)
    if _on_local_feature_brush_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_local_feature_brush_depsgraph_update)
    if _on_local_feature_brush_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_on_local_feature_brush_undo_post)
    if _on_local_feature_brush_redo_post not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(_on_local_feature_brush_redo_post)
    if _on_local_feature_brush_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_local_feature_brush_load_pre)
    if _on_local_feature_brush_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_local_feature_brush_load_post)
    if _display_distance_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_display_distance_load_pre)
    if _display_distance_save_pre not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(_display_distance_save_pre)
    if _display_distance_save_post not in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.append(_display_distance_save_post)
    save_post_fail_handlers = getattr(bpy.app.handlers, "save_post_fail", None)
    if (
        save_post_fail_handlers is not None
        and _display_distance_save_post not in save_post_fail_handlers
    ):
        save_post_fail_handlers.append(_display_distance_save_post)
    _rebuild_keymaps()


def _fill_preview_modal_active():
    operator = _runtime.fill_preview_modal_operator
    session = _runtime.fill_preview_modal_session
    return bool(
        _runtime.fill_preview_modal_handler_live
        and operator is not None
        and session is not None
    )


def _active_modal_owners():
    """Return modal owners that must terminate before RNA teardown."""
    owners = []
    seen = set()
    def add_owner(label):
        label = str(label)
        if label not in seen:
            seen.add(label)
            owners.append(label)
    for label in _lifecycle.modal_registry_owners():
        add_owner(label)
    if _fill_preview_modal_active():
        add_owner("Smart Fill")
    for label, state in (
        ("Guided Ridge", _runtime.guided_ridge_state),
        ("Tube Shape", _runtime.tube_preview_state),
    ):
        if isinstance(state, dict) and state.get("active"):
            add_owner(label)
    if (
        _runtime.local_feature_brush_stroke_operator is not None
        or _runtime.local_feature_brush_pending_stroke is not None
    ):
        add_owner("Local Feature")
    return tuple(owners)


def _lifecycle_modal_preflight():
    owners = list(_active_modal_owners())
    owners.extend(_lifecycle.modal_quiescence_owners())
    return {"safe": not owners, "owners": owners}


def _fill_preview_modal_owner_terminal(operator):
    if _runtime.fill_preview_modal_operator is operator:
        _runtime.fill_preview_modal_operator = None
        _runtime.fill_preview_modal_session = None
        _runtime.fill_preview_modal_context = None
        _runtime.fill_preview_modal_handler_live = False
        _runtime.fill_preview_modal_cancel_requested = False
        _runtime.fill_preview_modal_cancel_reason = ""
        return True
    return False


def _fill_preview_invoke_failure_checkpoint(stage):
    """Test-only native invoke seam; production leaves it unset."""
    if getattr(_runtime, "_test_invoke_failure_stage", None) == str(stage):
        _runtime._test_invoke_failure_stage = None
        raise RuntimeError("injected Smart Fill invoke failure: %s" % stage)


def unregister():
    _display_distance_cancel_all()
    _local_remesh_request_cancel("unregister")
    _open_boundary_loop_clear_repeat_token()
    _runtime.fill_preview_reload_blocked = False
    _runtime.fill_preview_reload_warning = ""
    # A public unregister cannot remove a Python modal handler.  Mark its
    # registry entry for a self-contained terminal event, then continue the
    # normal Blender teardown so addon_utils.disable never leaves a
    # disabled-with-classes mismatch.
    for entry in tuple(_runtime.modal_registry.values()):
        entry["teardown_requested"] = True
        _lifecycle.modal_schedule_terminal_event(
            operator=entry.get("operator"),
            token=entry.get("token"),
            state=entry.get("state"),
        )
    _lifecycle.modal_schedule_orphan_cleanup()
    _remove_registered_handlers()
    _runtime.guided_ridge_last_guide = None
    _fill_preview_restore_manual_analysis_profiles()
    if not _runtime.is_registered:
        _fill_preview_request_cancel(reason="unregister")
        _tube_preview_request_cancel(reason="unregister")
        _guided_ridge_request_cancel(reason="unregister")
        _local_feature_request_cancel_all("unregister", restore=True)
        _cancel_orphan_cleanup()
        _cancel_undo_orphan_cleanup()
        _runtime.retopo_undo_tombstones.clear()
        _runtime.retopo_debug_sessions.clear()
        _runtime.retopo_debug_retired_sessions.clear()
        _remove_registered_handlers()
        _restore_retopoflow_hooks()
        _stop_topology_color_draw()
        _unregister_tools()
        _lifecycle.modal_force_teardown_orphans()
        return
    _finish_all_states()
    _fill_preview_request_cancel(reason="unregister")
    _tube_preview_request_cancel(reason="unregister")
    _guided_ridge_request_cancel(reason="unregister")
    _local_feature_request_cancel_all("unregister", restore=True)
    _cancel_orphan_cleanup()
    _cancel_undo_orphan_cleanup()
    _cleanup_orphan_face_set_proxies()
    _runtime.retopo_undo_tombstones.clear()
    _runtime.retopo_debug_sessions.clear()
    _runtime.retopo_debug_retired_sessions.clear()
    _restore_retopoflow_hooks()
    _remove_registered_handlers()
    _stop_topology_color_draw()
    _remove_keymaps()
    _unregister_tools()
    _runtime.local_face_set_adjacency_cache.clear()
    _runtime.fill_preview_adjacency_cache.clear()
    _runtime.fill_preview_cursor_cache.clear()
    _runtime.local_feature_brush_cache.clear()
    _runtime.local_feature_brush_states.clear()
    _lifecycle.pending_restore_clear()
    _runtime.is_registered = False
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Repeated disable/reload can encounter a class already removed by
            # Blender. Continue exact teardown of the remaining registrations.
            pass
    _lifecycle.modal_force_teardown_orphans()
    if hasattr(bpy.types.Scene, "mfo_reference_object"):
        del bpy.types.Scene.mfo_reference_object


# Wrap only the Smart Fill modal boundary so every event, including timer and Esc,
# contributes to the responsiveness measurements shown in diagnostics.
_sfsf_modal_uninstrumented = VIEW3D_OT_mesh_focus_local_face_set_grow.modal
def _sfsf_modal_instrumented(self, context, event):
    state = _runtime.fill_preview_state
    started = time.perf_counter()
    event_type = getattr(event, "type", "")
    event_value = getattr(event, "value", None)
    if state is not None:
        metrics = state.setdefault("metrics", {})
        metrics["modal_entries"] = int(metrics.get("modal_entries", 0)) + 1
        if event_type == "ESC" and event_value in {None, "PRESS"}:
            metrics["cancel_requested_at"] = time.perf_counter()
    try:
        return _sfsf_modal_uninstrumented(self, context, event)
    finally:
        current = state if state is not None else _runtime.fill_preview_state
        if current is not None:
            elapsed = time.perf_counter() - started
            metrics = current.setdefault("metrics", {})
            metrics["modal_max_seconds"] = max(
                float(metrics.get("modal_max_seconds", 0.0)), elapsed
            )

VIEW3D_OT_mesh_focus_local_face_set_grow.modal = _sfsf_modal_instrumented


if __name__ == "__main__":
    register()
def _fill_preview_build_adjacency_cooperative(obj, prepared=None):
    """Build only the reusable surface graph needed by preview sessions.

    This preparation reads all loops once, but intentionally does not compute
    curvature, valley bands, partition ownership, or contour relaxation.  All
    geometry features are recomputed on the session's candidate+halo patch.
    """
    import numpy as np

    mesh = obj.data
    signature = _fill_preview_signature(obj)
    if signature is None:
        raise RuntimeError("preview target is unavailable")
    cached = _runtime.fill_preview_adjacency_cache.get(signature)
    if cached is not None:
        refreshed = False
        if cached.get("geometry_dirty"):
            refreshed = _fill_preview_refresh_cached_adjacency(obj, cached)
            if refreshed:
                cached.pop("visibility_refresh_verified", None)
                cached.pop("visibility_refresh_hidden_count", None)
                return cached
            _runtime.fill_preview_adjacency_cache.pop(signature, None)
            cached = None
        visibility = (
            None
            if cached is None
            else _fill_preview_cached_visibility_check(obj, cached)
        )
        if visibility is not None and visibility.get("matches"):
            return cached
        if cached is not None:
            _runtime.fill_preview_adjacency_cache.pop(signature, None)

    if prepared is None:
        vertex_count = len(mesh.vertices)
        face_count = len(mesh.polygons)
        coordinates = np.empty((vertex_count, 3), dtype=np.float32)
        loop_edges = np.empty(len(mesh.loops), dtype=np.int32)
        loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
        totals = np.empty(face_count, dtype=np.int32)
        hidden = np.empty(face_count, dtype=bool)
        mesh.vertices.foreach_get("co", coordinates.ravel())
        mesh.loops.foreach_get("edge_index", loop_edges)
        mesh.loops.foreach_get("vertex_index", loop_vertices)
        mesh.polygons.foreach_get("loop_total", totals)
        mesh.polygons.foreach_get("hide", hidden)
        centers = np.empty((face_count, 3), dtype=np.float32)
        normals = np.empty((face_count, 3), dtype=np.float32)
        mesh.polygons.foreach_get("center", centers.ravel())
        mesh.polygons.foreach_get("normal", normals.ravel())
    else:
        coordinates = prepared["coordinates"]
        loop_edges = prepared["loop_edges"]
        loop_vertices = prepared["loop_vertices"]
        totals = prepared["totals"]
        hidden = prepared["hidden"]
        centers = prepared["centers"]
        normals = prepared["normals"]
        vertex_count = len(coordinates)
        face_count = len(totals)

    # Hash the local coordinates while the raw array is already resident.
    # The transformed graph only needs the fingerprint, not the raw buffer.
    coordinate_fingerprint = _fill_preview_array_fingerprint(coordinates)
    transform = np.asarray(obj.matrix_world.to_3x3(), dtype=np.float64)
    world_matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    translation = world_matrix[:3, 3]
    # The following NumPy transform is indivisible, so publish an honest
    # indeterminate state before entering it.  The timer records the wall
    # time of the native call and returns on the next stage boundary.
    yield _fill_preview_prepare_token("world-space", phase="prepare-finalize")
    world_vertices = coordinates.astype(np.float64) @ transform.T + translation
    centers = centers.astype(np.float64) @ transform.T + translation
    normals = normals.astype(np.float64) @ np.linalg.inv(transform)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-20)
    if isinstance(prepared, dict):
        # ``prepared`` is the one-shot state payload owned by this modal
        # session. Release the largest raw buffer once its world-space copy
        # and fingerprint exist; topology arrays remain live below.
        prepared.pop("coordinates", None)
    del coordinates
    yield _fill_preview_prepare_token("world-space-ready", 1, 1, phase="prepare-finalize")

    yield _fill_preview_prepare_token("edge-pairs", phase="prepare-finalize")
    face_ids = np.repeat(np.arange(face_count, dtype=np.int32), totals)
    order = np.argsort(loop_edges, kind="stable")
    sorted_edges = loop_edges[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_edges)) + 1]
    lengths = np.diff(np.r_[starts, len(order)])
    pair_starts = starts[lengths == 2]
    paired_loops = order[pair_starts]
    first = face_ids[paired_loops]
    second = face_ids[order[pair_starts + 1]]
    face_starts = np.r_[0, np.cumsum(totals)[:-1]]
    paired_next = face_starts[first] + (
        (paired_loops - face_starts[first] + 1) % totals[first]
    )
    edge_v0 = loop_vertices[paired_loops]
    edge_v1 = loop_vertices[paired_next]
    yield _fill_preview_prepare_token("edge-pairs-ready", 1, 1, phase="prepare-finalize")

    yield _fill_preview_prepare_token("seam-detection", phase="prepare-finalize")
    # Preserve the existing unwelded-seam behavior.  Only coincident open
    # edges with opposite winding and compatible normals become bridges.
    border_loops = order[starts[lengths == 1]]
    seam_count = 0
    if len(border_loops):
        border_faces = face_ids[border_loops]
        next_loops = face_starts[border_faces] + (
            (border_loops - face_starts[border_faces] + 1) % totals[border_faces]
        )
        points_a = world_vertices[loop_vertices[border_loops]]
        points_b = world_vertices[loop_vertices[next_loops]]
        edge_lengths = np.linalg.norm(points_b - points_a, axis=1)
        tolerance = max(float(np.median(edge_lengths)) * 1.0e-4, 1.0e-12)
        quantized = np.rint(np.vstack((points_a, points_b)) / tolerance).astype(np.int64)
        _, point_ids = np.unique(quantized, axis=0, return_inverse=True)
        pa, pb = np.split(point_ids, 2)
        signatures = np.column_stack((np.minimum(pa, pb), np.maximum(pa, pb)))
        _, inverse, counts = np.unique(
            signatures, axis=0, return_inverse=True, return_counts=True
        )
        seam_order = np.argsort(inverse, kind="stable")
        seam_starts = np.r_[0, np.cumsum(counts)[:-1]][counts == 2]
        ia, ib = seam_order[seam_starts], seam_order[seam_starts + 1]
        fa, fb = border_faces[ia], border_faces[ib]
        match = (
            (pa[ia] == pb[ib])
            & (pb[ia] == pa[ib])
            & (pa[ia] != pb[ia])
            & (fa != fb)
            & (np.sum(normals[fa] * normals[fb], axis=1) > 0.5)
            & (np.linalg.norm(points_a[ia] - points_b[ib], axis=1) <= tolerance)
            & (np.linalg.norm(points_b[ia] - points_a[ib], axis=1) <= tolerance)
        )
        first = np.r_[first, fa[match]]
        second = np.r_[second, fb[match]]
        edge_v0 = np.r_[edge_v0, loop_vertices[border_loops[ia[match]]]]
        edge_v1 = np.r_[edge_v1, loop_vertices[next_loops[ia[match]]]]
        seam_count = int(np.count_nonzero(match))
        yield _fill_preview_prepare_token("seam-detection-ready", 1, 1, phase="prepare-finalize")

    yield _fill_preview_prepare_token("topology-filter", phase="prepare-finalize")
    topology_first = np.asarray(first, dtype=np.int32)
    topology_second = np.asarray(second, dtype=np.int32)
    topology_edge_v0 = np.asarray(edge_v0, dtype=np.int32)
    topology_edge_v1 = np.asarray(edge_v1, dtype=np.int32)
    topology_fingerprint = _fill_preview_topology_fingerprint(
        loop_edges, loop_vertices, totals
    )
    valid = ~(hidden[first] | hidden[second]) & (first != second)
    all_visible = bool(np.all(valid))
    if not all_visible:
        # Hidden faces need an immutable raw pair list so visibility changes
        # can rebuild the filtered CSR without re-sorting every loop.  When
        # the common all-visible case applies, share the arrays with the
        # visible graph and avoid four large duplicate allocations.
        topology_first = topology_first.copy()
        topology_second = topology_second.copy()
        topology_edge_v0 = topology_edge_v0.copy()
        topology_edge_v1 = topology_edge_v1.copy()
        first, second = first[valid], second[valid]
        edge_v0, edge_v1 = edge_v0[valid], edge_v1[valid]
    yield _fill_preview_prepare_token("topology-validity-ready", 1, 1, phase="prepare-finalize")
    yield _fill_preview_prepare_token("edge-metrics", phase="prepare-finalize")
    delta = centers[second] - centers[first]
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-20)
    sources = np.r_[first, second]
    destinations = np.r_[second, first]
    source_edges = np.r_[np.arange(len(first)), np.arange(len(first))]
    edge_v0_directed = np.r_[edge_v0, edge_v0]
    edge_v1_directed = np.r_[edge_v1, edge_v1]
    yield _fill_preview_prepare_token("edge-metrics-ready", 1, 1, phase="prepare-finalize")
    # The full graph already has the flat loop vertex array and per-face
    # offsets.  Keep that compact schema instead of allocating millions of
    # Python tuples/integers; the compatibility accessor presents the same
    # indexed sequence to the boundary resolvers.
    yield _fill_preview_prepare_token("face-vertex-schema", phase="prepare-finalize")
    face_vertex_flat = loop_vertices
    face_vertex_offsets = np.asarray(face_starts, dtype=np.int64)
    yield _fill_preview_prepare_token(
        "face-vertex-schema-ready", 1, 1, phase="prepare-finalize"
    )
    yield _fill_preview_prepare_token("csr-order", phase="prepare-finalize")
    order = np.argsort(sources, kind="stable")
    yield _fill_preview_prepare_token("csr-order-ready", 1, 1, phase="prepare-finalize")
    degree = np.bincount(sources, minlength=face_count)
    offsets = np.r_[0, np.cumsum(degree)].astype(np.int64)
    pair_edge_points = (
        np.stack((world_vertices[edge_v0], world_vertices[edge_v1]), axis=1)
        if len(edge_v0)
        else np.empty((0, 2, 3), dtype=np.float64)
    )
    yield _fill_preview_prepare_token("csr-ready", 1, 1, phase="prepare-finalize")
    yield _fill_preview_prepare_token("cache-fingerprints", phase="prepare-finalize")
    # The first make-result call reads the live Face Set values anyway.  Its
    # fingerprint is recorded there, avoiding a second full attribute read
    # during cold graph assembly.  A missing fingerprint conservatively uses
    # the ordinary geometry validation path on an early dirty notification.
    face_set_fingerprint = None
    yield _fill_preview_prepare_token("cache-fingerprints-ready", 1, 1, phase="prepare-finalize")
    yield _fill_preview_prepare_token("cache-build", phase="prepare-finalize")
    identity_face_ids, identity_face_ids_provenance = (
        make_mesh_global_identity_face_ids(face_count)
    )
    cached = {
        "signature": signature,
        "count": int(face_count),
        "face_edge_counts": totals.astype(np.int32, copy=False),
        "centers": centers,
        "normals": normals,
        "hidden": hidden,
        "world_vertices": world_vertices,
        "offsets": offsets,
        "neighbors": destinations[order].astype(np.int32, copy=False),
        "neighbor_lengths": np.r_[distance, distance][order],
        # Keep the directed-to-physical mapping so screen-boundary refinement
        # can remain bounded even when the source mesh is very large.
        "edge_indices": source_edges[order].astype(np.int32, copy=False),
        "edge_v0": edge_v0_directed[order].astype(np.int32, copy=False),
        "edge_v1": edge_v1_directed[order].astype(np.int32, copy=False),
        "first": first,
        "second": second,
        "pair_lengths": distance,
        "pair_v0": edge_v0,
        "pair_v1": edge_v1,
        "pair_edge_points": pair_edge_points,
        "face_vertex_flat": face_vertex_flat,
        "face_vertex_offsets": face_vertex_offsets,
        "face_vertex_counts": totals.astype(np.int32, copy=False),
        "face_ids": identity_face_ids,
        "_face_ids_identity_provenance": identity_face_ids_provenance,
        "seam_count": seam_count,
        "vertex_id_space": "mesh-global",
        "topology_fingerprint": topology_fingerprint,
        "topology_first": topology_first,
        "topology_second": topology_second,
        "topology_edge_v0": topology_edge_v0,
        "topology_edge_v1": topology_edge_v1,
        "coordinate_fingerprint": coordinate_fingerprint,
        "face_set_fingerprint": face_set_fingerprint,
        "geometry_dirty": False,
        "geometry_refresh_count": 0,
    }
    # Drop obsolete revisions for this mesh while retaining unrelated meshes.
    mesh_pointer = signature[1]
    for key in list(_runtime.fill_preview_adjacency_cache):
        if key[1] == mesh_pointer and key != signature:
            _runtime.fill_preview_adjacency_cache.pop(key, None)
    _runtime.fill_preview_adjacency_cache[signature] = cached
    yield _fill_preview_prepare_token("cache-ready", 1, 1, phase="prepare-finalize")
    return cached
