"""Mutable runtime state for Mesh Focus Orbit.

This module intentionally contains data only. Feature modules import it as a
module (never individual mutable values) so reloads do not create split
session state.
"""

addon_keymaps = []
shadow_analysis_view_tokens = {}
display_distance_sessions = {}
active_states = {}
last_tap_times = {}
session_cleanup_areas = set()
retopo_undo_tombstones = {}
local_face_set_adjacency_cache = {}
fill_preview_adjacency_cache = {}
fill_preview_cursor_cache = {}
local_feature_brush_states = {}
local_feature_brush_cache = {}
topology_color_cache = {}
topology_color_cache_dirty = set()
retopo_debug_sessions = {}
retopo_debug_retired_sessions = {}

# All cross-module mutable/session state has one owner here.  Feature modules
# must access these through the runtime module, rather than importing a scalar
# snapshot from another component.  Keeping this module out of normal reload
# preserves the identity of active sessions, handlers, timers and caches.
session_serial = 0
modal_registry = {}
modal_registry_serial = 0
modal_terminal_timers = {}
# Immutable evidence of actual modal return boundaries. Normal modal paths
# append before removing their owner; unload bookkeeping never erases this log.
modal_terminal_events = []
# A Blender RNA modal callback may still be unwinding after it has returned its
# terminal status. Keep retired tokens in a short, explicit main-loop grace
# window before lifecycle reload/install preflight is considered safe.
modal_quiescence_pending = {}
modal_quiescence_timer = None
modal_quiescence_generation = 0
modal_orphan_cleanup_timer = None
pending_restores = {}
pending_restore_timer = None
retopo_isolation_serial = 0
undo_orphan_cleanup_pending = False
undo_orphan_cleanup_retries = 0
last_undo_post_perf = None
retopo_undo_retry_pending = False

fill_preview_state = None
# The Python modal handler itself cannot be removed through a public Blender
# API.  Keep its owner/session in the process singleton so lifecycle code can
# refuse an unload while Blender may still dispatch an event to the old RNA
# class.  These are deliberately runtime attributes, not imported scalars.
fill_preview_modal_operator = None
fill_preview_modal_session = None
fill_preview_modal_context = None
fill_preview_modal_handler_live = False
fill_preview_modal_cancel_requested = False
fill_preview_modal_cancel_reason = ""
fill_preview_reload_blocked = False
fill_preview_reload_warning = ""
fill_preview_last_confirm_metrics = {}
fill_preview_draw_handler = None
fill_preview_text_draw_handler = None
fill_preview_shader = None

tube_preview_state = None
tube_preview_draw_handler = None
tube_preview_text_draw_handler = None
tube_preview_shader = None

guided_ridge_state = None
guided_ridge_last_guide = None
guided_ridge_curve_2d_shader = None
guided_ridge_curve_last_smoothing = -12.0
# A short-lived native Sculpt transaction is owned here while the editor modal
# is suspended.  Keeping the record outside the operator instance lets the
# editor return a terminal status before Blender starts the native stroke, then
# re-enter the same route/curve session after the one-shot operation completes.
guided_ridge_native_transaction = None
guided_ridge_native_transaction_timer = None

local_feature_brush_load_guard = False
local_feature_brush_pending_stroke = None
local_feature_brush_stroke_operator = None

is_registered = False
polyquilt_qsnap_class = None
polyquilt_qsnap_original_snap_objects = None
polyquilt_qsnap_filter_installed = False
retopoflow_nearest_filter_class = None
retopoflow_nearest_filter_original_update = None
retopoflow_nearest_filter_installed = False
retopoflow_automerge_class = None
retopoflow_automerge_original = None
retopoflow_automerge_installed = False
retopoflow_automerge_nearest_bmvert = None
retopoflow_automerge_filter = None
retopoflow_automerge_session_id = None
retopoflow_automerge_retopo_session_id = None
retopoflow_polypen_class = None
retopoflow_polypen_original_update = None
retopoflow_polypen_installed = False
retopoflow_polypen_owner = None
retopoflow_polypen_nearest_bmvert = None
retopoflow_polypen_filter = None
retopoflow_polypen_session_id = None
retopoflow_polypen_retopo_session_id = None
retopoflow_translate_preview_class = None
retopoflow_translate_preview_original = None
retopoflow_translate_preview_installed = False
retopoflow_translate_preview_owner = None
retopoflow_translate_preview_nearest_bmvert = None
retopoflow_translate_preview_filter = None
retopoflow_translate_preview_session_id = None
retopoflow_translate_preview_retopo_session_id = None

topology_color_draw_handler = None
topology_color_depth_shader_cache = None
topology_color_fallback_shader = None

# Timer callback identities must live in the process singleton so an old
# foundation module can be torn down safely before its replacement is loaded.
deferred_orphan_cleanup = None
deferred_undo_orphan_cleanup = None

# A canonical service reference is populated by registration after classes and
# preferences exist.  Other components use this callable, never a copied
# preference object.
addon_preferences_provider = None
