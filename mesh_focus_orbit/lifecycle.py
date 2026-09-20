"""Explicit registration lifecycle helpers.

This module deliberately performs no work while it is imported.  Blender
handlers, timers, draw callbacks, keymaps, and RNA classes are removed only
from the explicit unregister path, by identity.  This keeps importing an
isolated copy of the package from touching a running add-on.
"""

from __future__ import annotations

from . import runtime

_PACKAGE_ROOT = str(__name__).split(".", 1)[0] or "mesh_focus_orbit"


def _pending_restore_driver_key():
    """Return the durable key for this actual package root.

    Isolated test copies can use a different root package name.  Deriving the
    key from this module's own import name keeps those records out of the
    production package while remaining stable across reloads.
    """
    return f"{_PACKAGE_ROOT}.guided_ridge.pending_restore"


def _driver_namespace():
    try:
        import bpy
        return bpy.app.driver_namespace
    except (AttributeError, RuntimeError, TypeError):
        return None


def _callback_path(callback):
    module = getattr(callback, "__module__", "")
    qualname = getattr(callback, "__qualname__", getattr(callback, "__name__", ""))
    return f"{module}:{qualname}" if module and qualname else ""


def _resolve_callback(path):
    if not path or ":" not in str(path):
        return None
    try:
        import importlib
        module_name, qualname = str(path).split(":", 1)
        target = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
        return target if callable(target) else None
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return None


def _sync_pending_restore_driver():
    namespace = _driver_namespace()
    if namespace is None:
        return
    records = {}
    for key, record in tuple(runtime.pending_restores.items()):
        records[str(key)] = {
            "context": record.get("context"),
            "restore": record.get("restore"),
            "callback_path": record.get("callback_path", ""),
            "operator": record.get("operator"),
        }
    if records:
        namespace[_pending_restore_driver_key()] = records
    else:
        namespace.pop(_pending_restore_driver_key(), None)


def _hydrate_pending_restores():
    namespace = _driver_namespace()
    if namespace is None:
        return
    records = namespace.get(_pending_restore_driver_key())
    if not isinstance(records, dict):
        return
    for key, persisted in tuple(records.items()):
        if not isinstance(persisted, dict):
            continue
        callback = _resolve_callback(persisted.get("callback_path", ""))
        if callback is None:
            continue
        current = runtime.pending_restores.get(str(key), {})
        runtime.pending_restores[str(key)] = {
            "context": persisted.get("context"),
            "restore": persisted.get("restore"),
            "callback": callback,
            "callback_path": persisted.get("callback_path", ""),
            "operator": persisted.get("operator", current.get("operator")),
        }


def _pending_restore_tick():
    """Retry exact Guided Ridge tool restores until context or record expires."""
    _hydrate_pending_restores()
    for key, record in tuple(runtime.pending_restores.items()):
        callback = _resolve_callback(record.get("callback_path", "")) or record.get("callback")
        try:
            restored = bool(callback(record.get("context"), record.get("restore")))
        except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError, ValueError):
            restored = False
        if restored:
            runtime.pending_restores.pop(key, None)
    _sync_pending_restore_driver()
    if runtime.pending_restores:
        return 0.10
    runtime.pending_restore_timer = None
    return None


def pending_restore_register(key, context, restore, callback, *, operator=None):
    """Keep a failed exact restore independent of feature session lifetime."""
    runtime.pending_restores[str(key)] = {
        "context": context,
        "restore": restore,
        "callback": callback,
        "callback_path": _callback_path(callback),
        "operator": operator,
    }
    _sync_pending_restore_driver()
    if runtime.pending_restore_timer is None:
        try:
            import bpy
            bpy.app.timers.register(
                _pending_restore_tick,
                first_interval=0.10,
            )
            runtime.pending_restore_timer = _pending_restore_tick
        except (AttributeError, RuntimeError, TypeError, ValueError):
            runtime.pending_restore_timer = None
    return str(key)


def pending_restore_remove(key):
    runtime.pending_restores.pop(str(key), None)
    # Keep the process-durable namespace in lockstep even when another record
    # remains.  Otherwise a removed record can be resurrected on the next
    # reload/hydration pass.
    _sync_pending_restore_driver()
    if runtime.pending_restores:
        return False
    timer = runtime.pending_restore_timer
    if timer is not None:
        try:
            import bpy
            if bpy.app.timers.is_registered(_pending_restore_tick):
                bpy.app.timers.unregister(_pending_restore_tick)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    runtime.pending_restore_timer = None
    _sync_pending_restore_driver()
    return True


def pending_restore_clear():
    """Stop retry scheduling without discarding durable restore records."""
    if runtime.pending_restore_timer is not None:
        try:
            import bpy
            if bpy.app.timers.is_registered(_pending_restore_tick):
                bpy.app.timers.unregister(_pending_restore_tick)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    runtime.pending_restore_timer = None
    _sync_pending_restore_driver()


def pending_restore_resume():
    """Hydrate durable records and ensure their retry timer is running."""
    _hydrate_pending_restores()
    if runtime.pending_restores and runtime.pending_restore_timer is None:
        try:
            import bpy
            bpy.app.timers.register(_pending_restore_tick, first_interval=0.10)
            runtime.pending_restore_timer = _pending_restore_tick
        except (AttributeError, RuntimeError, TypeError, ValueError):
            runtime.pending_restore_timer = None


def modal_register(operator, kind, session=None, *, state=None):
    """Register one exact Blender modal instance in the process singleton."""
    runtime.modal_registry_serial = int(runtime.modal_registry_serial) + 1
    token = f"{kind}:{runtime.modal_registry_serial}:{id(operator):x}"
    entry = {
        "token": token,
        "operator": operator,
        "kind": str(kind),
        "session": session,
        "state": state,
        "cancel_requested": False,
        "cancel_reason": "",
        "teardown_requested": False,
        "terminal_timer": None,
        "terminal_timer_window_manager": None,
    }
    runtime.modal_registry[token] = entry
    if isinstance(state, dict):
        state["modal_token"] = token
    return token


def modal_entry(operator=None, token=None):
    for key, entry in tuple(runtime.modal_registry.items()):
        if token is not None and key == token:
            return entry
        if operator is not None and entry.get("operator") is operator:
            return entry
    return None


def modal_request_cancel(operator=None, token=None, reason="external-change"):
    """Mark an owner for its next modal event without clearing liveness."""
    entry = modal_entry(operator=operator, token=token)
    if entry is None:
        return None
    entry["cancel_requested"] = True
    entry["cancel_reason"] = str(reason)
    return entry


def modal_schedule_terminal_event(operator=None, token=None, state=None):
    """Attach one minimal event timer so a cancelled modal can return terminal."""
    entry = modal_entry(operator=operator, token=token)
    if entry is None or entry.get("terminal_timer") is not None:
        return entry
    state = state if isinstance(state, dict) else entry.get("state")
    try:
        window_manager = state.get("window_manager") if isinstance(state, dict) else None
        window = state.get("window") if isinstance(state, dict) else None
        if window_manager is None:
            import bpy
            window_manager = bpy.context.window_manager
        if window is None:
            import bpy
            window = bpy.context.window
        if window_manager is not None:
            timer = window_manager.event_timer_add(0.01, window=window)
            entry["terminal_timer"] = timer
            entry["terminal_timer_window_manager"] = window_manager
            runtime.modal_terminal_timers[entry["token"]] = timer
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        entry["terminal_timer"] = None
    return entry


def _modal_quiescence_tick():
    """Advance the post-return main-loop grace window.

    The owner is already retired from the modal registry when this runs, but
    Blender may still be unwinding the RNA callback on the same event turn.
    Two distinct app-timer callbacks are required before reload/install
    preflight can proceed.  This callback never force-clears a live owner.
    """
    pending = runtime.modal_quiescence_pending
    for record in tuple(pending.values()):
        record["ticks"] = int(record.get("ticks", 0)) + 1
    for token, record in tuple(pending.items()):
        if int(record.get("ticks", 0)) >= 2:
            pending.pop(token, None)
    if pending:
        return 0.01
    runtime.modal_quiescence_timer = None
    return None


def _modal_quiescence_timer_registered(callback=None):
    callback = callback or _modal_quiescence_tick
    try:
        import bpy
        return bool(bpy.app.timers.is_registered(callback))
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return False


def modal_quiescence_ensure():
    """Ensure a pending grace record has a live, current app-timer callback.

    Pending retirement is fail-closed: a lost or failed timer never makes
    reload/install appear safe. The exact callback identity is checked through
    Blender's public timer registry before reusing the stored reference.
    """
    pending = runtime.modal_quiescence_pending
    timer = runtime.modal_quiescence_timer
    if not pending:
        if timer is not None and not _modal_quiescence_timer_registered(timer):
            runtime.modal_quiescence_timer = None
        return True
    if _modal_quiescence_timer_registered(_modal_quiescence_tick):
        runtime.modal_quiescence_timer = _modal_quiescence_tick
        return True
    runtime.modal_quiescence_timer = None
    try:
        import bpy
        bpy.app.timers.register(_modal_quiescence_tick, first_interval=0.01)
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        # Keep the pending record and refuse safety until a later preflight can
        # retry registration.
        runtime.modal_quiescence_timer = None
        return False
    if not _modal_quiescence_timer_registered(_modal_quiescence_tick):
        runtime.modal_quiescence_timer = None
        return False
    runtime.modal_quiescence_timer = _modal_quiescence_tick
    return True


def modal_quiescence_is_safe():
    """Return true only after every retired modal crossed two timer turns."""
    modal_quiescence_ensure()
    return not bool(runtime.modal_quiescence_pending)


def modal_quiescence_owners():
    """Return diagnostics for retired callbacks still in grace."""
    modal_quiescence_ensure()
    return tuple(
        "Modal quiescence:%s" % token
        for token in tuple(runtime.modal_quiescence_pending)
    )


def _modal_quiescence_schedule(token, generation):
    runtime.modal_quiescence_pending[str(token)] = {
        "generation": int(generation),
        "ticks": 0,
    }
    modal_quiescence_ensure()


def modal_terminal(operator=None, token=None, status=None):
    """Record and clear one owner immediately before its modal terminal return."""
    entry = modal_entry(operator=operator, token=token)
    if entry is None:
        return False
    if status in {"FINISHED", "CANCELLED"}:
        events = runtime.modal_terminal_events
        token = entry.get("token")
        events.append({
            "token": token,
            "kind": entry.get("kind"),
            "status": str(status),
            "operator_id": id(entry.get("operator")),
        })
        if len(events) > 128:
            del events[:-128]
        runtime.modal_quiescence_generation = int(
            runtime.modal_quiescence_generation
        ) + 1
        _modal_quiescence_schedule(
            token,
            runtime.modal_quiescence_generation,
        )
    timer = entry.get("terminal_timer")
    if timer is not None:
        try:
            wm = entry.get("terminal_timer_window_manager")
            if wm is None:
                import bpy
                wm = bpy.context.window_manager
            wm.event_timer_remove(timer)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
    runtime.modal_terminal_timers.pop(entry["token"], None)
    entry["terminal_timer_window_manager"] = None
    runtime.modal_registry.pop(entry["token"], None)
    return True


def _modal_orphan_cleanup_tick():
    """Finish owners whose RNA class was removed by an explicit unregister.

    Blender 5.2 exposes no public modal-handler removal API.  The normal
    disable path therefore requests cancellation before class teardown and
    leaves this one-shot lifecycle callback to retire the exact owner record.
    Any surviving Python modal receives CANCELLED if Blender dispatches a
    later event; the callback itself keeps runtime state from becoming stale.
    """
    for entry in tuple(runtime.modal_registry.values()):
        if not entry.get("teardown_requested"):
            continue
        modal_terminal(token=entry.get("token"))
    if not any(
        str(entry.get("kind", "")) == "Smart Fill"
        for entry in tuple(runtime.modal_registry.values())
    ):
        runtime.fill_preview_modal_operator = None
        runtime.fill_preview_modal_session = None
        runtime.fill_preview_modal_context = None
        runtime.fill_preview_modal_handler_live = False
        runtime.fill_preview_modal_cancel_requested = False
        runtime.fill_preview_modal_cancel_reason = ""
    runtime.modal_orphan_cleanup_timer = None
    return None


def modal_schedule_orphan_cleanup():
    """Schedule exact owner retirement for disable/reload teardown."""
    if runtime.modal_orphan_cleanup_timer is not None:
        return True
    if not any(
        bool(entry.get("teardown_requested"))
        for entry in tuple(runtime.modal_registry.values())
    ):
        return False
    try:
        import bpy
        bpy.app.timers.register(_modal_orphan_cleanup_tick, first_interval=0.01)
        runtime.modal_orphan_cleanup_timer = _modal_orphan_cleanup_tick
        return True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def modal_force_teardown_orphans():
    """Retire teardown-requested owners at explicit add-on unload.

    Blender 5.2 may unregister an add-on from an app-timer callback while the
    modal handler is still owned by the old RNA class. In that narrow unload
    boundary no subsequent modal event is guaranteed, and the one-shot app
    timer can be discarded with class/module teardown. The explicit
    unregister path therefore performs idempotent bookkeeping cleanup for
    those exact owners only. It is not evidence that Blender delivered a
    terminal modal event and it must never be reported as a native
    FINISHED/CANCELLED return. Normal modal events alone use modal_terminal
    with a status immediately before returning that status.
    """
    for entry in tuple(runtime.modal_registry.values()):
        if entry.get("teardown_requested"):
            modal_terminal(token=entry.get("token"))
    if not any(
        str(entry.get("kind", "")) == "Smart Fill"
        for entry in tuple(runtime.modal_registry.values())
    ):
        runtime.fill_preview_modal_operator = None
        runtime.fill_preview_modal_session = None
        runtime.fill_preview_modal_context = None
        runtime.fill_preview_modal_handler_live = False
        runtime.fill_preview_modal_cancel_requested = False
        runtime.fill_preview_modal_cancel_reason = ""
    orphan_timer = runtime.modal_orphan_cleanup_timer
    if orphan_timer is not None:
        try:
            import bpy
            if bpy.app.timers.is_registered(_modal_orphan_cleanup_tick):
                bpy.app.timers.unregister(_modal_orphan_cleanup_tick)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    runtime.modal_orphan_cleanup_timer = None


def modal_registry_owners():
    return tuple(
        str(entry.get("kind", "modal"))
        for entry in tuple(runtime.modal_registry.values())
        if entry.get("operator") is not None
    )


def remove_handler_identity(handler_list, callback, *, owner_modules=()):
    """Remove *callback* only when the exact registered object is present.

    ``list.remove`` uses equality, which is too permissive for some mocked
    Blender callback objects.  Identity also makes cleanup safe when a reload
    has created a new function with the same name.  ``owner_modules`` is an
    optional final ownership guard used by the registration module.
    """
    if callback is None:
        return False
    allowed = set(owner_modules or ())
    for index, current in reversed(tuple(enumerate(handler_list))):
        if current is not callback:
            continue
        if allowed:
            module_name = getattr(current, "__module__", None)
            if module_name not in allowed:
                return False
        del handler_list[index]
        return True
    return False


def remove_registered_handlers(specs, *, owner_modules=()):
    """Remove a sequence of ``(handler_list, callback)`` registrations."""
    removed = 0
    for handler_list, callback in tuple(specs or ()):
        if remove_handler_identity(
            handler_list, callback, owner_modules=owner_modules
        ):
            removed += 1
    return removed


def snapshot_handler_identities(handler_lists):
    """Return a read-only identity snapshot for lifecycle regression tests."""
    return tuple(
        tuple(id(callback) for callback in handler_list)
        for handler_list in tuple(handler_lists or ())
    )
