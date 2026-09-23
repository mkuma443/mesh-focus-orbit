"""Mesh Focus Orbit Blender add-on package.

This is a thin package facade. Feature implementations are ordinary Python
modules with explicit imports; registration and cleanup are owned by the
registration module. The facade keeps the legacy public attribute contract
through a deliberate, read-only component lookup while avoiding a second
mutable state namespace.
"""

import importlib
import os
import sys
import types

bl_info = {
    "name": "Mesh Focus Orbit",
    "author": "OpenAI",
    "version": (3, 4, 7),
    "blender": (5, 2, 0),
    "location": "3D View",
    "description": "Mesh-centered orbit, Face Set tools, Smart Fill, and Guided Ridge",
    "category": "3D View",
}


def _load_components():
    """Load/reload children in dependency order without ``exec``.

    On a hot reload Blender classes and callbacks must be released while the
    old module objects are still reachable.  ``registration.unregister`` is
    therefore called before any child reload.  The runtime module is a
    process-local singleton and is intentionally retained across reloads.
    """
    package = __name__
    module_names = (
        "config",
        "runtime",
        "viewport",
        "lifecycle",
        "foundation",
        "guided_ridge.curve_sculpt",
        "guided_ridge.core",
        "local_feature",
        # Pure Smart Fill policy helpers must reload before geometry/preview;
        # those modules import their functions by name during a root reload.
        "smart_fill.invariants",
        "smart_fill.geometry",
        "smart_fill.preview",
        "tube_shape",
        "registration",
    )
    previous_registration = sys.modules.get(package + ".registration")
    previous_runtime = getattr(previous_registration, "_runtime", None)
    previous_lifecycle = getattr(previous_registration, "_lifecycle", None)
    if previous_registration is not None:
        # Reload safety is independent of the registration flag. A retired
        # modal can still be inside Blender's RNA callback after unregister,
        # and that grace state must block child reload even when the old
        # runtime already says is_registered == False.
        preflight_fn = getattr(previous_registration, "_lifecycle_modal_preflight", None)
        if callable(preflight_fn):
            preflight = preflight_fn()
        else:
            owners = []
            registry = getattr(previous_runtime, "modal_registry", {}) or {}
            owners.extend(
                str(entry.get("kind", "modal"))
                for entry in tuple(registry.values())
                if entry.get("operator") is not None
            )
            quiescence_safe = True
            if previous_lifecycle is not None:
                ensure = getattr(previous_lifecycle, "modal_quiescence_ensure", None)
                if callable(ensure):
                    quiescence_safe = bool(ensure())
                is_safe = getattr(previous_lifecycle, "modal_quiescence_is_safe", None)
                if callable(is_safe):
                    quiescence_safe = bool(is_safe()) and quiescence_safe
                owners.extend(
                    str(item)
                    for item in getattr(previous_lifecycle, "modal_quiescence_owners", lambda: ())()
                )
            preflight = {"safe": bool(quiescence_safe and not owners), "owners": owners}
        reload_safe = bool(preflight.get("safe", False))
        if not reload_safe:
            if previous_runtime is not None:
                previous_runtime.fill_preview_reload_blocked = True
                previous_runtime.fill_preview_reload_warning = (
                    "Mesh Focus Orbit reload refused: active modal "
                    + ", ".join(preflight.get("owners", ()))
                )
                warning = previous_runtime.fill_preview_reload_warning
            else:
                warning = (
                    "Mesh Focus Orbit reload refused: previous lifecycle is not quiescent"
                )
            # A root reload is an explicit lifecycle operation. Refuse before
            # unregister/reloading any child while a live Blender modal handler
            # or retired callback still owns callbacks.
            raise RuntimeError(warning)
        needs_unregister = bool(
            previous_runtime is not None
            and getattr(previous_runtime, "is_registered", False)
        )
        if needs_unregister:
            previous_registration.unregister()
    modules = {}
    for module_name in module_names:
        qualified = package + "." + module_name
        existing = sys.modules.get(qualified)
        if existing is not None and module_name != "runtime":
            modules[module_name] = importlib.reload(existing)
        else:
            modules[module_name] = importlib.import_module(qualified)
    return modules


_loaded = _load_components()
config = _loaded["config"]
runtime = _loaded["runtime"]
foundation = _loaded["foundation"]
guided_ridge_curve_sculpt = _loaded["guided_ridge.curve_sculpt"]
guided_ridge = _loaded["guided_ridge.core"]
local_feature = _loaded["local_feature"]
tube_shape = _loaded["tube_shape"]
smart_fill_geometry = _loaded["smart_fill.geometry"]
smart_fill_preview = _loaded["smart_fill.preview"]
registration = _loaded["registration"]
lifecycle = _loaded["lifecycle"]

_MFO_COMPONENT_FILES = (
    "foundation.py",
    "guided_ridge/curve_sculpt.py",
    "guided_ridge/core.py",
    "local_feature.py",
    "tube_shape.py",
    "smart_fill/geometry.py",
    "smart_fill/preview.py",
    "registration.py",
)
_MFO_COMPONENT_ORIGINS = {
    relative: os.path.join(
        os.path.dirname(__file__),
        *relative.split("/"),
    )
    for relative in _MFO_COMPONENT_FILES
}
_MFO_SUPPORT_FILES = ("config.py", "runtime.py", "viewport.py", "lifecycle.py")
_MFO_SUPPORT_ORIGINS = {
    relative: os.path.join(os.path.dirname(__file__), relative)
    for relative in _MFO_SUPPORT_FILES
}
_MFO_TOOL_ICON_DIR = getattr(registration, "_MFO_TOOL_ICON_DIR", "")

_COMPONENT_MODULES = (
    config,
    runtime,
    lifecycle,
    guided_ridge,
    guided_ridge_curve_sculpt,
    foundation,
    local_feature,
    tube_shape,
    smart_fill_geometry,
    smart_fill_preview,
    registration,
)



_COMPAT_EXPORTS = {
    # Each compatibility name has exactly one canonical owner.
    "_fill_preview_state": (runtime, "fill_preview_state"),
    "_tube_preview_state": (runtime, "tube_preview_state"),
    "_guided_ridge_state": (runtime, "guided_ridge_state"),
    "_is_registered": (runtime, "is_registered"),
    "_TempOrbitState": (foundation, "_TempOrbitState"),
    "_tool_event_coordinate": (foundation, "_tool_event_coordinate"),
    "_FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD": (
        smart_fill_preview, "_FILL_PREVIEW_TERMINAL_DELTA_FACE_THRESHOLD"
    ),
    "_MFO_TOOL_KEYMAP_NAMES": (registration, "_MFO_TOOL_KEYMAP_NAMES"),
    "_TOOL_CLASSES": (registration, "_TOOL_CLASSES"),
    "_MFO_TOOL_ICON_DIR": (registration, "_MFO_TOOL_ICON_DIR"),
    "VIEW3D_OT_mesh_focus_face_set_activate": (
        registration, "VIEW3D_OT_mesh_focus_face_set_activate"
    ),
    "VIEW3D_OT_mesh_focus_face_set_tool": (
        registration, "VIEW3D_OT_mesh_focus_face_set_tool"
    ),
    "VIEW3D_OT_mesh_focus_guided_ridge": (
        registration, "VIEW3D_OT_mesh_focus_guided_ridge"
    ),
    "VIEW3D_OT_mesh_focus_local_face_set_grow": (
        registration, "VIEW3D_OT_mesh_focus_local_face_set_grow"
    ),
    "VIEW3D_OT_mesh_focus_orbit_tool": (
        registration, "VIEW3D_OT_mesh_focus_orbit_tool"
    ),
    "TOOL_FACE_SET_OPERATOR_ID": (config, "TOOL_FACE_SET_OPERATOR_ID"),
    "GUIDED_RIDGE_CURVE_DEFAULT_SMOOTHING": (
        config, "GUIDED_RIDGE_CURVE_DEFAULT_SMOOTHING"
    ),
    "GUIDED_RIDGE_CURVE_NONZERO_MAX_SAMPLES": (
        guided_ridge, "GUIDED_RIDGE_CURVE_NONZERO_MAX_SAMPLES"
    ),
    "GUIDED_RIDGE_CURVE_SCULPT_STEP1": (
        config, "GUIDED_RIDGE_CURVE_SCULPT_STEP1"
    ),
    "GUIDED_RIDGE_CURVE_SCULPT_STEP2": (
        config, "GUIDED_RIDGE_CURVE_SCULPT_STEP2"
    ),
    "GUIDED_RIDGE_PREPARE_WORK_CHUNK": (
        config, "GUIDED_RIDGE_PREPARE_WORK_CHUNK"
    ),
    "GUIDED_RIDGE_UI_PROTOTYPE": (config, "GUIDED_RIDGE_UI_PROTOTYPE"),
    "_guided_ridge_safety_reason": (guided_ridge, "_guided_ridge_safety_reason"),
    "_guided_ridge_context_matches": (guided_ridge, "_guided_ridge_context_matches"),
    "_guided_ridge_surface_anchors": (guided_ridge, "_guided_ridge_surface_anchors"),
    "_guided_ridge_build_last_guide": (guided_ridge, "_guided_ridge_build_last_guide"),
    "_guided_ridge_current_signature": (guided_ridge, "_guided_ridge_current_signature"),
    "_guided_ridge_mesh_integrity": (guided_ridge, "_guided_ridge_mesh_integrity"),
}

def __getattr__(name):
    # Read-only compatibility lookup; no component broadcast/fan-out.
    export = _COMPAT_EXPORTS.get(name)
    if export is None:
        raise AttributeError(name)
    return getattr(export[0], export[1])

def register():
    return registration.register()

def unregister():
    return registration.unregister()
