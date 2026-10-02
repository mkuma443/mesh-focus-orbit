"""Local triangle density for a manually repaired patch in Edit Mode."""

import math
import statistics
import time

import bmesh
import bpy
from bpy.props import FloatProperty, IntProperty

from . import lifecycle as _lifecycle
from . import runtime as _runtime


def _target_length(bm, selected, boundary_edges):
    """Prefer the unselected mesh immediately outside the repaired patch."""
    rim_verts = {vert for edge in boundary_edges for vert in edge.verts}
    neighbors = {
        edge
        for vert in rim_verts
        for edge in vert.link_edges
        if edge not in boundary_edges
        and any(face not in selected for face in edge.link_faces)
    }
    samples = []
    for edge in neighbors:
        length = edge.calc_length()
        if length > 1.0e-12:
            samples.append(length)
    if not samples:
        for face in selected:
            for edge in face.edges:
                length = edge.calc_length()
                if length > 1.0e-12:
                    samples.append(length)
    if not samples:
        raise ValueError("Could not estimate a target edge length")
    return statistics.median(samples)


def _is_valid(element):
    try:
        return bool(element.is_valid)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _face_snapshot(face, original_id_layer):
    """Capture an external face by stable local ids and representative data."""
    return (
        tuple(int(vert[original_id_layer]) for vert in face.verts),
        tuple(tuple(float(value) for value in vert.co) for vert in face.verts),
        int(face.material_index),
        bool(face.smooth),
        bool(face.select),
        bool(face.hide),
    )


def _make_boundary_edges(selected, selected_set):
    return {
        edge
        for face in selected
        for edge in face.edges
        if any(link not in selected_set for link in edge.link_faces)
        or sum(link in selected_set for link in edge.link_faces) == 1
    }


def _source_vertex_snapshot(vert):
    return (
        int(vert.index),
        tuple(float(value) for value in vert.co),
    )


def _source_face_snapshot(face):
    return (
        tuple(_source_vertex_snapshot(vert) for vert in face.verts),
        int(face.material_index),
        bool(face.smooth),
        bool(face.select),
        bool(face.hide),
    )


def _patch_source_signature(bm, selected=None):
    """Describe the selected patch and its shared external boundary only."""
    bm.verts.index_update()
    if selected is None:
        selected = [
            face for face in bm.faces
            if face.select and not face.hide
        ]
    selected_set = set(selected)
    selected_faces = tuple(
        sorted(_source_face_snapshot(face) for face in selected)
    )
    selected_edges = {
        edge for face in selected for edge in face.edges
    }
    edge_signatures = tuple(
        sorted(
            (
                tuple(sorted(_source_vertex_snapshot(vert) for vert in edge.verts)),
                bool(edge.seam),
                bool(edge.smooth),
                bool(edge.select),
                bool(edge.hide),
            )
            for edge in selected_edges
        )
    )
    boundary_signatures = []
    for edge in _make_boundary_edges(selected, selected_set):
        external_faces = tuple(
            sorted(
                _source_face_snapshot(face)
                for face in edge.link_faces
                if face not in selected_set
            )
        )
        boundary_signatures.append(
            (
                tuple(sorted(_source_vertex_snapshot(vert) for vert in edge.verts)),
                bool(edge.seam),
                bool(edge.smooth),
                bool(edge.select),
                bool(edge.hide),
                external_faces,
            )
        )
    return (
        selected_faces,
        edge_signatures,
        tuple(sorted(boundary_signatures)),
    )


def _prepare_patch_state(bm, target, max_faces):
    # Create all temporary layers before retaining element references: adding a
    # BMesh custom-data layer can invalidate existing Python wrappers.
    region_layer = bm.faces.layers.int.new("__mfo_local_remesh_region")
    original_id_layer = bm.verts.layers.int.new(
        "__mfo_local_remesh_original_id"
    )
    try:
        selected = []
        for face in bm.faces:
            if face.select and not face.hide:
                selected.append(face)
        source_signature = _patch_source_signature(bm, selected)
        if not selected:
            raise ValueError("Select the repaired faces first")
        if max_faces < 1:
            raise ValueError("Maximum patch face count must be positive")
        triangulated_face_estimate = sum(len(face.verts) - 2 for face in selected)
        if triangulated_face_estimate > max_faces:
            raise ValueError(
                "Patch exceeds the face limit; increase the target edge length"
            )

        selected_set = set(selected)
        original_coordinates = {}
        patch_original_ids = set()
        original_ids_by_vert = {}
        next_original_id = 1

        def assign_original_id(vert):
            nonlocal next_original_id
            original_id = original_ids_by_vert.get(vert)
            if original_id is None:
                original_id = next_original_id
                next_original_id += 1
                original_ids_by_vert[vert] = original_id
                vert[original_id_layer] = original_id
                original_coordinates[original_id] = vert.co.copy()
            return original_id

        for face in selected:
            face[region_layer] = 1
            for vert in face.verts:
                patch_original_ids.add(assign_original_id(vert))

        boundary_edges = _make_boundary_edges(selected, selected_set)
        if target <= 0.0:
            target = _target_length(bm, selected_set, boundary_edges)
        if not math.isfinite(target) or target <= 1.0e-12:
            raise ValueError("Target edge length must be positive")

        boundary_snapshots = []
        for edge in boundary_edges:
            endpoint_ids = frozenset(
                assign_original_id(vert) for vert in edge.verts
            )
            external_faces = [
                face for face in edge.link_faces if face not in selected_set
            ]
            for face in external_faces:
                for vert in face.verts:
                    assign_original_id(vert)
            external_signatures = tuple(
                sorted(
                    _face_snapshot(face, original_id_layer)
                    for face in external_faces
                )
            )
            boundary_snapshots.append((endpoint_ids, external_signatures))

        return {
            "bm": bm,
            "target": float(target),
            "max_faces": int(max_faces),
            "region_layer": region_layer,
            "original_id_layer": original_id_layer,
            "patch_original_ids": patch_original_ids,
            "original_coordinates": original_coordinates,
            "selected": selected,
            "source_signature": source_signature,
            "region": set(selected),
            "boundary_snapshots": boundary_snapshots,
            "area_limit": 0.5 * float(target) * float(target),
            "passes": 0,
        }
    except Exception:
        try:
            bm.faces.layers.int.remove(region_layer)
        except (ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        try:
            bm.verts.layers.int.remove(original_id_layer)
        except (ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        raise


def _vertex_is_original(state, vert):
    """Distinguish original vertices from generated/interpolated ones."""
    try:
        original_id = int(vert[state["original_id_layer"]])
        original_co = state["original_coordinates"].get(original_id)
        return (
            original_co is not None
            and (vert.co - original_co).length <= 1.0e-10
        )
    except (ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def _add_region_face(state, face):
    if not _is_valid(face):
        return
    try:
        face[state["region_layer"]] = 1
        state["region"].add(face)
    except (ReferenceError, RuntimeError, TypeError, ValueError):
        return


def _collect_operation_geometry(state, result):
    """Keep the patch face set local to operation results and generated vertices."""
    for face in result.get("faces", ()):
        _add_region_face(state, face)
    generated_vertices = set()
    for element in result.get("geom", ()):
        if not _is_valid(element):
            continue
        if isinstance(element, bmesh.types.BMFace):
            _add_region_face(state, element)
        elif isinstance(element, bmesh.types.BMVert):
            if not _vertex_is_original(state, element):
                generated_vertices.add(element)
    for vert in result.get("verts", ()):
        if _is_valid(vert) and not _vertex_is_original(state, vert):
            generated_vertices.add(vert)

    # Subdivision can create faces that its geom result omits. Every returned
    # new vertex lies on an edge/fan wholly inside the patch, so its incident
    # faces are safe to tag without walking unrelated BMesh faces.
    for vert in generated_vertices:
        if not _is_valid(vert):
            continue
        for face in vert.link_faces:
            _add_region_face(state, face)


def _refresh_region(state):
    region = set()
    layer = state["region_layer"]
    for face in state["region"]:
        if not _is_valid(face):
            continue
        try:
            if face[layer] == 1:
                region.add(face)
        except (ReferenceError, RuntimeError, TypeError, ValueError):
            continue
    state["region"] = region
    return region


def _triangulate_patch(state):
    selected = [face for face in state["selected"] if _is_valid(face)]
    result = bmesh.ops.triangulate(
        state["bm"],
        faces=selected,
        quad_method="BEAUTY",
        ngon_method="BEAUTY",
    )
    _collect_operation_geometry(state, result)
    _refresh_region(state)


def _refine_patch_step(state):
    """Run one subdivision/poke pass and return whether refinement is complete."""
    region = _refresh_region(state)
    if len(region) > state["max_faces"]:
        raise ValueError(
            "Patch exceeds the face limit; increase the target edge length"
        )
    long_edges = {
        edge
        for face in region
        for edge in face.edges
        if len(edge.link_faces) == 2
        and all(link[state["region_layer"]] == 1 for link in edge.link_faces)
        and edge.calc_length() > state["target"] * 1.2
    }
    large_faces = [
        face for face in region if face.calc_area() > state["area_limit"]
    ]
    if not long_edges and not large_faces:
        return True
    if len(region) + len(large_faces) * 2 + len(long_edges) * 2 > state["max_faces"]:
        raise ValueError(
            "Patch exceeds the face limit; increase the target edge length"
        )
    if state["passes"] >= 16:
        raise ValueError("Patch did not reach the requested density within 16 passes")

    if long_edges:
        result = bmesh.ops.subdivide_edges(
            state["bm"],
            edges=list(long_edges),
            cuts=1,
            use_grid_fill=False,
        )
        _collect_operation_geometry(state, result)
        region = _refresh_region(state)
        large_faces = [
            face for face in region if face.calc_area() > state["area_limit"]
        ]
    if large_faces:
        result = bmesh.ops.poke(state["bm"], faces=large_faces)
        _collect_operation_geometry(state, result)
        _refresh_region(state)

    state["passes"] += 1
    return False


def _boundary_edge_map(state, region):
    edge_map = {}
    layer = state["original_id_layer"]
    for face in region:
        for edge in face.edges:
            endpoints = [(vert, int(vert[layer])) for vert in edge.verts]
            endpoint_ids = frozenset(original_id for _vert, original_id in endpoints)
            if len(endpoint_ids) != 2 or not all(
                original_id in state["patch_original_ids"]
                and _vertex_is_original(state, vert)
                for vert, original_id in endpoints
            ):
                continue
            edge_map.setdefault(endpoint_ids, set()).add(edge)
    return edge_map


def _verify_local_boundary(state, region):
    region_layer = state["region_layer"]
    original_id_layer = state["original_id_layer"]
    edge_map = _boundary_edge_map(state, region)
    for endpoint_ids, original_external_signatures in state["boundary_snapshots"]:
        edges = edge_map.get(endpoint_ids, set())
        if len(edges) != 1:
            raise ValueError("Patch boundary would change")
        edge = next(iter(edges))
        if not _is_valid(edge):
            raise ValueError("Patch boundary would change")
        if frozenset(int(vert[original_id_layer]) for vert in edge.verts) != endpoint_ids:
            raise ValueError("Patch boundary would change")
        external_faces = [
            face
            for face in edge.link_faces
            if not _is_valid(face) or face[region_layer] != 1
        ]
        current_signatures = tuple(
            sorted(
                _face_snapshot(face, original_id_layer)
                for face in external_faces
                if _is_valid(face)
            )
        )
        if current_signatures != original_external_signatures:
            raise ValueError("Patch boundary connectivity would change")


def _remove_temp_layers(state):
    if state is None:
        return
    region_layer = state.get("region_layer")
    if region_layer is not None:
        try:
            state["bm"].faces.layers.int.remove(region_layer)
        except (ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        state["region_layer"] = None
    original_id_layer = state.get("original_id_layer")
    if original_id_layer is not None:
        try:
            state["bm"].verts.layers.int.remove(original_id_layer)
        except (ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        state["original_id_layer"] = None


def _relax_and_verify(state):
    region = _refresh_region(state)
    result = bmesh.ops.triangulate(
        state["bm"],
        faces=list(region),
        quad_method="BEAUTY",
        ngon_method="BEAUTY",
    )
    _collect_operation_geometry(state, result)
    region = _refresh_region(state)
    if len(region) > state["max_faces"]:
        raise ValueError(
            "Patch exceeds the face limit; increase the target edge length"
        )

    region_vertices = {vert for face in region for vert in face.verts}
    original_seen = {}
    new_vertices = set()
    for vert in region_vertices:
        original_id = int(vert[state["original_id_layer"]])
        original_co = state["original_coordinates"].get(original_id)
        if original_co is not None and (vert.co - original_co).length <= 1.0e-10:
            if original_id in state["patch_original_ids"]:
                prior_vert = original_seen.get(original_id)
                if prior_vert is not None and prior_vert is not vert:
                    raise ValueError("An existing vertex would be duplicated")
                original_seen[original_id] = vert
        else:
            new_vertices.add(vert)

    if set(original_seen) != state["patch_original_ids"]:
        raise ValueError("An existing patch vertex would move")

    # Relax only generated vertices with a fully two-manifold fan inside the
    # selected patch. Original and shared boundary vertices never move.
    updates = {}
    layer = state["region_layer"]
    for vert in new_vertices:
        if any(
            len(edge.link_faces) != 2
            or any(face[layer] != 1 for face in edge.link_faces)
            for edge in vert.link_edges
        ):
            continue
        neighbors = [edge.other_vert(vert).co for edge in vert.link_edges]
        if neighbors:
            mean = sum(neighbors, vert.co.copy() * 0.0) / len(neighbors)
            updates[vert] = vert.co.lerp(mean, 0.12)
    for vert, co in updates.items():
        vert.co = co

    if any(len(face.verts) != 3 for face in region):
        raise ValueError("Patch triangulation failed")
    _verify_local_boundary(state, region)

    for face in region:
        face.select_set(True)
    count = len(region)
    _remove_temp_layers(state)
    return state["target"], count


def _build_patch(bm, target, max_faces, pass_counter=None):
    state = _prepare_patch_state(bm, target, max_faces)
    try:
        _triangulate_patch(state)
        while not _refine_patch_step(state):
            pass
        if pass_counter is not None:
            pass_counter.append(int(state["passes"]))
        return _relax_and_verify(state)
    finally:
        _remove_temp_layers(state)


def _pointer(value):
    try:
        return int(value.as_pointer())
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None


def _local_remesh_timer_ready(state, event, now=None):
    """Accept a TIMER tick after the prior modal step has had time to redraw."""
    if getattr(event, "type", None) != "TIMER":
        return False
    current_time = time.perf_counter() if now is None else float(now)
    return current_time >= float(state.get("next_tick_at", 0.0))


def _local_remesh_schedule_next_tick(state, now=None):
    current_time = time.perf_counter() if now is None else float(now)
    interval = max(0.03, float(state.get("timer_interval_s", 0.03)))
    state["next_tick_at"] = current_time + interval


def _local_remesh_context_matches(state, context, *, verify_source=False):
    if (
        context is None
        or _pointer(getattr(context, "window", None)) != state["window_pointer"]
        or _pointer(getattr(context, "scene", None)) != state["scene_pointer"]
        or getattr(context, "mode", None) != "EDIT_MESH"
    ):
        return False
    obj = getattr(context, "edit_object", None)
    mesh = getattr(obj, "data", None)
    if (
        obj is None
        or mesh is None
        or getattr(obj, "type", None) != "MESH"
        or _pointer(obj) != state["object_pointer"]
        or _pointer(mesh) != state["mesh_pointer"]
        or getattr(mesh, "shape_keys", None) is not None
    ):
        return False
    if verify_source:
        live_bm = bmesh.from_edit_mesh(mesh)
        counts = (len(live_bm.verts), len(live_bm.edges), len(live_bm.faces))
        if counts != state["source_counts"]:
            return False
        if _patch_source_signature(live_bm) != state.get("source_signature"):
            return False
    return True


def _local_remesh_publish_status(state, key, label, progress):
    state["phase"] = key
    state["phase_label"] = label
    phases = state["phase_events"]
    if not phases or phases[-1] != label:
        phases.append(label)
    state["progress_samples"].append((label, float(progress)))
    if len(state["progress_samples"]) > 64:
        del state["progress_samples"][:-64]
    text = f"Local Remesh | {label} | Esc to cancel"
    workspace = state.get("workspace")
    try:
        if workspace is not None:
            workspace.status_text_set(text)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    window_manager = state.get("window_manager")
    if state.get("progress_started") and window_manager is not None:
        try:
            window_manager.progress_update(float(progress))
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
    window = state.get("window")
    try:
        if window is not None and window.screen is not None:
            for area in window.screen.areas:
                area.tag_redraw()
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def _local_remesh_measure(state, label, callback):
    started = time.perf_counter()
    try:
        return callback()
    finally:
        elapsed = time.perf_counter() - started
        state["phase_times"][label] = (
            state["phase_times"].get(label, 0.0) + elapsed
        )
        state["unyielding_calls"].append((label, elapsed))
        if len(state["unyielding_calls"]) > 64:
            del state["unyielding_calls"][:-64]
        if elapsed > state["max_unyielding_s"]:
            state["max_unyielding_s"] = elapsed
            state["max_unyielding_stage"] = label


def _local_remesh_dispose_staged(state):
    """Release all staged resources and transient UI state, idempotently."""
    if not isinstance(state, dict) or state.get("resources_disposed"):
        return
    state["resources_disposed"] = True
    patch_state = state.get("patch_state")
    if patch_state is not None:
        try:
            _remove_temp_layers(patch_state)
        except (ReferenceError, RuntimeError, TypeError, ValueError):
            pass
        state["patch_state"] = None
    staged = state.get("staged")
    state["staged"] = None
    if staged is not None:
        try:
            staged.free()
        except (ReferenceError, RuntimeError, TypeError, ValueError):
            pass
    window_manager = state.get("window_manager")
    timer = state.get("timer")
    state["timer"] = None
    if window_manager is not None and timer is not None:
        try:
            window_manager.event_timer_remove(timer)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
    for key in ("candidate", "rollback"):
        mesh = state.get(key)
        state[key] = None
        if mesh is not None:
            try:
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                pass
    if state.get("progress_started") and window_manager is not None:
        try:
            window_manager.progress_end()
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pass
    state["progress_started"] = False
    workspace = state.get("workspace")
    try:
        if workspace is not None:
            workspace.status_text_set(None)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass
    window = state.get("window")
    try:
        if window is not None and window.screen is not None:
            for area in window.screen.areas:
                area.tag_redraw()
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        pass


def local_remesh_request_cancel(reason="lifecycle"):
    """Cancel staged Local Remesh work before load, undo, redo, or teardown."""
    cancelled = 0
    for entry in tuple(_runtime.modal_registry.values()):
        if str(entry.get("kind", "")) != "Local Remesh":
            continue
        token = entry.get("token")
        _lifecycle.modal_request_cancel(
            token=token,
            reason=str(reason),
        )
        state = entry.get("state")
        if isinstance(state, dict):
            state["cancel_reason"] = str(reason)
            state["final_passes"] = int(
                (state.get("patch_state") or {}).get("passes", 0)
            )
            _local_remesh_dispose_staged(state)
            _local_remesh_terminal_metrics(state, "CANCELLED")
            if _runtime.local_remesh_modal_state is state:
                _runtime.local_remesh_modal_state = None
        _lifecycle.modal_schedule_terminal_event(
            token=token,
            state=state,
        )
        cancelled += 1
    return cancelled


def _local_remesh_terminal_metrics(state, status, error=None):
    _runtime.local_remesh_last_metrics = {
        "status": str(status),
        "error": str(error) if error is not None else "",
        "cancel_reason": str(state.get("cancel_reason", "")),
        "target": float(state.get("target", 0.0)),
        "output_triangles": int(state.get("output_triangles", 0)),
        "refinement_passes": int(state.get("final_passes", 0)),
        "phase_events": tuple(state.get("phase_events", ())),
        "phase_times_s": dict(state.get("phase_times", {})),
        "timer_events_seen": int(state.get("timer_events_seen", 0)),
        "timer_ticks_processed": int(state.get("timer_ticks_processed", 0)),
        "unyielding_calls_s": tuple(state.get("unyielding_calls", ())),
        "max_unyielding_s": float(state.get("max_unyielding_s", 0.0)),
        "max_unyielding_stage": str(state.get("max_unyielding_stage", "")),
        "elapsed_s": max(0.0, time.perf_counter() - state.get("started_at", time.perf_counter())),
    }


class MESH_OT_mesh_focus_local_remesh(bpy.types.Operator):
    """Add sculptable triangle density only to selected repair faces."""

    bl_idname = "mesh.mesh_focus_local_remesh"
    bl_label = "Local Remesh"
    bl_description = "Densify selected repair faces; unavailable on meshes with Shape Keys"
    bl_options = {"REGISTER", "UNDO"}

    target_edge_length: FloatProperty(
        name="Target Edge Length",
        description="Zero estimates the median edge length from the surrounding mesh",
        default=0.0,
        min=0.0,
        options={"SKIP_SAVE"},
        precision=6,
        unit="LENGTH",
    )
    max_patch_faces: IntProperty(
        name="Maximum Patch Faces",
        description="Abort safely before an unexpectedly large patch is committed",
        default=200000,
        min=100,
        max=2000000,
    )

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "edit_object", None)
        return context.mode == "EDIT_MESH" and obj is not None and obj.type == "MESH"

    def invoke(self, context, event):
        if not self.poll(context):
            return {"CANCELLED"}
        if any(
            str(entry.get("kind", "")) == "Local Remesh"
            for entry in tuple(_runtime.modal_registry.values())
        ):
            self.report({"WARNING"}, "A Local Remesh operation is already active")
            return {"CANCELLED"}
        obj = context.edit_object
        mesh = obj.data
        if mesh.shape_keys is not None:
            self.report(
                {"ERROR"},
                "Local Remesh does not support meshes with Shape Keys",
            )
            return {"CANCELLED"}
        live_bm = bmesh.from_edit_mesh(mesh)
        state = {
            "operator": self,
            "window": context.window,
            "window_manager": context.window_manager,
            "workspace": context.workspace,
            "scene_pointer": _pointer(context.scene),
            "window_pointer": _pointer(context.window),
            "object_pointer": _pointer(obj),
            "mesh_pointer": _pointer(mesh),
            "live_bm": live_bm,
            "source_counts": (
                len(live_bm.verts),
                len(live_bm.edges),
                len(live_bm.faces),
            ),
            "staged": None,
            "patch_state": None,
            "candidate": None,
            "rollback": None,
            "timer": None,
            "target": 0.0,
            "output_triangles": 0,
            "phase": "prepare",
            "phase_label": "Preparing",
            "phase_events": [],
            "progress_samples": [],
            "phase_times": {},
            "timer_interval_s": 0.03,
            "next_tick_at": 0.0,
            "timer_events_seen": 0,
            "timer_ticks_processed": 0,
            "unyielding_calls": [],
            "max_unyielding_s": 0.0,
            "max_unyielding_stage": "",
            "started_at": time.perf_counter(),
            "progress_started": False,
            "resources_disposed": False,
            "announce_pending": True,
        }
        token = None
        try:
            context.window_manager.progress_begin(0.0, 1.0)
            state["progress_started"] = True
            _local_remesh_publish_status(state, "prepare", "Preparing", 0.02)
            _local_remesh_schedule_next_tick(state)
            state["timer"] = context.window_manager.event_timer_add(
                state["timer_interval_s"],
                window=context.window,
            )
            token = _lifecycle.modal_register(
                self,
                "Local Remesh",
                state=state,
            )
            state["modal_token"] = token
            _runtime.local_remesh_modal_state = state
            context.window_manager.modal_handler_add(self)
        except Exception as exc:
            _local_remesh_dispose_staged(state)
            if _runtime.local_remesh_modal_state is state:
                _runtime.local_remesh_modal_state = None
            if token is not None:
                _lifecycle.modal_terminal(
                    token=token,
                    status="CANCELLED",
                )
            self.report({"ERROR"}, f"Local Remesh could not start: {exc}")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _finish_modal(self, state, status, error=None):
        state["final_passes"] = int(
            (state.get("patch_state") or {}).get("passes", 0)
        )
        _local_remesh_dispose_staged(state)
        _local_remesh_terminal_metrics(state, status, error)
        if _runtime.local_remesh_modal_state is state:
            _runtime.local_remesh_modal_state = None
        _lifecycle.modal_terminal(
            token=state.get("modal_token"),
            status=status,
        )
        if status == "FINISHED":
            if int(state.get("final_passes", 0)) == 0:
                message = (
                    "Local Remesh: no subdivision needed at target density; "
                    "%s triangles, target %.6g"
                    % (state.get("output_triangles", 0), state.get("target", 0.0))
                )
            else:
                message = (
                    "Local Remesh: %s triangles, target %.6g"
                    % (state.get("output_triangles", 0), state.get("target", 0.0))
                )
            self.report(
                {"INFO"},
                message,
            )
        elif error is not None:
            self.report({"ERROR"}, f"Local Remesh: {error}")
        return {status}

    def _advance_modal(self, state, context):
        stage = state["phase"]
        staged = state.get("staged")
        patch_state = state.get("patch_state")
        if stage == "prepare":
            def prepare():
                state["staged"] = state["live_bm"].copy()
                state["patch_state"] = _prepare_patch_state(
                    state["staged"],
                    self.target_edge_length,
                    self.max_patch_faces,
                )
            _local_remesh_measure(state, "prepare", prepare)
            patch_state = state["patch_state"]
            state["source_signature"] = patch_state["source_signature"]
            state["target"] = patch_state["target"]
            state["selected_faces"] = len(patch_state["selected"])
            state["phase"] = "triangulate"
            _local_remesh_publish_status(state, "triangulate", "Triangulating", 0.20)
            return None
        if stage == "triangulate":
            _local_remesh_measure(
                state,
                "triangulate",
                lambda: _triangulate_patch(patch_state),
            )
            state["phase"] = "refine"
            _local_remesh_publish_status(state, "refine", "Subdivide pass 1/16", 0.28)
            return None
        if stage == "refine":
            done = _local_remesh_measure(
                state,
                "refine-pass-%d" % (patch_state["passes"] + 1),
                lambda: _refine_patch_step(patch_state),
            )
            if done:
                state["phase"] = "relax"
                _local_remesh_publish_status(state, "relax", "Relax and verify", 0.76)
            else:
                pass_number = int(patch_state["passes"]) + 1
                progress = min(0.72, 0.28 + 0.44 * pass_number / 16.0)
                _local_remesh_publish_status(
                    state,
                    "refine",
                    "Subdivide pass %d/16" % pass_number,
                    progress,
                )
            return None
        if stage == "relax":
            target, count = _local_remesh_measure(
                state,
                "relax-and-verify",
                lambda: _relax_and_verify(patch_state),
            )
            state["target"] = target
            state["output_triangles"] = count
            state["phase"] = "candidate"
            _local_remesh_publish_status(state, "candidate", "Building candidate", 0.88)
            return None
        if stage == "candidate":
            def build_candidate():
                candidate = bpy.data.meshes.new("MFO Local Remesh Candidate")
                state["candidate"] = candidate
                staged.to_mesh(candidate)
            _local_remesh_measure(state, "candidate-mesh", build_candidate)
            state["phase"] = "apply"
            _local_remesh_publish_status(state, "apply", "Applying to Edit Mesh", 0.96)
            return None
        if stage == "apply":
            _local_remesh_measure(
                state,
                "apply-and-rollback-guard",
                lambda: self._apply_candidate(state, context),
            )
            return self._finish_modal(state, "FINISHED")
        raise RuntimeError("Unknown Local Remesh stage: %s" % stage)

    def _apply_candidate(self, state, context):
        if not _local_remesh_context_matches(state, context, verify_source=True):
            raise RuntimeError("The edit mesh or selected patch changed during preparation")
        mesh = context.edit_object.data
        live_bm = bmesh.from_edit_mesh(mesh)
        rollback = bpy.data.meshes.new("MFO Local Remesh Rollback")
        state["rollback"] = rollback
        # The rollback image is taken from the live BMesh immediately before
        # Apply, so a failed candidate write cannot restore an older snapshot.
        live_bm.to_mesh(rollback)
        try:
            live_bm.clear()
            live_bm.from_mesh(state["candidate"])
            bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
        except Exception:
            live_bm.clear()
            live_bm.from_mesh(rollback)
            bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
            raise

    def modal(self, context, event):
        entry = _lifecycle.modal_entry(operator=self)
        if entry is None:
            orphaned_state = _runtime.local_remesh_modal_state
            if (
                isinstance(orphaned_state, dict)
                and orphaned_state.get("operator") is self
            ):
                _local_remesh_dispose_staged(orphaned_state)
                _runtime.local_remesh_modal_state = None
            return {"CANCELLED"}
        state = entry.get("state")
        if not isinstance(state, dict):
            return {"CANCELLED"}
        if entry.get("cancel_requested") or entry.get("teardown_requested"):
            reason = state.get("cancel_reason") or entry.get("cancel_reason") or "cancelled"
            state["cancel_reason"] = str(reason)
            return self._finish_modal(state, "CANCELLED")
        if event.type == "ESC" and event.value == "PRESS":
            state["cancel_reason"] = "cancelled by user"
            return self._finish_modal(state, "CANCELLED")
        if not _local_remesh_context_matches(state, context):
            return self._finish_modal(state, "CANCELLED", "Edit context changed")
        if getattr(event, "type", None) == "TIMER":
            state["timer_events_seen"] = int(state.get("timer_events_seen", 0)) + 1
        if not _local_remesh_timer_ready(state, event):
            # The staged snapshot owns this interaction. Consume user input so
            # no later edit can be overwritten by the staged result. Blender's
            # Event has no timer attribute, so TIMER cadence is gated by a
            # monotonic deadline instead of comparing timer wrappers.
            return {"RUNNING_MODAL"}
        state["timer_ticks_processed"] = int(
            state.get("timer_ticks_processed", 0)
        ) + 1
        if state.get("announce_pending"):
            state["announce_pending"] = False
            _local_remesh_schedule_next_tick(state)
            return {"RUNNING_MODAL"}
        try:
            result = self._advance_modal(state, context)
        except Exception as exc:
            return self._finish_modal(state, "CANCELLED", str(exc))
        finally:
            if not state.get("resources_disposed"):
                _local_remesh_schedule_next_tick(state)
        return result if result is not None else {"RUNNING_MODAL"}

    def execute(self, context):
        """Preserve deterministic operator execution for explicit EXEC calls."""
        if not self.poll(context):
            return {"CANCELLED"}
        mesh = context.edit_object.data
        if mesh.shape_keys is not None:
            self.report(
                {"ERROR"},
                "Local Remesh does not support meshes with Shape Keys",
            )
            return {"CANCELLED"}
        edit_bm = bmesh.from_edit_mesh(mesh)
        staged = edit_bm.copy()
        candidate = None
        backup = None
        pass_counter = []
        try:
            target, face_count = _build_patch(
                staged,
                self.target_edge_length,
                self.max_patch_faces,
                pass_counter,
            )
            candidate = bpy.data.meshes.new("MFO Local Remesh Candidate")
            staged.to_mesh(candidate)
            backup = bpy.data.meshes.new("MFO Local Remesh Rollback")
            edit_bm.to_mesh(backup)
            try:
                edit_bm.clear()
                edit_bm.from_mesh(candidate)
                bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
            except Exception:
                edit_bm.clear()
                edit_bm.from_mesh(backup)
                bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
                raise
        except Exception as exc:
            self.report({"ERROR"}, f"Local Remesh: {exc}")
            return {"CANCELLED"}
        finally:
            staged.free()
            if candidate is not None:
                bpy.data.meshes.remove(candidate)
            if backup is not None:
                bpy.data.meshes.remove(backup)
        if pass_counter and pass_counter[0] == 0:
            message = (
                "Local Remesh: no subdivision needed at target density; "
                f"{face_count} triangles, target {target:.6g}"
            )
        else:
            message = f"Local Remesh: {face_count} triangles, target {target:.6g}"
        self.report({"INFO"}, message)
        return {"FINISHED"}
