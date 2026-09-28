"""Local triangle density for a manually repaired patch in Edit Mode."""

import math
import statistics

import bmesh
import bpy
from bpy.props import FloatProperty, IntProperty


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
    samples = [edge.calc_length() for edge in neighbors if edge.calc_length() > 1.0e-12]
    if not samples:
        samples = [
            edge.calc_length()
            for face in selected
            for edge in face.edges
            if edge.calc_length() > 1.0e-12
        ]
    if not samples:
        raise ValueError("Could not estimate a target edge length")
    return statistics.median(samples)


def _build_patch(bm, target, max_faces):
    # Create temporary layers before taking BMesh element references: creating
    # a layer can invalidate Python wrappers for existing elements.
    region_layer = bm.faces.layers.int.new("__mfo_local_remesh_region")
    original_id_layer = bm.verts.layers.int.new("__mfo_local_remesh_original_id")
    original_coordinates = {}
    for original_id, vert in enumerate(bm.verts, 1):
        vert[original_id_layer] = original_id
        original_coordinates[original_id] = vert.co.copy()
    selected = {face for face in bm.faces if face.select and not face.hide}
    if not selected:
        raise ValueError("Select the repaired faces first")
    if max_faces < 1:
        raise ValueError("Maximum patch face count must be positive")
    triangulated_face_estimate = sum(len(face.verts) - 2 for face in selected)
    if triangulated_face_estimate > max_faces:
        raise ValueError("Patch exceeds the face limit; increase the target edge length")
    for face in selected:
        face[region_layer] = 1

    def vertex_signature(vert):
        original_id = vert[original_id_layer]
        original_co = original_coordinates.get(original_id)
        if original_co is not None and (vert.co - original_co).length <= 1.0e-10:
            return ("original", original_id)
        return ("new", tuple(round(float(value), 10) for value in vert.co))

    def face_signature(face):
        vertices = tuple(vertex_signature(vert) for vert in face.verts)
        cycles = [vertices[index:] + vertices[:index] for index in range(len(vertices))]
        return min(cycles), face.material_index

    def unselected_signatures():
        return sorted(
            face_signature(face)
            for face in bm.faces if face[region_layer] != 1
        )

    unselected_before = unselected_signatures()
    boundary_edges = {
        edge
        for face in selected
        for edge in face.edges
        if any(link not in selected for link in edge.link_faces)
        or sum(link in selected for link in edge.link_faces) == 1
    }
    if target <= 0.0:
        target = _target_length(bm, selected, boundary_edges)
    if not math.isfinite(target) or target <= 1.0e-12:
        raise ValueError("Target edge length must be positive")

    bmesh.ops.triangulate(bm, faces=list(selected), quad_method="BEAUTY", ngon_method="BEAUTY")
    area_limit = 0.5 * target * target
    for _ in range(16):
        region = {face for face in bm.faces if face[region_layer] == 1}
        if len(region) > max_faces:
            raise ValueError("Patch exceeds the face limit; increase the target edge length")
        long_edges = {
            edge
            for face in region
            for edge in face.edges
            if len(edge.link_faces) == 2
            and all(link[region_layer] == 1 for link in edge.link_faces)
            and edge.calc_length() > target * 1.2
        }
        large_faces = [face for face in region if face.calc_area() > area_limit]
        if not long_edges and not large_faces:
            break
        if len(region) + len(large_faces) * 2 + len(long_edges) * 2 > max_faces:
            raise ValueError("Patch exceeds the face limit; increase the target edge length")
        if long_edges:
            bmesh.ops.subdivide_edges(bm, edges=list(long_edges), cuts=1, use_grid_fill=False)
        region = {face for face in bm.faces if face[region_layer] == 1}
        large_faces = [face for face in region if face.calc_area() > area_limit]
        if large_faces:
            bmesh.ops.poke(bm, faces=large_faces)
    else:
        raise ValueError("Patch did not reach the requested density within 16 passes")

    region = {face for face in bm.faces if face[region_layer] == 1}
    bmesh.ops.triangulate(bm, faces=list(region), quad_method="BEAUTY", ngon_method="BEAUTY")
    region = {face for face in bm.faces if face[region_layer] == 1}
    if len(region) > max_faces:
        raise ValueError("Patch exceeds the face limit; increase the target edge length")

    # A small single-pass relax affects only newly created interior vertices.
    new_verts = []
    seen_original_ids = set()
    for vert in bm.verts:
        original_id = vert[original_id_layer]
        original_co = original_coordinates.get(original_id)
        if original_co is not None and (vert.co - original_co).length <= 1.0e-10:
            if original_id in seen_original_ids:
                raise ValueError("An existing vertex would move")
            seen_original_ids.add(original_id)
        else:
            new_verts.append(vert)
    updates = {}
    for vert in new_verts:
        if any(
            len(edge.link_faces) != 2
            or any(face[region_layer] != 1 for face in edge.link_faces)
            for edge in vert.link_edges
        ):
            continue
        neighbors = [edge.other_vert(vert).co for edge in vert.link_edges]
        if neighbors:
            mean = sum(neighbors, vert.co.copy() * 0.0) / len(neighbors)
            updates[vert] = vert.co.lerp(mean, 0.12)
    for vert, co in updates.items():
        vert.co = co

    if unselected_signatures() != unselected_before:
        raise ValueError("Unselected geometry would change")
    if seen_original_ids != original_coordinates.keys():
        raise ValueError("An existing vertex would move")
    if any(len(face.verts) != 3 for face in region):
        raise ValueError("Patch triangulation failed")
    for face in bm.faces:
        face.select_set(face[region_layer] == 1)
    count = len(region)
    bm.faces.layers.int.remove(region_layer)
    bm.verts.layers.int.remove(original_id_layer)
    return target, count


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

    def execute(self, context):
        if not self.poll(context):
            return {"CANCELLED"}
        mesh = context.edit_object.data
        if mesh.shape_keys is not None:
            self.report({"ERROR"}, "Local Remesh does not support meshes with Shape Keys")
            return {"CANCELLED"}
        edit_bm = bmesh.from_edit_mesh(mesh)
        staged = edit_bm.copy()
        candidate = None
        backup = None
        try:
            target, face_count = _build_patch(
                staged, self.target_edge_length, self.max_patch_faces
            )
            candidate = bpy.data.meshes.new("MFO Local Remesh Candidate")
            staged.to_mesh(candidate)
            # Keep a rollback copy until the edit mesh has been updated.
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
        self.report({"INFO"}, f"Local Remesh: {face_count} triangles, target {target:.6g}")
        return {"FINISHED"}
