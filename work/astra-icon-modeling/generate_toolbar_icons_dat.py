"""Export the Astra toolbar scenes to Blender's VCO triangle-icon format.

Run from Blender without opening the user's UI::

    blender --background mfo-toolbar-icons-astra.blend \
        --python generate_toolbar_icons_dat.py -- \
        --output-dir ../../assets/mfo-toolbar-icons-astra

The source blend remains read-only.  Each scene is projected through its own
orthographic camera, so the .dat geometry follows the supplied rendered icon
orientation rather than assuming a world axis.  The file format is Blender's
official ``VCO`` format: an 8-byte header, all quantized XY triangle vertices,
then one RGBA byte quadruplet per vertex.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import struct
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Color


SCENES = {
    "01 | MFO Focus Surface": "mfo-focus-surface",
    "02 | Face Set MFO": "mfo-face-set",
    "03 | Smart Face Set Fill": "mfo-smart-face-set-fill",
    "04 | Guided Ridge": "mfo-guided-ridge",
    "05 | Tube Shape": "mfo-tube-shape",
}


def _material_color(obj, material_index):
    color = (1.0, 1.0, 1.0, 1.0)
    slots = getattr(obj, "material_slots", ())
    if 0 <= int(material_index) < len(slots):
        material = slots[int(material_index)].material
        if material is not None:
            color = tuple(float(value) for value in material.diffuse_color[:])
            node_tree = getattr(material, "node_tree", None)
            if node_tree is not None:
                rgb = next(
                    (node.outputs[0].default_value[:] for node in node_tree.nodes
                     if node.type == "RGB" and node.outputs),
                    None,
                )
                if rgb is not None:
                    color = tuple(float(value) for value in rgb)
    if len(color) < 4:
        color = tuple(color[:3]) + (1.0,)
    return tuple(max(0.0, min(1.0, value)) for value in color[:4])


def _linear_to_srgb(value):
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def _color_bytes(color):
    return tuple(
        max(0, min(255, round(_linear_to_srgb(value) * 255.0)))
        for value in color[:3]
    ) + (max(0, min(255, round(color[3] * 255.0))),)


def _project(scene, camera, world_point):
    projected = world_to_camera_view(scene, camera, world_point)
    return (float(projected.x * 2.0 - 1.0), float(projected.y * 2.0 - 1.0))


@contextlib.contextmanager
def _scene_evaluation_context(scene):
    """Yield the depsgraph for ``scene`` and restore the caller's context.

    ``evaluated_depsgraph_get()`` follows the active window scene, not an
    arbitrary Scene datablock.  The icon source has one scene per feature, so
    evaluating every object under the current scene would silently skip
    modifiers in the other four scenes.  A temporary context override works in
    both UI and background Blender sessions and restores the original scene
    and view layer when the ``with`` block exits.
    """
    context = bpy.context
    view_layer_name = getattr(getattr(context, "view_layer", None), "name", "")
    view_layer = scene.view_layers.get(view_layer_name) or scene.view_layers[0]
    with context.temp_override(scene=scene, view_layer=view_layer):
        yield context.evaluated_depsgraph_get()


def _triangles_for_scene(scene):
    camera = scene.camera
    if camera is None:
        raise RuntimeError(f"scene has no camera: {scene.name}")
    with _scene_evaluation_context(scene) as depsgraph:
        camera = camera.evaluated_get(depsgraph)
        triangles = []
        camera_inverse = camera.matrix_world.inverted_safe()
        for obj in sorted(scene.objects, key=lambda value: value.name):
            if obj.type not in {"MESH", "CURVE"} or obj.hide_render:
                continue
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            try:
                if mesh is None:
                    continue
                mesh.transform(evaluated.matrix_world)
                vertices = mesh.vertices
                for polygon in mesh.polygons:
                    if len(polygon.vertices) < 3:
                        continue
                    color = _color_bytes(_material_color(obj, polygon.material_index))
                    first = int(polygon.vertices[0])
                    for offset in range(1, len(polygon.vertices) - 1):
                        indices = (first, int(polygon.vertices[offset]), int(polygon.vertices[offset + 1]))
                        projected = [_project(scene, camera, vertices[index].co) for index in indices]
                        area = (
                            (projected[0][0] - projected[1][0]) * (projected[1][1] - projected[2][1])
                            + (projected[0][1] - projected[1][1]) * (projected[2][0] - projected[1][0])
                        )
                        if abs(area) <= 1.0e-8:
                            continue
                        if area < 0.0:
                            projected[1], projected[2] = projected[2], projected[1]
                        depth = sum((camera_inverse @ vertices[index].co).z for index in indices) / 3.0
                        triangles.append((float(depth), projected, (color, color, color)))
            finally:
                evaluated.to_mesh_clear()
        # Farther camera-space Z is more negative; writing far-to-near preserves
        # the same painter ordering as Blender's official exporter.
        triangles.sort(key=lambda item: int(item[0] * 100.0))
        return triangles


def _mesh_triangle_count(mesh):
    return sum(max(0, len(polygon.vertices) - 2) for polygon in mesh.polygons)


def _modifier_evaluation_report(scene):
    """Report modifier evaluation under the scene-specific dependency graph."""
    records = []
    with _scene_evaluation_context(scene) as depsgraph:
        evaluated_scene = getattr(getattr(depsgraph, "scene_eval", None), "name", None)
        view_layer = getattr(getattr(depsgraph, "view_layer", None), "name", None)
        for obj in sorted(scene.objects, key=lambda value: value.name):
            if obj.type != "MESH" or not len(obj.modifiers):
                continue
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            try:
                records.append({
                    "object": obj.name,
                    "modifiers": tuple(modifier.type for modifier in obj.modifiers),
                    "source_vertices": len(obj.data.vertices),
                    "source_polygons": len(obj.data.polygons),
                    "source_triangles": _mesh_triangle_count(obj.data),
                    "evaluated_vertices": len(mesh.vertices) if mesh is not None else 0,
                    "evaluated_polygons": len(mesh.polygons) if mesh is not None else 0,
                    "evaluated_triangles": _mesh_triangle_count(mesh) if mesh is not None else 0,
                    "changed": bool(mesh is not None and (
                        len(mesh.vertices) != len(obj.data.vertices)
                        or len(mesh.polygons) != len(obj.data.polygons)
                        or _mesh_triangle_count(mesh) != _mesh_triangle_count(obj.data)
                    )),
                })
            finally:
                evaluated.to_mesh_clear()
    return {
        "scene": scene.name,
        "evaluated_scene": evaluated_scene,
        "view_layer": view_layer,
        "modifier_objects": records,
    }


def _write_icon(path, triangles):
    coords = bytearray()
    colors = bytearray()
    for _depth, points, tri_colors in triangles:
        encoded = []
        for x, y in points:
            encoded.append((
                max(0, min(255, round((x + 1.0) * 0.5 * 255.0))),
                max(0, min(255, round((y + 1.0) * 0.5 * 255.0))),
            ))
        area = (
            (encoded[0][0] - encoded[1][0]) * (encoded[1][1] - encoded[2][1])
            + (encoded[0][1] - encoded[1][1]) * (encoded[2][0] - encoded[1][0])
        )
        if area <= 0:
            continue
        for pair in encoded:
            coords.extend(bytes(pair))
        for color in tri_colors:
            colors.extend(bytes(color))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"VCO\x00\xff\xff\x00\x00" + bytes(coords) + bytes(colors))
    if len(coords) % 6 or len(colors) != len(coords) * 2:
        raise RuntimeError(f"invalid VCO payload: {path}")
    return len(coords) // 6


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    output_dir = os.path.abspath(args.output_dir)
    report = {}
    for scene_name, stem in SCENES.items():
        scene = bpy.data.scenes.get(scene_name)
        if scene is None:
            raise RuntimeError(f"missing scene: {scene_name}")
        triangles = _triangles_for_scene(scene)
        count = _write_icon(os.path.join(output_dir, stem + ".dat"), triangles)
        report[stem] = {
            "triangles": count,
            "path": os.path.join(output_dir, stem + ".dat"),
            "evaluation": _modifier_evaluation_report(scene),
        }
    if args.report:
        report_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(report)
    return report


if __name__ == "__main__":
    if "--" in sys.argv:
        sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1:]
    main()
