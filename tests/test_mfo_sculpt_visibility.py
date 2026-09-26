"""Focused Blender regression for normal MFO on hidden Sculpt faces.

The package is imported under a temporary name; no scene or add-on state is edited.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

from mathutils import Matrix, Vector


def _load_source():
    package_dir = Path(__file__).resolve().parents[1] / "mesh_focus_orbit"
    spec = importlib.util.spec_from_file_location(
        "mfo_sculpt_visibility_test_source",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeSculptObject:
    type = "MESH"
    mode = "SCULPT"
    matrix_world = Matrix.Identity(4)

    def __init__(self, second_visible=True):
        self.data = SimpleNamespace(
            polygons=[SimpleNamespace(hide=True), SimpleNamespace(hide=not second_visible)]
        )
        self.ray_origins = []

    def ray_cast(self, origin, _direction):
        self.ray_origins.append(tuple(origin))
        if origin.z <= 1.0:
            return True, Vector((0.0, 0.0, 1.0)), Vector((0.0, 0.0, 1.0)), 0
        if origin.z <= 2.0:
            return True, Vector((0.0, 0.0, 2.0)), Vector((0.0, 0.0, 1.0)), 1
        return False, Vector(), Vector(), -1


def run():
    module = _load_source()
    foundation = module.foundation
    origin = Vector((0.0, 0.0, 0.0))
    direction = Vector((0.0, 0.0, 1.0))

    obj = _FakeSculptObject(second_visible=True)
    hit = foundation._raycast_sculpt_visible_object(obj, origin, direction)
    assert hit is not None
    assert tuple(hit[0]) == (0.0, 0.0, 2.0)
    assert abs(hit[1] - 2.0) < 1.0e-6
    assert len(obj.ray_origins) == 2
    assert obj.ray_origins[1][2] > 1.0

    hidden_only = _FakeSculptObject(second_visible=False)
    assert foundation._raycast_sculpt_visible_object(
        hidden_only, origin, direction
    ) is None

    # The dispatcher must use the source-mesh visibility path only for the
    # active Sculpt object. Other visible mesh objects retain evaluated picking.
    sculpt = _FakeSculptObject()
    other = SimpleNamespace(type="MESH", mode="OBJECT")
    context = SimpleNamespace(
        mode="SCULPT",
        active_object=sculpt,
        view_layer=SimpleNamespace(objects=(sculpt, other)),
        evaluated_depsgraph_get=lambda: object(),
    )
    old_visible = foundation._object_is_visible
    old_sculpt_cast = foundation._raycast_sculpt_visible_object
    old_object_cast = foundation._raycast_object
    calls = []
    try:
        foundation._object_is_visible = lambda _context, _obj: True
        foundation._raycast_sculpt_visible_object = (
            lambda _obj, _origin, _direction: calls.append("sculpt")
            or (Vector((0.0, 0.0, 2.0)), 2.0)
        )
        foundation._raycast_object = (
            lambda _obj, _depsgraph, _origin, _direction: calls.append("other")
            or (Vector((0.0, 0.0, 3.0)), 3.0)
        )
        nearest = foundation._find_mesh_hit(context, (origin, direction))
        assert tuple(nearest[0]) == (0.0, 0.0, 2.0)
        assert calls == ["sculpt", "other"]
    finally:
        foundation._object_is_visible = old_visible
        foundation._raycast_sculpt_visible_object = old_sculpt_cast
        foundation._raycast_object = old_object_cast

    return {"passed": 3, "source_version": module.bl_info["version"]}


if __name__ == "__main__":
    print(run())
