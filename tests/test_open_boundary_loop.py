"""非表示面でできた開口の縁を選ぶ、Blender不要の回帰検証。"""

import ast
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "mesh_focus_orbit" / "registration.py"
INIT = ROOT / "mesh_focus_orbit" / "__init__.py"


def _load_helper(clock=time):
    source = REGISTRATION.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(REGISTRATION))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_open_boundary_loop_from_seed")
    namespace = {"time": clock}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(REGISTRATION), "exec"), namespace)
    return namespace["_open_boundary_loop_from_seed"]


class _Vert:
    def __init__(self):
        self.link_edges = []


class _Edge:
    def __init__(self, first, second, *, faces=1, hidden=False, face_hidden=None):
        self.verts = (first, second)
        self.link_faces = [SimpleNamespace(hide=value) for value in
                           (face_hidden if face_hidden is not None else [False] * faces)]
        self.hide = hidden
        first.link_edges.append(self)
        second.link_edges.append(self)


def _cycle(size, *, hidden_face=False):
    verts = [_Vert() for _ in range(size)]
    return [_Edge(verts[i], verts[(i + 1) % size],
                  face_hidden=[False, True] if hidden_face else [False])
            for i in range(size)]


def _grid():
    # 長方形の隠れ領域と、隣にある別の隠れ領域。面は全て残っている。
    vertices = {(x, y): _Vert() for y in range(5) for x in range(9)}
    edges = {}
    for y in range(4):
        for x in range(8):
            face = SimpleNamespace(hide=(y == 2 and 1 <= x <= 3)
                                   or (x == 6 and y in (1, 2)))
            corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
            for a, b in zip(corners, corners[1:] + corners[:1]):
                key = frozenset((a, b))
                if key not in edges:
                    edges[key] = _Edge(vertices[a], vertices[b], faces=0)
                edges[key].link_faces.append(face)
    # 期待値は生成した長方形の座標から指定し、探索関数を使って作らない。
    pairs = [((x, y), (x + 1, y)) for y in (2, 3) for x in range(1, 4)]
    pairs += [((x, 2), (x, 3)) for x in (1, 4)]
    expected = {edges[frozenset(pair)] for pair in pairs}
    return edges, expected


def test_patch_version():
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    info = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == "bl_info"
                                         for t in n.targets)))
    assert info["version"] == (3, 4, 7)


def test_real_boundary_remains_supported():
    first, second = _cycle(4), _cycle(5)
    selected, reason = _load_helper()(first[0])
    assert not reason and selected == set(first)
    assert not selected.intersection(second)


def test_hidden_face_rim_beats_land_shortcuts():
    edges, expected = _grid()
    seed = edges[frozenset(((1, 2), (2, 2)))]
    assert len(seed.link_faces) == 2
    selected, reason = _load_helper()(seed)
    assert not reason and selected == expected
    assert len(selected) == 8  # 別の開口や陸側の辺を含めない。


def test_repeated_calls_and_visibility_changes():
    edges, expected = _grid()
    helper = _load_helper()
    seed = edges[frozenset(((1, 2), (2, 2)))]
    for _ in range(3):
        assert helper(seed)[0] == expected
    hidden = next(f for f in seed.link_faces if f.hide)
    hidden.hide = False
    assert helper(seed)[0] is None
    hidden.hide = True
    assert helper(seed)[0] == expected


def test_surface_loose_hidden_and_nonmanifold_seeds_rejected():
    for edge in (
        _Edge(_Vert(), _Vert(), faces=2),
        _Edge(_Vert(), _Vert(), faces=0),
        _Edge(_Vert(), _Vert(), hidden=True),
        _Edge(_Vert(), _Vert(), face_hidden=[True, True]),
        _Edge(_Vert(), _Vert(), face_hidden=[False, True, True]),
    ):
        selected, reason = _load_helper()(edge)
        assert selected is None and "opening edge" in reason
    assert _load_helper()(None)[0] is None


def test_branched_and_broken_boundaries_rejected():
    edges = _cycle(4, hidden_face=True)
    _Edge(edges[0].verts[0], _Vert(), face_hidden=[False, True])
    assert "branched" in _load_helper()(edges[0])[1]
    edges = _cycle(4, hidden_face=True)
    edges[2].hide = True
    assert _load_helper()(edges[0])[0] is None


def test_touching_openings_do_not_merge():
    first = _cycle(4, hidden_face=True)
    v = first[0].verts[0]
    a, b, c = _Vert(), _Vert(), _Vert()
    second = [_Edge(v, a, face_hidden=[False, True]),
              _Edge(a, b, face_hidden=[False, True]),
              _Edge(b, c, face_hidden=[False, True]),
              _Edge(c, v, face_hidden=[False, True])]
    helper = _load_helper()
    for seed in first:
        selected, reason = helper(seed)
        assert not reason and selected == set(first)
    for seed in second:
        selected, reason = helper(seed)
        assert not reason and selected == set(second)


def test_multiple_cycles_sharing_a_path_are_ambiguous():
    a, b = _Vert(), _Vert()
    paths = []
    for _ in range(3):
        mid = _Vert()
        paths.extend((_Edge(a, mid, face_hidden=[False, True]),
                      _Edge(mid, b, face_hidden=[False, True])))
    selected, reason = _load_helper()(paths[0])
    assert selected is None and "branched" in reason


def test_long_boundary_does_not_depend_on_python_recursion():
    edges = _cycle(2000, hidden_face=True)
    selected, reason = _load_helper()(edges[0], time_limit=2.0)
    assert not reason and selected == set(edges)


def test_search_limits_return_no_partial_selection():
    edges = _cycle(4, hidden_face=True)
    helper = _load_helper()
    assert helper(edges[0], max_edges=4)[0] == set(edges)
    selected, reason = helper(edges[0], max_edges=3)
    assert selected is None and "edge limit" in reason
    ticks = iter((10.0, 11.0))
    clock = SimpleNamespace(perf_counter=lambda: next(ticks))
    selected, reason = _load_helper(clock)(edges[0], time_limit=0.5)
    assert selected is None and "time limit" in reason


def test_shortcut_uses_exact_modifiers():
    tree = ast.parse(REGISTRATION.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and n.args and isinstance(n.args[0], ast.Name)
             and n.args[0].id == "OPEN_BOUNDARY_LOOP_OPERATOR_ID"
             and len(n.args) >= 3 and isinstance(n.args[1], ast.Constant)]
    assert len(calls) == 1
    call = calls[0]
    assert [ast.literal_eval(n) for n in call.args[1:3]] == ["L", "PRESS"]
    kw = {n.arg: ast.literal_eval(n.value) for n in call.keywords}
    assert kw == {"any": False, "shift": True, "ctrl": False, "alt": True}


def run():
    tests = [value for name, value in globals().items()
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    return {"passed": len(tests), "blender_imported": False}


if __name__ == "__main__":
    print(run())
