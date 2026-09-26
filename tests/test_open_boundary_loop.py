"""非表示面でできた開口の縁を選ぶ、Blender不要の回帰検証。"""

import ast
import math
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "mesh_focus_orbit" / "registration.py"
INIT = ROOT / "mesh_focus_orbit" / "__init__.py"
PERFORMANCE_RESULTS = {}


def _load_helper(clock=time):
    source = REGISTRATION.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(REGISTRATION))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_open_boundary_loop_from_seed")
    namespace = {"time": clock}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(REGISTRATION), "exec"), namespace)
    return namespace["_open_boundary_loop_from_seed"]


def _load_arc_helpers(clock=time):
    source = REGISTRATION.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(REGISTRATION))
    names = {
        "_open_boundary_loop_from_seed",
        "_open_boundary_loop_clear_repeat_token",
        "_open_boundary_loop_edge_order_key",
        "_open_boundary_loop_ordered_cycle",
        "_open_boundary_loop_arc_options",
        "_open_boundary_loop_identity",
        "_open_boundary_loop_state_signature",
        "_open_boundary_loop_repeat_edges",
        "_open_boundary_loop_resolve_selection",
    }
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"time": clock, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REGISTRATION), "exec"),
         namespace)
    return namespace


def _bm_for_edges(edges, extra_verts=()):
    verts = list({vert for edge in edges for vert in edge.verts})
    verts.extend(extra_verts)
    faces = []
    seen_faces = set()
    for edge in edges:
        for face in edge.link_faces:
            if id(face) not in seen_faces:
                seen_faces.add(id(face))
                faces.append(face)
    return SimpleNamespace(verts=verts, edges=list(edges), faces=faces)


def _owner():
    return SimpleNamespace(data=object(), matrix_world=())


class _Vert:
    def __init__(self, index=-1, co=(0.0, 0.0, 0.0)):
        self.index = index
        self.co = co
        self.link_edges = []


class _Edge:
    def __init__(self, first, second, *, faces=1, hidden=False, face_hidden=None,
                 index=-1, length=1.0):
        self.index = index
        self.length = length
        self.verts = (first, second)
        self.link_faces = [SimpleNamespace(hide=value) for value in
                           (face_hidden if face_hidden is not None else [False] * faces)]
        self.hide = hidden
        first.link_edges.append(self)
        second.link_edges.append(self)


def _cycle(size, *, hidden_face=False, lengths=None):
    verts = [_Vert(i, (float(i), 0.0, 0.0)) for i in range(size)]
    return [_Edge(verts[i], verts[(i + 1) % size],
                  face_hidden=[False, True] if hidden_face else [False],
                  index=i, length=(lengths[i] if lengths is not None else 1.0))
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
    assert info["version"] == (3, 4, 12)


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


def test_two_selected_edges_choose_short_arc_and_toggle_by_edge_set():
    edges = _cycle(8, hidden_face=True, lengths=(1, 1, 2, 3, 4, 5, 6, 7))
    resolver = _load_arc_helpers()["_open_boundary_loop_resolve_selection"]
    bm, obj = _bm_for_edges(edges), _owner()
    selected = {edges[0], edges[4]}
    short, token, reason = resolver(
        bm, obj, tuple(selected), None, lambda edge: edge.length
    )
    assert not reason
    assert short == set(edges[:5])  # Independent inclusive-arc oracle.
    assert {edges[0], edges[4]} <= short

    long, token, reason = resolver(
        bm, obj, tuple(short), token, lambda edge: edge.length
    )
    assert not reason
    assert long == {edges[0], *edges[4:]}
    short_again, token, reason = resolver(
        bm, obj, tuple(long), token, lambda edge: edge.length
    )
    assert not reason and short_again == short


def test_two_edge_short_arc_still_toggles_to_the_other_arc():
    edges = _cycle(7, hidden_face=True, lengths=(1, 1, 3, 3, 3, 3, 3))
    resolver = _load_arc_helpers()["_open_boundary_loop_resolve_selection"]
    bm, obj = _bm_for_edges(edges), _owner()
    short, token, reason = resolver(
        bm, obj, (edges[1], edges[0]), None, lambda edge: edge.length
    )
    assert not reason and short == {edges[0], edges[1]}
    long, token, reason = resolver(
        bm, obj, tuple(short), token, lambda edge: edge.length
    )
    assert not reason and long == set(edges)
    short_again, _token, reason = resolver(
        bm, obj, tuple(long), token, lambda edge: edge.length
    )
    assert not reason and short_again == {edges[0], edges[1]}


def test_large_two_edge_cycle_ordering_is_linear_and_bounded():
    size = 75_000
    comparisons = [0]
    comparison_limit = size

    class CountingEdge(_Edge):
        def __eq__(self, other):
            comparisons[0] += 1
            assert comparisons[0] <= comparison_limit
            return self is other

        __hash__ = object.__hash__

    verts = [_Vert(i, (float(i), 0.0, 0.0)) for i in range(size)]
    edges = [CountingEdge(
        verts[i], verts[(i + 1) % size], face_hidden=[False, True],
        index=i, length=1.0,
    ) for i in range(size)]
    arc_options = _load_arc_helpers()["_open_boundary_loop_arc_options"]
    started = time.perf_counter()
    short_arc, long_arc, ordered, reason = arc_options(
        set(edges), (edges[0], edges[size // 2]), lambda edge: edge.length
    )
    elapsed = time.perf_counter() - started

    assert not reason
    assert len(ordered) == size
    assert all(id(actual) == id(expected)
               for actual, expected in zip(ordered, edges))
    assert all(id(actual) == id(expected) for actual, expected in zip(
        short_arc, edges[:size // 2 + 1]
    ))
    assert len(long_arc) + len(short_arc) == size + 2
    assert comparisons[0] <= size
    assert elapsed < 3.0
    PERFORMANCE_RESULTS["large_two_edge_cycle_75000_ms"] = round(elapsed * 1000, 3)
    PERFORMANCE_RESULTS["large_cycle_equality_comparisons"] = comparisons[0]


def test_equal_length_arcs_choose_deterministically_by_edge_index():
    edges = _cycle(6, hidden_face=True)
    resolver = _load_arc_helpers()["_open_boundary_loop_resolve_selection"]
    selected, _token, reason = resolver(
        _bm_for_edges(edges), _owner(), (edges[0], edges[3]), None,
        lambda edge: edge.length,
    )
    assert not reason
    assert selected == {edges[0], edges[1], edges[2], edges[3]}


def test_two_edges_from_different_loops_and_non_boundary_edge_are_rejected():
    first, second = _cycle(4, hidden_face=True), _cycle(5, hidden_face=True)
    resolver = _load_arc_helpers()["_open_boundary_loop_resolve_selection"]
    bm = _bm_for_edges(first + second)
    selected, token, reason = resolver(
        bm, _owner(), (first[0], second[0]), None, lambda edge: edge.length
    )
    assert selected is None and token is None
    assert "same unique opening loop" in reason

    surface = _Edge(_Vert(), _Vert(), faces=2, index=99)
    selected, token, reason = resolver(
        _bm_for_edges(first + [surface]), _owner(), (first[0], surface),
        None, lambda edge: edge.length,
    )
    assert selected is None and token is None and reason


def test_changed_selection_or_local_visibility_discards_repeat_token():
    edges = _cycle(8, hidden_face=True, lengths=(1, 1, 2, 3, 4, 5, 6, 7))
    resolver = _load_arc_helpers()["_open_boundary_loop_resolve_selection"]
    bm, obj = _bm_for_edges(edges), _owner()
    short, token, reason = resolver(
        bm, obj, (edges[0], edges[4]), None, lambda edge: edge.length
    )
    assert not reason

    changed, changed_token, reason = resolver(
        bm, obj, (edges[0], edges[3]), token, lambda edge: edge.length
    )
    assert not reason and changed == {edges[0], edges[1], edges[2], edges[3]}
    assert changed_token["anchors"] == {edges[0], edges[3]}
    assert changed != {edges[0], *edges[4:]}

    edges[0].link_faces[1].hide = False
    selected, stale_token, reason = resolver(
        bm, obj, tuple(short), token, lambda edge: edge.length
    )
    assert selected is None and stale_token is None and reason


def test_ring_external_face_visibility_branch_invalidates_repeat_token():
    edges = _cycle(8, hidden_face=True, lengths=(1, 1, 2, 3, 4, 5, 6, 7))
    spur = _Edge(
        edges[0].verts[0], _Vert(99, (0.0, 1.0, 0.0)),
        face_hidden=[False, False], index=99, length=1.0,
    )
    resolver = _load_arc_helpers()["_open_boundary_loop_resolve_selection"]
    bm, obj = _bm_for_edges(edges + [spur]), _owner()
    short, token, reason = resolver(
        bm, obj, (edges[0], edges[1]), None, lambda edge: edge.length
    )
    assert not reason and short == {edges[0], edges[1]}
    prior_signature = token["state_signature"]

    # This exterior edge is initially not a boundary. Hiding only its outer
    # face makes it a third visible-boundary edge at a ring vertex without
    # changing mesh counts or any ring edge/face signature.
    spur.link_faces[1].hide = True
    selected_before_repeat = frozenset(short)
    selected, stale_token, reason = resolver(
        bm, obj, tuple(short), token, lambda edge: edge.length
    )
    assert token["state_signature"] == prior_signature
    assert selected_before_repeat == frozenset((edges[0], edges[1]))
    assert selected is None and stale_token is None
    assert "branched" in reason


def test_topology_change_and_undo_reset_discard_repeat_token():
    edges = _cycle(7, hidden_face=True, lengths=(1, 1, 3, 3, 3, 3, 3))
    helpers = _load_arc_helpers()
    resolver = helpers["_open_boundary_loop_resolve_selection"]
    bm, obj = _bm_for_edges(edges), _owner()
    short, token, reason = resolver(
        bm, obj, (edges[0], edges[1]), None, lambda edge: edge.length
    )
    assert not reason and short == {edges[0], edges[1]}
    prior_signature = token["state_signature"]
    bm.verts.append(_Vert(99, (20.0, 0.0, 0.0)))
    same_short, fresh_token, reason = resolver(
        bm, obj, tuple(short), token, lambda edge: edge.length
    )
    assert not reason and same_short == {edges[0], edges[1]}
    assert fresh_token["state_signature"] != prior_signature

    tree = ast.parse(REGISTRATION.read_text(encoding="utf-8"))
    clear_node = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_open_boundary_loop_clear_repeat_token")
    namespace = {"_OPEN_BOUNDARY_LOOP_REPEAT_TOKEN": fresh_token}
    exec(compile(ast.Module(body=[clear_node], type_ignores=[]),
                 str(REGISTRATION), "exec"), namespace)
    namespace["_open_boundary_loop_clear_repeat_token"]()
    assert namespace["_OPEN_BOUNDARY_LOOP_REPEAT_TOKEN"] is None


def run():
    tests = [value for name, value in globals().items()
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    return {
        "passed": len(tests),
        "performance": dict(PERFORMANCE_RESULTS),
        "blender_imported": False,
    }


if __name__ == "__main__":
    print(run())
