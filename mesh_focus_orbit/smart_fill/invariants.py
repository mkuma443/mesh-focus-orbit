"""Pure Smart Fill selection invariants.

This module deliberately has no Blender imports.  Preview sessions use it to
keep one immutable visible universe and to merge accepted stages without
allowing hidden faces or an unrelated connected island to leak into a result.
"""

from collections import deque

import numpy as np


def visible_seed_component(count, offsets, neighbors, hidden, seed):
    """Return ``(visible_universe, seed_component, reason)``.

    Hidden faces are excluded before traversal and therefore act as stable
    barriers.  The component is computed from the session snapshot, not from a
    radius crop, so later wheel stages cannot change its identity.
    """
    try:
        count = int(count)
        offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
        neighbors = np.asarray(neighbors, dtype=np.int64).reshape(-1)
        hidden = np.asarray(hidden, dtype=bool).reshape(-1)
        seed = int(seed)
    except (TypeError, ValueError):
        return None, None, "schema"
    if (
        count <= 0
        or len(offsets) != count + 1
        or int(offsets[-1]) != len(neighbors)
        or len(hidden) != count
        or seed < 0
        or seed >= count
        or bool(hidden[seed])
    ):
        return None, None, "seed-or-schema"
    if np.any(neighbors < 0) or np.any(neighbors >= count):
        return None, None, "neighbor-range"
    visible = ~hidden
    component = np.zeros(count, dtype=bool)
    component[seed] = True
    pending = deque([seed])
    while pending:
        face = int(pending.popleft())
        for other in neighbors[int(offsets[face]) : int(offsets[face + 1])]:
            other = int(other)
            if visible[other] and not component[other]:
                component[other] = True
                pending.append(other)
    return visible, component, "ok"


def visible_seed_reached_component(count, hidden, seed, reached_ids):
    """Build the visible seed component from an existing graph walk.

    Production Smart Fill already maintains a Dijkstra frontier for the
    current radius. Reusing its reached rows avoids a second Python BFS over
    every polygon. The frontier is topological, so nearby disconnected sheets
    cannot enter the component.
    """
    try:
        count = int(count)
        hidden = np.asarray(hidden, dtype=bool).reshape(-1)
        seed = int(seed)
        reached = np.asarray(reached_ids, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return None, None, "schema"
    if (
        count <= 0
        or len(hidden) != count
        or seed < 0
        or seed >= count
        or bool(hidden[seed])
    ):
        return None, None, "seed-or-schema"
    visible = ~hidden
    component = np.zeros(count, dtype=bool)
    in_range = (reached >= 0) & (reached < count)
    reached = np.unique(reached[in_range])
    if len(reached):
        component[reached] = visible[reached]
    component[seed] = True
    return visible, component, "ok"


def monotonic_visible_selection(
    previous_ids, current_ids, visible_ids, component_ids, seed
):
    """Union a growing candidate with accepted visible faces.

    The operation is deterministic and idempotent.  ``visible_ids`` and
    ``component_ids`` are session snapshots; current hidden changes are not
    silently accepted by this helper and must invalidate the session first.
    """
    try:
        previous = np.asarray(previous_ids, dtype=np.int64).reshape(-1)
        current = np.asarray(current_ids, dtype=np.int64).reshape(-1)
        visible = np.asarray(visible_ids, dtype=bool).reshape(-1)
        component = np.asarray(component_ids, dtype=bool).reshape(-1)
        seed = int(seed)
    except (TypeError, ValueError):
        return None, "schema"
    if len(visible) != len(component):
        return None, "schema"
    if seed < 0 or seed >= len(component) or not component[seed]:
        return None, "seed-invalid"
    merged = np.unique(np.r_[previous, current]).astype(np.int64, copy=False)
    in_range = (merged >= 0) & (merged < len(component))
    valid = np.zeros(len(merged), dtype=bool)
    valid[in_range] = visible[merged[in_range]] & component[merged[in_range]]
    merged = merged[valid]
    if not np.any(merged == seed):
        merged = np.unique(np.r_[merged, np.asarray([seed], dtype=np.int64)])
    return merged.astype(np.int32, copy=False), "ok"


def selection_monotonic(previous_ids, current_ids):
    """Independent oracle used by tests for the subset/superset invariant."""
    previous = {int(value) for value in np.asarray(previous_ids).reshape(-1)}
    current = {int(value) for value in np.asarray(current_ids).reshape(-1)}
    return previous.issubset(current)


def map_global_faces_to_local(global_face_ids, accepted_global_ids):
    """Map accepted mesh-global polygon ids to compact geometry-row ids.

    Compact cursor/visibility graphs are not guaranteed to start at zero or
    to contain contiguous mesh ids.  Returning the matching row positions
    explicitly prevents callers from silently treating a local patch as a
    mesh-global ``arange``.
    """
    try:
        global_ids = np.asarray(global_face_ids, dtype=np.int64).reshape(-1)
        accepted = np.asarray(accepted_global_ids, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("global/local face mapping schema is invalid") from error
    if len(global_ids) == 0:
        return np.empty(0, dtype=np.int32)
    if len(np.unique(global_ids)) != len(global_ids):
        raise ValueError("global/local face mapping is not one-to-one")
    return np.flatnonzero(np.isin(global_ids, accepted)).astype(np.int32, copy=False)


def accepted_graph_rows_from_result(result):
    """Map one displayed result into immutable confirm-graph row space."""
    if not isinstance(result, dict):
        raise ValueError("accepted result schema is invalid")
    confirm = result.get("confirm_geometry")
    if not isinstance(confirm, dict):
        raise ValueError("accepted result has no confirm graph")
    face_ids = np.asarray(confirm.get("face_ids"), dtype=np.int64).reshape(-1)
    accepted_faces = np.asarray(result.get("faces", ()), dtype=np.int64).reshape(-1)
    expected_count = int(confirm.get("count", len(face_ids)))
    if (
        len(face_ids) == 0
        or len(face_ids) != expected_count
        or len(np.unique(face_ids)) != len(face_ids)
    ):
        raise ValueError("confirm graph mapping is invalid")
    accepted_global = np.unique(accepted_faces)
    rows = map_global_faces_to_local(face_ids, accepted_global)
    if len(rows) != len(accepted_global):
        raise ValueError("accepted ids are absent from confirm graph")
    return rows.astype(np.int32, copy=False)


def cached_result_contains_floor(result, floor_ids):
    """Return whether a cached candidate contains the prior accepted floor.

    ``floor_ids`` and the returned rows are both in the immutable confirm-graph
    row space.  This deliberately validates the full candidate provenance
    instead of unioning ``result['faces']`` into a cached boundary or confirm
    domain, which would make the displayed candidate and confirmation diverge.
    """
    try:
        floor = np.asarray(floor_ids, dtype=np.int64).reshape(-1)
        rows = accepted_graph_rows_from_result(result)
    except (TypeError, ValueError):
        return False
    if len(floor) == 0:
        return True
    if np.any(floor < 0):
        return False
    return bool(np.all(np.isin(floor, rows)))


def normal_cache_hit_allowed(result, previous_radius, radius, floor_ids):
    """Apply the Normal radius-cache/floor policy without mutating state."""
    if result is None or previous_radius is None:
        return False
    try:
        previous = float(previous_radius)
        current = float(radius)
    except (TypeError, ValueError):
        return False
    if current < previous:
        return False
    return cached_result_contains_floor(result, floor_ids)


def publish_bounded_result(cache, key, result, limit=3):
    """Publish one result while retaining the existing bounded-cache policy."""
    if not isinstance(cache, dict) or int(limit) <= 0:
        raise ValueError("bounded result cache schema is invalid")
    cache.pop(key, None)
    cache[key] = result
    while len(cache) > int(limit):
        cache.pop(next(iter(cache)), None)
    return cache


def visible_floor_enabled(strict_mode):
    """Whether the Normal-mode visible selection floor applies to a stage."""
    return not bool(strict_mode)
