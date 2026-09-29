"""Time-to-ball estimates, role (head) assignment and slot ordering.

All arithmetic is float32, in the order the policy's training environment used, so the tolerant comparisons
(1e-3) resolve the same way.
"""
from __future__ import annotations

import math

import numpy as np

F = np.float32
CAR_MAX_SPEED = F(2300.0)
TOUCH_RADIUS = F(92.75 + 85.0)  # ball radius + car reach used by the estimate
TOLERANCE = 1e-3
TAKEOVER_MARGIN = F(0.10)  # seconds a new head-0 leader must gain over the incumbent
INF = F(np.inf)


def _norm(v):
    return np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1] + v[..., 2] * v[..., 2])


def _dot(a, b):
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _clamp01(x):
    return np.clip(x, F(0.0), F(1.0))


def trajectory_times(s):
    """Per car: (time for a 2300 uu/s car to reach the moving ball, closest distance along that path, squared
    distance now)."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        rel = s['ball_pos'][None] - s['pos']
        bv = np.broadcast_to(s['ball_vel'][None], rel.shape)
        dist = _norm(rel)
        dist_sq = _dot(rel, rel)
        a = _dot(bv, bv) - CAR_MAX_SPEED * CAR_MAX_SPEED
        b = F(2.0) * (_dot(rel, bv) - CAR_MAX_SPEED * TOUCH_RADIUS)
        c = dist * dist - TOUCH_RADIUS * TOUCH_RADIUS
        n = dist.shape[0]
        t = np.full(n, INF, np.float32)
        closest = np.zeros(n, np.float32)
        for i in range(n):
            if dist[i] <= TOUCH_RADIUS:
                t[i], closest[i] = F(0.0), dist[i]
                continue
            ai, bi, ci = a[i], b[i], c[i]
            ti = INF
            if abs(ai) <= F(1e-6):
                if bi < F(-1e-6):
                    ti = -ci / bi
            else:
                disc = bi * bi - F(4.0) * ai * ci
                if disc >= F(0.0):
                    root = np.sqrt(disc)
                    inv2a = F(0.5) / ai
                    t0, t1 = (-bi - root) * inv2a, (-bi + root) * inv2a
                    lo, hi = min(t0, t1), max(t0, t1)
                    if ai > F(0.0):
                        if lo >= F(0.0):
                            ti = lo
                    elif hi >= F(0.0):
                        ti = hi
            if math.isfinite(ti) and ti < F(0.0):
                ti = F(0.0)
            t[i] = ti
            if math.isfinite(ti):
                use = ti
            else:
                use = F(0.0) if abs(ai) <= F(1e-6) else max(F(0.0), -bi / (F(2.0) * ai))
            closest[i] = _norm(rel[i] + bv[i] * F(use))
    return t, closest, dist_sq


def approach_times(s, t):
    """Approach-quality time per car: trajectory time plus a bounded correction for how well the car is set up to
    reach the intercept (momentum, facing, boost on long approaches, side of the ball, air, upside down)."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        finite = np.isfinite(t)
        future = np.where(finite, np.clip(t, F(0.0), F(6.0)), F(0.0))
        intercept = s['ball_pos'][None] + s['ball_vel'][None] * future[:, None]
        to_intercept = intercept - s['pos']
        dist = _norm(to_intercept)
        direction = to_intercept / np.where(dist > F(1e-3), dist, F(1.0))[:, None]
        flat_dir = direction.copy()
        flat_dir[:, 2] = 0
        flat_fwd = s['forward'].copy()
        flat_fwd[:, 2] = 0
        len_dir, len_fwd = _norm(flat_dir), _norm(flat_fwd)
        ok = (len_dir > F(1e-3)) & (len_fwd > F(1e-3))
        facing = np.where(ok, np.clip(_dot(flat_dir, flat_fwd) / np.where(ok, len_dir * len_fwd, F(1.0)), F(-1.0), F(1.0)),
                          F(1.0))
        speed = _dot(s['vel'], direction)
        momentum = F(-0.12) * _clamp01(speed / F(2300.0))
        turn = F(0.12) * (F(1.0) - facing) * F(0.5)
        away = F(0.10) * _clamp01(-speed / F(2300.0))
        long_approach = _clamp01((dist - F(900.0)) / F(2600.0))
        low_boost = F(1.0) - _clamp01(s['boost'] / F(50.0))
        slow = F(1.0) - _clamp01(np.maximum(speed, F(0.0)) / F(1800.0))
        boost_term = F(0.08) * long_approach * low_boost * slow
        bp, bv = s['ball_pos'], s['ball_vel']
        ball_speed = _norm(bv)
        centered = (abs(bp[0]) <= F(50.0)) and (abs(bp[1]) <= F(50.0)) and (ball_speed <= F(50.0))
        goal_y = np.where(s['team'] == 0, F(5120.0), F(-5120.0)).astype(np.float32)
        to_goal = np.stack([-intercept[:, 0], goal_y - intercept[:, 1], np.zeros_like(goal_y)], -1)
        len_goal = _norm(to_goal)
        side_ok = (not centered) & (len_dir > F(1e-3)) & (len_goal > F(1e-3))
        side_dot = np.clip(_dot(flat_dir, to_goal) / np.where(side_ok, len_dir * len_goal, F(1.0)), F(-1.0), F(1.0))
        side_term = np.where(side_ok, F(0.18) * (F(1.0) - side_dot) * F(0.5), F(0.0))
        grounded = (s['wheel_contacts'] >= 4) & ~s['is_jumping'] & ~s['is_flipping']
        low_ball = F(1.0) - _clamp01((intercept[:, 2] - F(220.0)) / F(700.0))
        elevated = _clamp01((s['pos'][:, 2] - F(120.0)) / F(500.0))
        air_term = np.where(grounded, F(0.0), F(0.10) * low_ball * elevated)
        upside_down = F(0.08) * _clamp01(-s['up'][:, 2])
        correction = np.clip(momentum + turn + away + boost_term + side_term + air_term + upside_down, F(-0.12), F(0.48))
        out = np.maximum(t + correction, F(0.0))
        out = np.where(dist > F(1e-3), out, t)
        out = np.where(t <= F(0.05), t, out)
        return np.where(finite & ~s['is_demoed'], out, INF).astype(np.float32)


def _less(ka, kb):
    """Tolerant lexicographic comparison. Keys: (bool-or-float value, tolerance or None) ..., then car id."""
    for (va, tol), (vb, _) in zip(ka[:-1], kb[:-1]):
        if tol is None:
            if va != vb:
                return va < vb
        elif abs(va - vb) > tol:  # inf - inf is nan: compares False, falls through
            return va < vb
    return ka[-1] < kb[-1]


def _insertion_sort(items, key):
    out = list(items)
    for i in range(1, len(out)):
        cur, k = out[i], key(out[i])
        j = i - 1
        while j >= 0 and _less(k, key(out[j])):
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = cur
    return out


def team_indices(s, team):
    return [i for i in range(len(s['team'])) if int(s['team'][i]) == team]


def assign_roles(s, team, history, keys=None):
    """Role order (car indices, head 0 first) for `team`, with head-0 hysteresis. `history` is the previous role
    order (list of car indices) or None. Returns the new order, which is also the next history."""
    t, closest, dist_sq = keys if keys is not None else trajectory_times(s)
    aq = approach_times(s, t)
    demoed = s['is_demoed']

    def key(i):
        return ((bool(demoed[i]), None), (float(aq[i]), TOLERANCE), (float(t[i]), TOLERANCE),
                (float(closest[i]), TOLERANCE), (float(dist_sq[i]), TOLERANCE), int(s['car_id'][i]))

    order = _insertion_sort(team_indices(s, team), key)
    if history and len(order) == 3 and len(history) == 3 and history[0] in order:
        pos = order.index(history[0])
        if pos > 0 and demoed[order[0]] == demoed[order[pos]]:
            advantage = aq[order[pos]] - aq[order[0]]
            if not advantage >= TAKEOVER_MARGIN:
                order = [order[pos]] + [c for c in order if c != order[pos]]
    return order


def opponent_order(s, team, keys=None):
    """Opponents of `team` for the observation's slots 3-5: trajectory time, closest distance, squared distance,
    car id (no demo priority)."""
    t, closest, dist_sq = keys if keys is not None else trajectory_times(s)

    def key(i):
        return ((float(t[i]), TOLERANCE), (float(closest[i]), TOLERANCE), (float(dist_sq[i]), TOLERANCE),
                int(s['car_id'][i]))

    return _insertion_sort(team_indices(s, 1 - team), key)
