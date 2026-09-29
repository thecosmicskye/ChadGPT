"""Team observation (396 floats) and per-car legal-action masks."""
from __future__ import annotations

import math

import numpy as np

from .actions import MASKS, MIRRORED_CONTROLS

F = np.float32
OBS_SIZE = 396
POS_COEF = F(1 / 2300.0)
VEL_COEF = F(1 / 2300.0)
ANG_COEF = F(1) / F(3.14159265358979323846)
PAD_COEF = F(1 / 10.0)
DOUBLE_JUMP_WINDOW = F(1.25)
# Boost-pad index permutation for a mirror in x (pads are listed by y, then x).
PAD_X_MIRROR = np.array((0, 2, 1, 4, 3, 6, 5, 7, 9, 8, 11, 10, 14, 13, 12, 18, 17, 16, 15, 21, 20, 19, 23, 22, 25, 24,
                         26, 28, 27, 30, 29, 32, 31, 33))
MIRROR_CONTROL_COLUMNS = list(MIRRORED_CONTROLS)
CONTEXT_COLUMNS = (380, 395)
TERMINAL_COLUMN = 395
SLOTS = 8

# Multi-timescale context
HALF_LIVES = (F(1.0), F(10.0), F(60.0))
LN2 = F(0.6931471805599453)
MAX_CAR_BALL_DISTANCE = F(13271.94033)


def has_flip_or_jump(s):
    return s['is_on_ground'] | (~s['has_flipped'] & ~s['has_double_jumped']
                                & (s['air_time_since_jump'] < DOUBLE_JUMP_WINDOW))


def action_masks(s):
    """(N, 324) legal actions per car."""
    index = (s['is_on_ground'].astype(np.int64) | ((s['boost'] != 0).astype(np.int64) << 1)
             | ((has_flip_or_jump(s) | s['turtled']).astype(np.int64) << 2))
    return MASKS[index]


def team_observation(s, team, roles, opponents, context, terminal_flag=1.0):
    """Observation of `team` with friendly slots in role order and the given opponent order.
    Returns (obs (396,) float32, mirrored)."""
    inverted = team == 1
    flip = np.array([-1, -1, 1], np.float32) if inverted else np.ones(3, np.float32)
    bpos, bvel, bang = s['ball_pos'] * flip, s['ball_vel'] * flip, s['ball_ang_vel'] * flip
    mirrored = bool(bpos[0] < 0)
    m = F(-1.0) if mirrored else F(1.0)
    fx = np.array([m, 1, 1], np.float32)
    fyz = np.array([1, m, m], np.float32)
    o = np.zeros(OBS_SIZE, np.float32)
    o[0:3] = bpos * fx * POS_COEF
    o[3:6] = bvel * fx * VEL_COEF
    o[6:9] = bang * fyz * ANG_COEF
    order = list(roles) + list(opponents)
    n = len(order)
    idx = np.array(order, dtype=np.int64)
    pos = s['pos'][idx] * flip * fx
    vel = s['vel'][idx] * flip * fx
    ang = s['ang_vel'][idx] * flip * fyz
    fwd = s['forward'][idx] * flip * fx
    up = s['up'][idx] * flip * fx
    torque = s['flip_rel_torque'][idx] * fx
    car3d = np.concatenate([pos * POS_COEF, fwd, up, vel * VEL_COEF, ang * ANG_COEF, torque], 1)
    o[9:9 + n * 18] = car3d.reshape(-1)
    pads = s['pads'][::-1] if inverted else s['pads']
    if mirrored:
        pads = pads[PAD_X_MIRROR]
    o[153:187] = pads * PAD_COEF
    cont = np.stack([np.minimum(s['jump_time'][idx], F(0.25)), s['air_time_since_jump'][idx],
                     np.minimum(s['flip_time'][idx], F(0.95)), s['demo_timer'][idx], s['boost'][idx] / F(100.0),
                     s['boosting_time'][idx], s['handbrake'][idx]], 1)
    o[187:187 + n * 7] = cont.reshape(-1)
    prev = s['prev'][idx].copy()
    prev[:, MIRROR_CONTROL_COLUMNS] *= m
    o[243:243 + n] = prev[:, 3]
    ints = np.stack([s['team'][idx].astype(np.float32), s['hitbox'][idx].astype(np.float32),
                     prev[:, 0], prev[:, 2], prev[:, 4]], 1)
    o[252:252 + n * 5] = ints.reshape(-1)
    hfj = has_flip_or_jump(s)[idx]
    bools = np.stack([s['is_on_ground'][idx], s['has_jumped'][idx], s['is_jumping'][idx], s['has_double_jumped'][idx],
                      s['has_flipped'][idx], s['is_flipping'][idx], hfj], 1).astype(np.float32)
    bools = np.concatenate([bools, prev[:, 5:8]], 1)
    o[292:292 + n * 10] = bools.reshape(-1)
    o[376:380] = -1.0
    o[376:376 + min(len(roles), 4)] = np.arange(min(len(roles), 4), dtype=np.float32)
    o[CONTEXT_COLUMNS[0]:CONTEXT_COLUMNS[1]] = context * (F(-1.0) if inverted else F(1.0))
    o[TERMINAL_COLUMN] = terminal_flag
    return o, mirrored


def context_signals(s):
    """The four continuous context signals of a state (ball y, ball y-velocity, boost advantage, nearest-car
    advantage; blue positive)."""
    blue_boost = orange_boost = F(0.0)
    blue_n = orange_n = 0
    blue_near = orange_near = F(np.inf)
    for i in range(len(s['team'])):
        b = np.clip(s['boost'][i] / F(100.0), F(0.0), F(1.0))
        d = s['pos'][i] - s['ball_pos']
        dist = np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if int(s['team'][i]) == 0:
            blue_boost, blue_n = blue_boost + b, blue_n + 1
            if not s['is_demoed'][i]:
                blue_near = min(blue_near, dist)
        else:
            orange_boost, orange_n = orange_boost + b, orange_n + 1
            if not s['is_demoed'][i]:
                orange_near = min(orange_near, dist)
    mean_blue = blue_boost / F(blue_n) if blue_n else F(0.0)
    mean_orange = orange_boost / F(orange_n) if orange_n else F(0.0)
    if np.isfinite(blue_near) and np.isfinite(orange_near):
        near = np.clip((orange_near - blue_near) / MAX_CAR_BALL_DISTANCE, F(-1.0), F(1.0))
    else:
        near = F(0.0)
    return np.array([np.clip(s['ball_pos'][1] / F(5120.0), F(-1.0), F(1.0)),
                     np.clip(s['ball_vel'][1] / F(6000.0), F(-1.0), F(1.0)),
                     np.clip(mean_blue - mean_orange, F(-1.0), F(1.0)), near], np.float32)


def retention(dt):
    # The bridge computes std::exp(-ln2 * dt / h) in float (the Windows C runtime's expf). The exponent is formed in
    # float32 the same way; the exp is taken in double and rounded once, which equals that expf on every dt = k/120
    # (k <= 240) and half-life. numpy's float32 exp on x86 differs in 333 of those 720 cases, including 1/120 s at
    # 60 s (one ulp high), which moved the 60 s context entries by up to 5e-5 against the bridge.
    return np.array([math.exp(float(-LN2 * F(dt) / h)) for h in HALF_LIVES], np.float32)


def fold_context(context, s, dt, impulse=0.0, keep=None):
    """One context update over `dt` seconds with a goal impulse (+1 blue goal, -1 orange goal). `keep` overrides
    the retention (3,). Returns the new (15,) context."""
    r = retention(dt) if keep is None else keep
    sig = context_signals(s)
    c = context.reshape(3, 5).copy()
    c[:, :4] = r[:, None] * c[:, :4] + (F(1.0) - r)[:, None] * sig[None, :]
    c[:, 4] = np.clip(r * c[:, 4] + F(impulse), F(-1.0), F(1.0))
    return c.reshape(15).astype(np.float32)
