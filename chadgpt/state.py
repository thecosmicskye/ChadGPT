"""Per-car game state the policy reads, as numpy float32 / bool arrays over the cars of the match.

Car order is the RLBot packet order. The runtime needs these fields (shapes for N cars):

    ball_pos, ball_vel, ball_ang_vel   (3,)
    pads                               (34,)   boost-pad cooldown remaining, seconds, packet order
    team, car_id, hitbox               (N,)    int
    pos, vel, ang_vel, forward, up     (N, 3)
    flip_rel_torque                    (N, 3)
    boost, boosting_time, handbrake, demo_timer, jump_time, air_time_since_jump, flip_time   (N,)
    is_demoed, has_jumped, is_jumping, has_flipped, has_double_jumped, is_flipping,
    is_on_ground, turtled              (N,)    bool
    wheel_contacts                     (N,)    int, 0-4
    prev                               (N, 8)  controls the car held over the last decision (world frame)
"""
from __future__ import annotations

import numpy as np

FLOAT_FIELDS = ('boost', 'boosting_time', 'handbrake', 'demo_timer', 'jump_time', 'air_time_since_jump', 'flip_time')
BOOL_FIELDS = ('is_demoed', 'has_jumped', 'is_jumping', 'has_flipped', 'has_double_jumped', 'is_flipping',
               'is_on_ground', 'turtled')
VEC_FIELDS = ('pos', 'vel', 'ang_vel', 'forward', 'up', 'flip_rel_torque')
INT_FIELDS = ('team', 'car_id', 'hitbox', 'wheel_contacts')
BALL_FIELDS = ('ball_pos', 'ball_vel', 'ball_ang_vel')


def empty_state(n_cars):
    s = {k: np.zeros(3, np.float32) for k in BALL_FIELDS}
    s['pads'] = np.zeros(34, np.float32)
    s.update({k: np.zeros(n_cars, np.float32) for k in FLOAT_FIELDS})
    s.update({k: np.zeros(n_cars, bool) for k in BOOL_FIELDS})
    s.update({k: np.zeros((n_cars, 3), np.float32) for k in VEC_FIELDS})
    s.update({k: np.zeros(n_cars, np.int64) for k in INT_FIELDS})
    s['prev'] = np.zeros((n_cars, 8), np.float32)
    return s
