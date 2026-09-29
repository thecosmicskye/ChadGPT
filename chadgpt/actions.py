"""The 324-action table, legal-action masks and controller conversion."""
from __future__ import annotations

import math

import numpy as np

NUM_ACTIONS = 324
# Control order in the table: throttle, steer, pitch, yaw, roll, jump, boost, handbrake.
THROTTLE, STEER, PITCH, YAW, ROLL, JUMP, BOOST, HANDBRAKE = range(8)
MIRRORED_CONTROLS = (STEER, YAW, ROLL)  # negate these when the team's frame was mirrored in x
MASK_SIZES = (75, 18, 102, 24, 279, 222, 306, 228)


def _round3(value):
    v = np.float32(value)
    return float(np.float32(np.round(v * np.float32(1000.0)) / np.float32(1000.0)))


def _rows():
    three, two = (-1.0, 0.0, 1.0), (0.0, 1.0)
    rows = []
    # Ground actions: throttle x steer, boost only at full throttle, handbrake on/off. Yaw mirrors steer.
    for throttle in three:
        for steer in three:
            for boost in two:
                for handbrake in two:
                    if boost and throttle != 1.0:
                        continue
                    rows.append((throttle, steer, 0.0, steer, 0.0, 0.0, boost, handbrake))
    # Air actions: at least one full input, pitch and roll not both zero (those are ground actions).
    for throttle in three:
        for pitch in three:
            for yaw in three:
                for roll in three:
                    if pitch == 0.0 and roll == 0.0:
                        continue
                    if max(abs(pitch), abs(yaw), abs(roll)) < 1.0:
                        continue
                    for boost in two:
                        if boost and throttle != 1.0:
                            continue
                        rows.append((throttle, yaw, pitch, yaw, roll, 0.0, boost, 1.0))
    # Jump / flip actions: yaw x (no direction or one of 16 directions) x boost x throttle.
    pi32 = np.float32(math.pi)
    for yaw in three:
        for direction in range(-1, 16):
            pitch = roll = np.float32(0.0)
            if direction >= 0:
                angle = np.float32(pi32 * np.float32(direction)) / np.float32(8.0)
                pitch, roll = np.sin(angle, dtype=np.float32), np.cos(angle, dtype=np.float32)
                scale = max(abs(pitch), abs(roll))
                pitch, roll = np.float32(pitch / scale), np.float32(roll / scale)
            for boost in two:
                for throttle in three:
                    if boost and throttle != 1.0:
                        continue
                    rows.append((throttle, yaw, float(pitch), yaw, float(roll), 1.0, boost, 1.0))
    return [tuple(_round3(x) for x in r) for r in rows]


def build_action_table():
    rows = _rows()
    order = sorted(range(len(rows)), key=lambda i: (rows[i][JUMP], -rows[i][THROTTLE], rows[i][BOOST]))  # stable
    table = np.array([rows[i] for i in order], dtype=np.float32)
    if table.shape != (NUM_ACTIONS, 8):
        raise AssertionError(f'action table has shape {table.shape}')
    return table


def build_masks(table):
    """masks[index] for index = on_ground | has_boost << 1 | can_jump << 2."""
    jump = table[:, JUMP] != 0
    boost = table[:, BOOST] != 0
    ground = ~jump & (table[:, PITCH] == 0) & (table[:, ROLL] == 0)
    air = (~jump & ~ground) | (ground & (table[:, THROTTLE] == table[:, BOOST])
                               & ((table[:, YAW] != 0) == (table[:, HANDBRAKE] != 0)))
    masks = np.zeros((8, NUM_ACTIONS), dtype=bool)
    for index in range(8):
        allowed = (ground if index & 1 else air).copy()
        if not index & 2:
            allowed &= ~boost
        if index & 4:
            allowed |= jump
        masks[index] = allowed
    if tuple(int(n) for n in masks.sum(1)) != MASK_SIZES:
        raise AssertionError(f'mask sizes {masks.sum(1).tolist()}')
    return masks


ACTION_TABLE = build_action_table()
MASKS = build_masks(ACTION_TABLE)
