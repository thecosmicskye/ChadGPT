"""RLBot v5 GamePacket -> policy state, with the temporal reconstruction RLBot does not report
."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import empty_state

F = np.float32
TICK = F(1.0 / 120.0)
# Rocket League / RocketSim car constants (seconds, per second).
JUMP_MIN_TIME = F(0.025)
JUMP_MAX_TIME = F(0.2)
JUMP_RESET_TIME_PAD = F(1 / 40.0)
JUMP_HOLD_GROUNDED = F(5.5 / 120.0)  # a jump held this soon after take-off still counts as grounded
FLIP_TORQUE_TIME = F(0.65)
DOUBLE_JUMP_WINDOW = F(1.25)
POWERSLIDE_RISE = F(5.0)
POWERSLIDE_FALL = F(2.0)
BOOST_MIN_TIME = F(0.1)
FLOOR_CONTACT_MAX_Z = F(25.0)
MAX_STEP_SECONDS = F(0.25)
DEMO_RESPAWN_TIME = F(3.0)  # RocketSim RESPAWN_TIME: the demolished car's timer starts here on the demolition tick

# Boost pads (x, y). Training (RLGym BOOST_LOCATIONS, GigaLearnCPP CommonValues::BOOST_LOCATIONS, RocketSim's own pad
# list) orders them by y with the midfield rows (y = +-1024/1036) as (-2048, 0, +2048); RLBot v5 (FieldInfo and the
# GamePacket's boost_pads) sorts them by y, then x, which puts the midfield rows as (-2048, +2048, 0) (the y of the
# centre pad differs by 12 uu). The observation reads the pad timers in training order.
TRAINING_PAD_XY = (
    (0, -4240), (-1792, -4184), (1792, -4184), (-3072, -4096), (3072, -4096), (-940, -3308), (940, -3308), (0, -2816),
    (-3584, -2484), (3584, -2484), (-1788, -2300), (1788, -2300), (-2048, -1036), (0, -1024), (2048, -1036),
    (-3584, 0), (-1024, 0), (1024, 0), (3584, 0), (-2048, 1036), (0, 1024), (2048, 1036), (-1788, 2300), (1788, 2300),
    (-3584, 2484), (3584, 2484), (0, 2816), (-940, 3310), (940, 3308), (-3072, 4096), (3072, 4096), (-1792, 4184),
    (1792, 4184), (0, 4240))
RLBOT_PAD_XY = (
    (0, -4240), (-1792, -4184), (1792, -4184), (-3072, -4096), (3072, -4096), (-940, -3308), (940, -3308), (0, -2816),
    (-3584, -2484), (3584, -2484), (-1788, -2302), (1788, -2302), (-2048, -1036), (2048, -1036), (0, -1024),
    (-3584, 0), (-1024, 0), (1024, 0), (3584, 0), (0, 1024), (-2048, 1036), (2048, 1036), (-1788, 2302), (1788, 2302),
    (-3584, 2484), (3584, 2484), (0, 2816), (-940, 3308), (940, 3308), (-3072, 4096), (3072, 4096), (-1792, 4184),
    (1792, 4184), (0, 4240))


def pad_order(packet_xy, training_xy=TRAINING_PAD_XY):
    """Packet pad index of every training pad, matched by (x, y) within 20 uu (a bijection), as a numpy index array:
    training_timers = packet_timers[pad_order(...)]."""
    out = []
    for tx, ty in training_xy:
        hits = [k for k, (x, y) in enumerate(packet_xy) if (x - tx) ** 2 + (y - ty) ** 2 < 400.0]
        if len(hits) != 1:
            raise ValueError(f'boost pad ({tx}, {ty}) matched {len(hits)} packet pads')
        out.append(hits[0])
    if sorted(out) != list(range(len(training_xy))):
        raise ValueError('boost pad matching is not a bijection')
    return np.array(out, np.int64)


PAD_FROM_PACKET = pad_order(RLBOT_PAD_XY)


def demo_timer_of(timeout):
    """RLBot demolished_timeout (-1 alive, 0 on the demolition frame, then the seconds left counting down from
    3 - 1/120) -> RocketSim's demo respawn timer (0 alive, 3 on the demolition tick, then counting down)."""
    if timeout < 0:
        return F(0.0)
    return DEMO_RESPAWN_TIME if timeout == 0 else F(timeout)


def rotation_basis(yaw, pitch, roll):
    """Rotator -> (forward, up) unit vectors, float32."""
    cy, sy = np.cos(F(yaw)), np.sin(F(yaw))
    cp, sp = np.cos(F(pitch)), np.sin(F(pitch))
    cr, sr = np.cos(F(roll)), np.sin(F(roll))
    forward = np.array([cp * cy, cp * sy, sp], np.float32)
    up = np.array([-cr * sp * cy - sr * sy, -cr * sp * sy + sr * cy, cp * cr], np.float32)
    return forward, up


def controls_of(ci):
    """flat.ControllerState -> (8,) float32 in table order."""
    if ci is None:
        return np.zeros(8, np.float32)
    return np.array([ci.throttle, ci.steer, ci.pitch, ci.yaw, ci.roll, 1.0 if ci.jump else 0.0,
                     1.0 if ci.boost else 0.0, 1.0 if ci.handbrake else 0.0], np.float32)


@dataclass
class Temporal:
    """What RLBot does not report about a car, integrated per packet from the controls the car applied."""
    has_jumped: bool = False
    has_double_jumped: bool = False
    has_flipped: bool = False
    is_jumping: bool = False
    is_flipping: bool = False
    is_boosting: bool = False
    last_controls: np.ndarray = field(default_factory=lambda: np.zeros(8, np.float32))
    flip_rel_torque: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    jump_time: np.float32 = F(0.0)
    air_time: np.float32 = F(0.0)
    air_time_since_jump: np.float32 = F(0.0)
    flip_time: np.float32 = F(0.0)
    boosting_time: np.float32 = F(0.0)
    handbrake: np.float32 = F(0.0)
    last_boost: np.float32 = F(0.0)

    @classmethod
    def from_packet(cls, car, controls):
        """Restart from what the packet shows (first packet, demolition, implausible gap)."""
        t = cls(has_jumped=car['has_jumped'], has_double_jumped=car['has_double_jumped'],
                has_flipped=car['has_flipped'])
        t.is_flipping = car['has_flipped'] and not car['is_on_ground'] and car['flip_time'] < FLIP_TORQUE_TIME
        t.last_controls = controls.copy()
        t.flip_time = max(F(0.0), car['flip_time']) if car['has_flipped'] else F(0.0)
        t.last_boost = car['boost']
        return t

    def step(self, car, c, dt):
        """Advance by dt seconds with applied controls c; updates `car` (a dict of this car's packet values)."""
        packet_ground = car['is_on_ground']
        grounded = packet_ground
        new_jump = car['has_jumped'] and not self.has_jumped
        if (new_jump or self.has_jumped) and c[5] == 1.0 and self.jump_time <= JUMP_HOLD_GROUNDED:
            grounded = True
        if new_jump:
            self.has_jumped, self.has_double_jumped, self.has_flipped, self.is_flipping = True, False, False, False
            self.flip_rel_torque = np.zeros(3, np.float32)
            self.flip_time = self.jump_time = self.air_time_since_jump = F(0.0)
            self.is_jumping = (not car['has_double_jumped']) and c[5] == 1.0 and self.jump_time < JUMP_MAX_TIME
        pressed = (not new_jump) and c[5] == 1.0 and self.last_controls[5] != 1.0

        if c[7] == 1.0:
            self.handbrake = self.handbrake + POWERSLIDE_RISE * dt
        else:
            self.handbrake = self.handbrake - POWERSLIDE_FALL * dt
        self.handbrake = F(min(max(self.handbrake, F(0.0)), F(1.0)))

        if self.last_boost > 0:
            if self.is_boosting:
                self.is_boosting = c[6] == 1.0 or self.boosting_time < BOOST_MIN_TIME
            elif c[6] == 1.0:
                self.is_boosting = True
        else:
            self.is_boosting = False
        self.boosting_time = self.boosting_time + dt if self.is_boosting else F(0.0)

        if not grounded and self.is_flipping:
            self.is_flipping = self.has_flipped and self.flip_time < FLIP_TORQUE_TIME
        elif grounded:
            self.is_flipping = False

        if grounded and not self.is_jumping:
            if not (self.has_jumped and self.jump_time < JUMP_MIN_TIME + JUMP_RESET_TIME_PAD):
                self.has_jumped, self.jump_time = False, F(0.0)

        if self.is_jumping:
            self.is_jumping = self.jump_time < JUMP_MIN_TIME or (c[5] == 1.0 and self.jump_time < JUMP_MAX_TIME)
        elif grounded and pressed:
            self.is_jumping, self.jump_time = True, F(0.0)
        if self.is_jumping:
            self.has_jumped = True
        if self.is_jumping or self.has_jumped:
            self.jump_time = self.jump_time + dt

        if grounded:
            self.has_double_jumped = self.has_flipped = False
            self.air_time = self.air_time_since_jump = self.flip_time = F(0.0)
            self.flip_rel_torque = np.zeros(3, np.float32)
        else:
            self.air_time = self.air_time + dt
            if self.has_jumped and not self.is_jumping:
                self.air_time_since_jump = self.air_time_since_jump + dt
            else:
                self.air_time_since_jump = F(0.0)
            if pressed and self.air_time_since_jump < DOUBLE_JUMP_WINDOW:
                stick = abs(c[3]) + abs(c[2]) + abs(c[4])
                if not self.has_double_jumped and not self.has_flipped:
                    if stick >= F(0.5):
                        self.flip_time, self.has_flipped, self.is_flipping = F(0.0), True, True
                        dx, dy = -c[2], c[3] + c[4]
                        if abs(dy) < F(0.1) and abs(dx) < F(0.1):
                            dx = dy = F(0.0)
                        else:
                            n = np.sqrt(dx * dx + dy * dy)
                            dx, dy = dx / n, dy / n
                        ticks = dt / TICK
                        self.flip_rel_torque = (np.array([-dy / ticks, dx / ticks, 0.0], np.float32) if ticks > 0
                                                else np.zeros(3, np.float32))
                    else:
                        self.has_double_jumped = True
            if self.is_flipping or self.has_flipped:
                self.flip_time = self.flip_time + dt

        if not car['has_jumped'] and grounded and not self.is_jumping:
            self.has_jumped = self.has_double_jumped = self.has_flipped = False
            self.is_jumping = self.is_flipping = False
            self.jump_time = self.air_time_since_jump = self.flip_time = F(0.0)
            self.flip_rel_torque = np.zeros(3, np.float32)

        # Once the game reports the dodge, its flag and elapsed time win over the reconstruction.
        if car['has_flipped']:
            packet_flip = max(F(0.0), car['flip_time'])
            self.has_flipped = True
            self.has_double_jumped = car['has_double_jumped']
            if packet_flip > 0 or self.flip_time <= 0:
                self.flip_time = packet_flip
            self.is_flipping = (not grounded) and self.flip_time < FLIP_TORQUE_TIME

        self.last_controls = c.copy()
        self.last_boost = car['boost']
        car['is_on_ground'] = grounded

    def apply(self, car):
        car.update(has_jumped=self.has_jumped, has_double_jumped=self.has_double_jumped,
                   has_flipped=self.has_flipped, is_jumping=self.is_jumping, is_flipping=self.is_flipping,
                   flip_rel_torque=self.flip_rel_torque.copy(), jump_time=self.jump_time,
                   air_time_since_jump=self.air_time_since_jump, flip_time=self.flip_time,
                   boosting_time=self.boosting_time, handbrake=self.handbrake, air_time=self.air_time,
                   is_boosting=self.is_boosting)


class PacketReader:
    """Keeps the per-car temporal state across packets and builds the policy state."""

    def __init__(self):
        self.temporal: dict[int, Temporal] = {}
        self.pad_from_packet = PAD_FROM_PACKET  # the hivemind re-derives it from FieldInfo (set_field_pads)
        # The latest packet's cars after temporal reconstruction, the packet's own ground flags and every car's
        # last_input: the shadow arena's input (chadgpt/shadow_arena.py)
        self.last_cars, self.last_ground_evidence = [], np.zeros(0, bool)
        self.last_controls = np.zeros((0, 8), np.float32)

    def clear(self):
        self.temporal.clear()

    def set_field_pads(self, field_xy):
        """Match the match's FieldInfo boost pads (x, y in packet order) to the training order."""
        self.pad_from_packet = pad_order(field_xy)
        return self.pad_from_packet

    def read(self, packet, dt, integrate):
        """packet -> state dict.
        integrate: the game clock is running, so advance the temporal state by dt seconds."""
        players = packet.players
        n = len(players)
        s = empty_state(n)
        cars, controls = [], np.zeros((n, 8), np.float32)
        slots = [0, 0]
        for i, p in enumerate(players):
            team = int(p.team)
            slot = slots[team] if team in (0, 1) else 0
            if team in (0, 1):
                slots[team] += 1
            ph = p.physics
            forward, up = rotation_basis(ph.rotation.yaw, ph.rotation.pitch, ph.rotation.roll)
            car = dict(team=team, car_id=(2 + 2 * slot) if team == 1 else (1 + 2 * slot),
                       pos=np.array([ph.location.x, ph.location.y, ph.location.z], np.float32),
                       vel=np.array([ph.velocity.x, ph.velocity.y, ph.velocity.z], np.float32),
                       ang_vel=np.array([ph.angular_velocity.x, ph.angular_velocity.y, ph.angular_velocity.z],
                                        np.float32),
                       forward=forward, up=up, boost=F(p.boost), is_on_ground=int(p.air_state) == 0,
                       has_jumped=bool(p.has_jumped), has_double_jumped=bool(p.has_double_jumped),
                       has_flipped=bool(p.has_dodged), flip_time=F(p.dodge_elapsed),
                       is_demoed=p.demolished_timeout >= 0, demo_timer=demo_timer_of(p.demolished_timeout),
                       is_jumping=False, is_flipping=False,
                       flip_rel_torque=np.zeros(3, np.float32), jump_time=F(0.0), air_time_since_jump=F(0.0),
                       boosting_time=F(0.0), handbrake=F(0.0), air_time=F(0.0), is_boosting=False,
                       rot=(ph.rotation.yaw, ph.rotation.pitch, ph.rotation.roll), is_supersonic=bool(p.is_supersonic))
            controls[i] = controls_of(p.last_input)
            cars.append(car)
        ground_evidence = np.array([c['is_on_ground'] for c in cars], bool)
        if integrate:
            for i, car in enumerate(cars):
                t = self.temporal.get(i)
                if t is None or car['is_demoed'] or dt < 0 or dt > MAX_STEP_SECONDS:
                    t = self.temporal[i] = Temporal.from_packet(car, controls[i])
                else:
                    t.step(car, controls[i], F(dt))
                t.apply(car)
        for i, car in enumerate(cars):
            for k in ('team', 'car_id', 'boost', 'is_on_ground', 'has_jumped', 'has_double_jumped', 'has_flipped',
                      'flip_time', 'is_demoed', 'demo_timer', 'is_jumping', 'is_flipping', 'jump_time',
                      'air_time_since_jump', 'boosting_time', 'handbrake', 'pos', 'vel', 'ang_vel', 'forward', 'up',
                      'flip_rel_torque'):
                s[k][i] = car[k]
            # Floor contact from height (normal up): the turtle bit of the action mask.
            s['turtled'][i] = car['world_contact'] = (not car['is_demoed']) and car['pos'][2] <= FLOOR_CONTACT_MAX_Z
            s['wheel_contacts'][i] = 4 if ground_evidence[i] else 0
        ball = packet.balls[0].physics if len(packet.balls) else None
        if ball is not None:
            s['ball_pos'][:] = (ball.location.x, ball.location.y, ball.location.z)
            s['ball_vel'][:] = (ball.velocity.x, ball.velocity.y, ball.velocity.z)
            s['ball_ang_vel'][:] = (ball.angular_velocity.x, ball.angular_velocity.y, ball.angular_velocity.z)
        if len(packet.boost_pads) == 34:  # packet order -> training order
            s['pads'][:] = np.array([bp.timer for bp in packet.boost_pads], np.float32)[self.pad_from_packet]
        s['prev'][:] = controls  # every car's last_input; the agent puts its own cars' held controls over these
        self.last_cars, self.last_ground_evidence, self.last_controls = cars, ground_evidence, controls
        return s
