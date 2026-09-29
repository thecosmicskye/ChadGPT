"""RocketSim shadow arena.

RLBot reports no wheel or surface contact: only `air_state`, a state machine that stays Jumping / DoubleJumping /
Dodging for a fixed time even after the car has landed. The policy was trained on RocketSim's contact bits (on
ground = 3+ wheels touching, the turtle bit from the car body's world contact, the four wheel contacts for the role
rules). The shadow arena recovers them the way the C++ RLBot bridge did (GigaLearnCPP-Leak src/RLBotV5Client.cpp,
class ShadowArena): after each decision the runtime sets a RocketSim arena to the packet state (every car, the
ball), gives every car the controls it will hold over the next frame, steps one tick, and reads the contacts back.
The next packet uses them. The packet stays authoritative for position, velocity, rotation, boost and the jump/flip
state; only `is_on_ground`, the wheel contacts and the world contact come from the arena, and a packet that says
OnGround always stays on the ground.

The bridge stepped on a worker thread and used whatever result was ready; this runtime steps synchronously after the
decision, which is the bridge's normal case (the result of packet t is used at packet t+1).

Rocket League's collision meshes are Psyonix data and do not ship with the bot. The arena runs when the 16 soccar
meshes are in the mesh folder (CHADGPT_COLLISION_MESHES, else collision_meshes/ in the bot's folder); on Windows
chadgpt/mesh_dump.py can dump them from the running game.
"""
from __future__ import annotations

import hashlib
import os
import struct
import threading
from pathlib import Path

import numpy as np

BOOST_MIN_TIME = 0.1
TELEPORT_DISTANCE_SQ = 250.0 * 250.0  # a car moving further than this between packets was reset (kickoff, respawn)
TURTLE_NORMAL_Z = float(np.float32(0.9))  # contactNormal.z > 0.9f selects floor-like contact
SOCCAR_MESH_COUNT = 16
DEMO_HOLD_MIN = 2.0 / 120.0  # least respawn timer for a demolished car, so the one-tick step keeps it demolished
LOG_EVERY = 600  # throttle for repeated failure / fallback logs (packets)
# SHA-256 of the 16 soccar meshes as RocketSim knows them (RLArenaCollisionDumper output, any order). Informational:
# a dump with other hashes still loads if it passes the checks in init_rocketsim.
KNOWN_SOCCAR_SHA256 = frozenset((
    '085f7533c53a73da664c129c3b1c93f5ce08ceb302aa5b993ff5fe95af5d6ae8',
    '0db5c559287bcd255b99e6f5db341b874ec9a5fcf6137247cf16d913e178db5c',
    '21c1d1eb450f9e09ee8d7d55e1d41bf563518f9b2deb2ab5369c47139faa2e4a',
    '3350567082d9a133f59f8471d941fde6cce20eb2b2bc26c2ab0566b7fb7a21e9',
    '36cc44e14a0e500aeb6167a4d5fce449b218b3d941df3ca46f398d269b01af3d',
    '4864911a7f0a3add929f41cc8f676cde416dcc3c07ab08a30742c9893e8a6ce8',
    '762ec145db3c317d82314ad94af6f8b591e822c26860ed3c0ee106aba5591eee',
    '776c24b1231b4f7ab5437af68e4cc1cc4d5cd7875b8cb386a67487162c3941a3',
    '7f4468be0b0835a48996f92d1a2398f1290acbf1152d39e1c53c2701d1fb6ce3',
    '8764d43b87ba134cacf1be454d59d493630188e79c0d88639258b359bb511c4e',
    '92c3224da44f1a359140647845a44457df35541458b92485e7a3232bb596566a',
    '952a90a0ff27c2733d507e59bdca8600968276eddad9a41b47a689010e805118',
    '9d42e5db7cde5c7be0783c256e15e819bd09351fcb2debd3205e3d055073c993',
    'a2262bfa2bfacca3caf3a8334f02b9b52624056853cc33a1340a23c3a33804e6',
    'b441f782f07e81c2444ed7a5c4ed618c26ef3e1940ac2c63c362154620631750',
    'c887df022d6a2b33e78a3cf8f5dd65d9cd61f4d72009781d40c32091119a06f3'))

_init_lock = threading.Lock()
_initialized_mesh_dir = None


class ShadowArenaError(RuntimeError):
    pass


def mesh_dir_from(version_dir):
    """The collision-mesh folder (it holds soccar/): CHADGPT_COLLISION_MESHES, else collision_meshes/ in version_dir."""
    value = os.environ.get('CHADGPT_COLLISION_MESHES') or Path(version_dir) / 'collision_meshes'
    return Path(value).expanduser().resolve()


def check_meshes(mesh_dir):
    """(ok, reason, known): ok when soccar/ holds exactly 16 well-formed .cmf files (int32 triangle and vertex counts,
    then 3 int32 indices per triangle and 3 float32 per vertex); known = how many match KNOWN_SOCCAR_SHA256."""
    folder = Path(mesh_dir) / 'soccar'
    files = sorted(folder.glob('*.cmf')) if folder.is_dir() else []
    if len(files) != SOCCAR_MESH_COUNT:
        return False, f'{len(files)} of {SOCCAR_MESH_COUNT} soccar meshes in {folder}', 0
    known = 0
    for f in files:
        b = f.read_bytes()
        if len(b) < 8:
            return False, f'{f.name} is truncated', 0
        tris, verts = struct.unpack('<ii', b[:8])
        if tris <= 0 or verts <= 0 or len(b) != 8 + 12 * tris + 12 * verts:
            return False, f'{f.name} is not a collision mesh file ({len(b)} bytes)', 0
        known += hashlib.sha256(b).hexdigest() in KNOWN_SOCCAR_SHA256
    return True, f'{SOCCAR_MESH_COUNT} soccar meshes in {folder}', known


def init_rocketsim(mesh_dir):
    """Import RocketSim and load the soccar meshes, once per process (thread-safe). Raises ShadowArenaError."""
    global _initialized_mesh_dir
    try:
        import RocketSim as rs
    except ImportError as e:
        raise ShadowArenaError(f'the shadow arena needs the rocketsim Python package ({e})') from e
    mesh_dir = Path(mesh_dir)
    with _init_lock:
        if _initialized_mesh_dir is None:
            ok, reason, _ = check_meshes(mesh_dir)
            if not ok:
                raise ShadowArenaError(f'the shadow arena needs the RocketSim soccar collision meshes: {reason}; set '
                                       'CHADGPT_COLLISION_MESHES to the folder that holds soccar/')
            rs.init(str(mesh_dir))  # RocketSim's global init: once per process, whatever happens next
            _initialized_mesh_dir = mesh_dir
            # Self-check: a car resting on the floor must see the floor after one tick.
            arena = rs.Arena(rs.GameMode.SOCCAR)
            car = arena.add_car(0, rs.CarConfig.OCTANE)
            state = rs.CarState()
            state.pos = rs.Vec(0.0, -2000.0, 17.0)
            car.set_state(state)
            arena.step(2)
            if not car.get_state().is_on_ground:
                _initialized_mesh_dir = False  # initialized with meshes that fail the check: unusable this process
                raise ShadowArenaError(f'RocketSim loaded the meshes in {mesh_dir} but a resting car has no floor '
                                       'contact')
        elif _initialized_mesh_dir is False:
            raise ShadowArenaError('RocketSim was initialized with meshes that failed the floor check')
        elif _initialized_mesh_dir != mesh_dir:
            raise ShadowArenaError(f'RocketSim is already initialized with {_initialized_mesh_dir}')
    return rs


class Derived:
    __slots__ = ('on_ground', 'wheels', 'world_contact', 'normal', 'state')

    def __init__(self, on_ground, wheels, world_contact, normal, state=None):
        self.on_ground, self.wheels, self.world_contact, self.normal, self.state = (
            bool(on_ground), tuple(bool(w) for w in wheels), bool(world_contact), tuple(normal), state)


class ShadowArena:
    """One per TeamController (it models every car of the match, both teams). Call per packet, in this order:
    begin_packet(cars, clock_running) -> bool; if True apply(s, ground_evidence); decide; then
    queue_prediction(cars, packet_controls, current_controls, ball).
    required=False (CHADGPT_SHADOW_ARENA=auto): a failed sync falls back to the packet's contacts and is logged;
    required=True (CHADGPT_SHADOW_ARENA=1): it raises."""

    def __init__(self, mesh_dir, required=False, refresh_packets=1, log=None):
        self.mesh_dir = Path(mesh_dir)
        self.rs = init_rocketsim(self.mesh_dir)
        self.required = required
        self.refresh_packets = max(1, int(refresh_packets))
        self.log = log or (lambda msg: None)
        self.arena = None
        self.cars = {}          # car id -> RocketSim car
        self.teams = {}         # car id -> team
        self.last_positions = {}
        self.packets_since_step = 0
        self.packet_seq = 0
        self.pending_seq = 0
        self.queue_current = False
        self.derived = {}       # car id -> Derived
        self.derived_seq = 0
        self.snapshot_ready = False
        self.pending_failure = None
        self.failures = 0
        self.fallback_run = 0   # consecutive clock-running packets that used the packet's own contacts
        self.fallback_runs = 0
        self.applied = False    # the last apply() used the arena (for tracing)
        self.age = -1           # packets between the applied snapshot and the current packet

    def warm(self, cars):
        """Build the arena for this roster (none known yet: a 3v3) and take the first (slow, lazily initializing)
        step now, off the inference thread; call before the arena is handed to a controller."""
        self._recreate(cars or [dict(car_id=(2 + 2 * k) if team else (1 + 2 * k), team=team)
                                for team in (0, 1) for k in range(3)])
        self.arena.step(1)

    # -- per packet -----------------------------------------------------------------------------------------
    def begin_packet(self, cars, physics_advancing):
        """cars: the packet's cars after temporal reconstruction (PacketReader.last_cars)."""
        self.packet_seq += 1
        current = self.packet_seq
        self.pending_seq = current
        self.queue_current = False
        if self.pending_failure is not None:
            reason, self.pending_failure = self.pending_failure, None
            self._note_failure(reason)
            self.packets_since_step = 0
            self._publish_fallback(cars, current)
            self.queue_current = physics_advancing
            return True

        teleported = len(self.last_positions) != len(cars)
        positions = {}
        for car in cars:
            positions[car['car_id']] = car['pos']
            last = self.last_positions.get(car['car_id'])
            if last is None or float(np.sum((last - car['pos']) ** 2)) > TELEPORT_DISTANCE_SQ:
                teleported = True
        self.last_positions = positions

        if not physics_advancing:  # countdown: RocketSim has no frozen state, publish the packet's own contacts
            self.packets_since_step = 0
            self.fallback_run = 0
            self._publish_fallback(cars, current)
            return True
        ready = self._complete(cars)
        if teleported or not ready:
            self.packets_since_step = 0
            self._publish_fallback(cars, current)
            self.queue_current = True
            self._note_fallback('teleport' if teleported else 'no snapshot')
            return True
        self._end_fallback_run()
        self.packets_since_step += 1
        if self.packets_since_step >= self.refresh_packets:
            self.packets_since_step = 0
            self.queue_current = True
        return ready

    def apply(self, s, ground_evidence, cars):
        """Overwrite the contact fields of state s with the snapshot. A packet that says OnGround stays grounded
        with four wheels (and keeps the packet's floor contact)."""
        self.applied, self.age = False, -1
        if len(ground_evidence) != len(cars) or not self._complete(cars):
            return False
        for i, car in enumerate(cars):
            if car['is_demoed']:  # no contacts for a demolished car: keep the packet's
                continue
            if ground_evidence[i]:
                s['is_on_ground'][i] = True
                s['wheel_contacts'][i] = 4
                continue
            d = self.derived[car['car_id']]
            s['is_on_ground'][i] = d.on_ground
            s['wheel_contacts'][i] = sum(d.wheels)
            s['turtled'][i] = d.world_contact and d.normal[2] > TURTLE_NORMAL_Z
        self.applied, self.age = True, max(0, self.packet_seq - self.derived_seq)
        return True

    def queue_prediction(self, cars, previous_controls, current_controls, ball):
        """Step one tick from this packet with the controls every car holds next (previous_controls: the packet's
        last_input, which RocketSim needs as the car's last controls; current_controls: this process's new controls
        for its own cars, the packet's last_input for the others). ball: (pos, vel, ang_vel, (yaw, pitch, roll))."""
        if not self.queue_current:
            return False
        self.queue_current = False
        try:
            self._process(cars, previous_controls, current_controls, ball, self.pending_seq)
        except Exception as e:  # noqa: BLE001 - any RocketSim failure degrades to packet contacts, or aborts
            self.snapshot_ready = False
            self.pending_failure = f'{type(e).__name__}: {e}'
        return True

    # -- internals ----------------------------------------------------------------------------------------------
    def _note_failure(self, reason):
        self.failures += 1
        if self.failures == 1 or self.failures % LOG_EVERY == 0:
            self.log(f'shadow arena sync FAILED ({reason}) count={self.failures}; falling back to packet contacts')
        if self.required:
            raise ShadowArenaError(f'the shadow arena is required and failed to sync ({reason})')

    def _note_fallback(self, reason):
        """A clock-running packet that used the packet's own contacts. One such packet after a kickoff reset or a
        respawn is normal; a longer run means the arena is not keeping up, so runs of 2+ are logged (throttled)."""
        self.fallback_run += 1
        if self.fallback_run == 2:
            self.fallback_runs += 1
            if self.fallback_runs == 1 or self.fallback_runs % LOG_EVERY == 0:
                self.log(f'shadow arena: packet contacts for 2+ packets in a row ({reason}), '
                         f'run {self.fallback_runs}')
        elif self.fallback_run % LOG_EVERY == 0:
            self.log(f'shadow arena: packet contacts for {self.fallback_run} packets in a row ({reason})')

    def _end_fallback_run(self):
        if self.fallback_run >= LOG_EVERY:
            self.log(f'shadow arena: back on arena contacts after {self.fallback_run} packets')
        self.fallback_run = 0

    def _complete(self, cars):
        return self.snapshot_ready and len(self.derived) == len(cars) and all(c['car_id'] in self.derived
                                                                              for c in cars)

    def _publish_fallback(self, cars, seq):
        derived = {}
        for car in cars:
            g = bool(car['is_on_ground'])
            wc = bool(car['world_contact'])
            derived[car['car_id']] = Derived(g, (g, g, g, g), wc, (0.0, 0.0, 1.0) if wc else (0.0, 0.0, 0.0))
        self.derived, self.derived_seq, self.snapshot_ready = derived, seq, True

    def _roster_matches(self, cars):
        return (self.arena is not None and len(self.cars) == len(cars)
                and all(self.teams.get(c['car_id']) == c['team'] for c in cars))

    def _recreate(self, cars):
        rs = self.rs
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.cars = {c['car_id']: self.arena.add_car(int(c['team']), rs.CarConfig.OCTANE) for c in cars}
        self.teams = {c['car_id']: c['team'] for c in cars}

    def _controls(self, c, boosting_override=False):
        rs = self.rs
        return rs.CarControls(throttle=float(c[0]), steer=float(c[1]), pitch=float(c[2]), yaw=float(c[3]),
                              roll=float(c[4]), jump=bool(c[5] != 0), boost=bool(c[6] != 0) or boosting_override,
                              handbrake=bool(c[7] != 0))

    def _car_state(self, car, last_controls):
        rs = self.rs
        st = rs.CarState()
        st.pos = rs.Vec(*map(float, car['pos']))
        st.vel = rs.Vec(*map(float, car['vel']))
        st.ang_vel = rs.Vec(*map(float, car['ang_vel']))
        st.rot_mat = rs.Angle(*map(float, car['rot'])).as_rot_mat()
        st.is_on_ground = bool(car['is_on_ground'])
        st.wheels_with_contact = (False, False, False, False)
        st.has_jumped = bool(car['has_jumped'])
        st.has_double_jumped = bool(car['has_double_jumped'])
        st.has_flipped = bool(car['has_flipped'])
        st.flip_rel_torque = rs.Vec(*map(float, car['flip_rel_torque']))
        st.jump_time = float(car['jump_time'])
        st.flip_time = float(car['flip_time'])
        st.is_flipping = bool(car['is_flipping'])
        st.is_jumping = bool(car['is_jumping'])
        st.air_time = float(car['air_time'])
        st.air_time_since_jump = float(car['air_time_since_jump'])
        st.boost = float(car['boost'])
        st.boosting_time = float(car['boosting_time'])
        st.is_supersonic = bool(car['is_supersonic'])
        st.handbrake_val = float(car['handbrake'])
        st.has_world_contact = bool(car['world_contact'])
        st.world_contact_normal = rs.Vec(0.0, 0.0, 1.0) if car['world_contact'] else rs.Vec(0.0, 0.0, 0.0)
        st.is_demoed = bool(car['is_demoed'])
        if st.is_demoed:
            # Deliberate departure from the C++ bridge, which left the respawn timer at 0 so RocketSim respawned the
            # car at a spawn point during the step: keep it demolished (its contacts are not applied either).
            st.demo_respawn_timer = max(float(car.get('demo_timer', 0.0)), DEMO_HOLD_MIN)
        st.last_controls = last_controls
        return st

    def _process(self, cars, previous_controls, current_controls, ball, seq):
        rs = self.rs
        if not self._roster_matches(cars):
            self._recreate(cars)
        for i, car in enumerate(cars):
            rs_car = self.cars[car['car_id']]
            rs_car.set_state(self._car_state(car, self._controls(previous_controls[i])))
            # The Python binding does not expose CarState.isBoosting. RocketSim keeps boosting while
            # boostingTime < BOOST_MIN_TIME even without the boost input; holding boost for this tick gives the
            # same force and the same timers as setting isBoosting, which the bridge did.
            keep_boosting = bool(car['is_boosting']) and float(car['boosting_time']) < BOOST_MIN_TIME
            rs_car.set_controls(self._controls(current_controls[i], keep_boosting))
        pos, vel, ang_vel, rot = ball
        bs = rs.BallState()
        bs.pos, bs.vel, bs.ang_vel = rs.Vec(*map(float, pos)), rs.Vec(*map(float, vel)), rs.Vec(*map(float, ang_vel))
        bs.rot_mat = rs.Angle(*map(float, rot)).as_rot_mat()
        self.arena.ball.set_state(bs)
        self.arena.step(1)
        completed = {}
        for car_id, rs_car in self.cars.items():
            st = rs_car.get_state()
            completed[car_id] = Derived(st.is_on_ground, st.wheels_with_contact, st.has_world_contact,
                                        st.world_contact_normal.as_tuple(), st)
        if seq >= self.derived_seq:
            self.derived, self.derived_seq, self.snapshot_ready = completed, seq, True
