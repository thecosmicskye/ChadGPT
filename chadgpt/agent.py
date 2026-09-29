"""ChadGPT as an RLBot v5 hivemind.

The match logic lives in `TeamController`, which takes GamePackets and returns controls, so it can be driven
without a game. `ChadGPTHivemind` wires it to RLBot's Python interface.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .packet import PAD_FROM_PACKET, PacketReader
from .mesh_dump import meshes_loadable, start_shadow_arena
from .policy import ChadGPTModel, TeamPolicy, find_checkpoint
from .shadow_arena import ShadowArenaError, init_rocketsim, mesh_dir_from
from .state import empty_state

F = np.float32
TICK = F(1.0 / 120.0)
TICK_SKIP = 4
PRESERVE_RECURRENT_ON_GOAL = True
MEMORY_SAVE_SECONDS = 10.0  # persistent memory: the writer saves at most this often (wall clock), and at exit
REWIND_FRAMES, REWIND_SECONDS = 120, 1.0
MOVE_EPS_SQ = 1.0
# flat.MatchPhase values
PHASE_COUNTDOWN, PHASE_KICKOFF, PHASE_ACTIVE, PHASE_ENDED = 1, 2, 3, 7


def env_flag(name, default):
    return os.environ.get(name, default).strip() not in ('', '0')


def sample_state():
    """A standing 3v3 (blue at y < 0, orange at y > 0), for warm-up and the self-check."""
    s = empty_state(6)
    s['team'][:] = [0, 0, 0, 1, 1, 1]
    s['car_id'][:] = [1, 3, 5, 2, 4, 6]
    s['pos'][:, 1] = [-1000, -2000, -3000, 1000, 2000, 3000]
    s['forward'][:, 0] = 1.0
    s['up'][:, 2] = 1.0
    return s


@dataclass
class BotSlot:
    index: int
    ticks: int = -1
    update: bool = True
    action: np.ndarray = field(default_factory=lambda: np.zeros(8, np.float32))
    controls: np.ndarray = field(default_factory=lambda: np.zeros(8, np.float32))
    last_packet_frame: int = -1
    last_head: int = -1

    def reset(self, frame, clear_head=False):
        self.ticks, self.update = -1, True
        self.action = np.zeros(8, np.float32)
        self.controls = np.zeros(8, np.float32)
        self.last_packet_frame = frame
        if clear_head:
            self.last_head = -1


def _try_lock(path):
    """Open and exclusively lock `path` without waiting; the open file (keep it) or None if another process holds it.
    The OS drops the lock when the process exits, so a crashed writer never blocks the next one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except OSError:
        f.close()
        return None


def _vsq(v):
    return v.x * v.x + v.y * v.y + v.z * v.z


class TeamController:
    """Per-packet match logic for the cars of one team."""

    def __init__(self, model: ChadGPTModel, team: int, indices, log=None, memory_file=None):
        self.policy = TeamPolicy(model, team)
        self.team = team
        self.bots = {i: BotSlot(i) for i in indices}
        self.reader = PacketReader()
        self.log = log or (lambda msg: None)
        self.last_blue = self.last_orange = -1
        self.last_batch_frame, self.last_batch_time = -1, 0.0
        self.last_temporal_frame = -1
        self.clock_running_last = False
        self.loop_running_last = False
        self.saw_countdown = False
        self.movement_started = False
        self.have_last_physics = False
        self.last_positions, self.last_ball = [], None
        self.round_active_start = -1
        self.decisions = 0
        self.last_decision = None  # (frame, heads, actions) of the latest decision, for tracing
        self.reasons, self.heads = {}, {}  # why each car re-decided this packet (h head change, t cadence, k kickoff)
        # One team decision every 4 frames, as in training (default); CHADGPT_TEAM_CADENCE=0: the bridge's per-car rhythm
        self.bridge_cadence = not env_flag('CHADGPT_TEAM_CADENCE', '1')
        self.goal_log = os.environ.get('CHADGPT_GOAL_LOG') or None  # optional: append one line per goal
        self.state_hook = None  # optional callable(state) for offline checks; never set by the live bot
        self.controls_hook = None  # optional callable(bots) that may replace the held controls (offline checks)
        # Shadow arena (chadgpt/shadow_arena.py): None until chadgpt.mesh_dump hands one over (install_shadow), then
        # swapped in at the next packet boundary. phase and decided let that background thread time its work.
        self.shadow = None
        self._pending_shadow = None
        self.phase = -1
        self.decided = threading.Event()
        self._warm_up()
        # Persistent memory (live bot only): the recurrent state is never cleared, is loaded from memory_file when the
        # model loads, and is saved by ONE writer process (the first to lock writer.lock next to it); other processes
        # only load it. CHADGPT_PERSIST_MEMORY=0 restores the per-match resets.
        self.memory_file = Path(memory_file) if memory_file and env_flag('CHADGPT_PERSIST_MEMORY', '1') else None
        self.memory_writer = None  # the open, locked writer.lock when this process is the writer
        self.last_memory_save = 0.0
        if self.memory_file is not None:
            self._load_memory()
            self.memory_writer = _try_lock(self.memory_file.parent / 'writer.lock')
            self.log(f'memory: {"writer" if self.memory_writer else "read-only (another process writes)"}')

    def _load_memory(self):
        """memory_file (recurrent.npy) if present, else the shipped warmed-up seed.npy next to it, else zeros."""
        import torch
        seed = self.memory_file.with_name('seed.npy')
        path = self.memory_file if self.memory_file.exists() else seed
        try:
            h = np.load(path)
        except FileNotFoundError:
            self.log(f'memory: starting fresh (no {self.memory_file.name} or {seed.name} in {seed.parent})')
            return
        except Exception as e:  # unreadable file: keep the fresh state rather than crash the bot
            self.log(f'memory: could not read {path} ({e}); starting fresh')
            return
        if h.shape != (self.policy.h.shape[-1],) or not np.all(np.isfinite(h)):
            self.log(f'memory: ignoring {path} (shape {h.shape})')
            return
        self.policy.h = torch.as_tensor(h, dtype=self.policy.h.dtype, device=self.policy.h.device).reshape(self.policy.h.shape)
        self.log(f'memory: loaded {path.name} ({path})')

    def save_memory(self, force=False):
        if self.memory_writer is None:
            return
        now = time.monotonic()
        if not force and now - self.last_memory_save < MEMORY_SAVE_SECONDS:
            return
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.memory_file.with_name(self.memory_file.name + '.tmp')
        with open(tmp, 'wb') as f:
            np.save(f, self.policy.h.detach().float().cpu().numpy().reshape(-1))
        os.replace(tmp, self.memory_file)  # atomic: a crash never leaves a half-written file
        self.last_memory_save = now

    def install_shadow(self, arena, note=''):
        """Hand over a ready ShadowArena (any thread); it takes effect at the start of the next packet."""
        self._pending_shadow = (arena, note)

    def _warm_up(self):
        """One throw-away decision so the first real one is not slowed by lazy allocation."""
        s = sample_state()
        for _ in range(2):
            self.policy.update_roles(s)
            self.policy.decide(s, commit=False)
        self.policy.clear_roles()

    # -- resets -------------------------------------------------------------------------------------------------
    def _reset_inactive(self, frame, reset_context=False, reset_recurrent=True):
        if reset_recurrent and self.memory_file is None:  # persistent memory is never cleared
            self.policy.reset_recurrent()
        if reset_context:
            self.policy.reset_context()
        self.reader.clear()
        self.last_temporal_frame = -1
        self.clock_running_last = self.loop_running_last = False
        self.saw_countdown = self.movement_started = self.have_last_physics = False
        self.round_active_start = -1
        self.last_positions, self.last_ball = [], None
        self.policy.clear_roles()
        for b in self.bots.values():
            b.reset(frame)

    # -- packet helpers -----------------------------------------------------------------------------------------
    @staticmethod
    def _stationary(packet):
        if not len(packet.balls):
            return False
        for p in packet.players:
            if _vsq(p.physics.velocity) > MOVE_EPS_SQ or _vsq(p.physics.angular_velocity) > MOVE_EPS_SQ:
                return False
        b = packet.balls[0].physics
        return _vsq(b.velocity) <= MOVE_EPS_SQ and _vsq(b.angular_velocity) <= MOVE_EPS_SQ

    def _movement(self, packet):
        """True once anything has moved since the round's first packet (sticky until a reset)."""
        if not len(packet.balls):
            return False
        positions = [p.physics.location for p in packet.players]
        ball = packet.balls[0].physics
        moved = any(_vsq(p.physics.velocity) > MOVE_EPS_SQ for p in packet.players) \
            or _vsq(ball.velocity) > MOVE_EPS_SQ
        if self.have_last_physics and len(self.last_positions) == len(positions):
            lb = self.last_ball
            if (ball.location.x - lb[0]) ** 2 + (ball.location.y - lb[1]) ** 2 + (ball.location.z - lb[2]) ** 2 \
                    > MOVE_EPS_SQ:
                moved = True
            for p, q in zip(positions, self.last_positions):
                if (p.x - q[0]) ** 2 + (p.y - q[1]) ** 2 + (p.z - q[2]) ** 2 > MOVE_EPS_SQ:
                    moved = True
                    break
        self.last_positions = [(p.x, p.y, p.z) for p in positions]
        self.last_ball = (ball.location.x, ball.location.y, ball.location.z)
        self.have_last_physics = True
        self.movement_started = self.movement_started or moved
        return self.movement_started

    # -- main entry ---------------------------------------------------------------------------------------------
    def step(self, packet):
        """GamePacket -> {car index: (8,) controls}."""
        mi = packet.match_info
        frame, now = int(mi.frame_num), float(mi.seconds_elapsed)
        self.phase = int(mi.match_phase)
        if self._pending_shadow is not None:  # packet boundary: the shadow arena takes over from here
            (self.shadow, note), self._pending_shadow = self._pending_shadow, None
            self.log(f'shadow arena ON at frame {frame} ({self.shadow.mesh_dir}{"; " + note if note else ""})')
        if self.last_batch_frame >= 0 and frame <= self.last_batch_frame:
            if frame == self.last_batch_frame:
                return self.outputs()
            if not (self.last_batch_frame - frame >= REWIND_FRAMES and self.last_batch_time - now >= REWIND_SECONDS):
                return self.outputs()  # stale packet
            self.log(f'timeline reset at frame {frame}')
            self._reset_inactive(frame, reset_context=True, reset_recurrent=True)
            self.last_blue = self.last_orange = -1
        self._compute(packet, frame)
        self.last_batch_frame, self.last_batch_time = frame, now
        if self.memory_writer is not None:
            self.save_memory()
        return self.outputs()

    def outputs(self):
        return {i: b.controls for i, b in self.bots.items()}

    def _compute(self, packet, frame):
        phase = int(packet.match_info.match_phase)
        if phase == PHASE_ENDED:
            self._reset_inactive(frame, reset_context=True, reset_recurrent=True)
            return
        round_active = phase in (PHASE_KICKOFF, PHASE_ACTIVE)
        countdown = phase == PHASE_COUNTDOWN
        blue = orange = 0
        for t in packet.teams:
            if t.team_index == 0:
                blue = t.score
            elif t.team_index == 1:
                orange = t.score
        score_changed = self.last_blue >= 0 and (blue != self.last_blue or orange != self.last_orange)
        blue_goal = self.last_blue >= 0 and blue > self.last_blue
        orange_goal = self.last_orange >= 0 and orange > self.last_orange
        if score_changed:
            new_match = blue < self.last_blue or orange < self.last_orange
            self.log(f'score {blue}-{orange} at frame {frame}{" (new match)" if new_match else ""}')
            if self.goal_log:  # the C++ bridge's kickoff telemetry line, read by RLBot evaluation scripts
                since = frame - self.round_active_start if self.round_active_start >= 0 else -1
                with open(self.goal_log, 'a') as f:
                    f.write(f'RLBot goal frame={frame} framesSinceKickoffRelease={since} score={blue}-{orange} '
                            f'scorer={"blue" if blue > self.last_blue else "orange"}\n')
            self._reset_inactive(frame, reset_context=new_match,
                                 reset_recurrent=new_match or not PRESERVE_RECURRENT_ON_GOAL)
            self.last_blue, self.last_orange = blue, orange
        released = round_active and not countdown
        if countdown:
            self.saw_countdown = True
        if released and self.saw_countdown:
            self.saw_countdown = False
        if self.round_active_start < 0 and not round_active and not self._stationary(packet):
            for b in self.bots.values():  # goal / replay: idle with zero controls until the kickoff setup
                b.reset(frame)
            self.policy.clear_roles()
            self.decided.set()  # no decisions during the replay: a quiet time for background work
            return
        if self.round_active_start < 0 and released and not score_changed:
            self.round_active_start = frame
        moving = self._movement(packet)
        clock = (not countdown) and moving
        just_started_clock = clock and not self.clock_running_last
        just_started_loop = not self.loop_running_last

        dt = F(0.0)
        if clock and not just_started_clock and self.last_temporal_frame >= 0:
            dt = F(max(0, frame - self.last_temporal_frame)) * TICK
        self.last_temporal_frame = frame if clock else -1
        s = self.reader.read(packet, dt, integrate=clock)
        self.last_blue, self.last_orange = blue, orange
        goal = blue_goal or orange_goal
        if dt > 0 or goal:
            self.policy.fold_context(s, dt, 1.0 if blue_goal else (-1.0 if orange_goal else 0.0))
        # Previous actions as in training: every car's controls over the last step. The packet's last_input for the
        # other cars (opponents included; PacketReader.read fills them), the held controls for this process's own cars.
        for i, b in self.bots.items():
            if i < len(s['prev']):
                s['prev'][i] = b.controls
        cars = self.reader.last_cars
        if self.shadow is not None and self.shadow.begin_packet(cars, clock):
            self.shadow.apply(s, self.reader.last_ground_evidence, cars)
        if self.state_hook is not None:
            self.state_hook(s)

        # Cadence and roles
        if just_started_loop:
            self.policy.clear_roles()
        release_frame = self.round_active_start >= 0 and frame - self.round_active_start <= 1
        for b in self.bots.values():
            delta = frame - b.last_packet_frame if b.last_packet_frame >= 0 else -1
            b.last_packet_frame = frame
            if just_started_loop:
                b.ticks, b.update, b.last_head = 0, True, -1
            else:
                b.ticks += max(0, delta) if delta >= 0 else 0
        pending = []
        self.reasons = {}
        if self.bridge_cadence:
            # C++ bridge rhythm: roles every packet; each car re-decides on its own 4-frame count or head change.
            roles = self.policy.update_roles(s)
            heads = {car: h for h, car in enumerate(roles)}
            for i, b in self.bots.items():
                if i not in heads:
                    b.controls = np.zeros(8, np.float32)
                    continue
                head = heads[i]
                if b.last_head != head or b.ticks >= TICK_SKIP or release_frame:
                    self.reasons[i] = 'h' if b.last_head != head else ('t' if b.ticks >= TICK_SKIP else 'k')
                    b.update, b.ticks = True, 0
                if b.update:
                    pending.append(i)
        else:
            # Training rhythm: the whole team decides together every 4 frames (roles, one forward pass, one
            # memory update), holding every car's controls in between.
            # At the kickoff release the team decides once (frame 0 of the round), which sets the 4-frame phase.
            release_frame = self.round_active_start == frame
            due = release_frame or any(b.update or b.ticks >= TICK_SKIP for b in self.bots.values())
            if due or self.policy.roles is None:
                roles = self.policy.update_roles(s)
                why = 'k' if release_frame else 't'
                for b in self.bots.values():
                    b.update, b.ticks = True, 0
            heads = {car: h for h, car in enumerate(self.policy.roles or [])}
            for i, b in self.bots.items():
                if i not in heads:
                    b.controls = np.zeros(8, np.float32)
                    continue
                if b.update:
                    self.reasons[i] = why
                    pending.append(i)
        self.heads = heads
        if pending:
            res = self.policy.decide(s, commit=clock)
            self.decisions += 1
            for i in pending:
                b = self.bots[i]
                b.action = res['controls'][i].astype(np.float32)
                b.last_head, b.update = heads[i], False
            self.last_decision = (frame, dict(heads), {i: int(res['actions'][i]) for i in pending})
        for b in self.bots.values():
            b.controls = b.action.copy()  # no action delay
        if self.controls_hook is not None:
            self.controls_hook(self.bots)
        if self.shadow is not None:
            # Predict the next packet's contacts from this one with the controls every car now holds.
            packet_controls = self.reader.last_controls
            current = packet_controls.copy()
            for i, b in self.bots.items():
                if i < len(current):
                    current[i] = b.controls
            ph = packet.balls[0].physics if len(packet.balls) else None
            ball = ((ph.location.x, ph.location.y, ph.location.z), (ph.velocity.x, ph.velocity.y, ph.velocity.z),
                    (ph.angular_velocity.x, ph.angular_velocity.y, ph.angular_velocity.z),
                    (ph.rotation.yaw, ph.rotation.pitch, ph.rotation.roll)) if ph is not None else (
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            self.shadow.queue_prediction(cars, packet_controls, current, ball)
        if pending:
            self.decided.set()
        self.clock_running_last = clock
        self.loop_running_last = True


def raise_priority(model, log):
    """What the C++ bridge did on Windows with a GPU (GGL_RLBOT_HIGH_PRIORITY and _REQUIRED, _CUDA_STREAM): high
    process priority, the highest priority for the inference thread and a high-priority CUDA stream. Failing to get
    them is an error, as it was for the bridge. CHADGPT_HIGH_PRIORITY=0 skips this."""
    if os.name != 'nt' or model.device.type != 'cuda' or not env_flag('CHADGPT_HIGH_PRIORITY', '1'):
        return False
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    if not kernel32.SetPriorityClass(ctypes.c_void_p(kernel32.GetCurrentProcess()), 0x00000080):  # HIGH_PRIORITY_CLASS
        raise OSError(f'could not set high process priority (Windows error {kernel32.GetLastError()})')
    if not kernel32.SetThreadPriority(ctypes.c_void_p(kernel32.GetCurrentThread()), 2):  # THREAD_PRIORITY_HIGHEST
        raise OSError(f'could not set the inference thread priority (Windows error {kernel32.GetLastError()})')
    model.use_high_priority_stream()
    log('high process and thread priority, high-priority CUDA stream')
    return True


def to_controller(flat, c):
    return flat.ControllerState(throttle=float(c[0]), steer=float(c[1]), pitch=float(c[2]), yaw=float(c[3]),
                                roll=float(c[4]), jump=bool(c[5]), boost=bool(c[6]), handbrake=bool(c[7]),
                                use_item=False)


def make_hivemind_class():
    import torch
    from rlbot import flat
    from rlbot.managers import Hivemind

    class ChadGPTHivemind(Hivemind):
        version_dir: Path = Path('.')
        device: str = 'auto'
        threads: int | str | None = None  # None / 'auto': chadgpt.policy.resolve_threads
        precision: str | None = None  # None: CHADGPT_PRECISION, else bf16

        def initialize(self):
            t0 = time.perf_counter()
            ckpt = find_checkpoint(self.version_dir)
            self.model = ChadGPTModel(ckpt, self.device, threads=self.threads, precision=self.precision)
            self.controller = TeamController(self.model, self.team, self.indices, log=self._logger.info,
                                             memory_file=Path(self.version_dir) / 'memory' / 'recurrent.npy')
            shadow = start_shadow_arena(self.controller, self.version_dir, self._logger.info)
            self._priority_set = False
            field_xy = [(p.location.x, p.location.y) for p in self.field_info.boost_pads]
            if len(field_xy) == 34:
                order = self.controller.reader.set_field_pads(field_xy)
                if not np.array_equal(order, PAD_FROM_PACKET):
                    self._logger.warning('FieldInfo boost pads are not in the standard RLBot order; mapped by position')
            self._logger.info('ChadGPT %s (%s) loaded on %s (%s) in %.1f s for team %d, cars %s; shadow arena %s',
                              self.version_dir.name, ckpt.name, self.model.device, self.model.precision,
                              time.perf_counter() - t0, self.team, self.indices, shadow)
            self.trace = None
            trace_dir = os.environ.get('CHADGPT_TRACE_DIR')
            if trace_dir:  # diagnostics only: one JSON line per packet
                Path(trace_dir).mkdir(parents=True, exist_ok=True)
                self.trace = open(Path(trace_dir) / f'trace_team{self.team}_{os.getpid()}.jsonl', 'w', buffering=1 << 16)
                pads = [[p.location.x, p.location.y, p.location.z, bool(p.is_full_boost)]
                        for p in self.field_info.boost_pads]
                self.trace.write(json.dumps(dict(event='init', team=self.team, indices=list(self.indices),
                                                 device=str(self.model.device), precision=self.model.precision,
                                                 threads=torch.get_num_threads(),
                                                 field_pads=pads)) + '\n')
                self._trace_n = 0

        def get_outputs(self, packet):
            if not self._priority_set:  # on the thread that runs inference, like the bridge
                self._priority_set = True
                raise_priority(self.model, self._logger.info)
            t0 = time.perf_counter()
            before = self.controller.decisions
            controls = self.controller.step(packet)
            out = {i: to_controller(flat, c) for i, c in controls.items()}
            if self.trace is not None:
                mi = packet.match_info
                row = dict(f=int(mi.frame_num), ph=int(mi.match_phase),
                           sc=[int(t.score) for t in packet.teams], ms=round((time.perf_counter() - t0) * 1e3, 3),
                           dec=self.controller.decisions - before,
                           sa=None if self.controller.shadow is None else int(self.controller.shadow.applied),
                           h={i: self.controller.heads.get(i, -1) for i in self.indices},
                           why=self.controller.reasons,
                           c={i: [round(float(x), 2) for x in c] for i, c in controls.items()})
                if self._trace_n % 30 == 0:
                    row['pads'] = [[int(bp.is_active), round(float(bp.timer), 4)] for bp in packet.boost_pads]
                self._trace_n += 1
                self.trace.write(json.dumps(row) + '\n')
            return out

        def retire(self):
            if getattr(self, 'controller', None) is not None:
                self.controller.save_memory(force=True)
            if getattr(self, 'trace', None) is not None:
                self.trace.close()

    return ChadGPTHivemind


def self_check(args):
    """Offline check (no game): load and verify the weights, start the hivemind class, make one team decision."""
    t0 = time.perf_counter()
    make_hivemind_class()  # imports torch and RLBot's interface as the bot does
    ckpt = find_checkpoint(args.version_dir)
    model = ChadGPTModel(ckpt, args.device, threads=args.threads, precision=args.precision)
    ctl = TeamController(model, 0, [0, 1, 2])
    s = sample_state()
    t1 = time.perf_counter()
    ctl.policy.update_roles(s)
    res = ctl.policy.decide(s, commit=False)
    ms = (time.perf_counter() - t1) * 1e3
    import torch
    where = 'bundled' if getattr(sys, 'frozen', False) else Path(args.version_dir).name
    print(f'ChadGPT check: {where} checkpoint {ckpt.name}, {len(model.hashes)} files match '
          f'SHA256SUMS; {model.device} {model.precision}'
          f'{f" ({model.cpu_weights} weights, timed {model.cpu_weight_ms})" if model.cpu_weights else ""}, '
          f'{torch.get_num_threads()} thread(s); one team decision '
          f'{ms:.1f} ms, actions {[int(res["actions"][i]) for i in range(3)]}; ready in {time.perf_counter() - t0:.1f} s. OK',
          flush=True)
    print(f'Shadow arena: {shadow_check(args.version_dir)}', flush=True)
    return 0


def shadow_check(version_dir):
    """The shadow arena's status for --check: RocketSim, the meshes, and a resting car's floor contact."""
    mode = os.environ.get('CHADGPT_SHADOW_ARENA', 'auto').strip().lower() or 'auto'
    if mode == '0':
        return 'off (CHADGPT_SHADOW_ARENA=0)'
    try:
        import RocketSim  # noqa: F401
    except ImportError as e:
        return f'unavailable (rocketsim package missing: {e})'
    mesh_dir = mesh_dir_from(version_dir)
    ok, reason, known = meshes_loadable(mesh_dir)
    if not ok:
        return (f'no meshes yet ({reason}); '
                + ('dumped from Rocket League on the first match' if os.name == 'nt' else 'off'))
    try:
        init_rocketsim(mesh_dir)
    except ShadowArenaError as e:
        return f'meshes do not load: {e}'
    return f'ready ({reason}, {known} of 16 known; a resting car touches the floor)'


def main(argv=None):
    ap = argparse.ArgumentParser(description='Run ChadGPT as an RLBot v5 hivemind.')
    frozen = getattr(sys, '_MEIPASS', None)  # a PyInstaller build carries its version folder inside
    ap.add_argument('--version-dir', default=os.environ.get('CHADGPT_VERSION_DIR') or frozen or '.',
                    help='version folder holding checkpoint/ and SHA256SUMS (default: current directory, or the '
                         'weights inside a release build)')
    ap.add_argument('--device', default=os.environ.get('CHADGPT_DEVICE', 'auto'),
                    help='auto (default: cuda when available, else cpu), cuda or cpu')
    ap.add_argument('--threads', default=os.environ.get('CHADGPT_THREADS', 'auto'),
                    help='torch CPU threads: auto (default; see chadgpt.policy.AUTO_THREADS) or a number')
    ap.add_argument('--precision', default=os.environ.get('CHADGPT_PRECISION', 'bf16'), choices=('bf16', 'fp32'),
                    help='bf16 (default): the C++ bridge\'s bfloat16 autocast (emulated on the CPU); fp32: float32')
    ap.add_argument('--agent-id', default='skye/chadgpt')
    ap.add_argument('--check', action='store_true', help='offline self-check (no game): load the weights, one decision')
    args = ap.parse_args(argv)
    if args.check:
        return self_check(args)
    cls = make_hivemind_class()
    cls.version_dir = Path(args.version_dir).resolve()
    cls.device, cls.threads, cls.precision = args.device, args.threads, args.precision
    try:
        cls(args.agent_id).run(wants_match_communications=False, wants_ball_predictions=False)
    except ConnectionError as e:  # the server closed the session (match stopped) while controls were being sent
        print(f'ChadGPT: connection to RLBot closed ({e}); exiting', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
