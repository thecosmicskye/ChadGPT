"""Turning the shadow arena on in a running bot, and the first-launch collision-mesh dump (Windows).

The shadow arena (chadgpt/shadow_arena.py) needs Rocket League's 16 soccar collision meshes, which are Psyonix data and
do not ship with the bot. When they are missing and Rocket League runs on Windows, a background thread dumps them from
the running game once with the bundled RLArenaCollisionDumper (third_party/RLArenaCollisionDumper, MIT), checks them,
loads them into RocketSim and hands a ready arena to the running controller, which swaps it in at a packet boundary.
Nothing here runs on the inference thread, and any failure leaves the bot playing on packet contacts.

Retry policy: one dump attempt per launch. Each attempt first bumps a counter in collision_meshes/dump_failures.txt
(so a crash counts too) and a good dump deletes it; after MAX_FAILED_LAUNCHES failed launches no more attempts are
made until that file is deleted.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .shadow_arena import SOCCAR_MESH_COUNT, ShadowArena, ShadowArenaError, check_meshes

DUMPER_NAME = 'RLArenaCollisionDumper.exe'
DUMPER_SHA256 = '91a40fc8d705a4c44b692c534f826253cf7b60b34bfacd4b90c6f6029875e5c5'  # v1.0.0 release asset
DUMP_TIMEOUT = 90.0  # seconds for the dumper run
GAME_WAIT = 300.0  # seconds to wait for Rocket League with a match running before giving up (not a failed launch)
PEER_WAIT = 300.0  # seconds to wait for another ChadGPT process that is dumping
MAX_FAILED_LAUNCHES = 3
FAILURE_MARKER = 'dump_failures.txt'
LOCK_NAME = 'dump.lock'
# flat.MatchPhase: the dump starts in a kickoff countdown or in play, when the game steps its physics every tick
PHASES_TICKING = (1, 2, 3)
# phases in which the cars' controls do not matter: inactive, countdown, goal scored, replay, ended
PHASES_QUIET = (0, 1, 4, 5, 7)
QUIET_WAIT = 30.0


def find_dumper():
    """The bundled dumper: third_party/RLArenaCollisionDumper/ in the repository, or inside a release build (stored
    there as .exe.bin so the bot's exe stays the only .exe in the build)."""
    roots = [Path(__file__).resolve().parents[1]]
    if getattr(sys, '_MEIPASS', None):
        roots.insert(0, Path(sys._MEIPASS))
    for root in roots:
        for name in (DUMPER_NAME, DUMPER_NAME + '.bin'):
            p = root / 'third_party' / 'RLArenaCollisionDumper' / name
            if p.is_file():
                return p
    return None


def meshes_loadable(mesh_dir):
    """check_meshes plus index bounds, so a bad dump never reaches RocketSim. (ok, reason, known)."""
    ok, reason, known = check_meshes(mesh_dir)
    if not ok:
        return ok, reason, known
    for f in sorted((Path(mesh_dir) / 'soccar').glob('*.cmf')):
        b = f.read_bytes()
        tris, verts = np.frombuffer(b[:8], '<i4')
        idx = np.frombuffer(b[8:8 + 12 * tris], '<i4')
        xyz = np.frombuffer(b[8 + 12 * tris:], '<f4')
        if idx.min() < 0 or idx.max() >= verts or not np.all(np.isfinite(xyz)):
            return False, f'{f.name} has out-of-range indices or non-finite vertices', 0
    return True, reason, known


# -- Windows helpers ---------------------------------------------------------------------------------------------------
def _no_window():
    return subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def rocket_league_pids():
    out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq RocketLeague.exe', '/FO', 'CSV', '/NH'],
                         capture_output=True, text=True, timeout=20, creationflags=_no_window()).stdout
    pids = []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 1 and parts[0].lower() == 'rocketleague.exe' and parts[1].isdigit():
            pids.append(int(parts[1]))
    return pids


def _process_dialog_text(pid):
    """Text of a visible top-level window owned by pid (the dumper's error message box), or None."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL('user32', use_last_error=True)  # private handle: argtypes set here stay local
    found = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [enum_proc, wintypes.LPARAM]
    user32.EnumChildWindows.argtypes = [wintypes.HWND, enum_proc, wintypes.LPARAM]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]

    def text_of(hwnd):
        buf = ctypes.create_unicode_buffer(1024)
        user32.GetWindowTextW(hwnd, buf, 1024)
        return buf.value

    def on_child(hwnd, _):
        t = text_of(hwnd)
        if t:
            found.append(t)
        return True

    def on_top(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(text_of(hwnd))
            user32.EnumChildWindows(hwnd, enum_proc(on_child), 0)
            return False
        return True

    user32.EnumWindows(enum_proc(on_top), 0)
    return ' | '.join(t for t in found if t) if found else None


def _resume_process(pid):
    """NtResumeProcess: undo a suspension the dumper may have left if it was killed while Rocket League was frozen."""
    import ctypes
    from ctypes import wintypes
    kernel32, ntdll = ctypes.WinDLL('kernel32'), ctypes.WinDLL('ntdll')
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    h = kernel32.OpenProcess(0x0800, False, pid)  # PROCESS_SUSPEND_RESUME
    if h:
        ntdll.NtResumeProcess(h)
        kernel32.CloseHandle(h)


def run_dumper(exe, workdir, rl_pid, log, timeout=DUMP_TIMEOUT):
    """Run the dumper in workdir; (ok, reason, seconds). Its output is workdir/collision_meshes/<mode>/mesh_N.cmf."""
    work_exe = Path(workdir) / DUMPER_NAME
    shutil.copyfile(exe, work_exe)
    lines = []
    t0 = time.perf_counter()
    proc = subprocess.Popen([str(work_exe)], cwd=workdir, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, creationflags=_no_window())
    reader = threading.Thread(target=lambda: lines.extend(l.decode('utf-8', 'replace').rstrip()
                                                          for l in proc.stdout), daemon=True)
    reader.start()
    reason = None
    while proc.poll() is None:
        if time.perf_counter() - t0 > timeout:
            reason = f'timed out after {timeout:.0f} s'
            break
        box = _process_dialog_text(proc.pid)
        if box:  # FATAL_ERROR shows a message box, then exits: close it now
            reason = f'dumper error: {box}'
            break
        time.sleep(0.1)
    if reason is not None:
        proc.kill()
        proc.wait(10)
        if 'timed out' in reason:
            _resume_process(rl_pid)
    reader.join(5)
    secs = time.perf_counter() - t0
    tail = ' / '.join(l for l in lines[-4:] if l)
    if reason is None and proc.returncode != 0:
        reason = f'dumper exit code {proc.returncode}'
    if reason is not None:
        return False, f'{reason} (output: {tail})', secs
    log(f'mesh dump: dumper finished in {secs:.1f} s ({tail})')
    return True, 'ok', secs


# -- the background enabler --------------------------------------------------------------------------------------------
class ShadowEnabler(threading.Thread):
    """Makes the shadow arena ready off the inference thread and hands it to controller.install_shadow."""

    def __init__(self, controller, mesh_dir, log, dump):
        super().__init__(name='chadgpt-shadow-arena', daemon=True)
        self.controller, self.mesh_dir, self.log, self.dump = controller, Path(mesh_dir), log, dump
        self.dumped = False  # this thread wrote the meshes (then a good RocketSim load clears the failure count)

    def run(self):
        try:
            if self._meshes_ready():
                self._enable()
        except Exception as e:  # noqa: BLE001 - never let this thread take the bot down
            self.log(f'shadow arena OFF: {type(e).__name__}: {e}')

    # -- steps ------------------------------------------------------------------------------------------------
    def _meshes_ready(self):
        ok, reason, _ = meshes_loadable(self.mesh_dir)
        if ok:
            return True
        if not self.dump:
            self.log(f'shadow arena OFF: no collision meshes ({reason}); set CHADGPT_COLLISION_MESHES to a folder '
                     'holding soccar/')
            return False
        return self._dump_once(reason)

    def _dump_once(self, why):
        from .agent import _try_lock
        marker = self.mesh_dir / FAILURE_MARKER
        failures = _read_count(marker)
        if failures >= MAX_FAILED_LAUNCHES:
            self.log(f'shadow arena OFF: the mesh dump failed on {failures} launches; delete {marker} to try again')
            return False
        exe = find_dumper()
        if exe is None:
            self.log(f'shadow arena OFF: no collision meshes ({why}) and no {DUMPER_NAME} bundled')
            return False
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        lock = _try_lock(self.mesh_dir / LOCK_NAME)
        if lock is None:
            return self._wait_for_peer()
        try:
            rl_pid = self._wait_for_game()
            if rl_pid is None:
                return False
            digest = hashlib.sha256(exe.read_bytes()).hexdigest()
            if digest != DUMPER_SHA256:
                self.log(f'shadow arena OFF: {exe} has sha256 {digest}, expected {DUMPER_SHA256}')
                return False
            _write_count(marker, failures + 1)  # counted before the run, so a crash counts as a failed launch
            self.log(f'mesh dump: no collision meshes ({why}); dumping them from Rocket League (pid {rl_pid}), '
                     f'attempt {failures + 1} of {MAX_FAILED_LAUNCHES}')
            with tempfile.TemporaryDirectory(prefix='chadgpt_meshes_') as tmp:
                ok, reason, secs = run_dumper(exe, tmp, rl_pid, self.log)
                if ok:
                    ok, reason = self._install(Path(tmp))
            if not ok:
                self.log(f'shadow arena OFF: the mesh dump failed ({reason}); failed launches '
                         f'{failures + 1} of {MAX_FAILED_LAUNCHES} ({marker})')
                return False
            ok, reason, known = meshes_loadable(self.mesh_dir)
            self.log(f'mesh dump: {reason}, {known} of {SOCCAR_MESH_COUNT} match the known RocketSim meshes')
            self.dumped = ok
            return ok
        finally:
            lock.close()

    def _install(self, tmp):
        """Move the dump's soccar meshes into mesh_dir/soccar (via a temporary folder, then a rename)."""
        base = next((tmp / d for d in ('collision_meshes', 'collision-meshes') if (tmp / d).is_dir()), None)
        if base is None:
            return False, 'the dumper wrote no collision_meshes folder'
        modes = sorted(p.name for p in base.iterdir() if p.is_dir())
        if 'soccar' not in modes:
            return False, f'the loaded arena is not soccar (dumped: {modes or "nothing"})'
        ok, reason, _ = meshes_loadable(base)
        if not ok:
            return False, reason
        target, partial = self.mesh_dir / 'soccar', self.mesh_dir / 'soccar.partial'
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(base / 'soccar', partial)
        if target.exists():
            shutil.rmtree(target)
        os.replace(partial, target)
        return True, 'ok'

    def _wait_for_game(self):
        """Rocket League's pid once it runs and this controller sees a countdown or play, else None."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < GAME_WAIT:
            pids = rocket_league_pids()
            if pids and self.controller.phase in PHASES_TICKING:
                return pids[0]
            time.sleep(0.5)
        self.log('shadow arena OFF: Rocket League (RocketLeague.exe) with a match running was not found; '
                 'no mesh dump this launch')
        return None

    def _wait_for_peer(self):
        """Another ChadGPT process holds the dump lock: wait for it to finish, then use its meshes if good."""
        from .agent import _try_lock
        self.log('mesh dump: another ChadGPT process is dumping; waiting for it')
        t0 = time.monotonic()
        while time.monotonic() - t0 < PEER_WAIT:
            time.sleep(2.0)
            lock = _try_lock(self.mesh_dir / LOCK_NAME)
            if lock is not None:
                lock.close()
                ok, reason, _ = meshes_loadable(self.mesh_dir)
                if not ok:
                    self.log(f'shadow arena OFF: the other process\'s mesh dump failed ({reason})')
                return ok
        self.log('shadow arena OFF: the other process\'s mesh dump did not finish')
        return False

    def _enable(self):
        """RocketSim init and a warmed arena, then handed to the controller. RocketSim holds the GIL through its init,
        the arena build and the first step (chunks of about 10-25 ms), which can delay the inference thread, so this
        runs while the cars' controls do not matter (inactive, kickoff countdown, goal and replay); after
        QUIET_WAIT seconds without such a window it runs right after a team decision instead."""
        t_wait = time.monotonic()
        while self.controller.phase not in PHASES_QUIET and time.monotonic() - t_wait < QUIET_WAIT:
            time.sleep(0.02)
        if self.controller.phase not in PHASES_QUIET:
            self.controller.decided.clear()
            self.controller.decided.wait(5.0)
        t0 = time.perf_counter()
        try:
            arena = ShadowArena(self.mesh_dir, required=False, log=self.log)
        except ShadowArenaError:
            if self.dumped:  # a dump RocketSim rejects: remove it so the next launch tries again (still counted)
                shutil.rmtree(self.mesh_dir / 'soccar', ignore_errors=True)
            raise
        arena.warm(self.controller.reader.last_cars)
        self.controller.install_shadow(arena, f'RocketSim ready in {(time.perf_counter() - t0) * 1e3:.0f} ms')
        if self.dumped:
            (self.mesh_dir / FAILURE_MARKER).unlink(missing_ok=True)


def _read_count(path):
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return 0


def _write_count(path, n):
    Path(path).write_text(f'{n}\nFailed collision-mesh dumps. Delete this file to let ChadGPT try again.\n')


def start_shadow_arena(controller, version_dir, log):
    """CHADGPT_SHADOW_ARENA: auto (default: on when meshes exist or can be dumped), 0 (off) or 1 (required: the meshes
    must be there and a failed arena step stops the bot). Returns a short status for the load log line."""
    from .shadow_arena import mesh_dir_from
    mode = os.environ.get('CHADGPT_SHADOW_ARENA', 'auto').strip().lower() or 'auto'
    mesh_dir = mesh_dir_from(version_dir)
    if mode == '0':
        log('shadow arena OFF (CHADGPT_SHADOW_ARENA=0)')
        return 'off'
    if mode == '1':
        arena = ShadowArena(mesh_dir, required=True, log=log)  # raises ShadowArenaError: the bot does not start
        controller.install_shadow(arena, 'required')
        return f'on, required ({mesh_dir})'
    ok, reason, _ = meshes_loadable(mesh_dir)
    dump = (not ok and os.name == 'nt' and 'CHADGPT_COLLISION_MESHES' not in os.environ)
    if ok:
        log(f'shadow arena: {reason}; loading RocketSim in the background')
    elif dump:
        log(f'shadow arena: no collision meshes yet ({reason}); will dump them from Rocket League in the background')
    ShadowEnabler(controller, mesh_dir, log, dump).start()
    return 'starting' if ok or dump else 'off'
