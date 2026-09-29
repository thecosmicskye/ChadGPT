"""Stateful team policy: recurrent memory, role history and context for one team."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from .actions import ACTION_TABLE, MIRRORED_CONTROLS
from .model import HIDDEN, load_checkpoint, parse_precision
from .observation import action_masks, fold_context, team_observation
from .roles import assign_roles, opponent_order, trajectory_times


# CPU threads for 'auto'. float32: a batch-1 forward is a
# chain of matrix-vector products bound by memory bandwidth, so more threads do not help (Ryzen 7 9800X3D, Apple M4
# Pro). bfloat16 (bf16 weights, oneDNN): one thread gains little over float32; four take about half the time
# (Ryzen 7 9800X3D); more help little and would compete with the game.
AUTO_THREADS = {'fp32': 1, 'bf16': 4}


CPU_WEIGHTS = ('bf16', 'fp32')


def choose_cpu_weights(model, choice=None, rounds=3, n=8):
    """Weight copies of the CPU's emulated bf16 path (CHADGPT_CPU_WEIGHTS): 'bf16' or 'fp32' as given, or 'auto'
    (default): time team decisions with each, alternating, and keep the faster. bfloat16 copies halve the bytes per
    decision (about half the time on AVX-512 BF16 x86 and Apple CPUs); on CPUs without fast bfloat16 kernels (e.g.
    AVX2-only x86) the float32 copies of the same values can be faster. Both round like CUDA's bf16 autocast; they
    differ only in summation order. Returns (choice, {weights: median ms} or None)."""
    choice = (choice or 'auto').strip().lower()
    if choice in CPU_WEIGHTS:
        model.net.cpu_weights = choice
        return choice, None
    if choice != 'auto':
        raise ValueError(f'CHADGPT_CPU_WEIGHTS must be auto, bf16 or fp32, not {choice!r}')
    obs, h = np.zeros((1, 396), np.float32), torch.zeros(1, HIDDEN, device=model.device)
    times = {w: [] for w in CPU_WEIGHTS}
    for w in CPU_WEIGHTS:   # build each set of copies and warm it up
        model.net.cpu_weights = w
        for _ in range(2):
            model.forward(obs, h)
    for _ in range(rounds):
        for w in CPU_WEIGHTS:
            model.net.cpu_weights = w
            for _ in range(n):
                t = time.perf_counter()
                model.forward(obs, h)
                times[w].append(time.perf_counter() - t)
    ms = {w: round(1e3 * sorted(v)[len(v) // 2], 2) for w, v in times.items()}
    best = min(CPU_WEIGHTS, key=ms.get)
    model.net.cpu_weights = best
    cache = model.net.__dict__.get('_bf16_programs', {})
    for key in [k for k in cache if k[1] != best]:   # free the copies not used
        del cache[key]
    return best, ms


def resolve_threads(threads, precision='bf16'):
    """'auto' (or None, 0, '') -> min(AUTO_THREADS[precision], CPU count); a number is used as given."""
    if threads in (None, '', 0, 'auto'):
        return min(AUTO_THREADS[precision], os.cpu_count() or 1)
    return max(1, int(threads))


def resolve_device(device):
    """'auto' (or empty) -> 'cuda' when torch sees a CUDA GPU, else 'cpu'."""
    if not device or device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def find_checkpoint(version_dir):
    """The single checkpoint folder under <version_dir>/checkpoint/."""
    folders = sorted(p for p in (Path(version_dir) / 'checkpoint').iterdir() if (p / 'META.json').is_file())
    if len(folders) != 1:
        raise ValueError(f'expected one checkpoint under {version_dir}/checkpoint, found {[p.name for p in folders]}')
    return folders[0]


class ChadGPTModel:
    """The network plus the fixed tables. Shared by both teams when one process drives both."""

    def __init__(self, checkpoint_dir, device='auto', threads=None, verify=True, precision=None):
        """device: 'auto' (CUDA when available, like the C++ bridge, else CPU), 'cuda' or 'cpu'.
        precision: 'bf16' (default: the bridge's bfloat16 autocast) or 'fp32'; None reads CHADGPT_PRECISION."""
        self.precision = parse_precision(os.environ.get('CHADGPT_PRECISION') if precision is None else precision)
        torch.set_num_threads(resolve_threads(threads, self.precision))
        try:
            torch.set_num_interop_threads(1)  # one forward pass at a time: no inter-op parallelism to use
        except RuntimeError:  # already set, or parallel work already ran in this process
            pass
        self.device = torch.device(resolve_device(device))
        self.net, self.meta, self.hashes = load_checkpoint(checkpoint_dir, self.device, verify=verify)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.net.precision = self.precision
        self.stream = None
        self.cpu_weights, self.cpu_weight_ms = None, None
        if self.device.type == 'cpu' and self.precision == 'bf16':
            self.cpu_weights, self.cpu_weight_ms = choose_cpu_weights(self, os.environ.get('CHADGPT_CPU_WEIGHTS'))
        for _ in range(3):  # warm-up: the first forward pass allocates and is several times slower
            self.forward(np.zeros((1, 396), np.float32), torch.zeros(1, HIDDEN, device=self.device))

    @torch.no_grad()
    def forward(self, obs, h):
        """obs (G, 396) numpy, h (G, 64) torch -> logits (G, 4, 324) torch float32 on CPU, next h."""
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                return self._forward(obs, h)
        return self._forward(obs, h)

    def _forward(self, obs, h):
        x = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).to(self.device)
        logits, h_next = self.net(x, h)
        return logits.float().cpu(), h_next

    def use_high_priority_stream(self):
        """Run inference on a high-priority CUDA stream (the C++ bridge's GGL_RLBOT_HIGH_PRIORITY_CUDA_STREAM)."""
        if self.device.type == 'cuda':
            self.stream = torch.cuda.Stream(device=self.device, priority=-1)


def deterministic_action(logits, mask):
    """Argmax of the masked, clamped softmax (first index on ties): the deterministic choice."""
    masked = logits + -1e10 * (~torch.as_tensor(mask)).float()
    probs = torch.softmax(masked, -1).clamp(1e-11, 1.0).masked_fill(~torch.as_tensor(mask), 0.0)
    probs = probs / probs.sum(-1, keepdim=True)
    return int(probs.argmax(-1)), probs


class TeamPolicy:
    """Decisions for the cars of one team, carrying the team's recurrent state, role history and context."""

    def __init__(self, model: ChadGPTModel, team: int):
        self.model, self.team = model, team
        self.reset()

    def reset(self):
        """Match start: everything cleared."""
        self.h = torch.zeros(1, HIDDEN, device=self.model.device)
        self.context = np.zeros(15, np.float32)
        self.roles = None

    def reset_recurrent(self):
        self.h = torch.zeros(1, HIDDEN, device=self.model.device)

    def reset_context(self):
        self.context = np.zeros(15, np.float32)

    def clear_roles(self):
        self.roles = None

    def fold_context(self, s, dt, impulse=0.0, keep=None):
        self.context = fold_context(self.context, s, dt, impulse, keep)

    def update_roles(self, s, keys=None):
        """Evaluate the team's role order (with head-0 hysteresis) and store it as the new history."""
        self.roles = assign_roles(s, self.team, self.roles, keys)
        return self.roles

    def heads(self):
        """{car index: head} for the current role order."""
        return {car: head for head, car in enumerate(self.roles or [])}

    def decide(self, s, commit=True, keys=None, context=None):
        """One forward pass for the team with the current roles (call update_roles first).

        Returns dict(obs, mirrored, logits (4, 324), actions {car: action index}, controls {car: (8,) world-frame
        controls}, heads {car: head}, masks). commit=False leaves the recurrent state unchanged."""
        keys = keys if keys is not None else trajectory_times(s)
        opponents = opponent_order(s, self.team, keys)
        obs, mirrored = team_observation(s, self.team, self.roles, opponents,
                                         self.context if context is None else context)
        logits, h_next = self.model.forward(obs[None], self.h)
        if commit:
            self.h = h_next
        logits = logits[0]
        masks = action_masks(s)
        actions, controls = {}, {}
        for head, car in enumerate(self.roles):
            head_logits = logits[min(head, logits.shape[0] - 1)]
            a, _ = deterministic_action(head_logits, masks[car])
            c = ACTION_TABLE[a].copy()
            if mirrored:
                c[list(MIRRORED_CONTROLS)] *= -1.0
            actions[car], controls[car] = a, c
        return dict(obs=obs, mirrored=mirrored, logits=logits, actions=actions, controls=controls,
                    heads=self.heads(), masks=masks, opponents=opponents)
