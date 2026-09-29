"""ChadGPT team policy network and raw_bf16_v1 weight loader."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

OBS_SIZE = 396
NUM_HEADS = 4
NUM_ACTIONS = 324
HIDDEN = 64
ENTITY_FEATURES = 44
# Per-player blocks of the observation: (first column, width); each block holds 8 player slots.
PLAYER_BLOCKS = ((9, 18), (187, 7), (243, 1), (252, 5), (292, 10))
HEAD_INDEX_COLUMNS = (376, 380)

# The inference contract this runtime implements (checked against META.json before loading).
CONTRACT = dict(
    obs_builder='InReverse2', schema_version=22, tick_skip=4, action_delay=0, action_head_tick_skips=[4, 4, 4, 4],
    canonical_mirror_mode=True, canonical_boost_timer_obs=True, head_assignment_mode=5,
    demo_aware_head_index_obs=True, sorted_action_mask_semantics=True, trajectory_context_obs=True,
    multi_timescale_context_obs=True, multi_timescale_context_feature_count=15,
    multi_timescale_context_half_lives_seconds=[1.0, 10.0, 60.0],
    multi_timescale_context_layout='ball_y,ball_vel_y,boost_advantage,nearest_ball_advantage,goal_impulse',
    terminal_objective_obs=True, terminal_objective_inference_mode=1,
    matrix_free_recurrent_hidden_size=64, matrix_free_recurrent_conditioned_gates=True,
    matrix_free_recurrent_observation_conditioned_gates=True, matrix_free_recurrent_dense_candidate_delta=True,
    relational_actor_attention_enabled=True, relational_actor_attention_heads=4,
    relational_actor_attention_hidden_size=128, masked_auxiliary_columns=[251, 372, 373, 374, 375],
)
CONTRACT_SCALES = dict(auxiliary_feature_scale=0.0, prev_action_feature_scale=1.0, trajectory_context_scale=1.0)
UNSUPPORTED_FLAGS = ('shared_role_actor', 'autoregressive_team_actions_enabled', 'shared_ego_actor_residual_enabled',
                     'ball_prediction_obs', 'learned_action_delay_enabled', 'scratch_memory_enabled')
MODEL_NAMES = ('shared_head', 'policy', 'recurrent_memory', 'recurrent_candidate_delta', 'relational_actor_attention')


def _mlp(n_in, widths, n_out=None, layer_norm=True):
    layers, last = [], n_in
    for w in widths:
        layers.append(nn.Linear(last, w))
        if layer_norm:
            layers.append(nn.LayerNorm(w, eps=1e-5))
        layers.append(nn.ReLU())
        last = w
    if n_out is not None:
        layers.append(nn.Linear(last, n_out))
    return nn.Sequential(*layers)


def attention_tokens(obs):
    """(G, 396) team observations -> (G, 9, 440) relational tokens: ball, then 8 player slots."""
    G = obs.shape[0]
    shared = obs.clone()
    parts = []
    for first, width in PLAYER_BLOCKS:
        parts.append(obs[:, first:first + 8 * width].reshape(G, 8, width))
        shared[:, first:first + 8 * width] = 0
    head_index = obs.new_full((G, 8, 1), -1.0)
    head_index[:, :4, 0] = obs[:, HEAD_INDEX_COLUMNS[0]:HEAD_INDEX_COLUMNS[1]]
    shared[:, HEAD_INDEX_COLUMNS[0]:HEAD_INDEX_COLUMNS[1]] = 0
    kind = obs.new_zeros(G, 8, 2)
    kind[..., 1] = 1.0
    players = torch.cat(parts + [head_index, kind], 2)
    ball = obs.new_zeros(G, 1, ENTITY_FEATURES)
    ball[:, 0, :9] = obs[:, :9]
    ball[:, 0, 42] = 1.0
    entities = torch.cat([ball, players], 1)
    return torch.cat([shared[:, None].expand(G, 9, OBS_SIZE), entities], 2)


PRECISIONS = ('fp32', 'bf16')


def parse_precision(value):
    """'bf16' (default; also bfloat16) or 'fp32' (also float32)."""
    v = (value or 'bf16').strip().lower()
    v = {'float32': 'fp32', 'bfloat16': 'bf16'}.get(v, v)
    if v not in PRECISIONS:
        raise ValueError(f'precision must be fp32 or bf16, not {value!r}')
    return v


def _bf16(x):
    """Round float32 values to bfloat16 (nearest even) and widen them back."""
    return x.to(torch.bfloat16).float()


# How cuBLAS runs the policy's output layer (7424 -> 1296) for one decision (batch 1) on the RTX 5090 the bridge ran
# on, measured with PyTorch 2.11/CUDA 12.8 (the bridge's) and 2.14/CUDA 13.0 alike:
# the 7424 products are summed in slices split at these offsets, each slice sum is rounded to bf16, the slice sums are
# added and rounded, and the bias is added afterwards with one more rounding.
SPLIT_K_BOUNDS = {(7424, 1296): (0, 1536, 3072, 4608, 6144, 7424)}


def _bf16_program(module, weight_dtype=torch.bfloat16):
    """nn.Sequential -> the steps _run_autocast_emulated runs, with every Linear's weight held as a contiguous copy
    (the output layer's as one copy per split-K slice). bfloat16 copies (default) halve the bytes to stream per
    decision; float32 copies (the same bf16 values) use the float32 CPU kernels instead, for CPUs whose bfloat16
    kernels are slow (see ChadGPTModel's CPU weight choice)."""
    steps = []
    for layer in module:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.detach().to(weight_dtype)
            bounds = SPLIT_K_BOUNDS.get((layer.in_features, layer.out_features))
            if bounds:
                slices = [(lo, hi, weight[:, lo:hi].contiguous()) for lo, hi in zip(bounds, bounds[1:])]
                steps.append(('split_k', slices, layer.bias.detach().float()))
            else:
                steps.append(('linear', weight.contiguous(), layer.bias.detach().to(weight_dtype)))
        else:
            steps.append(('other', layer))
    return steps


def _linear(x, weight, bias=None):
    """A Linear under bfloat16 autocast: input rounded to bf16, float32 accumulation, output rounded to bf16 once
    (bias included). bf16 weights: a bfloat16 F.linear (bf16 out); float32 weights: float32 kernels, then rounded."""
    if weight.dtype == torch.bfloat16:
        return torch.nn.functional.linear(x.to(torch.bfloat16), weight, bias)
    return _bf16(torch.nn.functional.linear(_bf16(x.float()), weight, bias))


def _linear_split_k(x, slices, bias):
    parts = [_linear(x[:, lo:hi], w) for lo, hi, w in slices]  # each slice sum rounded to bf16
    total = torch.stack(parts).double().sum(0)  # bf16 slice sums add exactly
    return _bf16(_bf16(total.float()) + bias)


def _run_autocast_emulated(steps, x):
    """nn.Sequential under CUDA bfloat16 autocast, on any device (steps from _bf16_program): a Linear rounds its
    input to bf16 and multiplies bf16 weights with float32 accumulation (a bfloat16 F.linear: oneDNN on x86 CPUs)
    and rounds its output (bias included) to bf16, except the output layer, which follows cuBLAS's split sum
    (SPLIT_K_BOUNDS); LayerNorm runs in float32 (autocast's float32 list); ReLU and Tanh keep their input's
    precision. Returns float32. Every row is computed as its own decision."""
    for step in steps:
        if step[0] == 'linear':
            x = _linear(x, step[1], step[2])
        elif step[0] == 'split_k':
            x = _linear_split_k(x, step[1], step[2])
        elif isinstance(step[1], nn.LayerNorm):
            layer = step[1]
            x = torch.nn.functional.layer_norm(x.float(), layer.normalized_shape, layer.weight, layer.bias, layer.eps)
        else:
            x = step[1](x)
    return x.float()


class TeamPolicyNet(nn.Module):
    precision = 'bf16'  # the C++ bridge's bfloat16 autocast (forward_bf16 on CUDA, emulated on CPU); or 'fp32'
    cpu_weights = 'bf16'  # weight copies of the emulated bf16 path: 'bf16' or 'fp32' (same values, float32 kernels)

    def __init__(self, policy_layers=(3712, 3712, 7424), width=3712):
        super().__init__()
        self.shared_head = _mlp(OBS_SIZE, (width, width))
        self.policy = _mlp(width, policy_layers, NUM_HEADS * NUM_ACTIONS)
        self.recurrent_memory = nn.Sequential(nn.Linear(2 * HIDDEN, 3 * HIDDEN))
        self.recurrent_candidate_delta = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.Tanh())
        self.relational_actor_attention = _mlp(OBS_SIZE + ENTITY_FEATURES, (128, 128), NUM_HEADS * (NUM_ACTIONS + 1),
                                               layer_norm=False)

    def memory(self, x, h):
        """Recurrent memory on the shared features x (G, 3712) with state h (G, 64) -> (features, next state)."""
        base = x[:, :HIDDEN]
        gate_in = torch.cat([h, base], 1)
        scale, decay, gain = self.recurrent_memory(gate_in).split(HIDDEN, 1)
        candidate = torch.tanh(base + scale * h + self.recurrent_candidate_delta(gate_in))
        h_next = torch.lerp(h, candidate, torch.sigmoid(decay))
        out = x.clone()
        out[:, :HIDDEN] = base + gain * h_next
        return out, h_next

    def attention(self, obs, mlp=None):
        G = obs.shape[0]
        tokens = (mlp or self.relational_actor_attention)(attention_tokens(obs)).view(G, 9, NUM_HEADS, NUM_ACTIONS + 1)
        weights = torch.softmax(tokens[..., 0].float(), 1)
        return (weights[..., None] * tokens[..., 1:]).sum(1).reshape(G, NUM_HEADS * NUM_ACTIONS)

    def forward(self, obs, h):
        """obs (G, 396), h (G, 64) -> logits (G, 4, 324), next recurrent state (G, 64)."""
        if self.precision == 'bf16':
            if obs.device.type == 'cuda':
                return self.forward_bf16(obs, h)
            return self.forward_bf16_emulated(obs, h)
        x, h_next = self.memory(self.shared_head(obs), h)
        logits = self.policy(x) + self.attention(obs)
        return logits.view(-1, NUM_HEADS, NUM_ACTIONS), h_next

    def forward_bf16(self, obs, h):
        """The C++ bridge's precision (GigaLearnCPP Model::Forward with halfPrec): the shared head, the policy and
        the relational attention each run under bfloat16 autocast on obs's device and return float32; the recurrent
        memory runs in float32 with autocast off, so the recurrent state stays float32. This is the bridge's
        arithmetic on CUDA; on the CPU, forward_bf16_emulated is used instead of CPU autocast, whose bf16 kernels
        round differently."""
        device = obs.device.type
        with torch.autocast(device, dtype=torch.bfloat16):
            shared = self.shared_head(obs)
        with torch.autocast(device, enabled=False):
            x, h_next = self.memory(shared.float(), h)
        with torch.autocast(device, dtype=torch.bfloat16):
            policy = self.policy(x)
            attention = self.attention(obs)
        # The bridge's logits are bfloat16: its attention term is rounded to bf16 and added to the bf16 policy output
        # in bf16. Keeping the float32 sum broke the bridge's exact top ties (5.4% of decisions, the first index wins)
        # and changed 3.5% of its actions on the 2026-09-27 trace.
        logits = (policy.to(torch.bfloat16) + attention.to(torch.bfloat16)).float()
        return logits.view(-1, NUM_HEADS, NUM_ACTIONS), h_next

    def forward_bf16_emulated(self, obs, h):
        """forward_bf16's numerics without CUDA: the same bf16 roundings, with the
        products accumulated in float32 by the CPU instead of by CUDA's bf16 GEMM kernels. The bfloat16 weight copies
        are made on the first call and kept (the float32 parameters stay for the other paths)."""
        cache = self.__dict__.setdefault('_bf16_programs', {})   # (device, cpu_weights) -> programs
        programs = cache.get((obs.device, self.cpu_weights))
        if programs is None:
            dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[self.cpu_weights]
            programs = cache[obs.device, self.cpu_weights] = {
                name: _bf16_program(getattr(self, name), dtype)
                for name in ('shared_head', 'policy', 'relational_actor_attention')}
        shared = _run_autocast_emulated(programs['shared_head'], obs)
        x, h_next = self.memory(shared, h)
        policy = _run_autocast_emulated(programs['policy'], x)
        attention = self.attention(obs, lambda t: _run_autocast_emulated(programs['relational_actor_attention'], t))
        logits = _bf16(_bf16(policy) + _bf16(attention))
        return logits.view(-1, NUM_HEADS, NUM_ACTIONS), h_next


def read_sha256sums(path):
    sums = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip().lstrip('*')] = digest.lower()
    return sums


def read_weights(path):
    """A checkpoint file's bytes. A file over GitHub's 100 MB limit is stored as <name>.part0, <name>.part1, ...
    (plain git, no LFS) and joined here."""
    path = Path(path)
    if path.is_file():
        return path.read_bytes()
    parts = []
    while (part := path.with_name(f'{path.name}.part{len(parts)}')).is_file():
        parts.append(part.read_bytes())
    if not parts:
        raise FileNotFoundError(f'{path} (or {path.name}.part0, ...) not found')
    return b''.join(parts)


def verify_checkpoint(checkpoint_dir, sha256sums=None):
    """Check every checkpoint file against SHA256SUMS (the version folder's, by default). Returns {file: sha256}."""
    checkpoint_dir = Path(checkpoint_dir)
    sums_path = Path(sha256sums) if sha256sums else checkpoint_dir.parent.parent / 'SHA256SUMS'
    expected = read_sha256sums(sums_path)
    prefix = checkpoint_dir.relative_to(sums_path.parent).as_posix()
    found = {}
    for name in ['META.json'] + [m['file'] for m in json.loads((checkpoint_dir / 'META.json').read_text())['models']]:
        key = f'{prefix}/{name}'
        if key not in expected:
            raise ValueError(f'{key} is not listed in {sums_path}')
        digest = hashlib.sha256(read_weights(checkpoint_dir / name)).hexdigest()
        if digest != expected[key]:
            raise ValueError(f'{key}: sha256 {digest} does not match {expected[key]} (incomplete download?)')
        found[name] = digest
    return found


def check_contract(meta):
    if meta.get('format') != 'raw_bf16_v1':
        raise ValueError(f"unsupported weight format {meta.get('format')!r}")
    inf = meta.get('checkpoint_inference', meta)
    for key, want in CONTRACT.items():
        if inf.get(key) != want:
            raise ValueError(f'checkpoint contract mismatch: {key}={inf.get(key)!r}, runtime implements {want!r}')
    for key, want in CONTRACT_SCALES.items():
        if float(inf.get(key, -1)) != want:
            raise ValueError(f'checkpoint contract mismatch: {key}={inf.get(key)!r}, runtime implements {want}')
    for flag in UNSUPPORTED_FLAGS:
        if inf.get(flag):
            raise ValueError(f'checkpoint uses {flag}, which this runtime does not implement')


def load_checkpoint(checkpoint_dir, device='cpu', verify=True, sha256sums=None):
    """raw_bf16_v1 checkpoint folder -> (TeamPolicyNet in eval mode, float32 parameters, META dict, hashes)."""
    checkpoint_dir = Path(checkpoint_dir)
    hashes = verify_checkpoint(checkpoint_dir, sha256sums) if verify else {}
    meta = json.loads((checkpoint_dir / 'META.json').read_text())
    check_contract(meta)
    specs = {m['name']: m for m in meta['models']}
    if set(specs) != set(MODEL_NAMES):
        raise ValueError(f'unexpected models in META.json: {sorted(specs)}')
    net = TeamPolicyNet(tuple(specs['policy']['layer_sizes']), specs['shared_head']['num_outputs'])
    for name in MODEL_NAMES:
        spec = specs[name]
        raw = read_weights(checkpoint_dir / spec['file'])
        if len(raw) != 2 * spec['numel']:
            raise ValueError(f'{name}: {len(raw)} bytes, expected {2 * spec["numel"]}')
        values = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
        module = getattr(net, name)
        if sum(p.numel() for p in module.parameters()) != values.numel():
            raise ValueError(f'{name}: parameter count does not match the file')
        nn.utils.vector_to_parameters(values, module.parameters())
    return net.to(device).eval().requires_grad_(False), meta, hashes
