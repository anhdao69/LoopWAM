"""Standalone DeepSpeed training with explicit optimizer-step and resume semantics."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .data import (DEFAULT_DATA_ROOT, DEFAULT_STATS, DEFAULT_TEXT_CACHE, ManifestDataset, build_dataset,
                   build_split_manifest, file_sha256, manifest_digest, save_manifest)
from .sampler import DistributedWindowSampler, MODES, resolve_mode


class EMA:
    """FP32 EMA of trainable parameters, updated once per successful optimizer step."""

    def __init__(self, model: torch.nn.Module, decay: float = .999):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0, 1)')
        self.decay = decay
        self.updates = 0
        self.shadow = {name: value.detach().float().clone() for name, value in model.named_parameters() if value.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = dict(model.named_parameters())
        # Batch foreach by device/dtype; no per-parameter scalar device reads.
        groups = defaultdict(lambda: ([], []))
        for name, average in self.shadow.items():
            values, sources = groups[(average.device, average.dtype)]
            values.append(average)
            sources.append(current[name].detach().to(device=average.device, dtype=average.dtype))
        for values, sources in groups.values():
            torch._foreach_lerp_(values, sources, 1 - self.decay)
        self.updates += 1

    def state_dict(self) -> dict:
        return {'decay': self.decay, 'updates': self.updates,
                'shadow': {name: value.detach().cpu() for name, value in self.shadow.items()}}

    @contextmanager
    def apply_to(self, model):
        """Temporarily use EMA storage without allocating another model copy.

        Restore the exact original data views, including ZeRO's flattened storage.
        This context is only valid for evaluation between optimizer updates.
        """
        parameters = dict(model.named_parameters())
        original = {name: parameters[name].data for name in self.shadow}
        try:
            for name, average in self.shadow.items():
                parameters[name].data = average
            yield
        finally:
            for name, value in original.items():
                parameters[name].data = value

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        if set(state['shadow']) != set(self.shadow):
            raise ValueError('EMA checkpoint parameter names differ from model')
        self.decay, self.updates = float(state['decay']), int(state['updates'])
        for name, value in state['shadow'].items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(f'EMA shape mismatch: {name}')
            self.shadow[name].copy_(value)


class WarmupConstantLR:
    """LR for update n is base_lr * min((n + 1) / warmup, 1), with zero-based n."""

    def __init__(self, optimizer, warmup_steps: int = 500):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.completed_steps = 0
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        self._apply()

    def _apply(self):
        multiplier = min((self.completed_steps + 1) / max(1, self.warmup_steps), 1.)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = base * multiplier

    def step(self, **kwargs):
        self.completed_steps += 1
        self._apply()

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return dict(warmup_steps=self.warmup_steps, completed_steps=self.completed_steps, base_lrs=self.base_lrs)

    def load_state_dict(self, state):
        if len(state['base_lrs']) != len(self.optimizer.param_groups):
            raise ValueError('LR checkpoint optimizer group mismatch')
        self.warmup_steps = int(state['warmup_steps'])
        self.completed_steps = int(state['completed_steps'])
        self.base_lrs = list(state['base_lrs'])
        self._apply()


class GradientCoverage:
    """Collect the first backward's tensor coverage without scalar device reads."""

    def __init__(self, model):
        self.expected = [name for name, p in model.named_parameters() if p.requires_grad and p.numel()]
        self.observed = {}
        self.hooks = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.numel():
                self.hooks.append(parameter.register_hook(lambda gradient, name=name: self._observe(name, gradient)))

    def _observe(self, name, gradient):
        if name not in self.observed:
            self.observed[name] = (torch.isfinite(gradient).all(), torch.count_nonzero(gradient) > 0)
        return gradient

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        names = sorted(self.observed)
        flags = torch.stack([value for name in names for value in self.observed[name]]).cpu().tolist() if names else []
        finite = {name: flags[2 * i] for i, name in enumerate(names)}
        nonzero = {name: flags[2 * i + 1] for i, name in enumerate(names)}
        def group(name):
            if 'lora' in name: return 'lora'
            if 'delta' in name or 'slot' in name: return 'slot'
            if 'norm' in name: return 'norm'
            if 'proprio_encoder' in name: return 'proprio'
            return 'shared_or_inherited'
        groups = {}
        for name in self.expected:
            row = groups.setdefault(group(name), {'expected': 0, 'observed': 0, 'nonzero': 0})
            row['expected'] += 1
            row['observed'] += int(name in self.observed)
            row['nonzero'] += int(nonzero.get(name, False))
        return {'groups': groups, 'missing': sorted(set(self.expected) - set(names)),
                'nonfinite': [name for name in names if not finite[name]],
                'zero_gradient': [name for name in names if not nonzero[name]],
                'complete': set(names) == set(self.expected) and all(finite.values())}


def build_parameter_groups(model: torch.nn.Module, lr: float = 5e-5, new_lr: float = 2e-4) -> list[dict]:
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower = name.lower()
        no_decay = parameter.ndim < 2 or any(token in lower for token in ('norm', 'lora', 'slot', 'delta', 'bias', 'gate', 'modulation', 'alpha'))
        is_new = any(token in lower for token in ('gate', 'reinject', 'alpha', 'hidden_projector', 'kd_projector'))
        rate, decay = new_lr if is_new else lr, 0. if no_decay else .01
        group = groups.setdefault((rate, decay), {'params': [], 'lr': rate, 'weight_decay': decay})
        group['params'].append(parameter)
    if not groups:
        raise ValueError('No trainable parameters')
    return list(groups.values())


def gradient_group_name(model, name: str) -> str:
    if name.startswith('proprio_encoder.'):
        return 'proprio'
    match = re.match(r'mot\.mixtures\.(video|action)\.(.*)', name)
    if match is None:
        return 'other'
    stream, suffix = match.groups()
    block = re.match(r'blocks\.(\d+)\.', suffix)
    if block is None:
        return f'{stream}/global'
    index = int(block.group(1))
    if 'lora' in suffix:
        return f'{stream}/lora'
    looped = getattr(model, 'meta', {}).get('arch') == 'loopwam'
    if 'delta' in suffix or (looped and 3 <= index < 9 and 'norm' in suffix):
        return f'{stream}/slot'
    blocks = model.mot.mixtures[stream].blocks
    section = 'prelude' if index < 3 else 'coda' if index >= len(blocks) - 3 else 'core_base'
    return f'{stream}/{section}'


@torch.no_grad()
def grouped_gradient_norms(model, *, distributed: bool = True) -> dict[str, torch.Tensor]:
    """Global accumulated, averaged, pre-clipping norms for ZeRO-1/2.

    DeepSpeed 0.18.7 exposes the local, non-overlapping gradient fragments via
    each parameter's HP mapping after backward. Summing only their squared norms
    avoids materializing/all-gathering every full gradient. One small vector
    all-reduce reconstructs the exact global group norms.
    """
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad and p.numel()]
    device = named[0][1].device
    groups = {gradient_group_name(model, name) for name, _ in named}
    sums = {name: torch.zeros((), device=device) for name in sorted(groups)}
    for name, parameter in named:
        if hasattr(parameter, '_hp_mapping'):
            mapping = parameter._hp_mapping
            gradient = None if mapping is None else mapping.get_lp_grad_fragment(parameter._index_in_param_group)
        else:
            gradient = parameter.grad
        if gradient is not None:
            sums[gradient_group_name(model, name)] += torch.linalg.vector_norm(gradient.detach().float()).square()
    keys = list(sums)
    values = torch.stack([sums[name] for name in keys])
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return dict(zip(keys, values.sqrt().unbind()))


def optimizer_update(model, optimizer, *, ema=None, scheduler=None, clip_grad: float = 1.) -> None:
    """CPU/reference update used to verify accumulation and EMA against full batches."""
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip_grad, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None:
        scheduler.step()
    if ema is not None:
        ema.update(model)


def deepspeed_config(micro_batch: int, grad_accum: int, world_size: int, zero_stage: int) -> dict:
    if micro_batch * grad_accum * world_size != 128:
        raise ValueError(f'Global batch must equal 128, got {micro_batch} * {grad_accum} * {world_size}')
    return {
        'train_batch_size': 128, 'train_micro_batch_size_per_gpu': micro_batch,
        'gradient_accumulation_steps': grad_accum, 'gradient_clipping': 1.,
        # Native autocast preserves FP32 model/master weights. bf16.enabled would
        # instead downcast the stored student weights and change this contract.
        'torch_autocast': {'enabled': True, 'dtype': 'bfloat16', 'lower_precision_safe_modules': []},
        'fp16': {'enabled': False}, 'bf16': {'enabled': False},
        'zero_optimization': {'stage': zero_stage, 'contiguous_gradients': True,
                              'overlap_comm': True, 'reduce_scatter': True,
                              'reduce_bucket_size': 50_000_000, 'allgather_bucket_size': 50_000_000,
                              'ignore_unused_parameters': True},
        'zero_allow_untested_optimizer': True, 'steps_per_print': 1000000,
        'wall_clock_breakdown': False, 'checkpoint': {'use_node_local_storage': False},
    }


def rng_state() -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def atomic_json(value: dict, path: Path):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def save_training_state(engine, ema, sampler, output: Path, metadata: dict, *, keep: int = 2) -> Path:
    """Commit latest only after every rank's optimizer, RNG, cursor and EMA exist."""
    rank = dist.get_rank()
    state_root = output / 'state'
    # A retry at the same absolute step must not overwrite the checkpoint that
    # latest currently names: a killed writer would otherwise corrupt resume.
    tag_payload = [None]
    if rank == 0:
        tag = f"step_{metadata['global_step']:08d}"
        if (state_root / tag).exists():
            tag += f'_{time.time_ns()}'
        tag_payload[0] = tag
    dist.broadcast_object_list(tag_payload, src=0)
    tag = tag_payload[0]
    destination = state_root / tag
    destination.mkdir(parents=True, exist_ok=True)
    engine.save_checkpoint(str(state_root), tag=tag, client_state=metadata, save_latest=False)
    torch.save({'rng': rng_state(), 'sampler': sampler.state_dict()}, destination / f'rank_{rank:05d}.pt')
    if rank == 0:
        torch.save(ema.state_dict(), destination / 'ema.pt')
        atomic_json(metadata, destination / 'metadata.json')
    dist.barrier()
    if rank == 0:
        latest = state_root / '.latest.tmp'
        latest.write_text(tag + '\n')
        latest.replace(state_root / 'latest')
        atomic_json({**metadata, 'tag': tag, 'state_path': str(state_root.resolve())}, state_root / 'latest.json')
        if keep > 0:
            prior = sorted(p for p in state_root.glob('step_*') if p.is_dir() and (p / 'metadata.json').exists())
            for stale in prior[:-keep]:
                if stale != destination:
                    shutil.rmtree(stale)
    dist.barrier()
    return destination


def teacher_identity(checkpoint: str | Path) -> dict:
    """Immutable file identity without reading a multi-gigabyte teacher on every rank."""
    path = Path(checkpoint).resolve(strict=True)
    info = path.stat()
    return {'path': str(path), 'size': info.st_size, 'mtime_ns': info.st_mtime_ns}


def validate_resume_contract(metadata: dict, expected: dict, *, state_root: str | Path, output: str | Path) -> None:
    for name in ('stats_sha256', 'manifest_sha256', 'seed', 'loss', 'world_size', 'global_batch', 'teacher_identity'):
        if name not in metadata or metadata[name] != expected[name]:
            raise ValueError(f'Resume mismatch for {name}: {metadata.get(name)!r} != {expected[name]!r}')
    # A branch into a different output directory is an explicit stage fork.
    # Continuing the same output must preserve its sampling experiment.
    if Path(state_root).resolve() == (Path(output).resolve() / 'state'):
        for name in ('mode', 'stage2_mode', 'stage3_mode'):
            if metadata.get(name) != expected[name]:
                raise ValueError(f'Same-output resume cannot change {name}; use a new --output for an explicit fork')


def load_training_state(engine, ema, sampler, state_root: str | Path, expected: dict) -> dict:
    state_root = Path(state_root)
    metadata = json.loads((state_root / 'latest.json').read_text())
    validate_resume_contract(metadata, expected, state_root=state_root, output=expected['output'])
    tag = metadata['tag']
    loaded, client_state = engine.load_checkpoint(str(state_root), tag=tag, load_module_strict=True,
                                                load_optimizer_states=True, load_lr_scheduler_states=True)
    if loaded is None or client_state['global_step'] != metadata['global_step']:
        raise RuntimeError('DeepSpeed checkpoint did not load the advertised step')
    rank_state = torch.load(state_root / tag / f'rank_{dist.get_rank():05d}.pt', map_location='cpu', weights_only=False)
    sampler.load_state_dict(rank_state['sampler'])
    ema.load_state_dict(torch.load(state_root / tag / 'ema.pt', map_location='cpu', weights_only=False))
    if ema.updates != metadata['global_step']:
        raise ValueError('EMA update count does not match absolute optimizer step')
    if engine.lr_scheduler.completed_steps != metadata['global_step']:
        raise ValueError('LR schedule does not match absolute optimizer step')
    restore_rng(rank_state['rng'])
    return metadata


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init', required=True, help='Converted canonical checkpoint matching the resumed architecture')
    p.add_argument('--resume', help='Parent/current run state directory, including all ZeRO shards')
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=MODES, default='fixed')
    p.add_argument('--stage2-mode', choices=MODES[:-1], default='coupled')
    p.add_argument('--stage3-mode', choices=MODES[:-1], default='decoupled')
    p.add_argument('--loss', choices=('L2', 'L3'), default='L3')
    p.add_argument('--max-steps', type=int, required=True, help='Absolute optimizer step endpoint, including parent steps')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=None, help='Default chosen to obtain global batch 128')
    p.add_argument('--zero-stage', type=int, choices=(1, 2), default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--cpu-threads', type=int, default=1)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--keep-checkpoints', type=int, default=2)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--diagnostic-every', type=int, default=1000, help='EMA open-loop panel interval; 0 disables for throughput probes')
    p.add_argument('--stats', default=DEFAULT_STATS)
    p.add_argument('--teacher', default='checkpoints/fastwam_release/libero_uncond_2cam224.pt')
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--text-cache', default=DEFAULT_TEXT_CACHE)
    p.add_argument('--manifest', help='Immutable split file; defaults to output/split_manifest.json')
    p.add_argument('--split-seed', type=int, default=42, help='Fixed across training seeds')
    p.add_argument('--time-budget-seconds', type=float, default=None)
    p.add_argument('--checkpoint-reserve-seconds', type=float, default=180.)
    p.add_argument('--timing-warmup-steps', type=int, default=1, help='Exclude initial cold updates from seconds_per_step')
    p.add_argument('--overfit-one-batch', action='store_true', help='Reuse the first global batch for the Phase-0 overfit diagnostic')
    p.add_argument('--local-rank', '--local_rank', type=int, default=None)
    return p


def train(args) -> dict:
    job_started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError('LoopWAM training requires CUDA; use CPU tests for infrastructure verification')
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank or 0))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(args.cpu_threads)
    # torchrun initializes rendezvous; DeepSpeed receives an already initialized PG.
    if not dist.is_initialized():
        # Rank-zero open-loop diagnostics intentionally leave peers at a barrier.
        dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank), timeout=timedelta(minutes=30))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', local_rank)
    if args.grad_accum is None:
        divisor = args.micro_batch * world_size
        if divisor <= 0 or 128 % divisor:
            raise ValueError('micro_batch * world_size must divide global batch 128')
        args.grad_accum = 128 // divisor
    config = deepspeed_config(args.micro_batch, args.grad_accum, world_size, args.zero_stage)
    if args.max_steps <= 0 or args.log_every <= 0 or args.save_every < 0 or args.diagnostic_every < 0 or args.workers < 0 or args.timing_warmup_steps < 0:
        raise ValueError('Invalid step, logging or worker count')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / 'state/latest.json').exists():
        raise ValueError('Output already contains training state; pass --resume to continue it')
    from fastwam.utils.misc import register_work_dir
    register_work_dir(output)
    manifest = build_split_manifest(args.data_root, args.split_seed, verify_parquet=rank == 0)
    base_metadata = {'version': 2, 'init': str(Path(args.init).resolve()), 'loss': args.loss, 'seed': args.seed,
                     'output': str(output),
                     'mode': args.mode, 'stage2_mode': args.stage2_mode, 'stage3_mode': args.stage3_mode,
                     'stats_sha256': file_sha256(args.stats), 'manifest_sha256': manifest_digest(manifest),
                     'world_size': world_size, 'global_batch': 128, 'max_steps': args.max_steps,
                     'dataset_counts': manifest['counts'], 'teacher': str(Path(args.teacher).resolve()),
                     'teacher_identity': teacher_identity(args.teacher),
                     'micro_batch': args.micro_batch, 'grad_accum': args.grad_accum, 'zero_stage': args.zero_stage}
    if args.resume:
        saved_metadata = json.loads((Path(args.resume) / 'latest.json').read_text())
        validate_resume_contract(saved_metadata, base_metadata, state_root=args.resume, output=output)
    if rank == 0:
        save_manifest(manifest, args.manifest or output / 'split_manifest.json')
        save_manifest(manifest, output / 'split_manifest.json')
        atomic_json(vars(args), output / 'train_args.json')
        atomic_json(config, output / 'deepspeed_config.json')
    dist.barrier()
    dataset = build_dataset(manifest, 'train', stats=args.stats, text_cache=args.text_cache)
    if rank == 0:
        print(f'[startup] dataset ready after {time.monotonic() - job_started:.2f}s', flush=True)
    validation = ManifestDataset(dataset.dataset, manifest, 'validation') if rank == 0 and args.diagnostic_every else None
    sampler = DistributedWindowSampler(len(dataset), rank=rank, world_size=world_size, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.micro_batch, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed + rank),
                        **({'prefetch_factor': 2} if args.workers else {}))
    random.seed(args.seed + rank)
    np.random.seed((args.seed + rank) % 2**32)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    from .model import load_model
    model = load_model(args.init, device=device, training=True,
                       teacher_checkpoint=args.teacher if args.loss == 'L3' else None,
                       loss_recipe=args.loss, mode=args.mode, seed=args.seed,
                       gradient_checkpointing=args.gradient_checkpointing)
    if rank == 0:
        print(f'[startup] model and teacher ready after {time.monotonic() - job_started:.2f}s', flush=True)
    bad = [n for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if bad:
        raise ValueError(f'Trainable master parameters must be float32: {bad[:5]}')
    groups = build_parameter_groups(model)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .95), eps=1e-8, fused=True)
    import deepspeed
    engine, _, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer,
                                                   model_parameters=[p for p in model.parameters() if p.requires_grad],
                                                   lr_scheduler=lambda opt: WarmupConstantLR(opt, 500),
                                                   config=config, dist_init_required=False)
    ema = EMA(model, decay=.999)
    coverage = GradientCoverage(model)
    step = 0
    if args.resume:
        saved = load_training_state(engine, ema, sampler, args.resume, base_metadata)
        step = saved['global_step']
    if step > args.max_steps:
        raise ValueError('Requested absolute endpoint precedes resumed step')
    if args.overfit_one_batch and args.resume:
        raise ValueError('The one-batch diagnostic does not support resume; use a new output')
    if int(engine.global_steps) != step:
        raise ValueError(f'DeepSpeed global step {engine.global_steps} disagrees with resume step {step}')
    engine.train()
    iterator = iter(loader)
    stop = {'requested': False, 'signal': None}
    def request_stop(signum, frame):
        stop.update(requested=True, signal=signum)
    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    torch.cuda.synchronize(device)
    started = time.monotonic()
    initialization_seconds = started - job_started
    if rank == 0:
        print(f'[startup] training ready after {initialization_seconds:.2f}s', flush=True)
    start_step, checkpoint_seconds, diagnostic_seconds = step, 0., 0.
    warmup_seconds, warmup_completed = 0., 0
    total_data_wait, warmup_data_wait = 0., 0.
    metric_sums, metric_counts = {}, {}
    interval_start, interval_step = started, step
    overfit_samples = []
    overfit_rng = None
    interrupted = False
    try:
        while step < args.max_steps:
            elapsed = time.monotonic() - job_started
            if args.time_budget_seconds is not None and elapsed >= max(0., args.time_budget_seconds - args.checkpoint_reserve_seconds):
                stop['requested'] = True
            # One collective at each optimizer boundary propagates signals/deadline.
            stopping = torch.tensor(int(stop['requested']), device=device)
            dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            if stopping.item():
                interrupted = True
                break
            model.mode = resolve_mode(step, args.mode, args.stage2_mode, args.stage3_mode)
            if args.overfit_one_batch:
                if overfit_rng is None:
                    overfit_rng = rng_state()
                else:
                    restore_rng(overfit_rng)
            step_metrics = {}
            group_norms = None
            step_data_wait = 0.
            finite = torch.ones((), dtype=torch.bool, device=device)
            for micro in range(args.grad_accum):
                if args.overfit_one_batch and len(overfit_samples) == args.grad_accum:
                    sample = overfit_samples[micro]
                else:
                    data_start = time.monotonic()
                    sample = next(iterator)
                    step_data_wait += time.monotonic() - data_start
                    if args.overfit_one_batch:
                        overfit_samples.append(sample)
                sample = to_device(sample, device)
                # DeepSpeed applies native bf16 autocast in forward and scales loss
                # by grad_accum in backward. Do not divide the loss a second time.
                loss, metrics = engine(sample, global_step=step)
                finite.logical_and_(torch.isfinite(loss.detach()))
                for key, value in {'loss': loss.detach(), **metrics}.items():
                    scalar = value.detach() if torch.is_tensor(value) else torch.tensor(value, device=device)
                    if scalar.numel() != 1:
                        raise ValueError(f'Metric {key} must be a scalar')
                    step_metrics[key] = step_metrics.get(key, 0) + scalar.float() / args.grad_accum
                engine.backward(loss)
                if micro == args.grad_accum - 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError(f'Nonfinite loss at absolute step {step}; last committed checkpoint is valid')
                    if step == 0 or (step + 1) % 100 == 0:
                        group_norms = grouped_gradient_norms(model)
                engine.step()
            # engine.step performs an optimizer update only at the accumulation boundary.
            step += 1
            if int(engine.global_steps) != step:
                raise RuntimeError('DeepSpeed optimizer/accumulation step drift')
            ema.update(model)
            sampler.advance(args.micro_batch * args.grad_accum)
            if coverage is not None:
                report = coverage.close()
                coverage = None
                gathered = [None] * world_size if rank == 0 else None
                dist.gather_object(report, gathered, dst=0)
                if rank == 0:
                    atomic_json({'global_step': step, 'ranks': gathered,
                                 'all_ranks_complete': all(row['complete'] for row in gathered)}, output / 'gradient_coverage.json')
            grad_norm = engine.get_global_grad_norm()
            if grad_norm is not None:
                step_metrics['grad_norm'] = torch.as_tensor(grad_norm, device=device).detach().float()
                if not torch.isfinite(step_metrics['grad_norm']).item():
                    raise FloatingPointError(f'Nonfinite gradient norm at step {step}; previous checkpoint remains valid')
            if step - start_step <= args.timing_warmup_steps:
                warmup_completed = step - start_step
                warmup_seconds = time.monotonic() - started - checkpoint_seconds - diagnostic_seconds
                warmup_data_wait += step_data_wait
            total_data_wait += step_data_wait
            step_metrics['data_wait_seconds_per_step'] = torch.tensor(step_data_wait, device=device)
            if group_norms is not None:
                step_metrics.update({f'gradient_norm/{name}': value for name, value in group_norms.items()})
            for key, value in step_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0) + value
                metric_counts[key] = metric_counts.get(key, 0) + 1
            if step % args.log_every == 0 or step == args.max_steps or group_norms is not None:
                keys = sorted(metric_sums)
                values = torch.stack([metric_sums[k] / metric_counts[k] for k in keys])
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                max_wait = (metric_sums['data_wait_seconds_per_step'] / metric_counts['data_wait_seconds_per_step']).clone()
                dist.all_reduce(max_wait, op=dist.ReduceOp.MAX)
                logged = dict(zip(keys + ['max_rank_data_wait_seconds_per_step'],
                                  torch.cat((values / world_size, max_wait[None])).cpu().tolist()))
                now = time.monotonic()
                record = {**logged, 'global_step': step, 'lr': scheduler.get_last_lr()[0],
                          'mode': model.mode, 'seconds_per_step': (now - interval_start) / (step - interval_step),
                          'samples_seen': step * 128, 'epoch_equivalent': step * 128 / len(dataset),
                          'elapsed_seconds': now - started, 'max_memory_allocated_gb': torch.cuda.max_memory_allocated(device) / 1e9}
                if group_norms is not None:
                    record['gradient_norm_scope'] = 'global_accumulated_pre_clip'
                if rank == 0:
                    with (output / 'metrics.jsonl').open('a') as handle:
                        handle.write(json.dumps(record, sort_keys=True) + '\n')
                    print(json.dumps(record, sort_keys=True), flush=True)
                metric_sums, metric_counts = {}, {}
                interval_start, interval_step = now, step
            if args.save_every and step % args.save_every == 0 and step < args.max_steps:
                checkpoint_start = time.monotonic()
                save_training_state(engine, ema, sampler, output, {**base_metadata, 'global_step': step, 'complete': False}, keep=args.keep_checkpoints)
                duration = time.monotonic() - checkpoint_start
                checkpoint_seconds += duration
                interval_start += duration
            if args.diagnostic_every and step % args.diagnostic_every == 0:
                from .diagnostics import run_open_loop
                diagnostic_start = time.monotonic()
                saved_rng = rng_state()
                dist.barrier()
                if rank == 0:
                    was_training = model.training
                    try:
                        with ema.apply_to(model):
                            model.eval()
                            diagnostic = run_open_loop(model, validation, output, step, teacher_checkpoint=args.teacher, seed=1234)
                            diagnostic.update(weights='ema', ema_decay=ema.decay, ema_updates=ema.updates,
                                              stats_sha256=base_metadata['stats_sha256'])
                            atomic_json(diagnostic, output / 'open_loop' / f'step_{step:08d}.json')
                    finally:
                        model.train(was_training)
                dist.barrier()
                restore_rng(saved_rng)
                duration = time.monotonic() - diagnostic_start
                diagnostic_seconds += duration
                interval_start += duration
        torch.cuda.synchronize(device)
        train_seconds = time.monotonic() - started - checkpoint_seconds - diagnostic_seconds
        complete = step == args.max_steps and not interrupted
        final_metadata = {**base_metadata, 'global_step': step, 'complete': complete,
                          'interrupted': interrupted, 'signal': stop['signal']}
        checkpoint_start = time.monotonic()
        # Save even when no updates occurred: exports and complete metadata must match.
        save_training_state(engine, ema, sampler, output, final_metadata, keep=args.keep_checkpoints)
        if rank == 0:
            model.export_checkpoint(output / 'raw.pt', step=step)
            model.export_checkpoint(output / 'ema.pt', use_state=ema.shadow, step=step)
        dist.barrier()
        checkpoint_seconds += time.monotonic() - checkpoint_start
        completed = step - start_step
        measured_steps = completed - warmup_completed
        measured_seconds = train_seconds - warmup_seconds
        data_wait = torch.tensor((total_data_wait - warmup_data_wait) / max(measured_steps, 1), device=device)
        max_data_wait = data_wait.clone()
        dist.all_reduce(data_wait, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_data_wait, op=dist.ReduceOp.MAX)
        timing = {**final_metadata, 'start_step': start_step, 'steps': completed, 'steps_completed': completed,
                  'training_seconds': train_seconds, 'checkpoint_seconds': checkpoint_seconds,
                  'initialization_seconds': initialization_seconds,
                  'diagnostic_seconds': diagnostic_seconds,
                  'seconds_per_step': measured_seconds / measured_steps if measured_steps else None,
                  'mean_seconds_per_step': train_seconds / completed if completed else None,
                  'timing_warmup_steps': warmup_completed, 'timing_measured_steps': measured_steps,
                  'timing_measured_seconds': measured_seconds,
                  'samples_per_second': measured_steps * 128 / measured_seconds if measured_steps and measured_seconds else None,
                  'mean_samples_per_second': completed * 128 / train_seconds if train_seconds else None,
                  'data_wait_seconds_per_step': float(data_wait / world_size) if measured_steps else None,
                  'max_rank_data_wait_seconds_per_step': float(max_data_wait) if measured_steps else None,
                  'max_memory_allocated_gb': torch.cuda.max_memory_allocated(device) / 1e9,
                  'raw_checkpoint': str(output / 'raw.pt'), 'ema_checkpoint': str(output / 'ema.pt'),
                  'state_path': str(output / 'state')}
        if rank == 0:
            atomic_json(timing, output / 'timing.json')
        return timing
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    args = parser().parse_args(argv)
    result = train(args)
    if dist.get_rank() == 0:
        print(json.dumps(result, sort_keys=True), flush=True)
    dist.destroy_process_group()
    return 0 if result['complete'] else 3
