import copy
import json
from collections import Counter

import numpy as np
import pytest
import torch

from fastwam.loop.data import ManifestDataset, build_split_manifest, manifest_indices, save_manifest
from fastwam.loop.sampler import DistributedWindowSampler, configurations_for_step, resolve_mode
from fastwam.loop.trainer import (EMA, GradientCoverage, WarmupConstantLR, build_parameter_groups,
                                 deepspeed_config, optimizer_update, teacher_identity, validate_resume_contract)


def synthetic_dataset(tmp_path):
    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    episodes = []
    for task in range(10):
        for demo in range(4):
            episodes.append({'episode_index': len(episodes), 'tasks': [f'task {task}'], 'length': 34 + demo})
    (root / 'meta' / 'episodes.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in episodes))
    (root / 'meta' / 'tasks.jsonl').write_text(''.join(json.dumps({'task_index': t, 'task': f'task {t}'}) + '\n' for t in range(10)))
    (root / 'meta' / 'info.json').write_text(json.dumps({'total_episodes': 40, 'total_frames': sum(e['length'] for e in episodes), 'total_tasks': 10, 'fps': 20}))
    return root


def test_episode_split_is_balanced_exhaustive_and_immutable(tmp_path):
    root = synthetic_dataset(tmp_path)
    manifest = build_split_manifest(root)
    assert manifest == build_split_manifest(root)
    held = [e for e in manifest['episodes'] if e['split'] == 'validation']
    assert Counter(e['task_id'] for e in held) == Counter({t: 2 for t in range(10)})
    train = manifest_indices(manifest, 'train')
    val = manifest_indices(manifest, 'validation')
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(1420))
    for episode in manifest['episodes']:
        ids = set(range(episode['window_start'], episode['window_stop']))
        assert ids <= set(train if episode['split'] == 'train' else val)
    path = tmp_path / 'split.json'
    save_manifest(manifest, path)
    save_manifest(manifest, path)
    changed = copy.deepcopy(manifest)
    changed['seed'] += 1
    with pytest.raises(ValueError, match='immutable'):
        save_manifest(changed, path)


def test_sampler_resume_is_independent_of_prefetch_and_microbatch():
    samplers = [DistributedWindowSampler(29, rank=r, world_size=4, seed=71) for r in range(4)]
    streams = [iter(s) for s in samplers]
    global_first = [next(streams[r]) for _ in range(7) for r in range(4)]
    assert len(set(global_first)) == 28
    # Prefetch may consume an iterator, but only explicit commit advances checkpoint state.
    for stream in streams:
        for _ in range(30):
            next(stream)
    for sampler in samplers:
        sampler.advance(7)
    resumed = []
    for rank, sampler in enumerate(samplers):
        new = DistributedWindowSampler(29, rank=rank, world_size=4, seed=71)
        new.load_state_dict(sampler.state_dict())
        resumed.append(iter(new))
    baseline = [iter(DistributedWindowSampler(29, rank=r, world_size=4, seed=71, consumed=7)) for r in range(4)]
    assert [[next(s) for s in resumed] for _ in range(25)] == [[next(s) for s in baseline] for _ in range(25)]
    assert samplers[0].state_dict()['consumed'] == 7


def test_stage_sampler_reproducible_and_uniform():
    assert configurations_for_step(7999, 'three_stage', 42) == ((4, 4),)
    assert configurations_for_step(8000, 'three_stage', 42)[1][0] in (1, 2, 3)
    assert resolve_mode(14000, 'three_stage') == 'decoupled'
    assert resolve_mode(8000, 'three_stage', 'konly', 'coupled') == 'konly'
    choices = Counter(configurations_for_step(step, 'decoupled', 42)[1] for step in range(9000))
    assert set(choices) == {(v, a) for v in range(1, 5) for a in range(1, v + 1)} - {(4, 4)}
    assert all(800 < count < 1200 for count in choices.values())
    assert configurations_for_step(14001, 'decoupled', 43) == configurations_for_step(14001, 'decoupled', 43)
    assert all(configurations_for_step(s, 'konly', 42)[1][0] == 4 for s in range(40))


def test_accumulated_update_equals_full_batch_and_ema_once():
    torch.manual_seed(3)
    full = torch.nn.Linear(3, 2)
    accum = copy.deepcopy(full)
    x, y = torch.randn(12, 3), torch.randn(12, 2)
    opt_full = torch.optim.AdamW(full.parameters(), lr=.02, betas=(.9, .95))
    opt_accum = torch.optim.AdamW(accum.parameters(), lr=.02, betas=(.9, .95))
    ema = EMA(accum, decay=.9)
    initial = {n: p.detach().clone() for n, p in accum.named_parameters()}
    for xx, yy in zip(x.chunk(4), y.chunk(4)):
        (torch.nn.functional.mse_loss(accum(xx), yy) / 4).backward()
    torch.nn.functional.mse_loss(full(x), y).backward()
    optimizer_update(accum, opt_accum, ema=ema)
    optimizer_update(full, opt_full)
    for n, p in full.named_parameters():
        torch.testing.assert_close(p, dict(accum.named_parameters())[n])
        torch.testing.assert_close(ema.shadow[n], initial[n] * .9 + p * .1)
    assert ema.updates == 1
    assert all(p.grad is None for p in accum.parameters())


def test_lr_and_ema_fork_preserve_absolute_progress():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = WarmupConstantLR(optimizer, warmup_steps=500)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-7)
    for _ in range(499):
        scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(5e-5)
    restored_opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    restored_schedule = WarmupConstantLR(restored_opt, warmup_steps=500)
    restored_opt.load_state_dict(optimizer.state_dict())
    restored_schedule.load_state_dict(scheduler.state_dict())
    restored_schedule.step()
    assert restored_opt.param_groups[0]['lr'] == pytest.approx(5e-5)
    ema = EMA(model)
    ema.update(model)
    restored_ema = EMA(model)
    restored_ema.load_state_dict(ema.state_dict())
    assert restored_ema.updates == 1
    for name in ema.shadow:
        torch.testing.assert_close(ema.shadow[name], restored_ema.shadow[name])


def test_optimizer_excludes_frozen_and_no_decay_adapters():
    class Parameters(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.norm = torch.nn.LayerNorm(3)
            self.lora_A = torch.nn.Parameter(torch.ones(2, 3))
            self.slot_delta = torch.nn.Parameter(torch.ones(4, 6, 3))
            self.gate = torch.nn.Parameter(torch.zeros(3))
            self.teacher = torch.nn.Linear(3, 3).requires_grad_(False)
    model = Parameters()
    groups = build_parameter_groups(model)
    lookup = {id(p): g for g in groups for p in g['params']}
    assert id(model.teacher.weight) not in lookup
    assert lookup[id(model.linear.weight)]['weight_decay'] == .01
    for p in [model.linear.bias, model.norm.weight, model.lora_A, model.slot_delta, model.gate]:
        assert lookup[id(p)]['weight_decay'] == 0
    assert lookup[id(model.gate)]['lr'] == 2e-4
    assert lookup[id(model.lora_A)]['lr'] == 5e-5


def test_ema_evaluation_restores_optimizer_parameter_views_after_error():
    model = torch.nn.Linear(3, 1)
    ema = EMA(model)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
    with torch.no_grad():
        for value in ema.shadow.values():
            value.add_(5)
    with pytest.raises(RuntimeError, match='evaluation failed'):
        with ema.apply_to(model):
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, original[name] + 5)
            raise RuntimeError('evaluation failed')
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, original[name])
        assert value.data_ptr() == pointers[name]


def test_window_selection_never_substitutes_a_failed_sample(tmp_path):
    manifest = build_split_manifest(synthetic_dataset(tmp_path))
    class StrictSource:
        def _get(self, index):
            if index == 0:
                raise OSError('corrupt original window')
            return {'index': index}
    source = StrictSource()
    for split in ('train', 'validation'):
        subset = ManifestDataset(source, manifest, split)
        for local_index, original_index in enumerate(subset.indices):
            if original_index == 0:
                with pytest.raises(OSError, match='corrupt original window'):
                    subset[local_index]
            else:
                sample = subset[local_index]
                assert sample['window_id'] == original_index == sample['index']
                episode = manifest['episodes'][sample['episode_id']]
                assert episode['split'] == split
                assert 0 <= sample['frame_index'] < episode['length']


def test_global_batch_and_fp32_autocast_contract():
    config = deepspeed_config(2, 16, 4, 2)
    assert config['train_batch_size'] == 128
    assert config['torch_autocast']['dtype'] == 'bfloat16'
    assert config['bf16']['enabled'] is False
    with pytest.raises(ValueError, match='Global batch'):
        deepspeed_config(3, 16, 4, 1)


def test_teacher_action_and_proprio_normalization_roundtrip():
    from pathlib import Path
    from fastwam.loop.data import DEFAULT_STATS
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json, SingleFieldLinearNormalizer
    if not Path(DEFAULT_STATS).exists():
        pytest.skip('Local teacher statistics are not installed')
    stats = load_dataset_stats_from_json(DEFAULT_STATS)
    for field, dimensions in (('action', 7), ('state', 8)):
        values = {key.removeprefix('global_'): value for key, value in stats[field]['default'].items() if key.startswith('global_')}
        normalizer = SingleFieldLinearNormalizer(values, mode='min/max')
        x = torch.lerp(values['min'], values['max'], torch.linspace(0, 1, 5)[:, None])
        assert x.shape == (5, dimensions)
        torch.testing.assert_close(normalizer.backward(normalizer.forward(x)), x, atol=1e-6, rtol=1e-6)
        if field == 'action':
            # Dataset gripper: 0 closed / 1 open. Teacher normalized targets
            # are -1/+1; simulator reverses that sign to +1 closed / -1 open.
            endpoints = x[[0, -1]]
            encoded = normalizer.forward(endpoints)
            torch.testing.assert_close(encoded[:, -1], torch.tensor([-1., 1.]))
            decoded = normalizer.backward(encoded)
            simulator = -(decoded[:, -1] * 2 - 1).sign()
            torch.testing.assert_close(simulator, torch.tensor([1., -1.]))


def test_gradient_coverage_detects_disconnected_trainable_parameters():
    class WithDisconnected(torch.nn.Linear):
        def __init__(self):
            super().__init__(3, 2)
            self.lora_unused = torch.nn.Parameter(torch.ones(2, 3))
    model = WithDisconnected()
    coverage = GradientCoverage(model)
    model(torch.ones(2, 3)).sum().backward()
    report = coverage.close()
    assert not report['complete']
    assert report['missing'] == ['lora_unused']
    assert report['groups']['lora']['observed'] == 0
    assert report['groups']['shared_or_inherited']['observed'] == 2


def test_resume_teacher_identity_and_explicit_fork_contract(tmp_path):
    teacher = tmp_path / 'teacher.pt'
    teacher.write_bytes(b'original-teacher')
    old_output, new_output = tmp_path / 'stage1', tmp_path / 'stage2'
    expected = dict(stats_sha256='stats', manifest_sha256='split', seed=42, loss='L3', world_size=4,
                    global_batch=128, teacher_identity=teacher_identity(teacher), mode='fixed',
                    stage2_mode='coupled', stage3_mode='decoupled')
    saved = copy.deepcopy(expected)
    validate_resume_contract(saved, expected, state_root=old_output / 'state', output=old_output)
    expected['mode'] = 'coupled'
    with pytest.raises(ValueError, match='Same-output resume'):
        validate_resume_contract(saved, expected, state_root=old_output / 'state', output=old_output)
    validate_resume_contract(saved, expected, state_root=old_output / 'state', output=new_output)
    teacher.write_bytes(b'different-teacher')
    expected['teacher_identity'] = teacher_identity(teacher)
    with pytest.raises(ValueError, match='teacher_identity'):
        validate_resume_contract(saved, expected, state_root=old_output / 'state', output=new_output)
    del saved['teacher_identity']
    with pytest.raises(ValueError, match='teacher_identity'):
        validate_resume_contract(saved, expected, state_root=old_output / 'state', output=new_output)
