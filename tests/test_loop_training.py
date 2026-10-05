import copy
import json
from collections import Counter

import numpy as np
import pytest
import torch

from fastwam.loop.data import ManifestDataset, build_split_manifest, manifest_indices, save_manifest
from fastwam.loop.sampler import DistributedWindowSampler, configurations_for_step, resolve_mode
from fastwam.loop.trainer import (EMA, GradientCoverage, WarmupConstantLR, build_parameter_groups,
                                 deepspeed_config, gradient_group_name, grouped_gradient_norms,
                                 optimizer_update, teacher_identity, validate_resume_contract)


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


def test_grouped_gradient_norms_cover_all_parameters_and_match_total():
    model = torch.nn.Module()
    model.meta = {'arch': 'loopwam'}
    model.mot = torch.nn.Module()
    model.mot.mixtures = torch.nn.ModuleDict()
    for stream in ('video', 'action'):
        expert = torch.nn.Module()
        expert.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(12)])
        model.mot.mixtures[stream] = expert
    model.proprio_encoder = torch.nn.Linear(3, 3)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    norms = grouped_gradient_norms(model, distributed=False)
    expected = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in model.parameters()]))
    torch.testing.assert_close(torch.stack(list(norms.values())).square().sum().sqrt(), expected)
    assert gradient_group_name(model, 'mot.mixtures.video.blocks.0.weight') == 'video/prelude'
    assert gradient_group_name(model, 'mot.mixtures.action.blocks.5.weight') == 'action/core_base'
    assert gradient_group_name(model, 'mot.mixtures.video.blocks.10.weight') == 'video/coda'
    assert gradient_group_name(model, 'mot.mixtures.video.blocks.5.norm_q.weight') == 'video/slot'
    assert gradient_group_name(model, 'mot.mixtures.action.blocks.5.lora_a') == 'action/lora'


def make_tiny_latent_cache(tmp_path):
    from fastwam.loop.data import (CACHE_FORMAT, LATENT_SHAPE, file_sha256, manifest_digest,
                                  tensor_payload_digest, window_identity, write_cache_contexts,
                                  write_cache_record)
    root = tmp_path / 'latents'
    manifest = {'episodes': [
        {'episode_id': 0, 'task_id': 0, 'task': 'first', 'window_start': 0, 'window_stop': 2, 'windows': 2},
        {'episode_id': 1, 'task_id': 1, 'task': 'second', 'window_start': 2, 'window_stop': 4, 'windows': 2}],
        'train_window_ids': [0, 1], 'validation_window_ids': [2, 3]}
    contexts = {task: {'context': torch.full((128, 4096), float(task + 1), dtype=torch.bfloat16),
                       'context_mask': torch.ones(128, dtype=torch.bool), 'prompt': f'prompt {task}'}
                for task in range(2)}
    for sample in contexts.values():
        sample['context'][90:] = 0  # zero original masked T5 rows, retain all-ones returned mask
    metadata = {'format': CACHE_FORMAT, 'manifest_sha256': manifest_digest(manifest), 'stats_sha256': 'stats-v1',
                'expected_windows': 4, 'contexts': {str(task): {
                    'processed_sha256': tensor_payload_digest(sample), 'prompt': sample['prompt']}
                    for task, sample in contexts.items()}}
    metadata['cache_id'] = manifest_digest(metadata)
    save_manifest(metadata, root / 'metadata.json')
    write_cache_contexts(root, metadata, contexts)
    samples = {}
    for window_id in range(4):
        sample = {**window_identity(manifest, window_id),
                  'input_latents': torch.full(LATENT_SHAPE, window_id / 8, dtype=torch.bfloat16),
                  'action': torch.arange(224, dtype=torch.float32).reshape(32, 7) + window_id,
                  'proprio': torch.arange(256, dtype=torch.float32).reshape(32, 8) - window_id,
                  'image_is_pad': torch.arange(9) >= (1 if window_id % 2 else 9),
                  'action_is_pad': torch.arange(32) >= (1 if window_id % 2 else 32),
                  'proprio_is_pad': torch.arange(33) >= (1 if window_id % 2 else 33)}
        samples[window_id] = copy.deepcopy(sample)
        write_cache_record(root, metadata, sample)
    save_manifest({'cache_id': metadata['cache_id'], 'metadata_sha256': file_sha256(root / 'metadata.json'),
                   'completed_windows': 4, 'window_ids_sha256': manifest_digest([0, 1, 2, 3])}, root / 'complete.json')
    return root, manifest, metadata, samples, contexts


def test_latent_cache_preserves_original_windows_features_padding_context_and_rng(tmp_path, monkeypatch):
    import fastwam.loop.data as data
    root, manifest, metadata, samples, contexts = make_tiny_latent_cache(tmp_path)
    monkeypatch.setattr(data, 'build_dataset', lambda *args, **kwargs: pytest.fail('Cache attempted raw image/data loading'))
    state = torch.get_rng_state().clone()
    train = data.CachedLatentDataset(root, manifest, 'train', metadata)
    validation = train.for_split('validation')
    assert train.indices == [0, 1] and validation.indices == [2, 3]
    assert data.manifest_indices(manifest, 'all') == [0, 1, 2, 3]
    # Includes adjacent episode boundaries and both end-padded windows.
    for dataset in (train, validation):
        for index, window_id in enumerate(dataset.indices):
            actual = dataset[index]
            assert 'video' not in actual
            for key, expected in samples[window_id].items():
                if isinstance(expected, torch.Tensor):
                    assert torch.equal(actual[key], expected)
                    assert actual[key].dtype == expected.dtype
                else:
                    assert actual[key] == expected
            assert torch.equal(actual['context'], contexts[actual['task_id']]['context'])
            assert actual['context_mask'].all() and not actual['context'][90:].any()
            assert actual['prompt'] == f"prompt {actual['task_id']}"
    assert torch.equal(state, torch.get_rng_state())


def test_latent_cache_rejects_incomplete_changed_sources_missing_and_corrupt_records(tmp_path):
    from fastwam.loop.data import CachedLatentDataset, cache_record_path, validate_complete_cache
    root, manifest, metadata, *_ = make_tiny_latent_cache(tmp_path)
    changed = copy.deepcopy(metadata)
    changed['stats_sha256'] = 'different'
    with pytest.raises(ValueError, match='provenance'):
        CachedLatentDataset(root, manifest, 'train', changed)
    complete = json.loads((root / 'complete.json').read_text())
    (root / 'complete.json').unlink()
    with pytest.raises(FileNotFoundError):
        CachedLatentDataset(root, manifest, 'train', metadata)
    (root / 'complete.json').write_text(json.dumps({**complete, 'window_ids_sha256': 'wrong-ids-same-count'}))
    with pytest.raises(ValueError, match='incomplete'):
        validate_complete_cache(root, metadata)
    (root / 'complete.json').write_text(json.dumps(complete))
    dataset = CachedLatentDataset(root, manifest, 'train', metadata)
    cache_record_path(root, 0).unlink()
    with pytest.raises(FileNotFoundError):
        dataset[0]
    payload = torch.load(cache_record_path(root, 1), weights_only=True)
    payload['sample']['action'][0, 0] += 1
    torch.save(payload, cache_record_path(root, 1))
    with pytest.raises(ValueError, match='checksum'):
        dataset[1]


def test_latent_cache_rejects_wrong_episode_and_context_even_with_self_consistent_checksums(tmp_path):
    from fastwam.loop.data import CachedLatentDataset, cache_record_path, tensor_payload_digest
    root, manifest, metadata, *_ = make_tiny_latent_cache(tmp_path)
    dataset = CachedLatentDataset(root, manifest, 'train', metadata)
    payload = torch.load(cache_record_path(root, 1), weights_only=True)
    payload['sample']['task_id'] = 1
    payload['sha256'] = tensor_payload_digest(payload['sample'])
    torch.save(payload, cache_record_path(root, 1))
    with pytest.raises(ValueError, match='identity'):
        dataset[1]
    context_path = root / 'contexts/task_00.pt'
    payload = torch.load(context_path, weights_only=True)
    payload['sample']['context'][0, 0] += 1
    payload['sha256'] = tensor_payload_digest(payload['sample'])
    torch.save(payload, context_path)
    with pytest.raises(ValueError, match='context checksum'):
        CachedLatentDataset(root, manifest, 'train', metadata)


def test_cached_records_compact_contiguous_views_before_serialization(tmp_path):
    from fastwam.loop.data import LATENT_SHAPE, cache_record_path, write_cache_record
    root, manifest, metadata, samples, _ = make_tiny_latent_cache(tmp_path)
    sample = samples[0]
    # A contiguous view still retains its much larger backing storage in torch.save.
    sample['input_latents'] = torch.zeros((8, *LATENT_SHAPE), dtype=torch.bfloat16)[3]
    sample['action'] = torch.zeros((128, 32, 7), dtype=torch.float32)[7]
    assert sample['action'].is_contiguous()
    assert sample['action'].untyped_storage().nbytes() > sample['action'].numel() * sample['action'].element_size()
    write_cache_record(root, metadata, sample)
    serialized = torch.load(cache_record_path(root, 0), weights_only=True)['sample']
    for value in serialized.values():
        if isinstance(value, torch.Tensor):
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


def test_latent_cache_forbids_random_preprocessing_and_live_cache_resume_switch(tmp_path):
    from fastwam.loop.data import validate_deterministic_preprocessing
    config = {'use_text_embed_cache': True, 'context_len': 128, 'processor': {
        'train_transforms': [{'_target_': 'fastwam.datasets.lerobot.transforms.image.ToTensor'},
                             {'_target_': 'torchvision.transforms.Resize', 'size': [224, 224]}]}}
    config['processor']['val_transforms'] = copy.deepcopy(config['processor']['train_transforms'])
    validate_deterministic_preprocessing(config)
    for name, value in [('drop_high_level_prob', .5), ('action_state_transforms', []),
                        ('train_transforms', [{'_target_': 'torchvision.transforms.RandomCrop', 'size': 224}])]:
        changed = copy.deepcopy(config)
        changed['processor'][name] = value
        with pytest.raises(ValueError, match='deterministic'):
            validate_deterministic_preprocessing(changed)
    metadata = dict(stats_sha256='s', manifest_sha256='m', seed=42, loss='L2', world_size=4,
                    global_batch=128, teacher_identity={'size': 1}, latent_cache_identity={'cache_id': 'one'})
    validate_resume_contract(metadata, metadata, state_root=tmp_path / 'old/state', output=tmp_path / 'new')
    for identity in (None, {'cache_id': 'two'}):
        with pytest.raises(ValueError, match='latent_cache_identity'):
            validate_resume_contract(metadata, {**metadata, 'latent_cache_identity': identity},
                                     state_root=tmp_path / 'old/state', output=tmp_path / 'new')


def test_overfit_warmup_override_is_isolated_from_benchmark_and_preserves_base_lr(tmp_path):
    from fastwam.loop.trainer import resolve_warmup_steps
    assert resolve_warmup_steps(False, None) == 500
    assert resolve_warmup_steps(True, None) == 500
    assert resolve_warmup_steps(True, 0) == 0
    for overfit, warmup in ((False, 0), (False, 500), (True, -1)):
        with pytest.raises(ValueError, match='overfit-one-batch'):
            resolve_warmup_steps(overfit, warmup)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    schedule = WarmupConstantLR(optimizer, resolve_warmup_steps(True, 0))
    assert optimizer.param_groups[0]['lr'] == 5e-5
    schedule.step()
    assert optimizer.param_groups[0]['lr'] == 5e-5
    benchmark = torch.optim.AdamW([parameter], lr=5e-5)
    WarmupConstantLR(benchmark, resolve_warmup_steps(False, None))
    assert benchmark.param_groups[0]['lr'] == pytest.approx(1e-7)
    metadata = dict(stats_sha256='s', manifest_sha256='m', seed=42, loss='L2', world_size=4,
                    global_batch=128, teacher_identity={'size': 1}, warmup_steps=500)
    with pytest.raises(ValueError, match='warmup_steps'):
        validate_resume_contract(metadata, {**metadata, 'warmup_steps': 0},
                                 state_root=tmp_path / 'old/state', output=tmp_path / 'new')


def test_diagnostic_budget_cli_and_same_output_immutability_with_explicit_fork_scope(tmp_path):
    from fastwam.loop.trainer import parser
    args = parser().parse_args(['--init', 'init.pt', '--output', 'run', '--max-steps', '1',
                                '--diagnostic-pairs', '4,4', '2,2', '1,1'])
    assert args.diagnostic_pairs == ['4,4', '2,2', '1,1']
    output = tmp_path / 'run'
    metadata = dict(stats_sha256='s', manifest_sha256='m', seed=42, loss='L2', world_size=4,
                    global_batch=128, teacher_identity={'size': 1}, mode='fixed', stage2_mode='coupled',
                    stage3_mode='decoupled', diagnostic_pairs=[[4, 4]])
    expected = {**metadata, 'diagnostic_pairs': [[4, 4], [2, 2], [1, 1]]}
    with pytest.raises(ValueError, match='diagnostic_pairs'):
        validate_resume_contract(metadata, expected, state_root=output / 'state', output=output)
    validate_resume_contract(metadata, expected, state_root=output / 'state', output=tmp_path / 'fork')
