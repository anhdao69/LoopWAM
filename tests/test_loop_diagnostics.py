"""Loop diagnostics must observe the actual pass without changing training state."""
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from fastwam.loop import diagnostics
from fastwam.loop.slots import SlotLinear
from test_loop_model import policy, CaptureTeacher


def _require(name):
    assert hasattr(diagnostics, name), f"Missing diagnostic implementation: {name}"
    return getattr(diagnostics, name)


def test_lora_norm_ratio_matches_dense_product_and_handles_zero_shared_norm():
    module = nn.Module()
    module.projection = SlotLinear(nn.Linear(2, 3, bias=False), rank=2)
    with torch.no_grad():
        module.projection.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))
        module.projection.lora_a.copy_(torch.tensor([[1., 2.], [0., 1.]]).expand(4, -1, -1))
        for slot in range(4):
            module.projection.lora_b[slot].copy_(torch.tensor([[1., 0.], [0., 2.], [0., 0.]]) * (slot + 1))
    records = _require('lora_norm_ratios')(module)
    assert len(records) == 4
    for slot, row in enumerate(records):
        assert row['slot'] == slot and row['module'] == 'projection'
        assert row['ratio'] == pytest.approx(3 * (slot + 1) / 2 ** .5, rel=1e-6)
    with torch.no_grad():
        module.projection.weight.zero_()
    records = diagnostics.lora_norm_ratios(module)
    assert all(row['ratio'] is None and row['status'] == 'zero_shared_norm' for row in records)
    json.dumps(records, allow_nan=False)


def test_cross_loop_cka_centers_features_and_reports_constant_states():
    features = torch.tensor([[1., 0.], [0., 1.], [2., 3.], [2., -1.]])
    result = _require('cross_loop_cka')(torch.stack((features, features * 2 + 3, torch.ones_like(features))))
    assert result['valid_loops'] == [True, True, False]
    assert result['matrix'][0][1] == pytest.approx(1., abs=1e-8)
    assert result['matrix'][0][2] is None and result['matrix'][2][2] is None


def test_loop_capture_keeps_full_pass_when_short_actions_follow():
    capture = _require('capture_loop_dynamics')
    torch.manual_seed(44)
    model = policy().eval()
    x = torch.randn(1, 2, 3, 2, 2)
    action = torch.randn(1, 4, 2)
    timestep = torch.tensor([500.])
    context, mask = torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        with capture(model.mot) as full:
            expected = model.denoise_configurations(x, action, timestep, timestep, context, mask, ((4, 4),))
        with capture(model.mot) as multiple:
            actual = model.denoise_configurations(x, action, timestep, timestep, context, mask,
                                                 ((4, 4), (4, 1), (2, 2)), return_video_exits=True)
    for kind in ('video', 'action'):
        assert set(full[kind]) == {1, 2, 3, 4}
        for loop in range(1, 5):
            torch.testing.assert_close(full[kind][loop]['features'], multiple[kind][loop]['features'])
            for field in ('state_l2', 'update_l2', 'state_rms', 'update_rms'):
                assert full[kind][loop][field] == pytest.approx(multiple[kind][loop][field])
                assert full[kind][loop][field] >= 0
    torch.testing.assert_close(expected[1][4, 4], actual[1][4, 4])
    assert '_video_layer' not in model.mot.__dict__ and '_action_layer' not in model.mot.__dict__


def test_state_and_update_norms_measure_each_loop_from_its_predecessor():
    model = policy().eval()
    with torch.no_grad():
        for parameter in model.mot.parameters():
            parameter.zero_()
        for expert in model.mot.mixtures.values():
            for block in expert.blocks[3:9]:
                block.modulation[:, 5] = 1
                for slot in range(4):
                    block.ffn[2].bias_delta[slot].fill_(slot + 1)
        video, action = model.mot.mixtures['video'], model.mot.mixtures['action']
        with diagnostics.capture_loop_dynamics(model.mot) as trace:
            _, cache, _ = model.mot.video_forward(
                torch.ones(1, 3, 12), video.get_freqs(3, 1, 1), torch.zeros(1, 6, 12),
                torch.zeros(1, 2, 12), torch.ones(1, 3, 2, dtype=torch.bool),
                torch.ones(3, 3, dtype=torch.bool), 1, 4)
            model.mot.action_forward(torch.ones(1, 2, 8), action.get_freqs(2), torch.zeros(1, 6, 8),
                                     torch.zeros(1, 2, 8), torch.ones(1, 2, 2, dtype=torch.bool), cache, 4, 4)
    for kind, root_elements in (('video', 6), ('action', 4)):
        for loop, state, update in ((1, 7, 6), (2, 19, 12), (3, 37, 18), (4, 61, 24)):
            row = trace[kind][loop]
            assert row['state_rms'] == pytest.approx(state)
            assert row['update_rms'] == pytest.approx(update)
            assert row['state_l2'] == pytest.approx(state * root_elements)
            assert row['update_l2'] == pytest.approx(update * root_elements)


def test_temporary_teacher_failure_restores_all_rngs_and_module_modes(monkeypatch, tmp_path):
    model = policy().train()
    model.mot.mixtures['video'].blocks[0].eval()
    modes = {name: module.training for name, module in model.named_modules()}
    before_torch, before_numpy, before_python = torch.get_rng_state(), np.random.get_state(), random.getstate()
    def fail_load(*args, **kwargs):
        torch.rand(4); np.random.rand(4); random.random()
        raise RuntimeError('teacher load failed')
    monkeypatch.setattr(diagnostics, 'load_teacher', fail_load)
    with pytest.raises(RuntimeError, match='teacher load failed'):
        diagnostics.run_open_loop(model, None, tmp_path, 1000, 'unused')
    assert torch.equal(before_torch, torch.get_rng_state())
    assert np.array_equal(before_numpy[1], np.random.get_state()[1])
    assert before_numpy[2:] == np.random.get_state()[2:]
    assert before_python == random.getstate()
    assert modes == {name: module.training for name, module in model.named_modules()}


def test_capture_failure_restores_existing_method_override():
    model = policy().eval()
    original = model.mot._video_layer
    def custom_layer(*args, **kwargs):
        return original(*args, **kwargs)
    model.mot._video_layer = custom_layer
    with pytest.raises(RuntimeError, match='interrupted'):
        with diagnostics.capture_loop_dynamics(model.mot):
            raise RuntimeError('interrupted')
    assert model.mot._video_layer is custom_layer
    assert '_action_layer' not in model.mot.__dict__


def test_open_loop_writes_loop_and_adapter_metrics_on_fixed_panel(tmp_path):
    _require('capture_loop_dynamics')
    torch.manual_seed(45)
    model = policy(teacher=CaptureTeacher()).train()
    class Panel:
        indices = list(range(20))
        manifest = {'episodes': [dict(split='validation', window_start=i, unpadded_windows=1) for i in range(20)]}
        def __getitem__(self, index):
            generator = torch.Generator().manual_seed(index)
            return dict(window_id=index, input_latents=torch.randn(2, 3, 2, 2, generator=generator),
                        action=torch.randn(32, 2, generator=generator),
                        context=torch.randn(4, 16, generator=generator), context_mask=torch.ones(4, dtype=torch.bool),
                        proprio=torch.randn(3, generator=generator), action_is_pad=torch.zeros(32, dtype=torch.bool),
                        image_is_pad=torch.zeros(9, dtype=torch.bool))
    rng = torch.get_rng_state().clone()
    record = diagnostics.run_open_loop(model, Panel(), tmp_path, 1000, 'unused', pairs=['2,2', '4,4', '1,1'])
    assert torch.equal(rng, torch.get_rng_state()) and model.training
    assert record['loop_dynamics']['configuration'] == [4, 4]
    assert record['loop_dynamics']['tau'] == .5
    assert record['loop_dynamics']['clips'] == 20
    assert record['evaluated_pairs'] == [[4, 4], [2, 2], [1, 1]]
    assert set(record['metrics']) == {'4_4', '2_2', '1_1'}
    for stream in ('video', 'action'):
        assert len(record['loop_dynamics'][stream]['loops']) == 4
        assert len(record['loop_dynamics'][stream]['cross_loop_cka']['matrix']) == 4
    assert len(record['lora_norm_ratios']) == 480
    assert json.loads((tmp_path / 'open_loop/step_00001000.json').read_text()) == record


def test_explicit_diagnostic_pairs_cover_continuations_konly_and_all_confirmation_budgets():
    from types import SimpleNamespace
    from fastwam.loop.diagnostics import diagnostic_pairs, normalize_diagnostic_pairs
    fixed = SimpleNamespace(meta={'arch': 'loopwam'}, mode='fixed')
    assert diagnostic_pairs(fixed, 10000) == ((4, 4),)  # preserve old probe defaults
    assert diagnostic_pairs(fixed, 10000, ['1,1', '2,2', '4,4']) == ((4, 4), (1, 1), (2, 2))
    konly = SimpleNamespace(meta={'arch': 'loopwam'}, mode='konly')
    assert diagnostic_pairs(konly, 9000, ['4,4', '4,2', '4,1', '2,2']) == ((4, 4), (4, 2), (4, 1), (2, 2))
    all_pairs = [(v, a) for v in range(1, 5) for a in range(1, v + 1)]
    actual = diagnostic_pairs(fixed, 1000, all_pairs)
    assert actual[0] == (4, 4) and len(actual) == 10 and set(actual) == set(all_pairs)
    assert normalize_diagnostic_pairs(['2,2']) == ((4, 4), (2, 2))
    for arch in ('untied30', 'untied12'):
        control = SimpleNamespace(meta={'arch': arch}, mode='fixed')
        assert diagnostic_pairs(control, 1, ['4,4']) == ((4, 4),)
        with pytest.raises(ValueError, match='controls'):
            diagnostic_pairs(control, 1, ['2,2'])
    for values in (['2,3'], ['0,0'], ['5,1'], ['1'], ['a,b'], ['4,4', '4,4'], []):
        with pytest.raises(ValueError):
            normalize_diagnostic_pairs(values)
