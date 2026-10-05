"""CPU Dynamo regression using the real Wan experts and inference path."""
import logging
import pytest
import torch

from fastwam.loop.schedule import PAIRS
from fastwam.models.wan22.fastwam import FastWAM
from test_loop_model import policy


def test_one_policy_compiles_all_ten_budgets_and_reuses_graphs(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    torch.set_num_threads(1)
    torch.manual_seed(29)
    torch._dynamo.reset()
    model = policy().eval()
    latents = torch.randn(1, 2, 1, 2, 2)
    # Only the frozen VAE is replaced: all Wan blocks, schedules, cache routing,
    # action preparation, Euler integration and Dynamo guards are real.
    monkeypatch.setattr(model, '_encode_input_image_latents_tensor', lambda **kw: latents)
    kwargs = dict(prompt=None, input_image=torch.zeros(1, 3, 16, 16), action_horizon=4,
                  context=torch.randn(1, 4, 16), context_mask=torch.ones(1, 4, dtype=torch.bool),
                  proprio=torch.randn(1, 3), seed=71, num_inference_steps=2)
    compile_original = torch.compile
    traces = []

    def eager_backend(graph, example_inputs):
        traces.append((model.mot.kv, model.mot.ka))
        return graph.forward

    def compile_cpu(fn, **kw):
        assert kw['fullgraph'] is True
        return compile_original(fn, backend=eager_backend, fullgraph=True)

    monkeypatch.setattr(torch, 'compile', compile_cpu)
    try:
        # Exercise the default eight-entry per-frame limit, even when another
        # caller has changed the process default. LoopWAM must scope its override.
        with torch._dynamo.config.patch(cache_size_limit=8):
            expected = {}
            for kv, ka in PAIRS:
                expected[kv, ka] = model.infer_action(**kwargs, Kv=kv, Ka=ka)['action']
                actual = model.infer_action(**kwargs, Kv=kv, Ka=ka, compile_action_infer=True)['action']
                torch.testing.assert_close(actual, expected[kv, ka], rtol=1e-5, atol=1e-5)
                assert torch._dynamo.config.cache_size_limit == 8
            first_pass_traces = len(traces)
            assert first_pass_traces == 4 + len(PAIRS)
            for kv, ka in reversed(PAIRS):
                actual = model.infer_action(**kwargs, Kv=kv, Ka=ka, compile_action_infer=True)['action']
                torch.testing.assert_close(actual, expected[kv, ka], rtol=1e-5, atol=1e-5)
            assert len(traces) == first_pass_traces
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize('initial_limit', [8, 32])
def test_compile_limit_is_restored_when_inference_fails(monkeypatch, initial_limit):
    model = policy().eval()

    def failed_inference(*args, **kwargs):
        assert torch._dynamo.config.cache_size_limit == max(initial_limit, len(PAIRS))
        raise RuntimeError('deliberate inference failure')

    monkeypatch.setattr(FastWAM, 'infer_action', failed_inference)
    with torch._dynamo.config.patch(cache_size_limit=initial_limit):
        with pytest.raises(RuntimeError, match='deliberate inference failure'):
            model.infer_action(None, torch.zeros(1, 3, 16, 16), 4, compile_action_infer=True)
        assert torch._dynamo.config.cache_size_limit == initial_limit
