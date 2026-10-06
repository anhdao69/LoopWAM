import importlib.machinery
import importlib.util
from pathlib import Path

import pytest


def module():
    p = Path(__file__).parents[1] / 'scripts/operations/bin/nvidia-smi'
    loader = importlib.machinery.SourceFileLoader('gpu_query', str(p))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec); loader.exec_module(mod)
    return mod


def record():
    return dict(job_id='12', node='gpu-node', captured_at=90, deadline=200,
        queries=[dict(args=['--query-gpu=name,driver_version', '--format=csv,noheader', '--id=0'],
                      stdout='NVIDIA H100 80GB HBM3, 580.105.08\n')])


def test_only_exact_static_query_returns_exact_captured_bytes():
    mod = module(); r = record()
    assert mod.cached_output(r, r['queries'][0]['args'], '12', 'gpu-node', 100) == r['queries'][0]['stdout']
    assert mod.cached_output(r, ['--query-gpu=utilization.gpu'], '12', 'gpu-node', 100) is None


@pytest.mark.parametrize('job,node,now', [('13', 'gpu-node', 100), ('12', 'other-node', 100),
                                       ('12', 'gpu-node', 201), ('12', 'gpu-node', 80)])
def test_cache_cannot_cross_job_node_or_validity_interval(job, node, now):
    mod = module(); r = record()
    assert mod.cached_output(r, r['queries'][0]['args'], job, node, now) is None
