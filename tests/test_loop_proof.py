"""Scientific launch evidence must be tied to complete, controlled measurements."""
import importlib.util
import json
from pathlib import Path
import time

import pytest

spec = importlib.util.spec_from_file_location('loopwam_check_overfit',
    Path(__file__).resolve().parents[1] / 'scripts/loopwam/check_overfit.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.fixture
def protocol(tmp_path):
    def save(path, value):
        path.write_text(json.dumps(value))
    roots = [tmp_path / name for name in ('baseline', 'followup')]
    for index, root in enumerate(roots):
        root.mkdir()
        args = dict(init='same.pt', seed=42, micro_batch=16, grad_accum=2, max_steps=300,
                    loss='L2', mode='fixed', overfit_one_batch=True, latent_cache=None,
                    zero_stage=1, gradient_checkpointing=False, stats='same.json')
        save(root / 'train_args.json', args)
        save(root / 'timing.json', dict(complete=True, global_step=300, start_step=0,
            global_batch=128, world_size=4, warmup_steps=500 if index == 0 else 0,
            manifest_sha256='manifest', stats_sha256='stats', teacher_identity={'path': 'teacher'}))
        save(root / 'gradient_coverage.json', dict(all_ranks_complete=True,
            ranks=[dict(complete=True, nonfinite=[], missing=[]) for _ in range(4)]))
        rows = [dict(global_step=1, loss=2., video_fm=1.5, grad_norm=1., lr=5e-5, **{'action_fm/4_4': .5})]
        rows += [dict(global_step=step, loss=.006, video_fm=.005, grad_norm=.1, lr=5e-5,
                      **{'action_fm/4_4': .001}) for step in range(10, 301, 10)]
        (root / 'metrics.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    path = tmp_path / 'protocol.json'
    save(path, dict(baseline=str(roots[0]), followup=str(roots[1]),
        criterion_recorded_before_followup_at=time.time()-10,
        followup_pass_criterion=dict(mean_final_30_loss_at_most_fraction_of_initial=.01,
                                    mean_final_30_each_fm_term_at_most=.03)))
    return path


def rewrite(path, update):
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))


def test_overfit_evidence_checks_controlled_measurements(protocol):
    result = checker.evaluate_record(protocol)
    assert result['status'] == 'pass'
    assert result['final_to_initial_loss_ratio'] == pytest.approx(.003)
    assert len(result['artifacts']) == 8


def test_overfit_decline_alone_does_not_satisfy_near_zero(protocol):
    path = protocol.parent / 'followup/metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows[-3:]:
        row['video_fm'], row['loss'] = .1, .101
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    assert checker.evaluate_record(protocol)['status'] == 'fail'


@pytest.mark.parametrize('defect', ['different_init', 'missing_rank', 'incomplete', 'late_criterion', 'different_lr'])
def test_overfit_refuses_uncontrolled_or_incomplete_evidence(protocol, defect):
    root = protocol.parent / 'followup'
    if defect == 'different_init':
        rewrite(root / 'train_args.json', lambda x: x.update(init='different.pt'))
    elif defect == 'missing_rank':
        rewrite(root / 'gradient_coverage.json', lambda x: x['ranks'].pop())
    elif defect == 'incomplete':
        rewrite(root / 'timing.json', lambda x: x.update(complete=False))
    elif defect == 'late_criterion':
        rewrite(protocol, lambda x: x.update(criterion_recorded_before_followup_at=time.time()+10))
    else:
        path = root / 'metrics.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]['lr'] = 1e-4
        path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        checker.evaluate_record(protocol)
