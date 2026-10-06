"""Independent evaluation must never race the campaign's manifest or outputs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts/operations/loopwam_secondary_evaluation.py'
    spec = importlib.util.spec_from_file_location('secondary', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def manifest(tmp_path):
    stats = tmp_path / 'stats.json'
    stats.write_text('{}')
    return dict(status='running', runs={'C1': {'status': 'trained'}}, decisions={},
                protocol={'evaluation_seeds': [42, 43]},
                initial_state_sha256={str(i): str(i) for i in range(10)},
                launch_arguments={'stats': str(stats), 'text_cache': str(tmp_path / 'text')})


def finish(root, name='C1', end=8000):
    d = root / name
    write(d / 'timing.json', {'complete': True, 'global_step': end})
    write(d / 'state/latest.json', {'complete': True, 'global_step': end})
    (d / 'ema.pt').write_bytes(b'checkpoint')


def evidence(root, m, task, folder):
    from fastwam.loop.evaluation import PROTOCOL, file_identity, sha256_file, summarize_tasks
    tasks = [dict(task_id=i, seed=43, status='complete', total_episodes=50, initial_state_sha256=str(i),
                  success_episodes=list(range(40)), failure_episodes=list(range(40, 50))) for i in range(10)]
    for t in tasks:
        write(folder / ('task_%d.json' % t['task_id']), t)
    summary = summarize_tasks(tasks, 43)
    summary.update(protocol=dict(PROTOCOL, weights='stage_end_ema', seed=43, kv=4, ka=4,
        checkpoint=file_identity(root / 'C1/ema.pt'),
        stats_sha256=sha256_file(m['launch_arguments']['stats'])),
        initial_state_sha256=m['initial_state_sha256'])
    write(folder / 'summary.json', summary)
    return summary


def test_only_complete_registered_endpoints_are_scheduled(tmp_path):
    mod = module(); m = manifest(tmp_path)
    assert mod.pending_tasks(tmp_path, m) == []
    finish(tmp_path)
    tasks = mod.pending_tasks(tmp_path, m)
    assert tasks == [dict(run='C1', pair=[4, 4], seed=43, variant='primary')]
    write(tmp_path / 'C1/state/latest.json', {'complete': False, 'global_step': 7999})
    assert mod.pending_tasks(tmp_path, m) == []


@pytest.mark.parametrize('status,kind', [('complete', None), ('stopped', 'gate_failed'), ('stopped', 'error')])
def test_terminal_campaign_has_no_work(tmp_path, status, kind):
    mod = module(); m = manifest(tmp_path); finish(tmp_path)
    m.update(status=status, stop_kind=kind)
    assert mod.pending_tasks(tmp_path, m) == []


def test_existing_canonical_destination_is_never_claimed(tmp_path):
    mod = module(); m = manifest(tmp_path); finish(tmp_path)
    task = mod.pending_tasks(tmp_path, m)[0]
    target = mod.destination(tmp_path, task)
    target.mkdir(parents=True)
    assert mod.pending_tasks(tmp_path, m) == []
    staged = tmp_path / 'staged'; staged.mkdir()
    assert mod.publish(staged, target) is False
    assert target.is_dir() and not target.is_symlink()


def test_publish_is_exclusive_and_visible_to_campaign_glob(tmp_path):
    mod = module(); staged = tmp_path / '.secondary/task'; staged.mkdir(parents=True)
    write(staged / 'summary.json', {'complete': True})
    target = tmp_path / 'C1/eval/kv4_ka4/seed43'
    assert mod.publish(staged, target)
    assert list(tmp_path.glob('*/eval/kv*_ka*/seed*/summary.json')) == [target / 'summary.json']
    other = tmp_path / 'other'; other.mkdir()
    assert not mod.publish(other, target)
    assert target.resolve() == staged.resolve()


def test_validation_preserves_manifest_and_rejects_incomplete_or_changed_evidence(tmp_path):
    mod = module(); m = manifest(tmp_path); finish(tmp_path)
    task = mod.pending_tasks(tmp_path, m)[0]; folder = tmp_path / '.staged'
    summary = evidence(tmp_path, m, task, folder)
    original = json.dumps(m, sort_keys=True)
    assert mod.validate_evaluation(tmp_path, m, task, folder).pct == 80
    assert json.dumps(m, sort_keys=True) == original
    summary['outcomes'].pop()
    write(folder / 'summary.json', summary)
    with pytest.raises(ValueError): mod.validate_evaluation(tmp_path, m, task, folder)
    evidence(tmp_path, m, task, folder)
    (tmp_path / 'C1/ema.pt').write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError): mod.validate_evaluation(tmp_path, m, task, folder)


def test_task_files_must_match_summary(tmp_path):
    mod = module(); m = manifest(tmp_path); finish(tmp_path)
    task = mod.pending_tasks(tmp_path, m)[0]; folder = tmp_path / '.staged'
    evidence(tmp_path, m, task, folder)
    (folder / 'task_9.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        mod.validate_evaluation(tmp_path, m, task, folder)


def test_individual_task_state_hash_must_match_canonical_states(tmp_path):
    mod = module(); m = manifest(tmp_path); finish(tmp_path)
    task = mod.pending_tasks(tmp_path, m)[0]; folder = tmp_path / '.staged'
    evidence(tmp_path, m, task, folder)
    path = folder / 'task_0.json'
    bad = json.loads(path.read_text()); bad['initial_state_sha256'] = 'different-states'
    write(path, bad)
    with pytest.raises(ValueError, match='initial-state'):
        mod.validate_evaluation(tmp_path, m, task, folder)


def test_source_drift_permanently_quarantines_inflight_artifacts(tmp_path, monkeypatch):
    mod = module(); folder = tmp_path / 'seed43'; folder.mkdir()
    (folder / 'task_0.json').write_text('untrusted partial task')
    monkeypatch.setattr(mod, 'executable_identity', lambda: {'sha256': 'changed'})
    monkeypatch.setattr(mod, 'sha256_file', lambda path: 'wrapper')
    with pytest.raises(ValueError, match='Source changed'):
        mod.verify_sources({'sha256': 'original'}, 'wrapper', folder)
    assert (folder / 'UNTRUSTED.json').exists()
    mod.prepare_staging(folder)
    assert folder.is_dir() and not (folder / 'task_0.json').exists()
    assert len(list(tmp_path.glob('seed43.untrusted.*/task_0.json'))) == 1


def test_command_uses_frozen_evaluator_and_no_profile_or_training(tmp_path):
    mod = module(); m = manifest(tmp_path); finish(tmp_path)
    task = mod.pending_tasks(tmp_path, m)[0]
    args = SimpleNamespace(deadline=12345, reserve=300)
    cmd = mod.evaluation_command(tmp_path, m, task, tmp_path / '.staged', args)
    assert cmd[1].endswith('scripts/loopwam/evaluate.py')
    assert cmd[cmd.index('--seed') + 1] == '43'
    assert cmd[cmd.index('--gpus') + 1] == '0,1,2,3'
    assert '--profile' not in cmd
    assert '--deadline' in cmd
