import copy
import json

import pytest

from fastwam.loop.runtime import InvocationLedger, aggregate_invocations, load_runtime_summary


def record(name, start, end, *, output='/run', training=10., initialization=2., checkpoint=1.,
           diagnostic=0., measured_steps=None, measured_seconds=None, status='completed'):
    steps = end - start
    measured_steps = max(0, steps - 1) if measured_steps is None else measured_steps
    measured_seconds = training - 2 if measured_seconds is None and measured_steps else measured_seconds or 0.
    return {'invocation_id': name, 'run_directory': output, 'start_step': start, 'global_step': end,
            'steps_completed': steps, 'training_seconds': training, 'initialization_seconds': initialization,
            'checkpoint_seconds': checkpoint, 'diagnostic_seconds': diagnostic,
            'timing_measured_steps': measured_steps, 'timing_measured_seconds': measured_seconds,
            'wall_seconds': training + initialization + checkpoint + diagnostic,
            'status': status, 'terminal': status != 'running', 'global_batch': 128}


def test_resumes_sum_elapsed_components_and_weight_warm_throughput_without_repeating_zero_step_work():
    records = [record('a', 0, 4, training=10, initialization=2, checkpoint=1, diagnostic=2,
                      measured_steps=3, measured_seconds=9, status='interrupted'),
               record('b', 4, 10, training=25, initialization=3, checkpoint=2, diagnostic=4,
                      measured_steps=5, measured_seconds=20),
               record('c', 10, 10, training=.5, initialization=1, checkpoint=1,
                      measured_steps=0, measured_seconds=99)]
    summary = aggregate_invocations(records)
    assert summary['invocation_count'] == 3
    assert summary['steps_completed'] == summary['unique_steps_completed'] == 10
    assert summary['zero_step_invocations'] == 1
    assert summary['total_wall_seconds'] == 51.5
    assert summary['total_training_seconds'] == 35.5
    assert summary['total_initialization_seconds'] == 6
    assert summary['total_checkpoint_seconds'] == 4
    assert summary['total_diagnostic_seconds'] == 6
    assert summary['timing_measured_steps'] == 8
    assert summary['timing_measured_seconds'] == 29
    assert summary['seconds_per_step'] == pytest.approx(29 / 8)
    assert summary['samples_per_second'] == pytest.approx(8 * 128 / 29)
    assert summary['history_complete']


def test_replayed_intervals_count_compute_but_not_duplicate_progress():
    records = [record('a', 0, 10), record('b', 8, 12), record('c', 9, 10, measured_steps=0)]
    summary = aggregate_invocations(records + [copy.deepcopy(records[0])])
    assert summary['invocation_count'] == 3
    assert summary['steps_completed'] == 15
    assert summary['unique_steps_completed'] == 12
    assert summary['replayed_steps'] == 3
    changed = copy.deepcopy(records[0])
    changed['training_seconds'] += 1
    with pytest.raises(ValueError, match='Conflicting'):
        aggregate_invocations(records + [changed])


def test_forks_account_only_child_invocations_and_never_follow_parent_state(tmp_path):
    parent, child = tmp_path / 'parent', tmp_path / 'child'
    parent_record = record('parent', 0, 10, output=str(parent))
    child_records = [record('child-a', 10, 14, output=str(child)), record('child-b', 14, 16, output=str(child))]
    for output, records in ((parent, [parent_record]), (child, child_records)):
        directory = output / 'runtime/invocations'
        directory.mkdir(parents=True)
        for item in records:
            item['resume_state_path'] = str(parent / 'state')
            (directory / f"{item['invocation_id']}.json").write_text(json.dumps(item))
    summary = load_runtime_summary(child)
    assert summary['steps_completed'] == summary['unique_steps_completed'] == 6
    assert summary['absolute_step_start'] == 10 and summary['absolute_step_end'] == 16
    assert summary['total_training_seconds'] == 20
    with pytest.raises(ValueError, match='fork costs'):
        aggregate_invocations([parent_record, *child_records])


def test_ledger_preserves_prior_launches_archives_legacy_once_and_records_allocation_wall_time(tmp_path):
    old = record('unused', 0, 2)
    old.pop('invocation_id')
    (tmp_path / 'timing.json').write_text(json.dumps(old))
    now = [100.]
    kwargs = {'clock': lambda: now[0], 'wall_clock': lambda: now[0] + 1000, 'job_id': 'slurm-42'}
    first = InvocationLedger(tmp_path, resume=tmp_path / 'state', requested_end_step=5, **kwargs)
    now[0] = 113.
    finished = first.update(record('ignored', 2, 5, output=str(tmp_path)), status='interrupted', terminal=True)
    # A caller cannot assign a different launch identity through timing metadata.
    assert finished['invocation_id'] == first.invocation_id
    assert finished['allocation_job_id'] == 'slurm-42'
    assert finished['wall_started_at'] < finished['wall_ended_at']
    assert finished['wall_seconds'] == 13
    first_path = tmp_path / 'runtime/invocations' / f'{first.invocation_id}.json'
    first_bytes = first_path.read_bytes()
    second = InvocationLedger(tmp_path, **kwargs)
    assert second.invocation_id != first.invocation_id
    assert first_path.read_bytes() == first_bytes
    files = list((tmp_path / 'runtime/invocations').glob('*.json'))
    assert len(files) == 3  # two launches and one archived legacy snapshot
    summary = load_runtime_summary(tmp_path)
    assert summary['has_legacy_records'] and not summary['history_complete']
    assert second.invocation_id in summary['partial_invocation_ids']


def test_failed_and_running_invocations_are_explicit_partial_observations(tmp_path):
    now = [0.]
    ledger = InvocationLedger(tmp_path, clock=lambda: now[0], wall_clock=lambda: 1000 + now[0])
    now[0] = 7
    ledger.fail(RuntimeError('startup failed'))
    summary = load_runtime_summary(tmp_path)
    assert summary['steps_completed'] == 0
    assert summary['seconds_per_step'] is None
    assert summary['total_wall_seconds'] == 7
    assert summary['total_initialization_seconds'] == 7
    assert summary['partial_invocation_ids'] == [ledger.invocation_id]
    assert not summary['history_complete']
