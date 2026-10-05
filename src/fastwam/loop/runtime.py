"""Durable invocation timing and accounting for continuations of one training run.

Step intervals describe work performed in this output directory. Fork parents are
references only: their elapsed time and inherited steps are never added here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_COMPONENTS = ('training_seconds', 'initialization_seconds', 'checkpoint_seconds', 'diagnostic_seconds')


def _utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec='microseconds')


def _atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w') as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalized_timing(timing):
    result = dict(timing)
    start = timing.get('start_step')
    steps = int(timing.get('steps_completed', timing.get('steps', 0)))
    if start is None and timing.get('global_step') is not None:
        start = int(timing['global_step']) - steps
    end = int(start) + steps if start is not None else None
    if steps < 0 or (start is not None and (start < 0 or timing.get('global_step', end) != end)):
        raise ValueError('Invalid absolute invocation step interval')
    measured_steps = int(timing.get('timing_measured_steps', max(0, steps - timing.get('timing_warmup_steps', 0))))
    if not 0 <= measured_steps <= steps:
        raise ValueError('Measured steps must belong to this invocation')
    measured_seconds = float(timing.get('timing_measured_seconds',
                                       (timing.get('seconds_per_step') or 0.) * measured_steps)) if measured_steps else 0.
    if measured_seconds < 0 or not math.isfinite(measured_seconds):
        raise ValueError('Measured seconds cannot be negative')
    result.update(start_step=start, end_step=end, steps_completed=steps,
                  timing_measured_steps=measured_steps, timing_measured_seconds=measured_seconds)
    for key in _COMPONENTS:
        result[key] = float(timing.get(key, 0.))
        if result[key] < 0 or not math.isfinite(result[key]):
            raise ValueError(f'Negative timing component: {key}')
    return result


def _legacy_record(timing, output):
    """Preserve the available old snapshot; earlier overwritten history is unknown."""
    digest = hashlib.sha256(json.dumps(timing, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    record = _normalized_timing(timing)
    record.update(version=1, invocation_id=f'legacy-{digest}', run_directory=str(Path(output).resolve()),
                  status='legacy_snapshot', terminal=True, legacy_snapshot=True,
                  wall_started_at=None, wall_ended_at=None, allocation_job_id=None,
                  wall_seconds=sum(record[key] for key in _COMPONENTS),
                  wall_seconds_source='component_sum', unattributed_seconds=0.)
    return record


class InvocationLedger:
    """Rank-zero writer. Each launch owns one file; previous launches stay intact."""

    def __init__(self, output, *, started_monotonic=None, started_wall=None, resume=None,
                 requested_end_step=None, job_id=None, clock=time.monotonic, wall_clock=time.time):
        self.output = Path(output).resolve()
        self.directory = self.output / 'runtime/invocations'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.clock, self.wall_clock = clock, wall_clock
        self.started_monotonic = clock() if started_monotonic is None else started_monotonic
        started_wall = wall_clock() if started_wall is None else started_wall
        self.invocation_id = uuid.uuid4().hex
        # Archive compatibility snapshots before a later launch can overwrite them.
        previous = self.output / 'timing.json'
        if previous.exists():
            old = json.loads(previous.read_text())
            previous_id = old.get('invocation_id')
            if previous_id is None:
                archived = _legacy_record(old, self.output)
                archive_path = self.directory / f"{archived['invocation_id']}.json"
                if not archive_path.exists():
                    _atomic_json(archived, archive_path)
            elif not (self.directory / f'{previous_id}.json').exists():
                # A copied compatibility snapshot still carries its original launch identity.
                _atomic_json(old, self.directory / f'{previous_id}.json')
        self.record = {'version': 1, 'invocation_id': self.invocation_id,
                       'run_directory': str(self.output), 'status': 'initializing', 'terminal': False,
                       'legacy_snapshot': False, 'wall_started_at': _utc(started_wall), 'wall_ended_at': None,
                       'allocation_job_id': job_id if job_id is not None else os.environ.get('SLURM_JOB_ID'),
                       'resume_state_path': str(Path(resume).resolve()) if resume else None,
                       'requested_end_step': requested_end_step, 'start_step': None, 'end_step': None,
                       'steps_completed': 0, 'timing_measured_steps': 0, 'timing_measured_seconds': 0.,
                       **{key: 0. for key in _COMPONENTS}}
        self.update({}, status='initializing')

    def update(self, timing, *, status='running', terminal=False):
        identity = {key: self.record[key] for key in ('invocation_id', 'run_directory', 'wall_started_at',
                    'allocation_job_id', 'resume_state_path', 'requested_end_step', 'legacy_snapshot')}
        previous_start = self.record.get('start_step')
        previous_steps = self.record.get('steps_completed', 0)
        candidate = _normalized_timing({**self.record, **timing})
        candidate.update(identity)
        if ((previous_start is not None and candidate['start_step'] != previous_start)
                or candidate['steps_completed'] < previous_steps):
            raise ValueError('Invocation step interval cannot move backwards or change its start')
        self.record = candidate
        observed_at = _utc(self.wall_clock())
        self.record.update(status=status, terminal=terminal, last_observed_at=observed_at,
                           wall_ended_at=observed_at if terminal else None,
                           wall_seconds=max(0., self.clock() - self.started_monotonic),
                           wall_seconds_source='monotonic')
        self.record['unattributed_seconds'] = max(0., self.record['wall_seconds'] - sum(self.record[k] for k in _COMPONENTS))
        _atomic_json(self.record, self.directory / f'{self.invocation_id}.json')
        return dict(self.record)

    def fail(self, error):
        if not self.record['terminal']:
            timing = {'error_type': type(error).__name__, 'error': str(error)[:500]}
            if self.record['status'] == 'initializing':
                timing['initialization_seconds'] = max(0., self.clock() - self.started_monotonic)
            self.update(timing, status='failed', terminal=True)


def aggregate_invocations(records):
    """Sum observed compute; weight warm throughput by measured steps and seconds.

    Duplicate invocation IDs must agree exactly. Replayed step intervals count as
    actual work while unique_steps_completed reports distinct progress. Unfinished
    launches contribute only their last durable observation, marked as incomplete.
    """
    unique = {}
    for record in records:
        invocation_id = record['invocation_id']
        if invocation_id in unique and unique[invocation_id] != record:
            raise ValueError(f'Conflicting timing records for invocation {invocation_id}')
        unique[invocation_id] = record
    records = list(unique.values())
    run_directories = {record.get('run_directory') for record in records if record.get('run_directory')}
    if len(run_directories) > 1:
        raise ValueError('Aggregate one output run at a time; fork costs must remain separate')
    normalized = [_normalized_timing(record) for record in records]
    intervals = sorted((r['start_step'], r['end_step']) for r in normalized if r['steps_completed'])
    union = 0
    previous_end = None
    for start, end in intervals:
        union += max(0, end - max(start, previous_end if previous_end is not None else start))
        previous_end = max(end, previous_end if previous_end is not None else end)
    steps = sum(r['steps_completed'] for r in normalized)
    measured_steps = sum(r['timing_measured_steps'] for r in normalized)
    measured_seconds = sum(r['timing_measured_seconds'] for r in normalized)
    measured_samples = sum(r['timing_measured_steps'] * r.get('global_batch', 128) for r in normalized)
    partial = [r['invocation_id'] for r in normalized if not r.get('terminal', False) or r.get('status') == 'failed']
    legacy = any(r.get('legacy_snapshot', False) for r in normalized)
    return {'version': 1, 'run_directory': next(iter(run_directories), None), 'invocation_count': len(records),
            'invocation_ids': sorted(unique), 'steps_completed': steps, 'unique_steps_completed': union,
            'replayed_steps': steps - union, 'zero_step_invocations': sum(r['steps_completed'] == 0 for r in normalized),
            'absolute_step_start': min((r['start_step'] for r in normalized if r['start_step'] is not None), default=None),
            'absolute_step_end': max((r['end_step'] for r in normalized if r['end_step'] is not None), default=None),
            'total_wall_seconds': sum(r.get('wall_seconds', 0.) for r in normalized),
            **{f'total_{key}': sum(r[key] for r in normalized) for key in _COMPONENTS},
            'total_unattributed_seconds': sum(r.get('unattributed_seconds', 0.) for r in normalized),
            'timing_measured_steps': measured_steps, 'timing_measured_seconds': measured_seconds,
            'seconds_per_step': measured_seconds / measured_steps if measured_steps else None,
            'samples_per_second': measured_samples / measured_seconds if measured_seconds and measured_steps else None,
            'partial_invocation_ids': partial, 'has_legacy_records': legacy,
            'history_complete': bool(normalized) and not partial and not legacy}


def load_runtime_summary(output):
    """Read the ledger, falling back to the one available legacy timing snapshot."""
    output = Path(output)
    paths = sorted((output / 'runtime/invocations').glob('*.json'))
    records = [json.loads(path.read_text()) for path in paths]
    if not records and (output / 'timing.json').exists():
        records = [_legacy_record(json.loads((output / 'timing.json').read_text()), output)]
    return aggregate_invocations(records)
