#!/usr/bin/env python3
"""Prefetch registered second-seed evaluations in another existing allocation.

This operational wrapper does not modify the frozen campaign, its manifest,
training schedule, or gate decisions. Publication is complete-only and exclusive.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

from fastwam.loop.campaign import Campaign, DELAY_PAIRS, executable_identity, resolve_deadline, run_matrix
from fastwam.loop.evaluation import ROOT, allocation_signals, atomic_json, run_process_group, sha256_file, summarize_tasks


def terminal(manifest):
    return (manifest.get('status') == 'complete' or
            manifest.get('status') == 'stopped' and manifest.get('stop_kind') in ('gate_failed', 'error'))


def destination(root, task):
    directory = 'eval_delay' if task['variant'] == 'delay' else 'eval'
    kv, ka = task['pair']
    return Path(root) / task['run'] / directory / f'kv{kv}_ka{ka}' / f"seed{task['seed']}"


def pending_tasks(root, manifest):
    """Only immutable endpoints already produced by the primary scheduler qualify."""
    root = Path(root)
    if terminal(manifest):
        return []
    seed = manifest['protocol']['evaluation_seeds'][1]
    result = []
    # Latest completed endpoints take priority over older optional control seeds.
    for run in reversed(run_matrix()):
        if run.id == 'P0-S' or manifest.get('runs', {}).get(run.id, {}).get('status') not in ('trained', 'complete'):
            continue
        directory = root / run.id
        try:
            timing = json.loads((directory / 'timing.json').read_text())
            state = json.loads((directory / 'state/latest.json').read_text())
        except FileNotFoundError:
            continue
        if not all(v.get('complete') and v.get('global_step') == run.end for v in (timing, state)):
            continue
        if not (directory / 'ema.pt').is_file():
            continue
        pairs = list(run.pairs)
        # Include extra budgets only after the primary scheduler requested them.
        for path in sorted((directory / 'eval').glob('kv*_ka*/seed*/summary.json')):
            protocol = json.loads(path.read_text()).get('protocol', {})
            pair = (protocol.get('kv'), protocol.get('ka'))
            if all(type(k) is int for k in pair) and 1 <= pair[1] <= pair[0] <= 4 and pair not in pairs:
                pairs.append(pair)
        variants = [('primary', pairs)]
        if run.id.startswith('F-Long-'):
            variants.append(('delay', DELAY_PAIRS))
        for variant, budgets in variants:
            for pair in budgets:
                task = dict(run=run.id, pair=list(pair), seed=seed, variant=variant)
                # lexists also protects a destination claimed by another worker.
                if not os.path.lexists(destination(root, task)):
                    result.append(task)
    return result


def validate_evaluation(root, manifest, task, folder):
    """Reuse the campaign's strict reader without calling its mutating constructor."""
    if not manifest.get('initial_state_sha256'):
        raise ValueError('Primary campaign must establish canonical initial states first')
    reader = object.__new__(Campaign)
    reader.root = Path(root).resolve()
    reader.args = SimpleNamespace(**manifest['launch_arguments'])
    reader.manifest = SimpleNamespace(data=manifest)
    reader.eval_path = lambda *args, **kwargs: Path(folder)
    value = reader.read_evidence(task['run'], [tuple(task['pair'])], [task['seed']], variant=task['variant'])
    summary = json.loads((Path(folder) / 'summary.json').read_text())
    tasks = [json.loads((Path(folder) / f'task_{i}.json').read_text()) for i in range(10)]
    for task_record in tasks:
        task_id = str(task_record['task_id'])
        if task_record.get('initial_state_sha256') != manifest['initial_state_sha256'].get(task_id):
            raise ValueError('Task initial-state provenance differs from canonical states: ' + task_id)
    expected = summarize_tasks(tasks, task['seed'], delay=task['variant'] == 'delay')
    for key in ('outcomes', 'episodes', 'successes', 'success_pct'):
        if summary.get(key) != expected[key]:
            raise ValueError('Task artifacts disagree with summary: ' + key)
    return value


def publish(staged, target):
    """An exclusive symlink makes the complete result visible in one atomic step."""
    staged, target = Path(staged).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(os.path.relpath(staged, target.parent.resolve()), target_is_directory=True)
        return True
    except FileExistsError:
        return False


def verify_sources(expected_source, script_sha, active_folder=None):
    actual_source = executable_identity()
    actual_script = sha256_file(Path(__file__))
    if actual_script != script_sha or actual_source != expected_source:
        if active_folder is not None:
            atomic_json(Path(active_folder) / 'UNTRUSTED.json', dict(detected_at=time.time(),
                expected_source=expected_source, actual_source=actual_source,
                expected_wrapper_sha256=script_sha, actual_wrapper_sha256=actual_script))
        raise ValueError('Source changed during secondary evaluation')


def prepare_staging(folder):
    folder = Path(folder)
    if (folder / 'UNTRUSTED.json').exists():
        folder.rename(folder.with_name(folder.name + '.untrusted.' + str(time.time_ns())))
    folder.mkdir(parents=True, exist_ok=True)


def evaluation_command(root, manifest, task, folder, args):
    launch = manifest['launch_arguments']
    kv, ka = task['pair']
    command = [sys.executable, str(ROOT / 'scripts/loopwam/evaluate.py'),
        '--checkpoint', str(Path(root) / task['run'] / 'ema.pt'), '--stats', launch['stats'],
        '--output', str(folder), '--seed', str(task['seed']), '--kv', str(kv), '--ka', str(ka),
        '--gpus', '0,1,2,3', '--text-cache', launch['text_cache'],
        '--deadline', str(args.deadline), '--deadline-reserve-seconds', str(args.reserve)]
    if task['variant'] == 'delay':
        command.append('--delay-injected')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--reserve', type=int, default=300)
    args = parser.parse_args()
    args.deadline = resolve_deadline(args.deadline)
    if args.deadline is None or args.reserve < 180:
        parser.error('An allocation deadline and at least 180 seconds reserve are required')
    root = Path(args.output).resolve()
    own = root / '.secondary_evaluations'
    own.mkdir(exist_ok=True)
    script_sha = sha256_file(Path(__file__))
    expected_source = json.loads((root / 'manifest.json').read_text())['protocol']['executable_source']
    metadata = dict(job_id=os.environ.get('SLURM_JOB_ID'), node=socket.gethostname(),
                    script_sha256=script_sha, executable_source_sha256=expected_source['sha256'],
                    started_at=time.time(), deadline=args.deadline, argv=sys.argv)

    def event(kind, **details):
        row = dict(metadata, status=kind, updated_at=time.time(), **details)
        atomic_json(own / 'status.json', row)
        with (own / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps(row, sort_keys=True), flush=True)

    with allocation_signals(), (own / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        event('starting')
        try:
            while time.time() < args.deadline - args.reserve - 30:
                verify_sources(expected_source, script_sha)
                manifest = json.loads((root / 'manifest.json').read_text())
                if terminal(manifest):
                    event('stopped', reason='Primary campaign reached a terminal state')
                    return
                tasks = pending_tasks(root, manifest)
                if not tasks:
                    atomic_json(own / 'status.json', dict(metadata, status='waiting', updated_at=time.time()))
                    time.sleep(15)
                    continue
                task = tasks[0]
                kv, ka = task['pair']
                folder = own / task['run'] / task['variant'] / f'kv{kv}_ka{ka}' / f"seed{task['seed']}"
                prepare_staging(folder)
                command = evaluation_command(root, manifest, task, folder, args)
                event('evaluating', task=task, command=command)
                last_check = 0.

                def tick():
                    nonlocal last_check
                    if time.monotonic() - last_check > 30:
                        last_check = time.monotonic()
                        verify_sources(expected_source, script_sha, folder)
                        current = json.loads((root / 'manifest.json').read_text())
                        if terminal(current):
                            raise RuntimeError('Primary campaign reached a terminal state')
                        atomic_json(own / 'status.json', dict(metadata, status='evaluating', task=task,
                                                            updated_at=time.time()))

                try:
                    with (folder / 'manager.log').open('a') as log:
                        run_process_group(command, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS='1'),
                            stdin=None, stdout=log, stderr=-2, deadline=args.deadline - args.reserve, on_tick=tick)
                finally:
                    verify_sources(expected_source, script_sha, folder)
                current = json.loads((root / 'manifest.json').read_text())
                if terminal(current):
                    event('stopped', reason='Primary campaign reached a terminal state before publication')
                    return
                evidence = validate_evaluation(root, current, task, folder)
                published = publish(folder, destination(root, task))
                event('published' if published else 'primary_already_claimed', task=task, success_pct=evidence.pct)
            event('allocation_end')
        except Exception as exc:
            event('stopped', reason=str(exc))
            raise


if __name__ == '__main__':
    main()
