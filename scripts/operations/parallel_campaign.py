#!/usr/bin/env python3
"""One gated campaign coordinator, with independent runs on exclusive Slurm slots."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

from fastwam.loop import campaign as core
from fastwam.loop.evaluation import ROOT, atomic_json, profile_cache_key, reusable_profile, run_process_group, sha256_file

BaseCampaign = core.Campaign
GROUPS = (
    (('C2', 'C3', 'S1-L2', 'S1-L3'), ('G0',)),
    (('S2-cont', 'S2-base'), ('G1', 'S1*')),
    (('S3-coupled', 'S3-late', 'S3-Konly', 'S3-2stage'), ('G2', 'GP', 'S2*')),
    (('F-Long-s1', 'F-Long-s2'), ('G3', 'S3*')),
)


def stage_group(run_id, decisions):
    for members, required in GROUPS:
        if run_id in members:
            missing = [key for key in required if decisions.get(key, {}).get('status') != 'pass']
            if missing:
                raise ValueError('Parallel stage requires passing decisions: ' + ', '.join(missing))
            return members
    return ()


class PeerCancelled(RuntimeError):
    pass


class AllocationSlots:
    def __init__(self, allocations, stopping=None):
        if not allocations or len({s['job_id'] for s in allocations}) != len(allocations):
            raise ValueError('Allocation slots must be distinct and nonempty')
        self.queue = queue.Queue()
        for allocation in allocations:
            self.queue.put(allocation)
        self.local = threading.local()
        self.stopping = stopping or threading.Event()

    def current(self):
        return getattr(self.local, 'allocation', None)

    @contextmanager
    def lease(self):
        if self.current() is not None:
            yield self.current()
            return
        while True:
            if self.stopping.is_set():
                raise PeerCancelled('A parallel worker stopped; no new allocation work may start')
            try:
                allocation = self.queue.get(timeout=.1)
                break
            except queue.Empty:
                continue
        self.local.allocation = allocation
        try:
            if self.stopping.is_set():
                raise PeerCancelled('A parallel worker stopped before dispatch')
            yield allocation
        finally:
            self.local.allocation = None
            self.queue.put(allocation)


class SynchronizedManifest:
    def __init__(self, manifest, lock):
        self.original, self.lock = manifest, lock
        self.data, self.runs, self.path = manifest.data, manifest.runs, manifest.path

    def save(self):
        with self.lock:
            return self.original.save()

    def update_run(self, *args, **kwargs):
        with self.lock:
            return self.original.update_run(*args, **kwargs)

    def record_source_drift(self, *args, **kwargs):
        with self.lock:
            return self.original.record_source_drift(*args, **kwargs)


def read_allocation(job_id):
    if not str(job_id).isdigit():
        raise ValueError('Expected numeric Slurm job ID')
    result = subprocess.run(['scontrol', 'show', 'job', str(job_id), '-o'], capture_output=True,
                            text=True, timeout=15)
    if result.returncode:
        return None
    fields = dict(item.split('=', 1) for item in result.stdout.split() if '=' in item)
    if fields.get('JobState') != 'RUNNING':
        return None
    if not fields.get('UserId', '').endswith('(' + str(os.getuid()) + ')'):
        raise ValueError('Allocation is not owned by the current user')
    tres = dict(item.split('=', 1) for item in fields.get('AllocTRES', '').split(',') if '=' in item)
    if tres.get('gres/gpu') != '4' or fields.get('NumNodes') != '1' or int(fields.get('NumCPUs', 0)) < 16:
        raise ValueError('Each slot requires one node, four GPUs and at least16 CPUs')
    from datetime import datetime
    return dict(job_id=str(job_id), node=fields['NodeList'],
                deadline=datetime.fromisoformat(fields['EndTime']).timestamp())


def select_allocations(current, authorized, lookup=read_allocation):
    primary = lookup(current)
    if primary is None:
        raise ValueError('Coordinator must run in an active Slurm allocation')
    selected = [primary]
    for job_id in dict.fromkeys(str(j) for j in authorized):
        if job_id == current:
            continue
        slot = lookup(job_id)
        if slot is None:
            continue
        if any(s['node'] == slot['node'] for s in selected):
            raise ValueError('Two allocation slots cannot share the same node')
        selected.append(slot)
    if len(selected) > 2:
        raise ValueError('This scheduler supports at most two authorized allocations')
    return selected


def slurm_command(slot, command):
    return ['srun', '--jobid=' + slot['job_id'], '--overlap', '--cpu-bind=none',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=16', '--gres=gpu:4',
            'bash', str(ROOT / 'scripts/operations/run_on_allocation.sh'), *map(str, command)]


def operations_identity():
    paths = set((ROOT / 'scripts/operations').glob('*.py')) | set((ROOT / 'scripts/operations').glob('*.sh'))
    paths.update((ROOT / 'scripts/operations/bin').glob('*'))
    files = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths) if p.is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(version=1, sha256=digest, files=files)


class ParallelCampaign(BaseCampaign):
    configuration = None

    def __init__(self, args):
        allocations = select_allocations(os.environ.get('SLURM_JOB_ID', ''), self.configuration['allocation_ids'])
        args.deadline = min([s['deadline'] for s in allocations] + ([args.deadline] if args.deadline else []))
        super().__init__(args)
        self.configure_parallel(allocations)
        try:
            actual = operations_identity()
            expected = self.manifest.data.get('parallel_operations_source', actual)
            if actual != expected:
                raise core.SourceDrift(expected, actual)
            self.ops_expected = expected
            self.manifest.data['parallel_operations_source'] = expected
            self.manifest.data['parallel_dispatch'] = dict(allocations=allocations, started_at=time.time(),
                common_deadline=self.deadline, policy='independent stage pipelines; four GPUs/global128 each')
            self.manifest.save()
            hardware = []
            for slot in allocations:
                self.execute(['/bin/true'], self.root / 'parallel_dispatch' / ('bootstrap_' + slot['job_id'] + '.log'), slot)
                metadata = json.loads((ROOT / 'outputs/loopwam_v1' / ('gpu_metadata_' + slot['job_id'] + '.json')).read_text())
                hardware.append(metadata)
            device_names = [h['queries'][0]['stdout'].strip() for h in hardware]
            if len(set(device_names)) != 1:
                raise ValueError('Parallel allocations have different GPU models or drivers')
            self.manifest.data['parallel_dispatch']['hardware'] = hardware
            self.manifest.save()
        except BaseException as exc:
            self.close()
            if isinstance(exc, core.SourceDrift):
                self.manifest.record_source_drift(exc, phase='resume')
            else:
                self.manifest.data.update(status='stopped', stop_kind=core.stop_kind(exc),
                    stop_reason=str(exc), stopped_at=time.time())
                self.manifest.save()
            raise

    def configure_parallel(self, allocations):
        self.mutex = threading.RLock()
        self.manifest = SynchronizedManifest(self.manifest, self.mutex)
        self.stopping = threading.Event()
        self.slots = AllocationSlots(allocations, self.stopping)
        self.pool = ThreadPoolExecutor(max_workers=len(allocations), thread_name_prefix='loopwam-pipeline')
        self.futures = {}
        self.profile_locks = {}
        self.first_error = None
        self.closed = False

    def init_checkpoint(self, arch):
        with self.mutex:
            return super().init_checkpoint(arch)

    def write_tables(self):
        with self.mutex:
            return super().write_tables()

    def verify_source(self, **kwargs):
        with self.mutex:
            super().verify_source(**kwargs)
            actual = operations_identity()
            if actual != self.ops_expected:
                error = core.SourceDrift(self.ops_expected, actual)
                self.manifest.record_source_drift(error, **kwargs)
                raise error

    def tick(self):
        if self.stopping.is_set():
            raise PeerCancelled('Another pipeline failed; terminating this remote step')

    def execute(self, command, log, slot):
        self.tick()
        self.require_trusted_artifacts()
        self.verify_source(phase='before_child', command=command, log=log)
        log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
        routed = slurm_command(slot, command)
        record = dict(at=time.time(), allocation=slot, command=list(map(str, command)), log=str(log))
        with self.mutex:
            events = self.root / 'parallel_dispatch/events.jsonl'
            events.parent.mkdir(exist_ok=True)
            with events.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        print('DISPATCH', json.dumps(record), flush=True)
        try:
            with log.open('a') as stream:
                run_process_group(routed, cwd=ROOT, env=dict(os.environ, SLURM_EXPORT_ENV='ALL'),
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    deadline=self.deadline - 30, on_tick=self.tick)
        finally:
            self.verify_source(phase='after_child', command=command, log=log)

    def command(self, cmd, log):
        cmd = list(map(str, cmd))
        with self.slots.lease() as slot:
            if '--profile-cache' in cmd:
                cache = Path(cmd[cmd.index('--profile-cache') + 1])
                with self.mutex:
                    lock = self.profile_locks.setdefault(str(cache), threading.Lock())
                # A short profile-only manager publishes the shared profile before
                # either worker runs episodes. The rollout holds no profile lock.
                with lock:
                    from types import SimpleNamespace
                    option = lambda key: cmd[cmd.index(key) + 1]
                    key = profile_cache_key(SimpleNamespace(gpus=option('--gpus'), stats=option('--stats'),
                        kv=int(option('--kv')), ka=int(option('--ka')), profile_architecture=option('--profile-architecture')))
                    if not cache.exists() or not reusable_profile(json.loads(cache.read_text()), key):
                        local = Path(option('--output')) / 'latency.json' if '--output' in cmd else None
                        if local is not None and local.exists() and reusable_profile(json.loads(local.read_text()), key):
                            atomic_json(cache, json.loads(local.read_text()))
                        else:
                            self.execute(cmd + ['--profile-only'], Path(log).with_name('profile_manager.log'), slot)
                        if not cache.exists() or not reusable_profile(json.loads(cache.read_text()), key):
                            raise ValueError('Profile-only child did not publish a valid architecture profile')
            return self.execute(cmd, log, slot)

    def evaluate(self, run_id, pair, seed, variant='primary'):
        output = self.eval_path(run_id, pair, seed, variant)
        if (output / 'summary.json').exists():
            self.read_evidence(run_id, [pair], [seed], variant=variant)
            if variant != 'primary' or seed != self.args.eval_seeds[0]:
                return
            profile = output / 'latency.json'
            expected = self.expected_profile_key(run_id, pair)
            if profile.exists() and expected and reusable_profile(json.loads(profile.read_text()), expected):
                return
        return super().evaluate(run_id, pair, seed, variant)

    def remaining(self, run_id):
        run = self.matrix[run_id]
        path = self.root / run_id / 'state/latest.json'
        step = json.loads(path.read_text())['global_step'] if path.exists() else run.start
        return max(0, run.end - step)

    def pipeline(self, run_id):
        try:
            with self.slots.lease():
                BaseCampaign.train(self, run_id)
                run = self.matrix[run_id]
                seeds = self.args.eval_seeds if run_id.startswith('F-Long-') else self.args.eval_seeds[:1]
                for pair in run.pairs:
                    for seed in seeds:
                        self.evaluate(run_id, pair, seed)
                if run_id.startswith('F-Long-'):
                    for pair in core.DELAY_PAIRS:
                        for seed in seeds:
                            self.evaluate(run_id, pair, seed, variant='delay')
                self.manifest.update_run(run_id, status='complete')
        except BaseException as exc:
            with self.mutex:
                if self.first_error is None and not isinstance(exc, PeerCancelled):
                    self.first_error = exc
            self.stopping.set()
            raise

    def train(self, run_id):
        if run_id in self.futures:
            return self.futures[run_id].result()
        if self.manifest.runs.get(run_id, {}).get('status') == 'complete':
            return self._validate_trained(self.matrix[run_id])
        members = stage_group(run_id, self.decisions)
        if not members:
            return super().train(run_id)
        pending = [name for name in members if self.manifest.runs.get(name, {}).get('status') != 'complete']
        for name in sorted(pending, key=lambda n: -self.remaining(n)):
            self.futures[name] = self.pool.submit(self.pipeline, name)
        return self.futures[run_id].result()

    def close(self):
        if not self.closed:
            self.stopping.set()
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.closed = True

    def run(self):
        try:
            return super().run()
        except BaseException:
            self.close()
            if self.first_error is not None:
                raise self.first_error
            raise
        finally:
            self.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', default='outputs/loopwam_v1/campaign')
    parser.add_argument('--plan', action='store_true')
    args, _ = parser.parse_known_args(argv)
    config_path = ROOT / 'outputs/loopwam_v1/parallel_allocations.json'
    if args.plan or not config_path.exists():
        return core.main(argv)
    config = json.loads(config_path.read_text())
    root = Path(args.output).resolve()
    if root != Path(config['campaign_root']).resolve():
        return core.main(argv)
    directory = root / 'parallel_dispatch'; directory.mkdir(exist_ok=True)
    with (directory / 'coordinator.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Check the existing controller before its constructor can mutate metadata.
        with (root / 'campaign.lock').open('a') as existing:
            fcntl.flock(existing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ParallelCampaign.configuration = config
        core.Campaign = ParallelCampaign
        try:
            # Construction itself starts bootstrap children, so it needs the
            # same signal cleanup as the run loop (core installs a nested guard).
            with core.allocation_signals():
                return core.main(argv)
        finally:
            core.Campaign = BaseCampaign


if __name__ == '__main__':
    main()
