"""Concurrency and gate contracts for the two-allocation coordinator."""
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts/operations/parallel_campaign.py'
    spec = importlib.util.spec_from_file_location('parallel_campaign_test', path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def test_stage_groups_require_all_scientific_dependencies():
    m = module()
    with pytest.raises(ValueError): m.stage_group('S2-cont', {})
    s1 = {'G1': {'status': 'pass'}, 'S1*': {'status': 'pass', 'winner': 'S1-L3'}}
    assert set(m.stage_group('S2-base', s1)) == {'S2-cont', 'S2-base'}
    for missing in ('G2', 'GP', 'S2*'):
        gates = {k: {'status': 'pass'} for k in ('G2', 'GP', 'S2*') if k != missing}
        with pytest.raises(ValueError): m.stage_group('S3-late', gates)
    assert len(m.stage_group('S3-late', {k: {'status': 'pass'} for k in ('G2', 'GP', 'S2*')})) == 4
    with pytest.raises(ValueError): m.stage_group('F-Long-s1', {'G3': {'status': 'fail'}})


def test_two_slots_execute_concurrently_without_reusing_an_allocation():
    m = module(); slots = m.AllocationSlots([{'job_id': '1'}, {'job_id': '2'}])
    barrier = threading.Barrier(2); seen=[]; lock=threading.Lock()
    def work():
        with slots.lease() as slot:
            with lock: seen.append(slot['job_id'])
            barrier.wait(timeout=2)
            time.sleep(.02)
    threads=[threading.Thread(target=work) for _ in range(2)]
    for t in threads:t.start()
    for t in threads:t.join(timeout=3)
    assert all(not t.is_alive() for t in threads)
    assert sorted(seen)==['1','2']
    with slots.lease() as outer:
        with slots.lease() as inner:assert inner==outer


def test_manifest_updates_from_workers_are_serialized(tmp_path):
    from fastwam.loop.campaign import Manifest
    m=module(); raw=Manifest(tmp_path/'manifest.json', {})
    wrapped=m.SynchronizedManifest(raw,threading.RLock())
    def update(name):
        for step in range(40):wrapped.update_run(name,status='training',step=step)
    threads=[threading.Thread(target=update,args=(name,)) for name in ('a','b')]
    for t in threads:t.start()
    for t in threads:t.join()
    data=json.loads(raw.path.read_text())
    assert data['runs']['a']['step']==data['runs']['b']['step']==39


def test_command_routes_four_gpus_and_preserves_argument_boundaries():
    m=module()
    original=['python','train.py','--output','a path with spaces','--micro-batch','16','--grad-accum','2']
    routed=m.slurm_command({'job_id':'123'},original)
    assert '--jobid=123' in routed and '--gres=gpu:4' in routed
    assert routed[-len(original):]==original


def test_expired_helpers_are_omitted_and_same_node_is_rejected():
    m=module()
    jobs={'1':dict(job_id='1',node='a',deadline=200), '2':None}
    assert m.select_allocations('1',['1','2'],jobs.get)==[jobs['1']]
    jobs['2']=dict(job_id='2',node='a',deadline=200)
    with pytest.raises(ValueError,match='same node'):m.select_allocations('1',['1','2'],jobs.get)


def fixture_campaign(tmp_path, monkeypatch):
    from fastwam.loop.campaign import Campaign, Manifest, run_matrix
    m=module(); c=object.__new__(m.ParallelCampaign)
    c.root=tmp_path; c.args=SimpleNamespace(eval_seeds=[42,43]); c.deadline=time.time()+60
    c.matrix={r.id:r for r in run_matrix()}
    c.manifest=Manifest(tmp_path/'manifest.json',{})
    c.decisions={k:{'status':'pass'} for k in ('G1','S1*','G2','S2*','GP','G3','S3*')}
    c.configure_parallel([dict(job_id='1'),dict(job_id='2')])
    monkeypatch.setattr(Campaign,'_validate_trained',lambda *a:None)
    return m,c,Campaign


def test_stage_pipeline_trains_both_and_evaluates_each_immediately(tmp_path,monkeypatch):
    m,c,base=fixture_campaign(tmp_path,monkeypatch)
    events=[]; lock=threading.Lock(); started=threading.Barrier(2)
    def train(self,name):
        with lock:events.append((name,'train',self.slots.current()['job_id']))
        started.wait(timeout=2)
        self.manifest.update_run(name,status='trained')
    def evaluate(self,name,pair,seed,variant='primary'):
        with lock:events.append((name,'eval',self.slots.current()['job_id']))
    monkeypatch.setattr(base,'train',train)
    monkeypatch.setattr(c,'evaluate',evaluate.__get__(c))
    try:
        c.train('S2-cont');c.train('S2-base')
    finally:c.close()
    assert {row[2] for row in events if row[1]=='train'}=={'1','2'}
    for name in ('S2-cont','S2-base'):
        e=[row for row in events if row[0]==name]
        assert [x[1] for x in e]==['train','eval','eval','eval']
        assert len({x[2] for x in e})==1
        assert c.manifest.runs[name]['status']=='complete'


def test_worker_failure_stops_peers_and_does_not_mark_complete(tmp_path,monkeypatch):
    m,c,base=fixture_campaign(tmp_path,monkeypatch)
    started=threading.Barrier(2); peer_stopped=threading.Event()
    def train(self,name):
        started.wait(timeout=2)
        if name=='S2-cont':raise ValueError('training failed')
        for _ in range(100):
            if self.stopping.is_set():peer_stopped.set();raise m.PeerCancelled('peer failed')
            time.sleep(.01)
        raise AssertionError('Peer was not cancelled')
    monkeypatch.setattr(base,'train',train)
    with pytest.raises(ValueError,match='training failed'):
        try:c.train('S2-cont')
        finally:c.close()
    assert peer_stopped.is_set()
    assert not any(v.get('status')=='complete' for v in c.manifest.runs.values())


def test_completed_run_never_launches_another_pipeline(tmp_path,monkeypatch):
    m,c,base=fixture_campaign(tmp_path,monkeypatch)
    c.manifest.update_run('S2-cont',status='complete')
    try:c.train('S2-cont')
    finally:c.close()
    assert c.futures=={}


def test_shared_profile_is_measured_once_before_concurrent_rollouts(tmp_path,monkeypatch):
    m,c,base=fixture_campaign(tmp_path,monkeypatch)
    cache=tmp_path/'cache.json';key={'hardware':'same'};events=[];lock=threading.Lock()
    monkeypatch.setattr(m,'profile_cache_key',lambda args:key)
    monkeypatch.setattr(m,'reusable_profile',lambda data,expected:data==expected)
    def execute(command,log,slot):
        if '--profile-only' in command:
            with lock:events.append('profile')
            time.sleep(.04)
            cache.write_text(json.dumps(key))
        else:
            assert cache.exists()
            with lock:events.append('rollout')
    monkeypatch.setattr(c,'execute',execute)
    cmd=['python','evaluate.py','--profile','--profile-cache',str(cache),'--profile-architecture','{}',
         '--gpus','0,1,2,3','--stats','stats.json','--kv','1','--ka','1']
    futures=[c.pool.submit(c.command,cmd,tmp_path/name/'manager.log') for name in ['a','b']]
    try:
        for future in futures:future.result(timeout=3)
    finally:c.close()
    assert events==['profile','rollout','rollout']


def test_complete_evaluation_is_validated_without_gpu_dispatch(tmp_path,monkeypatch):
    m,c,base=fixture_campaign(tmp_path,monkeypatch)
    folder=c.eval_path('C1',(4,4),42);folder.mkdir(parents=True)
    (folder/'summary.json').write_text('{}');(folder/'latency.json').write_text('{}')
    validated=[]
    monkeypatch.setattr(c,'read_evidence',lambda *a,**kw:validated.append(a))
    monkeypatch.setattr(c,'expected_profile_key',lambda *a:{'valid':True})
    monkeypatch.setattr(m,'reusable_profile',lambda *a:True)
    monkeypatch.setattr(c,'command',lambda *a:pytest.fail('Completed evaluation dispatched again'))
    try:c.evaluate('C1',(4,4),42)
    finally:c.close()
    assert len(validated)==1


def test_failed_allocation_bootstrap_records_terminal_error(tmp_path,monkeypatch):
    from fastwam.loop.campaign import Manifest
    m=module()
    def init(self,args):
        self.root=tmp_path;self.args=args;self.deadline=args.deadline
        self.manifest=Manifest(tmp_path/'manifest.json',{})
    monkeypatch.setattr(m.BaseCampaign,'__init__',init)
    monkeypatch.setattr(m,'select_allocations',lambda *a:[dict(job_id='1',deadline=time.time()+100)])
    monkeypatch.setattr(m.ParallelCampaign,'configuration',{'allocation_ids':['1']})
    monkeypatch.setattr(m.ParallelCampaign,'execute',lambda *a:(_ for _ in ()).throw(RuntimeError('bootstrap failed')))
    with pytest.raises(RuntimeError,match='bootstrap failed'):
        m.ParallelCampaign(SimpleNamespace(deadline=time.time()+100))
    state=json.loads((tmp_path/'manifest.json').read_text())
    assert state['status']=='stopped' and state['stop_kind']=='error'


def test_interrupted_rollout_seeds_missing_cache_from_valid_local_profile(tmp_path,monkeypatch):
    m,c,base=fixture_campaign(tmp_path,monkeypatch)
    output=tmp_path/'evaluation';output.mkdir()
    profile={'hardware':'same'};(output/'latency.json').write_text(json.dumps(profile))
    cache=tmp_path/'shared'/'cache.json'
    monkeypatch.setattr(m,'profile_cache_key',lambda args:profile)
    monkeypatch.setattr(m,'reusable_profile',lambda data,expected:data==expected)
    called=[]
    def execute(command,log,slot):
        assert '--profile-only' not in command
        assert json.loads(cache.read_text())==profile
        called.append(command)
    monkeypatch.setattr(c,'execute',execute)
    cmd=['python','evaluate.py','--profile','--profile-cache',str(cache),'--profile-architecture','{}',
         '--output',str(output),'--gpus','0,1,2,3','--stats','stats.json','--kv','1','--ka','1']
    try:c.command(cmd,output/'manager.log')
    finally:c.close()
    assert len(called)==1
