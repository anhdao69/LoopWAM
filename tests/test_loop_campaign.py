"""Campaign safety contracts; no model/GPU imports required."""
import json

import pytest


def records(successes=45, seed=0):
    return [dict(task_id=i, seed=seed, status="complete", total_episodes=50,
                 success_episodes=list(range(successes)),
                 failure_episodes=list(range(successes, 50))) for i in range(10)]


def test_matrix_keeps_fourteen_trainings_and_equal_stage3_budgets():
    from fastwam.loop.campaign import run_matrix
    runs = {r.id: r for r in run_matrix()}
    assert len(runs) == 14
    assert set(runs) == {"P0-S", "C1", "C2", "C3", "S1-L2", "S1-L3",
                         "S2-cont", "S2-base", "S3-coupled", "S3-late",
                         "S3-Konly", "S3-2stage", "F-Long-s1", "F-Long-s2"}
    assert runs["S3-late"].steps == 8000
    assert runs["S3-Konly"].steps == runs["S3-2stage"].steps == 14000
    assert runs["F-Long-s1"].steps == 22000
    assert (2,2) in runs["S3-Konly"].pairs
    assert len(runs["F-Long-s1"].pairs) == 10


@pytest.mark.parametrize("mutation", ["missing_task", "error", "missing_episode", "duplicate"])
def test_incomplete_or_error_evaluation_never_scores(mutation):
    from fastwam.loop.evaluation import summarize_tasks
    tasks = records()
    if mutation == "missing_task":
        tasks.pop()
    elif mutation == "error":
        tasks[0]["status"] = "error"
    elif mutation == "missing_episode":
        tasks[0]["failure_episodes"].pop()
    else:
        tasks[0]["success_episodes"].append(0)
    with pytest.raises(ValueError):
        summarize_tasks(tasks, seed=0)


def test_complete_summary_has_paired_outcomes_and_wilson_interval():
    from fastwam.loop.evaluation import summarize_tasks
    result = summarize_tasks(records(), seed=0)
    assert result["successes"] == 450
    assert result["episodes"] == 500
    assert result["success_pct"] == 90
    assert len(result["outcomes"]) == 500
    assert result["wilson95_pct"][0] < 90 < result["wilson95_pct"][1]


def test_close_effect_needs_second_seed_and_paired_significance():
    from fastwam.loop.campaign import Evidence, compare
    a = Evidence({(0, i): i < 460 for i in range(500)})
    b = Evidence({(0, i): i < 445 for i in range(500)})
    assert compare(a, b)["status"] == "needs_second_seed"
    a.outcomes.update({(1, i): i < 460 for i in range(500)})
    b.outcomes.update({(1, i): i < 445 for i in range(500)})
    result = compare(a, b)
    assert result["status"] == "clear"
    assert result["gap_pp"] == pytest.approx(3)
    assert result["mcnemar_p"] < 0.05


def test_unpaired_outcomes_refuse_comparison():
    from fastwam.loop.campaign import Evidence, compare
    with pytest.raises(ValueError, match="paired"):
        compare(Evidence({(0, 0): True}), Evidence({(1, 0): True}))


def test_gate_thresholds_and_missing_latency_stop():
    from fastwam.loop.campaign import gate_g0, gate_gp, gate_g3, Evidence
    def ev(n):
        return Evidence({(seed, i): i < n for seed in (0, 1) for i in range(500)})
    assert gate_g0(ev(460), ev(470), infrastructure_passed=True)["status"] == "pass"
    assert gate_g0(ev(430), ev(470), infrastructure_passed=True)["status"] == "fail"
    assert gate_g0(ev(460), ev(470), infrastructure_passed=False)["status"] == "blocked"
    assert gate_gp(ev(450), ev(450))["status"] == "fail"
    assert gate_g3({(4,1): ev(475), (1,1): ev(440), (4,2): ev(475), (2,2): ev(450)},
                   ev(450), ev(465), latencies={})["status"] == "blocked"


def test_manifest_refuses_changed_protocol_and_preserves_complete_run(tmp_path):
    from fastwam.loop.campaign import Manifest
    path = tmp_path / "manifest.json"
    state = Manifest(path, {"seed": 42})
    state.update_run("C1", status="complete", step=8000)
    reopened = Manifest(path, {"seed": 42})
    assert reopened.runs["C1"]["status"] == "complete"
    with pytest.raises(ValueError, match="protocol"):
        Manifest(path, {"seed": 43})
    with pytest.raises(ValueError, match="complete"):
        reopened.update_run("C1", status="training")


def test_g2_checks_both_shallow_truncations_and_full_depth_tax():
    from fastwam.loop.campaign import Evidence, gate_g2
    def ev(n):
        return Evidence({(s, i): i < n for s in (0, 1) for i in range(500)})
    base = {(4,4): ev(470), (2,2): ev(455), (1,1): ev(450)}
    cont = {(4,4): ev(475), (2,2): ev(430), (1,1): ev(420)}
    assert gate_g2(base, cont, ev(450))["status"] == "pass"
    cont[(2,2)] = ev(450)
    assert gate_g2(base, cont, ev(450))["status"] == "fail"
    cont[(2,2)] = ev(430)
    base[(4,4)] = ev(460)
    assert gate_g2(base, cont, ev(450))["status"] == "fail"


def test_g3_uses_measured_latency_segment_without_extrapolation():
    from fastwam.loop.campaign import Evidence, gate_g3
    def ev(n):
        return Evidence({(s, i): i < n for s in (0, 1) for i in range(500)})
    model = {(4,1): ev(475), (1,1): ev(440), (4,2): ev(475), (2,2): ev(450)}
    assert gate_g3(model, ev(450), ev(470),
                   {"C2": 100, "C3": 200, "4,1": 120, "4,2": 150})["status"] == "pass"
    assert gate_g3(model, ev(450), ev(470),
                   {"C2": 100, "C3": 200, "4,1": 220, "4,2": 250})["status"] == "fail"


def test_gp_accepts_only_clear_openloop_improvement():
    from fastwam.loop.campaign import Evidence, gate_gp
    ev = Evidence({(0, i): i < 450 for i in range(500)})
    assert gate_gp(ev, ev, ol1_c3=.8, ol1_c2=1)["status"] == "pass"
    assert gate_gp(ev, ev, ol1_c3=.95, ol1_c2=1)["status"] == "fail"


def test_second_seed_discordance_cannot_be_ignored():
    from fastwam.loop.campaign import Evidence, compare
    # 3 pp net effect, but high discordance: 206 wins vs176 losses in1000.
    a = Evidence({(s, i): i < 206 for s in (0,1) for i in range(500)})
    b = Evidence({(s, i): 206 <= i < 382 for s in (0,1) for i in range(500)})
    # Reset the second half to equal outcomes to give the stated 3pp aggregate.
    for i in range(500):
        a.outcomes[(1,i)] = b.outcomes[(1,i)] = False
    assert compare(a, b)["gap_pp"] == 3
    assert compare(a, b)["status"] == "ambiguous"


def test_smoke_finite_complete_and_decreasing_losses_required():
    from fastwam.loop.campaign import gate_smoke
    rows = [dict(global_step=(i+1)*100, loss=2-i*.01) for i in range(20)]
    assert gate_smoke(rows)["status"] == "pass"
    assert gate_smoke(rows[:-1])["status"] == "blocked"
    rows[10]["loss"] = float("nan")
    assert gate_smoke(rows)["status"] == "fail"
    for i, row in enumerate(rows):
        row["loss"] = 1+i*.01
    assert gate_smoke(rows)["status"] == "fail"


def test_konly_cannot_win_with_coupled_budget_collapse():
    from fastwam.loop.campaign import Campaign, Evidence
    campaign=object.__new__(Campaign)
    def score(run,pair):
        if run=='S3-Konly': return 200 if pair==(2,2) else 490
        return 450
    campaign.evidence=lambda run,pair,seeds: Evidence({(s,i):i<score(run,pair) for s in seeds for i in range(500)})
    campaign.read_evidence=lambda run,pairs,seeds: Evidence({(s,p,i):i<score(run,p) for s in seeds for p in pairs for i in range(500)})
    selected=campaign.select_stage3([42,43])
    assert selected['winner']!='S3-Konly'


def test_architecture_batches_preserve_global128_and_compute_accumulation():
    from fastwam.loop.campaign import resolve_batches
    batches = resolve_batches('loopwam=16,untied30=4,untied_v30a12=8', default_micro=8)
    assert batches['loopwam'] == {'micro_batch':16, 'grad_accum':2}
    assert batches['untied30'] == {'micro_batch':4, 'grad_accum':8}
    assert batches['untied12'] == {'micro_batch':8, 'grad_accum':4}
    assert resolve_batches('{"loopwam":16}', default_micro=8) == resolve_batches('loopwam=16', default_micro=8)
    for pair in batches.values():
        assert 4 * pair['micro_batch'] * pair['grad_accum'] == 128


@pytest.mark.parametrize('value', ['loopwam=3', 'loopwam=0', 'loopwam=-1', 'untied3=4', '{"loopwam":true}', 'loopwam=8,loopwam=16'])
def test_invalid_batch_map_refuses_to_change_experiment(value):
    from fastwam.loop.campaign import resolve_batches
    with pytest.raises(ValueError):
        resolve_batches(value, default_micro=8)


def test_deadline_preserves_reserve_and_resolves_slurm_endtime(monkeypatch):
    from fastwam.loop.campaign import resolve_deadline, remaining_training_seconds
    from types import SimpleNamespace
    from datetime import datetime
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    def scontrol(cmd, **kwargs):
        assert cmd == ['scontrol', 'show', 'job', '123', '-o']
        return SimpleNamespace(stdout='JobId=123 EndTime=2026-10-06T08:02:48 JobState=RUNNING')
    monkeypatch.setattr('fastwam.loop.campaign.subprocess.run', scontrol)
    assert resolve_deadline(None) == datetime.fromisoformat('2026-10-06T08:02:48').timestamp()
    assert resolve_deadline(2000) == 2000
    assert remaining_training_seconds(2000, now=1000, reserve=180) == 970
    with pytest.raises(TimeoutError):
        remaining_training_seconds(1200, now=1000, reserve=180)


def test_deadline_kills_descendant_process_group(tmp_path, monkeypatch):
    import os
    import sys
    import time
    from fastwam.loop.evaluation import run_process_group
    pid_path = tmp_path / 'child.pid'
    # The allocated clock expires after the child exists, avoiding a startup
    # race on a busy training node while exercising the real process cleanup.
    monkeypatch.setattr('fastwam.loop.evaluation.time.time', lambda: 2 if pid_path.exists() else 0)
    script = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(30)"
    with pytest.raises(TimeoutError):
        run_process_group([sys.executable,'-c',script,str(pid_path)], deadline=1)
    child = int(pid_path.read_text())
    stat = __import__('pathlib').Path(f'/proc/{child}/stat')
    assert not stat.exists() or stat.read_text().split()[2] == 'Z'


def campaign_args(tmp_path):
    from types import SimpleNamespace
    teacher = tmp_path / 'teacher.pt'
    stats = tmp_path / 'stats.json'
    teacher.write_bytes(b'fixture checkpoint; no model loading')
    stats.write_text('{}')
    return SimpleNamespace(output=str(tmp_path / 'campaign'), teacher=str(teacher), stats=str(stats),
        seed=42, eval_seeds=[42,43], micro_batch=8, micro_batch_map='loopwam=16', grad_accum=None,
        gradient_checkpointing=False, gpus='0,1,2,3', zero_stage=1, well_above_pp=3,
        converted_dir=str(tmp_path / 'converted'), latency_match_tolerance=.05,
        deadline=2000, checkpoint_reserve_seconds=180, infrastructure=str(tmp_path / 'proof.json'),
        workers=2, text_cache=str(tmp_path / 'text'))


def test_new_allocation_may_resume_but_batch_or_checkpointing_changes_cannot(tmp_path):
    from fastwam.loop.campaign import Campaign
    args = campaign_args(tmp_path)
    first = Campaign(args)
    first.manifest.update_run('C1', status='complete', step=8000)
    args.deadline = 100000
    resumed = Campaign(args)
    assert resumed.manifest.runs['C1']['status'] == 'complete'
    args.micro_batch_map = 'loopwam=8'
    with pytest.raises(ValueError, match='protocol'):
        Campaign(args)
    args.micro_batch_map = 'loopwam=16'
    args.gradient_checkpointing = True
    with pytest.raises(ValueError, match='protocol'):
        Campaign(args)


def test_pending_campaign_report_contains_all_fourteen_runs(tmp_path):
    import csv
    from fastwam.loop.campaign import Campaign
    campaign = Campaign(campaign_args(tmp_path))
    campaign.write_tables()
    rows = list(csv.DictReader((campaign.root / 'results.csv').open()))
    assert len(rows) == 14
    assert all(row['status'] == 'pending' and row['success_pct'] == '' for row in rows)
    assert set(row['run'] for row in rows) == set(campaign.matrix)


def test_expired_evaluation_keeps_finished_tasks_without_summary(tmp_path):
    import time
    from types import SimpleNamespace
    from fastwam.loop.evaluation import evaluate, atomic_json
    checkpoint, stats = tmp_path / 'checkpoint.pt', tmp_path / 'stats.json'
    checkpoint.write_bytes(b'fixture')
    stats.write_text('{}')
    output = tmp_path / 'evaluation'
    saved = records(seed=42)[0]
    atomic_json(output / 'task_0.json', saved)
    args = SimpleNamespace(output=str(output), checkpoint=str(checkpoint), stats=str(stats),
        text_cache=str(tmp_path), seed=42, kv=4, ka=4, teacher=False, profile=False,
        gpus='0,1,2,3', deadline=time.time()-1, deadline_reserve_seconds=180)
    with pytest.raises(TimeoutError):
        evaluate(args)
    assert json.loads((output / 'task_0.json').read_text()) == saved
    assert not (output / 'summary.json').exists()
    runtime = json.loads((output / 'runtime.json').read_text())
    assert runtime['status'] == 'interrupted' and runtime['completed_tasks'] == 1


def test_width_gate_stops_before_other_stage1_trainings(tmp_path, monkeypatch):
    from fastwam.loop.campaign import Campaign, Evidence, GateStopped
    args = campaign_args(tmp_path)
    __import__('pathlib').Path(args.infrastructure).write_text(json.dumps(dict(status='pass', all_14_tests_passed=True)))
    campaign = Campaign(args)
    smoke = campaign.root / 'P0-S/metrics.jsonl'
    smoke.parent.mkdir()
    smoke.write_text(''.join(json.dumps(dict(global_step=(i+1)*100, loss=2-i*.01))+'\n' for i in range(20)))
    trained = []
    monkeypatch.setattr(campaign, 'train', trained.append)
    monkeypatch.setattr(campaign, 'evaluate', lambda *args: None)
    monkeypatch.setattr(campaign, 'evidence', lambda run, pair, seeds:
        Evidence({(s,i):i<(475 if run=='teacher' else 430) for s in seeds for i in range(500)}))
    with pytest.raises(GateStopped, match='G0'):
        campaign.run()
    assert trained == ['P0-S', 'C1']


def test_profile_reuse_requires_matching_runtime_and_complete_measurement():
    from fastwam.loop.evaluation import reusable_profile
    key = dict(architecture='loopwam:r32:4,2', device_driver='H100,580', source_sha256='abc')
    result = dict(cache_key=key, status='complete', warmup=50, calls=500,
                  eager=dict(p50_ms=20,p90_ms=21,p99_ms=22), compiled=dict(p50_ms=10,p90_ms=11,p99_ms=12))
    assert reusable_profile(result, key)
    assert not reusable_profile(result, dict(key, source_sha256='changed'))
    assert not reusable_profile(result, dict(key, device_driver='4090,580'))
    result['calls'] = 10
    assert not reusable_profile(result, key)


def test_only_allocation_stops_are_eligible_for_future_continuation():
    import subprocess
    from fastwam.loop.campaign import stop_kind, GateStopped
    assert stop_kind(TimeoutError()) == 'allocation_end'
    assert stop_kind(subprocess.CalledProcessError(3, ['trainer'])) == 'allocation_end'
    assert stop_kind(GateStopped('G0 failed')) == 'gate_failed'
    assert stop_kind(subprocess.CalledProcessError(1, ['trainer'])) == 'error'
    assert stop_kind(ValueError('bad checkpoint')) == 'error'
    assert stop_kind(InterruptedError('cancelled externally')) == 'error'


@pytest.mark.parametrize('kind,additional,submitted', [('allocation_end',2,True), ('allocation_end',0,False),
                                                       ('gate_failed',2,False), ('error',2,False), ('complete',2,False)])
def test_continuation_script_only_queues_bounded_allocation_stops(tmp_path, kind, additional, submitted):
    """Run the real wrapper, substituting only training and the external scheduler."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    from fastwam.loop.evaluation import ROOT
    scripts = tmp_path / 'scripts/loopwam'
    scripts.mkdir(parents=True)
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'manifest.json').write_text(json.dumps(dict(launch_arguments={})))
    runner = scripts / 'run_campaign.sh'
    runner.write_text(f'#!{sys.executable}\n' + '''import json,os,sys,time
from pathlib import Path
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
path=Path(args['--output'])/'manifest.json'
record=json.loads(path.read_text())
kind=os.environ['FAKE_STOP_KIND']
record.update(status='complete' if kind=='complete' else 'stopped', stop_kind=kind,
              stopped_at=time.time(), allocation={'job_id':os.environ['SLURM_JOB_ID']})
path.write_text(json.dumps(record))
raise SystemExit(0 if kind=='complete' else 1)
''')
    runner.chmod(0o755)
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    capture = tmp_path / 'scheduler.json'
    scheduler = binaries / 'sbatch'
    scheduler.write_text(f'#!{sys.executable}\n' + '''import json,os,sys
from pathlib import Path
Path(os.environ['FAKE_SCHEDULER_CAPTURE']).write_text(json.dumps(sys.argv[1:]))
print('123456')
''')
    scheduler.chmod(0o755)
    env = dict(os.environ, LOOPWAM_REPO=str(tmp_path), SLURM_JOB_ID='99', FAKE_STOP_KIND=kind,
               FAKE_SCHEDULER_CAPTURE=str(capture), PATH=str(binaries)+os.pathsep+os.environ['PATH'])
    result = subprocess.run(['bash',str(ROOT/'scripts/loopwam/continue_campaign.sbatch'),
                             str(output),str(tmp_path/'proof.json'),str(additional)], env=env,
                            capture_output=True, text=True)
    assert result.returncode == (0 if kind=='complete' else 1), result.stderr
    assert capture.exists() == submitted
    if submitted:
        command = json.loads(capture.read_text())
        assert '--dependency=afterany:99' in command
        assert command[-1] == '1'
        chain = json.loads((output/'continuation_chain.jsonl').read_text())
        assert chain['submitted_job_id'] == '123456' and chain['remaining_additional'] == 1
