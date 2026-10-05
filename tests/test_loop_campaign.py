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
    assert len(runs["S3-Konly"].pairs) == 3
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
