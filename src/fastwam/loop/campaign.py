"""Resumable, gated 14-training LoopWAM screening campaign (LIBERO-Long only).

No conditional ablations, control continuations or all-suite training are
launched. A failed/ambiguous gate stops execution and leaves reviewable evidence.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .evaluation import (ROOT, PROTOCOL, atomic_json, file_identity, sha256_file, wilson,
                         run_process_group, allocation_signals)

ALL_PAIRS = tuple((v, a) for v in range(1, 5) for a in range(1, v + 1))
FIVE_PAIRS = ((4, 4), (4, 2), (4, 1), (2, 2), (1, 1))
COUPLED_PAIRS = ((1, 1), (2, 2), (4, 4))
ARCHITECTURES = ("loopwam", "untied30", "untied12", "untied_v30a12")


def resolve_batches(value=None, default_micro=8, grad_accum=None):
    """Resolve per-architecture micro batches with exact global batch128."""
    def valid(micro):
        if type(micro) is not int or micro <= 0 or 32 % micro:
            raise ValueError("Micro batches must be positive integer divisors of32 on four GPUs")
        return dict(micro_batch=micro, grad_accum=32 // micro)
    base = valid(default_micro)
    if grad_accum is not None and grad_accum != base["grad_accum"]:
        raise ValueError("Default micro batch and grad accumulation must give global batch128")
    overrides = {}
    if value:
        if isinstance(value, dict):
            overrides = value
        elif value.lstrip().startswith("{"):
            def unique(pairs):
                result = {}
                for key, val in pairs:
                    if key in result:
                        raise ValueError(f"Duplicate architecture: {key}")
                    result[key] = val
                return result
            overrides = json.loads(value, object_pairs_hook=unique)
        else:
            for item in value.split(","):
                key, micro = item.strip().split("=", 1)
                if key in overrides:
                    raise ValueError(f"Duplicate architecture: {key}")
                overrides[key] = int(micro)
    unknown = set(overrides) - set(ARCHITECTURES)
    if unknown:
        raise ValueError(f"Unknown architectures: {sorted(unknown)}")
    return {arch: valid(overrides.get(arch, default_micro)) for arch in ARCHITECTURES}


def resolve_deadline(explicit):
    if explicit is not None:
        if not math.isfinite(explicit) or explicit <= 0:
            raise ValueError("Deadline must be a positive epoch timestamp")
        return float(explicit)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    result = subprocess.run(["scontrol", "show", "job", job_id, "-o"],
                            capture_output=True, text=True, check=True, timeout=10)
    fields = dict(item.split("=", 1) for item in result.stdout.split() if "=" in item)
    end = fields.get("EndTime")
    if not end or end in ("Unknown", "N/A", "None", "UNLIMITED"):
        raise ValueError("Slurm EndTime unavailable; provide --deadline epoch explicitly")
    # scontrol emits cluster-local wall time; timestamp() uses the same node timezone.
    return datetime.fromisoformat(end).timestamp()


def remaining_training_seconds(deadline, *, now=None, reserve=180):
    if reserve < 180:
        raise ValueError("Checkpoint reserve must be at least180 seconds")
    if deadline is None:
        return None
    remaining = deadline - (time.time() if now is None else now) - 30
    if remaining <= reserve:
        raise TimeoutError("Insufficient allocation time for training plus checkpoint reserve")
    return remaining


@dataclass(frozen=True)
class Run:
    id: str
    arch: str = "loopwam"
    loss: str = "recipe"
    mode: str = "fixed"
    start: int = 0
    end: int = 8000
    fork: str | None = None
    pairs: tuple = ((4, 4),)
    seed_offset: int = 0

    @property
    def steps(self):
        return self.end - self.start


def run_matrix():
    return [
        Run("P0-S", loss="L2", end=2000),
        Run("C1", arch="untied30", loss="L3"),
        Run("C2", arch="untied12", loss="L3"),
        Run("C3", arch="untied_v30a12", loss="L3"),
        Run("S1-L2", loss="L2"), Run("S1-L3", loss="L3"),
        Run("S2-cont", start=8000, end=14000, fork="S1*", pairs=COUPLED_PAIRS),
        Run("S2-base", mode="coupled", start=8000, end=14000, fork="S1*", pairs=COUPLED_PAIRS),
        Run("S3-coupled", mode="coupled", start=14000, end=22000, fork="S2*", pairs=FIVE_PAIRS),
        Run("S3-late", mode="decoupled", start=14000, end=22000, fork="S2*", pairs=FIVE_PAIRS),
        Run("S3-Konly", mode="konly", start=8000, end=22000, fork="S1*", pairs=FIVE_PAIRS[:4]),
        Run("S3-2stage", mode="decoupled", start=8000, end=22000, fork="S1*", pairs=FIVE_PAIRS),
        Run("F-Long-s1", mode="three_stage", end=22000, pairs=ALL_PAIRS, seed_offset=1),
        Run("F-Long-s2", mode="three_stage", end=22000, pairs=ALL_PAIRS, seed_offset=2),
    ]


@dataclass
class Evidence:
    outcomes: dict

    @property
    def pct(self):
        if not self.outcomes:
            raise ValueError("Empty evidence")
        return 100 * sum(self.outcomes.values()) / len(self.outcomes)


def compare(a: Evidence, b: Evidence):
    if not a.outcomes or set(a.outcomes) != set(b.outcomes):
        raise ValueError("Comparisons require identical paired episode keys")
    gain = sum(bool(a.outcomes[k]) and not b.outcomes[k] for k in a.outcomes)
    loss = sum(bool(b.outcomes[k]) and not a.outcomes[k] for k in a.outcomes)
    discordant = gain + loss
    # Exact two-sided binomial McNemar test, stable even for thousands of pairs.
    if discordant:
        tail = min(gain, loss)
        terms = [math.lgamma(discordant + 1) - math.lgamma(k + 1)
                 - math.lgamma(discordant - k + 1) - discordant * math.log(2)
                 for k in range(tail + 1)]
        peak = max(terms)
        p = min(1.0, 2 * math.exp(peak) * sum(math.exp(t - peak) for t in terms))
    else:
        p = 1.0
    gap = round(a.pct - b.pct, 10)
    seeds = {k[0] for k in a.outcomes}
    if abs(gap) < 2:
        status = "tie"
    elif abs(gap) <= 4:
        status = "needs_second_seed" if len(seeds) < 2 else ("clear" if p < .05 else "ambiguous")
    else:
        status = "clear"
    return dict(status=status, gap_pp=gap, mcnemar_p=p, gain=gain, loss=loss,
                episodes=len(a.outcomes), eval_seeds=sorted(seeds))


def minimum(a, b, margin):
    evidence = compare(a, b)
    if evidence["status"] in ("needs_second_seed", "ambiguous"):
        status = "needs_second_seed" if evidence["status"] == "needs_second_seed" else "blocked"
    else:
        status = "pass" if evidence["gap_pp"] >= margin - 1e-9 else "fail"
    return dict(status=status, margin_pp=margin, comparison=evidence)


def combine(name, checks):
    statuses = {x["status"] for x in checks.values()}
    status = next((s for s in ("fail", "blocked", "needs_second_seed") if s in statuses), "pass")
    return dict(gate=name, status=status, checks=checks)


def gate_g0(c1, teacher, infrastructure_passed):
    return combine("G0", dict(infrastructure=dict(status="pass" if infrastructure_passed else "blocked"),
                              width=minimum(c1, teacher, -3)))


def gate_smoke(records, endpoint=2000):
    records = sorted({r["global_step"]: r for r in records}.values(), key=lambda r: r["global_step"])
    if len(records) < 20 or records[-1]["global_step"] != endpoint:
        return dict(status="blocked", reason="Need complete stage-end loss history and at least20 logged intervals")
    losses = [r.get("loss", float("nan")) for r in records]
    if not all(math.isfinite(v) for v in losses):
        return dict(status="fail", reason="Nonfinite smoke loss")
    first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    return dict(status="pass" if last < first else "fail", first_ten_mean=first, last_ten_mean=last,
                criterion="Finite loss; final10 logged-interval mean lower than initial10")


def gate_g1(winner, c1, c2):
    return combine("G1", dict(recovery=minimum(winner, c1, -2), control=minimum(winner, c2, 0)))


def gate_gp(c3, c2, ol1_c3=None, ol1_c2=None):
    check = minimum(c3, c2, 2)
    # 'Clearly better OL-1' is preregistered as a >10% reduction on the same holdout/noise grid.
    ol_pass = (ol1_c3 is not None and ol1_c2 is not None and ol1_c2 > 0
               and math.isfinite(ol1_c3) and math.isfinite(ol1_c2) and ol1_c3 < .9 * ol1_c2)
    if ol_pass:
        check = dict(status="pass", evidence="OL-1 reduction >10%", c3=ol1_c3, c2=ol1_c2)
    return combine("GP", dict(deep_video_premise=check))


def gate_g2(base, cont, c2, well_above_pp=3.0):
    return combine("G2", dict(full=minimum(base[(4,4)], cont[(4,4)], -1.5),
        k2_control=minimum(base[(2,2)], c2, 0), k1_control=minimum(base[(1,1)], c2, -3),
        k1_truncation=minimum(base[(1,1)], cont[(1,1)], well_above_pp),
        k2_truncation=minimum(base[(2,2)], cont[(2,2)], well_above_pp)))


def gate_g3(winner, c2, c3, latencies, latency_tolerance=.05):
    benefits = {"4,1_vs_1,1": minimum(winner[(4,1)], winner[(1,1)], 3),
                "4,2_vs_2,2": minimum(winner[(4,2)], winner[(2,2)], 3)}
    deep_status = "pass" if any(x["status"] == "pass" for x in benefits.values()) else next(
        (s for s in ("needs_second_seed", "blocked") if any(x["status"] == s for x in benefits.values())), "fail")
    latency_check = dict(status="blocked", reason="Measured comparable p50 latency required")
    if all(k in latencies for k in ("C2", "C3", "4,1", "4,2")):
        t2, t3 = latencies["C2"], latencies["C3"]
        points = {}
        for pair in ((4,1), (4,2)):
            t = latencies[f"{pair[0]},{pair[1]}"]
            same_latency = abs(t - t2) <= latency_tolerance * t2
            beat_control = minimum(winner[pair], c2, 2)
            matched = same_latency and beat_control["status"] == "pass"
            on_segment = t3 > t2 and t2 <= t <= t3
            line_pct = c2.pct + (c3.pct - c2.pct) * (t - t2) / (t3 - t2) if on_segment else None
            line_gap = winner[pair].pct - line_pct if on_segment else None
            above_line = on_segment and line_gap > 4
            # An interpolated success rate has no binary paired outcome, so do
            # not fabricate McNemar evidence for a 2--4pp line comparison.
            if matched or above_line:
                point_status = "pass"
            elif same_latency and beat_control["status"] in ("needs_second_seed", "blocked"):
                point_status = beat_control["status"]
            elif on_segment and 2 <= line_gap <= 4:
                point_status = "blocked"
            else:
                point_status = "fail"
            points[str(pair)] = dict(p50_ms=t, matched=matched, above_control_line=above_line,
                control_line_pct=line_pct, line_gap_pp=line_gap, comparison=beat_control, status=point_status)
        latency_status = "pass" if any(p["status"] == "pass" for p in points.values()) else next(
            (s for s in ("needs_second_seed", "blocked") if any(p["status"] == s for p in points.values())), "fail")
        latency_check = dict(status=latency_status, tolerance_fraction=latency_tolerance, points=points,
            line_rule="2--4pp above interpolated line needs a matched-latency experiment or reviewed statistical evidence")
    return combine("G3", dict(deep_video=dict(status=deep_status, pairs=benefits), latency=latency_check))


class Manifest:
    def __init__(self, path, protocol):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data["protocol"] != protocol:
                raise ValueError("Campaign protocol changed; use a fresh output directory")
        else:
            self.data = dict(protocol=protocol, runs={}, decisions={}, created_at=time.time())
            self.save()
        self.runs = self.data["runs"]

    def save(self):
        atomic_json(self.path, self.data)

    def update_run(self, run_id, **values):
        current = self.runs.get(run_id, {})
        if current.get("status") == "complete" and values.get("status", "complete") != "complete":
            raise ValueError(f"Cannot overwrite complete run {run_id}")
        current.update(values, updated_at=time.time())
        self.runs[run_id] = current
        self.save()


class GateStopped(RuntimeError):
    pass


def stop_kind(error):
    if isinstance(error, TimeoutError) or (isinstance(error, subprocess.CalledProcessError) and error.returncode == 3):
        return "allocation_end"
    return "gate_failed" if isinstance(error, GateStopped) else "error"


class Campaign:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.matrix = {run.id: run for run in run_matrix()}
        self.batches = resolve_batches(args.micro_batch_map, args.micro_batch, args.grad_accum)
        protocol = dict(matrix=[asdict(r) for r in run_matrix()], evaluation=PROTOCOL,
            teacher=file_identity(args.teacher), stats_sha256=sha256_file(args.stats),
            training_seed=args.seed, evaluation_seeds=args.eval_seeds, batch=128,
            batches=self.batches, gradient_checkpointing=args.gradient_checkpointing, gpus=args.gpus,
            zero_stage=args.zero_stage, well_above_pp=args.well_above_pp,
            converted_dir=str(Path(args.converted_dir).resolve()),
            latency_match_tolerance=args.latency_match_tolerance,
            teacher_reproduction="Long only; full-suite reproduction deferred",
            ol1_clear_improvement_fraction=.10)
        # JSON normalization avoids tuple/list differences on resume.
        self.manifest = Manifest(self.root / "manifest.json", json.loads(json.dumps(protocol)))
        self.decisions = self.manifest.data["decisions"]
        self.deadline = args.deadline
        self.manifest.data["allocation"] = dict(job_id=os.environ.get("SLURM_JOB_ID"),
            deadline=self.deadline, checkpoint_reserve_seconds=args.checkpoint_reserve_seconds,
            started_at=time.time())
        # Per-allocation deadlines and derived options must not constrain a later resume.
        self.manifest.data["launch_arguments"] = {k: v for k, v in vars(args).items()
            if k not in ("deadline", "plan", "report_only")}
        self.manifest.save()

    def command(self, cmd, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        print("RUN", shlex.join(map(str, cmd)), flush=True)
        with log.open("a") as stream:
            run_process_group(list(map(str, cmd)), cwd=ROOT, env=dict(os.environ,
                CUDA_VISIBLE_DEVICES=self.args.gpus), stdout=stream, stderr=subprocess.STDOUT,
                deadline=self.deadline - 30 if self.deadline is not None else None)

    def init_checkpoint(self, arch):
        name = "loopwam_r32" if arch == "loopwam" else arch
        path = Path(self.args.converted_dir).resolve() / f"{name}.pt"
        if not path.exists():
            self.command([sys.executable, "-m", "fastwam.loop.convert", "--teacher", self.args.teacher,
                          "--output", path, "--arch", arch], self.root / "logs" / f"convert_{arch}.log")
        identity = file_identity(path)
        saved = self.manifest.data.setdefault("initial_checkpoints", {})
        if arch in saved and saved[arch] != identity:
            raise ValueError(f"Converted initialization changed after first use: {path}")
        saved[arch] = identity
        self.manifest.save()
        return path

    def train(self, run_id):
        run = self.matrix[run_id]
        output = self.root / run_id
        previous = self.manifest.runs.get(run_id, {})
        if previous.get("status") in ("trained", "complete"):
            self._validate_trained(run)
            return
        init = self.init_checkpoint(run.arch)
        resume = None
        # A branch restarts from its own latest checkpoint after preemption.
        if (output / "state/latest.json").exists():
            resume = output / "state"
        elif run.fork:
            parent_id = self.decisions[run.fork]["winner"]
            resume = self.root / parent_id / "state"
            metadata = json.loads((resume / "latest.json").read_text())
            if metadata.get("global_step", metadata.get("step")) != run.start:
                raise ValueError(f"Fork {run_id} requires absolute step {run.start}: {resume}")
        loss = run.loss if run.loss != "recipe" else self.decisions["S1*"]["loss"]
        batch = self.batches[run.arch]
        budget = remaining_training_seconds(self.deadline, reserve=self.args.checkpoint_reserve_seconds)
        cmd = ["torchrun", "--standalone", "--nproc_per_node=4", "scripts/loopwam/train.py",
               "--init", init, "--output", output, "--mode", run.mode,
               "--loss", loss, "--max-steps", run.end, "--seed", self.args.seed + run.seed_offset,
               "--micro-batch", batch["micro_batch"], "--grad-accum", batch["grad_accum"],
               "--zero-stage", self.args.zero_stage, "--workers", self.args.workers,
               "--stats", self.args.stats, "--teacher", self.args.teacher]
        if self.args.gradient_checkpointing:
            cmd += ["--gradient-checkpointing"]
        if budget is not None:
            cmd += ["--time-budget-seconds", budget, "--checkpoint-reserve-seconds", self.args.checkpoint_reserve_seconds]
        if resume:
            cmd += ["--resume", resume]
        if run.mode == "three_stage":
            selected = self.decisions["S3*"]["winner"]
            stage2 = "konly" if selected == "S3-Konly" else "decoupled" if selected == "S3-2stage" else "coupled"
            stage3 = "konly" if selected == "S3-Konly" else "coupled" if selected == "S3-coupled" else "decoupled"
            cmd += ["--stage2-mode", stage2, "--stage3-mode", stage3]
        self.manifest.update_run(run_id, status="training", command=list(map(str, cmd)),
                                 fork=str(resume) if resume else None, loss=loss)
        command_started = time.time()
        try:
            self.command(cmd, output / "train.log")
            self._validate_trained(run)
            self.manifest.update_run(run_id, status="trained", step=run.end)
        except BaseException as exc:
            self.manifest.update_run(run_id, status="interrupted", error=str(exc))
            # torchrun wraps a worker's exit3 in its own exit1. Accept only a
            # fresh committed deadline-stop artifact, not an old partial run.
            timing_path = output / "timing.json"
            if isinstance(exc, subprocess.CalledProcessError) and budget is not None and timing_path.exists():
                timing = json.loads(timing_path.read_text())
                if (timing_path.stat().st_mtime >= command_started and timing.get("interrupted")
                        and not timing.get("complete") and timing.get("signal") is None
                        and self.deadline - time.time() <= self.args.checkpoint_reserve_seconds + 60):
                    raise TimeoutError(f"{run_id} saved its checkpoint before allocation end") from exc
            raise

    def _validate_trained(self, run):
        output = self.root / run.id
        timing = json.loads((output / "timing.json").read_text())
        if not timing.get("complete") or timing.get("global_step") != run.end:
            raise ValueError(f"{run.id} did not finish its fixed step boundary {run.end}")
        for name in ("ema.pt", "raw.pt", "state/latest.json"):
            if not (output / name).is_file():
                raise ValueError(f"Missing complete training artifact: {output / name}")

    def eval_path(self, run_id, pair, seed):
        return self.root / run_id / "eval" / f"kv{pair[0]}_ka{pair[1]}" / f"seed{seed}"

    def evaluate(self, run_id, pair, seed):
        output = self.eval_path(run_id, pair, seed)
        if (output / "summary.json").exists():
            self.read_evidence(run_id, [pair], [seed])
            if seed != self.args.eval_seeds[0] or (output / "latency.json").exists():
                return
        checkpoint = Path(self.args.teacher) if run_id == "teacher" else self.root / run_id / "ema.pt"
        cmd = [sys.executable, "scripts/loopwam/evaluate.py", "--checkpoint", checkpoint,
               "--stats", self.args.stats, "--output", output, "--seed", seed,
               "--kv", pair[0], "--ka", pair[1], "--gpus", self.args.gpus,
               "--text-cache", self.args.text_cache]
        if self.deadline is not None:
            cmd += ["--deadline", self.deadline, "--deadline-reserve-seconds", self.args.checkpoint_reserve_seconds]
        if run_id == "teacher":
            cmd += ["--teacher"]
        # Profiles are independent of eval seed and exclude concurrent rollout load.
        if seed == self.args.eval_seeds[0]:
            cmd += ["--profile"]
            arch = "teacher" if run_id == "teacher" else self.matrix[run_id].arch
            initialization = (file_identity(self.args.teacher) if arch == "teacher"
                              else self.manifest.data.get("initial_checkpoints", {}).get(arch))
            if initialization is not None:
                key = json.dumps(dict(arch=arch, initialization=initialization, pair=pair), sort_keys=True)
                cmd += ["--profile-cache", self.root / "latency" / f"{arch}_kv{pair[0]}_ka{pair[1]}.json",
                        "--profile-architecture", key]
        self.command(cmd, output / "manager.log")
        self.read_evidence(run_id, [pair], [seed])
        self.write_tables()

    def read_evidence(self, run_id, pairs, seeds):
        outcomes = {}
        for pair in pairs:
            for seed in seeds:
                summary = json.loads((self.eval_path(run_id, pair, seed) / "summary.json").read_text())
                if (summary.get("status") != "complete" or summary.get("episodes") != 500
                        or len(summary.get("outcomes", [])) != 500):
                    raise ValueError(f"Incomplete evaluation for {run_id}/{pair}/{seed}")
                protocol = summary["protocol"]
                if any(protocol.get(k) != v for k, v in PROTOCOL.items()):
                    raise ValueError("Evaluation does not match the registered LIBERO protocol")
                if protocol.get("seed") != seed or (protocol.get("kv"), protocol.get("ka")) != pair:
                    raise ValueError("Evaluation budget or seed differs from its output location")
                checkpoint = Path(self.args.teacher) if run_id == "teacher" else self.root / run_id / "ema.pt"
                if protocol.get("checkpoint") != file_identity(checkpoint):
                    raise ValueError("Evaluation checkpoint changed after measurement")
                if protocol.get("stats_sha256") != sha256_file(self.args.stats):
                    raise ValueError("Evaluation uses different normalization stats")
                states = summary.get("initial_state_sha256", {})
                if set(states) != {str(t) for t in range(10)}:
                    raise ValueError("Missing initial-state provenance")
                canonical = self.manifest.data.get("initial_state_sha256")
                if canonical is None:
                    self.manifest.data["initial_state_sha256"] = states
                    self.manifest.save()
                elif states != canonical:
                    raise ValueError("Evaluation initial states do not match the paired campaign")
                for row in summary["outcomes"]:
                    key = (seed, pair, row["task_id"], row["episode_id"])
                    if key in outcomes or row["seed"] != seed or not isinstance(row["success"], bool):
                        raise ValueError("Invalid paired outcome records")
                    outcomes[key] = row["success"]
                expected = {(seed, pair, t, e) for t in range(10) for e in range(50)}
                if not expected <= outcomes.keys():
                    raise ValueError("Missing paired outcomes")
        return Evidence(outcomes)

    def openloop_premise(self):
        """Accept a held-out OL-1 alternative only with matched recorded inputs."""
        records = []
        states = []
        for run_id in ("C3", "C2"):
            path = self.root / run_id / "open_loop/step_00008000.json"
            if not path.exists():
                return None, None
            value = json.loads(path.read_text())
            state = json.loads((self.root / run_id / "state/latest.json").read_text())
            if (value.get("global_step") != 8000 or state.get("global_step") != 8000
                    or not state.get("complete") or value.get("tau_grid") != [.1, .3, .5, .7, .9]
                    or state.get("stats_sha256") != sha256_file(self.args.stats)
                    or state.get("teacher") != str(Path(self.args.teacher).resolve())
                    or len(set(value.get("window_ids", []))) != 20):
                return None, None
            records.append(value)
            states.append(state)
        keys = ("window_ids", "seed", "panel")
        if any(records[0].get(k) is None or records[0][k] != records[1].get(k) for k in keys):
            return None, None
        if not states[0].get("manifest_sha256") or states[0]["manifest_sha256"] != states[1].get("manifest_sha256"):
            return None, None
        self.manifest.data["openloop_premise_evidence"] = dict(panel=records[0]["panel"],
            window_ids=records[0]["window_ids"], seed=records[0]["seed"],
            split_manifest_sha256=states[0]["manifest_sha256"], scope="20 fixed midpoint clips, not all validation windows")
        self.manifest.save()
        return tuple(r.get("metrics", {}).get("4_4", {}).get("ol1") for r in records)

    def evidence(self, run_id, pair, seeds):
        # Pair identity is deliberately removed for cross-budget comparisons.
        result = self.read_evidence(run_id, [pair], seeds)
        return Evidence({(key[0], key[2], key[3]): val for key, val in result.outcomes.items()})

    def resolve(self, name, required, decide):
        seeds = self.args.eval_seeds[:1]
        for attempt in range(2):
            for run_id, pairs in required.items():
                for pair in pairs:
                    for seed in seeds:
                        self.evaluate(run_id, pair, seed)
            result = decide(seeds)
            self.decisions[name] = result
            self.manifest.save()
            self.write_tables()
            if result["status"] != "needs_second_seed":
                break
            seeds = self.args.eval_seeds
        if result["status"] != "pass":
            raise GateStopped(f"{name}: {result['status']}. Evidence: {self.manifest.path}. No extra training launched.")
        return result

    def select_stage1(self, seeds):
        l2 = self.evidence("S1-L2", (4,4), seeds)
        l3 = self.evidence("S1-L3", (4,4), seeds)
        comparison = compare(l3, l2)
        status = comparison["status"]
        if status in ("needs_second_seed", "ambiguous"):
            return dict(status="needs_second_seed" if status == "needs_second_seed" else "blocked", comparison=comparison)
        winner = "S1-L3" if comparison["gap_pp"] >= 2 else "S1-L2"
        return dict(status="pass", winner=winner, loss=winner[-2:], comparison=comparison,
                    tie_rule="under 2pp: fewer loss terms")

    def select_stage3(self, seeds):
        coupled = {p: self.evidence("S3-coupled", p, seeds) for p in FIVE_PAIRS}
        eligible = []
        checks = {}
        for run_id in ("S3-late", "S3-coupled", "S3-2stage", "S3-Konly"):
            pair_constraints = [(4,4), (2,2)]
            check = combine(run_id, {str(p): minimum(self.evidence(run_id, p, seeds), coupled[p], -1.5)
                                     for p in pair_constraints})
            checks[run_id] = check
            if check["status"] in ("needs_second_seed", "blocked"):
                return dict(status=check["status"], checks=checks)
            if check["status"] == "pass":
                eligible.append(run_id)
        winner = eligible[0]
        primary = [(4,1), (4,2)]
        for run_id in eligible[1:]:
            comparison = compare(self.read_evidence(run_id, primary, seeds),
                                 self.read_evidence(winner, primary, seeds))
            if comparison["status"] in ("needs_second_seed", "ambiguous"):
                return dict(status="needs_second_seed" if comparison["status"] == "needs_second_seed" else "blocked",
                            checks=checks, comparison=comparison)
            if run_id == "S3-Konly" and winner == "S3-late":
                exceptions = [minimum(self.evidence(run_id, p, seeds), self.evidence(winner, p, seeds), 2) for p in primary]
                if any(c["status"] == "needs_second_seed" for c in exceptions):
                    return dict(status="needs_second_seed", checks=checks, konly=exceptions)
                if any(c["status"] == "blocked" for c in exceptions):
                    return dict(status="blocked", checks=checks, konly=exceptions)
                if any(c["status"] == "pass" for c in exceptions):
                    winner = run_id
            elif comparison["gap_pp"] >= 2:
                winner = run_id
            elif abs(comparison["gap_pp"]) < 2 and run_id == "S3-coupled":
                winner = run_id  # fewer sampling components, subject to the two named exceptions
        return dict(status="pass", winner=winner, constraints=checks,
                    konly_constraint="Both (4,4) and (2,2) retention constraints apply to every candidate")

    def latency_values(self, winner):
        values, profiles = {}, []
        for key, run_id, pair in (("C2", "C2", (4,4)), ("C3", "C3", (4,4)),
                                   ("4,1", winner, (4,1)), ("4,2", winner, (4,2))):
            path = self.eval_path(run_id, pair, self.args.eval_seeds[0]) / "latency.json"
            if not path.exists():
                return {}
            data = json.loads(path.read_text())
            if data.get("status") != "complete" or data.get("warmup") != 50 or data.get("calls") != 500:
                return {}
            values[key] = data["compiled"]["p50_ms"]
            profiles.append((data["device"], data["torch_version"], data["scope"], data["protocol"],
                             data.get("cache_key", {}).get("source_sha256"),
                             data.get("cache_key", {}).get("device_driver")))
        return values if all(p == profiles[0] for p in profiles) else {}

    def run(self):
        self.manifest.data["status"] = "running"
        for key in ("stop_kind", "stop_reason", "stopped_at"):
            self.manifest.data.pop(key, None)
        self.manifest.save()
        proof = json.loads(Path(self.args.infrastructure).read_text())
        infrastructure_passed = proof.get("status") == "pass" and proof.get("all_14_tests_passed") is True
        if not infrastructure_passed:
            raise GateStopped("P0-T infrastructure evidence must report status=pass and all_14_tests_passed=true")
        self.manifest.data["infrastructure_evidence"] = dict(path=str(Path(self.args.infrastructure).resolve()),
                                                           sha256=sha256_file(self.args.infrastructure), evidence=proof)
        self.manifest.save()
        self.train("P0-S")
        smoke = [json.loads(line) for line in (self.root / "P0-S/metrics.jsonl").read_text().splitlines()]
        self.decisions["P0-S"] = gate_smoke(smoke)
        self.manifest.save()
        if self.decisions["P0-S"]["status"] != "pass":
            raise GateStopped("P0-S finite/decreasing loss gate failed; inspect metrics before evaluation")
        self.evaluate("P0-S", (4,4), self.args.eval_seeds[0])
        self.manifest.update_run("P0-S", status="complete")
        for seed in self.args.eval_seeds:
            self.evaluate("teacher", (4,4), seed)
        teacher_pct = self.evidence("teacher", (4,4), self.args.eval_seeds).pct
        self.decisions["P0-R"] = dict(status="pass" if abs(teacher_pct - 95.2) <= 1.5 + 1e-9 else "fail",
            scope="Long only, two eval seeds", observed_pct=teacher_pct, reference_pct=95.2,
            tolerance_pp=1.5, full_libero="deferred")
        self.manifest.save()
        if self.decisions["P0-R"]["status"] != "pass":
            raise GateStopped("P0-R Long teacher reproduction failed; diagnose before student screening")
        self.train("C1")
        self.evaluate("C1", (4,4), self.args.eval_seeds[0])
        self.manifest.update_run("C1", status="complete")
        self.resolve("G0", {"C1": [(4,4)], "teacher": [(4,4)]}, lambda s:
            gate_g0(self.evidence("C1", (4,4), s), self.evidence("teacher", (4,4), s), infrastructure_passed))
        for run_id in ("C2", "C3", "S1-L2", "S1-L3"):
            self.train(run_id)
            self.evaluate(run_id, (4,4), self.args.eval_seeds[0])
            self.manifest.update_run(run_id, status="complete")
        selected = self.resolve("S1*", {"S1-L2": [(4,4)], "S1-L3": [(4,4)]}, self.select_stage1)
        winner = selected["winner"]
        self.resolve("G1", {r: [(4,4)] for r in (winner, "C1", "C2")}, lambda s:
            gate_g1(self.evidence(winner, (4,4), s), self.evidence("C1", (4,4), s), self.evidence("C2", (4,4), s)))
        # GP evidence is recorded now; a failed premise may still permit Stage 2 per Section 8.
        gp = gate_gp(self.evidence("C3", (4,4), self.args.eval_seeds[:1]),
                     self.evidence("C2", (4,4), self.args.eval_seeds[:1]), *self.openloop_premise())
        self.decisions["GP"] = gp
        self.manifest.save()
        for run_id in ("S2-cont", "S2-base"):
            self.train(run_id)
            for pair in COUPLED_PAIRS:
                self.evaluate(run_id, pair, self.args.eval_seeds[0])
            self.manifest.update_run(run_id, status="complete")
        self.resolve("G2", {"S2-base": COUPLED_PAIRS, "S2-cont": COUPLED_PAIRS, "C2": [(4,4)]}, lambda s:
            gate_g2({p: self.evidence("S2-base", p, s) for p in COUPLED_PAIRS},
                    {p: self.evidence("S2-cont", p, s) for p in COUPLED_PAIRS}, self.evidence("C2", (4,4), s),
                    self.args.well_above_pp))
        self.decisions["S2*"] = dict(status="pass", winner="S2-base", reason="Only trained elastic candidate in the 14-run matrix")
        self.manifest.save()
        self.evaluate("S2-base", (3,3), self.args.eval_seeds[0])
        self.resolve("GP", {"C3": [(4,4)], "C2": [(4,4)]}, lambda s:
            gate_gp(self.evidence("C3", (4,4), s), self.evidence("C2", (4,4), s), *self.openloop_premise()))
        stage3 = ("S3-coupled", "S3-late", "S3-Konly", "S3-2stage")
        for run_id in stage3:
            self.train(run_id)
            for pair in self.matrix[run_id].pairs:
                self.evaluate(run_id, pair, self.args.eval_seeds[0])
            self.manifest.update_run(run_id, status="complete")
        selected = self.resolve("S3*", {r: self.matrix[r].pairs for r in stage3}, self.select_stage3)
        winner = selected["winner"]
        for pair in ALL_PAIRS:
            self.evaluate(winner, pair, self.args.eval_seeds[0])
        self.resolve("G3", {winner: FIVE_PAIRS, "C2": [(4,4)], "C3": [(4,4)]}, lambda s:
            gate_g3({p: self.evidence(winner, p, s) for p in FIVE_PAIRS}, self.evidence("C2", (4,4), s),
                    self.evidence("C3", (4,4), s), self.latency_values(winner), self.args.latency_match_tolerance))
        for run_id in ("F-Long-s1", "F-Long-s2"):
            self.train(run_id)
            for pair in ALL_PAIRS:
                for seed in self.args.eval_seeds:
                    self.evaluate(run_id, pair, seed)
            self.manifest.update_run(run_id, status="complete")
        self.manifest.data["status"] = "complete"
        self.manifest.data.pop("stop_kind", None)
        self.manifest.data.pop("stop_reason", None)
        self.manifest.data.pop("stopped_at", None)
        self.manifest.save()
        self.write_tables()

    def write_tables(self):
        rows = []
        for path in sorted(self.root.glob("*/eval/kv*_ka*/seed*/summary.json")):
            data = json.loads(path.read_text())
            if data.get("status") != "complete":
                continue
            run_id = path.parents[3].name
            latency = path.parent / "latency.json"
            profile = json.loads(latency.read_text()).get("compiled", {}) if latency.exists() else {}
            rows.append(dict(run=run_id, status=self.manifest.runs.get(run_id, {}).get("status", "evaluated"),
                pair=path.parents[1].name, seed=data["seed"],
                success_pct=data["success_pct"], episodes=data["episodes"],
                wilson_low=data["wilson95_pct"][0], wilson_high=data["wilson95_pct"][1],
                p50_ms=profile.get("p50_ms", ""), p90_ms=profile.get("p90_ms", ""),
                p99_ms=profile.get("p99_ms", ""), eval_wall_seconds=data.get("wall_seconds", "")))
        columns = ["run", "status", "pair", "seed", "success_pct", "episodes", "wilson_low", "wilson_high",
                   "p50_ms", "p90_ms", "p99_ms", "eval_wall_seconds"]
        measured_runs = {row["run"] for row in rows}
        for run in self.matrix.values():
            if run.id not in measured_runs:
                rows.append(dict({key: "" for key in columns}, run=run.id,
                    status=self.manifest.runs.get(run.id, {}).get("status", "pending")))
        with (self.root / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        lines = ["# LoopWAM initial 14-run campaign", "", "LIBERO-Long selection evidence only. "
                 "Teacher reproduction is Long only; full-suite reproduction, delay-injected evaluation, "
                 "component latency splits and RTX 4090 profiles are not part of this first pass.", "",
                 f"G2 'well above' means at least {self.args.well_above_pp:g} pp at both K=1 and K=2. "
                 f"Matched latency means within {100*self.args.latency_match_tolerance:g}% of measured C2 p50.", "",
                 "| Training run | Architecture | Absolute steps | Status | Micro batch × accumulation × GPUs |",
                 "|---|---|---|---|---|"]
        for run in self.matrix.values():
            batch = self.batches[run.arch]
            state = self.manifest.runs.get(run.id, {}).get("status", "pending")
            lines.append(f"| {run.id} | {run.arch} | {run.start}→{run.end} | {state} | "
                         f"{batch['micro_batch']} × {batch['grad_accum']} ×4 |")
        lines += ["", "## Evaluation results", "",
                 "| Run | Budget | Eval seed | Success | Wilson 95% | Episodes | p50 / p90 / p99 ms |",
                 "|---|---|---:|---:|---|---:|---|"]
        for row in rows:
            if row["success_pct"] == "":
                continue
            latency = " / ".join(f"{row[k]:.2f}" if isinstance(row[k], (int,float)) else "—"
                                  for k in ("p50_ms", "p90_ms", "p99_ms"))
            lines.append(f"| {row['run']} | {row['pair']} | {row['seed']} | {row['success_pct']:.1f}% | "
                         f"{row['wilson_low']:.1f}–{row['wilson_high']:.1f}% | {row['episodes']} | {latency} |")
        lines += ["", "## Decisions", ""]
        for gate, result in self.decisions.items():
            lines.append(f"- {gate}: **{result['status']}**" + (f"; {result['winner']}" if "winner" in result else ""))
        estimates = self.runtime_estimates()
        lines += ["", "## Runtime evidence", "", "Estimates use matching measured architecture and mode only; "
                  "missing measurements remain unknown. Compilation and startup may add overhead.", "",
                  "```json", json.dumps(estimates, indent=2), "```", ""]
        (self.root / "results.md").write_text("\n".join(lines))
        atomic_json(self.root / "runtime_estimate.json", estimates)

    def runtime_estimates(self):
        measured = {}
        for run in self.matrix.values():
            path = self.root / run.id / "timing.json"
            if path.exists():
                timing = json.loads(path.read_text())
                value = timing.get("seconds_per_step")
                if isinstance(value, (int, float)) and value > 0 and math.isfinite(value):
                    loss = self.manifest.runs.get(run.id, {}).get("loss", run.loss)
                    measured[(run.arch, run.mode, loss)] = dict(source=run.id, seconds_per_step=value)
        remaining = {}
        for run in self.matrix.values():
            if self.manifest.runs.get(run.id, {}).get("status") == "complete":
                continue
            loss = run.loss if run.loss != "recipe" else self.decisions.get("S1*", {}).get("loss", "unknown")
            source = measured.get((run.arch, run.mode, loss))
            start = run.start
            state = self.root / run.id / "state/latest.json"
            if state.exists():
                start = max(start, json.loads(state.read_text()).get("global_step", start))
            steps = max(0, run.end - start)
            remaining[run.id] = dict(optimizer_steps=steps, estimated_training_seconds=steps * source["seconds_per_step"]
                if source else None, measured_source=source["source"] if source else None)
        eval_times = []
        for path in self.root.glob("*/eval/*/seed*/summary.json"):
            data = json.loads(path.read_text())
            if data.get("status") == "complete" and data.get("wall_seconds", 0) > 0:
                eval_times.append(data["wall_seconds"])
        return dict(remaining_training=remaining,
                    measured_eval_500_episode_seconds=eval_times,
                    eval_500_episode_seconds_mean=sum(eval_times)/len(eval_times) if eval_times else None,
                    note="No estimate for an unmeasured architecture/mode; gates may stop later work.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/loopwam_v1/campaign")
    parser.add_argument("--teacher", default="checkpoints/fastwam_release/libero_uncond_2cam224.pt")
    parser.add_argument("--stats", default="checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json")
    parser.add_argument("--text-cache", default="data/text_embeds_cache/libero")
    parser.add_argument("--converted-dir", default="checkpoints/loopwam_v1")
    parser.add_argument("--infrastructure", help="P0-T JSON evidence: status=pass, all_14_tests_passed=true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seeds", type=int, nargs=2, default=[42,43])
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--micro-batch-map", help='JSON object or arch=N list; e.g. loopwam=16,untied30=8')
    parser.add_argument("--grad-accum", type=int, default=None, help="Default auto-computed for global batch128")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--deadline", type=float, help="Allocation end as epoch seconds; default Slurm EndTime")
    parser.add_argument("--checkpoint-reserve-seconds", type=float, default=180)
    parser.add_argument("--zero-stage", type=int, choices=[1,2], default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--well-above-pp", type=float, default=3)
    parser.add_argument("--latency-match-tolerance", type=float, default=.05)
    parser.add_argument("--plan", action="store_true", help="Print the exact 14-run matrix, without running anything")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps([dict(asdict(r), training_steps=r.steps) for r in run_matrix()], indent=2))
        return
    if len(set(args.gpus.split(","))) != 4:
        parser.error("The registered training protocol requires 4 GPUs and global batch 128")
    try:
        resolve_batches(args.micro_batch_map, args.micro_batch, args.grad_accum)
        if args.checkpoint_reserve_seconds < 180:
            raise ValueError("Checkpoint reserve must be at least180 seconds")
        args.deadline = resolve_deadline(args.deadline)
    except (ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    if len(set(args.eval_seeds)) != 2:
        parser.error("Two distinct paired evaluation seeds are required")
    if not args.report_only and not args.infrastructure:
        parser.error("--infrastructure is required before any success-rate evaluation")
    campaign = Campaign(args)
    import fcntl
    with allocation_signals(), (campaign.root / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            campaign.write_tables()
            if not args.report_only:
                campaign.run()
        except Exception as exc:
            campaign.manifest.data.update(status="stopped", stop_kind=stop_kind(exc), stop_reason=str(exc),
                                          stopped_at=time.time())
            campaign.manifest.save()
            campaign.write_tables()
            raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
