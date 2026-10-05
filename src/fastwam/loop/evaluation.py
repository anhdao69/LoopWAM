"""Fixed LIBERO-Long protocol and persistent GPU evaluation workers.

The pure summary functions deliberately import neither torch nor LIBERO. A task
error has no success rate; all ten complete tasks are required for a summary.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = dict(suite="libero_10", tasks=10, initial_states=50, initial_state_indices=list(range(50)),
                action_horizon=32, euler_steps=10, sigma_shift=5.0, replan_steps=10,
                max_steps=700, num_steps_wait=30, compiled=True, text_cfg_scale=1.0,
                binarize_gripper=True, weights="stage_end_ema")


def check_deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Allocation deadline reached; completed task files are preserved")


def stop_process_groups(processes, grace_seconds=10):
    """Terminate every descendant in sessions created with start_new_session."""
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    until = time.monotonic() + grace_seconds
    while time.monotonic() < until and any(p.poll() is None for p in processes):
        time.sleep(.05)
    # Kill groups even when their leader exited: a compiler/simulator child
    # may have outlived the Python process that started it.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_process_group(command, *, deadline=None, on_tick=None, **kwargs):
    check_deadline(deadline)
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        while process.poll() is None:
            check_deadline(deadline)
            if on_tick is not None:
                on_tick()
            time.sleep(.1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        return process.returncode
    finally:
        stop_process_groups([process])


def profile_cache_key(args):
    """Weights share an architecture profile; runtime/code changes invalidate it."""
    from importlib.metadata import version
    device = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader",
                             "--id=" + args.gpus.split(",")[0]],
                            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/fastwam").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return dict(architecture=args.profile_architecture, device_driver=device, torch_version=version("torch"),
                source_sha256=digest.hexdigest(), stats_sha256=sha256_file(args.stats), protocol=PROTOCOL)


def reusable_profile(profile, key):
    if (profile.get("cache_key") != key or profile.get("status") != "complete"
            or profile.get("warmup") != 50 or profile.get("calls") != 500):
        return False
    return all(isinstance(profile.get(mode, {}).get(metric), (int, float))
               and math.isfinite(profile[mode][metric]) and profile[mode][metric] > 0
               for mode in ("eager", "compiled") for metric in ("p50_ms", "p90_ms", "p99_ms"))


@contextmanager
def allocation_signals():
    """Let manager finally blocks clean up worker sessions on SIGTERM/SIGINT."""
    def interrupted(signum, frame):
        # An external cancellation is not evidence of an allocation timeout.
        # In particular, cancelling a job must never authorize another one.
        raise InterruptedError(f"Process interrupted by signal {signum}")
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, n: int):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [100 * (center - half), 100 * (center + half)]


def validate_task(record, seed):
    if record.get("status") != "complete" or record.get("seed") != seed:
        raise ValueError("Incomplete, failed, or wrong-seed task evaluation")
    if record.get("total_episodes") != 50:
        raise ValueError("Each task requires exactly 50 episodes")
    success, failure = record["success_episodes"], record["failure_episodes"]
    if (len(success) + len(failure) != 50 or set(success) & set(failure)
            or sorted(success + failure) != list(range(50))):
        raise ValueError("Missing or duplicate episode outcomes")
    return set(success)


def summarize_tasks(tasks, seed):
    if len(tasks) != 10 or sorted(r.get("task_id", -1) for r in tasks) != list(range(10)):
        raise ValueError("Evaluation incomplete: exactly tasks 0..9 are required")
    outcomes = []
    for record in sorted(tasks, key=lambda x: x["task_id"]):
        success = validate_task(record, seed)
        outcomes.extend(dict(seed=seed, task_id=record["task_id"], episode_id=i,
                             success=i in success) for i in range(50))
    n_success = sum(r["success"] for r in outcomes)
    return dict(status="complete", seed=seed, episodes=500, successes=n_success,
                success_pct=n_success / 5, wilson95_pct=wilson(n_success, 500),
                outcomes=outcomes, task_seconds=sum(t.get("duration_seconds", 0) for t in tasks))


def _configuration(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    # The upstream evaluator registers the train config resolvers on import.
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="sim_libero", overrides=["task=libero_uncond_2cam224_1e-4"])
    cfg.seed = args.seed
    cfg.ckpt = str(Path(args.checkpoint).resolve())
    cfg.model.load_text_encoder = False
    cfg.model.skip_dit_load_from_pretrain = True
    cfg.model.action_dit_pretrained_path = None
    for name in ("video_scheduler", "action_scheduler"):
        cfg.model[name].train_shift = cfg.model[name].infer_shift = 5.0
    cfg.EVALUATION.task_suite_name = "libero_10"
    cfg.EVALUATION.num_trials = 50
    cfg.EVALUATION.num_inference_steps = 10
    cfg.EVALUATION.sigma_shift = 5.0
    cfg.EVALUATION.compile_action_infer = True
    cfg.EVALUATION.output_dir = str(args.output)
    cfg.EVALUATION.dataset_stats_path = str(args.stats)
    return cfg


def _load_runtime(args):
    import torch
    from hydra.utils import instantiate
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    libero_scripts = str(ROOT / "experiments/libero")
    if libero_scripts not in sys.path:
        sys.path.insert(0, libero_scripts)
    from experiments.libero import eval_libero_single as upstream
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.utils.pytorch_utils import set_global_seed
    cfg = _configuration(args)
    set_global_seed(args.seed)
    if args.teacher:
        from fastwam.loop.model import load_teacher
        model = load_teacher(args.checkpoint, device="cuda")
    else:
        from fastwam.loop.model import load_model
        model = load_model(args.checkpoint, device="cuda", training=False)
        model.set_budget(args.kv, args.ka)
        model.eval()
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(args.stats))
    cache = {}

    def encode_cached(prompt):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        contexts, masks = [], []
        for text in prompts:
            if text not in cache:
                hashed = hashlib.sha256(text.encode("utf-8")).hexdigest()
                path = Path(args.text_cache) / f"{hashed}.t5_len128.wan22ti2v5b.pt"
                payload = torch.load(path, map_location="cpu", weights_only=False)
                context, mask = payload["context"].clone(), payload["mask"].bool()
                if context.shape != (128, 4096) or mask.shape != (128,):
                    raise ValueError(f"Invalid cached text context: {path}")
                context[~mask] = 0
                cache[text] = (context.to(device=model.device, dtype=model.torch_dtype),
                               torch.ones_like(mask, device=model.device))
            contexts.append(cache[text][0])
            masks.append(cache[text][1])
        return torch.stack(contexts), torch.stack(masks)

    model.encode_prompt = encode_cached
    return upstream, cfg, model, processor


def _initial_states(upstream, task):
    import torch
    path = Path(upstream.get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    states = torch.load(path, map_location="cpu", weights_only=False)
    if len(states) < 50:
        raise ValueError(f"Need 50 distinct provided initial states: {path}")
    return states[:50], sha256_file(path)


def profile(args, upstream, cfg, model, processor):
    """Measure whole policy calls on one real observation, without GPU contention.

    Compilation and 50 warmups are excluded; preprocessing/denormalization is
    excluded. VAE, video prefill and all ten action steps are included together.
    """
    import numpy as np
    import torch
    suite = upstream.benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)
    states, state_hash = _initial_states(upstream, task)
    env, description = upstream.get_libero_env(task, upstream.LIBERO_ENV_RESOLUTION, args.seed)
    try:
        env.reset()
        obs = env.set_init_state(states[0])
        for _ in range(30):
            obs, *_ = env.step(upstream.get_libero_dummy_action())
        image, proprio, _ = upstream._obs_to_model_input(
            obs, cfg, processor, width=448, height=224, device="cuda", dtype=model.torch_dtype)
        kwargs = dict(prompt=upstream.DEFAULT_PROMPT.format(task=description), input_image=image,
                      proprio=proprio, action_horizon=32, num_inference_steps=10,
                      sigma_shift=5.0, seed=args.seed, rand_device="cpu")
        result = dict(status="complete", device=torch.cuda.get_device_name(), batch=1,
                      warmup=50, calls=500, initial_state_sha256=state_hash,
                      scope="infer_action total: VAE + video prefill + 10 action steps",
                      components="not independently instrumented", torch_version=torch.__version__,
                      protocol=PROTOCOL, kv=args.kv, ka=args.ka, checkpoint=file_identity(args.checkpoint))
        with torch.inference_mode():
            for mode in ("eager", "compiled"):
                def call():
                    return model.infer_action(**kwargs, compile_action_infer=(mode == "compiled"))
                for _ in range(50):
                    call()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(500):
                    started = time.perf_counter()
                    call()
                    torch.cuda.synchronize()
                    times.append(1000 * (time.perf_counter() - started))
                result[mode] = dict(zip(("p50_ms", "p90_ms", "p99_ms"),
                                        map(float, np.percentile(times, [50, 90, 99]))))
                result[mode]["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        atomic_json(Path(args.output) / "latency.json", result)
    finally:
        env.close()


def worker(args):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.libero.worker_pool import pop_task, write_worker_status
    upstream, cfg, model, processor = _load_runtime(args)
    output = Path(args.output)
    if args.profile_only:
        profile(args, upstream, cfg, model, processor)
        return
    suite = upstream.benchmark.get_benchmark_dict()["libero_10"]()
    if suite.n_tasks != 10:
        raise ValueError("LIBERO-Long must have ten tasks")
    while True:
        item = pop_task(output / "pending.txt", output / "queue.lock", output / "workers", args.worker)
        if item is None:
            return
        _, task_id = item
        cfg.EVALUATION.task_id = task_id
        started = time.monotonic()
        try:
            task = suite.get_task(task_id)
            states, state_hash = _initial_states(upstream, task)
            videos = output / "videos"
            videos.mkdir(exist_ok=True)
            result = upstream.run_single_task(task, states, model, processor, cfg,
                videos, output / "predicted_videos", action_horizon=32,
                input_w=448, input_h=224, model_device="cuda")
            result.update(status="complete", task_id=task_id, seed=args.seed,
                          total_episodes=50, initial_state_sha256=state_hash,
                          duration_seconds=time.monotonic() - started)
            validate_task(result, args.seed)
            atomic_json(output / f"task_{task_id}.json", result)
            write_worker_status(output / "workers", args.worker, "complete", str(task_id))
        except Exception:
            atomic_json(output / f"error_task_{task_id}.json", dict(status="error", task_id=task_id,
                        seed=args.seed, traceback=traceback.format_exc()))
            raise


def _evaluate(args, tick):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL, seed=args.seed, kv=args.kv, ka=args.ka, teacher=args.teacher,
                    checkpoint=file_identity(args.checkpoint), stats_sha256=sha256_file(args.stats),
                    text_cache=str(Path(args.text_cache).resolve()))
    metadata = output / "protocol.json"
    if metadata.exists() and json.loads(metadata.read_text()) != protocol:
        raise ValueError("Existing evaluation protocol/checkpoint differs; choose a new output directory")
    atomic_json(metadata, protocol)
    # An advisory file lock is released by the OS even if the manager is killed.
    import fcntl
    with (output / "manager.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        had_summary = (output / "summary.json").exists()
        if had_summary:
            tasks = [json.loads((output / f"task_{i}.json").read_text()) for i in range(10)]
            summarize_tasks(tasks, args.seed)
            if not args.profile or (output / "latency.json").exists():
                return
        cmd = [sys.executable, str(ROOT / "scripts/loopwam/evaluate.py"),
               "--checkpoint", str(args.checkpoint), "--stats", str(args.stats),
               "--output", str(output), "--seed", str(args.seed),
               "--kv", str(args.kv), "--ka", str(args.ka), "--text-cache", str(args.text_cache)]
        if args.teacher:
            cmd.append("--teacher")
        gpu_ids = args.gpus.split(",")
        base_env = os.environ.copy()
        base_env.setdefault("OMP_NUM_THREADS", "1")
        started = time.monotonic()
        deadline = args.deadline - args.deadline_reserve_seconds if args.deadline is not None else None
        cache_path = Path(args.profile_cache) if getattr(args, "profile_cache", None) else None
        cache_key = None
        if args.profile and cache_path is not None and not (output / "latency.json").exists():
            cache_key = profile_cache_key(args)
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
                if reusable_profile(cached, cache_key):
                    cached.update(reused_from=str(cache_path.resolve()),
                                  reused_for_checkpoint=file_identity(args.checkpoint))
                    atomic_json(output / "latency.json", cached)
        if args.profile and not (output / "latency.json").exists():
            env = dict(base_env, CUDA_VISIBLE_DEVICES=gpu_ids[0])
            with (output / "profile.log").open("a") as log:
                run_process_group(cmd + ["--profile-only", "--worker", "profile"], cwd=ROOT,
                    env=env, stdout=log, stderr=subprocess.STDOUT, deadline=deadline, on_tick=tick)
            if cache_path is not None:
                measured = json.loads((output / "latency.json").read_text())
                measured["cache_key"] = cache_key
                atomic_json(output / "latency.json", measured)
                atomic_json(cache_path, measured)
        if had_summary:
            return
        pending = []
        for i in range(10):
            path = output / f"task_{i}.json"
            if path.exists():
                validate_task(json.loads(path.read_text()), args.seed)
            else:
                pending.append(i)
        (output / "pending.txt").write_text("".join(f"libero_10,{i}\n" for i in pending))
        # Recover the directory lock only when no other manager can own this queue.
        stale = output / "queue.lock.lockdir"
        if stale.exists():
            stale.rmdir()
        workers = []
        try:
            check_deadline(deadline)
            for index, gpu in enumerate(gpu_ids[:len(pending)]):
                log = (output / f"worker_{index}.log").open("a")
                proc = subprocess.Popen(cmd + ["--worker", str(index)], cwd=ROOT,
                        env=dict(base_env, CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                workers.append((proc, log))
            while any(proc.poll() is None for proc, _ in workers):
                check_deadline(deadline)
                tick()
                failures = [proc.returncode for proc, _ in workers if proc.poll() not in (None, 0)]
                if failures:
                    raise RuntimeError(f"Evaluation worker failed: {failures}; see {output}")
                time.sleep(1)
            if any(proc.returncode != 0 for proc, _ in workers):
                raise RuntimeError(f"Evaluation worker failed; see {output}")
        finally:
            stop_process_groups([proc for proc, _ in workers])
            for proc, log in workers:
                log.close()
        tasks = [json.loads((output / f"task_{i}.json").read_text()) for i in range(10)]
        summary = summarize_tasks(tasks, args.seed)
        past = output / "attempts.jsonl"
        past_seconds = sum(json.loads(line)["wall_seconds"] for line in past.read_text().splitlines()) if past.exists() else 0
        elapsed = time.monotonic() - started
        summary.update(protocol=protocol, wall_seconds=past_seconds + elapsed, invocation_wall_seconds=elapsed,
                       initial_state_sha256={str(t["task_id"]): t["initial_state_sha256"] for t in tasks})
        atomic_json(output / "summary.json", summary)
        (output / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in summary["outcomes"]))
        print(json.dumps({k: v for k, v in summary.items() if k != "outcomes"}, indent=2))


def evaluate(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    updated = 0
    state = dict(status="running", started_at=started, deadline=args.deadline,
                 job_id=os.environ.get("SLURM_JOB_ID"), wall_seconds=0, completed_tasks=0)
    def tick(force=False):
        nonlocal updated
        now = time.time()
        if force or now - updated >= 5:
            state.update(wall_seconds=now-started, updated_at=now,
                         completed_tasks=sum((output / f"task_{i}.json").exists() for i in range(10)))
            atomic_json(output / "runtime.json", state)
            updated = now
    tick(True)
    try:
        with allocation_signals():
            _evaluate(args, tick)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, TimeoutError) else "error", error=str(exc))
        raise
    finally:
        tick(True)
        with (output / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(state, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--ka", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--text-cache", default="data/text_embeds_cache/libero")
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-cache", help="Reuse one measured architecture/budget profile with provenance")
    parser.add_argument("--profile-architecture", help="Architecture plus initialization identity; required with cache")
    parser.add_argument("--worker")
    parser.add_argument("--deadline", type=float, help="Allocation end epoch; manager stops before its reserve")
    parser.add_argument("--deadline-reserve-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    if args.ka > args.kv:
        parser.error("ka must be <= kv")
    if args.deadline_reserve_seconds < 0:
        parser.error("Deadline reserve must be nonnegative")
    if args.profile_cache and not args.profile_architecture:
        parser.error("--profile-cache requires --profile-architecture")
    if args.worker is not None:
        worker(args)
    else:
        try:
            evaluate(args)
        except TimeoutError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(3) from exc


if __name__ == "__main__":
    main()
