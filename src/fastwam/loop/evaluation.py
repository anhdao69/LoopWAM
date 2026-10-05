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
PROFILE_VERSION = 2
COMPONENT_WARMUP = 50
COMPONENT_CALLS = 100
PROTOCOL = dict(suite="libero_10", tasks=10, initial_states=50, initial_state_indices=list(range(50)),
                action_horizon=32, euler_steps=10, sigma_shift=5.0, replan_steps=10,
                max_steps=700, num_steps_wait=30, compiled=True, text_cfg_scale=1.0,
                binarize_gripper=True, weights="stage_end_ema")
DELAY_PROTOCOL = dict(version=1, controller="serial_receding_horizon_zero_order_command_hold",
    timing_scope="observation_preprocess+compiled_infer_action+action_postprocess",
    control_period_source="env.env.control_timestep", quantization="ceil_to_next_control_tick",
    hold="exact_last_processed_environment_command; initial_LIBERO_dummy_command",
    warmup_calls=50, warmup_rng="restore_python_numpy_torch_cpu_and_initialized_cuda",
    action_prefix="first_10_unchanged_after_delay", delay_counts_toward_max_steps=True,
    execution="discrete_event_emulation; not_buffered_or_asynchronous_hardware")


def delay_ticks(seconds, control_period):
    if (not math.isfinite(seconds) or seconds < 0 or not math.isfinite(control_period)
            or control_period <= 0):
        raise ValueError("Policy latency must be finite/nonnegative and control period positive")
    return math.ceil(seconds / control_period)


@contextmanager
def preserve_rng_state():
    """Warmup may initialize graphs, but must not advance paired policy RNGs."""
    import random
    import numpy as np
    import torch
    python_state, numpy_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def run_delayed_episode(env, initial_state, predict, *, max_steps=700, settling_steps=30,
                        replan_steps=10, image=None, warmup=None):
    """Serial receding-horizon control, holding commands while inference runs.

    predict consumes the frozen request observation and returns already processed
    environment commands plus measured wall seconds. Advancing the simulator
    afterward with the known held command emulates its non-pausing trajectory;
    the newly predicted actions are unavailable until the quantized return tick.
    This is not a buffered/asynchronous hardware controller.
    """
    period = float(env.env.control_timestep)
    delay_ticks(0., period)
    env.reset()
    obs = env.set_init_state(initial_state)
    held = [0, 0, 0, 0, 0, 0, -1]
    # Preserve upstream's initial settling convention, outside the 700-step cap.
    for _ in range(settling_steps):
        obs, _, _, _ = env.step(held.copy())
    if warmup is not None:
        with preserve_rng_state():
            warmup(obs)
    result = dict(success=False, control_period_seconds=period, control_steps=0,
                  delay_steps=0, policy_steps=0, replans=[], frames=[])

    def advance(command, kind):
        nonlocal obs
        if image is not None:
            result['frames'].append(image(obs))
        obs, _, done, _ = env.step(command.copy())
        result['control_steps'] += 1
        result[kind+'_steps'] += 1
        result['success'] = bool(done)

    while result['control_steps'] < max_steps and not result['success']:
        requested = result['control_steps']
        chunk, seconds = predict(obs)
        ticks = delay_ticks(seconds, period)
        commands = [[float(value) for value in row] for row in chunk[:replan_steps]]
        if len(commands) != replan_steps or any(len(row) != 7 or any(not math.isfinite(x) for x in row) for row in commands):
            raise ValueError("Expected ten finite, fully processed seven-dimensional commands")
        event = dict(request_step=requested, availability_step=requested+ticks,
                     policy_wall_ms=1000*seconds, delay_steps_scheduled=ticks,
                     quantization_overhead_ms=1000*(ticks*period-seconds),
                     held_command=held.copy(), delay_steps_executed=0, policy_steps_executed=0)
        result['replans'].append(event)
        for _ in range(min(ticks, max_steps-result['control_steps'])):
            advance(held, 'delay')
            event['delay_steps_executed'] += 1
            if result['success']:
                break
        if result['success'] or result['control_steps'] == max_steps:
            break
        for command in commands[:max_steps-result['control_steps']]:
            held = command
            advance(held, 'policy')
            event['policy_steps_executed'] += 1
            if result['success']:
                break
    return result


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
    architecture=getattr(args, 'profile_architecture', None) or json.dumps(dict(checkpoint=file_identity(args.checkpoint)))
    try:
        parsed=json.loads(architecture)
        if isinstance(parsed,dict):
            parsed.pop('pair',None)  # budget has its own field below
            architecture=parsed
    except (TypeError,json.JSONDecodeError):
        pass
    return dict(profile_version=PROFILE_VERSION, architecture=architecture, budget=[args.kv,args.ka],
                device_driver=device, torch_version=version("torch"),
                source_sha256=digest.hexdigest(), stats_sha256=sha256_file(args.stats), protocol=PROTOCOL)


def reusable_profile(profile, key):
    if (profile.get("cache_key") != key or profile.get("status") != "complete"
            or profile.get("warmup") != 50 or profile.get("calls") != 500 or not complete_components(profile)):
        return False
    return all(isinstance(profile.get(mode, {}).get(metric), (int, float))
               and math.isfinite(profile[mode][metric]) and profile[mode][metric] > 0
               for mode in ("eager", "compiled") for metric in ("p50_ms", "p90_ms", "p99_ms"))


def percentile_summary(values):
    ordered = sorted(values)
    if not ordered or any(not math.isfinite(x) or x < 0 for x in ordered):
        raise ValueError("Latency samples must be finite nonnegative values")
    def percentile(fraction):
        position = (len(ordered)-1)*fraction
        lower = int(position)
        upper = min(lower+1,len(ordered)-1)
        return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)
    return dict(samples=len(ordered),p50_ms=percentile(.5),p90_ms=percentile(.9),p99_ms=percentile(.99))


def aggregate_components(samples):
    """Aggregate per-call intervals; action sums and decode spans are different."""
    required = {'vae_encode_gpu_ms','video_prefill_gpu_ms','action_steps_gpu_ms',
                'action_decode_10_gpu_ms','total_gpu_timeline_ms','total_wall_ms'}
    metrics = {key: [] for key in ('vae_encode_gpu','video_prefill_gpu','action_step_gpu',
               'action_denoise_10_gpu','action_decode_10_gpu','total_gpu_timeline','total_wall',
               'unattributed_gpu_timeline','wall_minus_gpu_timeline','unattributed_wall')}
    per_step = [[] for _ in range(10)]
    if not samples:
        raise ValueError('No component observations')
    for row in samples:
        if set(row) != required or len(row['action_steps_gpu_ms']) != 10:
            raise ValueError('Missing component boundary or expected ten action denoises')
        steps = row['action_steps_gpu_ms']
        values = steps + [v for k,v in row.items() if k!='action_steps_gpu_ms']
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Nonfinite or negative component latency')
        denoise_sum = sum(steps)
        attributed = row['vae_encode_gpu_ms'] + row['video_prefill_gpu_ms'] + row['action_decode_10_gpu_ms']
        # Allow only sub-resolution event arithmetic; all ranges share one stream.
        if (row['action_decode_10_gpu_ms']+.05 < denoise_sum
                or row['total_gpu_timeline_ms']+.05 < attributed
                or row['total_wall_ms']+.05 < row['total_gpu_timeline_ms']):
            raise ValueError('Overlapping or inconsistent component intervals')
        for key in ('vae_encode_gpu','video_prefill_gpu','action_decode_10_gpu','total_gpu_timeline','total_wall'):
            metrics[key].append(row[key+'_ms'])
        metrics['action_step_gpu'].extend(steps)
        metrics['action_denoise_10_gpu'].append(denoise_sum)
        metrics['unattributed_gpu_timeline'].append(max(0.,row['total_gpu_timeline_ms']-attributed))
        metrics['wall_minus_gpu_timeline'].append(max(0.,row['total_wall_ms']-row['total_gpu_timeline_ms']))
        metrics['unattributed_wall'].append(max(0.,row['total_wall_ms']-attributed))
        for index,value in enumerate(steps):
            per_step[index].append(value)
    return dict(status='complete',calls=len(samples),denoise_steps=10,
                metrics={key:percentile_summary(values) for key,values in metrics.items()},
                action_steps_by_index=[percentile_summary(values) for values in per_step])


def complete_components(profile):
    if profile.get('profile_version') != PROFILE_VERSION or not isinstance(profile.get('components'),dict):
        return False
    required = ('vae_encode_gpu','video_prefill_gpu','action_step_gpu','action_denoise_10_gpu',
                'action_decode_10_gpu','total_gpu_timeline','total_wall','unattributed_gpu_timeline',
                'wall_minus_gpu_timeline','unattributed_wall')
    for mode in ('eager','compiled'):
        value = profile['components'].get(mode,{})
        if (value.get('status')!='complete' or value.get('calls')!=COMPONENT_CALLS
                or value.get('warmup')!=COMPONENT_WARMUP or value.get('denoise_steps')!=10):
            return False
        for name in required:
            stats = value.get('metrics',{}).get(name,{})
            count = COMPONENT_CALLS*10 if name=='action_step_gpu' else COMPONENT_CALLS
            if stats.get('samples')!=count or any(not isinstance(stats.get(k),(int,float))
                or not math.isfinite(stats[k]) or stats[k]<0 for k in ('p50_ms','p90_ms','p99_ms')):
                return False
    return True


def measure_components(model, call, mode):
    """Wrap Python boundaries outside compiled graphs, then restore each method.

    CUDA events describe elapsed stream intervals, including CPU enqueue gaps;
    they are not sums of kernel execution times. The decode span includes all
    ten denoises, Euler updates and intervening timestep preparation.
    """
    import torch
    saved = []
    events = {}
    active = False
    scheduler_steps = 0
    decode_end = None

    def replace(owner,name,function):
        saved.append((owner,name,name in vars(owner),getattr(owner,name)))
        setattr(owner,name,function)

    def wrap(owner,name,label):
        original=getattr(owner,name)
        def timed(*args,**kwargs):
            if not active:
                return original(*args,**kwargs)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            result=original(*args,**kwargs)
            end.record()
            events.setdefault(label,[]).append((begin,end))
            return result
        replace(owner,name,timed)

    scheduler=model.infer_action_scheduler
    original_step=scheduler.step
    def timed_step(*args,**kwargs):
        nonlocal scheduler_steps,decode_end
        result=original_step(*args,**kwargs)
        if active:
            scheduler_steps+=1
            if scheduler_steps==10:
                decode_end=torch.cuda.Event(enable_timing=True)
                decode_end.record()
        return result

    try:
        wrap(model,'_encode_input_image_latents_tensor','vae')
        if mode=='compiled':
            # Primary profiling has already created and warmed these graphs.
            wrap(model,'_prefill_video_cache_compiled','prefill')
            wrap(model,'_denoise_action_with_video_cache_compiled','action')
        else:
            wrap(model.mot,'prefill_video_cache_tensor','prefill')
            wrap(model,'_denoise_action_with_video_cache','action')
        replace(scheduler,'step',timed_step)
        for _ in range(COMPONENT_WARMUP):
            call()
        torch.cuda.synchronize()
        active=True
        samples=[]
        for _ in range(COMPONENT_CALLS):
            events={}; scheduler_steps=0; decode_end=None
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            wall_start=time.perf_counter()
            begin.record()
            call()
            end.record()
            torch.cuda.synchronize()  # one explicit synchronization per full call
            wall_ms=1000*(time.perf_counter()-wall_start)
            if ({key:len(value) for key,value in events.items()}!={'vae':1,'prefill':1,'action':10}
                    or scheduler_steps!=10 or decode_end is None):
                raise ValueError('Unexpected inference component boundaries; refusing partial component profile')
            elapsed=lambda pair: pair[0].elapsed_time(pair[1])
            samples.append(dict(vae_encode_gpu_ms=elapsed(events['vae'][0]),
                video_prefill_gpu_ms=elapsed(events['prefill'][0]),
                action_steps_gpu_ms=[elapsed(pair) for pair in events['action']],
                action_decode_10_gpu_ms=events['action'][0][0].elapsed_time(decode_end),
                total_gpu_timeline_ms=begin.elapsed_time(end),total_wall_ms=wall_ms))
        result=aggregate_components(samples)
        result.update(warmup=COMPONENT_WARMUP,method='CUDA events on the current stream, isolated after primary wall profiling',
            action_decode_scope='First denoise start through tenth Euler update end',
            unattributed_scope='Context/proprio/video preparation, cache clones, transfers and CPU enqueue gaps; not pure CPU time',
            note='Component instrumentation overhead is excluded from the primary500-call wall profile')
        return result
    finally:
        for owner,name,owned,original in reversed(saved):
            if owned:
                setattr(owner,name,original)
            else:
                delattr(owner,name)


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


def validate_task(record, seed, delay=False):
    if record.get("status") != "complete" or record.get("seed") != seed:
        raise ValueError("Incomplete, failed, or wrong-seed task evaluation")
    if record.get("total_episodes") != 50:
        raise ValueError("Each task requires exactly 50 episodes")
    if record.get('evaluation_kind', 'standard') != ('delay' if delay else 'standard'):
        raise ValueError('Delayed and primary task outcomes cannot be mixed')
    success, failure = record["success_episodes"], record["failure_episodes"]
    if (len(success) + len(failure) != 50 or set(success) & set(failure)
            or sorted(success + failure) != list(range(50))):
        raise ValueError("Missing or duplicate episode outcomes")
    return set(success)


def summarize_tasks(tasks, seed, delay=False):
    if len(tasks) != 10 or sorted(r.get("task_id", -1) for r in tasks) != list(range(10)):
        raise ValueError("Evaluation incomplete: exactly tasks 0..9 are required")
    outcomes = []
    for record in sorted(tasks, key=lambda x: x["task_id"]):
        success = validate_task(record, seed, delay=delay)
        details = {}
        if delay:
            if record.get('delay_protocol') != DELAY_PROTOCOL:
                raise ValueError('Delayed task protocol differs from the registered controller')
            rows = record.get('episode_results', [])
            if len(rows) != 50 or sorted(row.get('episode_id', -1) for row in rows) != list(range(50)):
                raise ValueError('Missing delayed episode timing records')
            details = {row['episode_id']: row for row in rows}
            for index,row in details.items():
                counts=[row.get(key) for key in ('control_steps','delay_steps','policy_steps')]
                if (any(type(value) is not int or value<0 for value in counts)
                        or counts[0]>PROTOCOL['max_steps'] or counts[0]!=sum(counts[1:])
                        or row.get('success') is not (index in success)):
                    raise ValueError('Invalid delayed control-step or outcome accounting')
                period=row['control_period_seconds']
                delay_ticks(0.,period)
                used=0
                for event in row['replans']:
                    ticks=delay_ticks(event['policy_wall_ms']/1000,period)
                    if (event['request_step']!=used or event['availability_step']!=used+ticks
                            or event['delay_steps_scheduled']!=ticks
                            or not 0<=event['delay_steps_executed']<=ticks
                            or not 0<=event['policy_steps_executed']<=PROTOCOL['replan_steps']):
                        raise ValueError('Invalid delayed control event chronology')
                    used+=event['delay_steps_executed']+event['policy_steps_executed']
                if used!=counts[0]:
                    raise ValueError('Missing delayed control events')
        outcomes.extend(dict(details.get(i, {}), seed=seed, task_id=record["task_id"], episode_id=i,
                             success=i in success) for i in range(50))
    n_success = sum(r["success"] for r in outcomes)
    summary = dict(status="complete", seed=seed, episodes=500, successes=n_success,
                success_pct=n_success / 5, wilson95_pct=wilson(n_success, 500),
                outcomes=outcomes, task_seconds=sum(t.get("duration_seconds", 0) for t in tasks))
    if delay:
        periods = {row['control_period_seconds'] for row in outcomes}
        if len(periods) != 1:
            raise ValueError('Delayed evaluation used inconsistent control periods')
        timings = [event['policy_wall_ms'] for row in outcomes for event in row['replans']]
        summary['delay'] = dict(protocol=DELAY_PROTOCOL, control_period_seconds=periods.pop(),
            decision_wall=percentile_summary(timings),
            control_steps=sum(row['control_steps'] for row in outcomes),
            delay_steps=sum(row['delay_steps'] for row in outcomes),
            policy_steps=sum(row['policy_steps'] for row in outcomes))
    return summary


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
        result = dict(status="wall_complete_components_pending", profile_version=PROFILE_VERSION,
                      device=torch.cuda.get_device_name(), batch=1,
                      warmup=50, calls=500, initial_state_sha256=state_hash,
                      scope="infer_action total: VAE + video prefill + 10 action steps",
                      components={}, torch_version=torch.__version__,
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
            # Finish BOTH primary wall profiles before any component wrapper is installed.
            atomic_json(Path(args.output) / "latency.json", result)
            try:
                for mode in ('eager','compiled'):
                    def component_call():
                        return model.infer_action(**kwargs, compile_action_infer=(mode=='compiled'))
                    result['components'][mode]=measure_components(model,component_call,mode)
                result['status']='complete'
            except Exception as exc:
                result.update(status='wall_complete_components_error',component_error=str(exc))
                atomic_json(Path(args.output) / 'latency.json',result)
                raise
        atomic_json(Path(args.output) / "latency.json", result)
    finally:
        env.close()


def run_delayed_task(upstream, task, states, model, processor, cfg, videos, warmed):
    """Reuse upstream preprocessing/action conversion; only the scheduler differs."""
    import torch
    env, description = upstream.get_libero_env(task, upstream.LIBERO_ENV_RESOLUTION, cfg.seed)
    result = dict(successes=0, success_episodes=[], failure_episodes=[], task_description=description,
                  evaluation_kind='delay', delay_protocol=DELAY_PROTOCOL, episode_results=[])
    try:
        # Text loading is initialization, just like graph compilation.
        with preserve_rng_state():
            model.encode_prompt(upstream.DEFAULT_PROMPT.format(task=description))
        def predict(obs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            chunk, _, predicted = upstream._predict_action_chunk(obs, description, model, processor, cfg,
                action_horizon=32, input_w=448, input_h=224, model_device='cuda')
            torch.cuda.synchronize()
            if predicted is not None:
                raise ValueError('Delay evaluation supports action-only inference')
            return chunk, time.perf_counter()-started
        def warmup(obs):
            for _ in range(DELAY_PROTOCOL['warmup_calls']):
                predict(obs)
            warmed['done'] = True
        for index, state in enumerate(states):
            episode = run_delayed_episode(env, state, predict, image=upstream.get_libero_image,
                warmup=None if warmed.get('done') else warmup)
            frames = episode.pop('frames')
            episode['episode_id'] = index
            success = episode['success']
            result['successes'] += int(success)
            result['success_episodes' if success else 'failure_episodes'].append(index)
            result['episode_results'].append(episode)
            upstream.save_rollout_video(videos, frames, f'task{cfg.EVALUATION.task_id}_trial{index}',
                                        success=success, task_description=description)
        return result
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
    warmed = {}
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
            if getattr(args, 'delay_injected', False):
                result = run_delayed_task(upstream, task, states, model, processor, cfg, videos, warmed)
            else:
                result = upstream.run_single_task(task, states, model, processor, cfg,
                    videos, output / "predicted_videos", action_horizon=32,
                    input_w=448, input_h=224, model_device="cuda")
            result.update(status="complete", task_id=task_id, seed=args.seed,
                          total_episodes=50, initial_state_sha256=state_hash,
                          duration_seconds=time.monotonic() - started)
            validate_task(result, args.seed, delay=getattr(args, 'delay_injected', False))
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
    protocol['weights'] = getattr(args, 'weights', 'stage_end_ema')
    delayed = getattr(args, 'delay_injected', False)
    if delayed:
        protocol['delay'] = DELAY_PROTOCOL
    metadata = output / "protocol.json"
    if metadata.exists() and json.loads(metadata.read_text()) != protocol:
        raise ValueError("Existing evaluation protocol/checkpoint differs; choose a new output directory")
    atomic_json(metadata, protocol)
    # An advisory file lock is released by the OS even if the manager is killed.
    import fcntl
    with (output / "manager.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        had_summary = (output / "summary.json").exists()
        latency_path = output / 'latency.json'
        cache_key = profile_cache_key(args) if args.profile else None
        profile_ready = bool(args.profile and latency_path.exists()
                             and reusable_profile(json.loads(latency_path.read_text()), cache_key))
        if had_summary:
            tasks = [json.loads((output / f"task_{i}.json").read_text()) for i in range(10)]
            summarize_tasks(tasks, args.seed, delay=delayed)
            if not args.profile or profile_ready:
                return
        cmd = [sys.executable, str(ROOT / "scripts/loopwam/evaluate.py"),
               "--checkpoint", str(args.checkpoint), "--stats", str(args.stats),
               "--output", str(output), "--seed", str(args.seed),
               "--kv", str(args.kv), "--ka", str(args.ka), "--text-cache", str(args.text_cache)]
        cmd += ['--weights', protocol['weights']]
        if delayed:
            cmd.append('--delay-injected')
        if args.teacher:
            cmd.append("--teacher")
        gpu_ids = args.gpus.split(",")
        base_env = os.environ.copy()
        base_env.setdefault("OMP_NUM_THREADS", "1")
        started = time.monotonic()
        deadline = args.deadline - args.deadline_reserve_seconds if args.deadline is not None else None
        cache_path = Path(args.profile_cache) if getattr(args, "profile_cache", None) else None
        if args.profile and cache_path is not None and not profile_ready:
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
                if reusable_profile(cached, cache_key):
                    cached.update(reused_from=str(cache_path.resolve()),
                                  reused_for_checkpoint=file_identity(args.checkpoint))
                    atomic_json(output / "latency.json", cached)
                    profile_ready = True
        if args.profile and not profile_ready:
            if latency_path.exists():
                old = json.loads(latency_path.read_text())
                atomic_json(output / f"latency.previous_v{old.get('profile_version',1)}.json",old)
            env = dict(base_env, CUDA_VISIBLE_DEVICES=gpu_ids[0])
            with (output / "profile.log").open("a") as log:
                run_process_group(cmd + ["--profile-only", "--worker", "profile"], cwd=ROOT,
                    env=env, stdout=log, stderr=subprocess.STDOUT, deadline=deadline, on_tick=tick)
            measured = json.loads((output / "latency.json").read_text())
            measured["cache_key"] = cache_key
            atomic_json(output / "latency.json", measured)
            if cache_path is not None:
                atomic_json(cache_path, measured)
        if had_summary or getattr(args,'profile_only',False):
            return
        pending = []
        for i in range(10):
            path = output / f"task_{i}.json"
            if path.exists():
                validate_task(json.loads(path.read_text()), args.seed, delay=delayed)
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
        summary = summarize_tasks(tasks, args.seed, delay=delayed)
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
                 job_id=os.environ.get("SLURM_JOB_ID"), wall_seconds=0, completed_tasks=0,
                 operation='profile' if getattr(args,'profile_only',False) else 'evaluation')
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


def profile_grid(args):
    """Phase0 all-ten-budget latency sweep; never creates rollout workers."""
    from copy import copy
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    record=dict(status='running',profile_version=PROFILE_VERSION,profiles=[],started_at=time.time())
    try:
        for kv in range(1,5):
            for ka in range(1,kv+1):
                selected=copy(args)
                selected.kv,selected.ka=kv,ka
                selected.profile=selected.profile_only=True
                selected.profile_all_budgets=False
                selected.output=str(root/f'kv{kv}_ka{ka}')
                if args.profile_cache:
                    cache=Path(args.profile_cache)
                    selected.profile_cache=str(cache.with_name(f'{cache.stem}_kv{kv}_ka{ka}{cache.suffix}'))
                evaluate(selected)
                path=Path(selected.output)/'latency.json'
                profile=json.loads(path.read_text())
                if profile.get('status')!='complete' or not complete_components(profile):
                    raise ValueError(f'Incomplete profile for budget{kv},{ka}')
                record['profiles'].append(dict(kv=kv,ka=ka,path=str(path),
                    eager=profile['eager'],compiled=profile['compiled']))
                atomic_json(root/'profile_grid.json',record)
        record['status']='complete'
    except Exception as exc:
        record.update(status='interrupted' if isinstance(exc,TimeoutError) else 'error',error=str(exc))
        raise
    finally:
        record['wall_seconds']=time.time()-record['started_at']
        atomic_json(root/'profile_grid.json',record)


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
    parser.add_argument('--weights', choices=('stage_end_ema', 'stage_end_raw'), default='stage_end_ema')
    parser.add_argument('--delay-injected', action='store_true', help='Serial receding-horizon command-hold latency emulation')
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-only", action="store_true", help="Measure latency without any rollouts")
    parser.add_argument("--profile-all-budgets", action="store_true", help="With --profile-only, sweep all ten LoopWAM budgets")
    parser.add_argument("--profile-cache", help="Reuse one measured architecture/budget profile with provenance")
    parser.add_argument("--profile-architecture", help="Architecture plus initialization identity; required with cache")
    parser.add_argument("--worker")
    parser.add_argument("--deadline", type=float, help="Allocation end epoch; manager stops before its reserve")
    parser.add_argument("--deadline-reserve-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    if args.ka > args.kv:
        parser.error("ka must be <= kv")
    if args.delay_injected and (args.profile or args.profile_only or args.profile_all_budgets):
        parser.error('Delay evaluation is separate from the primary latency profile')
    if args.deadline_reserve_seconds < 0:
        parser.error("Deadline reserve must be nonnegative")
    if args.profile_cache and not args.profile_architecture:
        if args.profile_all_budgets:
            args.profile_architecture=json.dumps(dict(arch='loopwam',initialization=file_identity(args.checkpoint)),sort_keys=True)
        else:
            parser.error("--profile-cache requires --profile-architecture")
    if args.profile_all_budgets and (not args.profile_only or args.teacher or args.worker):
        parser.error("--profile-all-budgets requires standalone --profile-only on a LoopWAM checkpoint")
    if args.profile_only:
        args.profile=True
    if args.worker is not None:
        worker(args)
    else:
        try:
            if args.profile_all_budgets:
                profile_grid(args)
            else:
                evaluate(args)
        except TimeoutError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(3) from exc


if __name__ == "__main__":
    main()
