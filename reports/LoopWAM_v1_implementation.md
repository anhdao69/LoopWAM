# LoopWAM v1: implementation and LIBERO-Long screening

Status (2026-10-05): implementation on branch `LoopWAM_v1`; production infrastructure validation is in progress. No completed screening result or selected recipe is claimed here. This report is updated as measured training and evaluation evidence becomes available.

## Scope and experimental contract

The implementation follows `plans/LoopWAM_v1.md` (v2 architecture) and the user's smaller initial route: P0-S; C1, C2, C3, S1-L2, S1-L3; S2-cont, S2-base; S3-coupled, S3-late, S3-Konly, S3-2stage; two end-to-end confirmation seeds. This is 14 training trajectories, with stage-dependent forks and gates. It excludes the initial video-KD, alternate-LR, r0, re-injection, deep-video-supervision and alternative-alignment ablations. LIBERO-10 is the training dataset and LIBERO-Long is the ten-task closed-loop evaluation suite.

The run budgets remain explicit optimizer-step endpoints. The 14 trajectories contain **142,000 incremental optimizer steps** at global batch 128 (18,176,000 sampled windows), excluding short infrastructure diagnostics. An interrupted trajectory resumes its optimizer state and does not count as another run. Gates can stop the route early; they do not automatically launch a fallback ablation.

| Phase | Run | Initialization or parent | Absolute optimizer steps | Incremental steps |
| --- | --- | --- | ---: | ---: |
| 0 | P0-S | Converted LoopWAM, L2 | 0–2,000 | 2,000 |
| 1 | C1 | Untied-30, L3 | 0–8,000 | 8,000 |
| 1 | C2 | Untied-12, L3 | 0–8,000 | 8,000 |
| 1 | C3 | Untied-V30/A12, L3 | 0–8,000 | 8,000 |
| 1 | S1-L2 | Converted LoopWAM, L2 | 0–8,000 | 8,000 |
| 1 | S1-L3 | Converted LoopWAM, L3 | 0–8,000 | 8,000 |
| 2 | S2-cont | Selected S1 checkpoint, fixed (4,4) | 8,000–14,000 | 6,000 |
| 2 | S2-base | Selected S1 checkpoint, coupled sampling | 8,000–14,000 | 6,000 |
| 3 | S3-coupled | S2-base, coupled sampling | 14,000–22,000 | 8,000 |
| 3 | S3-late | S2-base, decoupled sampling | 14,000–22,000 | 8,000 |
| 3 | S3-Konly | Selected S1 checkpoint, video budget fixed at 4 | 8,000–22,000 | 14,000 |
| 3 | S3-2stage | Selected S1 checkpoint, immediate decoupled sampling | 8,000–22,000 | 14,000 |
| 4 | F-Long-s1 | Converted initialization, selected schedule, seed 43 | 0–22,000 | 22,000 |
| 4 | F-Long-s2 | Converted initialization, selected schedule, seed 44 | 0–22,000 | 22,000 |

Screening training uses seed 42. Evaluation seeds are independently specified as 42 and 43. Confirmation runs follow the selected two-stage, action-only, coupled or three-stage schedule continuously from initialization. They do not inherit the screening optimizer state.

## Architecture and conversion

A separate `fastwam.loop` package preserves the original teacher execution as an independent numerical reference. The student uses the original Wan video/action preparation, positional encodings, velocity heads, schedulers and VAE. Video width is 2,048 with 16 × 128 attention heads and 8,192 FFN channels. Action width is 768, with 16 × 128 joint-attention heads, 6 × 128 text-attention heads and 3,072 FFN channels. The proprio encoder remains 8→4,096 because it creates a text-context token. The six action text-attention heads are selected from the teacher's **24** heads; this count is derived from the source tensors.

The looped model stores 3 prelude blocks, 6 core blocks and 3 coda blocks per stream: 12 unique blocks and `6 + 6K` effective layers. Core blocks have four explicit slots containing modulation deltas, copied norm parameters, bias deltas and rank-32 LoRA on every linear. A core linear uses `W + B_slot A_slot`, with no additional `alpha/r` multiplier, and `b + delta_b_slot`. Slots are function arguments rather than mutable module state, so activation-checkpoint recomputation cannot accidentally use the last slot executed during the forward pass. Each slot executes the original gated self-attention, cross-attention and gated FFN residuals once; the implementation adds no extra recurrent residual.

Video uses prefix loops 1…Kv. Action uses suffix slots Kv−Ka+1…Kv and reads the corresponding virtual video layers. First-frame keys/values retain autograd history in training. The full video pass supplies the final video FM loss and cached loop exits. A shorter video budget recomputes only its three coda blocks on first-frame tokens. This is numerically valid because the first frame never attends to future frames or actions. All ten budgets `1 ≤ Ka ≤ Kv ≤ 4` are supported.

Width conversion selects complete evenly spaced attention heads and FFN channels, then interpolates only action hidden axes with the original alpha-rescaling rule. Modulation/time-projection row groups are handled separately. Core weights are cycle-group means; biases/modulation/norms are restored by slot. Full-rank residual SVD is exact in numerical tests. Production rank 32 uses deterministic randomized SVD (oversampling 8, two power iterations); checkpoint metadata records this approximation and per-matrix captured residual energy.

The production LoopWAM artifact contains 1,079,946,183 parameters and occupies 2,160,261,923 bytes in bf16. This measured count includes video 891,250,880, action 188,658,439 and proprio 36,864 parameters. A serialization regression was found and fixed: the released proprio tensors are views into a 12,041,421,216-byte teacher storage, so conversion must clone them to avoid embedding the entire teacher storage in the compact checkpoint. A dedicated test checks this for both fp32 and bf16.

The initial rank-32 conversion captures less than 30% of the residual energy in 479 of 480 matrix/slot combinations. Weighted captured energy is 10.81% for video and 18.15% for action. This is a diagnostic warning for the recovery experiment, not evidence that the trained policy fails. Rank 64 is not automatically scheduled in the initial 14-run route.

### Measured initialization diagnostics

D1 completed on 1,000 deterministic, unpadded training clips. Mean centered linear CKA between layers in the six cycle groups is 0.9468 for video and 0.8998 for action; mean angular distances are 0.2881 and 0.6318 radians. These are similarities of token-mean teacher features, not proof that arbitrary layer replacement preserves behavior. D1 took 218.5 seconds after loading the model and dataset.

D2 used the same noisy inputs for every model over 20 held-out clips and five noise times. Initial action-velocity MSE against the teacher was 0.5573 for Untied-30, 0.6018 for LoopWAM r32, and 0.6446 with its adapters temporarily disabled. The adapter-disabled result is an initialization diagnostic; no r0 training ablation was added. Production full-rank adapters were not materialized; exact full-rank folding is covered by the numerical real-Wan tests. D2 took 52.1 seconds, including student loading.

![Measured teacher similarity and initial conversion fidelity](figures/loopwam_initialization.png)

Source artifacts: `outputs/loopwam_v1/initialization/{provenance,d1,d2}.json`. The corresponding PDF is `reports/figures/loopwam_initialization.pdf`.

### Implementation map

| File under `src/fastwam/loop/` | Responsibility |
| --- | --- |
| `convert.py` | Structured width conversion, untied controls, cycle means, residual SVD and canonical checkpoint metadata |
| `slots.py` | Shared linear/modulation parameters and explicit per-slot residuals/norms |
| `mot.py` | Virtual-layer schedules, differentiable first-frame caches, video exits and suffix-aligned action attention |
| `model.py` | Original FastWAM preparation/heads/schedulers, teacher bridge, multi-budget losses, inference and export |
| `losses.py` | Masked per-sample FM/KD reductions and sigma-shift guard |
| `data.py` | Immutable split, deterministic preprocessing, cached-record integrity and provenance |
| `sampler.py` | Rank-consistent budget sampling and resumable consumed-window ordering |
| `trainer.py` | DeepSpeed updates, LR, EMA, checkpoints, diagnostics and allocation-boundary handling |
| `diagnostics.py` | Fixed-panel OL1–OL3, loop dynamics, CKA and LoRA norm ratios |
| `evaluation.py` | LIBERO workers, episode evidence, strict summaries and latency measurement |
| `campaign.py` | The 14-run matrix, forks, scientific gates, evaluation dispatch and comparison tables |

The original FastWAM teacher files and the user's configuration edits remain separate from this implementation. The initial repository revision was `7faa711`; all implementation commits are on `LoopWAM_v1`.

## Data and losses

The dataset contains 388 demonstrations and 104,280 frame-start windows. A fixed task-stratified episode split reserves exactly two demonstrations per task: 368 training demonstrations/98,842 windows and 20 validation demonstrations/5,438 windows. Of these, 87,066 and 4,798 windows respectively are unpadded. The manifest records every original window ID, episode interval and metadata hash. All 388 parquet index/frame/episode/task columns were checked against the manifest. End padding retains original FastWAM semantics and is masked in the losses.

Preprocessing uses two 224×224 cameras concatenated horizontally, 33 observation steps subsampled to nine video frames, 32×7 actions and 8-dimensional proprioception. The loader decodes only the nine used video frames and bypasses upstream random-on-error fallback so a failed read cannot silently cross the validation split. Both student and teacher use the released normalization JSON. Stage 1 at 8,000 steps × 128 samples corresponds to 10.36 passes through the actual training-window count.

L2 is action FM plus future-video FM. L3 adds action-velocity KD. Teacher and student receive the exact same noisy video/action tensors and timestep tensors; each computes its own proprio context from the same normalized raw proprio. The teacher is frozen, in eval mode and under `no_grad`, with bf16 autocast. KD uses the same timestep weighting and padding reduction as action FM. Augmented samples receive zero KD contribution. All student/teacher train/inference sigma shifts must be 5.0; the existing action config's 1.0 is explicitly overridden in this workflow.

The elastic objective is `video_FM(4) + sum_b [action_FM(b) + action_KD(b)]` for L3, omitting KD for L2. The sum contains the full `(4,4)` budget and one deterministic, uniformly sampled additional budget per optimizer update. Coefficients are all one; the implementation does not silently average the two configurations or add a shallow video loss. All ranks and accumulation microbatches use the same sampled budget at a given absolute step. Teacher and full video computation are reused within the update.

Open-loop diagnostics use a documented fixed panel of one unpadded midpoint clip from each of the 20 held-out demonstrations, with saved window IDs and fixed noise. They compute OL1 against teacher action velocity at five fixed timesteps, OL2 first-ten action L1 after ten Euler steps and OL3 future-video velocity MSE. This panel is not an exhaustive evaluation of all 5,438 held-out windows. Closed-loop stage-end EMA evaluation remains the primary selection metric.

Additional diagnostics record global accumulated gradient norms before clipping, loop-state/update norms and cross-loop centered CKA at `(4,4), tau=0.5`, and each slot's `||BA||F / ||W||F`. The LoRA norm calculation uses rank-sized Gram matrices rather than materializing full residual matrices. Diagnostics preserve RNG and module modes, and the EMA context restores the exact original ZeRO parameter views.

### Frozen preprocessing cache

The dense-control probe exposed a sustained CPU bottleneck: its final two updates took 3.77–3.90 seconds, including 1.11–1.18 seconds waiting on the slowest loader rank. A shared cache of individual window encodings has been implemented; its full production build is in progress. It retains normalized actions/proprioception, padding masks, window identity and deduplicated text contexts alongside frozen VAE outputs. Every architecture will use the same cache, with strict dataset, preprocessing, normalization and VAE provenance.

A real H100 numerical check found batch-dependent bf16 encoder rounding: batch 16 versus batch 8 differed by 0.4084% relative L2 (maximum absolute difference 0.0625). Consequently, each cached window is encoded separately with a fixed batch size of one. This makes interrupted cache construction independent of batch membership. Singleton repeat, serialization/reload, noise and timestep draws, and first-frame causality were bit-identical in the checked padded and unpadded production clips. The cache preserves bf16 latents; converting them to fp32 would change noise generation. Cached training is not claimed to be bit-identical to the earlier batched-VAE throughput probes. Evidence: `outputs/loopwam_v1/cache_encoding_equivalence.json`.

Each record contains bf16 latents `[48,3,14,28]`, fp32 actions `[32,7]`, fp32 proprioception `[32,8]`, Boolean padding masks and original window/episode/task/frame IDs. The ten text contexts are stored once per task, retaining the original zero-padded embeddings and all-ones attention-mask behavior. Record checksums and a final complete marker prevent a partially written cache from being used for training. Offline construction can validate and refill missing or corrupt records; the training loader fails on corruption instead of silently falling back to a different preprocessing path. The manifest covers all 104,280 windows, including the held-out partition, and the loader enforces the original episode split.

## Optimization, resumption and evaluation

The standalone trainer uses global batch 128, AdamW beta (0.9,0.95), epsilon 1e-8, inherited LR 5e-5, 500-step linear warmup then constant LR, clipping 1.0, and EMA decay 0.999 updated once per optimizer update. Norms, biases, LoRA and slot deltas receive no weight decay. ZeRO-1/2 and microbatch/accumulation are configurable while preserving global batch and optimizer-step budgets. Native bf16 autocast keeps student/master parameters in fp32. The frozen teacher is deliberately outside the student's registered module tree, so it cannot enter the optimizer, EMA or training checkpoints.

Full-state checkpoints preserve raw weights, ZeRO optimizer shards, LR, EMA, absolute optimizer step, rank-specific RNG and the consumed-window cursor. The sampler's committed cursor is independent of dataloader prefetch. Forks preserve all training state and change only the intended sampling mode. A stop signal or time budget is handled at an optimizer boundary and checkpointed before exit. Final bf16 policy exports are distinct from resumable trainer states.

Closed-loop evaluation reuses FastWAM's LIBERO environment/action processing, with 50 fixed initial states per task, ten Euler steps, shift 5, replan every 10 actions, horizon 32 and maximum 700 steps. Persistent workers share tasks across available GPUs. Results retain per-episode paired outcomes, initial-state hashes, protocol/checkpoint provenance, Wilson intervals and measured wall time. Incomplete or failed tasks cannot become a completed 500-episode summary. Stage-end EMA checkpoints are evaluated immediately after each completed run. Comparisons in the 2–4 pp band require the prescribed second evaluation seed and paired McNemar test. The run table records pending/incomplete status explicitly.

## Verification evidence so far

- The integrated suite passed **114 tests in 130.56 seconds** after commit `db96a1a`. Tests cover actual Wan block restoration, full 30-layer velocity equality within 1e-3 fp32, tensor shapes/head maps, storage sharing, causal caches and action-output isolation, prefix exits, first-frame coda equality, all ten schedules, checkpointed gradients, identical KD inputs, per-sample masks, cache interruption/corruption, launch gates and strict bf16 save/reload across every budget. Evidence: `outputs/loopwam_v1/pytest.xml` and `pytest.log`.
- Actual first/last training windows were decoded and checked for video/action/proprio/context shapes and end-padding behavior.
- A single teacher simulator episode (task 0/state 0) succeeded using the intended protocol. This verifies integration only; it is not a benchmark success-rate estimate.
- Four-H100 production L3 updates passed at microbatch 8/accumulation 4 and microbatch 16/accumulation 2. Both preserve global batch 128. Every trainable tensor received finite, nonzero gradients on every rank, including all shared weights, slot parameters, LoRA, norms and proprio parameters.
- A ten-step coupled-sampling probe completed without distributed hangs. Same-output resume advanced absolute step 10 to 11 with LR, optimizer, EMA and sample cursor restored. An explicit fork then advanced step 11 to 12 and changed sampling to fixed while preserving state. The actual EMA open-loop callback passed on 20 held-out windows and three budgets in 47.1 seconds. Untied-30 completed a separate five-update production probe.
- A compile-cache defect was found before the campaign: PyTorch's default guard limit of eight could reject the ninth budget. The policy now scopes a sufficiently large Dynamo limit to its compiled inference call and restores the caller's setting even on an exception. A real-Wan CPU test runs all ten budgets through full-graph compilation, compares their eager outputs, and confirms that revisiting budgets creates no additional graphs. Production H100 `(4,4)` compilation was separately measured below.
- Independent review identified and corrected three issues: shallow OL3 must decode its own video exit; S3-Konly must satisfy the stated (2,2) retention constraint; same-output resume must reject a changed teacher or training recipe. The added Konly evaluation does not add a training trajectory.

### Controlled fixed-batch overfit diagnostic

The first 300-update L2 diagnostic used the standard 500-update warmup. Loss decreased from 2.78033 to 0.16219, but this was not accepted as near zero. Because the diagnostic ended before warmup completed, a follow-up changed only its warmup to zero; the nominal LR remained 5e-5. It reused the same initialization, first global batch of 128, fixed noise, seed, live VAE path, microbatch 16 and accumulation 2. The first forward losses matched exactly before any optimizer update.

Before starting that follow-up, its pass criterion was recorded: finite losses/gradients; mean combined loss over updates 271–300 no more than 1% of the initial loss; and each FM term no more than 0.03. It **passed** with mean combined loss 0.0115441 (0.4152% of initial), video 0.0114105 and action 0.000133641. The final logged interval had combined loss 0.0104683. This establishes that the implementation can memorize the fixed input/noise batch; it does not measure closed-loop generalization. The screening schedule remains unchanged at 500 warmup updates.

The baseline training portion took 517.27 seconds; the follow-up took 890.96 seconds while sharing GPUs with cache construction. The latter is explicitly excluded from throughput comparisons. Both peaked at 56.19 GB allocated per GPU. The passing evidence, criteria and hashed source artifacts are in `outputs/loopwam_v1/overfit_diagnostic_protocol.json`; the campaign refuses to launch P0-S without a matching passing evidence hash.

![Controlled fixed-batch warmup diagnostic](figures/loopwam_overfit.png)

### Preliminary throughput measurements

All measurements below use four H100 80GB GPUs, ZeRO-1, fp32 student/master weights, bf16 autocast and global batch 128. Cold startup, checkpoint writes and validation are separate. These short probes establish feasibility; long-run throughput will replace them in the campaign table.

| Probe | Microbatch/GPU | Accumulation | Measured update time | Peak allocated/GPU | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| LoopWAM L3 fixed, 2 updates | 8 | 4 | 3.52 s, one warm update | 50.33 GB | `runs/loopwam_validation/micro8` |
| LoopWAM L3 fixed, 3 updates | 16 | 2 | 2.20 s, last warm update | 68.23 GB | `runs/loopwam_validation/micro16` |
| LoopWAM L3 coupled, 10 updates | 16 | 2 | 3.235 s, mean of 9 warm updates | 70.99 GB | `runs/loopwam_validation/coupled16/timing_initial10.json` |
| Untied-30 L3 fixed, 5 updates | 8 | 4 | 3.460 s, mean of 4 warm updates | 72.76 GB | `runs/loopwam_validation/c1_micro8/timing.json` |

The coupled probe took 139.8 seconds to initialize and 31.6 seconds to save resumable state plus policy exports. Its warm throughput is 128 / 3.235 = 39.57 samples/s. The preserved initial timing artifact has an older inconsistent throughput field; the step-time numerator and denominator, and this explicit calculation, are used here. The current writer derives both fields from the same measured interval.

### Measured policy latency

The converted LoopWAM at `(Kv, Ka) = (4,4)` completed the production H100 profile: 50 warmups and 500 timed batch-one calls, with VAE, video prefill and ten Euler action steps included. Eager p50/p90/p99 was **307.69 / 316.10 / 327.73 ms**; compiled was **57.03 / 62.41 / 62.74 ms**. Peak allocated memory during inference was 3.80 GB. This is an initialization timing result, not a success-rate result.

Separate CUDA-event measurements after the primary wall-time measurements gave compiled median VAE 4.58 ms, video prefill 5.94 ms, and full ten-step action decoding 43.87 ms. Event intervals include CPU enqueue gaps; their medians should not be added as an exact decomposition of the median end-to-end latency. The primary timings exclude instrumentation overhead. Evidence: `outputs/loopwam_v1/profile_smoke/latency.json`.

## Runtime and limitations

### Launch and resume

The intended production command, after cache completion and the cached four-GPU validation, is:

```bash
bash scripts/loopwam/run_campaign.sh \
  --output outputs/loopwam_v1/campaign \
  --infrastructure outputs/loopwam_v1/infrastructure.json \
  --converted-dir checkpoints/loopwam_v1 \
  --latent-cache checkpoints/loopwam_v1/latent_cache \
  --micro-batch 8 --micro-batch-map loopwam=16,untied12=16 \
  --zero-stage 1 --workers 2 --checkpoint-reserve-seconds 300
```

This gives microbatch 16 × accumulation 2 × 4 GPUs for LoopWAM/Untied-12 and 8 × 4 × 4 for Untied-30/V30-A12. The allocation's end time is discovered from Slurm. Repeating the same command resumes the existing immutable protocol; it does not reinitialize a completed trajectory. The output directory retains `manifest.json`, per-run `train.log`, `metrics.jsonl`, `timing.json`, resumable `state/`, `raw.pt`, `ema.pt`, open-loop artifacts and episode-level evaluation files. `results.csv`, `results.md` and `runtime_estimate.json` are refreshed as evidence arrives.

To inspect the matrix without allocating GPUs or training:

```bash
bash scripts/loopwam/run_campaign.sh --plan
```

Infrastructure evidence is generated from executed tests and measured GPU artifacts, not from a manually asserted Boolean:

```bash
source scripts/activate_fastwam.sh
python -m pytest -q --junitxml=outputs/loopwam_v1/pytest.xml
python scripts/loopwam/check_overfit.py
python scripts/loopwam/verify_infrastructure.py \
  --junit outputs/loopwam_v1/pytest.xml \
  --overfit outputs/loopwam_v1/overfit_diagnostic_protocol.json
```

The standard workflow evaluates every completed trajectory immediately, then advances only if the relevant gate passes. G0 checks C1 against the reproduced teacher; G1 checks recovery against C1/C2; GP checks the deep-video premise; G2 checks elasticity and full-budget retention; G3 checks deep-video benefit at matched measured latency. The otherwise unspecified G2 phrase “well above” is registered as at least 3 pp at both K=1 and K=2, and the matched-latency tolerance is 5%. Ambiguous comparisons request the second evaluation seed automatically. A failed gate records the evidence and stops instead of scheduling the excluded second-round ablations.

Allocation 872809 provides four H100 80GB GPUs, 16 CPUs and 512GB RAM on evc102. It began 2026-10-05 12:02:48 and ends 2026-10-06 08:02:48 (cluster EDT). The user approved up to eight additional 20-hour, four-H100 continuation allocations. `scripts/loopwam/continue_campaign.sbatch` resumes the existing immutable campaign; its bounded chain continues only after an allocation deadline, and stops after a failed scientific gate or runtime error. Submission IDs are recorded separately when actually submitted.

Multiplying the preliminary 2.20–3.235 seconds/update by 142,000 updates gives roughly 87–128 hours of training alone. This is a planning range, not a measured total: the dense controls, data-loader steady state, different elastic modes, closed-loop evaluations, startup and Slurm queue delays remain to be measured. The campaign writes measured per-run timings, comparison CSV/Markdown and a runtime-estimate JSON; unknown quantities stay unknown. A calendar completion date and a best setup require those measurements and passing gates.

Only H100 measurement is available in this allocation. The all-ten-budget latency grid, an RTX4090 profile and delay-injected evaluation remain outstanding. Full LIBERO across the other suites is outside the user's first Long-only campaign. LIBERO-Long is a selection set; later headline generalization claims require benchmarks unused for selection. The requested 14-run route excludes the plan's additional 22k control continuations: the screening controls stop at 8k, so final 22k LoopWAM comparisons against them have unequal training budgets and must be labeled accordingly.
