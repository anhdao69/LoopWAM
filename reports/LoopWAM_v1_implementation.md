# LoopWAM v1: implementation and LIBERO-Long screening

Status (2026-10-05): implementation on branch `LoopWAM_v1`; production infrastructure validation is in progress. No completed screening result or selected recipe is claimed here. This report is updated as measured training and evaluation evidence becomes available.

## Scope and experimental contract

The implementation follows `plans/LoopWAM_v1.md` (v2 architecture) and the user's smaller initial route: P0-S; C1, C2, C3, S1-L2, S1-L3; S2-cont, S2-base; S3-coupled, S3-late, S3-Konly, S3-2stage; two end-to-end confirmation seeds. This is 14 training trajectories, with stage-dependent forks and gates. It excludes the initial video-KD, alternate-LR, r0, re-injection, deep-video-supervision and alternative-alignment ablations. LIBERO-10 is the training dataset and LIBERO-Long is the ten-task closed-loop evaluation suite.

The run budgets remain explicit optimizer-step endpoints: 2,000 smoke steps; 8,000 Stage-1 steps; Stage-2 forks from step8,000 to14,000; Stage-3 forks from14,000 to22,000; Konly and 2stage fork from8,000 and run14,000 additional steps. Confirmation trajectories each run22,000 steps. Total incremental training is142,000 optimizer steps at global batch128 (18,176,000 sampled windows), before any optional diagnosis. Gates can stop this route early; they are not permission to invent a winner or silently add ablations.

## Architecture and conversion

A separate `fastwam.loop` package preserves the original teacher execution as an independent numerical reference. The student uses the original Wan video/action preparation, positional encodings, velocity heads, schedulers and VAE. Video width is2,048 with16x128 attention heads and8,192 FFN channels. Action width is768, with16x128 joint-attention heads,6x128 text-attention heads and3,072 FFN channels. The proprio encoder remains8→4,096, because it creates a text-context token; slicing it to hidden width would be incorrect for this codebase.

The looped model stores3 prelude blocks,6 core blocks and3 coda blocks per stream. Core blocks have four explicit slots containing modulation deltas, copied norm parameters, bias deltas and rank32 LoRA on every linear. Slots are function arguments rather than mutable module state, so activation-checkpoint recomputation cannot accidentally use the last slot executed during the forward pass. No extra recurrent residual is added: each slot executes the original gated-self-attention, cross-attention and gated-FFN residuals once.

Video uses prefix loops1…Kv. Action uses suffix slotsKv−Ka+1…Kv and reads the same virtual video layers. First-frame keys/values retain autograd history in training. The full video pass supplies the final video FM loss and cached loop exits. A shorter video budget recomputes only its three coda blocks on first-frame tokens. This is numerically valid because the first frame never attends to future frames or actions. All ten budgets1≤Ka≤Kv≤4 are supported.

Width conversion selects complete evenly spaced attention heads and FFN channels, then interpolates only action hidden axes with the original alpha-rescaling rule. Modulation/time-projection row groups are handled separately. Core weights are cycle-group means; biases/modulation/norms are restored by slot. Full-rank residual SVD is exact in numerical tests. Production rank32 uses deterministic randomized SVD (oversampling8, two power iterations); checkpoint metadata records this approximation and per-matrix captured residual energy.

The production LoopWAM artifact contains1,079,946,183 parameters and occupies2,160,261,923 bytes in bf16. A serialization regression was found and fixed: the released proprio tensors are views into a12,041,421,216-byte teacher storage, so conversion must clone them to avoid embedding the entire teacher storage in the compact checkpoint. A dedicated test checks this for both fp32 and bf16.

The initial rank32 conversion captures less than30% of the residual energy in479 of480 matrix/slot combinations. This is a diagnostic warning for the recovery experiment, not evidence that the trained policy fails. Rank64 is not automatically scheduled in the initial14-run route.

## Data and losses

The dataset contains388 demonstrations and104,280 frame-start windows. A fixed task-stratified episode split reserves exactly two demonstrations per task:368 training demonstrations/98,842 windows and20 validation demonstrations/5,438 windows. Of these,87,066 and4,798 windows respectively are unpadded. The manifest records every original window ID, episode interval and metadata hash. All388 parquet index/frame/episode/task columns were checked against the manifest. End padding retains original FastWAM semantics and is masked in the losses.

Preprocessing uses two224x224 cameras concatenated horizontally,33 observation steps subsampled to nine video frames,32x7 actions and8-dimensional proprioception. The loader decodes only the nine used video frames and bypasses upstream random-on-error fallback so a failed read cannot silently cross the validation split. Both student and teacher use the released normalization JSON. Stage1 at8,000 steps×128 samples corresponds to10.36 passes through the actual training-window count.

L2 is action FM plus future-video FM. L3 adds action-velocity KD. Teacher and student receive the exact same noisy video/action tensors and timestep tensors; each computes its own proprio context from the same normalized raw proprio. The teacher is frozen, in eval mode and under no_grad, with bf16 autocast. KD uses the same timestep weighting and padding reduction as action FM. Augmented samples receive zero KD contribution. All student/teacher train/inference sigma shifts must be5.0; the existing action config's1.0 is explicitly overridden in this workflow.

Open-loop diagnostics use a documented fixed panel of one unpadded midpoint clip from each of the20 held-out demonstrations, with saved window IDs and fixed noise. They compute OL1 against teacher action velocity at five fixed timesteps, OL2 first-ten action L1 after ten Euler steps and OL3 future-video velocity MSE. This panel is not an exhaustive evaluation of all5,438 held-out windows. Closed-loop stage-end EMA evaluation remains the primary selection metric.

## Optimization, resumption and evaluation

The standalone trainer uses global batch128, AdamW beta(.9,.95), epsilon1e-8, inherited LR5e-5,500-step linear warmup then constant LR, clipping1.0, and EMA decay.999 updated once per optimizer update. Norms, biases, LoRA and slot deltas receive no weight decay. ZeRO1/2 and microbatch/accumulation are configurable while preserving global batch and optimizer-step budgets. Native bf16 autocast keeps student/master parameters in fp32. The frozen teacher is deliberately outside the student's registered module tree, so it cannot enter the optimizer, EMA or training checkpoints.

Full-state checkpoints preserve raw weights, ZeRO optimizer shards, LR, EMA, absolute optimizer step, rank-specific RNG and the consumed-window cursor. The sampler's committed cursor is independent of dataloader prefetch. Forks preserve all training state and change only the intended sampling mode. A stop signal or time budget is handled at an optimizer boundary and checkpointed before exit. Final bf16 policy exports are distinct from resumable trainer states.

Closed-loop evaluation reuses FastWAM's LIBERO environment/action processing, with fixed50 initial states per task, ten Euler steps, shift5, replan10, horizon32 and maximum700 steps. Persistent workers share tasks across available GPUs. Results retain per-episode paired outcomes, initial-state hashes, protocol/checkpoint provenance, Wilson intervals and measured wall time. Incomplete or failed tasks cannot become a completed500-episode summary. Stage-end EMA checkpoints are evaluated immediately after each completed run. Comparisons in the2–4pp band require the prescribed second eval seed and paired McNemar test. The run table records pending/incomplete status explicitly.

## Verification evidence so far

- The integrated CPU suite passed 72 tests at commit `1cfd1f5`. Subsequent focused checks cover deadline cancellation and causal action-output invariance. Tests cover actual Wan block restoration, full 30-layer velocity equality within 1e-3 fp32, tensor shapes/head maps, storage sharing, causal caches, prefix exits, first-frame coda equality, all ten schedules, checkpointed gradients, identical KD inputs, per-sample masks and strict bf16 save/reload across every budget.
- Actual first/last training windows were decoded and checked for video/action/proprio/context shapes and end-padding behavior.
- A single teacher simulator episode (task 0/state 0) succeeded using the intended protocol. This verifies integration only; it is not a benchmark success-rate estimate.
- Four-H100 production L3 updates passed at microbatch 8/accumulation 4 and microbatch 16/accumulation 2. Both preserve global batch 128. Every trainable tensor received finite, nonzero gradients on every rank, including all shared weights, slot parameters, LoRA, norms and proprio parameters.
- A ten-step coupled-sampling probe completed without distributed hangs. Same-output resume then advanced absolute step 10 to 11 with LR, optimizer, EMA and sample cursor restored. The actual EMA open-loop callback passed on 20 held-out windows and three budgets in 47.1 seconds. Explicit fork and control-model probes are still in progress.
- Independent review identified and corrected three issues: shallow OL3 must decode its own video exit; S3-Konly must satisfy the stated (2,2) retention constraint; same-output resume must reject a changed teacher or training recipe. The added Konly evaluation does not add a training trajectory.

### Preliminary throughput measurements

All measurements below use four H100 80GB GPUs, ZeRO-1, fp32 student/master weights, bf16 autocast and global batch 128. Cold startup, checkpoint writes and validation are separate. These short probes establish feasibility; long-run throughput will replace them in the campaign table.

| Probe | Microbatch/GPU | Accumulation | Measured update time | Peak allocated/GPU | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| LoopWAM L3 fixed, 2 updates | 8 | 4 | 3.52 s, one warm update | 50.33 GB | `runs/loopwam_validation/micro8` |
| LoopWAM L3 fixed, 3 updates | 16 | 2 | 2.20 s, last warm update | 68.23 GB | `runs/loopwam_validation/micro16` |
| LoopWAM L3 coupled, 10 updates | 16 | 2 | 3.235 s, mean of 9 warm updates | 70.99 GB | `runs/loopwam_validation/coupled16/timing_initial10.json` |

The coupled probe took 139.8 seconds to initialize and 31.6 seconds to save resumable state plus policy exports. Its warm throughput is 128 / 3.235 = 39.57 samples/s. The preserved initial timing artifact has an older inconsistent throughput field; the step-time numerator and denominator, and this explicit calculation, are used here. The current writer derives both fields from the same measured interval.

## Runtime and limitations

Allocation 872809 provides four H100 80GB GPUs, 16 CPUs and 512GB RAM on evc102. It began 2026-10-05 12:02:48 and ends 2026-10-06 08:02:48 (cluster EDT). The user approved up to eight additional 20-hour, four-H100 continuation allocations. `scripts/loopwam/continue_campaign.sbatch` resumes the existing immutable campaign; its bounded chain continues only after an allocation deadline, and stops after a failed scientific gate or runtime error. Submission IDs are recorded separately when actually submitted.

Multiplying the preliminary 2.20–3.235 seconds/update by 142,000 updates gives roughly 87–128 hours of training alone. This is a planning range, not a measured total: the dense controls, data-loader steady state, different elastic modes, closed-loop evaluations, startup and Slurm queue delays remain to be measured. The campaign writes measured per-run timings, comparison CSV/Markdown and a runtime-estimate JSON; unknown quantities stay unknown. A calendar completion date and a best setup require those measurements and passing gates.

Only H100 measurement is available in this allocation. An RTX4090 profile, component-level latency decomposition, 1,000-clip layer-similarity diagnostics and delay-injected evaluation are not yet validated. Full LIBERO across the other suites is outside the user's first Long-only campaign. LIBERO-Long is a selection set; later headline generalization claims require benchmarks unused for selection. The requested 14-run route excludes the plan's additional 22k control continuations: the screening controls stop at 8k, so final 22k LoopWAM comparisons against them have unequal training budgets and must be labeled accordingly.
