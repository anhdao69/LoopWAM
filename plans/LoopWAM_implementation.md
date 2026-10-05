# LoopWAM v1 implementation and execution ledger

Authority: `plans/LoopWAM_v1.md`, with the user's minimum 14-training route overriding the larger matrix. Work on branch `LoopWAM_v1`; preserve existing local setup. Execution is authorized by the request.

## Design

Reuse FastWAM video/action preparation, masks, velocity heads, VAE, schedulers and LIBERO simulator. Add a separate LoopMoT and LoopWAM subclass so teacher numerics remain available as a reference. Each core block owns one shared weight and four explicit slot parameters; pass slot IDs explicitly (no mutable active-slot state, which breaks checkpoint recomputation). Video caching stores differentiable first-frame keys/values for each virtual layer and full-pass exits; sampled shallow paths recompute just their three first-frame coda layers. Action slots use suffix alignment.

Converted checkpoints store canonical video/action/proprio state, architecture and selected-head metadata. Strict loading must reject missing or extra keys. Conversion equality is tested in fp32 on tiny real Wan blocks at full residual rank. Production conversion uses rank-32 truncated SVD with energy reporting. Teacher and students use shift 5 and the same normalization JSON.

## Tasks and interfaces

1. Conversion and slots: `fastwam.loop.slots` exposes a slotted DiT block with explicit `slot` argument and `attention_io`/`post_attention`; `fastwam.loop.convert` supplies width-sliced experts, folded experts and strict canonical checkpoints. Validate shapes, full-rank equality, exact norm/bias/modulation restoration, sharing and energy.
2. Core execution: `fastwam.loop.mot.LoopMoT` with schedule lookups, prefix caches, coda recomputation and differentiable action cache reads. `fastwam.loop.model.LoopWAM` provides distillation `forward(sample, global_step=...)`, checkpoint and inference integration. Validate causality, prefixes, all ten pairs, gradients, save/load, masking and shift contracts.
3. Dataset and training: deterministic two-demos-per-task holdout, immutable manifest and teacher stats; standalone distributed trainer with fp32 parameters, bf16 autocast, AdamW ZeRO-1/2, fixed global batch 128, accumulation, EMA .999, 500-step warmup/constant LR, full state forks/resume and timing. Cache deterministic VAE/text processing if profiling justifies it.
4. Campaign and evaluation: exactly P0-S, C1/C2/C3/S1-L2/S1-L3, S2-cont/S2-base, S3-coupled/S3-late/S3-Konly/S3-2stage and two confirmations. Immediate stage-end EMA eval, paired 500-episode outcomes, gates and table. Preserve total step budgets (Konly and 2stage +14k). Stop on failed gate; never silently choose a winner or enter conditional ablations.
5. Validate: CPU numerical tests, four-GPU distributed smoke, actual dataset forward/backward, then measured short throughput probes. Independently review the implementation and fix concrete defects before long training.
6. Execute and report: launch authorized training/evaluation in current allocation, write evidence-backed technical report and estimates from measured throughput; checkpoint before allocation ends. No unmeasured success/performance claims.

## Review focus

- Distillation proprio context must use each model's own encoder on identical raw normalized proprio.
- No future-token or action-label leakage into cached observation features.
- Slot IDs must survive activation checkpoint backward unchanged.
- Accumulation counts optimizer updates; EMA and LR update once per optimizer step.
- Forks preserve optimizer, EMA and absolute step; changed sampling starts at the correct boundary.
- Selection gates use paired outcomes; incomplete/failed evaluations never become zeros or winners.

## Initial findings / rulings

- Base commit 7faa711, existing user changes limited to configs and activation script. No tests supplied.
- Existing action configuration has shift 1; explicit LoopWAM shift 5 required.
- User's 14-run cap excludes initial video KD/LR/r0, reinjection/DSV and alignment/self-KD extras, plus control continuations.
- Stay in the requested checkout on its new branch to preserve the installed editable environment and data symlinks.
- Allocation 872809 on evc102: four H100 80GB, 16 CPUs, 512GB RAM; end 2026-10-06 08:02:48 cluster time.
- The plan says both fixed 8k and approximately ten epochs; retain the explicit 8k/14k/22k boundaries for the fair initial campaign and report measured window/epoch counts.
