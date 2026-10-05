# LoopWAM Plan 1 — Final Implementation Plan (v2)

Oct 5, 2026 · @Anh Dao

## 1. Summary and changes from v1

Build LoopWAM-1B from `libero_uncond_2cam224.pt` and grow it in three gated stages:

1. Recover the teacher at fixed (K\_v, K\_a) = (4, 4).
2. Make looping elastic with K\_v = K\_a.
3. Decouple the two loop counts (LD-MoT).

Start with three losses. Every other term must beat the simpler recipe in a LIBERO-Long screening run before it enters.

Working rules:

- One question per stage and one change per run, so every regression has a single cause.
- Minimal by default: ties (under 2 pp on LIBERO-Long, Section 6) go to the simpler recipe.
- Screen on LIBERO-Long (train on LIBERO-10, evaluate 500 episodes), then run the chosen recipe once on all four suites.
- Decisions are pre-registered (Section 8). Checkpoints are evaluated only at fixed stage-end steps, never picked by test score.

Fixed inputs:

- Codebase: FastWAM @7faa711 (MIT).
- Teacher: `libero_uncond_2cam224.pt` with `libero_uncond_2cam224_dataset_stats.json`; action and video sigma shift 5.0.
- Data: `yuanty/LIBERO-fastwam` (MuJoCo 3.3.2).

| # | v1 plan | v2 (this plan) | Why |
| --- | --- | --- | --- |
| 1 | Six loss terms, two action configurations, four video exits and three weight schedules, all from step 0 | Three terms (action FM, video FM, action KD) at the final exit. Video KD, hidden KD, motion KD and video deep supervision enter only through screening runs | Too many interacting terms to attribute a regression |
| 2 | Two stages: recovery, then elastic and decoupled at once | Three stages: fixed (4,4), coupled elastic (K\_v = K\_a), decoupled (K\_a ≤ K\_v) | Separates "looping is elastic" from "LD-MoT works" |
| 3 | Shared loop embedding φ(r) | Removed | The per-(loop, block) adaLN delta already encodes the loop index |
| 4 | Re-injection gate in the main model | Off by default; screened in Stage 2 | It does nothing at fixed K |
| 5 | Cosine LR over 30k steps | 500-step warmup, constant LR, EMA weights evaluated; cooldown only in the final run | Stage-2 and Stage-3 runs can fork from stage-end checkpoints without LR confounds |
| 6 | Controls: Untied-12, Untied-30 | Adds Untied-V30/A12, a fixed-depth continuation, a coupled continuation, an action-only elastic run and a two-stage control | Each tests one claim |
| 7 | Action text cross-attention: "6 heads" | 6 of 24 teacher heads, evenly spaced, then hidden-size interpolation | Unspecified in v1 |
| 8 | Untied-12 initialization unspecified | Efficient-WAM layer mapping for both experts | Fair equal-parameter control |
| 9 | Shift mismatch listed only as a risk | Training refuses to start unless teacher shift = student shift = 5.0 | Prevents silent KD corruption |
| 10 | — | `is_augmented` flag masks KD per sample (off for LIBERO) | Needed later for augmented LIBERO-Plus data |

## 2. Architecture: LoopWAM-1B

LoopWAM-1B has two experts with the same 3 + 6×K + 3 layout, coupled by one joint attention per layer as in Fast-WAM. It is about 1.08B parameters, excluding the VAE and T5. The minimal version keeps only the per-slot parameters that make conversion exact, plus rank-32 LoRA.

| Item | Video expert | Action expert |
| --- | --- | --- |
| Hidden width d | 2048 | 768 |
| Self-attention | 16 heads × 128 | Q/K/V projected 768 → 2048 (16 × 128, the video head space); O 2048 → 768 |
| Text cross-attention | 16 heads × 128 | 6 heads × 128 |
| FFN | 8192, GELU-tanh | 3072, GELU-tanh |
| Block type | Wan2.2 block: adaLN-6, q/k RMSNorm, affine cross-attention norm, 3D RoPE | Fast-WAM DiTBlock, 1D RoPE over 32 action tokens |
| Unique blocks | 3 prelude + 6 shared core + 3 coda = 12 | 3 + 6 + 3 = 12 |
| Effective depth | 6 + 6·K\_v (12 / 18 / 24 / 30) | 6 + 6·K\_a |
| Tokens per training sample (LIBERO) | 98 clean first-frame + 196 noisy future = 294 | 32 action tokens; proprio is one extra text-context token |
| Parameters (approx.) | 0.849B | 0.169B |

### Per-slot parameters

Each core block i has a parameter slot for every loop r = 1…4.

| Component per (block i, slot r) | Minimal model (M0) | Total size | Purpose |
| --- | --- | --- | --- |
| adaLN table delta Δm(i,r), 6 × d | On | ≈0.4M | Restores each teacher layer's modulation exactly |
| Cross-attention norm affine, q/k RMSNorm weights | On | ≈0.5M | Exact norm restoration |
| Bias deltas Δb(i,r) on every linear | On | ≈1.0M | Exact bias restoration (averaging would blur them, LoRA cannot restore them) |
| LoRA rank 32 on every linear | On; r = 0 and r = 64 are Stage-1 runs | ≈60M | Carries the rest of each teacher layer's identity |
| Re-injection gate α\_r ∈ R^d, initialized to 0 | Off; screened in Stage 2 | ≈0.01M | Stability under variable K |
| Loop embedding φ(r) | Removed | — | Redundant with Δm(i,r) |

### Block update

Shared block S\_i in slot r uses:

```latex
W_{i,r} = W_i + B_{i,r}A_{i,r},\quad b_{i,r} = b_i + \Delta b_{i,r},\quad [\mu_1, s_1, g_1, \mu_2, s_2, g_2] = m_i + \Delta m_{i,r} + E(t)
```

E(t) is Wan's per-token timestep modulation, computed once per call. The Wan update then runs unchanged: gated self-attention, ungated cross-attention, gated FFN. Optional re-injection, applied only when the flag is on:

```latex
u^{(r)} = s^{(r-1)} + \alpha_r \odot \big(z - s^{(r-1)}\big),\qquad z = \text{prelude output}
```

### Schedules and Loop-Decoupled MoT

- **Video stream (prefix-elastic).** P1–P3, then core loops r = 1…K\_v using slot r, then Q1–Q3. The first-frame K/V of every virtual layer is cached. A shorter K\_v is an early exit of the K\_v = 4 pass.
- **Action stream (suffix-aligned, late alignment).** P1–P3 read the video prelude K/V. Core loop r = 1…K\_a uses action slot σ = K\_v − K\_a + r and reads video (loop σ, block i). Q1–Q3 read the video coda computed from exit K\_v, then the head.
- **K\_a = K\_v gives σ(r) = r**, which is exactly Fast-WAM's layer-aligned coupling.
- **Alignment variants (Stage-3 runs only):**
  - early: σ(r) = r;
  - final: video K/V always from loop K\_v, action slot as in late.
- **Masks: unchanged from Fast-WAM.**
  - The first frame attends to itself.
  - Future frames attend to the first frame and to each other.
  - Actions attend to first-frame video and to all actions.
  - Video never attends to actions.

&#91;embedded content: LoopWAM-1B at inference · (K\_v, K\_a) = (4, 2), late alignment\]

The world pass buys depth once per observation; each denoising step pays only K\_a action loops, aligned to the deepest video loops.

```latex
O_a = \mathrm{softmax}\!\Big(\tfrac{Q_a\,[K_v^{(\sigma(r),i)}[F];\,K_a]^\top}{\sqrt{128}}\Big)\,[V_v^{(\sigma(r),i)}[F];\,V_a]
```

### Budgets per action chunk

Batch 1, LIBERO inputs, 10 denoising steps. FLOPs are analytic counts carried over from v1; latency is measured in Phase 0.

| Configuration (K\_v, K\_a) | Executed layer calls (video + 10 × action) | FLOPs | Weight bytes |
| --- | --- | --- | --- |
| Fast-WAM teacher | 30 + 300 = 330 | 2,046 GFLOP | 30.0 GB |
| (4,4) | 330 | 755 GFLOP | 12.1 GB |
| (4,2) | 210 | 619 GFLOP | 8.9 GB |
| (4,1) | 150 | 551 GFLOP | 7.2 GB |
| (2,2) | 198 | 453 GFLOP | 7.2 GB |
| (1,1) | 132 | 302 GFLOP | 4.8 GB |

Inference compiles one prefill graph per K\_v and one action-step graph per (K\_v, K\_a).

## 3. Conversion from the Fast-WAM teacher

One script, `loop/convert.py`, turns `libero_uncond_2cam224.pt` into LoopWAM-1B and every control model. With full-rank LoRA it must reproduce the width-sliced 30-layer model exactly; that is unit test 1.

### Steps

1. **Video width 3072 → 2048.**
   - Keep head indices H16 = round(linspace(0, 23, 16)) and 8,192 evenly spaced FFN channels of 14,336.
   - Slice every block tensor: self and cross Q/K/V/O, q/k norms, cross-attention norm, FFN, modulation table.
   - Slice the globals: patch embedding, text embedding, time embedding, the six row groups of the time projection, head.
   - Port Efficient-WAM's `_build_structured_sliced_wan_state_dict`.
2. **Action width 1024 → 768.**
   - Self-attention Q/K/V/O use the same H16, so action queries share the video key head space.
   - Text cross-attention uses H6 = round(linspace(0, 23, 6)).
   - Select heads first, then interpolate only the hidden axis 1024 → 768. Use Fast-WAM's per-tensor linear interpolation with √(d\_src/d\_tgt) rescaling (`scripts/preprocess_action_dit_backbone.py`).
   - Keep 3,072 evenly spaced FFN channels of 4,096.
   - Slice the action encoder Linear(7 → 768), the head and the proprio projection the same way.
3. **Depth folding (cycle grouping).**
   - Prelude P\_j ← teacher layer j (1–3); coda Q\_j ← layer 27 + j (28–30).
   - Core S\_i ← the mean of layers ℓ(r, i) = 3 + 6(r − 1) + i for r = 1…4. For example, S\_1 averages layers 4, 10, 16 and 22.
4. **Exact slot restoration.**
   - Δm(i,r) = m\_ℓ − mean\_r(m\_ℓ).
   - Copy the norm weights of layer ℓ into slot (i, r).
   - Δb(i,r) = b\_ℓ − mean\_r(b\_ℓ).
5. **SVD-initialized LoRA.** Residual R = W\_ℓ − W̄\_i; set B = U\_32·diag(S\_32) and A = V\_32ᵀ. Log the captured energy ‖BA‖²\_F / ‖R‖²\_F per slot.
6. **New paths start as no-ops.** Gates α\_r = 0 when re-injection is enabled.

### Control models (same script)

| Model | Video expert | Action expert | Initialization | Params (approx.) |
| --- | --- | --- | --- | --- |
| Untied-30 | 30 layers, width-sliced | 30 layers, width-sliced | Teacher layers 1–30 | 2.47B |
| Untied-12 | 12 layers | 12 layers | Efficient-WAM mapping \[1, 2, 4, 6, 8, 11, 14, 17, 20, 23, 26, 30\] for both experts | 1.02B |
| Untied-V30/A12 | 30 layers | 12 layers: teacher layers \[1, 2, 3, 22–27, 28, 29, 30\]; each reads the video layer of the same index | Teacher layers | 2.2B |
| LoopWAM-1B | 3 + 6×4 + 3 | 3 + 6×4 + 3 | Steps 3–6 | 1.08B |

Untied-V30/A12 is the untied twin of LoopWAM at (4,1): same action depth, same video layers read. If it does not beat Untied-12, deep video does not help a shallow action stream, and LD-MoT has no premise.

### Pre-training diagnostics (once, no training)

- **D1 layer similarity.** Angular distance and linear CKA between teacher block outputs (video and action) on 1,000 LIBERO-10 clips, plus SVD energy per slot. Flag slots under 30% energy.
- **D2 init fidelity.** Open-loop action-velocity MSE against the teacher on held-out clips, at a fixed τ grid, for four models:
  - Untied-30 at init (the cost of width slicing);
  - LoopWAM with full-rank LoRA (must equal Untied-30);
  - LoopWAM r = 32;
  - LoopWAM r = 0.
- **Grouping contingency.** Sequence grouping (each core block applied K times in a row) breaks prefix elasticity and the LD-MoT cache layout. Use it only if Stage 1 fails after r = 64 and a longer Stage 1.

## 4. Loss functions

Yes, the v1 loss was too much for a first implementation. Six terms, two action configurations, four video exits and three schedules all interact, so a regression could not be attributed to any one of them. v2 starts with three terms and adds the rest only through screening runs.

### Default recipe (L3)

```latex
\mathcal{L} = \sum_{c \in \mathcal{C}_{\text{step}}} \gamma_c \Big[ \mathcal{L}^a_{\mathrm{FM}}(c) + \beta\, \mathcal{L}^a_{\mathrm{KD}}(c) \Big] + \lambda_v\, \mathcal{L}^v_{\mathrm{FM}}(K_v{=}4)
```

- Stage 1: C\_step = {(4,4)}.
- Stages 2–3: C\_step = {(4,4), one sampled configuration}.
- Weights: γ = 1 for both configurations, β = 1, λ\_v = 1.

Video loss comes from the full K\_v = 4 path only, unless video deep supervision wins in Stage 2.

### Term definitions

| Term | Definition | Weight | Status |
| --- | --- | --- | --- |
| Action FM | Fast-WAM action flow matching: x\_τ = (1 − σ\_τ)A + σ\_τε, target ε − A, Fast-WAM timestep weighting, mean over 32 × 7 | 1 | Always on |
| Video FM | Same, for the 2 future latent frames; the clean first frame is excluded | λ\_v = 1 | Always on. Removing video co-training cost Fast-WAM 4 pp on LIBERO (97.6 → 93.5) |
| Action KD | MSE between student and teacher action velocity on the same x\_τ, τ, observation, text and proprio. Same timestep weighting as action FM | β = 1 | On by default; removed in S1-L2 |
| Video KD | MSE between student and teacher video velocity on future tokens | 1 | Screened in S1-L4 |
| Hidden KD | Efficient-WAM hidden loss, 1 − cos(P·h\_S, PCA256(LN(h\_T))), at video anchors: student prelude / loop exits / coda vs teacher layers 3 / 9, 15, 21, 27 / 30 | 0.1 constant in screening; 0.2 → 0.1 → 0 schedule if adopted | Conditional (S1-L5) |
| Motion KD | 1 − cos between frame-to-frame differences of per-frame mean hidden states at anchors 15, 21, 27, 30 | 0.1 | Conditional, lowest priority (S1-L6) |
| Video deep supervision | Video FM (plus video KD if adopted) decoded through the coda at the sampled exit K\_v′ | 1 | Screened in S2-dsv |
| Self-KD target | For the sampled configuration, the KD target is the student's own (4,4) velocity with stop-gradient, not the teacher's | — | Screened in S2-self and S3-self |

### The loss ladder

| Name | Terms | First tested in | Runs when |
| --- | --- | --- | --- |
| L2 | Action FM + video FM | P0-S (2k-step smoke test), S1-L2 (full screen) | Always; it is also the no-distillation control |
| L3 | L2 + action KD | S1-L3 | Always (default) |
| L4 | L3 + video KD | S1-L4 | Always |
| L5 | best of {L3, L4} + hidden KD | S1-L5 | Only if LoopWAM trails Untied-30 by > 2 pp, or its open-loop error is > 1.2× Untied-30's |
| L6 | L5 + motion KD | S1-L6 | Only if L5 beat its base by ≥ 2 pp |

### Implementation rules

- Reuse Fast-WAM's `training_loss` for both FM terms. The KD terms use the same weighting function, so their scales are comparable.
- The teacher runs in eval mode, under `no_grad`, in bf16. It receives exactly the student's ε, τ\_a, τ\_v and inputs. Training asserts teacher shift = student shift = 5.0.
- The student uses the teacher's dataset-stats file for normalization in every run. Otherwise the KD targets live in a different action space.
- Each term is averaged over its own valid elements and logged raw and weighted, per configuration.
- A per-sample `is_augmented` mask zeroes the KD terms. It is inactive for LIBERO and used later for augmented LIBERO-Plus data.

## 5. Training stages

Training has three stages, each answering one question and changing one thing; the loss set stays fixed across them:

1. Stage 1 recovers the teacher at fixed (4,4).
2. Stage 2 makes looping elastic with K\_v = K\_a.
3. Stage 3 decouples the loops (LD-MoT).

| Stage | Configurations trained each step | Question | Screening steps (LIBERO-10) | Final-run steps (4 suites) | Exit gate |
| --- | --- | --- | --- | --- | --- |
| 1. Recovery | (4,4) only | Can 12 unique blocks and 30 effective layers recover the dense model? | 0 → 8k | 0 → 7.5k | G1 |
| 2. Coupled elastic | (4,4) + one (K,K), K \~ U{1, 2, 3} | Is one shared-depth model usable at every K? | 8k → 14k | 7.5k → 17.5k | G2 |
| 3. Decoupled LD-MoT | (4,4) + one pair, uniform over (1,1), (2,1), (2,2), (3,1), (3,2), (3,3), (4,1), (4,2), (4,3) | Does deep world + shallow action beat shallow both? | 14k → 22k | 17.5k → 30k | G3 |

- **Coupled pairs stay in the Stage-3 pool** so elasticity learned in Stage 2 is not forgotten.
- **The sampler is seeded by the global step**, so every rank runs the same configuration and the same graph.
- **Transitions:** in screening, a stage ends at its step boundary and the next stage starts only if its gate passed. In the final run, transitions are fixed in advance from the screening curves.

### Optimization (all runs)

| Setting | Value |
| --- | --- |
| Global batch | 128 (gradient accumulation as needed) |
| Optimizer | AdamW, β = (0.9, 0.95), ε = 1e-8, weight decay 0.01. No decay on LoRA, slot deltas, norms or gates |
| Learning rate | 5e-5 for every teacher-derived tensor (including LoRA and slot deltas); 2e-4 for newly initialized modules (gates, hidden-KD projectors). S1-LR tests 1e-4 |
| Schedule | 500-step linear warmup, then constant. The final run adds a linear cooldown to 10% over its last 3k steps |
| EMA | Decay 0.999; EMA weights are evaluated. One raw-vs-EMA check in Phase 1 |
| Precision | bf16 autocast, fp32 master weights, gradient clip 1.0, ZeRO-1 |
| Noise schedule | Video shift 5.0, action shift 5.0 (asserted equal to the teacher's) |
| DDP | `find_unused_parameters=True`, because LoRA slots of unused loops get no gradient when K < 4 |

Constant LR with EMA replaces cosine for one reason. Stage-2 and Stage-3 screening runs fork from the previous stage's checkpoint, and every branch must start from the same LR state. EMA gives the evaluated weights the smoothing a cooldown would.

### One training step

1. Load a batch: clean first-frame latent, 2 future latent frames, 32 × 7 action chunk, cached T5 embedding, proprio.
2. Sample τ\_v, τ\_a, ε\_v, ε\_a; build the noisy future latents and the noisy action chunk.
3. Teacher forward under `no_grad`: action and video velocities, plus anchor hidden states if hidden KD is on.
4. Student video stream at K\_v = 4.
   - Run prelude → 4 core loops → coda.
   - Keep loop-exit states s(1)…s(4) and the first-frame K/V of every virtual layer.
   - Decode the video velocity → video FM (and video KD).
5. For a sampled configuration with K\_v′ < 4, rerun only the video coda on the first-frame tokens of s(K\_v′) to get its coda K/V.
   - This is valid because first-frame tokens never attend to future tokens.
   - With video deep supervision on, run the coda on all tokens and decode the video velocity at that exit.
6. Student action stream for each configuration in the step.
   - Prelude (reads video prelude K/V).
   - K\_a loops on slots σ(r) = K\_v − K\_a + r, reading video loop σ.
   - Coda (reads the coda K/V of exit K\_v), then head.
   - Compute action FM + β·action KD.
7. Sum the losses, backward, optimizer step, EMA update.

Stage 1 costs one teacher forward, one video pass and one action pass per step. Stages 2–3 add one action pass, plus a 3-block coda on 98 tokens when K\_v′ < 4. Measure step time and memory in the smoke run.

## 6. Screening protocol on LIBERO-Long

Every screening decision follows the same protocol:

- Train on LIBERO-10 only.
- Evaluate 500 LIBERO-Long episodes with EMA weights at fixed stage-end steps.
- Treat paired differences under 2 pp as ties, and resolve ties in favour of the simpler recipe.

### Data

- **Training set:** `libero_10_no_noops_lerobot` from `yuanty/LIBERO-fastwam`.
- **Open-loop validation:** hold out 2 demonstrations per task (episode-level split, 20 demos). Record the exact demo and window counts from the manifest.
- **Preprocessing:** identical to Fast-WAM's `libero_2cam` config:
  - `image` and `wrist_image` at 224×224, concatenated to 224×448;
  - 33-step clips, which give 9 video frames (3 latent frames) and 32 actions;
  - 7-D actions, 8-D proprio, min/max normalization from the teacher's stats file.
- **Step budget:** 8k steps at batch 128 for Stage 1, about 10 epochs of LIBERO-10. That assumes roughly 100k training windows (my estimate). Compute N\_windows on day 1 and set Stage 1 to about 10 epochs.

### Closed-loop evaluation

| Item | Setting |
| --- | --- |
| Episodes | 10 tasks × 50 fixed initial states = 500 per eval seed |
| Weights | EMA weights at the stage-end step only; no checkpoint picking |
| Sampler | 10 Euler steps, action shift 5.0, replan every 10 of 32 actions, max 700 steps, FastWAM's LIBERO eval defaults, compiled inference |
| Eval seeds | One per run. Add a second (1,000 episodes in total) for any decision with a 2–4 pp gap |
| Configurations | Stage 1: (4,4).\<br>Stage 2: K = 1, 2, 4 (K = 3 for the winner only).\<br>Stage 3: (4,4), (4,2), (4,1), (2,2), (1,1); the full 10-pair grid for the winner only |

### Open-loop metrics

Computed every 1k steps on the 20 held-out demos, with fixed noise seeds, for every evaluated configuration:

- **OL-1:** action-velocity MSE against the teacher at τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.
- **OL-2:** L1 between the first 10 actions of a full 10-step sample and the ground truth.
- **OL-3:** video-velocity MSE against the teacher on future tokens.

They give learning curves, early kill signals and tie-breakers. They never override success rate.

### Noise bands and statistics

With 500 episodes, the standard error is about 1.0 pp at 95% success and 1.3 pp at 90%. Compare runs paired: same initial states, same eval seed, McNemar test on per-episode outcomes.

| Paired gap in Long success rate | Reading | Action |
| --- | --- | --- |
| < 2 pp | Tie | Take the simpler recipe. Use OL-1/OL-2 only if they differ by > 10% |
| 2–4 pp | Likely | Run a second eval seed. Accept if the gap is still ≥ 2 pp on 1,000 episodes and McNemar p < 0.05 |
| > 4 pp | Clear | Accept |

Winners chosen by a margin under 3 pp get a second training seed in Phase 4.

### Selection-bias caveat

LIBERO-Long's initial states become the selection set. The paper's headline claims must rest on benchmarks never used for selection: the other three LIBERO suites, LIBERO-Plus, LIBERO-Pro and RoboTwin. Say so in the paper.

### Latency

Latency is measured once per architecture configuration, not per run, under these conditions:

- RTX 4090 and H100, batch 1, real LIBERO observations;
- eager and compiled;
- 50 warm-up and 500 timed calls;
- p50, p90 and p99, split into VAE, video prefill and action loop.

Delay-injected evaluation, where the simulator does not pause for the policy, is added in Phase 4.

## 7. Run matrix

Screening is 18 always-run trainings plus up to 8 conditional ones, plus three control continuations, in four phases. Stage-1 runs start from the converted teacher. Stage-2 runs fork from the Stage-1 winner, and Stage-3 runs from the Stage-2 winner. A two-seed confirmation and one full four-suite run follow.

Notation:

- Loss sets: L2 = action FM + video FM; L3 = L2 + action KD; L4 = L3 + video KD; L5 = + hidden KD; L6 = + motion KD.
- r = LoRA rank.
- "Fork X" = start from run X's checkpoint (weights, optimizer and EMA).
- "Recipe" = the loss set, LR and r chosen in Phase 1.

### Phase 0 — infrastructure (no success-rate decisions)

| ID | What | Pass condition |
| --- | --- | --- |
| P0-T | Unit-test suite (Section 9) | All 14 tests pass |
| P0-R | Teacher reproduction: full LIBERO (2,000 episodes) plus LIBERO-Long with 2 eval seeds | Average within 1 pp of 97.6; Long within 1.5 pp of 95.2 |
| P0-D1 | Layer similarity and SVD energy per slot | Report; flag slots under 30% energy |
| P0-D2 | Init fidelity: Untied-30 at init, and LoopWAM with full-rank / r32 / r0 LoRA | Full-rank LoopWAM equals Untied-30 to numerical precision |
| P0-S | Smoke test: LoopWAM r32, L2, (4,4), 2k steps; plus a 300-step overfit on one batch | Finite losses that decrease; every shared, slot and LoRA tensor gets gradient; overfit loss near zero. Record step time and memory |
| P0-L | Latency profile: teacher, Untied-12/30/V30A12, LoopWAM at all 10 pairs | Table of p50/p90/p99, eager and compiled |

### Phase 1 — Stage 1, fixed (4,4), 8k steps from the converted init

| ID | Model | Loss | Change | Question | Runs |
| --- | --- | --- | --- | --- | --- |
| C1 | Untied-30 | L3 | — | What does width slicing cost? This is LoopWAM's ceiling | Always |
| C2 | Untied-12 | L3 | — | Equal-parameter baseline | Always |
| C3 | Untied-V30/A12 | L3 | — | LD-MoT premise: does deep video help a 12-layer action stream? | Always |
| S1-L2 | LoopWAM r32 | L2 | — | Is teacher KD needed at all? | Always |
| S1-L3 | LoopWAM r32 | L3 | — | Default minimal recipe | Always |
| S1-L4 | LoopWAM r32 | L4 | — | Does video KD help? | Always |
| S1-LR | LoopWAM r32 | L3 | LR 1e-4 | Is 5e-5 too conservative for recovery? | Always |
| S1-R0 | LoopWAM r0 | L3 | No LoRA | Is per-loop LoRA needed? | Always |
| S1-L5 | LoopWAM r32 | best of L3/L4 + hidden KD | — | Does hidden KD close a conversion gap? | If the best LoopWAM < C1 − 2 pp, or its OL-1 > 1.2× C1's |
| S1-L6 | LoopWAM r32 | L5 + motion KD | — | Does motion KD add anything? | Only if S1-L5 beat its base by ≥ 2 pp |
| S1-R64 | LoopWAM r64 | best loss | — | Is LoRA capacity the bottleneck? | If S1-R0 trails S1-L3 by ≥ 3 pp |

Phase 1 output: the Stage-1 recipe and checkpoint S1\* at 8k.

### Phase 2 — Stage 2, coupled elastic, fork S1\* at 8k, +6k steps

| ID | Change from S2-base | Question | Runs |
| --- | --- | --- | --- |
| S2-cont | Keeps training fixed (4,4) only | Control: elastic tax at K = 4, and zero-shot truncation at K < 4 | Always |
| S2-base | (4,4) + one (K,K), K \~ U{1, 2, 3}; teacher KD on both | Is the model elastic? | Always |
| S2-reinj | + gated re-injection (α = 0 at start) | Does re-injection stabilize variable K? | Always |
| S2-dsv | + video FM (and video KD if in the recipe) at the sampled exit | Does video deep supervision help shallow K? | Always |
| S2-self | Sampled-configuration KD target = the student's own (4,4) output, stop-gradient | Can self-distillation replace the teacher for shallow K? | Always |
| S2-combo | Re-injection + deep supervision | Are the two gains additive? | If both S2-reinj and S2-dsv win |
| S2-γ | Sampled-configuration weight 0.5 | Protects K = 4 | If S2-base (4,4) < S2-cont (4,4) − 1.5 pp |

Evaluate every Phase-2 run at K = 1, 2, 4. Output: the Stage-2 recipe and checkpoint S2\* at 14k.

### Phase 3 — Stage 3, decoupled LD-MoT, +8k steps

| ID | Fork | Sampling | Alignment | Question | Runs |
| --- | --- | --- | --- | --- | --- |
| S3-coupled | S2\* | Stage-2 coupled sampling, continued | — | Control: does a coupled model already work at (4, K\_a) zero-shot? | Always |
| S3-late | S2\* | The 9 pairs with K\_a ≤ K\_v | Late | Main LD-MoT run | Always |
| S3-early | S2\* | Same | Early | Is late alignment the right choice? | Always |
| S3-Konly | S1\* at 8k, +14k | K\_v = 4 fixed; (4,4) + (4, K\_a), K\_a \~ U{1, 2, 3} | Late | Do the headline (4, K\_a) points need video elasticity at all? | Always |
| S3-2stage | S1\* at 8k, +14k | The 9 pairs immediately (v1 schedule) | Late | Is the coupled stage worth it, at equal 22k steps? | Always |
| S3-final | S2\* | The 9 pairs | Final | Alternative alignment | If S3-early ties S3-late |
| S3-self | S2\* | The 9 pairs, self-KD on the sampled pair | Late | Teacher-free shallow configurations | If S2-self was within 1 pp of S2-base |
| S3-wt | S2\* | Pairs with K\_a < K\_v weighted 2× | Late | Undertrained late alignment | If S3-late at (4,1) < (1,1) |
| C1/C2/C3-22k | Continue C1–C3 | Fixed | — | Equal-step controls at 22k | Always |

Evaluate at (4,4), (4,2), (4,1), (2,2), (1,1). S3-Konly is evaluated at (4,4), (4,2), (4,1); controls at their single configuration.

### Phase 4 — confirmation on LIBERO-Long

| ID | What | Evaluation |
| --- | --- | --- |
| F-Long-s1, F-Long-s2 | Chosen recipe end-to-end as one continuous 22k run (no forks), 2 training seeds | Full 10-pair grid, 2 eval seeds (1,000 episodes per configuration), delay-injected eval for (4,4), (4,1), (1,1) |
| Controls | Untied-12 and Untied-V30/A12 retrained with the chosen loss for 22k, if Phase 1 used a different loss | 2 eval seeds |

Phase 4 output: success-vs-measured-latency plot for one LoopWAM against the untied students and the teacher.

### Phase 5 — full LIBERO

See Section 10.

## 8. Decision rules and gates

Each stage ends with one pre-registered gate. A failed gate triggers a named diagnosis, not another stage.

&#91;embedded content: screening and final phases · 6 phases, 4 gates\]

A phase starts only after the previous gate passes. A failed gate leads to the fallback on the right, not to the next phase.

### Selection rule per stage

| Stage | Primary metric | Constraint | Tie-break (gap < 2 pp) |
| --- | --- | --- | --- |
| 1 | Long success rate at (4,4) | — | Fewer loss terms, then lower LoRA rank, then lower OL-1 |
| 2 | Mean Long success over K = 1, 2, 4 | Winner's (4,4) ≥ S2-cont's (4,4) − 1.5 pp | Fewer added components |
| 3 | Mean Long success at (4,1) and (4,2) | (4,4) and (2,2) ≥ S3-coupled's − 1.5 pp | See the two exceptions below |

Two deliberate exceptions to "simpler wins" in Stage 3:

- **S3-late vs S3-Konly.** Keep S3-late, which serves every (K\_v, K\_a), unless S3-Konly beats it by ≥ 2 pp at (4,1) or (4,2). One model for all budgets is part of the contribution.
- **3-stage vs S3-2stage.** On a tie, keep 3 stages. Total steps are equal, and the Stage-2 checkpoint gives the coupled-elastic row of the paper for free.

### Gates

| Gate | Pass condition (LIBERO-Long, paired) | If it fails |
| --- | --- | --- |
| G0 infrastructure | Phase 0 passes and C1 ≥ teacher − 3 pp | C1 failing means width slicing is the problem. Switch the main model to LoopWAM-L (full width 3072/1024, same layout, 2.58B) and rerun Phase 1 |
| G1 recovery | Best LoopWAM (4,4) ≥ C1 − 2 pp and ≥ C2 | Check D1/D2. If SVD energy is low, run S1-R64. Then try a 12k-step Stage 1, then LR. Sequence grouping is the last resort (it costs prefix elasticity) |
| GP premise | C3 ≥ C2 + 2 pp, or clearly better OL-1 | Deep video does not help a shallow action stream at this scale. Still run Stage 2, but drop LD-MoT as the headline: re-scope to an elastic looped WAM with K(τ) schedules, or a LoopWAM-L analysis paper |
| G2 elastic | Winner's (4,4) ≥ S2-cont − 1.5 pp; K = 2 ≥ C2; K = 1 ≥ C2 − 3 pp; K < 4 well above S2-cont's truncated K < 4 | Try S2-γ and S2-dsv. If K = 1 still collapses, restrict the sampler to K ≥ 2 and continue |
| G3 LD-MoT (paper go/no-go) | (4,1) beats (1,1), or (4,2) beats (2,2), by ≥ 3 pp (same action depth, deeper video), and beats C2 at matched measured p50 latency (or lies above the C2–C3 line on the success-vs-latency plot) | v1 fallbacks: any-depth backbone for the adaptive-compute line; LoopWAM-L analysis paper at a robotics venue; conversion recipe for edge deployment |

### Reading failures across stages

| Observation | Diagnosis |
| --- | --- |
| Stage 1 fails and C1 fails | Width slicing, not looping |
| Stage 1 fails, C1 passes | Depth folding or conversion (check D1/D2, LoRA rank) |
| Stage 1 passes, Stage 2 has good K = 4 but bad K = 1, 2 | Elastic-depth training |
| Stage 2 passes, Stage 3 has good (2,2) but bad (4,2) | LD-MoT alignment specifically (S3-early, S3-final, S3-wt) |
| S3-coupled already good at (4, K\_a) zero-shot | Decoupled training adds little; report it as such |
| C3 ≈ C2 | Deep video does not help; LD-MoT premise fails regardless of looping |

## 9. Unit tests and diagnostics

No success-rate run starts until all 14 unit tests pass. Every run logs the same diagnostics, so a failure can be localized to a stage, a stream or a loop.

### Unit tests (P0-T)

- [ ] 1\. Conversion equality: LoopWAM with full-rank LoRA equals Untied-30 (max absolute velocity difference < 1e-3 in fp32 on a fixed batch).
- [ ] 2\. Shapes: every sliced tensor matches its target size. H16 and H6 head indices are stored in checkpoint metadata.
- [ ] 3\. Storage sharing: each core block's tensors share one storage across all loop calls. Checkpoint size matches unique parameters × 2 bytes (bf16).
- [ ] 4\. Residual semantics: with re-injection α = 0 the loop equals plain block composition; no residual is added twice.
- [ ] 5\. Schedules: for each of the 10 (K\_v, K\_a) pairs, the list of (block, slot, video cache key) matches Section 2. (4,4) gives σ(r) = r.
- [ ] 6\. Branch isolation: replacing future frames or action labels leaves the inference action output unchanged (eval mode, fixed RNG).
- [ ] 7\. Cache causality: first-frame K/V are identical with and without future tokens present.
- [ ] 8\. Coda on first-frame tokens: its K/V equal the first-frame slice of an all-token coda pass.
- [ ] 9\. Prefix elasticity: exit states at K\_v′ < 4 equal the matching states of the K\_v = 4 pass.
- [ ] 10\. Sampler: identical (K\_v, K\_a) on every rank at every step; stage boundaries at the configured steps.
- [ ] 11\. Shift assert: training refuses to start when teacher and student shifts differ.
- [ ] 12\. Normalization: the student loads the teacher's stats file; normalize → denormalize round trip is exact; gripper sign is correct.
- [ ] 13\. Gradient coverage: after one step every shared, slot and LoRA tensor in use has a finite gradient, and DDP does not hang when K < 4.
- [ ] 14\. Save/load: outputs match before and after reload at every (K\_v, K\_a). Loading uses `strict=True` with an explicit key map.

### Logged diagnostics

| Group | What | Frequency | Catches |
| --- | --- | --- | --- |
| Losses | Every term, raw and weighted, per configuration | Every step | Term imbalance |
| Gradients | Norm per group: prelude, core, slot deltas, LoRA, coda; action vs video | 100 steps | Dead or exploding groups |
| Loop dynamics | State norm per loop, update norm between loops, cross-loop CKA on a fixed batch | 1k steps | Residual growth, loop stagnation |
| LoRA | ‖BA‖\_F / ‖W̄‖\_F per slot | 1k steps | LoRA doing all the work |
| Gates | Mean and max of α\_r (when enabled) | 1k steps | Gate collapse |
| Open loop | OL-1, OL-2, OL-3 for every evaluated configuration | 1k steps | Early regressions |
| System | Step time, peak memory, samples per second | 100 steps | Cost drift |

## 10. Final full LIBERO run and extensions

The chosen recipe is retrained from the converted teacher as one multi-task model on all four suites. It runs for 30k steps, with stage transitions fixed in advance, and is evaluated with three eval seeds.

| Item | Setting |
| --- | --- |
| Data | All four suites, no held-out split (Fast-WAM protocol) |
| Initialization | Freshly converted `libero_uncond_2cam224.pt`, never a screening checkpoint |
| Stages | 7.5k / 10k / 12.5k by default. Raise Stage 1 to the step where screening OL-1 plateaued, rounded up to the next 2.5k. If S3-Konly won, Stages 2 and 3 become one action-only elastic stage (17.5k → 30k) |
| Schedule | 500-step warmup, constant LR from Phase 1, linear cooldown to 10% over the last 3k steps, EMA 0.999 |
| Training seeds | 2 for the main model, 1 for each control |
| Evaluation | 4 suites × 10 tasks × 50 initial states × 3 eval seeds = 6,000 episodes per configuration. Main model: full 10-pair grid on seed 1, five key pairs on seed 2 |
| Controls | Untied-12, Untied-30 and Untied-V30/A12 with the same recipe and steps; the teacher re-evaluated under the same protocol |
| Reported | Per-suite and average success with Wilson intervals; parameters stored / unique / trainable / loaded (VAE included); FLOPs; VRAM; p50/p90/p99 latency eager and compiled; delay-injected success |

### Extensions (recipe unchanged)

| Benchmark | Teacher and data | Protocol notes |
| --- | --- | --- |
| LIBERO-Plus | No new teacher: evaluate the full LIBERO model zero-shot | Pin its environment and reproduce the teacher on standard LIBERO inside it first. Report by perturbation type |
| LIBERO-Pro | Same | Same; confirm from its repo that it is evaluation-only |
| RoboTwin 2.0 | `robotwin_uncond_3cam_384.pt`, shift 5.0. Rerun the conversion from this teacher | 3 cameras at 240×320 composed to 384×320 (about 120 first-frame tokens); 14-D action and proprio with z-score stats; recompute PCA bases if hidden KD is used; unseen instructions; 50 tasks × 100 × clean/randomized = 10,000 episodes per system |
| Augmented LIBERO-Plus training (optional) | LIBERO teacher | `is_augmented` mask: KD off on augmented samples, ground truth only |

On RoboTwin, train only the main model, Untied-12 and Untied-V30/A12, with hyperparameters frozen from LIBERO. Each system costs 10,000 evaluation episodes.

## 11. Implementation map (FastWAM @7faa711)

The whole plan runs from one training script and one config schema. Every screening run differs from the default only in the flags its row names in Section 7.

| File | Change |
| --- | --- |
| `src/fastwam/models/wan22/wan_video_dit.py` (DiTBlock) | Accept `slot=(i, r)`: adaLN delta, norm affines, bias deltas, LoRA. The teacher path stays untouched |
| `src/fastwam/models/wan22/action_dit.py` | Same for action blocks, plus the narrowed 6-head text cross-attention |
| `src/fastwam/models/wan22/mot.py` | Replace `expert.blocks[layer_idx]` with schedule lookups in `forward_joint_core`, `prefill_video_cache_tensor`, `forward_action_with_video_cache_tensor` (cache key = the σ map) and `forward`. Replace the equal-depth assertion with a schedule check |
| `src/fastwam/models/wan22/fastwam.py` | `infer_action(K_v, K_a)`; one compiled graph per prefill K\_v and per action (K\_v, K\_a); `load_checkpoint` with `strict=True` and an explicit key map |
| new `loop/convert.py` | Width slicing, cycle grouping, slot restoration, SVD-LoRA, control-model builders, metadata |
| new `loop/slots.py` | `SlotLinear` (shared W and b, per-slot Δb and LoRA A/B) and slot norms |
| new `loop/schedule.py` | Virtual schedules and σ maps (late / early / final) |
| new `loop/sampler.py` | Stage-aware sampler seeded by global step |
| new `distill/teacher.py` | Frozen teacher wrapper, shift assert, anchor hooks |
| new `distill/losses.py` | Action/video KD, hidden KD with PCA projectors, motion KD, `is_augmented` mask |
| new `tools/profile_latency.py`, `tools/flops.py`, `tools/diagnostics.py` | Latency (p50/p90/p99, eager/compiled, delay injection), FLOP counts, loop dynamics |
| new `configs/model/loopwam_1b.yaml`, `configs/task/libero10_loopwam_*.yaml`, `configs/task/libero_loopwam_full.yaml` | Model, screening and final configs |

### Config schema (default = S1-L3 continued through all stages)

```yaml
model:
  arch: loopwam            # loopwam | untied30 | untied12 | untied_v30a12
  video:  {width: 2048, heads: 16, ffn: 8192, prelude: 3, core: 6, coda: 3}
  action: {width: 768, self_heads: 16, cross_heads: 6, ffn: 3072, prelude: 3, core: 6, coda: 3}
  k_max: 4
  lora_rank: 32            # 0 | 32 | 64
  reinject: false
  align: late              # late | early | final
teacher:
  ckpt: checkpoints/fastwam_release/libero_uncond_2cam224.pt
  stats: checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
  sigma_shift: 5.0
loss:
  a_fm: 1.0
  v_fm: 1.0
  a_kd: 1.0
  v_kd: 0.0
  hid: 0.0
  mot: 0.0
  sampled_weight: 1.0
  sampled_target: teacher  # teacher | self
  sampled_video_fm: false  # video deep supervision
train:
  data: libero_10          # libero_10 (screening) | libero_all (final)
  global_batch: 128
  lr_inherited: 5.0e-5
  lr_new: 2.0e-4
  warmup: 500
  schedule: constant       # constant | constant_cooldown
  ema: 0.999
  stages:
    - {name: s1, until: 8000,  mode: fixed}
    - {name: s2, until: 14000, mode: coupled,   k: [1, 2, 3]}
    - {name: s3, until: 22000, mode: decoupled} # or: konly
```

### Forward-pass sketch

```python
def video_prefill(self, x, t_mod, ctx, Kv):
    """Prelude, Kv core loops, coda. Caches first-frame K/V of every virtual layer."""
    cache, exits = {}, {}
    for j, blk in enumerate(self.prelude):
        x, cache["pre", j] = blk(x, t_mod, ctx)
    z = s = x
    for r in range(1, Kv + 1):
        if self.reinject:
            s = s + self.alpha[r] * (z - s)
        for i, blk in enumerate(self.core):
            s, cache["core", r, i] = blk(s, t_mod, ctx, slot=(i, r))
        exits[r] = s
    cache["coda", Kv] = self.run_coda(s, t_mod, ctx)  # 3 K/V entries; rerun on exits[k] for k < Kv
    return cache, exits

def action_step(self, a, t_mod, ctx, cache, Kv, Ka, align="late"):
    """One denoising step: prelude, Ka core loops, coda, head."""
    for j, blk in enumerate(self.prelude):
        a = blk(a, t_mod, ctx, video_kv=cache["pre", j])
    z = s = a
    for r in range(1, Ka + 1):
        sig = r if align == "early" else Kv - Ka + r     # action slot
        vloop = Kv if align == "final" else sig           # video loop read
        if self.reinject:
            s = s + self.alpha[sig] * (z - s)
        for i, blk in enumerate(self.core):
            s = blk(s, t_mod, ctx, video_kv=cache["core", vloop, i], slot=(i, sig))
    for j, blk in enumerate(self.coda):
        s = blk(s, t_mod, ctx, video_kv=cache["coda", Kv][j])
    return self.head(s, t_mod)
```

Two engineering traps:

- LoRA slots of unused loops get no gradient when K < 4. Use `find_unused_parameters=True` or a zero-weighted touch of every slot.
- The sampled configuration must be identical on every rank, or collective ops deadlock.

## 12. Risks, order of work and budget

The two risks that can end the project both surface in Phase 1:

- width slicing failing on LIBERO-Long (C1);
- deep video not helping a shallow action stream (C3 vs C2).

Launch C1–C3 as soon as the conversion tests pass.

| Risk | Early signal | Response |
| --- | --- | --- |
| Width slicing loses LIBERO-Long | C1 < teacher − 3 pp | LoopWAM-L (full width) |
| Averaging destroys teacher layers | Low SVD energy in D1; LoopWAM < C1 − 2 pp | r = 64, longer Stage 1, sequence grouping last |
| Deep video does not help | C3 ≈ C2 | Re-scope before Stage 3 |
| Elastic tax at K = 4 | S2 (4,4) < S2-cont − 1.5 pp | Sampled weight 0.5 |
| Shallow K collapses | K = 1 far below C2 | Video deep supervision, re-injection; restrict to K ≥ 2 |
| Late alignment undertrained | S3-late (4,1) < (1,1) | S3-wt, S3-final |
| LIBERO-Long noise hides effects | Every variant within 2 pp | Second eval seed, OL metrics; move the decision to a RoboTwin subset |
| Selection bias on Long | — | Headline claims on unselected benchmarks only |
| No measured latency gain | p50 tracks executed layer calls | Claim only measured numbers; the story rests on (4,1) and (4,2) |
| Teacher shift mismatch | KD loss abnormally large at step 0 | Hard assert in the trainer |

### Order of work

1. Conversion script and unit tests 1–9; teacher reproduction in parallel.
2. Launch C1, C2 and C3 as soon as test 1 passes. They do not depend on any LoopWAM decision.
3. Smoke test, D1/D2 and latency profile.
4. Phase 1 LoopWAM runs, all in parallel. Then one conditional round (L5, L6, R64) if triggered.
5. Phase 2 in parallel, then Phase 3 in parallel; the C1–C3 continuations run alongside Phase 3.
6. Phase 4 confirmation, then the Phase 5 full run.

### Budget

Training in optimizer steps at global batch 128; evaluation in episodes. Stage-2/3 steps cost more than Stage-1 steps (a second action pass), and Untied-30 steps more than LoopWAM steps.

| Phase | Training runs (always / with conditionals) | Training steps | Evaluation episodes |
| --- | --- | --- | --- |
| 0 | 1 smoke run | 2k | 3,000 (teacher: 2,000 full LIBERO + 1,000 Long) |
| 1 | 8 / 11 | 64k–88k | 4,000–5,500 Long |
| 2 | 5 / 7 | 30k–42k | 8,000–11,000 Long |
| 3 | 5 + 3 continuations / 11 | 94k–118k | 15,500–23,000 Long |
| 4 | 2 / 4 | 44k–88k | about 25,000 Long, including delay-injected runs |
| Screening total | 24 / 34 | about 234k–338k | about 56k–68k Long, plus about 10% for second eval seeds |
| 5 (full LIBERO) | 2 main + 3 controls | 150k | about 114,000 across 4 suites |

Evaluation, not training, dominates the screening budget. Kill a run early, without closed-loop evaluation, if its OL-1 at the stage end is more than 1.5× the best run in the same phase.

