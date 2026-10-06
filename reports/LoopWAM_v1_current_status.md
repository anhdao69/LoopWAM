# LoopWAM v1 — current experiment status

**Snapshot: 2026-10-06T14:11:48-04:00 (America/New_York). Branch: `LoopWAM_v1`.**

## Summary

- Six of 14 registered training trajectories are complete: 42,000 of 142,000 incremental optimizer updates.
- Thirteen complete primary evaluation batches are available: 6,500 episodes, including the released teacher and both registered seeds for screening controls/candidates.
- Paired Stage-1 selection chooses **S1-L2**. Its pooled success is **75.3%**, versus **70.1%** for S1-L3.
- The unchanged G1 function returns **fail** in a read-only precheck: selected S1-L2 reaches 79.0% on seed 42; C1 reaches 93.8%, making the required recovery threshold 91.8%.
- The official controller is still running the required S1-L2 raw-versus-EMA diagnostic before recording G1. No Stage-2 training has started.
- The two-allocation scheduler is deployed and verified. Later training is held by the scientific gate, not a lack of scheduling support.
- No final best setup or valid all-14 completion ETA is available under the current go/no-go plan. Proceeding despite G1 would be an explicit exploratory change.

## Allocations and current work

| Job | Node/state | Current role | Allocation end |
| --- | --- | --- | --- |
| 872933 | evc102 / RUNNING | Coordinator and S1-L2 raw diagnostic, four H100s | 2026-10-07T04:53:11 |
| 873007 | evc104 / RUNNING | Available second training slot; currently idle | 2026-10-07T04:53:11 |
| 873269 | PENDING | Existing bounded continuation; dependency on both interactive jobs | Queue dependent |
| 878362 | PENDING | Unrelated user job; untouched | Queue dependent |

Raw diagnostic progress at capture: **5/10 complete tasks**. Partial task outcomes are not presented as a full-suite success rate.
The continuation wrapper skips a campaign stopped by a failed gate or error. It does not override G1. No new allocation was submitted for this publication.

## Completed LIBERO-Long results

Each seed uses ten tasks and 50 initial states per task (500 episodes). These are stage-end EMA outcomes. The smoke run is not a final candidate. Paired evaluation seeds are not independent training seeds.

| Run | Seed 42 | Seed 43 | Pooled success | Training wall time | Eval time: seed 42 / 43 |
| --- | ---: | ---: | ---: | --- | --- |
| teacher | 477/500 (95.4%) | 473/500 (94.6%) | 95.0% | Released checkpoint | 0h 27m 00s / 0h 20m 46s |
| P0-S | 85/500 (17.0%) | — | — | 0h 40m 50s | 0h 48m 19s / — |
| C1 | 469/500 (93.8%) | 446/500 (89.2%) | 91.5% | 3h 52m 07s | 0h 28m 58s / 0h 20m 13s |
| C2 | 343/500 (68.6%) | 292/500 (58.4%) | 63.5% | 1h 41m 56s | 0h 28m 06s / 0h 23m 00s |
| C3 | 463/500 (92.6%) | 453/500 (90.6%) | 91.6% | 3h 13m 38s | 0h 24m 09s / 0h 18m 46s |
| S1-L2 | 395/500 (79.0%) | 358/500 (71.6%) | 75.3% | 2h 31m 47s | 0h 26m 12s / 0h 23m 41s |
| S1-L3 | 379/500 (75.8%) | 322/500 (64.4%) | 70.1% | 3h 25m 38s | 0h 28m 18s / 0h 30m 02s |

Training wall time sums each run's own invocations, including initialization, checkpointing and diagnostics. S1-L3 resumed across three invocations; it has 8,000 unique updates and no replayed updates in the ledger. Evaluation wall times can include profiling and prior attempts. The interrupted duplicate seed-43 attempt is separately archived in the recovery record.

## All 14 registered training trajectories

| Run | Incremental updates | Absolute endpoint | Status |
| --- | ---: | ---: | --- |
| P0-S | 2,000 | 2,000 | complete |
| C1 | 8,000 | 8,000 | complete |
| C2 | 8,000 | 8,000 | complete |
| C3 | 8,000 | 8,000 | complete |
| S1-L2 | 8,000 | 8,000 | complete |
| S1-L3 | 8,000 | 8,000 | complete |
| S2-cont | 6,000 | 14,000 | not started; held by stage prerequisites |
| S2-base | 6,000 | 14,000 | not started; held by stage prerequisites |
| S3-coupled | 8,000 | 22,000 | not started; held by stage prerequisites |
| S3-late | 8,000 | 22,000 | not started; held by stage prerequisites |
| S3-Konly | 14,000 | 22,000 | not started; held by stage prerequisites |
| S3-2stage | 14,000 | 22,000 | not started; held by stage prerequisites |
| F-Long-s1 | 22,000 | 22,000 | not started; held by stage prerequisites |
| F-Long-s2 | 22,000 | 22,000 | not started; held by stage prerequisites |

## Scientific decisions

- **P0-S:** pass; finite, decreasing loss across the 2,000-update smoke run.
- **P0-R:** pass; released teacher reaches 95.0% pooled LIBERO-Long success, within the registered tolerance of 95.2%. Full LIBERO remains deferred.
- **G0:** pass; the width control is within the registered teacher margin.
- **S1 selection:** S1-L2 wins over S1-L3. Pooled L3−L2 gap is −5.2 percentage points; exact McNemar p=0.001158.
- **G1 precheck:** recovery fails by 12.8 points relative to the allowed threshold (79.0% versus 91.8%). The C2 comparison passes by 10.4 points. Official gate recording follows the raw diagnostic.
- **GP, G2, G3 and confirmations:** not completed. C3 versus C2 screening evidence is promising, but it does not establish LoopWAM recovery or permit skipping G1.

## Parallel execution and fairness

One coordinator owns the manifest and leases each allocation exclusively. Each pipeline trains on four GPUs, validates its endpoint, then evaluates immediately. Independent S2 runs, S3 variants and confirmation seeds can run concurrently after their prerequisites pass. The earlier standalone secondary evaluator has been retired.
Every experiment retains global batch 128, the registered optimizer/loss/seed/step schedule and checkpoint parent. LoopWAM and Untied-12 use microbatch 16 × accumulation 2 × four GPUs; other controls use 8 × 4 × four GPUs. ZeRO stage 1 and the frozen latent cache remain unchanged. Longer remaining trajectories are scheduled first within an independent group.
Both nodes report H100 80GB HBM3 and driver 580.105.08. Profiling runs without another workload on the same allocation; validated architecture/budget profiles are shared. The scheduler falls back to one pipeline on future continuation allocations if the authorized helper jobs have expired.

## Verification, provenance and limitations

- 99 relevant tests passed on the deployed version. Live probes verified both allocation routes, four visible GPUs per node, and removal of a cancelled remote Slurm step.
- The handoff changed only the launch route within the frozen executable set. Model, trainer, evaluator and configuration bytes were unchanged; the new operations helpers are separately hashed and checked around every child.
- Existing complete evaluations were validated and retained. The duplicate incomplete primary seed-43 attempt was archived before publishing the completed secondary result.
- Parallel training throughput has not been measured on eligible S2/S3 workloads because G1 is unmet. A twofold full-campaign speedup is not claimed.
- Screening controls have 8,000 updates; future confirmation trajectories would have 22,000. The released teacher has a different training history. These budgets limit architecture-only interpretations.

## Published evidence and navigation

- [Detailed implementation](LoopWAM_v1_implementation.md)
- [Parallel scheduler implementation and handoff](LoopWAM_v1_parallel_training.md)
- [Current campaign results and confidence intervals](current_status/results.md)
- [Results CSV](current_status/results.csv) and [training runtime CSV](current_status/training_runtime.csv)
- [Campaign manifest and decisions](current_status/manifest.json)
- [Snapshot inventory with source paths and SHA-256 hashes](current_status/snapshot_inventory.json)
- [Slurm snapshot](current_status/slurm_status.json)
- [Recorded G1 precheck](current_status/recovery/parallel_g1_precheck.json)

The `current_status/evaluations/` tree contains completed summaries, individual task outcomes, protocol/runtime metadata and available latency profiles. The `training/` tree contains metrics, endpoint timing, checkpoint metadata and invocation ledgers. Recovery audits and verification artifacts are included. Large model weights, distributed checkpoints, datasets, latent caches and rollout videos remain on the cluster; their paths and identities are recorded in the evidence.
