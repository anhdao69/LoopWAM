# LoopWAM v1: allocation handoff and interim evidence

**October 6, 13:58 EDT update:** the [parallel training scheduler is deployed](LoopWAM_v1_parallel_training.md). Six training endpoints are complete; the selected S1-L2 fails the G1 recovery precheck. Later stages are held under the registered gates. The measurements below retain their original timestamps.

Recorded 2026-10-06, 10:54 EDT. This is an interim report; the 14-run campaign has not finished.

## Completed LIBERO-Long evidence

Every student row below uses the stage-end EMA checkpoint, evaluation seed 42, ten tasks and 50 initial states per task. Teacher reproduction uses both registered evaluation seeds. Training wall time includes initialization, checkpointing and diagnostics; evaluation time includes any initial latency profiling.

| Run | Updates | Training wall time | Evaluation wall time | Successes | Success rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher, seed 42 | Released checkpoint | — | 27m 00s | 477/500 | 95.4% |
| Teacher, seed 43 | Released checkpoint | — | 20m 46s | 473/500 | 94.6% |
| P0-S | 2,000 | 40m 50s | 48m 19s | 85/500 | 17.0% |
| C1 Untied-30 | 8,000 | 3h 52m 07s | 28m 58s | 469/500 | 93.8% |
| C2 Untied-12 | 8,000 | 1h 41m 56s | 28m 06s | 343/500 | 68.6% |
| C3 V30/A12 | 8,000 | 3h 13m 38s | 24m 09s | 463/500 | 92.6% |
| S1-L2 | 8,000 | 2h 31m 47s | 26m 12s | 395/500 | 79.0% |

P0-R and G0 passed. C3 exceeds C2 by 24.0 percentage points on the first evaluation seed. S1-L2 exceeds C2 by 10.4 points but trails C1 by 14.8 points. G1 requires the selected LoopWAM candidate to reach C1 minus 2 points and at least C2; S1-L3 is still required before making that decision. These results do not establish that LoopWAM has recovered dense-model performance.

## Two-allocation execution

| Allocation | Node | Work | End time, EDT |
| --- | --- | --- | --- |
| 872933 | evc102 | Main gated campaign; four-GPU training and immediate first-seed evaluation | Oct 7, 04:53:11 |
| 873007 | evc104 | Registered second-seed evaluations of completed training endpoints | Oct 7, 04:53:11 |
| 873269 | Pending | Resume the main campaign after both interactive allocations end | Queue dependent |

The primary controller remains the only manifest writer, training scheduler and gate decision maker. Each training run keeps four GPUs, global batch 128, the original microbatch/accumulation settings, optimizer, data ordering and total update budget. The second allocation supplies additional paired evaluation evidence; it does not add training trajectories or advance a stage before its gate passes.

`scripts/operations/loopwam_secondary_evaluation.py` waits for a registered endpoint with complete timing/state markers and an EMA checkpoint. It runs the existing evaluator on seed 43 in a private staging directory. Before publication it checks all 500 outcomes, all ten task artifacts, per-task initial-state hashes, checkpoint identity, normalization and protocol against the main campaign. Publication uses an exclusive atomic symlink. If the main evaluator already created the destination, its files are retained; duplicate speculative work is recorded rather than overwriting evidence.

Completed second-seed results can be reused by the existing gate logic. Gates still start with seed 42 and request both seeds for the original close-effect conditions. Public tables distinguish individual evaluation seeds and only pool complete paired sets. During a publication race, one duplicate evaluation may be spent; this affects resource cost, not the selected checkpoint or episode set.

The secondary worker records its own source hash, allocation, node, command and event ledger. A detected source change marks its active staging directory untrusted; a later restart quarantines those artifacts. It stops on a terminal campaign state or its own allocation deadline. It resumes existing valid task files after an ordinary interruption.

## Metadata timeout and audited launch-script change

The first October 6 resume stopped before further training because the static `nvidia-smi --query-gpu=name,driver_version` request exceeded the campaign's ten-second timeout. The same read succeeded in 7.51 seconds on evc102 and 4.03 seconds on evc104 when checked independently. Both nodes report NVIDIA H100 80GB HBM3 and driver 580.105.08.

The launch script now invokes `scripts/operations/capture_gpu_metadata.py` with a 45-second timeout before campaign construction. It captures the real device-name/driver response and the four GPU UUIDs. `scripts/operations/bin/nvidia-smi` serves only that exact static request from a cache scoped to the matching allocation, hostname and allocation lifetime. Every other NVIDIA query executes `/usr/bin/nvidia-smi`, including GPU utilization, memory and process queries. No latency measurements or evaluation outcomes are cached by this wrapper.

After the initial operational resume, S1-L3 advanced from 1,754 to **1,896**. A rank-0 SIGUSR1 requested a collective, optimizer-boundary checkpoint. Both orchestration workers were stopped before editing the launch script. An audit verified that **only `scripts/loopwam/run_campaign.sh` changed within the frozen executable set**; model, trainer, evaluator and configuration bytes were unchanged. Original numerical artifacts therefore remain applicable. The audited manifest update and previous manifest are retained in `campaign/recovery/`.

- Previous executable-set SHA-256: `dd2c30054f41546bc3c4a592c73197d3077091ef66e0bd7ac0970d25c0a94028`.
- New executable-set SHA-256: `02741f8e43f6f7c357b566c68857da2a639cec66240381da35a28a4ade5f3753`.
- The inference source hash used by latency profiles is unchanged, so existing measured profiles remain valid.

Placing the capture in the standard launch script also fixes future invocations from the already submitted continuation job. A replacement batch job was considered but not submitted: a scheduler dry run forecast a substantially later start. Job 873269 and its original submission priority were preserved; only its dependency now includes both interactive allocations. No additional allocation was consumed by this handoff.

## Verification and remaining work

The first operations test pass covered 80 campaign/secondary checks. Independent review identified two additional provenance gaps: missing per-task initial-state comparison and reusable staging after detected source drift. Both were reproduced with failing tests and corrected. The final relevant suite passed **87 tests in 9.92 seconds**, including publication races, terminal-state scheduling, exact metadata caching and job/node/expiry rejection. The standard launch script's real four-H100 `--plan` invocation completed and emitted the unchanged 14-run matrix. [Published evidence snapshot](results_snapshot/2026-10-06/comparison.md).

Five training trajectories and their primary evaluations are complete. S1-L3 is resumed from step 1,896; its measured update rate is approximately 1.4–1.5 seconds on four H100s. The remaining 6,104 updates require about 2.4–2.6 hours of update compute, plus initialization, checkpoints and evaluation. A Stage-1 decision is expected this afternoon if execution remains healthy. Later stages and the final best setup remain conditional on the registered gates. The original campaign forecast remains provisional; two-node evaluation overlap does not imply a twofold training speedup.

Sources: live `campaign/manifest.json`, per-run `timing.json`, per-seed `summary.json`, `allocation_handoffs.jsonl`, `recovery/metadata_bootstrap_migration.json`, `.secondary_evaluations/events.jsonl`, and `outputs/loopwam_v1/operations_tests.xml`.
