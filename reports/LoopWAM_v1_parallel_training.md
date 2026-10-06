# LoopWAM v1: concurrent ablation training

Prepared October 6, 2026. This report describes the change from one training allocation plus a secondary evaluator to independent training/evaluation pipelines on both existing four-H100 allocations. The 14-run campaign is still in progress. This report supersedes the scheduling section of `LoopWAM_v1_two_allocations.md`; its earlier measurements remain historical evidence.

## Scientific scope and resource assignment

The user authorized both interactive allocations for training. The coordinator assigns an entire run pipeline to one allocation: four-rank training, endpoint validation, then the registered immediate evaluations. A second independent pipeline can execute on the other allocation. A single experiment still uses four GPUs and global batch 128.

| Stage | Concurrent work | Prerequisites |
| --- | --- | --- |
| Stage 1, if resumed with unfinished endpoints | C2, C3, S1-L2, S1-L3, at most two pipelines | G0 |
| Stage 2 | S2-cont and S2-base | G1 and selected S1 recipe |
| Stage 3 | Two of coupled, late, Konly and 2stage at a time | G2, GP and S2 parent |
| Confirmation | F-Long-s1 and F-Long-s2 | G3 and selected S3 recipe |

Stage 3 schedules the longer remaining trajectories first; Konly and 2stage each need 14,000 incremental updates, whereas coupled and late each need 8,000. This changes dispatch order within an independent group. Parent checkpoints, seeds, losses, budgets and selection rules are unchanged. Stage barriers await all required endpoint evidence before the original gate logic runs. Gate failure stops the campaign; the scheduler does not manufacture permission for later runs.

The initial slots are allocation **872933 / evc102** and **873007 / evc104**, each ending **October 7 at 04:53:11 EDT**. The scheduler verifies current ownership, RUNNING state, one node, four GPUs and at least 16 CPUs. It rejects two slots on the same node and different GPU-model/driver identities. Each actual dispatch is recorded with allocation, command, timestamp and log path in `campaign/parallel_dispatch/events.jsonl`.

## Implementation

`scripts/operations/parallel_campaign.py` subclasses the existing frozen campaign controller. The original `Campaign.run` remains the authority for the experimental sequence and all gates. Training and evaluation delegate to the original implementations with their existing command arguments. Dispatch uses `srun --jobid=<slot> --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:4`, followed by `scripts/operations/run_on_allocation.sh` and the unchanged child argument vector.

A bounded thread pool runs two pipelines. Allocation leases are exclusive and reentrant: a pipeline holds its allocation across training and evaluation, and its nested commands reuse that lease. A single coordinator writes the manifest. Manifest updates, initialization bookkeeping and report generation share a reentrant lock. Separate run directories retain independent optimizer, scheduler, RNG, EMA and distributed checkpoint state. A resumed run first uses its own committed state; a new branch uses the selected parent at the original absolute step boundary.

Microbatch settings remain those of the registered campaign: LoopWAM and Untied-12 use 16 × accumulation 2 × four GPUs; other controls use 8 × accumulation 4 × four GPUs. ZeRO stage 1, optimizer settings, training seed offsets, frozen latent cache, normalization and 500-episode evaluation protocol are unchanged. Concurrent runs do not combine their gradients or increase an experiment's batch size.

Shared latency profiles are protected by a per-architecture/budget lock. The first pipeline obtains a valid profile before releasing that lock for rollouts. A valid local profile from an interrupted evaluation can seed the shared cache atomically. Profile validation retains the existing architecture, budget, hardware, driver, software and inference-code checks. No two GPU workloads from this scheduler run on the same allocation at once, including profiling. Full hardware capture records retain each node's GPU UUIDs.

## Failure, cancellation and continuation

A worker failure sets a shared stop flag. The coordinator stops new dispatches, terminates peer child process groups, drains executor threads, and reports the first substantive error. Bootstrap children are covered by signal handling as well as the main run loop. Constructor failures record a terminal manifest status. External cancellation is an error, not evidence authorizing a new reservation.

The effective deadline is the earliest end time of the selected allocations. Both trainers retain their checkpoint reserve. The scheduler's child supervisor enforces the common deadline as a final limit. Completed endpoints are validated and reused; incomplete training resumes its own distributed state. An error can discard updates after the most recent committed checkpoint; it cannot mark that run complete.

An output-scoped configuration identifies this campaign and the two authorized helper IDs. On a future continuation allocation, the coordinator always includes its current allocation and omits expired helper jobs. It therefore falls back to one pipeline when only one allocation remains. This change makes no new Slurm reservations. Pending job **873269**, its dependency on both interactive jobs, its queue priority and the previously approved bounded continuation chain are retained. Unrelated job **878362** is untouched.

The old secondary evaluation worker is retired during handoff. Its already published seed-43 summaries remain validated inputs. Selection-triggered additional seed evaluations continue through the original campaign logic. The new scheduler does not promise that both allocations will always be training: each pipeline evaluates immediately after training, and stage decisions can require sequential evidence.

## Provenance and handoff

The deployment waits for both current S1-L3 endpoint evaluations to finish before stopping the old orchestration steps. It verifies that both steps have exited and takes the existing campaign and secondary-worker locks before changing the launch route. The previous manifest is archived. Within the frozen executable set, the only intended change is the final dispatch line in `scripts/loopwam/run_campaign.sh`; model, trainer, evaluator and configuration bytes are checked against the prior manifest.

The audited source migration is recorded in `campaign/recovery/parallel_training_migration.json`. The new scheduler additionally pins all operations Python/shell helpers and the metadata wrapper in `parallel_operations_source`. Both scientific and operational identities are verified before and after every child. Changes detected during a child invocation mark its artifacts untrusted.

## Verification

- **99 tests passed in 14.40 seconds** across campaign, secondary evaluation, metadata caching and the new parallel scheduler.
- Twelve new scheduler tests cover scientific prerequisites, concurrent pipelines, exclusive/reentrant slots, serialized manifest writes, unchanged argument boundaries, expired-helper fallback, duplicate-node rejection, immediate evaluation order, failure cancellation, completed-result reuse, shared-profile serialization, bootstrap error recording and interrupted-evaluation profile recovery.
- Independent review identified the constructor signal-cleanup gap and interrupted-profile recovery gap. Both received fixes and regression coverage; follow-up review found no additional important issues in the reviewed paths.
- A live concurrent Slurm probe reached job 872933 on evc102 and job 873007 on evc104, each with `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- A second live probe ran from allocation 872933 and timed out a remote sleeping child on allocation 873007. The remote Slurm step was confirmed absent afterward. Existing evaluation steps remained active.

Test evidence is stored in `outputs/loopwam_v1/parallel_tests.xml`, `parallel_tests.log`, `parallel_transport_probe.json` and `parallel_cleanup_probe.json`. These probes validate routing and cancellation; measured concurrent training throughput still requires eligible training runs after G1.

## Runtime expectations and remaining uncertainty

S1-L3 finished 8,000 updates at 13:26 EDT. Its last resumed segment completed 6,104 updates in 2h 33m 33s including initialization, checkpointing and diagnostics, with warm update time 1.439 seconds. Across its three invocations, measured training wall time is about 3h 25m 38s. Both endpoint evaluation seeds are in progress at the time this report was prepared.

If G1 passes, the first concurrent pair contains 6,000 additional updates per run. S2-cont's fixed-mode training is approximately 2.4 hours of update compute at the measured L3 rate; S2-base's elastic mode is less certain (the short coupled probe was about 1.69 seconds/update, or 2.8 hours). Initialization, checkpoints, diagnostics and each run's three immediate 500-episode evaluations add time. A provisional Stage-2 duration is roughly **3.5–5 hours** with both pipelines available.

The current two allocations have only their remaining shared lifetime. Once they expire, the existing continuation chain supplies one allocation at a time. It would therefore be incorrect to divide the entire remaining campaign estimate by two. This handoff can save roughly **8–13 hours** of remaining elapsed work if gates pass and both nodes stay busy. Queue delays, required second-seed gate evaluations and unmeasured later training modes still prevent a precise full-campaign finish date. Completion and the best setup will be reported from the registered gates and confirmation evidence, not inferred from a partially completed stage.
