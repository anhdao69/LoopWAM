# LoopWAM parallel training on two existing allocations

User authorization: split the ablation training across interactive allocations 872933 and 873007, each with four H100s. The existing 14-run plan and scientific gates remain the authority.

## Design

Use one campaign coordinator and two exclusive allocation slots. Independent runs within a stage execute concurrently; each slot performs a run's training and its required immediate evaluations before taking another run. The original Campaign.run method still orders stages and decides every gate. Stage 2 requires G1; Stage 3 requires G2 and GP; confirmations require G3. Each run keeps global batch128, four ranks, its original microbatch/accumulation, seed, parent checkpoint and optimizer-step boundary.

Implement an operations subclass of the frozen Campaign. It delegates numerical training/evaluation to the existing methods, dispatching commands through srun into an authorized active allocation. A bounded thread executor handles independent pipelines. All manifest mutations and report generation share a reentrant lock within the single coordinator. No second campaign process writes the manifest. Allocation slots prevent GPU oversubscription. Shared architecture/budget profiles are measured once behind a lock before rollouts; other workers reuse the validated profile. Profiles remain isolated from other work on their own node.

Run groups: Stage1 controls/recovery if incomplete; S2-cont/S2-base; all four S3 variants; F-Long-s1/F-Long-s2. Schedule longer remaining training trajectories first within a group. Completed runs are validated/reused. Stage barriers wait for every required pipeline and evaluation. The first worker failure cancels peers and drains children before the coordinator records its final status. Checkpoint deadlines use the earliest selected allocation end so no worker outlives its coordinator. Source drift marks in-flight artifacts untrusted.

An output-scoped operational configuration lists only the two authorized allocation IDs. The current coordinator allocation is always eligible; expired helpers are omitted. A later existing continuation allocation can therefore resume serially when only one slot is available. No additional cluster reservations are added by this change. Job873269 and its bounded continuation policy stay intact.

## Implementation and verification

- [x] Test gate prerequisites, exclusive slot ownership, real concurrent scheduling, failure cleanup, completed-result reuse and profile serialization.
- [x] Add operations scheduler, allocation command wrapper and output-scoped configuration.
- [x] Independently review scheduler; fix important findings and run campaign regression tests.
- [x] Stop old workers at a safe boundary, audit the launch-script-only source migration, and resume the single coordinator.
- [x] Verify actual srun routing to both authorized nodes, live work, reports and GitHub branch.

## Risks to verify

Concurrent manifest/report writes must be serialized. One checkpoint must never be trained by two workers. A worker failure must not leave a remote Slurm step active. Gate failures must launch no subsequent stage. Partial endpoints must resume their own state instead of reforking a parent. Hardware/driver and software identities must agree across nodes for profile reuse. Later single-allocation continuations must remain compatible with the original frozen protocol.

Deployment completed October 6 at 13:57 EDT. Both allocation dispatches verified; G1 precheck fails, so concurrent training remains scientifically ineligible. Full evidence and the live diagnostic status are recorded in `reports/LoopWAM_v1_parallel_training.md`.
