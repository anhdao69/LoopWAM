# LoopWAM: two existing allocations

Authorization: the user requested continuing the campaign with their two four-H100 interactive allocations on 2026-10-06. Jobs 872933 (evc102) and 873007 (evc104) both end 2026-10-07 04:53:11 EDT.

## Execution design

Resume the frozen campaign on 872933 from S1-L3 checkpoint 1,754. Keep its four-GPU/global-128 training protocol, run matrix, parent selection, and scientific gates unchanged. The second allocation precomputes the already registered evaluation seed 43 for completed training endpoints. This overlaps evaluation with training and supplies paired evidence for close decisions. It adds evaluation coverage, not training trajectories or ablations.

The main campaign remains the sole writer of its manifest and sole training scheduler. A small external operations script calls the existing evaluator into private staging directories. It validates all 500 outcomes, task artifacts, protocol, checkpoint and initial-state identities before publishing a completed directory through an atomic, exclusive symlink. If the primary evaluator has already claimed the destination, leave its output untouched. A duplicate race may waste one evaluation; it cannot overwrite results or fail the primary run.

The operations script is outside the frozen model/trainer/evaluator source set; its own SHA-256, arguments, job/node and events are recorded separately. It checks the original frozen source before and after each child. No campaign constructor is called, because that constructor writes allocation metadata before taking its lock. Stop on terminal campaign errors/gates, source changes, or the allocation deadline. Never start a later training stage independently.

Continuation job 873269 now depends on both interactive allocations ending. The existing maximum of eight continuation allocations is unchanged.

## Verification tasks

- [x] Fail tests for scheduling only complete registered endpoints, second-seed selection, terminal stops, strict evidence validation and publication races.
- [x] Implement the independent evaluation worker and run those tests plus existing campaign tests.
- [x] Launch on evc104; verify the actual evaluator is running and the primary checkpoint advances.
- [x] Record live results, runtime and this scheduling decision in the technical report.

## Operational findings and rulings

- A repeated NVIDIA static metadata query timeout stopped the first resume. Capture real static name/driver/UUID metadata once with a 45-second startup allowance and serve only the exact name/driver query for the matching node/job/lifetime. All dynamic NVIDIA calls use the real binary.
- Ruling: place capture in the standard launch script so the original queued continuation retains its priority. Gracefully checkpoint S1-L3 at 1,896 and stop both workers before changing that one frozen shell file; audit and record the old/new identities in the manifest. Model/trainer/evaluator/configuration bytes and inference-profile hashes stay unchanged. Cost: one extra checkpoint/reload; no lost optimizer updates.
- Independent review: fix per-task initial-state provenance and quarantine staging after source drift. Both regression tests were observed failing and then passing. Recheck terminal status immediately before publication. The proposed duplicate continuation entry point was removed because the standard launch-script fix makes it unnecessary.

## Limits

The second allocation accelerates evaluation coverage. Training remains four GPUs per run and sequential in the existing gated controller. Later stages remain conditional on G1/G2/G3; available hardware does not waive a failed scientific gate. Extra seed-43 results are reported separately and pooled only when both seeds are complete; gate selection keeps its original first-seed/close-effect rules.
