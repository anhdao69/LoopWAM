# Published result snapshot

Latest: [October 6 comparison](2026-10-06/comparison.md), [CSV](2026-10-06/comparison.csv), [source hashes and gate decisions](2026-10-06/snapshot_manifest.json), and [allocation handoff report](../LoopWAM_v1_two_allocations.md). Five runs have completed primary evaluation; S1-L3 is in progress. The second four-H100 allocation supplies registered second-seed evaluations.

## October 5 snapshot

Captured 2026-10-05T17:06:32.721815-04:00. The campaign is ongoing; this is a dated snapshot.

- [Comparison table](comparison.md) and [CSV](comparison.csv).
- [Phase-0 outcomes](phase0_summary.json), [latency](phase0_latency.json), and [training timing](phase0_training_timing.json).
- [Forecast and assumptions](forecast_after_phase0.json) and [evaluation workload](evaluation_workload.json).
- [Pre-launch test results](validation_tests.xml).
- [Source mapping and file hashes](snapshot_manifest.json).

Only P0-S has completed training and evaluation in this snapshot: 85/500 successes (17.0%). Teacher reproduction is in progress, and no final setup has been selected. Cluster-local paths in provenance files refer to the original artifacts. The live campaign writes updated results under `outputs/loopwam_v1/campaign/` on the cluster.
