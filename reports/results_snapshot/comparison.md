# LoopWAM initial 14-run campaign

LIBERO-Long selection evidence only. Teacher reproduction is Long only; full-suite reproduction and RTX 4090 profiles are outside this first pass. Phase 4 includes separate delay-injected confirmation evaluations. CUDA component intervals are measured in a separate pass and stored in each latency.json; primary latency remains the uninstrumented wall time.

G2 'well above' means at least 3 pp at both K=1 and K=2. Matched latency means within 5% of measured C2 p50.

| Training run | Architecture | Absolute steps | Status | Micro batch × accumulation × GPUs |
|---|---|---|---|---|
| P0-S | loopwam | 0→2000 | trained | 16 × 2 ×4 |
| C1 | untied30 | 0→8000 | pending | 8 × 4 ×4 |
| C2 | untied12 | 0→8000 | pending | 16 × 2 ×4 |
| C3 | untied_v30a12 | 0→8000 | pending | 8 × 4 ×4 |
| S1-L2 | loopwam | 0→8000 | pending | 16 × 2 ×4 |
| S1-L3 | loopwam | 0→8000 | pending | 16 × 2 ×4 |
| S2-cont | loopwam | 8000→14000 | pending | 16 × 2 ×4 |
| S2-base | loopwam | 8000→14000 | pending | 16 × 2 ×4 |
| S3-coupled | loopwam | 14000→22000 | pending | 16 × 2 ×4 |
| S3-late | loopwam | 14000→22000 | pending | 16 × 2 ×4 |
| S3-Konly | loopwam | 8000→22000 | pending | 16 × 2 ×4 |
| S3-2stage | loopwam | 8000→22000 | pending | 16 × 2 ×4 |
| F-Long-s1 | loopwam | 0→22000 | pending | 16 × 2 ×4 |
| F-Long-s2 | loopwam | 0→22000 | pending | 16 × 2 ×4 |

## Evaluation results

| Run | Budget | Eval seed | Success | Wilson 95% | Episodes | p50 / p90 / p99 ms |
|---|---|---:|---:|---|---:|---|
| P0-S | kv4_ka4 | 42 | 17.0% | 14.0–20.5% | 500 | 62.11 / 62.29 / 62.64 |

## Decisions

- P0-S: **pass**

## Training invocation accounting

| Run | Invocations | Seconds: wall / training / initialization / checkpoint / diagnostics | Warm s/update | History complete |
|---|---:|---|---:|---|
| P0-S | 1 | 2450.1 / 2266.2 / 38.0 / 51.1 / 93.7 | 1.128 | True |

## Auxiliary evaluations (excluded from selection)

Raw-vs-EMA diagnostic: pending. Delay confirmation evaluations: 0/12 complete.

Delay protocol: serial receding-horizon zero-order command hold, quantized to the actual environment control period. Every delay tick counts inside the 700-step cap; the initial 30 settling steps retain the primary convention. This emulates serial control and does not model buffered or asynchronous hardware execution. Decision-wall timing includes preprocessing and action postprocessing; primary policy latency excludes these.

| Run | Variant | Budget | Seed | Success | Wilson 95% | Episodes |
|---|---|---|---:|---:|---|---:|

## Success versus measured latency

Figure status: **incomplete**. Source data: `success_vs_latency.csv` and `success_vs_latency.json`. PDF/PNG are exported when measurements are available. Separate panels retain both confirmation training seeds; 22k confirmations and 8k screening controls have unequal training budgets.

## Runtime evidence

Estimates use matching measured architecture and mode only; missing measurements remain unknown. Compilation and startup may add overhead.

```json
{
  "remaining_training": {
    "P0-S": {
      "optimizer_steps": 0,
      "estimated_training_seconds": 0.0,
      "measured_sources": [
        "P0-S"
      ]
    },
    "C1": {
      "optimizer_steps": 8000,
      "estimated_training_seconds": null,
      "measured_sources": null
    },
    "C2": {
      "optimizer_steps": 8000,
      "estimated_training_seconds": null,
      "measured_sources": null
    },
    "C3": {
      "optimizer_steps": 8000,
      "estimated_training_seconds": null,
      "measured_sources": null
    },
    "S1-L2": {
      "optimizer_steps": 8000,
      "estimated_training_seconds": 9025.619293993088,
      "measured_sources": [
        "P0-S"
      ]
    },
    "S1-L3": {
      "optimizer_steps": 8000,
      "estimated_training_seconds": null,
      "measured_sources": null
    },
    "S2-cont": {
      "optimizer_steps": 6000,
      "estimated_training_seconds": null,
      "measured_sources": null
    },
    "S2-base": {
      "optimizer_steps": 6000,
      "estimated_training_seconds": null,
      "measured_sources": null
    },
    "S3-coupled": {
      "optimizer_steps": 8000,
      "estimated_training_seconds": null,
      "measured_sources": null
    },
    "S3-late": {
      "optimizer_steps": 8000,
      "estimated_training_seconds": null,
      "measured_sources": null
    },
    "S3-Konly": {
      "optimizer_steps": 14000,
      "estimated_training_seconds": null,
      "measured_sources": null
    },
    "S3-2stage": {
      "optimizer_steps": 14000,
      "estimated_training_seconds": null,
      "measured_sources": null
    },
    "F-Long-s1": {
      "optimizer_steps": 22000,
      "estimated_training_seconds": null,
      "measured_sources": null
    },
    "F-Long-s2": {
      "optimizer_steps": 22000,
      "estimated_training_seconds": null,
      "measured_sources": null
    }
  },
  "training_accounting": {
    "P0-S": {
      "version": 1,
      "run_directory": "/lustre/fs1/groups/yshang/an221229/checkpoints/FastWAM/loopwam_v1/campaign/P0-S",
      "invocation_count": 1,
      "invocation_ids": [
        "3c515fac08d549fab2fd2a20f3386a85"
      ],
      "steps_completed": 2000,
      "unique_steps_completed": 2000,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": 0,
      "absolute_step_end": 2000,
      "total_wall_seconds": 2450.13190552406,
      "total_training_seconds": 2266.169369895011,
      "total_initialization_seconds": 38.03032876737416,
      "total_checkpoint_seconds": 51.10823328047991,
      "total_diagnostic_seconds": 93.69212258793414,
      "total_unattributed_seconds": 1.1318509932607412,
      "timing_measured_steps": 1999,
      "timing_measured_seconds": 2255.276621086523,
      "seconds_per_step": 1.128202411749136,
      "samples_per_second": 113.45481862740577,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": true
    },
    "C1": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "C2": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "C3": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "S1-L2": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "S1-L3": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "S2-cont": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "S2-base": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "S3-coupled": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "S3-late": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "S3-Konly": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "S3-2stage": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "F-Long-s1": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    },
    "F-Long-s2": {
      "version": 1,
      "run_directory": null,
      "invocation_count": 0,
      "invocation_ids": [],
      "steps_completed": 0,
      "unique_steps_completed": 0,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": null,
      "absolute_step_end": null,
      "total_wall_seconds": 0,
      "total_training_seconds": 0,
      "total_initialization_seconds": 0,
      "total_checkpoint_seconds": 0,
      "total_diagnostic_seconds": 0,
      "total_unattributed_seconds": 0,
      "timing_measured_steps": 0,
      "timing_measured_seconds": 0,
      "seconds_per_step": null,
      "samples_per_second": null,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": false
    }
  },
  "observed_campaign_training_wall_seconds": 2450.13190552406,
  "evaluation_accounting": {
    "P0-S/eval/kv4_ka4/seed42": {
      "attempts": 1,
      "wall_seconds": 2905.2176628112793,
      "failed_or_interrupted_attempts": 0
    }
  },
  "observed_evaluation_wall_seconds": 2905.2176628112793,
  "measured_eval_500_episode_seconds": [
    2899.2439383752644
  ],
  "eval_500_episode_seconds_mean": 2899.2439383752644,
  "note": "Warm throughput is weighted across measured steps of matching architecture/mode/loss. Run totals count only their own invocations, including replayed work; fork parents are not added again. Partial/legacy history is flagged. No estimate for an unmeasured architecture/mode; gates may stop later work."
}
```
