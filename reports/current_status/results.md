# LoopWAM initial 14-run campaign

LIBERO-Long selection evidence only. Teacher reproduction is Long only; full-suite reproduction and RTX 4090 profiles are outside this first pass. Phase 4 includes separate delay-injected confirmation evaluations. CUDA component intervals are measured in a separate pass and stored in each latency.json; primary latency remains the uninstrumented wall time.

G2 'well above' means at least 3 pp at both K=1 and K=2. Matched latency means within 5% of measured C2 p50.

| Training run | Architecture | Absolute steps | Status | Micro batch × accumulation × GPUs |
|---|---|---|---|---|
| P0-S | loopwam | 0→2000 | complete | 16 × 2 ×4 |
| C1 | untied30 | 0→8000 | complete | 8 × 4 ×4 |
| C2 | untied12 | 0→8000 | complete | 16 × 2 ×4 |
| C3 | untied_v30a12 | 0→8000 | complete | 8 × 4 ×4 |
| S1-L2 | loopwam | 0→8000 | complete | 16 × 2 ×4 |
| S1-L3 | loopwam | 0→8000 | complete | 16 × 2 ×4 |
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
| C1 | kv4_ka4 | 42 | 93.8% | 91.3–95.6% | 500 | 41.37 / 44.72 / 44.93 |
| C1 | kv4_ka4 | 43 | 89.2% | 86.2–91.6% | 500 | — / — / — |
| C2 | kv4_ka4 | 42 | 68.6% | 64.4–72.5% | 500 | 21.94 / 21.98 / 22.14 |
| C2 | kv4_ka4 | 43 | 58.4% | 54.0–62.6% | 500 | — / — / — |
| C3 | kv4_ka4 | 42 | 92.6% | 90.0–94.6% | 500 | 25.46 / 25.51 / 25.60 |
| C3 | kv4_ka4 | 43 | 90.6% | 87.7–92.9% | 500 | — / — / — |
| P0-S | kv4_ka4 | 42 | 17.0% | 14.0–20.5% | 500 | 62.11 / 62.29 / 62.64 |
| S1-L2 | kv4_ka4 | 42 | 79.0% | 75.2–82.3% | 500 | 62.11 / 62.29 / 62.64 |
| S1-L2 | kv4_ka4 | 43 | 71.6% | 67.5–75.4% | 500 | — / — / — |
| S1-L3 | kv4_ka4 | 42 | 75.8% | 71.9–79.3% | 500 | 62.11 / 62.29 / 62.64 |
| S1-L3 | kv4_ka4 | 43 | 64.4% | 60.1–68.5% | 500 | — / — / — |
| teacher | kv4_ka4 | 42 | 95.4% | 93.2–96.9% | 500 | 55.43 / 56.61 / 57.28 |
| teacher | kv4_ka4 | 43 | 94.6% | 92.3–96.3% | 500 | — / — / — |

## Decisions

- P0-S: **pass**
- P0-R: **pass**
- G0: **pass**
- S1*: **pass**; S1-L2

## Training invocation accounting

| Run | Invocations | Seconds: wall / training / initialization / checkpoint / diagnostics | Warm s/update | History complete |
|---|---:|---|---:|---|
| P0-S | 1 | 2450.1 / 2266.2 / 38.0 / 51.1 / 93.7 | 1.128 | True |
| C1 | 1 | 13927.2 / 13243.5 / 219.7 / 327.0 / 137.0 | 1.649 | True |
| C2 | 1 | 6115.6 / 5818.1 / 93.3 / 130.0 / 74.0 | 0.727 | True |
| C3 | 1 | 11618.4 / 11120.1 / 116.4 / 285.7 / 95.7 | 1.389 | True |
| S1-L2 | 1 | 9106.5 / 8603.7 / 78.2 / 132.2 / 292.4 | 1.074 | True |
| S1-L3 | 3 | 12338.0 / 11504.1 / 440.0 / 191.3 / 202.6 | 1.434 | True |

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
    "S2-cont": {
      "optimizer_steps": 6000,
      "estimated_training_seconds": 6509.781403637339,
      "measured_sources": [
        "P0-S",
        "S1-L2"
      ]
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
      "run_directory": "/lustre/fs1/groups/yshang/an221229/checkpoints/FastWAM/loopwam_v1/campaign/C1",
      "invocation_count": 1,
      "invocation_ids": [
        "d7a18c1f61634d8ead09fd34c163be06"
      ],
      "steps_completed": 8000,
      "unique_steps_completed": 8000,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": 0,
      "absolute_step_end": 8000,
      "total_wall_seconds": 13927.20726509206,
      "total_training_seconds": 13243.53740270622,
      "total_initialization_seconds": 219.6621150933206,
      "total_checkpoint_seconds": 326.9703048206866,
      "total_diagnostic_seconds": 137.0110669415444,
      "total_unattributed_seconds": 0.026375530287623405,
      "timing_measured_steps": 7999,
      "timing_measured_seconds": 13187.427550399676,
      "seconds_per_step": 1.648634523115349,
      "samples_per_second": 77.64000947773694,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": true
    },
    "C2": {
      "version": 1,
      "run_directory": "/lustre/fs1/groups/yshang/an221229/checkpoints/FastWAM/loopwam_v1/campaign/C2",
      "invocation_count": 1,
      "invocation_ids": [
        "be21c93bb734480c985194ea1d798ab8"
      ],
      "steps_completed": 8000,
      "unique_steps_completed": 8000,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": 0,
      "absolute_step_end": 8000,
      "total_wall_seconds": 6115.620851434767,
      "total_training_seconds": 5818.116827938706,
      "total_initialization_seconds": 93.3010520003736,
      "total_checkpoint_seconds": 129.95711081847548,
      "total_diagnostic_seconds": 74.02589203044772,
      "total_unattributed_seconds": 0.21996864676475525,
      "timing_measured_steps": 7999,
      "timing_measured_seconds": 5812.739960389212,
      "seconds_per_step": 0.7266833304649596,
      "samples_per_second": 176.14274971479082,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": true
    },
    "C3": {
      "version": 1,
      "run_directory": "/lustre/fs1/groups/yshang/an221229/checkpoints/FastWAM/loopwam_v1/campaign/C3",
      "invocation_count": 1,
      "invocation_ids": [
        "7d2752722ad6427798b600ccbc39a101"
      ],
      "steps_completed": 8000,
      "unique_steps_completed": 8000,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": 0,
      "absolute_step_end": 8000,
      "total_wall_seconds": 11618.41342655383,
      "total_training_seconds": 11120.075686236843,
      "total_initialization_seconds": 116.42623336985707,
      "total_checkpoint_seconds": 285.7418887615204,
      "total_diagnostic_seconds": 95.70743777044117,
      "total_unattributed_seconds": 0.4621804151684046,
      "timing_measured_steps": 7999,
      "timing_measured_seconds": 11112.559817882255,
      "seconds_per_step": 1.389243632689368,
      "samples_per_second": 92.136466914886,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": true
    },
    "S1-L2": {
      "version": 1,
      "run_directory": "/lustre/fs1/groups/yshang/an221229/checkpoints/FastWAM/loopwam_v1/campaign/S1-L2",
      "invocation_count": 1,
      "invocation_ids": [
        "0598c341416549e78fdd64c50301026b"
      ],
      "steps_completed": 8000,
      "unique_steps_completed": 8000,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": 0,
      "absolute_step_end": 8000,
      "total_wall_seconds": 9106.508956931531,
      "total_training_seconds": 8603.71184791997,
      "total_initialization_seconds": 78.2351453434676,
      "total_checkpoint_seconds": 132.15067067742348,
      "total_diagnostic_seconds": 292.40395614132285,
      "total_unattributed_seconds": 0.007336849346756935,
      "timing_measured_steps": 7999,
      "timing_measured_seconds": 8592.18912450783,
      "seconds_per_step": 1.0741579103022665,
      "samples_per_second": 119.16311258554246,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": true
    },
    "S1-L3": {
      "version": 1,
      "run_directory": "/lustre/fs1/groups/yshang/an221229/checkpoints/FastWAM/loopwam_v1/campaign/S1-L3",
      "invocation_count": 3,
      "invocation_ids": [
        "62f0104bb8184be3ab37330156c073ee",
        "79bc90a664bc4de7aee97d4f092a33af",
        "ce3f5dd824834257b9b175b4e59add35"
      ],
      "steps_completed": 8000,
      "unique_steps_completed": 8000,
      "replayed_steps": 0,
      "zero_step_invocations": 0,
      "absolute_step_start": 0,
      "absolute_step_end": 8000,
      "total_wall_seconds": 12337.966388251632,
      "total_training_seconds": 11504.06717550382,
      "total_initialization_seconds": 439.9541825912893,
      "total_checkpoint_seconds": 191.29636139795184,
      "total_diagnostic_seconds": 202.6220547258854,
      "total_unattributed_seconds": 0.026614032685756683,
      "timing_measured_steps": 7997,
      "timing_measured_seconds": 11470.991623025388,
      "seconds_per_step": 1.4344118573246702,
      "samples_per_second": 89.2351798030543,
      "partial_invocation_ids": [],
      "has_legacy_records": false,
      "history_complete": true
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
  "observed_campaign_training_wall_seconds": 55555.84879378788,
  "evaluation_accounting": {
    "C2/eval/kv4_ka4/seed43": {
      "attempts": 1,
      "wall_seconds": 1380.3413202762604,
      "failed_or_interrupted_attempts": 0
    },
    "C2/eval/kv4_ka4/seed42": {
      "attempts": 3,
      "wall_seconds": 1689.2138948440552,
      "failed_or_interrupted_attempts": 0
    },
    "C3/eval/kv4_ka4/seed43": {
      "attempts": 1,
      "wall_seconds": 1125.9478478431702,
      "failed_or_interrupted_attempts": 0
    },
    "C3/eval/kv4_ka4/seed42": {
      "attempts": 3,
      "wall_seconds": 1458.5250792503357,
      "failed_or_interrupted_attempts": 0
    },
    "S1-L2/eval/kv4_ka4/seed43": {
      "attempts": 1,
      "wall_seconds": 1421.4275624752045,
      "failed_or_interrupted_attempts": 0
    },
    "S1-L2/eval/kv4_ka4/seed42": {
      "attempts": 5,
      "wall_seconds": 1580.1224570274353,
      "failed_or_interrupted_attempts": 0
    },
    "C1/eval/kv4_ka4/seed43": {
      "attempts": 1,
      "wall_seconds": 1213.2381584644318,
      "failed_or_interrupted_attempts": 0
    },
    "C1/eval/kv4_ka4/seed42": {
      "attempts": 8,
      "wall_seconds": 1754.466633796692,
      "failed_or_interrupted_attempts": 0
    },
    "S1-L3/eval/kv4_ka4/seed43": {
      "attempts": 1,
      "wall_seconds": 1802.1488237380981,
      "failed_or_interrupted_attempts": 0
    },
    "S1-L3/eval/kv4_ka4/seed42": {
      "attempts": 3,
      "wall_seconds": 1698.1304321289062,
      "failed_or_interrupted_attempts": 0
    },
    "teacher/eval/kv4_ka4/seed43": {
      "attempts": 1,
      "wall_seconds": 1245.7443771362305,
      "failed_or_interrupted_attempts": 0
    },
    "teacher/eval/kv4_ka4/seed42": {
      "attempts": 8,
      "wall_seconds": 1640.169153213501,
      "failed_or_interrupted_attempts": 0
    },
    "P0-S/eval/kv4_ka4/seed42": {
      "attempts": 5,
      "wall_seconds": 2915.1913936138153,
      "failed_or_interrupted_attempts": 0
    }
  },
  "observed_evaluation_wall_seconds": 20924.667133808136,
  "measured_eval_500_episode_seconds": [
    1380.331903796643,
    1685.5558422971517,
    1125.9372341260314,
    1448.8211576789618,
    1421.4144310243428,
    1571.9945692513138,
    1213.2254504412413,
    1737.6508393008262,
    1802.077245157212,
    1697.8384537361562,
    1245.7078777514398,
    1619.5733694024384,
    2899.2439383752644
  ],
  "eval_500_episode_seconds_mean": 1603.7978701799248,
  "note": "Warm throughput is weighted across measured steps of matching architecture/mode/loss. Run totals count only their own invocations, including replayed work; fork parents are not added again. Partial/legacy history is flagged. No estimate for an unmeasured architecture/mode; gates may stop later work."
}
```
