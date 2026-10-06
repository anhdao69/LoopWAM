#!/usr/bin/env bash
# One resumable entrypoint. Use --plan to inspect the exact 14-training matrix.
set -euo pipefail
LOOPWAM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LOOPWAM_ROOT"
source scripts/activate_fastwam.sh
if [[ -z "${CUDA_HOME:-}" ]]; then
  if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/lmod.sh
  fi
  module load cuda/cuda-12.6.0
  if [[ -z "${CUDA_HOME:-}" ]]; then
    LOOPWAM_NVCC="$(command -v nvcc)"
    export CUDA_HOME="$(dirname -- "$(dirname -- "$(readlink -f -- "$LOOPWAM_NVCC")")")"
  fi
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Capture static hardware identity once per allocation before the campaign's
# repeated 10-second metadata queries. Dynamic GPU queries still use NVIDIA's binary.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "$LOOPWAM_ROOT/outputs/loopwam_v1"
  export LOOPWAM_GPU_QUERY_CACHE="$LOOPWAM_ROOT/outputs/loopwam_v1/gpu_metadata_${SLURM_JOB_ID}.json"
  python scripts/operations/capture_gpu_metadata.py "$LOOPWAM_GPU_QUERY_CACHE" \
    >> "$LOOPWAM_ROOT/outputs/loopwam_v1/gpu_metadata_capture_${SLURM_JOB_ID}.log"
  export PATH="$LOOPWAM_ROOT/scripts/operations/bin:$PATH"
fi
exec python scripts/operations/parallel_campaign.py "$@"
