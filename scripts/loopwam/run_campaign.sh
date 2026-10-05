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
exec python -m fastwam.loop.campaign "$@"
