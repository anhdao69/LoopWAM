#!/usr/bin/env bash
# Execute an unchanged training/evaluation command in its assigned Slurm slot.
set -euo pipefail
LOOPWAM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LOOPWAM_ROOT"
source scripts/activate_fastwam.sh
if [[ -z "${CUDA_HOME:-}" ]]; then
  if ! type module >/dev/null 2>&1; then source /etc/profile.d/lmod.sh; fi
  module load cuda/cuda-12.6.0
  if [[ -z "${CUDA_HOME:-}" ]]; then
    LOOPWAM_NVCC="$(command -v nvcc)"
    export CUDA_HOME="$(dirname -- "$(dirname -- "$(readlink -f -- "$LOOPWAM_NVCC")")")"
  fi
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export LOOPWAM_GPU_QUERY_CACHE="$LOOPWAM_ROOT/outputs/loopwam_v1/gpu_metadata_${SLURM_JOB_ID:?}.json"
if ! python - "$LOOPWAM_GPU_QUERY_CACHE" <<'PY'
import json, os, socket, sys, time
from pathlib import Path
try:
    d=json.loads(Path(sys.argv[1]).read_text())
    valid=(d['job_id']==os.environ['SLURM_JOB_ID'] and d['node']==socket.gethostname()
           and d['captured_at'] <= time.time() < d['deadline'] and len(d['queries'])==2)
except (OSError, ValueError, KeyError):
    valid=False
raise SystemExit(0 if valid else 1)
PY
then
  python scripts/operations/capture_gpu_metadata.py "$LOOPWAM_GPU_QUERY_CACHE"
fi
export PATH="$LOOPWAM_ROOT/scripts/operations/bin:$PATH"
exec "$@"
