#!/usr/bin/env bash
# One resumable entrypoint. Use --plan to inspect the exact 14-training matrix.
set -euo pipefail
LOOPWAM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LOOPWAM_ROOT"
source scripts/activate_fastwam.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec python -m fastwam.loop.campaign "$@"
