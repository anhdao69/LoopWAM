#!/usr/bin/env bash
# Prepared continuation: submit only after the user approves new allocations.
# Usage after approval: sbatch scripts/loopwam/continue_campaign.sbatch OUTPUT PROOF [MAX_ADDITIONAL]
# MAX_ADDITIONAL defaults to0; ceiling7 means at most8 new allocations including this one.
# Example OUTPUT=outputs/loopwam_v1/campaign PROOF=outputs/loopwam_v1/infrastructure.json
#SBATCH --job-name=loopwam-v1
#SBATCH --account=yshang
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-loopwam-%j.log

set -euo pipefail
LOOPWAM_REPO="${LOOPWAM_REPO:-/lustre/fs1/home/an221229/code/FastWAM}"
cd "$LOOPWAM_REPO"
LOOPWAM_OUTPUT="${1:?Provide the existing campaign output directory}"
LOOPWAM_PROOF="${2:?Provide the infrastructure proof JSON}"
LOOPWAM_ADDITIONAL="${3:-0}"
exec python3 - "$LOOPWAM_OUTPUT" "$LOOPWAM_PROOF" "$LOOPWAM_ADDITIONAL" <<'PY'
import json
import os
from pathlib import Path
import sys
import subprocess
import time

output, proof = map(lambda value: str(Path(value).resolve()), sys.argv[1:3])
additional = int(sys.argv[3])
if not 0 <= additional <= 7:
    raise SystemExit('MAX_ADDITIONAL must be an integer from0 through7')
manifest = json.loads((Path(output) / 'manifest.json').read_text())
arguments = dict(manifest['launch_arguments'], output=output, infrastructure=proof)
# A new allocation has a new EndTime. The wrapper discovers it through scontrol.
for key in ('deadline', 'plan', 'report_only'):
    arguments.pop(key, None)
command = [str(Path('scripts/loopwam/run_campaign.sh').resolve())]
for key, value in arguments.items():
    if value is None or value is False:
        continue
    command.append('--' + key.replace('_', '-'))
    if value is True:
        continue
    command.extend(str(item) for item in (value if isinstance(value, list) else [value]))
started = time.time()
completed = subprocess.run(command, check=False)
manifest = json.loads((Path(output) / 'manifest.json').read_text())
job_id = os.environ.get('SLURM_JOB_ID')
eligible = (completed.returncode != 0 and manifest.get('status') == 'stopped'
            and manifest.get('stop_kind') == 'allocation_end'
            and manifest.get('stopped_at', 0) >= started
            and manifest.get('allocation', {}).get('job_id') == job_id)
if additional and eligible:
    if not job_id:
        raise SystemExit('Cannot create a continuation dependency without SLURM_JOB_ID')
    submit = ['sbatch', '--parsable', '--dependency=afterany:' + job_id,
              str(Path('scripts/loopwam/continue_campaign.sbatch').resolve()),
              output, proof, str(additional - 1)]
    result = subprocess.run(submit, capture_output=True, text=True, check=True)
    submitted = result.stdout.strip().split(';', 1)[0]
    if not submitted.isdigit():
        raise SystemExit('sbatch returned an unrecognized job ID: ' + result.stdout)
    record = dict(previous_job_id=job_id, submitted_job_id=submitted, submitted_at=time.time(),
                  remaining_additional=additional-1, command=submit)
    with (Path(output) / 'continuation_chain.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print('Submitted continuation job ' + submitted, flush=True)
else:
    print('No continuation submitted: status=' + str(manifest.get('status'))
          + ', stop_kind=' + str(manifest.get('stop_kind'))
          + ', remaining=' + str(additional), flush=True)
raise SystemExit(completed.returncode)
PY
