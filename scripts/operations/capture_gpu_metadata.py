#!/usr/bin/env python3
"""Capture static device provenance once, before the campaign's short queries."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from fastwam.loop.campaign import resolve_deadline
from fastwam.loop.evaluation import atomic_json


def main():
    path = Path(sys.argv[1])
    job_id = os.environ['SLURM_JOB_ID']
    deadline = resolve_deadline(None)
    queries = []
    for args in (['--query-gpu=name,driver_version', '--format=csv,noheader', '--id=0'],
                 ['--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader']):
        started = time.time()
        result = subprocess.run(['/usr/bin/nvidia-smi', *args], capture_output=True,
                                text=True, check=True, timeout=45)
        if not result.stdout.strip():
            raise ValueError('Empty NVIDIA metadata response')
        queries.append(dict(args=args, seconds=time.time() - started, stdout=result.stdout))
    rows = [line.split(',') for line in queries[1]['stdout'].strip().splitlines()]
    if len(rows) != 4 or len({r[1].strip() for r in rows}) != 4:
        raise ValueError('Expected four distinct GPUs in this allocation')
    first = next(row for row in rows if row[0].strip() == '0')
    if ', '.join(item.strip() for item in first[2:]) != queries[0]['stdout'].strip():
        raise ValueError('Repeated GPU identity queries disagree')
    record = dict(job_id=job_id, node=socket.gethostname(), captured_at=time.time(),
                  deadline=deadline, queries=queries)
    atomic_json(path, record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
