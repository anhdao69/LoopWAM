#!/usr/bin/env python
"""Launch with torchrun --nproc-per-node=4 scripts/loopwam/train.py --help."""
from fastwam.loop.trainer import main

if __name__ == '__main__':
    raise SystemExit(main())
