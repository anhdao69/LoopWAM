"""Step-seeded depth sampling and resumable, rank-disjoint window streams."""
from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

MODES = ('fixed', 'coupled', 'decoupled', 'konly', 'three_stage')
COUPLED_PAIRS = ((1, 1), (2, 2), (3, 3))
DECOUPLED_PAIRS = tuple((v, a) for v in range(1, 5) for a in range(1, v + 1) if (v, a) != (4, 4))


def resolve_mode(global_step: int, mode: str, stage2_mode: str = 'coupled', stage3_mode: str = 'decoupled') -> str:
    if global_step < 0:
        raise ValueError('global_step must be nonnegative')
    if mode not in MODES:
        raise ValueError(f'Unknown sampling mode: {mode}')
    if mode != 'three_stage':
        return mode
    if stage2_mode not in MODES[:-1] or stage3_mode not in MODES[:-1]:
        raise ValueError('Stage modes must be fixed/coupled/decoupled/konly')
    return 'fixed' if global_step < 8000 else stage2_mode if global_step < 14000 else stage3_mode


def sample_configuration(global_step: int, mode: str = 'fixed', seed: int = 42) -> tuple[int, int] | None:
    """Use no mutable RNG state: all ranks and resumed runs choose the same pair."""
    mode = resolve_mode(global_step, mode)
    if mode == 'fixed':
        return None
    pool = COUPLED_PAIRS if mode == 'coupled' else ((4, 1), (4, 2), (4, 3)) if mode == 'konly' else DECOUPLED_PAIRS
    key = hashlib.blake2b(f'loopwam:{seed}:{global_step}'.encode(), digest_size=16).digest()
    return random.Random(int.from_bytes(key, 'little')).choice(pool)


def configurations_for_step(global_step: int, mode: str = 'fixed', seed: int = 42) -> tuple[tuple[int, int], ...]:
    sampled = sample_configuration(global_step, mode, seed)
    return ((4, 4),) if sampled is None else ((4, 4), sampled)


class DistributedWindowSampler(Sampler[int]):
    """An endless sequence with an explicitly committed cursor.

    A common permutation is interleaved across ranks. Epoch tails are retained,
    and the cursor advances only after consumption, never during worker prefetch.
    Changing microbatch or accumulation sizes therefore preserves sample order.
    """

    def __init__(self, dataset_size: int, *, rank: int = 0, world_size: int = 1, seed: int = 42, consumed: int = 0):
        if dataset_size <= 0 or world_size < 1 or not 0 <= rank < world_size or consumed < 0:
            raise ValueError('Invalid dataset size, rank, world size or cursor')
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.consumed = int(consumed)

    def __iter__(self) -> Iterator[int]:
        position = self.consumed * self.world_size + self.rank
        cached_epoch, permutation = -1, None
        while True:
            epoch, offset = divmod(position, self.dataset_size)
            if epoch != cached_epoch:
                generator = torch.Generator().manual_seed((self.seed + epoch) % (2**63 - 1))
                permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
                cached_epoch = epoch
            yield permutation[offset]
            position += self.world_size

    def __len__(self) -> int:
        return (self.dataset_size + self.world_size - 1) // self.world_size

    def advance(self, samples: int) -> None:
        if samples < 0:
            raise ValueError('Cannot move the committed sampler cursor backwards')
        self.consumed += samples

    def state_dict(self) -> dict:
        return {key: getattr(self, key) for key in ('dataset_size', 'rank', 'world_size', 'seed', 'consumed')}

    def load_state_dict(self, state: dict) -> None:
        for key in ('dataset_size', 'rank', 'world_size', 'seed'):
            if state[key] != getattr(self, key):
                raise ValueError(f'Sampler resume mismatch for {key}: {state[key]} != {getattr(self, key)}')
        if state['consumed'] < 0:
            raise ValueError('Invalid saved sampler cursor')
        self.consumed = int(state['consumed'])
