"""Pure, auditable virtual-layer schedules (slot indices are zero based)."""
PAIRS = tuple((kv, ka) for kv in range(1, 5) for ka in range(1, kv + 1))
V30A12_LAYERS = (0, 1, 2, 21, 22, 23, 24, 25, 26, 27, 28, 29)


def validate_budget(kv, ka):
    if not isinstance(kv, int) or not isinstance(ka, int) or (kv, ka) not in PAIRS:
        raise ValueError(f'Expected integer 1 <= Ka <= Kv <= 4, got {(kv, ka)}')


def video_schedule(kv=4, arch='loopwam'):
    validate_budget(kv, kv)
    if arch in ('untied30', 'untied_v30a12', 'untied12'):
        return tuple((i, None, ('dense', i)) for i in range(12 if arch == 'untied12' else 30))
    if arch != 'loopwam':
        raise ValueError(f'Unknown architecture: {arch}')
    return (tuple((i, None, ('pre', i)) for i in range(3))
            + tuple((3+i, r-1, ('core', r, i)) for r in range(1, kv+1) for i in range(6))
            + tuple((9+i, None, ('coda', kv, i)) for i in range(3)))


def action_schedule(kv=4, ka=4, arch='loopwam', alignment='late'):
    validate_budget(kv, ka)
    if arch == 'untied_v30a12':
        return tuple((i, None, ('dense', v)) for i, v in enumerate(V30A12_LAYERS))
    if arch in ('untied30', 'untied12'):
        return video_schedule(kv, arch)
    if alignment not in ('late', 'early', 'final'):
        raise ValueError(f'Unknown alignment: {alignment}')
    core = []
    for r in range(1, ka+1):
        slot = r if alignment == 'early' else kv-ka+r
        video_loop = kv if alignment == 'final' else slot
        core.extend((3+i, slot-1, ('core', video_loop, i)) for i in range(6))
    return (tuple((i, None, ('pre', i)) for i in range(3)) + tuple(core)
            + tuple((9+i, None, ('coda', kv, i)) for i in range(3)))
