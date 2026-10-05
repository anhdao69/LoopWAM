#!/usr/bin/env python3
"""Export the measured Phase-0 controlled warmup diagnostic."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='outputs/loopwam_v1/overfit_diagnostic_protocol.json')
    parser.add_argument('--output', default='reports/figures/loopwam_overfit')
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if protocol.get('status') not in ('pass', 'fail'):
        raise ValueError('Validate the completed diagnostic before plotting it')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'pdf.fonttype': 42})
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    metrics = [('loss', 'Combined flow-matching loss'), ('video_fm', 'Future-video loss'),
               ('action_fm/4_4', 'Action loss')]
    for key, label, color in [('baseline', '500-update warmup', '#527ba2'),
                              ('followup', 'Diagnostic: no warmup', '#ce643a')]:
        rows = [json.loads(line) for line in (Path(protocol[key]) / 'metrics.jsonl').read_text().splitlines()]
        for axis, (metric, title) in zip(axes, metrics):
            axis.semilogy([row['global_step'] for row in rows], [row[metric] for row in rows],
                          color=color, label=label, linewidth=1.8)
            axis.set(title=title, xlabel='Optimizer update', ylabel='Logged mean loss')
            axis.grid(True, which='major', alpha=.2)
    axes[0].legend(frameon=False, loc='lower left')
    figure.suptitle('Fixed batch and fixed noise • global batch 128 • LR 5e−5\n'
                   'Implementation diagnostic; all 14 screening trajectories retain 500-update warmup', fontsize=11)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        figure.savefig(output.with_suffix(suffix))
    plt.close(figure)
    print(output.with_suffix('.png'))


if __name__ == '__main__':
    main()
