#!/usr/bin/env python3
"""Render measured conversion diagnostics; no training or policy evaluation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='outputs/loopwam_v1/initialization')
    parser.add_argument('--output', default='reports/figures/loopwam_initialization')
    args = parser.parse_args()
    source = Path(args.input)
    d1, d2 = [json.loads((source / f'd{i}.json').read_text()) for i in (1, 2)]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, stream in enumerate(('video', 'action')):
        for column, (key, title, cmap, maximum) in enumerate((
                ('linear_cka', 'Centered linear CKA', 'viridis', 1),
                ('angular_distance_radians', 'Angular distance (radians)', 'magma', None))):
            axis = axes[row, column]
            values = np.asarray(d1['similarity'][stream][key])
            im = axis.imshow(values, origin='lower', extent=(.5, 30.5, .5, 30.5),
                             vmin=0, vmax=maximum, cmap=cmap, interpolation='nearest')
            axis.set(title=f'{stream.capitalize()}: {title}', xlabel='Teacher layer', ylabel='Teacher layer')
            figure.colorbar(im, ax=axis, fraction=.045)
    names = ('untied30', 'loopwam_r32', 'loopwam_r0')
    labels = ('Untied-30', 'LoopWAM r32', 'Adapters disabled')
    colors = ('#2764a5', '#e07b24', '#747d8c')
    taus = d2['tau_grid']
    for name, label, color in zip(names, labels, colors):
        values = [d2['aggregate'][name][str(tau)]['mse'] for tau in taus]
        axes[0, 2].plot(taus, values, 'o-', label=label, color=color, linewidth=2)
    axes[0, 2].set(title='Initialization action-velocity error', xlabel='Noise time τ', ylabel='MSE against teacher')
    axes[0, 2].legend(frameon=False)
    totals = [d2['aggregate'][name]['all']['mse'] for name in names]
    bars = axes[1, 2].bar(labels, totals, color=colors)
    axes[1, 2].bar_label(bars, labels=[f'{value:.4f}' for value in totals], padding=4)
    axes[1, 2].set(title='Mean over five noise times', ylabel='Action-velocity MSE', ylim=(0, max(totals)*1.2))
    axes[1, 2].tick_params(axis='x', labelrotation=12)
    figure.suptitle(f'LoopWAM initialization — {d1["clips"]:,} training clips for layer similarity; '
                   f'{d2["clips"]} held-out clips for fidelity\n'
                   'Converted weights before training; adapter-disabled row is an inference diagnostic', fontsize=14)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('.png', '.pdf'):
        figure.savefig(output.with_suffix(extension), dpi=180)
    plt.close(figure)
    print(output.with_suffix('.png'))


if __name__ == '__main__':
    main()
