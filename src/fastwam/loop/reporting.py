"""Standalone success/latency exports from validated primary evidence only."""
from __future__ import annotations
import csv
from pathlib import Path
from .evaluation import atomic_json


def export_success_latency(root, data):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root/'success_vs_latency.json', data)
    columns = ['run','pair','training_steps','training_seed','loss_recipe','eval_seeds','episodes','success_pct',
               'wilson_low','wilson_high','p50_ms','p90_ms','p99_ms','profile_path','profile_sha256']
    with (root/'success_vs_latency.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data['points'])
    if not data['points']:
        for suffix in ('png','pdf'):
            (root/f'success_vs_latency.{suffix}').unlink(missing_ok=True)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(13,5.5),sharey=True)
    labels = dict(teacher='Released teacher', C1='C1 untied-30 · 8k/L3',
                  C2='C2 untied-12 · 8k/L3', C3='C3 V30/A12 · 8k/L3')
    styles = dict(teacher=('black','*'),C1=('#8c564b','s'),C2=('#d62728','^'),C3=('#2ca02c','D'))
    for ax, run in zip(axes, ('F-Long-s1','F-Long-s2')):
        loop = sorted((p for p in data['points'] if p['run']==run),key=lambda p:p['p50_ms'])
        for point in [p for p in data['points'] if p['run'] in labels]+loop:
            color, marker = styles.get(point['run'],('#1f77b4','o'))
            label = (labels[point['run']] if point['run'] in labels else
                     f"LoopWAM · 22k/{point.get('loss_recipe','unknown')}")
            if point in loop and point is not loop[0]:
                label = None
            elif label:
                label += f" · n={point['episodes']}"
            ax.errorbar(point['p50_ms'],point['success_pct'],
                yerr=[[max(0.,point['success_pct']-point['wilson_low'])],
                      [max(0.,point['wilson_high']-point['success_pct'])]],
                color=color,marker=marker,linestyle='none',capsize=3,markersize=6,label=label)
        for index, point in enumerate(loop):
            ax.annotate(point['pair'],(point['p50_ms'],point['success_pct']),
                        xytext=(5,6 if index%2==0 else -12),textcoords='offset points',fontsize=7)
        seed = loop[0]['training_seed'] if loop else 'pending'
        ax.set_title(f'{run} · training seed {seed}')
        ax.set_xlabel('Measured compiled policy p50 (ms)')
        ax.set_ylim(0,102)
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7,loc='lower right')
    axes[0].set_ylabel('LIBERO-Long success (%) · Wilson 95% CI')
    state = 'COMPLETE' if data['status']=='complete' else 'INCOMPLETE — available measurements only'
    fig.suptitle(f'Standard paused-simulator evaluation · {state}',fontsize=11)
    fig.text(.5,.02,'One trained LoopWAM per panel. Unequal training steps and possibly different loss recipes. '
             'Raw/delayed outcomes excluded.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.95))
    for suffix in ('png','pdf'):
        fig.savefig(root/f'success_vs_latency.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)
