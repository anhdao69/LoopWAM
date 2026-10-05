#!/usr/bin/env python3
"""Check the pre-recorded Phase-0 warmup diagnostic against measured artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_record(path):
    protocol = json.loads(Path(path).read_text())
    roots = [Path(protocol[key]) for key in ('baseline', 'followup')]
    arguments, timings, histories, artifacts = [], [], [], []
    for root in roots:
        paths = [root / name for name in ('train_args.json', 'timing.json', 'metrics.jsonl', 'gradient_coverage.json')]
        args, timing, coverage = [json.loads(p.read_text()) for p in (paths[0], paths[1], paths[3])]
        rows = [json.loads(line) for line in paths[2].read_text().splitlines()]
        if (not timing['complete'] or timing['global_step'] != 300 or timing['start_step'] != 0
                or timing['global_batch'] != 128 or timing['world_size'] != 4):
            raise ValueError(f'Incomplete or incompatible diagnostic: {root}')
        if (not coverage['all_ranks_complete'] or len(coverage['ranks']) != 4
                or any(not row['complete'] or row['nonfinite'] or row['missing'] for row in coverage['ranks'])):
            raise ValueError(f'Gradient coverage failed: {root}')
        if (len({row['global_step'] for row in rows}) != len(rows)
                or rows[0]['global_step'] != 1 or rows[-1]['global_step'] != 300):
            raise ValueError(f'Missing/duplicate diagnostic loss history: {root}')
        keys = ('loss', 'video_fm', 'action_fm/4_4', 'grad_norm')
        if any(not math.isfinite(row[key]) or row[key] < 0 for row in rows for key in keys):
            raise ValueError(f'Nonfinite or negative diagnostic metric: {root}')
        arguments.append(args); timings.append(timing); histories.append(rows)
        artifacts.extend(dict(path=str(p.resolve()), sha256=digest(p)) for p in paths)
    for key in ('init', 'seed', 'micro_batch', 'grad_accum', 'max_steps', 'loss', 'mode',
                'overfit_one_batch', 'latent_cache', 'zero_stage', 'gradient_checkpointing', 'stats'):
        if arguments[0].get(key) != arguments[1].get(key):
            raise ValueError(f'Warmup comparison changed another setting: {key}')
    for key in ('manifest_sha256', 'stats_sha256', 'teacher_identity'):
        if timings[0][key] != timings[1][key]:
            raise ValueError(f'Diagnostic provenance changed: {key}')
    if (not arguments[1]['overfit_one_batch'] or arguments[1]['loss'] != 'L2'
            or arguments[1]['mode'] != 'fixed' or arguments[1].get('latent_cache') is not None
            or timings[0].get('warmup_steps', 500) != 500 or timings[1]['warmup_steps'] != 0):
        raise ValueError('Expected the same live-L2 fixed-batch diagnostic with warmup 500 versus 0')
    if (roots[1] / 'train_args.json').stat().st_mtime < protocol['criterion_recorded_before_followup_at']:
        raise ValueError('Follow-up started before its recorded pass criterion')
    if any(not math.isclose(row['lr'], 5e-5, rel_tol=1e-8) for row in histories[1]):
        raise ValueError('The follow-up changed the nominal learning rate')
    for key in ('loss', 'video_fm', 'action_fm/4_4'):
        if not math.isclose(histories[0][0][key], histories[1][0][key], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f'First forward changed before the optimizer update: {key}')
    final = [row for row in histories[1] if row['global_step'] in (280, 290, 300)]
    if len(final) != 3:
        raise ValueError('Need three ten-step averages covering follow-up updates 271 through 300')
    means = {key: sum(row[key] for row in final) / 3 for key in ('loss', 'video_fm', 'action_fm/4_4')}
    rule = protocol['followup_pass_criterion']
    ratio = means['loss'] / histories[1][0]['loss']
    passed = (ratio <= rule['mean_final_30_loss_at_most_fraction_of_initial'] and
              max(means['video_fm'], means['action_fm/4_4']) <= rule['mean_final_30_each_fm_term_at_most'])
    return dict(status='pass' if passed else 'fail', final_30_means=means, final_to_initial_loss_ratio=ratio,
                initial_loss=histories[1][0]['loss'], artifacts=artifacts,
                note='Diagnostic-only warmup change; all screening runs retain 500 warmup updates.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='outputs/loopwam_v1/overfit_diagnostic_protocol.json')
    args = parser.parse_args()
    result = evaluate_record(args.protocol)
    path = Path(args.protocol)
    record = json.loads(path.read_text())
    record.update(status=result['status'], measured_result=result)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}, indent=2))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
