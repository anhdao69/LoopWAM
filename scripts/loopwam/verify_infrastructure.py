#!/usr/bin/env python3
"""Build P0-T evidence from real pytest and four-GPU validation artifacts.

This certifies the fourteen infrastructure requirements, not Phase-0 learning
or teacher reproduction. Run pytest with --junitxml before calling this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


REQUIREMENTS = {
    1: ('conversion equality', ['test_end_to_end_velocity_full_rank_conversion_equality']),
    2: ('target shapes and head maps', ['test_width_slicing_preserves_whole_heads_and_segmented_modulation']),
    3: ('shared storage and compact serialization', ['test_rank_zero_retains_exact_bias_and_shares_base_weights', 'test_checkpoint_export_drops_unused_teacher_backing_storage']),
    4: ('residual composition', ['test_full_rank_fold_reproduces_thirty_layer_composition']),
    5: ('all ten schedules and suffix alignment', ['test_schedule_all_pairs_and_suffix_slots']),
    6: ('future/action branch isolation', ['test_cache_causality_and_future_isolation', 'test_dense_separate_stream_equals_joint_attention']),
    7: ('first-frame cache causality', ['test_cache_causality_and_future_isolation']),
    8: ('first-frame coda equality', ['test_prefix_and_first_frame_coda_at_all_exits']),
    9: ('prefix elasticity', ['test_prefix_and_first_frame_coda_at_all_exits']),
    10: ('distributed deterministic sampling and boundaries', ['test_stage_sampler_reproducible_and_uniform', 'test_sampler_resume_is_independent_of_prefetch_and_microbatch']),
    11: ('teacher/student shift guard', ['test_shift_guard']),
    12: ('teacher normalization and gripper', ['test_teacher_action_and_proprio_normalization_roundtrip']),
    13: ('finite gradient coverage and shallow distributed training', ['test_teacher_receives_identical_noisy_inputs_and_own_context', 'test_checkpointed_gradient_equals_uncheckpointed']),
    14: ('strict checkpoint roundtrip at all budgets', ['test_canonical_save_reload_every_budget']),
}


def artifact(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--junit', required=True)
    parser.add_argument('--validation-root', default='runs/loopwam_validation')
    parser.add_argument('--conversion-audit', default='outputs/loopwam_v1/conversion_audit.json')
    parser.add_argument('--overfit', help='Passing warmup-diagnostic protocol, required to launch the campaign')
    parser.add_argument('--output', default='outputs/loopwam_v1/infrastructure.json')
    args = parser.parse_args()
    cases = list(ET.parse(args.junit).getroot().iter('testcase'))
    require(cases, 'No executed tests in JUnit report')
    require(all(not any(case.find(tag) is not None for tag in ('failure', 'error', 'skipped')) for case in cases),
            'JUnit evidence contains failed, errored or skipped tests')
    names = {case.attrib['name'].split('[', 1)[0] for case in cases}
    rows = []
    for index, (name, tests) in REQUIREMENTS.items():
        require(set(tests) <= names, f'Missing executed test for requirement {index}: {set(tests) - names}')
        rows.append(dict(id=index, requirement=name, tests=tests, status='pass'))
    root = Path(args.validation_root)
    paths = [Path(args.conversion_audit), root/'coupled16/gradient_coverage.json',
             root/'coupled16/timing_initial10.json', root/'coupled16/timing.json',
             root/'fork_fixed12/timing.json', root/'coupled16/open_loop/step_00000011.json']
    audit, coverage, coupled, resumed, forked, panel = [json.loads(path.read_text()) for path in paths]
    require(audit['strict_load'] and all(audit[k]['compact_storage'] and audit[k]['all_finite']
                                       for k in ('video', 'action', 'proprio')), 'Production conversion audit failed')
    require(coverage['all_ranks_complete'] and len(coverage['ranks']) == 4 and
            all(row['complete'] and not row['missing'] and not row['nonfinite'] for row in coverage['ranks']),
            'Four-rank gradient coverage failed')
    require(coupled['complete'] and coupled['mode'] == 'coupled' and coupled['steps'] >= 10 and
            coupled['world_size'] == 4, 'Missing successful production shallow-sampling probe')
    require(resumed['complete'] and resumed['start_step'] == 10 and resumed['global_step'] == 11,
            'Same-output resume did not advance step 10 to 11')
    require(forked['complete'] and forked['start_step'] == 11 and forked['global_step'] == 12 and
            forked['mode'] == 'fixed', 'Explicit fork did not preserve progress and change sampling')
    for field in ('manifest_sha256', 'stats_sha256', 'teacher_identity', 'global_batch', 'seed'):
        require(coupled[field] == resumed[field] == forked[field], f'Fork/resume provenance mismatch: {field}')
    require(panel['weights'] == 'ema' and panel['ema_updates'] == 11 and len(set(panel['window_ids'])) == 20,
            'Missing real EMA open-loop validation')
    record = dict(status='pass', all_14_tests_passed=True, requirements=rows,
                  scope='P0-T infrastructure only; overfit, P0-S and teacher reproduction have separate gates',
                  generated_at=time.time(), executed_pytest_cases=len(cases), junit=artifact(args.junit),
                  production_evidence=[artifact(path) for path in paths],
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    if args.overfit:
        from check_overfit import evaluate_record
        measured = evaluate_record(args.overfit)
        require(measured['status'] == 'pass' and json.loads(Path(args.overfit).read_text()).get('status') == 'pass',
                'Overfit near-zero gate has not passed')
        record['overfit_evidence'] = dict(status='pass', **artifact(args.overfit))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(f'P0-T: all 14 requirements passed; {len(cases)} executed tests; evidence: {output}')


if __name__ == '__main__':
    main()
