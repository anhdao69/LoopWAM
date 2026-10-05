#!/usr/bin/env python
"""Build the complete deterministic LoopWAM per-window VAE cache with torchrun.

Decode batches are independent of canonical singleton encoding. Batched BF16 VAE
kernels differ slightly numerically, so each original window is always encoded
alone, including interrupted refill. No student or teacher policy is loaded.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from fastwam.loop.data import (
    DEFAULT_DATA_ROOT, DEFAULT_STATS, DEFAULT_TEXT_CACHE, _RECORD_IDS, _RECORD_TENSORS,
    build_dataset, build_split_manifest, cache_provenance, file_sha256,
    manifest_digest, read_cache_record, save_manifest, validate_cached_sample,
    validate_cache_metadata, window_identity, write_cache_contexts, write_cache_record,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output', default='checkpoints/loopwam_v1/latent_cache')
    result.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    result.add_argument('--stats', default=DEFAULT_STATS)
    result.add_argument('--text-cache', default=DEFAULT_TEXT_CACHE)
    result.add_argument('--manifest')
    result.add_argument('--split-seed', type=int, default=42)
    result.add_argument('--micro-batch', type=int, default=16, help='Decode/transfer batch; canonical VAE encode batch is always one')
    result.add_argument('--workers', type=int, default=2)
    result.add_argument('--cpu-threads', type=int, default=1)
    result.add_argument('--time-budget-seconds', type=float)
    result.add_argument('--log-every', type=int, default=256, help='Completed windows per rank between progress logs')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.micro_batch <= 0 or args.workers < 0 or args.log_every <= 0:
        raise ValueError('Invalid batch size, workers, or log interval')
    started = time.monotonic()
    rank_device = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(rank_device)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device('cuda', rank_device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    if not dist.is_initialized():
        dist.init_process_group('nccl', device_id=device, timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = None
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    try:
        metadata_message = [None]
        if rank == 0:
            lock = open(root / '.build.lock', 'a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = build_split_manifest(args.data_root, args.split_seed, verify_parquet=True)
            if args.manifest:
                save_manifest(manifest, args.manifest)
            metadata, contexts = cache_provenance(manifest, stats=args.stats, text_cache=args.text_cache)
            if (root / 'metadata.json').exists():
                validate_cache_metadata(json.loads((root / 'metadata.json').read_text()), metadata)
            save_manifest(metadata, root / 'metadata.json')
            save_manifest(manifest, root / 'split_manifest.json')
            write_cache_contexts(root, metadata, contexts)
            # Completeness is re-established from validated original IDs on every invocation.
            (root / 'complete.json').unlink(missing_ok=True)
            metadata_message[0] = (metadata, manifest)
        dist.broadcast_object_list(metadata_message, src=0)
        metadata, manifest = metadata_message[0]
        total = metadata['expected_windows']
        episode_starts = [e['window_start'] for e in manifest['episodes']]
        assigned = list(range(rank, total, world))
        validated, pending = [], []
        for window_id in assigned:
            try:
                read_cache_record(root, metadata, window_identity(manifest, window_id, episode_starts))
                validated.append(window_id)
            except Exception:
                # Refill is explicitly permitted only in this offline builder.
                pending.append(window_id)
        print(json.dumps({'rank': rank, 'validated_existing': len(validated), 'pending': len(pending),
                          'scan_seconds': time.monotonic() - started}), flush=True)
        audit = {}
        if pending and not stop:
            from fastwam.loop.model import LoopWAM, _load_vae
            dataset = build_dataset(manifest, 'all', stats=args.stats, text_cache=args.text_cache)
            loader = DataLoader(Subset(dataset, pending), batch_size=args.micro_batch, shuffle=False,
                                num_workers=args.workers, pin_memory=True,
                                persistent_workers=args.workers > 0,
                                generator=torch.Generator().manual_seed(1729 + rank),
                                **({'prefetch_factor': 2} if args.workers else {}))
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            vae = _load_vae(device)
            encoder = SimpleNamespace(device=device, vae=vae)
            audit = {'vae_scale_dtype': str(vae.scale[0].dtype),
                     'vae_scale': [value.float().cpu().tolist() for value in vae.scale],
                     'cuda_device': torch.cuda.get_device_name(device), 'cuda_version': torch.version.cuda,
                     'cudnn_version': torch.backends.cudnn.version(), 'decode_batch_size': args.micro_batch,
                     'encode_batch_size': 1, 'world_size': world}
            previous_log = len(validated)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for batch in loader:
                    if stop or (args.time_budget_seconds and time.monotonic() - started >= args.time_budget_seconds):
                        break
                    videos = batch['video'].to(device=device, dtype=torch.float32, non_blocking=True)
                    for index, raw_id in enumerate(batch['window_id'].tolist()):
                        identity = window_identity(manifest, raw_id, episode_starts)
                        task = metadata['contexts'][str(identity['task_id'])]
                        if batch['prompt'][index] != task['prompt']:
                            raise ValueError(f'Loader prompt disagrees with immutable task mapping: {raw_id}')
                        # Same frozen encode function as live training. Each clip resets its causal state.
                        latent = LoopWAM._encode_video_latents(encoder, videos[index:index + 1])[0].cpu()
                        sample = {'input_latents': latent}
                        sample.update({key: batch[key][index].clone() for key in _RECORD_TENSORS if key != 'input_latents'})
                        sample.update({key: batch[key][index].item() for key in _RECORD_IDS})
                        validate_cached_sample(sample, identity)
                        write_cache_record(root, metadata, sample)
                        validated.append(raw_id)
                    if len(validated) - previous_log >= args.log_every:
                        elapsed = time.monotonic() - started
                        print(json.dumps({'rank': rank, 'completed': len(validated), 'assigned': len(assigned),
                                          'seconds': elapsed}), flush=True)
                        previous_log = len(validated)
            del loader, dataset, encoder, vae
            torch.cuda.empty_cache()
        summaries = [None] * world
        dist.all_gather_object(summaries, {'ids': validated, 'audit': audit})
        complete = False
        if rank == 0:
            all_ids = [window_id for summary in summaries for window_id in summary['ids']]
            complete = sorted(all_ids) == list(range(total))
            if complete:
                completion = {'cache_id': metadata['cache_id'], 'metadata_sha256': file_sha256(root / 'metadata.json'),
                              'completed_windows': len(all_ids),
                              'window_ids_sha256': manifest_digest(sorted(all_ids)),
                              'seconds_this_invocation': time.monotonic() - started,
                              'encoding_audit': [summary['audit'] for summary in summaries],
                              'bf16_note': 'Canonical singleton encodings; mathematically identical frozen encoder, small numerical differences from live batched BF16 kernels.'}
                save_manifest(completion, root / 'complete.json')
            print(json.dumps({'complete': complete, 'completed_windows': len(all_ids),
                              'expected_windows': total, 'seconds': time.monotonic() - started}), flush=True)
        complete_message = [complete]
        dist.broadcast_object_list(complete_message, src=0)
        return 0 if complete_message[0] else 3
    finally:
        if lock is not None:
            lock.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
