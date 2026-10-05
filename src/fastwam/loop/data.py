"""Auditable LIBERO-10 episode holdout using the original FastWAM processing."""
from __future__ import annotations

import hashlib
import functools
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_DATA_ROOT = 'data/libero_mujoco3.3.2/libero_10_no_noops_lerobot'
DEFAULT_STATS = 'checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json'
DEFAULT_TEXT_CACHE = 'data/text_embeds_cache/libero'


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_split_manifest(root: str | Path = DEFAULT_DATA_ROOT, seed: int = 42, *, verify_parquet: bool = False) -> dict:
    """Retain every frame-start window, including masked end padding as FastWAM does.

    Window IDs are the original unfiltered LeRobot row IDs. Each episode records
    its half-open interval and explicit IDs are stored for both splits. No window
    crosses an episode: LeRobot clamps future queries and supplies padding masks.
    """
    root = Path(root).resolve()
    info = json.loads((root / 'meta/info.json').read_text())
    tasks = _read_jsonl(root / 'meta/tasks.jsonl')
    raw_episodes = sorted(_read_jsonl(root / 'meta/episodes.jsonl'), key=lambda e: e['episode_index'])
    task_ids = {task['task']: task['task_index'] for task in tasks}
    if len(task_ids) != 10 or info['total_tasks'] != 10:
        raise ValueError('Screening requires exactly the ten LIBERO-10 tasks')
    if len(raw_episodes) != info['total_episodes']:
        raise ValueError('Episode metadata count does not match info.json')
    by_task = defaultdict(list)
    for expected_id, episode in enumerate(raw_episodes):
        if episode['episode_index'] != expected_id or len(episode['tasks']) != 1 or episode['length'] <= 0:
            raise ValueError('Expected contiguous episode IDs and one task per nonempty episode')
        by_task[task_ids[episode['tasks'][0]]].append(expected_id)
    if set(by_task) != set(task_ids.values()):
        raise ValueError('Every LIBERO-10 task must have demonstrations')
    heldout = set()
    for task_id in sorted(by_task):
        ids = by_task[task_id]
        if len(ids) <= 2:
            raise ValueError(f'Task {task_id} needs more than two demonstrations')
        # Each task has an independent seed so changes in another task cannot move its holdout.
        rng = np.random.default_rng(np.random.SeedSequence([seed, task_id]))
        heldout.update(int(i) for i in rng.choice(ids, size=2, replace=False))
    episodes, train_ids, validation_ids = [], [], []
    offset = 0
    for episode in raw_episodes:
        episode_id, length = episode['episode_index'], episode['length']
        task_id = task_ids[episode['tasks'][0]]
        split = 'validation' if episode_id in heldout else 'train'
        row = dict(episode_id=episode_id, task_id=task_id, task=episode['tasks'][0], length=length,
                   split=split, window_start=offset, window_stop=offset + length,
                   windows=length, unpadded_windows=max(0, length - 32))
        if verify_parquet:
            import pyarrow.parquet as pq
            path = root / info['data_path'].format(episode_chunk=episode_id // info['chunks_size'], episode_index=episode_id)
            table = pq.read_table(path, columns=['index', 'frame_index', 'episode_index', 'task_index'])
            expected = {'index': np.arange(offset, offset + length), 'frame_index': np.arange(length),
                        'episode_index': np.full(length, episode_id), 'task_index': np.full(length, task_id)}
            for key, values in expected.items():
                if not np.array_equal(table[key].to_numpy(), values):
                    raise ValueError(f'Parquet {path} disagrees with manifest for {key}')
        episodes.append(row)
        (validation_ids if split == 'validation' else train_ids).extend(range(offset, offset + length))
        offset += length
    if offset != info['total_frames']:
        raise ValueError('Episode lengths do not sum to total_frames')
    counts = {}
    for split in ('train', 'validation'):
        selected = [e for e in episodes if e['split'] == split]
        counts[split] = {'episodes': len(selected), 'windows': sum(e['windows'] for e in selected),
                         'unpadded_windows': sum(e['unpadded_windows'] for e in selected)}
    return {'version': 1, 'dataset_root': str(root), 'seed': seed, 'holdout_per_task': 2,
            'window_definition': 'one start per original frame; 33 observations, stride 1, 32 actions; end padding masked',
            'window_id_definition': 'original unfiltered LeRobot global index; local frame = window_id - episode.window_start',
            'metadata_sha256': {name: file_sha256(root / 'meta' / name) for name in ('info.json', 'episodes.jsonl', 'tasks.jsonl')},
            'counts': counts, 'episodes': episodes, 'train_window_ids': train_ids, 'validation_window_ids': validation_ids}


def manifest_indices(manifest: dict, split: str) -> list[int]:
    if split not in ('train', 'validation'):
        raise ValueError(f'Unknown split: {split}')
    return manifest[f'{split}_window_ids']


def save_manifest(manifest: dict, path: str | Path) -> None:
    """An existing split is immutable; concurrent writers must agree exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f'Split manifest is immutable and differs: {path}')
        return
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    try:
        os.link(temporary, path)
    except FileExistsError:
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f'Split manifest is immutable and differs: {path}')
    finally:
        temporary.unlink(missing_ok=True)


class ManifestDataset(Dataset):
    """Select original rows without filtering/reindexing the underlying episodes."""

    def __init__(self, dataset: Dataset, manifest: dict, split: str):
        self.dataset = dataset
        self.manifest = manifest
        self.indices = manifest_indices(manifest, split)
        self.episode_starts = np.asarray([e['window_start'] for e in manifest['episodes']])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        window_id = self.indices[index]
        # _get deliberately bypasses RobotVideoDataset's random-on-error retry.
        sample = self.dataset._get(window_id)
        episode = self.manifest['episodes'][int(np.searchsorted(self.episode_starts, window_id, side='right')) - 1]
        sample.update(window_id=window_id, episode_id=episode['episode_id'], task_id=episode['task_id'],
                      frame_index=window_id - episode['window_start'], is_augmented=False)
        return sample


def build_dataset(manifest: dict, split: str, *, stats: str | Path = DEFAULT_STATS,
                  text_cache: str | Path = DEFAULT_TEXT_CACHE) -> ManifestDataset:
    """Use teacher normalization and identical resize/normalization/camera order.

    Images are decoded only at offsets 0,4,...,32; resizing is per image and is
    identical to decoding 33 images and then discarding the unused 24.
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset

    class StrictBaseLerobotDataset(BaseLerobotDataset):
        presample_images = True

        def __getitem__(self, index):
            # No recovery via a different episode: surface the original read error.
            raw = self._split_lerobot_sample(self.multi_dataset[index])
            sample = {'idx': index, 'task': raw['task'],
                      'action': {m['key']: self._get_action(m, raw) for m in self.action_meta},
                      'state': {m['key']: self._get_state(m, raw) for m in self.state_meta},
                      'images': {m['key']: self._get_image(m, raw) for m in self.image_meta},
                      'action_is_pad': raw[f"{self.action_meta[0]['lerobot_key']}_is_pad"],
                      'state_is_pad': raw[f"{self.state_meta[0]['lerobot_key']}_is_pad"],
                      'image_is_pad': raw[f"{self.image_meta[0]['lerobot_key']}_is_pad"]}
            return self.processor.preprocess(sample)

    class StrictRobotVideoDataset(RobotVideoDataset):
        base_dataset_cls = StrictBaseLerobotDataset

    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(root / 'configs/data/libero_2cam.yaml')
    cfg = OmegaConf.create({'data': config}).data.train
    cfg.dataset_dirs = [manifest['dataset_root']]
    cfg.pretrained_norm_stats = str(Path(stats).resolve())
    cfg.val_set_proportion = 0.0
    cfg.is_training_set = split == 'train'
    cfg.skip_padding_as_possible = False
    if (cfg.num_frames != 33 or cfg.global_sample_stride != 1 or cfg.action_video_freq_ratio != 4
            or list(cfg.video_size) != [224, 448] or cfg.concat_multi_camera != 'horizontal'
            or [entry.key for entry in cfg.shape_meta.images] != ['image', 'wrist_image']
            or cfg.processor.norm_default_mode != 'min/max'):
        raise ValueError('LoopWAM requires the teacher libero_2cam preprocessing contract')
    cfg.text_embedding_cache_dir = str(Path(text_cache).resolve())
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    kwargs.pop('_target_')
    kwargs['shape_meta'] = OmegaConf.create(kwargs['shape_meta'])
    kwargs['processor'] = instantiate(kwargs['processor'])
    dataset = StrictRobotVideoDataset(**kwargs)
    # Ten fixed instructions: avoid deserializing a ~1 MB T5 tensor per window.
    dataset._get_cached_text_context = functools.lru_cache(maxsize=16)(dataset._get_cached_text_context)
    if len(dataset) != sum(e['windows'] for e in manifest['episodes']):
        raise ValueError('Loaded dataset length does not match the immutable manifest')
    # Verify the actual concatenated episode boundaries, not only metadata counts.
    actual = dataset.lerobot_dataset.episode_data_index
    if actual['from'].tolist() != [e['window_start'] for e in manifest['episodes']] or actual['to'].tolist() != [e['window_stop'] for e in manifest['episodes']]:
        raise ValueError('Loaded dataset episode indexing does not match manifest')
    return ManifestDataset(dataset, manifest, split)
