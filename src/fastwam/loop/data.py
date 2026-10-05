"""Auditable LIBERO-10 episode holdout using the original FastWAM processing."""
from __future__ import annotations

import hashlib
import functools
import importlib.metadata
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
    if split == 'all':
        return list(range(sum(e['windows'] for e in manifest['episodes'])))
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

    def for_split(self, split):
        return ManifestDataset(self.dataset, self.manifest, split)

    def __getitem__(self, index):
        window_id = self.indices[index]
        # _get deliberately bypasses RobotVideoDataset's random-on-error retry.
        sample = self.dataset._get(window_id)
        episode = self.manifest['episodes'][int(np.searchsorted(self.episode_starts, window_id, side='right')) - 1]
        sample.update(window_id=window_id, episode_id=episode['episode_id'], task_id=episode['task_id'],
                      frame_index=window_id - episode['window_start'], is_augmented=False)
        return sample


def resolved_preprocessing(manifest: dict, *, stats=DEFAULT_STATS, text_cache=DEFAULT_TEXT_CACHE) -> dict:
    """Resolve the exact deterministic configuration used by both cache and loader."""
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(root / 'configs/data/libero_2cam.yaml')
    cfg = OmegaConf.create({'data': config}).data.train
    cfg.dataset_dirs = [manifest['dataset_root']]
    cfg.pretrained_norm_stats = str(Path(stats).resolve())
    cfg.val_set_proportion = 0.0
    cfg.is_training_set = True  # train/validation transforms are checked identical
    cfg.skip_padding_as_possible = False
    cfg.text_embedding_cache_dir = str(Path(text_cache).resolve())
    if (cfg.num_frames != 33 or cfg.global_sample_stride != 1 or cfg.action_video_freq_ratio != 4
            or list(cfg.video_size) != [224, 448] or cfg.concat_multi_camera != 'horizontal'
            or [entry.key for entry in cfg.shape_meta.images] != ['image', 'wrist_image']
            or cfg.processor.norm_default_mode != 'min/max'):
        raise ValueError('LoopWAM requires the teacher libero_2cam preprocessing contract')
    result = OmegaConf.to_container(cfg, resolve=True)
    validate_deterministic_preprocessing(result)
    return result


def validate_deterministic_preprocessing(config: dict) -> None:
    processor = config['processor']
    allowed = [{'_target_': 'fastwam.datasets.lerobot.transforms.image.ToTensor'},
               {'_target_': 'torchvision.transforms.Resize', 'size': [224, 224]}]
    if (processor['train_transforms'] != allowed or processor['val_transforms'] != allowed
            or processor.get('action_state_transforms') is not None
            or processor.get('drop_high_level_prob', 1.0) != 1.0
            or processor.get('use_zh_instruction', False)
            or config.get('override_instruction') is not None
            or not config.get('use_text_embed_cache', False)
            or config.get('context_len') != 128):
        raise ValueError('Latent caching requires deterministic identical train/validation preprocessing and fixed instructions')


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

    kwargs = resolved_preprocessing(manifest, stats=stats, text_cache=text_cache)
    kwargs['is_training_set'] = split != 'validation'
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


CACHE_FORMAT = 'loopwam_latent_cache_v1'
LATENT_SHAPE = (48, 3, 14, 28)
_RECORD_TENSORS = {
    'input_latents': (torch.bfloat16, LATENT_SHAPE),
    'action': (torch.float32, (32, 7)), 'proprio': (torch.float32, (32, 8)),
    'image_is_pad': (torch.bool, (9,)), 'action_is_pad': (torch.bool, (32,)),
    'proprio_is_pad': (torch.bool, (33,)),
}
_RECORD_IDS = ('window_id', 'episode_id', 'task_id', 'frame_index', 'is_augmented')


def default_vae_checkpoint() -> Path:
    return (Path(os.environ.get('DIFFSYNTH_MODEL_BASE_PATH', 'checkpoints')) /
            'Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth').resolve()


def tensor_payload_digest(payload: dict) -> str:
    """Hash tensor bytes, shape, dtype and primitive fields without pickle metadata."""
    digest = hashlib.sha256()
    for key in sorted(payload):
        value = payload[key]
        digest.update(key.encode() + b'\0')
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode() + b'\0')
            digest.update(value.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
        digest.update(b'\0')
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _context_sources(manifest: dict, text_cache: str | Path) -> tuple[dict, dict]:
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    tasks = {e['task_id']: e['task'] for e in manifest['episodes']}
    sources, contexts = {}, {}
    for task_id, task in sorted(tasks.items()):
        prompt = DEFAULT_PROMPT.format(task=task)
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        path = Path(text_cache).resolve() / f'{prompt_sha}.t5_len128.wan22ti2v5b.pt'
        raw = torch.load(path, map_location='cpu', weights_only=True)
        context, mask = raw['context'].clone(), raw['mask'].bool()
        if context.shape != (128, 4096) or context.dtype != torch.bfloat16 or mask.shape != (128,):
            raise ValueError(f'Invalid teacher text context: {path}')
        context[~mask] = 0
        processed = {'context': context, 'context_mask': torch.ones_like(mask), 'prompt': prompt}
        sources[str(task_id)] = {'task': task, 'prompt': prompt, 'prompt_sha256': prompt_sha,
                                 'source_path': str(path), 'source_sha256': file_sha256(path),
                                 'processed_sha256': tensor_payload_digest(processed)}
        contexts[task_id] = processed
    return sources, contexts


def cache_provenance(manifest: dict, *, stats=DEFAULT_STATS, text_cache=DEFAULT_TEXT_CACHE,
                     vae_checkpoint=None) -> tuple[dict, dict]:
    """Compute on rank zero, then broadcast; hashes VAE content once per launch."""
    root = Path(__file__).resolve().parents[3]
    vae_path = Path(vae_checkpoint or default_vae_checkpoint()).resolve(strict=True)
    info = vae_path.stat()
    preprocessing = resolved_preprocessing(manifest, stats=stats, text_cache=text_cache)
    sources, contexts = _context_sources(manifest, text_cache)
    relative_sources = [
        'src/fastwam/datasets/lerobot/base_lerobot_dataset.py',
        'src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py',
        'src/fastwam/datasets/lerobot/lerobot/datasets/video_utils.py',
        'src/fastwam/datasets/lerobot/lerobot/datasets/utils.py',
        'src/fastwam/datasets/lerobot/robot_video_dataset.py',
        'src/fastwam/datasets/lerobot/processors/base_processor.py',
        'src/fastwam/datasets/lerobot/processors/fastwam_processor.py',
        'src/fastwam/datasets/lerobot/transforms/image.py',
        'src/fastwam/datasets/lerobot/transforms/action_state_merger.py',
        'src/fastwam/datasets/lerobot/utils/normalizer.py',
        'src/fastwam/datasets/dataset_utils.py',
        'src/fastwam/models/wan22/wan_video_vae.py',
        'src/fastwam/models/wan22/helpers/loader.py',
        'src/fastwam/models/wan22/helpers/state_dict_converters.py',
        'src/fastwam/loop/data.py',
        'scripts/loopwam/cache_latents.py',
    ]
    # Track the exact encoding method without invalidating the data on unrelated policy edits.
    import ast
    model_source = (root / 'src/fastwam/loop/model.py').read_text()
    encode_node = next(n for n in ast.walk(ast.parse(model_source))
                       if isinstance(n, ast.FunctionDef) and n.name == '_encode_video_latents')
    encode_sha = hashlib.sha256(ast.dump(encode_node).encode()).hexdigest()
    versions = {}
    for package in ('torch', 'torchvision', 'av', 'lerobot', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # Detect changed recordings without reading hundreds of GB of video content.
    dataset_root = Path(manifest['dataset_root'])
    inventory = []
    for directory in ('data', 'videos'):
        for path in sorted((dataset_root / directory).rglob('*')):
            if path.is_file():
                stat = path.stat()
                inventory.append([str(path.relative_to(dataset_root)), stat.st_size, stat.st_mtime_ns])
    identity = {'format': CACHE_FORMAT, 'manifest_sha256': manifest_digest(manifest),
                'expected_windows': sum(e['windows'] for e in manifest['episodes']),
                'stats_sha256': file_sha256(stats), 'dataset_files_sha256': manifest_digest(inventory),
                'dataset_file_count': len(inventory),
                'preprocessing': preprocessing, 'preprocessing_sha256': manifest_digest(preprocessing),
                'source_sha256': {name: file_sha256(root / name) for name in relative_sources},
                'encoder_method_sha256': encode_sha, 'contexts': sources, 'versions': versions,
                'vae': {'path': str(vae_path), 'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
                        'sha256': file_sha256(vae_path), 'scale': 'WanVideoVAE38.mean, WanVideoVAE38.inv_std; source-hashed buffers'},
                'encoding': {'input_dtype': 'torch.float32', 'weights_dtype': 'torch.bfloat16',
                             'autocast_dtype': 'torch.bfloat16', 'latent_dtype': 'torch.bfloat16',
                             'latent_shape': list(LATENT_SHAPE), 'encode_batch_size': 1,
                             'temporal_reset': 'per clip', 'posterior': 'mean, no sampling',
                             'cudnn_benchmark': False, 'cudnn_deterministic': True,
                             'cudnn_allow_tf32': True, 'cuda_matmul_allow_tf32': False},
                'encoding_backend': {'cuda_version': torch.version.cuda,
                                     'cudnn_version': torch.backends.cudnn.version(),
                                     'device_name': torch.cuda.get_device_name() if torch.cuda.is_initialized() else None,
                                     'device_capability': list(torch.cuda.get_device_capability()) if torch.cuda.is_initialized() else None}}
    return {**identity, 'cache_id': manifest_digest(identity)}, contexts


def validate_cache_metadata(metadata: dict, expected: dict) -> None:
    identity = {key: value for key, value in metadata.items() if key != 'cache_id'}
    if metadata.get('format') != CACHE_FORMAT or metadata.get('cache_id') != manifest_digest(identity):
        raise ValueError('Invalid latent cache metadata identity')
    if metadata != expected:
        changed = [key for key in sorted(set(metadata) | set(expected)) if metadata.get(key) != expected.get(key)]
        raise ValueError(f'Latent cache provenance mismatch: {changed}')


def cache_record_path(root: str | Path, window_id: int) -> Path:
    return Path(root) / 'records' / f'{window_id // 1000:03d}' / f'{window_id:08d}.pt'


def window_identity(manifest: dict, window_id: int, episode_starts=None) -> dict:
    starts = episode_starts if episode_starts is not None else [e['window_start'] for e in manifest['episodes']]
    ep_index = int(np.searchsorted(starts, window_id, side='right')) - 1
    if ep_index < 0 or window_id >= manifest['episodes'][ep_index]['window_stop']:
        raise ValueError(f'Window ID outside manifest: {window_id}')
    episode = manifest['episodes'][ep_index]
    return {'window_id': window_id, 'episode_id': episode['episode_id'], 'task_id': episode['task_id'],
            'frame_index': window_id - episode['window_start'], 'is_augmented': False}


def validate_cached_sample(sample: dict, expected_identity: dict) -> None:
    if set(sample) != set(_RECORD_TENSORS) | set(_RECORD_IDS):
        raise ValueError('Cached sample has missing or unexpected fields')
    if any(sample[key] != value for key, value in expected_identity.items()):
        raise ValueError('Cached sample window/episode/task identity mismatch')
    for key, (dtype, shape) in _RECORD_TENSORS.items():
        value = sample[key]
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != shape:
            raise ValueError(f'Cached sample tensor contract mismatch: {key}')
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Cached sample contains nonfinite values: {key}')


def write_cache_record(root: str | Path, metadata: dict, sample: dict) -> None:
    sample = {key: value.detach().cpu().clone(memory_format=torch.contiguous_format) if isinstance(value, torch.Tensor) else value
              for key, value in sample.items()}
    payload = {'cache_id': metadata['cache_id'], 'sample': sample, 'sha256': tensor_payload_digest(sample)}
    atomic_torch_save(payload, cache_record_path(root, sample['window_id']))


def read_cache_record(root: str | Path, metadata: dict, expected_identity: dict) -> dict:
    path = cache_record_path(root, expected_identity['window_id'])
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('cache_id') != metadata['cache_id'] or payload.get('sha256') != tensor_payload_digest(payload['sample']):
        raise ValueError(f'Latent cache record checksum/provenance mismatch: {path}')
    validate_cached_sample(payload['sample'], expected_identity)
    return payload['sample']


def write_cache_contexts(root: str | Path, metadata: dict, contexts: dict) -> None:
    for task_id, sample in contexts.items():
        digest = tensor_payload_digest(sample)
        if digest != metadata['contexts'][str(task_id)]['processed_sha256']:
            raise ValueError(f'Context changed for task {task_id}')
        atomic_torch_save({'cache_id': metadata['cache_id'], 'sample': sample, 'sha256': digest},
                          Path(root) / 'contexts' / f'task_{task_id:02d}.pt')


def validate_complete_cache(root: str | Path, metadata: dict) -> dict:
    root = Path(root)
    complete = json.loads((root / 'complete.json').read_text())
    expected_count = metadata['expected_windows']
    if (complete.get('cache_id') != metadata['cache_id'] or complete.get('completed_windows') != expected_count
            or complete.get('window_ids_sha256') != manifest_digest(list(range(expected_count)))
            or complete.get('metadata_sha256') != file_sha256(root / 'metadata.json')):
        raise ValueError('Latent cache completion manifest is invalid or incomplete')
    return {'path': str(root.resolve()), 'cache_id': metadata['cache_id'],
            'metadata_sha256': complete['metadata_sha256']}


class CachedLatentDataset(Dataset):
    """Read only complete, immutable per-window records; never decode or fall back."""

    def __init__(self, root: str | Path, manifest: dict, split: str, expected_metadata: dict):
        self.root, self.manifest = Path(root), manifest
        self.metadata = json.loads((self.root / 'metadata.json').read_text())
        validate_cache_metadata(self.metadata, expected_metadata)
        if manifest_digest(manifest) != self.metadata['manifest_sha256']:
            raise ValueError('Latent cache split manifest differs')
        self.identity = validate_complete_cache(self.root, self.metadata)
        self.indices = manifest_indices(manifest, split)
        self.episode_starts = np.asarray([e['window_start'] for e in manifest['episodes']])
        self.contexts = {}
        for task_id, source in self.metadata['contexts'].items():
            payload = torch.load(self.root / 'contexts' / f'task_{int(task_id):02d}.pt', map_location='cpu', weights_only=True)
            digest = tensor_payload_digest(payload['sample'])
            if (payload.get('cache_id') != self.metadata['cache_id'] or payload.get('sha256') != digest
                    or source['processed_sha256'] != digest or payload['sample']['prompt'] != source['prompt']):
                raise ValueError(f'Cached context checksum/provenance mismatch for task {task_id}')
            self.contexts[int(task_id)] = payload['sample']

    def __len__(self):
        return len(self.indices)

    def for_split(self, split):
        # Reuse the validated ten contexts, avoiding another raw dataset or decoder.
        result = object.__new__(type(self))
        result.__dict__ = {**self.__dict__, 'indices': manifest_indices(self.manifest, split)}
        return result

    def __getitem__(self, index):
        identity = window_identity(self.manifest, self.indices[index], self.episode_starts)
        sample = read_cache_record(self.root, self.metadata, identity)
        sample.update(self.contexts[identity['task_id']])
        return sample
