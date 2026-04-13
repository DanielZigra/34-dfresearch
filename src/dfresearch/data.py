"""
Dataset download, caching, and DataLoader utilities for dfresearch.

Supports image, image_human, video, and audio modalities.
Downloads from HuggingFace Hub using gasbench's download engine, with local caching.
Supports concurrent downloads across multiple datasets.

Dataset configs are pulled directly from BitMind-AI/gasbench at runtime,
so they stay in sync with the competition benchmark datasets automatically.
"""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

# Load .env if present (HF_TOKEN, CUDA_VISIBLE_DEVICES, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import requests
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dfresearch.transforms import resize_image, apply_random_augmentations, random_horizontal_flip

PROJECT_ROOT = Path(__file__).parent.parent.parent

GASBENCH_RAW_URL = (
    "https://raw.githubusercontent.com/BitMind-AI/gasbench/main"
    "/src/gasbench/dataset/configs"
)


def get_cache_dir() -> Path:
    """Root directory for datasets and mirrored gasbench YAML configs (like gasbench ``--cache-dir``)."""
    return Path(os.environ.get("DFRESEARCH_CACHE", "~/.cache/dfresearch")).expanduser()


def set_cache_dir(path: str | Path | None) -> None:
    """Set cache root via ``DFRESEARCH_CACHE`` (call before using loaders if using ``--cache-dir``)."""
    if path is None:
        return
    os.environ["DFRESEARCH_CACHE"] = str(Path(path).expanduser())


def _gasbench_config_cache_dir() -> Path:
    return get_cache_dir() / "gasbench_configs"

# Local override file: place custom datasets here and they get merged with gasbench.
# File format: same YAML as gasbench configs. See datasets.yaml.example.
LOCAL_DATASETS_DIR = PROJECT_ROOT / "datasets"

LABEL_MAP = {"real": 0, "synthetic": 1, "semisynthetic": 1}

# Default per-dataset sample cap when downloading for training.
# gasbench configs don't have max_samples — this is a training-only setting.
DEFAULT_SAMPLES_PER_DATASET = 500


# ---------------------------------------------------------------------------
# Config loading — synced from gasbench
# ---------------------------------------------------------------------------

def _load_gasbench_config(modality: str, force_refresh: bool = False) -> dict:
    """Fetch a single modality config from gasbench GitHub, with local cache."""
    filename = f"{modality}_datasets.yaml"
    cfg_root = _gasbench_config_cache_dir()
    cached_path = cfg_root / filename

    if cached_path.exists() and not force_refresh:
        with open(cached_path) as f:
            return yaml.safe_load(f)

    url = f"{GASBENCH_RAW_URL}/{filename}"
    print(f"  Fetching {modality} dataset config from gasbench...")

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        config = yaml.safe_load(resp.text)

        cfg_root.mkdir(parents=True, exist_ok=True)
        cached_path.write_text(resp.text)
        print(f"  Cached gasbench {modality} config ({len(config.get('datasets', []))} datasets)")
        return config

    except Exception as e:
        if cached_path.exists():
            print(f"  Warning: Could not fetch from gasbench ({e}), using cached config")
            with open(cached_path) as f:
                return yaml.safe_load(f)
        raise RuntimeError(
            f"Cannot load {modality} dataset config: {e}\n"
            f"Ensure you have internet access, or manually place the config at:\n"
            f"  {cached_path}"
        ) from e


def _load_local_datasets(modality: str) -> list[dict]:
    """
    Load custom datasets from the local datasets/ directory.

    Looks for datasets/{modality}.yaml (e.g. datasets/image.yaml).
    Returns empty list if no local overrides exist.
    """
    local_path = LOCAL_DATASETS_DIR / f"{modality}.yaml"
    if not local_path.exists():
        return []

    with open(local_path) as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", [])
    if datasets:
        print(f"  Loaded {len(datasets)} custom {modality} datasets from {local_path}")
    return datasets


def load_dataset_config(modality: str, force_refresh: bool = False) -> dict:
    """
    Load merged dataset config: gasbench (upstream) + local overrides.

    Gasbench configs are the source of truth for competition datasets.
    Local datasets from datasets/{modality}.yaml are appended on top,
    letting developers add their own training data without forking gasbench.

    Local datasets with the same 'name' as a gasbench dataset replace
    the upstream entry (useful for overriding settings).
    """
    config = _load_gasbench_config(modality, force_refresh=force_refresh)
    local_datasets = _load_local_datasets(modality)

    if local_datasets:
        existing_names = {d["name"] for d in config.get("datasets", [])}
        for ds in local_datasets:
            if ds["name"] in existing_names:
                config["datasets"] = [
                    ds if d["name"] == ds["name"] else d
                    for d in config["datasets"]
                ]
            else:
                config["datasets"].append(ds)

    return config


def refresh_configs():
    """Force re-fetch all configs from gasbench."""
    for modality in ("image", "image_human", "video", "audio"):
        _load_gasbench_config(modality, force_refresh=True)
    print("All configs refreshed from gasbench.")


# ---------------------------------------------------------------------------
# Download engine
# ---------------------------------------------------------------------------

def download_and_cache_dataset(
    dataset_name: str,
    hf_path: str,
    modality: str,
    max_samples: int = DEFAULT_SAMPLES_PER_DATASET,
    media_type: str = "synthetic",
    source_format: Optional[str] = None,
    data_columns: Optional[list[str]] = None,
    include_paths: Optional[list[str]] = None,
    exclude_paths: Optional[list[str]] = None,
    media_per_archive: int = 100,
    archives_per_dataset: int = 5,
) -> Path:
    """
    Download a dataset from HuggingFace and cache locally.

    Uses gasbench's download_and_extract for robust handling of all
    source formats (parquet, zip, tar, raw media files).
    """
    cache_path = get_cache_dir() / "datasets" / dataset_name / "samples"
    marker = cache_path / ".download_complete"

    if marker.exists():
        return cache_path

    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        from gasbench.dataset.config import BenchmarkDatasetConfig
        from gasbench.dataset.download import download_and_extract

        config = BenchmarkDatasetConfig(
            name=dataset_name,
            path=hf_path,
            modality=modality,
            media_type=media_type,
            source_format=source_format or "",
            data_columns=data_columns,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            media_per_archive=media_per_archive,
            archives_per_dataset=archives_per_dataset,
        )

        hf_token = os.environ.get("HF_TOKEN")

        eff_archives = archives_per_dataset
        eff_media = media_per_archive
        if eff_archives == -1 or eff_archives > max_samples:
            eff_archives = max(1, max_samples)
        if eff_media == -1 or eff_media > max_samples:
            eff_media = max_samples

        saved = 0
        for sample in download_and_extract(
            dataset=config,
            media_per_archive=eff_media,
            archives_per_dataset=eff_archives,
            hf_token=hf_token,
        ):
            if saved >= max_samples:
                break
            try:
                if modality in ("image", "image_human"):
                    img = sample.get("image")
                    if img is None:
                        continue
                    img.convert("RGB").save(cache_path / f"{saved:06d}.png")
                elif modality == "video":
                    video = sample.get("video_bytes")
                    if video is None:
                        continue
                    (cache_path / f"{saved:06d}.mp4").write_bytes(video)
                elif modality == "audio":
                    audio = sample.get("audio_bytes")
                    if audio is None:
                        continue
                    (cache_path / f"{saved:06d}.wav").write_bytes(audio)
                saved += 1
            except Exception:
                continue

        marker.touch()
        print(f"  Cached {saved} samples for {dataset_name}")
    except Exception as e:
        print(f"  Warning: Could not download {dataset_name}: {e}")

    return cache_path


def download_all_datasets(
    modality: str,
    max_workers: int = 4,
    max_samples_per_dataset: int = DEFAULT_SAMPLES_PER_DATASET,
    progress: bool = True,
) -> dict[str, int]:
    """
    Download all datasets for a modality using concurrent workers.

    Uses the gasbench dataset config as the source of truth.
    """
    config = load_dataset_config(modality)
    datasets_cfg = config["datasets"]

    if progress:
        print(f"\nDownloading {len(datasets_cfg)} {modality} datasets "
              f"(up to {max_workers} concurrent, max {max_samples_per_dataset}/dataset)...")
        print("=" * 60)

    results = {}

    def _download_one(ds_cfg):
        name = ds_cfg["name"]
        t0 = time.time()
        cache_path = download_and_cache_dataset(
            dataset_name=name,
            hf_path=ds_cfg["path"],
            modality=modality,
            max_samples=max_samples_per_dataset,
            media_type=ds_cfg.get("media_type", "synthetic"),
            source_format=ds_cfg.get("source_format"),
            data_columns=ds_cfg.get("data_columns"),
            include_paths=ds_cfg.get("include_paths"),
            exclude_paths=ds_cfg.get("exclude_paths"),
            media_per_archive=ds_cfg.get("media_per_archive", 100),
            archives_per_dataset=ds_cfg.get("archives_per_dataset", 5),
        )
        elapsed = time.time() - t0
        n = _count_media_files(cache_path, modality)
        return name, n, elapsed

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, cfg): cfg["name"] for cfg in datasets_cfg}

        for future in as_completed(futures):
            name = futures[future]
            try:
                name, count, elapsed = future.result()
                results[name] = count
                if progress:
                    print(f"  [{name}] {count} samples in {elapsed:.1f}s")
            except Exception as e:
                results[name] = 0
                if progress:
                    print(f"  [{name}] FAILED: {e}")

    if progress:
        total = sum(results.values())
        print(f"\n{'=' * 60}")
        print(f"Total: {total} {modality} samples across {len(results)} datasets")

    return results


def _count_media_files(cache_path: Path, modality: str) -> int:
    """Count media files in a cache directory."""
    if not cache_path.exists():
        return 0
    extensions = _modality_extensions(modality)
    return sum(1 for f in cache_path.iterdir()
               if f.suffix.lower() in extensions and not f.name.startswith("."))


def _modality_extensions(modality: str) -> set[str]:
    return {
        "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
        "image_human": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
        "video": {".mp4", ".avi", ".mkv", ".mov", ".webm"},
        "audio": {".wav", ".mp3", ".flac", ".npz", ".ogg"},
    }[modality]




# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class ImageDeepfakeDataset(Dataset):
    def __init__(self, samples, target_size=(224, 224), augment_level=0):
        self.samples = samples
        self.target_size = target_size
        self.augment_level = augment_level

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception:
            img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        img = resize_image(img, self.target_size)
        if self.augment_level > 0:
            img = apply_random_augmentations(img, level=self.augment_level)
            img = random_horizontal_flip(img)
        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return tensor, label


class VideoDeepfakeDataset(Dataset):
    def __init__(self, samples, target_size=(224, 224), num_frames=16, augment_level=0):
        self.samples = samples
        self.target_size = target_size
        self.num_frames = num_frames
        self.augment_level = augment_level

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(str(path), ctx=cpu(0))
            total = len(vr)
            if total >= self.num_frames:
                indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
            else:
                indices = list(range(total)) + [total - 1] * (self.num_frames - total)
            frames = vr.get_batch(indices).asnumpy()
        except Exception:
            frames = np.zeros(
                (self.num_frames, self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        processed = []
        for i in range(min(frames.shape[0], self.num_frames)):
            frame = resize_image(frames[i], self.target_size)
            if self.augment_level > 0:
                frame = apply_random_augmentations(frame, level=self.augment_level)
            processed.append(frame)
        while len(processed) < self.num_frames:
            processed.append(processed[-1] if processed else
                             np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8))
        frames_arr = np.stack(processed)
        tensor = torch.from_numpy(np.ascontiguousarray(frames_arr)).permute(0, 3, 1, 2)
        return tensor, label


class AudioDeepfakeDataset(Dataset):
    def __init__(self, samples, sample_rate=16000, duration_seconds=6.0):
        self.samples = samples
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration_seconds)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            if path.suffix == ".npz":
                data = np.load(path)
                waveform = data["audio"].astype(np.float32)
                sr = int(data.get("sr", self.sample_rate))
            else:
                import soundfile as sf
                waveform, sr = sf.read(str(path), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=-1)
            if sr != self.sample_rate:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
        except Exception:
            waveform = np.zeros(self.target_length, dtype=np.float32)
        if len(waveform) > self.target_length:
            start = random.randint(0, len(waveform) - self.target_length)
            waveform = waveform[start : start + self.target_length]
        elif len(waveform) < self.target_length:
            pad = self.target_length - len(waveform)
            waveform = np.pad(waveform, (0, pad), mode="constant")
        waveform = np.clip(waveform, -1.0, 1.0)
        tensor = torch.from_numpy(waveform)
        return tensor, label


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def gather_samples(
    modality: str,
    split: str = "train",
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> list[tuple[Path, int]]:
    """
    Gather all cached samples for a modality with labels.

    Uses gasbench configs as the dataset registry.
    Applies class balancing and reproducible train/val splits.
    """
    config = load_dataset_config(modality)
    real_samples = []
    fake_samples = []

    extensions = _modality_extensions(modality)

    for ds_cfg in config["datasets"]:
        media_type = ds_cfg["media_type"]
        label = LABEL_MAP.get(media_type, 1)
        cache_path = get_cache_dir() / "datasets" / ds_cfg["name"] / "samples"

        if not cache_path.exists():
            continue

        files = sorted(
            f for f in cache_path.iterdir()
            if f.suffix.lower() in extensions and not f.name.startswith(".")
        )

        for f in files:
            if label == 0:
                real_samples.append((f, label))
            else:
                fake_samples.append((f, label))

    rng = random.Random(seed)
    rng.shuffle(real_samples)
    rng.shuffle(fake_samples)

    if max_per_class is not None:
        real_samples = real_samples[:max_per_class]
        fake_samples = fake_samples[:max_per_class]

    def _split(samples):
        n = len(samples)
        split_idx = int(0.8 * n)
        return samples[:split_idx], samples[split_idx:]

    real_train, real_val = _split(real_samples)
    fake_train, fake_val = _split(fake_samples)

    if split == "train":
        out = real_train + fake_train
    elif split == "val":
        out = real_val + fake_val
    else:
        out = real_samples + fake_samples

    rng.shuffle(out)

    print(f"Total {modality} samples: {len(out)}")

    if len(out) == 0:
        print(f"WARNING: No {modality} samples found for split='{split}'. "
              f"Run `uv run prepare.py --modality {modality}` first.")

    return out


def make_dataloader(
    modality: str,
    split: str = "train",
    batch_size: int = 32,
    target_size: tuple[int, int] = (224, 224),
    num_frames: int = 16,
    augment_level: int = 1,
    max_per_class: Optional[int] = None,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for the specified modality and split."""
    samples = gather_samples(modality, split=split, max_per_class=max_per_class)

    if modality in ("image", "image_human"):
        dataset = ImageDeepfakeDataset(
            samples, target_size=target_size,
            augment_level=augment_level if split == "train" else 0)
    elif modality == "video":
        dataset = VideoDeepfakeDataset(
            samples, target_size=target_size, num_frames=num_frames,
            augment_level=augment_level if split == "train" else 0)
    elif modality == "audio":
        dataset = AudioDeepfakeDataset(samples)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    use_drop_last = (split == "train") and (len(dataset) > batch_size)
    eff_workers = min(num_workers, len(dataset))

    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(len(dataset), 1)),
        shuffle=(split == "train"),
        num_workers=eff_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=use_drop_last,
        persistent_workers=(eff_workers > 0),
    )
