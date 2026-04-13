# dfresearch

Autonomous deepfake detection research for [BitMind Subnet 34](https://bitmind.ai/subnet).

Give an AI agent a deepfake detection training setup and let it experiment autonomously. It modifies the training code, trains for 10 minutes, checks if the result improved, keeps or discards, and repeats. You wake up to a log of experiments and a better model.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [unconst/Arbos](https://github.com/unconst/Arbos). Datasets sourced from [BitMind-AI/gasbench](https://github.com/BitMind-AI/gasbench).

### TL;DR — train and submit in 4 commands

```bash
uv sync                                          # install deps
uv run prepare.py --modality image                # download datasets
uv run train_image.py                             # train (10 min, outputs submission-ready checkpoint)
uv run export.py --modality image --model efficientnet-b4  # package ZIP for submission
```

---

## How it works

```bash
┌──────────────────────────────────────────────────────────┐
│  SETUP                                                    │
│  1. Download datasets (uv run prepare.py)                 │
│  2. Create experiment branch                              │
│  3. Run baseline                                          │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  LOOP FOREVER                                             │
│                                                           │
│  1. Edit train_{modality}.py with an idea                 │
│  2. git commit                                            │
│  3. uv run train_{modality}.py > run.log 2>&1             │
│  4. Parse results: grep sn34_score run.log                │
│  5. sn34_score improved? → KEEP (advance branch)          │
│     sn34_score same/worse? → DISCARD (git reset)          │
│  6. Log to results.tsv                                    │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  EXPORT                                                   │
│  1. uv run export.py --modality image --model effnet-b4   │
│  2. gasbench run --image-model detector.zip --small       │
│  3. gascli d push --image-model detector.zip              │
└──────────────────────────────────────────────────────────┘
```

The repo has three types of files:

- **`prepare.py`** — Fixed constants, data downloading, evaluation metrics. Not modified.
- **`train_*.py`** — Training scripts the agent edits. One per modality. Contains model choice, hyperparameters, optimizer, training loop. **This is what the agent modifies.**
- **`program.md`** — Agent instructions. **This is what the human iterates on.**

## Quick start

**Requirements:** NVIDIA GPU (tested on A100/H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment and install dependencies

```bash
# Creates a .venv in the project root and installs everything from pyproject.toml
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

<details>
<summary>Alternative: using standard venv + pip</summary>

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Note: `uv run` commands below assume `uv`. If you used pip, replace `uv run <script>` with `python <script>`.

</details>

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your HuggingFace token — needed for gated datasets
# Get one at: https://huggingface.co/settings/tokens
```

### 4. Download datasets

```bash
uv run prepare.py --modality image              # image datasets
uv run prepare.py --modality video              # video datasets
uv run prepare.py --modality audio              # audio datasets
uv run prepare.py                               # all modalities
uv run prepare.py --modality image --workers 8  # faster with more workers
```

### 5. Verify data

```bash
uv run prepare.py --verify --modality image
```

### 6. Train (~10 min on GPU)

```bash
uv run train_image.py                 # trains EfficientNet-B4 baseline
uv run train_video.py                 # trains R3D-18 baseline
uv run train_audio.py                 # trains Wav2Vec2 baseline
```

### 7. Evaluate and export

```bash
uv run evaluate.py --modality image
uv run export.py --modality image --model efficientnet-b4
```

### Setting up results tracking

Before running the autoresearch loop, create a `results.tsv` file:

```bash
echo -e "commit\tsn34_score\taccuracy\tmemory_gb\tstatus\tdescription" > results.tsv
```

This file tracks all experiments. See `program.md` for the full autoresearch workflow.

### CLI entry point

The `dfresearch` CLI wraps all commands:

```bash
dfresearch prepare --modality image          # download datasets
dfresearch train --modality image            # train default model
dfresearch evaluate --modality image         # evaluate checkpoint
dfresearch export --modality image --model efficientnet-b4  # export for competition
```

## Running the agent

Point your AI coding agent (Claude, Codex, etc.) at this repo and prompt:

```bash
Hi, have a look at program.md and let's kick off a new image experiment!
Let's do the setup first.
```

The agent will create a branch, establish a baseline, and start experimenting autonomously. Each experiment takes ~10 minutes, so you get ~6/hour or ~50 overnight.

## Baseline models

### Image (3 baselines)

| Model | Params | Source | Description |
|---|---|---|---|
| **EfficientNet-B4** | 19M | [timm](https://github.com/huggingface/pytorch-image-models) | ImageNet-pretrained CNN. Standard baseline from FaceForensics++ and DFDC challenge literature. Proven, efficient, well-understood. |
| **CLIP ViT-L/14** | 304M | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | OpenAI's vision-language model. CLIP features generalize exceptionally well across unseen generators ([UniversalFakeDetect, CVPR 2023](https://arxiv.org/abs/2302.10174)). |
| **SMOGY Swin** | 87M | [Smogy/SMOGY-Ai-images-detector](https://huggingface.co/Smogy/SMOGY-Ai-images-detector) | Swin Transformer already fine-tuned for AI-image detection (98.2% accuracy). Arrives pre-trained for the task, unlike other baselines. CC-BY-NC-4.0 license. |

### Video (3 baselines)

| Model | Params | Source | Description |
|---|---|---|---|
| **R3D-18** | 33M | [torchvision](https://pytorch.org/vision/stable/models/video_resnet.html) | 3D ResNet-18 pretrained on Kinetics-400. 3D convolutions capture temporal artifacts between frames. Lightweight, fast to train. |
| **VideoMAE** | 87M | [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) | Self-supervised masked autoencoder pretrained on video. Strong spatiotemporal representations with excellent transfer learning. |
| **Hiera** | 52M | [facebook/hiera-base-224-hf](https://huggingface.co/facebook/hiera-base-224-hf) | Meta's fast hierarchical ViT pretrained with MAE. Learns spatial biases through pretraining rather than hand-designed modules. Processes frames independently then pools temporally. |

### Audio (3 baselines)

| Model | Params | Source | Description |
|---|---|---|---|
| **Wav2Vec2** | 95M | [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) | Self-supervised pretrained on 960h of speech. Dominant approach in ASVspoof challenges. Takes raw waveform input (16kHz, 6s). |
| **AST** | 87M | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | Audio Spectrogram Transformer — applies ViT to mel-spectrograms. AudioSet pretrained. Frequency-domain approach, complementary to Wav2Vec2. |
| **WavLM** | 95M | [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus) | Microsoft's SSL model that [outperforms Wav2Vec2 on ASVspoof5](https://arxiv.org/html/2408.07414v1). Same raw waveform API, better pretrained representations. Drop-in upgrade. |

## Switching models

Each training script defaults to one model. To switch:

```bash
# Image
uv run train_image.py --model clip-vit-l14
uv run train_image.py --model smogy-swin

# Video
uv run train_video.py --model videomae
uv run train_video.py --model hiera

# Audio
uv run train_audio.py --model ast
uv run train_audio.py --model wavlm
```

Or edit the `MODEL_NAME` constant at the top of the training script.

## Adding a new model

You can add any model from HuggingFace, timm, or custom PyTorch code. There are 4 files to touch.

### 1. Create the model file

Create a new file in `src/dfresearch/models/{modality}/`. The file must:
- Define a `nn.Module` class that takes **uint8 `[B, C, H, W]` input** and returns **`[B, 2]` logits**
- Handle normalization inside `forward()` (the input is always raw `[0, 255]`)
- Define `load_model(weights_path, num_classes=2)` at module level for gasbench

Example — adding DINOv2 for image detection:

```python
# src/dfresearch/models/image/dinov2.py

import torch
import torch.nn as nn
from safetensors.torch import load_file

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


class DINOv2Detector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        if not pretrained:
            self.backbone.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )
        self.register_buffer("mean", torch.tensor(DINO_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(DINO_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        features = self.backbone(x)
        return self.head(features)


def load_model(weights_path, num_classes=2):
    model = DINOv2Detector(num_classes=num_classes, pretrained=False)
    model.load_state_dict(load_file(weights_path))
    model.train(False)
    return model
```

> **Key rules for `forward()`**: Input is always uint8 `[0, 255]`. Output is always `[B, 2]` logits (not probabilities). Normalization happens inside the model, not in the data pipeline.

### 2. Register in the model registry

Edit `src/dfresearch/models/__init__.py`:

```python
from dfresearch.models.image.dinov2 import DINOv2Detector

MODEL_REGISTRY = {
    "image": {
        "efficientnet-b4": EfficientNetB4Detector,
        "clip-vit-l14": CLIPViTDetector,
        "dinov2-vitb14": DINOv2Detector,  # ← add here
    },
    ...
}
```

### 3. Register in the export map

Edit `export.py` — add the module path so export knows where to find `model.py`:

```python
MODEL_MODULES = {
    "image": {
        "efficientnet-b4": "dfresearch.models.image.efficientnet",
        "clip-vit-l14": "dfresearch.models.image.clip_vit",
        "dinov2-vitb14": "dfresearch.models.image.dinov2",  # ← add here
    },
    ...
}
```

### 4. Use it

```bash
uv run train_image.py --model dinov2-vitb14
uv run export.py --modality image --model dinov2-vitb14
```

### Allowed imports in submissions

Your `model.py` can only use these packages (enforced by gasbench):

`torch`, `torchvision`, `torchaudio`, `transformers`, `timm`, `einops`, `safetensors`, `PIL`, `cv2`, `numpy`, `scipy`

If your model needs a package not on this list, it won't run in the competition. See the [gasbench safetensors spec](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md#allowed-imports) for the full list.

### Tips for choosing models

- **timm models** (`timm.create_model("model_name", ...)`) — 1000+ image architectures. Good starting point. Try `convnext_base`, `swin_base_patch4_window7_224`, `efficientnetv2_m`.
- **HuggingFace transformers** — CLIP, DINOv2, SigLIP, VideoMAE, Wav2Vec2, Whisper, etc. Often the strongest pretrained representations.
- **Ensemble approach** — Train multiple models, average their logits in a wrapper `forward()`. Package as a single `model.py` with multiple backbones.
- **Smaller is faster** — The entrance exam has a timeout. A 3B parameter model might be more accurate but fail the time limit.

## Project structure

```
dfresearch/
├── README.md                          # This file
├── pyproject.toml                     # Dependencies
├── program.md                         # Autoresearch agent instructions
│
├── prepare.py                         # Data download & evaluation (DO NOT MODIFY)
├── train_image.py                     # Image training (agent modifies)
├── train_video.py                     # Video training (agent modifies)
├── train_audio.py                     # Audio training (agent modifies)
├── train_full.py                      # Full production training launcher
├── evaluate.py                        # Evaluate a checkpoint
├── export.py                          # Export to gasbench ZIP format
├── analysis.ipynb                     # Experiment visualization
│
└── src/dfresearch/
    ├── __init__.py
    ├── cli.py                         # CLI entry point
    ├── data.py                        # Dataset download & DataLoaders
    ├── transforms.py                  # Augmentations
    └── models/
        ├── __init__.py                # Model registry
        ├── image/
        │   ├── efficientnet.py        # EfficientNet-B4
        │   └── clip_vit.py            # CLIP ViT-L/14
        ├── video/
        │   ├── r3d.py                 # R3D-18
        │   └── videomae.py            # VideoMAE
        └── audio/
            ├── wav2vec2.py            # Wav2Vec2
            └── ast_model.py           # Audio Spectrogram Transformer
```

**No local dataset configs.** Dataset definitions are pulled directly from [gasbench](https://github.com/BitMind-AI/gasbench/tree/main/src/gasbench/dataset/configs) at runtime and cached in `~/.cache/dfresearch/gasbench_configs/`. This keeps dfresearch automatically in sync with the competition benchmark datasets.

## Datasets

Dataset configs are pulled directly from [BitMind-AI/gasbench](https://github.com/BitMind-AI/gasbench/tree/main/src/gasbench/dataset/configs) at runtime -- the same configs used by the competition benchmark. This means:

- **Always in sync.** When gasbench adds new datasets, dfresearch picks them up automatically.
- **No stale copies.** You never have to manually update dataset lists.
- **Cached locally.** Configs are fetched once and stored at `~/.cache/dfresearch/gasbench_configs/`.

To re-fetch configs (e.g., after gasbench adds new datasets):

```bash
uv run prepare.py --refresh-configs
```

### How downloading works

`prepare.py` downloads up to `--max-samples` samples (default: 500) from each dataset in the gasbench config. This gives you a diverse training set that covers the same distribution the competition evaluates on.

```bash
# Download 500 samples per dataset (default)
uv run prepare.py --modality image

# Download more for better training
uv run prepare.py --modality image --max-samples 1000

# Download fewer for quick experiments
uv run prepare.py --modality image --max-samples 100

# Check what's cached
uv run prepare.py --verify --modality image
```

The `--verify` flag shows a breakdown by dataset with real/synthetic/semisynthetic counts and class balance ratio.

### Current dataset coverage (from gasbench)

| Modality | Datasets | Types |
|----------|----------|-------|
| **Image** | 60+ datasets | Real (flickr30k, celeb-a-hq, ffhq, MS-COCO, ...), Synthetic (DALL-E 3, MidJourney, SFHQ, ...), Semisynthetic (face-swap, PICA-100K, ...) |
| **Video** | 45+ datasets | Real (UCF-101, Moments-in-Time, DFD, ...), Synthetic (VidProM, DeepAction, MovieGen, ...), Semisynthetic (DFD-fake, FakeParts, ...) |
| **Audio** | 35+ datasets | Real (Common Voice, LibriSpeech, GigaSpeech, ...), Synthetic (ElevenLabs, ShiftySpeech, FakeVoices, ...) |

See the [gasbench dataset configs](https://github.com/BitMind-AI/gasbench/tree/main/src/gasbench/dataset/configs) for the complete, authoritative list.

### Adding custom datasets

You can add your own HuggingFace datasets for training without modifying any code. Create a YAML file in the `datasets/` directory:

```bash
# Copy the example
cp datasets/image.yaml.example datasets/image.yaml
```

Edit `datasets/image.yaml`:

```yaml
datasets:
  - name: my-gan-faces
    path: myusername/gan-generated-faces
    modality: image
    media_type: synthetic

  - name: my-real-photos
    path: myusername/real-photo-collection
    modality: image
    media_type: real
```

Then download as usual — your datasets get merged with the gasbench datasets automatically:

```bash
uv run prepare.py --modality image
uv run prepare.py --verify --modality image  # your datasets show up here
```

Each entry needs four fields: `name` (unique ID), `path` (HuggingFace repo), `modality`, and `media_type` (`real`, `synthetic`, or `semisynthetic`). See `datasets/README.md` for the full format and advanced options.

## Two-phase workflow: Explore then Train

dfresearch is designed around a **two-phase workflow**:

### Phase 1: Explore (autoresearch loop, 10 min/experiment)

Fast iteration to find the best architecture, hyperparameters, and augmentation strategy:

```bash
# Download small training set (500 samples/dataset, fast)
uv run prepare.py --modality image

# Let the agent run experiments — each takes ~10 minutes
# See program.md for the full autonomous loop
uv run train_image.py > run.log 2>&1
```

This is where the autoresearch loop shines. The agent tries ~6 experiments/hour, keeps improvements, discards failures, and builds up knowledge in `STATE.md`.

### Phase 2: Full training (hours, using the best config found)

Once the agent has found a good configuration, train for real:

```bash
# Train the best image model for 4 hours with 5x more data
uv run train_full.py --modality image --hours 4 --max-samples 2000

# Train specific model found during exploration
uv run train_full.py --modality image --model clip-vit-l14 --hours 8 --max-samples 5000

# Overnight video training
uv run train_full.py --modality video --hours 12 --max-samples 3000

# Just download more data first (no training)
uv run train_full.py --modality audio --download-only --max-samples 5000

# Train with data already cached
uv run train_full.py --modality audio --hours 3 --skip-download
```

`train_full.py` handles the full pipeline in one command:
1. Downloads more training data (default 2000/dataset vs 500 during exploration)
2. Runs the training script with an extended time budget
3. Evaluates the final model
4. Exports to competition format (safetensors ZIP)

All output is logged to `runs/full_{modality}_{timestamp}.log`.

### When to switch from Phase 1 to Phase 2

- sn34_score has plateaued across 5+ experiments — you've found the local optimum
- You have a model scoring >= 80% accuracy (passes the entrance exam)
- The agent's `STATE.md` shows clear winners and the "next ideas" list is getting thin

## Competition workflow

### Scoring

Models are scored on BitMind Subnet 34 using the `sn34_score`:

```
sn34_score = sqrt(MCC_norm^1.2 * Brier_norm^1.8)
```

Where `MCC_norm = (MCC + 1) / 2` and `Brier_norm = 1 - brier_score`. This rewards both discrimination accuracy (MCC) and calibration quality (Brier).

### Submission format

Models must be submitted as a ZIP containing:

```
detector.zip
├── model_config.yaml    # Metadata and preprocessing config
├── model.py             # Model architecture with load_model() function
└── model.safetensors    # Trained weights
```

Training automatically generates all three files in `results/checkpoints/{modality}/`. The `export.py` script ZIPs them for submission.

### End-to-end competition flow

```bash
# 1. Train — checkpoint dir is submission-ready after this
uv run train_image.py

# 2. Evaluate locally
uv run evaluate.py --modality image

# 3. Package as ZIP for submission
uv run export.py --modality image --model efficientnet-b4

# 4. Test with gasbench
gasbench run --image-model results/exports/image_detector_efficientnet-b4.zip --small

# 5. Push to BitMind Subnet 34
gascli d push \
  --image-model results/exports/image_detector_efficientnet-b4.zip \
  --wallet-name your_wallet \
  --wallet-hotkey your_hotkey
```

### Entrance exam

Before your model is scored on the network, it must pass an entrance exam:

- Runs `gasbench run --small` internally
- Must achieve **>= 80% accuracy** to pass
- Maximum 1 hour 25 minute timeout

### Model requirements

- **Format**: Safetensors only (ONNX no longer accepted)
- **Input**: uint8 images/video, float32 audio (see [Safetensors spec](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md))
- **Output**: `[batch_size, 2]` logits for `[real, synthetic]`
- **Allowed imports**: torch, torchvision, torchaudio, transformers, timm, einops, safetensors, PIL, cv2, numpy, scipy (see full list in [gasbench docs](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md#allowed-imports))

## Design choices

- **Single training file per modality.** The agent only touches `train_{modality}.py`. This keeps scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for 10 minutes. This makes experiments directly comparable regardless of architecture changes.
- **Real competition metric.** We use the exact `sn34_score` formula from Subnet 34, so improvements here translate directly to better competition rankings.
- **Gasbench-compatible export.** One command to package your model in the exact format validators expect.
- **Self-contained.** No external dependencies beyond what's in `pyproject.toml`. One GPU, one training file, one metric.

## Tips for miners

1. **Start with the baseline.** Run the default training first to establish your starting point.
2. **Image is the easiest win.** More training data, faster iteration, simpler models. Start here.
3. **Augmentations matter.** Level 2-3 augmentations significantly improve generalization to unseen generators.
4. **CLIP features generalize well.** If you're struggling with cross-generator performance, try the CLIP baseline.
5. **Don't over-fit to the training set.** The competition includes private holdout datasets you can't see. Diverse training data and augmentations help.
6. **Submit early, iterate fast.** Push your first model as soon as it passes 80% accuracy. You can always update it.
7. **Watch memory.** The competition runs evaluation on cloud GPUs. Models that OOM will fail.

## Troubleshooting

**"No cached data found"** / **"No training data"**: Run `uv run prepare.py --modality {modality}` to download datasets. Use `--verify` to check cache status.

**Downloads are slow**: Increase concurrent workers: `uv run prepare.py --modality image --workers 8`. Check your internet connection and HuggingFace Hub access.

**Some datasets fail to download**: This is expected — some HuggingFace datasets require authentication or have rate limits. The system skips failures gracefully and continues with remaining datasets. You can re-run prepare to retry failed ones.

**OOM during training**: Reduce `BATCH_SIZE` or `MAX_PER_CLASS` in the training script. Increase `GRAD_ACCUM_STEPS` to compensate. For video models, reduce `NUM_FRAMES`.

**"No CUDA GPU found"**: Training will fall back to CPU but will be very slow. Ensure NVIDIA drivers and CUDA are installed. Check with `nvidia-smi`.

**Low sn34_score**: Check that your model outputs proper logits (not probabilities). Make sure augmentation is enabled during training but disabled during evaluation. Ensure both real and synthetic samples exist in the cache.

**Export fails**: Ensure you've run the training script first to generate `results/checkpoints/{modality}/model.safetensors`.

**gasbench entrance exam fails**: You need >= 80% accuracy. Check `uv run evaluate.py --modality {modality}` locally first.

## License

MIT
