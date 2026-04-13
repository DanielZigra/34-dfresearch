# dfresearch — Autonomous Deepfake Detection Research

This is an experiment to have an AI agent autonomously research and improve deepfake detection models for [BitMind Subnet 34](https://bitmind.ai/subnet).

## Context

BitMind runs a competitive deepfake detection subnet on Bittensor where miners submit models that detect AI-generated content across **image**, **video**, and **audio** modalities. Models are scored using the `sn34_score` metric — a geometric mean of normalized MCC and Brier score. Higher is better.

This repo provides baseline models and a training setup. Your job is to make them better.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a modality and run tag**: propose a tag based on today's date and modality (e.g. `image-mar14`). The branch `autoresearch/{tag}` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/{tag}` from current master/main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context and model descriptions.
   - `prepare.py` — fixed constants, data downloading, evaluation metrics. **Do not modify.**
   - `train_image.py` / `train_video.py` / `train_audio.py` — the file you modify (pick one based on modality).
   - `src/dfresearch/models/` — baseline model implementations.
   - `src/dfresearch/data.py` — data loading pipeline.
   - `src/dfresearch/transforms.py` — augmentation pipeline.
   - `STATE.md` — your working memory from previous sessions (if it exists).
4. **Verify data exists**: Check that `~/.cache/dfresearch/datasets/{modality}/` contains cached data. If not, tell the human to run `uv run prepare.py --modality {modality}`. Dataset configs are auto-synced from [gasbench](https://github.com/BitMind-AI/gasbench/tree/main/src/gasbench/dataset/configs) — no local config files to maintain.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row if it doesn't exist. The baseline will be recorded after the first run.
6. **Read STATE.md**: If it exists, read it carefully — it contains your notes from previous sessions about what worked, what failed, and what to try next.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Working memory (STATE.md)

You have **no memory between sessions**. The only continuity is what you write to `STATE.md`. If you don't write it there, your next session won't know about it.

After every experiment, update `STATE.md` with:
- **Current best**: the best sn34_score and what configuration achieved it
- **What worked**: patterns you've discovered (e.g. "LR 3e-4 > 1e-4 for efficientnet")
- **What failed**: ideas that didn't pan out, so you don't repeat them
- **Next ideas**: prioritized list of what to try next, based on your reflections
- **Key observations**: anything surprising about the data, loss curves, or model behavior

Keep it **short and high-signal** — under 100 lines. Prune old entries that are no longer relevant. This file is your brain between sessions.

Example STATE.md:

```markdown
# State — image experiments on autoresearch/image-mar14

## Current best
sn34_score: 0.741 | accuracy: 87.2% | model: efficientnet-b4 | commit: b2c3d4e

## What worked
- LR 3e-4 with augment_level=3 gave +0.018 over baseline
- Freezing backbone for first 2 epochs then unfreezing helped convergence
- Label smoothing 0.1 gave small but consistent improvement

## What failed
- CLIP ViT-L/14: worse than efficientnet at this data scale (needs more data)
- LR 5e-4: diverges after ~200 steps
- Focal loss: no improvement over cross-entropy, added complexity
- Batch size 64: OOM on this GPU

## Next ideas (priority order)
1. Try cosine annealing schedule (warmup 100 → cosine decay)
2. Add mixup augmentation (alpha=0.2)
3. Try efficientnet_b5 (more capacity, fits in VRAM)
4. Experiment with dropout 0.5 (currently 0.3)

## Observations
- Val loss plateaus around step 800, suggesting LR could decay earlier
- Real/synthetic class balance is 0.87 — slight imbalance toward real
- JPEG compression augmentation seems most impactful (saw biggest gap without it)
```

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock). You launch it as:

```bash
uv run train_image.py > run.log 2>&1    # for image experiments
uv run train_video.py > run.log 2>&1    # for video experiments
uv run train_audio.py > run.log 2>&1    # for audio experiments
```

**What you CAN do:**

- Modify the training script for the chosen modality (`train_image.py`, `train_video.py`, or `train_audio.py`). Everything is fair game: model choice, architecture changes, optimizer, hyperparameters, batch size, augmentation level, etc.
- Modify model files in `src/dfresearch/models/` — add new architectures, change existing ones, modify forward passes, add layers, etc.
- Modify `src/dfresearch/transforms.py` to add or tweak augmentations.
- Update `STATE.md` with your notes and observations.

**What you CANNOT do:**

- Modify `prepare.py`. It is read-only. It contains the fixed evaluation metrics and constants.
- Modify `src/dfresearch/data.py`. The data pipeline is fixed.
- Modify `evaluate.py` or `export.py`. These are stable tooling scripts.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_model` and `compute_sn34_score` functions are ground truth.

**The goal: get the highest `sn34_score` on the validation set.** Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the augmentations.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful sn34_score gains, but it should not blow up dramatically (stay under 40GB ideally).

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. When evaluating changes, weigh the complexity cost against improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline — run the training script as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
model:            efficientnet-b4
sn34_score:       0.723456
accuracy:         0.856000
mcc:              0.712000
brier:            0.134500
training_seconds: 600.1
total_seconds:    615.3
peak_vram_mb:     12340.0
num_steps:        1500
num_params_M:     19.3
num_epochs:       5
batch_size:       32
learning_rate:    0.0001
augment_level:    2
```

Extract the key metrics:

```bash
grep "^sn34_score:\|^accuracy:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 6 columns:

```
commit	sn34_score	accuracy	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. sn34_score achieved (e.g. 0.723456) — use 0.000000 for crashes
3. accuracy (e.g. 0.856000) — use 0.000000 for crashes
4. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Also save the full run log as a timestamped artifact:

```bash
cp run.log runs/$(date +%Y%m%d_%H%M%S)_run.log
```

This preserves detailed logs so you can reference past experiments' full output, not just the TSV summary.

## The experiment loop

LOOP FOREVER:

1. **Review state**: Read `STATE.md` and `results.tsv` to understand where you are
2. **Reflect**: Before choosing the next experiment, review the last 3-5 results. Ask yourself:
   - What patterns am I seeing? (e.g., "higher augmentation consistently helps")
   - Why did the last experiment succeed or fail?
   - Am I making progress or plateauing? If plateauing, try something more radical.
   - Are there any ideas from my "what failed" list that could work differently now?
3. **Plan**: Choose the next experiment based on your reflection. Write a 1-sentence hypothesis.
4. **Implement**: Modify the training script with the experimental change.
5. **Commit**: `git commit` with a descriptive message
6. **Run**: `uv run train_{modality}.py > run.log 2>&1`
7. **Parse**: `grep "^sn34_score:\|^accuracy:\|^peak_vram_mb:" run.log`
8. **Handle crashes**: If grep is empty, `tail -n 50 run.log` to diagnose. Fix if trivial, skip if fundamental.
9. **Log**: Record results in `results.tsv`. Save `run.log` to `runs/`.
10. **Decide**:
    - sn34_score improved → **keep** the commit, advance the branch
    - sn34_score same or worse → **discard** (`git reset --hard` to previous)
11. **Update STATE.md**: Write what you learned. Update best score, patterns, failed ideas, and next priorities.
12. **Go to 1**

## Experiment ideas (starting points)

### Image
- Switch between efficientnet-b4 and clip-vit-l14, compare baselines
- Try different learning rates: 5e-5, 1e-4, 3e-4, 5e-4
- Increase augment_level (more aggressive data augmentation)
- Freeze backbone and only train the head (linear probing)
- Try cosine annealing LR schedule
- Add label smoothing to cross-entropy loss
- Increase model capacity: try efficientnet_b5 or efficientnet_b7
- Add mixup or cutmix augmentation
- Try focal loss instead of cross-entropy

### Video
- Switch between r3d-18 and videomae
- Increase num_frames (more temporal context)
- Try different frame sampling strategies
- Add temporal attention on top of frame features
- Try 2D backbone + temporal pooling instead of 3D convolutions
- Experiment with different augmentation levels per frame

### Audio
- Switch between wav2vec2 and ast
- Try unfreezing the feature encoder (FREEZE_FEATURE_ENCODER = False)
- Experiment with larger wav2vec2 models (wav2vec2-large)
- Add SpecAugment-style augmentation
- Try weighted pooling instead of mean pooling over time
- Experiment with different spectrogram parameters for AST

## When to transition to full training

The 10-minute autoresearch loop is for finding the right configuration. Once you've plateaued, it's time for a full training run:

**Signs you're ready:**
- sn34_score has plateaued across 5+ consecutive experiments
- You've found a model that scores >= 80% accuracy
- Your "next ideas" list in STATE.md is getting thin
- Small hyperparameter tweaks give diminishing returns

When you think you've found the best configuration, tell the human. They can kick off a full run:

```bash
uv run train_full.py --modality {modality} --model {model_name} --hours 4 --max-samples 2000
```

This downloads more data (4x the exploration set), trains for hours instead of minutes, evaluates, and exports the model for competition. You don't need to run this — the human will.

## Competition context

When you achieve a good sn34_score, the model can be exported for BitMind competition:

```bash
uv run export.py --modality {modality} --model {model_name}
gasbench run --{modality}-model results/exports/{modality}_detector_{model_name}.zip --small
```

The entrance exam requires >=80% accuracy. The full benchmark uses sn34_score for TAO emissions ranking.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read STATE.md, look for patterns in results.tsv, review past run logs in runs/, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.
