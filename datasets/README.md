# Custom Datasets

Drop YAML files here to add your own datasets for training.
These get merged on top of the gasbench competition configs automatically.

## File naming

Create one file per modality:
- `image.yaml` — custom image datasets
- `video.yaml` — custom video datasets
- `audio.yaml` — custom audio datasets

## Format

Same format as gasbench dataset configs. Each dataset entry needs:

```yaml
datasets:
  - name: my-custom-dataset        # unique name (used as cache folder name)
    path: username/dataset-name     # HuggingFace Hub path
    modality: image                 # image, video, or audio
    media_type: synthetic           # real, synthetic, or semisynthetic
```

### Optional fields

```yaml
    source_format: parquet          # parquet, zip, tar, jpg, mp4, wav, etc.
    include_paths: ["subdir/"]      # only include rows matching these paths
    exclude_paths: ["bad_data/"]    # exclude rows matching these paths
```

## Example

Create `datasets/image.yaml`:

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

Then download:

```bash
uv run prepare.py --modality image
```

Your custom datasets will appear alongside the gasbench datasets in training.

## Overriding gasbench entries

If you use a `name` that matches an existing gasbench dataset, your local entry
replaces the upstream one. This lets you tweak settings (e.g., add `include_paths`
or change `media_type`) without forking gasbench.
