# Aircraft Seam Measurement

Python pipeline for measuring **gap** and **flush** on aircraft skin seams from
3D line-scanner point clouds. A UNet / Attention-UNet segmentation model
identifies the seam region in the depth map, and geometric routines compute
per-section gap / flush statistics.

## Project layout

```
src/
  pipeline/              # Online inference + measurement pipeline
    __main__.py          # CLI entry: runs the full gap/flush pipeline
    gap_flush.py         # Top-level orchestration (load -> infer -> measure -> export)
    seam_mapping/        # Point-cloud I/O, depth extraction, seam-mask inference
    seam_measurement/    # Geometry: sections, top/bottom surfaces, gap/flush math
    seam_training/       # Minimal training-module shim used at inference time
    outputs.py           # JSON / CSV / PNG export helpers
    viewer3d.py          # Optional Open3D result viewer
  seam_training/         # Full training package (model, data, train loop, utils)

scripts/
  preprocess/            # Manual cropping and mask-editing tools
  train/                 # Training entry points (single + LOOCV)
  eval/                  # Checkpoint comparison utilities
  draw/                  # Visualisation helpers (Open3D etc.)

data/      # Local input data (git-ignored)
outputs/   # Runs, checkpoints, metrics (git-ignored)
app/       # Vendor SDK integration (git-ignored; see APP_SDK_CUSTOMIZATION_NOTES.md)
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

For GPU training, install a torch build matching your CUDA toolkit from
<https://pytorch.org/get-started/locally/> instead of relying on the plain
`torch` pin in `requirements.txt`.

## Running the measurement pipeline

From the project root:

```bash
python -m src.pipeline \
  --pcd-path    <path/to/crop.pcd> \
  --checkpoint-path <path/to/latest.pth> \
  --show-3d-viewer
```

Commonly useful flags:

| Flag | Purpose |
| --- | --- |
| `--threshold`               | Segmentation probability threshold (default from training config). |
| `--seam-step`               | Section sampling step in pixels. |
| `--fast-mode`               | Use the faster section-extraction path for online runs. |
| `--no-profile-plots`        | Skip gap / flush PNG generation. |
| `--show-3d-viewer`          | Open an Open3D viewer on the result. |
| `--save-3d-viewer-bundle`   | Export `viewer_bundle.npz` so the viewer can be reopened later. |

Outputs (summary JSON, section CSV, optional PNGs and viewer bundle) are
written under `outputs/pipeline/` by default; override with `--output-root`.

## Training

Single real-data run:

```bash
python scripts/train/train_real_only_model.py
```

LOOCV experiment suite:

```bash
python scripts/train/run_loocv_experiments.py
```

Training configuration (input size, loss weights, patch sampling, etc.) lives
in `src/seam_training/config.json`.

## Tests

A small pytest suite covers the pure-numpy utilities (reference-frame
geometry, ASCII-PCD loader). It runs without torch / opencv / open3d so it
is safe to execute in a minimal CI environment.

```bash
pip install numpy pytest
pytest
```

See `.github/workflows/ci.yml` for the configuration that runs on push /
pull request.

## Data / model artefacts

`data/`, `outputs/`, and `app/` are intentionally git-ignored. The repository
ships with source code only; input point clouds, trained checkpoints, and the
vendor SDK stay on each developer's machine. See
[`APP_SDK_CUSTOMIZATION_NOTES.md`](APP_SDK_CUSTOMIZATION_NOTES.md) for the
SDK-side integration summary.
