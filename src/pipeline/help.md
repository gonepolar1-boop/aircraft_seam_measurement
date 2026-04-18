# Pipeline CLI quick reference

Run from the project root. Replace the paths below with those for your
environment (the examples use forward slashes, but Windows-style paths also
work).

```bash
python -m src.pipeline \
  --pcd-path        data/process/manual_crop/1/crop.pcd \
  --checkpoint-path outputs/model/<run_id>/checkpoints/latest.pth \
  --show-3d-viewer
```

See `python -m src.pipeline --help` for the full list of flags, and the
top-level `README.md` for an overview of the pipeline and output artefacts.
