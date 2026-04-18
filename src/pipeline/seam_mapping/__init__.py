"""Seam-mapping subpackage.

Uses lazy attribute loading so importing a single submodule (for example
``pipeline.seam_mapping.io`` in a minimal test environment) does not pull in
the heavier inference dependencies (torch, opencv) eagerly. This mirrors the
pattern already used in :mod:`pipeline`.
"""

from __future__ import annotations

__all__ = [
    "extract_mask_point_cloud",
    "load_point_map",
    "predict_mask_from_point_map",
    "preload_model",
]


def __getattr__(name: str):
    if name == "extract_mask_point_cloud":
        from .extraction import extract_mask_point_cloud

        return extract_mask_point_cloud
    if name == "load_point_map":
        from .io import load_point_map

        return load_point_map
    if name in {"predict_mask_from_point_map", "preload_model"}:
        from . import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
