# Re-export the model definitions from the full training package so the
# pipeline-side shim does not need to duplicate them.
#
# The pipeline-side ``seam_training`` subpackage intentionally stays thin:
# ``utils`` only carries the inference-time constants (no ``config.json``
# dependency), while the model architecture is a pure-PyTorch module with no
# such coupling and can safely be shared.

from seam_training.model import (  # noqa: F401
    AttentionGate,
    AttentionUNet,
    DecoderBlock,
    DoubleConv,
    EncoderBlock,
    SegmentationCriterion,
    UNet,
    build_model,
)

__all__ = [
    "AttentionGate",
    "AttentionUNet",
    "DecoderBlock",
    "DoubleConv",
    "EncoderBlock",
    "SegmentationCriterion",
    "UNet",
    "build_model",
]
