"""Shared matplotlib configuration for thesis figures.

Every figure-producing script under ``scripts/draw`` calls :func:`apply_style`
at import time so that they render consistently for ``outputs/thesis_figures``:

* PNG at 300 DPI, width ≥ 1200 px (typical 6.5" × ... in at 300 DPI)
* Chinese-capable serif (SimSun / Microsoft YaHei) with Times New Roman
  fallback for Latin characters
* Axis labels ≥ 10 pt, ticks ≥ 9 pt, titles ≥ 11 pt
* Palette tuned so lines remain distinguishable when printed in black & white
  (combines colour + linestyle + marker)

All helpers are idempotent - importing the module once is enough, but
calling :func:`apply_style` again after a third-party import that
overrode rcParams (e.g. seaborn) will re-assert the thesis style.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Output path every script writes to unless the caller overrides it.
THESIS_FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "thesis_figures"

# Candidate Chinese-capable fonts in preference order. The first one found on
# the system wins; otherwise matplotlib falls back to DejaVu Sans (and CJK
# characters will render as tofu, which is the expected behaviour on a fresh
# non-Windows box without CJK fonts installed).
_CJK_FONT_CANDIDATES = (
    "SimSun",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
)

# Colour-blind-friendly palette with built-in contrast when greyscaled.
PALETTE = {
    "primary": "#1f77b4",    # blue
    "secondary": "#d62728",  # red
    "accent": "#2ca02c",     # green
    "muted": "#7f7f7f",      # gray
    "highlight": "#ff7f0e",  # orange
    "purple": "#9467bd",
    "brown": "#8c564b",
}

LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def _pick_cjk_font() -> str | None:
    available = {fm.FontProperties(fname=path).get_name() for path in fm.findSystemFonts()}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            return name
    return None


def apply_style() -> None:
    """Set matplotlib rcParams for the thesis figure style."""
    cjk = _pick_cjk_font()
    family_stack = [cjk] if cjk else []
    family_stack.extend(["Times New Roman", "DejaVu Serif"])
    matplotlib.rcParams.update(
        {
            "font.family": family_stack,
            "axes.unicode_minus": False,   # so "-" renders correctly under CJK fonts
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "figure.dpi": 120,            # preview - the savefig dpi is 300
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.format": "png",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.0,
        }
    )


def ensure_output_dir(path: Path | None = None) -> Path:
    target = Path(path) if path is not None else THESIS_FIGURES_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def savefig(fig, save_path: Path | str, *, dpi: int = 300) -> Path:
    """Save a figure with the thesis defaults and close it.

    Returns the resolved :class:`Path` so callers can print it.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


# Apply defaults automatically on import.
apply_style()
