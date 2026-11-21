"""Shared matplotlib style so every report figure uses the same palette, ink and grid."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; no window needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, always assigned in this order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Diverging blue <-> red with a neutral gray midpoint (for correlations, -1 ... 1).
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])


def apply_style() -> None:
    """Thin marks, recessive axes and grid, text in ink colors rather than series colors."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "lines.linewidth": 1.5, "legend.frameon": False, "font.size": 9,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })


def save(fig: plt.Figure, path: Path) -> None:
    """One place for resolution and cleanup so all figures look the same in the report."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
