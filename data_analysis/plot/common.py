from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Charter", "DejaVu Serif"]

DEFAULT_OUTPUT_DIR = Path("data_analysis/output")
AXIS_LABEL_SIZE = 16
AXIS_NUMBER_SIZE = 14
PLOT_TITLE_SIZE = 20
LEGEND_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 16

COLORS = [
    "#2E86AB",
    "#F18F01",
    "#A23B72",
    "#06A77D",
    "#D4AF37",
    "#9B59B6",
    "#E74C3C",
    "#0E6453",
]

SEMANTIC_COLORS = {
    "user_interaction": "#2E86AB",
    "item_interaction": "#D4AF37",
    "item_property": "#06A77D",
}

DATASET_NAMING = {
    "ebnerd": "EB-NeRD",
    "mind": "MIND",
}


def dataset_label(dataset_name: str) -> str:
    return DATASET_NAMING.get(dataset_name, dataset_name)


def dataset_plot_title(dataset_name: str, plot_title: str) -> str:
    return f"{dataset_label(dataset_name)} {plot_title}"


def style_axis(ax, xlabel: str, ylabel: str, title: str, *, title_pad: float = 12) -> None:
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_SIZE, fontweight="bold", pad=title_pad)
    ax.tick_params(axis="both", which="major", labelsize=AXIS_NUMBER_SIZE)


def get_output_dir(output_dir: str | None = None) -> Path:
    path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: Figure, path: Path, close: bool = True) -> None:
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    if close:
        plt.close(fig)
