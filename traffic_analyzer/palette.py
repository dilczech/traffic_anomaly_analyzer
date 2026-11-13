"""Shared report and chart colours; semantic roles stay stable."""
PALETTE = {
    "header": "#33465B",
    "header_text": "#E8EFF6",
    "heading": "#3478A7",
    "text": "#263445",
    "fact": "#3478A7",
    "reference": "#7A8793",
    "anomaly": "#C45151",
    "quality": "#87719B",
    "observation": "#FFF3D6",
    "high_priority": "#F8E1DE",
    "stripe": "#F5F8FB",
    "grid": "#DFE6ED",
    "paper": "#FFFFFF",
}


def excel_colour(role):
    return PALETTE[role].lstrip("#")


def apply_plot_theme(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": PALETTE["text"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["heading"],
        "axes.edgecolor": PALETTE["reference"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "grid.color": PALETTE["grid"],
        "figure.facecolor": PALETTE["paper"],
        "axes.facecolor": PALETTE["paper"],
    })
