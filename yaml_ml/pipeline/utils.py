from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np


def plot_scores(scores: List[Dict[str, Dict[str, float]]], save_path: str = None):
    # Enable LaTeX-style fonts and customize aesthetics
    plt.style.use('seaborn-v0_8-muted')
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # Parse scores_dicts
    names = []
    score_types = set()
    scores_dicts = []

    for entry in scores:
        for name, score_dict in entry.items():
            names.append(name)
            scores_dicts.append(score_dict)
            score_types.update(score_dict.keys())

    score_types = sorted(score_types)
    num_types = len(score_types)

    # Prepare data for plots
    score_values = {score_type: [] for score_type in score_types}
    for score_dict in scores_dicts:
        for score_type in score_types:
            score_values[score_type].append(score_dict.get(score_type, 0))

    # Define a custom color palette
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, num_types))

    # Plot each score type in a separate subplot
    fig, axes = plt.subplots(num_types, 1, figsize=(6, 6 + num_types * 0.5), sharex=True)
    if num_types == 1:
        axes = [axes]  # Ensure axes is iterable when there is only one score type

    bar_width = 0.4

    for ax, score_type, color in zip(axes, score_types, colors):
        ax.bar(names, score_values[score_type], color=color, alpha=0.85, width=bar_width, edgecolor="black",
               linewidth=0.5)
        ax.set_ylabel(score_type.upper(), fontsize=12, labelpad=10)
        ax.grid(color="grey", alpha=0.3, ls="--", axis="y")
        ax.set_axisbelow(True)

        # Enhance tick visibility
        ax.tick_params(axis='both', direction='in', length=5, width=0.5)

        # Hide x-ticks except on the last subplot
        if ax != axes[-1]:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Set x-axis label only for the last subplot
    axes[-1].set_xlabel("Configurations", fontsize=12, labelpad=10)
    axes[-1].tick_params(axis='x', which='both', labelrotation=45)

    # Adjust spacing and add a title
    fig.suptitle(r"\textbf{Scores}", fontsize=16, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def transform_to_universal_score(value: float, score_type: str):
    pass


def rank_from_scores(scores):
    pass


def plot_ranking():
    pass





