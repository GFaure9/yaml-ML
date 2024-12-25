from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np


def plot_scores(scores: List[Dict[str, Dict[str, float]]], save_path: str = None):
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

    # Define colors for each score type
    colors = plt.cm.tab10(np.linspace(0, 1, num_types))

    # Plot each score type in a separate subplot
    fig, axes = plt.subplots(num_types, 1, figsize=(5, 5), sharex=True)
    if num_types == 1:
        axes = [axes]  # Ensure axes is iterable when there is only one score type

    bar_width = 0.3  # Thinner bars

    for ax, score_type, color in zip(axes, score_types, colors):
        ax.bar(names, score_values[score_type], color=color, alpha=0.8, width=bar_width)
        ax.set_ylabel(score_type.upper())
        ax.grid(color="grey", alpha=0.2, ls="--", axis="y")

        # Hide ticks except on the last subplot
        if ax != axes[-1]:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Set x-axis label only for the last subplot
    axes[-1].set_xlabel("Configurations")

    fig.suptitle("Scores", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust layout to fit the suptitle

    if not save_path:
        plt.show()

    else:
        plt.savefig(save_path)
        plt.close()


def transform_to_universal_score(value: float, score_type: str):
    pass


def rank_from_scores(scores):
    pass


def plot_ranking():
    pass





