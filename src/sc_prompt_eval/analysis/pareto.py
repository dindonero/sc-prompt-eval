from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


@dataclass
class ParetoPoint:
    """A point in the Pareto analysis."""
    model: str
    prompt: str
    f1: float
    cost_per_kloc: float
    time_per_kloc: float
    is_pareto_optimal: bool = False

    @property
    def label(self) -> str:
        return f"{self.model[:10]}+{self.prompt}"


def compute_pareto_frontier(points: List[ParetoPoint], maximize_y: bool = True) -> List[ParetoPoint]:
    """
    Compute the Pareto frontier for cost-accuracy tradeoff.

    Args:
        points: List of points with (cost, accuracy)
        maximize_y: If True, higher y (F1) is better

    Returns:
        List of Pareto-optimal points
    """
    if not points:
        return []

    # Sort by cost (x-axis, lower is better)
    sorted_points = sorted(points, key=lambda p: p.cost_per_kloc)

    frontier = []
    best_y = float('-inf') if maximize_y else float('inf')

    for point in sorted_points:
        if maximize_y:
            if point.f1 > best_y:
                point.is_pareto_optimal = True
                frontier.append(point)
                best_y = point.f1
        else:
            if point.f1 < best_y:
                point.is_pareto_optimal = True
                frontier.append(point)
                best_y = point.f1

    return frontier


def plot_pareto_front(
    points: List[ParetoPoint],
    output_path: str | Path,
    title: str = "Accuracy vs Cost Pareto Front",
    figsize: Tuple[int, int] = (10, 8),
    show_labels: bool = True,
) -> None:
    """
    Plot F1 vs Cost/KLOC with Pareto frontier highlighted.

    Args:
        points: List of ParetoPoint objects
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        show_labels: Whether to label each point
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color mapping for models
    models = list(set(p.model for p in points))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_colors = {m: c for m, c in zip(models, colors)}

    # Marker mapping for prompts
    prompts = list(set(p.prompt for p in points))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    prompt_markers = {p: markers[i % len(markers)] for i, p in enumerate(prompts)}

    # Plot all points
    for point in points:
        ax.scatter(
            point.cost_per_kloc,
            point.f1,
            c=[model_colors[point.model]],
            marker=prompt_markers[point.prompt],
            s=150 if point.is_pareto_optimal else 80,
            edgecolors='black' if point.is_pareto_optimal else 'none',
            linewidths=2 if point.is_pareto_optimal else 0,
            alpha=0.8,
            zorder=10 if point.is_pareto_optimal else 5,
        )

        if show_labels:
            ax.annotate(
                point.label,
                (point.cost_per_kloc, point.f1),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=7,
                alpha=0.7,
            )

    # Plot Pareto frontier line
    frontier = [p for p in points if p.is_pareto_optimal]
    if len(frontier) > 1:
        frontier_sorted = sorted(frontier, key=lambda p: p.cost_per_kloc)
        ax.plot(
            [p.cost_per_kloc for p in frontier_sorted],
            [p.f1 for p in frontier_sorted],
            'r--',
            linewidth=2,
            alpha=0.7,
            label='Pareto Frontier',
            zorder=1,
        )

    # Legends
    model_handles = [
        mpatches.Patch(color=model_colors[m], label=m)
        for m in models
    ]
    prompt_handles = [
        plt.Line2D([0], [0], marker=prompt_markers[p], color='gray',
                   linestyle='', markersize=8, label=p)
        for p in prompts
    ]

    legend1 = ax.legend(handles=model_handles, title='Models', loc='upper left')
    ax.add_artist(legend1)
    ax.legend(handles=prompt_handles, title='Prompts', loc='lower right')

    ax.set_xlabel('Cost ($/KLOC)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Set axis limits with padding
    costs = [p.cost_per_kloc for p in points]
    f1s = [p.f1 for p in points]
    ax.set_xlim(min(costs) * 0.9, max(costs) * 1.1)
    ax.set_ylim(min(f1s) * 0.9, min(max(f1s) * 1.1, 1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()




