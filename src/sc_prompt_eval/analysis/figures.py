from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def plot_vuln_heatmap(
    df: pd.DataFrame,
    output_path: str | Path,
    metric: str = 'f1',
    title: str = "Detection Performance by Vulnerability Type",
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    Create heatmap of performance by vulnerability type.

    Args:
        df: DataFrame with columns: category, model, prompt, and metric column
        output_path: Path to save figure
        metric: Metric column to visualize
        title: Plot title
        figsize: Figure size
    """
    # Pivot table: rows = DASP categories, columns = (model, prompt)
    df['method'] = df['model'].str[:10] + '+' + df['prompt']
    pivot = df.pivot_table(
        index='category',
        columns='method',
        values=metric,
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not pd.isna(val):
                text_color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=text_color, fontsize=8)

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Method (Model + Prompt)', fontsize=12)
    ax.set_ylabel('DASP Category', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric.upper(), fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cost_breakdown(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Cost Breakdown by Method",
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Create bar chart of costs by method.

    Args:
        df: DataFrame with columns: model, prompt, cost_per_kloc, avg_latency_s
        output_path: Path to save figure
    """
    df = df.copy()
    df['method'] = df['model'].str[:12] + '\n' + df['prompt']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Cost per KLOC
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    bars1 = ax1.bar(df['method'], df['cost_per_kloc'], color=colors)
    ax1.set_ylabel('Cost ($/KLOC)', fontsize=11)
    ax1.set_title('API Cost per KLOC', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, val in zip(bars1, df['cost_per_kloc']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'${val:.3f}', ha='center', va='bottom', fontsize=8)

    # Latency
    if 'avg_latency_s' in df.columns or 'time_per_kloc' in df.columns:
        time_col = 'time_per_kloc' if 'time_per_kloc' in df.columns else 'avg_latency_s'
        bars2 = ax2.bar(df['method'], df[time_col], color=colors)
        ax2.set_ylabel('Time (s/KLOC)', fontsize=11)
        ax2.set_title('Processing Time per KLOC', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars2, df[time_col]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{val:.1f}s', ha='center', va='bottom', fontsize=8)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()






