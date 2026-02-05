from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

from ..scoring.stats import ConfidenceInterval


def format_ci(ci: ConfidenceInterval, decimals: int = 3) -> str:
    """Format confidence interval for LaTeX."""
    fmt = f".{decimals}f"
    return f"{ci.mean:{fmt}} [{ci.lower:{fmt}}, {ci.upper:{fmt}}]"




def generate_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    columns: Optional[List[str]] = None,
    column_formats: Optional[Dict[str, str]] = None,
    highlight_best: Optional[List[str]] = None,
) -> str:
    """
    Generate a LaTeX table from DataFrame.

    Args:
        df: Input DataFrame
        caption: Table caption
        label: LaTeX label
        columns: Columns to include (default: all)
        column_formats: Dict mapping column -> format string
        highlight_best: Columns where best value should be bolded

    Returns:
        LaTeX table string
    """
    if columns:
        df = df[columns].copy()

    if column_formats is None:
        column_formats = {}

    # Find best values for highlighting
    best_values = {}
    if highlight_best:
        for col in highlight_best:
            if col in df.columns:
                best_values[col] = df[col].max()

    # Build table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    # Column spec
    col_spec = "l" + "c" * (len(df.columns) - 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header = " & ".join(df.columns)
    lines.append(f"{header} \\\\")
    lines.append("\\midrule")

    # Rows
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]

            # Format value
            if col in column_formats:
                formatted = column_formats[col].format(val)
            elif isinstance(val, float):
                formatted = f"{val:.3f}"
            else:
                formatted = str(val)

            # Highlight best
            if col in best_values and val == best_values[col]:
                formatted = f"\\textbf{{{formatted}}}"

            cells.append(formatted)

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_main_results_table(
    results_df: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """
    Generate the main results table for the paper.

    Expected columns: model, prompt, precision, recall, f1, cost_per_kloc
    """
    # Reorder and rename columns
    columns = ['model', 'prompt', 'precision', 'recall', 'f1', 'cost_per_kloc']
    df = results_df[[c for c in columns if c in results_df.columns]].copy()

    df.columns = ['Model', 'Prompt', 'Precision', 'Recall', 'F1', '\\$/KLOC']

    latex = generate_latex_table(
        df,
        caption="Main results: Detection performance and cost across prompt strategies",
        label="tab:main_results",
        highlight_best=['Precision', 'Recall', 'F1'],
        column_formats={
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1': '{:.3f}',
            '\\$/KLOC': '{:.4f}',
        }
    )

    Path(output_path).write_text(latex)
    return latex






