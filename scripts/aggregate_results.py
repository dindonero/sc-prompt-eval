#!/usr/bin/env python3
"""
Aggregate experiment results and generate paper-ready tables.

Usage:
    python scripts/aggregate_results.py outputs/experiment_v1
    python scripts/aggregate_results.py outputs/experiment_v1 --format latex
    python scripts/aggregate_results.py outputs/experiment_v1 --format csv --output results/
    python scripts/aggregate_results.py outputs/experiment_v1 --with-stats
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add src to path for stats imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from sc_prompt_eval.scoring.stats import (
    bootstrap_ci_f1,
    wilcoxon_signed_rank,
    holm_bonferroni_correction,
    ConfidenceInterval,
)


def load_parsed_results(output_dir: Path) -> List[Dict]:
    """Load all parsed result files from output directory."""
    results = []

    # Walk the directory structure: dataset/model/prompt/contract/run_{idx}_parsed.json
    for parsed_file in output_dir.rglob("*_parsed.json"):
        try:
            data = json.loads(parsed_file.read_text())

            # Extract path components
            parts = parsed_file.relative_to(output_dir).parts
            if len(parts) >= 4:
                dataset = parts[0]
                model = parts[1]
                prompt = parts[2]
                contract = parts[3]
                run_file = parts[4] if len(parts) > 4 else parsed_file.name

                # Parse run index from filename
                run_idx = 0
                if run_file.startswith("run_"):
                    try:
                        run_idx = int(run_file.split("_")[1])
                    except (IndexError, ValueError):
                        pass

                result = {
                    "dataset": dataset,
                    "model": model,
                    "prompt": prompt,
                    "contract": contract,
                    "run_idx": run_idx,
                    **data.get("metrics", {}),
                    "finding_count": data.get("finding_count", 0),
                    "parse_errors": len(data.get("parse_errors", [])),
                    "gt_count": len(data.get("ground_truth", [])),
                }

                # Add LOC if available
                if "loc" in data:
                    result.update(data["loc"])

                # Add instance-level metrics if available
                inst = data.get("metrics", {}).get("instance_metrics", {})
                if inst:
                    result["inst_tp"] = inst.get("tp", 0)
                    result["inst_fp"] = inst.get("fp", 0)
                    result["inst_fn"] = inst.get("fn", 0)
                    result["has_line_annotations"] = inst.get("has_line_annotations", False)

                # Add runner metadata
                meta = data.get("runner_metadata", {})
                result["latency_s"] = meta.get("latency_s", 0)
                result["api_calls"] = meta.get("api_calls", 1)

                results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {parsed_file}: {e}", file=sys.stderr)

    return results


def aggregate_by_model_prompt(results: List[Dict]) -> pd.DataFrame:
    """Aggregate results by (model, prompt) pairs."""
    df = pd.DataFrame(results)

    if df.empty:
        return pd.DataFrame()

    # Group by model and prompt
    grouped = df.groupby(["model", "prompt"]).agg({
        "tp": "sum",
        "fp": "sum",
        "fn": "sum",
        "finding_count": "sum",
        "gt_count": "sum",
        "parse_errors": "sum",
        "contract": "count",
        "latency_s": "mean",
        "api_calls": "sum",
    }).reset_index()

    grouped.rename(columns={"contract": "n_contracts"}, inplace=True)

    # Compute aggregate metrics
    grouped["precision"] = grouped["tp"] / (grouped["tp"] + grouped["fp"])
    grouped["recall"] = grouped["tp"] / (grouped["tp"] + grouped["fn"])
    grouped["f1"] = 2 * grouped["precision"] * grouped["recall"] / (grouped["precision"] + grouped["recall"])

    # Fill NaN with 0
    grouped = grouped.fillna(0)

    # Round percentages
    grouped["precision"] = (grouped["precision"] * 100).round(1)
    grouped["recall"] = (grouped["recall"] * 100).round(1)
    grouped["f1"] = (grouped["f1"] * 100).round(1)
    grouped["latency_s"] = grouped["latency_s"].round(2)

    return grouped


def aggregate_by_category(results: List[Dict]) -> pd.DataFrame:
    """Aggregate results by vulnerability category."""
    # Flatten ground truth categories
    category_results = []

    for r in results:
        # Get ground truth category if available
        gt = r.get("gt_categories", [])
        if not gt and "gt_category" in r:
            gt = [r["gt_category"]] if r["gt_category"] else []

        for cat in gt:
            category_results.append({
                "model": r["model"],
                "prompt": r["prompt"],
                "category": cat,
                "tp": 1 if cat in r.get("pred_categories", []) else 0,
                "fn": 0 if cat in r.get("pred_categories", []) else 1,
            })

    if not category_results:
        return pd.DataFrame()

    df = pd.DataFrame(category_results)

    grouped = df.groupby(["model", "prompt", "category"]).agg({
        "tp": "sum",
        "fn": "sum",
    }).reset_index()

    grouped["recall"] = grouped["tp"] / (grouped["tp"] + grouped["fn"])
    grouped["recall"] = (grouped["recall"] * 100).round(1)

    return grouped


def aggregate_instance_level(results: List[Dict]) -> pd.DataFrame:
    """Aggregate instance-level (line-matching) metrics."""
    # Filter to results with instance metrics
    inst_results = [r for r in results if r.get("has_line_annotations")]

    if not inst_results:
        return pd.DataFrame()

    df = pd.DataFrame(inst_results)

    grouped = df.groupby(["model", "prompt"]).agg({
        "inst_tp": "sum",
        "inst_fp": "sum",
        "inst_fn": "sum",
        "contract": "count",
    }).reset_index()

    grouped.rename(columns={"contract": "n_contracts"}, inplace=True)

    grouped["precision"] = grouped["inst_tp"] / (grouped["inst_tp"] + grouped["inst_fp"])
    grouped["recall"] = grouped["inst_tp"] / (grouped["inst_tp"] + grouped["inst_fn"])
    grouped["f1"] = 2 * grouped["precision"] * grouped["recall"] / (grouped["precision"] + grouped["recall"])

    grouped = grouped.fillna(0)
    grouped["precision"] = (grouped["precision"] * 100).round(1)
    grouped["recall"] = (grouped["recall"] * 100).round(1)
    grouped["f1"] = (grouped["f1"] * 100).round(1)

    return grouped


def compute_confidence_intervals(results: List[Dict]) -> pd.DataFrame:
    """Compute bootstrap confidence intervals for P/R/F1 by (model, prompt).

    Returns DataFrame with columns: model, prompt, precision_ci, recall_ci, f1_ci
    where each _ci column contains "mean [lower, upper]" strings.
    """
    # Group results by (model, prompt) and collect per-contract TP/FP/FN
    grouped_data = defaultdict(lambda: {"tp": [], "fp": [], "fn": []})

    for r in results:
        key = (r["model"], r["prompt"])
        grouped_data[key]["tp"].append(r.get("tp", 0) or 0)
        grouped_data[key]["fp"].append(r.get("fp", 0) or 0)
        grouped_data[key]["fn"].append(r.get("fn", 0) or 0)

    rows = []
    for (model, prompt), data in grouped_data.items():
        try:
            p_ci, r_ci, f1_ci = bootstrap_ci_f1(
                data["tp"], data["fp"], data["fn"],
                n_bootstrap=10000, ci=0.95
            )
            rows.append({
                "model": model,
                "prompt": prompt,
                "precision": f"{p_ci.mean*100:.1f}",
                "precision_ci": f"[{p_ci.lower*100:.1f}, {p_ci.upper*100:.1f}]",
                "recall": f"{r_ci.mean*100:.1f}",
                "recall_ci": f"[{r_ci.lower*100:.1f}, {r_ci.upper*100:.1f}]",
                "f1": f"{f1_ci.mean*100:.1f}",
                "f1_ci": f"[{f1_ci.lower*100:.1f}, {f1_ci.upper*100:.1f}]",
                "n_samples": len(data["tp"]),
            })
        except Exception as e:
            print(f"Warning: CI computation failed for {model}/{prompt}: {e}", file=sys.stderr)

    return pd.DataFrame(rows)


def compute_paired_comparisons(results: List[Dict], baseline_prompt: str = "P0_baseline") -> pd.DataFrame:
    """Compute paired statistical tests comparing each prompt to a baseline.

    Uses Wilcoxon signed-rank test with Holm-Bonferroni correction.

    Args:
        results: List of result dicts with model, prompt, contract, f1 scores
        baseline_prompt: Prompt ID to use as baseline for comparisons

    Returns:
        DataFrame with comparison results including p-values and significance
    """
    # Group results by (model, prompt, contract) -> f1
    f1_by_contract = defaultdict(dict)
    for r in results:
        key = (r["model"], r["prompt"])
        contract = r["contract"]
        f1 = r.get("f1", 0) or 0
        f1_by_contract[key][contract] = f1

    # Get all models
    models = list(set(r["model"] for r in results))

    comparisons = []
    for model in models:
        baseline_key = (model, baseline_prompt)
        if baseline_key not in f1_by_contract:
            continue

        baseline_scores = f1_by_contract[baseline_key]

        # Compare each other prompt to baseline
        for (m, prompt), scores in f1_by_contract.items():
            if m != model or prompt == baseline_prompt:
                continue

            # Align scores by contract
            common_contracts = set(baseline_scores.keys()) & set(scores.keys())
            if len(common_contracts) < 5:
                continue

            baseline_f1s = [baseline_scores[c] for c in common_contracts]
            prompt_f1s = [scores[c] for c in common_contracts]

            # Wilcoxon signed-rank test
            test_result = wilcoxon_signed_rank(baseline_f1s, prompt_f1s)

            comparisons.append({
                "model": model,
                "baseline": baseline_prompt,
                "prompt": prompt,
                "n_contracts": len(common_contracts),
                "baseline_mean_f1": sum(baseline_f1s) / len(baseline_f1s) * 100,
                "prompt_mean_f1": sum(prompt_f1s) / len(prompt_f1s) * 100,
                "diff": (sum(prompt_f1s) - sum(baseline_f1s)) / len(prompt_f1s) * 100,
                "wilcoxon_stat": test_result.statistic,
                "p_value": test_result.p_value,
                "effect_size": test_result.effect_size or 0,
            })

    if not comparisons:
        return pd.DataFrame()

    df = pd.DataFrame(comparisons)

    # Apply Holm-Bonferroni correction
    p_values = df["p_value"].tolist()
    significant = holm_bonferroni_correction(p_values, alpha=0.05)
    df["significant"] = significant
    df["p_value_str"] = df.apply(
        lambda r: f"{r['p_value']:.4f}{'*' if r['significant'] else ''}", axis=1
    )

    return df


def format_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Format DataFrame as LaTeX table."""
    if df.empty:
        return f"% Empty table: {label}\n"

    # Select relevant columns for main results table
    cols = ["model", "prompt", "precision", "recall", "f1"]
    display_cols = [c for c in cols if c in df.columns]

    subset = df[display_cols].copy()

    # Build LaTeX
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{" + "l" * len(display_cols) + "}",
        r"\toprule",
    ]

    # Header
    header = " & ".join([c.replace("_", " ").title() for c in display_cols]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for _, row in subset.iterrows():
        cells = []
        for col in display_cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.1f}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as Markdown table."""
    if df.empty:
        return "*(No data)*\n"

    try:
        return df.to_markdown(index=False)
    except ImportError:
        # Fallback if tabulate not installed
        return df.to_string(index=False)


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results for paper tables")
    parser.add_argument("output_dir", type=Path, help="Path to experiment output directory")
    parser.add_argument("--format", choices=["csv", "latex", "markdown", "all"], default="all",
                        help="Output format (default: all)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for generated files (default: same as output_dir)")
    parser.add_argument("--with-stats", action="store_true",
                        help="Compute bootstrap CIs and paired statistical tests")
    parser.add_argument("--baseline", type=str, default="P0_baseline",
                        help="Baseline prompt for paired comparisons (default: P0_baseline)")
    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Error: Output directory not found: {args.output_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or args.output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.output_dir}")
    results = load_parsed_results(args.output_dir)
    print(f"Loaded {len(results)} result files")

    if not results:
        print("No results found!", file=sys.stderr)
        sys.exit(1)

    # Generate aggregations
    print("\n=== Category-Level Results (Model x Prompt) ===")
    main_df = aggregate_by_model_prompt(results)
    print(format_markdown_table(main_df))

    print("\n=== Instance-Level Results (Line Matching) ===")
    inst_df = aggregate_instance_level(results)
    if not inst_df.empty:
        print(format_markdown_table(inst_df))
    else:
        print("(No instance-level annotations available)")

    # Statistical analysis (optional)
    ci_df = pd.DataFrame()
    comparison_df = pd.DataFrame()
    if args.with_stats:
        print("\n=== Bootstrap Confidence Intervals (95%) ===")
        ci_df = compute_confidence_intervals(results)
        if not ci_df.empty:
            print(format_markdown_table(ci_df))
        else:
            print("(Insufficient data for confidence intervals)")

        print(f"\n=== Paired Comparisons vs {args.baseline} ===")
        comparison_df = compute_paired_comparisons(results, baseline_prompt=args.baseline)
        if not comparison_df.empty:
            print(format_markdown_table(comparison_df))
        else:
            print("(Insufficient data for paired comparisons)")

    # Save outputs
    if args.format in ("csv", "all"):
        main_df.to_csv(out_path / "category_level_results.csv", index=False)
        print(f"\nSaved: {out_path / 'category_level_results.csv'}")

        if not inst_df.empty:
            inst_df.to_csv(out_path / "instance_level_results.csv", index=False)
            print(f"Saved: {out_path / 'instance_level_results.csv'}")

        if args.with_stats and not ci_df.empty:
            ci_df.to_csv(out_path / "confidence_intervals.csv", index=False)
            print(f"Saved: {out_path / 'confidence_intervals.csv'}")

        if args.with_stats and not comparison_df.empty:
            comparison_df.to_csv(out_path / "paired_comparisons.csv", index=False)
            print(f"Saved: {out_path / 'paired_comparisons.csv'}")

    if args.format in ("latex", "all"):
        latex_main = format_latex_table(
            main_df,
            caption="Category-level vulnerability detection results",
            label="tab:category-results"
        )
        (out_path / "category_level_results.tex").write_text(latex_main)
        print(f"Saved: {out_path / 'category_level_results.tex'}")

        if not inst_df.empty:
            latex_inst = format_latex_table(
                inst_df,
                caption="Instance-level vulnerability detection results",
                label="tab:instance-results"
            )
            (out_path / "instance_level_results.tex").write_text(latex_inst)
            print(f"Saved: {out_path / 'instance_level_results.tex'}")

    # Also save raw aggregated JSON for programmatic access
    summary = {
        "n_results": len(results),
        "models": list(main_df["model"].unique()) if not main_df.empty else [],
        "prompts": list(main_df["prompt"].unique()) if not main_df.empty else [],
        "category_level": main_df.to_dict(orient="records") if not main_df.empty else [],
        "instance_level": inst_df.to_dict(orient="records") if not inst_df.empty else [],
    }
    if args.with_stats:
        summary["confidence_intervals"] = ci_df.to_dict(orient="records") if not ci_df.empty else []
        summary["paired_comparisons"] = comparison_df.to_dict(orient="records") if not comparison_df.empty else []
    (out_path / "aggregated_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_path / 'aggregated_summary.json'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
