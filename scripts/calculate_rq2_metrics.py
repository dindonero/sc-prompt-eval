#!/usr/bin/env python3
"""Calculate RQ2 per-category metrics for all prompts P0-P6."""

import json
from pathlib import Path
from collections import defaultdict
import re


PROMPT_DIRS = {
    "P0_baseline": "P0",
    "P1_icl": "P1",
    "P2_structured": "P2",
    "P3_smartguard": "P3",
    "P4_tool_augmented": "P4",
    "P5_smartaudit": "P5",
}

# Map filename prefixes to DASP categories
CATEGORY_PREFIXES = {
    "access_control": "access_control",
    "arithmetic": "arithmetic",
    "bad_randomness": "bad_randomness",
    "denial_of_service": "denial_of_service",
    "front_running": "front_running",
    "other": "other",
    "reentrancy": "reentrancy",
    "short_addresses": "short_addresses",
    "time_manipulation": "time_manipulation",
    "unchecked_low_level_calls": "unchecked_low_level_calls",
}


def get_gt_category_from_filename(filename: str) -> str:
    """Extract ground truth category from contract filename."""
    name = filename.replace('.sol', '')

    # Try exact prefix matches first (longer prefixes first)
    for prefix in sorted(CATEGORY_PREFIXES.keys(), key=len, reverse=True):
        if name.startswith(prefix + "_") or name == prefix:
            return CATEGORY_PREFIXES[prefix]

    # Handle special cases
    if name.startswith("unchecked_"):
        return "unchecked_low_level_calls"
    if name.startswith("denial_"):
        return "denial_of_service"
    if name.startswith("short_"):
        return "short_addresses"
    if name.startswith("time_"):
        return "time_manipulation"
    if name.startswith("front_"):
        return "front_running"
    if name.startswith("bad_"):
        return "bad_randomness"

    return "unknown"


def load_prompt_metrics(prompt_dir: Path, prompt_label: str) -> list[dict]:
    """Load metrics from a prompt output directory."""
    runs = []
    if not prompt_dir.exists():
        print(f"  Warning: {prompt_dir} not found")
        return runs

    for contract_dir in prompt_dir.iterdir():
        if not contract_dir.is_dir():
            continue

        parsed_file = contract_dir / "run_0_parsed.json"
        if not parsed_file.exists():
            continue

        try:
            with open(parsed_file) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        contract_name = contract_dir.name
        gt_category = get_gt_category_from_filename(contract_name)

        # Get metrics - try category_metrics first, then root level
        metrics = data.get("metrics", {})
        cat_metrics = metrics.get("category_metrics", metrics)

        # Fall back to root level if category_metrics not present
        if not cat_metrics:
            cat_metrics = {
                "tp": data.get("tp", 0),
                "fp": data.get("fp", 0),
                "fn": data.get("fn", 0),
                "gt_categories": data.get("gt_categories", [gt_category]),
            }

        runs.append({
            "prompt": prompt_label,
            "contract_id": contract_name,
            "gt_category": gt_category,
            "tp": cat_metrics.get("tp", 0),
            "fp": cat_metrics.get("fp", 0),
            "fn": cat_metrics.get("fn", 0),
            "gt_categories": cat_metrics.get("gt_categories", [gt_category]),
        })

    return runs


def load_p6_metrics(p6_dir: Path) -> list[dict]:
    """Load P6 iAudit metrics."""
    runs = []
    if not p6_dir.exists():
        print(f"  Warning: P6 directory not found: {p6_dir}")
        return runs

    for contract_dir in p6_dir.iterdir():
        if not contract_dir.is_dir():
            continue

        parsed_file = contract_dir / "run_0_parsed.json"
        if not parsed_file.exists():
            continue

        try:
            with open(parsed_file) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        contract_name = contract_dir.name
        gt_category = get_gt_category_from_filename(contract_name)

        metrics = data.get("metrics", {}).get("category_metrics", {})

        runs.append({
            "prompt": "P6",
            "contract_id": contract_name,
            "gt_category": gt_category,
            "tp": metrics.get("tp", 0),
            "fp": metrics.get("fp", 0),
            "fn": metrics.get("fn", 0),
            "gt_categories": metrics.get("gt_categories", [gt_category]),
        })

    return runs


def calculate_per_category_metrics(runs: list[dict]) -> dict:
    """Calculate per-category metrics grouped by prompt."""
    # Structure: {prompt: {category: {tp, fp, fn, n_contracts}}}
    metrics = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_contracts": 0}))

    for run in runs:
        prompt = run.get("prompt", "unknown")
        gt_cat = run.get("gt_category", "unknown")

        metrics[prompt][gt_cat]["tp"] += run.get("tp", 0)
        metrics[prompt][gt_cat]["fp"] += run.get("fp", 0)
        metrics[prompt][gt_cat]["fn"] += run.get("fn", 0)
        metrics[prompt][gt_cat]["n_contracts"] += 1

    # Calculate precision, recall, F1
    results = {}
    for prompt, categories in metrics.items():
        results[prompt] = {}
        for category, counts in categories.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[prompt][category] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision * 100, 1),
                "recall": round(recall * 100, 1),
                "f1": round(f1 * 100, 1),
                "n_contracts": counts["n_contracts"]
            }

    return results


def generate_latex_table(results: dict) -> str:
    """Generate LaTeX table for RQ2."""
    prompts = ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]

    categories = [
        ("reentrancy", "Reentrancy", 32),
        ("access_control", "Access Control", 21),
        ("arithmetic", "Arithmetic", 23),
        ("unchecked_low_level_calls", "Unchecked Calls", 75),
        ("denial_of_service", "DoS", 7),
        ("bad_randomness", "Bad Randomness", 31),
        ("time_manipulation", "Time Manip.", 7),
        ("front_running", "Front Running", 7),
        ("short_addresses", "Short Addr.", 1),
        ("other", "Other", 3),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Per-category F1 scores (\%) across prompt strategies. Best per category in bold. P4 excluded due to content filtering (only 134 contracts).}")
    lines.append(r"\label{tab:rq2_category}")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Category} & \textbf{N} & \textbf{P0} & \textbf{P1} & \textbf{P2} & \textbf{P3} & \textbf{P5} & \textbf{P6}$^\dagger$ \\")
    lines.append(r"\midrule")

    for cat_key, cat_label, n_expected in categories:
        row_values = []
        f1_scores = []

        # Only include P0, P1, P2, P3, P5, P6 (skip P4)
        for prompt in ["P0", "P1", "P2", "P3", "P5", "P6"]:
            if prompt in results and cat_key in results[prompt]:
                f1 = results[prompt][cat_key]["f1"]
                f1_scores.append((prompt, f1))
            else:
                f1_scores.append((prompt, None))

        # Find max F1 for bolding (exclude None values)
        valid_f1s = [(p, f) for p, f in f1_scores if f is not None and f > 0]
        max_f1 = max([f for _, f in valid_f1s]) if valid_f1s else 0

        # Format row values
        for prompt, f1 in f1_scores:
            if f1 is None:
                # P6 doesn't support front_running, short_addresses, other
                if prompt == "P6" and cat_key in ["front_running", "short_addresses", "other"]:
                    row_values.append("n/a")
                else:
                    row_values.append("--")
            elif f1 == max_f1 and f1 > 0:
                row_values.append(f"\\textbf{{{f1:.1f}}}")
            else:
                row_values.append(f"{f1:.1f}")

        # Italicize P6-unsupported categories
        if cat_key in ["front_running", "short_addresses", "other"]:
            lines.append(f"\\textit{{{cat_label}}} & {n_expected} & {' & '.join(row_values)} \\\\")
        else:
            lines.append(f"{cat_label} & {n_expected} & {' & '.join(row_values)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{9}{l}{\scriptsize $^\dagger$P6 excludes categories not in Qian taxonomy (italicized rows)} \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    base_dir = Path("/home/chefmike/Desktop/phd/sc_prompt_eval_skeleton")
    output_base = base_dir / "outputs/experiment_v1/smartbugs_curated/o4-mini"

    all_runs = []

    # Load P0-P5
    print("Loading P0-P5 results...")
    for dir_name, label in PROMPT_DIRS.items():
        prompt_dir = output_base / dir_name
        runs = load_prompt_metrics(prompt_dir, label)
        print(f"  {label}: {len(runs)} contracts")
        all_runs.extend(runs)

    # Load P6
    print("Loading P6 results...")
    p6_dir = base_dir / "outputs/experiment_v1/smartbugs_curated/iaudit_local/P6_iaudit"
    p6_runs = load_p6_metrics(p6_dir)
    print(f"  P6: {len(p6_runs)} contracts")
    all_runs.extend(p6_runs)

    print(f"\nTotal runs: {len(all_runs)}")

    # Calculate metrics
    results = calculate_per_category_metrics(all_runs)

    # Print summary
    print("\n" + "="*80)
    print("RQ2 Per-Category Metrics Summary")
    print("="*80)

    for prompt in sorted(results.keys()):
        print(f"\n{prompt}:")
        total_tp, total_fp, total_fn = 0, 0, 0
        for cat in sorted(results[prompt].keys()):
            m = results[prompt][cat]
            print(f"  {cat:30s}: F1={m['f1']:5.1f}%, P={m['precision']:5.1f}%, R={m['recall']:5.1f}% (n={m['n_contracts']}, TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
            total_tp += m['tp']
            total_fp += m['fp']
            total_fn += m['fn']

        # Overall for this prompt
        overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
        print(f"  {'OVERALL':30s}: F1={overall_f1*100:5.1f}%, P={overall_p*100:5.1f}%, R={overall_r*100:5.1f}%")

    # Generate LaTeX
    print("\n" + "="*80)
    print("LaTeX Table")
    print("="*80)
    latex = generate_latex_table(results)
    print(latex)

    # Save results
    output_file = base_dir / "outputs/experiment_v1/rq2_per_category_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {output_file}")

    # Save LaTeX table
    latex_file = base_dir / "outputs/experiment_v1/rq2_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex)
    print(f"Saved LaTeX table to {latex_file}")


if __name__ == "__main__":
    main()
