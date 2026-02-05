from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .metrics import ScoreSummary, prf1, match_by_category


@dataclass
class ContractResult:
    """Results for a single contract across all runs."""
    contract_id: str
    dataset: str
    model: str
    prompt: str
    runs: List[dict] = field(default_factory=list)

    @property
    def avg_finding_count(self) -> float:
        counts = [r.get("finding_count", 0) for r in self.runs if "error" not in r]
        return np.mean(counts) if counts else 0.0

    @property
    def avg_cost(self) -> float:
        costs = [r.get("cost_usd", 0) for r in self.runs if r.get("cost_usd")]
        return np.mean(costs) if costs else 0.0

    @property
    def avg_latency(self) -> float:
        latencies = [r.get("latency_s", 0) for r in self.runs if r.get("latency_s")]
        return np.mean(latencies) if latencies else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.get("tokens", 0) for r in self.runs if r.get("tokens"))


@dataclass
class AggregatedResults:
    """Aggregated results across experiment."""
    by_model_prompt: Dict[Tuple[str, str], List[ContractResult]] = field(default_factory=dict)
    by_vuln_type: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add_result(self, result: ContractResult):
        key = (result.model, result.prompt)
        if key not in self.by_model_prompt:
            self.by_model_prompt[key] = []
        self.by_model_prompt[key].append(result)


def load_experiment_results(output_dir: str | Path) -> Dict:
    """Load results from experiment output directory."""
    output_dir = Path(output_dir)
    results_file = output_dir / "results_summary.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    return json.loads(results_file.read_text())


def load_parsed_findings(output_dir: str | Path, dataset: str, model: str, prompt: str, contract_id: str) -> List[List[dict]]:
    """Load all parsed findings for a contract across runs."""
    output_dir = Path(output_dir)
    contract_dir = output_dir / dataset / model / prompt / contract_id.replace("/", "_")

    findings_per_run = []
    run_idx = 0
    while True:
        parsed_file = contract_dir / f"run_{run_idx}_parsed.json"
        if not parsed_file.exists():
            break
        data = json.loads(parsed_file.read_text())
        findings_per_run.append(data.get("findings", []))
        run_idx += 1

    return findings_per_run


def aggregate_by_model_prompt(results: Dict) -> pd.DataFrame:
    """Aggregate results by (model, prompt) combination."""
    rows = []

    # Group runs by (dataset, model, prompt, contract)
    grouped = defaultdict(list)
    for run in results.get("runs", []):
        if "error" in run:
            continue
        key = (run["dataset"], run["model"], run["prompt"], run["contract_id"])
        grouped[key].append(run)

    # Aggregate by (model, prompt)
    model_prompt_stats = defaultdict(lambda: {
        "contracts": set(),
        "total_findings": 0,
        "total_cost": 0.0,
        "total_tokens": 0,
        "total_latency": 0.0,
        "api_calls": 0,
        "parse_errors": 0,
    })

    for (dataset, model, prompt, contract_id), runs in grouped.items():
        key = (model, prompt)
        stats = model_prompt_stats[key]
        stats["contracts"].add(contract_id)
        for run in runs:
            stats["total_findings"] += run.get("finding_count", 0)
            stats["total_cost"] += run.get("cost_usd", 0) or 0
            stats["total_tokens"] += run.get("tokens", 0) or 0
            stats["total_latency"] += run.get("latency_s", 0) or 0
            stats["api_calls"] += 1
            stats["parse_errors"] += run.get("parse_error_count", 0)

    for (model, prompt), stats in model_prompt_stats.items():
        n_contracts = len(stats["contracts"])
        n_calls = stats["api_calls"]
        rows.append({
            "model": model,
            "prompt": prompt,
            "contracts": n_contracts,
            "api_calls": n_calls,
            "avg_findings_per_contract": stats["total_findings"] / n_contracts if n_contracts else 0,
            "total_cost_usd": stats["total_cost"],
            "avg_cost_per_contract": stats["total_cost"] / n_contracts if n_contracts else 0,
            "total_tokens": stats["total_tokens"],
            "avg_tokens_per_call": stats["total_tokens"] / n_calls if n_calls else 0,
            "avg_latency_s": stats["total_latency"] / n_calls if n_calls else 0,
            "parse_error_rate": stats["parse_errors"] / n_calls if n_calls else 0,
        })

    return pd.DataFrame(rows)






