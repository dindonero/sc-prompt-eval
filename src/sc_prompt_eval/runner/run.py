"""Main experiment runner with specialized backends for different prompt strategies.

Prompt Type Detection and Routing (7 prompts):
- P0-P3: Simple single-call prompts (baseline, taxonomy, few-shot, structured CoT)
- P4: SmartGuard (RAG + CoT + Self-check) - Ding2025SmartGuard
- P5: Tool-augmented with Slither (Sun2023GPTScan, Ince2025GenDetect)
- P6: Multi-agent debate (Wei2025LLMSmartAudit, Ma2024iAudit)
"""
from __future__ import annotations

import json
import hashlib
import logging
import random
import re
import signal
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from tqdm import tqdm
from rich.console import Console

from ..config import ExperimentSpec, DatasetSpec
from ..datasets.base import ContractItem, Dataset
from ..datasets.smartbugs_curated import SmartBugsCurated
from ..datasets.benign import BenignContracts
from ..io import load_experiment_config
from ..parsing.findings import parse_findings, Finding
from ..parsing.json_utils import extract_json_balanced, extract_json_object, safe_json_loads
from ..prompts.registry import PromptRegistry
from ..models import make_adapter, LLMAdapter, LLMResponse
from ..scoring.metrics import (
    ScoreSummary, compute_metrics, compute_instance_metrics,
    get_category_from_text, normalize_category, DASP_CATEGORIES
)

# Import specialized runners
from .multiagent import run_smartaudit, SmartAuditResult, extract_final_findings
from .iaudit_runner import run_iaudit_pipeline, iAuditConfig, iAuditResult

console = Console()


def _print_verbose_response(response: LLMResponse, label: str = "API Response") -> None:
    """Print API response details for verbose mode."""
    console.print(f"\n      [cyan]━━━ {label} ━━━[/cyan]")
    console.print(f"      [dim]Tokens: {response.total_tokens} | Cost: ${response.cost_usd:.4f} | Latency: {response.latency_s:.2f}s[/dim]")
    # Print response text with truncation for very long responses
    text = response.text or ""
    if len(text) > 2000:
        console.print(f"      [white]{text[:2000]}[/white]")
        console.print(f"      [dim]... (truncated, {len(text)} chars total)[/dim]")
    else:
        console.print(f"      [white]{text}[/white]")
    console.print(f"      [cyan]━━━━━━━━━━━━━━━━━━━━━[/cyan]\n")


# Lazy import for Slither (may not be installed)
# Thread-safe with double-check locking
_slither_runner = None
_slither_timeout = 120  # Default timeout
_slither_lock = threading.Lock()

# Lazy import for function filter
# Thread-safe with double-check locking
_function_filter = None
_function_filter_lock = threading.Lock()

# Lazy-loaded scenarios for P5 (GPTScan-style)
# Thread-safe with double-check locking
_p5_scenarios = None
_p5_scenarios_lock = threading.Lock()


def get_p5_scenarios() -> dict:
    """Load GPTScan-style scenarios for P5 tool-augmented prompts. Thread-safe.

    Returns dict with 'scenarios' (list) and 'logic_checks' (list).
    Falls back to empty lists if file not found.
    """
    global _p5_scenarios
    if _p5_scenarios is not None:
        return _p5_scenarios

    with _p5_scenarios_lock:
        # Double-check after acquiring lock
        if _p5_scenarios is not None:
            return _p5_scenarios

        scenarios_file = Path(__file__).parent.parent.parent.parent / "data" / "scenarios.json"
        if scenarios_file.exists():
            try:
                data = json.loads(scenarios_file.read_text())
                _p5_scenarios = {
                    "scenarios": data.get("scenarios", []),
                    "logic_checks": data.get("logic_checks", []),
                }
                console.print(f"[green]Loaded {len(_p5_scenarios['scenarios'])} scenarios for P5[/green]")
            except (json.JSONDecodeError, IOError) as e:
                console.print(f"[yellow]Warning: Failed to load scenarios.json: {e}[/yellow]")
                _p5_scenarios = {"scenarios": [], "logic_checks": []}
        else:
            console.print("[yellow]Warning: scenarios.json not found, using empty scenarios[/yellow]")
            _p5_scenarios = {"scenarios": [], "logic_checks": []}

        return _p5_scenarios


def get_slither_runner(timeout: int = None):
    """Lazy load Slither runner to avoid import errors if not installed. Thread-safe.

    Args:
        timeout: Optional timeout override for Slither execution
    """
    global _slither_runner, _slither_timeout

    if _slither_runner is None:
        with _slither_lock:
            # Thread-safe timeout update and initialization (inside lock)
            if timeout is not None:
                _slither_timeout = timeout
            # Double-check after acquiring lock
            if _slither_runner is None:
                try:
                    from ..tools.slither_runner import SlitherRunner, check_slither_available
                    available, message = check_slither_available()
                    if available:
                        _slither_runner = SlitherRunner(timeout=_slither_timeout)
                        console.print(f"[green]Slither: {message}[/green]")
                    else:
                        console.print(f"[yellow]Slither: {message}[/yellow]")
                        _slither_runner = False
                except ImportError:
                    console.print("[yellow]Warning: Slither module not available. P5 will run without static analysis.[/yellow]")
                    _slither_runner = False
    return _slither_runner if _slither_runner else None


def get_function_filter():
    """Lazy load function filter for GPTScan-style pre-filtering. Thread-safe.

    Returns FunctionFilter instance or None if import fails.
    """
    global _function_filter
    if _function_filter is None:
        with _function_filter_lock:
            # Double-check after acquiring lock
            if _function_filter is None:
                try:
                    from ..tools.function_filter import FunctionFilter
                    scenarios = get_p5_scenarios().get("scenarios", [])
                    _function_filter = FunctionFilter(scenarios)
                    console.print(f"[green]Function filter: Loaded with {len(scenarios)} scenarios[/green]")
                except ImportError as e:
                    console.print(f"[yellow]Warning: Function filter not available: {e}[/yellow]")
                    _function_filter = False
    return _function_filter if _function_filter else None


@dataclass
class RunResult:
    """Result from a single contract/run execution."""
    dataset_name: str
    model_name: str
    prompt_id: str
    prompt_type: str
    contract_id: str
    run_idx: int
    findings: List[Finding]
    cost_usd: float
    tokens: int
    api_calls: int
    latency_s: float
    metrics: Dict
    extra_metadata: Dict
    ground_truth: List[Dict]
    error: Optional[str] = None


class ThreadSafeResultsAccumulator:
    """Thread-safe accumulator for experiment results.

    Allows multiple worker threads to safely add results that are
    aggregated into the final experiment summary.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._totals = {
            "contracts": 0,
            "api_calls": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "instance_tp": 0,
            "instance_fp": 0,
            "instance_fn": 0,
            "contracts_with_line_annotations": 0,
            "skipped_runs": 0,
            "parse_failed_runs": 0,
        }
        self._runs: List[Dict] = []
        self._contracts_counted: Set[str] = set()  # Track unique contracts

    def add_result(self, result: RunResult):
        """Thread-safe result addition."""
        with self._lock:
            # Only count each contract once
            contract_key = f"{result.dataset_name}:{result.model_name}:{result.prompt_id}:{result.contract_id}"
            if contract_key not in self._contracts_counted:
                self._totals["contracts"] += 1
                self._contracts_counted.add(contract_key)

            # Update totals
            self._totals["api_calls"] += result.api_calls
            self._totals["total_cost_usd"] += result.cost_usd
            self._totals["total_tokens"] += result.tokens

            # Category-level metrics
            self._totals["total_tp"] += result.metrics.get("tp", 0)
            self._totals["total_fp"] += result.metrics.get("fp", 0)
            self._totals["total_fn"] += result.metrics.get("fn", 0)

            # Instance-level metrics
            inst_metrics = result.metrics.get("instance_metrics", {})
            if inst_metrics.get("has_line_annotations"):
                self._totals["instance_tp"] += inst_metrics.get("tp", 0)
                self._totals["instance_fp"] += inst_metrics.get("fp", 0)
                self._totals["instance_fn"] += inst_metrics.get("fn", 0)
                # Only count contracts_with_line_annotations once per contract
                if result.run_idx == 0:
                    self._totals["contracts_with_line_annotations"] += 1

            # Build run record
            run_record = {
                "dataset": result.dataset_name,
                "model": result.model_name,
                "prompt": result.prompt_id,
                "prompt_type": result.prompt_type,
                "contract_id": result.contract_id,
                "run_idx": result.run_idx,
                "finding_count": len(result.findings),
                "cost_usd": result.cost_usd,
                "tokens": result.tokens,
                "api_calls": result.api_calls,
                "latency_s": result.latency_s,
                "precision": result.metrics.get("precision"),
                "recall": result.metrics.get("recall"),
                "f1": result.metrics.get("f1"),
                "tp": result.metrics.get("tp", 0),
                "fp": result.metrics.get("fp", 0),
                "fn": result.metrics.get("fn", 0),
                "gt_category": result.ground_truth[0]["category"] if result.ground_truth else None,
                "pred_categories": result.metrics.get("pred_categories", []),
                "parse_error_count": len(result.extra_metadata.get("parse_errors", [])),
            }

            # Add specialized metadata based on prompt type
            if result.prompt_type == 'smartaudit':
                run_record["agent_costs"] = result.extra_metadata.get("agent_costs", {})
                run_record["mode"] = result.extra_metadata.get("mode", "BA")
                run_record["role_exchanges"] = result.extra_metadata.get("role_exchanges_performed", 0)
            elif result.prompt_type == 'smartguard':
                run_record["patterns_retrieved"] = result.extra_metadata.get("num_patterns_retrieved", 0)
            elif result.prompt_type == 'tool_augmented':
                run_record["slither_available"] = result.extra_metadata.get("slither_available", False)
            elif result.prompt_type == 'icl_single_type':
                run_record["types_with_vulnerabilities"] = result.extra_metadata.get("types_with_vulnerabilities", [])
                run_record["vulnerability_types_checked"] = len(result.extra_metadata.get("vulnerability_types_checked", []))

            self._runs.append(run_record)

    def add_skip(self):
        """Thread-safe skip counter increment."""
        with self._lock:
            self._totals["skipped_runs"] += 1

    def add_parse_failure(self, result: RunResult):
        """Thread-safe parse failure counter increment."""
        with self._lock:
            self._totals["parse_failed_runs"] += 1
            # Still track cost/tokens but don't count in metrics
            self._totals["api_calls"] += result.api_calls
            self._totals["total_cost_usd"] += result.cost_usd
            self._totals["total_tokens"] += result.tokens
            self._runs.append({
                "dataset": result.dataset_name,
                "model": result.model_name,
                "prompt": result.prompt_id,
                "prompt_type": result.prompt_type,
                "contract_id": result.contract_id,
                "run_idx": result.run_idx,
                "error": "parse_failed",
                "parse_errors": result.extra_metadata.get("parse_errors", []),
            })

    def add_error(self, result: RunResult):
        """Thread-safe error result addition."""
        with self._lock:
            self._runs.append({
                "dataset": result.dataset_name,
                "model": result.model_name,
                "prompt": result.prompt_id,
                "prompt_type": result.prompt_type,
                "contract_id": result.contract_id,
                "run_idx": result.run_idx,
                "error": result.error,
            })

    def get_totals(self) -> Dict:
        """Thread-safe access to totals."""
        with self._lock:
            return dict(self._totals)

    def get_results(self) -> Dict:
        """Thread-safe access to full results."""
        with self._lock:
            return {
                "totals": dict(self._totals),
                "runs": list(self._runs),
            }


def _is_affirmative(value) -> bool:
    """Check if a value represents an affirmative response (case-insensitive).

    Handles various LLM response formats:
    - "yes", "Yes", "YES"
    - "true", "True", "TRUE"
    - "y", "Y"
    - True (boolean)
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("yes", "y", "true", "1")


def parse_icl_single_response(text: str, vuln_type: str) -> Dict:
    """Parse single-type ICL response from LLM.

    Args:
        text: Raw LLM response text (can be None)
        vuln_type: The vulnerability type being checked

    Returns:
        Dict with parsed response or empty result on failure
    """
    # Handle None or empty input
    if not text:
        return {
            "contains_vulnerability": "no",
            "vulnerability_type": vuln_type,
            "certainty_score": 0,
            "parse_error": True
        }

    # Try to extract JSON from the response
    try:
        # 1. First try: parse the whole text directly (handles compact JSON)
        stripped = text.strip()
        if stripped.startswith('{'):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass

        # 2. Look for JSON block in markdown code fence
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Extract balanced JSON object (handles nested braces)
        json_str = extract_json_balanced(text)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 4. Last resort: try parsing the whole text
        return json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError) as e:
        # Log parse failure for visibility (these become silent FNs otherwise)
        preview = text[:150] + '...' if len(text) > 150 else text
        logger.warning(f"JSON parse failed for {vuln_type}: {e}. Preview: {preview}")
        return {
            "contains_vulnerability": "no",
            "vulnerability_type": vuln_type,
            "certainty_score": 0,
            "parse_error": True
        }


def detect_prompt_type(prompt_id: str, template_path: str) -> str:
    """
    Detect prompt type from ID or template path.

    Prompt taxonomy (7 prompt types):
    - P0: Zero-shot baseline
    - P1: Few-shot in-context learning (ICL)
    - P2: Structured chain-of-thought reasoning
    - P3: SmartGuard (RAG + CoT + Self-check) - Zhang et al. 2024
    - P4: Tool-augmented GPTScan (Slither) - Sun et al. 2024
    - P5: LLM-SmartAudit multi-agent - Wei et al. 2024
    - P6: iAudit (fine-tuned Detector + Reasoner + Ranker-Critic) - Ma et al. 2024

    Returns one of: 'simple', 'icl_single_type', 'smartguard', 'tool_augmented', 'smartaudit', 'p6_iaudit', 'skip'
    """
    # Check by prompt ID first
    prompt_id_lower = prompt_id.lower()
    template_lower = template_path.lower()

    # P1 ICL single-type variant (per-vulnerability-type prompts)
    if 'p1_icl' in prompt_id_lower or 'icl_single' in template_lower:
        return 'icl_single_type'

    # P3: SmartGuard-style (RAG + CoT + Self-check in single prompt) - Zhang et al. 2024
    if 'p3' in prompt_id_lower and 'smartguard' in prompt_id_lower:
        return 'smartguard'
    if 'smartguard' in template_lower:
        return 'smartguard'

    # P4: Tool-augmented GPTScan (Slither) - Sun et al. 2024
    if 'p4' in prompt_id_lower or 'tool_augmented' in template_lower or 'tool-augmented' in prompt_id_lower:
        return 'tool_augmented'
    if 'stage' in template_lower and ('scenario' in template_lower or 'property' in template_lower or 'extraction' in template_lower):
        return 'tool_augmented'

    # P6: iAudit pipeline (Ma et al., 2024)
    # Check BEFORE P5 SmartAudit to catch 'p6_iaudit' specifically
    if 'p6' in prompt_id_lower and 'iaudit' in prompt_id_lower:
        return 'p6_iaudit'
    if 'iaudit' in template_lower:
        return 'p6_iaudit'

    # P5: LLM-SmartAudit multi-agent (Wei et al. 2024)
    if 'p5' in prompt_id_lower or 'multiagent' in prompt_id_lower or 'multi-agent' in prompt_id_lower or 'smartaudit' in prompt_id_lower:
        # P5 uses multiple templates, detect by project_manager or auditor being the entry point
        if 'project_manager' in template_lower or 'auditor' in template_lower:
            return 'smartaudit'
        # If it's counselor/expert/role_exchange/ba_mode/ta_mode, skip (handled by SmartAudit runner)
        if any(x in template_lower for x in ['counselor', 'expert', 'role_exchange', 'ba_mode', 'ta_mode']):
            return 'skip'
        return 'smartaudit'

    # P0-P2, or unknown: Simple single-call
    return 'simple'


def compute_metrics_for_run(
    predictions: List[Finding],
    ground_truth: List[dict],
    line_tolerance: int = 3,
) -> Dict:
    """Compute both category-level and instance-level metrics for a single run.

    This function returns TWO sets of metrics:
    1. Category-level: Did we detect the right vulnerability TYPES? (original behavior)
    2. Instance-level: Did we find each vulnerability at the right LOCATION AND TYPE?

    Category-level counts unique categories (e.g., 2 reentrancy vulns = 1 TP if detected).
    Instance-level counts each vulnerability instance with BOTH category AND line matching.

    IMPORTANT: Instance-level matching requires BOTH:
    - Same vulnerability category/type
    - Overlapping or nearby line numbers (within tolerance)
    This prevents false positives from different vulnerability types at the same location.

    Args:
        predictions: List of Finding objects from LLM
        ground_truth: List of ground truth dicts with 'category' and optionally 'lines'
        line_tolerance: Max line distance to consider a location match (default 3)

    Returns:
        Dict with both 'category_metrics' and 'instance_metrics' sub-dicts
    """
    # Handle empty ground truth (benign contracts)
    # Still extract prediction info for false positive analysis
    if not ground_truth:
        pred_categories = []
        unknown_predictions = 0

        for pred in predictions:
            cat = normalize_category(pred.category)
            if cat in DASP_CATEGORIES:
                pred_categories.append(cat)
            else:
                unknown_predictions += 1

        # Deduplicate for category-level
        pred_categories_unique = list(set(pred_categories))

        empty_metrics = {
            "precision": 0.0 if predictions else None,  # 0 precision if any FP
            "recall": None,  # Undefined with no ground truth
            "f1": None,
            "tp": 0, "fp": len(predictions), "fn": 0,
        }
        return {
            "category_metrics": {
                **empty_metrics,
                "pred_categories": pred_categories_unique,
                "gt_categories": [],
                "matched_categories": [],
                "missed_categories": [],
                "spurious_categories": pred_categories_unique,  # All predictions are spurious when no GT
                "unknown_predictions": unknown_predictions,
            },
            "instance_metrics": {
                **empty_metrics,
                "matched_instances": [],
                "unmatched_predictions": [
                    {"category": p.category, "title": p.title}
                    for p in predictions
                ],
                "unmatched_ground_truth": [],
            },
            # Backward compatibility: keep top-level metrics (category-level)
            **empty_metrics,
            "pred_categories": pred_categories_unique, "gt_categories": [],
        }

    # === Category-Level Metrics (original behavior) ===
    cat_result = compute_metrics(predictions, ground_truth, strict=True)

    # Extract ground truth categories for the output
    gt_categories = set()
    for g in ground_truth:
        if g.get("category"):
            cat = normalize_category(g["category"])
            if cat != 'unknown':
                gt_categories.add(cat)

    # Extract predicted categories for the output
    pred_categories = set(cat_result.matched_categories) | set(cat_result.spurious_categories)

    category_metrics = {
        "precision": cat_result.precision,
        "recall": cat_result.recall,
        "f1": cat_result.f1,
        "tp": cat_result.true_positives,
        "fp": cat_result.false_positives,
        "fn": cat_result.false_negatives,
        "pred_categories": list(pred_categories),
        "gt_categories": list(gt_categories),
        "matched_categories": cat_result.matched_categories,
        "missed_categories": cat_result.missed_categories,
        "spurious_categories": cat_result.spurious_categories,
        "unknown_predictions": cat_result.unknown_predictions,
    }

    # === Instance-Level Metrics (line-level matching with category requirement) ===
    # IMPORTANT: Instance matching requires BOTH:
    # 1. Same vulnerability category/type
    # 2. Overlapping/nearby line numbers (within tolerance)
    # This prevents false matches from different vulnerability types at the same line.
    has_line_annotations = any(g.get("lines") for g in ground_truth)

    if has_line_annotations:
        inst_result = compute_instance_metrics(
            predictions, ground_truth, line_tolerance=line_tolerance
        )
        instance_metrics = {
            "precision": inst_result.precision,
            "recall": inst_result.recall,
            "f1": inst_result.f1,
            "tp": inst_result.true_positives,
            "fp": inst_result.false_positives,
            "fn": inst_result.false_negatives,
            "matched_instances": inst_result.matched_instances,
            "unmatched_predictions": inst_result.unmatched_predictions,
            "unmatched_ground_truth": inst_result.unmatched_ground_truth,
            "has_line_annotations": True,
        }
    else:
        # No line annotations - instance metrics not meaningful
        instance_metrics = {
            "precision": None, "recall": None, "f1": None,
            "tp": None, "fp": None, "fn": None,
            "matched_instances": [],
            "unmatched_predictions": [],
            "unmatched_ground_truth": [],
            "has_line_annotations": False,
        }

    return {
        # Nested metrics for clarity
        "category_metrics": category_metrics,
        "instance_metrics": instance_metrics,
        # Backward compatibility: keep top-level metrics (category-level)
        "precision": cat_result.precision,
        "recall": cat_result.recall,
        "f1": cat_result.f1,
        "tp": cat_result.true_positives,
        "fp": cat_result.false_positives,
        "fn": cat_result.false_negatives,
        "pred_categories": list(pred_categories),
        "gt_categories": list(gt_categories),
        "matched_categories": cat_result.matched_categories,
        "missed_categories": cat_result.missed_categories,
        "spurious_categories": cat_result.spurious_categories,
        "unknown_predictions": cat_result.unknown_predictions,
    }

# Dataset registry
DATASET_REGISTRY = {
    "smartbugs_curated": SmartBugsCurated,
    "benign_contracts": BenignContracts,
}


def get_dataset(spec: DatasetSpec) -> Dataset:
    """Get dataset instance from spec.

    Args:
        spec: Dataset specification with kind, path, and name.

    Returns:
        Dataset instance.
    """
    if spec.kind not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset kind: {spec.kind}. Supported: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[spec.kind](spec.path, name=spec.name)


def ensure_dir(p: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_loc(source: str) -> Dict[str, int]:
    """Compute lines of code metrics for a contract.

    Returns:
        Dict with loc_total, loc_code, loc_comments, loc_blank
    """
    lines = source.split('\n')
    total = len(lines)
    blank = sum(1 for line in lines if not line.strip())
    comments = sum(1 for line in lines if line.strip().startswith('//') or line.strip().startswith('/*') or line.strip().startswith('*'))
    code = total - blank - comments
    return {
        "loc_total": total,
        "loc_code": code,
        "loc_comments": comments,
        "loc_blank": blank,
    }


def check_run_exists(
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    prompt_id: str,
    contract_id: str,
    run_idx: int,
) -> bool:
    """Check if a run output already exists (for resumable experiments).

    Returns:
        True if the parsed output file exists
    """
    run_dir = output_dir / dataset_name / model_name / prompt_id / contract_id.replace("/", "_")
    parsed_path = run_dir / f"run_{run_idx}_parsed.json"
    return parsed_path.exists()


def save_run_output(
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    prompt_id: str,
    contract_id: str,
    run_idx: int,
    response: Optional[LLMResponse],
    findings: List[Finding],
    parse_errors: List[str],
    ground_truth: List[dict] = None,
    metrics: Dict = None,
    extra_data: Dict = None,
    contract_source: str = None,
    rendered_prompt: str = None,
    model_info: Dict = None,
) -> None:
    """Save outputs for a single run.

    Args:
        extra_data: Additional data for specialized runners (multiagent, RAG, etc.)
        contract_source: Original contract source for LOC computation
        rendered_prompt: The rendered prompt (for reproducibility)
        model_info: Model configuration (name, provider, params) for reproducibility
    """
    import hashlib

    run_dir = output_dir / dataset_name / model_name / prompt_id / contract_id.replace("/", "_")
    ensure_dir(run_dir)

    # Save raw response (if available - specialized runners may not have single response)
    raw_path = run_dir / f"run_{run_idx}_raw.json"
    if response:
        raw_data = {
            "text": response.text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "cost_usd": response.cost_usd,
            "latency_s": response.latency_s,
        }
    else:
        raw_data = {"note": "Specialized runner - see extra_data for details"}

    # Add rendered prompt with hash for reproducibility/dedup
    if rendered_prompt:
        prompt_hash = hashlib.sha256(rendered_prompt.encode()).hexdigest()[:16]
        raw_data["rendered_prompt"] = rendered_prompt
        raw_data["prompt_hash"] = prompt_hash

    # Add model info for reproducibility
    if model_info:
        raw_data["model_info"] = model_info

    if extra_data:
        raw_data["extra"] = extra_data
    raw_path.write_text(json.dumps(raw_data, indent=2, default=str))

    # Save parsed findings with metrics and LOC
    parsed_path = run_dir / f"run_{run_idx}_parsed.json"
    parsed_data = {
        "findings": [f.model_dump() for f in findings],
        "parse_errors": parse_errors,
        "finding_count": len(findings),
        "ground_truth": ground_truth or [],
        "metrics": metrics or {},
    }

    # Add LOC metadata if contract source provided
    if contract_source:
        parsed_data["loc"] = compute_loc(contract_source)

    if extra_data:
        parsed_data["runner_metadata"] = extra_data
    parsed_path.write_text(json.dumps(parsed_data, indent=2, default=str))

    # Optionally save rendered prompt for debugging
    if rendered_prompt:
        prompt_path = run_dir / f"run_{run_idx}_prompt.txt"
        prompt_path.write_text(rendered_prompt)


def run_gptscan_multistage(
    adapter: "LLMAdapter",
    prompt_registry: "PromptRegistry",
    contract_source: str,
    scenarios: List[Dict],
    candidate_functions: List[str],
    slither_output: str = "",
    slither_instance: Any = None,
    enable_static_confirmation: bool = True,
    verbose: bool = False,
    stage1_template: str = "p4_stage1_scenario.j2",
    stage2_template: str = "p4_stage2_property.j2",
    stage3_template: str = "p4_stage3_extraction.j2",
) -> Tuple[List[Finding], float, int, Dict]:
    """
    Run GPTScan-style multi-stage pipeline.

    Stage 1: Scenario matching (per function) - batch all scenarios
    Stage 2: Property verification (for matched scenarios)
    Stage 3: Key variable/statement extraction (for confirmed vulnerabilities)
    Stage 4: Static confirmation (if enabled)

    Args:
        adapter: LLM adapter for generation
        prompt_registry: Registry for loading prompt templates
        contract_source: Solidity source code
        scenarios: List of scenario definitions from scenarios.json
        candidate_functions: List of function names to analyze
        slither_output: Optional Slither output for reference
        slither_instance: Optional Slither object for static confirmation
        enable_static_confirmation: Whether to run static confirmation (Step 6)
        verbose: Print detailed output
        stage1_template: Template for Stage 1 (scenario matching)
        stage2_template: Template for Stage 2 (property verification)
        stage3_template: Template for Stage 3 (key variable extraction)

    Returns:
        Tuple of (findings, total_cost, total_tokens, metadata)
    """
    total_cost = 0.0
    total_tokens = 0
    api_calls = 0
    findings = []

    # Extract function code from source
    func_filter = get_function_filter()
    if func_filter:
        all_functions = func_filter.extract_functions(contract_source)
        # Use name:start_line as key to handle overloaded functions
        # (same name, different parameters) without collision
        func_map = {f.name: f for f in all_functions}  # Primary lookup by name
        func_map_by_line = {f"{f.name}:{f.start_line}": f for f in all_functions}  # Unique lookup
        # Also build a list-based lookup for overloaded functions
        func_list_map = defaultdict(list)
        for f in all_functions:
            func_list_map[f.name].append(f)
    else:
        func_map = {}
        func_map_by_line = {}
        func_list_map = defaultdict(list)

    stage_metadata = {
        "stage1_results": {},
        "stage2_results": {},
        "stage3_results": {},
        "functions_analyzed": 0,
        "scenarios_matched": 0,
        "properties_confirmed": 0,
        # Raw LLM outputs for debugging/analysis
        "raw_llm_outputs": {
            "stage1": {},  # func_name -> raw response text
            "stage2": {},  # func_name -> {scenario_id -> raw response text}
            "stage3": {},  # func_name -> {scenario_id -> raw response text}
        },
    }

    # Process each candidate function
    # Handle overloaded functions by processing ALL functions with matching name
    for func_name in candidate_functions:
        # Get all functions with this name (handles overloads)
        matching_funcs = func_list_map.get(func_name, [])
        if not matching_funcs:
            continue

        for func_info in matching_funcs:
            # Use unique key for tracking (name:line) to handle overloads
            func_key = f"{func_info.name}:{func_info.start_line}"
            stage_metadata["functions_analyzed"] += 1
            function_code = func_info.full_text

            # STAGE 1: Scenario matching (batch all scenarios in one call)
            stage1_prompt = prompt_registry.render(
                stage1_template,
                contract_source=function_code,  # Pass function_code as contract_source for registry compatibility
                function_code=function_code,
                scenarios=scenarios,
            )

            response1 = adapter.generate(stage1_prompt)
            total_cost += response1.cost_usd or 0
            total_tokens += response1.total_tokens or 0
            api_calls += 1

            if verbose:
                console.print(f"[cyan]Stage 1 ({func_key}): {response1.text[:200]}...[/cyan]")

            # Save raw LLM output (use func_key for unique tracking)
            stage_metadata["raw_llm_outputs"]["stage1"][func_key] = response1.text

            # Parse scenario matches {"1": "Yes", "2": "No", ...}
            matched_scenarios = []
            try:
                # Extract JSON from response
                json_match = re.search(r'\{[^}]+\}', response1.text)
                if json_match:
                    matches = json.loads(json_match.group())
                    for idx, scenario in enumerate(scenarios, 1):
                        if _is_affirmative(matches.get(str(idx), "No")):
                            matched_scenarios.append(scenario)
                            stage_metadata["scenarios_matched"] += 1
            except (json.JSONDecodeError, AttributeError):
                pass

            stage_metadata["stage1_results"][func_key] = [s["id"] for s in matched_scenarios]

            # STAGE 2 & 3: For each matched scenario
            for scenario in matched_scenarios:
                # STAGE 2: Property verification
                stage2_prompt = prompt_registry.render(
                    stage2_template,
                    contract_source=function_code,  # Pass function_code as contract_source for registry compatibility
                    function_code=function_code,
                    scenario=scenario,
                )

                response2 = adapter.generate(stage2_prompt)
                total_cost += response2.cost_usd or 0
                total_tokens += response2.total_tokens or 0
                api_calls += 1

                if verbose:
                    console.print(f"[cyan]Stage 2 ({func_key}/{scenario['name']}): {response2.text}[/cyan]")

                # Save raw LLM output
                if func_key not in stage_metadata["raw_llm_outputs"]["stage2"]:
                    stage_metadata["raw_llm_outputs"]["stage2"][func_key] = {}
                stage_metadata["raw_llm_outputs"]["stage2"][func_key][scenario["id"]] = response2.text

                property_confirmed = _is_affirmative(response2.text.strip())

                if func_key not in stage_metadata["stage2_results"]:
                    stage_metadata["stage2_results"][func_key] = {}
                stage_metadata["stage2_results"][func_key][scenario["id"]] = property_confirmed

                if not property_confirmed:
                    continue

                stage_metadata["properties_confirmed"] += 1

                # STAGE 3: Key variable extraction
                stage3_prompt = prompt_registry.render(
                    stage3_template,
                    contract_source=function_code,  # Pass function_code as contract_source for registry compatibility
                    function_code=function_code,
                    scenario=scenario,
                )

                response3 = adapter.generate(stage3_prompt)
                total_cost += response3.cost_usd or 0
                total_tokens += response3.total_tokens or 0
                api_calls += 1

                if verbose:
                    console.print(f"[cyan]Stage 3 ({func_key}/{scenario['name']}): {response3.text[:300]}...[/cyan]")

                # Save raw LLM output
                if func_key not in stage_metadata["raw_llm_outputs"]["stage3"]:
                    stage_metadata["raw_llm_outputs"]["stage3"][func_key] = {}
                stage_metadata["raw_llm_outputs"]["stage3"][func_key][scenario["id"]] = response3.text

                # Parse extraction result
                extraction = {}
                try:
                    json_match = re.search(r'\{[\s\S]*\}', response3.text)
                    if json_match:
                        extraction = json.loads(json_match.group())
                except (json.JSONDecodeError, AttributeError):
                    pass

                if func_key not in stage_metadata["stage3_results"]:
                    stage_metadata["stage3_results"][func_key] = {}
                stage_metadata["stage3_results"][func_key][scenario["id"]] = extraction

                # Create finding from extraction
                key_vars = []
                if "VariableA" in extraction:
                    key_vars.append(extraction["VariableA"].get("name", ""))
                if "VariableB" in extraction:
                    key_vars.append(extraction["VariableB"].get("name", ""))

                key_stmts = extraction.get("key_statements", [])
                vuln_lines = extraction.get("vulnerable_line_numbers", [func_info.start_line])

                finding = Finding(
                    title=f"{scenario['name']} in {func_info.name}",
                    category=normalize_category(scenario.get("dasp_category", "other")),
                    severity=scenario.get("severity_default", "medium"),
                    confidence=0.8,  # High confidence from 3-stage confirmation
                    explanation=f"Function {func_info.name} (line {func_info.start_line}) {scenario['scenario']} and {scenario['property']}",
                    evidence={
                        "function": func_info.name,
                        "function_key": func_key,
                        "lines": vuln_lines if isinstance(vuln_lines, list) else [vuln_lines],
                        "file": "contract.sol",
                    },
                    key_variables=key_vars,
                    key_statements=key_stmts,
                    fix_suggestion=f"Review and fix the {scenario['name']} vulnerability pattern",
                    scenario_matched=scenario["name"],
                    property_verified=scenario["property"],
                    static_check_type=scenario.get("static_check", ""),
                    source="gptscan_multistage",
                )

                # STAGE 4: Static confirmation (if enabled)
                if enable_static_confirmation:
                    try:
                        from ..confirmation.base import run_confirmation_pipeline

                        finding = run_confirmation_pipeline(
                            finding=finding,
                            extraction=extraction,
                            scenario_id=scenario["id"],
                            slither_instance=slither_instance,
                            contract_source=contract_source,
                        )

                        if verbose:
                            status = "CONFIRMED" if finding.static_confirmed else "REJECTED"
                            console.print(f"[cyan]Stage 4 ({func_key}/{scenario['name']}): {status}[/cyan]")

                        # Track confirmation stats
                        if "confirmation_results" not in stage_metadata:
                            stage_metadata["confirmation_results"] = {}
                        if func_key not in stage_metadata["confirmation_results"]:
                            stage_metadata["confirmation_results"][func_key] = {}

                        stage_metadata["confirmation_results"][func_key][scenario["id"]] = {
                            "confirmed": finding.static_confirmed,
                            "confidence": finding.confidence,
                            "reason": getattr(finding, 'static_check_reason', ''),
                        }

                    except ImportError as e:
                        if verbose:
                            console.print(f"[yellow]Stage 4 skipped: {e}[/yellow]")

                findings.append(finding)

    # LINE-LEVEL CORRELATION: Only boost rejected findings if there's a confirmed
    # finding of the SAME vulnerability type at the same location.
    # This prevents incorrectly boosting unrelated vulnerabilities that happen
    # to be at the same line number but have different categories.
    if enable_static_confirmation and findings:
        # Group findings by function AND category
        func_cat_to_findings = defaultdict(list)
        for f in findings:
            func_name = f.evidence.function if f.evidence else ""
            category = getattr(f, 'category', 'unknown')
            func_cat_to_findings[(func_name, category)].append(f)

        # For each function+category group, check if any finding was confirmed
        for (func_name, category), func_cat_findings in func_cat_to_findings.items():
            confirmed_in_group = [f for f in func_cat_findings if getattr(f, 'static_confirmed', None) is True]
            rejected_in_group = [f for f in func_cat_findings if getattr(f, 'static_confirmed', None) is False]

            if confirmed_in_group and rejected_in_group:
                # There's a confirmed finding of the SAME type - boost rejected findings
                for f in rejected_in_group:
                    # Check if they share the same lines
                    confirmed_lines = set()
                    for cf in confirmed_in_group:
                        if cf.evidence and cf.evidence.lines:
                            confirmed_lines.update(cf.evidence.lines)

                    rejected_lines = set(f.evidence.lines) if f.evidence and f.evidence.lines else set()

                    # Only boost if same category AND (same lines OR no line info)
                    if confirmed_lines & rejected_lines or (not confirmed_lines and not rejected_lines):
                        f.confidence = max(f.confidence, 0.5)
                        f.static_check_reason = f"{f.static_check_reason} (boosted: same-type vuln confirmed at location)"

    # Count confirmed vs rejected vs boosted (before filtering)
    confirmed_count = sum(1 for f in findings if getattr(f, 'static_confirmed', None) is True)
    rejected_count = sum(1 for f in findings if getattr(f, 'static_confirmed', None) is False)
    boosted_count = sum(1 for f in findings if getattr(f, 'static_confirmed', None) is False and 'boosted' in getattr(f, 'static_check_reason', ''))
    total_before_filter = len(findings)

    # Filter out rejected findings (per GPTScan methodology)
    # Only keep findings that are confirmed OR have no static confirmation status (non-P5 findings)
    if enable_static_confirmation:
        findings = [
            f for f in findings
            if getattr(f, 'static_confirmed', None) is not False
        ]
        if verbose:
            console.print(f"[dim]Static confirmation: {confirmed_count} confirmed, {rejected_count} rejected (filtered out)[/dim]")

    metadata = {
        "approach": "gptscan_multistage",
        "api_calls": api_calls,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        **stage_metadata,
        "slither_output_length": len(slither_output),
        "enable_static_confirmation": enable_static_confirmation,
        "findings_confirmed_by_static": confirmed_count,
        "findings_rejected_by_static": rejected_count,
        "findings_rejected_filtered_out": rejected_count,
        "findings_boosted_by_correlation": boosted_count,
        "total_findings_before_confirmation": total_before_filter,
        "total_findings_after_filter": len(findings),
    }

    return findings, total_cost, total_tokens, metadata


def run_single_item(
    adapter: LLMAdapter,
    prompt_registry: PromptRegistry,
    prompt_id: str,
    template_path: str,
    contract_source: str,
    contract_id: Optional[str] = None,
    tools_config: Optional[Dict] = None,
    verbose: bool = False,
) -> Tuple[List[Finding], float, int, Dict]:
    """
    Run a single contract through the appropriate prompt backend.

    Args:
        adapter: LLM adapter for generation
        prompt_registry: Registry for loading prompt templates
        prompt_id: ID of the prompt (e.g., "P4_smartguard")
        template_path: Path to the Jinja2 template
        contract_source: Solidity source code to analyze
        contract_id: Optional contract identifier for logging/tracking
        tools_config: Optional dict with tool settings (top_k_patterns, slither_timeout, etc.)
        verbose: If True, print API responses to console

    Returns:
        Tuple of (findings, cost, tokens, extra_metadata)
    """
    # Extract tool config settings with defaults
    tc = tools_config or {}
    top_k_patterns = tc.get('top_k_patterns', 5)
    slither_timeout = tc.get('slither_timeout', 120)
    max_rounds = tc.get('max_rounds', 1)
    consensus_threshold = tc.get('consensus_threshold', 0.9)
    auditor_runs = tc.get('auditor_runs', 1)
    voting_threshold = tc.get('voting_threshold', 0.5)
    # P5 GPTScan settings
    enable_static_confirmation = tc.get('enable_static_confirmation', True)
    # P5 stage templates (configurable)
    p4_stage1_template = tc.get('p4_stage1_template', 'p4_stage1_scenario.j2')
    p4_stage2_template = tc.get('p4_stage2_template', 'p4_stage2_property.j2')
    p4_stage3_template = tc.get('p4_stage3_template', 'p4_stage3_extraction.j2')
    # GPTScan methodology alignment
    slither_mode = tc.get('slither_mode', 'confirmation_only')  # confirmation_only | pre_and_post | disabled
    enable_reachability_filter = tc.get('enable_reachability_filter', True)

    prompt_type = detect_prompt_type(prompt_id, template_path)

    if prompt_type == 'skip':
        # This template is part of a multi-template prompt, skip
        return [], 0.0, 0, {"skipped": True, "reason": "Part of multi-agent chain"}

    elif prompt_type == 'icl_single_type':
        # P2 ICL: Run separate prompt for each vulnerability type
        # Per ICL paper (Chachar2025) - use type-specific examples
        from ..parsing.findings import Evidence  # Import outside loop

        # Load ICL examples (default: synthetic examples to prevent data leakage)
        # Synthetic examples are hand-crafted and NOT from the evaluation dataset
        from ..paths import get_icl_examples_path
        examples_path = get_icl_examples_path()
        try:
            icl_examples = json.loads(examples_path.read_text())
            # Skip metadata key if present
            if "_metadata" in icl_examples:
                icl_source = icl_examples["_metadata"].get("source", "synthetic")
            else:
                icl_source = "unknown"
        except (FileNotFoundError, json.JSONDecodeError) as e:
            console.print(f"[yellow]Warning: Could not load ICL examples: {e}[/yellow]")
            icl_examples = {}
            icl_source = "none"

        all_findings = []
        total_cost = 0.0
        total_tokens = 0
        total_latency = 0.0
        per_type_results = {}
        raw_llm_outputs = {}  # Store raw responses per vulnerability type

        # Extended from Chachar2025 ICL paper to all 10 DASP categories
        # Original paper tested 3 types; we extend to full DASP taxonomy
        vuln_types = [
            "reentrancy",
            "access_control",
            "arithmetic",
            "unchecked_low_level_calls",
            "denial_of_service",
            "bad_randomness",
            "front_running",
            "time_manipulation",
            "short_addresses",
            "other",
        ]

        for vuln_type in vuln_types:
            examples = icl_examples.get(vuln_type, {})
            example_list = examples.get("examples", [])

            # Render type-specific prompt
            rendered = prompt_registry.render(
                template_path,
                contract_source=contract_source,
                vulnerability_type=vuln_type,
                vulnerability_description=examples.get("description", ""),
                example1_code=example_list[0].get("code", "") if len(example_list) > 0 else "",
                example2_code=example_list[1].get("code", "") if len(example_list) > 1 else "",
                example3_code=example_list[2].get("code", "") if len(example_list) > 2 else "",
            )

            response = adapter.generate(rendered)
            if verbose:
                _print_verbose_response(response, f"P2 ICL - {vuln_type}")
            total_cost += response.cost_usd or 0
            total_tokens += response.total_tokens or 0
            total_latency += response.latency_s or 0

            # Save raw LLM output
            raw_llm_outputs[vuln_type] = response.text

            # Parse single-type response
            result = parse_icl_single_response(response.text, vuln_type)
            per_type_results[vuln_type] = result

            if _is_affirmative(result.get("contains_vulnerability")):
                # Convert to Finding format
                location = result.get("vulnerability_location") or {}

                # Handle certainty_score properly: 0 is a valid score, not None
                certainty = result.get("certainty_score")
                confidence = (certainty if certainty is not None else 50) / 100.0

                finding = Finding(
                    title=f"{vuln_type.replace('_', ' ').title()} vulnerability",
                    category=vuln_type,
                    severity="high",
                    confidence=confidence,
                    evidence=Evidence(
                        lines=location.get("lines", []),
                        function=location.get("function", ""),
                    ),
                    explanation=location.get("description", ""),
                )
                all_findings.append(finding)

        metadata = {
            "approach": "icl_single_type",
            "icl_examples_source": icl_source,  # "synthetic" = no data leakage
            "vulnerability_types_checked": vuln_types,
            "per_type_results": per_type_results,
            "raw_llm_outputs": raw_llm_outputs,  # Raw LLM responses per vulnerability type
            "types_with_vulnerabilities": [
                vt for vt, r in per_type_results.items()
                if _is_affirmative(r.get("contains_vulnerability"))
            ],
            "api_calls": len(vuln_types),
            "latency_s": total_latency,
            "parse_errors": [
                vt for vt, r in per_type_results.items()
                if r.get("parse_error")
            ],
        }

        return all_findings, total_cost, total_tokens, metadata

    elif prompt_type == 'smartguard':
        # P4: SmartGuard-style (RAG + CoT + Self-check)
        # Based on Ding2025SmartGuard - combines:
        # 1. Demonstration Selection (CodeBERT retrieval)
        # 2. Demonstration Expansion (automatic CoT generation) - optional
        # 3. In-Context Learning (true k-shot ICL format)
        # 4. Iterative Self-Check (multi-turn validation) - optional
        from ..rag.retriever import get_retriever

        # Configuration from tools_config (with defaults)
        enable_cot_expansion = tc.get('enable_cot_expansion', False)  # Full SmartGuard (expensive)
        use_icl_template = tc.get('use_icl_template', True)  # True k-shot ICL format
        enable_iterative_selfcheck = tc.get('enable_iterative_selfcheck', False)  # Multi-turn self-check
        selfcheck_max_iterations = tc.get('selfcheck_max_iterations', 3)
        selfcheck_threshold = tc.get('selfcheck_convergence_threshold', 0.9)
        num_patterns = top_k_patterns  # From config (default: 5)

        total_cost = 0.0
        total_tokens = 0
        total_latency = 0.0
        api_calls = 0

        # Stage 1: Demonstration Selection (Retrieval)
        # Use singleton to avoid rebuilding index per contract
        # Note: Data leakage prevented by dataset separation (Qian patterns ≠ SmartBugs evaluation)
        retriever = get_retriever()
        retrieved_patterns = retriever.retrieve_patterns(
            contract_source,
            num_patterns=num_patterns,
        )

        # Stage 2: Demonstration Expansion (Auto CoT Generation)
        # This is optional and expensive - generates CoT for each pattern
        cot_metadata = {"enabled": enable_cot_expansion}

        if enable_cot_expansion and retrieved_patterns:
            from ..rag.cot_generator import CoTGenerator

            cot_generator = CoTGenerator(adapter)
            patterns_with_cot = []

            for pattern in retrieved_patterns:
                # Skip if pattern already has CoT (from pre-generation)
                if pattern.get('chain_of_thought'):
                    patterns_with_cot.append(pattern)
                    continue

                cot_result = cot_generator.generate_cot(
                    code=pattern.get('vulnerable_code', ''),
                    true_category=pattern.get('category', 'other'),
                    use_cache=True,  # Cache to avoid regeneration
                )

                patterns_with_cot.append({
                    **pattern,
                    'chain_of_thought': cot_result.reasoning,
                })

                # Track CoT generation costs (actual values from API calls)
                api_calls += cot_result.iterations
                total_tokens += cot_result.total_tokens
                total_cost += cot_result.total_cost
                total_latency += cot_result.total_latency

            retrieved_patterns = patterns_with_cot
            cot_metadata.update({
                "patterns_expanded": len(patterns_with_cot),
                "cot_api_calls": api_calls,
                "cot_tokens": total_tokens,
                "cot_cost_usd": total_cost,
            })

        # Stage 3: In-Context Learning Audit
        # Use true k-shot ICL template or legacy reference context template
        if use_icl_template:
            # True k-shot ICL format (SmartGuard paper Figure 4 style)
            rendered = prompt_registry.render(
                "p3_smartguard_icl.j2",
                contract_source=contract_source,
                demonstrations=retrieved_patterns
            )
        else:
            # Legacy format (patterns as reference context)
            rendered = prompt_registry.render(
                template_path,
                contract_source=contract_source,
                retrieved_patterns=retrieved_patterns
            )

        response = adapter.generate(rendered)
        if verbose:
            _print_verbose_response(response, "P4 SmartGuard ICL" if use_icl_template else "P4 SmartGuard")
        initial_findings, errors = parse_findings(response.text)

        api_calls += 1
        total_cost += response.cost_usd or 0
        total_tokens += response.total_tokens or 0
        total_latency += response.latency_s or 0

        # Stage 4: Iterative Self-Check (optional multi-turn validation)
        selfcheck_metadata = {"enabled": enable_iterative_selfcheck}

        if enable_iterative_selfcheck and initial_findings:
            from ..rag.selfcheck import SelfCheckRunner

            selfcheck = SelfCheckRunner(
                adapter,
                max_iterations=selfcheck_max_iterations,
                convergence_threshold=selfcheck_threshold,
            )

            selfcheck_result = selfcheck.run_selfcheck(
                initial_findings=initial_findings,
                contract_source=contract_source,
            )

            findings = selfcheck_result.final_findings

            # Accumulate self-check costs
            api_calls += selfcheck_result.total_api_calls
            total_cost += selfcheck_result.total_cost
            total_tokens += selfcheck_result.total_tokens
            total_latency += selfcheck_result.total_latency

            selfcheck_metadata.update({
                "iterations": selfcheck_result.iterations,
                "converged": selfcheck_result.converged,
                "convergence_iteration": selfcheck_result.convergence_iteration,
                "initial_findings": selfcheck_result.initial_finding_count,
                "final_findings": selfcheck_result.final_finding_count,
                "findings_removed": selfcheck_result.findings_removed,
                "iteration_history": selfcheck_result.iteration_history,
            })

            if verbose:
                console.print(f"  [cyan]Self-check:[/cyan] {selfcheck_result.initial_finding_count} → {selfcheck_result.final_finding_count} findings "
                            f"({selfcheck_result.iterations} iterations, converged={selfcheck_result.converged})")
        else:
            findings = initial_findings

        metadata = {
            "approach": "smartguard_icl" if use_icl_template else "smartguard_legacy",
            "components": ["rag", "icl", "cot", "selfcheck"] if enable_iterative_selfcheck else ["rag", "cot"],
            "demonstration_expansion": cot_metadata,
            "selfcheck": selfcheck_metadata,
            "num_patterns_retrieved": len(retrieved_patterns),
            "retrieved_categories": [p.get("category", "unknown") for p in retrieved_patterns],
            "parse_errors": errors,
            "raw_output": response.text,
            "rendered_prompt": rendered,  # Save for reproducibility
            "api_calls": api_calls,
            "latency_s": total_latency,
        }
        return findings, total_cost, total_tokens, metadata

    elif prompt_type == 'tool_augmented':
        # P5: GPTScan-style multi-stage pipeline
        # Per GPTScan paper (Sun2023): Static analysis is used for POST-LLM confirmation only,
        # NOT for pre-LLM candidate detection. Pre-LLM uses multi-dimensional filtering.

        # Load scenarios first (needed for both filtering and template)
        p5_scenarios = get_p5_scenarios()

        # PHASE 1: GPTScan-style pre-filtering by keywords/expressions (FNK/FCE/FCCE)
        # This is the PRIMARY candidate selection mechanism per GPTScan methodology
        func_filter = get_function_filter()
        prefilter_candidates = set()
        prefilter_by_scenario = {}
        all_functions = []

        if func_filter:
            # Extract all functions first
            all_functions = func_filter.extract_functions(contract_source)

            # Get candidates per scenario using keyword/expression matching
            prefilter_by_scenario = func_filter.filter_all_scenarios(contract_source)
            for scenario_candidates in prefilter_by_scenario.values():
                prefilter_candidates.update(scenario_candidates)

            # PHASE 2: Reachability analysis (GPTScan-style)
            # Filter to only functions reachable from external entry points
            if enable_reachability_filter and prefilter_candidates:
                reachable = func_filter.filter_reachable_functions(
                    contract_source, prefilter_candidates
                )
                prefilter_candidates = reachable

        # Initialize Slither-related variables
        slither = None
        slither_result = None
        slither_candidates = set()
        slither_output = ""
        slither_metadata = {"slither_mode": slither_mode}

        # PHASE 2b: Slither detector scanning (ONLY if legacy mode)
        # Per GPTScan paper: Slither is NOT used for pre-LLM detection
        if slither_mode == "pre_and_post":
            # Legacy mode: use Slither for both pre-LLM detection and post-LLM confirmation
            slither = get_slither_runner(timeout=slither_timeout)
            if slither:
                slither_result = slither.run(contract_source)
                slither_candidates = slither_result.get_candidate_functions()
                slither_output = slither_result.to_gptscan_prompt_text()
                slither_metadata.update({
                    "slither_candidates": list(slither_candidates),
                    "total_findings": len(slither_result.findings),
                    "slither_success": slither_result.success,
                    "slither_available": True,
                })
            else:
                slither_metadata["slither_available"] = False

        elif slither_mode == "confirmation_only":
            # GPTScan methodology: Slither only for post-LLM confirmation
            # Get runner but DON'T run detectors pre-LLM
            slither = get_slither_runner(timeout=slither_timeout)
            slither_metadata["slither_available"] = slither is not None
            # Run Slither to get findings (for saving) but don't use for candidate selection
            if slither:
                slither_result = slither.run(contract_source)
                slither_output = slither_result.to_gptscan_prompt_text()
                slither_metadata["slither_findings_count"] = len(slither_result.findings)
                slither_metadata["slither_raw_output"] = slither_result.raw_output
                slither_metadata["slither_findings"] = [
                    {
                        "check": f.check,
                        "impact": f.impact,
                        "confidence": f.confidence,
                        "description": f.description,
                        "functions": f.get_affected_functions(),
                        "lines": f.get_line_range(),
                    }
                    for f in slither_result.findings
                ]

        # slither_mode == "disabled": No Slither at all

        # PHASE 3: Combine candidates based on mode
        if slither_mode == "pre_and_post":
            # Legacy: union of pre-filter and Slither candidates
            if prefilter_candidates and slither_candidates:
                combined_candidates = prefilter_candidates.union(slither_candidates)
            elif prefilter_candidates:
                combined_candidates = prefilter_candidates
            elif slither_candidates:
                combined_candidates = slither_candidates
            else:
                combined_candidates = {f.name for f in all_functions} if all_functions else set()
        else:
            # GPTScan methodology: only pre-filter candidates (no Slither pre-LLM)
            if prefilter_candidates:
                combined_candidates = prefilter_candidates
            else:
                # Fallback: all public/external functions
                combined_candidates = {
                    f.name for f in all_functions
                    if f.visibility in ("public", "external")
                } if all_functions else set()

        candidate_functions = list(combined_candidates)

        # Track filtering metadata
        filtering_metadata = {
            "prefilter_candidates": list(prefilter_candidates),
            "prefilter_by_scenario": {k: list(v) for k, v in prefilter_by_scenario.items()},
            "slither_candidates": list(slither_candidates),
            "combined_candidates": candidate_functions,
            "prefilter_count": len(prefilter_candidates),
            "slither_count": len(slither_candidates),
            "combined_count": len(candidate_functions),
            "slither_mode": slither_mode,
            "reachability_filter_enabled": enable_reachability_filter,
        }

        # PHASE 4: Run multi-stage pipeline
        if candidate_functions:
            # Get Slither instance for static confirmation (if available)
            # This is where Slither is actually used in GPTScan methodology
            slither_instance = None
            if slither_mode != "disabled" and slither:
                if slither_result is not None and hasattr(slither_result, 'get_slither_instance'):
                    slither_instance = slither_result.get_slither_instance()
                else:
                    # For confirmation_only mode, create Slither instance lazily
                    try:
                        from ..tools.slither_runner import SlitherResult
                        # Create a minimal result to get the Slither instance
                        temp_result = slither.run(contract_source)
                        slither_instance = temp_result.get_slither_instance()
                        slither_metadata["slither_instance_created"] = True
                    except Exception as e:
                        slither_metadata["slither_instance_error"] = str(e)

            # GPTScan multi-stage pipeline (Scenario→Property→Extraction→Confirmation)
            # Note: slither_output is empty for confirmation_only mode (not shown to LLM)
            findings, total_cost, total_tokens, stage_metadata = run_gptscan_multistage(
                adapter=adapter,
                prompt_registry=prompt_registry,
                contract_source=contract_source,
                scenarios=p5_scenarios["scenarios"],
                candidate_functions=candidate_functions,
                slither_output="" if slither_mode == "confirmation_only" else slither_output,
                slither_instance=slither_instance,
                enable_static_confirmation=enable_static_confirmation,
                verbose=verbose,
                stage1_template=p4_stage1_template,
                stage2_template=p4_stage2_template,
                stage3_template=p4_stage3_template,
            )

            metadata = {
                **slither_metadata,
                **filtering_metadata,
                **stage_metadata,
                "slither_available": slither is not None,
                "function_filter_available": func_filter is not None,
                "slither_output_length": len(slither_output) if slither_output else 0,
                "slither_prompt_text": slither_output if slither_output else "",
                "scenarios_loaded": len(p5_scenarios["scenarios"]),
                "enable_static_confirmation": enable_static_confirmation,
            }
            return findings, total_cost, total_tokens, metadata
        else:
            # No candidate functions matched pre-filter - return empty results
            metadata = {
                **slither_metadata,
                **filtering_metadata,
                "slither_available": slither is not None,
                "function_filter_available": func_filter is not None,
                "scenarios_loaded": len(p5_scenarios["scenarios"]),
                "no_candidates": True,
                "reason": "No functions matched pre-filter criteria",
            }
            return [], 0.0, 0, metadata

    elif prompt_type == 'smartaudit':
        # P6: LLM-SmartAudit multi-agent (Wei et al. 2024)
        # Uses BA (Broad Analysis) or TA (Targeted Analysis) modes
        from pathlib import Path

        # Get SmartAudit config options
        smartaudit_mode = tc.get('smartaudit_mode', 'BA')
        smartaudit_max_rounds = tc.get('smartaudit_max_rounds', 3)
        smartaudit_consensus_threshold = tc.get('smartaudit_consensus_threshold', 0.9)
        smartaudit_enable_role_exchange = tc.get('smartaudit_enable_role_exchange', True)
        smartaudit_scenarios = tc.get('smartaudit_scenarios', None)
        smartaudit_scenario_path = tc.get('smartaudit_scenario_path', None)

        result = run_smartaudit(
            adapter, prompt_registry, contract_source,
            mode=smartaudit_mode,
            max_rounds=smartaudit_max_rounds,
            consensus_threshold=smartaudit_consensus_threshold,
            enable_role_exchange=smartaudit_enable_role_exchange,
            scenarios=smartaudit_scenarios,
            scenario_path=Path(smartaudit_scenario_path) if smartaudit_scenario_path else None,
        )
        findings, metadata = extract_final_findings(result)

        # Add raw outputs for analysis (full outputs saved)
        metadata["raw_outputs"] = result.raw_outputs
        metadata["api_calls"] = result.total_api_calls
        metadata["parse_errors"] = []  # SmartAudit handles parsing internally
        metadata["latency_s"] = result.total_latency_s

        return findings, result.total_cost, result.total_tokens, metadata

    elif prompt_type == 'p6_iaudit':
        # P6 iAudit: Full pipeline (Ma et al., 2024)
        # Architecture: Detector (fine-tuned) → Reasoner (fine-tuned) → Ranker-Critic
        # Expected F1: ~91% (paper reported 91.21%)

        # Get P6 iAudit config from tools_config
        iaudit_config = iAuditConfig(
            detector_model_path=tc.get('p6_iaudit_detector_model', ''),
            reasoner_model_path=tc.get('p6_iaudit_reasoner_model', ''),
            ranker_model=tc.get('p6_iaudit_ranker_model', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
            use_4bit=tc.get('p6_iaudit_use_4bit', True),
            max_ranker_rounds=tc.get('p6_iaudit_max_ranker_rounds', 5),
            detector_prompts=tc.get('p6_iaudit_detector_prompts', 5),
            reasoner_prompts=tc.get('p6_iaudit_reasoner_prompts', 10),
            hf_token=tc.get('p6_iaudit_hf_token', ''),
        )

        # Run iAudit pipeline
        result = run_iaudit_pipeline(
            contract_source=contract_source,
            config=iaudit_config,
            verbose=verbose,
        )

        metadata = {
            "approach": "p6_iaudit",
            "detector_label": result.detector_label,
            "detector_responses": result.detector_responses,
            "reasoner_explanations": result.reasoner_explanations,
            "selected_explanation": result.selected_explanation,
            "ranker_analysis": result.ranker_analysis,
            "ranker_rounds": result.ranker_rounds,
            "critic_agreed": result.critic_agreed,
            "api_calls": result.api_calls,
            "latency_s": result.total_latency_s,
            "detector_cost": result.detector_cost,
            "reasoner_cost": result.reasoner_cost,
            "ranker_cost": result.ranker_cost,
            "parse_errors": [],
        }

        return result.findings, result.total_cost_usd, result.total_tokens, metadata

    else:
        # Simple single-call prompt (P0-P3)
        rendered = prompt_registry.render(
            template_path,
            contract_source=contract_source
        )
        response = adapter.generate(rendered)
        if verbose:
            _print_verbose_response(response, f"Simple Prompt ({prompt_id})")
        findings, errors = parse_findings(response.text)
        return findings, response.cost_usd or 0, response.total_tokens or 0, {
            "parse_errors": errors,
            "raw_output": response.text,  # Save raw output for debugging/reproducibility
            "rendered_prompt": rendered,  # Save for reproducibility
            "api_calls": 1,
            "latency_s": response.latency_s or 0,
            # Include token breakdown for cost tracking
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "cost_usd": response.cost_usd,
        }


def _process_contract(
    item: ContractItem,
    adapter: LLMAdapter,
    prompt_reg: PromptRegistry,
    prompt_spec,  # PromptSpec
    dataset_spec: DatasetSpec,
    model_spec,  # ModelSpec
    spec: ExperimentSpec,
    out_dir: Path,
    prompt_type: str,
    verbose: bool,
) -> List[RunResult]:
    """Process all runs for a single contract. Thread-safe.

    Args:
        item: Contract to process
        adapter: LLM adapter
        prompt_reg: Prompt registry
        prompt_spec: Prompt specification
        dataset_spec: Dataset specification
        model_spec: Model specification
        spec: Full experiment specification
        out_dir: Output directory
        prompt_type: Detected prompt type
        verbose: Verbose output flag

    Returns:
        List of RunResult objects for each run
    """
    results = []

    for run_idx in range(spec.runs_per_item):
        # Check if this run already exists (for resumable experiments)
        if check_run_exists(
            out_dir, dataset_spec.name, model_spec.name,
            prompt_spec.id, item.id, run_idx
        ):
            # Return a skip marker
            results.append(RunResult(
                dataset_name=dataset_spec.name,
                model_name=model_spec.name,
                prompt_id=prompt_spec.id,
                prompt_type=prompt_type,
                contract_id=item.id,
                run_idx=run_idx,
                findings=[],
                cost_usd=0.0,
                tokens=0,
                api_calls=0,
                latency_s=0.0,
                metrics={},
                extra_metadata={},
                ground_truth=[],
                error="skipped_existing",
            ))
            continue

        try:
            # Use the unified run_single_item which routes to appropriate backend
            tools_dict = None
            if spec.tools:
                tools_dict = asdict(spec.tools)

            findings, cost, tokens, extra_metadata = run_single_item(
                adapter,
                prompt_reg,
                prompt_spec.id,
                prompt_spec.template_path,
                item.source,
                contract_id=item.id,
                tools_config=tools_dict,
                verbose=verbose,
            )

            # Skip if this was a skip prompt
            if extra_metadata.get("skipped"):
                continue

            # Check for critical parse errors (JSON couldn't be parsed at all)
            parse_errors = extra_metadata.get("parse_errors", [])
            has_critical_parse_error = any(
                "json_load_error" in str(err) for err in parse_errors
            )

            ground_truth = item.labels or []

            # If critical parse error, mark as failed and don't compute metrics
            if has_critical_parse_error:
                logger.warning(
                    f"Parse failed for {item.id} - excluding from metrics. "
                    f"Errors: {parse_errors}"
                )
                metrics = {
                    "parse_failed": True,
                    "tp": 0, "fp": 0, "fn": 0,
                    "precision": None, "recall": None, "f1": None,
                }
                extra_metadata["parse_failed"] = True
            else:
                # Compute metrics against ground truth
                metrics = compute_metrics_for_run(findings, ground_truth)

            # Build model_info for reproducibility
            model_info = {
                "name": model_spec.name,
                "provider": model_spec.provider,
                "params": model_spec.params,
            }

            # Save outputs with metrics and extra metadata
            save_run_output(
                out_dir,
                dataset_spec.name,
                model_spec.name,
                prompt_spec.id,
                item.id,
                run_idx,
                None,
                findings,
                parse_errors if isinstance(parse_errors, list) else [],
                ground_truth=ground_truth,
                metrics=metrics,
                extra_data=extra_metadata,
                contract_source=item.source,
                rendered_prompt=extra_metadata.get("rendered_prompt"),
                model_info=model_info,
            )

            results.append(RunResult(
                dataset_name=dataset_spec.name,
                model_name=model_spec.name,
                prompt_id=prompt_spec.id,
                prompt_type=prompt_type,
                contract_id=item.id,
                run_idx=run_idx,
                findings=findings,
                cost_usd=cost,
                tokens=tokens,
                api_calls=extra_metadata.get("api_calls", 1),
                latency_s=extra_metadata.get("latency_s", 0),
                metrics=metrics,
                extra_metadata=extra_metadata,
                ground_truth=ground_truth,
                error="parse_failed" if has_critical_parse_error else None,
            ))

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append(RunResult(
                dataset_name=dataset_spec.name,
                model_name=model_spec.name,
                prompt_id=prompt_spec.id,
                prompt_type=prompt_type,
                contract_id=item.id,
                run_idx=run_idx,
                findings=[],
                cost_usd=0.0,
                tokens=0,
                api_calls=0,
                latency_s=0.0,
                metrics={},
                extra_metadata={},
                ground_truth=[],
                error=str(e),
            ))

    return results


def run_experiment(
    spec: ExperimentSpec,
    *,
    prompt_root: str | Path,
    dry_run: bool = False,
    verbose: bool = False,
    workers: int = 1,
) -> Dict:
    """Run the full experiment.

    Args:
        spec: Experiment specification
        prompt_root: Root directory for prompt templates
        dry_run: If True, validate config without running
        verbose: If True, print API responses to console
        workers: Number of parallel workers (default: 1 for sequential)
    """
    out_dir = ensure_dir(spec.output_dir)
    prompt_reg = PromptRegistry(prompt_root)

    # Save experiment config
    config_path = out_dir / "experiment_config.json"
    config_path.write_text(json.dumps(asdict(spec), indent=2, default=str))

    # Apply random seed for reproducibility
    random.seed(spec.random_seed)
    np.random.seed(spec.random_seed)

    # Print experiment summary
    console.print(f"\n[bold blue]Experiment: {spec.experiment_id}[/bold blue]")
    console.print(f"Output dir: {out_dir}")
    console.print(f"Runs per item: {spec.runs_per_item}")
    console.print(f"Random seed: {spec.random_seed} (applied)")
    if workers > 1:
        console.print(f"[bold cyan]Parallel workers: {workers}[/bold cyan]")

    console.print("\n[bold]Datasets:[/bold]")
    for ds in spec.datasets:
        console.print(f"  - {ds.name} ({ds.kind}) @ {ds.path}")

    console.print("\n[bold]Models:[/bold]")
    for m in spec.models:
        console.print(f"  - {m.provider}:{m.name} | temp={m.params.get('temperature', 0.0)}")

    console.print("\n[bold]Prompts:[/bold]")
    for p in spec.prompts:
        console.print(f"  - {p.id} ({p.template_path})")

    if dry_run:
        console.print("\n[yellow]Dry-run mode: validating config and prompts only[/yellow]")
        # Validate all prompts render correctly with appropriate context
        for p in spec.prompts:
            try:
                prompt_type = detect_prompt_type(p.id, p.template_path)
                if prompt_type == 'skip':
                    console.print(f"  [dim]⊘[/dim] {p.id} (part of multi-agent chain)")
                    continue

                # Provide appropriate context for different prompt types
                render_context = {"contract_source": "// test contract"}
                if prompt_type == 'smartguard':
                    render_context["retrieved_patterns"] = []
                elif prompt_type == 'tool_augmented':
                    render_context["slither_output"] = ""
                    render_context["candidate_functions"] = []
                    # P4 stage templates require these additional variables
                    render_context["scenarios"] = [{"id": "test", "scenario": "Test scenario", "property": "Test property"}]
                    render_context["scenario"] = {"id": "test", "scenario": "Test scenario", "property": "Test property"}
                    render_context["function_code"] = "function test() public {}"
                    render_context["function_name"] = "test"
                elif prompt_type == 'icl_single_type':
                    render_context["vulnerability_type"] = "reentrancy"
                    render_context["vulnerability_description"] = "Test description"
                    render_context["example1_code"] = "// example 1"
                    render_context["example2_code"] = "// example 2"
                    render_context["example3_code"] = "// example 3"

                prompt_reg.render(p.template_path, **render_context)
                console.print(f"  [green]✓[/green] {p.id} renders OK ({prompt_type})")
            except Exception as e:
                console.print(f"  [red]✗[/red] {p.id} failed: {e}")
        return {"status": "dry_run_complete"}

    # Initialize thread-safe results accumulator
    accumulator = ThreadSafeResultsAccumulator()

    # Set up graceful shutdown for Ctrl+C
    shutdown_event = threading.Event()
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def signal_handler(signum, frame):
        if not shutdown_event.is_set():
            console.print("\n[yellow]Graceful shutdown requested... waiting for current tasks[/yellow]")
            shutdown_event.set()
        else:
            # Second Ctrl+C - force exit
            console.print("\n[red]Force shutdown[/red]")
            signal.signal(signal.SIGINT, original_sigint_handler)
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Main experiment loop
        for dataset_spec in spec.datasets:
            if shutdown_event.is_set():
                break

            console.print(f"\n[bold cyan]Loading dataset: {dataset_spec.name}[/bold cyan]")
            dataset = get_dataset(dataset_spec)
            contracts = list(dataset.iter_items())
            console.print(f"  Found {len(contracts)} contracts")

            for model_spec in spec.models:
                if shutdown_event.is_set():
                    break

                console.print(f"\n[bold magenta]Model: {model_spec.provider}:{model_spec.name}[/bold magenta]")
                adapter = make_adapter(model_spec.provider, model_spec.name, model_spec.params)

                for prompt_spec in spec.prompts:
                    if shutdown_event.is_set():
                        break

                    console.print(f"\n  [bold]Prompt: {prompt_spec.id}[/bold]")

                    # Detect prompt type for logging
                    prompt_type = detect_prompt_type(prompt_spec.id, prompt_spec.template_path)
                    if prompt_type == 'skip':
                        console.print(f"    [dim]Skipping {prompt_spec.id} (part of multi-agent chain)[/dim]")
                        continue

                    console.print(f"    [dim]Type: {prompt_type}[/dim]")

                    if workers == 1:
                        # Sequential execution (original behavior)
                        pbar = tqdm(contracts, desc=f"    {prompt_spec.id}", unit="contract")

                        for item in pbar:
                            if shutdown_event.is_set():
                                break

                            run_results = _process_contract(
                                item, adapter, prompt_reg, prompt_spec,
                                dataset_spec, model_spec, spec, out_dir,
                                prompt_type, verbose
                            )

                            for result in run_results:
                                if result.error == "skipped_existing":
                                    accumulator.add_skip()
                                elif result.error == "parse_failed":
                                    console.print(f"    [yellow]Parse failed for {result.contract_id} run {result.run_idx} - excluded from metrics[/yellow]")
                                    accumulator.add_parse_failure(result)
                                elif result.error:
                                    console.print(f"    [red]Error on {result.contract_id} run {result.run_idx}: {result.error}[/red]")
                                    accumulator.add_error(result)
                                else:
                                    accumulator.add_result(result)

                            # Update progress bar
                            totals = accumulator.get_totals()
                            pbar.set_postfix({
                                "cost": f"${totals['total_cost_usd']:.4f}",
                            })
                    else:
                        # Parallel execution
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            # Submit all contract processing tasks
                            futures = {}
                            for item in contracts:
                                if shutdown_event.is_set():
                                    break
                                future = executor.submit(
                                    _process_contract,
                                    item, adapter, prompt_reg, prompt_spec,
                                    dataset_spec, model_spec, spec, out_dir,
                                    prompt_type, verbose
                                )
                                futures[future] = item

                            # Progress bar tracking completed futures
                            pbar = tqdm(
                                as_completed(futures),
                                total=len(futures),
                                desc=f"    {prompt_spec.id}",
                                unit="contract"
                            )

                            for future in pbar:
                                if shutdown_event.is_set():
                                    # Cancel remaining futures
                                    for f in futures:
                                        f.cancel()
                                    break

                                item = futures[future]
                                try:
                                    run_results = future.result()
                                    for result in run_results:
                                        if result.error == "skipped_existing":
                                            accumulator.add_skip()
                                        elif result.error == "parse_failed":
                                            console.print(f"    [yellow]Parse failed for {result.contract_id} run {result.run_idx} - excluded from metrics[/yellow]")
                                            accumulator.add_parse_failure(result)
                                        elif result.error:
                                            console.print(f"    [red]Error on {result.contract_id} run {result.run_idx}: {result.error}[/red]")
                                            accumulator.add_error(result)
                                        else:
                                            accumulator.add_result(result)
                                except Exception as e:
                                    console.print(f"    [red]Future error for {item.id}: {e}[/red]")

                                # Update progress bar
                                totals = accumulator.get_totals()
                                pbar.set_postfix({
                                    "cost": f"${totals['total_cost_usd']:.4f}",
                                })

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)

    # Get final results from accumulator
    accumulated = accumulator.get_results()
    results = {
        "experiment_id": spec.experiment_id,
        "started_at": datetime.now().isoformat(),
        "runs": accumulated["runs"],
        "totals": accumulated["totals"],
    }

    # Compute overall CATEGORY-level metrics (backward compatible)
    tp = results["totals"]["total_tp"]
    fp = results["totals"]["total_fp"]
    fn = results["totals"]["total_fn"]
    overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    overall_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    results["totals"]["overall_precision"] = overall_precision
    results["totals"]["overall_recall"] = overall_recall
    results["totals"]["overall_f1"] = overall_f1

    # Compute overall INSTANCE-level metrics (requires BOTH category AND line match)
    inst_tp = results["totals"]["instance_tp"]
    inst_fp = results["totals"]["instance_fp"]
    inst_fn = results["totals"]["instance_fn"]
    instance_precision = inst_tp / (inst_tp + inst_fp) if (inst_tp + inst_fp) > 0 else 0.0
    instance_recall = inst_tp / (inst_tp + inst_fn) if (inst_tp + inst_fn) > 0 else 0.0
    instance_f1 = 2 * instance_precision * instance_recall / (instance_precision + instance_recall) if (instance_precision + instance_recall) > 0 else 0.0

    results["totals"]["instance_precision"] = instance_precision
    results["totals"]["instance_recall"] = instance_recall
    results["totals"]["instance_f1"] = instance_f1

    # Save final results
    results["completed_at"] = datetime.now().isoformat()
    results_path = out_dir / "results_summary.json"
    results_path.write_text(json.dumps(results, indent=2))

    # Print summary
    console.print(f"\n[bold green]Experiment complete![/bold green]")
    skipped = results['totals'].get('skipped_runs', 0)
    if skipped > 0:
        console.print(f"  [dim]Skipped {skipped} existing runs (resumable mode)[/dim]")
    parse_failed = results['totals'].get('parse_failed_runs', 0)
    if parse_failed > 0:
        console.print(f"  [yellow]Parse failures: {parse_failed} runs excluded from metrics[/yellow]")
    console.print(f"  Total API calls: {results['totals']['api_calls']}")
    console.print(f"  Total cost: ${results['totals']['total_cost_usd']:.4f}")
    console.print(f"  Total tokens: {results['totals']['total_tokens']:,}")
    console.print(f"\n[bold]Category-Level Metrics (by vulnerability type):[/bold]")
    console.print(f"  Precision: {overall_precision:.2%}")
    console.print(f"  Recall: {overall_recall:.2%}")
    console.print(f"  F1 Score: {overall_f1:.2%}")
    if results["totals"]["contracts_with_line_annotations"] > 0:
        console.print(f"\n[bold]Instance-Level Metrics (category + location match):[/bold]")
        console.print(f"  Precision: {instance_precision:.2%}")
        console.print(f"  Recall: {instance_recall:.2%}")
        console.print(f"  F1 Score: {instance_f1:.2%}")
        console.print(f"  (Based on {results['totals']['contracts_with_line_annotations']} contracts with line annotations)")
    console.print(f"\n  Results saved to: {out_dir}")

    return results


def main(config_path: str, *, dry_run: bool = False, verbose: bool = False, workers: int = 1) -> None:
    """Entry point for CLI.

    Args:
        config_path: Path to experiment YAML config
        dry_run: If True, validate config without running
        verbose: If True, print API responses to console
        workers: Number of parallel workers (default: 1 for sequential)
    """
    from ..paths import get_prompts_dir
    spec = load_experiment_config(config_path)
    prompt_root = get_prompts_dir()
    run_experiment(spec, prompt_root=prompt_root, dry_run=dry_run, verbose=verbose, workers=workers)
