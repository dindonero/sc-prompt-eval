"""
Cost estimation for experiment runs.
Provides --dry-run functionality to estimate costs before running.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

# Token estimation: ~4 chars per token for code
CHARS_PER_TOKEN = 4

# Pricing as of January 2026 (USD per 1M tokens)
# Updated for Azure OpenAI, Microsoft Foundry (Claude), and Azure AI Foundry (DeepSeek)
MODEL_PRICING = {
    # ==========================================================================
    # GPT-5 Series (Azure OpenAI 2026)
    # ==========================================================================
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-chat": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-pro": {"input": 15.00, "output": 120.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5.1-chat": {"input": 1.25, "output": 10.00},
    "gpt-5.1-codex": {"input": 2.00, "output": 15.00},
    "gpt-5.1-codex-max": {"input": 5.00, "output": 40.00},
    "gpt-5.1-codex-mini": {"input": 0.50, "output": 4.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.2-chat": {"input": 1.75, "output": 14.00},
    "gpt-5-codex": {"input": 2.00, "output": 15.00},
    "codex-mini": {"input": 0.50, "output": 4.00},

    # ==========================================================================
    # O-Series Reasoning Models (Azure OpenAI 2026)
    # ==========================================================================
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o3": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},

    # ==========================================================================
    # GPT-OSS Models (Azure - estimates)
    # ==========================================================================
    "gpt-oss-120b": {"input": 1.00, "output": 4.00},
    "gpt-oss-20b": {"input": 0.20, "output": 0.80},

    # ==========================================================================
    # Claude Models (Azure Microsoft Foundry - Jan 2026)
    # Note: Opus 4.5 pricing reduced 66% from legacy Opus 4.1
    # ==========================================================================
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-4-1": {"input": 15.00, "output": 75.00},

    # ==========================================================================
    # DeepSeek Models (Azure AI Foundry)
    # ==========================================================================
    "DeepSeek-R1": {"input": 1.35, "output": 5.40},
    "DeepSeek-R1-0528": {"input": 1.35, "output": 5.40},

    # ==========================================================================
    # Legacy OpenAI models (backwards compatibility)
    # ==========================================================================
    "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # ==========================================================================
    # Legacy Claude models (backwards compatibility, with corrected pricing)
    # ==========================================================================
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}

# Prompt template approximate sizes (in tokens)
# Updated to match current prompt structure from runner.py
PROMPT_OVERHEAD = {
    "p0_baseline": 200,                    # Simple zero-shot
    "p1_taxonomy_constrained": 400,        # + DASP vocabulary
    "p2_icl_single_type": 800,             # Per vulnerability type
    "p3_structured_reasoning": 600,        # + audit checklist
    "p4_smartguard": 1200,                 # RAG + CoT + Self-check
    "p5_tool_augmented": 800,              # + Slither output
    "p6_auditor": 400,                     # Multi-agent: initial audit
    "p6_reviewer": 600,                    # Multi-agent: review
    "p6_expert": 500,                      # Multi-agent: expert analysis
    "p6_critic": 700,                      # Multi-agent: final critique
}

# API calls per prompt strategy
# Keys are the base strategy (e.g., "P0", "P2") - prompt IDs like "P0_baseline" will be normalized
# Based on actual implementations in runner.py:
CALLS_PER_STRATEGY = {
    "P0": 1,           # Baseline: single zero-shot call
    "P1": 10,          # ICL single-type: 10 calls (one per DASP vulnerability type)
    "P2": 1,           # Structured reasoning/CoT: single call
    "P3": -1,          # SmartGuard: variable (use get_smartguard_calls), default ~24 with CoT+selfcheck
    "P4": 4,           # GPTScan tool-augmented: 3-4 stage calls (scenario+property+extraction+confirmation)
    "P5": 4,           # SmartAudit multi-agent: 4 agents (PM, Auditor, Expert, Counselor)
    "P6": 0,           # iAudit: local fine-tuned models, no API cost
}

# P3 SmartGuard CoT expansion settings (when enable_cot_expansion=true)
# Per pattern: 1 (zero-shot) + up to 6 (rethink) + 1 (hint if wrong) + 1 (confirm) = 2-9 calls
COT_CALLS_PER_PATTERN_MIN = 2   # Best case: zero-shot correct + confirmation
COT_CALLS_PER_PATTERN_AVG = 4   # Average case: 1-2 rethinks + confirmation
COT_CALLS_PER_PATTERN_MAX = 9   # Worst case: all rethinks + hint + confirmation

# P3 SmartGuard Iterative Self-Check settings (when enable_iterative_selfcheck=true)
# Per iteration: 1 (review) + 0.5 (rethink, ~50% of iterations need rethink)
SELFCHECK_CALLS_PER_ITERATION = 1.5  # Average: review + occasional rethink


def get_smartguard_calls(
    top_k_patterns: int = 5,
    enable_cot_expansion: bool = True,
    enable_iterative_selfcheck: bool = False,
    selfcheck_max_iterations: int = 3,
    estimate: str = "avg",
    cache_hit_rate: float = 0.0,
) -> int:
    """
    Calculate P3 SmartGuard API calls with all optional stages.

    Args:
        top_k_patterns: Number of patterns to retrieve (default: 5)
        enable_cot_expansion: If True, generate CoT for each pattern
        enable_iterative_selfcheck: If True, run multi-turn self-check
        selfcheck_max_iterations: Max iterations for self-check (default: 3)
        estimate: "min", "avg", or "max" for call estimation
        cache_hit_rate: Fraction of CoT calls that hit cache (0.0 = first run)

    Returns:
        Total API calls for P4 SmartGuard
    """
    total_calls = 0

    # Stage 1: CoT Expansion (if enabled)
    if enable_cot_expansion:
        if estimate == "min":
            cot_calls_per_pattern = COT_CALLS_PER_PATTERN_MIN
        elif estimate == "max":
            cot_calls_per_pattern = COT_CALLS_PER_PATTERN_MAX
        else:  # avg
            cot_calls_per_pattern = COT_CALLS_PER_PATTERN_AVG

        # Apply cache hit rate (cached CoTs = 0 API calls)
        effective_patterns = int(top_k_patterns * (1.0 - cache_hit_rate))
        total_calls += effective_patterns * cot_calls_per_pattern

    # Stage 2: ICL Audit (always 1 call)
    total_calls += 1

    # Stage 3: Iterative Self-Check (if enabled)
    if enable_iterative_selfcheck:
        # Estimate actual iterations based on estimate type
        if estimate == "min":
            actual_iterations = 1  # Converges immediately
        elif estimate == "max":
            actual_iterations = selfcheck_max_iterations
        else:  # avg
            actual_iterations = max(1, selfcheck_max_iterations - 1)  # Usually converges before max

        total_calls += int(actual_iterations * SELFCHECK_CALLS_PER_ITERATION)

    return total_calls


def get_strategy_calls(prompt_id: str) -> int:
    """Get API calls for a prompt strategy, handling various ID formats.

    Handles: "P0", "P0_baseline", "p0_baseline", "P2_icl", etc.
    """
    # Normalize: extract base strategy like "P0", "P2", "P6"
    prompt_upper = prompt_id.upper()

    # Try exact match first
    if prompt_upper in CALLS_PER_STRATEGY:
        return CALLS_PER_STRATEGY[prompt_upper]

    # Extract prefix (e.g., "P0" from "P0_baseline")
    if "_" in prompt_id:
        prefix = prompt_id.split("_")[0].upper()
        if prefix in CALLS_PER_STRATEGY:
            return CALLS_PER_STRATEGY[prefix]

    # Try matching just the number part (e.g., "0" from "P0_baseline")
    import re
    match = re.match(r"P?(\d+)", prompt_upper)
    if match:
        key = f"P{match.group(1)}"
        if key in CALLS_PER_STRATEGY:
            return CALLS_PER_STRATEGY[key]

    return 1  # Default fallback


@dataclass
class CostEstimate:
    """Cost estimate for an experiment."""
    total_contracts: int
    total_api_calls: int
    total_input_tokens: int
    total_output_tokens: int
    cost_by_model: Dict[str, float]
    cost_by_prompt: Dict[str, float]
    total_cost_usd: float

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "COST ESTIMATE (DRY RUN)",
            "=" * 60,
            f"Total contracts: {self.total_contracts:,}",
            f"Total API calls: {self.total_api_calls:,}",
            f"Est. input tokens: {self.total_input_tokens:,}",
            f"Est. output tokens: {self.total_output_tokens:,}",
            "",
            "Cost by Model:",
        ]
        for model, cost in sorted(self.cost_by_model.items()):
            lines.append(f"  {model}: ${cost:.2f}")

        lines.append("")
        lines.append("Cost by Prompt Strategy:")
        for prompt, cost in sorted(self.cost_by_prompt.items()):
            lines.append(f"  {prompt}: ${cost:.2f}")

        lines.extend([
            "",
            "-" * 60,
            f"TOTAL ESTIMATED COST: ${self.total_cost_usd:.2f}",
            "-" * 60,
            "",
            "Note: Actual costs may vary by ±20% based on:",
            "  - Contract length variation",
            "  - Model output verbosity",
            "  - API pricing changes",
        ])

        return "\n".join(lines)


def estimate_tokens_for_contract(source_code: str) -> int:
    """Estimate token count for a contract."""
    return len(source_code) // CHARS_PER_TOKEN


def get_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a model call."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        # Default to expensive pricing if model unknown
        pricing = {"input": 15.00, "output": 75.00}

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def estimate_experiment_cost(
    contracts: List[Dict],
    models: List[str],
    prompts: List[str],
    runs_per_contract: int = 1,
    enable_cot_expansion: bool = True,
    top_k_patterns: int = 5,
    enable_iterative_selfcheck: bool = False,
    selfcheck_max_iterations: int = 3,
    cot_cache_hit_rate: float = 0.0,
) -> CostEstimate:
    """
    Estimate total cost for an experiment.

    Args:
        contracts: List of contract dicts with 'source' key
        models: List of model names
        prompts: List of prompt strategy IDs (P0, P1, etc.)
        runs_per_contract: Number of runs per contract (for averaging)
        enable_cot_expansion: If True, P4 uses CoT expansion (default: True)
        top_k_patterns: Number of patterns for P4 RAG (affects CoT call count)
        enable_iterative_selfcheck: If True, P4 runs multi-turn self-check
        selfcheck_max_iterations: Max iterations for self-check (default: 3)
        cot_cache_hit_rate: Fraction of CoT calls that hit cache (0.0 = first run)

    Returns:
        CostEstimate with detailed breakdown
    """
    total_api_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    cost_by_model: Dict[str, float] = {m: 0.0 for m in models}
    cost_by_prompt: Dict[str, float] = {p: 0.0 for p in prompts}

    # Average contract size
    avg_contract_tokens = 0
    for contract in contracts:
        source = contract.get("source", "")
        avg_contract_tokens += estimate_tokens_for_contract(source)
    avg_contract_tokens = avg_contract_tokens // len(contracts) if contracts else 500

    # Estimate output tokens (typically 500-1500 per response)
    avg_output_tokens = 800

    for model in models:
        for prompt in prompts:
            # Check if this is P3 SmartGuard (RAG + CoT + Self-check)
            prompt_upper = prompt.upper()
            is_p3_smartguard = prompt_upper.startswith("P3") or prompt_upper == "P3"
            if is_p3_smartguard:
                calls_per_contract = get_smartguard_calls(
                    top_k_patterns=top_k_patterns,
                    enable_cot_expansion=enable_cot_expansion,
                    enable_iterative_selfcheck=enable_iterative_selfcheck,
                    selfcheck_max_iterations=selfcheck_max_iterations,
                    estimate="avg",
                    cache_hit_rate=cot_cache_hit_rate,
                )
            else:
                calls_per_contract = get_strategy_calls(prompt)

            # Get prompt overhead
            prompt_key = f"p{prompt[1:]}_" if prompt.startswith("P") else prompt
            overhead = 400  # Default
            for key, tokens in PROMPT_OVERHEAD.items():
                if prompt_key.lower() in key.lower() or key.lower() in prompt_key.lower():
                    overhead = tokens
                    break

            for contract in contracts:
                source = contract.get("source", "")
                contract_tokens = estimate_tokens_for_contract(source)

                for run in range(runs_per_contract):
                    # Calculate tokens for this call
                    input_tokens = contract_tokens + overhead
                    output_tokens = avg_output_tokens

                    # Multiply by calls per strategy
                    total_input = input_tokens * calls_per_contract
                    total_output = output_tokens * calls_per_contract

                    # Calculate cost
                    cost = get_model_cost(model, total_input, total_output)

                    # Update totals
                    total_api_calls += calls_per_contract
                    total_input_tokens += total_input
                    total_output_tokens += total_output
                    cost_by_model[model] += cost
                    cost_by_prompt[prompt] += cost

    total_cost = sum(cost_by_model.values())

    return CostEstimate(
        total_contracts=len(contracts),
        total_api_calls=total_api_calls,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        cost_by_model=cost_by_model,
        cost_by_prompt=cost_by_prompt,
        total_cost_usd=total_cost,
    )


def estimate_from_config(config_path: str | Path) -> CostEstimate:
    """Estimate cost from experiment config file."""
    import yaml

    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())

    # Extract models
    models = [m["name"] for m in config.get("models", [])]

    # Extract prompts (handle both list of dicts and list of strings)
    raw_prompts = config.get("prompts", [])
    if raw_prompts and isinstance(raw_prompts[0], dict):
        prompts = [p.get("id", f"P{i}") for i, p in enumerate(raw_prompts)]
    else:
        prompts = raw_prompts or ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

    # Load datasets (mock for estimation)
    datasets = config.get("datasets", [])

    # Create mock contracts based on dataset sizes
    contracts = []
    for ds in datasets:
        # Estimate contract counts
        if "smartbugs" in ds.get("name", "").lower():
            count = 143
            avg_lines = 50
        elif "solidifi" in ds.get("name", "").lower():
            count = 350
            avg_lines = 80
        else:
            count = ds.get("sample_size", 100)
            avg_lines = 60

        for i in range(count):
            # Mock contract with estimated size
            contracts.append({
                "source": "x" * (avg_lines * 40),  # ~40 chars per line
                "id": f"{ds.get('name', 'unknown')}_{i}",
            })

    runs = config.get("runs_per_item", config.get("runs_per_contract", 1))

    # Extract P4 settings from tools config
    tools_config = config.get("tools", {})
    enable_cot_expansion = tools_config.get("enable_cot_expansion", True)
    top_k_patterns = tools_config.get("top_k_patterns", 5)
    enable_iterative_selfcheck = tools_config.get("enable_iterative_selfcheck", False)
    selfcheck_max_iterations = tools_config.get("selfcheck_max_iterations", 3)

    return estimate_experiment_cost(
        contracts, models, prompts, runs,
        enable_cot_expansion=enable_cot_expansion,
        top_k_patterns=top_k_patterns,
        enable_iterative_selfcheck=enable_iterative_selfcheck,
        selfcheck_max_iterations=selfcheck_max_iterations,
    )


def print_cost_breakdown():
    """Print model pricing for reference."""
    print("=" * 60)
    print("MODEL PRICING (USD per 1M tokens)")
    print("=" * 60)
    print(f"{'Model':<35} {'Input':>10} {'Output':>10}")
    print("-" * 60)

    for model, prices in sorted(MODEL_PRICING.items()):
        print(f"{model:<35} ${prices['input']:>8.2f} ${prices['output']:>8.2f}")

    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        estimate = estimate_from_config(config_path)
        print(estimate)
    else:
        print("Usage: python cost_estimator.py <config.yaml>")
        print()
        print_cost_breakdown()
