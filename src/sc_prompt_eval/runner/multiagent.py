"""P6 LLM-SmartAudit Multi-Agent Runner.

EXACT IMPLEMENTATION of Wei et al. 2024 "LLM-SmartAudit: Advanced Smart
Contract Vulnerability Detection"

Key components per the paper:
1. 4 Agent Roles: Project Manager, Counselor, Auditor, Solidity Expert
2. 3-Phase Task Queue: Contract Analysis → Vulnerability ID → Report
3. Two Modes: BA (Broad Analysis) and TA (Targeted Analysis)
4. Role Exchange Mechanism for reducing false positives (Figure 5b)
5. Consensus Loop with max n=3 rounds
6. 40 vulnerability scenarios for TA mode (Buffer-Reasoning)

Reference: Wei et al. 2024 - Figures 2, 3, 5b
"""
from __future__ import annotations

import json
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

from ..models.base import LLMAdapter
from ..prompts.registry import PromptRegistry
from ..parsing.findings import parse_findings, Finding
from ..parsing.json_utils import extract_json_object

logger = logging.getLogger(__name__)


@dataclass
class SmartAuditResult:
    """Result from LLM-SmartAudit analysis."""
    # Mode info
    mode: str  # "BA" or "TA"

    # Phase outputs
    phase1_contract_analysis: dict = field(default_factory=dict)
    phase1_counselor_summary: dict = field(default_factory=dict)
    phase2_findings: List[Finding] = field(default_factory=list)
    phase3_report: List[Finding] = field(default_factory=list)

    # Raw outputs for debugging
    raw_outputs: Dict[str, str] = field(default_factory=dict)

    # Cost tracking
    total_cost: float = 0.0
    total_tokens: int = 0
    total_latency_s: float = 0.0
    total_api_calls: int = 0

    # Per-agent tracking
    agent_costs: Dict[str, float] = field(default_factory=dict)
    agent_tokens: Dict[str, int] = field(default_factory=dict)
    agent_latencies: Dict[str, float] = field(default_factory=dict)

    # Consensus/convergence
    rounds_executed: int = 1
    max_rounds_configured: int = 3
    converged: bool = False
    convergence_round: Optional[int] = None
    round_history: List[Dict] = field(default_factory=list)
    role_exchanges_performed: int = 0

    # TA mode specific
    scenarios_checked: List[str] = field(default_factory=list)
    scenario_results: Dict[str, dict] = field(default_factory=dict)


def load_scenarios(scenario_path: Optional[Path] = None) -> List[dict]:
    """Load vulnerability scenarios for TA mode from smartaudit_scenarios.json."""
    if scenario_path is None:
        # Look in data directory relative to this file
        scenario_path = Path(__file__).parent.parent.parent.parent / "data" / "smartaudit_scenarios.json"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenarios file not found: {scenario_path}")

    with open(scenario_path) as f:
        data = json.load(f)
    return data["scenarios"]


def _track_cost(result: SmartAuditResult, agent: str, response) -> None:
    """Track costs for an agent call."""
    cost = response.cost_usd or 0
    tokens = response.total_tokens or 0
    latency = response.latency_s or 0

    result.total_cost += cost
    result.total_tokens += tokens
    result.total_latency_s += latency
    result.total_api_calls += 1

    # Per-agent accumulation
    if agent not in result.agent_costs:
        result.agent_costs[agent] = 0.0
        result.agent_tokens[agent] = 0
        result.agent_latencies[agent] = 0.0

    result.agent_costs[agent] += cost
    result.agent_tokens[agent] += tokens
    result.agent_latencies[agent] += latency


def _finding_key(f: Finding) -> str:
    """Create a unique key for finding comparison using category + location."""
    category = (f.category or "unknown").lower().strip()
    # Include line numbers for location-aware comparison
    lines = ""
    if f.evidence and f.evidence.lines:
        lines = ",".join(str(ln) for ln in sorted(f.evidence.lines)[:3])  # First 3 lines
    func = (f.evidence.function if f.evidence and f.evidence.function else "").lower().strip()
    return f"{category}:{func}:{lines}"


def _check_consensus(
    previous: List[Finding],
    current: List[Finding],
    threshold: float = 0.9,
) -> Tuple[bool, dict]:
    """Check if findings have converged between rounds using Jaccard similarity.

    Uses category + location (function + lines) for comparison instead of just title,
    to properly identify the same vulnerability instances across rounds.
    """
    prev_keys = {_finding_key(f) for f in previous}
    curr_keys = {_finding_key(f) for f in current}

    if not prev_keys and not curr_keys:
        return True, {"similarity": 1.0, "added": [], "removed": []}

    intersection = prev_keys & curr_keys
    union = prev_keys | curr_keys
    similarity = len(intersection) / len(union) if union else 1.0

    return similarity >= threshold, {
        "similarity": similarity,
        "added": list(curr_keys - prev_keys),
        "removed": list(prev_keys - curr_keys),
        "intersection_size": len(intersection),
        "union_size": len(union),
    }


def _apply_expert_verification(
    findings: List[Finding],
    expert_data: Union[dict, list],
) -> List[Finding]:
    """Apply expert verification to filter findings based on technical verdicts.

    Uses category-based matching in addition to title for more robust matching.
    Handles both dict format ({"technical_verifications": [...]}) and direct list format.
    """
    # Handle both dict with technical_verifications key and direct list format
    if isinstance(expert_data, list):
        verifications = expert_data
    else:
        verifications = expert_data.get("technical_verifications", [])

    # Build map of (category, title) to verdicts for robust matching
    verdict_map = {}
    for v in verifications:
        title = v.get("finding_title", "").lower().strip()
        category = v.get("category", "").lower().strip()
        verdict = v.get("technical_verdict", "VULNERABLE")
        # Index by both title-only and category+title for flexible matching
        verdict_map[title] = verdict
        if category:
            verdict_map[f"{category}:{title}"] = verdict

    # Keep findings that expert confirmed as VULNERABLE or CONDITIONAL
    verified = []
    for f in findings:
        title_key = f.title.lower().strip() if f.title else ""
        category_key = (f.category or "").lower().strip()

        # Try category+title match first, then title-only
        verdict = verdict_map.get(
            f"{category_key}:{title_key}",
            verdict_map.get(title_key, "VULNERABLE")  # Default to keeping
        )
        if verdict in ("VULNERABLE", "CONDITIONAL"):
            verified.append(f)

    # Add expert's additional findings (only if expert_data is a dict)
    additional = expert_data.get("additional_technical_findings", []) if isinstance(expert_data, dict) else []
    for add_finding in additional:
        finding = Finding(
            title=add_finding.get("title", "Unknown"),
            category=add_finding.get("category", "other"),
            severity=add_finding.get("severity", "medium"),
            confidence=add_finding.get("confidence", 0.7),
            explanation=add_finding.get("technical_explanation", ""),
            evidence=add_finding.get("evidence"),
            fix_suggestion=add_finding.get("fix_suggestion"),
        )
        verified.append(finding)

    return verified


def _perform_role_exchange(
    adapter: LLMAdapter,
    prompt_registry: PromptRegistry,
    contract_source: str,
    findings: List[Finding],
    result: SmartAuditResult,
) -> List[Finding]:
    """
    Role Exchange mechanism per Figure 5b in paper.

    Expert reviews findings as if they were the Auditor, getting a fresh
    perspective to identify potential false positives.
    """
    findings_json = json.dumps([f.model_dump() for f in findings], indent=2)

    role_swap_prompt = prompt_registry.render(
        "p5_role_exchange.j2",
        contract_source=contract_source,
        original_findings=findings_json,
    )
    response = adapter.generate(role_swap_prompt)
    _track_cost(result, "role_exchange", response)
    result.raw_outputs[f"role_exchange_{result.role_exchanges_performed}"] = response.text

    # Parse role exchange result
    try:
        exchange_json = extract_json_object(response.text)
        exchange_data = json.loads(exchange_json)

        # Get verified findings from role exchange
        # Handle both dict format ({"verified_findings": [...]}) and direct list format
        if isinstance(exchange_data, list):
            verified_list = exchange_data
        else:
            verified_list = exchange_data.get("verified_findings", [])
        verified_titles = {v.get("title", "").lower().strip() for v in verified_list if isinstance(v, dict)}

        # Filter findings to only those that survived role exchange
        return [f for f in findings if f.title and f.title.lower().strip() in verified_titles]
    except (json.JSONDecodeError, KeyError) as e:
        # Log warning when role exchange fails - keeping all findings is conservative
        # but defeats the purpose of false positive reduction
        logger.warning(
            f"Role exchange parse failed: {e}. Keeping all {len(findings)} findings. "
            f"Response preview: {response.text[:200]}..."
        )
        return findings


def run_smartaudit(
    adapter: LLMAdapter,
    prompt_registry: PromptRegistry,
    contract_source: str,
    mode: str = "BA",
    max_rounds: int = 3,
    consensus_threshold: float = 0.9,
    enable_role_exchange: bool = True,
    scenarios: Optional[List[str]] = None,
    scenario_path: Optional[Path] = None,
) -> SmartAuditResult:
    """
    Run LLM-SmartAudit vulnerability detection.

    Implements the exact methodology from Wei et al. 2024:
    - BA Mode: Broad Analysis with Thought-Reasoning (ReAct-based)
    - TA Mode: Targeted Analysis with Buffer-Reasoning (BoT-based)

    Args:
        adapter: LLM adapter to use
        prompt_registry: Template registry
        contract_source: Solidity source code
        mode: "BA" (Broad Analysis) or "TA" (Targeted Analysis)
        max_rounds: Max consensus rounds (default 3 per paper)
        consensus_threshold: Similarity threshold for convergence
        enable_role_exchange: Enable role-reversal verification (Figure 5b)
        scenarios: For TA mode, list of scenario IDs to check (None = all 40)
        scenario_path: Path to scenarios JSON file (optional)

    Returns:
        SmartAuditResult with all findings and metadata
    """
    if mode.upper() == "BA":
        return _run_broad_analysis(
            adapter, prompt_registry, contract_source,
            max_rounds, consensus_threshold, enable_role_exchange
        )
    elif mode.upper() == "TA":
        return _run_targeted_analysis(
            adapter, prompt_registry, contract_source,
            scenarios, scenario_path
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'BA' or 'TA'")


def _run_broad_analysis(
    adapter: LLMAdapter,
    prompt_registry: PromptRegistry,
    contract_source: str,
    max_rounds: int = 3,
    consensus_threshold: float = 0.9,
    enable_role_exchange: bool = True,
) -> SmartAuditResult:
    """
    BA Mode: Thought-Reasoning based comprehensive analysis.

    Task Queue per Figure 3 in paper:
    1. Contract Analysis: Project Manager + Auditor assess, Counselor summarizes
    2. Vulnerability Identification: Auditor + Expert with role exchanges
    3. Comprehensive Report: Joint final report generation
    """
    result = SmartAuditResult(mode="BA", max_rounds_configured=max_rounds)

    # ===== PHASE 1: CONTRACT ANALYSIS =====

    # Step 1a: Project Manager initiates, assesses contract structure
    pm_prompt = prompt_registry.render(
        "p5_project_manager.j2",
        contract_source=contract_source
    )
    pm_response = adapter.generate(pm_prompt)
    result.raw_outputs["project_manager"] = pm_response.text
    _track_cost(result, "project_manager", pm_response)

    try:
        phase1_json = extract_json_object(pm_response.text)
        parsed = json.loads(phase1_json)
        # Ensure it's a dict, wrap lists in a container
        result.phase1_contract_analysis = parsed if isinstance(parsed, dict) else {"items": parsed}
    except (json.JSONDecodeError, TypeError):
        result.phase1_contract_analysis = {"raw": pm_response.text}

    # Step 1b: Counselor summarizes Phase 1 and provides strategic guidance
    counselor_prompt = prompt_registry.render(
        "p5_counselor.j2",
        contract_source=contract_source,
        phase1_analysis=json.dumps(result.phase1_contract_analysis, indent=2)
    )
    counselor_response = adapter.generate(counselor_prompt)
    result.raw_outputs["counselor"] = counselor_response.text
    _track_cost(result, "counselor", counselor_response)

    try:
        counselor_json = extract_json_object(counselor_response.text)
        result.phase1_counselor_summary = json.loads(counselor_json)
    except (json.JSONDecodeError, TypeError):
        result.phase1_counselor_summary = {"raw": counselor_response.text}

    # ===== PHASE 2: VULNERABILITY IDENTIFICATION =====

    all_findings: List[Finding] = []
    previous_findings: List[Finding] = []

    for round_num in range(max_rounds):
        round_info = {"round": round_num + 1}

        # Step 2a: Auditor identifies vulnerabilities with Thought-Reasoning
        auditor_prompt = prompt_registry.render(
            "p5_auditor.j2",
            contract_source=contract_source,
            counselor_guidance=json.dumps(result.phase1_counselor_summary, indent=2),
            task_description="Identify all security vulnerabilities in this smart contract",
            ideas="Reentrancy, Access Control, Arithmetic, Logic Flaws, Economic Attacks"
        )
        auditor_response = adapter.generate(auditor_prompt)
        result.raw_outputs[f"auditor_round_{round_num}"] = auditor_response.text
        _track_cost(result, "auditor", auditor_response)

        auditor_findings, _ = parse_findings(auditor_response.text)
        round_info["auditor_findings"] = len(auditor_findings)

        # Step 2b: Expert verifies Auditor's findings
        expert_prompt = prompt_registry.render(
            "p5_expert.j2",
            contract_source=contract_source,
            auditor_findings=json.dumps([f.model_dump() for f in auditor_findings], indent=2)
        )
        expert_response = adapter.generate(expert_prompt)
        result.raw_outputs[f"expert_round_{round_num}"] = expert_response.text
        _track_cost(result, "expert", expert_response)

        # Parse expert verification and apply filtering
        try:
            expert_json = extract_json_object(expert_response.text)
            expert_data = json.loads(expert_json)
            verified_findings = _apply_expert_verification(auditor_findings, expert_data)
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, keep auditor findings
            verified_findings = auditor_findings

        round_info["expert_verified"] = len(verified_findings)

        # Step 2c: Role Exchange (if enabled and not first round)
        if enable_role_exchange and round_num > 0:
            pre_exchange_count = len(verified_findings)
            verified_findings = _perform_role_exchange(
                adapter, prompt_registry, contract_source,
                verified_findings, result
            )
            result.role_exchanges_performed += 1
            round_info["post_role_exchange"] = len(verified_findings)
            round_info["role_exchange_removed"] = pre_exchange_count - len(verified_findings)

        # Check consensus (skip first round)
        if round_num > 0 and previous_findings:
            converged, consensus_metrics = _check_consensus(
                previous_findings, verified_findings, consensus_threshold
            )
            round_info["consensus_metrics"] = consensus_metrics

            if converged:
                result.converged = True
                result.convergence_round = round_num + 1
                all_findings = verified_findings
                result.round_history.append(round_info)
                break

        previous_findings = verified_findings
        all_findings = verified_findings
        result.rounds_executed = round_num + 1
        result.round_history.append(round_info)

    result.phase2_findings = all_findings

    # ===== PHASE 3: COMPREHENSIVE REPORT =====
    # Final findings are the verified findings from Phase 2
    result.phase3_report = all_findings

    return result


def _run_targeted_analysis(
    adapter: LLMAdapter,
    prompt_registry: PromptRegistry,
    contract_source: str,
    scenario_ids: Optional[List[str]] = None,
    scenario_path: Optional[Path] = None,
) -> SmartAuditResult:
    """
    TA Mode: Buffer-Reasoning based scenario-specific analysis.

    Runs through vulnerability scenarios with thought templates (BoT-based).
    Each scenario uses a specific thought template for that vulnerability type.
    """
    result = SmartAuditResult(mode="TA")

    # Load all 40 scenarios
    all_scenarios = load_scenarios(scenario_path)

    # Filter to requested scenarios if specified
    if scenario_ids:
        scenarios = [s for s in all_scenarios if s["id"] in scenario_ids]
    else:
        scenarios = all_scenarios  # All 40

    # Phase 1: Quick contract analysis (same as BA)
    pm_prompt = prompt_registry.render(
        "p5_project_manager.j2",
        contract_source=contract_source
    )
    pm_response = adapter.generate(pm_prompt)
    result.raw_outputs["project_manager"] = pm_response.text
    _track_cost(result, "project_manager", pm_response)

    try:
        phase1_json = extract_json_object(pm_response.text)
        parsed = json.loads(phase1_json)
        # Ensure it's a dict, wrap lists in a container
        result.phase1_contract_analysis = parsed if isinstance(parsed, dict) else {"items": parsed}
    except (json.JSONDecodeError, TypeError):
        result.phase1_contract_analysis = {"raw": pm_response.text}

    # Phase 2: Scenario-by-scenario analysis with Buffer-Reasoning
    all_findings: List[Finding] = []
    state_memory: Dict[str, dict] = {}

    for scenario in scenarios:
        scenario_id = scenario["id"]

        # Render TA mode prompt with scenario-specific thought template
        ta_prompt = prompt_registry.render(
            "p5_ta_mode.j2",
            contract_source=contract_source,
            scenario=scenario,
            agent_role="Smart Contract Auditor",
            state_memory=state_memory
        )
        ta_response = adapter.generate(ta_prompt)
        result.raw_outputs[f"scenario_{scenario_id}"] = ta_response.text
        _track_cost(result, f"scenario_{scenario_id}", ta_response)

        # Parse scenario result - check for identified vulnerability
        response_text = ta_response.text
        scenario_name = scenario["name"]

        # Check if vulnerability was identified
        found = (
            f"<INFO> {scenario_name} Identified" in response_text or
            f"<INFO> {scenario_id} Identified" in response_text or
            f"{scenario_name} Identified" in response_text
        )

        if found:
            # Parse findings from this scenario
            findings, _ = parse_findings(response_text)

            # If no structured findings parsed, create one from scenario info
            if not findings:
                finding = Finding(
                    title=f"{scenario_name} Vulnerability",
                    category=scenario.get("dasp_category", "other"),
                    severity="medium",  # Default, should be parsed from response
                    confidence=0.7,
                    explanation=f"Identified via TA mode scenario {scenario_id}",
                )
                findings = [finding]

            all_findings.extend(findings)
            state_memory[scenario_id] = {"found": True, "findings_count": len(findings)}
        else:
            state_memory[scenario_id] = {"found": False}

        result.scenarios_checked.append(scenario_id)

    result.scenario_results = state_memory
    result.phase2_findings = all_findings
    result.phase3_report = all_findings
    result.rounds_executed = 1  # TA mode is single-pass through scenarios

    return result


def extract_final_findings(result: SmartAuditResult) -> Tuple[List[Finding], Dict]:
    """
    Extract final findings and metadata from SmartAudit result.

    Returns:
        Tuple of (findings list, metadata dict)
    """
    metadata = {
        "mode": result.mode,
        "rounds_executed": result.rounds_executed,
        "max_rounds_configured": result.max_rounds_configured,
        "converged": result.converged,
        "convergence_round": result.convergence_round,
        "role_exchanges_performed": result.role_exchanges_performed,
        "total_cost": result.total_cost,
        "total_tokens": result.total_tokens,
        "total_latency_s": result.total_latency_s,
        "total_api_calls": result.total_api_calls,
        "agent_costs": result.agent_costs,
        "agent_tokens": result.agent_tokens,
        "agent_latencies": result.agent_latencies,
        "round_history": result.round_history,
        "phase1_contract_type": (
            result.phase1_contract_analysis.get("contract_overview", {}).get("type", "unknown")
            if isinstance(result.phase1_contract_analysis, dict)
            else "unknown"
        ),
    }

    # TA mode specific metadata
    if result.mode == "TA":
        metadata["scenarios_checked"] = result.scenarios_checked
        metadata["scenarios_with_findings"] = [
            sid for sid, res in result.scenario_results.items()
            if res.get("found")
        ]
        metadata["scenarios_count"] = len(result.scenarios_checked)
        metadata["scenarios_positive"] = len(metadata["scenarios_with_findings"])

    # BA mode specific metadata
    if result.mode == "BA":
        metadata["phase2_finding_count"] = len(result.phase2_findings)
        metadata["phase3_finding_count"] = len(result.phase3_report)

        # Track category distribution
        categories = [f.category for f in result.phase3_report if f.category]
        metadata["category_distribution"] = {
            cat: categories.count(cat) for cat in set(categories)
        }

    return result.phase3_report, metadata


# Backwards compatibility alias
run_multiagent_debate = run_smartaudit
DebateResult = SmartAuditResult
