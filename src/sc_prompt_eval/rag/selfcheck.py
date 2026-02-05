"""
Iterative Self-Check for P4 SmartGuard audit outputs.

Implements actual multi-turn verification of findings (not just prompt instructions).
Based on SmartGuard paper's self-check stage, adapted for post-audit validation.

Algorithm:
1. Take initial findings from audit
2. Ask LLM to critically review each finding against specific criteria
3. If findings changed significantly, trigger rethink prompt
4. Repeat until convergence or max iterations

This provides actual iterative validation, unlike the current P4 which only
embeds self-check instructions in the prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.base import LLMAdapter

from ..parsing.findings import Finding, parse_findings


@dataclass
class SelfCheckResult:
    """Result of iterative self-check process."""
    final_findings: List[Finding]
    iterations: int
    converged: bool
    convergence_iteration: Optional[int]

    # Per-iteration history
    iteration_history: List[Dict] = field(default_factory=list)

    # Cost tracking (cumulative across all API calls)
    total_cost: float = 0.0
    total_tokens: int = 0
    total_latency: float = 0.0
    total_api_calls: int = 0

    # Findings evolution
    initial_finding_count: int = 0
    final_finding_count: int = 0
    findings_removed: int = 0
    findings_added: int = 0


# Self-check review prompt - asks LLM to validate its own findings
SELFCHECK_REVIEW_PROMPT = """## SELF-CHECK: VALIDATE YOUR FINDINGS

You previously identified these potential vulnerabilities in a smart contract audit:

**Your Findings:**
```json
{findings_json}
```

**Contract Under Review:**
```solidity
{contract_source}
```

---

## CRITICAL REVIEW CRITERIA

For EACH finding above, carefully verify:

1. **Evidence Accuracy**: Are the line numbers and function names correct? Double-check by scanning the code.

2. **Category Correctness**: Is this actually a {category_if_single} vulnerability, or could it be misclassified?

3. **Exploitability Reality Check**: Can this ACTUALLY be exploited in practice?
   - Is there a concrete attack path?
   - What are the preconditions for exploitation?
   - Are there mitigating factors you may have missed?

4. **False Positive Check**: Could this be a non-issue that looks like a vulnerability?
   - Safe patterns that resemble vulnerable ones
   - Intentional design choices
   - Modern Solidity protections (0.8+ overflow checks, etc.)

5. **Context Consideration**: Did you miss any context that changes the assessment?
   - Upstream validation
   - Access control at caller level
   - Intended behavior

---

## YOUR REVIEW

For each finding, provide one of:
- **KEEP**: Finding is valid and accurate as-is
- **REMOVE**: False positive - explain why
- **MODIFY**: Needs correction - provide updated version

---

## OUTPUT VALIDATED FINDINGS

After your critical review, output ONLY the findings that survived validation.
Do NOT include findings you marked as REMOVE.
If you marked a finding as MODIFY, include the corrected version.

Provide your final validated findings as a JSON array:

```json
[
  {{
    "category": "DASP category",
    "title": "Vulnerability title",
    "evidence": {{
      "file": "contract.sol",
      "lines": [line_numbers],
      "function": "function_name"
    }},
    "explanation": "Clear explanation of the vulnerability",
    "self_check_verdict": "KEEP or MODIFIED"
  }}
]
```

Return `[]` if all findings were false positives.
"""


# Rethink prompt when findings are unstable between iterations
SELFCHECK_RETHINK_PROMPT = """## RETHINK: FINDINGS INSTABILITY DETECTED

Your findings have changed significantly between review iterations, suggesting uncertainty.

**Previous iteration findings:**
```json
{previous_findings}
```

**Current iteration findings:**
```json
{current_findings}
```

**Changes Detected:**
- Findings removed: {removed_count}
- Findings added: {added_count}
- Categories affected: {affected_categories}

---

## FINAL DETERMINATION

Your assessment is unstable. Please make a final, definitive decision:

1. **Why did your assessment change?** What new considerations emerged?

2. **Which version is more accurate?** Trust your more careful analysis.

3. **Are there any findings you're genuinely uncertain about?**
   - If uncertain, apply the conservative security principle: include the finding
   - Better to have a false positive than miss a real vulnerability

4. **Commit to your final answer.** No more changes after this.

---

## FINAL OUTPUT

Provide your DEFINITIVE final findings as a JSON array:

```json
[...]
```
"""


class SelfCheckRunner:
    """
    Iterative self-check for vulnerability findings.

    Makes actual API calls to verify findings, unlike prompt-based self-check
    instructions that execute in a single call.
    """

    def __init__(
        self,
        adapter: "LLMAdapter",
        max_iterations: int = 3,
        convergence_threshold: float = 0.9,
    ):
        """
        Initialize self-check runner.

        Args:
            adapter: LLM adapter for API calls
            max_iterations: Maximum review iterations (default 3)
            convergence_threshold: Jaccard similarity for stopping (default 0.9)
        """
        self.adapter = adapter
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def run_selfcheck(
        self,
        initial_findings: List[Finding],
        contract_source: str,
    ) -> SelfCheckResult:
        """
        Run iterative self-check on initial audit findings.

        Args:
            initial_findings: Findings from initial audit to validate
            contract_source: Original contract source code

        Returns:
            SelfCheckResult with validated findings and metrics
        """
        if not initial_findings:
            # No findings to validate
            return SelfCheckResult(
                final_findings=[],
                iterations=0,
                converged=True,
                convergence_iteration=0,
                initial_finding_count=0,
                final_finding_count=0,
            )

        current_findings = initial_findings
        iteration_history = []

        total_cost = 0.0
        total_tokens = 0
        total_latency = 0.0
        total_api_calls = 0

        converged = False
        convergence_iteration = None

        for i in range(self.max_iterations):
            # Build review prompt
            findings_json = json.dumps(
                [self._finding_to_dict(f) for f in current_findings],
                indent=2
            )

            # Determine category hint (if all findings same category)
            categories = {f.category for f in current_findings if f.category}
            category_hint = list(categories)[0] if len(categories) == 1 else "vulnerability"

            prompt = SELFCHECK_REVIEW_PROMPT.format(
                findings_json=findings_json,
                contract_source=contract_source,
                category_if_single=category_hint,
            )

            # Generate review
            response = self.adapter.generate(prompt)
            total_cost += response.cost_usd or 0
            total_tokens += response.total_tokens or 0
            total_latency += response.latency_s or 0
            total_api_calls += 1

            # Parse validated findings
            validated_findings, errors = parse_findings(response.text)

            # Check convergence using Jaccard similarity
            is_converged, metrics = self._check_convergence(
                current_findings, validated_findings
            )

            iteration_history.append({
                "iteration": i + 1,
                "input_count": len(current_findings),
                "output_count": len(validated_findings),
                "convergence_metrics": metrics,
                "cost": response.cost_usd or 0,
                "tokens": response.total_tokens or 0,
                "parse_errors": errors,
            })

            if is_converged:
                converged = True
                convergence_iteration = i + 1
                current_findings = validated_findings
                break

            # If not converged and not last iteration, trigger rethink
            if i < self.max_iterations - 1 and not is_converged:
                rethink_response = self._run_rethink(
                    current_findings, validated_findings, metrics, contract_source
                )

                total_cost += rethink_response.cost_usd or 0
                total_tokens += rethink_response.total_tokens or 0
                total_latency += rethink_response.latency_s or 0
                total_api_calls += 1

                rethink_findings, _ = parse_findings(rethink_response.text)

                # Use rethink output if it produced results, otherwise keep validated
                current_findings = rethink_findings if rethink_findings else validated_findings
            else:
                current_findings = validated_findings

        return SelfCheckResult(
            final_findings=current_findings,
            iterations=len(iteration_history),
            converged=converged,
            convergence_iteration=convergence_iteration,
            iteration_history=iteration_history,
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_latency=total_latency,
            total_api_calls=total_api_calls,
            initial_finding_count=len(initial_findings),
            final_finding_count=len(current_findings),
            findings_removed=max(0, len(initial_findings) - len(current_findings)),
            findings_added=max(0, len(current_findings) - len(initial_findings)),
        )

    def _run_rethink(
        self,
        previous: List[Finding],
        current: List[Finding],
        metrics: Dict,
        contract_source: str,
    ):
        """Run rethink prompt when findings are unstable."""
        prev_cats = {f.category for f in previous if f.category}
        curr_cats = {f.category for f in current if f.category}
        affected = prev_cats.symmetric_difference(curr_cats)

        prompt = SELFCHECK_RETHINK_PROMPT.format(
            previous_findings=json.dumps([self._finding_to_dict(f) for f in previous], indent=2),
            current_findings=json.dumps([self._finding_to_dict(f) for f in current], indent=2),
            removed_count=len(previous) - len(current) if len(previous) > len(current) else 0,
            added_count=len(current) - len(previous) if len(current) > len(previous) else 0,
            affected_categories=", ".join(affected) if affected else "none",
        )

        return self.adapter.generate(prompt)

    def _check_convergence(
        self,
        previous: List[Finding],
        current: List[Finding],
    ) -> Tuple[bool, Dict]:
        """Check if findings have converged using Jaccard similarity."""
        prev_keys = {self._finding_key(f) for f in previous}
        curr_keys = {self._finding_key(f) for f in current}

        if not prev_keys and not curr_keys:
            return True, {"similarity": 1.0, "added": [], "removed": []}

        intersection = prev_keys & curr_keys
        union = prev_keys | curr_keys
        similarity = len(intersection) / len(union) if union else 1.0

        return similarity >= self.convergence_threshold, {
            "similarity": similarity,
            "added": list(curr_keys - prev_keys),
            "removed": list(prev_keys - curr_keys),
            "intersection_size": len(intersection),
            "union_size": len(union),
        }

    def _finding_key(self, f: Finding) -> str:
        """Generate unique key for finding comparison."""
        cat = (f.category or "unknown").lower().replace(" ", "_")
        func = ""
        if f.evidence and f.evidence.function:
            func = f.evidence.function.lower()
        title_words = (f.title or "")[:30].lower().replace(" ", "_")
        return f"{cat}:{func}:{title_words}"

    def _finding_to_dict(self, f: Finding) -> Dict:
        """Convert Finding to dict for JSON serialization."""
        result = {
            "category": f.category,
            "title": f.title,
            "explanation": f.explanation,
        }
        if f.evidence:
            result["evidence"] = {
                "file": f.evidence.file or "contract.sol",
                "lines": f.evidence.lines or [],
                "function": f.evidence.function,
            }
        if f.severity:
            result["severity"] = f.severity
        if f.confidence is not None:
            result["confidence"] = f.confidence
        return result


def get_selfcheck_runner(adapter: "LLMAdapter", **kwargs) -> SelfCheckRunner:
    """Factory function to create a SelfCheckRunner."""
    return SelfCheckRunner(adapter, **kwargs)
