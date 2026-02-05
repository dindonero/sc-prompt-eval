"""
Base classes for static confirmation modules.

Per GPTScan paper Section 4.4, static confirmation modules verify
LLM-extracted key variables and statements using program analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..parsing.findings import Finding


@dataclass
class ConfirmationResult:
    """Result of static confirmation check."""

    confirmed: bool
    """Whether the vulnerability was confirmed by static analysis."""

    confidence: float = 0.5
    """Confidence level (0.0 to 1.0) of the confirmation."""

    reason: str = ""
    """Human-readable explanation of the confirmation result."""

    evidence: Optional[Dict[str, Any]] = None
    """Additional evidence from static analysis (variable names, CFG paths, etc.)."""

    check_type: str = ""
    """Type of check performed (DF, VC, OC, FA)."""

    def __post_init__(self):
        """Ensure confidence is in valid range."""
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class ExtractionData:
    """Structured extraction data from LLM Stage 3."""

    variable_a: Optional[Dict[str, str]] = None
    """Primary variable extracted (name, description)."""

    variable_b: Optional[Dict[str, str]] = None
    """Secondary variable extracted (name, description)."""

    key_statements: List[str] = field(default_factory=list)
    """Key statements identified by LLM."""

    vulnerable_lines: List[int] = field(default_factory=list)
    """Line numbers of vulnerable code."""

    raw_extraction: Optional[Dict] = None
    """Raw extraction JSON from LLM."""

    @classmethod
    def from_dict(cls, data: Dict) -> "ExtractionData":
        """Create ExtractionData from LLM extraction dict."""
        return cls(
            variable_a=data.get("VariableA"),
            variable_b=data.get("VariableB"),
            key_statements=data.get("key_statements", []),
            vulnerable_lines=data.get("vulnerable_line_numbers", []),
            raw_extraction=data,
        )

    @property
    def var_a_name(self) -> Optional[str]:
        """Get name of variable A."""
        if self.variable_a:
            return self.variable_a.get("name")
        return None

    @property
    def var_b_name(self) -> Optional[str]:
        """Get name of variable B."""
        if self.variable_b:
            return self.variable_b.get("name")
        return None


class BaseConfirmer(ABC):
    """
    Abstract base class for static confirmation modules.

    Each confirmer implements a specific type of program analysis
    to verify LLM-extracted vulnerability findings.
    """

    check_type: str = "BASE"
    """Identifier for this confirmation type (DF, VC, OC, FA)."""

    @abstractmethod
    def confirm(
        self,
        finding: "Finding",
        extraction: ExtractionData,
        slither_instance: Any,
        contract_source: str,
    ) -> ConfirmationResult:
        """
        Verify a finding using static analysis.

        Args:
            finding: The vulnerability finding to verify
            extraction: Extracted key variables/statements from LLM
            slither_instance: Slither analysis object (or None if unavailable)
            contract_source: Original Solidity source code

        Returns:
            ConfirmationResult with confirmation status and evidence
        """
        pass

    def _find_function_in_slither(self, slither_instance: Any, func_name: str):
        """
        Find a function by name in Slither's analyzed contracts.

        Args:
            slither_instance: Slither analysis object
            func_name: Name of function to find

        Returns:
            Slither Function object or None
        """
        if slither_instance is None:
            return None

        for contract in slither_instance.contracts:
            for func in contract.functions:
                if func.name == func_name:
                    return func

        return None

    def _find_variable_in_function(self, func: Any, var_name: str):
        """
        Find a variable by name within a function.

        Args:
            func: Slither Function object
            var_name: Name of variable to find

        Returns:
            Slither Variable object or None
        """
        if func is None or not var_name:
            return None

        # Check local variables
        for var in func.local_variables:
            if var.name == var_name:
                return var

        # Check state variables read/written
        for var in func.state_variables_read + func.state_variables_written:
            if var.name == var_name:
                return var

        # Check parameters
        for param in func.parameters:
            if param.name == var_name:
                return param

        return None


def run_confirmation_pipeline(
    finding: "Finding",
    extraction: Dict,
    scenario_id: str,
    slither_instance: Any,
    contract_source: str,
    confirmer_map: Optional[Dict[str, List[BaseConfirmer]]] = None,
) -> "Finding":
    """
    Run the full confirmation pipeline for a finding.

    Per GPTScan Table 1, each vulnerability type has specific
    static confirmation modules to apply.

    Args:
        finding: Finding to confirm
        extraction: Raw extraction dict from LLM
        scenario_id: ID of the matched scenario
        slither_instance: Slither analysis object
        contract_source: Original source code
        confirmer_map: Optional custom mapping of scenario->confirmers

    Returns:
        Updated Finding with confirmation results
    """
    from .dataflow import DataFlowConfirmer
    from .value_comparison import ValueComparisonConfirmer
    from .order_check import OrderCheckConfirmer
    from .function_args import FunctionArgConfirmer

    # Default confirmer map per DASP10 categories (adapted from GPTScan methodology)
    if confirmer_map is None:
        confirmer_map = {
            # DASP10 categories with appropriate static checks
            "reentrancy": [OrderCheckConfirmer()],  # OC: state update order vs external call
            "access_control": [ValueComparisonConfirmer()],  # VC: owner/sender checks
            "arithmetic": [DataFlowConfirmer()],  # DF: SafeMath usage, overflow checks
            "unchecked_low_level_calls": [ValueComparisonConfirmer()],  # VC: return value checked
            "denial_of_service": [DataFlowConfirmer()],  # DF: loop bounds, dependencies
            "bad_randomness": [DataFlowConfirmer()],  # DF: randomness source tracing
            "front_running": [FunctionArgConfirmer()],  # FA: msg.sender verification
            "time_manipulation": [ValueComparisonConfirmer()],  # VC: timestamp comparison
            "short_addresses": [ValueComparisonConfirmer()],  # VC: input length validation
            "other": [DataFlowConfirmer()],  # DF: storage pointer initialization
        }

    confirmers = confirmer_map.get(scenario_id, [])
    if not confirmers:
        # No specific confirmers for this scenario
        finding.static_confirmed = None
        return finding

    extraction_data = ExtractionData.from_dict(extraction)
    confirmation_results = []

    for confirmer in confirmers:
        result = confirmer.confirm(
            finding=finding,
            extraction=extraction_data,
            slither_instance=slither_instance,
            contract_source=contract_source,
        )
        confirmation_results.append(result)

        if not result.confirmed:
            # Per GPTScan: if any confirmation fails, demote finding
            finding.static_confirmed = False
            finding.confidence *= 0.3  # Reduce confidence significantly
            finding.static_check_reason = result.reason
            return finding

    # All confirmations passed
    finding.static_confirmed = True
    finding.confidence = min(finding.confidence * 1.2, 1.0)  # Boost confidence
    finding.static_check_reason = "All static checks passed"

    return finding
