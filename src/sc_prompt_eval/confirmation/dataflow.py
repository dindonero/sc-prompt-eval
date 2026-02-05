"""
Data Flow (DF) confirmation module.

Uses Slither's data dependency analysis to verify that
extracted variables have the expected data flow relationships.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .base import BaseConfirmer, ConfirmationResult, ExtractionData

if TYPE_CHECKING:
    from ..parsing.findings import Finding


class DataFlowConfirmer(BaseConfirmer):
    """
    Confirms variable dependencies using Slither's data flow analysis.

    Per GPTScan: DF checks whether key variables have data dependencies,
    e.g., whether a price variable depends on a reserve variable.
    """

    check_type = "DF"

    def confirm(
        self,
        finding: "Finding",
        extraction: ExtractionData,
        slither_instance: Any,
        contract_source: str,
    ) -> ConfirmationResult:
        """
        Verify data dependency between extracted variables.

        Args:
            finding: The vulnerability finding to verify
            extraction: Extracted key variables from LLM
            slither_instance: Slither analysis object
            contract_source: Original Solidity source code

        Returns:
            ConfirmationResult indicating whether data flow exists
        """
        var_a_name = extraction.var_a_name
        var_b_name = extraction.var_b_name

        if not var_a_name:
            return ConfirmationResult(
                confirmed=False,
                confidence=0.3,
                reason="Missing primary variable for data flow analysis",
                check_type=self.check_type,
            )

        # If only one variable provided, use single-variable check
        if not var_b_name:
            return self._single_var_check(
                var_a_name, contract_source, finding, slither_instance
            )

        if slither_instance is None:
            # Fall back to regex-based heuristic
            return self._fallback_check(
                var_a_name, var_b_name, contract_source, finding
            )

        # Find the function in Slither
        func_name = finding.evidence.function if finding.evidence else ""
        func = self._find_function_in_slither(slither_instance, func_name)

        if func is None:
            return ConfirmationResult(
                confirmed=False,
                confidence=0.4,
                reason=f"Function '{func_name}' not found in Slither analysis",
                check_type=self.check_type,
            )

        # Find the variables
        var_a = self._find_variable_in_function(func, var_a_name)
        var_b = self._find_variable_in_function(func, var_b_name)

        if var_a is None or var_b is None:
            return self._fallback_check(
                var_a_name, var_b_name, contract_source, finding
            )

        # Check data dependency using Slither's analysis
        try:
            from slither.analyses.data_dependency.data_dependency import is_dependent

            # Check if var_a depends on var_b
            if is_dependent(var_a, var_b, func):
                return ConfirmationResult(
                    confirmed=True,
                    confidence=0.9,
                    reason=f"Data dependency confirmed: {var_a_name} depends on {var_b_name}",
                    evidence={
                        "source_var": var_b_name,
                        "dependent_var": var_a_name,
                        "function": func_name,
                    },
                    check_type=self.check_type,
                )

            # Check reverse dependency
            if is_dependent(var_b, var_a, func):
                return ConfirmationResult(
                    confirmed=True,
                    confidence=0.85,
                    reason=f"Data dependency confirmed: {var_b_name} depends on {var_a_name}",
                    evidence={
                        "source_var": var_a_name,
                        "dependent_var": var_b_name,
                        "function": func_name,
                    },
                    check_type=self.check_type,
                )

            return ConfirmationResult(
                confirmed=False,
                confidence=0.7,
                reason=f"No data dependency found between {var_a_name} and {var_b_name}",
                check_type=self.check_type,
            )

        except ImportError:
            return self._fallback_check(
                var_a_name, var_b_name, contract_source, finding
            )

    def _fallback_check(
        self,
        var_a_name: str,
        var_b_name: str,
        contract_source: str,
        finding: "Finding",
    ) -> ConfirmationResult:
        """
        Fallback regex-based heuristic when Slither unavailable.

        Checks if both variables appear in the same assignment or expression.
        """
        import re

        # Check if both variables appear together in assignments
        # Pattern: var_a = ... var_b ... or var_a op var_b
        pattern = rf'{re.escape(var_a_name)}\s*=\s*[^;]*{re.escape(var_b_name)}'
        if re.search(pattern, contract_source):
            return ConfirmationResult(
                confirmed=True,
                confidence=0.6,
                reason=f"Heuristic: {var_a_name} appears to depend on {var_b_name} (regex match)",
                evidence={"method": "regex_fallback"},
                check_type=self.check_type,
            )

        # Check reverse
        pattern = rf'{re.escape(var_b_name)}\s*=\s*[^;]*{re.escape(var_a_name)}'
        if re.search(pattern, contract_source):
            return ConfirmationResult(
                confirmed=True,
                confidence=0.6,
                reason=f"Heuristic: {var_b_name} appears to depend on {var_a_name} (regex match)",
                evidence={"method": "regex_fallback"},
                check_type=self.check_type,
            )

        return ConfirmationResult(
            confirmed=False,
            confidence=0.4,
            reason=f"No relationship found between {var_a_name} and {var_b_name} (heuristic)",
            check_type=self.check_type,
        )

    def _single_var_check(
        self,
        var_name: str,
        contract_source: str,
        finding: "Finding",
        slither_instance: Any,
    ) -> ConfirmationResult:
        """
        Single-variable check for arithmetic and other vulnerabilities.

        When LLM only extracts one variable, we check:
        - For arithmetic: is the variable used in unprotected arithmetic?
        - For bad_randomness: is the variable derived from predictable sources?
        - For denial_of_service: is the variable used in unbounded operations?
        """
        import re

        func_name = finding.evidence.function if finding.evidence else ""
        scenario_id = getattr(finding, 'scenario_matched', '') or finding.category or ''

        # Extract function body if available
        func_pattern = rf'function\s+{re.escape(func_name)}\s*\([^)]*\)[^{{]*\{{([\s\S]*?)\}}'
        match = re.search(func_pattern, contract_source) if func_name else None
        search_scope = match.group(1) if match else contract_source

        if scenario_id.lower() == "arithmetic":
            # Check for unprotected arithmetic on the variable
            unsafe_patterns = [
                rf'{re.escape(var_name)}\s*\+\s*=',  # +=
                rf'{re.escape(var_name)}\s*\-\s*=',  # -=
                rf'{re.escape(var_name)}\s*\*\s*=',  # *=
                rf'{re.escape(var_name)}\s*=\s*[^;]*\+',  # = ... +
                rf'{re.escape(var_name)}\s*=\s*[^;]*\-',  # = ... -
                rf'{re.escape(var_name)}\s*=\s*[^;]*\*',  # = ... *
            ]
            safe_patterns = [
                r'SafeMath',
                r'using\s+.*Math',
                rf'add\s*\([^)]*{re.escape(var_name)}',
                rf'sub\s*\([^)]*{re.escape(var_name)}',
                rf'mul\s*\([^)]*{re.escape(var_name)}',
            ]

            has_unsafe = any(re.search(p, search_scope) for p in unsafe_patterns)
            has_safe = any(re.search(p, contract_source, re.IGNORECASE) for p in safe_patterns)

            if has_unsafe and not has_safe:
                return ConfirmationResult(
                    confirmed=True,
                    confidence=0.75,
                    reason=f"Variable '{var_name}' used in unprotected arithmetic",
                    evidence={"variable": var_name, "method": "single_var_check"},
                    check_type=self.check_type,
                )
            elif has_safe:
                return ConfirmationResult(
                    confirmed=False,
                    confidence=0.7,
                    reason=f"SafeMath or safe arithmetic detected for '{var_name}'",
                    evidence={"variable": var_name, "method": "single_var_check"},
                    check_type=self.check_type,
                )

        elif scenario_id.lower() == "bad_randomness":
            # Check if variable is derived from predictable sources
            predictable_sources = [
                rf'{re.escape(var_name)}\s*=\s*[^;]*block\.timestamp',
                rf'{re.escape(var_name)}\s*=\s*[^;]*block\.number',
                rf'{re.escape(var_name)}\s*=\s*[^;]*blockhash',
                rf'{re.escape(var_name)}\s*=\s*[^;]*block\.difficulty',
                rf'{re.escape(var_name)}\s*=\s*[^;]*now',
            ]
            if any(re.search(p, search_scope) for p in predictable_sources):
                return ConfirmationResult(
                    confirmed=True,
                    confidence=0.8,
                    reason=f"Variable '{var_name}' derived from predictable blockchain value",
                    evidence={"variable": var_name, "method": "single_var_check"},
                    check_type=self.check_type,
                )

        elif scenario_id.lower() == "denial_of_service":
            # Check for unbounded loops using the variable
            unbounded_patterns = [
                rf'for\s*\([^)]*{re.escape(var_name)}',
                rf'while\s*\([^)]*{re.escape(var_name)}',
                rf'\.length',  # Array iteration
            ]
            if any(re.search(p, search_scope) for p in unbounded_patterns):
                return ConfirmationResult(
                    confirmed=True,
                    confidence=0.7,
                    reason=f"Variable '{var_name}' used in potentially unbounded loop",
                    evidence={"variable": var_name, "method": "single_var_check"},
                    check_type=self.check_type,
                )

        # Default: try to find the variable in arithmetic context
        arith_context = rf'{re.escape(var_name)}\s*[\+\-\*\/]|[\+\-\*\/]\s*{re.escape(var_name)}'
        if re.search(arith_context, search_scope):
            return ConfirmationResult(
                confirmed=True,
                confidence=0.6,
                reason=f"Variable '{var_name}' found in arithmetic context (single-var check)",
                evidence={"variable": var_name, "method": "single_var_check"},
                check_type=self.check_type,
            )

        return ConfirmationResult(
            confirmed=False,
            confidence=0.4,
            reason=f"Could not verify vulnerability for single variable '{var_name}'",
            check_type=self.check_type,
        )
