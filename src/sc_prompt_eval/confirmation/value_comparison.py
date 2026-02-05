"""
Value Comparison (VC) confirmation module.

Checks if variables are compared in require/if statements,
verifying that proper validation exists (or is missing).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .base import BaseConfirmer, ConfirmationResult, ExtractionData

if TYPE_CHECKING:
    from ..parsing.findings import Finding


class ValueComparisonConfirmer(BaseConfirmer):
    """
    Confirms presence or absence of value comparisons in conditions.

    Per GPTScan: VC checks whether key variables are properly validated
    in require() or if() statements. For vulnerabilities like "approval_not_cleared",
    we check if the clearing logic is missing.
    """

    check_type = "VC"

    def confirm(
        self,
        finding: "Finding",
        extraction: ExtractionData,
        slither_instance: Any,
        contract_source: str,
    ) -> ConfirmationResult:
        """
        Verify value comparison patterns for extracted variables.

        For most vulnerabilities, we're confirming that a check is MISSING,
        which makes the code vulnerable.

        Args:
            finding: The vulnerability finding to verify
            extraction: Extracted key variables from LLM
            slither_instance: Slither analysis object
            contract_source: Original Solidity source code

        Returns:
            ConfirmationResult indicating whether the vulnerability is confirmed
        """
        var_a_name = extraction.var_a_name
        func_name = finding.evidence.function if finding.evidence else ""
        scenario_id = getattr(finding, 'scenario_matched', '') or finding.category or ''

        if not var_a_name:
            # For "missing check" scenarios, do function-level analysis
            # since the vulnerability is about the function lacking guards, not a specific variable
            if self._is_missing_check_scenario(scenario_id):
                return self._function_level_check(func_name, contract_source, finding, scenario_id)
            return ConfirmationResult(
                confirmed=False,
                confidence=0.3,
                reason="Missing variable name for value comparison check",
                check_type=self.check_type,
            )

        if slither_instance is None:
            return self._fallback_check(
                var_a_name, func_name, contract_source, finding
            )

        # Find the function in Slither
        func = self._find_function_in_slither(slither_instance, func_name)

        if func is None:
            return self._fallback_check(
                var_a_name, func_name, contract_source, finding
            )

        # Check if variable appears in conditions using Slither's CFG
        try:
            from slither.core.cfg.node import NodeType

            var_in_condition = False
            condition_type = None

            for node in func.nodes:
                # Check IF nodes
                if node.type == NodeType.IF:
                    if node.expression and var_a_name in str(node.expression):
                        var_in_condition = True
                        condition_type = "IF"
                        break
                # Check EXPRESSION nodes for require() and assert() calls
                elif node.type == NodeType.EXPRESSION:
                    expr_str = str(node.expression) if node.expression else ""
                    if ("require(" in expr_str or "assert(" in expr_str) and var_a_name in expr_str:
                        var_in_condition = True
                        condition_type = "require/assert"
                        break

            # For VC vulnerabilities, the issue is usually MISSING validation
            # So if we find validation, the vulnerability might be a false positive
            # Note: scenario_id already set at function start with category fallback

            if self._is_missing_check_scenario(scenario_id):
                # These scenarios are vulnerable when the check is ABSENT
                if var_in_condition:
                    return ConfirmationResult(
                        confirmed=False,
                        confidence=0.8,
                        reason=f"Variable '{var_a_name}' IS checked in {condition_type} - may be false positive",
                        evidence={
                            "variable": var_a_name,
                            "found_in": condition_type,
                            "function": func_name,
                        },
                        check_type=self.check_type,
                    )
                else:
                    return ConfirmationResult(
                        confirmed=True,
                        confidence=0.85,
                        reason=f"Variable '{var_a_name}' is NOT checked in any condition - vulnerability confirmed",
                        evidence={
                            "variable": var_a_name,
                            "function": func_name,
                            "missing_check": True,
                        },
                        check_type=self.check_type,
                    )
            else:
                # For other scenarios, presence of check might be part of the vulnerability
                if var_in_condition:
                    return ConfirmationResult(
                        confirmed=True,
                        confidence=0.75,
                        reason=f"Variable '{var_a_name}' found in {condition_type}",
                        evidence={
                            "variable": var_a_name,
                            "found_in": condition_type,
                            "function": func_name,
                        },
                        check_type=self.check_type,
                    )
                else:
                    return ConfirmationResult(
                        confirmed=False,
                        confidence=0.6,
                        reason=f"Variable '{var_a_name}' not found in any condition",
                        check_type=self.check_type,
                    )

        except ImportError:
            return self._fallback_check(
                var_a_name, func_name, contract_source, finding
            )

    def _is_missing_check_scenario(self, scenario_id: str) -> bool:
        """Determine if this scenario is vulnerable due to missing checks."""
        missing_check_scenarios = [
            "approval_not_cleared",
            "slippage",
            "unauthorized_transfer",
            "risky_first_deposit",
            "access_control",  # Vuln is ABSENCE of authorization checks
            "unchecked_low_level_calls",  # Vuln is ABSENCE of return value checks
            "short_addresses",  # Vuln is ABSENCE of input length validation
        ]
        return scenario_id.lower() in [s.lower() for s in missing_check_scenarios]

    def _fallback_check(
        self,
        var_name: str,
        func_name: str,
        contract_source: str,
        finding: "Finding",
    ) -> ConfirmationResult:
        """
        Fallback regex-based heuristic when Slither unavailable.
        """
        import re

        # Extract function body if possible
        func_pattern = rf'function\s+{re.escape(func_name)}\s*\([^)]*\)[^{{]*\{{([\s\S]*?)\}}'
        match = re.search(func_pattern, contract_source)
        func_body = match.group(1) if match else contract_source

        # Check for require/if/assert with the variable
        patterns = [
            rf'require\s*\([^)]*{re.escape(var_name)}[^)]*\)',
            rf'if\s*\([^)]*{re.escape(var_name)}[^)]*\)',
            rf'assert\s*\([^)]*{re.escape(var_name)}[^)]*\)',
        ]

        # Get scenario_id with category fallback
        scenario_id = getattr(finding, 'scenario_matched', '') or getattr(finding, 'category', '') or ''

        for pattern in patterns:
            if re.search(pattern, func_body, re.IGNORECASE):
                if self._is_missing_check_scenario(scenario_id):
                    return ConfirmationResult(
                        confirmed=False,
                        confidence=0.6,
                        reason=f"Variable '{var_name}' appears in condition (heuristic)",
                        evidence={"method": "regex_fallback"},
                        check_type=self.check_type,
                    )
                else:
                    return ConfirmationResult(
                        confirmed=True,
                        confidence=0.6,
                        reason=f"Variable '{var_name}' appears in condition (heuristic)",
                        evidence={"method": "regex_fallback"},
                        check_type=self.check_type,
                    )

        # No condition found
        if self._is_missing_check_scenario(scenario_id):
            return ConfirmationResult(
                confirmed=True,
                confidence=0.65,
                reason=f"No condition found for '{var_name}' - missing check (heuristic)",
                evidence={"method": "regex_fallback"},
                check_type=self.check_type,
            )
        else:
            return ConfirmationResult(
                confirmed=False,
                confidence=0.5,
                reason=f"No condition found for '{var_name}' (heuristic)",
                evidence={"method": "regex_fallback"},
                check_type=self.check_type,
            )

    def _function_level_check(
        self,
        func_name: str,
        contract_source: str,
        finding: "Finding",
        scenario_id: str,
    ) -> ConfirmationResult:
        """
        Function-level check when no specific variable is extracted.

        For "missing check" scenarios (access_control, unchecked_low_level_calls),
        we check if the function has ANY authorization guards or return value checks.
        """
        import re

        if not func_name:
            return ConfirmationResult(
                confirmed=True,
                confidence=0.5,
                reason="No function name provided, assuming missing checks (function-level)",
                evidence={"method": "function_level_fallback"},
                check_type=self.check_type,
            )

        # Extract function body
        func_pattern = rf'function\s+{re.escape(func_name)}\s*\([^)]*\)[^{{]*\{{([\s\S]*?)\}}'
        match = re.search(func_pattern, contract_source)
        func_body = match.group(1) if match else ""

        # Check for different types of missing checks based on scenario
        if scenario_id.lower() == "access_control":
            # Look for authorization patterns
            auth_patterns = [
                r'require\s*\(\s*msg\.sender\s*==',
                r'require\s*\(\s*owner\s*==',
                r'require\s*\(\s*_owner\s*==',
                r'onlyOwner',
                r'onlyAdmin',
                r'onlyRole',
                r'hasRole\s*\(',
                r'_checkRole\s*\(',
            ]
            for pattern in auth_patterns:
                if re.search(pattern, func_body, re.IGNORECASE):
                    return ConfirmationResult(
                        confirmed=False,
                        confidence=0.7,
                        reason=f"Function '{func_name}' has authorization check - may be false positive",
                        evidence={"method": "function_level_check", "pattern_found": pattern},
                        check_type=self.check_type,
                    )
            # No auth found = vulnerability confirmed
            return ConfirmationResult(
                confirmed=True,
                confidence=0.8,
                reason=f"Function '{func_name}' lacks authorization checks - access control vulnerability confirmed",
                evidence={"method": "function_level_check", "missing": "authorization"},
                check_type=self.check_type,
            )

        elif scenario_id.lower() == "unchecked_low_level_calls":
            # Look for return value handling
            if re.search(r'\(\s*bool\s+\w+\s*,', func_body):
                # Captures return value
                if re.search(r'require\s*\(\s*\w+\s*\)', func_body):
                    return ConfirmationResult(
                        confirmed=False,
                        confidence=0.7,
                        reason=f"Function '{func_name}' checks low-level call return - may be false positive",
                        evidence={"method": "function_level_check"},
                        check_type=self.check_type,
                    )
            return ConfirmationResult(
                confirmed=True,
                confidence=0.75,
                reason=f"Function '{func_name}' does not check low-level call return value",
                evidence={"method": "function_level_check", "missing": "return_check"},
                check_type=self.check_type,
            )

        elif scenario_id.lower() == "short_addresses":
            # Look for msg.data.length check
            if re.search(r'msg\.data\.length', func_body):
                return ConfirmationResult(
                    confirmed=False,
                    confidence=0.7,
                    reason=f"Function '{func_name}' checks msg.data.length - may be false positive",
                    evidence={"method": "function_level_check"},
                    check_type=self.check_type,
                )
            return ConfirmationResult(
                confirmed=True,
                confidence=0.7,
                reason=f"Function '{func_name}' does not validate input data length",
                evidence={"method": "function_level_check", "missing": "length_check"},
                check_type=self.check_type,
            )

        # Default: confirm for other missing check scenarios
        return ConfirmationResult(
            confirmed=True,
            confidence=0.6,
            reason=f"Function '{func_name}' lacks expected validation (function-level check)",
            evidence={"method": "function_level_check"},
            check_type=self.check_type,
        )
