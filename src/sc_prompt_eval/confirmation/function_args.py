"""
Function Argument (FA) confirmation module.

Validates function argument sources and checks for missing sender verification.
Used for vulnerabilities like front_running and price_manipulation_buying.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .base import BaseConfirmer, ConfirmationResult, ExtractionData

if TYPE_CHECKING:
    from ..parsing.findings import Finding


class FunctionArgConfirmer(BaseConfirmer):
    """
    Confirms function argument validation issues.

    Per GPTScan: FA checks whether function arguments are properly validated,
    e.g., whether msg.sender is checked against the beneficiary parameter.
    """

    check_type = "FA"

    def confirm(
        self,
        finding: "Finding",
        extraction: ExtractionData,
        slither_instance: Any,
        contract_source: str,
    ) -> ConfirmationResult:
        """
        Verify function argument validation patterns.

        For front_running: Check if msg.sender is verified against recipient
        For price_manipulation: Check if external call arguments are validated

        Args:
            finding: The vulnerability finding to verify
            extraction: Extracted key variables from LLM
            slither_instance: Slither analysis object
            contract_source: Original Solidity source code

        Returns:
            ConfirmationResult indicating whether the vulnerability is confirmed
        """
        var_a_name = extraction.var_a_name  # Usually the recipient/beneficiary param
        var_b_name = extraction.var_b_name  # Usually msg.sender or validation var

        func_name = finding.evidence.function if finding.evidence else ""
        scenario_id = getattr(finding, 'scenario_matched', '') or ''

        if slither_instance is None:
            return self._fallback_check(
                var_a_name, var_b_name, func_name,
                contract_source, scenario_id, finding
            )

        # Find the function in Slither
        func = self._find_function_in_slither(slither_instance, func_name)

        if func is None:
            return self._fallback_check(
                var_a_name, var_b_name, func_name,
                contract_source, scenario_id, finding
            )

        try:
            # Check for sender verification patterns
            if "front_running" in scenario_id.lower():
                return self._check_front_running(
                    func, var_a_name, var_b_name, contract_source
                )
            elif "price_manipulation" in scenario_id.lower():
                return self._check_price_manipulation(
                    func, var_a_name, var_b_name, contract_source
                )
            else:
                return self._generic_arg_check(
                    func, var_a_name, var_b_name, contract_source
                )

        except Exception as e:
            return self._fallback_check(
                var_a_name, var_b_name, func_name,
                contract_source, scenario_id, finding
            )

    def _check_front_running(
        self,
        func: Any,
        recipient_param: str,
        sender_check: str,
        contract_source: str,
    ) -> ConfirmationResult:
        """
        Check for missing msg.sender verification in mint/claim functions.

        Front-running vulnerability exists when:
        - Function has a recipient parameter
        - No check that msg.sender == recipient
        """
        import re
        from slither.core.cfg.node import NodeType

        # Check if msg.sender is compared to the recipient parameter
        msg_sender_check_found = False

        for node in func.nodes:
            # Check IF nodes and EXPRESSION nodes (for require/assert)
            expr_str = str(node.expression) if node.expression else ""
            is_condition = (
                node.type == NodeType.IF or
                (node.type == NodeType.EXPRESSION and ("require(" in expr_str or "assert(" in expr_str))
            )

            if is_condition:
                # Look for msg.sender == param or param == msg.sender
                if recipient_param:
                    pattern = rf'msg\.sender\s*==\s*{re.escape(recipient_param)}|{re.escape(recipient_param)}\s*==\s*msg\.sender'
                    if re.search(pattern, expr_str):
                        msg_sender_check_found = True
                        break

                # Also check for common patterns
                if "msg.sender" in expr_str:
                    msg_sender_check_found = True
                    break

        if msg_sender_check_found:
            return ConfirmationResult(
                confirmed=False,
                confidence=0.8,
                reason="msg.sender check found - front-running may be mitigated",
                evidence={
                    "recipient_param": recipient_param,
                    "sender_check_found": True,
                },
                check_type=self.check_type,
            )
        else:
            return ConfirmationResult(
                confirmed=True,
                confidence=0.85,
                reason=f"No msg.sender verification for '{recipient_param}' - front-running possible",
                evidence={
                    "recipient_param": recipient_param,
                    "sender_check_found": False,
                    "function": func.name,
                },
                check_type=self.check_type,
            )

    def _check_price_manipulation(
        self,
        func: Any,
        amount_var: str,
        result_var: str,
        contract_source: str,
    ) -> ConfirmationResult:
        """
        Check for price manipulation vulnerabilities in swap functions.

        Vulnerability exists when:
        - Function makes external calls to DEX APIs
        - No slippage protection or price validation
        """
        import re

        # Check for Uniswap/PancakeSwap API calls
        dex_patterns = [
            r'swap\w*\(',
            r'IUniswap',
            r'IPancake',
            r'getAmountsOut',
            r'getAmountOut',
        ]

        func_body = str(func.nodes) if func.nodes else ""

        has_dex_call = any(
            re.search(pattern, func_body, re.IGNORECASE)
            for pattern in dex_patterns
        )

        # Check for slippage protection
        slippage_patterns = [
            r'amountOutMin',
            r'minAmountOut',
            r'require\s*\([^)]*amount[^)]*>[^)]*\)',
        ]

        has_slippage_check = any(
            re.search(pattern, func_body, re.IGNORECASE)
            for pattern in slippage_patterns
        )

        if has_dex_call and not has_slippage_check:
            return ConfirmationResult(
                confirmed=True,
                confidence=0.8,
                reason="DEX call found without slippage protection - price manipulation possible",
                evidence={
                    "has_dex_call": has_dex_call,
                    "has_slippage_check": has_slippage_check,
                    "function": func.name,
                },
                check_type=self.check_type,
            )
        elif has_dex_call and has_slippage_check:
            return ConfirmationResult(
                confirmed=False,
                confidence=0.75,
                reason="DEX call with slippage protection found",
                evidence={
                    "has_dex_call": has_dex_call,
                    "has_slippage_check": has_slippage_check,
                },
                check_type=self.check_type,
            )
        else:
            return ConfirmationResult(
                confirmed=False,
                confidence=0.5,
                reason="No DEX call pattern detected",
                check_type=self.check_type,
            )

    def _generic_arg_check(
        self,
        func: Any,
        var_a: str,
        var_b: str,
        contract_source: str,
    ) -> ConfirmationResult:
        """Generic function argument validation check."""
        import re
        from slither.core.cfg.node import NodeType

        # Check if parameters are validated
        param_names = [p.name for p in func.parameters]

        validation_found = False
        for node in func.nodes:
            expr_str = str(node.expression) if node.expression else ""
            is_condition = (
                node.type == NodeType.IF or
                (node.type == NodeType.EXPRESSION and ("require(" in expr_str or "assert(" in expr_str))
            )

            if is_condition:
                for param in param_names:
                    if param in expr_str:
                        validation_found = True
                        break
                if validation_found:
                    break

        if not validation_found and len(param_names) > 0:
            return ConfirmationResult(
                confirmed=True,
                confidence=0.7,
                reason=f"Function parameters not validated: {param_names}",
                evidence={
                    "parameters": param_names,
                    "validation_found": False,
                },
                check_type=self.check_type,
            )
        else:
            return ConfirmationResult(
                confirmed=False,
                confidence=0.6,
                reason="Parameter validation found or no external parameters",
                check_type=self.check_type,
            )

    def _fallback_check(
        self,
        var_a_name: str,
        var_b_name: str,
        func_name: str,
        contract_source: str,
        scenario_id: str,
        finding: "Finding",
    ) -> ConfirmationResult:
        """
        Fallback regex-based heuristic when Slither unavailable.
        """
        import re

        # Extract function body
        func_pattern = rf'function\s+{re.escape(func_name)}\s*\([^)]*\)[^{{]*\{{([\s\S]*?)\}}'
        match = re.search(func_pattern, contract_source)
        func_body = match.group(1) if match else contract_source

        if "front_running" in scenario_id.lower():
            # Check for msg.sender verification
            sender_check = re.search(
                r'require\s*\([^)]*msg\.sender[^)]*\)|msg\.sender\s*==',
                func_body
            )

            if sender_check:
                return ConfirmationResult(
                    confirmed=False,
                    confidence=0.6,
                    reason="msg.sender check found (heuristic)",
                    evidence={"method": "regex_fallback"},
                    check_type=self.check_type,
                )
            else:
                return ConfirmationResult(
                    confirmed=True,
                    confidence=0.65,
                    reason="No msg.sender check found - front-running possible (heuristic)",
                    evidence={"method": "regex_fallback"},
                    check_type=self.check_type,
                )

        elif "price" in scenario_id.lower():
            # Check for DEX calls without slippage
            has_swap = re.search(r'swap|getAmount', func_body, re.IGNORECASE)
            has_min_check = re.search(r'amountOutMin|minAmount|require', func_body, re.IGNORECASE)

            if has_swap and not has_min_check:
                return ConfirmationResult(
                    confirmed=True,
                    confidence=0.6,
                    reason="Swap without slippage check (heuristic)",
                    evidence={"method": "regex_fallback"},
                    check_type=self.check_type,
                )
            else:
                return ConfirmationResult(
                    confirmed=False,
                    confidence=0.5,
                    reason="Slippage check found or no swap detected (heuristic)",
                    evidence={"method": "regex_fallback"},
                    check_type=self.check_type,
                )

        return ConfirmationResult(
            confirmed=False,
            confidence=0.4,
            reason="Generic fallback - could not determine vulnerability (heuristic)",
            evidence={"method": "regex_fallback"},
            check_type=self.check_type,
        )
