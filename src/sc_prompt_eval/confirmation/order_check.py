"""
Order Check (OC) confirmation module.

Verifies statement execution order using Slither's CFG analysis.
Used for vulnerabilities like wrong_interest_rate_order and wrong_checkpoint_order.
"""

from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING

from .base import BaseConfirmer, ConfirmationResult, ExtractionData

if TYPE_CHECKING:
    from ..parsing.findings import Finding


class OrderCheckConfirmer(BaseConfirmer):
    """
    Confirms statement execution order using CFG analysis.

    Per GPTScan: OC checks whether operations happen in the correct order,
    e.g., whether interest is accrued before or after balance update.
    """

    check_type = "OC"

    def confirm(
        self,
        finding: "Finding",
        extraction: ExtractionData,
        slither_instance: Any,
        contract_source: str,
    ) -> ConfirmationResult:
        """
        Verify execution order of key statements.

        Args:
            finding: The vulnerability finding to verify
            extraction: Extracted key statements from LLM
            slither_instance: Slither analysis object
            contract_source: Original Solidity source code

        Returns:
            ConfirmationResult indicating whether wrong order is confirmed
        """
        key_statements = extraction.key_statements

        if not key_statements or len(key_statements) < 2:
            return ConfirmationResult(
                confirmed=False,
                confidence=0.3,
                reason="Need at least 2 key statements for order check",
                check_type=self.check_type,
            )

        # Get first two statements (expected order: stmt1 should come before stmt2)
        stmt1 = key_statements[0]
        stmt2 = key_statements[1] if len(key_statements) > 1 else None

        if not stmt2:
            return ConfirmationResult(
                confirmed=False,
                confidence=0.3,
                reason="Second statement missing for order check",
                check_type=self.check_type,
            )

        func_name = finding.evidence.function if finding.evidence else ""

        if slither_instance is None:
            return self._fallback_check(
                stmt1, stmt2, func_name, contract_source, finding
            )

        # Find the function in Slither
        func = self._find_function_in_slither(slither_instance, func_name)

        if func is None:
            return self._fallback_check(
                stmt1, stmt2, func_name, contract_source, finding
            )

        # Use CFG to check ordering
        try:
            nodes = list(func.nodes)
            idx1 = self._find_statement_index(nodes, stmt1)
            idx2 = self._find_statement_index(nodes, stmt2)

            if idx1 is None or idx2 is None:
                return self._fallback_check(
                    stmt1, stmt2, func_name, contract_source, finding
                )

            # For wrong order vulnerabilities, stmt1 should come AFTER stmt2
            # but it currently comes BEFORE (which is the bug)
            if idx1 < idx2:
                # Statement 1 comes before statement 2 (current order)
                # This might be the WRONG order for these vulnerability types
                return ConfirmationResult(
                    confirmed=True,
                    confidence=0.85,
                    reason=f"Wrong order confirmed: '{stmt1[:50]}...' executes before '{stmt2[:50]}...'",
                    evidence={
                        "statement1_index": idx1,
                        "statement2_index": idx2,
                        "function": func_name,
                        "order": "stmt1_before_stmt2",
                    },
                    check_type=self.check_type,
                )
            else:
                # Statement 2 comes before statement 1 (correct order)
                return ConfirmationResult(
                    confirmed=False,
                    confidence=0.8,
                    reason=f"Correct order: '{stmt2[:50]}...' executes before '{stmt1[:50]}...'",
                    evidence={
                        "statement1_index": idx1,
                        "statement2_index": idx2,
                        "function": func_name,
                        "order": "stmt2_before_stmt1",
                    },
                    check_type=self.check_type,
                )

        except Exception as e:
            return self._fallback_check(
                stmt1, stmt2, func_name, contract_source, finding
            )

    def _find_statement_index(
        self, nodes: List[Any], statement: str
    ) -> Optional[int]:
        """
        Find the index of a statement in the CFG nodes.

        Args:
            nodes: List of Slither CFG nodes
            statement: Statement string to find

        Returns:
            Index of the node containing the statement, or None
        """
        # Normalize statement for comparison
        stmt_normalized = self._normalize_statement(statement)

        for idx, node in enumerate(nodes):
            if node.expression:
                node_str = self._normalize_statement(str(node.expression))
                if stmt_normalized in node_str or node_str in stmt_normalized:
                    return idx

        return None

    def _normalize_statement(self, stmt: str) -> str:
        """Normalize a statement for comparison."""
        import re
        # Remove whitespace and common variations
        return re.sub(r'\s+', '', stmt.lower())

    def _fallback_check(
        self,
        stmt1: str,
        stmt2: str,
        func_name: str,
        contract_source: str,
        finding: "Finding",
    ) -> ConfirmationResult:
        """
        Fallback line-number based heuristic when Slither unavailable.

        Searches for statements in the source and compares line positions.
        """
        import re

        # Extract function body
        func_pattern = rf'function\s+{re.escape(func_name)}\s*\([^)]*\)[^{{]*\{{([\s\S]*?)\}}'
        match = re.search(func_pattern, contract_source)
        func_body = match.group(1) if match else contract_source

        # Find positions of key patterns from statements
        pos1 = self._find_pattern_position(stmt1, func_body)
        pos2 = self._find_pattern_position(stmt2, func_body)

        if pos1 is None or pos2 is None:
            return ConfirmationResult(
                confirmed=False,
                confidence=0.4,
                reason=f"Could not locate statements in function body (heuristic)",
                evidence={"method": "regex_fallback"},
                check_type=self.check_type,
            )

        # Compare positions
        if pos1 < pos2:
            return ConfirmationResult(
                confirmed=True,
                confidence=0.65,
                reason=f"Wrong order detected: '{stmt1[:30]}...' appears before '{stmt2[:30]}...' (heuristic)",
                evidence={
                    "method": "regex_fallback",
                    "pos1": pos1,
                    "pos2": pos2,
                },
                check_type=self.check_type,
            )
        else:
            return ConfirmationResult(
                confirmed=False,
                confidence=0.65,
                reason=f"Correct order: '{stmt2[:30]}...' appears before '{stmt1[:30]}...' (heuristic)",
                evidence={
                    "method": "regex_fallback",
                    "pos1": pos1,
                    "pos2": pos2,
                },
                check_type=self.check_type,
            )

    def _find_pattern_position(self, stmt: str, text: str) -> Optional[int]:
        """Find position of a statement pattern in text."""
        import re

        # Extract key identifiers from statement
        # Try to find function calls, variable assignments, etc.
        identifiers = re.findall(r'\b(\w+)\s*\(', stmt)  # function calls
        if not identifiers:
            identifiers = re.findall(r'\b(\w+)\s*=', stmt)  # assignments
        if not identifiers:
            # Use any word that's not a keyword
            identifiers = re.findall(r'\b([a-z]\w{3,})\b', stmt, re.IGNORECASE)

        for ident in identifiers:
            match = re.search(rf'\b{re.escape(ident)}\b', text)
            if match:
                return match.start()

        return None
