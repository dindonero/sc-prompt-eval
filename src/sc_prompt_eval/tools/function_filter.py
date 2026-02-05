"""
GPTScan-style multi-dimensional function filtering.

Implements pre-analysis filtering from Sun2023 Table 2:
- FNK: Function Name contains Keyword
- FNI: Function Name Invokes (calls another function)
- FCE: Function Content contains Expression
- FCNE: Function Content does NOT contain Expression
- FCCE: Function Content contains Combination of Expressions
- FCNCE: Function Content does NOT contain Combination
- FPT: Function Parameters match Types (address, uint256, etc.)
- FPNC: Function is Public, Not analyzed with Caller
- FNM: Function has No access control Modifiers (detects ANY modifier)
- FA: Function Argument validation (confirmation-stage only in GPTScan)
- VC: Value Comparison - checks if variables are compared in require/if
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Any
from pathlib import Path


@dataclass
class FilterRule:
    """A single filtering rule."""
    rule_type: str  # FNK, FCE, FCNE, FCCE, etc.
    patterns: List[str]
    require_all: bool = False  # For combinations (FCCE)
    negate: bool = False  # For FCNE, FCNCE


@dataclass
class FunctionInfo:
    """Extracted function information."""
    name: str
    body: str
    full_text: str
    start_line: int
    end_line: int
    visibility: str = "public"  # public, external, internal, private
    modifiers: List[str] = field(default_factory=list)
    parameters: List[Dict[str, str]] = field(default_factory=list)


class FunctionFilter:
    """
    Multi-dimensional function filtering per GPTScan paper.

    Filters candidate functions BEFORE sending to LLM analysis to reduce
    the search space and improve precision.
    """

    # Common access control modifiers to detect
    ACCESS_MODIFIERS = [
        "onlyOwner", "onlyAdmin", "onlyRole", "onlyMinter",
        "onlyGovernance", "onlyAuthorized", "whenNotPaused",
        "nonReentrant", "initializer"
    ]

    def __init__(self, scenarios: Optional[List[Dict]] = None):
        """
        Initialize with optional scenario definitions.

        Args:
            scenarios: List of scenario dicts from scenarios.json
        """
        self.scenarios = scenarios or []
        self._compile_rules()

    def _compile_rules(self) -> None:
        """Build regex patterns from scenario definitions.

        GPTScan Table 2 multi-dimensional filtering: creates rules for ALL
        filtering_types specified in the scenario. Rules are later combined
        with AND logic (all must pass) per GPTScan methodology.
        """
        self.rules_by_scenario: Dict[str, List[FilterRule]] = {}

        for scenario in self.scenarios:
            rules = []
            scenario_id = scenario.get("id", "")

            # Get filtering types - these determine which rules to create
            filtering_types = scenario.get("filtering_type", [])
            if isinstance(filtering_types, str):
                filtering_types = [filtering_types]

            keywords = scenario.get("filtering_keywords", [])
            expressions = scenario.get("filtering_expressions", [])
            # Separate negative expressions for FCNE (GPTScan: different patterns for NOT checks)
            expressions_negative = scenario.get("filtering_expressions_negative", [])

            # FNK: Function Name contains Keyword
            if "FNK" in filtering_types and keywords:
                rules.append(FilterRule(
                    rule_type="FNK",
                    patterns=keywords
                ))

            # FNI: Function Name Invokes (calls specific function)
            if "FNI" in filtering_types and keywords:
                rules.append(FilterRule(
                    rule_type="FNI",
                    patterns=keywords
                ))

            # FCE: Function Content contains Expression
            if "FCE" in filtering_types and expressions:
                rules.append(FilterRule(
                    rule_type="FCE",
                    patterns=expressions
                ))

            # FCNE: Function Content does NOT contain Expression
            # Use filtering_expressions_negative if available, otherwise fall back to expressions
            fcne_patterns = expressions_negative if expressions_negative else expressions
            if "FCNE" in filtering_types and fcne_patterns:
                rules.append(FilterRule(
                    rule_type="FCNE",
                    patterns=fcne_patterns,
                    negate=True
                ))

            # FCCE: Function Content contains Combination of Expressions (ALL)
            if "FCCE" in filtering_types and expressions:
                rules.append(FilterRule(
                    rule_type="FCCE",
                    patterns=expressions,
                    require_all=True
                ))

            # FCNCE: Function Content does NOT contain Combination
            if "FCNCE" in filtering_types and expressions:
                rules.append(FilterRule(
                    rule_type="FCNCE",
                    patterns=expressions,
                    require_all=True,
                    negate=True
                ))

            # FNM: Function has No access control Modifiers
            if "FNM" in filtering_types:
                rules.append(FilterRule(
                    rule_type="FNM",
                    patterns=[]  # Checked programmatically
                ))

            # FPNC: Function is Public, Not analyzed with Caller
            if "FPNC" in filtering_types:
                rules.append(FilterRule(
                    rule_type="FPNC",
                    patterns=[]  # Checked programmatically
                ))

            # FPT: Function Parameters match Types
            if "FPT" in filtering_types:
                # Use filtering_expressions as type patterns if available,
                # otherwise use common vulnerable parameter types
                type_patterns = scenario.get("parameter_types", [
                    "address", "uint256", "bytes", "string"
                ])
                rules.append(FilterRule(
                    rule_type="FPT",
                    patterns=type_patterns
                ))

            # VC: Value Comparison - check if variables are compared in require/if
            # Per GPTScan: pre-filter for functions that have validation logic
            if "VC" in filtering_types:
                rules.append(FilterRule(
                    rule_type="VC",
                    patterns=expressions if expressions else []
                ))

            # FA: Function Argument validation
            # NOTE: Per GPTScan methodology, FA is primarily used in the
            # POST-LLM static confirmation stage (Section 4.4), not pre-filtering.
            # We include it here for scenario completeness but it passes through
            # to confirmation stage for actual validation.
            if "FA" in filtering_types:
                rules.append(FilterRule(
                    rule_type="FA",
                    patterns=[]  # Validated in confirmation stage with Slither
                ))

            self.rules_by_scenario[scenario_id] = rules

    def extract_functions(self, source: str) -> List[FunctionInfo]:
        """
        Parse Solidity source to extract function definitions.

        Uses ANTLR-based parsing per GPTScan methodology for accurate
        extraction of function metadata (modifiers, visibility, parameters).
        Falls back to regex-based extraction if ANTLR fails.

        Args:
            source: Solidity source code

        Returns:
            List of FunctionInfo objects
        """
        # Try ANTLR-based parsing first (GPTScan-aligned)
        try:
            return self._extract_functions_antlr(source)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"ANTLR parsing failed, falling back to regex: {e}"
            )
            return self._extract_functions_regex(source)

    def _extract_functions_antlr(self, source: str) -> List[FunctionInfo]:
        """
        Extract functions using ANTLR parser.

        Per GPTScan methodology (Section 3.2.1), uses ANTLR for accurate
        function boundary detection, modifier extraction, and parameter parsing.
        """
        from .solidity_parser import extract_functions as antlr_extract
        from .solidity_parser import strip_comments_and_strings

        antlr_funcs = antlr_extract(source)
        functions = []

        for af in antlr_funcs:
            # Convert SolidityFunction to FunctionInfo
            functions.append(FunctionInfo(
                name=af.name,
                body=af.body_text,
                full_text=af.full_text,
                start_line=af.start_line,
                end_line=af.end_line,
                visibility=af.visibility,
                modifiers=af.modifiers,
                parameters=af.parameters,
            ))

        return functions

    def _extract_functions_regex(self, source: str) -> List[FunctionInfo]:
        """
        Fallback regex-based function extraction.

        Used when ANTLR parsing fails (e.g., syntax errors in contract).
        """
        functions = []

        # Pattern to match function definitions with visibility and modifiers
        func_pattern = re.compile(
            r'function\s+(\w+)\s*'  # function name
            r'\(([^)]*)\)\s*'  # parameters
            r'((?:public|external|internal|private)?\s*'  # visibility
            r'(?:view|pure|payable)?\s*'  # state mutability
            r'(?:\w+\s*)*?)'  # modifiers
            r'(?:returns\s*\([^)]*\))?\s*'  # return type
            r'\{',  # opening brace
            re.DOTALL
        )

        for match in func_pattern.finditer(source):
            func_name = match.group(1)
            params_str = match.group(2)
            modifiers_str = match.group(3)

            # Find the matching closing brace
            body_start = match.end()
            brace_count = 1
            pos = body_start

            while pos < len(source) and brace_count > 0:
                if source[pos] == '{':
                    brace_count += 1
                elif source[pos] == '}':
                    brace_count -= 1
                pos += 1

            if brace_count == 0:
                body = source[body_start:pos-1]
                full_text = source[match.start():pos]

                # Calculate line numbers
                start_line = source[:match.start()].count('\n') + 1
                end_line = source[:pos].count('\n') + 1

                # Extract visibility
                visibility = "public"  # default
                for vis in ["public", "external", "internal", "private"]:
                    if vis in modifiers_str:
                        visibility = vis
                        break

                # Extract modifiers
                modifiers = []
                for mod in self.ACCESS_MODIFIERS:
                    if mod in modifiers_str:
                        modifiers.append(mod)

                # Extract parameters
                parameters = self._parse_parameters(params_str)

                functions.append(FunctionInfo(
                    name=func_name,
                    body=body,
                    full_text=full_text,
                    start_line=start_line,
                    end_line=end_line,
                    visibility=visibility,
                    modifiers=modifiers,
                    parameters=parameters
                ))

        return functions

    def _parse_parameters(self, params_str: str) -> List[Dict[str, str]]:
        """Parse function parameter string into list of {type, name} dicts."""
        params = []
        if not params_str.strip():
            return params

        for param in params_str.split(','):
            param = param.strip()
            if not param:
                continue

            # Handle: type [memory/storage/calldata] name
            parts = param.split()
            if len(parts) >= 2:
                # Last part is name, first is type
                param_type = parts[0]
                param_name = parts[-1]
                params.append({"type": param_type, "name": param_name})
            elif len(parts) == 1:
                # Just type, no name
                params.append({"type": parts[0], "name": ""})

        return params

    def filter_by_scenario(self, source: str, scenario_id: str) -> Set[str]:
        """
        Return function names matching a specific scenario's filter rules.

        Args:
            source: Solidity source code
            scenario_id: ID of scenario to filter for

        Returns:
            Set of function names that pass the filters
        """
        functions = self.extract_functions(source)
        rules = self.rules_by_scenario.get(scenario_id, [])

        if not rules:
            # No rules defined - return all functions
            return {f.name for f in functions}

        candidates = set()
        for func in functions:
            if self._matches_rules(func, rules):
                candidates.add(func.name)

        return candidates

    def filter_all_scenarios(self, source: str) -> Dict[str, Set[str]]:
        """
        Filter functions for all scenarios.

        Args:
            source: Solidity source code

        Returns:
            Dict mapping scenario_id to set of matching function names
        """
        functions = self.extract_functions(source)
        results = {}

        for scenario_id, rules in self.rules_by_scenario.items():
            candidates = set()
            for func in functions:
                if self._matches_rules(func, rules):
                    candidates.add(func.name)
            if candidates:
                results[scenario_id] = candidates

        return results

    def get_candidate_functions(self, source: str) -> Set[str]:
        """
        Get union of all functions matching any scenario.

        This is the main entry point for pre-filtering before analysis.

        Args:
            source: Solidity source code

        Returns:
            Set of function names that are candidates for at least one scenario
        """
        all_candidates = set()
        by_scenario = self.filter_all_scenarios(source)

        for candidates in by_scenario.values():
            all_candidates.update(candidates)

        return all_candidates

    def filter_reachable_functions(
        self, source: str, candidates: Set[str]
    ) -> Set[str]:
        """
        Filter candidates to only include functions reachable from external entry points.

        Per GPTScan paper: Uses call graph to determine which functions can be called
        by external users. Entry points are public/external functions.

        Implementation uses regex-based call graph construction. For more accurate
        results, consider using Slither's call_graph API in the confirmation stage.

        Args:
            source: Solidity source code
            candidates: Set of candidate function names from pre-filtering

        Returns:
            Filtered set containing only reachable functions
        """
        functions = self.extract_functions(source)
        func_map = {f.name: f for f in functions}

        # Build call graph: function_name -> set of called functions
        call_graph: Dict[str, Set[str]] = {}
        for func in functions:
            calls = self._extract_function_calls(func.body, func_map.keys())
            call_graph[func.name] = calls

        # Identify entry points (public/external functions)
        entry_points = {
            f.name for f in functions
            if f.visibility in ("public", "external")
        }

        # BFS/DFS to find all reachable functions from entry points
        reachable: Set[str] = set()
        stack = list(entry_points)

        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)

            # Add functions called by current
            for called in call_graph.get(current, set()):
                if called not in reachable:
                    stack.append(called)

        # Return intersection of candidates with reachable functions
        return candidates.intersection(reachable)

    def _extract_function_calls(
        self, body: str, known_functions: Set[str]
    ) -> Set[str]:
        """
        Extract function calls from a function body.

        Args:
            body: Function body text
            known_functions: Set of known function names in the contract

        Returns:
            Set of called function names
        """
        calls: Set[str] = set()

        # Pattern to match function calls: functionName(
        # This is a simplified pattern that works for most cases
        call_pattern = re.compile(r'\b(\w+)\s*\(')

        for match in call_pattern.finditer(body):
            call_name = match.group(1)
            # Only include if it's a known function in this contract
            if call_name in known_functions:
                calls.add(call_name)

        return calls

    def _matches_rules(self, func: FunctionInfo, rules: List[FilterRule]) -> bool:
        """
        Check if function matches all applicable rules.

        Uses AND logic across rule types per GPTScan multi-dimensional filtering:
        ALL rule types must pass for a function to be a candidate.
        Within a rule type, OR logic applies (any pattern match suffices).

        Example: filtering_type: ["FNK", "FCE", "FNM"]
        - Function must match FNK (any keyword) AND
        - Function must match FCE (any expression) AND
        - Function must match FNM (no modifiers)

        Args:
            func: Function information
            rules: List of filter rules

        Returns:
            True if function passes ALL filter conditions
        """
        if not rules:
            return True

        # Group rules by type
        rules_by_type: Dict[str, List[FilterRule]] = {}
        for rule in rules:
            if rule.rule_type not in rules_by_type:
                rules_by_type[rule.rule_type] = []
            rules_by_type[rule.rule_type].append(rule)

        # AND logic: ALL rule types must have at least one matching rule
        for rule_type, type_rules in rules_by_type.items():
            type_matched = False

            for rule in type_rules:
                if self._check_single_rule(func, rule):
                    type_matched = True
                    break

            # If this rule type didn't match, function fails filtering
            if not type_matched:
                return False

        # All rule types passed
        return True

    def _check_single_rule(self, func: FunctionInfo, rule: FilterRule) -> bool:
        """Check if function matches a single rule per GPTScan Table 2."""
        if rule.rule_type == "FNK":
            # Function Name contains Keyword
            return any(
                kw.lower() in func.name.lower()
                for kw in rule.patterns
            )

        elif rule.rule_type == "FCE":
            # Function Content contains Expression
            return any(
                re.search(pattern, func.body, re.IGNORECASE)
                for pattern in rule.patterns
            )

        elif rule.rule_type == "FCNE":
            # Function Content does NOT contain Expression
            return not any(
                re.search(pattern, func.body, re.IGNORECASE)
                for pattern in rule.patterns
            )

        elif rule.rule_type == "FCCE":
            # Function Content contains Combination (all patterns)
            if rule.require_all:
                return all(
                    re.search(pattern, func.body, re.IGNORECASE)
                    for pattern in rule.patterns
                )
            return any(
                re.search(pattern, func.body, re.IGNORECASE)
                for pattern in rule.patterns
            )

        elif rule.rule_type == "FCNCE":
            # Function Content does NOT contain Combination
            if rule.require_all:
                return not all(
                    re.search(pattern, func.body, re.IGNORECASE)
                    for pattern in rule.patterns
                )
            return not any(
                re.search(pattern, func.body, re.IGNORECASE)
                for pattern in rule.patterns
            )

        elif rule.rule_type == "FNM":
            # Function has No access control Modifiers
            # GPTScan: Detect ANY modifier, not just predefined ones
            # Check both parsed modifiers AND raw function signature
            if func.modifiers:
                return False  # Has known modifiers
            # Also check for modifier patterns in full_text (catches custom modifiers)
            # Pattern: word after visibility/mutability that isn't a keyword
            modifier_pattern = re.search(
                r'\)\s*(?:public|external|internal|private)?\s*'
                r'(?:view|pure|payable)?\s*'
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(|{|returns)',
                func.full_text
            )
            if modifier_pattern:
                potential_modifier = modifier_pattern.group(1)
                # Exclude Solidity keywords
                solidity_keywords = {
                    'returns', 'return', 'virtual', 'override', 'memory',
                    'storage', 'calldata', 'public', 'private', 'internal',
                    'external', 'view', 'pure', 'payable', 'constant'
                }
                if potential_modifier.lower() not in solidity_keywords:
                    return False  # Has a custom modifier
            return True  # No modifiers detected

        elif rule.rule_type == "FPNC":
            # Function is Public, Not analyzed with Caller
            return func.visibility in ["public", "external"]

        elif rule.rule_type == "FA":
            # Function Argument validation
            # Per GPTScan: FA is primarily for post-LLM static confirmation
            # In pre-filtering, we check for basic argument validation patterns
            # (msg.sender checks, require statements with arguments)
            if not rule.patterns:
                # No specific patterns - pass through to confirmation stage
                return True
            # Check for argument validation patterns in body
            arg_validation_patterns = [
                r'require\s*\([^)]*msg\.sender',
                r'require\s*\([^)]*_\w+\s*[!=<>]',
                r'if\s*\([^)]*msg\.sender',
            ]
            return any(
                re.search(pattern, func.body)
                for pattern in arg_validation_patterns
            )

        elif rule.rule_type == "FPT":
            # Function Parameters match Types
            # Check if function has parameters of specified types
            if not rule.patterns:
                return True
            func_param_types = {p.get("type", "").lower() for p in func.parameters}
            # Check if any required type is present
            return any(
                any(req_type.lower() in param_type for param_type in func_param_types)
                for req_type in rule.patterns
            )

        elif rule.rule_type == "FNI":
            # Function Name Invokes (calls specific function)
            # Check if function body calls any of the patterns
            return any(
                re.search(rf'\b{re.escape(pattern)}\s*\(', func.body)
                for pattern in rule.patterns
            )

        elif rule.rule_type == "VC":
            # Value Comparison - check if variables are compared in require/if
            # Per GPTScan: pre-filter for functions that have validation logic
            # Check for require/assert/if statements with comparisons
            vc_patterns = [
                r'require\s*\([^)]*[<>=!]+',
                r'assert\s*\([^)]*[<>=!]+',
                r'if\s*\([^)]*[<>=!]+',
            ]
            # If specific patterns provided, check those too
            if rule.patterns:
                for pattern in rule.patterns:
                    if re.search(pattern, func.body, re.IGNORECASE):
                        return True
            # Check for general comparison patterns
            return any(
                re.search(pattern, func.body)
                for pattern in vc_patterns
            )

        return False


def load_scenarios_from_file(path: str = "data/scenarios.json") -> List[Dict]:
    """Load scenarios from JSON file."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return data.get("scenarios", [])
    except Exception:
        return []


