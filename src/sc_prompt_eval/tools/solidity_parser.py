"""
ANTLR-based Solidity parser for GPTScan-style pre-filtering.

Per GPTScan methodology (Sun et al., ICSE 2024), uses ANTLR to parse
Solidity source code and extract function information with syntactic
precision. This enables accurate:
- Modifier detection (FNM filtering)
- Parameter type matching (FPT filtering)
- Visibility detection (FPNC filtering)
- Function body extraction for content matching (FCE/FCNE/FCCE)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from antlr4 import CommonTokenStream, InputStream, ParserRuleContext
from antlr4.error.ErrorListener import ErrorListener

logger = logging.getLogger(__name__)


@dataclass
class SolidityFunction:
    """Parsed function information from ANTLR AST."""

    name: str
    """Function identifier (or 'fallback'/'receive' for special functions)."""

    visibility: str = "internal"
    """Access level: public, external, internal, private."""

    state_mutability: Optional[str] = None
    """State mutability: view, pure, payable, or None (nonpayable)."""

    modifiers: List[str] = field(default_factory=list)
    """List of modifier names applied to this function."""

    parameters: List[Dict[str, str]] = field(default_factory=list)
    """Parameter list: [{'type': 'uint256', 'name': 'amount', 'location': 'memory'}, ...]"""

    return_types: List[str] = field(default_factory=list)
    """Return type list: ['uint256', 'bool', ...]"""

    body_text: str = ""
    """Function body text (without signature)."""

    full_text: str = ""
    """Complete function definition including signature."""

    start_line: int = 0
    """Starting line number (1-indexed)."""

    end_line: int = 0
    """Ending line number (1-indexed)."""

    is_virtual: bool = False
    """Whether function is marked virtual."""

    is_override: bool = False
    """Whether function is an override."""

    contract_name: Optional[str] = None
    """Name of the containing contract."""


class SilentErrorListener(ErrorListener):
    """Suppress ANTLR error output for graceful degradation."""

    def __init__(self):
        self.errors: List[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"Line {line}:{column} - {msg}")


class SolidityFunctionVisitor:
    """
    ANTLR visitor to extract function information from Solidity parse tree.

    Usage:
        visitor = SolidityFunctionVisitor()
        functions = visitor.extract_functions(source_code)
    """

    def __init__(self):
        self._source_lines: List[str] = []
        self._current_contract: Optional[str] = None

    def extract_functions(self, source: str) -> List[SolidityFunction]:
        """
        Parse Solidity source and extract all function definitions.

        Args:
            source: Complete Solidity source code

        Returns:
            List of SolidityFunction objects with parsed metadata
        """
        # Import here to avoid import errors if ANTLR not available
        from .antlr_generated import SolidityLexer, SolidityParser

        self._source_lines = source.split("\n")
        functions: List[SolidityFunction] = []

        try:
            # Create ANTLR input stream and lexer
            input_stream = InputStream(source)
            lexer = SolidityLexer(input_stream)

            # Suppress error output
            error_listener = SilentErrorListener()
            lexer.removeErrorListeners()
            lexer.addErrorListener(error_listener)

            # Create token stream and parser
            token_stream = CommonTokenStream(lexer)
            parser = SolidityParser(token_stream)
            parser.removeErrorListeners()
            parser.addErrorListener(error_listener)

            # Parse the source
            tree = parser.sourceUnit()

            if error_listener.errors:
                logger.debug(f"ANTLR parse warnings: {error_listener.errors[:3]}")

            # Extract functions from parse tree
            functions = self._extract_from_tree(tree)

        except Exception as e:
            logger.warning(f"ANTLR parsing failed: {e}, falling back to empty list")

        return functions

    def _extract_from_tree(self, tree: ParserRuleContext) -> List[SolidityFunction]:
        """Recursively extract functions from parse tree."""
        from .antlr_generated import SolidityParser

        functions: List[SolidityFunction] = []

        # Check if this is a contract definition
        if isinstance(tree, SolidityParser.ContractDefinitionContext):
            # Get contract name
            if tree.identifier():
                self._current_contract = tree.identifier().getText()

        # Check if this is a function definition
        if isinstance(tree, SolidityParser.FunctionDefinitionContext):
            func = self._parse_function(tree)
            if func:
                functions.append(func)

        # Recurse into children
        for i in range(tree.getChildCount()):
            child = tree.getChild(i)
            if isinstance(child, ParserRuleContext):
                functions.extend(self._extract_from_tree(child))

        return functions

    def _parse_function(
        self, ctx: Any  # SolidityParser.FunctionDefinitionContext
    ) -> Optional[SolidityFunction]:
        """Parse a FunctionDefinitionContext into SolidityFunction."""
        from .antlr_generated import SolidityParser

        try:
            # Get function name
            name = ""
            if ctx.identifier():
                name = ctx.identifier().getText()
            elif ctx.Fallback():
                name = "fallback"
            elif ctx.Receive():
                name = "receive"

            if not name:
                return None

            # Get visibility
            # Default to "public" for pre-filtering purposes (Solidity 0.4.x default)
            # Note: Solidity 0.5+ defaults to no visibility (requires explicit)
            # For GPTScan-style filtering, being permissive is safer
            visibility = "public"
            vis_contexts = ctx.visibility()
            if vis_contexts:
                visibility = vis_contexts[0].getText()

            # Get state mutability
            state_mutability = None
            mut_contexts = ctx.stateMutability()
            if mut_contexts:
                state_mutability = mut_contexts[0].getText()

            # Get modifiers
            modifiers: List[str] = []
            mod_contexts = ctx.modifierInvocation()
            if mod_contexts:
                for mod_ctx in mod_contexts:
                    # Get the identifier path (modifier name)
                    if mod_ctx.identifierPath():
                        mod_name = mod_ctx.identifierPath().getText()
                        modifiers.append(mod_name)

            # Get parameters using ctx.arguments (first parameter list)
            parameters: List[Dict[str, str]] = []
            if ctx.arguments:
                parameters = self._parse_parameters(ctx.arguments)

            # Get return types using ctx.returnParameters (second parameter list)
            return_types: List[str] = []
            if ctx.returnParameters:
                ret_params = self._parse_parameters(ctx.returnParameters)
                return_types = [p.get("type", "") for p in ret_params]

            # Check virtual/override
            is_virtual = len(ctx.Virtual()) > 0 if hasattr(ctx, "Virtual") else False
            is_override = (
                len(ctx.overrideSpecifier()) > 0
                if hasattr(ctx, "overrideSpecifier")
                else False
            )

            # Get source positions
            start_line = ctx.start.line if ctx.start else 0
            end_line = ctx.stop.line if ctx.stop else start_line

            # Get full text and body
            full_text = self._get_text_from_context(ctx)
            body_text = ""
            if ctx.block():
                body_text = self._get_text_from_context(ctx.block())

            return SolidityFunction(
                name=name,
                visibility=visibility,
                state_mutability=state_mutability,
                modifiers=modifiers,
                parameters=parameters,
                return_types=return_types,
                body_text=body_text,
                full_text=full_text,
                start_line=start_line,
                end_line=end_line,
                is_virtual=is_virtual,
                is_override=is_override,
                contract_name=self._current_contract,
            )

        except Exception as e:
            logger.debug(f"Failed to parse function: {e}")
            return None

    def _parse_parameters(
        self, param_list_ctx: Any  # SolidityParser.ParameterListContext
    ) -> List[Dict[str, str]]:
        """Parse a parameter list into structured format."""
        from .antlr_generated import SolidityParser

        parameters: List[Dict[str, str]] = []

        if not param_list_ctx:
            return parameters

        try:
            # Get all parameter declarations using getTypedRuleContexts
            params = param_list_ctx.getTypedRuleContexts(
                SolidityParser.ParameterDeclarationContext
            )

            for param in params:
                if not param:
                    continue

                param_info: Dict[str, str] = {}

                # Get type name
                if hasattr(param, "typeName") and param.typeName():
                    param_info["type"] = param.typeName().getText()

                # Get parameter name (optional)
                if hasattr(param, "identifier") and param.identifier():
                    param_info["name"] = param.identifier().getText()

                # Get data location (memory, storage, calldata)
                if hasattr(param, "dataLocation") and param.dataLocation():
                    param_info["location"] = param.dataLocation().getText()

                if param_info:
                    parameters.append(param_info)

        except Exception as e:
            logger.debug(f"Failed to parse parameters: {e}")

        return parameters

    def _get_text_from_context(self, ctx: ParserRuleContext) -> str:
        """Extract original source text for a context."""
        if not ctx or not ctx.start or not ctx.stop:
            return ""

        try:
            start_line = ctx.start.line - 1  # 0-indexed
            end_line = ctx.stop.line  # 1-indexed (exclusive)
            start_col = ctx.start.column
            stop_col = ctx.stop.column + len(ctx.stop.text)

            if start_line == end_line - 1:
                # Single line
                return self._source_lines[start_line][start_col:stop_col]
            else:
                # Multi-line
                lines = []
                for i in range(start_line, min(end_line, len(self._source_lines))):
                    if i == start_line:
                        lines.append(self._source_lines[i][start_col:])
                    elif i == end_line - 1:
                        lines.append(self._source_lines[i][:stop_col])
                    else:
                        lines.append(self._source_lines[i])
                return "\n".join(lines)
        except (IndexError, AttributeError):
            return ""


def extract_functions(source: str) -> List[SolidityFunction]:
    """
    Convenience function to extract functions from Solidity source.

    Args:
        source: Complete Solidity source code

    Returns:
        List of SolidityFunction objects
    """
    visitor = SolidityFunctionVisitor()
    return visitor.extract_functions(source)


def strip_comments_and_strings(code: str) -> str:
    """
    Remove comments and string literals from Solidity code.

    This is important for accurate content-based filtering (FCE, FCNE, etc.)
    to avoid matching patterns inside comments or string literals.

    Args:
        code: Solidity source code

    Returns:
        Code with comments and strings removed
    """
    import re

    # Remove multi-line comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

    # Remove single-line comments
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)

    # Remove string literals (simple approach - doesn't handle escapes perfectly)
    code = re.sub(r'"[^"]*"', '""', code)
    code = re.sub(r"'[^']*'", "''", code)

    return code
