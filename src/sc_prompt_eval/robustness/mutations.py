"""
Mutation operators for robustness testing.

These operators apply semantics-preserving transformations to smart contracts
to test whether LLM vulnerability detection is robust to surface-level changes.
"""
from __future__ import annotations

import re
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MutationResult:
    """Result of applying a mutation."""
    original: str
    mutated: str
    operator: str
    changes_made: int
    description: str


class MutationOperator(ABC):
    """Base class for mutation operators."""

    name: str = "base"

    @abstractmethod
    def apply(self, code: str, seed: int = None) -> MutationResult:
        """Apply mutation to code."""
        pass


class WhitespaceMutation(MutationOperator):
    """Add/remove whitespace and blank lines (semantics-preserving)."""

    name = "whitespace"

    def apply(self, code: str, seed: int = None) -> MutationResult:
        if seed is not None:
            random.seed(seed)

        lines = code.split('\n')
        mutated_lines = []
        changes = 0

        for line in lines:
            # Randomly add extra blank lines
            if random.random() < 0.1:
                mutated_lines.append('')
                changes += 1

            # Randomly modify indentation (add/remove spaces)
            stripped = line.lstrip()
            if stripped and random.random() < 0.2:
                current_indent = len(line) - len(stripped)
                # Add or remove some spaces (keep valid indentation)
                delta = random.choice([-1, 0, 1]) * 4
                new_indent = max(0, current_indent + delta)
                mutated_lines.append(' ' * new_indent + stripped)
                if new_indent != current_indent:
                    changes += 1
            else:
                mutated_lines.append(line)

        return MutationResult(
            original=code,
            mutated='\n'.join(mutated_lines),
            operator=self.name,
            changes_made=changes,
            description=f"Modified whitespace in {changes} locations"
        )


class CommentMutation(MutationOperator):
    """Add/remove/modify comments (semantics-preserving)."""

    name = "comment"

    FILLER_COMMENTS = [
        "// TODO: review this",
        "// checked",
        "// important logic here",
        "// see documentation",
        "// verified",
        "// handles edge case",
        "/* placeholder */",
        "// note: tested",
    ]

    def apply(self, code: str, seed: int = None) -> MutationResult:
        if seed is not None:
            random.seed(seed)

        lines = code.split('\n')
        mutated_lines = []
        changes = 0

        for line in lines:
            # Remove existing single-line comments with some probability
            if '//' in line and random.random() < 0.3:
                # Only remove if it's a trailing comment
                if not line.strip().startswith('//'):
                    line = line.split('//')[0].rstrip()
                    changes += 1

            # Add random comments before some lines
            if random.random() < 0.1 and line.strip():
                indent = len(line) - len(line.lstrip())
                comment = random.choice(self.FILLER_COMMENTS)
                mutated_lines.append(' ' * indent + comment)
                changes += 1

            mutated_lines.append(line)

        return MutationResult(
            original=code,
            mutated='\n'.join(mutated_lines),
            operator=self.name,
            changes_made=changes,
            description=f"Modified {changes} comments"
        )


class VariableRenameMutation(MutationOperator):
    """Rename local variables (semantics-preserving within scope)."""

    name = "variable_rename"

    # Patterns to match variable declarations
    VAR_PATTERNS = [
        r'(uint\d*|int\d*|address|bool|bytes\d*|string)\s+(\w+)',
        r'(mapping\s*\([^)]+\))\s+(\w+)',
    ]

    PREFIXES = ['_', 'local_', 'var_', 'tmp_', 'the_']
    SUFFIXES = ['_val', '_var', '_data', '1', '2']

    def apply(self, code: str, seed: int = None) -> MutationResult:
        if seed is not None:
            random.seed(seed)

        mutated = code
        changes = 0

        # Find local variable declarations
        for pattern in self.VAR_PATTERNS:
            matches = list(re.finditer(pattern, mutated))

            for match in matches:
                var_name = match.group(2)

                # Skip common reserved names
                if var_name in ['msg', 'block', 'tx', 'this', 'super', 'owner', 'sender']:
                    continue

                # Skip if it looks like a state variable (no local context)
                if random.random() < 0.7:  # Only rename 30% of variables
                    continue

                # Generate new name
                if random.random() < 0.5:
                    new_name = random.choice(self.PREFIXES) + var_name
                else:
                    new_name = var_name + random.choice(self.SUFFIXES)

                # Replace all occurrences (simple approach - may cause issues in complex code)
                # Use word boundary to avoid partial matches
                mutated = re.sub(r'\b' + var_name + r'\b', new_name, mutated)
                changes += 1

        return MutationResult(
            original=code,
            mutated=mutated,
            operator=self.name,
            changes_made=changes,
            description=f"Renamed {changes} variables"
        )


class FunctionReorderMutation(MutationOperator):
    """Reorder function definitions (semantics-preserving in Solidity)."""

    name = "function_reorder"

    def apply(self, code: str, seed: int = None) -> MutationResult:
        if seed is not None:
            random.seed(seed)

        # Simple regex to find function boundaries
        # This is a simplified approach - production code would need proper parsing
        function_pattern = r'(function\s+\w+\s*\([^)]*\)[^{]*\{)'

        # Find all function starts
        matches = list(re.finditer(function_pattern, code))

        if len(matches) < 2:
            return MutationResult(
                original=code,
                mutated=code,
                operator=self.name,
                changes_made=0,
                description="No functions to reorder"
            )

        # Extract function bodies (simplified - assumes balanced braces)
        functions = []
        for i, match in enumerate(matches):
            start = match.start()
            # Find matching closing brace
            depth = 0
            end = start
            in_string = False
            for j, char in enumerate(code[start:], start):
                if char == '"' and (j == start or code[j-1] != '\\'):
                    in_string = not in_string
                if not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            end = j + 1
                            break
            functions.append((start, end, code[start:end]))

        if len(functions) < 2:
            return MutationResult(
                original=code,
                mutated=code,
                operator=self.name,
                changes_made=0,
                description="Could not parse functions"
            )

        # Shuffle functions
        function_bodies = [f[2] for f in functions]
        shuffled = function_bodies.copy()
        random.shuffle(shuffled)

        # Rebuild code
        mutated = code
        # Replace in reverse order to preserve positions
        for i in range(len(functions) - 1, -1, -1):
            start, end, _ = functions[i]
            mutated = mutated[:start] + shuffled[i] + mutated[end:]

        changes = sum(1 for a, b in zip(function_bodies, shuffled) if a != b)

        return MutationResult(
            original=code,
            mutated=mutated,
            operator=self.name,
            changes_made=changes,
            description=f"Reordered {changes} functions"
        )


class NamingStyleMutation(MutationOperator):
    """Change naming conventions (camelCase <-> snake_case)."""

    name = "naming_style"

    def _camel_to_snake(self, name: str) -> str:
        """Convert camelCase to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _snake_to_camel(self, name: str) -> str:
        """Convert snake_case to camelCase."""
        components = name.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])

    def apply(self, code: str, seed: int = None) -> MutationResult:
        if seed is not None:
            random.seed(seed)

        mutated = code
        changes = 0

        # Find potential variable/function names
        identifiers = re.findall(r'\b([a-z][a-zA-Z0-9_]*)\b', code)
        unique_ids = set(identifiers)

        # Skip Solidity keywords and common names
        skip = {'function', 'contract', 'pragma', 'solidity', 'public', 'private',
                'internal', 'external', 'view', 'pure', 'payable', 'returns',
                'return', 'require', 'assert', 'revert', 'if', 'else', 'for',
                'while', 'mapping', 'address', 'uint', 'int', 'bool', 'string',
                'bytes', 'msg', 'block', 'tx', 'this', 'true', 'false', 'memory',
                'storage', 'calldata', 'emit', 'event', 'modifier', 'constructor'}

        for identifier in unique_ids:
            if identifier in skip or len(identifier) < 4:
                continue

            if random.random() > 0.2:  # Only change 20% of names
                continue

            # Determine current style and convert
            if '_' in identifier:
                new_name = self._snake_to_camel(identifier)
            elif any(c.isupper() for c in identifier[1:]):
                new_name = self._camel_to_snake(identifier)
            else:
                continue

            if new_name != identifier:
                mutated = re.sub(r'\b' + identifier + r'\b', new_name, mutated)
                changes += 1

        return MutationResult(
            original=code,
            mutated=mutated,
            operator=self.name,
            changes_made=changes,
            description=f"Changed naming style for {changes} identifiers"
        )


def apply_mutations(
    code: str,
    operators: List[MutationOperator] = None,
    seed: int = None
) -> Tuple[str, List[MutationResult]]:
    """
    Apply multiple mutation operators to code.

    Args:
        code: Source code to mutate
        operators: List of mutation operators (default: all)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (mutated code, list of mutation results)
    """
    if operators is None:
        operators = [
            WhitespaceMutation(),
            CommentMutation(),
            VariableRenameMutation(),
            FunctionReorderMutation(),
            NamingStyleMutation(),
        ]

    results = []
    current = code

    for i, op in enumerate(operators):
        op_seed = seed + i if seed is not None else None
        result = op.apply(current, seed=op_seed)
        results.append(result)
        current = result.mutated

    return current, results


def generate_mutants(
    code: str,
    n_mutants: int = 5,
    base_seed: int = 42
) -> List[Tuple[str, List[MutationResult]]]:
    """
    Generate multiple mutant versions of code.

    Args:
        code: Source code
        n_mutants: Number of mutants to generate
        base_seed: Base random seed

    Returns:
        List of (mutated code, mutation results) tuples
    """
    mutants = []

    for i in range(n_mutants):
        # Use different subset of operators for variety
        all_operators = [
            WhitespaceMutation(),
            CommentMutation(),
            VariableRenameMutation(),
            FunctionReorderMutation(),
            NamingStyleMutation(),
        ]

        # Randomly select 2-4 operators
        random.seed(base_seed + i)
        n_ops = random.randint(2, 4)
        selected = random.sample(all_operators, n_ops)

        mutated, results = apply_mutations(code, selected, seed=base_seed + i * 100)
        mutants.append((mutated, results))

    return mutants
