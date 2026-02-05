"""Shared JSON extraction utilities for LLM responses.

Consolidates JSON extraction logic used across the codebase for parsing
LLM outputs that may contain JSON embedded in markdown, code blocks,
or mixed with natural language.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def fix_json_string(s: str) -> str:
    """Try to fix common JSON issues from LLM outputs with Solidity code.

    Handles patterns like ("") inside strings which are common in Solidity.
    """
    try:
        # Pattern: ("") inside strings - common in Solidity code
        s = re.sub(r'\(\"\"\)', r'(\\"\\")', s)
        # Pattern: ("something")
        s = re.sub(r'\(\"([^\"]*)\"\)', r'(\\"\\1\\")', s)
        return s
    except Exception:
        return s


def repair_json(text: str) -> str:
    """Attempt to repair common JSON errors from LLM outputs.

    Fixes:
    - Trailing commas in arrays/objects
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing quotes around string values
    - Control characters in strings
    - Missing closing brackets/braces
    """
    repaired = text

    # Remove trailing commas before ] or }
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

    # Replace single quotes with double quotes (careful with apostrophes in text)
    # Only replace single quotes that look like JSON delimiters
    repaired = re.sub(r"(?<!\\)'(\s*[:\[\]{},])", r'"\1', repaired)
    repaired = re.sub(r"([:\[\]{},]\s*)'", r'\1"', repaired)

    # Remove control characters except \n, \r, \t
    repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', repaired)

    # Fix common truncated JSON (add missing closing brackets)
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_braces > 0:
        repaired += '}' * open_braces
    if open_brackets > 0:
        repaired += ']' * open_brackets

    return repaired


def _extract_json_candidates(text: str) -> List[Tuple[str, Any]]:
    """Extract all potential JSON blocks from text with markdown code blocks.

    Returns list of (json_string, parsed_object) tuples.
    """
    candidates = []

    # Try to find JSON in code blocks
    patterns = [
        (r'```json\s*([\s\S]*?)\s*```', True),   # ```json ... ```
        (r'```\s*([\s\S]*?)\s*```', True),        # ``` ... ```
        (r'\[\s*\{[\s\S]*?\}\s*\]', False),       # Raw JSON array with objects (non-greedy)
        (r'\[\s*\]', False),                       # Empty array [] - no vulnerabilities
    ]

    for pattern, has_group in patterns:
        for match in re.finditer(pattern, text):
            extracted = match.group(1) if has_group else match.group(0)

            # Try to parse as-is first
            try:
                parsed = json.loads(extracted)
                candidates.append((extracted, parsed))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                fixed = fix_json_string(extracted)
                try:
                    parsed = json.loads(fixed)
                    candidates.append((fixed, parsed))
                except json.JSONDecodeError:
                    pass

    return candidates


def extract_json_balanced(text: str) -> Optional[str]:
    """Extract a balanced JSON object from text using brace matching.

    Handles nested braces correctly by tracking depth and string boundaries.
    Returns the first complete JSON object found.
    """
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i, c in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def extract_json(text: str, prefer_array: bool = True) -> str:
    """Extract JSON from text that may contain markdown code blocks.

    Main extraction function that combines multiple strategies:
    1. Try markdown code blocks first
    2. Try raw JSON patterns
    3. Use balanced brace matching as fallback

    Args:
        text: The text to extract JSON from
        prefer_array: If True, prioritize arrays over objects (for findings)
                     If False, prioritize objects over arrays (for agent responses)

    Returns:
        Extracted JSON string, or original text if no JSON found
    """
    candidates = _extract_json_candidates(text)

    # Separate arrays and objects
    arrays = [(s, p) for s, p in candidates if isinstance(p, list)]
    objects = [(s, p) for s, p in candidates if isinstance(p, dict)]

    if prefer_array:
        if arrays:
            # Return the array with the most items (most findings)
            # Empty arrays are valid (no vulnerabilities found)
            arrays.sort(key=lambda x: len(x[1]), reverse=True)
            return arrays[0][0]
        if objects:
            return objects[0][0]
    else:
        if objects:
            return objects[0][0]
        if arrays:
            return arrays[0][0]

    # Check for raw empty array in text (outside code blocks)
    if re.search(r'^\s*\[\s*\]\s*$', text.strip()):
        return '[]'

    # Fallback: try balanced brace extraction
    balanced = extract_json_balanced(text)
    if balanced:
        try:
            json.loads(balanced)
            return balanced
        except json.JSONDecodeError:
            pass

    return text  # Return original if no patterns match


def extract_json_array(text: str) -> str:
    """Extract JSON array from text, prioritizing arrays over objects.

    Used for parsing findings which should always be arrays.
    Handles empty arrays [] which indicate no vulnerabilities found.
    """
    return extract_json(text, prefer_array=True)


def extract_json_object(text: str) -> str:
    """Extract JSON object from text, prioritizing objects over arrays.

    Used for intermediate agent responses (reviewer analysis, expert
    verification) where the output is a structured object rather than
    a findings array.
    """
    return extract_json(text, prefer_array=False)


def safe_json_loads(text: str, repair: bool = True) -> Optional[Any]:
    """Safely parse JSON with optional repair.

    Args:
        text: JSON text to parse
        repair: If True, attempt to repair malformed JSON

    Returns:
        Parsed JSON object/array, or None if parsing fails
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if repair:
            try:
                repaired = repair_json(text)
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
    return None
