from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

# Mapping for string confidence/severity to numeric values
CONFIDENCE_MAP = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.3,
    "very high": 0.95,
    "very low": 0.1,
}


class Evidence(BaseModel):
    file: str = "contract.sol"
    lines: List[int] = Field(default_factory=list)
    function: str = ""

    @field_validator('function', mode='before')
    @classmethod
    def parse_function(cls, v):
        """Handle None or non-string function values."""
        if v is None:
            return ""
        return str(v)

    @field_validator('file', mode='before')
    @classmethod
    def parse_file(cls, v):
        """Handle None or non-string file values."""
        if v is None:
            return "contract.sol"
        return str(v)

    @field_validator('lines', mode='before')
    @classmethod
    def parse_lines(cls, v):
        """Convert various line formats to list of ints."""
        if v is None:
            return []
        if isinstance(v, list):
            # Already a list, ensure all items are ints
            result = []
            for item in v:
                try:
                    result.append(int(item))
                except (ValueError, TypeError):
                    pass
            return result
        if isinstance(v, int):
            return [v]
        if isinstance(v, str):
            # Handle formats like "27", "27, 28", "27-28", "[27, 28]"
            v = v.strip().strip('[]')
            if not v:
                return []
            # Try splitting by comma
            if ',' in v:
                parts = v.split(',')
            elif '-' in v and v.count('-') == 1:
                # Range like "27-30"
                try:
                    start, end = v.split('-')
                    return list(range(int(start.strip()), int(end.strip()) + 1))
                except (ValueError, TypeError):
                    parts = [v]
            else:
                parts = [v]
            result = []
            for p in parts:
                try:
                    result.append(int(p.strip()))
                except ValueError:
                    pass
            return result
        return []


class Finding(BaseModel):
    """A vulnerability finding from LLM analysis.

    Uses DASP category taxonomy for classification.
    """
    title: str
    category: str = ""  # DASP category (reentrancy, arithmetic, etc.)
    severity: str = "medium"
    confidence: float = 0.5
    evidence: Optional[Evidence] = None
    explanation: str = ""
    fix_suggestion: str = ""

    # GPTScan multi-stage pipeline fields
    key_variables: List[str] = Field(default_factory=list)
    key_statements: List[str] = Field(default_factory=list)
    scenario_matched: str = ""  # Which GPTScan scenario matched
    property_verified: str = ""  # Which property was confirmed
    static_check_type: str = ""  # DF, VC, OC, FA
    source: str = ""  # slither, manual, gptscan_multistage

    # Static confirmation results
    static_confirmed: Optional[bool] = None  # True=confirmed, False=rejected, None=not checked
    static_check_reason: str = ""  # Reason for confirmation/rejection

    model_config = {"extra": "ignore"}  # Ignore unknown fields from LLM output

    @field_validator('key_variables', mode='before')
    @classmethod
    def parse_key_variables(cls, v):
        """Handle various key_variables formats."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x]
        if isinstance(v, str):
            return [v] if v else []
        return []

    @field_validator('key_statements', mode='before')
    @classmethod
    def parse_key_statements(cls, v):
        """Handle various key_statements formats."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x]
        if isinstance(v, str):
            return [v] if v else []
        return []

    @field_validator('evidence', mode='before')
    @classmethod
    def parse_evidence(cls, v):
        """Handle evidence field from various formats."""
        if v is None:
            return None
        if isinstance(v, Evidence):
            return v
        if isinstance(v, dict):
            return Evidence.model_validate(v)
        return None

    @field_validator('category', mode='before')
    @classmethod
    def parse_category(cls, v):
        """Normalize category field to DASP taxonomy."""
        if v is None:
            return ""

        # Convert to lowercase and normalize
        cat = str(v).lower().strip().replace('-', '_').replace(' ', '_')

        # Valid DASP categories
        valid_categories = {
            'reentrancy', 'access_control', 'arithmetic',
            'unchecked_low_level_calls', 'denial_of_service',
            'bad_randomness', 'front_running', 'time_manipulation',
            'short_addresses', 'other'
        }

        # If already valid, return as-is
        if cat in valid_categories:
            return cat

        # Try to extract valid category from malformed output
        # e.g., "dasp_category_(reentrancy)" -> "reentrancy"
        for valid_cat in valid_categories:
            if valid_cat in cat:
                return valid_cat

        # Common aliases
        aliases = {
            'integer_overflow': 'arithmetic',
            'integer_underflow': 'arithmetic',
            'overflow': 'arithmetic',
            'underflow': 'arithmetic',
            'tx_origin': 'access_control',
            'tx.origin': 'access_control',
            'timestamp': 'time_manipulation',
            'block_timestamp': 'time_manipulation',
            'dos': 'denial_of_service',
            'race_condition': 'front_running',
            'tod': 'front_running',
            'unchecked_call': 'unchecked_low_level_calls',
            'unchecked_return': 'unchecked_low_level_calls',
            'low_level_call': 'unchecked_low_level_calls',
        }

        for alias, valid_cat in aliases.items():
            if alias in cat:
                return valid_cat

        return cat  # Return normalized but unmatched category

    @field_validator('confidence', mode='before')
    @classmethod
    def parse_confidence(cls, v):
        """Convert string confidence to float."""
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            # Try to parse as number first
            try:
                return float(v)
            except ValueError:
                # Map string values like "High", "Medium", "Low"
                return CONFIDENCE_MAP.get(v.lower().strip(), 0.5)
        return 0.5

def fix_json_string(s: str) -> str:
    """Try to fix common JSON issues from LLM outputs."""
    import re

    # Try to fix common problematic patterns, return original if it breaks
    try:
        # Pattern: ("") inside strings - common in Solidity code
        s = re.sub(r'\(\"\"\)', r'(\\"\\")', s)
        # Pattern: ("something")
        s = re.sub(r'\(\"([^\"]*)\"\)', r'(\\"\\1\\")', s)
        return s
    except (re.error, TypeError):
        return s


def _extract_json_candidates(text: str) -> list:
    """Extract all potential JSON blocks from text with markdown code blocks.

    Returns list of (json_string, parsed_object) tuples.
    """
    import re

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


def extract_json_from_text(text: str) -> str:
    """Extract JSON from text that may contain markdown code blocks.

    Prioritizes JSON arrays over objects since findings should always be arrays.
    For prompts like P3_structured that output multiple JSON blocks (checklist object,
    then findings array), this ensures we get the findings array.

    Handles empty arrays [] which indicate no vulnerabilities found.
    """
    import re

    candidates = _extract_json_candidates(text)

    # Prioritize: arrays first (findings), then objects
    # Among arrays, prefer longer ones (more findings), but accept empty arrays
    arrays = [(s, p) for s, p in candidates if isinstance(p, list)]
    objects = [(s, p) for s, p in candidates if isinstance(p, dict)]

    if arrays:
        # Return the array with the most items (most findings)
        # Empty arrays are valid (no vulnerabilities found)
        arrays.sort(key=lambda x: len(x[1]), reverse=True)
        return arrays[0][0]

    if objects:
        return objects[0][0]

    # Check for raw empty array in text (outside code blocks)
    if re.search(r'^\s*\[\s*\]\s*$', text.strip()):
        return '[]'

    return text  # Return original if no patterns match


def extract_json_object_from_text(text: str) -> str:
    """Extract JSON object from text that may contain markdown code blocks.

    Prioritizes JSON objects over arrays. Used for intermediate agent responses
    (reviewer analysis, expert verification) where the output is a structured
    object rather than a findings array.
    """
    import re

    candidates = _extract_json_candidates(text)

    # Prioritize: objects first, then arrays
    objects = [(s, p) for s, p in candidates if isinstance(p, dict)]
    arrays = [(s, p) for s, p in candidates if isinstance(p, list)]

    if objects:
        return objects[0][0]

    if arrays:
        return arrays[0][0]

    return text  # Return original if no patterns match


def repair_json(text: str) -> str:
    """Attempt to repair common JSON errors from LLM outputs.

    Fixes:
    - Trailing commas in arrays/objects
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing quotes around string values
    - Control characters in strings
    - Unescaped newlines/tabs inside strings
    - Markdown code fence prefixes (```json)
    """
    import re

    repaired = text

    # Strip markdown code fence prefix if present at start
    repaired = re.sub(r'^```(?:json)?\s*\n?', '', repaired.strip())
    repaired = re.sub(r'\n?```\s*$', '', repaired.strip())

    # Remove trailing commas before ] or }
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

    # Replace single quotes with double quotes (careful with apostrophes in text)
    # Only replace single quotes that look like JSON delimiters
    repaired = re.sub(r"(?<!\\)'(\s*[:\[\]{},])", r'"\1', repaired)
    repaired = re.sub(r"([:\[\]{},]\s*)'", r'\1"', repaired)

    # Remove control characters except \n, \r, \t
    repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', repaired)

    # Escape unescaped newlines and tabs inside strings
    # This is tricky - we need to find strings and escape their contents
    def escape_string_contents(match):
        content = match.group(1)
        # Escape actual newlines/tabs that aren't already escaped
        content = content.replace('\n', '\\n')
        content = content.replace('\r', '\\r')
        content = content.replace('\t', '\\t')
        return '"' + content + '"'

    # Match strings (simplistic but covers most cases)
    # This regex finds "..." patterns where ... doesn't contain unescaped quotes
    repaired = re.sub(r'"((?:[^"\\]|\\.)*)(?:\n|\r|\t)((?:[^"\\]|\\.)*)"',
                      lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') +
                               m.group(2).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"',
                      repaired)

    # Fix common truncated JSON (add missing closing brackets)
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_braces > 0:
        repaired += '}' * open_braces
    if open_brackets > 0:
        repaired += ']' * open_brackets

    return repaired


def parse_findings(text: str) -> Tuple[List[Finding], List[str]]:
    """Best-effort parse: returns (findings, errors).

    Includes JSON repair for common LLM output errors.
    """
    import logging
    logger = logging.getLogger(__name__)

    errors: List[str] = []

    # Try to extract JSON from text
    json_text = extract_json_from_text(text.strip())

    # First attempt: parse directly
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        # Second attempt: repair and retry
        repaired_text = repair_json(json_text)
        try:
            data = json.loads(repaired_text)
            logger.info("JSON parse succeeded after repair")
        except json.JSONDecodeError as e2:
            # Log the failure for visibility
            preview = text[:200] + '...' if len(text) > 200 else text
            logger.warning(f"JSON parse failed even after repair: {e2}. Preview: {preview}")
            errors.append(f"json_load_error: {e2} (repair attempted)")
            return [], errors

    if not isinstance(data, list):
        errors.append("top_level_not_list")
        return [], errors

    out: List[Finding] = []
    for i, item in enumerate(data):
        try:
            out.append(Finding.model_validate(item))
        except ValidationError as e:
            errors.append(f"item_{i}_validation_error: {e}")
    return out, errors

