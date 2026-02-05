from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union, Set


# =============================================================================
# DASP Top 10 Taxonomy (Canonical for SmartBugs Curated)
# https://dasp.co/
# =============================================================================

DASP_CATEGORIES = [
    "reentrancy",
    "access_control",
    "arithmetic",
    "unchecked_low_level_calls",
    "denial_of_service",
    "bad_randomness",
    "front_running",
    "time_manipulation",
    "short_addresses",
    "other",
]

# Keywords in finding titles/explanations that indicate specific DASP categories
# Used for semantic matching when LLM outputs natural language descriptions
# Order matters! More specific patterns should come first to avoid false matches
CATEGORY_KEYWORDS = {
    # Check unchecked_low_level_calls FIRST - it often gets confused with reentrancy
    # because explanations mention "reentrancy" as a possible consequence
    'unchecked_low_level_calls': [
        'unchecked return', 'unchecked call', 'unchecked send',
        'return value not checked', 'ignoring return',
        'unchecked low-level', 'low-level call',
        'uncontrolled external', 'external call without',
        'uncontrolled ether', 'unchecked external'
    ],
    'short_addresses': [
        'short address', 'short-address', 'parameter attack',
        'input validation', 'calldata length'
    ],
    'denial_of_service': [
        'denial of service', 'dos with', 'dos via', 'dos)', 'gas limit',
        'block gas limit', 'failed call', 'send revert', '(dos',
        'unbounded loop', 'gas exhaustion'
    ],
    'bad_randomness': [
        'weak random', 'predictable random', 'blockhash', 'block.blockhash',
        'insecure random', 'pseudo-random', 'pseudorandom', 'predictable',
        'weak randomness', 'block.difficulty', 'coinbase'
    ],
    'front_running': [
        'front run', 'front-run', 'frontrun', 'transaction order depend',
        'race condition', 'toctou', 'transaction ordering'
    ],
    'arithmetic': [
        'integer overflow', 'integer underflow', 'overflow', 'underflow',
        'arithmetic', 'safemath', 'safe math'
    ],
    'time_manipulation': [
        'timestamp depend', 'block.timestamp depend', 'now depend',
        'time manipulation', 'timestamp manipul', 'block timestamp',
        'miner manipulation'
    ],
    'access_control': [
        'missing owner', 'unprotected function', 'lack of access control',
        'missing access control', 'authorization', 'unrestricted',
        'anyone can call', 'no access control', 'suicide', 'constructor name',
        'access control', 'delegate call', 'delegatecall', 'owner privilege',
        'unprotected owner', 'tx.origin', 'selfdestruct', 'visibility'
    ],
    # reentrancy should come AFTER unchecked_low_level_calls
    'reentrancy': [
        'reentran', 're-entran', 'recursive call', 'cross-function',
        'state update after external call'
    ],
    'other': [
        'uninitialized storage', 'storage pointer', 'floating pragma',
        'outdated compiler'
    ],
}


def get_category_from_text(title: str, explanation: str = "") -> str:
    """Extract DASP category from finding title/explanation using keyword matching.

    This is the primary method for semantic matching of LLM outputs to
    ground truth categories.

    Args:
        title: Finding title or vulnerability name
        explanation: Optional detailed explanation

    Returns:
        DASP category string or 'unknown' if no match
    """
    text = (title + " " + explanation).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return 'unknown'


def normalize_category(category: str) -> str:
    """Normalize a category string to valid DASP category.

    Handles common variations and aliases.
    """
    if not category:
        return 'unknown'

    cat_lower = category.lower().strip().replace('-', '_').replace(' ', '_')

    # Direct match
    if cat_lower in DASP_CATEGORIES:
        return cat_lower

    # Common aliases
    aliases = {
        'integer_overflow': 'arithmetic',
        'integer_underflow': 'arithmetic',
        'overflow': 'arithmetic',
        'underflow': 'arithmetic',
        'unchecked_call': 'unchecked_low_level_calls',
        'unchecked_return': 'unchecked_low_level_calls',
        'dos': 'denial_of_service',
        'randomness': 'bad_randomness',
        'weak_randomness': 'bad_randomness',
        'timestamp': 'time_manipulation',
        'block_timestamp': 'time_manipulation',
        'frontrunning': 'front_running',
        'race_condition': 'front_running',
        'authorization': 'access_control',
        'tx_origin': 'access_control',
        'short_address': 'short_addresses',
    }

    return aliases.get(cat_lower, 'unknown')


@dataclass
class ScoreSummary:
    """Summary of precision, recall, F1 scores."""
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass
class MetricsResult:
    """Detailed metrics result with category breakdowns."""
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_categories: List[str] = field(default_factory=list)
    missed_categories: List[str] = field(default_factory=list)
    spurious_categories: List[str] = field(default_factory=list)
    unknown_predictions: int = 0  # Count of predictions that couldn't be categorized


@dataclass
class InstanceMetricsResult:
    """Instance-level metrics with granular matching by line numbers.

    This provides more accurate evaluation when contracts have multiple
    vulnerabilities of the same type at different locations.
    """
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_instances: List[Dict] = field(default_factory=list)  # Details of each match
    unmatched_predictions: List[Dict] = field(default_factory=list)
    unmatched_ground_truth: List[Dict] = field(default_factory=list)


def prf1(tp: int, fp: int, fn: int) -> ScoreSummary:
    """Compute precision, recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ScoreSummary(precision, recall, f1, tp, fp, fn)


def match_by_category(pred: List[dict], gold: List[dict]) -> ScoreSummary:
    """Contract-level matching by DASP category.

    Note: Unknown predictions ARE counted as false positives.
    This is the stricter evaluation mode.
    """
    pred_categories = []
    unknown_count = 0

    for p in pred:
        cat = normalize_category(p.get("category", ""))
        if cat == 'unknown':
            unknown_count += 1
        else:
            pred_categories.append(cat)

    pred_set = set(pred_categories)
    gold_set = {normalize_category(g.get("category", "")) for g in gold}
    gold_set.discard('unknown')

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set) + unknown_count  # Unknown predictions count as FP
    fn = len(gold_set - pred_set)
    return prf1(tp, fp, fn)


def _get_lines(obj: Any) -> Set[int]:
    """Extract line numbers from a finding or ground truth object."""
    lines = set()

    if isinstance(obj, dict):
        # Check various possible field names
        if 'evidence' in obj and isinstance(obj['evidence'], dict):
            raw_lines = obj['evidence'].get('lines', [])
        elif 'line_numbers' in obj:
            raw_lines = obj['line_numbers']
        elif 'lines' in obj:
            raw_lines = obj['lines']
        else:
            raw_lines = []
    elif hasattr(obj, 'evidence') and obj.evidence:
        raw_lines = obj.evidence.lines if hasattr(obj.evidence, 'lines') else []
    elif hasattr(obj, 'line_numbers'):
        raw_lines = obj.line_numbers
    else:
        raw_lines = []

    # Convert to set of ints
    if isinstance(raw_lines, (list, tuple, set)):
        for line in raw_lines:
            try:
                lines.add(int(line))
            except (ValueError, TypeError):
                pass
    elif isinstance(raw_lines, int):
        lines.add(raw_lines)

    return lines


def _get_function(obj: Any) -> str:
    """Extract function name from a finding or ground truth object."""
    if isinstance(obj, dict):
        if 'evidence' in obj and isinstance(obj['evidence'], dict):
            return obj['evidence'].get('function', '') or ''
        return obj.get('function_name', '') or obj.get('function', '') or ''
    elif hasattr(obj, 'evidence') and obj.evidence:
        return obj.evidence.function if hasattr(obj.evidence, 'function') else ''
    elif hasattr(obj, 'function_name'):
        return obj.function_name or ''
    return ''


def _lines_overlap(lines1: Set[int], lines2: Set[int], tolerance: int = 3) -> bool:
    """Check if two sets of line numbers overlap or are within tolerance.

    Args:
        lines1: First set of line numbers
        lines2: Second set of line numbers
        tolerance: Maximum distance between lines to consider a match

    Returns:
        True if lines overlap or are within tolerance
    """
    if not lines1 or not lines2:
        # If either has no lines, we can't do line-level matching
        return False

    # Direct overlap
    if lines1 & lines2:
        return True

    # Check if any lines are within tolerance
    for l1 in lines1:
        for l2 in lines2:
            if abs(l1 - l2) <= tolerance:
                return True

    return False


def compute_instance_metrics(
    predictions: List[Any],
    ground_truth: List[Any],
    line_tolerance: int = 3,
) -> InstanceMetricsResult:
    """
    Compute instance-level metrics with granular matching by line numbers.

    This evaluates each vulnerability instance separately, using line number
    overlap to match predictions to ground truth. This handles cases where
    a contract has multiple vulnerabilities of the same type at different locations.

    Matching rules:
    1. Same category AND overlapping/nearby lines (within tolerance) = TP
    2. Prediction with no matching ground truth = FP (including unknown category)
    3. Ground truth with no matching prediction = FN

    Args:
        predictions: List of Finding objects or dicts
        ground_truth: List of ground truth vulnerability annotations
        line_tolerance: Maximum line distance to consider a match (default 3)

    Returns:
        InstanceMetricsResult with detailed matching information
    """
    # Normalize predictions
    pred_list = []
    for p in predictions:
        title = ""
        explanation = ""
        explicit_category = ""

        if hasattr(p, 'title'):
            title = p.title or ""
        elif isinstance(p, dict):
            title = p.get('title', '')

        if hasattr(p, 'explanation'):
            explanation = p.explanation or ""
        elif isinstance(p, dict):
            explanation = p.get('explanation', '')

        if hasattr(p, 'category'):
            explicit_category = p.category or ""
        elif isinstance(p, dict):
            explicit_category = p.get('category', '')

        # Determine category
        if explicit_category:
            cat = normalize_category(explicit_category)
        else:
            cat = get_category_from_text(title, explanation)

        pred_list.append({
            'category': cat,
            'lines': _get_lines(p),
            'function': _get_function(p),
            'title': title,
            'original': p,
            'matched': False,
        })

    # Normalize ground truth
    gt_list = []
    for g in ground_truth:
        if isinstance(g, dict):
            cat = normalize_category(g.get('category', ''))
        elif hasattr(g, 'category'):
            cat = normalize_category(g.category or '')
        else:
            cat = 'unknown'

        gt_list.append({
            'category': cat,
            'lines': _get_lines(g),
            'function': _get_function(g),
            'original': g,
            'matched': False,
        })

    # Match predictions to ground truth
    matched_instances = []

    for pred in pred_list:
        if pred['matched']:
            continue

        for gt in gt_list:
            if gt['matched']:
                continue

            # Must have same category (unknown predictions won't match)
            if pred['category'] != gt['category'] or pred['category'] == 'unknown':
                continue

            # Check line overlap
            if _lines_overlap(pred['lines'], gt['lines'], line_tolerance):
                pred['matched'] = True
                gt['matched'] = True
                matched_instances.append({
                    'category': pred['category'],
                    'pred_lines': list(pred['lines']),
                    'gt_lines': list(gt['lines']),
                    'function': pred['function'] or gt['function'],
                })
                break

            # If no lines available, fall back to function name matching
            if not pred['lines'] and not gt['lines']:
                pred_func = pred['function'].lower()
                gt_func = gt['function'].lower()
                if pred_func and gt_func and (pred_func in gt_func or gt_func in pred_func):
                    pred['matched'] = True
                    gt['matched'] = True
                    matched_instances.append({
                        'category': pred['category'],
                        'pred_lines': [],
                        'gt_lines': [],
                        'function': pred['function'] or gt['function'],
                        'matched_by': 'function_name',
                    })
                    break

    # Collect unmatched
    unmatched_predictions = [
        {'category': p['category'], 'lines': list(p['lines']), 'title': p['title']}
        for p in pred_list if not p['matched']
    ]
    unmatched_ground_truth = [
        {'category': g['category'], 'lines': list(g['lines'])}
        for g in gt_list if not g['matched']
    ]

    tp = len(matched_instances)
    fp = len(unmatched_predictions)
    fn = len(unmatched_ground_truth)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return InstanceMetricsResult(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        matched_instances=matched_instances,
        unmatched_predictions=unmatched_predictions,
        unmatched_ground_truth=unmatched_ground_truth,
    )


def compute_metrics(
    predictions: List[Any],
    ground_truth: List[Any],
    strict: bool = True,
) -> MetricsResult:
    """
    Compute vulnerability detection metrics using DASP taxonomy.

    Uses category-level matching where predictions and ground truth are
    collapsed into sets of unique categories. For granular instance-level
    matching, use compute_instance_metrics() instead.

    IMPORTANT: Predictions that cannot be categorized (unknown) ARE counted
    as false positives. This prevents precision inflation from uncategorizable
    hallucinations.

    Args:
        predictions: List of Finding objects or dicts with title/explanation
        ground_truth: List of dicts with 'category' field (DASP category)
        strict: If True (default), unknown predictions count as FP.
                If False, unknown predictions are ignored (legacy behavior).

    Returns:
        MetricsResult with precision, recall, F1, and detailed breakdown
    """
    # Extract categories from ground truth
    gold_categories = set()
    for g in ground_truth:
        category = None
        if isinstance(g, dict):
            category = g.get('category', '')
        elif hasattr(g, 'category'):
            category = g.category or ''

        if category:
            cat = normalize_category(category)
            if cat != 'unknown':
                gold_categories.add(cat)

    # Extract predicted categories using semantic matching
    pred_categories = set()
    unknown_count = 0

    for p in predictions:
        title = ""
        explanation = ""
        explicit_category = ""

        if hasattr(p, 'title'):
            title = p.title or ""
        elif isinstance(p, dict):
            title = p.get('title', '')

        if hasattr(p, 'explanation'):
            explanation = p.explanation or ""
        elif isinstance(p, dict):
            explanation = p.get('explanation', '')

        if hasattr(p, 'category'):
            explicit_category = p.category or ""
        elif isinstance(p, dict):
            explicit_category = p.get('category', '')

        # Try explicit category first
        if explicit_category:
            cat = normalize_category(explicit_category)
            if cat != 'unknown':
                pred_categories.add(cat)
                continue
            elif strict:
                unknown_count += 1
                continue

        # Fall back to semantic matching from title/explanation
        cat = get_category_from_text(title, explanation)
        if cat != 'unknown':
            pred_categories.add(cat)
        elif strict:
            # Unknown predictions count as FP in strict mode
            unknown_count += 1

    # Compute matches
    matched = pred_categories & gold_categories
    spurious = pred_categories - gold_categories
    missed = gold_categories - pred_categories

    tp = len(matched)
    fp = len(spurious) + unknown_count  # Include unknown predictions as FP
    fn = len(missed)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return MetricsResult(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        matched_categories=list(matched),
        missed_categories=list(missed),
        spurious_categories=list(spurious),
        unknown_predictions=unknown_count,
    )
