#!/usr/bin/env python3
"""
Build patterns_database.json from Qian et al. dataset.

Source: Qian et al. "Smart Contract Vulnerability Detection using Deep Learning"
Dataset: 12,515 real-world contracts from Etherscan with 8 vulnerability types.

This is the same corpus SmartGuard (Ding et al. 2025) used for their RAG system.
No train/test split needed - this corpus is used exclusively for RAG retrieval,
while SmartBugs Curated is used for evaluation (completely separate datasets).

DASP Mapping:
- reentrancy (RE) → reentrancy
- unchecked external call (UC) → unchecked_low_level_calls
- integer overflow (OF) → arithmetic
- block number dependency (BN) → bad_randomness
- ether strict equality (SE) → denial_of_service
- timestamp dependency (TP) → time_manipulation
- dangerous delegatecall (DE) → access_control
- ether frozen (EF) → denial_of_service
"""

import json
from pathlib import Path
from typing import Dict, List


# =============================================================================
# Qian to DASP Category Mapping
# =============================================================================

CATEGORY_MAP = {
    "reentrancy (RE)": "reentrancy",
    "unchecked external call (UC)": "unchecked_low_level_calls",
    "integer overflow (OF)": "arithmetic",
    "block number dependency (BN)": "bad_randomness",
    "ether strict equality (SE)": "denial_of_service",
    "timestamp dependency (TP)": "time_manipulation",
    "dangerous delegatecall (DE)": "access_control",
    "ether frozen (EF)": "denial_of_service",
}

# Human-readable titles for DASP categories
CATEGORY_TITLES = {
    'reentrancy': 'Reentrancy Vulnerability',
    'access_control': 'Access Control Weakness',
    'arithmetic': 'Integer Overflow/Underflow',
    'unchecked_low_level_calls': 'Unchecked Return Value',
    'denial_of_service': 'Denial of Service',
    'bad_randomness': 'Weak Randomness',
    'front_running': 'Front-Running / Transaction Order Dependence',
    'time_manipulation': 'Timestamp Dependence',
    'short_addresses': 'Short Address Attack',
    'other': 'Other Vulnerability',
}

# Detection hints for each category
DETECTION_HINTS = {
    'reentrancy': 'Look for external calls (call, send, transfer) before state updates',
    'access_control': 'Check for missing modifiers, tx.origin usage, or unprotected functions',
    'arithmetic': 'Look for unchecked math operations on uint types without SafeMath',
    'unchecked_low_level_calls': 'Check if return values of call/send/delegatecall are verified',
    'denial_of_service': 'Look for unbounded loops, failed call dependencies, or gas limits',
    'bad_randomness': 'Check for block.timestamp, blockhash, or block.number used for randomness',
    'front_running': 'Look for transaction-order dependent logic or price manipulation risks',
    'time_manipulation': 'Check for critical logic depending on block.timestamp or now',
    'short_addresses': 'Look for missing input validation on address parameters',
    'other': 'General code quality or logic issues',
}

# Qian-specific descriptions for each vulnerability type
QIAN_DESCRIPTIONS = {
    "reentrancy (RE)": "Reentrancy vulnerability allowing multiple withdrawals before balance update",
    "unchecked external call (UC)": "External call return value not checked, may silently fail",
    "integer overflow (OF)": "Arithmetic operation may overflow or underflow without SafeMath",
    "block number dependency (BN)": "Contract uses block.number for randomness, predictable by miners",
    "ether strict equality (SE)": "Strict balance equality check can be manipulated via selfdestruct",
    "timestamp dependency (TP)": "Contract logic depends on block.timestamp, manipulable by miners",
    "dangerous delegatecall (DE)": "Delegatecall to user-controlled address may allow storage hijacking",
    "ether frozen (EF)": "Ether may become permanently locked in contract",
}


def add_smartbugs_patterns(
    smartbugs_dir: Path,
    patterns_by_category: Dict[str, List[Dict]],
    categories: List[str]
) -> None:
    """Add patterns from SmartBugs for missing categories.

    Only adds patterns for specified categories to minimize test set contamination.

    Args:
        smartbugs_dir: Path to SmartBugs dataset
        patterns_by_category: Dict to add patterns to
        categories: List of category names to include (e.g., ['front_running', 'short_addresses'])
    """
    for category in categories:
        category_dir = smartbugs_dir / category
        if not category_dir.exists():
            print(f"Warning: SmartBugs category not found: {category_dir}")
            continue

        if category not in patterns_by_category:
            patterns_by_category[category] = []

        sol_files = list(category_dir.glob("*.sol"))
        print(f"Adding SmartBugs {category}: {len(sol_files)} contracts (for RAG coverage)")

        for sol_file in sol_files:
            try:
                code = sol_file.read_text(encoding='utf-8', errors='replace')
                truncated = code[:4000] if len(code) > 4000 else code

                pattern = {
                    'category': category,
                    'title': CATEGORY_TITLES.get(category, 'Unknown Vulnerability'),
                    'description': f"Vulnerable code example from {sol_file.name}",
                    'vulnerable_code': truncated,
                    'detection_hint': DETECTION_HINTS.get(category, ''),
                    'source_contract': sol_file.name,
                    'source_dataset': 'smartbugs_curated',
                }

                patterns_by_category[category].append(pattern)

            except Exception as e:
                print(f"  Warning: Failed to read {sol_file.name}: {e}")


def build_qian_patterns(qian_dir: Path, output_path: Path, smartbugs_dir: Path = None) -> Dict[str, List[Dict]]:
    """Build pattern database from Qian et al. dataset.

    Args:
        qian_dir: Path to Qian dataset (data/qian_dataset/Dataset_1/Dataset/)
        output_path: Path for output JSON file
        smartbugs_dir: Optional path to SmartBugs for missing categories

    Returns:
        Dict mapping DASP categories to lists of pattern entries
    """
    patterns_by_category: Dict[str, List[Dict]] = {}

    print(f"Building patterns from: {qian_dir}")
    print("=" * 60)

    for qian_category in CATEGORY_MAP.keys():
        category_dir = qian_dir / qian_category
        if not category_dir.exists():
            print(f"Warning: Category directory not found: {category_dir}")
            continue

        dasp_category = CATEGORY_MAP[qian_category]
        if dasp_category not in patterns_by_category:
            patterns_by_category[dasp_category] = []

        # Process all .sol files in category
        sol_files = list(category_dir.glob("*.sol"))
        print(f"Processing {qian_category}: {len(sol_files)} contracts -> {dasp_category}")

        for sol_file in sol_files:
            try:
                code = sol_file.read_text(encoding='utf-8', errors='replace')

                # Truncate very long contracts (keep first 4000 chars for TF-IDF)
                # This is enough for similarity matching while keeping database size manageable
                truncated = code[:4000] if len(code) > 4000 else code

                pattern = {
                    'category': dasp_category,
                    'title': CATEGORY_TITLES.get(dasp_category, 'Unknown Vulnerability'),
                    'description': QIAN_DESCRIPTIONS.get(qian_category, f"Vulnerable code from {sol_file.name}"),
                    'vulnerable_code': truncated,
                    'detection_hint': DETECTION_HINTS.get(dasp_category, ''),
                    'source_contract': sol_file.name,
                    'source_dataset': 'qian_et_al',
                    'original_category': qian_category,
                }

                patterns_by_category[dasp_category].append(pattern)

            except Exception as e:
                print(f"  Warning: Failed to read {sol_file.name}: {e}")

    # Add missing categories from SmartBugs (minimal contamination)
    if smartbugs_dir and smartbugs_dir.exists():
        print("\n" + "-" * 60)
        print("Adding missing categories from SmartBugs (for RAG coverage):")
        add_smartbugs_patterns(
            smartbugs_dir,
            patterns_by_category,
            categories=['front_running', 'short_addresses']
        )

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(patterns_by_category, indent=2))

    # Print summary
    print("\n" + "=" * 60)
    print(f"Pattern database built: {output_path}")
    print(f"\nPatterns by DASP category:")

    total = 0
    for cat in sorted(patterns_by_category.keys()):
        count = len(patterns_by_category[cat])
        total += count
        print(f"  {cat}: {count} patterns")

    print(f"\nTotal: {total} patterns")
    print(f"Note: No train/test split - separate corpus from test set (SmartBugs)")

    return patterns_by_category


if __name__ == "__main__":
    import sys

    # Default paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    default_qian_dir = project_root / "data" / "qian_dataset" / "Dataset_1" / "Dataset"
    default_smartbugs_dir = project_root / "data" / "smartbugs_curated" / "dataset"
    default_output = project_root / "data" / "patterns_database.json"

    # Allow custom paths as arguments
    if len(sys.argv) > 1:
        qian_dir = Path(sys.argv[1])
    else:
        qian_dir = default_qian_dir

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = default_output

    if not qian_dir.exists():
        print(f"Error: Qian dataset directory not found: {qian_dir}")
        print("\nExpected structure:")
        print("  data/qian_dataset/Dataset_1/Dataset/")
        print("    ├── reentrancy (RE)/")
        print("    ├── unchecked external call (UC)/")
        print("    ├── integer overflow (OF)/")
        print("    ├── ...")
        print("\nUsage: python build_qian_patterns.py [qian_dir] [output_path]")
        sys.exit(1)

    # Use SmartBugs for missing categories if available
    smartbugs_dir = default_smartbugs_dir if default_smartbugs_dir.exists() else None

    build_qian_patterns(qian_dir, output_path, smartbugs_dir)
