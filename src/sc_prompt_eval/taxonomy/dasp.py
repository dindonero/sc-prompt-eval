"""DASP taxonomy helpers.

DASP (Decentralized Application Security Project) Top 10 is the canonical
taxonomy used by SmartBugs Curated dataset. This module provides helpers
for working with DASP categories.

Reference: https://dasp.co/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class DASPCategory:
    """DASP vulnerability category."""
    id: str
    name: str
    description: str


# DASP Top 10 categories (in order from DASP.co)
DASP_TAXONOMY: List[DASPCategory] = [
    DASPCategory(
        "reentrancy",
        "Reentrancy",
        "Exploitation of external calls to re-enter vulnerable functions"
    ),
    DASPCategory(
        "access_control",
        "Access Control",
        "Missing or weak authorization checks on sensitive functions"
    ),
    DASPCategory(
        "arithmetic",
        "Arithmetic Issues",
        "Integer overflow, underflow, and other arithmetic errors"
    ),
    DASPCategory(
        "unchecked_low_level_calls",
        "Unchecked Return Values",
        "Ignoring return values of low-level calls (send, call, delegatecall)"
    ),
    DASPCategory(
        "denial_of_service",
        "Denial of Service",
        "Attacks that prevent legitimate contract usage (gas limits, failed calls)"
    ),
    DASPCategory(
        "bad_randomness",
        "Bad Randomness",
        "Use of predictable values for random number generation"
    ),
    DASPCategory(
        "front_running",
        "Front Running",
        "Transaction order dependence and race conditions"
    ),
    DASPCategory(
        "time_manipulation",
        "Time Manipulation",
        "Dependence on block.timestamp for critical logic"
    ),
    DASPCategory(
        "short_addresses",
        "Short Address Attack",
        "Exploitation of incorrectly padded function arguments"
    ),
    DASPCategory(
        "other",
        "Other",
        "Miscellaneous vulnerabilities not in other categories"
    ),
]

# Quick lookup by category ID
DASP_BY_ID = {cat.id: cat for cat in DASP_TAXONOMY}

# Valid category IDs
DASP_IDS = {cat.id for cat in DASP_TAXONOMY}


