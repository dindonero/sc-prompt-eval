from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Dict, Optional

from .base import ContractItem, Dataset

# =============================================================================
# DASP Top 10 Taxonomy (native to SmartBugs Curated)
# https://dasp.co/
# =============================================================================

# Valid DASP categories in SmartBugs Curated dataset
DASP_CATEGORIES: List[str] = [
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

# Human-readable names for DASP categories
DASP_NAMES: Dict[str, str] = {
    "reentrancy": "Reentrancy",
    "access_control": "Access Control",
    "arithmetic": "Arithmetic (Integer Overflow/Underflow)",
    "unchecked_low_level_calls": "Unchecked Return Values",
    "denial_of_service": "Denial of Service",
    "bad_randomness": "Bad Randomness",
    "front_running": "Front Running",
    "time_manipulation": "Time Manipulation",
    "short_addresses": "Short Address Attack",
    "other": "Other",
}


# =============================================================================
# Annotation Stripping (Critical for evaluation validity)
# =============================================================================

# Patterns for SmartBugs annotation comments that leak ground truth labels
ANNOTATION_PATTERNS = [
    # Multi-line comment annotations: /* @vulnerable_at_lines: 18,20 */
    re.compile(r'/\*[^*]*@vulnerable_at_lines[^*]*\*/', re.MULTILINE),
    # Single line @vulnerable_at_lines in block comments
    re.compile(r'\*\s*@vulnerable_at_lines.*$', re.MULTILINE),
    # Inline vulnerability markers: // <yes> <report> REENTRANCY
    re.compile(r'//\s*<yes>\s*<report>\s*\w+.*$', re.MULTILINE),
    # Alternative marker format: // <yes> REENTRANCY
    re.compile(r'//\s*<yes>\s+[A-Z_]+.*$', re.MULTILINE),
    # Source attribution that may hint at vulnerability type
    re.compile(r'\*\s*@vulnerable_at_lines:.*$', re.MULTILINE),
    # SWC Registry URLs leak vulnerability classification (SWC-101, SWC-104, etc.)
    re.compile(r'\*\s*@source:.*SWC-\d+.*$', re.MULTILINE),
    re.compile(r'\*\s*@source:.*swc-registry.*$', re.MULTILINE | re.IGNORECASE),
    # Alternative: remove entire @source lines with SmartContractSecurity references
    re.compile(r'\*\s*@source:.*smartcontractsecurity.*$', re.MULTILINE | re.IGNORECASE),
]


def strip_annotations(source: str) -> str:
    """Remove SmartBugs ground-truth annotations from contract source.

    This is CRITICAL for evaluation validity - without stripping, LLMs can
    trivially 'detect' vulnerabilities by reading the annotation comments.

    This function preserves line numbers (unlike strip_all_comments) which is
    essential for accurate instance-level metric matching with ground truth.

    Removes:
    - @vulnerable_at_lines comments
    - // <yes> <report> CATEGORY markers
    - SWC registry URLs that leak vulnerability classification
    - Other ground-truth leaking annotations
    """
    result = source
    for pattern in ANNOTATION_PATTERNS:
        result = pattern.sub('', result)

    # Clean up any resulting empty comment blocks
    result = re.sub(r'/\*\s*\*/', '', result)
    # Clean up multiple blank lines left behind
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)

    return result


def strip_all_comments(source: str) -> str:
    """Remove ALL comments from Solidity source code.

    This is the preferred method for evaluation to prevent any label leakage.
    Removes all single-line (//) and multi-line (/* */) comments, including
    NatSpec documentation (/// and /** */).

    Args:
        source: Solidity source code

    Returns:
        Source code with all comments removed
    """
    # Remove multi-line comments (non-greedy to handle nested-looking patterns)
    result = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    # Remove single-line comments (including /// NatSpec)
    result = re.sub(r'//.*$', '', result, flags=re.MULTILINE)
    # Clean up excessive blank lines
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
    return result


class SmartBugsCurated(Dataset):
    """Loader for SmartBugs Curated dataset with DASP taxonomy.

    SmartBugs Curated uses the DASP (Decentralized Application Security Project)
    Top 10 taxonomy natively.

    Ground truth sources (in priority order):
    1. vulnerabilities.json - Line-level annotations with multi-instance support
    2. Folder structure - Category-level only (fallback)

    Expected layout:
      root/
        vulnerabilities.json   # Detailed annotations (optional but recommended)
        dataset/
          <category>/
            <contract>.sol

    CRITICAL: This loader strips ground-truth annotation comments from source
    code before returning. Without this, LLMs could trivially "detect"
    vulnerabilities by reading the // <yes> <report> markers.
    """

    def __init__(self, root: str, name: str = "smartbugs_curated"):
        """Initialize SmartBugs Curated dataset loader.

        Args:
            root: Path to dataset root directory
            name: Dataset name for identification
        """
        super().__init__(name)
        self.root = Path(root)
        self._vuln_db: Optional[Dict[str, List[Dict]]] = None
        self._load_vulnerabilities_json()

    def _load_vulnerabilities_json(self) -> None:
        """Load vulnerabilities.json for line-level ground truth.

        This provides:
        - Line numbers for each vulnerability
        - Multiple vulnerability instances per contract
        - More accurate evaluation than folder-based labels
        """
        vuln_file = self.root / "vulnerabilities.json"
        if not vuln_file.exists():
            # Try alternate location: <root>/dataset/vulnerabilities.json
            vuln_file = self.root / "dataset" / "vulnerabilities.json"

        if not vuln_file.exists():
            # If root ends with /dataset, try parent directory
            # This handles config paths like "data/smartbugs_curated/dataset"
            if self.root.name == "dataset":
                vuln_file = self.root.parent / "vulnerabilities.json"

        if vuln_file.exists():
            try:
                vuln_list = json.loads(vuln_file.read_text())
                # Index by path for fast lookup
                self._vuln_db = {}
                for entry in vuln_list:
                    path = entry.get("path", "")
                    # Normalize path (remove leading dataset/ if present)
                    if path.startswith("dataset/"):
                        path = path[8:]  # Remove "dataset/" prefix
                    self._vuln_db[path] = entry.get("vulnerabilities", [])
                    # Also store with full path for matching
                    self._vuln_db[entry.get("path", "")] = entry.get("vulnerabilities", [])
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load vulnerabilities.json: {e}")
                self._vuln_db = None

    def _get_ground_truth(self, relative_path: str, folder_category: str) -> List[Dict]:
        """Get ground truth labels, preferring vulnerabilities.json over folder.

        Args:
            relative_path: Path relative to dataset root
            folder_category: Category inferred from folder structure

        Returns:
            List of ground truth labels with category and optionally lines
        """
        # Try vulnerabilities.json first (line-level, multi-instance)
        if self._vuln_db is not None:
            # Try various path formats
            path_variants = [
                relative_path,
                str(relative_path),
                f"dataset/{relative_path}",
                relative_path.replace("dataset/", ""),
            ]

            for path_key in path_variants:
                if path_key in self._vuln_db:
                    vulns = self._vuln_db[path_key]
                    if vulns:
                        return [
                            {
                                "category": v.get("category", folder_category).lower(),
                                "lines": v.get("lines", []),
                                "title": DASP_NAMES.get(
                                    v.get("category", folder_category).lower(),
                                    v.get("category", folder_category)
                                ),
                            }
                            for v in vulns
                        ]

        # Fallback to folder-based (category-only, single instance)
        category_name = DASP_NAMES.get(folder_category, folder_category.replace("_", " ").title())
        return [{
            "category": folder_category,
            "title": category_name,
            "lines": [],  # No line info from folder
        }]

    def iter_items(self) -> Iterable[ContractItem]:
        for sol in self.root.rglob("*.sol"):
            raw_source = sol.read_text(encoding="utf-8", errors="ignore")

            # CRITICAL: Strip annotation markers to prevent label leakage
            # Uses strip_annotations() (not strip_all_comments) to preserve line numbers
            # for accurate instance-level metric matching with ground truth
            source = strip_annotations(raw_source)

            # Extract category from path (parent folder name)
            relative_path = sol.relative_to(self.root)

            # Handle both direct layout and dataset/ subdirectory layout
            parts = relative_path.parts
            if parts and parts[0] == "dataset":
                folder_category = parts[1].lower() if len(parts) > 1 else "unknown"
            else:
                folder_category = parts[0].lower() if parts else "unknown"

            # Validate against known DASP categories
            if folder_category not in DASP_CATEGORIES:
                folder_category = "other"

            # Get ground truth (prefers vulnerabilities.json, falls back to folder)
            ground_truth = self._get_ground_truth(str(relative_path), folder_category)

            # Determine if we have line-level annotations
            has_line_annotations = any(gt.get("lines") for gt in ground_truth)

            yield ContractItem(
                id=str(relative_path),
                source=source,
                filename=sol.name,
                labels=ground_truth,
                metadata={
                    "path": str(sol),
                    "category": folder_category,
                    "annotation_markers_stripped": True,  # Leakage markers removed, general comments preserved
                    "has_line_annotations": has_line_annotations,
                    "vulnerability_count": len(ground_truth),
                },
            )
