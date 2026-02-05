"""
Benign contracts dataset for false positive measurement.

This dataset contains well-written Solidity contracts that use secure patterns.
They should NOT contain vulnerabilities, and any detections on these contracts
are false positives.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Optional

from .base import ContractItem, Dataset
from .smartbugs_curated import strip_annotations


class BenignContracts(Dataset):
    """Loader for benign (non-vulnerable) contracts for FP measurement.

    These contracts demonstrate secure Solidity patterns:
    - Checks-Effects-Interactions for reentrancy protection
    - Proper access control with msg.sender (not tx.origin)
    - Solidity 0.8+ for overflow protection
    - Pull-over-push payment pattern
    - Commit-reveal for randomness
    - Bounded loops with pagination
    - Return value checking for low-level calls

    Any vulnerability detections on these contracts are FALSE POSITIVES.
    """

    def __init__(self, root: str, name: str = "benign_contracts"):
        """Initialize benign contracts loader.

        Args:
            root: Path to benign contracts directory
            name: Dataset name for identification
        """
        super().__init__(name)
        self.root = Path(root)
        self._metadata: Optional[Dict] = None
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata about benign contracts."""
        metadata_file = self.root / "metadata.json"
        if metadata_file.exists():
            try:
                self._metadata = json.loads(metadata_file.read_text())
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load benign contracts metadata: {e}")
                self._metadata = None

    def _get_contract_info(self, filename: str) -> Dict:
        """Get metadata for a specific contract."""
        if self._metadata and "contracts" in self._metadata:
            for contract in self._metadata["contracts"]:
                if contract.get("name") == filename:
                    return contract
        return {}

    def iter_items(self) -> Iterable[ContractItem]:
        """Iterate over benign contracts.

        All contracts in this dataset should have empty ground truth labels
        since they are designed to be non-vulnerable.
        """
        for sol in self.root.glob("*.sol"):
            raw_source = sol.read_text(encoding="utf-8", errors="ignore")

            # Strip annotation markers for consistency with SmartBugs evaluation
            source = strip_annotations(raw_source)

            # Get contract info from metadata
            info = self._get_contract_info(sol.name)

            yield ContractItem(
                id=sol.name,
                source=source,
                filename=sol.name,
                labels=[],  # Empty - these are benign contracts
                metadata={
                    "path": str(sol),
                    "is_benign": True,
                    "safe_patterns": info.get("safe_patterns", []),
                    "expected_vulnerabilities": [],  # None expected
                    "description": info.get("description", ""),
                    "vulnerability_count": 0,
                },
            )

    def get_contract_count(self) -> int:
        """Return number of benign contracts."""
        return len(list(self.root.glob("*.sol")))

    def get_safe_patterns_summary(self) -> Dict[str, int]:
        """Return count of each safe pattern demonstrated."""
        pattern_counts: Dict[str, int] = {}

        if self._metadata and "contracts" in self._metadata:
            for contract in self._metadata["contracts"]:
                for pattern in contract.get("safe_patterns", []):
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        return pattern_counts
