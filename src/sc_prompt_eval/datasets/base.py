from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class ContractItem:
    """A single contract from a dataset."""
    id: str
    source: str
    filename: str = "Contract.sol"
    # ground truth fields (optional; depends on dataset)
    labels: Optional[List[dict]] = None  # list of findings, ideally in the same schema
    metadata: Optional[dict] = None


class Dataset(ABC):
    """Abstract base class for datasets."""

    def __init__(self, name: str = "dataset"):
        self.name = name

    @abstractmethod
    def iter_items(self) -> Iterable[ContractItem]:
        """Iterate over all items in the dataset."""
        raise NotImplementedError
