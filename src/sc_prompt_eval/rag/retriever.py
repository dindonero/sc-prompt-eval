"""
CodeBERT-based RAG retriever for vulnerability patterns.

Implements SmartGuard-style retrieval using CodeBERT embeddings
to find semantically similar vulnerability patterns.

Reference: Ding et al. 2025 "SmartGuard" - RAG + CoT + Self-check

DATA LEAKAGE PREVENTION:
The RAG corpus (Qian et al. dataset, 4,290 patterns) is COMPLETELY SEPARATE
from the evaluation set (SmartBugs Curated, 143 contracts). These datasets
share no contracts, eliminating data leakage by design. No leave-one-out
mechanism is needed because there is no overlap to exclude.
"""

import json
import os
import tempfile
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Lazy load to avoid import overhead when module is imported but retriever not used
# Thread-safe singleton pattern with double-check locking
_model = None
_cosine_similarity = None
_model_lock = threading.Lock()


def _ensure_codebert():
    """Lazy load CodeBERT model and cosine similarity function. Thread-safe."""
    global _model, _cosine_similarity
    if _model is None:
        with _model_lock:
            # Double-check after acquiring lock
            if _model is None:
                from sentence_transformers import SentenceTransformer
                from sklearn.metrics.pairwise import cosine_similarity
                _model = SentenceTransformer('microsoft/codebert-base')
                _cosine_similarity = cosine_similarity
    return _model, _cosine_similarity


class CodeBERTRetriever:
    """
    CodeBERT-based retriever for vulnerability patterns.

    Matches SmartGuard methodology:
    1. Pre-compute 768-dim embeddings for all patterns (cached to disk)
    2. Encode query contract with CodeBERT
    3. Return top-k by cosine similarity
    """

    def __init__(self, patterns_path: Optional[Path] = None, cache_embeddings: bool = True):
        """Initialize retriever with patterns database.

        Args:
            patterns_path: Path to patterns_database.json. If None, uses default location.
            cache_embeddings: If True, cache embeddings to disk for faster subsequent runs.
        """
        if patterns_path is None:
            from ..paths import get_patterns_database_path
            patterns_path = get_patterns_database_path()

        self.patterns_path = Path(patterns_path)
        self.cache_embeddings = cache_embeddings
        self.patterns_db: Dict[str, List[Dict]] = {}
        self.all_patterns: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

        self._load_patterns()
        self._load_or_compute_embeddings()

    def _load_patterns(self):
        """Load patterns database from JSON file.

        Applies quality filters:
        1. Excludes patterns with truncated code (`...` markers)
        2. Strips NatSpec annotation symbols (`@param`, `@return`, etc.)
        """
        if self.patterns_path.exists():
            self.patterns_db = json.loads(self.patterns_path.read_text())

            truncated_count = 0
            cleaned_count = 0

            # Flatten for indexing with quality filtering
            for category, patterns in self.patterns_db.items():
                for pattern in patterns:
                    code = pattern.get('vulnerable_code', '')

                    # Filter 1: Skip truncated patterns (reduces fidelity as examples)
                    if '...' in code or '…' in code:
                        truncated_count += 1
                        continue

                    # Filter 2: Clean NatSpec annotations (prevent label leakage)
                    import re
                    if '@' in code:
                        # Remove NatSpec tags like @param, @return, @dev, @notice
                        original_code = code
                        code = re.sub(r'///\s*@\w+[^\n]*', '///', code)  # /// @param x -> ///
                        code = re.sub(r'\*\s*@\w+[^\n]*', '*', code)      # * @return -> *
                        if code != original_code:
                            cleaned_count += 1
                            pattern = {**pattern, 'vulnerable_code': code}

                    self.all_patterns.append({**pattern, 'category': category})

            if truncated_count > 0:
                print(f"Filtered out {truncated_count} truncated patterns (contained '...')")
            if cleaned_count > 0:
                print(f"Cleaned NatSpec annotations from {cleaned_count} patterns")
        else:
            print(f"Warning: Patterns database not found at {self.patterns_path}")

    def _get_cache_path(self) -> Path:
        """Get path for cached embeddings."""
        return self.patterns_path.parent / "pattern_embeddings_codebert.npy"

    def _load_or_compute_embeddings(self):
        """Load cached embeddings or compute them if not available."""
        if not self.all_patterns:
            return

        cache_path = self._get_cache_path()

        # Try loading cached embeddings
        if self.cache_embeddings and cache_path.exists():
            try:
                self.embeddings = np.load(cache_path)
                if len(self.embeddings) == len(self.all_patterns):
                    print(f"Loaded {len(self.embeddings)} cached CodeBERT embeddings")
                    return
                else:
                    print(f"Cache size mismatch ({len(self.embeddings)} vs {len(self.all_patterns)}), recomputing...")
            except Exception as e:
                print(f"Failed to load cached embeddings: {e}")

        # Compute embeddings
        model, _ = _ensure_codebert()
        codes = [p.get('vulnerable_code', '') for p in self.all_patterns]

        print(f"Computing CodeBERT embeddings for {len(codes)} patterns...")
        self.embeddings = model.encode(codes, show_progress_bar=True, batch_size=32)

        # Cache to disk (atomic write via temp file + rename)
        if self.cache_embeddings:
            try:
                # Write to temp file first, then atomic rename
                fd, tmp_path = tempfile.mkstemp(
                    dir=cache_path.parent,
                    suffix='.npy.tmp'
                )
                os.close(fd)  # np.save will open the file itself
                np.save(tmp_path, self.embeddings)
                os.replace(tmp_path, cache_path)  # Atomic on POSIX
                print(f"Cached embeddings to {cache_path}")
            except Exception as e:
                # Cleanup temp file on failure
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                print(f"Warning: Failed to cache embeddings: {e}")

    def retrieve_patterns(
        self,
        contract_code: str,
        num_patterns: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Retrieve most similar vulnerability patterns using CodeBERT cosine similarity.

        Args:
            contract_code: The Solidity code to analyze
            num_patterns: Number of patterns to retrieve (k in top-k)
            similarity_threshold: Minimum similarity score (0-1) to include pattern

        Returns:
            List of pattern dictionaries ranked by similarity, each containing:
            - category: DASP vulnerability category
            - title: Pattern title
            - description: Vulnerability description
            - vulnerable_code: Example vulnerable code
            - detection_hint: How to detect this pattern
            - similarity_score: Cosine similarity with target contract
        """
        if self.embeddings is None or len(self.all_patterns) == 0:
            return []

        model, cosine_similarity = _ensure_codebert()

        # Encode query contract
        query_embedding = model.encode([contract_code])

        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        sorted_indices = np.argsort(similarities)[::-1]

        # Collect top-k patterns
        retrieved = []
        for idx in sorted_indices:
            if len(retrieved) >= num_patterns:
                break

            pattern = self.all_patterns[idx]
            similarity = float(similarities[idx])

            # Skip if below threshold
            if similarity < similarity_threshold:
                continue

            retrieved.append({
                **pattern,
                'similarity_score': round(similarity, 4),
            })

        return retrieved

    def get_pattern_count(self) -> int:
        """Return total number of patterns in the index."""
        return len(self.all_patterns)

    def get_category_counts(self) -> Dict[str, int]:
        """Return count of patterns per category."""
        return {cat: len(patterns) for cat, patterns in self.patterns_db.items()}


# Singleton instance for convenience
# Thread-safe with double-check locking
_retriever: Optional[CodeBERTRetriever] = None
_retriever_lock = threading.Lock()


def get_retriever() -> CodeBERTRetriever:
    """Get or create the singleton retriever instance. Thread-safe."""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            # Double-check after acquiring lock
            if _retriever is None:
                _retriever = CodeBERTRetriever()
    return _retriever


def reset_retriever():
    """Reset the singleton retriever. Useful for testing."""
    global _retriever
    with _retriever_lock:
        _retriever = None


def retrieve_patterns(contract_code: str, num_patterns: int = 5, **kwargs) -> List[Dict]:
    """Convenience function to retrieve patterns using CodeBERT similarity."""
    return get_retriever().retrieve_patterns(contract_code, num_patterns, **kwargs)
