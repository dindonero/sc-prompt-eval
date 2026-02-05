"""
Automatic Chain-of-Thought generation for retrieved demonstrations.

Implements SmartGuard's Demonstration Expansion stage (Ding et al. 2025):
1. Zero-shot CoT: "Let's think step by step" prompting
2. Self-check: Verify predicted label matches ground truth
3. Rethink loop: Up to N_t iterations if prediction is wrong
4. Label hint: If still wrong, provide true label and ask for reasoning
5. Confirmation: Final verification of the generated explanation

This generates CoT explanations for retrieved patterns BEFORE they are
used as in-context examples, improving the quality of demonstrations.
"""

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Type hint for adapter (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..adapters.base import LLMAdapter


@dataclass
class CoTResult:
    """Result of CoT generation with metadata."""
    reasoning: str           # Generated CoT explanation
    iterations: int          # How many rethink cycles needed
    used_true_label: bool    # Whether we had to provide the answer
    predicted_category: Optional[str]  # What the model predicted
    confidence: float        # Self-reported or inferred confidence
    # Token/cost accounting for efficiency metrics (accumulated across all iterations)
    total_tokens: int = 0    # Total tokens used across all API calls
    total_cost: float = 0.0  # Total cost in USD across all API calls
    total_latency: float = 0.0  # Total latency in seconds across all API calls


# Prompt templates for SmartGuard-style CoT generation
COT_ZERO_SHOT_PROMPT = """You are analyzing a Solidity smart contract code snippet for potential vulnerabilities.

**Code to analyze:**
```solidity
{code}
```

Let's think step by step about what vulnerabilities might exist in this code.

1. First, identify what the code does (purpose, state changes, external interactions)
2. Then, check for common vulnerability patterns (reentrancy, access control, arithmetic, etc.)
3. Finally, determine if any vulnerabilities are present

After your step-by-step analysis, provide your conclusion:

VERDICT: VULNERABLE or SAFE
CATEGORY: (if vulnerable, specify: reentrancy, access_control, arithmetic, unchecked_low_level_calls, denial_of_service, bad_randomness, front_running, time_manipulation, short_addresses, or other)
CONFIDENCE: (0-100%)
"""

RETHINK_PROMPT = """Your previous analysis may have missed something. Please rethink your analysis step by step.

Look more carefully at:
- External calls (call, send, transfer) and their ordering with state updates
- Access control (onlyOwner, require statements, tx.origin usage)
- Arithmetic operations on unsigned integers (overflow/underflow in pre-0.8)
- Return value checks on low-level calls
- Block properties used for critical logic (timestamp, blockhash)

Re-analyze the code and provide your updated conclusion:

VERDICT: VULNERABLE or SAFE
CATEGORY: (if vulnerable, specify the category)
CONFIDENCE: (0-100%)
"""

LABEL_HINT_PROMPT = """The correct classification for this code is: **{category}** vulnerability.

Please explain step by step WHY this code contains a {category} vulnerability.
Focus on:
1. The specific code pattern that creates the vulnerability
2. How an attacker could exploit it
3. What conditions make the exploit possible

Provide a clear, educational explanation that demonstrates understanding of this vulnerability type."""

CONFIRMATION_PROMPT = """Review your analysis one more time.

Is your explanation:
- Accurate in identifying the vulnerable code?
- Clear in explaining the attack vector?
- Complete in describing the conditions for exploitation?

Provide your final, refined explanation."""


class CoTGenerator:
    """
    Generate Chain-of-Thought explanations for vulnerability patterns.

    Implements SmartGuard's Demonstration Expansion stage with self-check
    algorithm to ensure high-quality CoT demonstrations.
    """

    def __init__(
        self,
        adapter: "LLMAdapter",
        max_rethink: int = 6,  # N_t from SmartGuard paper
        max_confirm: int = 1,  # N_c from SmartGuard paper
        cache_path: Optional[Path] = None,
    ):
        """
        Initialize CoT generator.

        Args:
            adapter: LLM adapter for generation
            max_rethink: Maximum rethink iterations (default 6 per paper)
            max_confirm: Maximum confirmation rounds (default 1 per paper)
            cache_path: Optional path to cache generated CoTs
        """
        self.adapter = adapter
        self.max_rethink = max_rethink
        self.max_confirm = max_confirm

        # Set up caching (thread-safe with lock)
        if cache_path is None:
            from ..paths import get_cot_cache_path
            cache_path = get_cot_cache_path()
        self.cache_path = cache_path
        self._cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._load_cache()

    def _load_cache(self):
        """Load cached CoTs from disk. Thread-safe."""
        with self._cache_lock:
            if self.cache_path.exists():
                try:
                    self._cache = json.loads(self.cache_path.read_text())
                except (json.JSONDecodeError, IOError):
                    self._cache = {}

    def _save_cache(self):
        """Save cache to disk. Thread-safe with atomic write."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Take a snapshot of cache under lock
            with self._cache_lock:
                cache_snapshot = dict(self._cache)

            # Write to temp file first, then atomic rename
            fd, tmp_path = tempfile.mkstemp(
                dir=self.cache_path.parent,
                suffix='.json.tmp'
            )
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(cache_snapshot, f, indent=2)
                os.replace(tmp_path, self.cache_path)  # Atomic on POSIX
            except Exception:
                # Cleanup temp file on failure
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except IOError:
            pass  # Silently fail cache writes

    def _get_cache_key(self, code: str, category: str) -> str:
        """Generate cache key from model name, code and category.

        Includes model_name to prevent cross-model cache pollution,
        since different models produce different CoT reasoning.
        """
        import hashlib
        model_name = getattr(self.adapter, 'model_name', 'unknown')
        content = f"{model_name}:{category}:{code}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[str]:
        """Thread-safe cache read."""
        with self._cache_lock:
            return self._cache.get(key)

    def _set_in_cache(self, key: str, value: str):
        """Thread-safe cache write."""
        with self._cache_lock:
            self._cache[key] = value

    def _remove_from_cache(self, key: str):
        """Thread-safe cache removal."""
        with self._cache_lock:
            if key in self._cache:
                del self._cache[key]

    def generate_cot(
        self,
        code: str,
        true_category: str,
        use_cache: bool = True,
    ) -> CoTResult:
        """
        Generate CoT explanation using SmartGuard's self-check algorithm.

        Args:
            code: Solidity code snippet
            true_category: The actual vulnerability category (ground truth)
            use_cache: Whether to use cached results

        Returns:
            CoTResult with generated reasoning and metadata
        """
        # Check cache first (thread-safe)
        cache_key = self._get_cache_key(code, true_category)
        TRUNCATION_MARKER = "[Output truncated - consider increasing max_tokens]"
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                # Skip cached entries that were truncated (may have been cached by older version)
                if TRUNCATION_MARKER not in cached:
                    return CoTResult(
                        reasoning=cached,
                        iterations=0,  # Cached, no iterations
                        used_true_label=False,
                        predicted_category=true_category,
                        confidence=0.9,
                        total_tokens=0,  # No API calls for cached results
                        total_cost=0.0,
                        total_latency=0.0,
                    )
                else:
                    # Truncated cache entry found - remove it and regenerate
                    import logging
                    logging.getLogger(__name__).info(
                        f"Removing truncated cache entry for category '{true_category}'"
                    )
                    self._remove_from_cache(cache_key)
                    self._save_cache()

        # Track cumulative token/cost/latency across all API calls
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0.0

        def _accumulate_response(resp):
            """Helper to accumulate response metrics."""
            nonlocal total_tokens, total_cost, total_latency
            total_tokens += resp.total_tokens or 0
            total_cost += resp.cost_usd or 0.0
            total_latency += resp.latency_s or 0.0

        # Step 1: Zero-shot CoT generation
        prompt = COT_ZERO_SHOT_PROMPT.format(code=code)
        response = self.adapter.generate(prompt)
        _accumulate_response(response)
        current_text = response.text

        predicted = self._extract_category(current_text)
        iterations = 1
        used_true_label = False

        # Step 2: Self-check rethink loop
        if not self._categories_match(predicted, true_category):
            for i in range(self.max_rethink):
                rethink_prompt = f"{current_text}\n\n{RETHINK_PROMPT}"
                rethink_response = self.adapter.generate(rethink_prompt)
                _accumulate_response(rethink_response)
                current_text = rethink_response.text
                predicted = self._extract_category(current_text)
                iterations += 1

                if self._categories_match(predicted, true_category):
                    break

        # Step 3: If still wrong after rethinking, provide true label
        if not self._categories_match(predicted, true_category):
            used_true_label = True
            hint_prompt = LABEL_HINT_PROMPT.format(category=true_category)
            full_prompt = f"Code:\n```solidity\n{code}\n```\n\n{hint_prompt}"
            hint_response = self.adapter.generate(full_prompt)
            _accumulate_response(hint_response)
            current_text = hint_response.text
            iterations += 1

        # Step 4: Final confirmation
        for _ in range(self.max_confirm):
            confirm_prompt = f"{current_text}\n\n{CONFIRMATION_PROMPT}"
            confirm_response = self.adapter.generate(confirm_prompt)
            _accumulate_response(confirm_response)
            current_text = confirm_response.text
            iterations += 1

        # Extract final reasoning (clean up for use as demonstration)
        final_reasoning = self._clean_reasoning(current_text, true_category)

        # Check for truncation - don't cache truncated responses
        # The truncation marker is added by OpenAI adapter when finish_reason == "length"
        TRUNCATION_MARKER = "[Output truncated - consider increasing max_tokens]"
        is_truncated = TRUNCATION_MARKER in final_reasoning

        # Cache the result only if not truncated (thread-safe)
        if use_cache and not is_truncated:
            self._set_in_cache(cache_key, final_reasoning)
            self._save_cache()
        elif is_truncated:
            import logging
            logging.getLogger(__name__).warning(
                f"CoT generation truncated for category '{true_category}' - "
                f"response not cached. Consider increasing max_tokens."
            )

        return CoTResult(
            reasoning=final_reasoning,
            iterations=iterations,
            used_true_label=used_true_label,
            predicted_category=predicted,
            confidence=0.9 if not used_true_label else 0.7,
            total_tokens=total_tokens,
            total_cost=total_cost,
            total_latency=total_latency,
        )

    def generate_cot_batch(
        self,
        patterns: List[Dict],
        use_cache: bool = True,
    ) -> List[Dict]:
        """
        Generate CoTs for a batch of patterns.

        Args:
            patterns: List of pattern dicts with 'vulnerable_code' and 'category'
            use_cache: Whether to use cached results

        Returns:
            List of patterns with 'chain_of_thought' field added
        """
        results = []
        for pattern in patterns:
            code = pattern.get('vulnerable_code', '')
            category = pattern.get('category', 'other')

            cot_result = self.generate_cot(code, category, use_cache=use_cache)

            results.append({
                **pattern,
                'chain_of_thought': cot_result.reasoning,
                'cot_metadata': {
                    'iterations': cot_result.iterations,
                    'used_true_label': cot_result.used_true_label,
                    'confidence': cot_result.confidence,
                }
            })

        return results

    def _extract_category(self, text: str) -> Optional[str]:
        """Extract predicted category from LLM response."""
        # Try to find CATEGORY: pattern
        match = re.search(r'CATEGORY:\s*(\w+)', text, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # Try common category keywords
        text_lower = text.lower()
        categories = [
            'reentrancy', 'access_control', 'arithmetic',
            'unchecked_low_level_calls', 'denial_of_service',
            'bad_randomness', 'front_running', 'time_manipulation',
            'short_addresses', 'other'
        ]
        for cat in categories:
            if cat in text_lower:
                return cat

        return None

    def _categories_match(self, predicted: Optional[str], true: str) -> bool:
        """Check if predicted category matches true category."""
        if not predicted:
            return False

        # Normalize both
        predicted = predicted.lower().strip()
        true = true.lower().strip()

        # Direct match
        if predicted == true:
            return True

        # Handle common variations (LLM outputs vary)
        aliases = {
            # Arithmetic variations
            'overflow': 'arithmetic',
            'underflow': 'arithmetic',
            'integer_overflow': 'arithmetic',
            'integer_underflow': 'arithmetic',
            'integer': 'arithmetic',  # Common LLM shorthand
            # Reentrancy variations
            're-entrancy': 'reentrancy',
            'reentrance': 'reentrancy',
            're-entrance': 'reentrancy',
            # Unchecked calls variations
            'unchecked_call': 'unchecked_low_level_calls',
            'unchecked_return': 'unchecked_low_level_calls',
            'unchecked': 'unchecked_low_level_calls',
            # DoS variations
            'dos': 'denial_of_service',
            'denial': 'denial_of_service',
            # Time manipulation variations
            'timestamp': 'time_manipulation',
            'block_timestamp': 'time_manipulation',
            'time': 'time_manipulation',
            # Randomness variations
            'randomness': 'bad_randomness',
            'random': 'bad_randomness',
            # Front-running variations
            'frontrunning': 'front_running',
            'front-running': 'front_running',
            'frontrun': 'front_running',
        }

        return aliases.get(predicted, predicted) == aliases.get(true, true)

    def _clean_reasoning(self, text: str, category: str) -> str:
        """Clean up reasoning text for use as demonstration."""
        # Remove VERDICT/CATEGORY/CONFIDENCE lines (we already know the answer)
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Skip meta lines
            if re.match(r'^(VERDICT|CATEGORY|CONFIDENCE):', line.strip(), re.IGNORECASE):
                continue
            cleaned_lines.append(line)

        result = '\n'.join(cleaned_lines).strip()

        # Ensure it mentions the category
        if category.lower() not in result.lower():
            result = f"This code contains a {category} vulnerability.\n\n{result}"

        return result


def get_cot_generator(adapter: "LLMAdapter", **kwargs) -> CoTGenerator:
    """Factory function to create a CoT generator."""
    return CoTGenerator(adapter, **kwargs)
