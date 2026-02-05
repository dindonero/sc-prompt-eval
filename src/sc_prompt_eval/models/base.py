from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class LLMResponse:
    text: str
    # Optional accounting for efficiency metrics
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_s: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None

class LLMAdapter:
    """Provider-agnostic interface.

    Implementations should be *pure* (no dataset logic): given prompt -> response.
    """
    def __init__(self, model_name: str, **params):
        self.model_name = model_name
        self.params = params

    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError
