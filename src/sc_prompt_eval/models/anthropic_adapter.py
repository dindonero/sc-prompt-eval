from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMAdapter, LLMResponse

# Pricing per 1M tokens (as of Jan 2026 - Azure Microsoft Foundry)
# Note: Opus 4.5 had a 66% price reduction from legacy Opus 4.1
ANTHROPIC_PRICING = {
    # Azure Microsoft Foundry (Jan 2026) - updated pricing
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-4-1": {"input": 15.00, "output": 75.00},
    # Legacy model IDs (with corrected pricing)
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


class AnthropicAdapter(LLMAdapter):
    """Anthropic API adapter for Claude models."""

    def __init__(self, model_name: str, **params):
        super().__init__(model_name, **params)
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.temperature = params.get("temperature", 0.0)
        self.max_tokens = params.get("max_tokens", 2000)
        self.top_p = params.get("top_p", 1.0)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD based on token usage."""
        pricing = ANTHROPIC_PRICING.get(self.model_name, {"input": 15.0, "output": 75.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def generate(self, prompt: str) -> LLMResponse:
        """Generate response from Anthropic model."""
        start_time = time.time()

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            messages=[{"role": "user", "content": prompt}],
        )

        latency = time.time() - start_time

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        cost = self._calculate_cost(input_tokens, output_tokens)

        # Extract text from content blocks
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return LLMResponse(
            text=text,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_s=latency,
            raw=response.model_dump(),
        )
