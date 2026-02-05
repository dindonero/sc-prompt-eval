from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMAdapter, LLMResponse

# Pricing per 1M tokens (as of Jan 2026 - Azure OpenAI)
OPENAI_PRICING = {
    # GPT-5 Series (Azure 2026)
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-chat": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-pro": {"input": 15.00, "output": 120.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5.1-chat": {"input": 1.25, "output": 10.00},
    "gpt-5.1-codex": {"input": 2.00, "output": 15.00},
    "gpt-5.1-codex-max": {"input": 5.00, "output": 40.00},
    "gpt-5.1-codex-mini": {"input": 0.50, "output": 4.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.2-chat": {"input": 1.75, "output": 14.00},
    "gpt-5-codex": {"input": 2.00, "output": 15.00},
    "codex-mini": {"input": 0.50, "output": 4.00},
    # O-Series Reasoning Models (Azure 2026)
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o3": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # GPT-OSS (Azure - estimates)
    "gpt-oss-120b": {"input": 1.00, "output": 4.00},
    "gpt-oss-20b": {"input": 0.20, "output": 0.80},
    # Legacy models (backwards compatibility)
    "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "chatgpt-4o-latest": {"input": 5.00, "output": 15.00},
}


class OpenAIAdapter(LLMAdapter):
    """OpenAI API adapter for GPT models (also supports Azure OpenAI)."""

    # Models that require max_completion_tokens instead of max_tokens
    # and don't support temperature/top_p (o-series reasoning models + gpt-5-nano)
    O_SERIES_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini", "gpt-5-nano"}

    # Models that require max_completion_tokens but DO support temperature/top_p
    MAX_COMPLETION_TOKENS_MODELS = {"gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-5-pro", "gpt-5.1", "gpt-5.2"}

    def __init__(self, model_name: str, **params):
        super().__init__(model_name, **params)

        # Support Azure OpenAI or custom endpoints via base_url
        api_key_env = params.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        base_url = params.get("base_url")

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        self.temperature = params.get("temperature", 0.0)
        self.max_tokens = params.get("max_tokens", 2000)
        self.top_p = params.get("top_p", 1.0)

        # Check if this is an o-series model (reasoning models - no temperature/top_p)
        self.is_o_series = any(model_name.startswith(prefix) for prefix in self.O_SERIES_MODELS)

        # Check if this model requires max_completion_tokens instead of max_tokens
        self.uses_max_completion_tokens = self.is_o_series or any(
            model_name.startswith(prefix) for prefix in self.MAX_COMPLETION_TOKENS_MODELS
        )

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD based on token usage."""
        pricing = OPENAI_PRICING.get(self.model_name, {"input": 10.0, "output": 30.0})
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def generate(self, prompt: str) -> LLMResponse:
        """Generate response from OpenAI model."""
        start_time = time.time()

        # Different parameter handling for different model types:
        # - O-series (o1, o3, o4-mini): max_completion_tokens, NO temperature/top_p
        # - GPT-5 series: max_completion_tokens, WITH temperature/top_p
        # - Legacy (GPT-4, etc.): max_tokens, WITH temperature/top_p
        if self.is_o_series:
            # O-series: no temperature/top_p support
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_tokens,
            )
        elif self.uses_max_completion_tokens:
            # GPT-5 series: use max_completion_tokens but supports temperature/top_p
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                top_p=self.top_p,
            )
        else:
            # Legacy models: use max_tokens
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )

        latency = time.time() - start_time

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        cost = self._calculate_cost(prompt_tokens, completion_tokens)

        # Handle response content - o-series models may return empty content if truncated
        content = response.choices[0].message.content
        if content is None:
            content = ""

        # Check for truncation (finish_reason != 'stop')
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length" and not content:
            # Model hit token limit before producing output - likely still reasoning
            content = "[Output truncated - consider increasing max_tokens]"

        return LLMResponse(
            text=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_s=latency,
            raw=response.model_dump(),
        )
