from .base import LLMAdapter, LLMResponse

# Lazy imports to avoid requiring all dependencies
_OpenAIAdapter = None
_AnthropicAdapter = None

def _get_openai_adapter():
    global _OpenAIAdapter
    if _OpenAIAdapter is None:
        from .openai_adapter import OpenAIAdapter
        _OpenAIAdapter = OpenAIAdapter
    return _OpenAIAdapter

def _get_anthropic_adapter():
    global _AnthropicAdapter
    if _AnthropicAdapter is None:
        from .anthropic_adapter import AnthropicAdapter
        _AnthropicAdapter = AnthropicAdapter
    return _AnthropicAdapter


class LocalAdapter(LLMAdapter):
    """Placeholder adapter for local model providers (e.g., iAudit).

    This adapter is not used directly - specialized runners like run_iaudit_pipeline
    handle model loading and inference internally. This class exists to allow
    the experiment runner to proceed without errors when provider='local'.
    """
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.params = kwargs

    def call(self, prompt: str, **kwargs) -> LLMResponse:
        raise NotImplementedError(
            "LocalAdapter.call() should not be called directly. "
            "Use the specialized runner (e.g., run_iaudit_pipeline) instead."
        )


def make_adapter(provider: str, model_name: str, params: dict) -> LLMAdapter:
    """Factory function to create model adapters."""
    if provider == "openai":
        OpenAIAdapter = _get_openai_adapter()
        return OpenAIAdapter(model_name, **params)
    elif provider == "anthropic":
        AnthropicAdapter = _get_anthropic_adapter()
        return AnthropicAdapter(model_name, **params)
    elif provider == "local":
        return LocalAdapter(model_name, **params)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic, local")


# For backwards compatibility, expose classes but delay import errors
class _LazyAdapter:
    def __init__(self, name, getter):
        self._name = name
        self._getter = getter

    def __call__(self, *args, **kwargs):
        return self._getter()(*args, **kwargs)

    def __repr__(self):
        return f"<LazyAdapter {self._name}>"


OpenAIAdapter = _LazyAdapter("OpenAIAdapter", _get_openai_adapter)
AnthropicAdapter = _LazyAdapter("AnthropicAdapter", _get_anthropic_adapter)

__all__ = ["LLMAdapter", "LLMResponse", "OpenAIAdapter", "AnthropicAdapter", "make_adapter"]
