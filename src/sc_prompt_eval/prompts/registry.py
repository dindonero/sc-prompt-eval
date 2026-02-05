from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from jinja2 import Environment, FileSystemLoader, StrictUndefined

@dataclass(frozen=True)
class RenderedPrompt:
    prompt_id: str
    text: str

class PromptRegistry:
    def __init__(self, prompt_root: str | Path):
        self.prompt_root = Path(prompt_root)
        self.env = Environment(
            loader=FileSystemLoader(str(self.prompt_root)),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_path: str, *, contract_source: str, **kwargs) -> str:
        tmpl = self.env.get_template(template_path)
        return tmpl.render(contract_source=contract_source, **kwargs)
