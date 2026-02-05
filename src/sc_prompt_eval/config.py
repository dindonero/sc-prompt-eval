from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class ModelSpec:
    name: str               # e.g., "gpt-4.1", "claude-3-5-sonnet", "llama3.1-70b"
    provider: str           # e.g., "openai", "anthropic", "google", "hf"
    params: Dict[str, Any]  # temperature, top_p, max_tokens, etc.

@dataclass
class PromptSpec:
    id: str                 # e.g., "p0_baseline"
    template_path: str      # prompts/*.j2
    description: str = ""

@dataclass
class DatasetSpec:
    name: str
    kind: str               # "smartbugs_curated", "solidifi", "ctf_suite", ...
    path: str

@dataclass
class ToolConfig:
    """Configuration for tool-augmented prompts (P3, P4, P5, P6)."""
    # P3 SmartGuard RAG settings (Zhang et al. 2024)
    top_k_patterns: int = 5  # Number of similar patterns to retrieve
    enable_cot_expansion: bool = True  # Full SmartGuard with CoT generation (default: enabled)

    # P3 True k-shot ICL and Iterative Self-Check (SmartGuard paper alignment)
    use_icl_template: bool = True  # Use p3_smartguard_icl.j2 (true k-shot format)
    enable_iterative_selfcheck: bool = False  # Multi-turn self-check after initial audit
    selfcheck_max_iterations: int = 3  # Max iterations for self-check convergence
    selfcheck_convergence_threshold: float = 0.9  # Jaccard similarity for stopping

    # P4 Slither/GPTScan settings (Sun et al. 2024)
    slither_timeout: int = 120  # Seconds before Slither times out
    enable_static_confirmation: bool = True  # Post-LLM static confirmation modules
    # P4 stage templates (GPTScan multi-stage pipeline)
    p4_stage1_template: str = "p4_stage1_scenario.j2"
    p4_stage2_template: str = "p4_stage2_property.j2"
    p4_stage3_template: str = "p4_stage3_extraction.j2"
    # GPTScan methodology alignment:
    # - "confirmation_only": Slither used only for post-LLM static confirmation (GPTScan paper)
    # - "pre_and_post": Slither used for both pre-LLM detection and post-LLM confirmation (legacy)
    # - "disabled": No Slither at all
    slither_mode: str = "confirmation_only"
    enable_reachability_filter: bool = True  # Filter candidates by reachability from entry points

    # P5 LLM-SmartAudit settings (Wei et al. 2024)
    # Exact implementation of multi-agent framework with BA/TA modes
    smartaudit_mode: str = "BA"  # "BA" (Broad Analysis) or "TA" (Targeted Analysis)
    smartaudit_max_rounds: int = 3  # Max consensus rounds (paper default = 3)
    smartaudit_consensus_threshold: float = 0.9  # Jaccard similarity for convergence
    smartaudit_enable_role_exchange: bool = True  # Role exchange per Figure 5b
    smartaudit_scenarios: Optional[List[str]] = None  # For TA mode: scenario IDs (None = all 40)
    smartaudit_scenario_path: Optional[str] = None  # Custom path to scenarios JSON

    # P6 iAudit: Fine-tuned pipeline (Ma et al., 2024)
    # Architecture: Detector → Reasoner → Ranker-Critic
    # Expected F1: ~91% (paper reported 91.21%)
    p6_iaudit_detector_model: str = ""  # Path to merged detector model (CodeLlama-13b fine-tuned)
    p6_iaudit_reasoner_model: str = ""  # Path to merged reasoner model (CodeLlama-13b fine-tuned)
    p6_iaudit_ranker_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Ranker-Critic model
    p6_iaudit_use_4bit: bool = True  # Use 4-bit quantization for memory efficiency
    p6_iaudit_max_ranker_rounds: int = 5  # Maximum Ranker-Critic debate rounds
    p6_iaudit_detector_prompts: int = 5  # Number of detector prompt variations
    p6_iaudit_reasoner_prompts: int = 10  # Number of reasoner paths (5 with call, 5 without)
    p6_iaudit_hf_token: str = ""  # HuggingFace token for model access


@dataclass
class ExperimentSpec:
    experiment_id: str
    random_seed: int
    runs_per_item: int
    prompts: List[PromptSpec]
    models: List[ModelSpec]
    datasets: List[DatasetSpec]
    output_dir: str
    notes: str = ""
    # Tool configuration (optional, uses defaults if not specified)
    tools: Optional[ToolConfig] = None
