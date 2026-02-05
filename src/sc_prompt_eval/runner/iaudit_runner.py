"""
iAudit Integration Module for sc_prompt_eval

Integrates the full iAudit pipeline (Ma et al., 2024) into the prompt evaluation framework.
Architecture: Detector (fine-tuned) → Reasoner (fine-tuned) → Ranker-Critic (Mixtral-8x7B)

Expected performance: ~91% F1 (paper reported 91.21%)

Usage:
    Configure in YAML with:
    - iaudit_detector_model: path to merged detector model
    - iaudit_reasoner_model: path to merged reasoner model
    - iaudit_ranker_model: model name for Ranker-Critic (default: Mixtral-8x7B)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..parsing.findings import Evidence, Finding

logger = logging.getLogger(__name__)

# Lazy path setup for iAudit imports (only when pipeline runs)
_iaudit_path_setup_done = False


def _setup_iaudit_path() -> bool:
    """Lazily add iaudit to sys.path only when needed. Returns True if successful."""
    global _iaudit_path_setup_done
    if _iaudit_path_setup_done:
        return True

    iaudit_path = Path(__file__).parent.parent.parent.parent / "iaudit"
    if iaudit_path.exists():
        if str(iaudit_path) not in sys.path:
            sys.path.insert(0, str(iaudit_path))
        _iaudit_path_setup_done = True
        return True
    else:
        logger.warning(f"iAudit path not found: {iaudit_path}. Ranker-Critic will be unavailable.")
        return False


@dataclass
class iAuditConfig:
    """Configuration for iAudit pipeline."""
    # Model paths (on server or local)
    detector_model_path: str = ""  # Path to merged detector model
    reasoner_model_path: str = ""  # Path to merged reasoner model
    ranker_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Ranker-Critic model

    # Inference settings
    use_4bit: bool = True  # Use 4-bit quantization for memory efficiency
    max_ranker_rounds: int = 5  # Maximum Ranker-Critic debate rounds

    # Number of inference paths
    detector_prompts: int = 5  # Number of detector prompt variations
    reasoner_prompts: int = 10  # Number of reasoner paths (5 with call, 5 without)

    # HuggingFace token for model access
    hf_token: str = ""


@dataclass
class iAuditResult:
    """Result from iAudit pipeline."""
    findings: List[Finding] = field(default_factory=list)

    # Stage outputs
    detector_label: str = ""  # "vulnerable" or "safe"
    detector_responses: List[str] = field(default_factory=list)
    reasoner_explanations: List[str] = field(default_factory=list)
    selected_explanation: str = ""
    ranker_analysis: str = ""

    # Cost and performance tracking
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_latency_s: float = 0.0
    api_calls: int = 0

    # Per-stage breakdown
    detector_cost: float = 0.0
    reasoner_cost: float = 0.0
    ranker_cost: float = 0.0

    # Debate metadata
    ranker_rounds: int = 0
    critic_agreed: bool = False


def _extract_label_from_response(text: str) -> str:
    """Extract safe/vulnerable label from detector response."""
    match = re.search(r"(?i)the label is (safe|vulnerable)", text)
    if match:
        return match.group(1).lower()
    # Fallback: look for keywords
    text_lower = text.lower()
    if "vulnerable" in text_lower:
        return "vulnerable"
    if "safe" in text_lower:
        return "safe"
    return "unknown"


def _majority_vote(labels: List[str]) -> str:
    """Get majority vote from detector responses."""
    vulnerable_count = sum(1 for l in labels if l == "vulnerable")
    safe_count = sum(1 for l in labels if l == "safe")
    return "vulnerable" if vulnerable_count > safe_count else "safe"


def _parse_vulnerability_category(explanation: str) -> str:
    """Extract DASP category from explanation text.

    Maps both SmartBugs/DASP categories and Qian dataset categories.
    Qian categories: BN (block number), DE (delegatecall), EF (ether frozen),
                     SE (strict equality), OF (overflow), RE (reentrancy),
                     TP (timestamp), UC (unchecked call)
    """
    # Map common vulnerability patterns to DASP categories
    # Order matters - more specific patterns first
    category_patterns = {
        "reentrancy": [r"reentran", r"recursive call", r"external call.*before.*state", r"call.*back"],
        "unchecked_low_level_calls": [
            r"unchecked.*call", r"return value.*ignored", r"low.?level",
            r"unchecked external call", r"\.call\(", r"\.send\(", r"\.transfer\(",
            r"check.*return", r"ignore.*return", r"external call.*check",
            r"call\.value", r"send\s*\(", r"delegatecall"
        ],
        "access_control": [r"access control", r"missing owner", r"unprotected", r"authorization",
                          r"only.*owner", r"require.*msg\.sender", r"permission"],
        "arithmetic": [r"overflow", r"underflow", r"integer", r"arithmetic", r"safemath"],
        "denial_of_service": [r"denial.?of.?service", r"dos", r"gas limit", r"loop.*external",
                              r"ether frozen", r"frozen", r"strict equality", r"balance.*equal"],
        "bad_randomness": [r"random", r"blockhash", r"predictable", r"block\.number", r"block number"],
        "front_running": [r"front.?run", r"transaction order", r"mempool", r"race condition"],
        "time_manipulation": [r"timestamp", r"block\.timestamp", r"time.*depend", r"now\s"],
        "short_addresses": [r"short address", r"parameter.*length", r"msg\.data"],
        "other": [r"uninitialized", r"floating pragma", r"deprecated"],
    }

    text_lower = explanation.lower()
    for category, patterns in category_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return category
    return "other"


def _extract_selected_reason_index(ranker_response: str) -> int:
    """Extract the reason index from ranker's JSON response.

    Ranker outputs like: {"id": "Reason 1", "score": 9, "analysis": "..."}
    Returns 0-based index (Reason 1 -> 0, Reason 2 -> 1, etc.)
    """
    try:
        # Try to parse as JSON
        data = json.loads(ranker_response.strip().rstrip('</s>'))
        reason_id = data.get('id', '').lower()
        # Extract number from "reason 1", "Reason 2", etc.
        match = re.search(r'reason\s*(\d+)', reason_id)
        if match:
            return int(match.group(1)) - 1  # Convert to 0-based
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass

    # Fallback: try regex on raw text
    match = re.search(r'reason\s*(\d+)', ranker_response.lower())
    if match:
        return int(match.group(1)) - 1

    return 0  # Default to first reason


def run_iaudit_pipeline(
    contract_source: str,
    config: iAuditConfig,
    verbose: bool = False,
) -> iAuditResult:
    """
    Run the full iAudit pipeline on a single contract.

    Pipeline:
    1. Detector: Binary classification (safe/vulnerable) with majority voting
    2. Reasoner: Generate multiple explanations (if vulnerable)
    3. Ranker-Critic: Select best explanation through debate

    Args:
        contract_source: Solidity source code
        config: iAudit configuration
        verbose: Print progress information

    Returns:
        iAuditResult with findings and metadata
    """
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    result = iAuditResult()
    start_time = time.time()

    # Set HuggingFace token - check env vars first, then config
    hf_token = config.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("Access_Token", "")
    if hf_token:
        os.environ["Access_Token"] = hf_token

    # Check if models are available
    if not config.detector_model_path or not Path(config.detector_model_path).exists():
        if verbose:
            print(f"[iAudit] Detector model not found at {config.detector_model_path}")
        # Return empty result - models not available
        result.findings = []
        return result

    if verbose:
        print(f"[iAudit] Loading Detector model from {config.detector_model_path}")

    # ===== Stage 1: Detector (Binary Classification) =====
    torch_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if config.use_4bit else None

    detector_start = time.time()

    # Load detector model
    detector_tokenizer = AutoTokenizer.from_pretrained(
        config.detector_model_path,
        token=config.hf_token or os.environ.get("Access_Token", ""),
    )
    detector_model = AutoModelForCausalLM.from_pretrained(
        config.detector_model_path,
        device_map={"": 0},  # Force to GPU 0 (leave GPU 1 for Mixtral)
        torch_dtype=torch_dtype,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        token=config.hf_token or os.environ.get("Access_Token", ""),
    )
    detector_model.eval()

    # Generate detector prompts (5 variations per paper)
    detector_prompts = _generate_detector_prompts(contract_source, config.detector_prompts)

    detector_responses = []
    detector_labels = []

    with torch.no_grad():
        for prompt in detector_prompts:
            inputs = detector_tokenizer(prompt, return_tensors="pt")
            outputs = detector_model.generate(
                input_ids=inputs["input_ids"].to(detector_model.device),
                attention_mask=inputs["attention_mask"].to(detector_model.device),
                max_new_tokens=512,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                eos_token_id=detector_tokenizer.eos_token_id,
                pad_token_id=detector_tokenizer.eos_token_id,
            )
            response_text = detector_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract response after "### Response:"
            if "### Response:" in response_text:
                response_text = response_text.split("### Response:")[1].strip()
            detector_responses.append(response_text)
            detector_labels.append(_extract_label_from_response(response_text))

    # Majority voting
    final_label = _majority_vote(detector_labels)
    result.detector_label = final_label
    result.detector_responses = detector_responses
    result.detector_cost = time.time() - detector_start
    result.api_calls += config.detector_prompts

    if verbose:
        print(f"[iAudit] Detector verdict: {final_label} (votes: {detector_labels})")

    # If safe, return empty findings
    if final_label == "safe":
        result.total_latency_s = time.time() - start_time
        return result

    # Free detector model memory
    del detector_model
    del detector_tokenizer
    torch.cuda.empty_cache()

    # ===== Stage 2: Reasoner (Explanation Generation) =====
    if not config.reasoner_model_path or not Path(config.reasoner_model_path).exists():
        if verbose:
            print(f"[iAudit] Reasoner model not found at {config.reasoner_model_path}")
        # Create basic finding without detailed explanation
        result.findings = [Finding(
            title="Vulnerability Detected",
            category="other",
            severity="high",
            confidence=0.8,
            explanation="Detector identified this contract as vulnerable (no reasoner available).",
        )]
        result.total_latency_s = time.time() - start_time
        return result

    if verbose:
        print(f"[iAudit] Loading Reasoner model from {config.reasoner_model_path}")

    reasoner_start = time.time()

    # Load reasoner model
    reasoner_tokenizer = AutoTokenizer.from_pretrained(
        config.reasoner_model_path,
        token=config.hf_token or os.environ.get("Access_Token", ""),
    )
    reasoner_model = AutoModelForCausalLM.from_pretrained(
        config.reasoner_model_path,
        device_map={"": 0},  # Force to GPU 0 (leave GPU 1 for Mixtral)
        torch_dtype=torch_dtype,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        token=config.hf_token or os.environ.get("Access_Token", ""),
    )
    reasoner_model.eval()

    # Generate reasoner prompts (10 paths: 5 with call context, 5 without)
    reasoner_prompts = _generate_reasoner_prompts(contract_source, config.reasoner_prompts)

    reasoner_explanations = []

    with torch.no_grad():
        for prompt in reasoner_prompts:
            inputs = reasoner_tokenizer(prompt, return_tensors="pt")
            outputs = reasoner_model.generate(
                input_ids=inputs["input_ids"].to(reasoner_model.device),
                attention_mask=inputs["attention_mask"].to(reasoner_model.device),
                max_new_tokens=1024,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                eos_token_id=reasoner_tokenizer.eos_token_id,
                pad_token_id=reasoner_tokenizer.eos_token_id,
            )
            response_text = reasoner_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "### Response:" in response_text:
                response_text = response_text.split("### Response:")[1].strip()
            reasoner_explanations.append(response_text)

    result.reasoner_explanations = reasoner_explanations
    result.reasoner_cost = time.time() - reasoner_start
    result.api_calls += config.reasoner_prompts

    if verbose:
        print(f"[iAudit] Reasoner generated {len(reasoner_explanations)} explanations")

    # Free reasoner model memory
    del reasoner_model
    del reasoner_tokenizer
    torch.cuda.empty_cache()

    # ===== Stage 3: Ranker-Critic Selection =====
    if verbose:
        print(f"[iAudit] Loading Ranker-Critic model: {config.ranker_model}")

    ranker_start = time.time()

    try:
        # Setup iAudit path for Ranker/Critic imports (lazy initialization)
        if not _setup_iaudit_path():
            raise ImportError("iAudit path not available")

        # Load Ranker-Critic model (Mixtral-8x7B)
        # Force to GPU 1 to avoid conflict with student's job on GPU 0
        ranker_tokenizer = AutoTokenizer.from_pretrained(config.ranker_model)
        ranker_model = AutoModelForCausalLM.from_pretrained(
            config.ranker_model,
            device_map={"": 1},  # Force all layers to GPU 1 (has 90 GiB free)
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )

        # Import Ranker and Critic from iaudit (path set up above)
        from selection.Ranker import Ranker
        from selection.Criticor import Critic

        ranker = Ranker(ranker_model, ranker_tokenizer)
        critic = Critic(ranker_model, ranker_tokenizer)

        # Run Ranker-Critic debate
        feedback = None
        selected_response = None

        for round_num in range(config.max_ranker_rounds):
            result.ranker_rounds = round_num + 1

            # Ranker selects or merges
            item = {"code": contract_source, "reason_pred": reasoner_explanations}
            if feedback is None or "merge" not in feedback.get("feedback", "").lower():
                _, response = ranker.rank(item, "vulnerable", feedback=feedback)
            else:
                _, response = ranker.merge(item, "vulnerable", feedback=feedback)

            selected_response = response
            if verbose:
                print(f"[iAudit] Round {round_num + 1}: Ranker selected explanation")

            # Critic reviews
            critic_response = critic.critic(item, response)

            if "yes" in critic_response.lower():
                result.critic_agreed = True
                if verbose:
                    print(f"[iAudit] Critic agreed at round {round_num + 1}")
                break
            else:
                feedback = {"pre_answer": response, "feedback": critic_response}

        result.ranker_analysis = selected_response or ""
        result.ranker_cost = time.time() - ranker_start
        result.api_calls += result.ranker_rounds * 2  # ranker + critic per round

        # Extract the actual reasoner explanation (not the ranker's JSON meta-analysis)
        if selected_response and reasoner_explanations:
            reason_idx = _extract_selected_reason_index(selected_response)
            reason_idx = min(reason_idx, len(reasoner_explanations) - 1)  # Bounds check
            result.selected_explanation = reasoner_explanations[reason_idx]
        else:
            result.selected_explanation = reasoner_explanations[0] if reasoner_explanations else ""

        # Free ranker model memory
        del ranker_model
        del ranker_tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        # Log error properly (not just when verbose) and track partial cost
        logger.warning(
            f"Ranker-Critic stage failed: {e}. Falling back to first reasoner explanation. "
            "This is a degraded execution path."
        )
        if verbose:
            print(f"[iAudit] Ranker-Critic failed: {e}, using first explanation")
        result.selected_explanation = reasoner_explanations[0] if reasoner_explanations else ""
        result.ranker_cost = time.time() - ranker_start  # Track time spent even on failure

    # ===== Create Final Findings =====
    if result.selected_explanation:
        # Parse category from actual reasoner explanation (not ranker's JSON)
        category = _parse_vulnerability_category(result.selected_explanation)
        finding = Finding(
            title=f"{category.replace('_', ' ').title()} Vulnerability",
            category=category,
            severity="high",
            confidence=0.9 if result.critic_agreed else 0.7,
            explanation=result.selected_explanation,
            evidence=Evidence(lines=[], function=""),
        )
        result.findings = [finding]

    result.total_latency_s = time.time() - start_time

    if verbose:
        print(f"[iAudit] Pipeline complete in {result.total_latency_s:.1f}s")
        print(f"[iAudit] Final findings: {len(result.findings)}")

    return result


def _generate_detector_prompts(contract_source: str, num_prompts: int = 5) -> List[str]:
    """Generate detector prompt variations (per iAudit paper)."""
    base_template = """### Instruction:
You are a smart contract security auditor. Analyze the following Solidity code and determine if it contains any security vulnerabilities.

{variation}

```solidity
{code}
```

Respond with: "The label is [safe/vulnerable]" followed by a brief explanation.

### Response:"""

    variations = [
        "Focus on reentrancy, access control, and arithmetic vulnerabilities.",
        "Check for unchecked external calls and denial of service risks.",
        "Look for front-running, time manipulation, and bad randomness issues.",
        "Analyze state changes, external interactions, and control flow.",
        "Examine authorization checks, input validation, and data handling.",
    ]

    prompts = []
    for i in range(min(num_prompts, len(variations))):
        prompt = base_template.format(
            variation=variations[i],
            code=contract_source[:8000],  # Truncate if too long
        )
        prompts.append(prompt)

    return prompts


def _generate_reasoner_prompts(contract_source: str, num_prompts: int = 10) -> List[str]:
    """Generate reasoner prompt variations (per iAudit paper: 5 with call, 5 without)."""
    base_template = """### Instruction:
You are a smart contract security expert. The following Solidity code has been identified as vulnerable.

{context_info}

Explain the vulnerability in detail:
1. What type of vulnerability is present?
2. Where exactly is the vulnerability located?
3. How could an attacker exploit it?
4. What is the potential impact?

```solidity
{code}
```

### Response:"""

    with_call_variations = [
        "Consider the call sequences and function interactions.",
        "Analyze how external calls could be exploited.",
        "Focus on state changes before and after external calls.",
        "Examine the control flow between functions.",
        "Consider reentrancy through callback mechanisms.",
    ]

    without_call_variations = [
        "Focus on the contract's internal logic.",
        "Analyze access control and authorization.",
        "Check arithmetic operations and overflow risks.",
        "Examine input validation and edge cases.",
        "Consider denial of service scenarios.",
    ]

    prompts = []

    # First half: with call context
    for i in range(num_prompts // 2):
        idx = i % len(with_call_variations)
        prompt = base_template.format(
            context_info=with_call_variations[idx],
            code=contract_source[:8000],
        )
        prompts.append(prompt)

    # Second half: without call context
    for i in range(num_prompts - num_prompts // 2):
        idx = i % len(without_call_variations)
        prompt = base_template.format(
            context_info=without_call_variations[idx],
            code=contract_source[:8000],
        )
        prompts.append(prompt)

    return prompts
