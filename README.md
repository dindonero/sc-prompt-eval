# SC-Prompt-Eval: LLM Prompt Engineering for Smart Contract Vulnerability Detection

A research framework for systematically evaluating prompt engineering strategies for smart contract vulnerability detection using Large Language Models.

## Overview

This tool implements and evaluates **6 distinct prompt engineering strategies (P0-P6)** for detecting vulnerabilities in Solidity smart contracts. Each strategy represents a different approach from the research literature:

| Prompt | Strategy | Description | Paper Reference |
|--------|----------|-------------|-----------------|
| **P0** | Zero-shot Baseline | Single-pass JSON output format | Baseline |
| **P1** | In-Context Learning | Per-vulnerability-type examples (10 API calls) | Few-shot learning |
| **P2** | Structured Reasoning | Chain-of-thought with audit checklist | CoT prompting |
| **P3** | SmartGuard RAG | Retrieved patterns + CoT + self-check | Zhang et al. 2024 |
| **P4** | Tool-Augmented | Slither static analysis + multi-stage LLM | Sun et al. 2024 (GPTScan) |
| **P5** | Multi-Agent | 4 specialized agents (PM, Counselor, Auditor, Expert) | Wei et al. 2024 (LLM-SmartAudit) |
| **P6** | Fine-tuned Pipeline | Detector + Reasoner + Ranker-Critic | Ma et al. 2024 (iAudit) |

## Key Results (experiment_v1)

Evaluated on **SmartBugs Curated dataset** (143 contracts, 10 DASP vulnerability categories):

| Prompt | Precision | Recall | F1 | Cost |
|--------|-----------|--------|------|------|
| P0_baseline | 31.0% | 78.7% | 44.5% | $1.89 |
| P1_icl | 25.0% | 96.5% | 39.7% | ~$19 |
| P2_structured | 44.2% | **96.3%** | 60.6% | $2.25 |
| **P3_smartguard** | **63.0%** | 88.0% | **73.5%** | ~$43 |
| P4_tool_augmented | 26.4% | 24.6% | 25.5% | $4.13 |
| P5_smartaudit | 23.6% | 86.0% | 37.1% | $24.27 |
| P6_iaudit | 46.7% | 46.7% | 46.7% | Local |

**Best F1:** P3_smartguard (73.5%) | **Best Cost-Effectiveness:** P2_structured (60.6% F1 @ $2.25)

## Features

- **6 Prompt Strategies** via Jinja2 templates
- **Multi-Model Support**: OpenAI, Anthropic, Azure OpenAI, local models
- **Multiple Datasets**: SmartBugs Curated, benign contracts, custom datasets
- **RAG Integration**: CodeBERT embeddings for pattern retrieval (P3)
- **Static Analysis**: Optional Slither integration (P4)
- **Multi-Agent Orchestration**: Role-based agents with consensus (P5)
- **Comprehensive Metrics**: Precision, Recall, F1, cost tracking, latency
- **Reproducible**: Seeded RNG, detailed logging, raw output preservation

## Installation

### Prerequisites
- Python 3.10+
- (Optional) Slither for P4 static analysis
- (Optional) GPU for P6 local models

### Setup

```bash
# Clone repository
git clone https://github.com/dindonero/sc-prompt-eval.git
cd sc-prompt-eval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### API Key Configuration

Edit `.env` with your credentials:

```bash
# Required for OpenAI/Azure models
OPENAI_API_KEY=sk-your-key-here
AZURE_OPENAI_API_KEY=your-azure-key

# Required for Anthropic models
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional: For P6 iAudit fine-tuned models
HUGGINGFACE_TOKEN=hf_your-token
```

## Quick Start

### Validate Configuration (Dry Run)

```bash
python -m sc_prompt_eval.cli --config configs/mini_pilot.yaml --dry-run
```

### Run Mini Pilot (10 contracts, quick test)

```bash
python -m sc_prompt_eval.cli --config configs/mini_pilot.yaml
```

### Run Full Experiment (143 contracts)

```bash
python -m sc_prompt_eval.cli --config configs/full_experiment.yaml --workers 4
```

### CLI Options

```
python -m sc_prompt_eval.cli --config <config.yaml> [options]

Options:
  --config PATH     Path to experiment YAML (required)
  --dry-run         Validate config and prompt rendering only
  --verbose         Enable verbose output with API responses
  --workers N       Number of parallel workers (default: 1)
```

## Configuration Guide

### YAML Config Structure

```yaml
experiment_id: "my_experiment"
output_dir: "outputs/my_experiment"
runs_per_item: 1              # Repetitions per (contract, model, prompt)
random_seed: 42               # For reproducibility
notes: "Description of this experiment"

datasets:
  - name: "smartbugs_curated"
    kind: "smartbugs_curated"
    path: "data/smartbugs_curated/dataset"

models:
  - name: "gpt-4o"
    provider: "openai"
    params:
      max_tokens: 4000
      temperature: 0.0        # Not supported for o-series models

prompts:
  - id: "P0_baseline"
    template_path: "p0_baseline.j2"
    description: "Zero-shot baseline"

tools:
  # P3 SmartGuard RAG settings
  top_k_patterns: 5
  enable_cot_expansion: true

  # P4 Slither settings
  slither_timeout: 120

  # P5 Multi-agent settings
  smartaudit_mode: "BA"       # "BA" (Broad) or "TA" (Targeted)
  smartaudit_max_rounds: 3
```

### Adding Custom Prompts

1. **Create Jinja2 template** in `prompts/` directory:

```jinja2
{# prompts/p7_custom.j2 #}
You are a smart contract security auditor.

Analyze the following Solidity contract for vulnerabilities.
Focus on: {{ focus_areas | default('all DASP categories') }}

Contract:
```solidity
{{ contract_source }}
```

Return findings as JSON array with schema:
[{"category": "...", "title": "...", "severity": "...", "explanation": "..."}]
```

2. **Register in config**:

```yaml
prompts:
  - id: "P7_custom"
    template_path: "p7_custom.j2"
    description: "My custom prompt strategy"
```

3. **Available template variables**:
   - `{{ contract_source }}` - The Solidity source code
   - `{{ retrieved_patterns }}` - RAG patterns (P3)
   - `{{ slither_output }}` - Static analysis results (P4)
   - `{{ agent_role }}` - Agent name for multi-agent (P5)
   - Custom variables via `ToolConfig`

### Adding Custom Datasets

1. **Create dataset directory**:

```
data/my_dataset/
├── category_1/
│   ├── contract1.sol
│   └── contract2.sol
├── category_2/
│   └── contract3.sol
└── vulnerabilities.json  # Optional ground truth
```

2. **Ground truth format** (`vulnerabilities.json`):

```json
{
  "contract1.sol": [
    {
      "category": "reentrancy",
      "title": "Reentrancy in withdraw()",
      "evidence": {
        "lines": [42, 43],
        "function": "withdraw"
      }
    }
  ]
}
```

3. **Register in config**:

```yaml
datasets:
  - name: "my_dataset"
    kind: "smartbugs_curated"  # Uses same loader format
    path: "data/my_dataset"
```

### Changing Models

#### Supported Providers

| Provider | Models | Notes |
|----------|--------|-------|
| `openai` | gpt-4o, gpt-4o-mini, o1, o3-mini, o4-mini | Azure endpoints supported |
| `anthropic` | claude-opus-4-5, claude-sonnet-4-5, claude-3-5-sonnet | |
| `local` | Custom fine-tuned models | For P6 iAudit |

#### OpenAI Configuration

```yaml
models:
  - name: "gpt-4o"
    provider: "openai"
    params:
      max_tokens: 4000
      temperature: 0.0
      top_p: 1.0
```

#### Azure OpenAI Configuration

```yaml
models:
  - name: "o4-mini"
    provider: "openai"
    params:
      base_url: "https://your-resource.openai.azure.com/openai/v1/"
      api_key_env: "AZURE_OPENAI_API_KEY"
      max_tokens: 16000
```

#### Anthropic Configuration

```yaml
models:
  - name: "claude-opus-4-5"
    provider: "anthropic"
    params:
      max_tokens: 4000
      temperature: 0.0
```

### Tool Configuration (P3-P6)

#### P3 SmartGuard (RAG)

```yaml
tools:
  top_k_patterns: 5           # Number of patterns to retrieve
  enable_cot_expansion: true  # Generate CoT for patterns
  use_icl_template: true      # Use few-shot ICL format
```

#### P4 Tool-Augmented (Slither)

```yaml
tools:
  slither_timeout: 120        # Seconds before timeout
  enable_static_confirmation: true
  slither_mode: "confirmation_only"  # or "pre_filter"
```

#### P5 Multi-Agent (SmartAudit)

```yaml
tools:
  smartaudit_mode: "BA"       # "BA" (Broad Analysis) or "TA" (Targeted)
  smartaudit_max_rounds: 3    # Max consensus iterations
```

## Output Structure

```
outputs/experiment_v1/
├── experiment_config.json    # Config used for this run
├── final_results.json        # Aggregate metrics
├── results_summary.json      # Detailed per-contract metrics
├── RESULTS.md                # Human-readable summary
└── smartbugs_curated/
    └── o4-mini/
        ├── P0_baseline/
        │   └── contract_name/
        │       ├── run_0_raw.json     # Raw LLM response
        │       ├── run_0_parsed.json  # Normalized findings
        │       └── run_0_prompt.txt   # Prompt sent to LLM
        ├── P1_icl/
        ├── P2_structured/
        └── ...
```

### Output JSON Schema

**Parsed findings** (`run_0_parsed.json`):

```json
{
  "findings": [
    {
      "category": "reentrancy",
      "title": "Reentrancy vulnerability in withdraw()",
      "severity": "high",
      "confidence": "high",
      "evidence": {
        "file": "Contract.sol",
        "lines": [42, 43],
        "function": "withdraw"
      },
      "explanation": "External call before state update...",
      "fix_suggestion": "Use checks-effects-interactions pattern..."
    }
  ],
  "metrics": {
    "tp": 1,
    "fp": 0,
    "fn": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "cost_usd": 0.013,
  "latency_s": 2.45,
  "tokens": {
    "input": 1500,
    "output": 450
  }
}
```

## Project Structure

```
sc-prompt-eval/
├── src/sc_prompt_eval/       # Core package
│   ├── cli.py                # Command-line interface
│   ├── config.py             # Configuration dataclasses
│   ├── runner/               # Experiment orchestration
│   │   ├── run.py            # Main runner with prompt routing
│   │   ├── multiagent.py     # P5 multi-agent implementation
│   │   ├── iaudit_runner.py  # P6 fine-tuned pipeline
│   │   └── cost_estimator.py # Token counting and cost tracking
│   ├── models/               # LLM adapters
│   │   ├── openai_adapter.py
│   │   └── anthropic_adapter.py
│   ├── datasets/             # Dataset loaders
│   │   ├── base.py           # ContractItem dataclass
│   │   └── smartbugs_curated.py
│   ├── prompts/              # Template registry
│   │   └── registry.py       # Jinja2 loader
│   ├── rag/                  # RAG components for P3
│   │   ├── retriever.py      # CodeBERT embeddings
│   │   └── cot_generator.py  # Chain-of-thought generation
│   ├── parsing/              # Output parsing
│   │   └── findings.py       # JSON extraction and normalization
│   ├── scoring/              # Metrics computation
│   │   └── metrics.py        # Precision, Recall, F1
│   └── tools/                # Static analysis integration
│       └── slither.py        # Slither wrapper for P4
├── prompts/                  # Jinja2 prompt templates
│   ├── p0_baseline.j2
│   ├── p1_icl_single_type.j2
│   ├── p2_structured_reasoning.j2
│   ├── p3_smartguard.j2
│   ├── p4_stage*.j2
│   └── p5_*.j2
├── configs/                  # Experiment configurations
│   ├── mini_pilot.yaml       # Quick test (10 contracts)
│   └── full_experiment.yaml  # Full evaluation (143 contracts)
├── data/                     # Datasets
│   ├── smartbugs_curated/    # Main evaluation dataset
│   ├── smartbugs_mini/       # Subset for testing
│   ├── benign_contracts/     # False positive testing
│   ├── patterns_database.json # RAG patterns (18MB)
│   └── icl_examples.json     # Few-shot examples
├── schemas/                  # JSON validation schemas
├── outputs/                  # Experiment results
│   └── experiment_v1/        # Pre-computed results
└── tests/                    # Unit tests
```

## Vulnerability Categories (DASP Top 10)

| Category | Description |
|----------|-------------|
| `reentrancy` | External calls before state updates |
| `access_control` | Missing or improper access restrictions |
| `arithmetic` | Integer overflow/underflow |
| `unchecked_low_level_calls` | Unchecked return values from call/send |
| `denial_of_service` | Gas limit, unexpected revert |
| `bad_randomness` | Predictable random number generation |
| `front_running` | Transaction ordering dependence |
| `time_manipulation` | Block timestamp dependence |
| `short_addresses` | Short address/parameter attack |
| `other` | Miscellaneous vulnerabilities |

## External Data (Optional)

### Qian Dataset for Full P3/P6 Support

The RAG retrieval system uses patterns from the Qian et al. dataset. The `patterns_database.json` file included in this repo contains extracted patterns. For the full training dataset:

1. Download from: [Qian et al. Smart Contract Dataset](https://github.com/example/qian-dataset)
2. Place in `data/qian_dataset/`
3. The retriever will automatically use it for P3 pattern expansion

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{sc_prompt_eval,
  title = {SC-Prompt-Eval: LLM Prompt Engineering for Smart Contract Vulnerability Detection},
  author = {Donato, Miguel},
  year = {2026},
  url = {https://github.com/dindonero/sc-prompt-eval}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- SmartBugs Curated dataset maintainers
- Zhang et al. (SmartGuard), Sun et al. (GPTScan), Wei et al. (LLM-SmartAudit), Ma et al. (iAudit) for prompt strategy inspiration
