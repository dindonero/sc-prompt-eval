[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# sc-prompt-eval: A Prompt Engineering Framework for LLM-Based Smart Contract Vulnerability Detection

A research framework for systematically evaluating prompt engineering strategies
for detecting vulnerabilities in Solidity smart contracts using Large Language
Models. The framework implements six strategies drawn from the recent
literature -- ranging from a zero-shot baseline through in-context learning,
chain-of-thought reasoning, retrieval-augmented generation (RAG), tool-augmented
analysis, multi-agent orchestration, and fine-tuned pipelines -- and benchmarks
them on the SmartBugs Curated dataset (143 contracts, 10 DASP vulnerability
categories) under controlled, reproducible conditions.

## Key Results

Evaluated on **SmartBugs Curated** with GPT-4o-mini (Azure):

| Strategy | F1 | Precision | Recall | Cost |
|---|---|---|---|---|
| P0 Baseline | 34.2% | 23.9% | 60.1% | $1.90 |
| P1 ICL | 39.5% | 24.9% | 96.5% | $19 |
| P2 Structured CoT | 59.6% | 44.1% | 91.6% | $2.39 |
| **P3 SmartGuard** | **72.5%** | **61.9%** | **87.4%** | $43 |
| P4 Tool-Augmented | 26.8% | 29.5% | 24.6% | $4.13 |
| P5 SmartAudit | 38.3% | 24.6% | 86.0% | $24.27 |
| P6 iAudit | 46.7% | 46.7% | 46.7% | Local |

**Best F1:** P3 SmartGuard (72.5%) | **Best cost-effectiveness:** P2 Structured CoT (59.6% F1 at $2.39)

## Prompt Strategies

| ID | Strategy | Description | Reference |
|---|---|---|---|
| **P0** | Zero-shot Baseline | Single-pass JSON output format | Baseline |
| **P1** | In-Context Learning | Per-vulnerability-type examples (10 API calls) | Few-shot learning |
| **P2** | Structured Reasoning | Chain-of-thought with audit checklist | CoT prompting |
| **P3** | SmartGuard RAG | Retrieved patterns + CoT + self-check | Zhang et al. 2024 |
| **P4** | Tool-Augmented | Slither static analysis + multi-stage LLM | Sun et al. 2024 (GPTScan) |
| **P5** | Multi-Agent | 4 specialized agents (PM, Counselor, Auditor, Expert) | Wei et al. 2024 (LLM-SmartAudit) |
| **P6** | Fine-tuned Pipeline | Detector + Reasoner + Ranker-Critic | Ma et al. 2024 (iAudit) |

## Installation

### Prerequisites

- Python 3.10+
- (Optional) [Slither](https://github.com/crytic/slither) for P4 static analysis
- (Optional) GPU for P6 local models

### Setup

```bash
# Clone repository
git clone https://github.com/dindonero/sc-prompt-eval.git
cd sc-prompt-eval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install package in development mode (includes all dependencies)
pip install -e .

# For P6 iAudit local model support:
# pip install -e ".[iaudit]"

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

### 1. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your OpenAI / Azure / Anthropic keys
```

### 2. Validate configuration (dry run)

```bash
python -m sc_prompt_eval.cli --config configs/mini_pilot.yaml --dry-run
```

### 3. Run a small pilot (10 contracts)

```bash
python -m sc_prompt_eval.cli --config configs/mini_pilot.yaml
```

### 4. Run the full experiment (143 contracts)

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

## Configuration

Experiments are defined via YAML configuration files. See `configs/` for examples.

```yaml
experiment_id: "my_experiment"
output_dir: "outputs/my_experiment"
runs_per_item: 1
random_seed: 42

datasets:
  - name: "smartbugs_curated"
    kind: "smartbugs_curated"
    path: "data/smartbugs_curated/dataset"

models:
  - name: "gpt-4o"
    provider: "openai"
    params:
      max_tokens: 4000
      temperature: 0.0

prompts:
  - id: "P0_baseline"
    template_path: "p0_baseline.j2"
    description: "Zero-shot baseline"
```

### Supported Providers

| Provider | Models | Notes |
|---|---|---|
| `openai` | gpt-4o, gpt-4o-mini, o1, o3-mini, o4-mini | Azure endpoints supported |
| `anthropic` | claude-opus-4-5, claude-sonnet-4-5, claude-3-5-sonnet | Direct API |
| `local` | Custom fine-tuned models | For P6 iAudit |

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
│   ├── models/               # LLM adapters (OpenAI, Anthropic)
│   ├── datasets/             # Dataset loaders
│   ├── prompts/              # Jinja2 template registry
│   ├── rag/                  # RAG components for P3
│   ├── parsing/              # Output parsing and normalization
│   ├── scoring/              # Metrics (Precision, Recall, F1)
│   └── tools/                # Static analysis integration (Slither)
├── prompts/                  # Jinja2 prompt templates (P0-P6)
├── configs/                  # Experiment YAML configurations
├── data/                     # Datasets and RAG patterns
│   ├── smartbugs_curated/    # Main evaluation dataset
│   ├── benign_contracts/     # False positive testing
│   └── patterns_database.json # RAG pattern corpus
├── schemas/                  # JSON validation schemas
├── outputs/                  # Experiment results
└── tests/                    # Unit tests
```

## Output Format

Results are written per-contract with raw LLM responses, parsed findings, and
aggregate metrics:

```
outputs/experiment_v1/
├── experiment_config.json
├── final_results.json
├── results_summary.json
└── smartbugs_curated/
    └── o4-mini/
        └── P0_baseline/
            └── contract_name/
                ├── run_0_raw.json
                ├── run_0_parsed.json
                └── run_0_prompt.txt
```

## Paper

> **Prompt Engineering for LLM-Based Smart Contract Vulnerability Detection**
> Dinis Araujo, NOVA IMS, 2026
>
> Paper: [arXiv preprint (coming soon)](#)

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{araujo2026prompt,
  title   = {Prompt Engineering for {LLM}-Based Smart Contract Vulnerability Detection},
  author  = {Araujo, Dinis},
  year    = {2026},
  url     = {https://github.com/dindonero/sc-prompt-eval},
  note    = {arXiv preprint (coming soon)}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [SmartBugs](https://github.com/smartbugs/smartbugs) for the curated vulnerability dataset
- Zhang et al. (SmartGuard), Sun et al. (GPTScan), Wei et al. (LLM-SmartAudit), Ma et al. (iAudit) for the prompt strategy designs evaluated in this work
