"""
Comprehensive test suite for the sc_prompt_eval tool.
Tests all components without requiring actual API calls.
"""
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sc_prompt_eval.models.base import LLMAdapter, LLMResponse
from sc_prompt_eval.prompts.registry import PromptRegistry
from sc_prompt_eval.parsing.findings import parse_findings, Finding
from sc_prompt_eval.scoring.metrics import compute_metrics
from sc_prompt_eval.scoring.aggregation import aggregate_by_model_prompt
from sc_prompt_eval.scoring.stats import bootstrap_ci, mcnemar_test, wilcoxon_signed_rank
from sc_prompt_eval.analysis.pareto import compute_pareto_frontier, ParetoPoint
from sc_prompt_eval.analysis.tables import generate_latex_table
from sc_prompt_eval.runner.voting import aggregate_findings_by_voting, normalize_finding_key
from sc_prompt_eval.runner.multiagent import extract_final_findings, DebateResult
from sc_prompt_eval.rag.retriever import CodeBERTRetriever, retrieve_patterns
from sc_prompt_eval.robustness.mutations import (
    WhitespaceMutation, CommentMutation, VariableRenameMutation,
    apply_mutations, generate_mutants
)
from sc_prompt_eval.datasets.loader import (
    ContractSample, GroundTruth, InMemoryDataset, create_test_dataset
)


class MockLLMAdapter(LLMAdapter):
    """Mock adapter that returns predefined responses."""

    def __init__(self, responses: list = None):
        super().__init__("mock-model")
        self.responses = responses or []
        self.call_count = 0

    def generate(self, prompt: str) -> LLMResponse:
        if self.call_count < len(self.responses):
            response_text = self.responses[self.call_count]
        else:
            response_text = "[]"
        self.call_count += 1
        return LLMResponse(
            text=response_text,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
        )


def test_prompt_registry():
    """Test prompt template loading and rendering."""
    print("Testing PromptRegistry...")

    prompt_root = Path(__file__).parent.parent / 'prompts'
    registry = PromptRegistry(prompt_root)

    # Test loading p0_baseline template (known to work without extra params)
    template_name = "p0_baseline.j2"

    # Test rendering
    contract = "pragma solidity ^0.8.0; contract Test {}"
    rendered = registry.render(template_name, contract_source=contract)

    assert contract in rendered, "Contract should be in rendered output"
    assert len(rendered) > len(contract), "Rendered should include template text"
    print(f"  ✓ Loaded and rendered {template_name}")

    # Test p5 with slither_output parameter (needs candidate_functions too)
    rendered_p5 = registry.render(
        "p5_tool_augmented.j2",
        contract_source=contract,
        slither_output="No issues found",
        candidate_functions=[]  # Empty list for testing
    )
    assert "No issues found" in rendered_p5
    print("  ✓ Rendered p5_tool_augmented.j2 with slither_output")

    print("  ✓ PromptRegistry tests passed")


def test_findings_parser():
    """Test parsing of LLM vulnerability findings."""
    print("Testing findings parser...")

    # Test valid JSON array
    valid_json = '''
    [
        {
            "category": "reentrancy",
            "title": "Reentrancy",
            "severity": "high",
            "confidence": 0.9,
            "evidence": {"lines": [10, 11, 12], "function": "withdraw"},
            "explanation": "External call before state update"
        }
    ]
    '''
    findings, errors = parse_findings(valid_json)
    assert len(findings) == 1, f"Expected 1 finding, got {len(findings)}"
    assert findings[0].category == "reentrancy"
    assert findings[0].severity == "high"
    print("  ✓ Parsed valid JSON")

    # Test embedded JSON in text
    embedded = '''
    Here is my analysis:
    ```json
    [{"category": "arithmetic", "title": "Integer Overflow", "severity": "medium", "confidence": 0.8}]
    ```
    '''
    findings, errors = parse_findings(embedded)
    assert len(findings) == 1
    assert findings[0].category == "arithmetic"
    print("  ✓ Parsed embedded JSON")

    # Test empty array
    findings, errors = parse_findings("[]")
    assert len(findings) == 0
    print("  ✓ Handled empty array")

    # Test malformed input
    findings, errors = parse_findings("This is not JSON at all")
    assert len(findings) == 0
    print("  ✓ Handled malformed input gracefully")

    print("  ✓ Findings parser tests passed")


def test_metrics_computation():
    """Test vulnerability detection metrics."""
    print("Testing metrics computation...")

    # Create ground truth and predictions
    ground_truth = [
        GroundTruth(category="reentrancy", title="Reentrancy"),
        GroundTruth(category="arithmetic", title="Integer Overflow"),
    ]

    # Perfect prediction
    predictions = [
        Finding(category="reentrancy", title="Reentrancy", severity="high", confidence=0.9),
        Finding(category="arithmetic", title="Integer Overflow", severity="medium", confidence=0.8),
    ]

    metrics = compute_metrics(predictions, ground_truth)
    assert metrics.precision == 1.0, f"Expected precision 1.0, got {metrics.precision}"
    assert metrics.recall == 1.0, f"Expected recall 1.0, got {metrics.recall}"
    print("  ✓ Perfect prediction metrics correct")

    # Partial prediction (1 TP, 1 FP, 1 FN)
    predictions = [
        Finding(category="reentrancy", title="Reentrancy", severity="high", confidence=0.9),
        Finding(category="access_control", title="tx.origin", severity="medium", confidence=0.7),  # FP
    ]

    metrics = compute_metrics(predictions, ground_truth)
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    print("  ✓ Partial prediction metrics correct")

    print("  ✓ Metrics computation tests passed")


def test_voting_aggregation():
    """Test self-consistency voting."""
    print("Testing voting aggregation...")

    # Simulate 5 runs
    run1 = [
        Finding(category="reentrancy", title="Reentrancy", severity="high", confidence=0.9),
        Finding(category="arithmetic", title="Overflow", severity="medium", confidence=0.7),
    ]
    run2 = [
        Finding(category="reentrancy", title="Reentrancy", severity="high", confidence=0.85),
    ]
    run3 = [
        Finding(category="reentrancy", title="Reentrancy", severity="high", confidence=0.95),
        Finding(category="access_control", title="tx.origin", severity="medium", confidence=0.6),
    ]
    run4 = [
        Finding(category="reentrancy", title="Reentrancy", severity="high", confidence=0.88),
    ]
    run5 = [
        Finding(category="reentrancy", title="Reentrancy", severity="high", confidence=0.92),
        Finding(category="arithmetic", title="Overflow", severity="medium", confidence=0.75),
    ]

    all_findings = [run1, run2, run3, run4, run5]
    aggregated, stats = aggregate_findings_by_voting(all_findings, threshold=0.5)

    # reentrancy appears in all 5 runs (5/5 = 100%)
    # arithmetic appears in 2 runs (2/5 = 40%, below 50%)
    # access_control appears in 1 run (1/5 = 20%, below 50%)

    categories = [f.category for f in aggregated]
    assert "reentrancy" in categories, "reentrancy should pass voting (5/5)"
    print(f"  ✓ reentrancy passed voting (5/5 votes)")

    assert stats["n_runs"] == 5
    print(f"  ✓ Voting stats recorded correctly")

    print("  ✓ Voting aggregation tests passed")


def test_rag_system():
    """Test RAG TF-IDF retriever with Qian et al. patterns."""
    print("Testing RAG system...")

    # Test TF-IDF retriever
    retriever = CodeBERTRetriever()

    # Verify patterns loaded from Qian et al. dataset
    pattern_count = retriever.get_pattern_count()
    assert pattern_count > 0, "Should have patterns from Qian et al. dataset"
    print(f"  ✓ Patterns database has {pattern_count} patterns")

    # Check category distribution
    cat_counts = retriever.get_category_counts()
    assert len(cat_counts) > 0, "Should have multiple categories"
    print(f"  ✓ Patterns span {len(cat_counts)} categories")

    # Test search by code
    reentrancy_code = """
    function withdraw() public {
        uint amount = balances[msg.sender];
        msg.sender.call{value: amount}("");
        balances[msg.sender] = 0;
    }
    """

    retrieved_patterns = retriever.retrieve_patterns(reentrancy_code, num_patterns=5)
    assert len(retrieved_patterns) > 0, "Retriever should find patterns"
    print(f"  ✓ TF-IDF retriever found {len(retrieved_patterns)} patterns")

    # Verify pattern format
    for p in retrieved_patterns:
        assert "category" in p, "Pattern should have category"
        assert "vulnerable_code" in p, "Pattern should have vulnerable_code"
        assert "similarity_score" in p, "Pattern should have similarity_score"
    print("  ✓ Pattern format correct")

    # Test template format from retrieve_patterns function
    template_data = retrieve_patterns(reentrancy_code, num_patterns=3)
    assert isinstance(template_data, list)
    assert len(template_data) > 0
    print("  ✓ Convenience function works")

    print("  ✓ RAG system tests passed")


def test_mutation_operators():
    """Test robustness mutation operators."""
    print("Testing mutation operators...")

    sample_code = '''
pragma solidity ^0.8.0;

contract Test {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() public {
        uint256 amount = balances[msg.sender];
        balances[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }
}
'''

    # Test whitespace mutation
    ws = WhitespaceMutation()
    result = ws.apply(sample_code, seed=42)
    assert result.mutated != sample_code or result.changes_made == 0
    print(f"  ✓ Whitespace mutation: {result.changes_made} changes")

    # Test comment mutation
    cm = CommentMutation()
    result = cm.apply(sample_code, seed=42)
    print(f"  ✓ Comment mutation: {result.changes_made} changes")

    # Test variable rename
    vr = VariableRenameMutation()
    result = vr.apply(sample_code, seed=42)
    print(f"  ✓ Variable rename mutation: {result.changes_made} changes")

    # Test combined mutations
    mutated, results = apply_mutations(sample_code, seed=42)
    total_changes = sum(r.changes_made for r in results)
    print(f"  ✓ Combined mutations: {total_changes} total changes across {len(results)} operators")

    # Test mutant generation
    mutants = generate_mutants(sample_code, n_mutants=3, base_seed=42)
    assert len(mutants) == 3
    print(f"  ✓ Generated {len(mutants)} mutant variants")

    print("  ✓ Mutation operators tests passed")


def test_dataset_loader():
    """Test dataset loading functionality."""
    print("Testing dataset loader...")

    # Test in-memory dataset
    dataset = create_test_dataset()
    samples = dataset.load()

    assert len(samples) == 4, f"Expected 4 samples, got {len(samples)}"
    print(f"  ✓ Created test dataset with {len(samples)} samples")

    # Check vulnerable samples
    vuln_samples = [s for s in samples if s.is_vulnerable]
    assert len(vuln_samples) == 3, f"Expected 3 vulnerable samples, got {len(vuln_samples)}"
    print(f"  ✓ {len(vuln_samples)} samples have vulnerabilities")

    # Check categories
    all_categories = set()
    for s in samples:
        all_categories.update(s.categories)
    print(f"  ✓ Unique categories in dataset: {all_categories}")

    # Test iteration
    count = 0
    for sample in dataset.iter_samples():
        count += 1
    assert count == len(samples)
    print("  ✓ Iterator works correctly")

    print("  ✓ Dataset loader tests passed")


def test_statistical_analysis():
    """Test statistical analysis functions."""
    print("Testing statistical analysis...")

    # Test bootstrap CI
    scores = [0.8, 0.85, 0.75, 0.9, 0.82, 0.88, 0.79, 0.84]
    ci = bootstrap_ci(scores, n_bootstrap=1000)
    assert ci.lower < ci.mean < ci.upper
    assert 0 <= ci.lower <= 1
    assert 0 <= ci.upper <= 1
    print(f"  ✓ Bootstrap CI: {ci.mean:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]")

    # Test McNemar test
    pred_a = [True, True, False, True, False, True, True, False]
    pred_b = [True, False, False, True, True, True, False, False]
    result = mcnemar_test(pred_a, pred_b)
    print(f"  ✓ McNemar test: p={result.p_value:.4f}, significant={result.significant}")

    # Test Wilcoxon signed-rank
    scores_a = [0.8, 0.85, 0.75, 0.9, 0.82]
    scores_b = [0.7, 0.8, 0.72, 0.85, 0.78]
    result = wilcoxon_signed_rank(scores_a, scores_b)
    print(f"  ✓ Wilcoxon test: statistic={result.statistic:.4f}, p={result.p_value:.4f}")

    print("  ✓ Statistical analysis tests passed")


def test_pareto_analysis():
    """Test Pareto frontier computation."""
    print("Testing Pareto analysis...")

    # Create test points using actual ParetoPoint fields
    points = [
        ParetoPoint(model="ModelA", prompt="P0", f1=0.75, cost_per_kloc=0.01, time_per_kloc=1.0),
        ParetoPoint(model="ModelA", prompt="P1", f1=0.82, cost_per_kloc=0.05, time_per_kloc=2.0),
        ParetoPoint(model="ModelA", prompt="P2", f1=0.85, cost_per_kloc=0.10, time_per_kloc=3.0),
        ParetoPoint(model="ModelB", prompt="P0", f1=0.70, cost_per_kloc=0.03, time_per_kloc=1.5),  # dominated
        ParetoPoint(model="ModelB", prompt="P1", f1=0.88, cost_per_kloc=0.08, time_per_kloc=2.5),
    ]

    frontier = compute_pareto_frontier(points, maximize_y=True)
    frontier_labels = [p.label for p in frontier]

    # ModelB+P0 should be dominated (higher cost, lower accuracy than ModelA+P0)
    assert "ModelB+P0" not in frontier_labels
    print(f"  ✓ Dominated point correctly excluded")

    # ModelA+P0 should be on frontier (lowest cost at its accuracy level)
    assert "ModelA+P0" in frontier_labels
    print(f"  ✓ Pareto frontier: {frontier_labels}")

    print("  ✓ Pareto analysis tests passed")


def test_latex_table_generation():
    """Test LaTeX table generation."""
    print("Testing LaTeX table generation...")

    import pandas as pd

    # Create sample results
    data = {
        'Model': ['Claude', 'GPT-4.5', 'Claude', 'GPT-4.5'],
        'Prompt': ['P0', 'P0', 'P1', 'P1'],
        'F1': [0.75, 0.72, 0.82, 0.80],
        'Cost': [0.01, 0.02, 0.03, 0.04],
    }
    df = pd.DataFrame(data)

    latex = generate_latex_table(
        df,
        caption="Vulnerability Detection Results",
        label="tab:results",
        highlight_best=['F1']
    )

    assert "\\begin{table}" in latex
    assert "\\caption{" in latex
    assert "\\label{tab:results}" in latex
    assert "Claude" in latex
    print("  ✓ Generated valid LaTeX table")

    print("  ✓ LaTeX table generation tests passed")


def test_aggregation():
    """Test results aggregation."""
    print("Testing results aggregation...")

    # Create mock experiment results in the expected format
    results = {
        "runs": [
            {
                'model': 'claude-opus',
                'prompt': 'P0',
                'dataset': 'smartbugs',
                'contract_id': 'test1',
                'finding_count': 2,
                'cost_usd': 0.01,
                'tokens': 500,
                'latency_s': 1.5,
            },
            {
                'model': 'claude-opus',
                'prompt': 'P0',
                'dataset': 'smartbugs',
                'contract_id': 'test2',
                'finding_count': 3,
                'cost_usd': 0.01,
                'tokens': 600,
                'latency_s': 1.8,
            },
            {
                'model': 'gpt-4.5',
                'prompt': 'P0',
                'dataset': 'smartbugs',
                'contract_id': 'test1',
                'finding_count': 2,
                'cost_usd': 0.02,
                'tokens': 550,
                'latency_s': 2.0,
            },
        ]
    }

    # Test model/prompt aggregation
    df = aggregate_by_model_prompt(results)
    assert len(df) == 2  # claude-opus/P0 and gpt-4.5/P0

    claude_row = df[df['model'] == 'claude-opus'].iloc[0]
    assert claude_row['api_calls'] == 2
    assert claude_row['avg_findings_per_contract'] == 2.5
    print(f"  ✓ Model/prompt aggregation: {len(df)} groups")

    print("  ✓ Aggregation tests passed")


def test_full_pipeline_mock():
    """Test full pipeline with mock LLM."""
    print("Testing full pipeline (mocked)...")

    # Create test dataset
    dataset = create_test_dataset()
    samples = dataset.load()[:2]  # Just 2 samples for speed

    # Create mock adapter with realistic responses
    mock_responses = [
        json.dumps([{
            "category": "reentrancy",
            "title": "Reentrancy",
            "severity": "high",
            "confidence": 0.9,
        }]),
        json.dumps([{
            "category": "access_control",
            "title": "Authorization through tx.origin",
            "severity": "medium",
            "confidence": 0.85
        }]),
    ]
    adapter = MockLLMAdapter(mock_responses)

    # Create prompt registry
    prompt_root = Path(__file__).parent.parent / 'prompts'
    registry = PromptRegistry(prompt_root)

    results = []
    for sample in samples:
        # Render prompt using p0_baseline.j2 (the actual template name)
        prompt = registry.render('p0_baseline.j2', contract_source=sample.source)

        # Generate (mocked)
        response = adapter.generate(prompt)

        # Parse findings
        findings, _ = parse_findings(response.text)

        # Compute metrics
        ground_truth = sample.vulnerabilities
        metrics = compute_metrics(findings, ground_truth)

        results.append({
            'contract_id': sample.id,
            'findings': len(findings),
            'ground_truth': len(ground_truth),
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1': metrics.f1,
        })

    print(f"  ✓ Processed {len(results)} contracts")
    for r in results:
        print(f"    - {r['contract_id']}: F1={r['f1']:.2f} (found {r['findings']}/{r['ground_truth']})")

    print("  ✓ Full pipeline tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SC_PROMPT_EVAL COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()

    tests = [
        test_prompt_registry,
        test_findings_parser,
        test_metrics_computation,
        test_voting_aggregation,
        test_leave_one_out_exclusion,
        test_rag_system,
        test_mutation_operators,
        test_dataset_loader,
        test_statistical_analysis,
        test_pareto_analysis,
        test_latex_table_generation,
        test_aggregation,
        test_full_pipeline_mock,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
