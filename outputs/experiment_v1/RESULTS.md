# Experiment V1 Results - Smart Contract Vulnerability Detection

**Dataset:** SmartBugs Curated (143 contracts)
**Models:** o4-mini (Azure OpenAI) for P0-P5, Local Mixtral-8x7B for P6
**Date:** 2026-02-05
**Last Updated:** 2026-02-05

## Main Results (Category-Level Metrics)

These are the official results matching the paper. Metrics are computed using category-level matching across all contracts.

### P0-P5: o4-mini (Azure OpenAI)

| Prompt | Total | Valid | Err | Findings | TP | FP | FN | Precision | Recall | F1 | Cost |
|--------|------:|------:|----:|---------:|---:|---:|---:|----------:|-------:|---:|-----:|
| P0_baseline | 143 | 141 | 2 | 360 | 86 | 274 | 57 | 23.9% | 60.1% | 34.2% | $1.90 |
| P1_icl | 143 | 142 | 1 | 555 | 138 | 417 | 5 | 24.9% | **96.5%** | 39.5% | ~$19* |
| P2_structured | 143 | 136 | 7 | 297 | 131 | 166 | 12 | 44.1% | 91.6% | 59.6% | $2.39 |
| **P3_smartguard** | 143 | 142 | 1 | 202 | 125 | 77 | 18 | **61.9%** | 87.4% | **72.5%** | ~$43* |
| P4_tool_augmented | 143 | 134 | 9 | 112 | 33 | 79 | 101 | 29.5% | 24.6% | 26.8% | $4.13 |
| P5_smartaudit | 143 | 143 | 0 | 500 | 123 | 377 | 20 | 24.6% | 86.0% | 38.3% | $24.27 |

*Estimated based on API call count (P1: 1430 calls, P3: 3265 calls @ $0.0133/call)

### P6: Local Fine-tuned CodeLlama-13b (iAudit Pipeline)

**Note:** P6 was trained on Qian dataset which lacks `front_running`, `short_addresses`, and `other` categories. Results exclude these 8 contracts for fair comparison.

| Matching | Total | Valid | Err | Findings | TP | FP | FN | Precision | Recall | F1 | Cost |
|----------|------:|------:|----:|---------:|---:|---:|---:|----------:|-------:|---:|-----:|
| Qian-compatible | 135 | 135 | 0 | 135 | 63 | 72 | 72 | 46.7% | 46.7% | 46.7% | Local |

**P6 Category Accuracy (Qian-compatible categories only):**

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| bad_randomness | 7 | 8 | **87.5%** |
| reentrancy | 27 | 31 | **87.1%** |
| denial_of_service | 4 | 6 | 66.7% |
| arithmetic | 9 | 15 | 60.0% |
| access_control | 9 | 18 | 50.0% |
| time_manipulation | 1 | 5 | 20.0% |
| unchecked_low_level_calls | 6 | 52 | 11.5% |

**Qian -> DASP Category Mapping:**

| Qian Dataset | DASP/SmartBugs |
|--------------|----------------|
| reentrancy (RE) | reentrancy |
| integer overflow (OF) | arithmetic |
| timestamp dependency (TP) | time_manipulation |
| unchecked external call (UC) | unchecked_low_level_calls |
| block number dependency (BN) | bad_randomness |
| dangerous delegatecall (DE) | access_control |
| ether frozen (EF) | denial_of_service |
| ether strict equality (SE) | denial_of_service |
| *(not in Qian)* | front_running |
| *(not in Qian)* | short_addresses |
| *(not in Qian)* | other |

## Error Summary

| Prompt | Errors | Error Type | Contracts Affected |
|--------|-------:|------------|-------------------|
| P0_baseline | 2 | 1 empty findings, 1 JSON parse | `front_running_FindThisHash.sol`, `unchecked_low_level_calls_0xec329ffc97d75fe03428ae155fc7793431487f63.sol` |
| P1_icl | 1 | Parse error (short_addresses) | `unchecked_low_level_calls_0x663e4229142a27f00bafb5d087e1e730648314c3.sol` |
| P2_structured | 7 | Empty findings (detection failures) | `access_control_parity_wallet_bug_2.sol`, `bad_randomness_blackjack.sol`, `front_running_ERC20.sol`, `front_running_FindThisHash.sol`, `front_running_eth_tx_order_dependence_minimal.sol`, `time_manipulation_timed_crowdsale.sol`, `unchecked_low_level_calls_etherpot_lotto.sol` |
| P3_smartguard | 1 | Malformed JSON ("fifty" instead of 50) | `unchecked_low_level_calls_0x84d9ec85c9c568eb332b7226a8f826d897e0a4a8.sol` |
| P4_tool_augmented | 9 | Azure content filter blocked | 9 contracts blocked as "jailbreak" |
| P5_smartaudit | 0 | None | - |
| P6_iaudit | 0 | None | - |

**Total Errors:** 20 (2.0% of 992 total runs)

## Key Findings

### Best Performers
- **Best F1 Score:** P3_smartguard (**72.5%**)
- **Best Precision:** P3_smartguard (**61.9%**)
- **Best Recall:** P1_icl (**96.5%**)

### Prompt Analysis

1. **P0_baseline** - Simple single-pass prompt with JSON output
   - Moderate recall (60.1%), low precision (23.9%)
   - F1: 34.2%
   - Cost-effective baseline ($1.90)

2. **P1_icl** - Per-vulnerability-type ICL with 10 API calls per contract
   - Highest recall (96.5%) - finds almost everything
   - Very low precision (24.9%) - many false positives
   - F1: 39.5%
   - Expensive due to multiple API calls (~$19)

3. **P2_structured** - Chain-of-thought with systematic audit checklist
   - High recall (91.6%) with decent precision (44.1%)
   - Second-best F1 (59.6%)
   - Most cost-effective for performance ($2.39)

4. **P3_smartguard** - RAG-enhanced with pattern retrieval and self-check
   - **Winner**: Best F1 (72.5%) and precision (61.9%)
   - Strong recall (87.4%)
   - Higher cost due to RAG + self-check iterations (~$43)

5. **P4_tool_augmented** - GPTScan-aligned: pre-filter + 3-stage LLM + static confirmation
   - Poor performance: F1 26.8%, Recall 24.6%
   - 9 contracts failed due to Azure content filter
   - Static confirmation rejected many valid LLM findings

6. **P5_smartaudit** - LLM-SmartAudit multi-agent with 4 specialized agents
   - High recall (86.0%) but low precision (24.6%)
   - F1: 38.3%
   - Most expensive ($24.27) - does not justify cost

7. **P6_iaudit** - iAudit fine-tuned pipeline (Detector + Reasoner + Ranker-Critic)
   - F1: **46.7%** (135 contracts, excluding non-Qian categories)
   - Only 1 finding per contract
   - **Excellent on:** reentrancy (87.1%), bad_randomness (87.5%)
   - **Poor on:** unchecked_low_level_calls (11.5%), time_manipulation (20.0%)
   - Local execution (no API cost)
   - Note: Trained on Qian dataset taxonomy (excludes front_running, short_addresses, other)

## Cost-Effectiveness Analysis

Ordered by F1/$ (descending):

| Prompt | F1 | Cost | F1/$ |
|--------|---:|-----:|-----:|
| P2_structured | 59.6% | $2.39 | **24.9** |
| P0_baseline | 34.2% | $1.90 | 18.0 |
| P4_tool_augmented | 26.8% | $4.13 | 6.5 |
| P1_icl | 39.5% | ~$19 | 2.1 |
| P3_smartguard | 72.5% | ~$43 | 1.7 |
| P5_smartaudit | 38.3% | $24.27 | 1.6 |
| P6_iaudit | 46.7%* | Local | Inf |

*P6 evaluated on 135 Qian-compatible contracts only

**Most Cost-Effective (API):** P2_structured (24.9 F1 points per dollar)
**Most Cost-Effective (Local):** P6_iaudit (no API cost, F1: 46.7%)

## Notes

- **Metrics Type:** Category-level (vulnerability type match between prediction and ground truth)
- **Ground Truth:** SmartBugs Curated dataset categories (reentrancy, access_control, arithmetic, etc.)
- **P4 Issues:**
  - Azure content filter flagged GPTScan's "attack scenario" language as potential jailbreak
  - Static confirmation heuristics too aggressive
- **Cost Estimation:** P1 and P3 did not track cost; estimated from API call count using P0 baseline ($0.0133/call)

## Recommendations

1. **For highest accuracy:** Use P3_smartguard (F1: 72.5%)
2. **For best cost-effectiveness:** Use P2_structured (F1: 59.6% @ $2.39)
3. **For maximum coverage:** Use P1_icl (Recall: 96.5%)
4. **For offline/local use:** P6_iaudit (F1: 46.7%, no API cost, excellent on reentrancy/bad_randomness)
5. **Avoid:** P4_tool_augmented (content filter issues) and P5_smartaudit (poor ROI)

## Files

- `RESULTS.md` - This summary
- `final_results.json` - Machine-readable metrics matching this summary
- `smartbugs_curated/o4-mini/P*/` - Individual contract outputs
- Each contract has `run_0_raw.json` and `run_0_parsed.json`
