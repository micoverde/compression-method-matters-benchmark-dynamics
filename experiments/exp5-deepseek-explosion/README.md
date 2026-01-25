# Experiment 5: DeepSeek Output Token Explosion Under Prompt Compression

## Critical Note: Contradictory Findings

**WARNING**: The pilot results CONTRADICT the 38× explosion finding from Article 4. This requires careful investigation before drawing conclusions.

| Source | DeepSeek at r=0.3 | Direction |
|--------|-------------------|-----------|
| Article 4 (N=16,270) | 798 tokens (38× baseline) | EXPLOSION |
| Exp 5 Pilot (N=10) | 68 tokens (0.55× baseline) | REDUCTION |

**We must reconcile this 69× discrepancy before updating any claims.**

## Possible Explanations for Discrepancy

1. **Benchmark difference**: Article 4 used 73% MBPP, pilot used 100% HumanEval
2. **Sample size**: N=10 is statistically underpowered (CV=96% requires N>100)
3. **Compression artifacts**: Different prompt structures may trigger different behaviors
4. **Model version drift**: DeepSeek API may have been updated
5. **Methodological error**: Pilot or Article 4 may have implementation bugs

## Research Question

Why does DeepSeek-Chat exhibit significantly different output behavior under prompt compression compared to other LLM providers?

**Sub-question (NEW)**: Why do pilot results contradict Article 4's findings?

## Hypotheses

| ID | Hypothesis | Status |
|----|------------|--------|
| H1 | MoE/MLA Architecture Fragility | UNCLEAR - pilot shows no explosion |
| H2 | GRPO Training (Primary) | UNCLEAR - needs MBPP replication |
| H3 | Hidden System Prompts | TESTABLE - pilot used consistent system prompt |
| H4 | Tokenizer Differences | PARTIAL - tokenizer analysis pending |
| H5 | Uncertainty Handling | UNCLEAR - short prompts may not trigger |

## Experimental Design

### Pilot Study (Completed - CONTRADICTORY RESULTS)
- **Date**: January 25, 2026
- **Trials**: 100 (10 samples × 2 ratios × 5 models)
- **Models**: DeepSeek-Chat, GPT-4o-mini, Claude-3.5-Sonnet*, Gemini-1.5-Flash*, Mistral-Large
- **Compression Ratios**: r=1.0 (baseline), r=0.3 (aggressive)
- **Benchmark**: HumanEval (first 10 problems)
- **Compression Method**: First-N-words truncation (matching Article 4)

*Claude and Gemini failed due to API issues (40/100 trials failed)

### Compression Algorithm (Matching Article 4)
```python
def compress_prompt(prompt: str, ratio: float) -> str:
    """First N words only - destroys trailing instruction markers"""
    if ratio >= 1.0:
        return prompt
    words = prompt.split()
    keep_count = max(1, int(len(words) * ratio))
    return " ".join(words[:keep_count])
```

## Results Summary

### Pilot Findings

| Model | Baseline (r=1.0) | Compressed (r=0.3) | Change |
|-------|------------------|-------------------|--------|
| DeepSeek-Chat | 122.8 tokens | 67.8 tokens | **0.55× (DECREASED)** |
| GPT-4o-mini | 68.3 tokens | 52.3 tokens | 0.77× (decreased) |
| Mistral-Large | 64.1 tokens | 61.9 tokens | 0.97× (stable) |

### Statistical Limitations
- N=10 per condition is severely underpowered
- DeepSeek CV=95.9% requires N>255 for 80% power
- Cannot draw population-level conclusions from this pilot

## Required Next Steps

1. **MBPP Replication**: Run pilot with MBPP prompts (Article 4's primary benchmark)
2. **Larger Sample**: Increase to N=50 minimum per condition
3. **Article 4 Audit**: Review Article 4's raw data for potential issues
4. **Methodology Comparison**: Compare exact prompts between Article 4 and pilot

## Data Files

| File | Description |
|------|-------------|
| `data/pilot_results_20260125_052638.jsonl` | Raw pilot results (60 successful, 40 failed) |

## Reproduction

```bash
pip install httpx

export DEEPSEEK_API_KEY="..."
export OPENAI_API_KEY="..."
export MISTRAL_API_KEY="..."

python pilot_experiment.py
```

## Team

- **Principal Researcher**: Warren Johnson, Microsoft Research
- **PhD Team**: 6 researchers (statistical rigor, experimental design, hypothesis testing)

## Status

- [x] Pilot experiment complete
- [x] **CRITICAL**: Results contradict Article 4 - investigation required
- [ ] MBPP replication study
- [ ] Article 4 methodology audit
- [ ] Reconciliation of contradictory findings
- [ ] Updated claims based on reconciled data

## Citation

```bibtex
@misc{johnson2026deepseek,
  title={Provider-Dependent Output Behavior Under Prompt Compression: A Replication Study},
  author={Johnson, Warren and Microsoft Research Team},
  year={2026},
  note={Experiment 5 - Investigating contradictions with Article 4}
}
```
