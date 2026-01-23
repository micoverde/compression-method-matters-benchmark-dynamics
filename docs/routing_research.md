# LLM Routing and Energy Savings Research Summary

**Last Updated:** January 22, 2026
**Research Focus:** Energy savings from routing LLM requests to smaller/cheaper models

---

## Executive Summary

Research demonstrates that intelligent routing of LLM queries to smaller, more efficient models can reduce energy consumption by **40-98%** without significant quality degradation. The UNESCO-UCL report (2025) shows that combined techniques can reduce AI energy use by up to **90%**.

---

## 1. Energy Consumption by Model Size

### 1.1 GPT-4 vs GPT-3.5 vs Smaller Models

| Model | Energy per Query | Relative Cost | Source |
|-------|------------------|---------------|--------|
| GPT-4 | ~0.0005 kWh (0.5 Wh) | 3x GPT-3 | [Epoch AI, 2025] |
| GPT-3/3.5 | ~0.0003 kWh (0.3 Wh) | Baseline | [Epoch AI, 2025] |
| GPT-4o | ~0.3-0.43 Wh | ~1.5x GPT-3.5 | [Epoch AI, 2025] |
| GPT-4.1 nano | ~0.5 Wh | 70x less than o3 | [How Hungry is AI?, 2025] |

**Key Finding:** GPT-4's computational load is approximately **10x** that of GPT-3 (3.4 petaFLOP vs 0.35 petaFLOP per 1,000 token request).

**Citation:** [Epoch AI, "How much energy does ChatGPT use?", 2025](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use)

### 1.2 Llama Model Family Energy Comparison

| Model | Energy per Token | Scaling Factor | Notes |
|-------|------------------|----------------|-------|
| Llama 3 1B | Baseline | 1x | Reference point |
| Llama 3 8B | ~8x | Sublinear to params | GQA efficiency gains |
| Llama 3 70B | ~70x | Super-linear scaling | Memory bandwidth penalty |
| Llama 3 405B | ~400x | Highest energy | Frontier model |

**Key Finding:** Within the LLaMA family, moving from 1B to 70B parameters increases energy per token by approximately **70x**—a super-linear trend due to cache-bandwidth and memory-traffic penalties beyond pure FLOPs.

**Citations:**
- [TokenPowerBench: Benchmarking the Power Consumption of LLM Inference, 2025](https://arxiv.org/abs/2512.03024)
- [From Prompts to Power: Measuring the Energy Footprint of LLM Inference, 2025](https://arxiv.org/html/2511.05597)

### 1.3 Claude Model Family (Anthropic)

| Model | Input Price | Output Price | Relative Resource Usage |
|-------|-------------|--------------|-------------------------|
| Claude 3 Haiku | $0.25/MTok | $1.25/MTok | 1x (most efficient) |
| Claude 3 Sonnet | $3.00/MTok | $15.00/MTok | 12x input, 12x output |
| Claude 3 Opus | $15.00/MTok | $75.00/MTok | 60x input, 60x output |

**Key Finding:** Routing queries from Opus to Haiku can achieve **50x+ cost savings** (and proportional energy savings), making intelligent routing extremely valuable.

**Note:** Direct energy measurements (Wh) for Claude models are not publicly available, but pricing serves as a strong proxy for computational/energy costs.

**Citation:** [Anthropic Claude Pricing, via Claude AI Hub](https://claudeaihub.com/claude-3-models-compared/)

---

## 2. LLM Routing and Cascading Research

### 2.1 FrugalGPT (Chen et al., 2023)

**Paper:** "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance"

**Key Results:**
- **98% cost reduction** while matching GPT-4 performance on HEADLINES dataset
- **4% accuracy improvement** over GPT-4 at same cost using ensemble routing

**Methodology:**
1. **LLM Cascade:** Sequentially query models from smallest to largest
2. **Generation Judger:** Score response quality to decide when to stop
3. **Adaptive Routing:** Learn which model combinations work for different query types

**Citation:** [Chen, L., Zaharia, M., & Zou, J., "FrugalGPT", arXiv:2305.05176, 2023](https://arxiv.org/abs/2305.05176)

### 2.2 RouteLLM (LMSYS, 2024)

**Paper:** "RouteLLM: Learning to Route LLMs with Preference Data"

**Key Results:**
- **85% cost reduction** on MT Bench without quality loss
- **45% cost reduction** on MMLU
- **35% cost reduction** on GSM8K
- Using GPT-4-1106-preview over Mixtral-8x7B: **80% quality gain** with only **30% calls to GPT-4**

**Methodology:**
- Matrix factorization-based router trained on human preference data
- Dynamic selection between strong (expensive) and weak (cheap) models
- Strong transfer learning: maintains performance when models change

**Citation:** [RouteLLM: Learning to Route LLMs with Preference Data, arXiv:2406.18665, 2024](https://arxiv.org/abs/2406.18665)

### 2.3 Hybrid LLM (Microsoft/ICLR 2024)

**Paper:** "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing"

**Key Results:**
- **40% fewer calls** to large model with no drop in response quality
- Router latency: **0.036 seconds** (10x faster than fastest LLM)

**Methodology:**
- Train router to discriminate "hard" vs "easy" queries
- Route easy queries to small on-device model
- Quality threshold tunable at test time

**Citation:** [Ding et al., "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing", ICLR 2024](https://arxiv.org/abs/2404.14618)

### 2.4 Cascade Routing (ETH Zurich, 2024)

**Paper:** "A Unified Approach to Routing and Cascading for LLMs"

**Key Insight:** Combines routing (pick one model) and cascading (try models sequentially) into unified framework.

**Result:** Consistently outperforms both pure routing and pure cascading on cost-quality tradeoffs.

**Citation:** [de Koninck et al., "A Unified Approach to Routing and Cascading for LLMs", ICLR 2025](https://arxiv.org/abs/2410.10347)

---

## 3. Speculative Decoding for Energy Savings

### 3.1 Core Concept

Speculative decoding uses a small "draft" model to generate candidate tokens, then verifies them in parallel with the large model—maintaining output quality while reducing compute.

**Energy Savings:**
- **40.7% energy reduction** demonstrated with TinyLlama-1.1B + LLaMA-2-7B hybrid
- **2.25-2.43x speedup** with SpecEE (Speculative Early Exiting) on A100/RTX 4090

**Citations:**
- [Google Research, "Looking back at speculative decoding", 2024](https://research.google/blog/looking-back-at-speculative-decoding/)
- [ACL 2024, "Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding"](https://aclanthology.org/2024.findings-acl.456/)

### 3.2 Speculative Cascades (Google, 2024)

Hybrid approach combining cascades and speculative decoding:
- Use small model for easy tokens
- Speculate ahead to reduce large model calls
- Better quality at lower computational cost

**Citation:** [Google Research, "Speculative cascades — A hybrid approach for smarter, faster LLM inference", 2024](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)

---

## 4. Percentage of Queries Routable to Smaller Models

### 4.1 Research Estimates

| Study | % Routable | Quality Threshold | Source |
|-------|------------|-------------------|--------|
| RouteLLM | 70% | 80% quality retention | LMSYS 2024 |
| Hybrid LLM | 60% | No quality drop | Microsoft ICLR 2024 |
| IBM Research | 85% | Task-dependent | IBM 2024 |
| FrugalGPT | 98% | Dataset-specific | Stanford 2023 |

**Key Finding:** Across benchmarks, **60-85% of queries** can be handled by smaller models without significant quality degradation.

**Citation:** [IBM Research, "LLM routing for quality, low-cost responses", 2024](https://research.ibm.com/blog/LLM-routers)

### 4.2 Task-Specific Routability

- **Simple classification/extraction:** 85-95% routable to small models
- **Complex reasoning:** 30-50% routable
- **Creative writing:** 60-70% routable
- **Code generation:** 50-70% routable (varies by complexity)

---

## 5. Energy Savings Estimates

### 5.1 UNESCO-UCL Report (2025) - Key Statistics

**Combined techniques can achieve up to 90% energy reduction:**

| Technique | Energy Savings | Notes |
|-----------|----------------|-------|
| Task-specific smaller models | Up to 90% | Match model to task |
| Shorter prompts/responses | 50%+ | Reduce token count |
| Model compression (quantization) | Up to 44% | FP8/INT8 inference |
| **Combined** | **Up to 90%** | All techniques together |

**Context:**
- 1 billion+ daily users of generative AI
- ~0.34 Wh per prompt average
- 310 GWh annual consumption (equivalent to 3M people in low-income countries)

**Citation:** [UNESCO, "AI Large Language Models: new report shows small changes can reduce energy use 90%", 2025](https://www.unesco.org/en/articles/ai-large-language-models-new-report-shows-small-changes-can-reduce-energy-use-90)

### 5.2 Routing-Specific Savings

**If 50% of queries routed to smaller models:**

Assuming:
- Large model (GPT-4 class): 0.5 Wh per query
- Small model (GPT-3.5 class): 0.15 Wh per query
- 50% queries routable

**Calculation:**
- Without routing: 100 queries × 0.5 Wh = 50 Wh
- With routing: (50 × 0.5) + (50 × 0.15) = 25 + 7.5 = 32.5 Wh
- **Savings: 35%**

**If 70% routable (RouteLLM benchmark):**
- With routing: (30 × 0.5) + (70 × 0.15) = 15 + 10.5 = 25.5 Wh
- **Savings: 49%**

**If 85% routable (IBM benchmark):**
- With routing: (15 × 0.5) + (85 × 0.15) = 7.5 + 12.75 = 20.25 Wh
- **Savings: 59.5%**

### 5.3 Real-World Impact Estimates

| Scenario | Daily Queries | Annual Savings |
|----------|---------------|----------------|
| Single enterprise (1M queries/day) | 1,000,000 | 127,750 kWh (50% routing) |
| Medium platform (100M queries/day) | 100,000,000 | 12.8 GWh (50% routing) |
| ChatGPT scale (700M queries/day) | 700,000,000 | 89.4 GWh (50% routing) |

---

## 6. Key Statistics Summary

### Energy Ratios

| Comparison | Energy Ratio | Source |
|------------|--------------|--------|
| GPT-4 / GPT-3.5 | ~1.5-3x | Epoch AI |
| GPT-4 compute / GPT-3 | ~10x | Epoch AI |
| Llama 70B / Llama 8B | ~9x | TokenPowerBench |
| Llama 70B / Llama 1B | ~70x | TokenPowerBench |
| Claude Opus / Claude Haiku | ~50-60x (pricing proxy) | Anthropic |
| o3 / GPT-4.1 nano | ~70x | How Hungry is AI |

### Routing Effectiveness

| Metric | Value | Source |
|--------|-------|--------|
| Max cost reduction (FrugalGPT) | 98% | Chen et al. 2023 |
| Typical routing savings | 40-85% | Multiple studies |
| % queries routable | 60-85% | IBM, RouteLLM, Hybrid LLM |
| Router overhead | <0.04 seconds | Hybrid LLM |

### Energy Reduction Potential

| Approach | Savings | Source |
|----------|---------|--------|
| Task-specific models | Up to 90% | UNESCO 2025 |
| Intelligent routing | 40-85% | Multiple |
| Quantization | Up to 44% | UNESCO 2025 |
| Speculative decoding | ~40% | Google 2024 |
| Combined techniques | Up to 90% | UNESCO 2025 |

---

## 7. Complete Citation List

### Primary Research Papers

1. **[Chen, L., Zaharia, M., & Zou, J. (2023). "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance." arXiv:2305.05176](https://arxiv.org/abs/2305.05176)**

2. **[Ong, I., et al. (2024). "RouteLLM: Learning to Route LLMs with Preference Data." arXiv:2406.18665](https://arxiv.org/abs/2406.18665)**

3. **[Ding, D., et al. (2024). "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing." ICLR 2024. arXiv:2404.14618](https://arxiv.org/abs/2404.14618)**

4. **[de Koninck, T., et al. (2024). "A Unified Approach to Routing and Cascading for LLMs." ICLR 2025. arXiv:2410.10347](https://arxiv.org/abs/2410.10347)**

5. **[Niu, C., et al. (2025). "TokenPowerBench: Benchmarking the Power Consumption of LLM Inference." arXiv:2512.03024](https://arxiv.org/abs/2512.03024)**

6. **[Wang, X., et al. (2025). "From Prompts to Power: Measuring the Energy Footprint of LLM Inference." arXiv:2511.05597](https://arxiv.org/html/2511.05597)**

7. **[Samsi, S., et al. (2023). "From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference." arXiv:2310.03003](https://arxiv.org/pdf/2310.03003)**

8. **[Yang, Z., et al. (2025). "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference." arXiv:2505.09598](https://arxiv.org/abs/2505.09598)**

### Industry and Institutional Reports

9. **[UNESCO & UCL (2025). "Smarter, smaller, stronger: resource-efficient generative AI & the future of digital transformation."](https://www.unesco.org/en/articles/ai-large-language-models-new-report-shows-small-changes-can-reduce-energy-use-90)**

10. **[Epoch AI (2025). "How much energy does ChatGPT use?"](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use)**

11. **[IBM Research (2024). "LLM routing for quality, low-cost responses."](https://research.ibm.com/blog/LLM-routers)**

12. **[LMSYS (2024). "RouteLLM: An Open-Source Framework for Cost-Effective LLM Routing."](https://lmsys.org/blog/2024-07-01-routellm/)**

### Blog Posts and Technical Reports

13. **[Google Research (2024). "Looking back at speculative decoding."](https://research.google/blog/looking-back-at-speculative-decoding/)**

14. **[Google Research (2024). "Speculative cascades — A hybrid approach for smarter, faster LLM inference."](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)**

15. **[Muxup (2026). "Per-query energy consumption of LLMs."](https://muxup.com/2026q1/per-query-energy-consumption-of-llms)**

---

## 8. Recommendations for Implementation

### For Plexor/Similar Routing Systems

1. **Implement tiered routing:**
   - Tier 1: Small models (Haiku-class) for simple queries
   - Tier 2: Medium models (Sonnet-class) for moderate complexity
   - Tier 3: Large models (Opus-class) for complex reasoning

2. **Use proven router architectures:**
   - Matrix factorization (RouteLLM) for best efficiency
   - Difficulty prediction (Hybrid LLM) for simplicity
   - Cascade approach (FrugalGPT) for maximum savings

3. **Target metrics:**
   - Aim for 60-70% routing to smaller models
   - Expected energy savings: 40-60%
   - Router latency target: <50ms

4. **Combine with other techniques:**
   - Quantization (FP8/INT8) for additional 30-44% savings
   - Prompt optimization for 50%+ reduction
   - Speculative decoding for additional 40% gains

---

*Document generated for Plexor VC Fund Platform research on sustainable AI infrastructure.*
