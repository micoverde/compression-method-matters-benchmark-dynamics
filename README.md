# Green Tokens: Measuring the Energy Cost of LLM Prompt Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2026.XXXXX)

This repository contains the experimental data, analysis scripts, and measurement tools for the paper:

> **"Green Tokens: Quantifying the Energy Impact of Prompt Compression in Large Language Model Inference"**

## Key Findings

1. **Output Length Paradox**: Prompt compression reduces token count by 17%, but saves only **4.9% energy** due to compensatory model behavior (longer outputs, harder inference).

2. **Per-Token Energy Penalty**: Minimal prompts consume **15.2% more energy per token** than full prompts (p < 0.01 with autocorrelation correction).

3. **Routing Dominates Compression**: Model routing provides **7× greater CO₂ savings** than compression (36% vs 4.9%).

4. **Global Impact**: Combined compression + routing could save **12.8 million MT CO₂/year** (equivalent to removing 2.8 million cars).

## Repository Structure

```
green-tokens-energy-research/
├── data/
│   ├── phase1/              # API-based energy proxy measurements
│   └── phase2/              # Direct GPU power measurements (NVML)
│       ├── phase2_20260123_011028.jsonl  # Raw trial data (600 trials)
│       ├── phase2_summary.json           # Aggregated results
│       └── idle_test.csv                 # GPU idle power baseline
├── scripts/
│   ├── article4_experiment_runner.py     # Phase 1 experiment runner
│   ├── power_monitor.py                  # NVML power monitoring class
│   ├── calibration_protocol.py           # Energy proxy validation
│   ├── co2_calculations.py               # Datacenter CO₂ projections
│   └── phase2_statistical_analysis.py    # Statistical analysis
├── analysis/
│   ├── statistical_analysis.json         # Full statistical results
│   └── co2_savings_analysis.json         # CO₂ savings scenarios
├── docs/
│   ├── datacenter_research.md            # AI energy consumption research
│   └── routing_research.md               # Model routing literature review
└── figures/                              # Generated figures for paper
```

## Experiment Details

### Phase 2: Direct GPU Energy Measurement

- **Hardware**: NVIDIA RTX 4090 (24GB VRAM)
- **Model**: TinyLlama-1.1B-Chat-v1.0
- **Inference**: vLLM v0.14.0
- **Power Monitoring**: NVML at 10Hz sampling rate
- **Trials**: 600 (5 problems × 4 compression levels × 30 repetitions)

### Compression Levels

| Level | Description | Compression Ratio |
|-------|-------------|-------------------|
| Full | Complete prompt with examples | 1.0 (baseline) |
| Medium | Reduced examples | ~0.7 |
| Terse | Minimal context | ~0.5 |
| Minimal | Function signature only | ~0.3 |

## Reproducing Results

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running Phase 2 Analysis

```bash
# Statistical analysis
python scripts/phase2_statistical_analysis.py

# CO2 savings calculations
python scripts/co2_calculations.py
```

### Data Format

Each trial in `phase2_20260123_011028.jsonl` contains:

```json
{
  "trial": 1,
  "timestamp": "2026-01-23T01:10:30.123456",
  "problem": "HE_0",
  "compression": "full",
  "prompt_tokens": 49,
  "completion_tokens": 256,
  "total_tokens": 305,
  "inference_energy_j": 115.03,
  "avg_power_w": 194.3,
  "peak_power_w": 210.5,
  "duration_s": 0.592,
  "success": true
}
```

## Statistical Methods

- **Autocorrelation Correction**: Newey-West HAC estimator for serial correlation in power measurements
- **Multiple Comparisons**: Bonferroni correction for pairwise t-tests
- **Effect Sizes**: Cohen's d reported for all comparisons

## Citation

If you use this data or code, please cite:

```bibtex
@article{greentokens2026,
  title={Green Tokens: Quantifying the Energy Impact of Prompt Compression in Large Language Model Inference},
  author={[Authors]},
  journal={arXiv preprint arXiv:2026.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA for NVML power monitoring tools
- The vLLM team for efficient inference
- TinyLlama authors for the open model
- IEA, Goldman Sachs, and EPA for energy/emissions data

## Contact

For questions about the data or methodology, please open an issue or contact the authors.
