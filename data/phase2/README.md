# Phase 2: Direct GPU Energy Measurement Data

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Date | 2026-01-23 |
| GPU | NVIDIA RTX 4090 (24GB) |
| Model | TinyLlama-1.1B-Chat-v1.0 |
| Inference Engine | vLLM v0.14.0 |
| Power Sampling | NVML at 10Hz |
| Total Trials | 600 |

## Files

### `phase2_20260123_011028.jsonl`

Raw experimental data in JSON Lines format. Each line is one trial.

**Schema:**
```json
{
  "trial": "int - Trial number (1-600)",
  "timestamp": "string - ISO 8601 timestamp",
  "problem": "string - HumanEval problem ID (HE_0 through HE_4)",
  "compression": "string - Compression level (full/medium/terse/minimal)",
  "prompt_len": "int - Prompt character length",
  "prompt_tokens": "int - Input tokens",
  "completion_tokens": "int - Output tokens",
  "total_tokens": "int - Total tokens (input + output)",
  "latency_s": "float - Inference latency in seconds",
  "success": "bool - Whether inference succeeded",
  "total_energy_j": "float - Total GPU energy (Joules)",
  "inference_energy_j": "float - Inference-only energy (above idle baseline)",
  "avg_power_w": "float - Average power draw (Watts)",
  "peak_power_w": "float - Peak power draw (Watts)",
  "duration_s": "float - Measurement duration (seconds)",
  "samples": "int - Number of power samples collected"
}
```

### `phase2_summary.json`

Aggregated results by compression level.

### `idle_test.csv`

GPU idle power baseline measurements (used for energy attribution).

## Compression Levels

| Level | Description | Avg Prompt Tokens |
|-------|-------------|-------------------|
| full | Complete prompt with docstring and examples | 49 |
| medium | Reduced context | 35 |
| terse | Minimal context | 32 |
| minimal | Function signature only | 19 |

## Key Results

| Compression | Energy (J) | Tokens | mJ/token |
|-------------|------------|--------|----------|
| full | 115.0 | 304.5 | 377.8 |
| medium | 112.0 | 274.8 | 407.8 |
| terse | 109.8 | 269.7 | 407.0 |
| minimal | 109.5 | 251.6 | 435.3 |

**Finding**: Minimal prompts use 15.2% MORE energy per token than full prompts.
