#!/usr/bin/env python3
"""
Experiment 5 Pilot - DeepSeek Token Explosion Validation
Tests 5 models with 20 samples each at r=1.0 and r=0.3
Total: 100 trials, ~$1 budget
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx

# Configuration
MODELS = {
    "deepseek-chat": {
        "provider": "deepseek",
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_API_KEY",
    },
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "key_env": "ANTHROPIC_API_KEY",
    },
    "gemini-1.5-flash": {
        "provider": "google",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "key_env": "GOOGLE_API_KEY",
    },
    "mistral-large-latest": {
        "provider": "mistral",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "key_env": "MISTRAL_API_KEY",
    },
}

COMPRESSION_RATIOS = [1.0, 0.3]
SAMPLES_PER_CONDITION = 10
MAX_TOKENS = 4096  # Increased from 1024 to avoid ceiling effects (Team rec)

# Minimal system prompt for consistency across providers (PhD 3 recommendation)
SYSTEM_PROMPT = "You are a code completion assistant. Output only the function implementation."

# Sample HumanEval-style prompts
SAMPLE_PROMPTS = [
    {"id": "HumanEval/0", "prompt": 'Complete this Python function. Only provide the implementation code.\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n\nImplementation:'},
    {"id": "HumanEval/1", "prompt": 'Complete this Python function. Only provide the implementation code.\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    """Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those groups into separate strings and return the list of those.\n    >>> separate_paren_groups("( ) (( )) (( )( ))")\n    ["()", "(())", "(()())"]\n    """\n\nImplementation:'},
    {"id": "HumanEval/2", "prompt": 'Complete this Python function. Only provide the implementation code.\n\ndef truncate_number(number: float) -> float:\n    """Given a positive floating point number, it can be decomposed into an integer part\n    and a decimal part. Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    """\n\nImplementation:'},
    {"id": "HumanEval/3", "prompt": 'Complete this Python function. Only provide the implementation code.\n\ndef below_zero(operations: List[int]) -> bool:\n    """You are given a list of deposit and withdrawal operations on a bank account.\n    Detect if at any point the balance goes below zero, return True, else False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    """\n\nImplementation:'},
    {"id": "HumanEval/4", "prompt": 'Complete this Python function. Only provide the implementation code.\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    """Calculate Mean Absolute Deviation around the mean of the dataset.\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    """\n\nImplementation:'},
    {"id": "HumanEval/5", "prompt": 'Complete this Python function. Only provide the implementation code.\n\ndef intersperse(numbers: List[int], delimiter: int) -> List[int]:\n    """Insert a number between every two consecutive elements of input list.\n    >>> intersperse([], 4)\n    []\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    """\n\nImplementation:'},
    {"id": "HumanEval/6", "prompt": 'Complete this Python function. Only provide the implementation code.\n\ndef parse_nested_parens(paren_string: str) -> List[int]:\n    """Return the deepest level of nesting of parentheses for each group.\n    >>> parse_nested_parens("(()()) ((())) () ((())()())")\n    [2, 3, 1, 3]\n    """\n\nImplementation:'},
    {"id": "HumanEval/7", "prompt": 'Complete this Python function. Only provide the implementation code.\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    """Filter a list of strings, returning only those containing the substring.\n    >>> filter_by_substring([], "a")\n    []\n    >>> filter_by_substring(["abc", "bacd", "cde", "array"], "a")\n    ["abc", "bacd", "array"]\n    """\n\nImplementation:'},
    {"id": "HumanEval/8", "prompt": 'Complete this Python function. Only provide the implementation code.\n\ndef sum_product(numbers: List[int]) -> Tuple[int, int]:\n    """Return a tuple of the sum and product of all integers in the list.\n    >>> sum_product([])\n    (0, 1)\n    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n    """\n\nImplementation:'},
    {"id": "HumanEval/9", "prompt": 'Complete this Python function. Only provide the implementation code.\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    """Return a list of rolling maximum element found until given moment.\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    """\n\nImplementation:'},
]


def compress_prompt(prompt: str, ratio: float) -> str:
    """Word-based compression matching Article 4 methodology.

    Uses FIRST N WORDS only (not first+last) to match Article 4's compression.
    This destroys instruction markers at end of prompt, triggering the
    verbose compensation behavior documented in Article 4.

    PhD 6 finding: The previous first+last method preserved instruction markers,
    which is why pilot showed 2.6x vs Article 4's 38x explosion.
    """
    if ratio >= 1.0:
        return prompt

    words = prompt.split()
    keep_count = max(1, int(len(words) * ratio))

    # Article 4 style: keep FIRST N words only (destroys trailing instructions)
    compressed_words = words[:keep_count]
    return " ".join(compressed_words)


async def call_openai(client: httpx.AsyncClient, model: str, prompt: str, api_key: str) -> dict:
    """Call OpenAI-compatible API (OpenAI, DeepSeek, Mistral)"""
    config = MODELS[model]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }

    start_time = time.time()
    response = await client.post(config["endpoint"], headers=headers, json=payload, timeout=120.0)
    latency_ms = int((time.time() - start_time) * 1000)

    if response.status_code != 200:
        return {"error": response.text, "latency_ms": latency_ms}

    data = response.json()

    return {
        "text": data["choices"][0]["message"]["content"],
        "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
        "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
        "latency_ms": latency_ms,
    }


async def call_anthropic(client: httpx.AsyncClient, model: str, prompt: str, api_key: str) -> dict:
    """Call Anthropic Claude API"""
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }

    start_time = time.time()
    response = await client.post(MODELS[model]["endpoint"], headers=headers, json=payload, timeout=120.0)
    latency_ms = int((time.time() - start_time) * 1000)

    if response.status_code != 200:
        return {"error": response.text, "latency_ms": latency_ms}

    data = response.json()

    return {
        "text": data["content"][0]["text"],
        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
        "latency_ms": latency_ms,
    }


async def call_google(client: httpx.AsyncClient, model: str, prompt: str, api_key: str) -> dict:
    """Call Google Gemini API"""
    url = f"{MODELS[model]['endpoint']}?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {
            "maxOutputTokens": MAX_TOKENS,
            "temperature": 0.0,
        },
    }

    start_time = time.time()
    response = await client.post(url, json=payload, timeout=120.0)
    latency_ms = int((time.time() - start_time) * 1000)

    if response.status_code != 200:
        return {"error": response.text, "latency_ms": latency_ms}

    data = response.json()

    # Extract text and token counts
    text = ""
    if "candidates" in data and data["candidates"]:
        parts = data["candidates"][0].get("content", {}).get("parts", [])
        if parts:
            text = parts[0].get("text", "")

    usage = data.get("usageMetadata", {})

    return {
        "text": text,
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0),
        "latency_ms": latency_ms,
    }


async def call_model(client: httpx.AsyncClient, model: str, prompt: str) -> dict:
    """Route to appropriate API based on model"""
    config = MODELS[model]
    api_key = os.environ.get(config["key_env"], "")

    if not api_key:
        return {"error": f"Missing API key: {config['key_env']}", "latency_ms": 0}

    if config["provider"] == "anthropic":
        return await call_anthropic(client, model, prompt, api_key)
    elif config["provider"] == "google":
        return await call_google(client, model, prompt, api_key)
    else:  # openai, deepseek, mistral
        return await call_openai(client, model, prompt, api_key)


async def run_pilot():
    """Run the pilot experiment"""
    results = []

    print("=" * 70)
    print("EXPERIMENT 5 PILOT - DeepSeek Token Explosion Validation")
    print("=" * 70)
    print(f"Models: {len(MODELS)}")
    print(f"Compression ratios: {COMPRESSION_RATIOS}")
    print(f"Samples per condition: {SAMPLES_PER_CONDITION}")
    print(f"Total trials: {len(MODELS) * len(COMPRESSION_RATIOS) * SAMPLES_PER_CONDITION}")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        trial_count = 0
        total_trials = len(MODELS) * len(COMPRESSION_RATIOS) * SAMPLES_PER_CONDITION

        for model in MODELS:
            print(f"\n>>> Testing {model}...")

            for ratio in COMPRESSION_RATIOS:
                for i, sample in enumerate(SAMPLE_PROMPTS[:SAMPLES_PER_CONDITION]):
                    trial_count += 1

                    # Compress prompt
                    compressed = compress_prompt(sample["prompt"], ratio)
                    input_tokens_approx = len(compressed.split())

                    # Call model
                    print(f"  [{trial_count}/{total_trials}] {model} r={ratio} {sample['id']}...", end=" ", flush=True)

                    try:
                        response = await call_model(client, model, compressed)

                        if "error" in response:
                            print(f"ERROR: {response['error'][:50]}")
                            result = {
                                "model": model,
                                "sample_id": sample["id"],
                                "compression_ratio": ratio,
                                "status": "error",
                                "error": response["error"],
                                "latency_ms": response["latency_ms"],
                                "timestamp": datetime.now().isoformat(),
                            }
                        else:
                            output_tokens = response["output_tokens"]
                            print(f"OK - {output_tokens} tokens, {response['latency_ms']}ms")

                            result = {
                                "model": model,
                                "sample_id": sample["id"],
                                "compression_ratio": ratio,
                                "status": "success",
                                "input_tokens": response["input_tokens"],
                                "output_tokens": output_tokens,
                                "latency_ms": response["latency_ms"],
                                "response_length": len(response["text"]),
                                "response_preview": response["text"][:200],
                                "timestamp": datetime.now().isoformat(),
                            }

                        results.append(result)

                    except Exception as e:
                        print(f"EXCEPTION: {str(e)[:50]}")
                        results.append({
                            "model": model,
                            "sample_id": sample["id"],
                            "compression_ratio": ratio,
                            "status": "exception",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        })

                    # Rate limiting
                    await asyncio.sleep(1.5)

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"pilot_results_{timestamp}.jsonl"

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n>>> Results saved to {output_file}")

    # Analyze results
    analyze_pilot(results)

    return results


def analyze_pilot(results: list):
    """Quick analysis of pilot results"""
    print("\n" + "=" * 70)
    print("PILOT ANALYSIS")
    print("=" * 70)

    # Group by model and ratio
    stats = {}

    for r in results:
        if r["status"] != "success":
            continue

        key = (r["model"], r["compression_ratio"])
        if key not in stats:
            stats[key] = {"tokens": [], "latency": []}

        stats[key]["tokens"].append(r["output_tokens"])
        stats[key]["latency"].append(r["latency_ms"])

    # Print summary table
    print(f"\n{'Model':<30} {'Ratio':<8} {'Avg Tokens':<12} {'Avg Latency':<12} {'N':<5}")
    print("-" * 70)

    for (model, ratio), data in sorted(stats.items()):
        avg_tokens = sum(data["tokens"]) / len(data["tokens"]) if data["tokens"] else 0
        avg_latency = sum(data["latency"]) / len(data["latency"]) if data["latency"] else 0
        n = len(data["tokens"])
        print(f"{model:<30} {ratio:<8} {avg_tokens:<12.1f} {avg_latency:<12.0f} {n:<5}")

    # Calculate explosion ratios
    print("\n" + "=" * 70)
    print("OUTPUT TOKEN EXPLOSION (r=0.3 vs r=1.0)")
    print("=" * 70)

    for model in MODELS:
        baseline = stats.get((model, 1.0), {}).get("tokens", [])
        compressed = stats.get((model, 0.3), {}).get("tokens", [])

        if baseline and compressed:
            baseline_avg = sum(baseline) / len(baseline)
            compressed_avg = sum(compressed) / len(compressed)
            explosion = compressed_avg / baseline_avg if baseline_avg > 0 else 0

            marker = "🔴" if explosion > 5 else "🟡" if explosion > 2 else "🟢"
            print(f"{marker} {model:<30} {baseline_avg:.0f} → {compressed_avg:.0f} tokens ({explosion:.1f}×)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(run_pilot())
