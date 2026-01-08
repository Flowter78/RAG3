# topk_impact_test_lmstudio.py
# Usage: python compare_topk.py.py
#
# Requires: pip install requests

import re
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests

# =========================
# Configuration
# =========================

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "openai/gpt-oss-20b"
TIMEOUT_S = 180

N_RUNS = 8  # number of generations per condition

SYSTEM_PROMPT = (
    "Tu es libre d’explorer différentes formulations. "
    "Tu peux varier le vocabulaire, le ton et les idées. "
    "Ne cherche pas à être neutre ou standardisé."
)

USER_PROMPT = "Écris un court paragraphe (5–6 phrases) décrivant une ville futuriste."

# Keep everything identical except top_k
COMMON_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 220,
    "stream": False,
}

TOPK_CONDITIONS = {
    "top_k_low_5": 5,
    "top_k_high_200": 200,
}

OUTPUT_JSON = "topk_experiment_results.json"

# =========================
# LM Studio call
# =========================

def call_llm(system_prompt: str, user_prompt: str, params: Dict) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        **params,
    }
    r = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# =========================
# Simple text metrics
# =========================

_word_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _word_re.findall(text)]

def unique_word_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def repetition_rate(tokens: List[str]) -> float:
    """Share of tokens that are repeats (1 - unique ratio)."""
    if not tokens:
        return 0.0
    return 1.0 - unique_word_ratio(tokens)

def entropy(tokens: List[str]) -> float:
    """Shannon entropy over word distribution (higher = more diverse)."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def jaccard_similarity(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def summarize_runs(texts: List[str]) -> Dict:
    token_lists = [tokenize(t) for t in texts]
    lengths = [len(toks) for toks in token_lists]

    uniq_ratios = [unique_word_ratio(toks) for toks in token_lists]
    reps = [repetition_rate(toks) for toks in token_lists]
    ents = [entropy(toks) for toks in token_lists]

    # Average pairwise Jaccard (lower means more variety across runs)
    sims = []
    for i in range(len(token_lists)):
        for j in range(i + 1, len(token_lists)):
            sims.append(jaccard_similarity(token_lists[i], token_lists[j]))
    avg_sim = sum(sims) / len(sims) if sims else 1.0

    return {
        "runs": len(texts),
        "avg_length_words": sum(lengths) / len(lengths) if lengths else 0.0,
        "avg_unique_word_ratio": sum(uniq_ratios) / len(uniq_ratios) if uniq_ratios else 0.0,
        "avg_repetition_rate": sum(reps) / len(reps) if reps else 0.0,
        "avg_entropy_words": sum(ents) / len(ents) if ents else 0.0,
        "avg_pairwise_jaccard": avg_sim,
    }


# =========================
# Experiment
# =========================

def main():
    print("Top-K impact test (LM Studio)\n")
    print(f"BASE_URL: {BASE_URL}")
    print(f"MODEL:    {MODEL}")
    print(f"Runs per condition: {N_RUNS}\n")
    print("Only top_k changes between conditions.\n")

    all_results = {
        "base_url": BASE_URL,
        "model": MODEL,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "common_params": COMMON_PARAMS,
        "conditions": {},
    }

    for name, top_k in TOPK_CONDITIONS.items():
        params = {**COMMON_PARAMS, "top_k": top_k}

        print("=" * 80)
        print(f"Condition: {name} (top_k={top_k})")
        print("=" * 80)

        texts = []
        for i in range(N_RUNS):
            print(f"  run {i+1}/{N_RUNS} ...", end="", flush=True)
            try:
                out = call_llm(SYSTEM_PROMPT, USER_PROMPT, params)
            except Exception as e:
                print("\n  [ERROR]", e)
                out = f"[ERROR] {e}"
            print(" done.")
            texts.append(out)
            time.sleep(0.2)

        # Print outputs
        for i, t in enumerate(texts, 1):
            print("-" * 80)
            print(f"{name} | run {i}")
            print(t.strip())

        # Metrics
        metrics = summarize_runs(texts)
        print("\n[Metrics]")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        all_results["conditions"][name] = {
            "top_k": top_k,
            "params": params,
            "outputs": texts,
            "metrics": metrics,
        }

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Quick interpretation hints:")
    print("- avg_unique_word_ratio higher  => richer vocabulary (often with high top_k)")
    print("- avg_entropy_words higher      => more diverse word distribution")
    print("- avg_pairwise_jaccard lower    => outputs differ more across runs")
    print("=" * 80)
    print(f"\nSaved results to: {OUTPUT_JSON}\n")


if __name__ == "__main__":
    main()
