# compare_system_prompts_lmstudio.py
# Usage: python compare_system_prompts_lmstudio.py
#
# Requires: pip install requests

import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import requests

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "openai/gpt-oss-20b"

QUESTION = "Explique ce qu’est le deep learning à quelqu’un qui n’a jamais fait d’informatique."

# --- The 3 system prompts you asked for ---
SYSTEM_PROMPTS = {
    "A_reasoning_masked": (
        "Tu réfléchis étape par étape avant de répondre, "
        "mais tu ne dois jamais révéler ton raisonnement interne. "
        "Tu fournis uniquement la réponse finale, claire, pédagogique et concise."
    ),
    "B_force_errors": (
        "Tu réponds rapidement sans vérifier tes affirmations. "
        "Tu privilégies la fluidité et la confiance "
        "même si certaines informations peuvent être approximatives ou incorrectes."
    ),
    "C_adversarial_constraints": (
        "Tu dois expliquer le concept demandé tout en respectant strictement ces contraintes :\n"
        "- pas d’analogies\n"
        "- pas de métaphores\n"
        "- pas de jargon technique\n"
        "- pas plus de 5 phrases\n"
        "Si une contrainte est difficile à respecter, "
        "tu dois quand même fournir la meilleure réponse possible."
    ),
}

# --- Comparison (evaluator) prompt (used as system prompt for scoring) ---
EVAL_SYSTEM_PROMPT = """Tu es un évaluateur neutre et rigoureux de modèles de langage.

On te fournit plusieurs réponses à une même question,
chacune générée avec un system prompt différent.

Ton rôle est de comparer ces réponses selon les critères suivants :
1. Clarté pour un non-spécialiste
2. Cohérence logique
3. Précision des informations
4. Respect des contraintes imposées par le system prompt
5. Risque d’erreurs ou d’hallucinations
6. Qualité pédagogique globale

Pour chaque réponse :
- donne une note de 1 à 5 pour chaque critère
- justifie brièvement chaque note (1–2 phrases max)

Ensuite :
- indique quelle réponse est la plus utile dans un contexte pédagogique
- laquelle est la plus dangereuse si prise pour argent comptant
- laquelle montre le mieux l’impact du system prompting

Reste factuel, critique et structuré.
N’invente aucune information absente des réponses analysées.
"""

# Generation parameters you can tweak (these are request-level)
GEN_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.95,
    # top_k is not part of the official OpenAI ChatCompletions spec,
    # but some LM Studio backends accept it. We'll send it anyway.
    "top_k": 40,
    "max_tokens": 500,
    "stream": False,
}

EVAL_PARAMS = {
    "temperature": 0.2,
    "top_p": 1.0,
    "top_k": 0,
    "max_tokens": 900,
    "stream": False,
}

TIMEOUT_S = 180


def post_chat(messages: List[Dict[str, str]], params: Dict[str, Any]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        **params,
    }
    r = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def safe_post_chat(messages: List[Dict[str, str]], params: Dict[str, Any]) -> str:
    try:
        return post_chat(messages, params)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP error calling LM Studio: {e}") from e
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Unexpected response format: {e}") from e


def print_block(title: str, text: str):
    print("=" * 80)
    print(title)
    print("-" * 80)
    print(text.strip())
    print()


def main():
    print("LM Studio system-prompt comparator\n")
    print(f"BASE_URL: {BASE_URL}")
    print(f"MODEL:    {MODEL}")
    print(f"QUESTION: {QUESTION}\n")

    results: Dict[str, Dict[str, Any]] = {}
    answers_for_eval: List[str] = []

    # 1) Generate answers for each system prompt
    for key, sys_prompt in SYSTEM_PROMPTS.items():
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": QUESTION},
        ]
        print(f"[1/2] Generating answer: {key} ...")
        answer = safe_post_chat(messages, GEN_PARAMS)
        results[key] = {
            "system_prompt": sys_prompt,
            "answer": answer,
            "gen_params": GEN_PARAMS.copy(),
        }
        answers_for_eval.append((key, answer))
        time.sleep(0.2)

    # Print answers
    for key, _ in SYSTEM_PROMPTS.items():
        print_block(f"ANSWER {key}", results[key]["answer"])

    # 2) Ask model to evaluate/compare the three answers
    eval_user_prompt = "Question:\n" + QUESTION + "\n\n"
    for key, ans in answers_for_eval:
        eval_user_prompt += f"Réponse {key}:\n{ans.strip()}\n\n"

    eval_messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": eval_user_prompt},
    ]

    print("[2/2] Evaluating answers ...")
    evaluation = safe_post_chat(eval_messages, EVAL_PARAMS)
    print_block("EVALUATION", evaluation)

    # Save everything to a JSON file for later comparisons
    out = {
        "base_url": BASE_URL,
        "model": MODEL,
        "question": QUESTION,
        "system_prompts": SYSTEM_PROMPTS,
        "generation_params": GEN_PARAMS,
        "evaluation_params": EVAL_PARAMS,
        "results": results,
        "evaluation_text": evaluation,
    }

    out_path = "lmstudio_prompt_compare_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}\n")


if __name__ == "__main__":
    main()
