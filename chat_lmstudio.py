import requests
import sys
import json

# =========================
# Configuration globale
# =========================

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = "Tu es un assistant utile et concis."

GENERATION_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": False,
    "top_k": 40,
}

REQUEST_TIMEOUT = 180

SESSION_INFO = {
    "model": MODEL,
    "system_prompt": SYSTEM_PROMPT,
    "generation_params": GENERATION_PARAMS,
}



# =========================
# Fonctions utilitaires
# =========================

def call_llm(messages):
    """
    Envoie une requête au modèle via LM Studio
    et retourne le texte de la réponse.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        **GENERATION_PARAMS,
    }

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def init_conversation():
    """
    Initialise l'historique de conversation
    avec le system prompt.
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}]


# =========================
# Boucle principale
# =========================

def chat():
    print("Chat LM Studio")
    print("Commandes : /exit pour quitter, /reset pour vider l'historique\n")
    print(json.dumps(SESSION_INFO, indent=2))

    messages = init_conversation()

    while True:
        user_input = input("toi > ").strip()

        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input == "/reset":
            messages = init_conversation()
            print("(historique vidé)\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            answer = call_llm(messages)
        except requests.exceptions.RequestException as e:
            print(f"\n[Erreur HTTP] {e}\n")
            continue
        except (KeyError, IndexError, ValueError) as e:
            print(f"\n[Erreur de parsing] {e}\n")
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"\nia  > {answer}\n")


# =========================
# Point d'entrée
# =========================

if __name__ == "__main__":
    try:
        chat()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
