import requests
import sys

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "openai/gpt-oss-20b"

def chat():
    print("Chat LM Studio (tape /exit pour quitter, /reset pour vider l'historique)\n")
    messages = [{"role": "system", "content": "Tu es un assistant utile et concis."}]

    while True:
        user = input("toi > ").strip()
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            messages = [messages[0]]
            print("(historique vidé)\n")
            continue

        messages.append({"role": "user", "content": user})

        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": False
        }

        try:
            r = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            answer = data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"\n[Erreur HTTP] {e}\n")
            continue
        except (KeyError, IndexError, ValueError) as e:
            print(f"\n[Réponse inattendue] {e}\n{r.text}\n")
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"\nia  > {answer}\n")

if __name__ == "__main__":
    try:
        chat()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
