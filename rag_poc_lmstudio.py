#Ce projet est basé sur ChromaDB 0.4.x
#pip install chromadb==0.4.24 pypdf==5.1.0 requests==2.32.3

import os
import requests
from pypdf import PdfReader
import chromadb
import glob
import logging
from pypdf.errors import PdfReadError, PdfStreamError #éviter les fichiers corrompus 

# Silence ChromaDB logs (IDs déjà existants, infos internes)
logging.getLogger("chromadb").setLevel(logging.ERROR) 

# ======================
# CONFIG
# ======================
LM_BASE_URL = "http://127.0.0.1:1234/v1"
LM_MODEL = "openai/gpt-oss-20b"

#PDF_PATHS = glob.glob("data/*.pdf")
PDF_PATHS = glob.glob("data/**/*.pdf", recursive=True)
#print(PDF_PATHS)

EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

CHUNK_SIZE = 700          # en caractères (simple pour démarrer)
CHUNK_OVERLAP = 150
TOP_K = 10
TIMEOUT_S = 180

CHROMA_BATCH_SIZE = 500  # stable
REBUILD_INDEX = False  # True UNIQUEMENT ajout de nouveaux pdfs



#gestion des metadata sur les chunks=> version année, date du document. 
#dans le retrive on utilise la cosinne car dans cette étude c'est la meilleur (similarité )
#system prompt dynamique
#les question qu'on s'est posé, quelle elemenet on ts'est intéressé (littérature, choix de la similarité cosinus, etc)

PROMPT_BASE = (
    "Tu es un assistant qui répond à partir des extraits fournis. "
    "Si l'info n'est pas dans les extraits, tu dis 'Je ne trouve pas dans les documents fournis'. "
    "Tu ajoutes des citations sous la forme [source:page]."    
)
PROMPT_DEFINITION = (
   "Donne une définition claire et concise."
   "Une phrase principale maximum, puis un court complément si nécessaire."
   "Pas d’exemple non présent dans les extraits."
)
PROMPT_REVISION = (
    "Aide à la révision."
    "Réponse courte, factuelle, facile à mémoriser."
    "Utilise des listes à puces si pertinent."
)
PROMPT_LISTE = (
    "Structure la réponse sous forme de liste claire."
    "Chaque point doit correspondre à une information explicite des extraits."
)
PROMPT_EXPLICATION = (
    "Explique de manière progressive et logique."
    "Ne fais aucune hypothèse."
    "Ne complète pas avec des connaissances externes."
)

SYSTEM_PROMPT = PROMPT_BASE

PROMPTS = {
    "definition": PROMPT_BASE + PROMPT_DEFINITION,
    "revision": PROMPT_BASE + PROMPT_REVISION,
    "liste": PROMPT_BASE + PROMPT_LISTE,
    "explication": PROMPT_BASE + PROMPT_EXPLICATION,
}




# ======================
# TEXT UTILS
# ======================
def chunk_text(text: str, size: int, overlap: int):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def embed_texts(texts):
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }

    r = requests.post(
        f"{LM_BASE_URL}/embeddings",
        json=payload,
        timeout=TIMEOUT_S
    )
    r.raise_for_status()

    data = r.json()["data"]
    return [d["embedding"] for d in data]


def read_pdf(path: str):
    pages = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            pages.append((i + 1, page.extract_text() or ""))
    except (PdfReadError, PdfStreamError, Exception) as e:
        print(f"[WARN] PDF ignoré (invalide) : {path}")
        print(f"       Raison : {e}")
    return pages

def detect_intent(question: str):
    q = question.lower()

    if q.startswith(("qu'est-ce", "définir", "definition")):
        return "definition"

    if q.startswith(("donne", "liste", "objectifs", "principes")):
        return "liste"

    if any(word in q for word in ["pourquoi", "comment", "expliquer"]):
        return "explication"

    return "revision"



# ======================
# INDEX BUILD
# ======================

def build_index():
    print("→ Using LM Studio embeddings:", EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path="./chroma_db")

    # Si on ne veut PAS rebuild
    if not REBUILD_INDEX:
        try:
            col = client.get_collection("docs")
            print("→ Index existant chargé (pas de rebuild).")
            return col
        except Exception:
            print("→ Aucun index existant, création nécessaire.")


    # OBLIGATOIRE en ChromaDB 1.5.x
    col = client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"}
    )

    ids, docs, metas = [], [], []
    doc_id = 0

    for pdf_path in PDF_PATHS:
        pages = read_pdf(pdf_path)
        base = os.path.basename(pdf_path)

        for page_num, page_text in pages:
            if not page_text.strip():
                continue

            for c_idx, chunk in enumerate(
                chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            ):
                if len(chunk) < 100:
                    continue

                doc_id += 1
                ids.append(f"{base}:{page_num}:{c_idx}:{doc_id}")
                docs.append(chunk)
                metas.append({
                    "source": base,
                    "page": page_num,
                    "chunk_id": c_idx
                })

    print(f"→ Nombre de chunks à encoder : {len(docs)}")
    if not docs:
        raise RuntimeError("Aucun texte extrait des PDFs.")

    print("→ Encoding chunks...")
    embs = embed_texts(docs)
    print("→ Encoding done.")

#    col.add(
#        ids=ids,
#        documents=docs,
#        metadatas=metas,
#        embeddings=embs
#    )
#
#    return col

    print("→ Adding chunks to ChromaDB...")
    for i in range(0, len(ids), CHROMA_BATCH_SIZE):
        col.add(
            ids=ids[i:i + CHROMA_BATCH_SIZE],
            documents=docs[i:i + CHROMA_BATCH_SIZE],
            metadatas=metas[i:i + CHROMA_BATCH_SIZE],
            embeddings=embs[i:i + CHROMA_BATCH_SIZE],
        )
    print(f"  - Batch {i} → {min(i + CHROMA_BATCH_SIZE, len(ids))}")

    return col


# ======================
# RETRIEVE
# ======================

def retrieve(col, question: str, k: int):
    q_emb = embed_texts([question])[0]

    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )

    chunks = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        chunks.append((doc, meta))

    return chunks


# ======================
# GENERATE (LM STUDIO)
# ======================
def ask_lmstudio(question: str, retrieved_chunks, system_prompt: str):
    context_lines = []
    for i, (chunk, meta) in enumerate(retrieved_chunks, 1):
        context_lines.append(f"EXTRAIT {i} [{meta['source']}:{meta['page']}]\n{chunk}\n")

    user_content = (
        "Question:\n" + question + "\n\n"
        "Extraits des documents (à utiliser comme SEULE source):\n\n"
        + "\n".join(context_lines)
    )

    payload = {
        "model": LM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
        "stream": False,
    }

    r = requests.post(f"{LM_BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ======================
# MAIN DEMO
# ======================
if __name__ == "__main__":
    try:
        print("Building index...")
        col = build_index()
        print("OK.\n")

        while True:
            q = input("toi > ").strip()
            if not q:
                continue
            if q == "/exit":
                break
            
            intent = detect_intent(q)           #Détection de l’intention utilisateur
            print(intent)
            system_prompt = PROMPTS[intent]     #Sélection du system prompt adapté
            chunks = retrieve(col, q, TOP_K)    #Retrieval
            answer = ask_lmstudio(q, chunks, system_prompt)    #Génération avec prompt dynamique

            print(f"\nia  > {answer}\n")
    except KeyboardInterrupt:
        print("\nBye.")
