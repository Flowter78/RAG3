import os
import requests
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
import glob
import sys


# ======================
# CONFIG
# ======================
LM_BASE_URL = "http://127.0.0.1:1234/v1"
LM_MODEL = "openai/gpt-oss-20b"

PDF_PATHS = glob.glob("data/*.pdf")
print(PDF_PATHS)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # léger, marche bien en local
CHUNK_SIZE = 700          # en caractères (simple pour démarrer)
CHUNK_OVERLAP = 150
TOP_K = 10
TIMEOUT_S = 180

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond uniquement à partir des extraits fournis. "
    "Si l'info n'est pas dans les extraits, tu dis 'Je ne trouve pas dans les documents fournis'. "
    "Tu ajoutes des citations sous la forme [source:page]."
)

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

def read_pdf(path: str):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append((i + 1, page.extract_text() or ""))
    return pages

# ======================
# INDEX BUILD
# ======================
def build_index():
    print("→ Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    print("→ Embedding model loaded.")

    # Use persistent storage instead of in-memory
    #client = chromadb.Client()
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete existing collection if it exists (for clean rebuild)
    try:
        client.delete_collection("docs")
    except:
        pass
    
    col = client.create_collection("docs")
    #col = client.get_or_create_collection("docs")

    ids = []
    docs = []
    metas = []

    doc_id = 0
    for pdf_path in PDF_PATHS:
        pages = read_pdf(pdf_path)
        base = os.path.basename(pdf_path)

        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            for c_idx, chunk in enumerate(chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)):
                doc_id += 1
                ids.append(f"{base}:{page_num}:{c_idx}:{doc_id}")
                docs.append(chunk)
                metas.append({"source": base, "page": page_num})  
    print(f"→ Nombre de chunks à encoder : {len(docs)}")
    if not docs:
        raise RuntimeError("Aucun texte extrait des PDFs (docs est vide).")

    print("→ Encoding chunks...")
    embs = embedder.encode(docs, normalize_embeddings=True).tolist()
    print("→ Encoding done.")

    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    return col, embedder

# ======================
# RETRIEVE
# ======================
def retrieve(col, embedder, question: str, k: int):
    q_emb = embedder.encode([question], normalize_embeddings=True).tolist()[0]
    res = col.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    chunks = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        chunks.append((doc, meta))
    return chunks

# ======================
# GENERATE (LM STUDIO)
# ======================
def ask_lmstudio(question: str, retrieved_chunks):
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
            {"role": "system", "content": SYSTEM_PROMPT},
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
        col, embedder = build_index()
        print("OK.\n")

        while True:
            q = input("toi > ").strip()
            if not q:
                continue
            if q == "/exit":
                break
#           print("\n[DEBUG] Extraits récupérés :")
#           for i, (_, meta) in enumerate(chunks, 1):
#                print(f"  {i}. {meta['source']} p.{meta['page']}")
            chunks = retrieve(col, embedder, q, TOP_K)
            answer = ask_lmstudio(q, chunks)
            print(f"\nia  > {answer}\n")
    except KeyboardInterrupt:
        print("\nBye.")
