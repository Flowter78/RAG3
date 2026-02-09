"""
Application Streamlit pour le système RAG avec LM Studio
Interface utilisateur pour interroger des documents PDF via RAG
"""

import os
import streamlit as st
import requests
from pypdf import PdfReader
import chromadb
import glob
import logging
from pypdf.errors import PdfReadError, PdfStreamError
import time

# Silence ChromaDB logs
logging.getLogger("chromadb").setLevel(logging.ERROR)

# ======================
# CONFIG
# ======================
LM_BASE_URL = "http://127.0.0.1:1234/v1"
LM_MODEL = "qwen/qwen3-vl-4b"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
TOP_K = 10
TIMEOUT_S = 500
CHROMA_BATCH_SIZE = 500

# System prompts
PROMPT_BASE = (
    "Tu es un assistant qui répond à partir des extraits fournis. "
    "Si l'info n'est pas dans les extraits, tu dis 'Je ne trouve pas dans les documents fournis'. "
    "Tu ajoutes des citations sous la forme [source:page]."
)

PROMPTS = {
    "definition": PROMPT_BASE + "Donne une définition claire et concise. Une phrase principale maximum, puis un court complément si nécessaire. Pas d'exemple non présent dans les extraits.",
    "revision": PROMPT_BASE + "Aide à la révision. Réponse courte, factuelle, facile à mémoriser. Utilise des listes à puces si pertinent.",
    "liste": PROMPT_BASE + "Structure la réponse sous forme de liste claire. Chaque point doit correspondre à une information explicite des extraits.",
    "explication": PROMPT_BASE + "Explique de manière progressive et logique. Ne fais aucune hypothèse. Ne complète pas avec des connaissances externes.",
}

SYSTEM_PROMPT_QCM = (
    "Tu réponds à une question à choix multiples (QCM). "
    "Analyse chaque proposition (A, B, C, D, E) en te basant UNIQUEMENT sur les extraits fournis. "
    "Pour chaque proposition, indique si elle est EXACTE ou FAUSSE. "
    "Format de réponse attendu:\n"
    "A. [EXACTE/FAUSSE] - justification courte\n"
    "B. [EXACTE/FAUSSE] - justification courte\n"
    "etc.\n\n"
    "Puis conclus avec: RÉPONSE FINALE: [lettres des propositions exactes, ex: A, C, D]"
)


# ======================
# FONCTIONS UTILITAIRES
# ======================

def chunk_text(text: str, size: int, overlap: int):
    """Découpe le texte en chunks avec chevauchement"""
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
    """Génère les embeddings via LM Studio"""
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
    """Lit un PDF et retourne les pages avec leur numéro"""
    pages = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            pages.append((i + 1, page.extract_text() or ""))
    except (PdfReadError, PdfStreamError, Exception) as e:
        st.warning(f"PDF ignoré (invalide) : {path} - {e}")
    return pages


def detect_intent(question: str):
    """Détecte l'intention de l'utilisateur"""
    q = question.lower()
    
    if q.startswith(("qu'est-ce", "définir", "definition")):
        return "definition"
    if q.startswith(("donne", "liste", "objectifs", "principes")):
        return "liste"
    if any(word in q for word in ["pourquoi", "comment", "expliquer"]):
        return "explication"
    
    return "revision"


@st.cache_resource
def build_index(pdf_folder: str, rebuild: bool = False):
    """Construit l'index ChromaDB à partir des PDFs"""
    
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Si on ne veut pas rebuild
    if not rebuild:
        try:
            col = client.get_collection("docs")
            return col, "Index existant chargé"
        except Exception:
            pass
    
    # Création de la collection
    col = client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Récupération des PDFs
    pdf_paths = glob.glob(f"{pdf_folder}/**/*.pdf", recursive=True)
    
    if not pdf_paths:
        return None, f"Aucun PDF trouvé dans {pdf_folder}"
    
    ids, docs, metas = [], [], []
    doc_id = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf_path in enumerate(pdf_paths):
        status_text.text(f"Traitement de {os.path.basename(pdf_path)}...")
        pages = read_pdf(pdf_path)
        base = os.path.basename(pdf_path)
        
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            
            for c_idx, chunk in enumerate(chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)):
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
        
        progress_bar.progress((idx + 1) / len(pdf_paths))
    
    if not docs:
        return None, "Aucun texte extrait des PDFs"
    
    status_text.text(f"Génération des embeddings pour {len(docs)} chunks...")
    embs = embed_texts(docs)
    
    status_text.text("Ajout des chunks à ChromaDB...")
    for i in range(0, len(ids), CHROMA_BATCH_SIZE):
        col.add(
            ids=ids[i:i + CHROMA_BATCH_SIZE],
            documents=docs[i:i + CHROMA_BATCH_SIZE],
            metadatas=metas[i:i + CHROMA_BATCH_SIZE],
            embeddings=embs[i:i + CHROMA_BATCH_SIZE],
        )
    
    progress_bar.empty()
    status_text.empty()
    
    return col, f"Index créé avec succès ({len(docs)} chunks, {len(pdf_paths)} PDFs)"


def retrieve(col, question: str, k: int):
    """Récupère les k chunks les plus pertinents"""
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


def ask_lmstudio(question: str, retrieved_chunks, system_prompt: str):
    """Génère une réponse via LM Studio"""
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
# INTERFACE STREAMLIT
# ======================

def main():
    st.set_page_config(
        page_title="RAG avec LM Studio",
        page_icon="",
        layout="wide"
    )
    
    st.title(" Système RAG - Questions sur Documents PDF")
    st.markdown("*Posez vos questions sur vos documents PDF en utilisant LM Studio*")
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        pdf_folder = st.text_input(
            "Dossier des PDFs",
            value="data",
            help="Chemin vers le dossier contenant vos PDFs"
        )
        
        rebuild_index = st.checkbox(
            "Reconstruire l'index",
            value=False,
            help="Cochez pour forcer la reconstruction de l'index"
        )
        
        top_k = st.slider(
            "Nombre de chunks à récupérer",
            min_value=3,
            max_value=20,
            value=TOP_K,
            help="Plus de chunks = plus de contexte mais plus lent"
        )
        
        intent_mode = st.selectbox(
            "Mode de prompt",
            options=["Auto", "Définition", "Révision", "Liste", "Explication", "QCM"],
            help="Auto détecte automatiquement l'intention"
        )
        
        st.divider()
        
        # Informations sur la connexion
        st.subheader("🔌 Connexion LM Studio")
        st.text(f"URL: {LM_BASE_URL}")
        st.text(f"Modèle: {LM_MODEL}")
        
        # Test de connexion
        if st.button("Tester la connexion"):
            try:
                r = requests.get(f"{LM_BASE_URL}/models", timeout=5)
                if r.status_code == 200:
                    st.success(" LM Studio connecté")
                else:
                    st.error(" Erreur de connexion")
            except Exception as e:
                st.error(f" Impossible de se connecter: {e}")
    
    # Zone principale
    st.divider()
    
    # Initialisation de l'index
    if "collection" not in st.session_state or rebuild_index:
        with st.spinner("Construction de l'index..."):
            col, message = build_index(pdf_folder, rebuild_index)
            
            if col is None:
                st.error(f" {message}")
                st.stop()
            else:
                st.session_state.collection = col
                st.success(f" {message}")
    
    # Historique des conversations
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📄 Sources utilisées"):
                    for source in message["sources"]:
                        st.text(f"• {source['source']} (page {source['page']})")
    
    # Input de l'utilisateur
    if question := st.chat_input("Posez votre question..."):
        # Ajout du message utilisateur
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Génération de la réponse
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les documents..."):
                try:
                    # Détection de l'intention et sélection du prompt
                    if intent_mode == "Auto":
                        intent = detect_intent(question)
                        system_prompt = PROMPTS[intent]
                    elif intent_mode == "QCM":
                        system_prompt = SYSTEM_PROMPT_QCM
                    else:
                        intent_map = {
                            "Définition": "definition",
                            "Révision": "revision",
                            "Liste": "liste",
                            "Explication": "explication"
                        }
                        system_prompt = PROMPTS[intent_map[intent_mode]]
                    
                    # Retrieval
                    chunks = retrieve(st.session_state.collection, question, top_k)
                    
                    # Génération
                    answer = ask_lmstudio(question, chunks, system_prompt)
                    
                    # Affichage de la réponse
                    st.markdown(answer)
                    
                    # Sources
                    sources = [{"source": meta["source"], "page": meta["page"]} 
                              for _, meta in chunks[:5]]  # Top 5 sources
                    
                    with st.expander(" Sources utilisées"):
                        unique_sources = {}
                        for s in sources:
                            key = f"{s['source']}:{s['page']}"
                            if key not in unique_sources:
                                unique_sources[key] = s
                                st.text(f"• {s['source']} (page {s['page']})")
                    
                    # Ajout au contexte
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f" Erreur: {e}")
    
    # Bouton pour effacer l'historique
    if st.session_state.messages:
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button(" Effacer l'historique"):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()