"""
Script d'évaluation d'UN SEUL modèle sur le dataset QCM.
Produit un fichier results_MODEL_TIMESTAMP.csv

Usage: 
    python 2_rag_eval_system.py
    (Configurez le modèle dans la section CONFIG ci-dessous)
"""

import os
import requests
from pypdf import PdfReader
import chromadb
import glob
import logging
from pypdf.errors import PdfReadError, PdfStreamError
import pandas as pd
import re
from typing import List, Dict
from datetime import datetime
import time

# Silence ChromaDB logs
logging.getLogger("chromadb").setLevel(logging.ERROR)

# ======================
# CONFIG - MODIFIEZ ICI
# ======================
LM_BASE_URL = "http://127.0.0.1:1234/v1"

# MODÈLE À ÉVALUER - Changez cette ligne pour tester un autre modèle + penser à rendre dispo le modèle sur lm_studio
CURRENT_MODEL = "qwen/qwen3-vl-4b"

# Exemples:
# CURRENT_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b"
# CURRENT_MODEL = "mistralai/mistral-7b-instruct-v0.3"
# CURRENT_MODEL = "openai/gpt-oss-20b"
# CURRENT_MODEL = "llama-3-8b-gpt-4o-ru1.0"
# CURRENT_MODEL = "qwen/qwen3-vl-4b"
# CURRENT_MODEL = "mistralai/ministral-3-3b"
# CURRENT_MODEL = "mistralai/ministral-3-3b:2"
# CURRENT_MODEL = "mistralai/ministral-3-3b:3"

PDF_PATHS = glob.glob("data/**/*.pdf", recursive=True)
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

CHUNK_SIZE = 1000 # 400 700 1000
CHUNK_OVERLAP = 150
TOP_K = 10
TIMEOUT_S = 500
CHROMA_BATCH_SIZE = 500
REBUILD_INDEX = False

# Dataset QCM
CSV_PATH = "dataset_qcm.csv"
NUM_QUESTIONS = 150  # None = toutes les questions, ou un nombre (ex: 10 pour tester)

# Dossier de sortie
OUTPUT_DIR = "evaluation_results"

# ======================
# PROMPTS
# ======================
PROMPT_BASE = (
    "Tu es un assistant qui répond à partir des extraits fournis. "
    "Si l'info n'est pas dans les extraits, tu dis 'Je ne trouve pas dans les documents fournis'. "
    "Tu ajoutes des citations sous la forme [source:page]."
)

PROMPT_QCM = (
    "Tu réponds à une question à choix multiples (QCM). "
    "Analyse chaque proposition (A, B, C, D, E) en te basant UNIQUEMENT sur les extraits fournis. "
    "Pour chaque proposition, indique si elle est EXACTE ou FAUSSE. "
    "Format de réponse attendu:\n"
    "A. [EXACTE/FAUSSE] - justification courte\n"
    "B. [EXACTE/FAUSSE] - justification courte\n"
    "etc.\n\n"
    "Puis conclus avec: RÉPONSE FINALE: [lettres des propositions exactes, ex: A, C, D]"
)

SYSTEM_PROMPT_QCM = PROMPT_BASE + PROMPT_QCM

# ======================
# FONCTIONS UTILITAIRES
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

def build_index():
    print("→ Using LM Studio embeddings:", EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path="./chroma_db")

    if not REBUILD_INDEX:
        try:
            col = client.get_collection("docs")
            print("→ Index existant chargé (pas de rebuild).")
            return col
        except Exception:
            print("→ Aucun index existant, création nécessaire.")

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

def retrieve(col, question: str, k: int):
    q_emb = embed_texts([question])[0]
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        chunks.append((doc, meta, dist))
    
    return chunks

def ask_lmstudio(question: str, retrieved_chunks, system_prompt: str, model_name: str):
    """Génère une réponse avec le modèle spécifié"""
    context_lines = []
    for i, (chunk, meta, dist) in enumerate(retrieved_chunks, 1):
        context_lines.append(f"EXTRAIT {i} [{meta['source']}:{meta['page']}]\n{chunk}\n")

    user_content = (
        "Question:\n" + question + "\n\n"
        "Extraits des documents (à utiliser comme SEULE source):\n\n"
        + "\n".join(context_lines)
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
        "stream": False,
    }

    start_time = time.time()
    r = requests.post(f"{LM_BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    inference_time = time.time() - start_time
    
    response_content = r.json()["choices"][0]["message"]["content"]
    return response_content, inference_time

# ======================
# ÉVALUATION QCM
# ======================

def extract_answer_from_response(response: str) -> List[str]:
    """Extrait les lettres des réponses du texte généré"""
    # Cherche "RÉPONSE FINALE:" ou patterns similaires (avec ou sans ** markdown)
    patterns = [
        r"\*\*RÉPONSE FINALE\*\*\s*:\s*([A-E,\s]+)",  # **RÉPONSE FINALE**: A, B
        r"RÉPONSE FINALE\s*:\s*([A-E,\s]+)",           # RÉPONSE FINALE: A, B
        r"\*\*Réponse finale\*\*\s*:\s*([A-E,\s]+)",   # **Réponse finale**: A, B
        r"Réponse finale\s*:\s*([A-E,\s]+)",           # Réponse finale: A, B
        r"RÉPONSES?\s*:\s*([A-E,\s]+)",                # RÉPONSE: A, B
        r"Les propositions? exactes? sont?\s*:\s*([A-E,\s]+)",  # Les propositions exactes sont: A, B
        r"Answer\s*:\s*([A-E,\s]+)",                   # Answer: A, B
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            letters = re.findall(r'[A-E]', match.group(1).upper())
            return sorted(set(letters))
    
    # Si pas de pattern trouvé, cherche les mentions EXACTE (avec ou sans crochets)
    exacte_matches = re.findall(r'([A-E])\.\s*(?:\[)?EXACTE(?:\])?', response, re.IGNORECASE)
    if exacte_matches:
        return sorted(set(exacte_matches))
    
    return []

def parse_qcm_answer_key(answer_key: str) -> List[str]:
    """Parse la colonne attendu du CSV"""
    if pd.isna(answer_key) or answer_key == "":
        return []
    # Gère les formats: "A", "A,B,C", "A, B, C", etc.
    letters = re.findall(r'[A-E]', str(answer_key).upper())
    return sorted(set(letters))

def calculate_metrics(predicted: List[str], expected: List[str]) -> Dict:
    """Calcule les métriques de performance"""
    pred_set = set(predicted)
    exp_set = set(expected)
    
    # Exact match
    exact_match = pred_set == exp_set
    
    # Precision, Recall, F1
    if len(pred_set) == 0:
        precision = 0.0
        recall = 0.0 if len(exp_set) > 0 else 1.0
    else:
        tp = len(pred_set & exp_set)
        precision = tp / len(pred_set)
        recall = tp / len(exp_set) if len(exp_set) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted": predicted,
        "expected": expected
    }

def evaluate_model(model_name: str, csv_path: str, col, num_questions: int = None, output_dir: str = OUTPUT_DIR):
    """Évalue UN modèle sur le dataset de QCM et sauvegarde les résultats"""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Charger le CSV
    df = pd.read_csv(csv_path)
    
    # Limiter le nombre de questions si spécifié
    if num_questions:
        df = df.head(num_questions)
    
    print(f"\n{'='*70}")
    print(f"ÉVALUATION DU MODÈLE: {model_name}")
    print(f"Nombre de questions: {len(df)}")
    print(f"{'='*70}\n")
    
    results = []
    total_inference_time = 0
    
    for idx, row in df.iterrows():
        question = row['question']
        options = row['option']
        answer_key = row.get('attendu', '')
        
        print(f"[{idx+1}/{len(df)}] Question: {question[:60]}...")
        
        # Construire la question complète
        full_question = f"{question}\n\nOptions:\n{options}"
        
        try:
            # Retrieve
            chunks = retrieve(col, full_question, TOP_K)
            avg_distance = sum([c[2] for c in chunks]) / len(chunks) if chunks else 0
            
            # Generate avec le modèle
            response, inference_time = ask_lmstudio(full_question, chunks, SYSTEM_PROMPT_QCM, model_name)
            total_inference_time += inference_time
            
            # Extract answers
            predicted = extract_answer_from_response(response)
            expected = parse_qcm_answer_key(answer_key)
            
            # Calculate metrics
            metrics = calculate_metrics(predicted, expected)
            
            # Store results
            result = {
                "question_id": idx,
                "question": question,
                "options": options,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "full_response": response,
                "exact_match": metrics["exact_match"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "avg_retrieval_distance": avg_distance,
                "num_chunks_retrieved": len(chunks),
                "inference_time": inference_time
            }
            results.append(result)
            
            # Affichage
            status = "✓" if metrics["exact_match"] else "✗"
            print(f"  {status} Attendu: {expected} | Prédit: {predicted} | F1: {metrics['f1']:.2f} | Temps: {inference_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ ERREUR: {e}")
            result = {
                "question_id": idx,
                "question": question,
                "error": str(e)
            }
            results.append(result)
    
    # Sauvegarder les résultats
    results_df = pd.DataFrame(results)
    model_safe_name = model_name.replace("/", "_")
    results_file = os.path.join(output_dir, f"results_{model_safe_name}_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    
    # Afficher les statistiques
    valid_results = [r for r in results if "error" not in r]
    
    if valid_results:
        exact_match_count = sum([r["exact_match"] for r in valid_results])
        avg_f1 = sum([r["f1"] for r in valid_results]) / len(valid_results)
        
        print(f"\n{'='*70}")
        print(f"RÉSULTATS - {model_name}")
        print(f"{'='*70}")
        print(f"Questions évaluées: {len(valid_results)}/{len(df)}")
        print(f"Exact Match: {exact_match_count}/{len(valid_results)} ({exact_match_count/len(valid_results):.2%})")
        print(f"F1-Score moyen: {avg_f1:.4f}")
        print(f"Temps total: {total_inference_time:.2f}s")
        print(f"Temps moyen/question: {total_inference_time/len(valid_results):.2f}s")
        print(f"{'='*70}\n")
        print(f"✓ Résultats sauvegardés: {results_file}")
        print(f"{'='*70}\n")
    
    return results_df

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    try:
        # Build index (une seule fois)
        print("Building index...")
        col = build_index()
        print("OK.\n")
        
        # Évaluer le modèle
        results_df = evaluate_model(
            model_name=CURRENT_MODEL,
            csv_path=CSV_PATH,
            col=col,
            num_questions=NUM_QUESTIONS
        )
        
        print("\n Évaluation terminée!")
        print(f"\n Pour évaluer un autre modèle:")
        print(f"   1. Modifiez CURRENT_MODEL dans la CONFIG")
        print(f"   2. Relancez: python 2_rag_eval_system.py")
        print(f"\n Pour comparer tous les modèles:")
        print(f"   python 3_rag_comparaison.py")
        
    except KeyboardInterrupt:
        print("\n Interruption utilisateur.")
    except FileNotFoundError as e:
        print(f"\n✗ ERREUR: {e}")
    except Exception as e:
        print(f"\n✗ ERREUR: {e}")
        import traceback
        traceback.print_exc()