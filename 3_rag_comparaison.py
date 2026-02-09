"""
Script de comparaison de TOUS les modèles évalués.
Scanne tous les fichiers results_*.csv et génère un fichier comparison.csv

Usage:
    python 3_rag_comparaison.py
"""

import pandas as pd
import glob
import os
from datetime import datetime
import json

# ======================
# CONFIG
# ======================
RESULTS_DIR = "evaluation_results"
OUTPUT_FILE = None  # None = auto-généré avec timestamp

# ======================
# FONCTIONS
# ======================

def extract_model_name_from_filename(filename: str) -> str:
    """
    Extrait le nom du modèle depuis le nom de fichier.
    Format attendu: results_model_name_timestamp.csv
    
    Exemples:
    - results_openai_gpt-oss-20b_20260115_141544.csv → openai/gpt-oss-20b
    - results_deepseek_deepseek-r1-0528-qwen3-8b_20260115_141544.csv → deepseek/deepseek-r1-0528-qwen3-8b
    """
    # Enlever le préfixe et suffixe
    name = filename.replace('results_', '').replace('.csv', '')
    parts = name.split('_')
    
    # Le timestamp est toujours au format: YYYYMMDD_HHMMSS (2 derniers éléments)
    # On cherche l'index où commence le timestamp
    timestamp_idx = None
    for i in range(len(parts) - 1):
        # Vérifier si c'est un timestamp (8 chiffres pour la date)
        if len(parts[i]) == 8 and parts[i].isdigit():
            timestamp_idx = i
            break
    
    if timestamp_idx is not None:
        # Tout avant le timestamp est le nom du modèle
        model_parts = parts[:timestamp_idx]
    else:
        # Fallback: tout sauf les 2 derniers (date et heure)
        model_parts = parts[:-2]
    
    # Reconstituer avec des slashes (convention des modèles)
    # Le premier underscore devient un slash, les autres restent
    if len(model_parts) >= 2:
        return f"{model_parts[0]}/{('_'.join(model_parts[1:]))}"
    else:
        return '_'.join(model_parts)

def calculate_stats_from_results(df: pd.DataFrame) -> dict:
    """Calcule les statistiques à partir d'un DataFrame de résultats"""
    
    # Filtrer les résultats valides (sans erreur)
    valid_results = df[~df['exact_match'].isna()].copy()
    
    if len(valid_results) == 0:
        return None
    
    # Convertir exact_match en booléen si nécessaire
    if valid_results['exact_match'].dtype == 'object':
        valid_results['exact_match'] = valid_results['exact_match'].map({
            'True': True, 'False': False, True: True, False: False
        })
    
    stats = {
        "total_questions": len(df),
        "successful_evaluations": len(valid_results),
        "failed_evaluations": len(df) - len(valid_results),
        "exact_match_accuracy": float(valid_results['exact_match'].sum() / len(valid_results)),
        "avg_precision": float(valid_results['precision'].mean()),
        "avg_recall": float(valid_results['recall'].mean()),
        "avg_f1": float(valid_results['f1'].mean()),
        "avg_retrieval_distance": float(valid_results['avg_retrieval_distance'].mean()),
        "total_inference_time": float(valid_results['inference_time'].sum()),
        "avg_inference_time": float(valid_results['inference_time'].mean())
    }
    
    return stats

def compare_all_models(results_dir: str = RESULTS_DIR, output_file: str = None):
    """
    Compare tous les fichiers results_*.csv trouvés dans le dossier
    et génère un fichier comparison.csv
    """
    
    print(f"\n{'='*70}")
    print("COMPARAISON DES MODÈLES ÉVALUÉS")
    print(f"{'='*70}\n")
    
    # Trouver tous les fichiers results_*.csv (exclure les _fixed)
    pattern = os.path.join(results_dir, "results_*.csv")
    result_files = glob.glob(pattern)
    result_files = [f for f in result_files if not f.endswith('_fixed.csv')]
    
    if not result_files:
        print(f" Aucun fichier de résultats trouvé dans {results_dir}")
        print(f" Lancez d'abord: python 2_rag_eval_system.py")
        return None
    
    print(f" {len(result_files)} fichier(s) de résultats trouvé(s):\n")
    
    all_stats = []
    
    for file_path in sorted(result_files):
        filename = os.path.basename(file_path)
        print(f" Traitement: {filename}")
        
        try:
            # Charger les résultats
            df = pd.read_csv(file_path)
            
            # Extraire le nom du modèle
            model_name = extract_model_name_from_filename(filename)
            print(f"   Modèle: {model_name}")
            
            # Calculer les stats
            stats = calculate_stats_from_results(df)
            
            if stats is not None:
                stats["model_name"] = model_name
                stats["results_file"] = filename
                all_stats.append(stats)
                
                print(f"   ✓ Exact Match: {stats['exact_match_accuracy']:.2%}")
                print(f"   ✓ F1 moyen: {stats['avg_f1']:.4f}")
                print(f"   ✓ Temps moyen: {stats['avg_inference_time']:.2f}s")
            else:
                print(f"    Aucune donnée valide dans ce fichier")
                
        except Exception as e:
            print(f"    Erreur: {e}")
        
        print()
    
    if not all_stats:
        print(" Aucune statistique générée")
        return None
    
    # Créer le DataFrame de comparaison
    comparison_df = pd.DataFrame(all_stats)
    
    # Réorganiser les colonnes pour plus de clarté
    column_order = [
        "model_name",
        "exact_match_accuracy",
        "avg_f1",
        "avg_precision",
        "avg_recall",
        "avg_inference_time",
        "total_inference_time",
        "avg_retrieval_distance",
        "successful_evaluations",
        "failed_evaluations",
        "total_questions",
        "results_file"
    ]
    comparison_df = comparison_df[column_order]
    
    # Trier par F1-Score décroissant
    comparison_df = comparison_df.sort_values('avg_f1', ascending=False).reset_index(drop=True)
    
    # Générer le nom de fichier de sortie
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(results_dir, f"comparison_{timestamp}.csv")
    
    # Sauvegarder
    comparison_df.to_csv(output_file, index=False)
    
    # Afficher le résumé
    print(f"{'='*70}")
    print("RÉSULTATS DE LA COMPARAISON")
    print(f"{'='*70}\n")
    print(f"✓ {len(all_stats)} modèle(s) comparé(s)")
    print(f"✓ Fichier généré: {output_file}\n")
    
    print(" CLASSEMENT PAR F1-SCORE:\n")
    for idx, row in comparison_df.iterrows():
        print(f"  {idx+1}. {row['model_name']}")
        print(f"     • Exact Match: {row['exact_match_accuracy']:.2%}")
        print(f"     • F1-Score: {row['avg_f1']:.4f}")
        print(f"     • Precision: {row['avg_precision']:.4f}")
        print(f"     • Recall: {row['avg_recall']:.4f}")
        print(f"     • Temps moyen: {row['avg_inference_time']:.2f}s")
        print()
    
    print(f"{'='*70}")
    print(" STATISTIQUES GLOBALES")
    print(f"{'='*70}\n")
    print(f"  Meilleur F1-Score: {comparison_df['avg_f1'].max():.4f} ({comparison_df.iloc[0]['model_name']})")
    print(f"  F1-Score moyen: {comparison_df['avg_f1'].mean():.4f}")
    print(f"  Modèle le plus rapide: {comparison_df.loc[comparison_df['avg_inference_time'].idxmin(), 'model_name']}")
    print(f"  Temps moyen le plus court: {comparison_df['avg_inference_time'].min():.2f}s")
    print(f"\n{'='*70}\n")
    
    # Sauvegarder aussi en JSON pour faciliter la lecture
    json_file = output_file.replace('.csv', '.json')
    comparison_dict = comparison_df.to_dict(orient='records')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_dict, f, indent=2, ensure_ascii=False)
    print(f"✓ Version JSON: {json_file}")
    
    print(f"\n Pour visualiser les résultats:")
    print(f"   python 4_rag_visualization.py")
    print()
    
    return comparison_df

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    try:
        comparison_df = compare_all_models(RESULTS_DIR, OUTPUT_FILE)
        
        if comparison_df is not None:
            print(" Comparaison terminée avec succès!")
        else:
            print(" Aucune comparaison générée")
            
    except Exception as e:
        print(f"\n ERREUR: {e}")
        import traceback
        traceback.print_exc()