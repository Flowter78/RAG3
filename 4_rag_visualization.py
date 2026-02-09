"""
Script de visualisation des résultats de comparaison.
Lit le fichier comparison.csv et génère les graphiques PNG.

Usage:
    python 4_rag_visualization.py
    (Charge automatiquement le fichier comparison le plus récent)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from datetime import datetime

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ======================
# CONFIG
# ======================
RESULTS_DIR = "evaluation_results"
OUTPUT_SUBDIR = "visualizations"  # Sous-dossier pour les images

class RAGResultsVisualizer:
    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = results_dir
        self.comparison_df = None
        self.model_results = {}
        self.timestamp = None
        self.output_dir = None
        
    def load_latest_comparison(self):
        """Charge le fichier de comparaison le plus récent"""
        comparison_files = glob.glob(os.path.join(self.results_dir, "comparison_*.csv"))
        
        if not comparison_files:
            raise FileNotFoundError(
                f"Aucun fichier de comparaison trouvé dans {self.results_dir}\n"
                f" Lancez d'abord: python 3_rag_comparaison.py"
            )
        
        latest_comparison = max(comparison_files, key=os.path.getctime)
        self.comparison_df = pd.read_csv(latest_comparison)
        
        # Extraire le timestamp
        self.timestamp = os.path.basename(latest_comparison).replace("comparison_", "").replace(".csv", "")
        
        # Créer le dossier de sortie
        self.output_dir = os.path.join(self.results_dir, f"{OUTPUT_SUBDIR}_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"✓ Fichier de comparaison chargé: {os.path.basename(latest_comparison)}")
        print(f"  Nombre de modèles: {len(self.comparison_df)}")
        print(f"  Dossier de sortie: {self.output_dir}\n")
        
        # Charger aussi les résultats détaillés si disponibles
        self._load_detailed_results()
        
    def _load_detailed_results(self):
        """Charge les résultats détaillés pour chaque modèle"""
        print(" Chargement des résultats détaillés...")
        
        for _, row in self.comparison_df.iterrows():
            model_name = row['model_name']
            
            # Le nom du fichier est dans la colonne results_file
            if 'results_file' in row:
                result_file = os.path.join(self.results_dir, row['results_file'])
                if os.path.exists(result_file):
                    self.model_results[model_name] = pd.read_csv(result_file)
                    print(f"  ✓ {model_name}")
                else:
                    print(f"   Fichier introuvable: {row['results_file']}")
            else:
                # Fallback: chercher par nom de modèle
                model_safe_name = model_name.replace("/", "_")
                pattern = os.path.join(self.results_dir, f"results_{model_safe_name}_*.csv")
                result_files = glob.glob(pattern)
                
                if result_files:
                    latest_result = max(result_files, key=os.path.getctime)
                    self.model_results[model_name] = pd.read_csv(latest_result)
                    print(f"  ✓ {model_name}")
        
        print()
    
    def plot_metrics_comparison(self):
        """1. Graphique de comparaison des métriques principales"""
        print(" 1/6 - Comparaison des métriques...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparaison des Performances des Modèles', fontsize=16, fontweight='bold')
        
        metrics = [
            ('exact_match_accuracy', 'Exact Match Accuracy', axes[0, 0]),
            ('avg_f1', 'F1-Score Moyen', axes[0, 1]),
            ('avg_precision', 'Precision Moyenne', axes[1, 0]),
            ('avg_recall', 'Recall Moyen', axes[1, 1])
        ]
        
        for metric, title, ax in metrics:
            data = self.comparison_df.sort_values(metric, ascending=False)
            
            bars = ax.barh(range(len(data)), data[metric], 
                          color=sns.color_palette("husl", len(data)))
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels([name.split('/')[-1] for name in data['model_name']])
            ax.set_xlabel('Score')
            ax.set_title(title, fontweight='bold')
            ax.set_xlim(0, 1)
            
            # Ajouter les valeurs sur les barres
            for i, (idx, row) in enumerate(data.iterrows()):
                ax.text(row[metric] + 0.02, i, f"{row[metric]:.2%}", 
                       va='center', fontweight='bold')
            
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "01_metrics_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Sauvegardé: {save_path}")
        plt.close()
    
    def plot_speed_vs_accuracy(self):
        """2. Graphique vitesse vs précision"""
        print(" 2/6 - Vitesse vs Précision...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = self.comparison_df['avg_inference_time']
        y = self.comparison_df['exact_match_accuracy']
        labels = [name.split('/')[-1] for name in self.comparison_df['model_name']]
        
        colors = sns.color_palette("husl", len(self.comparison_df))
        
        for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
            ax.scatter(xi, yi, s=300, alpha=0.6, color=colors[i], 
                      edgecolors='black', linewidth=2)
            ax.annotate(label, (xi, yi), xytext=(10, 10), 
                       textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', 
                               facecolor=colors[i], alpha=0.3))
        
        ax.set_xlabel('Temps d\'inférence moyen (secondes)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Exact Match Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Compromis Vitesse vs Précision', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Zones de performance
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, 
                  label='Seuil acceptable (70%)')
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, 
                  label='Seuil excellent (80%)')
        ax.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "02_speed_vs_accuracy.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Sauvegardé: {save_path}")
        plt.close()
    
    def plot_detailed_metrics_radar(self):
        """3. Graphique radar pour comparaison multi-dimensionnelle"""
        print(" 3/6 - Graphique radar multi-dimensionnel...")
        
        categories = ['Exact Match', 'F1-Score', 'Precision', 'Recall', 'Vitesse']
        num_vars = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = sns.color_palette("husl", len(self.comparison_df))
        
        for idx, (_, row) in enumerate(self.comparison_df.iterrows()):
            # Normaliser la vitesse (inverser car plus petit = mieux)
            max_time = self.comparison_df['avg_inference_time'].max()
            speed_score = 1 - (row['avg_inference_time'] / max_time)
            
            values = [
                row['exact_match_accuracy'],
                row['avg_f1'],
                row['avg_precision'],
                row['avg_recall'],
                speed_score
            ]
            values += values[:1]
            
            label = row['model_name'].split('/')[-1]
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('Comparaison Multi-dimensionnelle des Modèles', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "03_radar_chart.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Sauvegardé: {save_path}")
        plt.close()
    
    def plot_performance_per_question(self):
        """4. Performance par question"""
        print("📈 4/6 - Performance par question...")
        
        if not self.model_results:
            print("    Résultats détaillés non disponibles, graphique ignoré")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(16, 8))
            
            num_questions = len(next(iter(self.model_results.values())))
            x = np.arange(num_questions)
            width = 0.8 / len(self.model_results)
            
            colors = sns.color_palette("husl", len(self.model_results))
            
            for i, (model_name, results_df) in enumerate(self.model_results.items()):
                valid_results = results_df[~results_df['exact_match'].isna()].copy()
                
                # Convertir en int
                if valid_results['exact_match'].dtype == 'object':
                    valid_results['exact_match'] = valid_results['exact_match'].map({
                        'True': 1, 'False': 0, True: 1, False: 0
                    })
                else:
                    valid_results['exact_match'] = valid_results['exact_match'].astype(int)
                
                exact_matches = valid_results['exact_match'].values
                positions = x[:len(exact_matches)] + (i - len(self.model_results)/2 + 0.5) * width
                
                label = model_name.split('/')[-1]
                ax.bar(positions, exact_matches, width, label=label, 
                      color=colors[i], alpha=0.7)
            
            ax.set_xlabel('Numéro de Question', fontsize=12, fontweight='bold')
            ax.set_ylabel('Réponse Correcte (1 = Oui, 0 = Non)', fontsize=12, fontweight='bold')
            ax.set_title('Performance par Question pour Chaque Modèle', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f"Q{i+1}" for i in range(num_questions)], 
                              rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, "04_performance_per_question.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Sauvegardé: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"    Erreur: {e}")
    
    def plot_f1_distribution(self):
        """5. Distribution des F1-scores"""
        print(" 5/6 - Distribution des F1-scores...")
        
        if not self.model_results:
            print("    Résultats détaillés non disponibles, graphique ignoré")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            data_to_plot = []
            labels = []
            
            for model_name, results_df in self.model_results.items():
                valid_results = results_df[~results_df['f1'].isna()]
                if len(valid_results) > 0:
                    data_to_plot.append(valid_results['f1'].values)
                    labels.append(model_name.split('/')[-1])
            
            if not data_to_plot:
                print("    Aucune donnée F1 valide")
                return
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
            
            colors = sns.color_palette("husl", len(data_to_plot))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_xlabel('Modèle', fontsize=12, fontweight='bold')
            ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
            ax.set_title('Distribution des F1-Scores par Modèle', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, "05_f1_distribution.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Sauvegardé: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"    Erreur: {e}")
    
    def plot_confusion_analysis(self):
        """6. Analyse de confusion"""
        print(" 6/6 - Analyse de confusion...")
        
        if not self.model_results:
            print("    Résultats détaillés non disponibles, graphique ignoré")
            return
        
        try:
            fig, axes = plt.subplots(1, len(self.model_results), 
                                    figsize=(6*len(self.model_results), 5))
            
            if len(self.model_results) == 1:
                axes = [axes]
            
            for ax, (model_name, results_df) in zip(axes, self.model_results.items()):
                # Convertir en booléen
                if results_df['exact_match'].dtype == 'object':
                    results_df['exact_match'] = results_df['exact_match'].map({
                        'True': True, 'False': False, True: True, False: False
                    })
                
                tp = int((results_df['exact_match'] == True).sum())
                fp_fn = int((results_df['exact_match'] == False).sum())
                total = len(results_df)
                
                confusion_data = np.array([[tp, fp_fn]])
                
                sns.heatmap(confusion_data, annot=True, fmt='d', cmap='YlGnBu', 
                           ax=ax, cbar=True, square=False,
                           xticklabels=['Correct', 'Incorrect'],
                           yticklabels=['Total'],
                           annot_kws={"fontsize": 14})
                
                ax.set_title(f"{model_name.split('/')[-1]}\nRésultats (Total: {total})", 
                            fontweight='bold', fontsize=11)
                
                accuracy = tp / total * 100 if total > 0 else 0
                ax.text(0.5, -0.3, f"Accuracy: {accuracy:.1f}%", 
                       ha='center', transform=ax.transAxes, 
                       fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, "06_confusion_analysis.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Sauvegardé: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"    Erreur: {e}")
    
    def generate_all_visualizations(self):
        """Génère tous les graphiques"""
        print(f"\n{'='*70}")
        print("GÉNÉRATION DES VISUALISATIONS")
        print(f"{'='*70}\n")
        
        self.plot_metrics_comparison()
        self.plot_speed_vs_accuracy()
        self.plot_detailed_metrics_radar()
        self.plot_performance_per_question()
        self.plot_f1_distribution()
        self.plot_confusion_analysis()
        
        print(f"\n{'='*70}")
        print(" VISUALISATIONS GÉNÉRÉES")
        print(f"{'='*70}")
        print(f"\n Tous les graphiques sont dans:")
        print(f"   {self.output_dir}")
        print(f"\n{'='*70}\n")

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    try:
        viz = RAGResultsVisualizer(RESULTS_DIR)
        viz.load_latest_comparison()
        viz.generate_all_visualizations()
        
        print(" Visualisation terminée avec succès!")
        
    except FileNotFoundError as e:
        print(f"\n ERREUR: {e}")
    except Exception as e:
        print(f"\n ERREUR: {e}")
        import traceback
        traceback.print_exc()