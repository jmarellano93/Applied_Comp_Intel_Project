# Module 8 General Function: Automatically converts the JSON results from Module 7 into publication-ready boxplots and a LaTeX table for your paper.

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np


def generate_artifacts():
    GEN_DIR = r"/old_shit"
    json_file = os.path.join(GEN_DIR, "validation_results_shallow.json")

    print(f"Loading {json_file}...")
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}. Run Module 07 first.")
        return

    # 1. LaTeX Table Generation
    print("\n--- GENERATING LATEX TABLE ---")
    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
    latex_str += "\\textbf{Method} & \\textbf{Mean Acc (\\%)} & \\textbf{W-Stat} & \\textbf{p-value} & \\textbf{Sig. ($p<0.05$)} \\\\\n\\hline\n"

    gp_scores = results['GP_Rule']
    latex_str += f"\\textbf{{Symbolic GP Rule}} & \\textbf{{{np.mean(gp_scores) * 100:.2f}}} & - & - & - \\\\\n\\hline\n"

    methods = [m for m in results.keys() if m != 'GP_Rule']
    plot_data = []

    for m in methods:
        base_scores = results[m]
        mean_acc = np.mean(base_scores) * 100
        try:
            stat, p_val = stats.wilcoxon(gp_scores, base_scores, alternative='greater')
        except ValueError:
            stat, p_val = 0, 1.0

        sig = "\\textbf{Yes}" if p_val < 0.05 else "No"
        latex_str += f"{m.replace('_', '\\_')} & {mean_acc:.2f} & {stat:.1f} & {p_val:.4f} & {sig} \\\\\n"

        # Prepare data for plotting
        for i in range(len(gp_scores)):
            plot_data.append({'Dataset_ID': i, 'Method': 'GP Rule', 'Accuracy': gp_scores[i] * 100})
            plot_data.append({'Dataset_ID': i, 'Method': m, 'Accuracy': base_scores[i] * 100})

    latex_str += "\\hline\n\\end{tabular}\n\\caption{Wilcoxon Signed-Rank Test Results across 51 Datasets}\n\\label{tab:wilcoxon_results}\n\\end{table}"
    print(latex_str)

    # 2. Boxplot Generation
    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Method', y='Accuracy', data=df, palette='Set2')
    plt.title("Generalization Performance: Symbolic GP Rule vs Baseline Initialization Heuristics (51 Datasets)")
    plt.ylabel("Balanced Accuracy (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot directly into the old_shit directory
    plot_path = os.path.join(GEN_DIR, "baseline_comparison_boxplots.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nVisual artifacts exported to '{plot_path}'.")

if __name__ == "__main__":
    generate_artifacts()