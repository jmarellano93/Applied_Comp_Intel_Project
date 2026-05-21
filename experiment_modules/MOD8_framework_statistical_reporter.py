"""
Module 8: Framework Statistical Reporter

Dynamically ingests validation JSON artifacts from the /reports namespace.
Generates multiobjective LaTeX tables (saved to /reports) and visual
distributions (saved to /visualizations).
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# =============================================================================
# 1. DYNAMIC PATHING & CONFIGURATION
# =============================================================================

class StatisticalConfig(BaseModel):
    alpha_level: float = Field(default=0.05, description="Significance threshold for p-values.")

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent / "generated_files" / "experimental_results_analysis_visualizations"

    @property
    def reports_dir(self) -> Path:
        d = self.base_dir / "reports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def vis_dir(self) -> Path:
        d = self.base_dir / "visualizations"
        d.mkdir(parents=True, exist_ok=True)
        return d

# =============================================================================
# 2. STATISTICAL ENGINE & ARTIFACT GENERATION
# =============================================================================

class ArtifactGenerator:
    def __init__(self, config: StatisticalConfig):
        self.cfg = config
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    def safe_wilcoxon(self, dist_a: List[float], dist_b: List[float], alternative: str = 'two-sided') -> Tuple[float, float]:
        if np.allclose(dist_a, dist_b, rtol=1e-7):
            return 0.0, 1.0
        try:
            stat, p_val = stats.wilcoxon(dist_a, dist_b, alternative=alternative)
            return float(stat), float(p_val)
        except ValueError:
            return 0.0, 1.0

    def generate_latex_table(self, results: Dict[str, Any], topology: str, activation: str) -> None:
        distributions = results.get("Raw_Distributions", {})
        if not distributions or "GP_Rule_1" not in distributions:
            return

        gp_acc = distributions["GP_Rule_1"]["acc"]
        gp_loss = distributions["GP_Rule_1"]["loss"]
        baselines = [k for k in distributions.keys() if "GP_Rule" not in k]

        latex_str = [
            "\\begin{table}[htpb]",
            "\\centering",
            "\\caption{Multiobjective Wilcoxon Signed-Rank Test Results: " + f"{topology.capitalize()} ({activation.capitalize()}) " + "}",
            "\\resizebox{\\textwidth}{!}{",
            "\\begin{tabular}{l | c c c | c c c}",
            "\\hline",
            "\\textbf{Method} & \\textbf{Mean Acc (\\%)} & \\textbf{W-Stat} & \\textbf{$p$-value} & \\textbf{Mean Loss} & \\textbf{W-Stat} & \\textbf{$p$-value} \\\\",
            "\\hline",
            f"\\textbf{{Symbolic GP Rule}} & \\textbf{{{np.mean(gp_acc)*100:.2f}}} & - & - & \\textbf{{{np.mean(gp_loss):.4f}}} & - & - \\\\",
            "\\hline"
        ]

        for base in baselines:
            base_acc = distributions[base]["acc"]
            base_loss = distributions[base]["loss"]

            acc_stat, acc_pval = self.safe_wilcoxon(gp_acc, base_acc, alternative='greater')
            acc_sig = "\\textbf{Yes}" if acc_pval < self.cfg.alpha_level else "No"

            loss_stat, loss_pval = self.safe_wilcoxon(gp_loss, base_loss, alternative='less')
            loss_sig = "\\textbf{Yes}" if loss_pval < self.cfg.alpha_level else "No"

            latex_str.append(f"{base.replace('_', '\\_')} & {np.mean(base_acc)*100:.2f} & {acc_stat:.1f} & {acc_pval:.4f} ({acc_sig}) & {np.mean(base_loss):.4f} & {loss_stat:.1f} & {loss_pval:.4f} ({loss_sig}) \\\\")

        latex_str.extend(["\\hline", "\\end{tabular}", "}", "\\label{tab:multiobjective_results_" + activation + "}", "\\end{table}\n"])

        table_path = self.cfg.reports_dir / f"Latex_Table_{topology}_{activation}.tex"
        with open(table_path, "w") as f:
            f.write("\n".join(latex_str))

    def generate_visual_distributions(self, results: Dict[str, Any], topology: str, activation: str) -> None:
        distributions = results.get("Raw_Distributions", {})
        if not distributions:
            return

        records = []
        for method, metrics in distributions.items():
            label = "Symbolic Rule" if "GP_Rule" in method else method.replace("_", " ")
            if "GP_Rule" in method and method != "GP_Rule_1":
                continue

            for i in range(len(metrics["acc"])):
                records.append({"Method": label, "Accuracy": metrics["acc"][i] * 100, "Epochs": metrics["epochs"][i], "Loss": metrics["loss"][i]})

        df = pd.DataFrame(records)
        base_filename = f"{topology}_{activation}"

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="Method", y="Accuracy", palette="viridis")
        plt.title(f"Generalization Accuracy Distributions: {activation.upper()}")
        plt.ylabel("Validation Accuracy (%)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.cfg.vis_dir / f"Boxplot_Accuracy_{base_filename}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x="Method", y="Epochs", palette="magma", cut=0)
        plt.title(f"Convergence Efficiency (Epochs to Target Threshold): {activation.upper()}")
        plt.ylabel("Epochs")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.cfg.vis_dir / f"Violinplot_Epochs_{base_filename}.png", dpi=300)
        plt.close()

        mean_df = df.groupby("Method").mean().reset_index()
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=mean_df, x="Epochs", y="Accuracy", hue="Method", s=200, palette="tab10")
        for _, row in mean_df.iterrows():
            plt.text(row['Epochs'] + 0.1, row['Accuracy'], row['Method'], fontsize=9)

        plt.title(f"Pareto Efficiency Landscape (Top-Left is Optimal): {activation.upper()}")
        plt.xlabel("Mean Epochs to Convergence (Lower is Better)")
        plt.ylabel("Mean Validation Accuracy % (Higher is Better)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.cfg.vis_dir / f"Pareto_Scatter_{base_filename}.png", dpi=300)
        plt.close()

    def process_all_artifacts(self) -> None:
        if not self.cfg.reports_dir.exists():
            print("Reports directory not found.")
            return

        json_files = list(self.cfg.reports_dir.glob("statistical_validation_*.json"))
        if not json_files:
            print(f"No JSON artifacts found in {self.cfg.reports_dir}. Run Module 7 first.")
            return

        print(f"--- INITIALIZING MODULE 8: STATISTICAL REPORTER ---")
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)

            topology = data.get("Metadata", {}).get("Topology", "unknown")
            activation = data.get("Metadata", {}).get("Activation", "unknown")

            self.generate_latex_table(data, topology, activation)
            self.generate_visual_distributions(data, topology, activation)
            print(f"Processed artifacts for: {topology} | {activation}")

        print(f"\n--- ALL ARTIFACTS EXPORTED TO partitioned namespaces. ---")

if __name__ == "__main__":
    generator = ArtifactGenerator(StatisticalConfig())
    generator.process_all_artifacts()