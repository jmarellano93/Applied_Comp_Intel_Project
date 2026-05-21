"""
Module 9: Framework Qualitative Analyzer

Integrates PyTorch DatasetManager to calculate empirical partial derivatives
based on the exact center of mass of the target data topography. Exports text
derivatives to /reports and asymptotic topography surfaces to /visualizations.
"""

import os
import re
import glob
import math
from pathlib import Path
from typing import Dict, List

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pydantic import BaseModel, Field

# CRITICAL METHODOLOGICAL UPDATE: Import empirical environment logic
from MOD3_pm_dataset_manager import CacheConfig, DatasetManager

# =============================================================================
# 1. DYNAMIC PATHING & CONFIGURATION
# =============================================================================

class QualitativeConfig(BaseModel):
    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent / "generated_files" / "experimental_results_analysis_visualizations"

    @property
    def rule_dir(self) -> Path:
        d = self.base_dir / "rules"
        d.mkdir(parents=True, exist_ok=True)
        return d

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
# 2. SYMBOLIC PARSING ENGINE
# =============================================================================

class SymbolicEngine:
    def __init__(self):
        self.feature_names = [
            "n_d_ratio", "feat_kurtosis", "iqr_dev", "pc_eigen",
            "target_entropy", "hopkins", "silhouette", "davies_bouldin"
        ]
        self.symbols = {name: sp.Symbol(name) for name in self.feature_names}

        self.parse_dict = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'protected_div': lambda x, y: x / y if y != 0 else 1.0,
            'neg': lambda x: -x,
            'math': sp,
            'protected_sqrt': lambda x: sp.sqrt(sp.Abs(x)),
            'protected_log': lambda x: sp.log(sp.Abs(x)),
            'protected_exp': lambda x: sp.exp(x),
        }
        self.parse_dict.update(self.symbols)

    def parse_rule_to_sympy(self, rule_string: str) -> sp.Expr:
        clean_str = rule_string.replace("math.sin", "sp.sin").replace("math.cos", "sp.cos")
        self.parse_dict['sp'] = sp
        try:
            expression = eval(clean_str, {"__builtins__": None}, self.parse_dict)
            return sp.simplify(expression)
        except Exception as e:
            raise ValueError(f"Failed to parse symbolic string '{clean_str}': {e}")

    def extract_top_rule(self, filepath: Path) -> str:
        with open(filepath, 'r', encoding="utf-8") as f:
            content = f.read()
        equations = re.findall(r"Equation:\s*(.+)", content)
        if not equations:
            raise ValueError(f"No parseable equations found in {filepath.name}")
        return equations[0]

# =============================================================================
# 3. QUALITATIVE ANALYSIS & SURFACE MAPPING
# =============================================================================

class QualitativeAnalyzer:
    def __init__(self, config: QualitativeConfig):
        self.cfg = config
        self.engine = SymbolicEngine()
        self.empirical_means = self._compute_empirical_feature_means()

    def _compute_empirical_feature_means(self) -> Dict[sp.Symbol, float]:
        """Calculates the exact average topography of the dataset universe."""
        print("Pre-fetching Dataset Environment to calculate Empirical Gradients...")
        manager = DatasetManager(CacheConfig())

        all_meta_vectors = []
        for did in manager.dataset_cache.keys():
            _, meta = manager.get_dataset(did)
            all_meta_vectors.append(meta.numpy())

        empirical_array = np.mean(all_meta_vectors, axis=0)

        # Map NumPy averages to SymPy symbols
        return {
            self.engine.symbols[name]: float(val)
            for name, val in zip(self.engine.feature_names, empirical_array)
        }

    def identify_dominant_features(self, expr: sp.Expr) -> List[sp.Symbol]:
        """Evaluates gradient magnitude at the true empirical center of mass."""
        present_symbols = list(expr.free_symbols)
        if len(present_symbols) <= 2:
            return present_symbols

        gradients = {}
        for sym in present_symbols:
            derivative = sp.diff(expr, sym)
            try:
                # Use real-world mean substitutions instead of an arbitrary 0.5 scalar
                grad_mag = abs(float(derivative.subs(self.empirical_means)))
            except TypeError:
                grad_mag = 0.0
            gradients[sym] = grad_mag

        sorted_symbols = sorted(gradients.items(), key=lambda item: item[1], reverse=True)
        return [sorted_symbols[0][0], sorted_symbols[1][0]]

    def analyze_activation_family(self, filepath: Path) -> None:
        activation = filepath.name.split('_')[3]
        print(f"\nAnalyzing Symbolic Architecture for: [{activation.upper()}]")

        try:
            rule_str = self.engine.extract_top_rule(filepath)
            expr = self.engine.parse_rule_to_sympy(rule_str)
        except ValueError as e:
            print(e)
            return

        derivatives_text = [f"QUALITATIVE ANALYSIS: {activation.upper()}", f"Equation: {expr}\n", "Empirical Partial Derivatives:"]
        present_symbols = list(expr.free_symbols)

        for sym in present_symbols:
            deriv = sp.diff(expr, sym)
            derivatives_text.append(f"d(Variance) / d({sym.name}) = {deriv}")

        txt_out = self.cfg.reports_dir / f"Analytical_Derivatives_{activation}.txt"
        with open(txt_out, 'w') as f:
            f.write("\n".join(derivatives_text))

        if not present_symbols:
            print("Rule is a mathematical constant. No surface to plot.")
            return

        dominant_syms = self.identify_dominant_features(expr)

        if len(dominant_syms) == 1:
            sym_x = dominant_syms[0]
            x_vals = np.linspace(0.1, 1.0, 100)
            f_lamb = sp.lambdify(sym_x, expr, "numpy")

            plt.figure(figsize=(8, 6))
            y_vals = f_lamb(x_vals)
            if isinstance(y_vals, (float, int)): y_vals = np.full_like(x_vals, y_vals)

            plt.plot(x_vals, y_vals, color='blue', linewidth=2)
            plt.title(f"Sensitivity Curve: {activation.upper()}")
            plt.xlabel(sym_x.name)
            plt.ylabel('Optimal Variance ($\\sigma^2$)')
            plt.grid(True)
            plt.savefig(self.cfg.vis_dir / f"Sensitivity_Curve_{activation}.png", dpi=300)
            plt.close()

        elif len(dominant_syms) >= 2:
            sym_x, sym_y = dominant_syms[0], dominant_syms[1]

            # Lock remaining non-dominant dimensions at their EMPIRICAL mean
            subs_dict = {sym: self.empirical_means[sym] for sym in present_symbols if sym not in dominant_syms}
            expr_2d = expr.subs(subs_dict)

            X_vals = np.linspace(0.1, 1.0, 50)
            Y_vals = np.linspace(0.1, 1.0, 50)
            X_mesh, Y_mesh = np.meshgrid(X_vals, Y_vals)

            f_lamb = sp.lambdify((sym_x, sym_y), expr_2d, "numpy")
            Z_mesh = f_lamb(X_mesh, Y_mesh)

            if isinstance(Z_mesh, (float, int)):
                Z_mesh = np.full_like(X_mesh, Z_mesh)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis', edgecolor='none', alpha=0.9)

            ax.set_title(f"Asymptotic Topography: {activation.upper()}")
            ax.set_xlabel(sym_x.name)
            ax.set_ylabel(sym_y.name)
            ax.set_zlabel('Variance ($\\sigma^2$)')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Variance Magnitude')

            plt.savefig(self.cfg.vis_dir / f"Asymptotic_Surface_{activation}.png", dpi=300)
            plt.close()

    def run(self) -> None:
        if not self.cfg.rule_dir.exists():
            print(f"Rule directory not found: {self.cfg.rule_dir}")
            return

        rule_files = list(self.cfg.rule_dir.glob("Final_Discovered_Rules_*.txt"))
        print(f"--- INITIALIZING MODULE 9: SYMBOLIC ANALYZER ---")
        print(f"Discovered {len(rule_files)} Artifacts for Symbolic Extraction.")

        for rf in rule_files:
            self.analyze_activation_family(rf)

        print(f"\n--- QUALITATIVE EXTRAPOLATION COMPLETE ---")

if __name__ == "__main__":
    analyzer = QualitativeAnalyzer(QualitativeConfig())
    analyzer.run()