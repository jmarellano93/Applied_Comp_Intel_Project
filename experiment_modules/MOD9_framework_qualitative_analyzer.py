"""
Module 9: Framework Qualitative Analyzer.

For every Pareto rule artifact emitted by MOD6, MOD9 produces:
    * A text file of empirical partial derivatives per rank
        → ``EARV/reports/MOD9_qualitative_analysis/``
    * 1-D sensitivity curves or 2-D asymptotic surfaces showing the variance
      landscape with non-dominant features pinned at their empirical means
        → ``EARV/visualizations/MOD9_topography/``

The "empirical center of mass" approach computes each meta-feature's mean
across the Phase A dataset bench, then substitutes those values for
non-dominant symbols in the rule. Dominant symbols (top-2 by gradient
magnitude at the empirical mean) become the free axes of the surface plot.

Critical bug fixes versus the previous version:
    * The SymPy parser now includes ``sin``/``cos`` keys directly. The
      previous code rewrote ``math.sin → sp.sin`` via string replacement,
      but MOD6's rule files emit bare ``sin(...)`` / ``cos(...)`` (because
      DEAP uses ``func.__name__`` for primitive serialization).
    * All five Pareto ranks are now analyzed per activation, not just rank 1.
    * Activation token extraction uses regex rather than positional split.

Mathematical Notes:
    * Empirical means: ``mu_j = (1/D) * sum_d m_j(d)`` over the D Phase A
      datasets, for each of the 8 meta-features.
    * Dominant feature selection: ``argmax_j |df/dx_j| (mu)`` and the runner-up.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required to enable 3d projection
from pydantic import BaseModel, Field

from MOD3_pm_dataset_manager import CacheConfig, DatasetManager


# =============================================================================
# 1. CONFIGURATION & PATHING
# =============================================================================

# Regex anchors the activation token (group 1) and topology token (group 2)
# in filenames produced by MOD6 of the form:
#     Final_Discovered_Rules_<activation>_<topology>_<YYYYMMDD>(_<HHMM>)?.txt
# Topology is constrained to the three known values so the trailing timestamp
# segment can never be ambiguously absorbed into the topology token.
_ACTIVATION_TOPOLOGY_FROM_FILENAME = re.compile(
    r"^Final_Discovered_Rules_([a-zA-Z]+)_(shallow|deep_narrow|funnel)_\d+(?:_\d+)?\.txt$"
)

# Retain the old single-capture regex for backward compatibility with legacy
# rule files that lack a topology token (pre Q-G era artifacts).
_ACTIVATION_FROM_FILENAME = re.compile(
    r"^Final_Discovered_Rules_([a-zA-Z]+)_\d+(?:_\d+)?\.txt$"
)


class QualitativeConfig(BaseModel):
    """Configuration for the qualitative analyzer.

    Attributes:
        max_ranks: Number of Pareto ranks to analyze per activation.
        surface_resolution: Grid resolution for 2-D surface plots.
    """

    max_ranks: int = Field(default=5, gt=0)
    surface_resolution: int = Field(default=50, gt=10)

    @property
    def base_dir(self) -> Path:
        """Returns the EARV root (direct child of ``experiment_modules/``)."""
        return (
            Path(__file__).resolve().parent
            / "experimental_results_analysis_visualizations"
        )

    @property
    def rule_dir(self) -> Path:
        """Directory where MOD6 rule artifacts live.

        Defaults to ``generated_files/GA_rule_files_testing/`` to stay
        consistent with ``MOD7_pipeline_driver.DriverMatrixConfig``. When
        promoting real production rules, update both default factories
        in lock-step (here and in MOD7_pipeline_driver) to point at
        ``generated_files/GA_rule_files/``.
        """
        d = (
            Path(__file__).resolve().parent
            / "generated_files" / "GA_rule_files"
        )
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def reports_dir(self) -> Path:
        """MOD9-owned TXT output subfolder."""
        d = self.base_dir / "reports" / "MOD9_qualitative_analysis"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def vis_dir(self) -> Path:
        """MOD9-owned PNG output subfolder."""
        d = self.base_dir / "visualizations" / "MOD9_topography"
        d.mkdir(parents=True, exist_ok=True)
        return d


# =============================================================================
# 2. SYMBOLIC PARSING ENGINE
# =============================================================================

class SymbolicEngine:
    """SymPy parser for DEAP-format GP rule strings.

    The DEAP textual format uses bare function names (``add``, ``mul``,
    ``sin``, ``cos``, ``protected_log`` ...) which we map to SymPy primitives
    via ``parse_dict``. The eight meta-feature names become free symbols.
    """

    def __init__(self) -> None:
        self.feature_names: List[str] = [
            "n_d_ratio", "feat_kurtosis", "iqr_dev", "pc_eigen",
            "target_entropy", "hopkins", "silhouette", "davies_bouldin",
        ]
        self.symbols: Dict[str, sp.Symbol] = {
            name: sp.Symbol(name, real=True) for name in self.feature_names
        }

        # CRITICAL: sin/cos are bare-name entries (DEAP serializes by __name__).
        # Both math.sin/.cos and SymPy's sin/cos go through the same key.
        self.parse_dict: Dict[str, object] = {
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            "protected_div": lambda x, y: x / y if y != 0 else sp.Integer(1),
            "neg": lambda x: -x,
            "sin": sp.sin,
            "cos": sp.cos,
            "protected_sqrt": lambda x: sp.sqrt(sp.Abs(x)),
            "protected_log": lambda x: sp.log(sp.Abs(x) + sp.Rational(1, 10**5)),
            "protected_exp": sp.exp,
        }
        self.parse_dict.update(self.symbols)

    def parse_rule_to_sympy(self, rule_string: str) -> sp.Expr:
        """Parses a DEAP-format rule string into a SymPy expression.

        Args:
            rule_string: e.g. ``"sin(protected_log(mul(hopkins, iqr_dev)))"``.

        Returns:
            Simplified SymPy expression.

        Raises:
            ValueError: If the string cannot be parsed (unknown symbol,
                syntax error, etc.).
        """
        # Defensive: strip whitespace and CR/LF.
        clean = rule_string.strip()
        try:
            # Closed eval: no builtins, parse_dict as locals.
            expression = eval(clean, {"__builtins__": {}}, self.parse_dict)
            return sp.simplify(expression)
        except Exception as exc:
            raise ValueError(f"Failed to parse symbolic string '{clean}': {exc}")

    def extract_all_rules(self, filepath: Path, max_ranks: int = 5) -> List[str]:
        """Returns up to ``max_ranks`` equation strings, in rank order.

        Args:
            filepath: Path to a ``Final_Discovered_Rules_*.txt`` artifact.
            max_ranks: Hard cap on number of equations to return.

        Returns:
            List of equation strings. Empty list if file has no equations.
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read()
        equations = re.findall(r"Equation:\s*(.+)", content)
        return equations[:max_ranks]


# =============================================================================
# 3. QUALITATIVE ANALYZER
# =============================================================================

class QualitativeAnalyzer:
    """Top-level driver for symbolic surface analysis of Pareto rules."""

    def __init__(
        self,
        config: QualitativeConfig,
        manager: Optional[DatasetManager] = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            config: A validated ``QualitativeConfig``.
            manager: Optional pre-loaded ``DatasetManager``. If None, a Phase A
                manager will be constructed and populated. Injecting a
                fake/mocked manager is the recommended path for unit tests.
        """
        self.cfg = config
        self.engine = SymbolicEngine()
        self.empirical_means = self._compute_empirical_feature_means(manager)

    def _compute_empirical_feature_means(
        self, manager: Optional[DatasetManager],
    ) -> Dict[sp.Symbol, float]:
        """Computes per-feature mean across the Phase A dataset bench.

        Args:
            manager: Optional pre-loaded ``DatasetManager``. If None, one is
                built and loaded here.

        Returns:
            ``{sp.Symbol: float}`` mapping feature symbols to empirical means.
        """
        if manager is None:
            print("Pre-fetching Phase A bench to anchor empirical gradients...")
            manager = DatasetManager(CacheConfig())
            manager.load_all_to_ram()

        all_meta_vectors: List[np.ndarray] = []
        for did in manager.dataset_cache.keys():
            _, meta = manager.get_dataset(did)
            all_meta_vectors.append(meta.detach().cpu().numpy())

        if not all_meta_vectors:
            # Defensive fallback: zero-vector if cache is empty.
            empirical_array = np.zeros(len(self.engine.feature_names))
        else:
            empirical_array = np.mean(np.stack(all_meta_vectors, axis=0), axis=0)

        return {
            self.engine.symbols[name]: float(val)
            for name, val in zip(self.engine.feature_names, empirical_array)
        }

    # -------------------------------------------------------------------------
    # Dominant feature identification
    # -------------------------------------------------------------------------

    def identify_dominant_features(self, expr: sp.Expr) -> List[sp.Symbol]:
        """Returns up to 2 most-dominant features by ∂f/∂x at the empirical mean.

        Args:
            expr: A SymPy expression containing zero or more free symbols.

        Returns:
            List of length 0, 1, or 2 sympy symbols. Constant expressions
            return ``[]``; rules with ≤2 free symbols return all of them.
        """
        present_symbols = list(expr.free_symbols)
        if len(present_symbols) <= 2:
            return present_symbols

        gradients: Dict[sp.Symbol, float] = {}
        for sym in present_symbols:
            derivative = sp.diff(expr, sym)
            try:
                grad_mag = abs(float(derivative.subs(self.empirical_means)))
            except (TypeError, ValueError):
                grad_mag = 0.0
            gradients[sym] = grad_mag

        sorted_symbols = sorted(gradients.items(), key=lambda item: item[1], reverse=True)
        return [sorted_symbols[0][0], sorted_symbols[1][0]]

    # -------------------------------------------------------------------------
    # Per-rank analysis
    # -------------------------------------------------------------------------

    def _emit_derivatives_text(
        self, expr: sp.Expr, activation: str, topology: str, rank: int,
    ) -> None:
        """Writes the analytical derivatives text artifact for one rank.

        Args:
            expr: SymPy expression of the rule.
            activation: Activation token.
            topology: Topology token ('shallow', 'deep_narrow', 'funnel', or
                'legacy' for pre-Q-G rule files).
            rank: 1-based Pareto rank.
        """
        lines: List[str] = [
            f"QUALITATIVE ANALYSIS: {activation.upper()} / {topology} — Rank {rank}",
            f"Equation: {expr}",
            "",
            "Empirical Partial Derivatives:",
        ]
        present_symbols = list(expr.free_symbols)
        if not present_symbols:
            lines.append("  (expression is constant — no partial derivatives)")
        else:
            for sym in present_symbols:
                deriv = sp.diff(expr, sym)
                lines.append(f"  d(Variance) / d({sym.name}) = {deriv}")

        out_path = (
            self.cfg.reports_dir
            / f"Analytical_Derivatives_{activation}_{topology}_Rank{rank}.txt"
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def _emit_sensitivity_curve(
        self, expr: sp.Expr, sym_x: sp.Symbol,
        activation: str, topology: str, rank: int,
    ) -> None:
        """1-D sensitivity curve for a single-symbol rule.

        Post-normalisation, meta-feature values are z-scored. We plot the
        sensitivity over the symmetric range [-2, +2] standard deviations,
        which covers ~95% of the Phase A meta-feature distribution.
        """
        x_vals = np.linspace(-2.0, 2.0, 100)
        f_lamb = sp.lambdify(sym_x, expr, "numpy")
        y_vals = f_lamb(x_vals)
        if isinstance(y_vals, (float, int)):
            y_vals = np.full_like(x_vals, float(y_vals))

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, color="blue", linewidth=2)
        plt.title(f"Sensitivity Curve: {activation.upper()} / {topology} — Rank {rank}")
        plt.xlabel(f"{sym_x.name} (z-score)")
        plt.ylabel(r"Optimal Variance ($\sigma^2$)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            self.cfg.vis_dir
            / f"Sensitivity_Curve_{activation}_{topology}_Rank{rank}.png",
            dpi=300,
        )
        plt.close()

    def _emit_asymptotic_surface(
        self,
        expr: sp.Expr,
        present_symbols: List[sp.Symbol],
        dominant_syms: List[sp.Symbol],
        activation: str,
        topology: str,
        rank: int,
    ) -> None:
        """2-D asymptotic surface with non-dominant features pinned at empirical means.

        Surface coordinates span the z-score range [-2, +2] for both axes.
        """
        sym_x, sym_y = dominant_syms[0], dominant_syms[1]
        subs_dict = {
            sym: self.empirical_means[sym]
            for sym in present_symbols if sym not in dominant_syms
        }
        expr_2d = expr.subs(subs_dict)

        res = self.cfg.surface_resolution
        X_vals = np.linspace(-2.0, 2.0, res)
        Y_vals = np.linspace(-2.0, 2.0, res)
        X_mesh, Y_mesh = np.meshgrid(X_vals, Y_vals)

        f_lamb = sp.lambdify((sym_x, sym_y), expr_2d, "numpy")
        Z_mesh = f_lamb(X_mesh, Y_mesh)
        if isinstance(Z_mesh, (float, int)):
            Z_mesh = np.full_like(X_mesh, float(Z_mesh))

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            X_mesh, Y_mesh, Z_mesh, cmap="viridis", edgecolor="none", alpha=0.9,
        )

        ax.set_title(
            f"Asymptotic Topography: {activation.upper()} / {topology} — Rank {rank}"
        )
        ax.set_xlabel(f"{sym_x.name} (z-score)")
        ax.set_ylabel(f"{sym_y.name} (z-score)")
        ax.set_zlabel(r"Variance ($\sigma^2$)")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Variance Magnitude")

        plt.tight_layout()
        plt.savefig(
            self.cfg.vis_dir
            / f"Asymptotic_Surface_{activation}_{topology}_Rank{rank}.png",
            dpi=300,
        )
        plt.close()

    def analyze_rank(
        self, expr: sp.Expr, activation: str, topology: str, rank: int,
    ) -> None:
        """Emits all artifacts for a single (activation, topology, rank) triple."""
        self._emit_derivatives_text(expr, activation, topology, rank)

        present_symbols = list(expr.free_symbols)
        if not present_symbols:
            print(f"  Rank {rank}: constant rule — no surface to plot.")
            return

        dominant_syms = self.identify_dominant_features(expr)

        if len(dominant_syms) == 1:
            self._emit_sensitivity_curve(
                expr, dominant_syms[0], activation, topology, rank,
            )
        elif len(dominant_syms) >= 2:
            self._emit_asymptotic_surface(
                expr, present_symbols, dominant_syms,
                activation, topology, rank,
            )

    def analyze_activation_family(self, filepath: Path) -> None:
        """Analyzes all ranks in one rule artifact.

        Args:
            filepath: Path to a ``Final_Discovered_Rules_*.txt`` artifact.
                Filename is expected to encode both the activation and the
                topology; the legacy single-token form is supported with a
                topology fallback for backward compatibility.

        Returns:
            None. Writes one TXT + one PNG per (activation, topology, rank).
        """
        # Preferred: (activation, topology) form from the new MOD6 outputs.
        match = _ACTIVATION_TOPOLOGY_FROM_FILENAME.match(filepath.name)
        if match:
            activation = match.group(1)
            topology = match.group(2)
        else:
            # Legacy fallback: pre-Q-G files with only an activation token.
            legacy = _ACTIVATION_FROM_FILENAME.match(filepath.name)
            if not legacy:
                print(f"Could not extract activation/topology from filename: {filepath.name}")
                return
            activation = legacy.group(1)
            topology = "legacy"

        print(
            f"\nAnalyzing [{activation.upper()} / {topology}] "
            f"across up to {self.cfg.max_ranks} ranks"
        )

        rule_strings = self.engine.extract_all_rules(filepath, self.cfg.max_ranks)
        if not rule_strings:
            print(f"  No equations found in {filepath.name}")
            return

        for rank_idx, rule_str in enumerate(rule_strings, start=1):
            try:
                expr = self.engine.parse_rule_to_sympy(rule_str)
            except ValueError as exc:
                print(f"  Rank {rank_idx} parse error: {exc}")
                continue

            self.analyze_rank(expr, activation, topology, rank_idx)
            print(f"  Rank {rank_idx} processed: {expr}")

    def run(self) -> None:
        """Top-level driver: scans ``rule_dir`` and processes every artifact."""
        if not self.cfg.rule_dir.exists():
            print(f"Rule directory not found: {self.cfg.rule_dir}")
            return

        rule_files = list(self.cfg.rule_dir.glob("Final_Discovered_Rules_*.txt"))
        print("--- INITIALIZING MODULE 9: SYMBOLIC ANALYZER ---")
        print(f"Source: {self.cfg.rule_dir}")
        print(f"TXT → {self.cfg.reports_dir}")
        print(f"PNG → {self.cfg.vis_dir}")
        print(f"Discovered {len(rule_files)} rule artifact(s).")

        for rf in rule_files:
            self.analyze_activation_family(rf)

        print("\n--- QUALITATIVE EXTRAPOLATION COMPLETE ---")


if __name__ == "__main__":
    # Production usage: rules live under EARV/rules/ (after MOD6 promotion).
    # For testing against GA_rule_files_testing, instantiate QualitativeConfig
    # and override rule_dir manually or by symlink.
    analyzer = QualitativeAnalyzer(QualitativeConfig())
    analyzer.run()