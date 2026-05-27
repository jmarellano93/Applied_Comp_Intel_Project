"""
Module 8: Framework Statistical Reporter.

Ingests validation JSON artifacts emitted by MOD7 and produces:
    * LaTeX Wilcoxon tables (5 GP ranks vs 6 baselines per activation)
        -> ``EARV/reports/MOD8_statistical_reports/``
    * Distribution plots (Accuracy boxplot, Epochs violinplot, Pareto scatter)
        -> ``EARV/visualizations/MOD8_distributions/``
    * **NEW** Cluster-stratified analysis (per-cell, per-cluster Wilcoxon and
      binomial win-rate with Holm-Bonferroni correction)
        -> ``EARV/reports/MOD8_cluster_stratified/``
    * **NEW** PCA(2) projection of the 25 Phase B datasets colored by cluster
        -> ``EARV/visualizations/MOD8_cluster_stratified/``

All output paths sit under the canonical EARV (Experimental Results Analysis
& Visualizations) directory, which is a direct child of ``experiment_modules/``.

Mathematical Notes:
    * Wilcoxon signed-rank test: tests H_0 that paired differences come from
      a symmetric distribution centered at zero. Used here because the same
      Phase B datasets are evaluated by both GP and baselines under matched
      seeds (paired observations).
    * ``safe_wilcoxon`` short-circuits zero-variance pairs (identical
      distributions) which would otherwise raise ``ValueError`` in
      ``scipy.stats.wilcoxon``.
    * **Cluster-stratified analysis** clusters the 25 Phase B datasets into
      K groups using K-means on the Phase-A Z-score-normalized 8-D meta-feature
      space (Phase-A params reused from MOD2 so the cluster geometry matches
      how GP rules saw the data). Per-cluster Wilcoxon and binomial tests
      are then computed for each (topology, activation, rule, baseline,
      cluster) tuple. Holm-Bonferroni correction is applied per cell to
      control family-wise error rate across ~K*5*6 simultaneous tests.

Cluster-stratification prerequisites:
    The MOD7 JSON ``Raw_Distributions`` block must include per-trial ``did``
    (dataset id) and ``seed`` arrays. If absent, the cluster analysis is
    skipped silently with a warning; the standard pooled outputs remain
    unaffected.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION & PATHING
# =============================================================================

class StatisticalConfig(BaseModel):
    """Configuration for the statistical reporter.

    Attributes:
        alpha_level: Significance threshold for Wilcoxon p-values.
        max_gp_ranks: Number of GP ranks to display per activation.
        k_clusters: Number of K-means clusters for stratified analysis.
        k_diagnostic_range: K values to scan for silhouette diagnostics.
        cluster_random_state: K-means random_state (cluster reproducibility).
        run_cluster_stratified: Master switch for cluster-stratified analysis.
    """

    alpha_level: float = Field(default=0.05, gt=0.0, lt=1.0)
    max_gp_ranks: int = Field(default=5, gt=0)
    k_clusters: int = Field(default=3, gt=1)
    k_diagnostic_range: Tuple[int, ...] = Field(default=(2, 3, 4, 5))
    cluster_random_state: int = Field(default=42)
    run_cluster_stratified: bool = Field(default=True)

    @property
    def base_dir(self) -> Path:
        """Returns the EARV root (direct child of ``experiment_modules/``)."""
        return (
            Path(__file__).resolve().parent
            / "experimental_results_analysis_visualizations"
        )

    @property
    def mod7_reports_dir(self) -> Path:
        """Source of MOD7 JSON artifacts."""
        d = self.base_dir / "reports" / "MOD7_validation_matrix"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def reports_dir(self) -> Path:
        """MOD8-owned LaTeX output subfolder (pooled)."""
        d = self.base_dir / "reports" / "MOD8_statistical_reports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def vis_dir(self) -> Path:
        """MOD8-owned PNG output subfolder (pooled)."""
        d = self.base_dir / "visualizations" / "MOD8_distributions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cluster_reports_dir(self) -> Path:
        """Output subfolder for cluster-stratified CSV + LaTeX."""
        d = self.base_dir / "reports" / "MOD8_cluster_stratified"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cluster_vis_dir(self) -> Path:
        """Output subfolder for cluster-stratified PCA figure."""
        d = self.base_dir / "visualizations" / "MOD8_cluster_stratified"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def generated_files_dir(self) -> Path:
        """Source of MOD2 normalization params + Phase B dataset metadata."""
        return Path(__file__).resolve().parent / "generated_files"


# =============================================================================
# 2. STATISTICAL ENGINE & ARTIFACT GENERATION
# =============================================================================

# The 8 meta-features that GP rules see, in the canonical order used by
# MOD2/MOD3/MOD6/MOD7. Used as column selectors when loading Phase B metadata.
_META_FEATURE_COLS: Tuple[str, ...] = (
    "n_d_ratio", "feat_kurtosis", "iqr_dev", "pc_eigen",
    "target_entropy", "hopkins", "silhouette", "davies_bouldin",
)


class ArtifactGenerator:
    """Statistical reporter: emits LaTeX tables, distribution plots, and
    cluster-stratified analyses."""

    def __init__(self, config: StatisticalConfig) -> None:
        self.cfg = config
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

        # Cluster machinery is lazy-initialized in _fit_global_clusters
        # so that the original pooled outputs work even when the cluster files
        # are missing.
        self._cluster_assignments: Optional[pd.Series] = None
        self._cluster_diagnostics: Optional[Dict[str, Any]] = None
        self._cluster_phase_b_normalized: Optional[np.ndarray] = None
        self._cluster_phase_b_df: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # Wilcoxon protection
    # -------------------------------------------------------------------------

    def safe_wilcoxon(
        self,
        dist_a: List[float],
        dist_b: List[float],
        alternative: str = "two-sided",
    ) -> Tuple[float, float]:
        """Wilcoxon signed-rank with zero-variance guard.

        Args:
            dist_a: First paired sample.
            dist_b: Second paired sample (must be same length as ``dist_a``).
            alternative: ``'two-sided'``, ``'greater'``, or ``'less'``.

        Returns:
            Tuple ``(W_statistic, p_value)``. Returns ``(0.0, 1.0)`` when
            distributions are numerically identical or when SciPy raises
            ``ValueError`` (e.g. all-zero differences).
        """
        if len(dist_a) == 0 or len(dist_b) == 0:
            return 0.0, 1.0
        if np.allclose(dist_a, dist_b, rtol=1e-7):
            return 0.0, 1.0
        try:
            stat, p_val = stats.wilcoxon(dist_a, dist_b, alternative=alternative)
            return float(stat), float(p_val)
        except ValueError:
            return 0.0, 1.0

    # -------------------------------------------------------------------------
    # LaTeX table: 5 GP ranks x 6 baselines (pooled)
    # -------------------------------------------------------------------------

    def _extract_gp_rule_keys(self, distributions: Dict[str, Any]) -> List[str]:
        """Returns the GP rule keys present in the JSON, sorted by rank order."""
        keys = [k for k in distributions.keys() if k.startswith("GP_Rule_")]
        # Sort by trailing integer for stable rank ordering.
        return sorted(keys, key=lambda k: int(k.split("_")[-1]))[: self.cfg.max_gp_ranks]

    def generate_latex_table(
        self, results: Dict[str, Any], topology: str, activation: str,
    ) -> None:
        """Writes a LaTeX table with one row per GP rank + one row per baseline.

        Args:
            results: Parsed MOD7 JSON content.
            topology: Topology token (for caption + filename).
            activation: Activation token (for caption + filename).
        """
        distributions = results.get("Raw_Distributions", {})
        if not distributions:
            return

        gp_keys = self._extract_gp_rule_keys(distributions)
        if not gp_keys:
            return

        baselines = [k for k in distributions.keys() if not k.startswith("GP_Rule_")]
        gp_rank1 = gp_keys[0]
        ref_acc = distributions[gp_rank1]["acc"]
        ref_loss = distributions[gp_rank1]["loss"]

        latex: List[str] = [
            "\\begin{table}[htpb]",
            "\\centering",
            "\\caption{Multiobjective Wilcoxon Signed-Rank Test Results: "
            f"{topology.capitalize()} / {activation.capitalize()} "
            f"(GP Rank-1 as paired comparator; $\\alpha = {self.cfg.alpha_level}$)"
            "}",
            "\\resizebox{\\textwidth}{!}{",
            "\\begin{tabular}{l | c c c | c c c}",
            "\\hline",
            "\\textbf{Method} & \\textbf{Mean Acc (\\%)} & \\textbf{W-Stat} "
            "& \\textbf{$p$-value} & \\textbf{Mean Loss} & \\textbf{W-Stat} "
            "& \\textbf{$p$-value} \\\\",
            "\\hline",
        ]

        for rk_idx, rk in enumerate(gp_keys, start=1):
            acc = distributions[rk]["acc"]
            loss = distributions[rk]["loss"]
            mean_acc = np.mean(acc) * 100 if acc else 0.0
            mean_loss = np.mean(loss) if loss else 0.0
            label = f"GP Rule (Rank {rk_idx})"
            if rk_idx == 1:
                latex.append(
                    f"\\textbf{{{label}}} & \\textbf{{{mean_acc:.2f}}} & - & - "
                    f"& \\textbf{{{mean_loss:.4f}}} & - & - \\\\"
                )
            else:
                latex.append(
                    f"{label} & {mean_acc:.2f} & - & - & {mean_loss:.4f} & - & - \\\\"
                )
        latex.append("\\hline")

        for base in baselines:
            base_acc = distributions[base]["acc"]
            base_loss = distributions[base]["loss"]
            mean_acc = np.mean(base_acc) * 100 if base_acc else 0.0
            mean_loss = np.mean(base_loss) if base_loss else 0.0

            acc_stat, acc_pval = self.safe_wilcoxon(ref_acc, base_acc, alternative="greater")
            loss_stat, loss_pval = self.safe_wilcoxon(ref_loss, base_loss, alternative="less")

            acc_sig = "\\textbf{Yes}" if acc_pval < self.cfg.alpha_level else "No"
            loss_sig = "\\textbf{Yes}" if loss_pval < self.cfg.alpha_level else "No"
            label = base.replace("_", "\\_")

            latex.append(
                f"{label} & {mean_acc:.2f} & {acc_stat:.1f} & "
                f"{acc_pval:.4f} ({acc_sig}) & {mean_loss:.4f} & "
                f"{loss_stat:.1f} & {loss_pval:.4f} ({loss_sig}) \\\\"
            )

        latex.extend([
            "\\hline",
            "\\end{tabular}",
            "}",
            f"\\label{{tab:multiobjective_results_{topology}_{activation}}}",
            "\\end{table}\n",
        ])

        table_path = self.cfg.reports_dir / f"Latex_Table_{topology}_{activation}.tex"
        with open(table_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(latex))

    # -------------------------------------------------------------------------
    # Distribution plots (pooled)
    # -------------------------------------------------------------------------

    def _build_long_dataframe(
        self, results: Dict[str, Any],
    ) -> pd.DataFrame:
        """Flattens the JSON Raw_Distributions block into a tidy DataFrame.

        Each row: (Method, Accuracy, Epochs, Loss).
        """
        distributions = results.get("Raw_Distributions", {})
        gp_keys = self._extract_gp_rule_keys(distributions)
        baselines = [k for k in distributions.keys() if not k.startswith("GP_Rule_")]

        records: List[Dict[str, Any]] = []
        for rk_idx, rk in enumerate(gp_keys, start=1):
            metrics = distributions[rk]
            label = f"GP-R{rk_idx}"
            for i in range(len(metrics["acc"])):
                records.append({
                    "Method": label,
                    "Accuracy": metrics["acc"][i] * 100.0,
                    "Epochs": metrics["epochs"][i],
                    "Loss": metrics["loss"][i],
                })
        for base in baselines:
            metrics = distributions[base]
            label = base.replace("_", " ")
            for i in range(len(metrics["acc"])):
                records.append({
                    "Method": label,
                    "Accuracy": metrics["acc"][i] * 100.0,
                    "Epochs": metrics["epochs"][i],
                    "Loss": metrics["loss"][i],
                })
        return pd.DataFrame(records)

    def generate_visual_distributions(
        self, results: Dict[str, Any], topology: str, activation: str,
    ) -> None:
        """Emits three PNG distributions per activation: Accuracy, Epochs, Pareto."""
        df = self._build_long_dataframe(results)
        if df.empty:
            return

        base_filename = f"{topology}_{activation}"
        method_order = df["Method"].drop_duplicates().tolist()

        plt.figure(figsize=(16, 6))
        sns.boxplot(
            data=df, x="Method", y="Accuracy",
            order=method_order,
            hue="Method", palette="viridis", legend=False,
        )
        plt.title(f"Generalization Accuracy Distributions: {activation.upper()}")
        plt.ylabel("Validation Accuracy (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.cfg.vis_dir / f"Boxplot_Accuracy_{base_filename}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(16, 6))
        sns.violinplot(
            data=df, x="Method", y="Epochs",
            order=method_order,
            hue="Method", palette="magma", cut=0, legend=False,
        )
        plt.title(f"Convergence Efficiency (Epochs to Target): {activation.upper()}")
        plt.ylabel("Epochs")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.cfg.vis_dir / f"Violinplot_Epochs_{base_filename}.png", dpi=300)
        plt.close()

        mean_df = df.groupby("Method", sort=False).mean(numeric_only=True).reset_index()
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=mean_df, x="Epochs", y="Accuracy",
            hue="Method", s=220, palette="tab20", legend=False,
        )
        for _, row in mean_df.iterrows():
            plt.text(row["Epochs"] + 0.15, row["Accuracy"], row["Method"], fontsize=9)

        plt.title(f"Pareto Efficiency Landscape (Top-Left is Optimal): {activation.upper()}")
        plt.xlabel("Mean Epochs to Convergence (Lower is Better)")
        plt.ylabel("Mean Validation Accuracy % (Higher is Better)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.cfg.vis_dir / f"Pareto_Scatter_{base_filename}.png", dpi=300)
        plt.close()

    # =========================================================================
    # 3. CLUSTER-STRATIFIED ANALYSIS  (NEW)
    # =========================================================================

    # -------------------------------------------------------------------------
    # Data loading helpers
    # -------------------------------------------------------------------------

    def _load_phase_b_meta_features(self) -> Optional[pd.DataFrame]:
        """Loads the 25 Phase B datasets and their raw meta-features.

        Returns:
            DataFrame indexed by ``did`` with the 8 meta-feature columns in
            canonical order, or None if the source CSV is missing.
        """
        csv_path = self.cfg.generated_files_dir / "Phase_B_Validation_Datasets.csv"
        if not csv_path.exists():
            print(f"[CLUSTER] Phase B CSV missing at {csv_path}; cluster analysis disabled.")
            return None

        df = pd.read_csv(csv_path)
        required = ["did", *_META_FEATURE_COLS]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(
                f"[CLUSTER] Phase B CSV missing required columns: {missing}; "
                "cluster analysis disabled."
            )
            return None

        return df[required].set_index("did")

    def _load_phase_a_norm_params(self) -> Optional[pd.DataFrame]:
        """Loads MOD2's Phase-A Z-score parameters (mean and std per meta-feature).

        Returns:
            DataFrame with index = meta-feature name, columns = ``['mean', 'std']``,
            or None if the source file is missing or unparseable.
        """
        csv_path = self.cfg.generated_files_dir / "meta_feature_normalization_params.csv"
        if not csv_path.exists():
            print(
                f"[CLUSTER] Normalization params missing at {csv_path}; "
                "cluster analysis will fit fresh Z-score on Phase B (DEGRADED MODE)."
            )
            return None

        df = pd.read_csv(csv_path)
        # MOD2 writes a flexible-shape CSV; normalize to (mean, std) columns.
        if {"feature", "mean", "std"}.issubset(df.columns):
            return df.set_index("feature")[["mean", "std"]]
        # Fall back: try to interpret first column as feature name.
        df = df.set_index(df.columns[0])
        if {"mean", "std"}.issubset(df.columns):
            return df[["mean", "std"]]

        print(f"[CLUSTER] Could not parse normalization params from {csv_path}; degraded fit.")
        return None

    def _normalize_phase_b(
        self,
        phase_b_df: pd.DataFrame,
        norm_params: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """Applies Phase-A Z-score params to Phase B meta-features.

        Args:
            phase_b_df: Raw Phase B meta-features, indexed by ``did``.
            norm_params: Phase-A (mean, std) per feature, or None for fresh fit.

        Returns:
            ``(25, 8)`` ndarray of normalized meta-features ready for K-means.
        """
        feature_matrix = phase_b_df[list(_META_FEATURE_COLS)].values.astype(np.float64)

        if norm_params is None:
            mu = feature_matrix.mean(axis=0)
            sd = feature_matrix.std(axis=0)
        else:
            mu = np.array([norm_params.loc[c, "mean"] for c in _META_FEATURE_COLS])
            sd = np.array([norm_params.loc[c, "std"] for c in _META_FEATURE_COLS])

        # Guard against zero-std (degenerate feature).
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (feature_matrix - mu) / sd

    # -------------------------------------------------------------------------
    # Clustering
    # -------------------------------------------------------------------------

    def _fit_global_clusters(self) -> bool:
        """Fits global K-means on Phase B meta-features. Populates self state.

        Returns:
            True iff clustering succeeded and ``self._cluster_assignments``
            is now a populated Series mapping did -> cluster_id.
        """
        phase_b_df = self._load_phase_b_meta_features()
        if phase_b_df is None:
            return False

        norm_params = self._load_phase_a_norm_params()
        normalized = self._normalize_phase_b(phase_b_df, norm_params)

        n_datasets = normalized.shape[0]
        print(
            f"[CLUSTER] Loaded {n_datasets} Phase B datasets with "
            f"{normalized.shape[1]} meta-features."
        )
        norm_source = "Phase-A params (MOD2)" if norm_params is not None else "Phase-B fresh fit (DEGRADED)"
        print(f"[CLUSTER] Normalization source: {norm_source}")

        diagnostics: Dict[str, Any] = {
            "k_silhouettes": {},
            "k_inertias": {},
            "normalization_source": "phase_a_mod2" if norm_params is not None else "phase_b_fresh",
            "n_datasets": int(n_datasets),
        }

        for k in self.cfg.k_diagnostic_range:
            if k >= n_datasets:
                continue
            try:
                km_k = KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=self.cfg.cluster_random_state,
                ).fit(normalized)
                sil = silhouette_score(normalized, km_k.labels_)
                diagnostics["k_silhouettes"][k] = float(sil)
                diagnostics["k_inertias"][k] = float(km_k.inertia_)
            except Exception as e:
                diagnostics["k_silhouettes"][k] = None
                diagnostics["k_inertias"][k] = None
                print(f"[CLUSTER] Diagnostic K={k} failed: {e}")

        km = KMeans(
            n_clusters=self.cfg.k_clusters,
            n_init=10,
            random_state=self.cfg.cluster_random_state,
        ).fit(normalized)

        self._cluster_assignments = pd.Series(
            data=km.labels_.astype(int),
            index=phase_b_df.index,
            name="cluster",
        )
        diagnostics["k_selected"] = int(self.cfg.k_clusters)
        diagnostics["cluster_sizes"] = {
            int(c): int((km.labels_ == c).sum())
            for c in range(self.cfg.k_clusters)
        }
        diagnostics["centroids_normalized"] = km.cluster_centers_.tolist()
        diagnostics["dataset_assignments"] = {
            int(did): int(cluster)
            for did, cluster in self._cluster_assignments.items()
        }
        # Record dataset names per cluster for the diagnostic report.
        raw_phase_b = pd.read_csv(
            self.cfg.generated_files_dir / "Phase_B_Validation_Datasets.csv"
        )
        if "name" in raw_phase_b.columns:
            name_map = raw_phase_b.set_index("did")["name"].to_dict()
            diagnostics["dataset_names"] = {
                int(did): str(name_map.get(int(did), "UNKNOWN"))
                for did in self._cluster_assignments.index
            }

        self._cluster_diagnostics = diagnostics
        self._cluster_phase_b_normalized = normalized
        self._cluster_phase_b_df = phase_b_df

        self._write_cluster_diagnostics()
        self._plot_phase_b_pca()

        return True

    def _write_cluster_diagnostics(self) -> None:
        """Writes a human-readable diagnostics file."""
        if self._cluster_diagnostics is None:
            return
        diag = self._cluster_diagnostics
        out: List[str] = [
            "=" * 70,
            "MOD8 CLUSTER DIAGNOSTICS",
            "=" * 70,
            f"Normalization source: {diag['normalization_source']}",
            f"Phase B dataset count: {diag['n_datasets']}",
            f"K selected:            {diag['k_selected']}",
            "",
            "Silhouette scores (higher = better-separated clusters):",
        ]
        for k in sorted(diag["k_silhouettes"]):
            v = diag["k_silhouettes"][k]
            inertia = diag["k_inertias"][k]
            v_str = f"{v:.4f}" if v is not None else "FAILED"
            i_str = f"{inertia:.2f}" if inertia is not None else "FAILED"
            tag = "  <- selected" if k == diag["k_selected"] else ""
            out.append(f"  K={k}: silhouette={v_str}  inertia={i_str}{tag}")

        out.extend([
            "",
            "Cluster sizes (selected K):",
        ])
        for c, n in sorted(diag["cluster_sizes"].items()):
            out.append(f"  Cluster {c}: {n} datasets")

        out.extend([
            "",
            "Dataset assignments (did -> cluster):",
        ])
        names = diag.get("dataset_names", {})
        for did, c in sorted(diag["dataset_assignments"].items()):
            name = names.get(did, "")
            out.append(f"  did={did:>6}  cluster={c}  name={name}")

        path = self.cfg.cluster_reports_dir / "cluster_diagnostics.txt"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"[CLUSTER] Diagnostics written to {path}")

    def _plot_phase_b_pca(self) -> None:
        """Writes a PCA(2) scatter of the 25 Phase B datasets, colored by cluster."""
        if self._cluster_diagnostics is None or self._cluster_phase_b_normalized is None:
            return
        normalized = self._cluster_phase_b_normalized
        df = self._cluster_phase_b_df
        assignments = self._cluster_assignments

        pca = PCA(n_components=2, random_state=self.cfg.cluster_random_state)
        coords = pca.fit_transform(normalized)
        var_explained = pca.explained_variance_ratio_

        plot_df = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "cluster": assignments.values,
            "did": df.index.values,
        })
        names = self._cluster_diagnostics.get("dataset_names", {})
        plot_df["name"] = plot_df["did"].map(names)

        plt.figure(figsize=(11, 8))
        palette = sns.color_palette("Set1", n_colors=self.cfg.k_clusters)
        sns.scatterplot(
            data=plot_df, x="PC1", y="PC2", hue="cluster",
            palette=palette, s=140, edgecolor="black", linewidth=0.6,
        )
        for _, row in plot_df.iterrows():
            label = row["name"] if isinstance(row["name"], str) else str(row["did"])
            plt.annotate(
                label, (row["PC1"], row["PC2"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points",
            )

        plt.title(
            f"Phase B Meta-Feature Space - PCA(2), K={self.cfg.k_clusters} clusters "
            f"(PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%})"
        )
        plt.xlabel(f"PC1 ({var_explained[0]:.1%})")
        plt.ylabel(f"PC2 ({var_explained[1]:.1%})")
        plt.legend(title="Cluster", loc="best")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = self.cfg.cluster_vis_dir / "Phase_B_PCA_clusters.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[CLUSTER] PCA figure written to {path}")

    # -------------------------------------------------------------------------
    # Per-cell stratified statistics
    # -------------------------------------------------------------------------

    def _trials_have_dataset_id(self, distributions: Dict[str, Any]) -> bool:
        """Returns True iff every method's metrics include the ``did`` and ``seed`` fields."""
        for method, metrics in distributions.items():
            if "did" not in metrics or "seed" not in metrics:
                return False
            if not (len(metrics["acc"]) == len(metrics["did"]) == len(metrics["seed"])):
                return False
        return True

    @staticmethod
    def _holm_bonferroni(p_values: List[float]) -> List[float]:
        """Holm-Bonferroni step-down correction.

        Args:
            p_values: Raw p-values in the family.

        Returns:
            List of corrected p-values, same length as input, same order as
            input. Each value is monotonically non-decreasing in the ranked
            sequence, clipped to [0, 1].

        Mathematical Notes:
            For ranked p-values p_(1) <= p_(2) <= ... <= p_(m):
                p_(i)_adj = max over j<=i of min((m - j + 1) * p_(j), 1).
            This controls the family-wise error rate at level alpha.
        """
        m = len(p_values)
        if m == 0:
            return []
        order = np.argsort(p_values)
        ranked = np.array(p_values, dtype=float)[order]
        adjusted = np.empty(m, dtype=float)
        running_max = 0.0
        for i, p in enumerate(ranked):
            corrected = min((m - i) * p, 1.0)
            running_max = max(running_max, corrected)
            adjusted[i] = running_max
        result = np.empty(m, dtype=float)
        result[order] = adjusted
        return result.tolist()

    def _compute_cluster_stratified_results(
        self,
        results: Dict[str, Any],
        topology: str,
        activation: str,
    ) -> Optional[pd.DataFrame]:
        """Computes per-cluster Wilcoxon + binomial stats for one (topology, activation) cell.

        Args:
            results: Parsed MOD7 JSON content.
            topology: Topology token.
            activation: Activation token.

        Returns:
            Long-format DataFrame with one row per (rule, baseline, cluster)
            tuple, or None if the cluster analysis cannot run for this cell.
        """
        if self._cluster_assignments is None:
            return None

        distributions = results.get("Raw_Distributions", {})
        if not distributions:
            return None
        if not self._trials_have_dataset_id(distributions):
            print(
                f"[CLUSTER] {topology}/{activation}: JSON lacks per-trial 'did'/'seed'; "
                "stratified analysis skipped for this cell."
            )
            return None

        gp_keys = self._extract_gp_rule_keys(distributions)
        if not gp_keys:
            return None
        baselines = [k for k in distributions.keys() if not k.startswith("GP_Rule_")]

        # Build a tall per-trial DataFrame
        rows: List[Dict[str, Any]] = []
        for method, metrics in distributions.items():
            n = len(metrics["acc"])
            for i in range(n):
                rows.append({
                    "method": method,
                    "did": int(metrics["did"][i]),
                    "seed": int(metrics["seed"][i]),
                    "acc": float(metrics["acc"][i]),
                    "loss": float(metrics["loss"][i]),
                })
        trials = pd.DataFrame(rows)
        trials["cluster"] = trials["did"].map(self._cluster_assignments)
        trials = trials.dropna(subset=["cluster"])
        trials["cluster"] = trials["cluster"].astype(int)

        cluster_ids = sorted(trials["cluster"].unique())
        raw_pvals_acc: List[float] = []
        raw_pvals_loss: List[float] = []
        records: List[Dict[str, Any]] = []

        for cluster_id in cluster_ids:
            sub = trials[trials["cluster"] == cluster_id]
            for gp_key in gp_keys:
                gp_sub = sub[sub["method"] == gp_key].sort_values(["did", "seed"])
                if gp_sub.empty:
                    continue
                gp_acc = gp_sub["acc"].tolist()
                gp_loss = gp_sub["loss"].tolist()

                for base in baselines:
                    base_sub = sub[sub["method"] == base].sort_values(["did", "seed"])
                    if base_sub.empty or len(base_sub) != len(gp_sub):
                        continue
                    base_acc = base_sub["acc"].tolist()
                    base_loss = base_sub["loss"].tolist()

                    acc_stat, acc_pval = self.safe_wilcoxon(
                        gp_acc, base_acc, alternative="greater",
                    )
                    loss_stat, loss_pval = self.safe_wilcoxon(
                        gp_loss, base_loss, alternative="less",
                    )

                    wins = int(sum(1 for g, b in zip(gp_loss, base_loss) if g < b))
                    n_trials = len(gp_loss)
                    win_rate = wins / n_trials if n_trials > 0 else 0.0

                    raw_pvals_acc.append(acc_pval)
                    raw_pvals_loss.append(loss_pval)

                    records.append({
                        "topology": topology,
                        "activation": activation,
                        "rule": gp_key,
                        "baseline": base,
                        "cluster": int(cluster_id),
                        "n_trials": int(n_trials),
                        "gp_acc_mean": float(np.mean(gp_acc)),
                        "baseline_acc_mean": float(np.mean(base_acc)),
                        "gp_loss_mean": float(np.mean(gp_loss)),
                        "baseline_loss_mean": float(np.mean(base_loss)),
                        "wilcoxon_acc_stat": float(acc_stat),
                        "wilcoxon_acc_pval_raw": float(acc_pval),
                        "wilcoxon_loss_stat": float(loss_stat),
                        "wilcoxon_loss_pval_raw": float(loss_pval),
                        "binomial_wins": int(wins),
                        "binomial_win_rate": float(win_rate),
                    })

        if not records:
            return None

        # Holm-Bonferroni per cell (family = all tests in this (topology, activation))
        acc_corrected = self._holm_bonferroni(raw_pvals_acc)
        loss_corrected = self._holm_bonferroni(raw_pvals_loss)
        for rec, pa_corr, pl_corr in zip(records, acc_corrected, loss_corrected):
            rec["wilcoxon_acc_pval_holm"] = float(pa_corr)
            rec["wilcoxon_loss_pval_holm"] = float(pl_corr)
            rec["acc_significant"] = bool(pa_corr < self.cfg.alpha_level)
            rec["loss_significant"] = bool(pl_corr < self.cfg.alpha_level)

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # LaTeX rendering for cluster-stratified results
    # -------------------------------------------------------------------------

    def _write_cluster_latex(
        self, cell_df: pd.DataFrame, topology: str, activation: str,
    ) -> None:
        """Writes a LaTeX table summarizing per-cluster results for one cell."""
        if cell_df.empty:
            return
        df = cell_df[cell_df["rule"] == "GP_Rule_1"].copy()
        if df.empty:
            return

        cluster_ids = sorted(df["cluster"].unique())
        baselines = df["baseline"].drop_duplicates().tolist()

        latex: List[str] = [
            "\\begin{table}[htpb]",
            "\\centering",
            "\\caption{Cluster-Stratified Performance: "
            f"{topology.capitalize()} / {activation.capitalize()} "
            "(GP Rank-1 vs Baselines per meta-feature cluster, "
            f"K={self.cfg.k_clusters}; Holm-Bonferroni "
            f"$\\alpha={self.cfg.alpha_level}$)"
            "}",
            "\\resizebox{\\textwidth}{!}{",
            "\\begin{tabular}{l | c | r r r | c}",
            "\\hline",
            "\\textbf{Baseline} & \\textbf{Cluster} "
            "& \\textbf{n} & \\textbf{Wins} & \\textbf{Win-Rate} "
            "& \\textbf{$p_{Holm}$ (Loss)} \\\\",
            "\\hline",
        ]
        for base in baselines:
            base_label = base.replace("_", "\\_")
            for c in cluster_ids:
                row = df[(df["baseline"] == base) & (df["cluster"] == c)]
                if row.empty:
                    continue
                row = row.iloc[0]
                sig_marker = "\\textbf{*}" if row["loss_significant"] else ""
                latex.append(
                    f"{base_label} & {int(c)} "
                    f"& {int(row['n_trials'])} & {int(row['binomial_wins'])} "
                    f"& {row['binomial_win_rate']:.2f} "
                    f"& {row['wilcoxon_loss_pval_holm']:.4f} {sig_marker} \\\\"
                )
            latex.append("\\hline")

        latex.extend([
            "\\end{tabular}",
            "}",
            f"\\label{{tab:cluster_stratified_{topology}_{activation}}}",
            "\\end{table}\n",
        ])
        path = self.cfg.cluster_reports_dir / f"Latex_Cluster_{topology}_{activation}.tex"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(latex))

    # -------------------------------------------------------------------------
    # Cluster-stratified driver
    # -------------------------------------------------------------------------

    def generate_cluster_stratified_report(
        self,
        results: Dict[str, Any],
        topology: str,
        activation: str,
        master_rows: List[Dict[str, Any]],
    ) -> None:
        """Generates cluster-stratified outputs for one (topology, activation) cell.

        Args:
            results: Parsed MOD7 JSON content.
            topology: Topology token.
            activation: Activation token.
            master_rows: Mutable list that collects rows for the master CSV.
                Caller writes the CSV once at the end.
        """
        cell_df = self._compute_cluster_stratified_results(results, topology, activation)
        if cell_df is None or cell_df.empty:
            return
        for _, row in cell_df.iterrows():
            master_rows.append(row.to_dict())
        self._write_cluster_latex(cell_df, topology, activation)

    # -------------------------------------------------------------------------
    # Driver
    # -------------------------------------------------------------------------

    def process_all_artifacts(self) -> None:
        """Iterates every MOD7 JSON and emits LaTeX + plots + cluster-stratified output."""
        json_files = list(self.cfg.mod7_reports_dir.glob("statistical_validation_*.json"))
        if not json_files:
            print(
                f"No MOD7 artifacts in {self.cfg.mod7_reports_dir}. "
                "Run MOD7_pipeline_driver first."
            )
            return

        print("--- INITIALIZING MODULE 8: STATISTICAL REPORTER ---")
        print(f"Source: {self.cfg.mod7_reports_dir}")
        print(f"LaTeX (pooled)  -> {self.cfg.reports_dir}")
        print(f"Plots (pooled)  -> {self.cfg.vis_dir}")
        if self.cfg.run_cluster_stratified:
            print(f"Cluster reports -> {self.cfg.cluster_reports_dir}")
            print(f"Cluster visuals -> {self.cfg.cluster_vis_dir}")

        cluster_ok = False
        if self.cfg.run_cluster_stratified:
            print("\n--- CLUSTER SETUP ---")
            cluster_ok = self._fit_global_clusters()
            if not cluster_ok:
                print("[CLUSTER] Setup failed; only pooled outputs will be produced.")

        master_rows: List[Dict[str, Any]] = []

        for file_path in json_files:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            topology = data.get("Metadata", {}).get("Topology", "unknown")
            activation = data.get("Metadata", {}).get("Activation", "unknown")

            # Pooled outputs (unchanged)
            self.generate_latex_table(data, topology, activation)
            self.generate_visual_distributions(data, topology, activation)

            # Cluster-stratified outputs (new)
            if cluster_ok:
                self.generate_cluster_stratified_report(
                    data, topology, activation, master_rows,
                )

            print(f"  Processed {topology} / {activation}")

        if master_rows:
            master_df = pd.DataFrame(master_rows)
            master_path = self.cfg.cluster_reports_dir / "cluster_master_results.csv"
            master_df.to_csv(master_path, index=False)
            print(
                f"\n[CLUSTER] Master CSV ({len(master_df)} rows) written to {master_path}"
            )

        print("\n--- ALL MOD8 ARTIFACTS EXPORTED ---")


# =============================================================================
# 4. CLI
# =============================================================================

def main() -> None:
    """CLI entry point for MOD8."""
    parser = argparse.ArgumentParser(
        description="MOD8: Statistical reporter for MOD7 validation outputs."
    )
    parser.add_argument(
        "--skip_cluster_stratified", action="store_true",
        help="Bypass the cluster-stratified analysis; produce only pooled outputs.",
    )
    parser.add_argument(
        "--k_clusters", type=int, default=3,
        help="Number of K-means clusters for stratified analysis (default: 3).",
    )
    args = parser.parse_args()

    config = StatisticalConfig(
        run_cluster_stratified=not args.skip_cluster_stratified,
        k_clusters=args.k_clusters,
    )
    generator = ArtifactGenerator(config)
    generator.process_all_artifacts()


if __name__ == "__main__":
    main()