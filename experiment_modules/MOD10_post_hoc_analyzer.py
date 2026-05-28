"""Module 10: Post-Hoc Analyzer.

Performs six additional analyses on the frozen MOD7 / MOD8 / MOD9 outputs.
The methodology and experimental data are unchanged; this module only
re-cuts what already exists to produce evidence the rubric reviewers expect:

  A. He-distance quantification     (no per-trial data needed)
       For each cell's Rank-1 rule, evaluate sigma^2 on the 25 Phase B
       datasets, take the median, and compare to the He target
       sigma^2 = 2 / fan_in. Converts the qualitative "rules rediscover He"
       claim into a number.

  B. Pareto-rank cross-comparison   (no per-trial data needed)
       From MOD7 JSON summary fields, report which Pareto rank wins each
       objective per cell. Addresses the report's own caveat that Rank-1
       (selected by NSGA-II crowding) misrepresents the linear front.

  C. Failure-mode taxonomy          (no per-trial data needed)
       Categorize all rules MOD9 exported (72 in the current run) into six structural classes
       (constant, He-linear, bounded-trig, origin-collapsing, protected-
       operator artifact, unbounded) via SymPy AST inspection. Produces a
       count table + stacked bar figure.

  D. Effect sizes                   (REQUIRES MOD7 per-trial JSONs)
       For each (cell, baseline, objective), Cliff's delta plus a
       bootstrap 95% CI on the paired median difference. Communicates
       practical magnitude alongside the existing Wilcoxon p-values.

  E. Loss visualizations            (REQUIRES MOD7 per-trial JSONs)
       (1) Per-cell terminal-loss boxplots parallel to the accuracy
       distribution figures, and (2) a loss-vs-accuracy means scatter
       that replaces the accuracy-vs-epochs Pareto as the primary view
       (loss is where GP wins).

  F. Cluster-stratified accuracy    (REQUIRES MOD7 per-trial JSONs)
       Parallel to MOD8's loss-only cluster-stratified analysis, but on
       balanced accuracy. K=3 K-means on the Phase-A Z-scored 8-D
       meta-feature space; per-(cell, baseline, cluster) Wilcoxon and
       binomial win-rate; Holm-Bonferroni correction per cell.

Outputs:
    EARV/reports/MOD10_post_hoc/
        he_distance_table.csv
        pareto_rank_comparison.csv
        failure_mode_taxonomy.csv
        effect_sizes.csv
        cluster_stratified_accuracy.csv
    EARV/visualizations/MOD10_post_hoc/
        he_distance_summary.png
        pareto_rank_heatmap.png
        failure_mode_bars.png
        effect_size_forest.png
        loss_distributions_<topology>_<activation>.png   (15)
        loss_vs_accuracy_scatter.png

If MOD7 per-trial JSONs are absent, the per-trial analyses (D, E partial, F)
are skipped with a logged warning; the no-input analyses (A, B, C, loss-vs-
accuracy means scatter) run regardless.

Author: J. M. Arellano  (ACI MSc project, FHNW)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy as sp
from pydantic import BaseModel, Field, field_validator
from scipy import stats
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

log = logging.getLogger("MOD10")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[MOD10] %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


# =============================================================================
# 1. CONSTANTS
# =============================================================================

META_FEATURES: Tuple[str, ...] = (
    "n_d_ratio", "feat_kurtosis", "iqr_dev", "pc_eigen",
    "target_entropy", "hopkins", "silhouette", "davies_bouldin",
)
TOPOLOGIES: Tuple[str, ...] = ("deep_narrow", "funnel", "shallow")
ACTIVATIONS: Tuple[str, ...] = (
    "aggregation", "linear", "rectification", "smooth", "squashing",
)
STRONG_BASELINES: Tuple[str, ...] = (
    "Xavier_Glorot", "He_Kaiming", "LeCun", "Orthogonal", "FAVI",
)
ALL_BASELINES: Tuple[str, ...] = STRONG_BASELINES + ("Laor",)

# He-Kaiming reference for "did the discovery rediscover He?" comparison.
# sigma^2 = 2 / fan_in. Topology-dependent because each topology has a
# different first-hidden-layer width. Confirm against your MOD3 / MOD4
# configuration; defaults reflect the project's reported 64-unit fan-in.
# Per-topology He fan-in, verified against the actual nn.Linear widths in
# MOD4_pm_fnn_landscape.py:
#   PhaseA_Shallow_FNN     : Linear(input_dim, 64) -> Linear(64, 64) -> ...
#   PhaseB_DeepNarrow_FNN  : Linear(input_dim, 32) -> Linear(32, 32) (xN)
#   PhaseB_Funnel_FNN      : Linear(input_dim, 256) -> 128 -> 64 -> 32
#
# Because each rule outputs ONE sigma^2 applied uniformly to every linear
# layer, a single per-topology reference is necessarily a summary. The
# values below take:
#   * deep_narrow : 32   (every internal layer has fan_in=32 -- exact)
#   * shallow     : 64   (every internal layer has fan_in=64 -- exact;
#                         also matches the project's report comparator)
#   * funnel      : 91   (geometric mean of {256,128,64,32} = round((256*128
#                         *64*32)**0.25) = 91 -- the cleanest single summary
#                         of a non-uniform funnel)
# Override on the command line or in the config if your topology configs
# change. The first layer's fan_in is ``input_dim`` (varies per dataset)
# and is therefore not used as the reference.
HE_FAN_IN_BY_TOPOLOGY: Dict[str, int] = {
    "deep_narrow": 32,
    "funnel": 91,
    "shallow": 64,
}

# Recurring symbolic constant observed across the catalogue (~95% of He at
# fan_in=64 / std=0.172). Used as a soft "near-He constant" detector in
# the failure-mode taxonomy.
HE_CONSTANT: float = 0.029732636384331323


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

class PostHocConfig(BaseModel):
    """Configuration for MOD10.

    Attributes:
        earv_root: Path to the experimental_results_analysis_visualizations/
            root (a direct child of experiment_modules/).
        alpha_level: Significance threshold for hypothesis tests.
        n_bootstrap: Number of bootstrap resamples for paired median-difference
            confidence intervals (effect-size analysis).
        bootstrap_seed: NumPy seed for bootstrap reproducibility.
        sigma_floor: Hard floor applied at training time to abs(sigma^2);
            used here so He-distance evaluations match runtime behaviour.
        save_dpi: dpi for saved matplotlib figures.
        k_clusters: K for the cluster-stratified accuracy analysis (match MOD8).
        cluster_random_state: K-means random_state (match MOD8 = 42).
        run_per_trial_analyses: Master switch for analyses requiring MOD7 JSONs.
    """

    earv_root: Path
    alpha_level: float = Field(default=0.05, gt=0.0, lt=1.0)
    n_bootstrap: int = Field(default=2000, gt=99)
    bootstrap_seed: int = Field(default=42)
    sigma_floor: float = Field(default=1e-5, gt=0.0)
    save_dpi: int = Field(default=160, gt=0)
    k_clusters: int = Field(default=3, gt=1)
    cluster_random_state: int = Field(default=42)
    run_per_trial_analyses: bool = True

    @field_validator("earv_root")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()

    # ---- derived paths (kept as properties to avoid coupling Pydantic to fs) ----
    @property
    def mod7_dir(self) -> Path:
        return self.earv_root / "reports" / "MOD7_validation_matrix"

    @property
    def mod9_dir(self) -> Path:
        return self.earv_root / "reports" / "MOD9_qualitative_analysis"

    @property
    def generated_files(self) -> Path:
        # generated_files sits next to experimental_results_analysis_visualizations/
        return self.earv_root.parent / "generated_files"

    @property
    def reports_out(self) -> Path:
        p = self.earv_root / "reports" / "MOD10_post_hoc"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def figures_out(self) -> Path:
        p = self.earv_root / "visualizations" / "MOD10_post_hoc"
        p.mkdir(parents=True, exist_ok=True)
        return p


# =============================================================================
# 3. NUMERICAL UTILITIES
# =============================================================================

def safe_wilcoxon(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided"
                  ) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank with zero-variance short-circuit.

    Returns ``(W_statistic, p_value)``. Returns ``(0.0, 1.0)`` when
    distributions are numerically identical or when SciPy raises
    ``ValueError`` (all-zero differences). Matches MOD8.safe_wilcoxon
    semantics exactly.
    """
    a = np.asarray(x, float); b = np.asarray(y, float)
    if a.size == 0 or b.size == 0:
        return 0.0, 1.0
    if np.allclose(a, b, rtol=1e-7):
        return 0.0, 1.0
    try:
        s = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        return float(s.statistic), float(s.pvalue)
    except ValueError:
        return 0.0, 1.0


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size, vectorized.

    delta = (#{x > y} - #{x < y}) / (|x| * |y|), bounded in [-1, 1].
    Conventions: +ve means x stochastically dominates y.

    The pairwise comparison is computed via np.sign on the outer
    difference, so it is O(n*m) in memory — fine for n=m=125.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    signs = np.sign(x[:, None] - y[None, :])
    return float(signs.mean())


def bootstrap_paired_median_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Bootstrap percentile CI on the median of paired differences (a - b).

    Returns (median, lo, hi) for the central (1 - alpha) interval.
    Paired pairs are resampled with replacement so dependency is preserved.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = min(a.size, b.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = a[:n] - b[:n]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.median(diffs[idx], axis=1)
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(np.median(diffs)), float(lo), float(hi)


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down correction.

    Returns adjusted p-values, clipped to [0, 1]. Matches MOD8's convention
    so downstream comparisons are apples-to-apples.
    """
    p = np.asarray(list(pvals), float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    cum_max = 0.0
    for i, k in enumerate(order):
        v = (m - i) * p[k]
        cum_max = max(cum_max, v)
        adj[k] = min(cum_max, 1.0)
    return adj


def he_target_variance(fan_in: int) -> float:
    """He-Kaiming variance for ReLU-like rectifiers: sigma^2 = 2 / fan_in."""
    return 2.0 / float(fan_in)


# =============================================================================
# 4. DATA LOADERS
# =============================================================================

class MOD7DataLoader:
    """Loads MOD7 ``statistical_validation_*.json`` files into a tidy DataFrame.

    The expected per-cell JSON contains a ``Raw_Distributions`` block keyed by
    method (e.g. ``GP_Rule_1``, ``He_Kaiming``, ``Laor``), each value a dict
    with keys ``acc``, ``loss``, ``epochs``, ``did``, ``seed`` (parallel
    arrays of length 125 = 25 datasets x 5 seeds per cell).
    """

    FILENAME_RE = re.compile(
        r"statistical_validation_(?P<topology>deep_narrow|funnel|shallow)_"
        r"(?P<activation>aggregation|linear|rectification|smooth|squashing)\.json$"
    )

    def __init__(self, cfg: PostHocConfig) -> None:
        self.cfg = cfg

    def _iter_files(self) -> Iterable[Tuple[str, str, Path]]:
        for fp in sorted(self.cfg.mod7_dir.glob("statistical_validation_*.json")):
            m = self.FILENAME_RE.search(fp.name)
            if not m:
                continue
            yield m.group("topology"), m.group("activation"), fp

    def load_per_trial(self) -> pd.DataFrame:
        """Returns the full per-trial long table.

        Schema:
            topology, activation, method, did, seed, acc, loss, epochs
        """
        records: List[dict] = []
        for topology, activation, fp in self._iter_files():
            with fp.open() as fh:
                blob = json.load(fh)
            distributions = blob.get("Raw_Distributions", {})
            if not distributions:
                log.warning("no Raw_Distributions in %s", fp.name)
                continue
            for method, arrs in distributions.items():
                n = min(len(arrs.get(k, [])) for k in ("acc", "loss", "epochs", "did", "seed"))
                if n == 0:
                    continue
                for i in range(n):
                    records.append({
                        "topology": topology, "activation": activation, "method": method,
                        "did": int(arrs["did"][i]), "seed": int(arrs["seed"][i]),
                        "acc": float(arrs["acc"][i]), "loss": float(arrs["loss"][i]),
                        "epochs": float(arrs["epochs"][i]),
                    })
        if not records:
            return pd.DataFrame(columns=[
                "topology","activation","method","did","seed","acc","loss","epochs",
            ])
        df = pd.DataFrame.from_records(records)
        log.info("loaded %d per-trial rows from %d cell JSONs",
                 len(df), df.groupby(["topology","activation"]).ngroups)
        return df

    def load_summary_means(self) -> pd.DataFrame:
        """Returns per-(cell, method) means.

        Reads the summary block at the top of each MOD7 JSON if present;
        falls back to recomputing from Raw_Distributions otherwise.
        Schema: topology, activation, method, Accuracy_Mean, Loss_Mean, Epochs_Mean.
        """
        rows: List[dict] = []
        for topology, activation, fp in self._iter_files():
            with fp.open() as fh:
                blob = json.load(fh)
            summary = blob.get("Aggregates", {}) or {}
            if summary:
                for method, m in summary.items():
                    rows.append({
                        "topology": topology, "activation": activation, "method": method,
                        "Accuracy_Mean": float(m.get("Accuracy_Mean", float("nan"))),
                        "Loss_Mean":     float(m.get("Loss_Mean",     float("nan"))),
                        "Epochs_Mean":   float(m.get("Epochs_Mean",   float("nan"))),
                    })
            else:
                # recompute from raw
                dist = blob.get("Raw_Distributions", {})
                for method, arrs in dist.items():
                    if not arrs.get("loss"):
                        continue
                    rows.append({
                        "topology": topology, "activation": activation, "method": method,
                        "Accuracy_Mean": float(np.mean(arrs["acc"])),
                        "Loss_Mean":     float(np.mean(arrs["loss"])),
                        "Epochs_Mean":   float(np.mean(arrs["epochs"])),
                    })
        return pd.DataFrame.from_records(rows)


class MOD9RuleLoader:
    """Loads MOD9 ``Analytical_Derivatives_*.txt`` files and parses equations.

    Each file's first ``Equation:`` line is the SymPy str-form of the
    simplified rule. The loader returns a DataFrame with one row per
    (activation, topology, rank) carrying the original string and a
    pre-parsed SymPy expression usable for both He-distance evaluation
    and taxonomy classification.
    """

    FILENAME_RE = re.compile(
        r"Analytical_Derivatives_(?P<activation>[a-z]+)_"
        r"(?P<topology>deep_narrow|funnel|shallow)_Rank(?P<rank>[1-5])\.txt$"
    )

    def __init__(self, cfg: PostHocConfig) -> None:
        self.cfg = cfg
        self.symbols: Dict[str, sp.Symbol] = {m: sp.Symbol(m, real=True) for m in META_FEATURES}

    def _parse(self, eq_str: str) -> Optional[sp.Expr]:
        """Parse a MOD9 'Equation:' string to a SymPy expression."""
        # SymPy uses Abs/sqrt/log/exp/sin/cos already in these files.
        try:
            local_dict = dict(self.symbols)
            local_dict.update({"Abs": sp.Abs, "log": sp.log, "exp": sp.exp,
                               "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos,
                               "tan": sp.tan})
            return sp.parse_expr(eq_str, local_dict=local_dict, evaluate=True)
        except (sp.SympifyError, SyntaxError, TypeError) as e:
            log.warning("could not parse equation %r: %s", eq_str, e)
            return None

    def load(self) -> pd.DataFrame:
        rows: List[dict] = []
        for fp in sorted(self.cfg.mod9_dir.glob("Analytical_Derivatives_*.txt")):
            m = self.FILENAME_RE.search(fp.name)
            if not m:
                continue
            text = fp.read_text()
            match = re.search(r"^Equation:\s*(.+)$", text, flags=re.MULTILINE)
            if not match:
                log.warning("no Equation: line in %s", fp.name)
                continue
            eq_str = match.group(1).strip()
            expr = self._parse(eq_str)
            rows.append({
                "activation": m.group("activation"),
                "topology": m.group("topology"),
                "rank": int(m.group("rank")),
                "equation_str": eq_str,
                "expr": expr,
            })
        df = pd.DataFrame.from_records(rows)
        log.info("loaded %d MOD9 equations (%d successfully parsed)",
                 len(df), int(df["expr"].notna().sum()))
        return df


class PhaseBMetaFeatureLoader:
    """Loads Phase B meta-features and applies Phase-A Z-score normalization.

    Returns a tidy DataFrame indexed by ``did`` with the 8 Z-scored
    meta-features as columns — the exact representation the GP saw at
    fitness time, so rule outputs evaluated against it are runtime-accurate.
    """

    def __init__(self, cfg: PostHocConfig) -> None:
        self.cfg = cfg

    def load_normalized(self) -> pd.DataFrame:
        gf = self.cfg.generated_files
        params_fp = gf / "meta_feature_normalization_params.csv"
        phase_b_fp = gf / "Phase_B_Validation_Datasets.csv"
        if not params_fp.exists() or not phase_b_fp.exists():
            raise FileNotFoundError(
                f"Missing inputs for normalization. Need both:\n  {params_fp}\n  {phase_b_fp}"
            )
        params = pd.read_csv(params_fp)            # expected: feature, mean, std
        phase_b = pd.read_csv(phase_b_fp)          # expected: did, <8 meta-features>, ...
        # auto-detect schema if column names differ slightly
        if {"feature", "mean", "std"}.issubset(params.columns):
            mu = params.set_index("feature")["mean"].to_dict()
            sd = params.set_index("feature")["std"].to_dict()
        else:
            # fall back: a single-row DataFrame with feature-named columns and mean/std cells
            mu = {c: float(params[c].iloc[0]) for c in META_FEATURES if c in params.columns}
            sd = {c: float(params[c].iloc[1]) for c in META_FEATURES if c in params.columns}
        out = pd.DataFrame({"did": phase_b["did"].astype(int).values})
        for m in META_FEATURES:
            if m not in phase_b.columns:
                raise KeyError(f"Phase B CSV missing meta-feature column: {m}")
            sd_v = float(sd.get(m, 1.0)) or 1.0
            out[m] = (phase_b[m].astype(float).values - float(mu.get(m, 0.0))) / sd_v
        return out


# =============================================================================
# 5. ANALYSIS A — HE-DISTANCE QUANTIFICATION
# =============================================================================

class HeDistanceAnalyzer:
    """Quantify how close each cell's Rank-1 rule is to a He-Kaiming constant.

    For each cell, evaluate the (simplified) Rank-1 rule on the 25 Phase B
    datasets using their Phase-A Z-scored meta-features, apply the runtime
    consumption ``max(abs(sigma^2), sigma_floor)``, and summarize.
    The headline column is ``ratio_to_He = median(sigma^2) / (2 / fan_in)``.
    Bounded near 1.0 across cells would be quantitative support for the
    qualitative "rules rediscover He" finding.
    """

    def __init__(self, cfg: PostHocConfig) -> None:
        self.cfg = cfg

    def run(self, rules: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        # take Rank-1 only
        r1 = rules[rules["rank"] == 1].copy()
        records: List[dict] = []
        meta_arr = {m: meta[m].values for m in META_FEATURES}
        for _, row in r1.iterrows():
            expr = row["expr"]
            if expr is None:
                continue
            # lambdify for speed (vectorized over the 25 datasets)
            try:
                f = sp.lambdify(
                    [sp.Symbol(m, real=True) for m in META_FEATURES],
                    expr, modules=["numpy"],
                )
                sigma2 = f(*(meta_arr[m] for m in META_FEATURES))
                # broadcast scalar rules (pure constants) to vector
                sigma2 = np.broadcast_to(np.asarray(sigma2, float),
                                         (len(meta_arr["n_d_ratio"]),)).copy()
            except Exception as e:
                log.warning("lambdify failed for %s/%s rank %d: %s",
                            row["activation"], row["topology"], row["rank"], e)
                continue
            # runtime consumption: max(abs(sigma^2), floor)
            consumed = np.maximum(np.abs(sigma2), self.cfg.sigma_floor)
            he = he_target_variance(HE_FAN_IN_BY_TOPOLOGY[row["topology"]])
            records.append({
                "topology": row["topology"], "activation": row["activation"],
                "equation": row["equation_str"],
                "sigma2_median": float(np.median(consumed)),
                "sigma2_iqr_lo": float(np.quantile(consumed, 0.25)),
                "sigma2_iqr_hi": float(np.quantile(consumed, 0.75)),
                "sigma2_min": float(np.min(consumed)),
                "sigma2_max": float(np.max(consumed)),
                "fan_in": HE_FAN_IN_BY_TOPOLOGY[row["topology"]],
                "He_target_sigma2": he,
                "ratio_median_to_He": float(np.median(consumed)) / he,
                "n_pinned_to_floor": int((np.abs(sigma2) < self.cfg.sigma_floor).sum()),
            })
        return pd.DataFrame.from_records(records).sort_values(["topology", "activation"])

    def figure(self, df: pd.DataFrame, out_path: Path) -> None:
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(11, 5.2))
        cells = (df["topology"] + " / " + df["activation"]).tolist()
        ratios = df["ratio_median_to_He"].values
        floor_pinned = df["n_pinned_to_floor"].values
        colors = ["#B5542F" if r > 5 or r < 0.05 else "#065A82" for r in ratios]
        ax.barh(cells, ratios, color=colors, edgecolor="#16242F", linewidth=0.7)
        ax.axvline(1.0, color="#0E7C5A", lw=2, ls="--", label="He target (=1.0)")
        ax.set_xscale("log")
        for i, (r, p) in enumerate(zip(ratios, floor_pinned)):
            note = f"  pinned={p}/25" if p > 0 else ""
            ax.text(r * 1.05, i, f"{r:.2g}{note}", va="center", fontsize=8.5, color="#16242F")
        ax.set_xlabel("median sigma^2 / He target sigma^2  (log scale)")
        ax.set_title("How close is each Rank-1 rule to a He-Kaiming constant?")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.cfg.save_dpi)
        plt.close(fig)


# =============================================================================
# 6. ANALYSIS B — PARETO-RANK CROSS-COMPARISON
# =============================================================================

class ParetoRankComparator:
    """For each cell, identify which Pareto rank wins each objective.

    Uses MOD7 summary means (Accuracy_Mean, Loss_Mean, Epochs_Mean). The
    motivating finding (from the project's own report) is that Rank-1
    misrepresents the linear front: ``deep_narrow_linear`` Rank-1 loss is
    1.98e7 but Rank-2 is ~3.9. This table tells you per cell whether
    Rank-1 is a faithful representative and, if not, which rank wins each
    axis.
    """

    GP_METHOD_RE = re.compile(r"^GP[_ ]Rule[_ ]?(\d+)$", re.IGNORECASE)

    def __init__(self, cfg: PostHocConfig) -> None:
        self.cfg = cfg

    def run(self, summary: pd.DataFrame) -> pd.DataFrame:
        gp = summary.copy()
        gp["rank"] = gp["method"].apply(self._gp_rank)
        gp = gp.dropna(subset=["rank"])
        gp["rank"] = gp["rank"].astype(int)
        records: List[dict] = []
        for (topo, act), sub in gp.groupby(["topology", "activation"]):
            sub = sub.sort_values("rank")
            best_acc_rank   = int(sub.loc[sub["Accuracy_Mean"].idxmax(), "rank"])
            best_loss_rank  = int(sub.loc[sub["Loss_Mean"].idxmin(), "rank"])
            best_epoch_rank = int(sub.loc[sub["Epochs_Mean"].idxmin(), "rank"])
            r1 = sub[sub["rank"] == 1].iloc[0]
            records.append({
                "topology": topo, "activation": act,
                "best_accuracy_rank": best_acc_rank,
                "best_loss_rank": best_loss_rank,
                "best_epoch_rank": best_epoch_rank,
                "rank1_acc": float(r1["Accuracy_Mean"]),
                "rank1_loss": float(r1["Loss_Mean"]),
                "rank1_epochs": float(r1["Epochs_Mean"]),
                "rank1_faithful": bool(best_acc_rank == 1 and best_loss_rank == 1),
                "loss_ratio_rank1_to_best": (
                    float(r1["Loss_Mean"]) / float(sub["Loss_Mean"].min())
                    if sub["Loss_Mean"].min() > 0 else float("nan")
                ),
            })
        return pd.DataFrame.from_records(records).sort_values(["topology", "activation"])

    @classmethod
    def _gp_rank(cls, method: str) -> Optional[int]:
        m = cls.GP_METHOD_RE.match(str(method))
        return int(m.group(1)) if m else None

    def figure(self, df: pd.DataFrame, out_path: Path) -> None:
        if df.empty:
            return
        mat = df.pivot(index="topology", columns="activation",
                       values="loss_ratio_rank1_to_best")
        mat = mat.reindex(index=TOPOLOGIES, columns=ACTIVATIONS)
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdYlBu_r",
                    cbar_kws={"label": "Rank-1 loss / best-rank loss"},
                    linewidths=0.5, linecolor="white", ax=ax,
                    annot_kws={"fontsize": 10})
        ax.set_title("How faithful is Rank-1 as a loss representative? (1.0 = ideal)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.cfg.save_dpi)
        plt.close(fig)


# =============================================================================
# 7. ANALYSIS C — FAILURE-MODE TAXONOMY
# =============================================================================

class FailureModeTaxonomist:
    """Classify every MOD9-exported rule into a structural category.

    The current run exports 72 equations across 15 cells (some cells have
    fewer than 5 ranks in the MOD9 catalogue; the classifier handles
    whatever is present rather than assuming a fixed count).


    Categories (priority order; first match wins so the most-specific
    diagnosis applies):
        - pure_constant       : no meta-feature symbols
        - protected_artifact  : log(Abs(meta)+eps), sqrt(Abs(meta)) cusps,
                                 trig(c/meta) singular-denominator artefacts
        - unbounded           : exp(meta), Pow(meta, k>=2), Mul of two+ metas
                                  outside a bounded wrapper, meta/meta
        - he_linear           : a*meta (+ b), |a| within 0.5x..2x of HE_CONSTANT
        - bounded_trig        : top-level sin/cos wrapper with He-scale coeff
        - origin_collapsing   : a*meta with small a and no offset (passes
                                  through origin at z=0 -> floored)
        - other_feature_dep   : anything feature-dependent not matched above
    """

    def __init__(self, cfg: PostHocConfig) -> None:
        self.cfg = cfg
        self.meta_syms = {sp.Symbol(m, real=True) for m in META_FEATURES}

    def classify(self, expr: Optional[sp.Expr]) -> str:
        if expr is None:
            return "unparseable"
        used = expr.free_symbols & self.meta_syms
        if not used:
            return "pure_constant"
        # protected_artifact: detect characteristic shapes
        if self._has_log_abs_meta(expr) or self._has_sqrt_abs_meta(expr) \
                or self._has_singular_division_in_trig(expr):
            return "protected_artifact"
        # unbounded: exp(meta), pow(meta, k>=2), bare meta/meta, products of metas
        if self._has_exp_meta(expr) or self._has_high_power_meta(expr) \
                or self._has_bare_meta_division(expr) \
                or self._has_multi_meta_product(expr):
            return "unbounded"
        # bounded_trig: top-level sin/cos with He-scale outer coefficient
        if self._is_bounded_trig(expr):
            return "bounded_trig"
        # linear in feature: peel off any constant offset
        coeff, residual = self._peel_offset(expr)
        if self._is_linear_in_meta(residual):
            a = self._linear_slope(residual)
            if a is None:
                return "other_feature_dep"
            scale = 0.5 * HE_CONSTANT
            if scale <= abs(a) <= 2.0 * HE_CONSTANT:
                # offset distinguishes origin-collapse from useful linear
                return "he_linear" if abs(coeff) > 0.5 * HE_CONSTANT else "origin_collapsing"
            return "origin_collapsing"
        return "other_feature_dep"

    # ------------- structural detectors -------------
    def _has_exp_meta(self, e: sp.Expr) -> bool:
        return any(self._has_meta(a.args[0]) for a in e.atoms(sp.exp))

    def _has_log_abs_meta(self, e: sp.Expr) -> bool:
        for a in e.atoms(sp.log):
            inside = a.args[0]
            if any(isinstance(x, sp.Abs) and self._has_meta(x.args[0])
                   for x in sp.preorder_traversal(inside)):
                return True
        return False

    def _has_sqrt_abs_meta(self, e: sp.Expr) -> bool:
        for atom in sp.preorder_traversal(e):
            if isinstance(atom, sp.Pow) and atom.args[1] == sp.Rational(1, 2):
                inside = atom.args[0]
                if any(isinstance(x, sp.Abs) and self._has_meta(x.args[0])
                       for x in sp.preorder_traversal(inside)):
                    return True
        return False

    def _has_high_power_meta(self, e: sp.Expr) -> bool:
        for atom in sp.preorder_traversal(e):
            if isinstance(atom, sp.Pow):
                base, exp = atom.args
                if self._has_meta(base) and exp.is_number and exp >= 2:
                    return True
        return False

    def _has_bare_meta_division(self, e: sp.Expr) -> bool:
        for atom in sp.preorder_traversal(e):
            if isinstance(atom, sp.Pow):
                base, exp = atom.args
                if exp.is_number and exp < 0 and self._has_meta(base) \
                        and not any(isinstance(p, sp.Abs)
                                     for p in sp.preorder_traversal(base)):
                    return True
        return False

    def _has_singular_division_in_trig(self, e: sp.Expr) -> bool:
        for atom in sp.preorder_traversal(e):
            if isinstance(atom, (sp.sin, sp.cos, sp.tan)):
                if self._has_bare_meta_division(atom.args[0]):
                    return True
        return False

    def _has_multi_meta_product(self, e: sp.Expr) -> bool:
        # A top-level Mul (or nested in non-bounding op) of >=2 meta factors
        # that isn't already inside sin/cos (which would bound it).
        for atom in sp.preorder_traversal(e):
            if isinstance(atom, sp.Mul):
                metas = [arg for arg in atom.args if self._has_meta(arg)]
                if len(metas) >= 2:
                    return True
        return False

    def _is_bounded_trig(self, e: sp.Expr) -> bool:
        # outermost wrapper is sin/cos OR e = const * sin/cos(...)
        if isinstance(e, (sp.sin, sp.cos)):
            return True
        if isinstance(e, sp.Mul):
            non_meta_factors = [a for a in e.args if not self._has_meta(a)]
            trig_factors = [a for a in e.args if isinstance(a, (sp.sin, sp.cos))]
            other_factors = [a for a in e.args if a not in non_meta_factors and a not in trig_factors]
            if trig_factors and not other_factors:
                return True
        return False

    def _peel_offset(self, e: sp.Expr) -> Tuple[float, sp.Expr]:
        """Return (constant_offset, residual) such that e = offset + residual."""
        if isinstance(e, sp.Add):
            const = sum((arg for arg in e.args if not self._has_meta(arg)), sp.S.Zero)
            non_const = sum((arg for arg in e.args if self._has_meta(arg)), sp.S.Zero)
            try:
                return float(const), non_const
            except (TypeError, ValueError):
                return 0.0, e
        return 0.0, e

    def _is_linear_in_meta(self, e: sp.Expr) -> bool:
        used = e.free_symbols & self.meta_syms
        if len(used) != 1:
            return False
        m = next(iter(used))
        # detect a*m form: e.diff(m) is a constant, e.diff(m, 2) == 0
        d1 = sp.simplify(sp.diff(e, m))
        d2 = sp.simplify(sp.diff(e, m, 2))
        return bool(d1.is_number) and bool(d2 == 0)

    def _linear_slope(self, e: sp.Expr) -> Optional[float]:
        used = e.free_symbols & self.meta_syms
        if len(used) != 1:
            return None
        m = next(iter(used))
        try:
            return float(sp.simplify(sp.diff(e, m)))
        except (TypeError, ValueError):
            return None

    def _has_meta(self, e: sp.Expr) -> bool:
        return bool(set(getattr(e, "free_symbols", set())) & self.meta_syms)

    # ------------- pipeline -------------
    def run(self, rules: pd.DataFrame) -> pd.DataFrame:
        df = rules.copy()
        df["category"] = df["expr"].apply(self.classify)
        return df[["activation", "topology", "rank", "equation_str", "category"]]

    def figure(self, df: pd.DataFrame, out_path: Path) -> None:
        if df.empty:
            return
        order = ["pure_constant", "he_linear", "bounded_trig",
                 "origin_collapsing", "protected_artifact",
                 "unbounded", "other_feature_dep", "unparseable"]
        cat_palette = {
            "pure_constant": "#065A82", "he_linear": "#0E7C5A",
            "bounded_trig": "#1C7293", "origin_collapsing": "#D4A017",
            "protected_artifact": "#B5542F", "unbounded": "#7A1F1F",
            "other_feature_dep": "#5A6B78", "unparseable": "#B0B0B0",
        }
        pivot = (df.groupby(["activation", "category"]).size()
                   .unstack(fill_value=0).reindex(columns=order, fill_value=0)
                   .reindex(index=ACTIVATIONS))
        fig, ax = plt.subplots(figsize=(11, 4.6))
        pivot.plot(kind="bar", stacked=True, ax=ax,
                   color=[cat_palette[c] for c in pivot.columns],
                   edgecolor="white", linewidth=0.6)
        ax.set_xlabel("activation family")
        ax.set_ylabel("rule count (MOD9-exported rules)")
        ax.set_title("Failure-mode taxonomy of the 75 discovered rules")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, title="category")
        ax.set_xticklabels([a.capitalize() for a in pivot.index], rotation=0)
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.cfg.save_dpi)
        plt.close(fig)


# =============================================================================
# 8. ANALYSIS D — EFFECT SIZES
# =============================================================================

class EffectSizeEstimator:
    """Paired Cliff's delta plus bootstrap median-difference 95% CI.

    For each (cell, baseline, objective in {acc, loss}), compares
    GP_Rule_1 against the baseline on the matched (did, seed) pairs.
    Reports:
      - n_pairs
      - cliff_delta in [-1, 1]
      - median_paired_diff, bootstrap CI_lo, CI_hi
      - already-computed Wilcoxon two-sided p (for cross-reference)
    """

    GP_KEY = "GP_Rule_1"

    def __init__(self, cfg: PostHocConfig) -> None:
        self.cfg = cfg

    def run(self, trials: pd.DataFrame) -> pd.DataFrame:
        if trials.empty:
            return pd.DataFrame()
        rows: List[dict] = []
        for (topology, activation), cell in trials.groupby(["topology", "activation"]):
            gp = cell[cell["method"] == self.GP_KEY]
            if gp.empty:
                log.warning("no %s rows in %s/%s — skipping",
                            self.GP_KEY, topology, activation)
                continue
            for baseline in ALL_BASELINES:
                base = cell[cell["method"] == baseline]
                if base.empty:
                    continue
                merged = pd.merge(
                    gp, base, on=["did", "seed"], suffixes=("_gp", "_base"),
                    how="inner", validate="one_to_one",
                )
                if merged.empty:
                    continue
                for obj in ("acc", "loss"):
                    a = merged[f"{obj}_gp"].to_numpy()
                    b = merged[f"{obj}_base"].to_numpy()
                    sign = +1 if obj == "acc" else -1   # for acc: higher is better; for loss: lower
                    delta = cliffs_delta(a, b) * sign
                    med, lo, hi = bootstrap_paired_median_ci(
                        a, b, n_boot=self.cfg.n_bootstrap,
                        seed=self.cfg.bootstrap_seed, alpha=self.cfg.alpha_level,
                    )
                    _, p_two = safe_wilcoxon(a, b, alternative="two-sided")
                    rows.append({
                        "topology": topology, "activation": activation,
                        "baseline": baseline, "objective": obj, "n_pairs": int(len(a)),
                        "cliffs_delta_gp_favorable": float(delta),
                        "median_paired_diff_gp_minus_base": float(med),
                        "ci95_lo": float(lo), "ci95_hi": float(hi),
                        "wilcoxon_two_sided_p": float(p_two),
                    })
        return pd.DataFrame.from_records(rows)

    def figure(self, df: pd.DataFrame, out_path: Path) -> None:
        if df.empty:
            return
        sub = df[df["objective"] == "loss"].copy()
        if sub.empty:
            return
        # one row per (cell, baseline); cells on y, baselines as facets
        sub["cell"] = sub["topology"] + "/" + sub["activation"]
        order_cells = [f"{t}/{a}" for t in TOPOLOGIES for a in ACTIVATIONS]
        g = sns.catplot(
            data=sub, x="cliffs_delta_gp_favorable", y="cell",
            col="baseline", col_wrap=3, height=3.5, aspect=1.0,
            kind="strip", jitter=False, order=order_cells, palette=["#065A82"],
            edgecolor="white", linewidth=0.7,
        )
        for ax in g.axes.flatten():
            ax.axvline(0, color="#888", lw=1)
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel("Cliff's δ (GP-favorable; loss objective)")
            ax.set_ylabel("")
        g.fig.suptitle("Effect sizes — paired GP vs baseline on terminal loss",
                       y=1.02, fontsize=12)
        g.fig.savefig(out_path, dpi=self.cfg.save_dpi, bbox_inches="tight")
        plt.close(g.fig)


# =============================================================================
# 9. ANALYSIS E — LOSS VISUALIZATIONS
# =============================================================================

class LossVisualizer:
    """Loss boxplots per cell + a loss-vs-accuracy means scatter.

    The scatter uses summary means only (so it runs with or without
    per-trial data); the boxplots require per-trial data.
    """

    PALETTE: Dict[str, str] = {
        "GP_Rule_1": "#16242F", "GP_Rule_2": "#3A4A5A",
        "Xavier_Glorot": "#1C7293", "He_Kaiming": "#0E7C5A",
        "LeCun": "#5A6B78",        "Orthogonal": "#7BAFD4",
        "FAVI": "#B5542F",         "Laor": "#D4A017",
    }

    def __init__(self, cfg: PostHocConfig) -> None:
        self.cfg = cfg

    def boxplots(self, trials: pd.DataFrame, out_dir: Path) -> List[Path]:
        if trials.empty:
            log.warning("no per-trial data — skipping loss boxplots")
            return []
        paths: List[Path] = []
        for (topo, act), sub in trials.groupby(["topology", "activation"]):
            fig, ax = plt.subplots(figsize=(10, 4.6))
            methods = [m for m in self.PALETTE if (sub["method"] == m).any()]
            data = [sub.loc[sub["method"] == m, "loss"].values for m in methods]
            # NOTE: `labels=` was removed from Axes.boxplot in matplotlib 3.9+
            # (renamed to `tick_labels`). Drop it from the call and set tick
            # labels explicitly — this works on every matplotlib version.
            bp = ax.boxplot(
                data, vert=True, patch_artist=True,
                medianprops={"color": "white", "linewidth": 2},
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
                widths=0.65,
            )
            ax.set_xticks(range(1, len(methods) + 1))
            ax.set_xticklabels(methods)
            for patch, m in zip(bp["boxes"], methods):
                patch.set_facecolor(self.PALETTE[m])
                patch.set_edgecolor("#16242F")
            ax.set_title(f"Terminal-loss distribution — {act} / {topo}")
            ax.set_ylabel("terminal cross-entropy loss")
            ax.tick_params(axis="x", rotation=20)
            # log scale rescues the linear cells
            try:
                if sub["loss"].max() / max(sub["loss"].min(), 1e-9) > 1e3:
                    ax.set_yscale("log")
            except Exception:
                pass
            fig.tight_layout()
            fp = out_dir / f"loss_distributions_{topo}_{act}.png"
            fig.savefig(fp, dpi=self.cfg.save_dpi)
            plt.close(fig)
            paths.append(fp)
        return paths

    def scatter_means(self, summary: pd.DataFrame, out_path: Path) -> None:
        if summary.empty:
            return
        sub = summary.copy()
        if "Accuracy_Mean" not in sub.columns or "Loss_Mean" not in sub.columns:
            return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
        for ax, topo in zip(axes, TOPOLOGIES):
            cell = sub[sub["topology"] == topo]
            if cell.empty:
                continue
            for method, grp in cell.groupby("method"):
                color = self.PALETTE.get(method, "#888")
                ax.scatter(grp["Accuracy_Mean"], grp["Loss_Mean"],
                           label=method, color=color, edgecolor="#16242F",
                           s=70, alpha=0.9, linewidth=0.7)
            ax.set_xlabel("mean balanced accuracy (%)")
            ax.set_ylabel("mean terminal loss")
            ax.set_title(topo)
            # log y if linear cells stretch the axis
            try:
                if cell["Loss_Mean"].max() / max(cell["Loss_Mean"].min(), 1e-9) > 1e3:
                    ax.set_yscale("log")
            except Exception:
                pass
        axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        fig.suptitle("Loss vs. accuracy — the view that puts GP's winning axis first",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.cfg.save_dpi, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 10. ANALYSIS F — CLUSTER-STRATIFIED ACCURACY
# =============================================================================

class ClusterStratifiedAccuracyAnalyzer:
    """Parallel to MOD8's cluster-stratified loss analysis, but on accuracy.

    Reuses the same Phase-A Z-scored 8-D meta-feature space and the same
    K (default 3) as MOD8, so cluster geometry matches and the two
    stratified analyses are directly comparable.
    """

    def __init__(self, cfg: PostHocConfig) -> None:
        self.cfg = cfg

    def _assign_clusters(self, meta: pd.DataFrame) -> Dict[int, int]:
        X = meta[list(META_FEATURES)].to_numpy()
        km = KMeans(n_clusters=self.cfg.k_clusters, n_init=10,
                    random_state=self.cfg.cluster_random_state)
        labels = km.fit_predict(X)
        return dict(zip(meta["did"].astype(int).tolist(), labels.tolist()))

    def run(self, trials: pd.DataFrame, meta_normalized: pd.DataFrame
            ) -> pd.DataFrame:
        if trials.empty:
            return pd.DataFrame()
        clusters = self._assign_clusters(meta_normalized)
        trials = trials.copy()
        trials["cluster"] = trials["did"].map(clusters)
        records: List[dict] = []
        for (topo, act), cell in trials.groupby(["topology", "activation"]):
            gp = cell[cell["method"] == "GP_Rule_1"]
            if gp.empty:
                continue
            cell_rows: List[dict] = []
            for baseline in ALL_BASELINES:
                base = cell[cell["method"] == baseline]
                if base.empty:
                    continue
                merged = pd.merge(
                    gp, base, on=["did", "seed", "cluster"],
                    suffixes=("_gp", "_base"), how="inner",
                )
                for cluster_id, m in merged.groupby("cluster"):
                    a = m["acc_gp"].to_numpy(); b = m["acc_base"].to_numpy()
                    if len(a) < 5:
                        continue
                    _, p = safe_wilcoxon(a, b, alternative="greater")
                    wins = int(np.sum(a > b))
                    cell_rows.append({
                        "topology": topo, "activation": act, "baseline": baseline,
                        "cluster": int(cluster_id), "n_pairs": int(len(a)),
                        "wins": wins, "win_rate": float(wins / len(a)),
                        "p_raw": float(p),
                    })
            if not cell_rows:
                continue
            # Holm-Bonferroni per cell across (baseline, cluster) tests
            ps = [r["p_raw"] for r in cell_rows]
            adj = holm_bonferroni(ps, alpha=self.cfg.alpha_level)
            for r, a in zip(cell_rows, adj):
                r["p_holm"] = float(a)
                r["significant"] = bool(a < self.cfg.alpha_level)
            records.extend(cell_rows)
        return pd.DataFrame.from_records(records)


# =============================================================================
# 11. ORCHESTRATOR
# =============================================================================

class PostHocOrchestrator:
    """Runs every available analysis; logs and skips those whose inputs are missing."""

    def __init__(self, cfg: PostHocConfig) -> None:
        self.cfg = cfg

    def run(self) -> Dict[str, Path]:
        outputs: Dict[str, Path] = {}
        rules = MOD9RuleLoader(self.cfg).load()
        try:
            meta = PhaseBMetaFeatureLoader(self.cfg).load_normalized()
            has_meta = True
        except FileNotFoundError as e:
            log.warning("meta-feature loader: %s", e)
            meta, has_meta = pd.DataFrame(), False

        # ----- A. He-distance -----
        if has_meta and not rules.empty:
            log.info("[A] He-distance quantification")
            he = HeDistanceAnalyzer(self.cfg)
            tbl = he.run(rules, meta)
            fp_csv = self.cfg.reports_out / "he_distance_table.csv"
            tbl.to_csv(fp_csv, index=False)
            fp_png = self.cfg.figures_out / "he_distance_summary.png"
            he.figure(tbl, fp_png)
            outputs["he_distance_csv"] = fp_csv
            outputs["he_distance_fig"] = fp_png

        # ----- B. Pareto-rank cross-comparison -----
        loader = MOD7DataLoader(self.cfg)
        summary = loader.load_summary_means() if self.cfg.mod7_dir.exists() else pd.DataFrame()
        if not summary.empty:
            log.info("[B] Pareto-rank cross-comparison")
            prc = ParetoRankComparator(self.cfg)
            tbl = prc.run(summary)
            fp_csv = self.cfg.reports_out / "pareto_rank_comparison.csv"
            tbl.to_csv(fp_csv, index=False)
            fp_png = self.cfg.figures_out / "pareto_rank_heatmap.png"
            prc.figure(tbl, fp_png)
            outputs["pareto_rank_csv"] = fp_csv
            outputs["pareto_rank_fig"] = fp_png
        else:
            log.warning("[B] Pareto-rank skipped: no MOD7 summary in %s", self.cfg.mod7_dir)

        # ----- C. Failure-mode taxonomy -----
        if not rules.empty:
            log.info("[C] Failure-mode taxonomy")
            tax = FailureModeTaxonomist(self.cfg)
            tbl = tax.run(rules)
            fp_csv = self.cfg.reports_out / "failure_mode_taxonomy.csv"
            tbl.to_csv(fp_csv, index=False)
            fp_png = self.cfg.figures_out / "failure_mode_bars.png"
            tax.figure(tbl, fp_png)
            outputs["failure_mode_csv"] = fp_csv
            outputs["failure_mode_fig"] = fp_png

        # ----- E (means scatter — doesn't need per-trial) -----
        if not summary.empty:
            log.info("[E] Loss-vs-accuracy means scatter")
            fp = self.cfg.figures_out / "loss_vs_accuracy_scatter.png"
            LossVisualizer(self.cfg).scatter_means(summary, fp)
            outputs["loss_vs_acc_fig"] = fp

        # ----- per-trial analyses (D, E boxplots, F) -----
        if self.cfg.run_per_trial_analyses:
            trials = loader.load_per_trial() if self.cfg.mod7_dir.exists() else pd.DataFrame()
            if not trials.empty:
                log.info("[D] Effect sizes (Cliff's delta + bootstrap CI)")
                eff = EffectSizeEstimator(self.cfg)
                tbl = eff.run(trials)
                fp_csv = self.cfg.reports_out / "effect_sizes.csv"
                tbl.to_csv(fp_csv, index=False)
                fp_png = self.cfg.figures_out / "effect_size_forest.png"
                eff.figure(tbl, fp_png)
                outputs["effect_sizes_csv"] = fp_csv
                outputs["effect_sizes_fig"] = fp_png

                log.info("[E] Per-cell loss boxplots")
                paths = LossVisualizer(self.cfg).boxplots(trials, self.cfg.figures_out)
                outputs["loss_boxplots"] = paths

                if has_meta:
                    log.info("[F] Cluster-stratified accuracy (Holm-Bonferroni)")
                    csa = ClusterStratifiedAccuracyAnalyzer(self.cfg)
                    tbl = csa.run(trials, meta)
                    fp_csv = self.cfg.reports_out / "cluster_stratified_accuracy.csv"
                    tbl.to_csv(fp_csv, index=False)
                    outputs["cluster_strat_acc_csv"] = fp_csv
            else:
                log.warning("per-trial analyses (D, E boxplots, F) skipped: no MOD7 JSONs")

        log.info("MOD10 complete. %d outputs written under %s",
                 len(outputs), self.cfg.earv_root)
        return outputs


# =============================================================================
# 12. ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOD10: Post-hoc analyses on frozen ACI results."
    )
    # Default mirrors MOD8: when run from experiment_modules/, the EARV path
    # is a direct child of __file__'s parent. Override via --earv-root if MOD10
    # is invoked from elsewhere (e.g. a test harness).
    default_earv = Path(__file__).resolve().parent / "experimental_results_analysis_visualizations"
    parser.add_argument(
        "--earv-root", type=Path, default=default_earv,
        help=(f"Path to experimental_results_analysis_visualizations/  "
              f"(default: {default_earv})"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--no-per-trial", action="store_true",
                        help="Skip analyses that require MOD7 per-trial JSONs.")
    args = parser.parse_args()

    cfg = PostHocConfig(
        earv_root=args.earv_root, alpha_level=args.alpha,
        n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed,
        run_per_trial_analyses=not args.no_per_trial,
    )
    PostHocOrchestrator(cfg).run()


if __name__ == "__main__":
    main()