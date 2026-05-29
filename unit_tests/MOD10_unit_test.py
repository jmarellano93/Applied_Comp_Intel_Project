"""Unit tests for MOD10_post_hoc_analyzer.

Covers:
    * Numeric helpers (cliffs_delta, bootstrap_paired_median_ci,
      holm_bonferroni, safe_wilcoxon).
    * FailureModeTaxonomist classifier on a curated set of MOD9 equations
      whose categories are known a priori.
    * HeDistanceAnalyzer end-to-end on synthetic Phase B meta-features
      with a pure-constant rule (sanity).
    * ParetoRankComparator on a fabricated MOD7 summary frame.
    * PhaseBMetaFeatureLoader normalization against a hand-computed
      Z-score baseline.
    * Schema robustness of MOD7DataLoader against an in-memory JSON.

Run from inside ``experiment_modules/``::

    pytest -xvs MOD10_unit_test.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import sympy as sp

# --- Path bootstrap (added after relocating this test to unit_tests/) -------
# The modules under test live in the sibling experiment_modules/ directory.
# Add it to sys.path so the bare ``import MOD...`` below resolves regardless of
# the current working directory or how pytest is launched.
import sys as _sys
from pathlib import Path as _Path
_MODULES_DIR = _Path(__file__).resolve().parent.parent / "experiment_modules"
if str(_MODULES_DIR) not in _sys.path:
    _sys.path.insert(0, str(_MODULES_DIR))
# ---------------------------------------------------------------------------

from MOD10_post_hoc_analyzer import (
    META_FEATURES, HE_CONSTANT, HE_FAN_IN_BY_TOPOLOGY,
    PostHocConfig, cliffs_delta, bootstrap_paired_median_ci,
    holm_bonferroni, safe_wilcoxon, he_target_variance,
    FailureModeTaxonomist, HeDistanceAnalyzer, ParetoRankComparator,
    PhaseBMetaFeatureLoader, MOD7DataLoader, MOD9RuleLoader,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _cfg(tmp_path: Path) -> PostHocConfig:
    return PostHocConfig(earv_root=tmp_path, n_bootstrap=300)


@pytest.fixture
def tax(tmp_path) -> FailureModeTaxonomist:
    return FailureModeTaxonomist(_cfg(tmp_path))


# -----------------------------------------------------------------------------
# 1. NUMERIC HELPERS
# -----------------------------------------------------------------------------

class TestCliffsDelta:
    def test_full_dominance(self):
        x = np.array([10, 11, 12]); y = np.array([0, 1, 2])
        assert cliffs_delta(x, y) == pytest.approx(1.0)

    def test_reverse_dominance(self):
        x = np.array([0, 1, 2]); y = np.array([10, 11, 12])
        assert cliffs_delta(x, y) == pytest.approx(-1.0)

    def test_identical(self):
        x = np.array([1, 2, 3]); y = np.array([1, 2, 3])
        # ties contribute 0 to the sign sum, so delta exactly 0
        assert cliffs_delta(x, y) == pytest.approx(0.0)

    def test_empty(self):
        assert np.isnan(cliffs_delta(np.array([]), np.array([1.0])))


class TestBootstrapCI:
    def test_constant_diff_produces_tight_ci(self):
        a = np.full(100, 5.0); b = np.full(100, 3.0)
        med, lo, hi = bootstrap_paired_median_ci(a, b, n_boot=500, seed=0)
        assert med == pytest.approx(2.0)
        assert lo == pytest.approx(2.0)
        assert hi == pytest.approx(2.0)

    def test_zero_diff(self):
        a = np.full(50, 1.0)
        med, lo, hi = bootstrap_paired_median_ci(a, a, n_boot=300, seed=1)
        assert med == 0.0 and lo == 0.0 and hi == 0.0

    def test_random_paired_centered(self):
        rng = np.random.default_rng(7)
        a = rng.normal(0, 1, 200); b = a - 1.0 + rng.normal(0, 0.05, 200)
        med, lo, hi = bootstrap_paired_median_ci(a, b, n_boot=500, seed=2)
        # paired diff median is ~+1.0 by construction
        assert lo < 1.0 < hi
        assert med == pytest.approx(1.0, abs=0.1)


class TestHolmBonferroni:
    def test_monotone_nondecreasing_in_sorted_order(self):
        p = np.array([0.001, 0.01, 0.04, 0.06, 0.20])
        adj = holm_bonferroni(p)
        # adjusted p must be non-decreasing in the sorted-by-raw order
        order = np.argsort(p)
        assert np.all(np.diff(adj[order]) >= -1e-12)

    def test_clipping_to_one(self):
        adj = holm_bonferroni([0.9, 0.95, 0.99])
        assert np.all(adj <= 1.0 + 1e-12)


class TestSafeWilcoxon:
    def test_zero_diff_no_raise(self):
        x = np.array([1.0, 2.0, 3.0])
        stat, p = safe_wilcoxon(x, x)
        # MOD8 convention: (0.0, 1.0) on zero-variance, not (nan, 1.0)
        assert stat == 0.0 and p == 1.0

    def test_normal_paired(self):
        rng = np.random.default_rng(42)
        x = rng.normal(1, 1, 30); y = rng.normal(0, 1, 30)
        _, p = safe_wilcoxon(x, y, alternative="greater")
        assert 0.0 <= p <= 1.0


# -----------------------------------------------------------------------------
# 2. FAILURE-MODE TAXONOMIST
# -----------------------------------------------------------------------------

class TestTaxonomy:
    """Each case is a real MOD9-shaped equation with a known category."""

    @pytest.fixture(autouse=True)
    def _init(self, tax):
        self.tax = tax
        self.symbols = {m: sp.Symbol(m, real=True) for m in META_FEATURES}

    def _parse(self, s: str) -> sp.Expr:
        return sp.parse_expr(s, local_dict={**self.symbols,
                                            "Abs": sp.Abs, "log": sp.log,
                                            "exp": sp.exp, "sqrt": sp.sqrt,
                                            "sin": sp.sin, "cos": sp.cos})

    @pytest.mark.parametrize("eq,expected", [
        # pure constants
        ("-0.0383", "pure_constant"),
        ("0.0223", "pure_constant"),
        # unbounded — linear-family pathology
        ("n_d_ratio**2", "unbounded"),
        ("feat_kurtosis*n_d_ratio*target_entropy", "unbounded"),
        ("exp(target_entropy/feat_kurtosis)", "unbounded"),
        ("hopkins*iqr_dev*n_d_ratio*pc_eigen", "unbounded"),
        # protected_artifact
        ("0.0297*sin(0.398/davies_bouldin)", "protected_artifact"),
        ("0.0297*log(Abs(hopkins) + 1/100000)", "protected_artifact"),
        ("0.0297*sqrt(Abs(feat_kurtosis)) + 0.00466", "protected_artifact"),
        # bounded_trig with He-scale envelope
        ("0.0297*cos(hopkins)", "bounded_trig"),
        # he_linear
        ("0.0297*pc_eigen - 0.00557", "origin_collapsing"),
        # origin-collapsing (linear through 0, He-scale-ish slope, no offset)
        ("0.0297*silhouette", "origin_collapsing"),
        ("0.0432*n_d_ratio", "origin_collapsing"),
    ])
    def test_classification(self, eq, expected):
        assert self.tax.classify(self._parse(eq)) == expected

    def test_unparseable(self):
        assert self.tax.classify(None) == "unparseable"


# -----------------------------------------------------------------------------
# 3. HE-DISTANCE
# -----------------------------------------------------------------------------

class TestHeDistance:
    def test_pure_constant_rule_evaluates_exactly(self, tmp_path):
        cfg = _cfg(tmp_path)
        rng = np.random.default_rng(0)
        meta = pd.DataFrame({"did": np.arange(25), **{m: rng.normal(0, 1, 25)
                                                       for m in META_FEATURES}})
        rules = pd.DataFrame([{
            "activation": "rectification", "topology": "deep_narrow", "rank": 1,
            "equation_str": "0.0297", "expr": sp.Float(0.0297),
        }])
        tbl = HeDistanceAnalyzer(cfg).run(rules, meta)
        assert len(tbl) == 1
        row = tbl.iloc[0]
        assert row["sigma2_median"] == pytest.approx(0.0297, rel=1e-9)
        assert row["He_target_sigma2"] == pytest.approx(
            he_target_variance(HE_FAN_IN_BY_TOPOLOGY["deep_narrow"]), rel=1e-9)
        assert row["n_pinned_to_floor"] == 0

    def test_negative_rule_is_floored_via_abs(self, tmp_path):
        cfg = _cfg(tmp_path)
        meta = pd.DataFrame({"did": np.arange(25),
                             **{m: np.zeros(25) for m in META_FEATURES}})
        rules = pd.DataFrame([{
            "activation": "smooth", "topology": "shallow", "rank": 1,
            "equation_str": "-0.0297", "expr": sp.Float(-0.0297),
        }])
        tbl = HeDistanceAnalyzer(cfg).run(rules, meta)
        # abs(-0.0297) -> 0.0297, not the floor
        assert tbl.iloc[0]["sigma2_median"] == pytest.approx(0.0297, rel=1e-9)


# -----------------------------------------------------------------------------
# 4. PARETO-RANK COMPARATOR
# -----------------------------------------------------------------------------

class TestParetoRankComparator:
    def test_linear_rank1_unfaithful(self, tmp_path):
        # mirrors the project's deep_narrow_linear story: Rank-1 loss is
        # extreme, Rank-2 is well-behaved.
        summary = pd.DataFrame([
            {"topology":"deep_narrow","activation":"linear","method":"GP_Rule_1","Accuracy_Mean":70.57,"Loss_Mean":1.98e7,"Epochs_Mean":29.0},
            {"topology":"deep_narrow","activation":"linear","method":"GP_Rule_2","Accuracy_Mean":71.0, "Loss_Mean":3.9,    "Epochs_Mean":28.0},
            {"topology":"deep_narrow","activation":"linear","method":"GP_Rule_3","Accuracy_Mean":70.8, "Loss_Mean":4.5,    "Epochs_Mean":27.5},
        ])
        prc = ParetoRankComparator(_cfg(tmp_path))
        tbl = prc.run(summary)
        row = tbl.iloc[0]
        assert row["best_loss_rank"] == 2
        assert row["best_epoch_rank"] == 3
        assert not row["rank1_faithful"]
        assert row["loss_ratio_rank1_to_best"] > 1e6

    def test_strength_cell_rank1_faithful(self, tmp_path):
        summary = pd.DataFrame([
            {"topology":"deep_narrow","activation":"squashing","method":"GP_Rule_1","Accuracy_Mean":77.32,"Loss_Mean":0.4766,"Epochs_Mean":18.5},
            {"topology":"deep_narrow","activation":"squashing","method":"GP_Rule_2","Accuracy_Mean":77.07,"Loss_Mean":0.5408,"Epochs_Mean":19.0},
        ])
        tbl = ParetoRankComparator(_cfg(tmp_path)).run(summary)
        row = tbl.iloc[0]
        assert row["best_loss_rank"] == 1
        assert row["rank1_faithful"]
        assert row["loss_ratio_rank1_to_best"] == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# 5. PHASE B META-FEATURE NORMALIZATION
# -----------------------------------------------------------------------------

class TestPhaseBLoader:
    def test_zscore_correctness(self, tmp_path):
        # build minimal generated_files/ with a known mean/std
        gen = tmp_path.parent / "generated_files"
        gen.mkdir(parents=True, exist_ok=True)
        params = pd.DataFrame([{"feature": m, "mean": 10.0, "std": 2.0}
                               for m in META_FEATURES])
        params.to_csv(gen / "meta_feature_normalization_params.csv", index=False)
        raw = {"did": [1, 2, 3]}
        for m in META_FEATURES:
            raw[m] = [10.0, 12.0, 8.0]   # z = 0, +1, -1
        pd.DataFrame(raw).to_csv(gen / "Phase_B_Validation_Datasets.csv", index=False)

        # earv_root.parent must equal gen.parent (per PostHocConfig.generated_files)
        cfg = PostHocConfig(earv_root=tmp_path)
        out = PhaseBMetaFeatureLoader(cfg).load_normalized()
        assert list(out["did"]) == [1, 2, 3]
        for m in META_FEATURES:
            np.testing.assert_allclose(out[m].values, [0.0, 1.0, -1.0], atol=1e-12)


# -----------------------------------------------------------------------------
# 6. MOD7 DATA LOADER (in-memory JSON round trip)
# -----------------------------------------------------------------------------

class TestMOD7Loader:
    def test_loads_per_trial_from_synthetic_json(self, tmp_path):
        d = tmp_path / "reports" / "MOD7_validation_matrix"
        d.mkdir(parents=True)
        blob = {"Raw_Distributions": {
            "GP_Rule_1": {"acc": [80.0, 75.0], "loss": [0.5, 0.6],
                          "epochs": [10, 12], "did": [1, 2], "seed": [0, 0]},
            "Laor":      {"acc": [75.0, 70.0], "loss": [0.7, 0.8],
                          "epochs": [12, 14], "did": [1, 2], "seed": [0, 0]},
        }}
        (d / "statistical_validation_funnel_squashing.json").write_text(json.dumps(blob))
        cfg = PostHocConfig(earv_root=tmp_path)
        df = MOD7DataLoader(cfg).load_per_trial()
        assert len(df) == 4
        assert set(df["method"]) == {"GP_Rule_1", "Laor"}
        assert set(df["topology"]) == {"funnel"}
        assert set(df["activation"]) == {"squashing"}


# -----------------------------------------------------------------------------
# 7. MOD9 RULE LOADER (in-memory file round trip)
# -----------------------------------------------------------------------------

class TestMOD9Loader:
    def test_parses_known_equation(self, tmp_path):
        d = tmp_path / "reports" / "MOD9_qualitative_analysis"
        d.mkdir(parents=True)
        (d / "Analytical_Derivatives_squashing_deep_narrow_Rank1.txt").write_text(
            "Equation: 0.0297326363843313*sqrt(Abs(feat_kurtosis)) + 0.00465900679264226\n"
            "Partial: ...\n"
        )
        cfg = PostHocConfig(earv_root=tmp_path)
        df = MOD9RuleLoader(cfg).load()
        assert len(df) == 1
        assert df.iloc[0]["expr"] is not None
        assert df.iloc[0]["rank"] == 1


# -----------------------------------------------------------------------------
# 8. ORCHESTRATOR (smoke test: doesn't crash on empty inputs)
# -----------------------------------------------------------------------------

class TestOrchestratorSmoke:
    def test_empty_runs_without_raising(self, tmp_path):
        from MOD10_post_hoc_analyzer import PostHocOrchestrator
        cfg = PostHocConfig(earv_root=tmp_path, run_per_trial_analyses=False)
        outputs = PostHocOrchestrator(cfg).run()
        # nothing to produce, but the call should not raise
        assert isinstance(outputs, dict)
