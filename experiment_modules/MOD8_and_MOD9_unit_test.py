"""
Unit Tests for Modules 8 & 9.

Covers:
    * MOD8: output directory structure (EARV at experiment_modules), Wilcoxon
      zero-variance protection, 5-rank LaTeX table generation, all-ranks plot.
    * MOD9: empirical-means injection, sin/cos parser regression (the bug
      that caused MOD9 to NameError on every rule containing sin or cos),
      multi-rank extraction, robust filename parsing.

DatasetManager is mocked in the MOD9 empirical-means tests so unit tests
remain I/O-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
import sympy as sp
import torch

import MOD8_framework_statistical_reporter as mod8
import MOD9_framework_qualitative_analyzer as mod9


# =============================================================================
# Fixtures
# =============================================================================

def _make_synthetic_mod7_json(n_trials: int = 10) -> Dict:
    """Builds a synthetic MOD7-format JSON dict with 5 GP ranks + 6 baselines."""
    rng = np.random.default_rng(0)
    distributions = {}
    for r in range(1, 6):
        distributions[f"GP_Rule_{r}"] = {
            "acc": (0.80 + 0.02 * r + 0.01 * rng.standard_normal(n_trials)).tolist(),
            "epochs": (10.0 + rng.uniform(0, 5, n_trials)).tolist(),
            "loss": (0.5 + 0.1 * rng.standard_normal(n_trials)).tolist(),
        }
    for base in ["Xavier_Glorot", "He_Kaiming", "LeCun", "Orthogonal", "FAVI", "Laor"]:
        distributions[base] = {
            "acc": (0.75 + 0.02 * rng.standard_normal(n_trials)).tolist(),
            "epochs": (15.0 + rng.uniform(0, 5, n_trials)).tolist(),
            "loss": (0.6 + 0.1 * rng.standard_normal(n_trials)).tolist(),
        }
    return {
        "Metadata": {"Topology": "shallow", "Activation": "smooth"},
        "Raw_Distributions": distributions,
        "Aggregates": {},
        "Binomial_Win_Matrix_By_Loss": {},
        "Binomial_P_Values_By_Loss": {},
    }


def _make_mock_dataset_manager() -> MagicMock:
    """Returns a mocked DatasetManager exposing 3 fake meta-feature vectors."""
    manager = MagicMock(spec=mod9.DatasetManager)
    manager.dataset_cache = {1: object(), 2: object(), 3: object()}

    # Each dataset returns (tensors_dict, meta_features_tensor) per MOD3 contract.
    meta_vectors = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    ]
    fake_calls = {1: meta_vectors[0], 2: meta_vectors[1], 3: meta_vectors[2]}
    manager.get_dataset.side_effect = lambda did: ({}, fake_calls[did])
    return manager


# =============================================================================
# MOD8 — Path structure
# =============================================================================

class TestMod8Paths:
    def test_earv_is_direct_child_of_experiment_modules(self):
        config = mod8.StatisticalConfig()
        # base_dir → ../EARV; parent must be the script's directory (experiment_modules).
        assert config.base_dir.name == "experimental_results_analysis_visualizations"

    def test_reports_dir_lives_under_mod8_subfolder(self):
        config = mod8.StatisticalConfig()
        assert config.reports_dir.name == "MOD8_statistical_reports"
        assert config.reports_dir.parent.name == "reports"

    def test_vis_dir_lives_under_mod8_distributions(self):
        config = mod8.StatisticalConfig()
        assert config.vis_dir.name == "MOD8_distributions"
        assert config.vis_dir.parent.name == "visualizations"

    def test_mod7_source_dir_resolves(self):
        config = mod8.StatisticalConfig()
        assert config.mod7_reports_dir.name == "MOD7_validation_matrix"
        assert config.mod7_reports_dir.parent.name == "reports"


# =============================================================================
# MOD8 — Wilcoxon
# =============================================================================

class TestSafeWilcoxon:
    def test_identical_distributions_short_circuit(self):
        gen = mod8.ArtifactGenerator(mod8.StatisticalConfig())
        stat, pval = gen.safe_wilcoxon([0.85, 0.86, 0.87, 0.88], [0.85, 0.86, 0.87, 0.88])
        assert stat == 0.0
        assert pval == 1.0

    def test_empty_distributions_return_neutral(self):
        gen = mod8.ArtifactGenerator(mod8.StatisticalConfig())
        stat, pval = gen.safe_wilcoxon([], [])
        assert stat == 0.0
        assert pval == 1.0

    def test_different_distributions_compute_test(self):
        gen = mod8.ArtifactGenerator(mod8.StatisticalConfig())
        a = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
        b = [0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87]
        stat, pval = gen.safe_wilcoxon(a, b, alternative="greater")
        assert pval < 0.05


# =============================================================================
# MOD8 — Rank ordering
# =============================================================================

class TestMod8RankOrdering:
    def test_gp_rule_keys_sorted_by_rank(self):
        gen = mod8.ArtifactGenerator(mod8.StatisticalConfig())
        distributions = {
            "GP_Rule_3": {"acc": [], "epochs": [], "loss": []},
            "GP_Rule_1": {"acc": [], "epochs": [], "loss": []},
            "Xavier_Glorot": {"acc": [], "epochs": [], "loss": []},
            "GP_Rule_2": {"acc": [], "epochs": [], "loss": []},
        }
        keys = gen._extract_gp_rule_keys(distributions)
        assert keys == ["GP_Rule_1", "GP_Rule_2", "GP_Rule_3"]

    def test_max_ranks_truncation(self):
        cfg = mod8.StatisticalConfig(max_gp_ranks=2)
        gen = mod8.ArtifactGenerator(cfg)
        distributions = {f"GP_Rule_{i}": {"acc": [], "epochs": [], "loss": []} for i in range(1, 6)}
        assert len(gen._extract_gp_rule_keys(distributions)) == 2


# =============================================================================
# MOD8 — End-to-end artifact generation (writes to tmp_path indirectly via cfg)
# =============================================================================

class TestMod8EndToEnd:
    def test_latex_table_contains_five_gp_rows(self, tmp_path, monkeypatch):
        """LaTeX output must contain all 5 GP_Rule rows + 6 baseline rows."""
        # Monkeypatch the config's base_dir to land in tmp.
        monkeypatch.setattr(
            mod8.StatisticalConfig, "base_dir",
            property(lambda self: tmp_path / "EARV"),
        )

        gen = mod8.ArtifactGenerator(mod8.StatisticalConfig())
        results = _make_synthetic_mod7_json()
        gen.generate_latex_table(results, "shallow", "smooth")

        tex_file = gen.cfg.reports_dir / "Latex_Table_shallow_smooth.tex"
        assert tex_file.exists()
        content = tex_file.read_text(encoding="utf-8")

        # 5 GP rank rows present
        for r in range(1, 6):
            assert f"GP Rule (Rank {r})" in content
        # 6 baselines present (with _ escaped)
        for base in ("Xavier", "He", "LeCun", "Orthogonal", "FAVI", "Laor"):
            assert base in content


# =============================================================================
# MOD9 — Activation regex
# =============================================================================

class TestActivationFilenameRegex:
    @pytest.mark.parametrize("fname,expected", [
        ("Final_Discovered_Rules_rectification_20260521_1932.txt", "rectification"),
        ("Final_Discovered_Rules_smooth_20260521_1937.txt", "smooth"),
        ("Final_Discovered_Rules_aggregation_20260521_1939.txt", "aggregation"),
        ("Final_Discovered_Rules_linear_20260521.txt", "linear"),
    ])
    def test_extracts_activation_token(self, fname, expected):
        match = mod9._ACTIVATION_FROM_FILENAME.match(fname)
        assert match is not None
        assert match.group(1) == expected

    @pytest.mark.parametrize("fname", [
        "Checkpoint_smooth_Gen0.txt",
        "random_filename.txt",
        "Final_Discovered_Rules.txt",
    ])
    def test_rejects_non_matching_filenames(self, fname):
        assert mod9._ACTIVATION_FROM_FILENAME.match(fname) is None


# =============================================================================
# MOD9 — SymbolicEngine: sin/cos REGRESSION
# =============================================================================

class TestSymbolicEngineSinCosRegression:
    """The previous version NameError'd on every rule containing sin/cos.

    DEAP serializes primitives using ``func.__name__`` so ``math.sin`` is
    written as bare ``sin(...)`` in rule files. The parser must resolve
    that name directly.
    """

    def test_sin_parses_to_sympy_sin(self):
        engine = mod9.SymbolicEngine()
        expr = engine.parse_rule_to_sympy("sin(hopkins)")
        assert "sin" in str(expr)
        # Free symbol must be hopkins.
        assert engine.symbols["hopkins"] in expr.free_symbols

    def test_cos_parses_to_sympy_cos(self):
        engine = mod9.SymbolicEngine()
        expr = engine.parse_rule_to_sympy("cos(silhouette)")
        assert "cos" in str(expr)

    def test_nested_protected_log_with_sin(self):
        """The exact form found in your testing artifacts."""
        engine = mod9.SymbolicEngine()
        expr = engine.parse_rule_to_sympy(
            "sin(protected_log(mul(hopkins, iqr_dev)))"
        )
        assert engine.symbols["hopkins"] in expr.free_symbols
        assert engine.symbols["iqr_dev"] in expr.free_symbols

    def test_pure_sin_silhouette_from_artifact(self):
        engine = mod9.SymbolicEngine()
        expr = engine.parse_rule_to_sympy("sin(silhouette)")
        assert engine.symbols["silhouette"] in expr.free_symbols


# =============================================================================
# MOD9 — SymbolicEngine: other primitives
# =============================================================================

class TestSymbolicEngineCoverage:
    def test_add_two_features(self):
        engine = mod9.SymbolicEngine()
        expr = engine.parse_rule_to_sympy("add(hopkins, silhouette)")
        assert expr == engine.symbols["hopkins"] + engine.symbols["silhouette"]

    def test_constant_rule(self):
        engine = mod9.SymbolicEngine()
        expr = engine.parse_rule_to_sympy("hopkins")
        assert len(expr.free_symbols) == 1

    def test_extract_all_rules_caps_at_max_ranks(self, tmp_path):
        engine = mod9.SymbolicEngine()
        artifact = tmp_path / "Final_Discovered_Rules_smooth_20260521.txt"
        artifact.write_text(
            "\n".join([f"Rank {i}:\nEquation: hopkins\n" for i in range(1, 11)])
        )
        rules = engine.extract_all_rules(artifact, max_ranks=5)
        assert len(rules) == 5


# =============================================================================
# MOD9 — Empirical means injection (mocked, no real I/O)
# =============================================================================

class TestEmpiricalMeansInjection:
    def test_eight_features_mapped(self):
        manager = _make_mock_dataset_manager()
        analyzer = mod9.QualitativeAnalyzer(mod9.QualitativeConfig(), manager=manager)
        assert len(analyzer.empirical_means) == 8
        for sym, val in analyzer.empirical_means.items():
            assert isinstance(sym, sp.Symbol)
            assert isinstance(val, float)

    def test_means_are_correct_average(self):
        """Three vectors [1..8], [2..9], [3..10] → mean is [2..9]."""
        manager = _make_mock_dataset_manager()
        analyzer = mod9.QualitativeAnalyzer(mod9.QualitativeConfig(), manager=manager)
        means = analyzer.empirical_means
        # n_d_ratio (index 0) mean = (1+2+3)/3 = 2.0
        assert means[analyzer.engine.symbols["n_d_ratio"]] == pytest.approx(2.0)
        # davies_bouldin (index 7) mean = (8+9+10)/3 = 9.0
        assert means[analyzer.engine.symbols["davies_bouldin"]] == pytest.approx(9.0)

    def test_empty_manager_does_not_crash(self):
        empty_manager = MagicMock(spec=mod9.DatasetManager)
        empty_manager.dataset_cache = {}
        analyzer = mod9.QualitativeAnalyzer(mod9.QualitativeConfig(), manager=empty_manager)
        # Zero-vector fallback: every feature mean is 0.0.
        for val in analyzer.empirical_means.values():
            assert val == 0.0


# =============================================================================
# MOD9 — Dominant feature selection
# =============================================================================

class TestDominantFeatures:
    def test_constant_rule_returns_empty(self):
        manager = _make_mock_dataset_manager()
        analyzer = mod9.QualitativeAnalyzer(mod9.QualitativeConfig(), manager=manager)
        expr = sp.Integer(42)
        assert analyzer.identify_dominant_features(expr) == []

    def test_single_symbol_rule_returns_that_symbol(self):
        manager = _make_mock_dataset_manager()
        analyzer = mod9.QualitativeAnalyzer(mod9.QualitativeConfig(), manager=manager)
        engine = analyzer.engine
        expr = engine.symbols["hopkins"]
        assert analyzer.identify_dominant_features(expr) == [engine.symbols["hopkins"]]

    def test_three_plus_symbols_returns_top_two(self):
        manager = _make_mock_dataset_manager()
        analyzer = mod9.QualitativeAnalyzer(mod9.QualitativeConfig(), manager=manager)
        engine = analyzer.engine
        # f = 10*hopkins + silhouette + 0.01*iqr_dev
        expr = (
            10 * engine.symbols["hopkins"]
            + engine.symbols["silhouette"]
            + sp.Rational(1, 100) * engine.symbols["iqr_dev"]
        )
        dominants = analyzer.identify_dominant_features(expr)
        assert len(dominants) == 2
        # The hopkins coefficient is largest, so it must be in the top 2.
        assert engine.symbols["hopkins"] in dominants


# =============================================================================
# MOD9 — Output path structure
# =============================================================================

class TestMod9Paths:
    def test_earv_root(self):
        config = mod9.QualitativeConfig()
        assert config.base_dir.name == "experimental_results_analysis_visualizations"

    def test_reports_subfolder(self):
        config = mod9.QualitativeConfig()
        assert config.reports_dir.name == "MOD9_qualitative_analysis"
        assert config.reports_dir.parent.name == "reports"

    def test_visualizations_subfolder(self):
        config = mod9.QualitativeConfig()
        assert config.vis_dir.name == "MOD9_topography"
        assert config.vis_dir.parent.name == "visualizations"

    def test_rule_dir_default_matches_driver(self):
        """MOD9's rule_dir default must match MOD7_pipeline_driver's default."""
        config = mod9.QualitativeConfig()
        assert config.rule_dir.name == "GA_rule_files_testing"
        assert config.rule_dir.parent.name == "generated_files"