"""
Unit Tests for Module 7: Framework Validation Matrix.

Validates binomial p-value math, sigma-squared sanitization, GP rule
compilation, and the six surviving baseline initialization schemes
(LSUV is not part of the baseline roster; its single-pass implementation
was non-canonical and the remaining six methods span the spectrum).
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

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

import MOD7_framework_validation_matrix as mod7


# =============================================================================
# Binomial p-value math
# =============================================================================

class TestBinomial:
    def test_zero_wins_returns_one(self):
        assert mod7.calculate_binomial_p_value(0, 125) == 1.0

    def test_zero_trials_returns_one(self):
        assert mod7.calculate_binomial_p_value(0, 0) == 1.0

    def test_50pct_win_rate_not_significant(self):
        # 62/125 = 49.6% — well above alpha
        assert mod7.calculate_binomial_p_value(62, 125) > 0.05

    def test_60pct_win_rate_significant(self):
        # 75/125 = 60% — should fall below alpha
        assert mod7.calculate_binomial_p_value(75, 125) < 0.05

    def test_unanimous_wins_near_zero(self):
        # 125/125 wins under H0=0.5 is astronomically improbable.
        assert mod7.calculate_binomial_p_value(125, 125) < 1e-30


# =============================================================================
# Sigma-squared sanitization
# =============================================================================

class TestSanitizeSigmaSquared:
    @pytest.mark.parametrize("val,expected", [
        (0.5, 0.5), (0.0, 0.0), (-0.3, 0.3),
    ])
    def test_finite_inputs(self, val, expected):
        assert mod7.sanitize_sigma_squared(val) == pytest.approx(expected)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_raises(self, bad):
        with pytest.raises(ValueError, match="finite"):
            mod7.sanitize_sigma_squared(bad)


# =============================================================================
# GP rule compilation
# =============================================================================

class TestCompileGpRule:
    def test_simple_addition_compiles_and_evaluates(self):
        func = mod7.compile_gp_rule("add(n_d_ratio, hopkins)")
        result = func(1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
        assert result == pytest.approx(3.0)

    def test_protected_div_floor(self):
        func = mod7.compile_gp_rule("protected_div(n_d_ratio, hopkins)")
        # hopkins=0 should trip the 1e-5 floor and return 1.0
        assert func(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) == 1.0

    def test_sin_compiles(self):
        """Regression: bare sin() in rule files must compile (DEAP emits __name__)."""
        func = mod7.compile_gp_rule("sin(hopkins)")
        result = func(0, 0, 0, 0, 0, math.pi / 2, 0, 0)
        assert result == pytest.approx(1.0)


# =============================================================================
# Baseline initialization schemes
# =============================================================================

@pytest.fixture
def small_model():
    """A 2-layer test model with 10 inputs and 3 outputs."""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3),
    )


@pytest.fixture
def mock_dataset():
    return {"X_train": torch.randn(50, 10)}


@pytest.fixture
def mock_m_vals():
    return np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.7, 0.4, 0.6])


class TestBaselineInitialization:
    @pytest.mark.parametrize("method", list(mod7.BASELINE_METHODS))
    def test_each_baseline_runs_without_error(self, small_model, mock_dataset, mock_m_vals, method):
        """All six remaining baselines must apply without raising."""
        mod7.apply_baseline_initialization(small_model, method, mock_dataset, mock_m_vals)

    @pytest.mark.parametrize("method", list(mod7.BASELINE_METHODS))
    def test_baseline_yields_finite_weights(self, small_model, mock_dataset, mock_m_vals, method):
        mod7.apply_baseline_initialization(small_model, method, mock_dataset, mock_m_vals)
        for layer in small_model.modules():
            if isinstance(layer, nn.Linear):
                assert torch.isfinite(layer.weight).all()
                assert torch.all(layer.bias == 0.0)

    def test_lsuv_no_longer_supported(self, small_model, mock_dataset, mock_m_vals):
        """LSUV was removed — passing it must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown baseline initialization method"):
            mod7.apply_baseline_initialization(small_model, "LSUV", mock_dataset, mock_m_vals)


# =============================================================================
# Output path resolution
# =============================================================================

def test_validation_reports_directory_under_earv():
    """The MOD7 output directory must sit under EARV/reports/MOD7_validation_matrix."""
    reports_dir = mod7.get_validation_reports_directory()
    assert reports_dir.name == "MOD7_validation_matrix"
    assert reports_dir.parent.name == "reports"
    assert reports_dir.parent.parent.name == "experimental_results_analysis_visualizations"


# =============================================================================
# Quick-test constants
# =============================================================================

def test_quick_test_constants_are_collapsed():
    """QUICK_TEST_SEED and QUICK_TEST_DATASET_LIMIT must be small enough for ~1h runs."""
    assert len(mod7.QUICK_TEST_SEED) == 1
    assert mod7.QUICK_TEST_DATASET_LIMIT <= 10
    assert len(mod7.FULL_TRIAL_SEEDS) >= 5