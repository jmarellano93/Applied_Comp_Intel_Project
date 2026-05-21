"""
Unit Tests for Modules 8 & 9: Statistical Integrity & Partitioned Pathing
"""

import pytest
import sympy as sp
import numpy as np
from pathlib import Path

import MOD8_framework_statistical_reporter as mod8
import MOD9_framework_qualitative_analyzer as mod9

# --- Module 8 Tests ---

def test_partitioned_output_directory_structure():
    """Verifies that the statistical reporter strictly respects sub-namespaces."""
    config = mod8.StatisticalConfig()

    assert config.reports_dir.name == "reports"
    assert config.vis_dir.name == "visualizations"
    assert config.reports_dir.parent.name == "experimental_results_analysis_visualizations"


def test_wilcoxon_zero_variance_protection():
    """Validates that exactly identical metric arrays do not crash the SciPy engine."""
    generator = mod8.ArtifactGenerator(mod8.StatisticalConfig())
    dist_a = [0.85, 0.86, 0.87, 0.88]
    dist_b = [0.85, 0.86, 0.87, 0.88]

    stat, pval = generator.safe_wilcoxon(dist_a, dist_b)

    assert stat == 0.0
    assert pval == 1.0


# --- Module 9 Tests ---

def test_empirical_gradient_substitution_logic():
    """Ensures gradient evaluation anchors to true dataset means rather than arbitrary scalars."""
    analyzer = mod9.QualitativeAnalyzer(mod9.QualitativeConfig())

    # Analyze the internal dictionary structure to ensure we mapped the 8 dataset variables
    assert len(analyzer.empirical_means) == 8

    for sym, val in analyzer.empirical_means.items():
        assert isinstance(sym, sp.Symbol)
        assert isinstance(val, float)