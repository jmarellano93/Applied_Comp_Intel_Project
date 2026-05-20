"""
Unit Tests for Module 6: MOGP Engine

Validates mathematical operator protections, Pydantic configuration schemas,
environmental logging fidelity, and strict path generation.
"""

import os
import pytest
import numpy as np
from pydantic import ValidationError
from pathlib import Path

# Assuming module is saved as MOD6_om_mogp_engine_final.py
import MOD6_om_mogp_engine_final as mod6

def test_pydantic_configuration_bounds():
    """Verify that runtime hyperparameters strictly respect EA boundaries."""
    # Valid config
    config = mod6.MOGPConfig()
    assert config.population_size == 200

    # Boundary failure: Mut + Cx > 1.0
    with pytest.raises(ValidationError):
        mod6.MOGPConfig(crossover_probability=0.8, mutation_probability=0.5)

    # Boundary failure: Negative populations
    with pytest.raises(ValidationError):
        mod6.MOGPConfig(population_size=-10)

def test_protected_math_operators():
    """Ensure mathematical primitives do not crash under infinite/NaN domains."""
    # Protected Division
    assert mod6.protected_div(10.0, 0.0) == 1.0
    assert mod6.protected_div(10.0, 2.0) == 5.0

    # Protected Sqrt (Abs value handling)
    assert mod6.protected_sqrt(-9.0) == 3.0

    # Protected Log
    assert mod6.protected_log(0.0) == 0.0
    assert mod6.protected_log(-np.e) == pytest.approx(1.0)

    # Protected Exp (Clipping)
    assert mod6.protected_exp(100.0) == pytest.approx(np.exp(10.0))
    assert mod6.protected_exp(-100.0) == pytest.approx(np.exp(-10.0))

def test_sigma_squared_sanitization():
    """Ensure raw rule variance output is finite and positive."""
    # Standard positive handling
    assert mod6.sanitize_sigma_squared(0.05) == 0.05
    # Negative handling (Absolute mapping)
    assert mod6.sanitize_sigma_squared(-0.05) == 0.05

    # Infinite/NaN prevention
    with pytest.raises(ValueError):
        mod6.sanitize_sigma_squared(np.inf)
    with pytest.raises(ValueError):
        mod6.sanitize_sigma_squared(np.nan)

def test_environment_logger():
    """Verify that the hardware/software snapshot is captured as a string."""
    config = mod6.MOGPConfig()
    env_str = mod6.get_system_environment(config)

    assert "Python Version:" in env_str
    assert "PyTorch Version:" in env_str
    assert "Random Seed: 42" in env_str
    assert "Git Commit:" in env_str

def test_dynamic_generated_directory(tmp_path):
    """Verify output mapping strictly relies on script execution environment."""
    import sys

    original_file = mod6.__file__
    try:
        mod6.__file__ = str(tmp_path / "dummy.py")
        expected_dir = tmp_path / "generated_files"

        resolved_dir = mod6.get_generated_directory()
        assert resolved_dir == expected_dir
        assert expected_dir.exists()
    finally:
        mod6.__file__ = original_file