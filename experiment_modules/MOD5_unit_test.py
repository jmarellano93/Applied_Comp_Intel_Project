"""
Unit Tests for Module 5: MOGP Prototype Engine

Validates Pydantic configuration bounds specifically tailored for the
lightweight prototype scale, as well as system logger integrations.
"""

import os
import pytest
import numpy as np
from pydantic import ValidationError
from pathlib import Path

# Assuming module is saved as MOD5_om_mogp_engine_prototype.py
import MOD5_om_mogp_engine_prototype as mod5

def test_prototype_configuration_scale():
    """Verify that prototype hyperparameters adhere to miniature scale bounds."""
    config = mod5.MOGPConfig()
    assert config.population_size == 20
    assert config.generations == 3
    assert config.datasets_per_rule == 3
    assert config.max_epochs == 5
    assert len(config.activation_functions) == 2

def test_pydantic_probability_validation():
    """Verify mutation and crossover probability bounds."""
    with pytest.raises(ValidationError):
        mod5.MOGPConfig(crossover_probability=0.9, mutation_probability=0.2) # Sum > 1.0

def test_sigma_squared_sanitization():
    """Ensure raw rule variance output is finite and strictly positive."""
    assert mod5.sanitize_sigma_squared(1.5) == 1.5
    assert mod5.sanitize_sigma_squared(-2.0) == 2.0  # Absolute fallback

    with pytest.raises(ValueError):
        mod5.sanitize_sigma_squared(np.inf)
    with pytest.raises(ValueError):
        mod5.sanitize_sigma_squared(np.nan)

def test_environment_logger_execution():
    """Verify that hardware/software context captures successfully."""
    config = mod5.MOGPConfig()
    env_str = mod5.get_system_environment(config)

    assert "Python Version:" in env_str
    assert "PyTorch Version:" in env_str
    assert "Random Seed: 42" in env_str

def test_dynamic_generated_directory(tmp_path):
    """Verify output mapping strictly relies on script execution environment."""
    import sys

    original_file = mod5.__file__
    try:
        mod5.__file__ = str(tmp_path / "dummy.py")
        expected_dir = tmp_path / "generated_files"

        resolved_dir = mod5.get_generated_directory()
        assert resolved_dir == expected_dir
        assert expected_dir.exists()
    finally:
        mod5.__file__ = original_file