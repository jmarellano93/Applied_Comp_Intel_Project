"""
Unit Tests for Module 6: Production Genetic Artifact Target Routing
"""

import pytest
from pathlib import Path
import MOD6_om_mogp_engine_final as mod6

def test_get_generated_directory_nesting_logic():
    """
    Validates that the output directory properly nests the execution trace
    within the explicit partitioned GA_rule_files subdirectory.
    """
    output_path = mod6.get_generated_directory()
    assert isinstance(output_path, Path)

    # Ensure path maps to exactly experiment_modules/generated_files/GA_rule_files
    trailing_paths = output_path.parts[-2:]
    assert trailing_paths == ('generated_files', 'GA_rule_files'), \
        "Pathing engine failed to nest inside GA_rule_files directory."

def test_production_hyperparameter_boundaries():
    """Ensures Pydantic schema validation defaults protect methodological integrity."""
    config = mod6.MOGPConfig()

    # Verify strict production-level variables are enforced
    assert config.population_size == 150
    assert config.generations == 20
    assert config.datasets_per_rule == 20