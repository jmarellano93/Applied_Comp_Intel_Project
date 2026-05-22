"""
Unit Tests for Module 5: Prototype Genetic Artifact Routing
"""

import pytest
from pathlib import Path
import MOD5_om_mogp_engine_prototype as mod5

def test_get_generated_directory_nesting_logic():
    """
    Validates that the output directory properly nests the execution trace
    within the explicit partitioned GA_rule_files subdirectory.
    """
    output_path = mod5.get_generated_directory()
    assert isinstance(output_path, Path)

    # Ensure path maps to exactly experiment_modules/generated_files/GA_rule_files
    trailing_paths = output_path.parts[-2:]
    assert trailing_paths == ('generated_files', 'GA_rule_files'), \
        "Pathing engine failed to nest inside GA_rule_files directory."