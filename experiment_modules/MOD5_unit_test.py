"""
Unit Tests for Module 5: Genetic Artifact Target Routing
"""

import pytest
from pathlib import Path

# Assuming module filename is MOD6_om_mogp_engine_final.py
import MOD6_om_mogp_engine_final as mod6

def test_get_generated_directory_nesting_logic():
    """
    Validates that the output directory properly nests the execution trace
    within the explicit GA_rule_files subdirectory.
    """
    output_path = mod6.get_generated_directory()

    # Assert that the pathing resolves locally rather than hardcoded C:\
    assert isinstance(output_path, Path)

    # Extract the last two nested directory names
    trailing_paths = output_path.parts[-2:]

    # Ensure they map strictly to the new requirement
    assert trailing_paths == ('generated_files', 'GA_rule_files'), \
        "Pathing engine failed to nest inside GA_rule_files directory."