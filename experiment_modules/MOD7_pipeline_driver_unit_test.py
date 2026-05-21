"""
Unit Tests for Module 7 Driver: Pipeline Matrix Orchestrator
Validates dynamic regex artifact scanning, new explicit subdirectory pathing,
and process token execution.
"""

from pathlib import Path
import pytest
from unittest.mock import patch

import MOD7_pipeline_driver as drv

def test_dynamic_regex_rule_extraction_from_ga_folder(tmp_path):
    """Ensures the driver scrapes equations from the newly nested GA_rule_files directory."""
    ga_dir = tmp_path / "generated_files" / "GA_rule_files"
    ga_dir.mkdir(parents=True)

    artifact = ga_dir / "Final_Discovered_Rules_smooth_20260521.txt"
    artifact.write_text(
        "Rank 1:\nEquation: add(n_d_ratio, 1.0)\nFitness: [1, 2, 3]\n\n"
        "Rank 2:\nEquation: protected_div(pc_eigen, 0.5)\n"
    )

    config = drv.DriverMatrixConfig(module_directory=tmp_path, rule_directory=ga_dir)
    driver = drv.PipelineDriver(config)

    extracted = driver.extract_rules_from_artifact("smooth")

    assert len(extracted) == 2
    assert extracted[0] == "add(n_d_ratio, 1.0)"
    assert extracted[1] == "protected_div(pc_eigen, 0.5)"


@patch("subprocess.run")
def test_missing_artifact_bypass(mock_sub_run, tmp_path):
    """Ensures pipeline skips gracefully if an activation family has no rules."""
    ga_dir = tmp_path / "generated_files" / "GA_rule_files"
    ga_dir.mkdir(parents=True)

    config = drv.DriverMatrixConfig(
        module_directory=tmp_path, rule_directory=ga_dir, activation_targets=["linear"]
    )
    driver = drv.PipelineDriver(config)
    driver.execute_matrix_sweep()

    # Since the GA_rule_files directory is empty, subprocess should never be called
    mock_sub_run.assert_not_called()