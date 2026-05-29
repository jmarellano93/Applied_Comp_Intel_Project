"""
Unit Tests for Module 7 Driver: Pipeline Matrix Orchestrator.

Validates regex artifact scanning, default rule_directory wiring to
``GA_rule_files_testing``, ``--quick_test`` propagation, and subprocess
command construction.
"""

from pathlib import Path

import pytest
from unittest.mock import patch

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

import MOD7_pipeline_driver as drv


# =============================================================================
# Defaults
# =============================================================================

def test_default_rule_directory_points_to_production_folder():
    """The default rule directory must be GA_rule_files (production) under generated_files."""
    config = drv.DriverMatrixConfig()
    assert config.rule_directory.name == "GA_rule_files"
    assert config.rule_directory.parent.name == "generated_files"


def test_quick_test_defaults_false():
    """quick_test must default to False so production runs aren't accidentally collapsed."""
    config = drv.DriverMatrixConfig()
    assert config.quick_test is False


def test_topology_default_shallow():
    """Backward-compat: the scalar `topology` field still defaults to shallow."""
    config = drv.DriverMatrixConfig()
    assert config.topology == "shallow"


def test_topology_targets_full_sweep_default():
    """topology_targets defaults to the full 3-topology sweep."""
    config = drv.DriverMatrixConfig()
    assert config.topology_targets == ["shallow", "deep_narrow", "funnel"]


def test_topology_targets_override():
    """Subset of topologies is honoured when explicitly provided."""
    config = drv.DriverMatrixConfig(topology_targets=["shallow"])
    assert config.topology_targets == ["shallow"]


# =============================================================================
# Rule extraction
# =============================================================================

def test_dynamic_regex_rule_extraction_from_testing_folder(tmp_path):
    """Driver scrapes equations from the GA_rule_files_testing directory.

    The filename embeds (activation, topology); the extractor
    requires both for unambiguous file selection.
    """
    ga_dir = tmp_path / "generated_files" / "GA_rule_files_testing"
    ga_dir.mkdir(parents=True)

    artifact = ga_dir / "Final_Discovered_Rules_smooth_shallow_20260521.txt"
    artifact.write_text(
        "Rank 1:\nEquation: add(n_d_ratio, 1.0)\nFitness: [1, 2, 3]\n\n"
        "Rank 2:\nEquation: protected_div(pc_eigen, 0.5)\n"
    )

    config = drv.DriverMatrixConfig(module_directory=tmp_path, rule_directory=ga_dir)
    driver = drv.PipelineDriver(config)
    extracted = driver.extract_rules_from_artifact("smooth", topology="shallow")

    assert len(extracted) == 2
    assert extracted[0] == "add(n_d_ratio, 1.0)"
    assert extracted[1] == "protected_div(pc_eigen, 0.5)"


def test_extraction_caps_at_five_rules(tmp_path):
    """Extraction must return at most 5 equations even when more are present."""
    ga_dir = tmp_path / "generated_files" / "GA_rule_files_testing"
    ga_dir.mkdir(parents=True)
    artifact = ga_dir / "Final_Discovered_Rules_linear_shallow_20260521.txt"
    artifact.write_text(
        "\n".join([f"Rank {i}:\nEquation: hopkins\n" for i in range(1, 11)])
    )

    config = drv.DriverMatrixConfig(module_directory=tmp_path, rule_directory=ga_dir)
    driver = drv.PipelineDriver(config)
    assert len(driver.extract_rules_from_artifact("linear", topology="shallow")) == 5


def test_extraction_disambiguates_by_topology(tmp_path):
    """The (activation, topology) pair selects exactly the right artifact.

    A directory containing the SAME activation under TWO topologies must
    return only the file matching the requested topology.
    """
    ga_dir = tmp_path / "generated_files" / "GA_rule_files_testing"
    ga_dir.mkdir(parents=True)

    art_shallow = ga_dir / "Final_Discovered_Rules_rectification_shallow_20260521.txt"
    art_funnel = ga_dir / "Final_Discovered_Rules_rectification_funnel_20260521.txt"
    art_shallow.write_text("Rank 1:\nEquation: rule_shallow_only\n")
    art_funnel.write_text("Rank 1:\nEquation: rule_funnel_only\n")

    config = drv.DriverMatrixConfig(module_directory=tmp_path, rule_directory=ga_dir)
    driver = drv.PipelineDriver(config)

    rules_shallow = driver.extract_rules_from_artifact("rectification", topology="shallow")
    rules_funnel = driver.extract_rules_from_artifact("rectification", topology="funnel")

    assert rules_shallow == ["rule_shallow_only"]
    assert rules_funnel == ["rule_funnel_only"]


def test_latest_timestamp_wins(tmp_path):
    """When multiple artifacts exist for the same (activation, topology), the
    most-recently modified one is read.
    """
    ga_dir = tmp_path / "generated_files" / "GA_rule_files_testing"
    ga_dir.mkdir(parents=True)

    older = ga_dir / "Final_Discovered_Rules_smooth_shallow_20240101.txt"
    older.write_text("Rank 1:\nEquation: OLD_RULE\n")

    newer = ga_dir / "Final_Discovered_Rules_smooth_shallow_20260101.txt"
    newer.write_text("Rank 1:\nEquation: NEW_RULE\n")

    # Force newer's mtime higher than older's.
    import os
    old_time = older.stat().st_mtime
    os.utime(newer, (old_time + 100, old_time + 100))

    config = drv.DriverMatrixConfig(module_directory=tmp_path, rule_directory=ga_dir)
    driver = drv.PipelineDriver(config)
    assert driver.extract_rules_from_artifact("smooth", topology="shallow")[0] == "NEW_RULE"


# =============================================================================
# Subprocess command construction
# =============================================================================

@patch("subprocess.run")
def test_missing_artifact_bypass(mock_sub_run, tmp_path):
    """Pipeline skips (topology, activation) pairs whose artifact is absent."""
    ga_dir = tmp_path / "generated_files" / "GA_rule_files_testing"
    ga_dir.mkdir(parents=True)

    config = drv.DriverMatrixConfig(
        module_directory=tmp_path, rule_directory=ga_dir,
        activation_targets=["linear"], topology_targets=["shallow"],
    )
    driver = drv.PipelineDriver(config)
    driver.execute_matrix_sweep()

    mock_sub_run.assert_not_called()


@patch("subprocess.run")
def test_quick_test_flag_propagated_to_mod7(mock_sub_run, tmp_path):
    """When quick_test=True, the subprocess command must include --quick_test."""
    ga_dir = tmp_path / "generated_files" / "GA_rule_files_testing"
    ga_dir.mkdir(parents=True)
    artifact = ga_dir / "Final_Discovered_Rules_linear_shallow_20260521.txt"
    artifact.write_text("Rank 1:\nEquation: hopkins\n")

    # Fake subprocess.run returning a success result.
    class _Result:
        returncode = 0
    mock_sub_run.return_value = _Result()

    config = drv.DriverMatrixConfig(
        module_directory=tmp_path,
        rule_directory=ga_dir,
        activation_targets=["linear"],
        topology_targets=["shallow"],
        quick_test=True,
    )
    driver = drv.PipelineDriver(config)
    driver.execute_matrix_sweep()

    assert mock_sub_run.called
    cmd = mock_sub_run.call_args[0][0]
    assert "--quick_test" in cmd
    assert "--rule_strs" in cmd
    # --quick_test must come before --rule_strs so it isn't swallowed by nargs='+'.
    assert cmd.index("--quick_test") < cmd.index("--rule_strs")


@patch("subprocess.run")
def test_quick_test_disabled_does_not_propagate(mock_sub_run, tmp_path):
    """When quick_test=False, the flag must NOT be in the subprocess command."""
    ga_dir = tmp_path / "generated_files" / "GA_rule_files_testing"
    ga_dir.mkdir(parents=True)
    artifact = ga_dir / "Final_Discovered_Rules_linear_shallow_20260521.txt"
    artifact.write_text("Rank 1:\nEquation: hopkins\n")

    class _Result:
        returncode = 0
    mock_sub_run.return_value = _Result()

    config = drv.DriverMatrixConfig(
        module_directory=tmp_path,
        rule_directory=ga_dir,
        activation_targets=["linear"],
        topology_targets=["shallow"],
        quick_test=False,
    )
    driver = drv.PipelineDriver(config)
    driver.execute_matrix_sweep()

    cmd = mock_sub_run.call_args[0][0]
    assert "--quick_test" not in cmd