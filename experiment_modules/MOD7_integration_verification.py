"""
Driver↔MOD7 integration verification.

Validates the CLI contract between MOD7_pipeline_driver and
MOD7_framework_validation_matrix, plus the default rule-directory wiring.

This file is a pytest test file. Run via:
    python -m pytest integration_verification.py -v
"""

import os
import subprocess
import sys

import pytest


def test_driver_to_mod7_interface_alignment() -> None:
    """Validates that the parameter tokens emitted by the Driver exist in MOD7's CLI parser."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mod7_path = os.path.join(current_dir, "MOD7_framework_validation_matrix.py")

    if not os.path.exists(mod7_path):
        pytest.skip(f"MOD7 not present at {mod7_path}")

    cmd = [sys.executable, mod7_path, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # The driver emits exactly --rule_strs (plural) — guard against accidental
    # regression to the singular form.
    assert "--rule_strs" in result.stdout
    assert "--rule_str " not in result.stdout

    # The quick-test flag must be surfaced for MSTR3 → driver → MOD7 propagation.
    assert "--quick_test" in result.stdout


def test_partitioned_namespace_generation() -> None:
    """Validates that the Driver's default rule_directory points at GA_rule_files_testing.

    The driver's default reflects the *current* testing phase. When you
    promote real production rules, change DriverMatrixConfig.rule_directory's
    default factory to point at GA_rule_files/ and update this assertion.
    """
    import MOD7_pipeline_driver as drv

    config = drv.DriverMatrixConfig()

    # The rule directory's parent must be generated_files/ (rules are data, not analysis artifacts).
    assert config.rule_directory.parent.name == "generated_files"
    # The default is the testing-mode directory.
    assert config.rule_directory.name == "GA_rule_files_testing"


def test_driver_quick_test_flag_surface() -> None:
    """Asserts the driver itself accepts --quick_test for MSTR3 forwarding."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    drv_path = os.path.join(current_dir, "MOD7_pipeline_driver.py")

    if not os.path.exists(drv_path):
        pytest.skip(f"Driver not present at {drv_path}")

    cmd = [sys.executable, drv_path, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "--quick_test" in result.stdout
    assert "--rule_directory" in result.stdout