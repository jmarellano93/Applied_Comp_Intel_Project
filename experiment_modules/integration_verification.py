"""
Integration verification mapping the communication interface between the
Pipeline Matrix Orchestrator and the hardware-accelerated Validation Script.
"""
import subprocess
import sys
import os
import pytest

def test_driver_to_mod7_interface_alignment():
    """Validates that the parameter tokens outputted by the Driver exist in Module 7's CLI parser."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mod7_path = os.path.join(current_dir, "MOD7_framework_validation_matrix.py")

    if os.path.exists(mod7_path):
        cmd = [sys.executable, mod7_path, "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        assert "--rule_strs" in result.stdout
        assert "--rule_str " not in result.stdout

def test_partitioned_namespace_generation():
    """Validates that the Driver generates the newly partitioned output namespaces."""
    import MOD7_pipeline_driver as drv

    # Initialize the config which natively resolves the directory tree
    config = drv.DriverMatrixConfig()

    # Ensure the parent directory represents the new architectural node
    assert config.rule_directory.parent.name == "experimental_results_analysis_visualizations"
    assert config.rule_directory.name == "rules"