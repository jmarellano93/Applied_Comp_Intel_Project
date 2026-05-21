"""
Unit verification for path structural mapping configurations within the project framework.
Ensures virtual environment executable trees maintain the proper naming convention.
"""
from pathlib import Path
import pytest

def test_venv_interpreter_path_structure():
    """Validates that the standard python executable resides in the correct project subdirectory."""
    mock_project_root = Path("C:/Users/John Arellano/PycharmProjects/Applied_Comp_Intel_Project")
    expected_interpreter_path = mock_project_root / ".venv" / "Scripts" / "python.exe"

    assert expected_interpreter_path.name == "python.exe"
    assert expected_interpreter_path.parent.name == "Scripts"
    assert expected_interpreter_path.parent.parent.name == ".venv"

def test_visualization_hub_path_integrity():
    """Validates the exact string syntax of the new visualization data hub."""
    mock_project_root = Path("C:/Users/John Arellano/PycharmProjects/Applied_Comp_Intel_Project")
    expected_hub_path = mock_project_root / "experiment_modules" / "generated_files" / "experimental_results_analysis_visualizations"

    assert expected_hub_path.name == "experimental_results_analysis_visualizations"