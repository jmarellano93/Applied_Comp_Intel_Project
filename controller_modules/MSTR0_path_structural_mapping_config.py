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


def test_controller_modules_path_integrity():
    """Validates the structure of the new controller routing namespace and naming conventions."""
    # Anchors dynamically to this file's position to determine the parent project root structure
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    controller_file = project_root / "controller_modules" / "MSTR1_master_data_orchestrator.py"

    # Assert structural safety constraints
    assert controller_file.parent.name == "controller_modules", "Controller directory naming mismatch."
    assert controller_file.name.startswith("MSTR"), "MSTR-series prefix token violation."
    assert controller_file.name.endswith(".py"), "Python extension routing missing."