"""
Unit Tests for Module 1: OpenML-CC18 Pipeline Selector

Utilizes pytest and unittest.mock to guarantee mathematical constraints
and architectural routing without executing costly network calls.
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

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

import MOD1_pipeline_selector as mod1

def test_pipeline_config_validation():
    """Test boundary constraints and validation within the Pydantic Config."""
    config = mod1.PipelineConfig(max_features=100)
    assert config.max_features == 100

    with pytest.raises(ValidationError):
        mod1.PipelineConfig(max_features=-10)

def test_setup_directories(tmp_path):
    """Test that directories are dynamically created without strict hardcoding."""
    dataset_dir = str(tmp_path / "openml_cc18_datasets")
    generated_dir = str(tmp_path / "generated_files")

    config = mod1.PipelineConfig(dataset_dir=dataset_dir, generated_dir=generated_dir)
    mod1.setup_directories(config)

    assert os.path.exists(dataset_dir)
    assert os.path.exists(generated_dir)

@patch('MOD1_pipeline_selector.openml')
def test_fetch_and_filter_metadata(mock_openml):
    """
    Ensure vectorized filtering accurately applies mathematical bounds
    (instances and feature caps) before download execution.
    """
    mock_suite = MagicMock()
    mock_suite.data = [1, 2, 3, 4]
    mock_openml.study.get_suite.return_value = mock_suite

    mock_metadata = pd.DataFrame({
        'did': [1, 2, 3, 4],
        'name': ['valid', 'too_many_rows', 'too_few_rows', 'too_many_cols'],
        'NumberOfInstances': [1000, 20000, 100, 1000],
        'NumberOfFeatures': [50, 50, 50, 300]
    })
    mock_openml.datasets.list_datasets.return_value = mock_metadata

    config = mod1.PipelineConfig(min_instances=500, max_instances=15000, max_features=200)
    filtered_df = mod1.fetch_and_filter_metadata(config)

    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]['name'] == 'valid'
    assert 'n_d_ratio' in filtered_df.columns
    assert filtered_df.iloc[0]['n_d_ratio'] == 1000 / 50.0

@patch('MOD1_pipeline_selector.openml')
def test_download_and_process_dataset_cleaning(mock_openml, tmp_path):
    """
    Verify that NaNs and zero-variance/constant columns are strictly dropped
    during data processing, ensuring mathematical stability downstream.
    """
    dataset_dir = str(tmp_path / "datasets")
    os.makedirs(dataset_dir)
    config = mod1.PipelineConfig(dataset_dir=dataset_dir)

    mock_dataset = MagicMock()
    mock_dataset.default_target_attribute = 'target_col'

    raw_X = pd.DataFrame({
        'feat1': [1.0, 2.0, 3.0, None],
        'feat2': [5.0, 5.0, 5.0, 5.0],
        'feat3': [10.0, 20.0, 30.0, 40.0]
    })
    raw_y = pd.Series([0, 1, 0, 1])

    mock_dataset.get_data.return_value = (raw_X, raw_y, None, None)
    mock_openml.datasets.get_dataset.return_value = mock_dataset

    log_result = mod1.download_and_process_dataset(did=99, name="test_data", config=config)

    assert log_result['status'] == 'downloaded'

    expected_path = os.path.join(dataset_dir, "99_test_data.csv")
    assert os.path.exists(expected_path)

    processed_df = pd.read_csv(expected_path)
    assert len(processed_df) == 3, "Failed to drop NaN rows"
    assert 'feat2' not in processed_df.columns, "Failed to drop zero-variance column"
    assert 'feat3' in processed_df.columns
    assert 'target_col' in processed_df.columns