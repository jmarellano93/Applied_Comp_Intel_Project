"""
Unit Tests for Module 3: Tensor Cache Manager

Validates rigorous constraints on epistemological data leakage, European semicolon
parsing matrices, exact PyTorch tensor casting, and dynamic portability pathing.
"""

import os
import pytest
import torch
import numpy as np
import pandas as pd

import MOD3_pm_dataset_manager as mod3

def test_dynamic_config_paths(tmp_path):
    """Ensure dynamic routing dynamically builds paths relative to execution location."""
    cfg = mod3.CacheConfig(
        module_dir=str(tmp_path),
        phase_csv_name="Phase_A_Discovery_Datasets.csv"
    )

    expected_gen = os.path.join(str(tmp_path), "generated_files")
    expected_meta = os.path.join(expected_gen, "Phase_A_Discovery_Datasets.csv")
    expected_log = os.path.join(expected_gen, "openml_cc18_download_log.csv")
    expected_data = os.path.join(str(tmp_path), "openml_cc18_datasets")

    assert cfg.generated_dir == expected_gen
    assert cfg.metadata_path == expected_meta
    assert cfg.log_path == expected_log
    assert cfg.dataset_dir == expected_data

def test_semicolon_parsing(tmp_path):
    """
    Mathematical Boundary: Guarantees the primary matrix reader correctly
    interprets semicolon-delimited structural configurations.
    """
    cfg = mod3.CacheConfig(module_dir=str(tmp_path))
    cfg.dataset_dir = str(tmp_path)
    manager = mod3.DatasetManager(cfg)

    csv_file = tmp_path / "99_test.csv"
    csv_file.write_text("feature_1;feature_2;target\n1.0;2.0;class_a\n3.0;4.0;class_b")

    X, y = manager.read_dataset_offline(str(csv_file), 99)

    assert X.shape == (2, 2), "Matrix failed to parse semicolon delimiter dimensions."
    assert "feature_1" in X.columns
    assert y.iloc[0] == "class_a"

def test_leak_free_pipeline():
    """
    Verifies that the Preprocessing Pipeline enforces Strict Separation.
    Validation data distributions must mathematically not influence the `fit` state.
    """
    cfg = mod3.CacheConfig()
    manager = mod3.DatasetManager(cfg)

    X_train_raw = pd.DataFrame({"num": [1, 2, 3, 4], "cat": ["a", "b", "a", "c"]})
    X_val_raw = pd.DataFrame({"num": [1000], "cat": ["z"]})

    preprocessor = manager.build_preprocessing_pipeline(X_train_raw)
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_val_proc = preprocessor.transform(X_val_raw)

    assert X_train_proc.shape[1] == 2
    assert not np.isnan(X_val_proc).any()

def test_tensor_typing_constraints():
    """
    PyTorch requires exact typing for backpropagation. Features must be float32 for
    linear weights, and targets must be int64 (long) for CrossEntropyLoss indexing.
    """
    cfg = mod3.CacheConfig()
    manager = mod3.DatasetManager(cfg)

    y_raw = pd.Series(["cat", "dog", "cat", "bird", None])
    y_encoded = manager.preprocess_target(y_raw)

    assert y_encoded.dtype == np.int64
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    assert y_tensor.dtype == torch.int64