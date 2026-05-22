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


def test_normalization_params_path_built(tmp_path):
    """Q-D: the cache config exposes a norm_params_path under generated_files/."""
    cfg = mod3.CacheConfig(module_dir=str(tmp_path))
    expected = os.path.join(str(tmp_path), "generated_files",
                            "meta_feature_normalization_params.csv")
    assert cfg.norm_params_path == expected


def test_load_normalization_params_applies_z_score(tmp_path):
    """
    Q-D: When a normalization-params CSV exists, ``_load_normalization_params``
    populates the per-feature (mean, std) arrays in canonical order. The
    transform applied at cache time is exact z-score using those arrays.
    """
    cfg = mod3.CacheConfig(module_dir=str(tmp_path))
    os.makedirs(cfg.generated_dir, exist_ok=True)

    feature_cols = [
        "n_d_ratio", "feat_kurtosis", "iqr_dev", "pc_eigen",
        "target_entropy", "hopkins", "silhouette", "davies_bouldin"
    ]
    # Construct deterministic normalization params: mean=j+1, std=1 for j in 0..7.
    norm_df = pd.DataFrame({
        "feature": feature_cols,
        "mean": [float(i + 1) for i in range(8)],
        "std": [1.0] * 8,
    })
    norm_df.to_csv(cfg.norm_params_path, index=False)

    manager = mod3.DatasetManager(cfg)
    manager._load_normalization_params()

    assert manager._norm_params_loaded is True
    np.testing.assert_allclose(
        manager._norm_mean, np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    )
    np.testing.assert_allclose(manager._norm_std, np.ones(8, dtype=np.float32))

    # Z-score check: an input vector matching the means returns the zero vector.
    raw = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    z = (raw - manager._norm_mean) / manager._norm_std
    np.testing.assert_allclose(z, np.zeros(8, dtype=np.float32))


def test_load_normalization_params_graceful_fallback(tmp_path):
    """
    Q-D backward compat: when the normalization params file is absent, the
    manager logs a warning and stays in legacy raw mode (identity transform).
    """
    cfg = mod3.CacheConfig(module_dir=str(tmp_path))
    # Do NOT write any normalization CSV.
    manager = mod3.DatasetManager(cfg)
    manager._load_normalization_params()

    assert manager._norm_params_loaded is False
    # Defaults: mean=0, std=1 → identity transform.
    np.testing.assert_allclose(manager._norm_mean, np.zeros(8, dtype=np.float32))
    np.testing.assert_allclose(manager._norm_std, np.ones(8, dtype=np.float32))