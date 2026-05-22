"""
Unit Tests for Module 2: Pipeline Meta-Extractor

Validates numerical precision and centroid partitioning exclusivity constraints.
"""

import pytest
import numpy as np
import pandas as pd

import MOD2_pipeline_meta_extractor as mod2

def test_config_initialization():
    """Validates structural path mapping based on project root."""
    cfg = mod2.ExtractionConfig(
        project_root=r"C:\test_env",
        output_dir=r"C:\test_env\generated"
    )
    assert cfg.dataset_dir == r"C:\test_env\openml_cc18_datasets"
    assert cfg.log_path == r"C:\test_env\generated\openml_cc18_download_log.csv"
    assert cfg.n_discovery_datasets == 20

def test_hopkins_statistic_uniform():
    """
    Mathematical Boundary: A perfectly uniform distribution should yield
    a Hopkins statistic very close to 0.5.
    """
    np.random.seed(42)
    X_uniform = np.random.uniform(0, 10, (500, 5))
    h_stat = mod2.calculate_hopkins_vectorized(X_uniform, seed=42)

    assert 0.35 <= h_stat <= 0.65, f"Hopkins violated uniform bound: {h_stat}"

def test_hopkins_statistic_clustered():
    """
    Mathematical Boundary: A heavily clustered dataset should yield
    a Hopkins statistic closely approaching 1.0.
    """
    np.random.seed(42)
    cluster_1 = np.random.normal(0, 0.1, (250, 5))
    cluster_2 = np.random.normal(10, 0.1, (250, 5))
    X_clustered = np.vstack([cluster_1, cluster_2])

    h_stat = mod2.calculate_hopkins_vectorized(X_clustered, seed=42)
    assert h_stat > 0.85, f"Hopkins failed to detect extreme clustering: {h_stat}"

def test_extract_meta_features():
    """Ensure exact extraction of the Elite 8 metrics without NaN leakage."""
    cfg = mod2.ExtractionConfig()
    X = pd.DataFrame(np.random.rand(100, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    y = pd.Series(np.random.choice([0, 1], size=100))

    features = mod2.extract_meta_features(X, y, cfg)

    expected_keys = [
        "n_d_ratio", "feat_kurtosis", "iqr_dev", "pc_eigen",
        "target_entropy", "hopkins", "silhouette", "davies_bouldin"
    ]

    for key in expected_keys:
        assert key in features
        assert not np.isnan(features[key])

    assert features["n_d_ratio"] == pytest.approx(100 / 5)

def test_centroid_partitioning():
    """
    Guarantees structural partitioning strictly outputs 20 Phase A datasets
    and mutually exclusively relegates the remainder to Phase B.
    """
    cfg = mod2.ExtractionConfig(n_discovery_datasets=20)

    dummy_data = {
        "did": list(range(1, 51)),
        "name": [f"ds_{i}" for i in range(1, 51)]
    }
    for feature in ["n_d_ratio", "feat_kurtosis", "iqr_dev", "pc_eigen",
                    "target_entropy", "hopkins", "silhouette", "davies_bouldin"]:
        dummy_data[feature] = np.random.rand(50)

    meta_df = pd.DataFrame(dummy_data)
    phase_a, phase_b = mod2.partition_datasets(meta_df, cfg)

    assert len(phase_a) == 20
    assert len(phase_b) == 30

    a_dids = set(phase_a["did"])
    b_dids = set(phase_b["did"])
    assert len(a_dids.intersection(b_dids)) == 0