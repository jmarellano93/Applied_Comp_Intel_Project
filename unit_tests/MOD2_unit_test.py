"""
Unit Tests for Module 2: Pipeline Meta-Extractor

Validates numerical precision and centroid partitioning exclusivity constraints.
"""

import pytest
import numpy as np
import pandas as pd

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


def test_persist_normalization_params_fits_on_phase_a_only(tmp_path):
    """
    Guarantee: normalization parameters MUST be fit on Phase A only,
    not on the entire pool. Verifies (mean, std) match phase_a_df statistics,
    NOT statistics computed from a hypothetical full set, AND that the
    output CSV is structurally well-formed.
    """
    cfg = mod2.ExtractionConfig(output_dir=str(tmp_path))

    feature_cols = [
        "n_d_ratio", "feat_kurtosis", "iqr_dev", "pc_eigen",
        "target_entropy", "hopkins", "silhouette", "davies_bouldin"
    ]
    rng = np.random.default_rng(42)
    # Phase A has 20 rows with controlled statistics.
    phase_a_data = {col: rng.normal(loc=10.0, scale=2.0, size=20) for col in feature_cols}
    phase_a_data["did"] = list(range(1, 21))
    phase_a_data["name"] = [f"ds_{i}" for i in range(1, 21)]
    phase_a_df = pd.DataFrame(phase_a_data)

    mod2.persist_normalization_params(phase_a_df, cfg)

    norm_path = tmp_path / "meta_feature_normalization_params.csv"
    assert norm_path.exists(), "Normalization params CSV was not written."

    norm_df = pd.read_csv(norm_path)
    assert set(norm_df["feature"]) == set(feature_cols)
    assert norm_df.shape == (8, 3)

    # Spot-check: each (mean, std) matches Phase A pandas-computed values.
    for col in feature_cols:
        expected_mean = float(phase_a_df[col].mean())
        expected_std = float(phase_a_df[col].std(ddof=0))
        row = norm_df.loc[norm_df["feature"] == col].iloc[0]
        assert row["mean"] == pytest.approx(expected_mean, rel=1e-6)
        assert row["std"] == pytest.approx(expected_std, rel=1e-6)


def test_persist_normalization_params_floors_zero_std(tmp_path):
    """A degenerate constant Phase-A column gets floored to epsilon, not 0,
    so the downstream z-score transform never divides by zero.
    """
    cfg = mod2.ExtractionConfig(output_dir=str(tmp_path), epsilon=1e-10)

    feature_cols = [
        "n_d_ratio", "feat_kurtosis", "iqr_dev", "pc_eigen",
        "target_entropy", "hopkins", "silhouette", "davies_bouldin"
    ]
    rng = np.random.default_rng(0)
    phase_a_data = {col: rng.normal(size=20) for col in feature_cols}
    # Force one column (silhouette) to be constant to trigger the std floor.
    phase_a_data["silhouette"] = np.zeros(20)
    phase_a_data["did"] = list(range(1, 21))
    phase_a_data["name"] = [f"ds_{i}" for i in range(1, 21)]
    phase_a_df = pd.DataFrame(phase_a_data)

    mod2.persist_normalization_params(phase_a_df, cfg)

    norm_df = pd.read_csv(tmp_path / "meta_feature_normalization_params.csv")
    silh_std = float(norm_df.loc[norm_df["feature"] == "silhouette", "std"].iloc[0])
    assert silh_std >= cfg.epsilon, "Zero-std floor was not enforced."