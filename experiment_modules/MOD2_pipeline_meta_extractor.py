"""
Module 2: Pipeline Meta-Extractor

Extracts intrinsic mathematical meta-features from downloaded datasets and partitions 
them using K-Means spatial representation to guarantee distributional coverage.
"""

import os
import glob
import logging
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.spatial import cKDTree
from scipy.stats import kurtosis, iqr, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - ACI-META - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# FUNCTIONAL BLOCK: Extraction Configuration
# 4A) WHAT IT DOES: Secures the input/output paths and enforces the exact volume
#     of datasets relegated to the Phase A mathematical discovery phase.
# 4B) PARAMETERS:
#     - n_discovery_datasets (20)
#     - random_seed (42)
#     - epsilon (1e-10)
# 4C) METHODOLOGICAL JUSTIFICATION:
#     - 20 Datasets guarantees sufficient topological diversity during GP crossover
#       while preventing out-of-memory constraints during the 80,000 FNN evaluations.
#     - epsilon is a strict mathematical guard against Floating Point exceptions
#       (division by zero) during ratio configurations.
# =============================================================================
class ExtractionConfig(BaseModel):
    project_root: str = Field(default=r"C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project")
    output_dir: str = Field(default=r"C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project\experiment_modules\generated_files")

    dataset_dir: str = ""
    log_path: str = ""

    n_discovery_datasets: int = Field(default=20, description="Exact number of datasets for Phase A.")
    random_seed: int = Field(default=42, description="Universal seed for mathematical determinism.")
    epsilon: float = Field(default=1e-10, description="Numerical stability guardrail.")

    @model_validator(mode='after')
    def build_paths(self) -> 'ExtractionConfig':
        path_root = os.path.join(self.project_root, "openml_cc18_datasets")
        path_module = os.path.join(self.project_root, "experiment_modules", "openml_cc18_datasets")

        if os.path.exists(path_module) and len(glob.glob(os.path.join(path_module, "*.csv"))) > 0:
            self.dataset_dir = path_module
        else:
            self.dataset_dir = path_root

        self.log_path = os.path.join(self.output_dir, "openml_cc18_download_log.csv")
        return self


# =============================================================================
# FUNCTIONAL BLOCK: Hopkins Statistic Calculation
# 4A) WHAT IT DOES: Calculates the dataset's clustering tendency by measuring spatial
#     randomness against a uniformly generated synthetic distribution using KD-Trees.
# 4B) PARAMETERS: sample_ratio (0.1), eps (1e-10)
# 4C) METHODOLOGICAL JUSTIFICATION: Setting sample_ratio to 10% ensures accurate
#     spatial representation of the feature topography without forcing a catastrophic
#     $O(N^2)$ pairwise distance calculation across the entire matrix.
# =============================================================================
def calculate_hopkins_vectorized(X: np.ndarray, seed: int, sample_ratio: float = 0.1, eps: float = 1e-10) -> float:
    n, d = X.shape
    m = max(1, int(n * sample_ratio))

    rng = np.random.default_rng(seed)
    X_min, X_max = X.min(axis=0), X.max(axis=0)

    mask = (X_max == X_min)
    X_max[mask] = X_max[mask] + eps

    sim_points = rng.uniform(X_min, X_max, (m, d))
    real_indices = rng.choice(n, m, replace=False)
    real_points = X[real_indices]

    tree = cKDTree(X)

    u_dist, _ = tree.query(sim_points, k=1)
    w_dist, _ = tree.query(real_points, k=2)
    w_dist = w_dist[:, 1]

    power = min(d, 5)
    u_sum = np.sum(u_dist ** power)
    w_sum = np.sum(w_dist ** power)

    denominator = u_sum + w_sum
    if denominator < eps:
        return 0.5

    return u_sum / denominator


# =============================================================================
# FUNCTIONAL BLOCK: Meta-Feature Extraction Matrix
# 4A) WHAT IT DOES: Extracts the 'Elite 8' meta-features representing dimensionality,
#     distribution shape, variance, and clustering tendencies.
# 4B) PARAMETERS: k_proxy = max(2, min(5, n // 10))
# 4C) METHODOLOGICAL JUSTIFICATION: The k_proxy parameter bounds the KMeans clustering
#     evaluation between 2 and 5 clusters to prevent micro-clustering noise on small
#     datasets, providing a stable baseline for the Silhouette and Davies-Bouldin scores.
# =============================================================================
def extract_meta_features(X: pd.DataFrame, y: pd.Series, cfg: ExtractionConfig) -> Dict[str, float]:
    n, d = X.shape
    X_arr = X.to_numpy()

    n_d_ratio = n / (d + cfg.epsilon)

    feat_kurtosis = float(np.mean(kurtosis(X_arr, axis=0, fisher=True, bias=False, nan_policy='omit')))
    iqr_dev = float(np.std(iqr(X_arr, axis=0, nan_policy='omit')))

    _, counts = np.unique(y.to_numpy(), return_counts=True)
    target_entropy = float(entropy(counts))

    n_comp = min(n, d)
    pca = PCA(n_components=n_comp, random_state=cfg.random_seed)
    try:
        pca.fit(X_arr)
        pc_eigen = float(np.mean(pca.explained_variance_))
    except Exception:
        pc_eigen = 0.0

    hopkins = calculate_hopkins_vectorized(X_arr, seed=cfg.random_seed, eps=cfg.epsilon)

    k_proxy = max(2, min(5, n // 10))
    try:
        km = KMeans(n_clusters=k_proxy, random_state=cfg.random_seed, n_init=1)
        cluster_labels = km.fit_predict(X_arr)
        silhouette = float(silhouette_score(X_arr, cluster_labels))
        davies_bouldin = float(davies_bouldin_score(X_arr, cluster_labels))
    except Exception:
        silhouette = 0.0
        davies_bouldin = 10.0

    return {
        "n_d_ratio": n_d_ratio,
        "feat_kurtosis": np.nan_to_num(feat_kurtosis, nan=0.0),
        "iqr_dev": np.nan_to_num(iqr_dev, nan=0.0),
        "pc_eigen": np.nan_to_num(pc_eigen, nan=0.0),
        "target_entropy": target_entropy,
        "hopkins": hopkins,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin
    }


# =============================================================================
# FUNCTIONAL BLOCK: Dataset Centroid Partitioning
# 4A) WHAT IT DOES: Groups the complete dataset matrix into 20 structural centroids,
#     selecting the closest real dataset to each centroid to form Phase A.
# 4B) PARAMETERS: strategy='median' (Imputer), metric='euclidean' (cdist).
# 4C) METHODOLOGICAL JUSTIFICATION: Using KMeans centroids on scaled meta-features
#     mathematically guarantees that the 20 discovery datasets in Phase A represent
#     the widest possible variety of topological structures (fat, wide, sparse, dense),
#     ensuring the GP rule learns generalized heuristics rather than overfitting to one data type.
# =============================================================================
def partition_datasets(meta_df: pd.DataFrame, cfg: ExtractionConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Clustering {len(meta_df)} datasets into {cfg.n_discovery_datasets} representational centroids.")

    feature_cols = [
        "n_d_ratio", "feat_kurtosis", "iqr_dev", "pc_eigen",
        "target_entropy", "hopkins", "silhouette", "davies_bouldin"
    ]

    X_meta = meta_df[feature_cols].copy()
    imputer = SimpleImputer(strategy='median')
    X_meta_imp = imputer.fit_transform(X_meta)

    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta_imp)

    kmeans = KMeans(n_clusters=cfg.n_discovery_datasets, random_state=cfg.random_seed, n_init=10)
    kmeans.fit(X_meta_scaled)
    centroids = kmeans.cluster_centers_

    from scipy.spatial.distance import cdist
    distances = cdist(centroids, X_meta_scaled, metric='euclidean')

    phase_a_indices = []
    available_indices = set(range(len(meta_df)))

    for i in range(cfg.n_discovery_datasets):
        sorted_indices = np.argsort(distances[i])
        for idx in sorted_indices:
            if idx in available_indices:
                phase_a_indices.append(idx)
                available_indices.remove(idx)
                break

    phase_a_df = meta_df.iloc[phase_a_indices].reset_index(drop=True)
    phase_b_df = meta_df.iloc[list(available_indices)].reset_index(drop=True)

    return phase_a_df, phase_b_df


def execute_meta_pipeline(cfg: ExtractionConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    if not os.path.exists(cfg.log_path):
        raise FileNotFoundError(f"Missing crucial download log file: {cfg.log_path}")

    log_df = pd.read_csv(cfg.log_path)
    target_mapping = dict(zip(log_df['did'], log_df['target']))

    logger.info(f"Searching for downloaded datasets in: {cfg.dataset_dir}")
    csv_files = glob.glob(os.path.join(cfg.dataset_dir, "*.csv"))
    logger.info(f"Detected {len(csv_files)} dataset architectures.")

    if len(csv_files) == 0:
        raise ValueError(f"Total architectural collapse: No CSV files found in {cfg.dataset_dir}")

    extracted_records = []

    for file in csv_files:
        filename = os.path.basename(file)
        try:
            did = int(filename.split('_')[0])
            name = filename.split('_')[1].replace('.csv', '')
        except ValueError:
            logger.warning(f"Bypassing anomalous file structure: {filename}")
            continue

        target_col = target_mapping.get(did, "class")

        try:
            try:
                df = pd.read_csv(file)
            except pd.errors.EmptyDataError:
                logger.error(f"Mathematical extraction bypassed Dataset {did}: File is completely empty.")
                continue

            if pd.isna(target_col) or target_col not in df.columns:
                target_col = df.columns[-1]

            X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])

            if X.shape[1] == 0 or X.shape[0] == 0:
                logger.error(f"Mathematical extraction bypassed Dataset {did}: 0 numerical features detected.")
                continue

            y = df[target_col].astype(str)

            features = extract_meta_features(X, y, cfg)
            features.update({"did": did, "name": name})
            extracted_records.append(features)

        except Exception as e:
            logger.error(f"Mathematical extraction collapsed on Dataset {did}: {str(e)}")
            continue

    if not extracted_records:
        raise ValueError("Total architectural collapse: No features extracted.")

    full_mf_df = pd.DataFrame(extracted_records)

    front_cols = ["did", "name"]
    back_cols = [c for c in full_mf_df.columns if c not in front_cols]
    full_mf_df = full_mf_df[front_cols + back_cols]

    phase_a, phase_b = partition_datasets(full_mf_df, cfg)

    path_a = os.path.join(cfg.output_dir, "Phase_A_Discovery_Datasets.csv")
    path_b = os.path.join(cfg.output_dir, "Phase_B_Validation_Datasets.csv")

    phase_a.to_csv(path_a, index=False)
    phase_b.to_csv(path_b, index=False)

    logger.info("--- PIPELINE METRICS EXPORTED ---")
    logger.info(f"Phase A (Discovery) Volume:  {len(phase_a)} datasets -> Saved to {path_a}")
    logger.info(f"Phase B (Validation) Volume: {len(phase_b)} datasets -> Saved to {path_b}")


if __name__ == "__main__":
    try:
        pipeline_cfg = ExtractionConfig()
        execute_meta_pipeline(pipeline_cfg)
    except Exception as e:
        logger.critical(f"FATAL EXTRACTION ERROR: {e}")