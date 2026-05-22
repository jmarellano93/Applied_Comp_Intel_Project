"""
Module 3: Dataset Ingestion & Tensor Cache Manager

Handles the offline ingestion, preprocessing (leak-free), and RAM-caching of datasets.
Features dynamic environment pathing to ensure portability across operating systems
and execution environments.
"""

import os
import glob
import logging
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field, model_validator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - ACI-CACHE - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# FUNCTIONAL BLOCK: Cache Configuration
# 4A) WHAT IT DOES: Maps paths and sets fundamental topological bounds for testing.
# 4B) PARAMETERS: test_size (0.2), random_seed (42).
# 4C) METHODOLOGICAL JUSTIFICATION:
#     - test_size=0.2 enforces the standard 80/20 Pareto principle for Train/Test
#       splits. This ratio guarantees the network has enough dense data to converge,
#       while providing a sufficiently large validation set to prove generalization.
# =============================================================================
class CacheConfig(BaseModel):
    module_dir: str = Field(default=MODULE_DIR, description="Dynamically resolved directory of the current module.")
    phase_csv_name: str = Field(default="Phase_A_Discovery_Datasets.csv", description="Target dataset partition list to load.")

    generated_dir: str = ""
    dataset_dir: str = ""
    metadata_path: str = ""
    log_path: str = ""
    norm_params_path: str = ""

    test_size: float = Field(default=0.2, description="Validation split ratio (80/20).")
    random_seed: int = Field(default=42, description="Seed for deterministic stratification and splitting.")

    @model_validator(mode='after')
    def build_paths(self) -> 'CacheConfig':
        self.generated_dir = os.path.join(self.module_dir, "generated_files")
        self.dataset_dir = os.path.join(self.module_dir, "openml_cc18_datasets")
        self.metadata_path = os.path.join(self.generated_dir, self.phase_csv_name)
        self.log_path = os.path.join(self.generated_dir, "openml_cc18_download_log.csv")
        # Normalization params are produced by MOD2 from Phase A only and consumed here.
        self.norm_params_path = os.path.join(
            self.generated_dir, "meta_feature_normalization_params.csv"
        )
        return self


class DatasetManager:
    def __init__(self, config: CacheConfig):
        self.cfg = config
        self.dataset_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.meta_features_cache: Dict[int, torch.Tensor] = {}
        self._target_mapping = self._build_target_mapping()
        # Z-score normalizer parameters (mean, std per feature). Populated from
        # disk by _load_normalization_params if MOD2 has produced them.
        self._norm_mean: np.ndarray = np.zeros(8, dtype=np.float32)
        self._norm_std: np.ndarray = np.ones(8, dtype=np.float32)
        self._norm_params_loaded: bool = False

    def _build_target_mapping(self) -> Dict[int, str]:
        if not os.path.exists(self.cfg.log_path):
            logger.warning(f"Download log missing at {self.cfg.log_path}. Relying on positional fallback.")
            return {}

        log_df = pd.read_csv(self.cfg.log_path)
        return dict(zip(log_df['did'], log_df['target']))

    # =============================================================================
    # FUNCTIONAL BLOCK: Normalization Parameter Loading
    # 4A) WHAT IT DOES: Reads MOD2-produced normalization params (per-feature
    #     mean, std fit on Phase A only) and stores them as instance arrays.
    # 4B) PARAMETERS: None (uses self.cfg.norm_params_path).
    # 4C) METHODOLOGICAL JUSTIFICATION: Applying identical (mean, std) to both
    #     Phase A and Phase B at cache load time guarantees the GP terminal-set
    #     values seen during discovery match those seen during validation
    #     under the same affine transform, with fit parameters derived from
    #     Phase A only (no distributional leakage). When the params file is
    #     absent the manager falls back to raw values with a warning, preserving
    #     compatibility with legacy rule artifacts produced before normalization
    #     was introduced.
    # =============================================================================
    def _load_normalization_params(self) -> None:
        if not os.path.exists(self.cfg.norm_params_path):
            logger.warning(
                f"Normalization params not found at {self.cfg.norm_params_path}. "
                "Falling back to RAW meta-features (legacy mode). "
                "Run MOD2 to regenerate normalization parameters."
            )
            return

        try:
            norm_df = pd.read_csv(self.cfg.norm_params_path)
            feature_order = [
                "n_d_ratio", "feat_kurtosis", "iqr_dev", "pc_eigen",
                "target_entropy", "hopkins", "silhouette", "davies_bouldin"
            ]
            # Sort by canonical feature order to guarantee alignment.
            norm_df = norm_df.set_index("feature").loc[feature_order].reset_index()
            self._norm_mean = norm_df["mean"].values.astype(np.float32)
            self._norm_std = norm_df["std"].values.astype(np.float32)
            self._norm_params_loaded = True
            logger.info(
                f"Loaded z-score normalization params from {self.cfg.norm_params_path}."
            )
        except Exception as e:
            logger.error(
                f"Failed to load normalization params ({e}); reverting to raw values."
            )
            self._norm_params_loaded = False

    def find_local_dataset_file(self, did: int) -> str:
        pattern = os.path.join(self.cfg.dataset_dir, f"{did}_*.csv")
        matches = glob.glob(pattern)

        if not matches:
            pattern_fallback = os.path.join(self.cfg.dataset_dir, f"{did}*.csv")
            matches = glob.glob(pattern_fallback)

        if not matches:
            raise FileNotFoundError(f"Local CSV for Dataset {did} missing from {self.cfg.dataset_dir}")

        return matches[0]

    def read_dataset_offline(self, file_path: str, did: int) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            df = pd.read_csv(file_path, sep=";")
            if df.shape[1] == 1:
                df = pd.read_csv(file_path, sep=",")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, sep=";", encoding="latin1")
            if df.shape[1] == 1:
                df = pd.read_csv(file_path, sep=",", encoding="latin1")

        target_col = self._target_mapping.get(did)

        if target_col and target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            target_col = df.columns[-1]
            y = df[target_col]
            X = df.drop(columns=[target_col])

        return X, y

    # =============================================================================
    # FUNCTIONAL BLOCK: Preprocessing Pipeline Construction
    # 4A) WHAT IT DOES: Imputes missing values, encodes categories, and scales features.
    # 4B) PARAMETERS: strategy="median", strategy="most_frequent", handle_unknown="use_encoded_value".
    # 4C) METHODOLOGICAL JUSTIFICATION:
    #     - Median imputation is used rather than Mean because it is statistically robust
    #       to extreme numerical outliers.
    #     - StandardScaler mathematically restricts inputs to a Gaussian distribution (mean=0, std=1).
    #       This is absolutely mandatory for Neural Networks to prevent catastrophic gradient
    #       explosion and to ensure stable initial weight distributions.
    # =============================================================================
    def build_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        cat_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
        num_cols = X.select_dtypes(exclude=["object", "string", "category", "bool"]).columns.tolist()

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols)
            ],
            remainder="drop"
        )

        return preprocessor

    def preprocess_target(self, y: pd.Series) -> np.ndarray:
        y = y.replace(["?", "NA", "N/A", "nan", "NaN", "None", ""], np.nan)
        y = y.fillna("MISSING_CLASS").astype(str)
        return LabelEncoder().fit_transform(y).astype(np.int64)

    def load_all_to_ram(self) -> None:
        if not os.path.exists(self.cfg.metadata_path):
            raise FileNotFoundError(f"Partition file missing: {self.cfg.metadata_path}")

        # Load normalization params produced by MOD2 (Phase-A-only fit).
        self._load_normalization_params()

        metadata_df = pd.read_csv(self.cfg.metadata_path)
        logger.info(f"Initiating RAM cache sequence for {len(metadata_df)} datasets...")

        for _, row in metadata_df.iterrows():
            did = int(row["did"])

            try:
                file_path = self.find_local_dataset_file(did)
                X_raw, y_raw = self.read_dataset_offline(file_path, did)

                X_raw = X_raw.replace(["?", "NA", "N/A", "nan", "NaN", "None", ""], np.nan)
                y = self.preprocess_target(y_raw)

                stratify_array = y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) > 1 else None

                X_train_raw, X_val_raw, y_train, y_val = train_test_split(
                    X_raw, y,
                    test_size=self.cfg.test_size,
                    random_state=self.cfg.random_seed,
                    stratify=stratify_array
                )

                preprocessor = self.build_preprocessing_pipeline(X_train_raw)
                X_train = preprocessor.fit_transform(X_train_raw)
                X_val = preprocessor.transform(X_val_raw)

                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                self.dataset_cache[did] = {
                    "X_train": torch.tensor(X_train, dtype=torch.float32),
                    "y_train": torch.tensor(y_train, dtype=torch.long),
                    "X_val": torch.tensor(X_val, dtype=torch.float32),
                    "y_val": torch.tensor(y_val, dtype=torch.long)
                }

                # Build raw 8-vector in canonical feature order, then apply
                # Phase-A z-score transform if normalization params are loaded.
                raw_vec = np.array([
                    row["n_d_ratio"], row["feat_kurtosis"], row["iqr_dev"],
                    row["pc_eigen"], row["target_entropy"], row["hopkins"],
                    row["silhouette"], row["davies_bouldin"]
                ], dtype=np.float32)

                if self._norm_params_loaded:
                    normalized_vec = (raw_vec - self._norm_mean) / self._norm_std
                else:
                    normalized_vec = raw_vec

                self.meta_features_cache[did] = torch.tensor(normalized_vec, dtype=torch.float32)

            except Exception as e:
                logger.error(f"Ingestion collapsed on Dataset {did}: {e}")

        logger.info(f"RAM Cache complete. Successfully pinned {len(self.dataset_cache)} PyTorch tensor suites.")

    def get_dataset(self, did: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if did not in self.dataset_cache:
            raise KeyError(f"Dataset {did} absent from tensor cache. Execute load_all_to_ram() first.")
        return self.dataset_cache[did], self.meta_features_cache[did]


if __name__ == "__main__":
    try:
        config = CacheConfig()
        manager = DatasetManager(config)
        manager.load_all_to_ram()
    except Exception as e:
        logger.critical(f"FATAL CACHE ERROR: {e}")