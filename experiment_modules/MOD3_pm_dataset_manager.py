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

# Configure scientific logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - ACI-CACHE - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dynamically resolve the directory containing this script (experiment_modules)
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


class CacheConfig(BaseModel):
    """
    Configuration schema for RAM ingestion paths and split parameters.
    Utilizes dynamic relative pathing for maximum portability.
    """
    module_dir: str = Field(
        default=MODULE_DIR,
        description="Dynamically resolved directory of the current module."
    )
    phase_csv_name: str = Field(
        default="Phase_A_Discovery_Datasets.csv",
        description="Target dataset partition list to load."
    )

    # Internal paths resolved post-validation
    generated_dir: str = ""
    dataset_dir: str = ""
    metadata_path: str = ""
    log_path: str = ""

    test_size: float = Field(default=0.2, description="Validation split ratio (80/20).")
    random_seed: int = Field(default=42, description="Seed for deterministic stratification and splitting.")

    @model_validator(mode='after')
    def build_paths(self) -> 'CacheConfig':
        """Dynamically maps operational paths relative to the script's execution location."""
        self.generated_dir = os.path.join(self.module_dir, "generated_files")
        self.dataset_dir = os.path.join(self.module_dir, "openml_cc18_datasets")
        self.metadata_path = os.path.join(self.generated_dir, self.phase_csv_name)
        self.log_path = os.path.join(self.generated_dir, "openml_cc18_download_log.csv")
        return self


class DatasetManager:
    """
    Ingests raw CC18 CSVs, applies strictly isolated topological transformations 
    (Universal StandardScaler Baseline), and pins PyTorch tensors into system RAM.
    """

    def __init__(self, config: CacheConfig):
        self.cfg = config
        self.dataset_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.meta_features_cache: Dict[int, torch.Tensor] = {}

        # Load the target mapping offline to prevent OpenML API network calls
        self._target_mapping = self._build_target_mapping()

    def _build_target_mapping(self) -> Dict[int, str]:
        if not os.path.exists(self.cfg.log_path):
            logger.warning(f"Download log missing at {self.cfg.log_path}. Relying on positional fallback.")
            return {}

        log_df = pd.read_csv(self.cfg.log_path)
        return dict(zip(log_df['did'], log_df['target']))

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

                self.meta_features_cache[did] = torch.tensor([
                    row["n_d_ratio"], row["feat_kurtosis"], row["iqr_dev"],
                    row["pc_eigen"], row["target_entropy"], row["hopkins"],
                    row["silhouette"], row["davies_bouldin"]
                ], dtype=torch.float32)

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