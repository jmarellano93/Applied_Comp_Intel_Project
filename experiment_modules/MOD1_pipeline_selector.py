"""
Module 1: OpenML-CC18 Pipeline Selector

Handles the primary dataset retrieval, feature reduction, and size filtering tasks.
Strictly enforces mathematical constraints on data dimensionality prior to download
to optimize network I/O and local storage.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import openml
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# FUNCTIONAL BLOCK: Pipeline Configuration
# WHAT IT DOES: Validates and stores the absolute mathematical boundaries
#  and architectural paths for dataset ingestion.
# PARAMETERS:
#  - min_instances (500)
#  - max_instances (15000)
#  - max_features (200)
# METHODOLOGICAL JUSTIFICATION:
#  - min_instances (500): Guarantees that after an 80/20 Train/Test split,
#    the neural network still receives at least 400 training samples, preventing
#    severe data starvation and stochastic gradient noise.
#  - max_instances (15000): Caps the upper bound to prevent the $O(N^2)$ time
#    complexity of spatial meta-feature extraction (like Hopkins statistic)
#    from stalling the pipeline.
#  - max_features (200): Phase A architectures have a 64-neuron input bound.
#    Restricting features to <= 200 ensures we do not force massive, destructive
#    compression bottlenecks in the first layer.
#  - OPENML API-KEY: Paste the API-Key associated with your OPENML account.

# =============================================================================
class PipelineConfig(BaseModel):
    api_key: str = Field(default="13d5e3978a2db3c8e91409ca7e75b7bd", description="OpenML API Key")
    suite_id: int = Field(default=99, description="OpenML Suite ID for CC18 Curated Classification")

    min_instances: int = Field(default=500, description="Minimum allowed instances (rows)")
    max_instances: int = Field(default=15000, description="Maximum allowed instances (rows)")
    max_features: int = Field(default=200, description="Maximum allowed features (columns)")

    dataset_dir: str = Field(default="openml_cc18_datasets", description="Directory for raw dataset CSVs")
    generated_dir: str = Field(default="generated_files", description="Directory for metadata and logs")

    @field_validator('max_features')
    def validate_features(cls, v):
        if v <= 0:
            raise ValueError("Maximum features must be strictly positive.")
        return v


def setup_directories(config: PipelineConfig) -> None:
    os.makedirs(config.dataset_dir, exist_ok=True)
    os.makedirs(config.generated_dir, exist_ok=True)
    logger.info(f"Directories verified: '{config.dataset_dir}', '{config.generated_dir}'")


# =============================================================================
# FUNCTIONAL BLOCK: Metadata Fetching & Filtering
# WHAT IT DOES: Downloads the CC18 suite metadata and drops datasets that
#  violate our dimensional constraints via vectorized pandas operations.
# PARAMETERS: Requires a valid PipelineConfig containing threshold parameters.
# METHODOLOGICAL JUSTIFICATION: Filtering the metadata *before* executing
#  the physical data download prevents massive network I/O waste, blocking
#  gigabytes of unusable data (like raw image matrices) from consuming local RAM.
# =============================================================================
def fetch_and_filter_metadata(config: PipelineConfig) -> pd.DataFrame:
    openml.config.apikey = config.api_key
    logger.info(f"Fetching OpenML suite {config.suite_id}...")

    suite = openml.study.get_suite(config.suite_id)
    metadata_df = openml.datasets.list_datasets(data_id=suite.data, output_format="dataframe")

    initial_count = len(metadata_df)
    logger.info(f"Found {initial_count} initial datasets in suite.")

    mask_compliant = (
            (metadata_df['NumberOfInstances'] >= config.min_instances) &
            (metadata_df['NumberOfInstances'] <= config.max_instances) &
            (metadata_df['NumberOfFeatures'] <= config.max_features)
    )

    filtered_df = metadata_df.loc[mask_compliant].copy()
    filtered_df["n_d_ratio"] = filtered_df["NumberOfInstances"] / filtered_df["NumberOfFeatures"]

    dropped_count = initial_count - len(filtered_df)
    logger.info(f"Filtered out {dropped_count} datasets violating dimensional constraints.")

    return filtered_df


# =============================================================================
# FUNCTIONAL BLOCK: Dataset Download & Initial Cleaning
# WHAT IT DOES: Downloads the CSV, drops NaN rows, and strips zero-variance columns.
# PARAMETERS: did (OpenML ID), name (Dataset Name), config (PipelineConfig).
# METHODOLOGICAL JUSTIFICATION:
#  - Dropping NaNs at the row level prevents structural failures during spatial calculations.
#  - Dropping zero-variance (constant) columns is mathematically required because
#    a constant feature provides zero information gain for gradient descent and
#    causes division-by-zero crashes during StandardScaler operations later.
# =============================================================================
def download_and_process_dataset(did: int, name: str, config: PipelineConfig) -> Dict[str, Any]:
    log_entry = {
        "did": did, "name": name, "target": None, "status": "failed",
        "file": None, "n_rows": None, "n_columns": None, "error": None
    }

    try:
        dataset = openml.datasets.get_dataset(did, download_data=True, download_qualities=False, download_features_meta_data=False)
        target_attr = dataset.default_target_attribute
        log_entry["target"] = target_attr

        X, y, _, _ = dataset.get_data(target=target_attr, dataset_format="dataframe")
        target_col_name = target_attr if target_attr is not None else "target"
        X[target_col_name] = y

        X.dropna(inplace=True)
        X = X.loc[:, X.nunique() > 1]

        if X.empty:
            raise ValueError("Dataset collapsed to 0 rows or 0 columns after cleaning.")

        filepath = os.path.join(config.dataset_dir, f"{did}_{name}.csv")
        X.to_csv(filepath, index=False)

        log_entry.update({"status": "downloaded", "file": filepath, "n_rows": X.shape[0], "n_columns": X.shape[1]})
        logger.info(f"Successfully processed dataset {did}: {name} ({X.shape[0]} rows, {X.shape[1]} cols)")

    except Exception as e:
        log_entry["error"] = str(e)
        logger.error(f"Failed to process dataset {did} ({name}): {e}")

    return log_entry


def run_pipeline(config: PipelineConfig) -> None:
    setup_directories(config)
    filtered_metadata = fetch_and_filter_metadata(config)

    metadata_path = os.path.join(config.generated_dir, "openml_cc18_metadata.csv")
    filtered_metadata.to_csv(metadata_path, index=False)
    logger.info(f"Saved compliant metadata to {metadata_path}")

    download_log = []
    for _, row in filtered_metadata.iterrows():
        did = row["did"]
        name = row["name"]
        log_result = download_and_process_dataset(did, name, config)
        download_log.append(log_result)

    log_df = pd.DataFrame(download_log)
    log_path = os.path.join(config.generated_dir, "openml_cc18_download_log.csv")
    log_df.to_csv(log_path, index=False)
    logger.info(f"Pipeline complete. Download log saved to {log_path}")


if __name__ == "__main__":
    try:
        pipeline_cfg = PipelineConfig()
        run_pipeline(pipeline_cfg)
    except Exception as e:
        logger.critical(f"Pipeline execution halted due to critical failure: {e}")