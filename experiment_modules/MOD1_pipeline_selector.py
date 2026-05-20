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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    """
    Configuration schema for the Pipeline Selector.
    Validates and stores mathematical boundaries and architectural paths.
    """
    api_key: str = Field(default="13d5e3978a2db3c8e91409ca7e75b7bd", description="OpenML API Key")
    suite_id: int = Field(default=99, description="OpenML Suite ID for CC18 Curated Classification")

    # Mathematical Thresholds
    min_instances: int = Field(default=500, description="Minimum allowed instances (rows)")
    max_instances: int = Field(default=15000, description="Maximum allowed instances (rows)")
    max_features: int = Field(default=200, description="Maximum allowed features (columns)")

    # Directory Paths
    dataset_dir: str = Field(default="openml_cc18_datasets", description="Directory for raw dataset CSVs")
    generated_dir: str = Field(default="generated_files", description="Directory for metadata and logs")

    @field_validator('max_features')
    def validate_features(cls, v):
        if v <= 0:
            raise ValueError("Maximum features must be strictly positive.")
        return v


def setup_directories(config: PipelineConfig) -> None:
    """
    Ensures that required output directories exist.

    Args:
        config (PipelineConfig): Validated configuration parameters.
    """
    os.makedirs(config.dataset_dir, exist_ok=True)
    os.makedirs(config.generated_dir, exist_ok=True)
    logger.info(f"Directories verified: '{config.dataset_dir}', '{config.generated_dir}'")


def fetch_and_filter_metadata(config: PipelineConfig) -> pd.DataFrame:
    """
    Retrieves OpenML suite metadata and applies vectorized filters based on
    dimensionality constraints to prevent unnecessary network I/O.

    Args:
        config (PipelineConfig): Validated configuration parameters.

    Returns:
        pd.DataFrame: Filtered metadata DataFrame containing only compliant datasets.
    """
    openml.config.apikey = config.api_key
    logger.info(f"Fetching OpenML suite {config.suite_id}...")

    suite = openml.study.get_suite(config.suite_id)
    metadata_df = openml.datasets.list_datasets(data_id=suite.data, output_format="dataframe")

    initial_count = len(metadata_df)
    logger.info(f"Found {initial_count} initial datasets in suite.")

    # Apply mathematical constraints using vectorized boolean indexing
    # JUSTIFICATION FOR PRE-DOWNLOAD FEATURE FILTERING:
    # 1. Architectural Constraint: The Phase A FNN input layer is constrained to 64 neurons. 
    #    Feeding >200 features forces massive compression, inducing underfitting that confounds variance calculations.
    # 2. Network/Storage Efficiency: Prevents the expensive download of massive datasets (e.g., CIFAR_10, MNIST)
    #    that will ultimately be discarded, optimizing Big-O time and space complexity of the pipeline.
    mask_compliant = (
            (metadata_df['NumberOfInstances'] >= config.min_instances) &
            (metadata_df['NumberOfInstances'] <= config.max_instances) &
            (metadata_df['NumberOfFeatures'] <= config.max_features)
    )

    filtered_df = metadata_df.loc[mask_compliant].copy()

    # Calculate N/D ratio safely (features is guaranteed > 0 by OpenML, but we use safe division implicitly)
    filtered_df["n_d_ratio"] = filtered_df["NumberOfInstances"] / filtered_df["NumberOfFeatures"]

    dropped_count = initial_count - len(filtered_df)
    logger.info(f"Filtered out {dropped_count} datasets violating dimensional constraints.")

    return filtered_df


def download_and_process_dataset(did: int, name: str, config: PipelineConfig) -> Dict[str, Any]:
    """
    Downloads a single dataset, applies data integrity checks (NaN/Constant removal),
    and saves it to the configured directory.

    Args:
        did (int): OpenML Dataset ID.
        name (str): Dataset Name.
        config (PipelineConfig): Validated configuration parameters.

    Returns:
        Dict[str, Any]: A structured log dictionary detailing the operation's outcome.
    """
    log_entry = {
        "did": did,
        "name": name,
        "target": None,
        "status": "failed",
        "file": None,
        "n_rows": None,
        "n_columns": None,
        "error": None
    }

    try:
        dataset = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                              download_features_meta_data=False)
        target_attr = dataset.default_target_attribute
        log_entry["target"] = target_attr

        # Fetch data as Pandas DataFrame for vectorized processing
        X, y, _, _ = dataset.get_data(target=target_attr, dataset_format="dataframe")

        # Merge target column for unified processing
        target_col_name = target_attr if target_attr is not None else "target"
        X[target_col_name] = y

        # 1. Integrity Check: Drop rows with missing values (NaN)
        X.dropna(inplace=True)

        # 2. Dimensionality Reduction: Drop constant features (zero-variance)
        # We exploit pandas nunique() for vectorized column filtering
        X = X.loc[:, X.nunique() > 1]

        # 3. Post-Cleaning Dimensionality Check: Ensure dataset is not empty
        if X.empty:
            raise ValueError("Dataset collapsed to 0 rows or 0 columns after cleaning.")

        # Export to CSV
        filepath = os.path.join(config.dataset_dir, f"{did}_{name}.csv")
        X.to_csv(filepath, index=False)

        # Update log
        log_entry.update({
            "status": "downloaded",
            "file": filepath,
            "n_rows": X.shape[0],
            "n_columns": X.shape[1]
        })
        logger.info(f"Successfully processed dataset {did}: {name} ({X.shape[0]} rows, {X.shape[1]} cols)")

    except Exception as e:
        log_entry["error"] = str(e)
        logger.error(f"Failed to process dataset {did} ({name}): {e}")

    return log_entry


def run_pipeline(config: PipelineConfig) -> None:
    """
    Orchestrates the OpenML data fetching, filtering, and processing pipeline.

    Args:
        config (PipelineConfig): Validated configuration parameters.
    """
    setup_directories(config)

    # 1. Fetch and Pre-Filter Metadata
    filtered_metadata = fetch_and_filter_metadata(config)

    # Save Metadata to Generated Directory
    metadata_path = os.path.join(config.generated_dir, "openml_cc18_metadata.csv")
    filtered_metadata.to_csv(metadata_path, index=False)
    logger.info(f"Saved compliant metadata to {metadata_path}")

    # 2. Download and Process Datasets
    download_log = []

    for _, row in filtered_metadata.iterrows():
        did = row["did"]
        name = row["name"]
        log_result = download_and_process_dataset(did, name, config)
        download_log.append(log_result)

    # 3. Save Download Log
    log_df = pd.DataFrame(download_log)
    log_path = os.path.join(config.generated_dir, "openml_cc18_download_log.csv")
    log_df.to_csv(log_path, index=False)
    logger.info(f"Pipeline complete. Download log saved to {log_path}")


if __name__ == "__main__":
    # Initialize Pydantic Configuration Model
    try:
        pipeline_cfg = PipelineConfig()
        run_pipeline(pipeline_cfg)
    except Exception as e:
        logger.critical(f"Pipeline execution halted due to critical failure: {e}")