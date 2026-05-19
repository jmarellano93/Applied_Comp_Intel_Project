# Module 1 General Function: Handles the primary CC-18 suite retrieval, feature reduction, and size filtering tasks.

import os
import re
import openml
import pandas as pd
from tqdm import tqdm

# Configuration
openml.config.apikey = "13d5e3978a2db3c8e91409ca7e75b7bd"

# Output directory
OUTPUT_DIR = "../openml_cc18_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Get the OpenML-CC18 Benchmark Suite
# Suite ID 99 = OpenML-CC18 Curated Classification benchmark
print("Fetching OpenML-CC18 suite information...")
suite = openml.study.get_suite(99)

dataset_ids = suite.data

print(f"Found {len(dataset_ids)} datasets in OpenML-CC18.")

# Optional sanity check
if len(dataset_ids) != 72:
    print(f"Warning: expected 72 datasets, but found {len(dataset_ids)}.")

# 2. List datasets with basic statistics
print("Fetching dataset metadata...")
df = openml.datasets.list_datasets(
    data_id=dataset_ids,
    output_format="dataframe"
)

# 3. Add simple meta-features
df["n_d_ratio"] = df["NumberOfInstances"] / df["NumberOfFeatures"]

df["missing_ratio"] = (
    df["NumberOfMissingValues"] /
    (df["NumberOfInstances"] * df["NumberOfFeatures"])
)

# 4. Save metadata for all 72 datasets
metadata_path = os.path.join(OUTPUT_DIR, "openml_cc18_metadata.csv")
df.to_csv(metadata_path, index=False)

print(f"Saved metadata to: {metadata_path}")

# 5. Helper function to make safe filenames
def safe_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^\w\-\.]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


# 6. Download every dataset in the suite
download_log = []

print("Downloading all OpenML-CC18 datasets...")

for did in tqdm(dataset_ids):
    try:
        dataset = openml.datasets.get_dataset(did, download_data=True)

        # Get dataset name from metadata if available
        dataset_name = df.loc[df["did"] == did, "name"].iloc[0]
        filename = f"{did}_{safe_filename(dataset_name)}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Convert OpenML dataset to pandas DataFrame
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format="dataframe"
        )

        # Add target column to the exported CSV
        if dataset.default_target_attribute is not None:
            X[dataset.default_target_attribute] = y
        else:
            X["target"] = y

        # Save dataset
        X.to_csv(filepath, index=False)

        download_log.append({
            "did": did,
            "name": dataset_name,
            "target": dataset.default_target_attribute,
            "status": "downloaded",
            "file": filepath,
            "n_rows": X.shape[0],
            "n_columns": X.shape[1]
        })

    except Exception as e:
        download_log.append({
            "did": did,
            "name": df.loc[df["did"] == did, "name"].iloc[0]
                    if did in df["did"].values else None,
            "target": None,
            "status": "failed",
            "file": None,
            "n_rows": None,
            "n_columns": None,
            "error": str(e)
        })

        print(f"\nFailed to download dataset {did}: {e}")

# 7. Save download log
log_df = pd.DataFrame(download_log)
log_path = os.path.join(OUTPUT_DIR, "openml_cc18_download_log.csv")
log_df.to_csv(log_path, index=False)

print("\nDownload complete.")
print(f"Datasets saved in: {OUTPUT_DIR}")
print(f"Download log saved to: {log_path}")

print("\nDownload summary:")
print(log_df["status"].value_counts())
