# Module 3 General Function: Ingests arrays, performs ordinal scaling, and manages the in-RAM tensor cache.

import os
import glob
import openml
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


class DatasetManager:
    """
    Handles the ingestion, caching, and distribution of datasets for the ACI project.
    Loads data into RAM to prevent I/O bottlenecking during the 240,000 inner-loop evaluations.
    """

    def __init__(self, metadata_csv_path, dataset_directory):
        self.metadata_csv_path = metadata_csv_path
        self.dataset_directory = dataset_directory

        # In-memory RAM caches
        self.dataset_cache = {}
        self.meta_features_cache = {}

    def find_local_dataset_file(self, did, dataset_name):
        """
        Finds a previously downloaded dataset CSV.

        Supports filenames produced by the previous downloader, e.g.:
            3_kr-vs-kp.csv
            31_credit-g.csv

        Falls back to any CSV beginning with the dataset ID.
        """
        did = int(did)

        # Primary pattern: any file beginning with "<did>_"
        pattern = os.path.join(self.dataset_directory, f"{did}_*.csv")
        matches = glob.glob(pattern)

        if matches:
            return matches[0]

        # Fallback pattern: any CSV beginning with the dataset ID
        pattern = os.path.join(self.dataset_directory, f"{did}*.csv")
        matches = glob.glob(pattern)

        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"No local CSV file found for dataset ID {did} ({dataset_name}) "
            f"in {self.dataset_directory}"
        )

    def read_local_dataset_csv(self, file_path):
        """
        Reads a locally saved OpenML dataset CSV.

        The previously generated downloader saved files with comma delimiters.
        A semicolon fallback is included for compatibility with older local exports.
        """
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")

        # If the file was semicolon-delimited, pandas may read it as one wide string column.
        if df.shape[1] == 1:
            try:
                df_semicolon = pd.read_csv(file_path, sep=";")
            except UnicodeDecodeError:
                df_semicolon = pd.read_csv(file_path, sep=";", encoding="latin1")

            if df_semicolon.shape[1] > 1:
                df = df_semicolon

        return df

    def split_features_and_target(self, df, did):
        """
        Splits a locally saved CSV into X and y.

        Uses OpenML metadata only, with download_data=False, to recover the target column name.
        This does not re-download the dataset contents.
        """
        dataset = openml.datasets.get_dataset(did, download_data=False)
        target_col = dataset.default_target_attribute

        if target_col is not None and target_col in df.columns:
            y = df[target_col]
            X_df = df.drop(columns=[target_col])
        elif "TARGET" in df.columns:
            y = df["TARGET"]
            X_df = df.drop(columns=["TARGET"])
        elif "target" in df.columns:
            y = df["target"]
            X_df = df.drop(columns=["target"])
        else:
            # Fallback for manually saved files where the target is the last column
            target_col = df.columns[-1]
            print(
                f"Warning: OpenML target column not found for dataset {did}. "
                f"Using last column as target: {target_col}"
            )
            y = df[target_col]
            X_df = df.drop(columns=[target_col])

        return X_df, y

    def preprocess_features(self, X_df):
        """
        Encodes categorical features, imputes missing values, and converts features
        into a numeric NumPy array suitable for PyTorch.
        """
        X_df = X_df.copy()

        # Normalize placeholder missing values
        X_df = X_df.replace(
            ["?", "NA", "N/A", "nan", "NaN", "None", ""],
            np.nan
        )

        cat_cols = X_df.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        num_cols = [col for col in X_df.columns if col not in cat_cols]

        # Coerce numeric columns safely
        for col in num_cols:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        # Impute numeric columns
        if len(num_cols) > 0:
            X_df[num_cols] = SimpleImputer(strategy="median").fit_transform(
                X_df[num_cols]
            )

        # Impute and encode categorical columns
        if len(cat_cols) > 0:
            X_df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
                X_df[cat_cols]
            )

            X_df[cat_cols] = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ).fit_transform(X_df[cat_cols])

        # Final safety conversion
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        X_df = X_df.fillna(0)

        X = X_df.to_numpy(dtype="float32")

        return X

    def preprocess_target(self, y):
        """
        Encodes target labels into integer class IDs for PyTorch.
        """
        y = pd.Series(y).replace(
            ["?", "NA", "N/A", "nan", "NaN", "None", ""],
            np.nan
        )

        y = y.fillna("MISSING_TARGET").astype(str)

        encoder_y = LabelEncoder()
        y_encoded = encoder_y.fit_transform(y)

        return y_encoded.astype("int64")

    def load_all_to_ram(self):
        """
        Iterates through the Phase A metadata, loads the corresponding data files,
        performs an 80/20 train/test split, and converts everything to PyTorch tensors.
        """
        print(f"Loading Phase A metadata from {self.metadata_csv_path}...")
        metadata_df = pd.read_csv(self.metadata_csv_path)

        for _, row in metadata_df.iterrows():
            did = int(row["did"])
            name = row["name"]

            try:
                # Extract the 8-dimensional meta-feature vector (M)
                meta_features = torch.tensor([
                    row["n_d_ratio"],
                    row["feat_kurtosis"],
                    row["iqr_dev"],
                    row["pc_eigen"],
                    row["target_entropy"],
                    row["hopkins"],
                    row["silhouette"],
                    row["davies_bouldin"]
                ], dtype=torch.float32)

                self.meta_features_cache[did] = meta_features

                # Locate and load the actual dataset
                file_path = self.find_local_dataset_file(did, name)
                df = self.read_local_dataset_csv(file_path)

                # Isolate features and target
                X_df, y_raw = self.split_features_and_target(df, did)

                # Encode/impute features and encode targets
                X = self.preprocess_features(X_df)
                y = self.preprocess_target(y_raw)

                # Standard 80/20 split for inner-loop PyTorch training/validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )

                # Scale inputs; crucial for FNN gradient stability
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                # Final safety guard against NaN/Inf values
                X_train = np.nan_to_num(
                    X_train,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                X_val = np.nan_to_num(
                    X_val,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                # Cast to PyTorch tensors and push to dictionary cache
                self.dataset_cache[did] = {
                    "X_train": torch.tensor(X_train, dtype=torch.float32),
                    "y_train": torch.tensor(y_train, dtype=torch.long),
                    "X_val": torch.tensor(X_val, dtype=torch.float32),
                    "y_val": torch.tensor(y_val, dtype=torch.long)
                }

            except Exception as e:
                print(f"Error loading dataset {name} (ID: {did}): {e}")

        print(f"Successfully cached {len(self.dataset_cache)} datasets in RAM.")

    def get_dataset(self, did):
        """
        Returns the PyTorch tensors and the meta-feature vector for a specific dataset ID.
        """
        if did not in self.dataset_cache:
            raise KeyError(
                f"Dataset ID {did} not found in cache. Did you run load_all_to_ram()?"
            )

        return self.dataset_cache[did], self.meta_features_cache[did]


# --- Quick Test Block ---
if __name__ == "__main__":
    PHASE_A_CSV = "Phase_A_Discovery_Datasets.csv"
    DATA_DIR = r"C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project\openml_cc18_datasets"

    manager = DatasetManager(
        metadata_csv_path=PHASE_A_CSV,
        dataset_directory=DATA_DIR
    )

    manager.load_all_to_ram()

    print("\nCached dataset IDs:")
    print(list(manager.dataset_cache.keys()))

    if len(manager.dataset_cache) > 0:
        sample_did = list(manager.dataset_cache.keys())[0]
        tensors, meta_features = manager.get_dataset(sample_did)

        print(f"\nSample dataset ID: {sample_did}")
        print("X_train shape:", tensors["X_train"].shape)
        print("y_train shape:", tensors["y_train"].shape)
        print("X_val shape:", tensors["X_val"].shape)
        print("y_val shape:", tensors["y_val"].shape)
        print("Meta-features shape:", meta_features.shape)
