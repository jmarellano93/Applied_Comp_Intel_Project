import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
class DatasetManager:
    """
    Handles the ingestion, caching, and distribution of datasets for the ACI project.
    Loads data into RAM to prevent I/O bottlenecking during the 40,000 inner-loop evaluations.
    """

    def __init__(self, metadata_csv_path, dataset_directory):
        self.metadata_csv_path = metadata_csv_path
        self.dataset_directory = dataset_directory

        # In-memory RAM caches
        self.dataset_cache = {}
        self.meta_features_cache = {}

    def load_all_to_ram(self):
        """
        Iterates through the Phase A metadata, loads the corresponding European-formatted
        data files, performs an 80/20 train/test split, and converts everything to PyTorch tensors.
        """
        print(f"Loading Phase A metadata from {self.metadata_csv_path}...")
        metadata_df = pd.read_csv(self.metadata_csv_path)

        for _, row in metadata_df.iterrows():
            did = int(row['did'])
            name = row['name']

            # Extract the 8-dimensional meta-feature vector (M)
            meta_features = torch.tensor([
                row['n_d_ratio'],
                row['feat_kurtosis'],
                row['iqr_dev'],
                row['pc_eigen'],
                row['target_entropy'],
                row['hopkins'],
                row['silhouette'],
                row['davies_bouldin']
            ], dtype=torch.float32)

            self.meta_features_cache[did] = meta_features

            # Construct the file path for the actual dataset
            file_path = os.path.join(self.dataset_directory, f"{did}_{name}.csv")

            if not os.path.exists(file_path):
                print(f"Warning: Data file {file_path} not found. Skipping.")
                continue

            # Load actual data using the semicolon delimiter for European formatting
            df = pd.read_csv(file_path, sep=';')

            # Isolate the feature dataframe
            X_df = df.drop(columns=['TARGET'])

            # Identify any string/categorical columns and encode them to numeric values
            cat_cols = X_df.select_dtypes(include=['object', 'string', 'category']).columns
            if len(cat_cols) > 0:
                X_df[cat_cols] = OrdinalEncoder().fit_transform(X_df[cat_cols])

            # Now it is safe to force the standard NumPy array for features
            X = X_df.to_numpy(dtype='float32')

            # The target column may contain string labels (e.g., 'tested_positive').
            # We must encode these into integers (0, 1, 2...) for PyTorch.
            encoder_y = LabelEncoder()
            y_encoded = encoder_y.fit_transform(df['TARGET'])
            y = y_encoded.astype('int64')

            # Standard 80/20 split for inner-loop PyTorch training/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale inputs (Crucial for FNN gradient stability)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            # Cast to PyTorch Tensors and push to dictionary cache
            self.dataset_cache[did] = {
                'X_train': torch.tensor(X_train, dtype=torch.float32),
                'y_train': torch.tensor(y_train, dtype=torch.long),
                'X_val': torch.tensor(X_val, dtype=torch.float32),
                'y_val': torch.tensor(y_val, dtype=torch.long)
            }

        print(f"Successfully cached {len(self.dataset_cache)} datasets in RAM.")

    def get_dataset(self, did):
        """
        Returns the PyTorch tensors and the meta-feature vector for a specific dataset ID.
        """
        if did not in self.dataset_cache:
            raise KeyError(f"Dataset ID {did} not found in cache. Did you run load_all_to_ram()?")

        return self.dataset_cache[did], self.meta_features_cache[did]


# --- Quick Test Block ---
if __name__ == "__main__":
    # Update these paths to match your local PyCharm environment
    PHASE_A_CSV = "Phase_A_Discovery_Datasets.csv"
    DATA_DIR = r"C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project\Potential_Dataset_Repository"

    manager = DatasetManager(metadata_csv_path=PHASE_A_CSV, dataset_directory=DATA_DIR)
    manager.load_all_to_ram()