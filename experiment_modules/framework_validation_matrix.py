# Module 7 General Function: Executes the out-of-distribution validation matrix against the 7 baseline heuristics.

import numpy as np
import scipy.stats as stats
import torch
import math
import json
import os
import warnings
from sklearn.cluster import KMeans
from pm_dataset_manager import DatasetManager
from pm_fnn_landscape import PyTorchEvaluator

warnings.filterwarnings("ignore")


def rank_1_rule(pc_eigen, target_entropy):
    """
    The translated Rank 1 equation from the MOGP engine:
    neg(mul(-0.07458949361120437, protected_div(pc_eigen, target_entropy)))
    """
    # Protect division just like the GP engine did
    denominator = target_entropy if abs(target_entropy) > 1e-5 else 1.0
    return 0.07458949361120437 * (pc_eigen / denominator)


def apply_baseline_initialization(model, method, dataset, m_vals):
    """
    Dynamically intercepts the PyTorch model and applies the chosen
    Static or Data-Driven SOTA initialization heuristic.
    """
    X_train = dataset['X_train']
    iqr_dev = m_vals[2]  # Extracted from the Dataset Manager

    with torch.no_grad():
        if method == 'Xavier_Glorot':
            for layer in [model.layer1, model.layer2, model.output_layer]:
                torch.nn.init.xavier_normal_(layer.weight)

        elif method == 'He_Kaiming':
            for layer in [model.layer1, model.layer2, model.output_layer]:
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        elif method == 'LeCun':
            for layer in [model.layer1, model.layer2, model.output_layer]:
                fan_in = layer.weight.size(1)
                torch.nn.init.normal_(layer.weight, mean=0., std=math.sqrt(1.0 / fan_in))

        elif method == 'Orthogonal':
            for layer in [model.layer1, model.layer2, model.output_layer]:
                torch.nn.init.orthogonal_(layer.weight)

        elif method == 'FAVI':
            # Feature-Adaptive Variance Initialization
            # Modifies topological variance scaling using the dataset's IQR deviation
            for layer in [model.layer1, model.layer2, model.output_layer]:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
                std = math.sqrt(2.0 / (fan_in + fan_out)) * (1.0 + iqr_dev)
                torch.nn.init.normal_(layer.weight, mean=0., std=std)

        elif method == 'Laor':
            # Cluster-Driven proxy weights using KMeans for the input layer
            num_clusters = model.layer1.weight.size(0)
            n_samples = X_train.shape[0]
            actual_clusters = min(num_clusters, n_samples)

            kmeans = KMeans(n_clusters=actual_clusters, n_init=1, random_state=42)
            kmeans.fit(X_train.numpy())
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

            # Prevent crashing if a dataset has fewer samples than neurons
            if actual_clusters < num_clusters:
                pad = torch.randn((num_clusters - actual_clusters, X_train.shape[1]))
                centers = torch.cat([centers, pad], dim=0)

            model.layer1.weight.copy_(centers)

            # Initialize subsequent layers with Kaiming
            for layer in [model.layer2, model.output_layer]:
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        elif method == 'LSUV':
            # Layer-Sequential Unit-Variance
            for layer in [model.layer1, model.layer2, model.output_layer]:
                torch.nn.init.orthogonal_(layer.weight)

            # Forward pass a mini-batch to scale weights to unit variance
            batch = X_train[:128]
            x = batch
            for layer in [model.layer1, model.layer2]:
                x = layer(x)
                if hasattr(model, 'activation'):
                    x = model.activation(x)
                var = torch.var(x)
                if var > 1e-6:
                    layer.weight.data /= torch.sqrt(var)

        # METHODOLOGY FIX: Ensure all biases are zeroed out across all methods to isolate weight impacts
        for layer in [model.layer1, model.layer2, model.output_layer]:
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)


def main():
    GEN_DIR = r"C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project\generated_files"
    DATA_DIR = r"C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project\openml_cc18_datasets"

    print("Loading Phase B Validation Data (51 Unseen Datasets)...")
    # Point the DatasetManager to the generated_files directory for the CSV
    manager = DatasetManager(os.path.join(GEN_DIR, "Phase_B_Validation_Datasets.csv"), DATA_DIR)
    manager.load_all_to_ram()

    baselines = ['Xavier_Glorot', 'He_Kaiming', 'LeCun', 'Orthogonal', 'LSUV', 'FAVI', 'Laor']
    results = {method: [] for method in baselines}
    results['GP_Rule'] = []

    print("\n--- COMMENCING MASSIVE PHASE B VALIDATION (1 vs 7 Matrix) ---")

    for idx, did in enumerate(manager.dataset_cache.keys()):
        dataset, meta_features = manager.get_dataset(did)
        m_vals = meta_features.numpy()
        pc_eigen = m_vals[3]
        target_entropy = m_vals[4]

        sigma_squared = rank_1_rule(pc_eigen, target_entropy)
        evaluator_gp = PyTorchEvaluator(dataset, sigma_squared=sigma_squared, max_epochs=30)
        acc_gp, _ = evaluator_gp.evaluate_fitness()
        results['GP_Rule'].append(acc_gp)

        for method in baselines:
            evaluator_base = PyTorchEvaluator(dataset, sigma_squared=1e-5, max_epochs=30)
            apply_baseline_initialization(evaluator_base.model, method, dataset, m_vals)
            acc_base, _ = evaluator_base.evaluate_fitness()
            results[method].append(acc_base)

        print(
            f"[{idx + 1}/{len(manager.dataset_cache)}] Dataset {did} Processed -> GP: {acc_gp * 100:.1f}% | Kaiming: {results['He_Kaiming'][-1] * 100:.1f}% | Laor: {results['Laor'][-1] * 100:.1f}%")

    # ... [Keep your console printing for the Wilcoxon stats the same] ...

    # --- JSON EXPORT BLOCK FOR MODULE 8 ---
    # Route the JSON payload to the generated_files directory
    json_path = os.path.join(GEN_DIR, "validation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f)
    print(f"\nRaw results successfully exported to '{json_path}' for qualitative analysis.")


if __name__ == "__main__":
    main()