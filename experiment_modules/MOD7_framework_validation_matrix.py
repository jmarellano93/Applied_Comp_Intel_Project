import os
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import math
import json
import argparse
import warnings
from sklearn.cluster import KMeans
from MOD3_pm_dataset_manager import DatasetManager
from MOD4_pm_fnn_landscape import PyTorchEvaluator

warnings.filterwarnings("ignore")


def rank_1_rule(pc_eigen, target_entropy):
    """
    The exact Rank 1 equation from the 24-hour MOGP run:
    mul(0.0730307591260062, protected_div(protected_sqrt(0.32445421948953435), target_entropy))
    """
    # Protect division just like the GP engine did
    denominator = target_entropy if abs(target_entropy) > 1e-5 else 1.0

    # Calculate the exact mathematical representation of the GP tree
    numerator_constant = 0.0730307591260062 * math.sqrt(0.32445421948953435)

    return numerator_constant / denominator


def apply_baseline_initialization(model, method, dataset, m_vals):
    """
    Dynamically intercepts the PyTorch model and applies the chosen
    Static or Data-Driven SOTA initialization heuristic to all linear layers.
    """
    X_train = dataset['X_train']
    iqr_dev = m_vals[2]

    # Dynamically extract all linear layers regardless of architecture (Phase A or B)
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    with torch.no_grad():
        if method == 'Xavier_Glorot':
            for layer in linear_layers: torch.nn.init.xavier_normal_(layer.weight)
        elif method == 'He_Kaiming':
            for layer in linear_layers: torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif method == 'LeCun':
            for layer in linear_layers: torch.nn.init.normal_(layer.weight, mean=0.,
                                                              std=math.sqrt(1.0 / layer.weight.size(1)))
        elif method == 'Orthogonal':
            for layer in linear_layers: torch.nn.init.orthogonal_(layer.weight)
        elif method == 'FAVI':
            for layer in linear_layers:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
                std = math.sqrt(2.0 / (fan_in + fan_out)) * (1.0 + iqr_dev)
                torch.nn.init.normal_(layer.weight, mean=0., std=std)
        elif method == 'Laor':
            first_layer = linear_layers[0]
            num_clusters = first_layer.weight.size(0)
            actual_clusters = min(num_clusters, X_train.shape[0])
            kmeans = KMeans(n_clusters=actual_clusters, n_init=1, random_state=42)
            kmeans.fit(X_train.numpy())
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            if actual_clusters < num_clusters:
                centers = torch.cat([centers, torch.randn((num_clusters - actual_clusters, X_train.shape[1]))], dim=0)
            first_layer.weight.copy_(centers)
            for layer in linear_layers[1:]: torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif method == 'LSUV':
            for layer in linear_layers: torch.nn.init.orthogonal_(layer.weight)
            current_input = X_train[:128]
            for m in model.modules():
                if isinstance(m, nn.Linear) and m != linear_layers[-1]:
                    current_input = m(current_input)
                    if hasattr(model, 'activation'):
                        current_input = model.activation(current_input)
                    var = torch.var(current_input)
                    if var > 1e-6: m.weight.data /= torch.sqrt(var)

        # METHODOLOGY FIX: Ensure all biases are zeroed out across all methods
        for layer in linear_layers:
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)


def main():
    parser = argparse.ArgumentParser(description="Run Phase B Validation Matrix.")
    parser.add_argument('--topology', type=str, default='shallow', choices=['shallow', 'deep_narrow', 'funnel'],
                        help="The FNN architecture to stress-test the Rank 1 Rule against.")
    args = parser.parse_args()

    # Dynamic Pathing to ensure old_shit handles the I/O
    GEN_DIR = r"/old_shit"
    DATA_DIR = r"C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project\openml_cc18_datasets"

    print("Loading Phase B Validation Data (51 Unseen Datasets)...")
    manager = DatasetManager(os.path.join(GEN_DIR, "Phase_B_Validation_Datasets.csv"), DATA_DIR)
    manager.load_all_to_ram()

    baselines = ['Xavier_Glorot', 'He_Kaiming', 'LeCun', 'Orthogonal', 'LSUV', 'FAVI', 'Laor']

    results = {method: [] for method in baselines}
    results['GP_Rule'] = []

    print(f"\n--- COMMENCING MASSIVE PHASE B VALIDATION (Topology: {args.topology.upper()}) ---")

    for idx, did in enumerate(manager.dataset_cache.keys()):
        dataset, meta_features = manager.get_dataset(did)
        m_vals = meta_features.numpy()

        pc_eigen = m_vals[3]
        target_entropy = m_vals[4]

        # 1. Evaluate the GP Discovered Rule
        sigma_squared = rank_1_rule(pc_eigen, target_entropy)

        # NOTE: activation_name is frozen to 'smooth' (GELU) based on the Plan B strategy
        evaluator_gp = PyTorchEvaluator(dataset, sigma_squared=sigma_squared, activation_name='smooth',
                                        topology=args.topology, max_epochs=30)
        acc_gp, _ = evaluator_gp.evaluate_fitness()
        results['GP_Rule'].append(acc_gp)

        # 2. Evaluate All 7 Baselines
        for method in baselines:
            evaluator_base = PyTorchEvaluator(dataset, sigma_squared=1e-5, activation_name='smooth',
                                              topology=args.topology, max_epochs=30)
            apply_baseline_initialization(evaluator_base.model, method, dataset, m_vals)
            acc_base, _ = evaluator_base.evaluate_fitness()
            results[method].append(acc_base)

        print(
            f"[{idx + 1}/{len(manager.dataset_cache)}] Dataset {did} Processed -> GP: {acc_gp * 100:.1f}% | Kaiming: {results['He_Kaiming'][-1] * 100:.1f}% | Laor: {results['Laor'][-1] * 100:.1f}%")

    # --- JSON EXPORT BLOCK FOR MODULE 8 ---
    json_path = os.path.join(GEN_DIR, f"validation_results_{args.topology}.json")
    with open(json_path, "w") as f:
        json.dump(results, f)
    print(f"\nRaw results successfully exported to '{json_path}'. Run Module 8 to generate Boxplots!")


if __name__ == "__main__":
    main()