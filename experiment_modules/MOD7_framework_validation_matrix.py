"""
Module 7: Framework Validation Matrix (Hardware Accelerated)

Evaluates top GP rules against SOTA baselines using stratified trials.
Routes final JSON output matrices explicitly to the /reports sub-namespace
within the experimental results directory.
"""

import math
import json
import argparse
import random
import operator
import warnings
from pathlib import Path
from typing import Dict, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from deap import gp
from scipy.stats import binom

from MOD3_pm_dataset_manager import CacheConfig, DatasetManager
from MOD4_pm_fnn_landscape import PyTorchEvaluator

warnings.filterwarnings("ignore")

# =============================================================================
# 1. DYNAMIC PATHING & CONFIGURATION
# =============================================================================

def get_reports_directory() -> Path:
    """Returns the dynamically resolved /reports namespace directory."""
    reports_dir = Path(__file__).resolve().parent / "generated_files" / "experimental_results_analysis_visualizations" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir

# =============================================================================
# 2. PROTECTED MATHEMATICAL OPERATORS (MOGP COMPILER DEPS)
# =============================================================================

def protected_div(left: float, right: float) -> float:
    return left / right if abs(right) > 1e-5 else 1.0

def protected_sqrt(x: float) -> float:
    return math.sqrt(abs(x))

def protected_log(x: float) -> float:
    return math.log(abs(x)) if abs(x) > 1e-5 else 0.0

def protected_exp(x: float) -> float:
    return math.exp(float(np.clip(x, -10.0, 10.0)))

def sanitize_sigma_squared(value: float) -> float:
    sigma_squared = float(value)
    if not np.isfinite(sigma_squared):
        raise ValueError("sigma_squared must be finite.")
    return abs(sigma_squared)

def compile_gp_rule(rule_string: str) -> Callable:
    pset = gp.PrimitiveSet("MAIN", 8)
    pset.renameArguments(
        ARG0="n_d_ratio", ARG1="feat_kurtosis", ARG2="iqr_dev", ARG3="pc_eigen",
        ARG4="target_entropy", ARG5="hopkins", ARG6="silhouette", ARG7="davies_bouldin"
    )
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(protected_log, 1)
    pset.addPrimitive(protected_exp, 1)
    pset.addEphemeralConstant("rand101", lambda: random.uniform(-1.0, 1.0))

    tree = gp.PrimitiveTree.from_string(rule_string, pset)
    return gp.compile(tree, pset)

def seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_binomial_p_value(wins: int, total_trials: int) -> float:
    if total_trials == 0 or wins == 0:
        return 1.0
    return 1.0 - binom.cdf(wins - 1, total_trials, 0.5)

def apply_baseline_initialization(model: nn.Module, method: str, dataset: Dict[str, torch.Tensor], m_vals: np.ndarray) -> None:
    X_train = dataset['X_train']
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    device = next(model.parameters()).device

    with torch.no_grad():
        if method == 'Xavier_Glorot':
            for layer in linear_layers: nn.init.xavier_normal_(layer.weight)
        elif method == 'He_Kaiming':
            for layer in linear_layers: nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif method == 'LeCun':
            for layer in linear_layers: nn.init.normal_(layer.weight, mean=0., std=math.sqrt(1.0 / layer.weight.size(1)))
        elif method == 'Orthogonal':
            for layer in linear_layers: nn.init.orthogonal_(layer.weight)
        elif method == 'FAVI':
            for layer in linear_layers:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                nn.init.normal_(layer.weight, mean=0., std=math.sqrt(2.0 / (fan_in + fan_out)) * (1.0 + m_vals[2]))
        elif method == 'Laor':
            first_layer = linear_layers[0]
            clusters = min(first_layer.weight.size(0), X_train.shape[0])
            try:
                kmeans = KMeans(n_clusters=clusters, n_init=1, random_state=42).fit(X_train.numpy())
                centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
                if clusters < first_layer.weight.size(0):
                    centers = torch.cat([centers, torch.randn((first_layer.weight.size(0) - clusters, X_train.shape[1]))], dim=0)
                first_layer.weight.copy_(centers)
                for layer in linear_layers[1:]: nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            except:
                for layer in linear_layers: nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif method == 'LSUV':
            for layer in linear_layers: nn.init.orthogonal_(layer.weight)
            x = X_train[:128].to(device)
            for child in getattr(model, 'model', model.children()):
                x_prev = x
                x = child(x)
                if isinstance(child, nn.Linear) and child != linear_layers[-1]:
                    var = torch.var(x)
                    if var > 1e-6:
                        child.weight.data /= torch.sqrt(var)
                        x = child(x_prev)

        for layer in linear_layers:
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', type=str, required=True, choices=['shallow', 'deep_narrow', 'funnel'])
    parser.add_argument('--activation', type=str, required=True)
    parser.add_argument('--rule_strs', type=str, nargs='+', required=True)
    args = parser.parse_args()

    rule_strings = args.rule_strs[:5]
    reports_dir = get_reports_directory()

    print(f"\n--- STATISTICAL VALIDATION MATRIX [{args.activation.upper()}] ---")
    print(f"Compiling {len(rule_strings)} GP Rules for 125-Trial Sweep...")
    gp_funcs = [compile_gp_rule(r) for r in rule_strings]

    manager = DatasetManager(CacheConfig(phase_csv_name="Phase_B_Validation_Datasets.csv"))
    manager.load_all_to_ram()

    baselines = ['Xavier_Glorot', 'He_Kaiming', 'LeCun', 'Orthogonal', 'LSUV', 'FAVI', 'Laor']
    trial_seeds = [42, 43, 44, 45, 46]
    total_trials = 0

    trial_metrics = {f"GP_Rule_{i+1}": {"acc": [], "epochs": [], "loss": []} for i in range(len(gp_funcs))}
    for base in baselines:
        trial_metrics[base] = {"acc": [], "epochs": [], "loss": []}

    win_matrix = {f"GP_Rule_{i+1}": {base: 0 for base in baselines} for i in range(len(gp_funcs))}

    for did in manager.dataset_cache.keys():
        dataset, meta = manager.get_dataset(did)
        m_vals = meta.numpy()

        for seed in trial_seeds:
            total_trials += 1

            current_gp_losses = []
            for i, func in enumerate(gp_funcs):
                try: sigma = sanitize_sigma_squared(func(*m_vals))
                except: sigma = 1e-5

                seed_runtime(seed)
                evaluator = PyTorchEvaluator(dataset, sigma, args.activation, args.topology, pin_memory=True, use_amp=True, torch_compile=False)
                acc, epochs, loss = evaluator.evaluate_fitness(return_loss=True)

                trial_metrics[f"GP_Rule_{i+1}"]["acc"].append(acc)
                trial_metrics[f"GP_Rule_{i+1}"]["epochs"].append(epochs)
                trial_metrics[f"GP_Rule_{i+1}"]["loss"].append(loss)
                current_gp_losses.append(loss)

            for base in baselines:
                seed_runtime(seed)
                evaluator = PyTorchEvaluator(dataset, 1e-5, args.activation, args.topology, pin_memory=True, use_amp=True, torch_compile=False)
                apply_baseline_initialization(evaluator.model, base, dataset, m_vals)
                acc, epochs, base_loss = evaluator.evaluate_fitness(return_loss=True)

                trial_metrics[base]["acc"].append(acc)
                trial_metrics[base]["epochs"].append(epochs)
                trial_metrics[base]["loss"].append(base_loss)

                for i, gp_loss in enumerate(current_gp_losses):
                    if gp_loss < base_loss:
                        win_matrix[f"GP_Rule_{i+1}"][base] += 1

    aggregates = {}
    for method, data in trial_metrics.items():
        aggregates[method] = {
            "Accuracy_Mean": float(np.mean(data["acc"])),
            "Accuracy_StdDev": float(np.std(data["acc"])),
            "Epochs_Mean": float(np.mean(data["epochs"])),
            "Epochs_StdDev": float(np.std(data["epochs"])),
            "Loss_Mean": float(np.mean(data["loss"])),
            "Loss_StdDev": float(np.std(data["loss"]))
        }

    final_report = {
        "Metadata": {
            "Topology": args.topology,
            "Activation": args.activation,
            "Total_Trials_Per_Method": total_trials
        },
        "Aggregates": aggregates,
        "Binomial_Win_Matrix_By_Loss": win_matrix,
        "Binomial_P_Values_By_Loss": {rule: {} for rule in win_matrix.keys()},
        "Raw_Distributions": trial_metrics
    }

    print(f"\nCompleted {total_trials} independent trials.")
    for rule, comparisons in win_matrix.items():
        print(f"\n{rule} Performance (Cross-Entropy Loss vs Baselines):")
        for base, wins in comparisons.items():
            p_val = calculate_binomial_p_value(wins, total_trials)
            final_report["Binomial_P_Values_By_Loss"][rule][base] = float(p_val)
            sig = "***" if p_val < 0.05 else ""
            print(f"  vs {base}: {wins}/{total_trials} wins (p={p_val:.4f}) {sig}")

    json_path = reports_dir / f"statistical_validation_{args.topology}_{args.activation}.json"
    with json_path.open("w") as f:
        json.dump(final_report, f, indent=4)
    print(f"\nComprehensive Rigor Report exported to {json_path}")

if __name__ == "__main__":
    main()