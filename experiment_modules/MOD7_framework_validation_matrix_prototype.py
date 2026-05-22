"""
Module 7: Framework Validation Matrix - **PROTOTYPE**.

A scaled-down derivative of MOD7_framework_validation_matrix.py used for
pre-flight pipeline validation BEFORE committing to the full production
validation sweep. The evaluation logic (FNNTrainer pool, baseline
initializers, sigma sanitisation, binomial significance testing) is
byte-identical to the production framework. Only three things differ:

    1. Scale
        Trial seeds:        [42]              (vs production's [42,43,44,45,46])
        Phase B datasets:   5 (first by id)   (vs production's 25)
        => 25x fewer trainings per (topology, activation) cell. For the
           full 18-pair sweep this is ~990 trainings (~10-20 minutes)
           vs production ~24,750 trainings (~3.5-7 hours).

    2. Output routing
        Writes JSON reports to
        ``experimental_results_analysis_visualizations/reports/MOD7_validation_matrix_prototype/``
        so prototype outputs are CLEANLY SEPARATED from the paper-grade
        production reports under ``MOD7_validation_matrix/``.

    3. Status / label strings
        Console output is prefixed "MOD7 PROTOTYPE" and JSON metadata
        carries ``"Prototype": true`` so the artifacts are unambiguous
        when reviewed months later.

Purpose:
    Verify the MOD5 -> MOD7 -> JSON pipeline produces well-formed
    validation reports in roughly 10-20 minutes, surfacing any contract
    breakage (rule compilation failures, dataset cache issues, baseline
    initializer errors) BEFORE the multi-hour production validation run.

Mathematical content:
    * Each per-trial training uses Adam (lr=5e-3), ``max_epochs=30``,
      balanced-accuracy early-stop at ``target_acc=0.85``.
    * Binomial p-value: one-tailed test of ``H_0: P(GP wins) = 0.5``
      versus ``H_1: P(GP wins) > 0.5`` via ``1 - F(wins-1; trials, 0.5)``.
      Note: at prototype scale (5 trials per cell) p-values are
      INDICATIVE only, not paper-grade. Use production scale for
      publication claims.
"""

from __future__ import annotations

import argparse
import json
import math
import operator
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deap import gp
from scipy.stats import binom
from sklearn.cluster import KMeans

from MOD3_pm_dataset_manager import CacheConfig, DatasetManager
from MOD4_pm_fnn_landscape import FNNTrainer

warnings.filterwarnings("ignore")


# =============================================================================
# 1. DYNAMIC PATHING & RUNTIME CONSTANTS
# =============================================================================

BASELINE_METHODS: Tuple[str, ...] = (
    "Xavier_Glorot",
    "He_Kaiming",
    "LeCun",
    "Orthogonal",
    "FAVI",
    "Laor",
)

# Prototype-scoped scale: always used by this file, NOT user-configurable.
# (Production framework has --quick_test for the same purpose, but here the
# prototype scale is the *only* mode.)
PROTOTYPE_SEEDS: List[int] = [42]
PROTOTYPE_DATASET_LIMIT: int = 5

# Kept for source-of-truth compatibility with the production framework's
# constants. The prototype does not use them but importing tests / future
# tooling may still rely on these symbols being present.
QUICK_TEST_SEED: List[int] = [42]
QUICK_TEST_DATASET_LIMIT: int = 5
FULL_TRIAL_SEEDS: List[int] = [42, 43, 44, 45, 46]


def get_validation_reports_directory() -> Path:
    """Returns the absolute path of the MOD7 PROTOTYPE reports subfolder.

    Returns:
        Path under ``experimental_results_analysis_visualizations/reports/MOD7_validation_matrix_prototype/``.
        Strict separation from the production ``MOD7_validation_matrix/``
        directory prevents prototype-scale JSONs from being mistaken for
        paper-grade artifacts.
    """
    earv_dir = (
        Path(__file__).resolve().parent
        / "experimental_results_analysis_visualizations"
    )
    reports_dir = earv_dir / "reports" / "MOD7_validation_matrix_prototype"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


# =============================================================================
# 2. PROTECTED MATHEMATICAL OPERATORS (must match MOD6's primitive set)
# =============================================================================

def protected_div(left: float, right: float) -> float:
    """Division with denominator floor at 1e-5."""
    return left / right if abs(right) > 1e-5 else 1.0


def protected_sqrt(x: float) -> float:
    """Square root of absolute value."""
    return math.sqrt(abs(x))


def protected_log(x: float) -> float:
    """Log of absolute value; returns 0 on near-zero input."""
    return math.log(abs(x)) if abs(x) > 1e-5 else 0.0


def protected_exp(x: float) -> float:
    """Exp with input clipped to [-10, 10] to prevent overflow."""
    return math.exp(float(np.clip(x, -10.0, 10.0)))


def sanitize_sigma_squared(value: float) -> float:
    """Ensures variance is finite and non-negative.

    Raises:
        ValueError: If ``value`` is non-finite.
    """
    sigma_squared = float(value)
    if not np.isfinite(sigma_squared):
        raise ValueError("sigma_squared must be finite.")
    return abs(sigma_squared)


def compile_gp_rule(rule_string: str) -> Callable[..., float]:
    """Compiles a GP rule string into a callable mapping meta-features → variance.

    Args:
        rule_string: The DEAP-formatted equation, e.g.
            ``"sin(protected_log(mul(hopkins, iqr_dev)))"``.

    Returns:
        Callable that takes the 8 meta-features (positional) and returns a
        float variance estimate.

    Raises:
        deap.gp.PrimitiveTree.from_string errors on malformed strings.
    """
    pset = gp.PrimitiveSet("MAIN", 8)
    pset.renameArguments(
        ARG0="n_d_ratio", ARG1="feat_kurtosis", ARG2="iqr_dev", ARG3="pc_eigen",
        ARG4="target_entropy", ARG5="hopkins", ARG6="silhouette", ARG7="davies_bouldin",
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
    """Resets the full RNG stack for cross-trial reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_binomial_p_value(wins: int, total_trials: int) -> float:
    """One-tailed binomial p-value for H_0: P(win) = 0.5 vs H_1: P(win) > 0.5.

    Args:
        wins: Number of trials in which the GP rule beat the baseline.
        total_trials: Total trial count.

    Returns:
        p-value in [0, 1]. Returns 1.0 if either input is zero (no signal).

    Mathematical Notes:
        p = P(X >= wins | X ~ Binomial(total_trials, 0.5))
          = 1 - F(wins - 1; total_trials, 0.5)
    """
    if total_trials == 0 or wins == 0:
        return 1.0
    return float(1.0 - binom.cdf(wins - 1, total_trials, 0.5))


# =============================================================================
# 3. BASELINE INITIALIZATION SCHEMES
# =============================================================================

def apply_baseline_initialization(
    model: nn.Module,
    method: str,
    dataset: Dict[str, torch.Tensor],
    m_vals: np.ndarray,
) -> None:
    """Overwrites all Linear-layer weights in ``model`` per the named scheme.

    Args:
        model: An nn.Module containing one or more ``nn.Linear`` layers.
        method: One of ``BASELINE_METHODS``.
        dataset: ``{"X_train": ..., ...}`` — needed for the data-aware ``Laor`` scheme.
        m_vals: 8-vector of meta-features — needed for ``FAVI`` scaling.

    Returns:
        None. Modifies ``model`` parameters in place.

    Mathematical Notes:
        * Xavier_Glorot: W ~ N(0, sqrt(2/(fan_in + fan_out)))
        * He_Kaiming:    W ~ N(0, sqrt(2/fan_in))             [ReLU regime]
        * LeCun:         W ~ N(0, sqrt(1/fan_in))
        * Orthogonal:    QR decomposition of a Gaussian matrix
        * FAVI:          W ~ N(0, sqrt(2/(fan_in+fan_out)) * (1 + iqr_dev))
        * Laor:          first layer = KMeans cluster centers of X_train;
                         deeper layers fall back to He_Kaiming.
    """
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    X_train = dataset["X_train"]

    with torch.no_grad():
        if method == "Xavier_Glorot":
            for layer in linear_layers:
                nn.init.xavier_normal_(layer.weight)
        elif method == "He_Kaiming":
            for layer in linear_layers:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif method == "LeCun":
            for layer in linear_layers:
                std = math.sqrt(1.0 / float(layer.weight.size(1)))
                nn.init.normal_(layer.weight, mean=0.0, std=std)
        elif method == "Orthogonal":
            for layer in linear_layers:
                nn.init.orthogonal_(layer.weight)
        elif method == "FAVI":
            iqr_dev = float(m_vals[2]) if len(m_vals) > 2 else 0.0
            for layer in linear_layers:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                std = math.sqrt(2.0 / (fan_in + fan_out)) * (1.0 + iqr_dev)
                nn.init.normal_(layer.weight, mean=0.0, std=std)
        elif method == "Laor":
            first_layer = linear_layers[0]
            n_neurons = first_layer.weight.size(0)
            n_clusters = min(n_neurons, int(X_train.shape[0]))
            try:
                kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42).fit(
                    X_train.detach().cpu().numpy()
                )
                centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
                if n_clusters < n_neurons:
                    padding = torch.randn((n_neurons - n_clusters, int(X_train.shape[1])))
                    centers = torch.cat([centers, padding], dim=0)
                first_layer.weight.copy_(centers)
                for layer in linear_layers[1:]:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            except Exception:
                # KMeans can fail on degenerate data; fall back to Kaiming.
                for layer in linear_layers:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        else:
            raise ValueError(f"Unknown baseline initialization method: {method!r}")

        for layer in linear_layers:
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


# =============================================================================
# 4. POOLED TRAINER UTILITIES
# =============================================================================

def _get_or_build_trainer(
    pool: Dict[int, FNNTrainer],
    manager: DatasetManager,
    did: int,
    activation: str,
    topology: str,
) -> Tuple[FNNTrainer, np.ndarray]:
    """Returns a cached ``FNNTrainer`` for ``did``, building on first miss.

    Args:
        pool: Mutable mapping keyed by dataset id.
        manager: DatasetManager exposing ``get_dataset(did)``.
        did: Dataset id.
        activation: Activation token (binds the model topology).
        topology: 'shallow' | 'deep_narrow' | 'funnel'.

    Returns:
        Tuple ``(trainer, meta_feature_vector)``.
    """
    tensors, meta_tensor = manager.get_dataset(did)
    m_vals = meta_tensor.detach().cpu().numpy()

    trainer = pool.get(did)
    if trainer is None:
        trainer = FNNTrainer(
            dataset_dict=tensors,
            activation_name=activation,
            topology=topology,
        )
        pool[did] = trainer
    return trainer, m_vals


def _compute_validation_loss(trainer: FNNTrainer) -> float:
    """One forward pass on the validation set; returns the CE loss.

    Args:
        trainer: A trained ``FNNTrainer`` (post ``.evaluate()`` call).

    Returns:
        Cross-entropy loss as a Python float.
    """
    trainer.model.eval()
    with torch.no_grad():
        val_logits = trainer.model(trainer.X_val)
        loss = F.cross_entropy(val_logits, trainer.y_val)
    return float(loss.item())


# =============================================================================
# 5. CORE EVALUATION LOOP
# =============================================================================

def _evaluate_gp_rule_on_trainer(
    trainer: FNNTrainer, sigma_squared: float,
) -> Tuple[float, float, float]:
    """GP-rule evaluation: reset weights to N(0, sqrt(sigma²)), train, return (acc, epochs, loss)."""
    trainer.reset_weights(sigma_squared)
    acc, epochs = trainer.evaluate()
    if epochs >= 999.0:
        # NaN-collapse sentinel; loss is meaningless here.
        return 0.0, 999.0, 999.0
    loss = _compute_validation_loss(trainer)
    return float(acc), float(epochs), float(loss)


def _evaluate_baseline_on_trainer(
    trainer: FNNTrainer,
    method: str,
    dataset: Dict[str, torch.Tensor],
    m_vals: np.ndarray,
) -> Tuple[float, float, float]:
    """Baseline evaluation: apply named init scheme, train, return (acc, epochs, loss)."""
    apply_baseline_initialization(trainer.model, method, dataset, m_vals)
    acc, epochs = trainer.evaluate()
    if epochs >= 999.0:
        return 0.0, 999.0, 999.0
    loss = _compute_validation_loss(trainer)
    return float(acc), float(epochs), float(loss)


def _select_phase_b_ids(
    manager: DatasetManager, quick_test: bool = True,
) -> List[int]:
    """Returns the dataset ids over which the prototype validation sweep runs.

    The prototype ALWAYS uses the first ``PROTOTYPE_DATASET_LIMIT`` dataset
    ids — the ``quick_test`` parameter is retained for signature parity with
    the production framework but its value is ignored.

    Args:
        manager: A loaded ``DatasetManager``.
        quick_test: Retained for API parity, ignored in the prototype.

    Returns:
        First ``PROTOTYPE_DATASET_LIMIT`` Phase B dataset ids.
    """
    all_ids = list(manager.dataset_cache.keys())
    return all_ids[:PROTOTYPE_DATASET_LIMIT]


# =============================================================================
# 6. MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """CLI entry point. Parses args, runs the PROTOTYPE validation matrix, writes JSON.

    The prototype always uses ``PROTOTYPE_SEEDS=[42]`` and the first
    ``PROTOTYPE_DATASET_LIMIT=5`` Phase B datasets. There is no
    ``--quick_test`` flag because the prototype IS the quick test.
    """
    parser = argparse.ArgumentParser(
        description="MOD7 PROTOTYPE: Framework Validation Matrix (scaled smoke test, CPU)."
    )
    parser.add_argument(
        "--topology", type=str, required=True,
        choices=["shallow", "deep_narrow", "funnel"],
        help="FNN topology under test.",
    )
    parser.add_argument(
        "--activation", type=str, required=True,
        help="Activation function name (one of the 6 canonical tokens).",
    )
    parser.add_argument(
        "--rule_strs", type=str, nargs="+", required=True,
        help="Up to 5 DEAP-format GP rule strings (top Pareto front).",
    )
    args = parser.parse_args()

    rule_strings = args.rule_strs[:5]
    reports_dir = get_validation_reports_directory()

    trial_seeds = PROTOTYPE_SEEDS

    print(f"\n--- STATISTICAL VALIDATION MATRIX [PROTOTYPE / {args.activation.upper()}] ---")
    print("MODE: PROTOTYPE (always quick: 1 seed x 5 Phase B datasets)")
    print(f"Compiling {len(rule_strings)} GP Rules for cross-validation sweep...")
    gp_funcs = [compile_gp_rule(r) for r in rule_strings]

    # Phase B dataset bench
    manager = DatasetManager(CacheConfig(phase_csv_name="Phase_B_Validation_Datasets.csv"))
    manager.load_all_to_ram()
    dataset_ids = _select_phase_b_ids(manager)
    print(f"Phase B dataset count for this run: {len(dataset_ids)} | Seeds: {trial_seeds}")

    # Per-dataset trainer pool (lifetime = this CLI invocation)
    trainer_pool: Dict[int, FNNTrainer] = {}

    # Trial-metric accumulators
    rule_keys = [f"GP_Rule_{i+1}" for i in range(len(gp_funcs))]
    trial_metrics: Dict[str, Dict[str, List[float]]] = {
        k: {"acc": [], "epochs": [], "loss": []} for k in rule_keys
    }
    for base in BASELINE_METHODS:
        trial_metrics[base] = {"acc": [], "epochs": [], "loss": []}

    win_matrix: Dict[str, Dict[str, int]] = {
        k: {base: 0 for base in BASELINE_METHODS} for k in rule_keys
    }

    total_trials = 0

    # Outer sweep: dataset × seed
    for did in dataset_ids:
        trainer, m_vals = _get_or_build_trainer(
            trainer_pool, manager, did, args.activation, args.topology,
        )
        tensors, _ = manager.get_dataset(did)

        for seed in trial_seeds:
            total_trials += 1
            current_gp_losses: List[float] = []

            # GP rules
            for i, func in enumerate(gp_funcs):
                try:
                    sigma = sanitize_sigma_squared(func(*m_vals))
                except Exception:
                    sigma = 1e-5

                seed_runtime(seed)
                acc, epochs, loss = _evaluate_gp_rule_on_trainer(trainer, sigma)
                trial_metrics[rule_keys[i]]["acc"].append(acc)
                trial_metrics[rule_keys[i]]["epochs"].append(epochs)
                trial_metrics[rule_keys[i]]["loss"].append(loss)
                current_gp_losses.append(loss)

            # Baselines
            for base in BASELINE_METHODS:
                seed_runtime(seed)
                acc, epochs, base_loss = _evaluate_baseline_on_trainer(
                    trainer, base, tensors, m_vals,
                )
                trial_metrics[base]["acc"].append(acc)
                trial_metrics[base]["epochs"].append(epochs)
                trial_metrics[base]["loss"].append(base_loss)

                for i, gp_loss in enumerate(current_gp_losses):
                    if gp_loss < base_loss:
                        win_matrix[rule_keys[i]][base] += 1

    # Aggregate metrics
    aggregates: Dict[str, Dict[str, float]] = {}
    for method, data in trial_metrics.items():
        aggregates[method] = {
            "Accuracy_Mean": float(np.mean(data["acc"])) if data["acc"] else 0.0,
            "Accuracy_StdDev": float(np.std(data["acc"])) if data["acc"] else 0.0,
            "Epochs_Mean": float(np.mean(data["epochs"])) if data["epochs"] else 0.0,
            "Epochs_StdDev": float(np.std(data["epochs"])) if data["epochs"] else 0.0,
            "Loss_Mean": float(np.mean(data["loss"])) if data["loss"] else 0.0,
            "Loss_StdDev": float(np.std(data["loss"])) if data["loss"] else 0.0,
        }

    # Build per-rule binomial p-values (INDICATIVE only at prototype scale)
    p_values: Dict[str, Dict[str, float]] = {k: {} for k in rule_keys}
    print(f"\nCompleted {total_trials} independent trials per (rule, baseline) cell.")
    print("NOTE: p-values at prototype scale are INDICATIVE only "
          "(use production framework for paper-grade significance).")
    for rule_key, comparisons in win_matrix.items():
        print(f"\n{rule_key} Performance (Cross-Entropy Loss vs Baselines):")
        for base, wins in comparisons.items():
            p_val = calculate_binomial_p_value(wins, total_trials)
            p_values[rule_key][base] = float(p_val)
            sig = "***" if p_val < 0.05 else ""
            print(f"  vs {base}: {wins}/{total_trials} wins (p={p_val:.4f}) {sig}")

    # Assemble + write JSON
    final_report = {
        "Metadata": {
            "Topology": args.topology,
            "Activation": args.activation,
            "Total_Trials_Per_Method": total_trials,
            "Prototype": True,
            "Quick_Test_Mode": True,  # for backward-compat with MOD8 ingest
            "Rules_Evaluated": rule_strings,
            "Baseline_Methods": list(BASELINE_METHODS),
        },
        "Aggregates": aggregates,
        "Binomial_Win_Matrix_By_Loss": win_matrix,
        "Binomial_P_Values_By_Loss": p_values,
        "Raw_Distributions": trial_metrics,
    }

    json_path = reports_dir / (
        f"statistical_validation_prototype_{args.topology}_{args.activation}.json"
    )
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(final_report, fh, indent=4)
    print(f"\nPrototype report exported to {json_path}")

    # Release pool explicitly so subprocess exit closes promptly.
    trainer_pool.clear()


if __name__ == "__main__":
    main()