"""
Module 5: Optimization Method (OM) - MOGP Meta-Grid Search Engine Prototype.

Executes the genetic programming outer-loop across the 6 canonical activation
functions. Explicitly CPU-routed for Phase A (Shallow FNN).

Methodological configuration (production):
    * Population: 150
    * Generations: 15
    * Datasets evaluated per rule: 20
    * Max epochs per inner training: 30
    * Cumulative PyTorch CPU evaluations: ~244,800

Architectural notes versus the MOD5 prototype:
    * The inner FNN trainer is now pooled per (dataset_id, activation) via
      ``FNNTrainer.reset_weights(...)`` — model construction, DataLoader
      instantiation, and validation-tensor binding amortize across the
      entire GA run rather than re-occurring per individual.
    * Generation 0 has its own tqdm bar so the user sees progress
      immediately. Each subsequent generation has its own bar nested below.
    * Per-batch parameter NaN/Inf scans replaced with per-epoch checks.
    * ``torch.set_num_threads`` / ``set_num_interop_threads`` are pinned at
      startup to avoid OMP contention on shallow CPU gemms.
"""

from __future__ import annotations

import datetime
import math
import operator
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import deap
import numpy as np
import torch
from deap import algorithms, base, creator, gp, tools
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

from MOD3_pm_dataset_manager import CacheConfig, DatasetManager
from MOD4_pm_fnn_landscape import FNNTrainer

warnings.filterwarnings("ignore")


# =============================================================================
# 1. RUNTIME CONFIGURATION & SYSTEM LOGGING
# =============================================================================

class MOGPConfig(BaseModel):
    """Runtime configuration validated via Pydantic.

    Mathematical Notes:
        * cxpb + mutpb <= 1.0 for ``algorithms.varOr`` (DEAP requires that
          their sum bound the probability of any offspring being modified).
    """

    # --- Outer GA dimensions ---
    population_size: int = Field(default=20, gt=0)
    generations: int = Field(default=10, gt=0)
    datasets_per_rule: int = Field(default=3, gt=0)
    max_epochs: int = Field(default=30, gt=0)
    batch_size: int = Field(default=64, gt=0)

    # --- Variation operator probabilities ---
    crossover_probability: float = Field(default=0.6, ge=0.0, le=1.0)
    mutation_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tree_height: int = Field(default=6, gt=0)

    # --- Topology + reproducibility ---
    topology: str = Field(default="shallow")
    random_seed: int = Field(default=42)
    target_acc: float = Field(default=0.85, ge=0.0, le=1.0)
    activation_functions: List[str] = Field(
        default=["rectification", "squashing", "smooth", "aggregation", "trigonometric", "linear"]
    )

    # --- CPU thread pinning ---
    num_threads: int = Field(
        default=4, ge=1,
        description="torch.set_num_threads value. 4 is a good default for shallow MLPs on 6c/12t.",
    )

    # --- Checkpointing ---
    checkpoint_every_n_gen: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "MOGPConfig":
        if self.crossover_probability + self.mutation_probability > 1.0:
            raise ValueError("Crossover + Mutation probability must be <= 1.0")
        return self


def get_system_environment(config: MOGPConfig) -> str:
    """Captures hardware and software dependencies for reproducibility."""
    import platform
    import sklearn

    details = [
        "--- SYSTEM & ENVIRONMENT LOG ---",
        f"Python Version: {sys.version.split(' ')[0]}",
        f"PyTorch Version: {torch.__version__}",
        f"DEAP Version: {deap.__version__}",
        f"Scikit-Learn Version: {sklearn.__version__}",
        f"OS Platform: {platform.platform()}",
        f"Hardware: {platform.machine()} / {platform.processor()}",
        "Execution Target: CPU-ONLY (Phase A shallow FNN)",
        f"PyTorch Threads -> intra_op: {torch.get_num_threads()} | inter_op: {torch.get_num_interop_threads()}",
        f"Random Seed: {config.random_seed}",
    ]
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        details.append(f"Git Commit: {commit_hash}")
    except Exception:
        details.append("Git Commit: N/A (Not a valid git repository)")
    details.append("--------------------------------")
    return "\n".join(details)


def get_generated_directory() -> Path:
    """Resolves the output directory relative to this script."""
    module_dir = Path(__file__).resolve().parent
    generated_dir = module_dir / "generated_files" / "GA_rule_files"
    generated_dir.mkdir(parents=True, exist_ok=True)
    return generated_dir


# =============================================================================
# 2. PROTECTED MATHEMATICAL OPERATORS
# =============================================================================

def protected_div(left: float, right: float) -> float:
    """Division with denominator floor at 1e-5 (returns 1.0 on near-zero)."""
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


# =============================================================================
# 3. DEAP TOOLBOX CONSTRUCTION
# =============================================================================

def build_primitive_set() -> gp.PrimitiveSet:
    """Builds the symbolic search space mapping meta-features -> variance."""
    primitive_set = gp.PrimitiveSet("MAIN", 8)
    primitive_set.renameArguments(
        ARG0="n_d_ratio", ARG1="feat_kurtosis", ARG2="iqr_dev", ARG3="pc_eigen",
        ARG4="target_entropy", ARG5="hopkins", ARG6="silhouette", ARG7="davies_bouldin",
    )
    primitive_set.addPrimitive(operator.add, 2)
    primitive_set.addPrimitive(operator.sub, 2)
    primitive_set.addPrimitive(operator.mul, 2)
    primitive_set.addPrimitive(protected_div, 2)
    primitive_set.addPrimitive(operator.neg, 1)
    primitive_set.addPrimitive(math.sin, 1)
    primitive_set.addPrimitive(math.cos, 1)
    primitive_set.addPrimitive(protected_sqrt, 1)
    primitive_set.addPrimitive(protected_log, 1)
    primitive_set.addPrimitive(protected_exp, 1)
    primitive_set.addEphemeralConstant("rand101", lambda: random.uniform(-1.0, 1.0))
    return primitive_set


def ensure_deap_creators() -> None:
    """Initializes DEAP's global multi-objective fitness schema."""
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)


def build_toolbox(primitive_set: gp.PrimitiveSet, config: MOGPConfig) -> base.Toolbox:
    """Constructs the DEAP toolbox with NSGA-II selection and bounded GP variation."""
    ensure_deap_creators()
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=primitive_set, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primitive_set)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitive_set)
    limit_decorator = gp.staticLimit(key=operator.attrgetter("height"), max_value=config.max_tree_height)
    toolbox.decorate("mate", limit_decorator)
    toolbox.decorate("mutate", limit_decorator)
    return toolbox


# =============================================================================
# 4. INNER EVALUATION WRAPPER  (pool-aware)
# =============================================================================

def get_phase_a_dataset_ids(manager: DatasetManager, datasets_per_rule: int) -> List[int]:
    """Deterministically slice the first ``datasets_per_rule`` dataset ids.

    Raises:
        ValueError: If the cache holds fewer datasets than requested.
    """
    dataset_ids = list(manager.dataset_cache.keys())
    if len(dataset_ids) < datasets_per_rule:
        raise ValueError(
            f"Required {datasets_per_rule} datasets, but cache holds {len(dataset_ids)}."
        )
    return dataset_ids[:datasets_per_rule]


def _get_or_build_trainer(
    pool: Dict[int, FNNTrainer],
    manager: DatasetManager,
    did: int,
    activation: str,
    config: MOGPConfig,
) -> Tuple[FNNTrainer, torch.Tensor]:
    """Returns a cached ``FNNTrainer`` for ``did``, building on first miss.

    Args:
        pool: Mutable mapping keyed by dataset id.
        manager: Provides ``get_dataset(did) -> (tensors_dict, meta_features)``.
        did: Dataset id.
        activation: Activation token (binds the model topology).
        config: Runtime configuration.

    Returns:
        Tuple ``(trainer, meta_features)``. Meta features are fetched on
        every call because they may be lazily materialized by the manager.
    """
    tensors, meta_features = manager.get_dataset(did)
    trainer = pool.get(did)
    if trainer is None:
        trainer = FNNTrainer(
            dataset_dict=tensors,
            activation_name=activation,
            topology=config.topology,
            max_epochs=config.max_epochs,
            target_acc=config.target_acc,
            batch_size=config.batch_size,
        )
        pool[did] = trainer
    return trainer, meta_features


def evaluate_rule(
    individual: gp.PrimitiveTree,
    manager: DatasetManager,
    activation: str,
    toolbox: base.Toolbox,
    config: MOGPConfig,
    trainer_pool: Dict[int, FNNTrainer],
) -> Tuple[float, float, int]:
    """Compiles a GP tree and evaluates it across the dataset bench.

    Args:
        individual: DEAP ``PrimitiveTree`` representing the candidate
            variance rule.
        manager: ``DatasetManager`` exposing ``get_dataset(did)``.
        activation: Current activation function name.
        toolbox: DEAP toolbox (used for ``compile``).
        config: Runtime configuration.
        trainer_pool: Per-(activation, dataset_id) pool of stateful
            ``FNNTrainer`` instances. Mutated on first miss for each
            ``did``.

    Returns:
        Tuple ``(avg_balanced_acc, avg_epochs_to_threshold, tree_size)``.
        On any compile-time exception or NaN gradient observed, returns
        ``(0.0, 999.0, tree_size)`` as the GP penalty sentinel.
    """
    rule_func = toolbox.compile(expr=individual)
    dataset_ids = get_phase_a_dataset_ids(manager, config.datasets_per_rule)

    total_acc = 0.0
    total_epochs = 0.0
    tree_size = len(individual)

    for did in dataset_ids:
        trainer, meta_features = _get_or_build_trainer(
            trainer_pool, manager, did, activation, config
        )
        m_vals = meta_features.detach().cpu().numpy()

        try:
            sigma_squared = sanitize_sigma_squared(rule_func(*m_vals))
        except Exception:
            return 0.0, 999.0, tree_size

        # Reproducibility seed: pins randperm shuffle and weight init.
        torch.manual_seed(config.random_seed + did)
        trainer.reset_weights(sigma_squared)

        acc, epochs = trainer.evaluate()

        if epochs == 999 or acc == 0.0:
            return 0.0, 999.0, tree_size

        total_acc += float(acc)
        total_epochs += float(epochs)

    n = float(len(dataset_ids))
    return total_acc / n, total_epochs / n, tree_size


# =============================================================================
# 5. ARTIFACT EXPORT & CHECKPOINTING
# =============================================================================

def export_pareto_rules(
    hall_of_fame: tools.ParetoFront, activation: str, config: MOGPConfig,
    output_dir: Path, env_details: str, top_k: int = 10,
    is_checkpoint: bool = False, gen: int = 0,
) -> Path:
    """Writes structured Pareto-front rules to disk with fsync barrier."""
    activation_token = "".join(c if c.isalnum() else "_" for c in activation.lower())
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if is_checkpoint:
        filename = f"Checkpoint_{activation_token}_Gen{gen}.txt"
    else:
        filename = f"Final_Discovered_Rules_{activation_token}_{timestamp}.txt"
    output_path = output_dir / filename

    with output_path.open("w", encoding="utf-8") as file:
        file.write("APPLIED COMPUTATIONAL INTELLIGENCE: MODULE 6 RESULTS\n")
        file.write(f"Activation Function: {activation.upper()}\n")
        file.write(
            f"Status: {'CHECKPOINT (Generation ' + str(gen) + ')' if is_checkpoint else 'FINAL COMPLETE RUN'}\n"
        )
        file.write("=" * 72 + "\n")
        file.write(env_details + "\n")
        file.write("=" * 72 + "\n")
        file.write(f"Population: {config.population_size} | Generations: {config.generations}\n")
        file.write(
            f"Datasets/Rule: {config.datasets_per_rule} | Batch: {config.batch_size} | Topology: {config.topology}\n"
        )
        file.write("=" * 72 + "\n\n")
        file.write("Top Discovered Rules-of-Thumb (Pareto Optimal):\n\n")

        if len(hall_of_fame) == 0:
            file.write("No Pareto-optimal rules discovered yet.\n")
        else:
            for rank, ind in enumerate(hall_of_fame[:top_k], start=1):
                file.write(f"Rank {rank}:\nEquation: {str(ind)}\n")
                file.write(
                    f"Fitness: [Acc: {ind.fitness.values[0]:.4f}, "
                    f"Epochs: {ind.fitness.values[1]:.2f}, "
                    f"Bloat: {ind.fitness.values[2]}]\n\n"
                )
        file.flush()
        os.fsync(file.fileno())

    return output_path


# =============================================================================
# 6. CUSTOM EVOLUTIONARY LOOP  (with visible Gen-0 progress)
# =============================================================================

def _evaluate_invalid(
    invalid_ind: List[gp.PrimitiveTree],
    toolbox: base.Toolbox,
    desc: str,
) -> None:
    """Evaluates and assigns fitness to ``invalid_ind`` with a live progress bar.

    Mutates ``invalid_ind`` in place by setting ``fitness.values``.
    """
    with tqdm(total=len(invalid_ind), desc=desc, unit="ind", leave=False) as bar:
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
            bar.update(1)


def custom_eaMuPlusLambda(
    population: List, toolbox: base.Toolbox, config: MOGPConfig,
    halloffame: tools.ParetoFront, activation: str,
    output_dir: Path, env_details: str,
) -> None:
    """Executes the EA block with generation-level checkpointing."""
    # ---- Generation 0 -----------------------------------------------------
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    _evaluate_invalid(invalid_ind, toolbox, desc=f"[{activation.upper()}] Gen 0 init")

    if halloffame is not None:
        halloffame.update(population)
    export_pareto_rules(
        halloffame, activation, config, output_dir, env_details,
        is_checkpoint=True, gen=0,
    )

    # ---- Generations 1..G -------------------------------------------------
    with tqdm(total=config.generations,
              desc=f"[{activation.upper()}] Evolutions", unit="gen") as pbar:
        for gen in range(1, config.generations + 1):
            offspring = algorithms.varOr(
                population, toolbox, config.population_size,
                config.crossover_probability, config.mutation_probability,
            )
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            _evaluate_invalid(
                invalid_ind, toolbox,
                desc=f"[{activation.upper()}] Gen {gen} offspring",
            )

            if halloffame is not None:
                halloffame.update(offspring)

            population[:] = toolbox.select(
                population + offspring, config.population_size
            )

            if gen % config.checkpoint_every_n_gen == 0 or gen == config.generations:
                export_pareto_rules(
                    halloffame, activation, config, output_dir, env_details,
                    is_checkpoint=True, gen=gen,
                )
            pbar.update(1)


# =============================================================================
# 7. ORCHESTRATION ENGINE
# =============================================================================

def run_gp_for_activation(
    activation: str, manager: DatasetManager, toolbox: base.Toolbox,
    config: MOGPConfig, output_dir: Path, env_details: str,
) -> Path:
    """Runs one full GP cycle for a single activation.

    The trainer pool is local to this function call — its lifetime exactly
    matches the GA cycle for ``activation``. Garbage-collected on return.

    Returns:
        Path to the final exported Pareto-front artifact.
    """
    trainer_pool: Dict[int, FNNTrainer] = {}

    # Late binding: re-register evaluate with this activation's trainer pool.
    toolbox.register(
        "evaluate", evaluate_rule,
        manager=manager, activation=activation, toolbox=toolbox,
        config=config, trainer_pool=trainer_pool,
    )

    population = toolbox.population(n=config.population_size)
    hall_of_fame = tools.ParetoFront()

    custom_eaMuPlusLambda(
        population, toolbox, config, hall_of_fame, activation,
        output_dir, env_details,
    )

    final_path = export_pareto_rules(
        hall_of_fame=hall_of_fame, activation=activation, config=config,
        output_dir=output_dir, env_details=env_details, top_k=10,
        is_checkpoint=False,
    )

    # Explicit pool teardown — releases ~20 FNN model graphs per activation.
    trainer_pool.clear()
    return final_path


def seed_runtime(seed: int) -> None:
    """Reseeds Python and NumPy RNGs. PyTorch is reseeded per-evaluation."""
    random.seed(seed)
    np.random.seed(seed)


def configure_torch_threads(config: MOGPConfig) -> None:
    """Pins PyTorch thread counts before any tensor ops execute.

    Mathematical Notes:
        For shallow MLPs the gemm sizes (~64x64) sit below the threshold
        where intra-op parallelism beats single-thread vector code. We cap
        intra_op at ``config.num_threads`` and pin inter_op to 1 to avoid
        OMP-vs-thread-pool oversubscription on Windows.
    """
    torch.set_num_threads(min(config.num_threads, os.cpu_count() or config.num_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # set_num_interop_threads can only be called before parallel work starts.
        pass


def main() -> None:
    """Top-level entry point for the MOGP meta-grid search."""
    config = MOGPConfig()
    configure_torch_threads(config)
    env_details = get_system_environment(config)
    output_dir = get_generated_directory()

    print("Loading Phase A Dataset Cache (Pre-Scaling to RAM)...")
    manager_config = CacheConfig()
    manager = DatasetManager(manager_config)
    manager.load_all_to_ram()

    primitive_set = build_primitive_set()
    toolbox = build_toolbox(primitive_set, config)

    print("\n--- Commencing Module 6 MOGP Meta-Grid Search ---")
    print(env_details)
    print(f"Target Checkpoint / Export Directory: {output_dir}")
    print(f"Activation Functions: {', '.join(config.activation_functions)}")

    total_evals = (
        config.population_size
        * config.generations
        * config.datasets_per_rule
        * len(config.activation_functions)
    )
    print(f"Approx. Cumulative PyTorch CPU Evaluations: {total_evals:,}")

    exported_files: List[Path] = []
    for activation in config.activation_functions:
        seed_runtime(config.random_seed)
        output_path = run_gp_for_activation(
            activation=activation, manager=manager, toolbox=toolbox,
            config=config, output_dir=output_dir, env_details=env_details,
        )
        exported_files.append(output_path)

    print("\n--- MODULE 6 MOGP META-GRID SEARCH COMPLETE ---")
    print("Final Discovered Rule Artifacts:")
    for path in exported_files:
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
