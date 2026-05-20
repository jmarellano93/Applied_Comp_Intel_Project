"""
Module 5: Optimization Method (OM) - MOGP Meta-Grid Search Engine Prototype.

Executes a reduced genetic programming outer-loop across a small activation-function
subset. This module is an architectural mirror of Module 6, intentionally configured
for a lightweight demonstration of the final MOGP engine behavior. It includes
generation-level checkpointing, exact stochastic seed resetting, environment logging,
and real-time progress bars.

Methodological configuration (Prototype Scale):
    - Population: 20
    - Generations: 3
    - Datasets evaluated per rule: 3
    - Max FNN epochs: 5
    - Activations: 'rectification', 'linear'
"""

import datetime
import math
import operator
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import deap
from deap import algorithms, base, creator, gp, tools
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

from MOD3_pm_dataset_manager import CacheConfig, DatasetManager
from MOD4_pm_fnn_landscape import PyTorchEvaluator

warnings.filterwarnings("ignore")

# =============================================================================
# 1. RUNTIME CONFIGURATION & SYSTEM LOGGING
# =============================================================================

class MOGPConfig(BaseModel):
    """
    Runtime configuration for the Module 5 MOGP Prototype engine.
    Values are intentionally kept small for rapid validation.
    """
    population_size: int = Field(default=20, gt=0)
    generations: int = Field(default=3, gt=0)
    datasets_per_rule: int = Field(default=3, gt=0)
    max_epochs: int = Field(default=5, gt=0)
    batch_size: int = Field(default=64, gt=0)
    crossover_probability: float = Field(default=0.6, ge=0.0, le=1.0)
    mutation_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tree_height: int = Field(default=6, gt=0)
    topology: str = Field(default="shallow")
    random_seed: int = Field(default=42)
    activation_functions: List[str] = Field(
        default=["rectification", "linear"]
    )

    @model_validator(mode='after')
    def validate_probabilities(self) -> 'MOGPConfig':
        """Ensures crossover and mutation probabilities do not exceed absolute bounds."""
        if self.crossover_probability + self.mutation_probability > 1.0:
            raise ValueError("Crossover + Mutation probability must be <= 1.0")
        return self


def get_system_environment(config: MOGPConfig) -> str:
    """
    Captures exact hardware and software dependencies for auditability.

    Returns:
        str: Formatted multiline string detailing the environment state.
    """
    import sklearn
    import platform

    details = [
        "--- SYSTEM & ENVIRONMENT LOG ---",
        f"Python Version: {sys.version.split(' ')[0]}",
        f"PyTorch Version: {torch.__version__}",
        f"DEAP Version: {deap.__version__}",
        f"Scikit-Learn Version: {sklearn.__version__}",
        f"OS Platform: {platform.platform()}",
        f"Hardware: {platform.machine()} / {platform.processor()}",
        f"CUDA Available: {torch.cuda.is_available()}"
    ]

    if torch.cuda.is_available():
        details.append(f"GPU Device: {torch.cuda.get_device_name(0)}")

    details.append(f"Random Seed: {config.random_seed}")

    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        details.append(f"Git Commit: {commit_hash}")
    except Exception:
        details.append("Git Commit: N/A (Not a valid git repository)")

    details.append("--------------------------------")
    return "\n".join(details)


def get_generated_directory() -> Path:
    """Dynamically resolves the output directory relative to this script."""
    module_dir = Path(__file__).resolve().parent
    generated_dir = module_dir / "generated_files"
    generated_dir.mkdir(parents=True, exist_ok=True)
    return generated_dir


# =============================================================================
# 2. PROTECTED MATHEMATICAL OPERATORS
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
    """Ensures variance strictly adheres to finite, positive constraints."""
    sigma_squared = float(value)
    if not np.isfinite(sigma_squared):
        raise ValueError("sigma_squared must be finite.")
    return abs(sigma_squared)


# =============================================================================
# 3. DEAP TOOLBOX CONSTRUCTION
# =============================================================================

def build_primitive_set() -> gp.PrimitiveSet:
    """Builds the symbolic search space mapping M -> variance."""
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
# 4. INNER EVALUATION WRAPPER
# =============================================================================

def get_phase_a_dataset_ids(manager: DatasetManager, datasets_per_rule: int) -> List[int]:
    dataset_ids = list(manager.dataset_cache.keys())
    if len(dataset_ids) == 0:
        raise ValueError("Dataset cache is empty. Run manager.load_all_to_ram().")
    if len(dataset_ids) < datasets_per_rule:
        raise ValueError(f"Required {datasets_per_rule} datasets, but cache holds {len(dataset_ids)}")
    return dataset_ids[:datasets_per_rule]

def evaluate_rule(
    individual: gp.PrimitiveTree, manager: DatasetManager, activation: str,
    toolbox: base.Toolbox, config: MOGPConfig
) -> Tuple[float, float, int]:
    """Compiles the GP tree and evaluates its variance initialization."""
    rule_func = toolbox.compile(expr=individual)
    dataset_ids = get_phase_a_dataset_ids(manager, config.datasets_per_rule)

    total_acc, total_epochs = 0.0, 0.0
    tree_size = len(individual)

    for did in dataset_ids:
        tensors, meta_features = manager.get_dataset(did)
        m_vals = meta_features.detach().cpu().numpy()

        try:
            sigma_squared = sanitize_sigma_squared(rule_func(*m_vals))
        except Exception:
            return 0.0, 999.0, tree_size

        evaluator = PyTorchEvaluator(
            tensors, sigma_squared=sigma_squared, activation_name=activation,
            topology=config.topology, max_epochs=config.max_epochs, batch_size=config.batch_size,
        )

        acc, epochs = evaluator.evaluate_fitness()
        total_acc += float(acc)
        total_epochs += float(epochs)

    avg_acc = total_acc / float(len(dataset_ids))
    avg_epochs = total_epochs / float(len(dataset_ids))

    return avg_acc, avg_epochs, tree_size


# =============================================================================
# 5. ARTIFACT EXPORT & CHECKPOINTING
# =============================================================================

def export_pareto_rules(
    hall_of_fame: tools.ParetoFront, activation: str, config: MOGPConfig,
    output_dir: Path, env_details: str, top_k: int = 10, is_checkpoint: bool = False, gen: int = 0
) -> Path:
    """Exports structured results with 'Prototype' prefix labeling."""
    activation_token = "".join(c if c.isalnum() else "_" for c in activation.lower())
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if is_checkpoint:
        filename = f"Checkpoint_Prototype_{activation_token}_Gen{gen}.txt"
    else:
        filename = f"Prototype_Rule_{activation_token}_{timestamp}.txt"

    output_path = output_dir / filename

    with output_path.open("w", encoding="utf-8") as file:
        file.write("APPLIED COMPUTATIONAL INTELLIGENCE: MODULE 5 (PROTOTYPE) RESULTS\n")
        file.write(f"Activation Function: {activation.upper()}\n")
        file.write(f"Status: {'CHECKPOINT (Generation ' + str(gen) + ')' if is_checkpoint else 'FINAL PROTOTYPE RUN'}\n")
        file.write("=" * 72 + "\n")
        file.write(env_details + "\n")
        file.write("=" * 72 + "\n")
        file.write(f"Population: {config.population_size} | Generations: {config.generations}\n")
        file.write(f"Datasets/Rule: {config.datasets_per_rule} | Batch: {config.batch_size} | Topology: {config.topology}\n")
        file.write("=" * 72 + "\n\n")
        file.write("Top Discovered Rules-of-Thumb (Pareto Optimal):\n\n")

        if len(hall_of_fame) == 0:
            file.write("No Pareto-optimal rules discovered yet.\n")
        else:
            for rank, ind in enumerate(hall_of_fame[:top_k], start=1):
                file.write(f"Rank {rank}:\nEquation: {str(ind)}\n")
                file.write(f"Fitness: [Acc: {ind.fitness.values[0]:.4f}, Epochs: {ind.fitness.values[1]:.2f}, Bloat: {ind.fitness.values[2]}]\n\n")

        file.flush()
        os.fsync(file.fileno())

    return output_path


# =============================================================================
# 6. CUSTOM EVOLUTIONARY LOOP
# =============================================================================

def custom_eaMuPlusLambda(
    population: List, toolbox: base.Toolbox, config: MOGPConfig,
    halloffame: tools.ParetoFront, activation: str, output_dir: Path, env_details: str
) -> None:
    """Provides a generation-level loop enabling progress bars and checkpointing."""
    # Evaluate initial valid/invalid population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    export_pareto_rules(halloffame, activation, config, output_dir, env_details, is_checkpoint=True, gen=0)

    # Initialize TQDM tracking across Generations
    with tqdm(total=config.generations, desc=f"[{activation.upper()}] Evolutions", unit="gen") as pbar:
        for gen in range(1, config.generations + 1):

            # Generate Offspring
            offspring = algorithms.varOr(
                population, toolbox, config.population_size,
                config.crossover_probability, config.mutation_probability
            )

            # Evaluate Offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if halloffame is not None:
                halloffame.update(offspring)

            # Select Next Generation
            population[:] = toolbox.select(population + offspring, config.population_size)

            # Enforce Prototype Checkpoint
            export_pareto_rules(halloffame, activation, config, output_dir, env_details, is_checkpoint=True, gen=gen)
            pbar.update(1)


# =============================================================================
# 7. ORCHESTRATION ENGINE
# =============================================================================

def run_gp_for_activation(
    activation: str, manager: DatasetManager, toolbox: base.Toolbox,
    config: MOGPConfig, output_dir: Path, env_details: str
) -> Path:

    toolbox.register("evaluate", evaluate_rule, manager=manager, activation=activation, toolbox=toolbox, config=config)

    population = toolbox.population(n=config.population_size)
    hall_of_fame = tools.ParetoFront()

    # Execute Custom Loop
    custom_eaMuPlusLambda(population, toolbox, config, hall_of_fame, activation, output_dir, env_details)

    # Export Final Clean Artifact
    return export_pareto_rules(
        hall_of_fame=hall_of_fame, activation=activation, config=config,
        output_dir=output_dir, env_details=env_details, top_k=5, is_checkpoint=False
    )

def seed_runtime(seed: int) -> None:
    """Resets RNG strictly. Must be called before EVERY activation run for fairness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main() -> None:
    config = MOGPConfig()
    env_details = get_system_environment(config)
    output_dir = get_generated_directory()

    print("Loading Phase A Dataset Cache (Pre-Scaling to RAM)...")
    manager_config = CacheConfig()
    manager = DatasetManager(manager_config)
    manager.load_all_to_ram()

    primitive_set = build_primitive_set()
    toolbox = build_toolbox(primitive_set, config)

    print("\n--- Commencing Module 5 MOGP Prototype Search ---")
    print(env_details)
    print(f"Target Checkpoint / Export Directory: {output_dir}")
    print(f"Activation Functions (Prototype List): {', '.join(config.activation_functions)}")

    total_evals = config.population_size * config.generations * config.datasets_per_rule * len(config.activation_functions)
    print(f"Approx. Cumulative PyTorch FNN Evaluations (Prototype Scale): {total_evals:,}")

    exported_files = []

    for activation in config.activation_functions:
        # STRICT RESET: Ensures prototype activations start from identical stochastic landscapes
        seed_runtime(config.random_seed)

        output_path = run_gp_for_activation(
            activation=activation, manager=manager, toolbox=toolbox,
            config=config, output_dir=output_dir, env_details=env_details
        )
        exported_files.append(output_path)

    print("\n--- MODULE 5 MOGP PROTOTYPE SEARCH COMPLETE ---")
    print("Final Discovered Rule Artifacts:")
    for path in exported_files:
        print(f" - {path.name}")

if __name__ == "__main__":
    main()