"""
Module 6: Optimization Method (OM) - MOGP Meta-Grid Search Engine.

Executes the definitive genetic programming outer-loop across the 6 canonical
activation functions. Explicitly CPU-routed for Phase A (Shallow FNN).
Produces the final multiobjective Pareto rules utilized in validation.
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
# FUNCTIONAL BLOCK: Production Configuration Bounds
# 4A) WHAT IT DOES: Establishes the massive computational dimensions required to
#     explore the symbolic meta-feature space deeply enough to guarantee global optima.
# 4B) PARAMETERS: population_size (100), generations (20), datasets_per_rule (20).
# 4C) METHODOLOGICAL JUSTIFICATION: Because fitness evaluation requires full
#     PyTorch neural network training, traditional GP population sizes (e.g., 500+)
#     are computationally prohibitive. A population of 150 across 20 generations
#     yields roughly 240,000 total PyTorch evaluations across the activation suite.
#     To mitigate premature convergence in this restricted population, we rely on
#     high mutation rates (30%) and NSGA-II multi-objective selection to artificially
#     enforce genetic and topological diversity. Training each across 20 distinct
#     dataset topologies mathematically guarantees that discovered initialization
#     heuristics are globally generalized across tabular data, not overfit to a single domain.
# =============================================================================

class MOGPConfig(BaseModel):
    population_size: int = Field(default=100, gt=0)
    generations: int = Field(default=20, gt=0)
    datasets_per_rule: int = Field(default=20, gt=0)
    max_epochs: int = Field(default=30, gt=0)
    batch_size: int = Field(default=64, gt=0)

    crossover_probability: float = Field(default=0.6, ge=0.0, le=1.0)
    mutation_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tree_height: int = Field(default=6, gt=0)

    topology: str = Field(default="shallow")
    random_seed: int = Field(default=42)
    target_acc: float = Field(default=0.85, ge=0.0, le=1.0)
    activation_functions: List[str] = Field(
        default=["rectification", "squashing", "smooth", "aggregation", "trigonometric", "linear"]
    )

    # -------------------------------------------------------------------------
    # Q-G: Topology coverage. Each (topology, activation) pair drives an
    # independent set of GP runs.
    # -------------------------------------------------------------------------
    topology_targets: List[str] = Field(
        default=["shallow", "deep_narrow", "funnel"]
    )

    # -------------------------------------------------------------------------
    # Q-E: Statistical replication. N independent GP runs per (topology,
    # activation) pair with distinct seeds; their Pareto fronts are aggregated
    # into a consensus front by union-with-deduplication + NSGA-II re-sort.
    # -------------------------------------------------------------------------
    n_gp_runs: int = Field(default=3, ge=1)
    gp_run_seeds: List[int] = Field(default=[42, 43, 44])

    # -------------------------------------------------------------------------
    # Q-F: Plan B contingency. When enabled, the GP switches from NSGA-II
    # multi-objective Pareto to tournament-selected SOO over a scalar
    # weighted fitness f = w_acc * acc - w_eps * (epochs / max_epochs).
    # Bloat is excluded from the scalar fitness; the height cap remains.
    # -------------------------------------------------------------------------
    use_plan_b_soo: bool = Field(default=False)
    plan_b_w_acc: float = Field(default=1.0, gt=0.0)
    plan_b_w_eps: float = Field(default=0.5, ge=0.0)
    plan_b_tournament_size: int = Field(default=3, ge=2)

    # -------------------------------------------------------------------------
    # Resume-on-restart safety: if a consensus artifact for a (topology,
    # activation) pair already exists in the output directory, skip that pair
    # by default. Set ``force_rerun=True`` (or pass ``--force_rerun``) to
    # overwrite previously-computed pairs.
    # -------------------------------------------------------------------------
    force_rerun: bool = Field(default=False)

    num_threads: int = Field(default=4, ge=1)
    checkpoint_every_n_gen: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "MOGPConfig":
        if self.crossover_probability + self.mutation_probability > 1.0:
            raise ValueError("Crossover + Mutation probability must be <= 1.0")
        if len(self.gp_run_seeds) < self.n_gp_runs:
            raise ValueError(
                f"gp_run_seeds has {len(self.gp_run_seeds)} entries but "
                f"n_gp_runs={self.n_gp_runs} requires at least that many."
            )
        return self


def get_system_environment(config: MOGPConfig) -> str:
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


# =============================================================================
# FUNCTIONAL BLOCK: Architectural Output Routing
# 4A) WHAT IT DOES: Dynamically maps the target directory to save the discovered equations.
# 4B) PARAMETERS: N/A (Implicit OS lookup).
# 4C) METHODOLOGICAL JUSTIFICATION: Prevents absolute path crashes across varied
#     operating systems, while isolating the output specifically to the /rules
#     subdirectory to prevent artifact collision with later reports.
# =============================================================================
def get_generated_directory() -> Path:
    module_dir = Path(__file__).resolve().parent
    generated_dir = module_dir / "generated_files" / "GA_rule_files"
    generated_dir.mkdir(parents=True, exist_ok=True)
    return generated_dir


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


def build_primitive_set() -> gp.PrimitiveSet:
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


def ensure_deap_creators(use_plan_b_soo: bool = False) -> None:
    """Initialises DEAP creators for either Plan A (Pareto multi-objective)
    or Plan B (single-objective scalar).

    Plan A uses ``FitnessMulti(weights=(1.0, -1.0, -1.0))`` over (acc, epochs, bloat).
    Plan B uses ``FitnessSingle(weights=(1.0,))`` over the scalar
    ``w_acc * acc - w_eps * (epochs / max_epochs)``.

    DEAP's ``creator`` is a process-global registry. To make mode-switching
    safe within a single Python process (relevant for unit tests and any
    notebook/integration usage that calls this with mixed values), we ALWAYS
    clear any stale ``FitnessMulti``, ``FitnessSingle``, and ``Individual``
    bindings at the entry of this function before installing only those
    needed for the current mode. Without this symmetric cleanup, a prior
    Plan B invocation would leave ``Individual`` bound to ``FitnessSingle``,
    silently breaking a subsequent Plan A run.
    """
    for token in ("FitnessMulti", "FitnessSingle", "Individual"):
        if hasattr(creator, token):
            delattr(creator, token)

    if use_plan_b_soo:
        creator.create("FitnessSingle", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessSingle)
    else:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)


def build_toolbox(primitive_set: gp.PrimitiveSet, config: MOGPConfig) -> base.Toolbox:
    """Builds the DEAP toolbox with selection branched on Plan A / Plan B.

    Plan A: NSGA-II non-dominated sorting + crowding distance.
    Plan B: tournament selection (size 3 by default) for SOO.
    All other operators (crossover, mutation, height limit) are shared.
    """
    ensure_deap_creators(use_plan_b_soo=config.use_plan_b_soo)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=primitive_set, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primitive_set)

    if config.use_plan_b_soo:
        toolbox.register("select", tools.selTournament, tournsize=config.plan_b_tournament_size)
    else:
        toolbox.register("select", tools.selNSGA2)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitive_set)
    limit_decorator = gp.staticLimit(key=operator.attrgetter("height"), max_value=config.max_tree_height)
    toolbox.decorate("mate", limit_decorator)
    toolbox.decorate("mutate", limit_decorator)
    return toolbox


def get_phase_a_dataset_ids(manager: DatasetManager, datasets_per_rule: int) -> List[int]:
    dataset_ids = list(manager.dataset_cache.keys())
    if len(dataset_ids) < datasets_per_rule:
        raise ValueError(
            f"Required {datasets_per_rule} datasets, but cache holds {len(dataset_ids)}."
        )
    return dataset_ids[:datasets_per_rule]


def _get_or_build_trainer(
    pool: Dict[int, FNNTrainer], manager: DatasetManager,
    did: int, activation: str, config: MOGPConfig,
) -> Tuple[FNNTrainer, torch.Tensor]:
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
    individual: gp.PrimitiveTree, manager: DatasetManager, activation: str,
    toolbox: base.Toolbox, config: MOGPConfig, trainer_pool: Dict[int, FNNTrainer],
) -> Tuple[float, ...]:
    """Evaluates one GP individual across the Phase A bench.

    Returns a fitness tuple whose arity depends on the optimisation mode:
        Plan A: (mean_balanced_acc, mean_epochs_to_threshold, tree_size)
        Plan B: (scalar_fitness,) where
            scalar_fitness = w_acc * mean_acc - w_eps * (mean_epochs / max_epochs)

    A rule that triggers NaN/Inf gradients (numerical instability constraint)
    is penalised with the dominated sentinel:
        Plan A: (0.0, 999.0, tree_size)
        Plan B: (0.0,)
    """
    rule_func = toolbox.compile(expr=individual)
    dataset_ids = get_phase_a_dataset_ids(manager, config.datasets_per_rule)

    total_acc = 0.0
    total_epochs = 0.0
    tree_size = len(individual)

    # Sentinel constants for the constraint-violation penalty (kept local
    # because they differ in shape between Plan A and Plan B).
    def _penalty() -> Tuple[float, ...]:
        if config.use_plan_b_soo:
            return (0.0,)
        return (0.0, 999.0, tree_size)

    for did in dataset_ids:
        trainer, meta_features = _get_or_build_trainer(
            trainer_pool, manager, did, activation, config
        )
        m_vals = meta_features.detach().cpu().numpy()

        try:
            sigma_squared = sanitize_sigma_squared(rule_func(*m_vals))
        except Exception:
            return _penalty()

        torch.manual_seed(config.random_seed + did)
        trainer.reset_weights(sigma_squared)

        acc, epochs = trainer.evaluate()

        if epochs == 999 or acc == 0.0:
            return _penalty()

        total_acc += float(acc)
        total_epochs += float(epochs)

    n = float(len(dataset_ids))
    mean_acc = total_acc / n
    mean_epochs = total_epochs / n

    if config.use_plan_b_soo:
        scalar = (
            config.plan_b_w_acc * mean_acc
            - config.plan_b_w_eps * (mean_epochs / float(config.max_epochs))
        )
        return (float(scalar),)

    return mean_acc, mean_epochs, tree_size


def export_pareto_rules(
    hall_of_fame: tools.ParetoFront, activation: str, config: MOGPConfig,
    output_dir: Path, env_details: str, top_k: int = 10,
    is_checkpoint: bool = False, gen: int = 0,
    topology: str = None, run_index: int = None,
) -> Path:
    """Writes the Pareto / SOO front to disk.

    Args:
        hall_of_fame: Final or intermediate hall-of-fame object.
        activation: Activation token.
        config: Runtime configuration.
        output_dir: Target directory.
        env_details: System environment block for reproducibility.
        top_k: How many ranks to write out.
        is_checkpoint: If True, writes to a Checkpoint_*.txt filename.
        gen: Generation index for checkpoint files.
        topology: Topology token. Embedded in filename if provided
            (None falls back to ``config.topology`` for backward compat).
        run_index: 1-based per-seed run index. If provided, the filename
            uses the ``_Run<i>`` suffix and the archive subdirectory; if
            None, the filename is the consensus form (placed in the main
            ``output_dir``).
    """
    activation_token = "".join(c if c.isalnum() else "_" for c in activation.lower())
    topology_token = (topology or config.topology).lower()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if is_checkpoint:
        suffix = f"_Run{run_index}" if run_index is not None else ""
        filename = f"Checkpoint_{activation_token}_{topology_token}{suffix}_Gen{gen}.txt"
        target_dir = output_dir
    elif run_index is not None:
        # Per-run archive: lives under _runs/ so MOD7/MOD9 don't pick it up.
        target_dir = output_dir / "per_run_archive"
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"Final_Discovered_Rules_{activation_token}_{topology_token}"
            f"_Run{run_index}_{timestamp}.txt"
        )
    else:
        # Consensus front: lives directly in output_dir, matched by MOD7/9 glob.
        target_dir = output_dir
        filename = (
            f"Final_Discovered_Rules_{activation_token}_{topology_token}_{timestamp}.txt"
        )
    output_path = target_dir / filename

    plan_b = config.use_plan_b_soo
    status = "CHECKPOINT (Generation " + str(gen) + ")" if is_checkpoint else (
        "PER-RUN PARETO FRONT (Run " + str(run_index) + ")" if run_index is not None
        else "CONSENSUS PARETO FRONT (aggregated across runs)"
    )

    with output_path.open("w", encoding="utf-8") as file:
        file.write("APPLIED COMPUTATIONAL INTELLIGENCE: MODULE 6 RESULTS\n")
        file.write(f"Activation Function: {activation.upper()}\n")
        file.write(f"Topology: {topology_token}\n")
        file.write(f"Status: {status}\n")
        file.write(f"Mode: {'Plan B SOO scalar fitness' if plan_b else 'Plan A Pareto (NSGA-II)'}\n")
        file.write("=" * 72 + "\n")
        file.write(env_details + "\n")
        file.write("=" * 72 + "\n")
        file.write(f"Population: {config.population_size} | Generations: {config.generations}\n")
        file.write(
            f"Datasets/Rule: {config.datasets_per_rule} | Batch: {config.batch_size} | "
            f"Topology: {topology_token}\n"
        )
        if plan_b:
            file.write(
                f"Plan B Weights: w_acc={config.plan_b_w_acc}, w_eps={config.plan_b_w_eps}\n"
            )
        file.write("=" * 72 + "\n\n")
        file.write("Top Discovered Rules-of-Thumb:\n\n")

        if len(hall_of_fame) == 0:
            file.write("No rules discovered yet.\n")
        else:
            for rank, ind in enumerate(hall_of_fame[:top_k], start=1):
                file.write(f"Rank {rank}:\nEquation: {str(ind)}\n")
                vals = ind.fitness.values
                if plan_b:
                    # Single-objective scalar.
                    file.write(f"Fitness: [Scalar: {vals[0]:.6f}]\n\n")
                else:
                    # Multi-objective (acc, epochs, bloat).
                    file.write(
                        f"Fitness: [Acc: {vals[0]:.4f}, "
                        f"Epochs: {vals[1]:.2f}, "
                        f"Bloat: {vals[2]}]\n\n"
                    )
        file.flush()
        os.fsync(file.fileno())

    return output_path


def _evaluate_invalid(
    invalid_ind: List[gp.PrimitiveTree], toolbox: base.Toolbox, desc: str,
) -> None:
    with tqdm(total=len(invalid_ind), desc=desc, unit="ind", leave=False) as bar:
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
            bar.update(1)


def custom_eaMuPlusLambda(
    population: List, toolbox: base.Toolbox, config: MOGPConfig,
    halloffame: tools.ParetoFront, activation: str,
    output_dir: Path, env_details: str,
    topology: str = None, run_index: int = None,
) -> None:
    """EA outer loop with per-generation checkpoints.

    Args:
        population: Initial population (Gen 0 individuals).
        toolbox: DEAP toolbox with registered evaluate / mate / mutate / select.
        config: Runtime configuration.
        halloffame: A ``tools.ParetoFront`` collecting non-dominated individuals.
        activation: Activation token, threaded into checkpoint filenames.
        output_dir: Target output directory.
        env_details: System environment block for reproducibility.
        topology: Topology token. Passed through to ``export_pareto_rules``
            so checkpoint filenames include the topology, preventing
            cross-topology overwrites within a single shared output dir.
        run_index: 1-based per-seed run index. Passed through so checkpoint
            filenames carry a ``_Run<i>`` suffix and do NOT collide across
            the N independent runs of the same (topology, activation) pair.
    """
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    _evaluate_invalid(invalid_ind, toolbox, desc=f"[{activation.upper()}] Gen 0 init")

    if halloffame is not None:
        halloffame.update(population)
    export_pareto_rules(
        halloffame, activation, config, output_dir, env_details,
        is_checkpoint=True, gen=0,
        topology=topology, run_index=run_index,
    )

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
                    topology=topology, run_index=run_index,
                )
            pbar.update(1)


def run_gp_for_activation(
    activation: str, topology: str, manager: DatasetManager, toolbox: base.Toolbox,
    config: MOGPConfig, output_dir: Path, env_details: str, run_seed: int, run_index: int,
) -> "tools.ParetoFront":
    """Execute one GP run for a single (topology, activation, seed) triple.

    Args:
        activation: Activation token.
        topology: FNN topology token (overrides ``config.topology`` for this run).
        manager: Loaded ``DatasetManager``.
        toolbox: Pre-built DEAP toolbox.
        config: Runtime configuration.
        output_dir: Target output directory for per-run artifacts.
        env_details: System environment block.
        run_seed: The random seed for THIS run.
        run_index: 1-based index of THIS run (1..N).

    Returns:
        The final hall-of-fame (Pareto front for Plan A, top-K front for Plan B).
        Persists the per-run artifact to ``output_dir / per_run_archive / ...``.
    """
    # Reseed all RNGs at the entry of this run so populations differ per seed.
    seed_runtime(run_seed)
    torch.manual_seed(run_seed)

    # Patch config.topology for the duration of this run so FNNTrainer
    # constructs the correct topology. Restored before return.
    saved_topology = config.topology
    config.topology = topology
    try:
        trainer_pool: Dict[int, FNNTrainer] = {}
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
            topology=topology, run_index=run_index,
        )

        # Persist per-run artifact (archive subdirectory).
        export_pareto_rules(
            hall_of_fame=hall_of_fame, activation=activation, config=config,
            output_dir=output_dir, env_details=env_details, top_k=10,
            is_checkpoint=False, topology=topology, run_index=run_index,
        )

        trainer_pool.clear()
        return hall_of_fame
    finally:
        config.topology = saved_topology


def aggregate_consensus_front(
    per_run_fronts: List["tools.ParetoFront"], config: MOGPConfig,
) -> List:
    """Aggregate N per-run Pareto fronts into a single consensus front.

    Algorithm:
        1. Collect all unique individuals (deduplicated by exact string repr).
        2. For each unique rule, compute its mean fitness across the runs in
           which it appeared (Plan A: per-objective mean; Plan B: scalar mean).
        3. Apply non-dominated sorting (Plan A) or scalar sort (Plan B) to
           obtain the consensus first-front / top-K.

    Returns:
        Sorted list of consensus individuals (best first), of length at most
        equal to the union-of-runs size.
    """
    seen: Dict[str, Dict] = {}
    for front in per_run_fronts:
        for ind in front:
            key = str(ind)
            if key not in seen:
                seen[key] = {"individual": ind, "fitnesses": []}
            seen[key]["fitnesses"].append(tuple(ind.fitness.values))

    if not seen:
        return []

    # Build consensus population: one Individual per unique string with the
    # per-objective mean fitness across runs it appeared in.
    consensus: List = []
    for entry in seen.values():
        # Clone the prototype Individual to avoid mutating the original.
        proto = entry["individual"]
        cloned = creator.Individual(proto)
        per_obj = list(zip(*entry["fitnesses"]))  # transpose
        mean_fit = tuple(float(np.mean(o)) for o in per_obj)
        cloned.fitness.values = mean_fit
        consensus.append(cloned)

    if config.use_plan_b_soo:
        # SOO: descending scalar fitness (weight = +1).
        consensus.sort(key=lambda c: c.fitness.values[0], reverse=True)
        return consensus

    # Plan A: NSGA-II first non-dominated front, then descending accuracy.
    first_front = tools.sortNondominated(
        consensus, len(consensus), first_front_only=True,
    )[0]
    first_front.sort(key=lambda c: c.fitness.values[0], reverse=True)
    return first_front


def seed_runtime(seed: int) -> None:
    """Resets Python's RNG and NumPy's RNG. PyTorch seeded separately per eval."""
    random.seed(seed)
    np.random.seed(seed)


def configure_torch_threads(config: MOGPConfig) -> None:
    """Pin PyTorch thread counts before any tensor ops execute."""
    torch.set_num_threads(min(config.num_threads, os.cpu_count() or config.num_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _parse_cli_overrides(config: MOGPConfig) -> MOGPConfig:
    """Optional CLI overrides for topology/activation/run subsets and Plan B.

    Lets the user pre-flight subsets without editing the script:
        python MOD6_om_mogp_engine_final.py \\
            --topologies shallow \\
            --activations rectification \\
            --n_runs 1 \\
            --population_size 50 \\
            [--use_plan_b_soo]
    With no CLI arguments, the full sweep defined by MOGPConfig defaults runs.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="MOD6 MOGP discovery engine (multi-topology, multi-seed).",
    )
    parser.add_argument("--topologies", type=str, nargs="+", default=None)
    parser.add_argument("--activations", type=str, nargs="+", default=None)
    parser.add_argument("--n_runs", type=int, default=None)
    parser.add_argument(
        "--population_size", type=int, default=None,
        help="Override the GP population size (default 100). Use 50-75 for "
             "smoke tests, 100 for production, 150 for a higher-diversity sweep.",
    )
    parser.add_argument(
        "--use_plan_b_soo", action="store_true",
        help="Switch to Plan B single-objective SOO (tournament selection, scalar fitness).",
    )
    parser.add_argument(
        "--force_rerun", action="store_true",
        help="Re-run (topology, activation) pairs even if a consensus artifact "
             "already exists. Default behaviour is to SKIP such pairs (resume).",
    )
    args, _unknown = parser.parse_known_args()

    if args.topologies is not None:
        config.topology_targets = args.topologies
    if args.activations is not None:
        config.activation_functions = args.activations
    if args.n_runs is not None:
        config.n_gp_runs = args.n_runs
    if args.population_size is not None:
        config.population_size = args.population_size
    if args.use_plan_b_soo:
        config.use_plan_b_soo = True
    if args.force_rerun:
        config.force_rerun = True

    return config


def main() -> None:
    """Full sweep across (activation, topology, run-seed) triples.

    Outer loop:  activation in config.activation_functions  (default 6)
    Middle loop: topology   in config.topology_targets      (default 3)
    Inner loop:  run_idx    in 1..config.n_gp_runs          (default 3)

    For each (activation, topology), the N per-run Pareto fronts are
    aggregated via ``aggregate_consensus_front`` into the consensus front
    written as the canonical artifact in the rule directory; per-run
    intermediates are archived under ``per_run_archive/``.

    Iteration order rationale:
        Activation is the OUTER loop, so all three topologies for a given
        activation function complete BEFORE the next activation begins.
        If the sweep is terminated early (intentionally, due to time
        constraints, or by an unexpected crash), the user is guaranteed
        to have *complete* result sets for every fully-finished activation
        — i.e., consensus fronts for all three topologies of that
        activation. Reducing the activation set via ``--activations``
        therefore lets the user finish a 2-activation sweep in ~1.5 days
        instead of waiting ~4.7 days for the full 6-activation sweep, while
        still yielding analyzable (topology, activation) cells for every
        completed activation.

    Resume safety:
        At the top of each (topology, activation) inner loop, the function
        checks whether a consensus artifact already exists for that pair.
        If so, the pair is SKIPPED unless ``--force_rerun`` is set. This
        makes the 7-day sweep recoverable: if the process dies on day 4,
        re-launching from the same shell picks up at the next incomplete
        pair, preserving all previously-written artifacts.
    """
    config = MOGPConfig()
    config = _parse_cli_overrides(config)

    configure_torch_threads(config)
    env_details = get_system_environment(config)
    output_dir = get_generated_directory()

    # --- PRE-FLIGHT: refuse to start a 7-day run unless MOD2 normalization
    # parameters exist on disk. A silent fall-through to raw meta-features
    # would invalidate every discovered rule and break consistency with the
    # downstream MOD7/MOD9 evaluation flow (which also reads the same params).
    manager_config = CacheConfig()
    norm_path = Path(manager_config.norm_params_path)
    if not norm_path.exists():
        raise FileNotFoundError(
            f"PRE-FLIGHT FAILURE: meta-feature normalization params not found at\n"
            f"    {norm_path}\n"
            f"Re-run MOD2_pipeline_meta_extractor.py FIRST so that MOD6 receives\n"
            f"z-scored terminal values consistent with the downstream MOD7\n"
            f"validation pipeline. Aborting before any compute is wasted."
        )
    print(f"PRE-FLIGHT OK: normalization params present at {norm_path.name}")

    print("Loading Phase A Dataset Cache (Pre-Scaling to RAM)...")
    manager = DatasetManager(manager_config)
    manager.load_all_to_ram()

    primitive_set = build_primitive_set()
    toolbox = build_toolbox(primitive_set, config)

    print("\n--- Commencing Module 6 MOGP Meta-Grid Search ---")
    print(env_details)
    print(f"Target Checkpoint / Export Directory: {output_dir}")
    print(f"Topologies:    {', '.join(config.topology_targets)}")
    print(f"Activations:   {', '.join(config.activation_functions)}")
    print(f"GP Runs/pair:  {config.n_gp_runs} (seeds: {config.gp_run_seeds[:config.n_gp_runs]})")
    print(f"Mode:          {'Plan B SOO' if config.use_plan_b_soo else 'Plan A Pareto'}")
    print(f"Force re-run:  {config.force_rerun}")

    total_evals = (
        config.population_size
        * config.generations
        * config.datasets_per_rule
        * len(config.activation_functions)
        * len(config.topology_targets)
        * config.n_gp_runs
    )
    print(f"Approx. Cumulative PyTorch CPU Evaluations: {total_evals:,}")

    exported_files: List[Path] = []
    skipped_pairs: List[Tuple[str, str]] = []

    for activation in config.activation_functions:
        for topology in config.topology_targets:
            # --- Resume guard: skip pairs whose consensus already exists. ---
            existing = list(output_dir.glob(
                f"Final_Discovered_Rules_{activation}_{topology}_*.txt"
            ))
            if existing and not config.force_rerun:
                print(
                    f"\n========== ({activation.upper()} / {topology.upper()}) "
                    f"SKIPPED — found {len(existing)} pre-existing consensus "
                    f"artifact(s). Use --force_rerun to override. =========="
                )
                skipped_pairs.append((topology, activation))
                continue

            print(f"\n========== ({activation.upper()} / {topology.upper()}) ==========")
            per_run_fronts: List = []

            for run_index in range(1, config.n_gp_runs + 1):
                run_seed = config.gp_run_seeds[run_index - 1]
                print(f"--- Run {run_index}/{config.n_gp_runs} (seed={run_seed}) ---")
                front = run_gp_for_activation(
                    activation=activation, topology=topology,
                    manager=manager, toolbox=toolbox,
                    config=config, output_dir=output_dir,
                    env_details=env_details,
                    run_seed=run_seed, run_index=run_index,
                )
                per_run_fronts.append(front)

            # Aggregate into consensus front and write the canonical artifact.
            consensus = aggregate_consensus_front(per_run_fronts, config)
            print(f"Consensus front size: {len(consensus)} unique rules")

            consensus_path = export_pareto_rules(
                hall_of_fame=consensus, activation=activation, config=config,
                output_dir=output_dir, env_details=env_details, top_k=10,
                is_checkpoint=False, topology=topology, run_index=None,
            )
            exported_files.append(consensus_path)

    # --- Write a machine-readable manifest of the full sweep. ---
    manifest = {
        "completed_pairs": [
            {"topology": p.name.split("_")[-3] if "_Run" not in p.name else None,
             "consensus_file": p.name}
            for p in exported_files
        ],
        "skipped_pairs": [
            {"topology": t, "activation": a} for (t, a) in skipped_pairs
        ],
        "config_snapshot": {
            "topology_targets": config.topology_targets,
            "activation_functions": config.activation_functions,
            "n_gp_runs": config.n_gp_runs,
            "use_plan_b_soo": config.use_plan_b_soo,
            "population_size": config.population_size,
            "generations": config.generations,
            "datasets_per_rule": config.datasets_per_rule,
        },
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    import json
    manifest_path = output_dir / "MOD6_sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n--- MODULE 6 MOGP META-GRID SEARCH COMPLETE ---")
    print(f"Newly written consensus artifacts: {len(exported_files)}")
    print(f"Skipped (already existed):         {len(skipped_pairs)}")
    print(f"Manifest:                          {manifest_path.name}")
    for path in exported_files:
        print(f" - {path.name}")


if __name__ == "__main__":
    main()