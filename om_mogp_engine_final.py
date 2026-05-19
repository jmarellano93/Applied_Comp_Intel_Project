# Module 6 General Function: The complete evolutionary Optimization Method (OM) engine running at full scale.

import operator
import math
import random
import numpy as np
import datetime
from deap import algorithms, base, creator, tools, gp
import warnings
import os

# Import your custom modules
from experiment_modules.pm_dataset_manager import DatasetManager
from experiment_modules.pm_fnn_landscape import PyTorchEvaluator

warnings.filterwarnings("ignore")


# ==========================================
# 1. PROTECTED MATHEMATICAL OPERATORS
# ==========================================
def protected_div(left, right):
    try:
        return left / right if abs(right) > 1e-5 else 1.0
    except ZeroDivisionError:
        return 1.0


def protected_sqrt(x):
    return math.sqrt(abs(x))


def protected_log(x):
    return math.log(abs(x)) if abs(x) > 1e-5 else 0.0


def protected_exp(x):
    try:
        return math.exp(max(min(x, 10), -10))
    except OverflowError:
        return 0.0


# ==========================================
# 2. DEFINING THE SEARCH SPACE
# ==========================================
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

# ==========================================
# 3. MULTI-OBJECTIVE FITNESS SETUP (NSGA-II)
# ==========================================
# METHODOLOGY FIX: Pure Multi-Objective Constraints (1.0, -1.0, -1.0)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# ==========================================
# 4. THE EVALUATION LOOP (INNER LOOP)
# ==========================================
def evaluate_rule(individual, manager):
    rule_func = toolbox.compile(expr=individual)

    total_acc = 0.0
    total_epochs = 0
    num_datasets = len(manager.dataset_cache)
    tree_size = len(individual)

    for did in manager.dataset_cache.keys():
        dataset, meta_features = manager.get_dataset(did)
        m_vals = meta_features.numpy()

        try:
            sigma_squared = rule_func(*m_vals)
        except Exception:
            return 0.0, 999, tree_size

        evaluator = PyTorchEvaluator(dataset, sigma_squared=sigma_squared, max_epochs=30)
        acc, epochs = evaluator.evaluate_fitness()

        total_acc += acc
        total_epochs += epochs

    avg_acc = total_acc / num_datasets
    avg_epochs = total_epochs / num_datasets

    return avg_acc, avg_epochs, tree_size


toolbox.register("evaluate", evaluate_rule)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))


# ==========================================
# 5. MAIN EXECUTION & AUTOMATED EXPORT
# ==========================================
def main():
    GEN_DIR = r"C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project\generated_files"
    DATA_DIR = r"C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project\openml_cc18_datasets"

    # Ensure directory exists just in case
    os.makedirs(GEN_DIR, exist_ok=True)

    print("Loading Phase A Data to RAM...")
    # Point the DatasetManager to the generated_files directory for the CSV
    manager = DatasetManager(os.path.join(GEN_DIR, "Phase_A_Discovery_Datasets.csv"), DATA_DIR)
    manager.load_all_to_ram()

    toolbox.register("evaluate", evaluate_rule, manager=manager)

    # FINAL METHODOLOGY CONSTRAINTS
    POP_SIZE = 300
    NGEN = 40

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)

    print(f"\n--- Commencing Multi-Objective Symbolic Regression (FINAL RUN) ---")
    print(f"Population: {POP_SIZE} | Generations: {NGEN} | Datasets Evaluated Per Rule: 20\n")
    print(
        "This will execute up to 240,000 PyTorch training sessions. Please ensure your machine has adequate cooling.\n")

    algorithms.eaMuPlusLambda(pop, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, cxpb=0.6, mutpb=0.3, ngen=NGEN, stats=stats,
                              halloffame=hof, verbose=True)

    print("\n--- EVOLUTION COMPLETE ---")

    # Route the output text file to the generated_files directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(GEN_DIR, f"Final_Discovered_Rules_{timestamp}.txt")

    with open(output_filename, "w") as f:
        f.write("APPLIED COMPUTATIONAL INTELLIGENCE: MILESTONE 4 RESULTS\n")
        f.write(f"Run Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters: Population {POP_SIZE}, Generations {NGEN}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Top 10 Discovered Rules-of-Thumb (Pareto Optimal):\n\n")

        for i, ind in enumerate(hof[:10]):
            rank_text = f"Rank {i + 1}:\n"
            eq_text = f"Equation: {str(ind)}\n"
            fit_text = f"Fitness: [Acc: {ind.fitness.values[0]:.4f}, Epochs: {ind.fitness.values[1]:.2f}, Bloat: {ind.fitness.values[2]}]\n\n"

            print(rank_text + eq_text + fit_text.strip())
            f.write(rank_text + eq_text + fit_text)

    print(f"\nResults successfully exported and secured in: {output_filename}")


if __name__ == "__main__":
    main()