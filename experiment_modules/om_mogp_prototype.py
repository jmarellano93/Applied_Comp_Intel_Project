# Module 5 General Function: The exploratory Optimization Method (OM) sub-engine used for hyperparameter calibration.

import operator
import math
import random
import numpy as np
from deap import algorithms, base, creator, tools, gp
import warnings

# Import your custom modules
from pm_dataset_manager import DatasetManager
from pm_fnn_landscape import PyTorchEvaluator

warnings.filterwarnings("ignore")


# ==========================================
# 1. PROTECTED MATHEMATICAL OPERATORS
# ==========================================
# GP will generate wild equations. We must protect against zero-division and complex numbers
# to prevent the engine from crashing during evaluation.

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
        return math.exp(max(min(x, 10), -10))  # Cap exponents to prevent Overflow errors
    except OverflowError:
        return 0.0


# ==========================================
# 2. DEFINING THE SEARCH SPACE (TERMINALS & PRIMITIVES)
# ==========================================
# We have 8 inputs (the Elite 8 Meta-features).
pset = gp.PrimitiveSet("MAIN", 8)
pset.renameArguments(
    ARG0="n_d_ratio", ARG1="feat_kurtosis", ARG2="iqr_dev", ARG3="pc_eigen",
    ARG4="target_entropy", ARG5="hopkins", ARG6="silhouette", ARG7="davies_bouldin"
)

# Add mathematical building blocks
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
# Add Ephemeral Random Constants (floating point numbers between -1.0 and 1.0)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1.0, 1.0))

# ==========================================
# 3. MULTI-OBJECTIVE FITNESS SETUP (NSGA-II)
# ==========================================
# Objectives: 1. Maximize Accuracy, 2. Minimize Epochs, 3. Minimize Equation Bloat
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# ==========================================
# 4. THE EVALUATION LOOP (THE INJECTION)
# ==========================================
def evaluate_rule(individual, manager):
    """
    Compiles the GP syntax tree into a mathematical equation.
    Tests that exact equation across all 10 Phase A datasets.
    """
    rule_func = toolbox.compile(expr=individual)

    total_acc = 0.0
    total_epochs = 0
    num_datasets = len(manager.dataset_cache)

    # Calculate Bloat (Number of nodes in the equation)
    tree_size = len(individual)

    for did in manager.dataset_cache.keys():
        dataset, meta_features = manager.get_dataset(did)

        # 1. Read the dataset's specific topology
        m_vals = meta_features.numpy()

        # 2. Feed the topology into the GP rule to generate the Variance
        try:
            sigma_squared = rule_func(*m_vals)
        except Exception:
            # If the math still fails somehow, severely penalize it
            return 0.0, 999, tree_size

        # 3. Inject Variance into PyTorch and Train
        evaluator = PyTorchEvaluator(dataset, sigma_squared=sigma_squared, max_epochs=30)
        acc, epochs = evaluator.evaluate_fitness()

        total_acc += acc
        total_epochs += epochs

    # Return the average performance across all 10 manifolds + the structural bloat
    avg_acc = total_acc / num_datasets
    avg_epochs = total_epochs / num_datasets

    return avg_acc, avg_epochs, tree_size


# Register evolutionary operators
toolbox.register("evaluate", evaluate_rule)
toolbox.register("select", tools.selNSGA2)  # Pareto-front sorting
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Enforce a hard limit on tree depth to prevent exponential memory consumption
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))


# ==========================================
# 5. MAIN EXECUTION (THE OUTER LOOP)
# ==========================================
def main():
    print("Loading Phase A Data to RAM...")
    manager = DatasetManager("../generated_files/Phase_A_Discovery_Datasets.csv",
                             r"C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project\openml_cc18_datasets")
    manager.load_all_to_ram()

    # Inject the manager into the evaluation function
    toolbox.register("evaluate", evaluate_rule, manager=manager)

    # Downsized Methodology Constraints
    POP_SIZE = 150  # Start small for Milestone 3 testing (Scale to 150 later)
    NGEN = 20  # Start small for Milestone 3 testing (Scale to 20 later)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.ParetoFront()  # Tracks the best multi-objective solutions

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)

    print(f"\n--- Commencing Multi-Objective Symbolic Regression ---")
    print(f"Population: {POP_SIZE} | Generations: {NGEN} | Datasets Evaluated Per Rule: 10\n")

    # The actual evolutionary loop using NSGA-II selection
    algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=POP_SIZE, lambda_=POP_SIZE,
        cxpb=0.6, mutpb=0.3,  # 60% Crossover, 30% Mutation
        ngen=NGEN, stats=stats, halloffame=hof, verbose=True
    )

    print("\n--- EVOLUTION COMPLETE ---")
    print("\nTop 3 Discovered Rules-of-Thumb (Pareto Optimal):")
    for i, ind in enumerate(hof[:3]):
        print(f"\nRank {i + 1}:")
        print(f"Equation: {str(ind)}")
        print(
            f"Fitness: [Acc: {ind.fitness.values[0]:.3f}, Epochs: {ind.fitness.values[1]:.1f}, Bloat: {ind.fitness.values[2]}]")


if __name__ == "__main__":
    main()