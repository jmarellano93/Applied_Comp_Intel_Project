import numpy as np
import scipy.stats as stats
import torch
import warnings
from dataset_manager import DatasetManager
from fnn_problem_model import PyTorchEvaluator

warnings.filterwarnings("ignore")


def rank_1_rule(pc_eigen, target_entropy):
    """
    The translated Rank 1 equation from the MOGP engine:
    neg(mul(-0.07458949361120437, protected_div(pc_eigen, target_entropy)))
    """
    # Protect division just like the GP engine did
    denominator = target_entropy if abs(target_entropy) > 1e-5 else 1.0
    return 0.07458949361120437 * (pc_eigen / denominator)


def main():
    print("Loading Phase B Validation Data (15 Unseen Datasets)...")
    manager = DatasetManager("Phase_B_Validation_Datasets.csv",
                             r"C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project\openml_cc18_datasets")
    manager.load_all_to_ram()

    gp_scores = []
    baseline_scores = []

    print("\n--- COMMENCING PHASE B VALIDATION ---")

    for did in manager.dataset_cache.keys():
        dataset, meta_features = manager.get_dataset(did)
        m_vals = meta_features.numpy()

        # M-vals mapping from GP: ARG3 is pc_eigen, ARG4 is target_entropy
        pc_eigen = m_vals[3]
        target_entropy = m_vals[4]

        # 1. Evaluate the GP Discovered Rule
        sigma_squared = rank_1_rule(pc_eigen, target_entropy)
        evaluator_gp = PyTorchEvaluator(dataset, sigma_squared=sigma_squared, max_epochs=30)
        acc_gp, _ = evaluator_gp.evaluate_fitness()
        gp_scores.append(acc_gp)

        # 2. Evaluate Baseline (PyTorch Default - Kaiming Uniform)
        # By passing a negative sigma squared, our FNN class will default to 1e-5,
        # but to truly test baseline, we bypass the custom injection.
        evaluator_base = PyTorchEvaluator(dataset, sigma_squared=1.0, max_epochs=30)
        # Reset weights to PyTorch Default (Kaiming Uniform) to erase the injection
        with torch.no_grad():
            for layer in [evaluator_base.model.layer1, evaluator_base.model.layer2, evaluator_base.model.output_layer]:
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        acc_base, _ = evaluator_base.evaluate_fitness()
        baseline_scores.append(acc_base)

        print(f"Dataset {did} -> GP Rule: {acc_gp:.4f} | PyTorch Baseline: {acc_base:.4f}")

    # --- STATISTICAL ANALYSIS ---
    print("\n--- WILCOXON SIGNED-RANK TEST RESULTS ---")

    # Calculate the differences
    differences = [gp - base for gp, base in zip(gp_scores, baseline_scores)]

    # Perform the Wilcoxon test
    stat, p_value = stats.wilcoxon(gp_scores, baseline_scores, alternative='greater')

    print(f"Mean GP Accuracy:       {np.mean(gp_scores) * 100:.2f}%")
    print(f"Mean Baseline Accuracy: {np.mean(baseline_scores) * 100:.2f}%")
    print(f"Wilcoxon W-Statistic:   {stat}")
    print(f"P-Value:                {p_value:.5f}")

    if p_value < 0.05:
        print("\nCONCLUSION: SUCCESS! The Symbolic GP Rule statistically outperforms the standard baseline (p < 0.05).")
    else:
        print(
            "\nCONCLUSION: The GP Rule performed well, but the difference did not achieve strict statistical significance (p >= 0.05).")


if __name__ == "__main__":
    main()