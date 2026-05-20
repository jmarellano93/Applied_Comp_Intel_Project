import os
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def qualitative_analysis():
    GEN_DIR = r"/old_shit"

    # Ensure directory exists
    os.makedirs(GEN_DIR, exist_ok=True)

    # 1. Define Symbolic Variables
    pc_eigen, target_entropy = sp.symbols('pc_eigen target_entropy')

    # Define the winning rule derived from MOGP
    # Rule: 0.0745 * (pc_eigen / target_entropy)
    rule_expr = 0.07458949361120437 * (pc_eigen / target_entropy)

    print("--- QUALITATIVE INTERPRETATION FRAMEWORK ---\n")
    print(f"Analyzed Equation [f(M)]: {rule_expr}")

    # 2. Directional Sensitivity (Partial Derivatives)
    d_pc = sp.diff(rule_expr, pc_eigen)
    d_entropy = sp.diff(rule_expr, target_entropy)

    print("\nDirectional Sensitivity (Partial Derivatives):")
    print(f"d(Variance) / d(PC Eigenvariance) = {d_pc}")
    print(f"d(Variance) / d(Target Entropy)   = {d_entropy}")

    # 3. Asymptotic Verification (3D Surface Plotting)
    # We plot the physical behavior of the rule across typical bounds
    eigen_vals = np.linspace(0.1, 1.0, 50)
    entropy_vals = np.linspace(0.1, 3.0, 50)  # Avoid exact zero to prevent infinity

    EIGEN, ENTROPY = np.meshgrid(eigen_vals, entropy_vals)

    # Lambdify the sympy expression for rapid numpy evaluation
    variance_func = sp.lambdify((pc_eigen, target_entropy), rule_expr, "numpy")
    VARIANCE = variance_func(EIGEN, ENTROPY)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(EIGEN, ENTROPY, VARIANCE, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_title('Asymptotic Verification: Data-Aware Initialization Landscape')
    ax.set_xlabel('Principal Component Eigenvariance')
    ax.set_ylabel('Target Entropy (Class Imbalance)')
    ax.set_zlabel('Optimal Weight Variance ($\sigma^2$)')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Variance Magnitude')

    # Save the 3D surface plot to the old_shit directory
    plot_path = os.path.join(GEN_DIR, "asymptotic_surface.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nAsymptotic Verification surface plot exported to '{plot_path}'.")


if __name__ == "__main__":
    qualitative_analysis()
