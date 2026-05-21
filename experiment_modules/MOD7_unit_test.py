"""
Unit Tests for Module 7: Framework Validation Matrix
Validates binomial distribution math, deterministic locking, and cross-device hardware alignment.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

import MOD7_framework_validation_matrix as mod7

def test_lsuv_hardware_device_alignment():
    """
    Validates that the empirical LSUV baseline respects the hardware device
    of the target model, preventing CPU/GPU memory violations.
    """
    # Create a small dummy dataset intentionally anchored to CPU
    mock_dataset = {'X_train': torch.randn(200, 10)}
    mock_m_vals = np.zeros(8)

    # Initialize a test model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    # Force the model to GPU if available to test cross-device logic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    try:
        # If LSUV fails to move the mock CPU data to the GPU, it will throw a RuntimeError
        mod7.apply_baseline_initialization(model, 'LSUV', mock_dataset, mock_m_vals)
    except RuntimeError as e:
        pytest.fail(f"LSUV Device Alignment Failed: {e}")


def test_binomial_cumulative_distribution_math():
    """Validates statistical significance calculations against known binomial probabilities."""
    p_baseline = mod7.calculate_binomial_p_value(62, 125)
    assert p_baseline > 0.05, "50% win rate incorrectly flagged as significant."

    p_significant = mod7.calculate_binomial_p_value(75, 125)
    assert p_significant < 0.05, "60% win rate (N=125) failed to register significance."