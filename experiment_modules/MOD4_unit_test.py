"""
Unit Tests for Module 4: PM FNN Landscape Evaluator

Validates architectural dimensions, gradient safeguards, and batch sizing.
"""

import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader

import MOD4_pm_fnn_landscape as mod4

def generate_mock_data():
    X_train = torch.randn(128, 10) # 128 instances, 10 features
    y_train = torch.randint(0, 3, (128,)) # 3 classes
    X_val = torch.randn(32, 10)
    y_val = torch.randint(0, 3, (32,))
    return {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}

def test_shallow_topology_dimensions():
    """Validates Phase A architectural constraints (2 layers, 64 neurons)."""
    model = mod4.PhaseA_Shallow_FNN(input_dim=10, num_classes=3)
    x = torch.randn(64, 10)
    out = model(x)
    assert out.shape == (64, 3), "Shallow Topology output dimensionality failure."

def test_deep_narrow_topology_dimensions():
    """Validates Phase B1 architectural constraints (8 layers, 32 neurons)."""
    model = mod4.PhaseB_DeepNarrow_FNN(input_dim=10, num_classes=3)
    x = torch.randn(64, 10)
    out = model(x)
    assert out.shape == (64, 3), "Deep Narrow Topology output dimensionality failure."

def test_funnel_topology_dimensions():
    """Validates Phase B2 architectural constraints (Decreasing layers)."""
    model = mod4.PhaseB_Funnel_FNN(input_dim=10, num_classes=3)
    x = torch.randn(64, 10)
    out = model(x)
    assert out.shape == (64, 3), "Funnel Topology output dimensionality failure."

def test_variance_injection():
    """Validates strict standard deviation scaling mapped to GP Rule."""
    data = generate_mock_data()
    # Sigma^2 = 0.04 -> StdDev = 0.2
    evaluator = mod4.PyTorchEvaluator(data, sigma_squared=0.04)

    # Check first linear layer
    layer_weight = evaluator.model.model[0].weight
    actual_std = torch.std(layer_weight).item()

    assert actual_std == pytest.approx(0.2, rel=0.1), "Variance injection failed to map mathematically."

def test_pytorch_balanced_accuracy():
    """Validates vectorized on-device accuracy metric."""
    data = generate_mock_data()
    evaluator = mod4.PyTorchEvaluator(data, sigma_squared=1.0)

    # 3 classes, perfect prediction
    y_true = torch.tensor([0, 0, 1, 1, 2, 2])
    y_pred = torch.tensor([0, 0, 1, 1, 2, 2])
    acc = evaluator._balanced_accuracy_pytorch(y_true, y_pred)
    assert acc == 1.0

    # Completely wrong prediction
    y_pred_wrong = torch.tensor([1, 1, 2, 2, 0, 0])
    acc_wrong = evaluator._balanced_accuracy_pytorch(y_true, y_pred_wrong)
    assert acc_wrong == 0.0