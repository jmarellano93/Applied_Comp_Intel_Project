"""
Unit Tests for Module 4: FNN Landscape Evaluator

Validates the structural instantiation of neural models and the safety
of the amortized stateful FNNTrainer.
"""

import pytest
import torch
import torch.nn as nn

# --- Path bootstrap (added after relocating this test to unit_tests/) -------
# The modules under test live in the sibling experiment_modules/ directory.
# Add it to sys.path so the bare ``import MOD...`` below resolves regardless of
# the current working directory or how pytest is launched.
import sys as _sys
from pathlib import Path as _Path
_MODULES_DIR = _Path(__file__).resolve().parent.parent / "experiment_modules"
if str(_MODULES_DIR) not in _sys.path:
    _sys.path.insert(0, str(_MODULES_DIR))
# ---------------------------------------------------------------------------

import MOD4_pm_fnn_landscape as mod4

def test_activation_routing_bounds():
    """Ensures unrecognized activation tokens fallback to standard non-linear spaces safely."""
    act_smooth = mod4.get_activation("smooth")
    assert isinstance(act_smooth, nn.GELU)

    # Validates boundary fallback logic
    act_unknown = mod4.get_activation("random_garbage_string")
    assert isinstance(act_unknown, nn.GELU), "Fallback routing failed."


def test_topology_factory_integrity():
    """Validates the factory pattern correctly allocates input and target vectors."""
    model_shallow = mod4._build_fnn_model(input_dim=15, num_classes=3, activation_name="smooth", topology="shallow")

    # Assert starting input size maps
    first_layer = next(model_shallow.modules())
    if isinstance(first_layer, nn.Sequential):
        first_linear = first_layer[0]
        assert first_linear.in_features == 15

    # Assert final dimension maps to classes
    last_layer = list(model_shallow.modules())[-1]
    assert last_layer.out_features == 3


def test_trainer_initialization_constraints():
    """Verifies FNNTrainer rejects badly dimensioned matrices prior to entering PyTorch's execution graph."""
    bad_dict = {
        "X_train": torch.randn(10, 5),
        "y_train": torch.randint(0, 2, (10,)),
        "X_val": torch.randn(5, 4), # Dims mismatch!
        "y_val": torch.randint(0, 2, (5,))
    }

    with pytest.raises(ValueError, match="share feature dim"):
        mod4.FNNTrainer(dataset_dict=bad_dict)