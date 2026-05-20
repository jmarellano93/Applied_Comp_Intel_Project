"""
Module 4: Problem Model (PM) - FNN Landscape Evaluator

Defines the core Neural Network topologies and handles the inner-loop fitness
evaluation. Enforces strict batching, PyTorch-native metric tracking, and
gradient explosion safeguards.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple
from pydantic import BaseModel, Field
import warnings

warnings.filterwarnings("ignore")


# --- CUSTOM ACTIVATION REPRESENTATIVES ---

class SineActivation(nn.Module):
    """Trigonometric mapping for periodic spatial features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def get_activation(activation_name: str) -> nn.Module:
    """Routes the strictly constrained 6-activation roster."""
    activations = {
        'linear': nn.Identity(),
        'rectification': nn.ReLU(),
        'squashing': nn.Tanh(),
        'smooth': nn.GELU(),
        'aggregation': nn.SiLU(),
        'trigonometric': SineActivation()
    }
    return activations.get(activation_name.lower(), nn.GELU())


# --- TOPOLOGY DEFINITIONS ---

class PhaseA_Shallow_FNN(nn.Module):
    """Discovery Topology: 2 Hidden Layers, 64 Neurons. Prioritizes rapid iteration."""

    def __init__(self, input_dim: int, num_classes: int, activation_name: str = 'smooth'):
        super().__init__()
        act = get_activation(activation_name)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), act,
            nn.Linear(64, 64), act,
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PhaseB_DeepNarrow_FNN(nn.Module):
    """Stress Test Topology: 8 Hidden Layers, 32 Neurons. Tests gradient persistence."""

    def __init__(self, input_dim: int, num_classes: int, activation_name: str = 'smooth'):
        super().__init__()
        act = get_activation(activation_name)
        layers = [nn.Linear(input_dim, 32), act]
        for _ in range(7):
            layers.extend([nn.Linear(32, 32), act])
        layers.append(nn.Linear(32, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PhaseB_Funnel_FNN(nn.Module):
    """Compression Topology: Decreasing layers to test feature bottlenecking."""

    def __init__(self, input_dim: int, num_classes: int, activation_name: str = 'smooth'):
        super().__init__()
        act = get_activation(activation_name)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), act,
            nn.Linear(256, 128), act,
            nn.Linear(128, 64), act,
            nn.Linear(64, 32), act,
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# --- INNER LOOP EVALUATOR ---

class PyTorchEvaluator:
    """
    Manages the PyTorch training loop. Implements mini-batching, native
    PyTorch accuracy metrics (avoiding CPU transfers), and gradient clipping.
    """

    def __init__(self, dataset_dict: Dict[str, torch.Tensor], sigma_squared: float,
                 activation_name: str = 'smooth', topology: str = 'shallow',
                 max_epochs: int = 30, target_acc: float = 0.85, batch_size: int = 64):

        self.X_train = dataset_dict['X_train']
        self.y_train = dataset_dict['y_train']
        self.X_val = dataset_dict['X_val']
        self.y_val = dataset_dict['y_val']

        self.input_dim = self.X_train.shape[1]
        self.num_classes = len(torch.unique(self.y_train))

        # Guard against zero/negative variance mathematically
        self.sigma_squared = max(abs(sigma_squared), 1e-5)
        self.max_epochs = max_epochs
        self.target_acc = target_acc
        self.batch_size = batch_size
        self.activation_name = activation_name

        if topology == 'deep_narrow':
            self.model = PhaseB_DeepNarrow_FNN(self.input_dim, self.num_classes, self.activation_name)
        elif topology == 'funnel':
            self.model = PhaseB_Funnel_FNN(self.input_dim, self.num_classes, self.activation_name)
        else:
            self.model = PhaseA_Shallow_FNN(self.input_dim, self.num_classes, self.activation_name)

        self._inject_custom_variance()

    def _inject_custom_variance(self) -> None:
        """Applies the GP-discovered variance to all linear topologies."""
        std_dev = math.sqrt(self.sigma_squared)
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=std_dev)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _balanced_accuracy_pytorch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Calculates balanced accuracy natively on-device to bypass CPU bottlenecks."""
        accs = []
        for c in range(self.num_classes):
            mask = (y_true == c)
            if mask.sum() > 0:
                accs.append((y_pred[mask] == c).float().mean().item())
        return sum(accs) / len(accs) if accs else 0.0

    def evaluate_fitness(self) -> Tuple[float, int]:
        """
        Executes the batch-driven training loop and returns validation metrics.
        Returns: Tuple of (Max Validation Accuracy, Epochs to Threshold).
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)

        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        epochs_to_threshold = self.max_epochs
        final_val_acc = 0.0

        for epoch in range(self.max_epochs):
            self.model.train()

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Mathematical Stability: Exploding Gradient Safeguard
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Mathematical Stability: NaN Gradient Check
                for param in self.model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        return 0.0, 999

                optimizer.step()

            # Validation Pass
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.X_val)
                _, predicted = torch.max(val_outputs, 1)

                val_acc = self._balanced_accuracy_pytorch(self.y_val, predicted)

                if val_acc >= self.target_acc and epochs_to_threshold == self.max_epochs:
                    epochs_to_threshold = epoch + 1
                final_val_acc = val_acc

        return final_val_acc, epochs_to_threshold