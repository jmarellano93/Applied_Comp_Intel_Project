"""
Module 4: Problem Model (PM) - FNN Landscape Evaluator.

Defines the core Neural Network topologies and inner-loop fitness evaluators.
Utilizes object-oriented PyTorch graphs to map discovered initialization
variances to validation accuracies across diverse dataset structures.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# =============================================================================
# FUNCTIONAL BLOCK: Custom Activation Representatives
# WHAT IT DOES: Maps string tokens to PyTorch native activation modules,
#     including a custom Sine activation for periodic feature mappings.
# PARAMETERS: activation_name (e.g., "linear", "rectification", "smooth").
# METHODOLOGICAL JUSTIFICATION: Strict boundary mapping prevents the outer
#     Genetic Algorithm from proposing invalid string tokens that would crash
#     the PyTorch execution graph. GELU ("smooth") is used as the fallback
#     standard due to its non-zero gradient properties for negative inputs.
# =============================================================================
class SineActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def get_activation(activation_name: str) -> nn.Module:
    activations: Dict[str, nn.Module] = {
        "linear": nn.Identity(),
        "rectification": nn.ReLU(),
        "squashing": nn.Tanh(),
        "smooth": nn.GELU(),
        "aggregation": nn.SiLU(),
        "trigonometric": SineActivation(),
    }
    return activations.get(activation_name.lower(), nn.GELU())


# =============================================================================
# FUNCTIONAL BLOCK: Topology Definitions & Factory
# WHAT IT DOES: Defines the FNN architectures (Shallow, Deep Narrow, Funnel)
#     and provides a factory pattern to instantiate them based on arguments.
# PARAMETERS: input_dim (dataset features), num_classes (target categories).
# METHODOLOGICAL JUSTIFICATION: Hardcoding network architectures (e.g.,
#     exactly 2 layers of 64 neurons for Phase A) ensures that the initialization
#     variance is the *only* independent variable affecting the convergence
#     trajectory, providing strict experimental control for the heuristic evaluation.
# =============================================================================
class PhaseA_Shallow_FNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, activation_name: str = "smooth"):
        super().__init__()
        act = get_activation(activation_name)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), act,
            nn.Linear(64, 64), act,
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PhaseB_DeepNarrow_FNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, activation_name: str = "smooth"):
        super().__init__()
        act = get_activation(activation_name)
        layers: List[nn.Module] = [nn.Linear(input_dim, 32), act]
        for _ in range(7):
            layers.extend([nn.Linear(32, 32), act])
        layers.append(nn.Linear(32, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PhaseB_Funnel_FNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, activation_name: str = "smooth"):
        super().__init__()
        act = get_activation(activation_name)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), act,
            nn.Linear(256, 128), act,
            nn.Linear(128, 64), act,
            nn.Linear(64, 32), act,
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _build_fnn_model(
    input_dim: int, num_classes: int, activation_name: str, topology: str,
) -> nn.Module:
    if topology == "deep_narrow":
        return PhaseB_DeepNarrow_FNN(input_dim, num_classes, activation_name)
    if topology == "funnel":
        return PhaseB_Funnel_FNN(input_dim, num_classes, activation_name)
    return PhaseA_Shallow_FNN(input_dim, num_classes, activation_name)


# =============================================================================
# FUNCTIONAL BLOCK: Legacy Evaluator
# WHAT IT DOES: Provides a single-shot execution context for compiling and
#     evaluating a PyTorch model. Evaluates fitness per individual.
# PARAMETERS: sigma_squared, max_epochs, target_acc, batch_size.
# METHODOLOGICAL JUSTIFICATION: This single-shot evaluator is maintained strictly
#     to provide API compatibility for isolated validation environments and tests
#     that do not require the high-throughput amortized pooling of FNNTrainer.
# =============================================================================
class PyTorchEvaluator:
    def __init__(self, dataset_dict: Dict[str, torch.Tensor], sigma_squared: float,
                 activation_name: str = "smooth", topology: str = "shallow",
                 max_epochs: int = 30, target_acc: float = 0.85, batch_size: int = 64,
                 pin_memory: bool = True, num_workers: int = 0, use_amp: bool = True,
                 torch_compile: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_train = dataset_dict["X_train"]
        self.y_train = dataset_dict["y_train"]
        self.X_val = dataset_dict["X_val"]
        self.y_val = dataset_dict["y_val"]
        self.input_dim = self.X_train.shape[1]
        self.num_classes = len(torch.unique(self.y_train))
        self.sigma_squared = max(abs(sigma_squared), 1e-5)
        self.max_epochs = max_epochs
        self.target_acc = target_acc
        self.batch_size = batch_size
        self.activation_name = activation_name
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.use_amp = use_amp and (self.device.type == "cuda")

        self.model = _build_fnn_model(self.input_dim, self.num_classes, self.activation_name, topology)
        self.model.to(self.device)
        self._inject_custom_variance()

        if torch_compile:
            torch._dynamo.config.suppress_errors = True
            self.model = torch.compile(self.model)

    def _inject_custom_variance(self) -> None:
        std_dev = math.sqrt(self.sigma_squared)
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=std_dev)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _balanced_accuracy_pytorch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        accs = []
        for c in range(self.num_classes):
            mask = (y_true == c)
            if mask.sum() > 0:
                accs.append((y_pred[mask] == c).float().mean().item())
        return sum(accs) / len(accs) if accs else 0.0

    def evaluate_fitness(self, return_loss: bool = False) -> tuple:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)

        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=self.pin_memory, num_workers=self.num_workers,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        epochs_to_threshold = self.max_epochs
        final_val_acc = 0.0

        X_val_dev = self.X_val.to(self.device)
        y_val_dev = self.y_val.to(self.device)

        for epoch in range(self.max_epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                for param in self.model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        if return_loss:
                            return 0.0, 999.0, 999.0
                        return 0.0, 999.0
                scaler.step(optimizer)
                scaler.update()

            self.model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    val_outputs = self.model(X_val_dev)
                    val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = self._balanced_accuracy_pytorch(y_val_dev, val_preds)
                final_val_acc = val_acc
                if val_acc >= self.target_acc:
                    epochs_to_threshold = epoch + 1
                    break

        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                final_outputs = self.model(X_val_dev)
                final_loss = criterion(final_outputs, y_val_dev).item()

        if return_loss:
            return float(final_val_acc), float(epochs_to_threshold), float(final_loss)
        return float(final_val_acc), float(epochs_to_threshold)


# =============================================================================
# FUNCTIONAL BLOCK: Pool-Friendly Evaluator (FNNTrainer)
# WHAT IT DOES: A stateful CPU trainer reused across thousands of weight
#     initializations. Amortizes dataset binding and DataLoader instantiation.
# PARAMETERS: dataset_dict, activation_name, topology, max_epochs, batch_size.
# METHODOLOGICAL JUSTIFICATION: In standard execution, instantiating a model
#     and moving data to device carries an O(N) overhead. By allocating the model
#     and dataset tensors in contiguous memory once, and simply calling `reset_weights`
#     per individual, the CPU focuses strictly on matrix multiplication, drastically
#     accelerating the 80,000+ total PyTorch evaluations required in MOGP.
# =============================================================================
class FNNTrainer:
    __slots__ = (
        "X_train", "y_train", "X_val", "y_val",
        "n_train", "input_dim", "num_classes",
        "max_epochs", "target_acc", "batch_size",
        "model", "_linear_params", "_all_params",
        "_val_support", "_val_support_nonzero_mask", "_val_n_classes_present",
        "_criterion",
    )

    def __init__(
        self,
        dataset_dict: Dict[str, torch.Tensor],
        activation_name: str = "smooth",
        topology: str = "shallow",
        max_epochs: int = 30,
        target_acc: float = 0.85,
        batch_size: int = 64,
    ) -> None:
        for key in ("X_train", "y_train", "X_val", "y_val"):
            if key not in dataset_dict:
                raise KeyError(f"dataset_dict missing required key: {key!r}")

        device = torch.device("cpu")

        self.X_train = dataset_dict["X_train"].to(device=device).contiguous()
        self.y_train = dataset_dict["y_train"].to(device=device).long().contiguous()
        self.X_val = dataset_dict["X_val"].to(device=device).contiguous()
        self.y_val = dataset_dict["y_val"].to(device=device).long().contiguous()

        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X_train and y_train must share dim 0.")
        if self.X_val.shape[0] != self.y_val.shape[0]:
            raise ValueError("X_val and y_val must share dim 0.")
        if self.X_train.shape[1] != self.X_val.shape[1]:
            raise ValueError("X_train and X_val must share feature dim.")

        self.n_train = int(self.X_train.shape[0])
        self.input_dim = int(self.X_train.shape[1])
        self.num_classes = int(torch.unique(torch.cat([self.y_train, self.y_val])).numel())

        self.max_epochs = int(max_epochs)
        self.target_acc = float(target_acc)
        self.batch_size = int(batch_size)

        self.model = _build_fnn_model(self.input_dim, self.num_classes, activation_name, topology).to(device)
        self._linear_params = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        self._all_params = list(self.model.parameters())

        support = torch.zeros(self.num_classes, dtype=torch.float32, device=device)
        support.scatter_add_(0, self.y_val, torch.ones_like(self.y_val, dtype=torch.float32))
        self._val_support = support
        self._val_support_nonzero_mask = support > 0
        self._val_n_classes_present = int(self._val_support_nonzero_mask.sum().item())

        self._criterion = nn.CrossEntropyLoss()

    def reset_weights(self, sigma_squared: float) -> None:
        std_dev = math.sqrt(max(abs(float(sigma_squared)), 1e-5))
        with torch.no_grad():
            for m in self._linear_params:
                m.weight.normal_(0.0, std_dev)
                if m.bias is not None:
                    m.bias.zero_()

    def _balanced_accuracy(self, y_pred: torch.Tensor) -> float:
        if self._val_n_classes_present == 0:
            return 0.0
        correct_mask = (y_pred == self.y_val).to(torch.float32)
        correct_per_class = torch.zeros(self.num_classes, dtype=torch.float32, device=self.y_val.device)
        correct_per_class.scatter_add_(0, self.y_val, correct_mask)
        acc_per_class = correct_per_class[self._val_support_nonzero_mask] / \
            self._val_support[self._val_support_nonzero_mask]
        return float(acc_per_class.mean().item())

    def _has_nonfinite_gradients(self) -> bool:
        for p in self._all_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    def evaluate(self) -> Tuple[float, float]:
        optimizer = optim.Adam(self._all_params, lr=0.005)
        epochs_to_threshold = float(self.max_epochs)
        final_val_acc = 0.0
        batch = self.batch_size

        for epoch in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(self.n_train, device=self.X_train.device)

            for start in range(0, self.n_train, batch):
                idx = perm[start:start + batch]
                batch_X = self.X_train.index_select(0, idx)
                batch_y = self.y_train.index_select(0, idx)

                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch_X)
                loss = self._criterion(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self._all_params, max_norm=1.0)
                optimizer.step()

            if self._has_nonfinite_gradients():
                return 0.0, 999.0

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(self.X_val)
                val_preds = torch.argmax(val_logits, dim=1)
            val_acc = self._balanced_accuracy(val_preds)
            final_val_acc = val_acc

            if val_acc >= self.target_acc:
                epochs_to_threshold = float(epoch + 1)
                break

        return float(final_val_acc), float(epochs_to_threshold)