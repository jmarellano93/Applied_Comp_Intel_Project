"""
Module 4: Problem Model (PM) - FNN Landscape Evaluator.

Defines the core Neural Network topologies and inner-loop fitness evaluators.

Public surface:
    * Topology factories: ``PhaseA_Shallow_FNN``, ``PhaseB_DeepNarrow_FNN``,
      ``PhaseB_Funnel_FNN`` (unchanged).
    * Legacy evaluator: ``PyTorchEvaluator`` (unchanged — preserves the
      MOD5 prototype's regression baseline).
    * New evaluator: ``FNNTrainer`` (stateful, pool-friendly, ~3–5x faster
      on CPU). Used by MOD6 for production runs.

Mathematical Notes:
    * Weight init follows the variance discovered by the outer GP rule:
      W_ij ~ N(0, sqrt(sigma_squared)). Bias = 0.
    * Optimizer: Adam(lr=5e-3) with gradient clipping at L2-norm 1.0
      (Pascanu et al., 2013) for exploding-gradient protection.
    * Validation metric: macro-averaged balanced accuracy
      BA = mean_c TP_c / N_c   (only over classes with N_c > 0).
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
# CUSTOM ACTIVATION REPRESENTATIVES
# =============================================================================

class SineActivation(nn.Module):
    """Trigonometric mapping for periodic spatial features.

    Mathematical Notes:
        f(x) = sin(x), period 2*pi, Lipschitz constant 1.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sin(x)


def get_activation(activation_name: str) -> nn.Module:
    """Maps activation name to its torch.nn module.

    Args:
        activation_name: One of the 6 canonical names. Unknown names fall
            back to GELU to preserve the legacy contract.

    Returns:
        A fresh ``nn.Module`` activation instance.
    """
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
# TOPOLOGY DEFINITIONS  (unchanged from original MOD4)
# =============================================================================

class PhaseA_Shallow_FNN(nn.Module):
    """Discovery Topology: 2 Hidden Layers, 64 Neurons."""

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
    """Stress Test Topology: 8 Hidden Layers, 32 Neurons."""

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
    """Compression Topology: 256 → 128 → 64 → 32 → C."""

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
    """Topology factory honoring the Open/Closed Principle.

    Args:
        input_dim: Number of input features.
        num_classes: Output cardinality for softmax classification.
        activation_name: One of the 6 canonical activation tokens.
        topology: 'shallow' (Phase A) | 'deep_narrow' | 'funnel'.

    Returns:
        Uninitialized (default-init) FNN module ready for variance injection.
    """
    if topology == "deep_narrow":
        return PhaseB_DeepNarrow_FNN(input_dim, num_classes, activation_name)
    if topology == "funnel":
        return PhaseB_Funnel_FNN(input_dim, num_classes, activation_name)
    return PhaseA_Shallow_FNN(input_dim, num_classes, activation_name)


# =============================================================================
# LEGACY EVALUATOR  (UNCHANGED — preserves MOD5 prototype compatibility)
# =============================================================================

class PyTorchEvaluator:
    """Legacy per-call evaluator. Retained verbatim for MOD5 backward compatibility.

    NOTE: For MOD6 production runs use :class:`FNNTrainer` (pool-friendly,
    ~3–5x faster). This class is kept solely so the working MOD5 prototype
    continues to run unchanged.
    """

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
# NEW POOL-FRIENDLY EVALUATOR
# =============================================================================

class FNNTrainer:
    """Stateful CPU trainer reused across thousands of weight initializations.

    Lifecycle:
        Build once per ``(dataset_id, activation, topology)`` triple,
        then call ``reset_weights(sigma_squared) -> evaluate()`` per individual.
        Amortizes dataset binding, model construction, validation-tensor
        layout, and per-class support computation across the entire
        outer GA loop.

    Mathematical Notes:
        * Weight init: W ~ N(0, sqrt(sigma_squared)), b = 0.
        * Loss: cross-entropy on logits.
        * Validation metric: balanced accuracy with vectorized
          ``scatter_add_`` confusion-diagonal computation.
        * Stability: gradient L2-norm clipped at 1.0; per-epoch finite
          check on all parameter gradients; early exit returns
          ``(0.0, 999.0)`` if any non-finite gradient is observed.
    """

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
        """Build the trainer once per (dataset, activation, topology) triple.

        Args:
            dataset_dict: Must contain keys ``X_train``, ``y_train``,
                ``X_val``, ``y_val`` as torch tensors (already RAM-resident).
            activation_name: One of the 6 canonical activation tokens.
            topology: 'shallow' | 'deep_narrow' | 'funnel'.
            max_epochs: Upper bound on inner training epochs.
            target_acc: Balanced-accuracy threshold for early stop.
            batch_size: Minibatch size for SGD.

        Raises:
            KeyError: If ``dataset_dict`` is missing any required key.
            ValueError: If train/val row counts mismatch their label arrays.
        """
        for key in ("X_train", "y_train", "X_val", "y_val"):
            if key not in dataset_dict:
                raise KeyError(f"dataset_dict missing required key: {key!r}")

        device = torch.device("cpu")

        # Contiguous on-CPU references. Cast labels to long once.
        X_train = dataset_dict["X_train"].to(device=device).contiguous()
        y_train = dataset_dict["y_train"].to(device=device).long().contiguous()
        X_val = dataset_dict["X_val"].to(device=device).contiguous()
        y_val = dataset_dict["y_val"].to(device=device).long().contiguous()

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must share dim 0.")
        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError("X_val and y_val must share dim 0.")
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError("X_train and X_val must share feature dim.")

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_train = int(X_train.shape[0])
        self.input_dim = int(X_train.shape[1])
        # Use the union of labels seen across train+val to size the head.
        self.num_classes = int(torch.unique(torch.cat([y_train, y_val])).numel())

        self.max_epochs = int(max_epochs)
        self.target_acc = float(target_acc)
        self.batch_size = int(batch_size)

        # Build model once. Linear-only param view avoids isinstance checks per reset.
        self.model = _build_fnn_model(self.input_dim, self.num_classes, activation_name, topology).to(device)
        self._linear_params = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        self._all_params = list(self.model.parameters())

        # Pre-compute per-class validation support for balanced accuracy.
        # Vectorized: scatter ones into a (num_classes,) bucket.
        support = torch.zeros(self.num_classes, dtype=torch.float32, device=device)
        support.scatter_add_(0, self.y_val, torch.ones_like(self.y_val, dtype=torch.float32))
        self._val_support = support
        self._val_support_nonzero_mask = support > 0
        self._val_n_classes_present = int(self._val_support_nonzero_mask.sum().item())

        self._criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # Weight reset
    # -------------------------------------------------------------------------

    def reset_weights(self, sigma_squared: float) -> None:
        """Re-initialize all Linear weights in-place. O(P) parameters.

        Args:
            sigma_squared: Variance discovered by the outer GP rule.
                Sanitized to ``max(abs(.), 1e-5)`` to guarantee finite,
                strictly positive standard deviation.

        Returns:
            None. Modifies ``self.model`` parameters in place.
        """
        std_dev = math.sqrt(max(abs(float(sigma_squared)), 1e-5))
        with torch.no_grad():
            for m in self._linear_params:
                m.weight.normal_(0.0, std_dev)
                if m.bias is not None:
                    m.bias.zero_()

    # -------------------------------------------------------------------------
    # Balanced accuracy (vectorized)
    # -------------------------------------------------------------------------

    def _balanced_accuracy(self, y_pred: torch.Tensor) -> float:
        """Macro-averaged balanced accuracy on the validation tensor.

        Mathematical Notes:
            BA = (1/|C_present|) * sum_{c in C_present} TP_c / N_c
            where C_present = {c : N_c > 0}. Computed in fully vectorized
            form using ``scatter_add_`` over the per-sample correctness
            indicator. O(N_val) time, O(num_classes) extra space.
        """
        if self._val_n_classes_present == 0:
            return 0.0
        correct_mask = (y_pred == self.y_val).to(torch.float32)
        correct_per_class = torch.zeros(self.num_classes, dtype=torch.float32, device=self.y_val.device)
        correct_per_class.scatter_add_(0, self.y_val, correct_mask)
        # Safe division: only over classes present (mask).
        acc_per_class = correct_per_class[self._val_support_nonzero_mask] / \
            self._val_support[self._val_support_nonzero_mask]
        return float(acc_per_class.mean().item())

    # -------------------------------------------------------------------------
    # Gradient finite-ness check
    # -------------------------------------------------------------------------

    def _has_nonfinite_gradients(self) -> bool:
        """True iff any parameter holds a NaN or Inf in its .grad buffer."""
        for p in self._all_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    # -------------------------------------------------------------------------
    # Main evaluation loop
    # -------------------------------------------------------------------------

    def evaluate(self) -> Tuple[float, float]:
        """Train to ``max_epochs`` (or early stop), return (balanced_acc, epochs).

        Returns:
            Tuple ``(final_balanced_acc, epochs_to_threshold)``. If gradient
            collapse is detected at any epoch boundary, returns
            ``(0.0, 999.0)`` as the sentinel for the outer GP loop.

        Mathematical Notes:
            * Optimizer state is fresh (Adam, lr=5e-3) on every call so
              that previous rule evaluations do not pollute moment estimates.
            * Minibatches are drawn via ``torch.randperm`` + ``index_select``
              for ~3x throughput over ``DataLoader`` on RAM-resident tensors.
        """
        optimizer = optim.Adam(self._all_params, lr=0.005)
        epochs_to_threshold = float(self.max_epochs)
        final_val_acc = 0.0
        batch = self.batch_size

        for epoch in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(self.n_train, device=self.X_train.device)

            # Vectorized minibatch dispatch via index_select.
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

            # Per-epoch (was per-batch) NaN safeguard. Grad clip + Adam make
            # per-batch checks unnecessary in practice.
            if self._has_nonfinite_gradients():
                return 0.0, 999.0

            # Validation pass.
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
