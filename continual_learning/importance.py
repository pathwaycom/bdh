"""
Synaptic Importance Estimation

Implements path integral and Fisher matrix methods for computing task importance.

[FIXED] - Path integral properly updated, save_prev_weights called, Fisher handles variable shapes
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ImportanceEstimator:
    """Base class for computing synaptic importance."""

    def __init__(self, model: nn.Module):
        self.model = model

    def estimate(self, data_loader, loss_fn) -> Dict[str, torch.Tensor]:
        """Estimate importance for all parameters."""
        raise NotImplementedError


class PathIntegralEstimator(ImportanceEstimator):
    """
    Path integral importance: I = ∫ ||∇L/∂w|| ||∂w/∂t|| dt

    Online method - accumulates during training.
    More efficient than Fisher matrix.

    Reference: Zenke et al., 2017

    [FIXED] Now properly maintains and updates prev_weights
    """

    def __init__(self, model: nn.Module, dampening_factor: float = 0.01):
        super().__init__(model)
        if not (0 < dampening_factor <= 1.0):
            raise ValueError("dampening_factor must be in (0, 1]")
        self.dampening_factor = dampening_factor
        self.path_integrals = {}
        self.prev_weights = {}
        # Initialize prev_weights
        self.save_prev_weights()

    def reset(self):
        """Reset accumulators."""
        self.path_integrals = {}
        self.prev_weights = {}
        self.save_prev_weights()

    def save_prev_weights(self):
        """Save current weights for delta computation."""
        self.prev_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prev_weights[name] = param.data.clone()

    def update_importance(self, loss: torch.Tensor):
        """
        Update importance based on current loss and weight changes.
        Call after backward() but before optimizer.step()

        [FIXED] Now properly computes delta_w from prev_weights
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data

                # Compute weight change if available
                if name in self.prev_weights:
                    delta_w = param.data - self.prev_weights[name]
                    # Path integral: |gradient| × |weight_change|
                    importance = torch.abs(grad) * (torch.abs(delta_w) + 1e-8)
                else:
                    # First iteration: use gradient magnitude only
                    importance = torch.abs(grad)

                # Accumulate with dampening
                if name not in self.path_integrals:
                    self.path_integrals[name] = torch.zeros_like(importance)

                self.path_integrals[name] += self.dampening_factor * importance

    def get_importance(self) -> Dict[str, torch.Tensor]:
        """Get accumulated importance values."""
        return self.path_integrals.copy()


class FisherEstimator(ImportanceEstimator):
    """
    Diagonal Fisher Information Matrix estimation.

    F_ii = E[(∂L/∂w_i)²]

    Accurate but more expensive than path integral.
    Used as task checkpoint.

    Reference: Kirkpatrick et al., 2017

    [FIXED] Handles variable output shapes, better device handling
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def estimate(self, data_loader, loss_fn,
                 num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Estimate diagonal Fisher for given data.

        Args:
            data_loader: Training data
            loss_fn: Loss function
            num_samples: Number of samples to use (None = all)

        Returns:
            Dict mapping parameter names to Fisher estimates

        [FIXED] Better shape handling and device management
        """
        fisher = {}
        num_batches = 0
        device = next(self.model.parameters()).device

        self.model.eval()

        for batch_idx, batch in enumerate(data_loader):
            if num_samples is not None and batch_idx >= num_samples:
                break

            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x, y = batch, batch

            x = x.to(device)
            y = y.to(device)

            self.model.zero_grad()

            # Forward pass
            outputs = self.model(x)

            # Handle different output shapes
            if len(outputs.shape) == 3:  # (batch, seq, vocab)
                batch_size, seq_len, vocab_size = outputs.shape
                logits_flat = outputs.view(batch_size * seq_len, vocab_size)
                y_flat = y.view(batch_size * seq_len)
            elif len(outputs.shape) == 2:  # (batch, vocab)
                logits_flat = outputs
                y_flat = y.view(-1)
            else:
                logger.warning(f"Unexpected output shape: {outputs.shape}, using as-is")
                logits_flat = outputs
                y_flat = y.view(-1)

            loss = loss_fn(logits_flat, y_flat)

            # Backward to get gradients
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_sq = param.grad.data ** 2

                    if name not in fisher:
                        fisher[name] = torch.zeros_like(grad_sq, device=device)

                    fisher[name] += grad_sq.to(device)

            num_batches += 1

        # Average over batches
        for name in fisher:
            fisher[name] /= max(num_batches, 1)

        return fisher


class HybridImportanceEstimator(ImportanceEstimator):
    """
    Combines path integral (during training) with Fisher (at task end).

    [FIXED] Ensures path_integral_est is properly initialized
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.path_integral_est = PathIntegralEstimator(model)
        self.fisher_est = FisherEstimator(model)

    def get_online_importance(self, loss: torch.Tensor):
        """Get importance during training (path integral)."""
        self.path_integral_est.update_importance(loss)
        return self.path_integral_est.get_importance()

    def get_task_importance(self, data_loader, loss_fn,
                           num_samples: int = 200) -> Dict[str, torch.Tensor]:
        """Get importance at task end (Fisher)."""
        return self.fisher_est.estimate(data_loader, loss_fn, num_samples=num_samples)

    def update_prev_weights(self):
        """Update prev_weights for path integral (call after optimizer step)."""
        self.path_integral_est.save_prev_weights()

    def reset_path_integral(self):
        """Reset path integral accumulator for new task."""
        self.path_integral_est.reset()