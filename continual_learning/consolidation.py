"""
Elastic Weight Consolidation and Metaplasticity

Implements task consolidation and automatic plasticity reduction.

[FIXED] - Lambda scaling capped, better error handling, safer buffer management
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class ElasticWeightConsolidation(nn.Module):
    """
    Elastic Weight Consolidation loss.

    Penalizes changes to important parameters:
    L_EWC = (λ/2) Σ F_i (w_i - w*_i)^2

    Where:
        - F_i: Fisher importance
        - w_i: Current weight
        - w*_i: Reference weight from previous task

    Reference: Kirkpatrick et al., 2017

    [FIXED] Lambda now caps at max_lambda to prevent exponential growth
    """

    def __init__(self, lambda_ewc: float = 1000.0, max_lambda: float = 5000.0):
        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.max_lambda = max_lambda
        self.is_first_task = True
        self.task_count = 0

    def consolidate_task(self, model: nn.Module, fisher_dict: Dict[str, torch.Tensor]):
        """
        Consolidate after task completion.
        Save reference weights and Fisher importance.

        [FIXED] Better name handling and buffer registration
        """
        if not self.is_first_task:
            # Scale lambda but cap it to prevent explosion
            self.lambda_ewc = min(self.lambda_ewc * 1.2, self.max_lambda)

        device = next(model.parameters()).device

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Sanitize names (replace '.' with '_' to avoid nested buffer names)
                safe_name = name.replace('.', '_')
                safe_ref_name = f"_ref_{safe_name}"
                safe_fisher_name = f"_fisher_{safe_name}"

                # Register buffer for reference weight if does not exist
                if not hasattr(model, safe_ref_name):
                    ref_buffer = torch.zeros_like(param.data, device=device)
                    model.register_buffer(safe_ref_name, ref_buffer)

                model._buffers[safe_ref_name].copy_(param.data)

                # Register buffer for Fisher importance if exists in fisher_dict
                if name in fisher_dict:
                    if not hasattr(model, safe_fisher_name):
                        fisher_buffer = torch.zeros_like(param.data, device=device)
                        model.register_buffer(safe_fisher_name, fisher_buffer)

                    fisher_data = fisher_dict[name].to(device)
                    model._buffers[safe_fisher_name].copy_(fisher_data)

        self.is_first_task = False
        self.task_count += 1

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC loss.

        [FIXED] Better device handling and safer buffer access
        """
        if self.is_first_task:
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device)

        device = next(model.parameters()).device
        ewc_loss = torch.tensor(0.0, device=device)

        for name, param in model.named_parameters():
            if param.requires_grad:
                safe_name = name.replace('.', '_')
                safe_ref_name = f"_ref_{safe_name}"
                safe_fisher_name = f"_fisher_{safe_name}"

                if hasattr(model, safe_ref_name) and hasattr(model, safe_fisher_name):
                    weight_ref = getattr(model, safe_ref_name)
                    fisher = getattr(model, safe_fisher_name)

                    # Weight deviation
                    delta_w = param - weight_ref

                    # EWC penalty: F * (w - w*)^2
                    ewc_loss = ewc_loss + (fisher * (delta_w ** 2)).sum()

        return (self.lambda_ewc / 2.0) * ewc_loss


class MetaplasticityConsolidation(nn.Module):
    """
    Automatic consolidation through metaplasticity.

    Reduces plasticity of important synapses after task completion.
    Inspired by: Fusi et al. synaptic tagging and capture model.

    [FIXED] Better handling of modules without consolidation support
    """

    def __init__(self, consolidation_rate: float = 0.98):
        super().__init__()
        if not (0.0 <= consolidation_rate <= 1.0):
            raise ValueError("consolidation_rate must be in [0, 1]")
        self.consolidation_rate = consolidation_rate

    def apply_consolidation(self, model: nn.Module, strength: float = 1.0):
        """
        Apply consolidation to all adaptive layers.

        Args:
            model: Neural network model
            strength: Consolidation strength multiplier (0-1)
        """
        if strength < 0 or strength > 1:
            raise ValueError("strength must be in [0, 1]")

        for module in model.modules():
            if hasattr(module, 'plasticity_state') and hasattr(module, 'update_plasticity_state'):
                # Compute importance-weighted consolidation
                if hasattr(module, 'importance') and module.importance is not None:
                    importance = torch.sigmoid(module.importance)
                else:
                    importance = torch.ones_like(module.plasticity_state)

                # Decay rate based on importance and strength
                decay_rate = (
                    self.consolidation_rate +
                    (1 - self.consolidation_rate) * importance * strength
                )

                # Update plasticity
                module.plasticity_state.data *= decay_rate
                module.plasticity_state.data.clamp_(min=0.01, max=1.0)

    def progressive_consolidation(self, model: nn.Module,
                                 epoch: int, num_consolidation_epochs: int = 10):
        """
        Gradually consolidate over multiple epochs.

        Args:
            model: Neural network model
            epoch: Current epoch in consolidation
            num_consolidation_epochs: Total consolidation epochs
        """
        if num_consolidation_epochs <= 0:
            raise ValueError("num_consolidation_epochs must be > 0")

        progress = min(epoch / num_consolidation_epochs, 1.0)

        for module in model.modules():
            if hasattr(module, 'plasticity_state') and hasattr(module, 'update_plasticity_state'):
                if hasattr(module, 'importance') and module.importance is not None:
                    importance = torch.sigmoid(module.importance)
                else:
                    importance = torch.ones_like(module.plasticity_state)

                # Progressive decay
                target_decay = (
                    self.consolidation_rate +
                    progress * (1 - self.consolidation_rate) * importance
                )

                module.plasticity_state.data *= target_decay
                module.plasticity_state.data.clamp_(min=0.01, max=1.0)