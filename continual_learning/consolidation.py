import torch
import torch.nn as nn
from typing import Dict

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
    """

    def __init__(self, lambda_ewc: float = 1000.0):
        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.is_first_task = True
        self.task_count = 0

    def consolidate_task(self, model: nn.Module, fisher_dict: Dict[str, torch.Tensor]):
        """
        Consolidate after task completion.
        Save reference weights and Fisher importance.
        """
        if not self.is_first_task:
            # Increase penalty for multiple tasks
            self.lambda_ewc *= 1.2

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Sanitize names (replace '.' with '_')
                safe_ref_name = f"_ref_{name}".replace('.', '_')
                safe_fisher_name = f"_fisher_{name}".replace('.', '_')

                # Register buffer for reference weight if does not exist
                if not hasattr(model, safe_ref_name):
                    ref_buffer = torch.zeros_like(param.data)
                    model.register_buffer(safe_ref_name, ref_buffer)
                model._buffers[safe_ref_name].copy_(param.data)

                # Register buffer for Fisher importance if exists in fisher_dict
                if name in fisher_dict:
                    if not hasattr(model, safe_fisher_name):
                        fisher_buffer = torch.zeros_like(param.data)
                        model.register_buffer(safe_fisher_name, fisher_buffer)
                    model._buffers[safe_fisher_name].copy_(fisher_dict[name])

        self.is_first_task = False
        self.task_count += 1

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC loss."""
        if self.is_first_task:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if param.requires_grad:
                safe_ref_name = f"_ref_{name}".replace('.', '_')
                safe_fisher_name = f"_fisher_{name}".replace('.', '_')

                if hasattr(model, safe_ref_name) and hasattr(model, safe_fisher_name):
                    weight_ref = getattr(model, safe_ref_name)
                    fisher = getattr(model, safe_fisher_name)

                    # Weight deviation
                    delta_w = param - weight_ref
                    # EWC penalty
                    ewc_loss = ewc_loss + (fisher * (delta_w ** 2)).sum()

        return (self.lambda_ewc / 2.0) * ewc_loss


class MetaplasticityConsolidation:
    """
    Automatic consolidation through metaplasticity.

    Reduces plasticity of important synapses after task completion.

    Inspired by: Fusi et al. synaptic tagging and capture model.
    """

    def __init__(self, consolidation_rate: float = 0.98):
        self.consolidation_rate = consolidation_rate

    def apply_consolidation(self, model: nn.Module, strength: float = 1.0):
        """Apply consolidation to all adaptive layers."""
        for module in model.modules():
            if hasattr(module, 'plasticity_state'):
                # Compute importance-weighted consolidation
                if hasattr(module, 'importance'):
                    importance = torch.sigmoid(module.importance)
                else:
                    importance = torch.ones_like(module.plasticity_state)

                # Decay rate based on importance
                decay_rate = (
                    self.consolidation_rate +
                    (1 - self.consolidation_rate) * importance
                )
                # Update plasticity
                module.plasticity_state.data *= decay_rate
                module.plasticity_state.data.clamp_(min=0.01, max=1.0)

    def progressive_consolidation(self, model: nn.Module,
                                 epoch: int, num_consolidation_epochs: int = 10):
        """Gradually consolidate over multiple epochs."""
        progress = epoch / num_consolidation_epochs
        for module in model.modules():
            if hasattr(module, 'plasticity_state'):
                target_decay = (
                    self.consolidation_rate +
                    progress * (1 - self.consolidation_rate)
                )
                module.plasticity_state.data *= target_decay
                module.plasticity_state.data.clamp_(min=0.01, max=1.0)
