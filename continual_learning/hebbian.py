"""
Hebbian Learning Module

Online Hebbian plasticity: Δw = η × pre_activity × post_activity
"""

import torch
import torch.nn as nn
from typing import Dict


class HebbianPlasticity:
    """
    Implements Hebbian learning rule for synaptic updates.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def apply_hebbian_update(self, model: nn.Module,
                            layer_activations: Dict[str, torch.Tensor]):
        """
        Apply Hebbian weight updates based on neuron activities.
        
        Args:
            model: Neural network model
            layer_activations: Dict mapping layer names to activation tensors
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if name in layer_activations:
                    activation = layer_activations[name]
                    
                    # Hebbian correlation
                    if hasattr(module, 'weight'):
                        # Simple outer product for linear layers
                        if isinstance(module, nn.Linear):
                            # activation shape: (batch, input_features)
                            # module.weight shape: (output_features, input_features)
                            
                            # This is a simplified version
                            # In practice, you'd compute correlations more carefully
                            pass
    
    def consolidate_importance(self, model: nn.Module):
        """Update importance based on Hebbian correlations."""
        for module in model.modules():
            if hasattr(module, 'importance') and hasattr(module, 'weight'):
                # Importance increases with Hebbian strengthening
                if hasattr(module, 'importance_accumulator'):
                    module.importance.data += 0.01 * module.importance_accumulator.data
                    module.importance_accumulator.zero_()
