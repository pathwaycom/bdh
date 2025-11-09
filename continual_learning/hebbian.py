import torch
import torch.nn as nn
from typing import Dict

class HebbianPlasticity:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def apply_hebbian_update(self, model: nn.Module, layer_activations: Dict[str, torch.Tensor]):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if name in layer_activations:
                    activation = layer_activations[name]
                    if hasattr(module, 'weight'):
                        # Compute Hebbian update: outer product of activations
                        # activation shape [batch_size, input_features]
                        # weight shape [output_features, input_features]
                        pre_syn = activation  # presynaptic inputs
                        post_syn = module.weight.data @ pre_syn.t()  # postsynaptic outputs
                        delta = torch.mm(post_syn.t(), pre_syn)
                        module.weight.data += self.learning_rate * delta / activation.size(0)

    def consolidate_importance(self, model: nn.Module):
        for module in model.modules():
            if hasattr(module, 'importance') and hasattr(module, 'importance_accumulator'):
                module.importance.data += 0.01 * module.importance_accumulator.data
                module.importance_accumulator.zero_()
