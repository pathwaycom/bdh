import torch
import torch.nn as nn
from typing import Optional


class AdaptiveSynapse(nn.Module):
    """
    Enhanced synaptic connection with consolidation support.
    
    Attributes:
        weight: Current synaptic strength
        weight_ref: Reference weight from last task
        importance: Task importance (Fisher or path integral)
        plasticity_state: Consolidation level (0-1)
        learning_rate_scale: Per-synapse learning rate modifier
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        # Main parameters
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
        # Continual learning state
        self.register_buffer('weight_ref', 
                           torch.zeros_like(self.weight, dtype=dtype))
        self.register_buffer('importance', 
                           torch.zeros_like(self.weight, dtype=dtype))
        self.register_buffer('plasticity_state', 
                           torch.ones((out_features, in_features), dtype=dtype))
        self.register_buffer('learning_rate_scale', 
                           torch.ones_like(self.weight, dtype=dtype))
        
        # Online importance accumulators
        self.register_buffer('path_integral', 
                           torch.zeros_like(self.weight, dtype=dtype))
        self.register_buffer('importance_accumulator', 
                           torch.zeros_like(self.weight, dtype=dtype))
        self.register_buffer('prev_weight', 
                           torch.zeros_like(self.weight, dtype=dtype))
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor(5.0)))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.sqrt(torch.tensor(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Standard linear forward pass."""
        return torch.nn.functional.linear(input, self.weight, self.bias)
    
    def forward_with_consolidation(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with plasticity gating.
        Consolidated synapses transmit more reliably.
        """
        # Gate weights by plasticity state
        gated_weight = self.weight * self.plasticity_state
        return torch.nn.functional.linear(input, gated_weight, self.bias)
    
    def update_plasticity_state(self, decay_rate: float = 0.98):
        """
        Reduce plasticity after task (consolidation).
        
        Args:
            decay_rate: Multiplicative decay (0.98 = 2% decay)
        """
        self.plasticity_state.data *= decay_rate
        self.plasticity_state.data.clamp_(min=0.01, max=1.0)
    
    def consolidate_weights(self):
        """Save current weights as reference for next task."""
        self.weight_ref.copy_(self.weight.data)


class AdaptiveLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with consolidation support."""
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True):
        super().__init__(in_features, out_features, bias)
        
        # Continual learning buffers
        self.register_buffer('weight_ref', 
                           torch.zeros_like(self.weight, dtype=self.weight.dtype))
        self.register_buffer('importance', 
                           torch.zeros_like(self.weight, dtype=self.weight.dtype))
        self.register_buffer('plasticity_state', 
                           torch.ones_like(self.weight, dtype=self.weight.dtype))
        self.register_buffer('learning_rate_scale', 
                           torch.ones_like(self.weight, dtype=self.weight.dtype))
        
        self.register_buffer('path_integral', 
                           torch.zeros_like(self.weight, dtype=self.weight.dtype))
        self.register_buffer('importance_accumulator', 
                           torch.zeros_like(self.weight, dtype=self.weight.dtype))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.weight, self.bias)
    
    def forward_with_consolidation(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with consolidation gating."""
        gated_weight = self.weight * self.plasticity_state
        return nn.functional.linear(input, gated_weight, self.bias)
    
    def update_plasticity_state(self, decay_rate: float = 0.98):
        self.plasticity_state.data *= decay_rate
        self.plasticity_state.data.clamp_(min=0.01, max=1.0)
    
    def consolidate_weights(self):
        self.weight_ref.copy_(self.weight.data)
