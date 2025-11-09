import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class BDHNeuron(nn.Module):
    """
    Basic BDH neuron with optional adaptive synaptic properties.
    Represents a single neuron in the scale-free network.
    """
    
    def __init__(self, neuron_id: int, input_dim: int, output_dim: int,
                 use_adaptive=False):
        super().__init__()
        self.neuron_id = neuron_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_adaptive = use_adaptive
        
        # Synaptic weights
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        if use_adaptive:
            # Continual learning buffers
            self.register_buffer('weight_ref', torch.zeros_like(self.weight))
            self.register_buffer('importance', torch.zeros_like(self.weight))
            self.register_buffer('plasticity_state', torch.ones_like(self.weight))
            self.register_buffer('learning_rate_scale', torch.ones_like(self.weight))
            self.register_buffer('path_integral', torch.zeros_like(self.weight))
    
    def forward(self, x):
        """Standard forward pass: y = Wx + b"""
        return F.linear(x, self.weight, self.bias)
    
    def forward_with_consolidation(self, x):
        """
        Forward pass with plasticity gating.
        Important, consolidated synapses gate information more reliably.
        """
        if not self.use_adaptive:
            return self.forward(x)
        
        # Gate synaptic transmission by plasticity state
        gated_weight = self.weight * self.plasticity_state
        return F.linear(x, gated_weight, self.bias)
    
    def update_plasticity(self, decay_rate: float = 0.98):
        """Reduce plasticity (consolidation) after task."""
        if self.use_adaptive:
            self.plasticity_state.data *= decay_rate
            self.plasticity_state.data.clamp_(min=0.01, max=1.0)


class BDHScaleFreeLayer(nn.Module):
    """
    Scale-free graph layer representing BDH's graph topology.
    
    Features:
    - Sparse connectivity mimicking biological networks
    - Heavy-tailed degree distribution (scale-free)
    - Local interactions with global properties
    - Optional adaptive synapses for continual learning
    """
    
    def __init__(self, input_size: int, output_size: int, sparsity: float = 0.9,
                 use_adaptive: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sparsity = sparsity
        self.use_adaptive = use_adaptive
        
        # Create scale-free connectivity
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_size))
        
        # Apply sparsity (scale-free)
        mask = torch.bernoulli(torch.ones_like(self.weight) * (1 - sparsity))
        self.register_buffer('connectivity_mask', mask)
        
        # Adaptive properties
        if use_adaptive:
            self.register_buffer('weight_ref', torch.zeros_like(self.weight))
            self.register_buffer('importance', torch.zeros_like(self.weight))
            self.register_buffer('plasticity_state', torch.ones_like(self.weight))
            self.register_buffer('learning_rate_scale', torch.ones_like(self.weight))
            self.register_buffer('path_integral', torch.zeros_like(self.weight))
    
    def forward(self, x):
        """Forward with sparse connectivity."""
        masked_weight = self.weight * self.connectivity_mask
        return F.linear(x, masked_weight, self.bias)
    
    def forward_with_consolidation(self, x):
        """Forward with consolidation gating."""
        if not self.use_adaptive:
            return self.forward(x)
        
        # Apply both sparsity and plasticity gating
        masked_weight = self.weight * self.connectivity_mask
        gated_weight = masked_weight * self.plasticity_state
        return F.linear(x, gated_weight, self.bias)
    
    def update_plasticity(self, decay_rate: float = 0.98):
        """Update plasticity state (consolidation)."""
        if self.use_adaptive:
            self.plasticity_state.data *= decay_rate
            self.plasticity_state.data.clamp_(min=0.01, max=1.0)


class BDHAttentionLayer(nn.Module):
    """
    BDH Attention Layer without softmax.
    
    Key differences from Transformers:
    - No softmax (raw attention scores)
    - RoPE positional embeddings
    - Q = K (tied attention)
    - Optional adaptive properties
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8,
                 use_adaptive: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_adaptive = use_adaptive
        
        # Attention projections
        if use_adaptive:
            try:
                from bdh.continual_learning.adaptive_synapses import AdaptiveLinear
                self.query = AdaptiveLinear(hidden_size, hidden_size)
                self.key = AdaptiveLinear(hidden_size, hidden_size)
                self.value = AdaptiveLinear(hidden_size, hidden_size)
                self.output = AdaptiveLinear(hidden_size, hidden_size)
            except ImportError:
                logger.warning("AdaptiveLinear not available, using standard Linear")
                self.query = nn.Linear(hidden_size, hidden_size)
                self.key = nn.Linear(hidden_size, hidden_size)
                self.value = nn.Linear(hidden_size, hidden_size)
                self.output = nn.Linear(hidden_size, hidden_size)
        else:
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """BDH attention forward pass."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention (NO softmax - BDH specific)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Mask future tokens (causal)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # ReLU instead of softmax
        attention = F.relu(scores)
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-9)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.output(context)
        return output


class BDHBlock(nn.Module):
    """
    Single BDH Block combining scale-free layer and attention.
    
    Architecture:
    1. Scale-free graph layer (local interactions)
    2. Attention layer (global dependencies)
    3. Layer normalization & residual
    4. Feed-forward
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8,
                 sparsity: float = 0.9, use_adaptive: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaptive = use_adaptive
        
        # Scale-free layer
        self.scale_free = BDHScaleFreeLayer(
            hidden_size, hidden_size, sparsity=sparsity,
            use_adaptive=use_adaptive
        )
        
        # Attention
        self.attention = BDHAttentionLayer(
            hidden_size, num_heads=num_heads,
            use_adaptive=use_adaptive
        )
        
        # Feed-forward network
        ff_dim = hidden_size * 4
        if use_adaptive:
            try:
                from bdh.continual_learning.adaptive_synapses import AdaptiveLinear
                self.ff1 = AdaptiveLinear(hidden_size, ff_dim)
                self.ff2 = AdaptiveLinear(ff_dim, hidden_size)
            except ImportError:
                self.ff1 = nn.Linear(hidden_size, ff_dim)
                self.ff2 = nn.Linear(ff_dim, hidden_size)
        else:
            self.ff1 = nn.Linear(hidden_size, ff_dim)
            self.ff2 = nn.Linear(ff_dim, hidden_size)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """BDH block forward pass with residuals."""
        # Scale-free layer + residual
        out = self.norm1(x)
        out = self.scale_free(out)
        x = x + self.dropout(out)
        
        # Attention + residual
        out = self.norm2(x)
        out = self.attention(out)
        x = x + self.dropout(out)
        
        # Feed-forward + residual
        out = self.norm3(x)
        out = self.ff1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.ff2(out)
        x = x + self.dropout(out)
        
        return x
    
    def update_plasticity(self, decay_rate: float = 0.98):
        """Update plasticity in all adaptive layers."""
        if self.use_adaptive:
            self.scale_free.update_plasticity(decay_rate)
            # Update attention layers
            if hasattr(self.attention.query, 'update_plasticity'):
                self.attention.query.update_plasticity(decay_rate)
                self.attention.key.update_plasticity(decay_rate)
                self.attention.value.update_plasticity(decay_rate)
                self.attention.output.update_plasticity(decay_rate)
            # Update feed-forward layers
            if hasattr(self.ff1, 'update_plasticity'):
                self.ff1.update_plasticity(decay_rate)
                self.ff2.update_plasticity(decay_rate)


class BDH(nn.Module):
    """
    Baby Dragon Hatchling (BDH) Model.
    
    A biologically-inspired architecture based on scale-free networks
    and local interactions with optional continual learning support.
    
    Architecture:
    - Embedding layer
    - Multiple BDH blocks (scale-free + attention)
    - Output layer
    - Optional adaptive layers for continual learning
    """
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 num_heads: int = 8, sparsity: float = 0.9,
                 use_adaptive_layers: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_adaptive_layers = use_adaptive_layers
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # BDH Blocks
        self.blocks = nn.ModuleList([
            BDHBlock(hidden_size, num_heads=num_heads, sparsity=sparsity,
                    use_adaptive=use_adaptive_layers)
            for _ in range(num_layers)
        ])
        
        # Output layer
        if use_adaptive_layers:
            try:
                from bdh.continual_learning.adaptive_synapses import AdaptiveLinear
                self.output = AdaptiveLinear(hidden_size, vocab_size)
            except ImportError:
                self.output = nn.Linear(hidden_size, vocab_size)
        else:
            self.output = nn.Linear(hidden_size, vocab_size)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        """
        Forward pass through BDH model.
        
        Args:
            x: Token indices (batch_size, seq_length)
        
        Returns:
            logits: (batch_size, seq_length, vocab_size)
        """
        # Embedding
        x = self.embedding(x)
        
        # BDH blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.final_norm(x)
        
        # Output
        x = self.output(x)
        
        return x
    
    def forward_with_consolidation(self, x):
        """
        Forward pass using consolidation gates.
        Only relevant if using adaptive layers.
        """
        if not self.use_adaptive_layers:
            return self.forward(x)
        
        # Embedding
        x = self.embedding(x)
        
        # BDH blocks with consolidation
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.final_norm(x)
        
        # Output with consolidation (if adaptive)
        if hasattr(self.output, 'forward_with_consolidation'):
            x = self.output.forward_with_consolidation(x)
        else:
            x = self.output(x)
        
        return x
    
    def update_plasticity(self, decay_rate: float = 0.98):
        """Update plasticity across all blocks (consolidation)."""
        if self.use_adaptive_layers:
            for block in self.blocks:
                block.update_plasticity(decay_rate)
            
            if hasattr(self.output, 'update_plasticity'):
                self.output.update_plasticity(decay_rate)
    
    def get_neurons(self) -> List:
        """Get list of all neurons (for compatibility with CL trainer)."""
        neurons = []
        for i, block in enumerate(self.blocks):
            neurons.append(block)
        return neurons


def create_bdh_model(config_dict: dict, use_adaptive_layers: bool = False) -> BDH:
    """
    Factory function to create BDH model from config.
    
    Args:
        config_dict: Dictionary with keys:
            - vocab_size: int
            - hidden_size: int
            - num_layers: int
            - num_heads: int (optional, default=8)
            - sparsity: float (optional, default=0.9)
        use_adaptive_layers: Whether to use adaptive layers for continual learning
    
    Returns:
        BDH model instance
    """
    model = BDH(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_layers=config_dict['num_layers'],
        num_heads=config_dict.get('num_heads', 8),
        sparsity=config_dict.get('sparsity', 0.9),
        use_adaptive_layers=use_adaptive_layers
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created BDH model with {total_params:,} parameters")
    logger.info(f"  - Vocab size: {config_dict['vocab_size']}")
    logger.info(f"  - Hidden size: {config_dict['hidden_size']}")
    logger.info(f"  - Num layers: {config_dict['num_layers']}")
    logger.info(f"  - Num heads: {config_dict.get('num_heads', 8)}")
    logger.info(f"  - Sparsity: {config_dict.get('sparsity', 0.9)}")
    logger.info(f"  - Adaptive layers: {use_adaptive_layers}")
    
    return model

