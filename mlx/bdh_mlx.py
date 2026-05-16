# Copyright 2025 Pathway Technology, Inc.
# MLX implementation of Baby Dragon Hatchling (BDH)

import dataclasses
import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n: int, theta: float, dtype=mx.float32) -> mx.array:
    """Generate frequency array for RoPE."""
    def quantize(t, q=2):
        return (t / q).astype(mx.int32).astype(dtype) * q

    arange = mx.arange(0, n, 1, dtype=dtype)
    return (
        1.0
        / (theta ** (quantize(arange) / n))
        / (2 * math.pi)
    )


class Attention(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = get_freqs(N, theta=2**16, dtype=mx.float32).reshape(1, 1, 1, N)

    @staticmethod
    def phases_cos_sin(phases: mx.array) -> Tuple[mx.array, mx.array]:
        """Convert phases to cosine and sine components."""
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = mx.cos(phases)
        phases_sin = mx.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases: mx.array, v: mx.array) -> mx.array:
        """Apply Rotary Position Embedding."""
        # Interleave negative of odd indices with even indices
        v_rot_parts = mx.stack([-v[..., 1::2], v[..., ::2]], axis=-1)
        v_rot = v_rot_parts.reshape(v.shape)
        
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).astype(v.dtype) + (v_rot * phases_sin).astype(v.dtype)

    def __call__(self, Q: mx.array, K: mx.array, V: mx.array) -> mx.array:
        """Forward pass of attention mechanism."""
        assert self.freqs.dtype == mx.float32
        assert K is Q
        _, _, T, _ = Q.shape

        r_phases = (
            mx.arange(0, T, dtype=self.freqs.dtype).reshape(1, 1, -1, 1)
        ) * self.freqs
        
        QR = self.rope(r_phases, Q)
        KR = QR

        # Current attention with causal mask
        scores = (QR @ KR.transpose(0, 1, 3, 2))
        # Apply causal mask (tril with diagonal=-1)
        mask = mx.tril(mx.ones((T, T)), k=-1)
        scores = scores * mask.reshape(1, 1, T, T)
        
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        # Modules (must be initialized first for proper parameter registration)
        self.attn = Attention(config)
        self.ln = nn.LayerNorm(D, affine=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        
        # Trainable parameters (registered via __setattr__)
        self.decoder = mx.random.normal((nh * N, D), scale=0.02)
        self.encoder = mx.random.normal((nh, D, N), scale=0.02)
        self.encoder_v = mx.random.normal((nh, D, N), scale=0.02)
        self.lm_head = mx.random.normal((D, config.vocab_size), scale=0.02)
        self.lm_gate = mx.random.normal((D, 1), scale=0.02)
        
        # Initialize embedding weights
        self.embed.weight = mx.random.normal(self.embed.weight.shape, scale=0.02)

    def __call__(self, idx: mx.array, targets: Optional[mx.array] = None) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass of BDH model."""
        C = self.config
        B, T = idx.shape
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx)
        x = mx.expand_dims(x, axis=1)  # B, 1, T, D

        # Layer normalization helps with training
        x = self.ln(x)

        for level in range(C.n_layer):
            # Hierarchical encoding
            x_latent = x @ self.encoder  # B, nh, T, N

            # Sparse activation
            x_sparse = mx.maximum(x_latent, 0)  # ReLU

            # Attention mechanism
            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
            )
            yKV = self.ln(yKV)

            # Value encoding
            y_latent = yKV @ self.encoder_v
            y_sparse = mx.maximum(y_latent, 0)  # ReLU
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            # Dropout
            xy_sparse = self.drop(xy_sparse)

            # MLP decoder
            # PyTorch: xy_sparse is (B, nh, T, N) -> transpose(1,2) -> (B, T, nh, N)
            # MLX: xy_sparse is (B, nh, T, N) -> transpose(0,2,1,3) -> (B, T, nh, N)
            yMLP = (
                xy_sparse.transpose(0, 2, 1, 3).reshape(B, T, N * nh) @ self.decoder
            )  # B, T, D
            yMLP = mx.expand_dims(yMLP, axis=1)  # B, 1, T, D
            
            y = self.ln(yMLP)
            x = self.ln(x + y)

        # Output projection
        logits = x.reshape(B, T, D) @ self.lm_head

        loss = None
        if targets is not None:
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), 
                targets.reshape(-1),
                reduction='mean'
            )

        return logits, loss

    def generate(
        self,
        idx: mx.array,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> mx.array:
        """Generate text autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                # Top-k filtering
                top_logits = mx.sort(logits, axis=-1)[:, -top_k:]
                kth_value = top_logits[:, [0]]
                logits = mx.where(logits < kth_value, -float('inf'), logits)
            
            # Sample from the distribution
            probs = mx.softmax(logits, axis=-1)
            idx_next = mx.random.categorical(mx.log(probs), num_samples=1)
            idx = mx.concatenate([idx, idx_next], axis=1)
            
        return idx

