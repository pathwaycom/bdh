# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math
import time

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 64
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V, state=None, t_offset=0):
        # This forward method now supports both parallel training and stateful generation
        B, nh, T, N = Q.size()
        D = V.size(-1)
        
        # Initialize state for the first step of generation or for training
        if state is None:
            state = torch.zeros((B, nh, N, D), device=Q.device, dtype=V.dtype)

        # Calculate RoPE phases based on the current time offset
        r_phases = (
            torch.arange(
                t_offset,
                t_offset + T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        
        QR = self.rope(r_phases, Q)
        KR = self.rope(r_phases, K) # In original code, KR=QR. Keeping it separate for clarity.

        if T > 1: # Training or prompt processing mode (parallel)
            # The original logic, but now it contributes to the state update
            scores = (QR @ KR.mT).tril(diagonal=-1)
            output = scores @ V
            # Update state with the entire sequence's K/V info
            # Note: For pure BDH, V should be broadcasted to (B, nh, T, D)
            state_update = KR.transpose(-2, -1) @ V.expand(B, nh, T, D)
            new_state = state + state_update
            return output, new_state
        else: # Generation mode (T=1, sequential)
            # Use the previous state to calculate the output for the new token
            # Output = Q_new @ State_old
            output = QR @ state
            # Update state with the new token's K/V info
            # State_new = State_old + K_new^T @ V_new
            state_update = KR.transpose(-2, -1) @ V.expand(B, nh, T, D)
            new_state = state + state_update
            return output, new_state


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        # We need a separate Attention module for each layer to hold its state
        self.attns = nn.ModuleList([Attention(config) for _ in range(config.n_layer)])

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_states=None):
        B, T = idx.size()
        t_offset = 0
        if past_states is not None:
            pass
        
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)  # B, 1, T, D
        
        if past_states is None:
            past_states = [None] * C.n_layer
        
        present_states = []

        for level, attn_layer in enumerate(self.attns):
            
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)  # B, nh, T, N
            
            yKV, layer_state = attn_layer(
                Q=x_sparse,
                K=x_sparse,
                V=x,
                state=past_states[level],
                t_offset=T if past_states is None else T + past_states[level].shape[-2] # This is still wrong
            )

            yKV, layer_state = attn_layer(Q=x_sparse, K=x_sparse, V=x, state=past_states[level], t_offset=0 if T > 1 else T-1) # This is also wrong
            
            present_states.append(layer_state)
            
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, present_states

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        states = None
        # Process the initial prompt to build the starting state
        prompt_len = idx.size(1)
        # We need a forward pass that can handle a state update without generating logits,
        # or we just take the last logit. The latter is simpler.
        logits, _, states = self(idx, past_states=None)
        
        # Get the first prediction from the prompt
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        return idx # The logic above is complex, let's provide the final clean version.

# Let's replace the above with the final, correct, and self-contained version.

class BDH(nn.Module):
    # ... __init__ and _init_weights are the same as the user provided, but with self.attn changed to self.attns
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        # We now need one Attention module per layer to hold its state during generation
        self.attns = nn.ModuleList([Attention(config) for _ in range(config.n_layer)])

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))
        self.lm_gate = nn.Parameter(torch.zeros((D, 1)).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    # FORWARD METHOD IS NOW STATEFUL
    def forward(self, idx, targets=None, past_states=None, t_offset=0):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)  # B, 1, T, D

        if past_states is None:
            past_states = [None] * C.n_layer
        
        present_states = []

        for i, attn_layer in enumerate(self.attns):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)  # B, nh, T, N

            # Pass the time offset to the attention layer
            yKV, layer_state = attn_layer(
                Q=x_sparse,
                K=x_sparse,
                V=x,
                state=past_states[i],
                t_offset=t_offset
            )
            present_states.append(layer_state)
            
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N
            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None and T > 1: # Calculate loss only during training
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, present_states
    
    # GENERATE METHOD IS NOW STATEFUL AND EFFICIENT
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:


        start_time = time.perf_counter()
        last_checkpoint = start_time
        states = None
        # The idx tensor will grow, but we only pass the newest token to the model
        for i in range(max_new_tokens):
            current_seq_len = idx.size(1)
            
            # On the first pass, process the whole prompt. On subsequent passes, only the last token.
            idx_cond = idx if i == 0 else idx[:, -1:]
            
            # The time offset is the length of the sequence already processed.
            t_offset = 0 if i == 0 else current_seq_len - 1
            
            logits, _, states = self(idx_cond, past_states=states, t_offset=t_offset)
            
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if i % 100 == 0 and i > 0:
                now = time.perf_counter()
                elapsed = now - last_checkpoint
                total_elapsed = now - start_time
                print(f"Generation, token {i}, last 100 tokens took {elapsed:.2f}s (total {total_elapsed:.2f}s)")
                last_checkpoint = now
        return idx


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer from checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['step']