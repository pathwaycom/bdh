# Copyright 2025 Pathway Technology, Inc.

"""
BDH (Binary Data Handler) Model Implementation

This module implements a neural network architecture designed for processing binary data sequences.
The model uses a transformer-like architecture with attention mechanisms to learn patterns in byte sequences.
This is useful for tasks like text generation, where we treat text as sequences of bytes.

Key concepts for ML beginners:
- Neural networks learn to predict the next item in a sequence by finding patterns in data
- Attention mechanisms allow the model to focus on relevant parts of the input when making predictions
- Embeddings convert discrete tokens (like bytes) into continuous vectors that the network can process
"""

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    """
    Configuration class for the BDH model architecture.
    
    This dataclass stores all the hyperparameters (settings) that control the model's structure.
    Hyperparameters are values set before training that determine how the model learns.
    Changing these values changes the model's capacity (ability to learn) and training behavior.
    
    Attributes:
        n_layer: Number of transformer layers in the model. Each layer processes the input
                 sequentially, allowing the model to learn increasingly complex patterns.
                 More layers = more capacity but slower training and more memory usage.
                 
        n_embd: Embedding dimension - the size of vectors used to represent each byte.
                Higher values allow richer representations but require more computation.
                Think of this as how many "features" we use to describe each byte.
                
        dropout: Dropout probability - randomly sets some neurons to zero during training.
                 This prevents overfitting (memorizing training data instead of learning patterns).
                 Value between 0 (no dropout) and 1 (all neurons dropped). 0.1 means 10% are dropped.
                 
        n_head: Number of attention heads. Attention heads allow the model to focus on different
                aspects of the input simultaneously (like looking at syntax and semantics separately).
                More heads = more parallel processing but more parameters.
                
        mlp_internal_dim_multiplier: Controls the size of internal layers in the feed-forward network.
                                      This is multiplied by n_embd to determine the hidden layer size.
                                      Larger values = more capacity in the MLP (Multi-Layer Perceptron).
                                      
        vocab_size: Size of the vocabulary - number of unique tokens the model can process.
                    For byte-level models, this is 256 (one for each possible byte value 0-255).
                    The model learns to predict which of these 256 values comes next.
    """
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    """
    Generate frequency values for Rotary Position Embedding (RoPE).
    
    RoPE is a technique that encodes position information directly into attention calculations.
    Instead of adding position embeddings, we rotate the query and key vectors based on their positions.
    This helps the model understand the order of elements in sequences.
    
    The frequencies determine how fast the rotation changes as we move through positions.
    Higher frequencies = faster rotation = model can distinguish positions that are closer together.
    
    Args:
        n: Number of frequency components to generate. More components = finer-grained position encoding.
        theta: Base frequency parameter. Higher values = slower rotation = model focuses on longer-range patterns.
        dtype: Data type for the tensor (e.g., float32). Important for numerical precision.
        
    Returns:
        A tensor of frequencies that will be used to rotate attention vectors based on position.
    """
    def quantize(t, q=2):
        """
        Quantize (round down to nearest multiple of q) the input tensor.
        
        Quantization reduces the number of unique frequency values, which can help with
        generalization and reduce overfitting to specific positions.
        q=2 means we round down to the nearest even number.
        """
        return (t / q).floor() * q

    # Generate frequencies using a geometric progression (exponential decay)
    # The formula creates frequencies that decrease exponentially, which is useful because
    # position differences matter more for nearby tokens than distant ones
    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    """
    Attention mechanism with Rotary Position Embedding (RoPE).
    
    Attention is the core mechanism that allows the model to focus on relevant parts of the input
    when making predictions. For example, when predicting the next word after "The cat sat on the",
    the model should pay attention to "cat" and "sat" to predict "mat" or "floor".
    
    This implementation uses RoPE, which encodes position information through rotations rather than
    addition. This is more mathematically elegant and often performs better than standard position embeddings.
    
    The attention mechanism works by:
    1. Computing how much each position should attend to every other position (attention scores)
    2. Using these scores to weight and combine the value vectors
    3. This creates a context-aware representation that considers the entire sequence
    """
    def __init__(self, config):
        """
        Initialize the attention module.
        
        Sets up the frequency buffers needed for RoPE. These frequencies are pre-computed
        and stored as buffers (non-trainable parameters) since they don't change during training.
        """
        super().__init__()
        self.config = config
        nh = config.n_head  # Number of attention heads
        D = config.n_embd   # Embedding dimension
        # N is the dimension of the expanded space for attention computation
        # This is larger than D to give the model more capacity in the attention mechanism
        N = config.mlp_internal_dim_multiplier * D // nh
        # Store frequencies as a buffer (non-trainable, but part of the model state)
        # The view(1, 1, 1, N) reshapes for broadcasting across batches, heads, and sequence positions
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        """
        Convert phase values to cosine and sine components for rotation.
        
        In RoPE, we rotate vectors using rotation matrices, which are built from cos and sin values.
        The phase represents how much to rotate (based on position and frequency).
        
        The modulo operation (phases % 1) ensures phases are in [0, 1), then we scale to [0, 2π)
        because trigonometric functions are periodic with period 2π.
        
        Args:
            phases: Tensor of phase values (position * frequency for each position-frequency pair)
            
        Returns:
            Tuple of (cosine, sine) tensors that will be used to rotate the attention vectors
        """
        # Normalize phases to [0, 1) range, then scale to [0, 2π) for trigonometric functions
        phases = (phases % 1) * (2 * math.pi)
        # Compute cosine and sine components for rotation
        # These will be used to rotate vectors: rotated = original * cos + rotated_original * sin
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        """
        Apply Rotary Position Embedding (RoPE) to a vector.
        
        RoPE rotates pairs of dimensions in the vector based on position. This encodes position
        information directly into the vector representation, allowing the model to understand
        where each token is in the sequence.
        
        The rotation is done in 2D planes: for dimensions [0, 1, 2, 3, ...], we rotate
        (0,1), (2,3), (4,5), etc. as pairs. This is more efficient than rotating all dimensions.
        
        Args:
            phases: Phase values (position * frequency) that determine rotation angles
            v: The vector to rotate (typically query or key vectors in attention)
            
        Returns:
            The rotated vector, which now contains position information encoded through rotation
        """
        # Create rotated version by swapping pairs and negating every other element
        # [..., 1::2] gets odd indices (1, 3, 5, ...), [..., ::2] gets even indices (0, 2, 4, ...)
        # The negative sign on odd indices creates the rotation effect
        # This is equivalent to rotating in 2D planes: (dim0, dim1), (dim2, dim3), etc.
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        # Get rotation coefficients (cos and sin) based on position-dependent phases
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        # Apply rotation: combine original vector (scaled by cos) with rotated vector (scaled by sin)
        # This is the standard 2D rotation formula: x' = x*cos - y*sin, y' = x*sin + y*cos
        # The result maintains the vector's magnitude but changes its direction based on position
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        """
        Compute attention with RoPE.
        
        Attention computes how much each position should focus on every other position.
        The output is a weighted combination of value vectors, where weights come from
        how similar query and key vectors are (after rotation with position information).
        
        Args:
            Q: Query vectors - "what am I looking for?" (shape: batch, heads, sequence_length, dim)
            K: Key vectors - "what do I contain?" (here, K is the same as Q for self-attention)
            V: Value vectors - "what information do I provide?" (shape: batch, heads, sequence_length, dim)
            
        Returns:
            Attention output - context-aware representations weighted by attention scores
        """
        # Ensure frequencies are in float32 for numerical stability
        assert self.freqs.dtype == torch.float32
        # This is self-attention: queries and keys come from the same source
        assert K is Q
        # Extract sequence length T from the input shape
        # Shape is (batch, heads, sequence_length, dimension)
        _, _, T, _ = Q.size()

        # Compute rotation phases for each position in the sequence
        # We create a tensor of positions [0, 1, 2, ..., T-1] and multiply by frequencies
        # This gives us position-dependent rotation angles for each frequency component
        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)  # Reshape for broadcasting: (1, 1, T, 1)
        ) * self.freqs  # Broadcast multiply: each position gets rotated by all frequencies
        # Apply RoPE to queries and keys to encode position information
        QR = self.rope(r_phases, Q)
        KR = QR  # Since K == Q, rotated keys are the same as rotated queries

        # Compute attention scores: how similar is each query to each key?
        # QR @ KR.mT computes dot products between all query-key pairs
        # tril(diagonal=-1) creates a lower triangular matrix, ensuring each position only
        # attends to previous positions (causal masking - can't see the future when predicting)
        scores = (QR @ KR.mT).tril(diagonal=-1)
        # Weight and combine value vectors using attention scores
        # Higher scores = more attention = that value contributes more to the output
        return scores @ V


class BDH(nn.Module):
    """
    BDH (Baby Dragon Hatchling) Model - A transformer-like architecture for byte sequences.
    
    This model processes sequences of bytes (0-255) and learns to predict the next byte.
    It uses a sparse attention mechanism where the model learns to encode and decode
    information through learned projection matrices.
    
    Architecture overview:
    1. Embedding: Convert byte tokens to dense vectors
    2. Multiple transformer layers: Each layer refines the representation
    3. Language model head: Converts final representation to probability distribution over bytes
    
    The "sparse" aspect comes from using ReLU activation, which creates sparse (mostly zero)
    representations that can be more efficient and interpretable than dense representations.
    """
    def __init__(self, config: BDHConfig):
        """
        Initialize the BDH model with the given configuration.
        
        Creates all the learnable parameters (weights) that will be optimized during training.
        These parameters start with small random values and are updated via gradient descent
        to minimize prediction error on the training data.
        """
        super().__init__()
        # Ensure vocabulary size is specified (needed for embedding and output layers)
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head  # Number of attention heads
        D = config.n_embd   # Embedding dimension
        # N is the expanded dimension for sparse representations
        # Larger than D to allow the model to represent information in a higher-dimensional space
        N = config.mlp_internal_dim_multiplier * D // nh
        
        # Decoder: Projects from sparse high-dimensional space back to embedding dimension
        # This learns how to combine information from the sparse representation
        # Shape: (nh * N, D) - takes expanded representation and compresses to D dimensions
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        
        # Encoder: Projects from embedding dimension to sparse high-dimensional space
        # This learns how to expand the representation to capture more information
        # Shape: (nh, D, N) - expands D-dimensional vectors to N-dimensional sparse vectors per head
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        # Attention mechanism: Allows the model to focus on relevant parts of the sequence
        self.attn = Attention(config)

        # Layer normalization: Stabilizes training by normalizing activations
        # elementwise_affine=False means no learnable scale/shift (just normalization)
        # This helps gradients flow better and speeds up training
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        
        # Embedding layer: Converts byte tokens (0-255) to dense D-dimensional vectors
        # Each of the 256 possible bytes gets a learnable vector representation
        self.embed = nn.Embedding(config.vocab_size, D)
        
        # Dropout: Randomly zeros some activations during training to prevent overfitting
        # This forces the model to learn robust features that don't depend on specific neurons
        self.drop = nn.Dropout(config.dropout)
        
        # Encoder for value vectors: Similar to self.encoder but used for value projection
        # This allows the model to learn different encodings for attention values vs queries/keys
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        # Language model head: Final projection from embeddings to vocabulary logits
        # Converts the D-dimensional representation to a 256-dimensional vector
        # Each dimension represents the model's confidence that the next byte is that value
        # Shape: (D, vocab_size) - projects from embedding space to vocabulary space
        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        # Initialize all weights with small random values
        # This ensures the model starts with diverse, non-zero gradients
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for different layer types.
        
        Proper weight initialization is crucial for training. If weights are too large,
        gradients explode; if too small, gradients vanish. The standard deviation of 0.02
        is a common choice that works well for transformer models.
        
        Args:
            module: The neural network module to initialize (Linear layer, Embedding, etc.)
        """
        if isinstance(module, nn.Linear):
            # Linear layers: Initialize weights with small random values from normal distribution
            # mean=0.0, std=0.02 ensures weights start small but non-zero
            # Small initial weights help prevent exploding gradients early in training
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Biases start at zero - this is standard practice for most layers
            # Zero bias means the layer starts neutral, learning bias values during training
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embedding layers: Also initialized with small random values
            # This ensures each token starts with a unique but small vector representation
            # The model will learn to make these representations meaningful during training
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass through the model.
        
        This is where the model processes input sequences and makes predictions.
        The input is a sequence of byte indices, and the output is a probability distribution
        over possible next bytes for each position.
        
        Args:
            idx: Input tensor of byte indices (shape: batch_size, sequence_length)
                 Each value is an integer 0-255 representing a byte
            targets: Optional target byte indices for computing loss during training
                     If provided, the model computes how wrong its predictions are
                     
        Returns:
            logits: Raw prediction scores for each possible next byte (before softmax)
                    Shape: (batch_size, sequence_length, vocab_size)
                    Higher values = model thinks that byte is more likely
            loss: Cross-entropy loss if targets provided, None otherwise
                  Lower loss = better predictions = model is learning correctly
        """
        C = self.config

        # Extract dimensions from input
        B, T = idx.size()  # B = batch size, T = sequence length
        D = C.n_embd       # Embedding dimension
        nh = C.n_head      # Number of attention heads
        # N is the expanded dimension for sparse representations
        N = D * C.mlp_internal_dim_multiplier // nh

        # Step 1: Convert byte indices to dense vector representations (embeddings)
        # Each byte (0-255) is mapped to a D-dimensional vector that the model learns
        # unsqueeze(1) adds a dimension for heads: (B, T, D) -> (B, 1, T, D)
        x = self.embed(idx).unsqueeze(1)

        # Step 2: Normalize embeddings (helps with training stability)
        # Layer normalization centers and scales the activations, preventing extreme values
        # This is done early to ensure the rest of the network receives well-scaled inputs
        x = self.ln(x)  # Shape: (B, 1, T, D)

        # Step 3: Process through multiple transformer layers
        # Each layer refines the representation, allowing the model to learn increasingly
        # complex patterns. More layers = deeper understanding but more computation.
        for level in range(C.n_layer):
            # 3a: Encode input to sparse high-dimensional representation
            # This expands the D-dimensional vectors to N-dimensional vectors per head
            # The model learns which dimensions to activate for different patterns
            x_latent = x @ self.encoder  # (B, 1, T, D) @ (nh, D, N) -> (B, nh, T, N)

            # 3b: Apply ReLU to create sparse representation
            # ReLU sets negative values to zero, creating a sparse (mostly zeros) representation
            # Sparsity can help the model focus on important features and reduce overfitting
            # Only positive activations pass through, creating a "rectified" representation
            x_sparse = F.relu(x_latent)  # Shape: (B, nh, T, N)

            # 3c: Apply attention mechanism
            # Attention allows each position to focus on relevant parts of the sequence
            # Q and K are both x_sparse (self-attention), V is the original x
            # This creates context-aware representations that consider the whole sequence
            yKV = self.attn(
                Q=x_sparse,  # Queries: "what am I looking for?"
                K=x_sparse,  # Keys: "what do I contain?" (same as queries for self-attention)
                V=x,         # Values: "what information do I provide?"
            )
            # Normalize attention output for stability
            yKV = self.ln(yKV)

            # 3d: Encode attention output to sparse representation
            # Similar to step 3a, but for the attention output
            # This allows the model to learn different sparse patterns for attended information
            y_latent = yKV @ self.encoder_v  # (B, nh, T, D) @ (nh, D, N) -> (B, nh, T, N)
            y_sparse = F.relu(y_latent)  # Create sparse representation of attention output

            # 3e: Combine sparse representations through element-wise multiplication
            # This is a gating mechanism: x_sparse acts as a gate for y_sparse
            # Only features that are active in both representations remain active
            # This creates a more selective, focused representation
            xy_sparse = x_sparse * y_sparse  # Shape: (B, nh, T, N)

            # 3f: Apply dropout for regularization
            # Randomly zeros some activations during training to prevent overfitting
            # This forces the model to learn redundant, robust features
            xy_sparse = self.drop(xy_sparse)

            # 3g: Decode sparse representation back to embedding dimension
            # Transpose and reshape to combine all heads: (B, nh, T, N) -> (B, 1, T, nh*N)
            # Then project back to D dimensions using the decoder matrix
            # This learns how to combine information from the sparse representation
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # Result shape: (B, 1, T, D)
            
            # 3h: Normalize the MLP output
            y = self.ln(yMLP)
            
            # 3i: Residual connection: add the layer's output to its input
            # This creates a "highway" for gradients to flow through, enabling deeper networks
            # The model can learn to make small adjustments (y) to the input (x)
            # Without residual connections, deep networks are hard to train
            x = self.ln(x + y)  # Normalize the sum for stability

        # Step 4: Project final representation to vocabulary logits
        # Reshape to remove head dimension: (B, 1, T, D) -> (B, T, D)
        # Then project to vocabulary size: (B, T, D) @ (D, vocab_size) -> (B, T, vocab_size)
        # Each of the 256 values represents the model's confidence for that byte
        logits = x.view(B, T, D) @ self.lm_head
        
        # Step 5: Compute loss if targets are provided (training mode)
        loss = None
        if targets is not None:
            # Cross-entropy loss measures how wrong the predictions are
            # It compares the predicted probability distribution to the true next byte
            # Lower loss = predictions are closer to truth = model is learning
            # Reshape to (batch*sequence, vocab_size) and (batch*sequence,) for the loss function
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Generate new tokens (bytes) by sampling from the model's predictions.
        
        This is the text generation process: the model predicts the next byte, we sample
        from that prediction, add it to the sequence, and repeat. This creates new sequences
        that follow patterns the model learned during training.
        
        The generation is autoregressive: each new token is based on all previous tokens.
        This is why language models can create coherent text - they consider the full context.
        
        Args:
            idx: Initial sequence of byte indices to start generation from (the "prompt")
                 Shape: (batch_size, sequence_length)
            max_new_tokens: Maximum number of new bytes to generate
                           The model will generate exactly this many tokens
            temperature: Controls randomness in sampling
                         - temperature = 1.0: Use model's confidence as-is (default)
                         - temperature < 1.0: Make predictions more confident (less random)
                         - temperature > 1.0: Make predictions less confident (more random)
                         Lower temperature = more conservative, higher = more creative
            top_k: If specified, only consider the top-k most likely tokens
                   This prevents the model from choosing very unlikely tokens
                   None means consider all 256 possible bytes
                   
        Returns:
            Extended sequence with original prompt + generated tokens
            Shape: (batch_size, original_length + max_new_tokens)
        """
        # @torch.no_grad() disables gradient computation for efficiency
        # We don't need gradients during generation (only during training)
        # This saves memory and speeds up inference
        
        # Generate tokens one at a time, building on the previous sequence
        for _ in range(max_new_tokens):
            # Use the current sequence as input (includes original prompt + generated tokens so far)
            idx_cond = idx
            
            # Get model's predictions for the next token
            # The model outputs logits (raw scores) for all 256 possible bytes
            logits, _ = self(idx_cond)  # Shape: (batch, sequence_length, vocab_size)
            
            # Take only the last position's predictions (we only care about the next token)
            # Divide by temperature to adjust randomness
            # Higher temperature = divide by larger number = smaller differences = more random
            # Lower temperature = divide by smaller number = larger differences = more confident
            logits = logits[:, -1, :] / temperature  # Shape: (batch, vocab_size)
            
            # Optional: Apply top-k filtering to restrict to most likely tokens
            if top_k is not None:
                # Find the k-th highest logit value
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set all logits below the k-th highest to negative infinity
                # This makes their probability zero after softmax
                # This prevents the model from choosing very unlikely tokens
                logits[logits < values[:, [-1]]] = float("-inf")
            
            # Convert logits to probabilities using softmax
            # Softmax ensures probabilities sum to 1.0 and are all positive
            # Higher logits become higher probabilities, but the relationship is non-linear
            # This creates a probability distribution over the 256 possible bytes
            probs = F.softmax(logits, dim=-1)  # Shape: (batch, vocab_size)
            
            # Sample one token from the probability distribution
            # multinomial randomly picks a token, but more likely tokens are picked more often
            # This adds randomness while still favoring the model's confident predictions
            # Without sampling, we'd always pick the most likely token (greedy decoding)
            idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (batch, 1)
            
            # Append the new token to the sequence
            # This becomes the input for the next iteration
            idx = torch.cat((idx, idx_next), dim=1)  # Shape: (batch, sequence_length + 1)
        
        # Return the complete sequence: original prompt + all generated tokens
        return idx
