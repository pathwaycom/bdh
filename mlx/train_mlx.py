# Copyright 2025 Pathway Technology, Inc.
# MLX training script for BDH with Hugging Face datasets

import os
import time
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from datasets import load_dataset

import bdh_mlx


# Training Configuration
BDH_CONFIG = bdh_mlx.BDHConfig(
    n_layer=6,
    n_embd=256,
    n_head=4,
    mlp_internal_dim_multiplier=128,
    vocab_size=256,  # Byte-level encoding
    dropout=0.1,
)

# Dataset-specific configuration for Internal Knowledge Map
# Supports phased training: 'system', 'instruction', 'both', or 'full'
TRAINING_MODE = "both"  # Options: "system", "instruction", "both", "full"
BLOCK_SIZE = 8192  # Increased for long-form content as recommended
BATCH_SIZE = 1  # Reduced per dataset recommendations (1-4)
MAX_ITERS = 5000  # Adjusted for smaller batch size
EPOCHS = 3  # Number of epochs through the dataset
LEARNING_RATE = 5e-5  # Much lower LR for stability with complex dataset
WEIGHT_DECAY = 0.05
LOG_FREQ = 50
EVAL_FREQ = 250
SAVE_FREQ = 500
GRAD_CLIP = 1.0

# Checkpoint directory
CHECKPOINT_DIR = "checkpoints_mlx"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def encode_text_to_bytes(text: str) -> List[int]:
    """Convert text to byte-level tokens."""
    return list(bytearray(text, "utf-8"))


def decode_bytes_to_text(tokens: List[int]) -> str:
    """Convert byte-level tokens back to text."""
    return bytes(tokens).decode("utf-8", errors="backslashreplace")


class DataLoader:
    """Efficient data loader for byte-level text data with support for structured dataset."""
    
    def __init__(self, data: np.ndarray, batch_size: int, block_size: int, is_structured: bool = False):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.data_len = len(data)
        self.is_structured = is_structured
        
    def get_batch(self) -> Tuple[mx.array, mx.array]:
        """Get a random batch of data."""
        # Random starting indices, ensuring we don't go past the end
        max_start = max(0, self.data_len - self.block_size - 1)
        
        if max_start == 0:
            # If data is shorter than block size, pad it
            x = np.zeros((self.batch_size, self.block_size), dtype=np.int64)
            y = np.zeros((self.batch_size, self.block_size), dtype=np.int64)
            for b in range(self.batch_size):
                actual_len = min(self.data_len - 1, self.block_size)
                x[b, :actual_len] = self.data[:actual_len]
                y[b, :actual_len] = self.data[1:actual_len + 1]
        else:
            ix = np.random.randint(0, max_start, size=self.batch_size)
            
            # Extract sequences
            x = np.stack([self.data[i:i + self.block_size] for i in ix])
            y = np.stack([self.data[i + 1:i + 1 + self.block_size] for i in ix])
        
        # Convert to MLX arrays
        return mx.array(x), mx.array(y)


def load_and_prepare_dataset(
    dataset_name: str = "Severian/Internal-Knowledge-Map",
    training_mode: str = "both"
) -> Tuple[DataLoader, DataLoader, int, dict]:
    """
    Load dataset from Hugging Face and prepare train/val splits.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        training_mode: How to construct training text
            - "system": Use only system field (Phase 1 training)
            - "instruction": Use only instruction field (Phase 2 training)
            - "both": Use system + instruction (recommended for phased approach)
            - "full": Use system + instruction + response (complete training)
    
    Returns:
        train_loader, val_loader, total_bytes, metadata
    """
    print(f"Loading dataset: {dataset_name}")
    print(f"Training mode: {training_mode}")
    
    try:
        # Load the dataset
        ds = load_dataset(dataset_name)
        
        # Get the first split available
        split_name = list(ds.keys())[0]
        sample = ds[split_name][0]
        print(f"Dataset split: {split_name}")
        print(f"Available fields: {list(sample.keys())}")
        
        # Check for Internal Knowledge Map structure
        has_ikm_structure = 'system' in sample and 'instruction' in sample and 'response' in sample
        
        if has_ikm_structure:
            print("\n✓ Detected Internal Knowledge Map structure!")
            print(f"  - System field: {len(sample['system'])} chars (avg)")
            print(f"  - Instruction field: {len(sample['instruction'])} chars (avg)")
            print(f"  - Response field: {len(sample['response'])} chars (avg)")
            
            # Construct text based on training mode
            texts = []
            for item in ds[split_name]:
                if training_mode == "system":
                    # Phase 1: Focus on system guidelines
                    text = f"{item['system']}\n\n"
                elif training_mode == "instruction":
                    # Phase 2: Focus on instructions
                    text = f"{item['instruction']}\n\n"
                elif training_mode == "both":
                    # Combined: System context + Instruction
                    text = f"### System:\n{item['system']}\n\n### Instruction:\n{item['instruction']}\n\n"
                elif training_mode == "full":
                    # Full training: Everything including response
                    text = (f"### System:\n{item['system']}\n\n"
                           f"### Instruction:\n{item['instruction']}\n\n"
                           f"### Response:\n{item['response']}\n\n"
                           f"---\n\n")
                else:
                    raise ValueError(f"Unknown training_mode: {training_mode}")
                
                texts.append(text)
            
            all_text = "".join(texts)
            metadata = {
                'structure': 'ikm',
                'mode': training_mode,
                'num_examples': len(ds[split_name])
            }
            
        else:
            # Fallback for non-IKM datasets
            print("\nUsing standard text concatenation mode")
            text_fields = ['text', 'content', 'data', 'body', 'system', 'instruction']
            text_field = None
            
            for field in text_fields:
                if field in sample:
                    text_field = field
                    break
            
            if text_field is None:
                for key, value in sample.items():
                    if isinstance(value, str):
                        text_field = key
                        break
            
            if text_field is None:
                raise ValueError(f"Could not find text field. Available: {sample.keys()}")
            
            print(f"Using text field: '{text_field}'")
            all_text = "\n\n".join([item[text_field] for item in ds[split_name]])
            metadata = {
                'structure': 'standard',
                'field': text_field,
                'num_examples': len(ds[split_name])
            }
        
        print(f"\nTotal characters in dataset: {len(all_text):,}")
        
        # Convert to bytes
        all_bytes = np.array(encode_text_to_bytes(all_text), dtype=np.uint8)
        print(f"Total bytes: {len(all_bytes):,}")
        
        # Split into train (90%) and validation (10%)
        split_idx = int(0.9 * len(all_bytes))
        train_data = all_bytes[:split_idx]
        val_data = all_bytes[split_idx:]
        
        print(f"Train bytes: {len(train_data):,}")
        print(f"Validation bytes: {len(val_data):,}")
        
        # Create data loaders
        train_loader = DataLoader(train_data, BATCH_SIZE, BLOCK_SIZE, is_structured=has_ikm_structure)
        val_loader = DataLoader(val_data, BATCH_SIZE, BLOCK_SIZE, is_structured=has_ikm_structure)
        
        return train_loader, val_loader, len(all_bytes), metadata
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check the dataset name and ensure it's accessible.")
        raise


def evaluate_model(model: bdh_mlx.BDH, val_loader: DataLoader, num_batches: int = 10) -> float:
    """Evaluate model on validation set."""
    total_loss = 0.0
    
    for _ in range(num_batches):
        x, y = val_loader.get_batch()
        _, loss = model(x, y)
        total_loss += loss.item()
    
    return total_loss / num_batches


def save_checkpoint(model: bdh_mlx.BDH, optimizer: optim.Optimizer, step: int, loss: float):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"bdh_mlx_step_{step}.npz")
    
    print(f"Saving checkpoint to {checkpoint_path}")
    
    # Flatten parameter tree for saving
    def flatten_params(params, prefix=""):
        flat = {}
        for k, v in params.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, f"{key}_"))
            else:
                flat[key] = v
        return flat
    
    flat_params = flatten_params(model.parameters())
    
    mx.savez(
        checkpoint_path,
        step=mx.array([step]),
        loss=mx.array([loss]),
        **flat_params
    )


def generate_sample(model: bdh_mlx.BDH, prompt: str = "The meaning of", max_tokens: int = 200):
    """Generate a text sample from the model."""
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    
    # Encode prompt
    prompt_tokens = encode_text_to_bytes(prompt)
    idx = mx.array([prompt_tokens])
    
    # Generate
    output = model.generate(idx, max_new_tokens=max_tokens, temperature=0.8, top_k=50)
    
    # Decode
    output_tokens = output[0].tolist()
    generated_text = decode_bytes_to_text(output_tokens)
    
    print(generated_text)
    print(f"{'='*60}\n")


def train():
    """Main training loop."""
    print("="*80)
    print("BDH-MLX Training for Internal Knowledge Map Dataset")
    print("="*80)
    print(f"\nModel Configuration: {BDH_CONFIG}")
    print(f"\nTraining Configuration:")
    print(f"  Training Mode: {TRAINING_MODE}")
    print(f"  Block size (context): {BLOCK_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Max iterations: {MAX_ITERS}")
    print(f"  Epochs: {EPOCHS}\n")
    
    # Load dataset
    train_loader, val_loader, dataset_size, metadata = load_and_prepare_dataset(
        training_mode=TRAINING_MODE
    )
    
    print(f"\nDataset metadata: {metadata}")
    
    # Initialize model
    model = bdh_mlx.BDH(BDH_CONFIG)
    
    # Count parameters (flatten nested dict structure)
    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += count_params(v)
            else:
                total += v.size
        return total
    
    num_params = count_params(model.parameters())
    print(f"\nModel parameters: {num_params:,}\n")
    
    # Initialize optimizer
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Loss and gradient function
    def loss_fn(model, x, y):
        _, loss = model(x, y)
        return loss
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")
    
    if TRAINING_MODE == "system":
        print("📚 Phase 1: Training on SYSTEM guidelines")
        print("   Focus: Learning contextual frameworks and systemic knowledge\n")
    elif TRAINING_MODE == "instruction":
        print("🎯 Phase 2: Training on INSTRUCTIONS")
        print("   Focus: Parsing specific prompts and tailoring responses\n")
    elif TRAINING_MODE == "both":
        print("🔄 Combined Training: SYSTEM + INSTRUCTION")
        print("   Focus: Contextual understanding + specific prompt handling\n")
    else:
        print("📖 Full Training: SYSTEM + INSTRUCTION + RESPONSE")
        print("   Focus: Complete understanding of the knowledge map\n")
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    loss_acc = 0.0
    loss_steps = 0
    
    for step in range(MAX_ITERS):
        # Get batch
        x, y = train_loader.get_batch()
        
        # Forward and backward pass
        loss, grads = loss_and_grad_fn(model, x, y)
        
        # Gradient clipping (handle nested dict structure)
        if GRAD_CLIP > 0:
            def clip_grads(grad_dict):
                clipped = {}
                for k, v in grad_dict.items():
                    if isinstance(v, dict):
                        clipped[k] = clip_grads(v)
                    else:
                        clipped[k] = mx.clip(v, -GRAD_CLIP, GRAD_CLIP)
                return clipped
            grads = clip_grads(grads)
        
        # Update parameters
        optimizer.update(model, grads)
        
        # Evaluate the updated parameters
        mx.eval(model.parameters())
        
        # Accumulate loss
        loss_acc += loss.item()
        loss_steps += 1
        
        # Logging
        if (step + 1) % LOG_FREQ == 0:
            avg_loss = loss_acc / loss_steps
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * BATCH_SIZE * BLOCK_SIZE / elapsed
            
            print(f"Step {step + 1}/{MAX_ITERS} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Tokens/sec: {tokens_per_sec:.0f} | "
                  f"Time: {elapsed:.1f}s")
            
            loss_acc = 0.0
            loss_steps = 0
        
        # Evaluation
        if (step + 1) % EVAL_FREQ == 0:
            print("\nEvaluating on validation set...")
            val_loss = evaluate_model(model, val_loader)
            print(f"Validation loss: {val_loss:.4f}\n")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss! Saving checkpoint...\n")
                save_checkpoint(model, optimizer, step + 1, val_loss)
            
            # Generate sample
            generate_sample(model)
        
        # Periodic checkpoint
        if (step + 1) % SAVE_FREQ == 0:
            save_checkpoint(model, optimizer, step + 1, loss.item())
    
    # Final evaluation and generation
    print("\n" + "="*80)
    print("Training Completed!")
    print("="*80)
    
    final_val_loss = evaluate_model(model, val_loader, num_batches=50)
    print(f"\nFinal validation loss: {final_val_loss:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    save_checkpoint(model, optimizer, MAX_ITERS, final_val_loss)
    
    # Generate final samples based on training mode
    print("\n" + "="*80)
    print("Generating Final Samples")
    print("="*80 + "\n")
    
    if TRAINING_MODE == "system":
        prompts = [
            "### System:\nTask Overview:",
            "### System:\nGuidelines:",
            "### System:\nObjective:",
        ]
    elif TRAINING_MODE == "instruction":
        prompts = [
            "### Instruction:\nAnalyze",
            "### Instruction:\nExplain",
            "### Instruction:\nDescribe",
        ]
    elif TRAINING_MODE == "both":
        prompts = [
            "### System:\nTask Overview: Analyze and explore\n\n### Instruction:\n",
            "### System:\nGuidelines: Focus on core interactions\n\n### Instruction:\n",
            "### System:\nObjective: Generate insights\n\n### Instruction:\n",
        ]
    else:  # full
        prompts = [
            "### System:\nTask Overview:",
            "### Instruction:\nAnalyze the ethical implications",
            "### Response:\n",
        ]
    
    for prompt in prompts:
        generate_sample(model, prompt, max_tokens=200)
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Training mode used: {TRAINING_MODE}")
    print("\n" + "="*80)
    
    if TRAINING_MODE == "system":
        print("\n💡 Next Step: Consider training Phase 2 with TRAINING_MODE='instruction'")
    elif TRAINING_MODE == "instruction":
        print("\n✓ Phase 2 complete! Model should understand both system and instructions.")
    else:
        print("\n✓ Training complete with combined/full approach!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(1337)
    mx.random.seed(1337)
    
    train()

