# Copyrighth Pathway Technology, Inc.

import os
from contextlib import nullcontext

import bdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.suppress_errors = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration
BDH_CONFIG = bdh.BDHConfig()
TOKENS_TO_GENERATE = 4000
BLOCK_SIZE = 1024
EFFECTIVE_BATCH_SIZE = 32
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 10
CHECKPOINT_FREQ = 100
EVAL_FREQ = 100
EVAL_ITERS = 20

# Training mode: 'scratch', 'continue', 'evaluate'
mode = 'evaluate'  # Change this to 'continue' or 'evaluate'
checkpoint_path = 'checkpoint_500.pt'  # Path to checkpoint for 'continue' and 'evaluate' modes

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

# Global vocabulary mappings
char_to_id = {}
id_to_char = {}


# Fetch the tiny Shakespeare dataset
def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


def build_vocabulary():
    """Build character-level vocabulary from input data."""
    global char_to_id, id_to_char
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}
    return len(chars)


def get_batch(split):
    # treat the file as characters
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    data = [char_to_id[ch] for ch in text]
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.tensor(data[i : i + BLOCK_SIZE], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i + 1 : i + 1 + BLOCK_SIZE], dtype=torch.long) for i in ix])
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()


@torch.no_grad()
def estimate_loss(model):
    """Estimate loss over multiple batches for both train and eval splits."""
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            with ctx:
                logits, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

import torch

# Assume these are passed in or defined in a config object
# BATCH_SIZE = 16 # Fallback
# BLOCK_SIZE = 1024
# DTYPE = "bfloat16"

def get_optimal_batch_size(model, block_size: int, dtype: str):
    """Get optimal batch size based on model size and available VRAM."""
    if not torch.cuda.is_available():
        return 16 # Return a default fallback

    # It's better to get the config from the model itself
    config = model.config

    # --- 1. Calculate Model Memory Usage ---
    param_count = sum(p.numel() for p in model.parameters())
    # Parameters (4 bytes) + Gradients (4 bytes) + AdamW Optimizer (8 bytes)
    model_memory = param_count * (4 + 4 + 8)

    # --- 2. Calculate Activation Memory Usage (per batch item) ---
    activation_bytes = 4 if dtype == "float32" else 2
    
    # Large tensors in the main loop (shape ~ B, T, n)
    N = config.mlp_internal_dim_multiplier * config.n_embd // config.n_head
    n_total = config.n_head * N

    main_activations = block_size * n_total * config.n_layer * 6 * activation_bytes
    
    # Don't forget the final logits tensor (shape ~ B, T, vocab_size)
    logits_memory = block_size * config.vocab_size * 4 # Logits are often float32
    
    activation_per_batch_item = main_activations + logits_memory

    # --- 3. Determine Max Batch Size ---
    total_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = total_memory * 0.95  # Use 80% of VRAM as a safety margin
    memory_for_batches = available_memory - model_memory
    
    if memory_for_batches <= 0:
        max_batch = 0
    else:
        max_batch = int(memory_for_batches / activation_per_batch_item)

    # --- Debug logging ---
    print(f"VRAM Debug:")
    print(f"  Total VRAM: {total_memory / 1e9:.2f} GB")
    print(f"  Model memory: {model_memory / 1e9:.2f} GB")
    print(f"  Activation per batch item: {activation_per_batch_item / 1e6:.2f} MB")
    print(f"  Memory available for batches: {memory_for_batches / 1e9:.2f} GB")
    print(f"  Max batch calculated: {max_batch}")

    if max_batch == 0:
        print("Warning: Model parameters alone exceed 80% of VRAM. Batch size set to 1.")
        return 1

    # --- Find nearest power of 2 for efficiency ---
    optimal = 1
    while optimal * 2 <= max_batch:
        optimal *= 2
    return max(1, optimal)


if __name__ == "__main__":
    fetch_data()

    # Build vocabulary
    vocab_size = build_vocabulary()
    BDH_CONFIG.vocab_size = vocab_size
    BDH_CONFIG.vocab_size = vocab_size
    print(f"Built vocabulary with {vocab_size} characters")

    model = bdh.BDH(BDH_CONFIG).to(device)

    # Auto-adjust batch size based on model size and VRAM
    BATCH_SIZE = get_optimal_batch_size(model, BLOCK_SIZE, dtype)
    gradient_accumulation_steps = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
    print(f"Using batch size: {BATCH_SIZE}, effective: {EFFECTIVE_BATCH_SIZE}, accumulation steps: {gradient_accumulation_steps}")
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    start_step = 0
    if mode == 'continue':
        start_step = bdh.load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Loaded checkpoint from step {start_step}")
    elif mode == 'evaluate':
        bdh.load_checkpoint(model, optimizer, checkpoint_path)
        print("Loaded checkpoint for evaluation")
        model.eval()
        prompt_text = "To be or "
        prompt = torch.tensor([char_to_id[ch] for ch in prompt_text], dtype=torch.long, device=device).unsqueeze(0)
        ret = model.generate(prompt, max_new_tokens=TOKENS_TO_GENERATE, top_k=3)
        ret_decoded = ''.join([id_to_char[i.item()] for i in ret.squeeze(0)])
        print(ret_decoded)
        exit()

    for step in range(start_step, MAX_ITERS):
        loss_acc = 0
        for micro_step in range(gradient_accumulation_steps):
            x, y = get_batch("train")
            with ctx:
                logits, loss, _ = model(x, y)
            loss = loss / gradient_accumulation_steps
            loss_acc += loss
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if step % LOG_FREQ == 0:
            print(f"Step: {step}/{MAX_ITERS} loss {loss_acc.item():.3}")
        if step % EVAL_FREQ == 0 and step > 0:
            losses = estimate_loss(model)
            print(f"Step: {step}/{MAX_ITERS} train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}")
        if step % CHECKPOINT_FREQ == 0 and step > 0:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step}, f'checkpoint_{step}.pt')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': MAX_ITERS}, 'final_checkpoint.pt')
    print("Training done, now generating a sample ")
    model.eval()
    prompt_text = "To be or "
    prompt = torch.tensor([char_to_id[ch] for ch in prompt_text], dtype=torch.long, device=device).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=TOKENS_TO_GENERATE, top_k=3)
    ret_decoded = ''.join([id_to_char[i.item()] for i in ret.squeeze(0)])

    print(ret_decoded)
