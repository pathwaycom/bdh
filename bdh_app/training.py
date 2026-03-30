# Copyright Pathway Technology, Inc.

"""
Training module for BDH.

This module provides a CLI-compatible entrypoint via main().
"""

import os
import argparse
import csv
from contextlib import nullcontext

import bdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: For monitoring memory usage
try:
    import psutil  # type: ignore[import-not-found]
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Install psutil for memory monitoring: pip install psutil")

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
else:
    device = torch.device("cpu")

# Data type selection
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        dtype = "bfloat16"
    else:
        dtype = "float16"
elif torch.backends.mps.is_available():
    dtype = "float32"
else:
    dtype = "float32"

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if device.type == "cuda"
    else nullcontext()
)

scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16" and device.type == "cuda"))

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {device} with dtype {dtype}")

BDH_CONFIG = bdh.BDHConfig(
    mlp_internal_dim_multiplier=48,
    n_layer=5,
)

BLOCK_SIZE = 384
BATCH_SIZE = 12

if device.type == "cpu":
    print("WARNING: Training on CPU will be very slow. Consider using a GPU if available.")
    print("Memory optimizations applied: reduced batch size, block size, and model dimensions")
elif device.type == "mps":
    print("MPS detected. Using float32 (required for MPS training).")
    print(f"Model config: n_layer={BDH_CONFIG.n_layer}, mlp_mult={BDH_CONFIG.mlp_internal_dim_multiplier}")
    print(f"Training config: batch_size={BATCH_SIZE}, block_size={BLOCK_SIZE}")
    print("Note: float32 uses 2x more memory than float16. Monitor RAM usage.")

MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 10

input_file_path = os.path.join(os.path.dirname(__file__), "..", "input.txt")
input_file_path = os.path.abspath(input_file_path)


def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


def get_batch(split):
    if not os.path.exists(input_file_path):
        fetch_data()
    assert os.path.exists(input_file_path), f"Dataset missing at {input_file_path}"

    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def estimate_loss(model, split, eval_steps=10):
    losses = []
    model.eval()
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


class Trainer:
    def __init__(self, max_iters, batch_size, out_dir):
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.out_dir = out_dir

    def run(self):
        global MAX_ITERS, BATCH_SIZE
        MAX_ITERS = self.max_iters
        BATCH_SIZE = self.batch_size

        checkpoint_dir = os.path.join(self.out_dir, "checkpoints")
        log_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        print(f"Training outputs will be saved to: {self.out_dir}")
        print(f"  Checkpoints: {checkpoint_dir}")
        print(f"  Logs: {log_dir}")

        fetch_data()
        model = bdh.BDH(BDH_CONFIG).to(device)

        if device.type == "cuda":
            model = torch.compile(model)
        elif device.type == "mps":
            print("Skipping torch.compile on MPS")
        else:
            print("Skipping torch.compile on CPU")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        x, y = get_batch("train")
        loss_acc = 0
        loss_steps = 0

        csv_path = os.path.join(log_dir, "training_log.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "train_loss", "val_loss"])

        checkpoint_intervals = [int(MAX_ITERS * (i / 10.0)) for i in range(1, 11)]
        checkpoint_intervals = sorted(list(set(s for s in checkpoint_intervals if s > 0)))
        print(f"Checkpoints will be saved at steps: {checkpoint_intervals}")

        for step in range(MAX_ITERS):
            with ctx:
                _, loss = model(x, y)

            x, y = get_batch("train")
            loss_acc += loss.item()
            loss_steps += 1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if device.type == "cuda" and step % 50 == 0:
                torch.cuda.empty_cache()

            if step % LOG_FREQ == 0:
                avg_loss = loss_acc / loss_steps
                val_loss = estimate_loss(model, "val", eval_steps=5)
                print(f"Step: {step}/{MAX_ITERS} train {avg_loss:.4f} | val {val_loss:.4f}")
                csv_writer.writerow([step, avg_loss, val_loss])
                csv_file.flush()
                loss_acc = 0
                loss_steps = 0

            target_step = step + 1
            if target_step in checkpoint_intervals:
                ckpt_name = f"bdh_checkpoint_step_{target_step}.pt"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': BDH_CONFIG,
                    'step': target_step,
                    'loss': loss.item(),
                }
                torch.save(checkpoint, ckpt_path)
                print(f"  --> Checkpoint saved: {ckpt_path}")

        csv_file.close()
        print("Training done.")

        final_model_path = os.path.join(checkpoint_dir, "bdh_model_final.pt")
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model weights saved to {final_model_path}")

        model.eval()
        print("Generating sample...")
        prompt = torch.tensor(
            bytearray("To be or ", "utf-8"),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
        ret = model.generate(prompt, max_new_tokens=100, top_k=3)
        try:
            ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
                errors="backslashreplace"
            )
            print(ret_decoded)
        except Exception:
            print("(Could not decode output bytes to string)")


def run_training(max_iters, batch_size, out_dir):
    Trainer(max_iters, batch_size, out_dir).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train BDH model")
    parser.add_argument("--max_iters", type=int, default=3000, help="Maximum number of training iterations")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--out_dir", type=str, default="outputs/training", help="Base output directory")

    args = parser.parse_args(argv)
    run_training(args.max_iters, args.batch_size, args.out_dir)


if __name__ == "__main__":
    main()
