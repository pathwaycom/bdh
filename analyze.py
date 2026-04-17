# Copyright Pathway Technology, Inc.
"""Interpretability analysis for a trained BDH model.

Quantifies the sparsity, activation frequency, and selectivity of individual
"neurons" (units of the `xy_sparse` tensor inside each BDH layer). Surfaces
candidate monosemantic neurons by showing the byte-context windows that most
strongly activate them.

Run after `train.py`:

    python train.py          # produces bdh_checkpoint.pt + input.txt
    python analyze.py        # reads them, prints a report, writes analysis.png

Set BDH_CHECKPOINT=path/to/ckpt.pt to point at a non-default checkpoint.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import fields

import numpy as np
import torch

import bdh


# ──────────────────────────────────────────────────────────────────────
# Instrumented forward — captures xy_sparse per layer without editing bdh.py
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def instrumented_forward(model: bdh.BDH, idx: torch.Tensor) -> list[torch.Tensor]:
    """Run BDH.forward and return xy_sparse for each layer.

    Mirrors BDH.forward exactly (same math, same order) but records the
    post-gating sparse activations that the README describes as the
    network's "locally interacting neuron particles."
    """
    C = model.config
    B, T = idx.size()

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    per_layer_xy_sparse: list[torch.Tensor] = []

    for _ in range(C.n_layer):
        x_latent = x @ model.encoder
        x_sparse = torch.relu(x_latent)

        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = torch.relu(y_latent)
        xy_sparse = x_sparse * y_sparse  # (B, nh, T, N)

        per_layer_xy_sparse.append(xy_sparse.detach())

        # continue exactly like BDH.forward so downstream layers see the same state
        N = C.mlp_internal_dim_multiplier * C.n_embd // C.n_head
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * C.n_head) @ model.decoder
        y = model.ln(yMLP)
        x = model.ln(x + y)

    return per_layer_xy_sparse


# ──────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────

def load_checkpoint(path: str, device: torch.device) -> tuple[bdh.BDH, bdh.BDHConfig]:
    ckpt = torch.load(path, map_location=device, weights_only=False)

    cfg_dict = ckpt["config"]
    valid_fields = {f.name for f in fields(bdh.BDHConfig)}
    cfg = bdh.BDHConfig(**{k: v for k, v in cfg_dict.items() if k in valid_fields})

    model = bdh.BDH(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def sample_batches(data: np.ndarray, n_batches: int, batch_size: int, block_size: int, seed: int = 0):
    """Yield (idx_tensor, absolute_start_positions) across n_batches samples."""
    rng = np.random.default_rng(seed)
    max_start = len(data) - block_size
    for _ in range(n_batches):
        starts = rng.integers(0, max_start, size=batch_size)
        x = np.stack([data[s : s + block_size].astype(np.int64) for s in starts])
        yield torch.from_numpy(x), starts


def run_analysis(
    model: bdh.BDH,
    cfg: bdh.BDHConfig,
    data: np.ndarray,
    device: torch.device,
    n_batches: int = 10,
    batch_size: int = 8,
    block_size: int = 256,
    fire_threshold: float = 1e-3,
) -> dict:
    """Accumulate per-neuron statistics across a sample of val-set positions."""
    nh = cfg.n_head
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // nh
    n_neurons = nh * N
    n_layer = cfg.n_layer

    # Per-layer, per-neuron accumulators (live on GPU while iterating)
    act_sum = torch.zeros((n_layer, n_neurons), dtype=torch.float64, device=device)
    fire_count = torch.zeros((n_layer, n_neurons), dtype=torch.float64, device=device)
    max_val = torch.full((n_layer, n_neurons), float("-inf"), dtype=torch.float32, device=device)
    # absolute byte position of the token that caused max_val
    max_pos = torch.zeros((n_layer, n_neurons), dtype=torch.int64, device=device)

    total_positions = 0

    for batch_idx, (idx, starts) in enumerate(
        sample_batches(data, n_batches, batch_size, block_size)
    ):
        idx = idx.to(device)
        # absolute position of each (batch, time) slot in the underlying byte stream
        starts_t = torch.from_numpy(starts).to(device)
        time_t = torch.arange(block_size, device=device)
        abs_pos = starts_t.unsqueeze(1) + time_t.unsqueeze(0)  # (B, T)
        abs_pos_flat = abs_pos.reshape(-1)                      # (B*T,)

        xy_list = instrumented_forward(model, idx)

        for layer_idx, xy in enumerate(xy_list):
            # xy: (B, nh, T, N) → (B*T, n_neurons)
            B_, nh_, T_, N_ = xy.shape
            act = xy.permute(0, 2, 1, 3).reshape(B_ * T_, nh_ * N_).float()

            act_sum[layer_idx] += act.sum(dim=0).to(torch.float64)
            fire_count[layer_idx] += (act > fire_threshold).sum(dim=0).to(torch.float64)

            # Track argmax per neuron across this batch, fold into running max
            batch_max, batch_argmax = act.max(dim=0)  # (n_neurons,), (n_neurons,)
            better = batch_max > max_val[layer_idx]
            max_val[layer_idx] = torch.where(better, batch_max, max_val[layer_idx])
            max_pos[layer_idx] = torch.where(
                better, abs_pos_flat[batch_argmax], max_pos[layer_idx]
            )

        total_positions += B_ * T_
        print(f"  processed batch {batch_idx + 1}/{n_batches} "
              f"({total_positions} token positions)")

    mean_act = (act_sum / total_positions).cpu().numpy()
    firing_rate = (fire_count / total_positions).cpu().numpy()
    max_val_np = max_val.cpu().numpy()
    max_pos_np = max_pos.cpu().numpy()

    # Selectivity: max / mean. High = neuron fires rarely but strongly.
    # Guard the mean against zeros; neurons that never fired get selectivity = 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        selectivity = np.where(mean_act > 0, max_val_np / mean_act, 0.0)

    return {
        "n_layer": n_layer,
        "n_neurons_per_layer": n_neurons,
        "nh": nh,
        "N": N,
        "total_positions": total_positions,
        "fire_threshold": fire_threshold,
        "mean_act": mean_act,           # (n_layer, n_neurons)
        "firing_rate": firing_rate,     # (n_layer, n_neurons)
        "max_val": max_val_np,          # (n_layer, n_neurons)
        "max_pos": max_pos_np,          # (n_layer, n_neurons)
        "selectivity": selectivity,     # (n_layer, n_neurons)
    }


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────

def bytes_to_printable(buf: bytes) -> str:
    """Render a byte window for console output — escapes control chars, keeps ASCII readable."""
    out_chars = []
    for b in buf:
        if b in (0x0a, 0x0d):
            out_chars.append("\\n" if b == 0x0a else "\\r")
        elif 0x20 <= b < 0x7f:
            out_chars.append(chr(b))
        else:
            out_chars.append(f"\\x{b:02x}")
    return "".join(out_chars)


def print_summary(stats: dict):
    n_layer = stats["n_layer"]
    n_neurons = stats["n_neurons_per_layer"]
    total = stats["total_positions"]

    print()
    print("=" * 72)
    print(" BDH INTERPRETABILITY ANALYSIS")
    print("=" * 72)
    print(f"  Layers:              {n_layer}")
    print(f"  Neurons per layer:   {n_neurons:,} ({stats['nh']} heads × {stats['N']} units)")
    print(f"  Token positions:     {total:,}")
    print(f"  Fire threshold:      {stats['fire_threshold']}")
    print()
    print(f"  {'layer':>5}  {'mean firing %':>13}  {'median firing %':>15}  "
          f"{'% neurons <1% fire':>18}  {'median selectivity':>19}")
    print("  " + "-" * 68)
    for layer in range(n_layer):
        fr = stats["firing_rate"][layer]
        sel = stats["selectivity"][layer]
        # Neurons that never fired have undefined selectivity — drop them from the median
        sel_active = sel[fr > 0]
        median_sel = float(np.median(sel_active)) if sel_active.size else float("nan")
        print(
            f"  {layer:>5}  "
            f"{100 * fr.mean():>12.2f}%  "
            f"{100 * np.median(fr):>14.2f}%  "
            f"{100 * np.mean(fr < 0.01):>17.2f}%  "
            f"{median_sel:>19.2f}"
        )
    print()


def print_top_neurons(
    stats: dict,
    data: np.ndarray,
    top_n: int = 5,
    window: int = 24,
):
    """For each layer, print the most-selective neurons with the byte window that
    activated them most. These are candidate monosemantic detectors."""
    n_layer = stats["n_layer"]
    print("Top-selective neurons per layer (candidate monosemantic detectors):")
    print()
    for layer in range(n_layer):
        sel = stats["selectivity"][layer]
        fr = stats["firing_rate"][layer]
        # Require at least some firing to avoid picking up degenerate never-activating neurons
        valid = fr > 1e-4
        if not np.any(valid):
            print(f"  Layer {layer}: no neurons met minimum firing rate.")
            continue
        sel_masked = np.where(valid, sel, -np.inf)
        top_idx = np.argpartition(-sel_masked, top_n)[:top_n]
        top_idx = top_idx[np.argsort(-sel_masked[top_idx])]

        print(f"  Layer {layer}")
        print(f"  {'neuron':>8}  {'firing %':>9}  {'selectivity':>11}  context (±{window} bytes around max)")
        print(f"  {'-' * 8:>8}  {'-' * 9:>9}  {'-' * 11:>11}  {'-' * (2 * window + 8)}")
        for neuron_idx in top_idx:
            pos = int(stats["max_pos"][layer, neuron_idx])
            lo = max(0, pos - window)
            hi = min(len(data), pos + window + 1)
            ctx = bytes_to_printable(bytes(data[lo:hi].tolist()))
            head = neuron_idx // stats["N"]
            unit = neuron_idx % stats["N"]
            print(
                f"  h{head:02d}·u{unit:05d}  "
                f"{100 * fr[neuron_idx]:>8.2f}%  "
                f"{sel[neuron_idx]:>11.1f}  "
                f"{ctx}"
            )
        print()


def save_figure(stats: dict, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return

    n_layer = stats["n_layer"]
    fig, axes = plt.subplots(2, n_layer, figsize=(3.5 * n_layer, 6), squeeze=False)

    for layer in range(n_layer):
        fr = stats["firing_rate"][layer]
        sel = stats["selectivity"][layer]
        sel_active = sel[fr > 0]

        ax = axes[0, layer]
        ax.hist(fr, bins=50, range=(0, max(0.01, float(fr.max()))))
        ax.set_title(f"Layer {layer}: firing rate")
        ax.set_xlabel("fraction of tokens firing")
        ax.set_ylabel("neuron count")

        ax = axes[1, layer]
        if sel_active.size:
            hi = float(np.quantile(sel_active, 0.99))
            ax.hist(np.clip(sel_active, 0, hi), bins=50, range=(0, hi))
        ax.set_title(f"Layer {layer}: selectivity (max/mean)")
        ax.set_xlabel("max/mean activation (clipped at p99)")
        ax.set_ylabel("neuron count")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved figure to {path}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "BDH_CHECKPOINT",
            os.path.join(os.path.dirname(__file__), "bdh_checkpoint.pt"),
        ),
    )
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "input.txt"),
    )
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--fire-threshold", type=float, default=1e-3)
    parser.add_argument("--top-n", type=int, default=5,
                        help="neurons to surface per layer")
    parser.add_argument("--context-window", type=int, default=24,
                        help="bytes of context on each side of the max-activating position")
    parser.add_argument("--figure", default="analysis.png",
                        help="path to save histogram figure (set to '' to skip)")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}. Run train.py first."
        )
    if not os.path.exists(args.input):
        raise SystemExit(
            f"Data file not found: {args.input}. Run train.py first to fetch it."
        )

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint, device)

    data = np.memmap(args.input, dtype=np.uint8, mode="r")
    # val split, matching train.py's 90/10 split
    val_data = np.asarray(data[int(0.9 * len(data)) :])
    print(f"Val data: {len(val_data):,} bytes")

    print(f"Analyzing {args.n_batches} batches of {args.batch_size}×{args.block_size}...")
    stats = run_analysis(
        model, cfg, val_data, device,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        fire_threshold=args.fire_threshold,
    )

    print_summary(stats)
    print_top_neurons(stats, val_data, top_n=args.top_n, window=args.context_window)

    if args.figure:
        save_figure(stats, args.figure)


if __name__ == "__main__":
    main()
