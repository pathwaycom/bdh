"""
Checkpoint evaluation and reporting module for BDH.
"""

import os
import re
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import torch

import bdh

plt.switch_backend('Agg')


def find_checkpoints(out_dir=".", min_step=0, max_step=100000):
    checkpoints = []
    pattern = re.compile(r'bdh_checkpoint_step_(\d+)\.pt')

    if not os.path.exists(out_dir):
        return []

    for filename in os.listdir(out_dir):
        match = pattern.match(filename)
        if match:
            step = int(match.group(1))
            if min_step <= step <= max_step:
                checkpoints.append((step, os.path.join(out_dir, filename)))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get('config', bdh.BDHConfig())
    model = bdh.BDH(config)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    metadata = {
        'step': checkpoint.get('step', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown'),
    }
    return model, config, metadata


def generate_text(model, prompt_text, max_new_tokens=200, temperature=1.0, top_k=3, device="cpu"):
    model.eval()
    model = model.to(device)
    prompt_bytes = bytearray(prompt_text, "utf-8")
    prompt_tensor = torch.tensor(prompt_bytes, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        generated = model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    generated_bytes = bytes(generated.to(torch.uint8).to("cpu").squeeze(0))
    return generated_bytes.decode(errors="backslashreplace")


def plot_loss(csv_path, output_path, checkpoint_steps=None):
    if not os.path.exists(csv_path):
        print(f"Warning: Log file {csv_path} not found. Skipping plot.")
        return False

    try:
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(10, 6))
        if "loss" in df.columns:
            plt.plot(df["step"], df["loss"], label="Training Loss")
        else:
            plt.plot(df["step"], df["train_loss"], label="Training Loss")
            if "val_loss" in df.columns:
                plt.plot(df["step"], df["val_loss"], label="Validation Loss")
        if checkpoint_steps:
            for step in checkpoint_steps:
                plt.axvline(step, color='gray', alpha=0.2, linestyle='--')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        print(f"Loss plot saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error plotting loss: {e}")
        return False


def sanitize_preview(text, limit=160):
    cleaned = text.replace("\r", " ").replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned[:limit] + ("..." if len(cleaned) > limit else "")


def plot_checkpoint_samples(results, output_path):
    if not results:
        return False
    steps = [results[0], results[len(results) // 2], results[-1]]
    fig, axes = plt.subplots(3, 1, figsize=(12, 5.5))
    for ax, r in zip(axes, steps):
        preview = sanitize_preview(r["text"], limit=200)
        ax.axis("off")
        ax.text(
            0,
            0.5,
            f"Step {r['step']} | Loss {r.get('loss', 'N/A')}\n{preview}",
            fontfamily="monospace",
            fontsize=9,
            va="center",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    return True


def plot_output_quality(results, output_path):
    if not results:
        return False
    steps = []
    ratios = []
    for r in results:
        text = r["text"]
        if not text:
            continue
        steps.append(r["step"])
        ratios.append(len(set(text)) / max(1, len(text)))
    plt.figure(figsize=(10, 4))
    plt.plot(steps, ratios, marker="o", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Unique character ratio")
    plt.title("Output Diversity over Checkpoints")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


class Evaluator:
    def __init__(self, input_dir, output_dir, prompt, device="auto", log_file=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.prompt = prompt
        self.device = device
        self.log_file = log_file

    def _resolve_device(self):
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and torch.backends.mps.is_available():
                device = "mps"
        return device

    def run(self):
        device = self._resolve_device()
        print(f"Using device: {device}")

        os.makedirs(self.output_dir, exist_ok=True)
        figs_dir = os.path.join(self.output_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)

        checkpoints = find_checkpoints(self.input_dir)
        print(f"Found {len(checkpoints)} checkpoints in {self.input_dir}")

        results = []
        for step, path in checkpoints:
            print(f"Eval step {step}...", end="\r")
            model, _, meta = load_checkpoint(path)
            text = generate_text(model, self.prompt, device=device)
            results.append({
                'step': step,
                'loss': meta['loss'],
                'text': text
            })

        csv_file = None
        if os.path.basename(os.path.normpath(self.input_dir)) == "checkpoints":
            parent = os.path.dirname(os.path.normpath(self.input_dir))
            log_dir = os.path.join(parent, "logs")
            csv_candidate = os.path.join(log_dir, "training_log.csv")
            if os.path.exists(csv_candidate):
                csv_file = csv_candidate
        if not csv_file or not os.path.exists(csv_file):
            candidate = os.path.join(self.input_dir, "training_log.csv")
            if os.path.exists(candidate):
                csv_file = candidate

        loss_plot_path = os.path.join(figs_dir, "loss_curve.png")
        has_plot = False
        if csv_file:
            checkpoint_steps = [r["step"] for r in results]
            has_plot = plot_loss(csv_file, loss_plot_path, checkpoint_steps=checkpoint_steps)

        samples_path = os.path.join(figs_dir, "checkpoint_samples.png")
        plot_checkpoint_samples(results, samples_path)
        quality_path = os.path.join(figs_dir, "output_quality.png")
        plot_output_quality(results, quality_path)

        report_path = os.path.join(self.output_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# BDH Training Report\n\n")
            if has_plot:
                f.write("## Training Loss\n\n")
                f.write(f"![Loss Curve](figs/loss_curve.png)\n\n")
            f.write("## Quick Before vs After\n\n")
            if results:
                early = results[0]
                mid = results[len(results) // 2]
                late = results[-1]
                f.write("| Stage | Step | Sample Output |\n")
                f.write("|-------|------|---------------|\n")
                f.write(f"| Early | {early['step']} | {sanitize_preview(early['text'], 160)} |\n")
                f.write(f"| Mid | {mid['step']} | {sanitize_preview(mid['text'], 160)} |\n")
                f.write(f"| Late | {late['step']} | {sanitize_preview(late['text'], 160)} |\n")
                f.write("\n")
            f.write("## Checkpoint Samples\n\n")
            f.write(f"![Checkpoint Samples](figs/checkpoint_samples.png)\n\n")
            f.write("## Output Diversity\n\n")
            f.write(f"![Output Diversity](figs/output_quality.png)\n\n")
            f.write("## Checkpoint Evaluations\n\n")
            f.write(f"**Prompt:** `{self.prompt}`\n\n")
            f.write("| Step | Loss | Generated Text Preview |\n")
            f.write("|------|------|------------------------|\n")
            for r in results:
                preview = r['text'][:100].replace('\n', ' ').replace('|', '\\|')
                f.write(f"| {r['step']} | {r.get('loss', 'N/A')} | {preview}... |\n")
            f.write("\n\n## Full Generation Outputs\n\n")
            for r in results:
                f.write(f"### Step {r['step']}\n")
                f.write(f"**Loss:** {r.get('loss', 'N/A')}\n\n")
                f.write("```\n")
                f.write(r['text'])
                f.write("\n```\n\n")

        print(f"\nReport generated at {report_path}")

        log_path = self.log_file or os.path.join(self.output_dir, "checkpoint_generations.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {self.prompt}\n")
            f.write("=" * 80 + "\n")
            for r in results:
                f.write(f"\nSTEP {r['step']} | Loss: {r.get('loss', 'N/A')}\n")
                f.write("-" * 80 + "\n")
                f.write(r["text"])
                f.write("\n")
        print(f"Consolidated log saved to {log_path}")


def run_evaluation(input_dir, output_dir, prompt, device="auto", log_file=None):
    Evaluator(input_dir, output_dir, prompt, device, log_file).run()


def run_multi_prompt_evaluation(input_dir, output_dir, prompts, device="auto"):
    eval_dir = os.path.join(output_dir, "multi_prompt")
    os.makedirs(eval_dir, exist_ok=True)
    for idx, prompt in enumerate(prompts, start=1):
        prompt_tag = f"prompt_{idx}"
        eval_out = os.path.join(eval_dir, prompt_tag)
        Evaluator(
            input_dir=input_dir,
            output_dir=eval_out,
            prompt=prompt,
            device=device,
            log_file=None,
        ).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate BDH checkpoints")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation", help="Directory to save report and visuals")
    parser.add_argument("--prompt", type=str, default="To be or not to be")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_file", type=str, default=None, help="Path to write consolidated generations log")
    args = parser.parse_args(argv)
    run_evaluation(args.input_dir, args.output_dir, args.prompt, args.device, args.log_file)


if __name__ == "__main__":
    main()
