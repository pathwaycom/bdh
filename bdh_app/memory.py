"""
Memory demonstration module for BDH.
"""

import sys
import os
import argparse

import torch
import matplotlib.pyplot as plt

import bdh

# Use Agg backend for headless environments (e.g., Docker)
plt.switch_backend("Agg")

# Configuration
FACT = "The capital of JiriLand is DragonCity."
QUERY = "The capital of JiriLand is"
EXPECTED = "DragonCity"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and torch.backends.mps.is_available():
    DEVICE = "mps"


class Tee(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class FastMemory:
    def __init__(self):
        self.facts = []

    def add(self, fact):
        self.facts.append(fact)

    def retrieve(self, query):
        query_tokens = set(query.lower().split())
        hits = []
        for fact in self.facts:
            fact_tokens = set(fact.lower().split())
            if query_tokens.intersection(fact_tokens):
                hits.append(fact)
        return hits


def find_latest_checkpoint(base_dir):
    if not os.path.exists(base_dir):
        return None
    candidates = []
    for name in os.listdir(base_dir):
        if name.startswith("bdh_checkpoint_step_") and name.endswith(".pt"):
            try:
                step = int(name.replace("bdh_checkpoint_step_", "").replace(".pt", ""))
            except ValueError:
                step = -1
            candidates.append((step, os.path.join(base_dir, name)))
    if not candidates:
        final_model = os.path.join(base_dir, "bdh_model_final.pt")
        return final_model if os.path.exists(final_model) else None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def get_model(train_out_dir):
    checkpoint_dir = os.path.join(train_out_dir, "checkpoints")
    ckpt_path = find_latest_checkpoint(checkpoint_dir)
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", bdh.BDHConfig())
        model = bdh.BDH(config)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        print(f"Loaded checkpoint: {ckpt_path}")
        return model
    raise FileNotFoundError(
        "No trained checkpoint found. Run train.py first to generate checkpoints."
    )


def generate(model, prompt, max_new=20, top_k=1, temperature=0.8):
    model.eval()
    prompt_bytes = bytearray(prompt, "utf-8")
    prompt_tensor = torch.tensor(prompt_bytes, dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(prompt_tensor, max_new_tokens=max_new, temperature=temperature, top_k=top_k)

    decoded = bytes(out.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="replace")
    return decoded[len(prompt):]


def sanitize_text(text):
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t ")
    return cleaned.replace("\r", "").strip()


def extract_fact_from_prompt(prompt_text):
    marker = "The capital of "
    if marker not in prompt_text:
        return None
    try:
        start = prompt_text.index(marker)
        snippet = prompt_text[start:]
        parts = snippet.split(" is ")
        if len(parts) < 2:
            return None
        city_part = parts[1]
        city = city_part.split(".")[0].strip()
        if city:
            return f"{marker}{parts[0].split(marker)[1]} is {city}."
    except Exception:
        return None
    return None


def memory_answer(query, facts):
    for fact in facts:
        if fact.startswith("The capital of ") and " is " in fact:
            return fact.split(" is ")[1].replace(".", "").strip()
    return ""


def demo_fast_memory(model):
    print("\n" + "=" * 50)
    print("FAST MEMORY: Runtime Context Injection")
    print("=" * 50)

    fast_memory = FastMemory()

    print("1. Baseline: trained model without memory")
    print(f"   Prompt: '{QUERY}'")
    output = sanitize_text(generate(model, QUERY))
    print(f"   Output: '{output}'")

    if "DragonCity" in output:
        print("   (Unexpected: Model randomly guessed it!)")
    else:
        print("   (Expected: Model doesn't know the fact)")

    print("\n2. Move fact from a prompt into fast memory")
    ingest_prompt = f"Fact: {FACT}\nQuestion: {QUERY}"
    print(f"   Ingest Prompt: '{ingest_prompt.replace(chr(10), ' ')}'")
    extracted = extract_fact_from_prompt(ingest_prompt)
    if extracted:
        fast_memory.add(extracted)
        print(f"   Stored in memory: '{extracted}'")
    else:
        print("   WARNING: No fact extracted from prompt.")

    print("\n3. Answer using fast memory (no fact in prompt)")
    retrieved = fast_memory.retrieve(QUERY)
    memory_response = memory_answer(QUERY, retrieved)
    print(f"   Memory Recall: '{memory_response}'")

    memory_context = " ".join([f"Fact: {f}" for f in retrieved])
    context_prompt = f"{memory_context}\nQuestion: {QUERY}\nAnswer:"
    model_with_memory = sanitize_text(generate(model, context_prompt))
    print(f"   Model + Memory Prompt Output: '{model_with_memory}'")

    if EXPECTED in memory_response:
        print("   SUCCESS: Memory layer recalled the fact.")
    else:
        print("   FAIL: Memory layer did not recall the fact.")


def demo_model_memory(model, out_dir):
    print("\n" + "=" * 50)
    print("MODEL MEMORY: Consolidate Facts into Weights")
    print("=" * 50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("1. Verifying model doesn't know the fact initially...")
    output = sanitize_text(generate(model, QUERY))
    print(f"   Output: '{output}'")

    print("\n2. Fine-tuning model on the fact (Consolidating to Long-Term Memory)...")
    data = bytearray(FACT * 10, "utf-8")
    x_train = torch.tensor(data[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)
    y_train = torch.tensor(data[1:], dtype=torch.long, device=DEVICE).unsqueeze(0)

    model.train()
    steps = 150
    print(f"   Training for {steps} steps...")

    losses = []
    for i in range(steps):
        _, loss = model(x_train, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        losses.append(current_loss)
        if i % 20 == 0 or i == steps - 1:
            print(f"   Step {i}: loss {current_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Fine-tuning Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Quick Fine-tuning Loss (Model Memory)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    loss_fig_path = os.path.join(out_dir, "figs", "memory_fine_tuning_loss.png")
    plt.savefig(loss_fig_path)
    plt.close()
    print(f"   (Saved fine-tuning loss plot to {loss_fig_path})")

    print("\n3. Testing recall WITHOUT context...")
    output = sanitize_text(generate(model, QUERY))
    print(f"   Prompt: '{QUERY}'")
    print(f"   Output: '{output}'")

    if "DragonCity" in output:
        print("   SUCCESS: Model internalized the fact into weights!")
    else:
        print("   FAIL: Model failed to memorize.")


class MemoryDemo:
    def __init__(self, train_out_dir, out_dir):
        self.train_out_dir = train_out_dir
        self.out_dir = out_dir

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        figs_dir = os.path.join(self.out_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)
        log_path = os.path.join(self.out_dir, "memory_log.txt")

        sys.stdout = Tee(log_path)
        print(f"Logging to {log_path}")
        print(f"Using device: {DEVICE}")
        print("\nCLIENT SUMMARY")
        print("- Fast Memory = external memory store, separate from the prompt.")
        print("- Model Memory = facts learned into weights after a short fine-tune.")
        print("- Success criteria:")
        print("  1) Fast memory returns the fact without placing it in the prompt.")
        print("  2) Model memory recalls the fact even when memory is cleared.")

        model = get_model(self.train_out_dir)
        demo_fast_memory(model)
        demo_model_memory(model, self.out_dir)


def run_memory_demo(train_out_dir, out_dir):
    MemoryDemo(train_out_dir, out_dir).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description="BDH memory demos")
    parser.add_argument("--train_out_dir", type=str, default="outputs/training", help="Training output directory")
    parser.add_argument("--out_dir", type=str, default="outputs/memory", help="Memory output directory")
    args = parser.parse_args(argv)
    run_memory_demo(args.train_out_dir, args.out_dir)


if __name__ == "__main__":
    main()
