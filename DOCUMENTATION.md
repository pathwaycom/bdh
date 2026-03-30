# Baby Dragon Hatchling (BDH) Guide

This guide explains what the project does, how to run it end-to-end, and how to interpret the outputs.

## What This Generates

- Training checkpoints at every 10% of the run.
- A consolidated generation log and a Markdown report comparing checkpoints.
- A loss curve plot (loss vs steps).
- A memory run log and a small loss plot for the model-memory fine-tune.

## Deliverables Summary

1. **End-to-End Training**: `train.py` supports configurable steps and automatic checkpointing at every 10% of progress.
2. **Checkpoint Comparison**: `compare_checkpoints.py` evaluates all checkpoints with a fixed prompt and creates a report + consolidated log.
3. **Memory Script**: `memory.py` demonstrates:
   - **Fast Memory (Context)**: A lightweight memory store ingests facts from a prompt and answers from memory without keeping the fact in the user prompt.
   - **Model Memory (Weights)**: Facts consolidated into model weights via a short fine-tune.
4. **One-Click Run**: `main.py` runs training, evaluation, and memory demos in order.

## How to Run

### Using Docker (Recommended for Reproducibility)

1.  **Build the CPU image**:

    ```bash
    docker build -t bdh-project -f Dockerfile .
    ```

2.  **Build the GPU image** (CUDA):

    ```bash
    docker build -t bdh-project-gpu -f Dockerfile.gpu .
    ```

3.  **Run Training** (saves checkpoints to `outputs/`):

    ```bash
    docker run -v $(pwd)/outputs:/app/outputs bdh-project python train.py --max_iters 1000
    ```

4.  **Compare Checkpoints**:

    ```bash
      docker run -v $(pwd)/outputs:/app/outputs bdh-project python compare_checkpoints.py --input_dir outputs/training/checkpoints --output_dir outputs/evaluation
    ```

5.  **Run Memory Script** (requires training checkpoints):
    ```bash
    docker run -v $(pwd)/outputs:/app/outputs bdh-project python memory.py
    ```
6.  **Run All in Order**:
    ```bash
    docker run -v $(pwd)/outputs:/app/outputs/checkpoints bdh-project python main.py --max_iters 1000
    ```

**GPU usage**:

```bash
docker run --gpus all -v $(pwd)/outputs:/app/outputs bdh-project-gpu python main.py --max_iters 1000
```

### Running Locally

Ensure you have Python and PyTorch installed.

1.  **Install requirements**:

    ```bash
    pip install numpy torch requests psutil pandas matplotlib
    ```

2.  **Train**:

    ```bash
    python train.py --max_iters 1000 --out_dir outputs
    ```

    - You can control training length with `--max_iters` (e.g., 100 for a quick run, 3000 for full training).
    - CPU training is slower; more iterations mean longer runtime. Use smaller `--max_iters` for quick tests.

3.  **Visuals & Logs**:

    - Training logs (loss vs step) are saved to `outputs/training/logs/training_log.csv`.
    - Run `python compare_checkpoints.py --input_dir outputs/training/checkpoints` to see the text generation progress.
    - Consolidated generations are written to `outputs/evaluation/checkpoint_generations.log`.

4.  **Run Memory Script** (requires training checkpoints):
    ```bash
    python memory.py
    ```
5.  **Run All in Order**:
    ```bash
    python main.py --max_iters 1000
    ```

## Outputs and Where to Find Them

- Training outputs: `outputs/training/`
  - Checkpoints: `outputs/training/checkpoints/`
  - Training log CSV: `outputs/training/logs/training_log.csv`
- Evaluation outputs: `outputs/evaluation/`
  - Report: `outputs/evaluation/report.md`
  - Consolidated generations log: `outputs/evaluation/checkpoint_generations.log`
  - Loss plot: `outputs/evaluation/figs/loss_curve.png`
  - Checkpoint samples: `outputs/evaluation/figs/checkpoint_samples.png`
  - Output diversity plot: `outputs/evaluation/figs/output_quality.png`
  - Multi-prompt runs: `outputs/evaluation/multi_prompt/`
- Memory outputs: `outputs/memory/`
  - Memory log: `outputs/memory/memory_log.txt`
  - Fine-tune loss plot: `outputs/memory/figs/memory_fine_tuning_loss.png`

## Results Interpretation

- **Training**: You will see the loss decrease over time. Around 100-200 steps, the model starts forming recognizable words. By 1000 steps, it should produce Shakespeare-like structure.
- **Checkpoint Comparison**: Early checkpoints produce noise; later checkpoints produce English-like text.
- **Fast Memory**: The memory layer recalls facts without putting them in the user prompt (and the log also shows the model attempt with memory-injected prompt for transparency).
- **Model Memory**: After fine-tuning, the model recalls the fact even without any context provided at runtime.

## What the Loss Curve Means

- The loss curve (`outputs/evaluation/figs/loss_curve.png`) shows how prediction error changes over training steps.
- Lower loss means the model is better at predicting the next character/byte.
- A steady downward trend is expected; it indicates the model is learning patterns from the dataset.
- Minor bumps are normal (training is noisy), but the overall trend should go down.
- If validation loss is shown, it should track training loss but stay slightly higher.

## Additional Visuals Included

- **Checkpoint samples**: A small panel showing early/mid/final outputs for quick comparison.
- **Output diversity**: Unique character ratio per checkpoint to show output variation over time.
- **Before vs After table**: Inline table in the report with early/mid/late outputs.
- **Multi-prompt checks**: Same checkpoints evaluated on 3 prompts for consistency.
