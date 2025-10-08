# BDH-MLX Training Guide for Internal Knowledge Map Dataset

## Overview

This guide explains how to train the BDH model using your **Internal Knowledge Map** dataset with the phased training methodology.

## Dataset Structure

Your dataset has three key fields:
- **system**: Task overviews, guidelines, and objectives (contextual framework)
- **instruction**: Specific prompts and questions (detailed instructions)
- **response**: Comprehensive answers (complete knowledge)

## Training Modes

The training script supports four modes:

### 1. **"system"** - Phase 1 Training
```python
TRAINING_MODE = "system"
```
- **Focus**: System guidelines only
- **Purpose**: Build foundational understanding of contextual frameworks
- **Use case**: First phase of phased training
- **Output format**: Raw system text

### 2. **"instruction"** - Phase 2 Training
```python
TRAINING_MODE = "instruction"
```
- **Focus**: Instructions only
- **Purpose**: Learn to parse and respond to specific prompts
- **Use case**: Second phase after system training
- **Output format**: Raw instruction text

### 3. **"both"** - Combined Training (RECOMMENDED)
```python
TRAINING_MODE = "both"
```
- **Focus**: System + Instruction together
- **Purpose**: Learn context AND specific prompts simultaneously
- **Use case**: Single-pass training with both contexts
- **Output format**: 
```
### System:
[system text]

### Instruction:
[instruction text]
```

### 4. **"full"** - Complete Training
```python
TRAINING_MODE = "full"
```
- **Focus**: System + Instruction + Response
- **Purpose**: Complete understanding including expected outputs
- **Use case**: Traditional supervised learning
- **Output format**: 
```
### System:
[system text]

### Instruction:
[instruction text]

### Response:
[response text]
---
```

## Recommended Configurations

### For 64GB Mac Silicon (Your Setup)

**Option 1: Combined Training (Fastest, Recommended)**
```python
TRAINING_MODE = "both"
BLOCK_SIZE = 4096
BATCH_SIZE = 2
MAX_ITERS = 5000
LEARNING_RATE = 5e-5
EPOCHS = 3
```

**Option 2: Full Training (Most Complete)**
```python
TRAINING_MODE = "full"
BLOCK_SIZE = 4096
BATCH_SIZE = 2
MAX_ITERS = 8000
LEARNING_RATE = 3e-5
EPOCHS = 3
```

**Option 3: Phased Training (Most Methodical)**

Phase 1:
```python
TRAINING_MODE = "system"
BLOCK_SIZE = 4096
BATCH_SIZE = 2
MAX_ITERS = 3000
LEARNING_RATE = 1e-4
```

Then Phase 2:
```python
TRAINING_MODE = "instruction"
BLOCK_SIZE = 4096
BATCH_SIZE = 2
MAX_ITERS = 3000
LEARNING_RATE = 5e-5
```

### For Smaller Macs (16-32GB)

```python
TRAINING_MODE = "both"
BLOCK_SIZE = 2048  # Reduced context
BATCH_SIZE = 1
MAX_ITERS = 5000
LEARNING_RATE = 5e-5
```

## How to Change Training Mode

Edit `train_mlx.py` and change the `TRAINING_MODE` variable at the top:

```python
# Line ~29 in train_mlx.py
TRAINING_MODE = "both"  # Change this to: "system", "instruction", "both", or "full"
```

## Expected Performance

### Training Speed (64GB Mac Silicon)
- **Tokens/second**: ~3,000-5,000
- **Time per 1000 iterations**: ~15-25 minutes
- **Full training (5000 iters)**: ~1.5-2.5 hours

### Dataset Statistics
- **Total examples**: ~4,685 entries
- **Total bytes**: ~5M bytes
- **Context window**: 4096 bytes (~4KB per sample)
- **Training samples**: ~90% (~4.2M bytes)
- **Validation samples**: ~10% (~500K bytes)

## Training Process

1. **Start training**:
```bash
python train_mlx.py
```

2. **Monitor progress**:
   - Loss logged every 50 iterations
   - Validation every 250 iterations
   - Sample generation every 250 iterations
   - Checkpoints saved every 500 iterations

3. **Checkpoints saved to**: `checkpoints_mlx/`

## Interpreting Results

### Good Signs
- ✓ Loss decreasing steadily
- ✓ Validation loss following training loss
- ✓ Generated samples become more coherent
- ✓ Model learns the "### System:" and "### Instruction:" structure

### Warning Signs
- ⚠️ Validation loss increasing (overfitting)
- ⚠️ Training loss stuck (learning rate too low)
- ⚠️ Loss oscillating wildly (learning rate too high)
- ⚠️ Generated text is gibberish (needs more training)

## Sample Generation Prompts

The script will generate samples based on training mode:

**System mode**:
- "### System:\nTask Overview:"
- "### System:\nGuidelines:"

**Instruction mode**:
- "### Instruction:\nAnalyze"
- "### Instruction:\nExplain"

**Both mode**:
- "### System:\nTask Overview: Analyze and explore\n\n### Instruction:\n"
- "### System:\nGuidelines: Focus on core interactions\n\n### Instruction:\n"

**Full mode**:
- "### System:\nTask Overview:"
- "### Instruction:\nAnalyze the ethical implications"
- "### Response:\n"

## Advanced: Custom Generation

After training, load the model and generate:

```python
import mlx.core as mx
import bdh_mlx

# Load model
config = bdh_mlx.BDHConfig()
model = bdh_mlx.BDH(config)

# Load checkpoint (you'll need to implement loading)
# ...

# Create prompt
prompt_text = """### System:
Task Overview: Analyze ethical implications

### Instruction:
Explain the concept of knowledge"""

prompt_bytes = list(bytearray(prompt_text, "utf-8"))
idx = mx.array([prompt_bytes])

# Generate
output = model.generate(idx, max_new_tokens=500, temperature=0.8, top_k=50)

# Decode
output_text = bytes(output[0].tolist()).decode("utf-8", errors="backslashreplace")
print(output_text)
```

## Memory Management

### If you get OOM (Out of Memory):

1. **Reduce BLOCK_SIZE**:
   - Try 2048 or 1024

2. **Reduce BATCH_SIZE**:
   - Try 1

3. **Reduce model size**:
```python
BDH_CONFIG = bdh_mlx.BDHConfig(
    n_layer=4,      # Reduced from 6
    n_embd=128,     # Reduced from 256
    n_head=2,       # Reduced from 4
)
```

## Phased Training Strategy

According to your dataset methodology:

### Traditional Phased Approach

**Step 1**: Train on system (context building)
```bash
# Edit train_mlx.py: TRAINING_MODE = "system"
python train_mlx.py
# Takes ~1-1.5 hours
```

**Step 2**: Train on instruction (fine-tuning)
```bash
# Edit train_mlx.py: TRAINING_MODE = "instruction"
# Load checkpoint from Phase 1 (you'll need to implement loading)
python train_mlx.py
# Takes ~1-1.5 hours
```

### Modern Combined Approach (Recommended)

**Single Pass**: Train on both simultaneously
```bash
# Edit train_mlx.py: TRAINING_MODE = "both"
python train_mlx.py
# Takes ~1.5-2.5 hours
# Often works better than phased for transformers!
```

## Tips for Best Results

1. **Start with "both" mode** - It's simpler and often works better
2. **Monitor the generated samples** - They tell you if the model is learning
3. **Save checkpoints frequently** - Don't lose progress
4. **Experiment with learning rate** - 5e-5 is a good starting point
5. **Increase context window** if you have memory - Longer context = better understanding
6. **Be patient** - Good results take 2-3 epochs minimum

## Troubleshooting

### Problem: Dataset not loading
**Solution**: Make sure you're authenticated with HuggingFace:
```bash
huggingface-cli login
```

### Problem: Training is slow
**Solution**: 
- Close other applications
- Check Activity Monitor for GPU usage
- Reduce BLOCK_SIZE to 2048

### Problem: Loss not decreasing
**Solution**:
- Increase learning rate to 1e-4
- Check if data is loading correctly
- Verify model is not too small

### Problem: Validation loss increasing
**Solution**:
- Reduce learning rate
- Add more dropout
- Stop training (early stopping)

## Next Steps After Training

1. **Test generation** with various prompts
2. **Evaluate on specific tasks** from your dataset
3. **Fine-tune** on downstream tasks if needed
4. **Share your results** - This is a unique training approach!

## Questions?

The training script will guide you through:
- ✓ Automatic dataset structure detection
- ✓ Mode-specific sample generation
- ✓ Progress tracking
- ✓ Checkpoint management

Just run `python train_mlx.py` and watch the magic happen! 🐉

