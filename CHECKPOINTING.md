## Activation checkpointing for BDH

Set `checkpoint_layers=True` in `BDHConfig` to recompute each layer's activations
during the backward pass instead of storing them. It only applies in training mode.

Measured on Tiny Shakespeare (byte-level, ~25M params, batch 4, context 256, 2000 steps,
RTX 3050 4 GB, 3 seeds):

| Seed | Val loss (default) | Val loss (checkpoint_layers=True) |
| --- | --- | --- |
| 1337 | 1.513 | 1.513 |
| 1 | 1.530 | 1.530 |
| 2 | 1.515 | 1.515 |

| | default | checkpoint_layers=True |
| --- | --- | --- |
| Peak GPU memory | 3,355 MB | 976 MB (-71%) |
| Training time (seed 1337) | 1,018 s | 1,366 s (+34%) |

The forward pass is deterministic, so the recomputed activations are identical.
`test_checkpoint.py` checks the outputs and gradients match.
Uses `torch.utils.checkpoint` with `use_reentrant=False`.
