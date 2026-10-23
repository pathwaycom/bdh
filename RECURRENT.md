# Recurrent (Hebbian synapse) inference for BDH

`bdh_recurrent.py` runs BDH one token at a time with a fixed-size state instead of
re-reading the whole text for every new token.

## Idea

BDH attention has no softmax:

    y_t = sum over s<t of (QR_t . QR_s) x_s

where QR is the sparse neuron activity after RoPE. Regrouping the sum:

    y_t = QR_t @ S_t,    S_t = sum over s<t of QR_s^T x_s

`S` has shape (N x D) per head, per layer. After each token, `QR^T x` is added to it:
the connection between a neuron and a value strengthens when that neuron fires for
that token. This is the Hebbian synapse view described in the BDH paper
(arXiv 2509.26507). The result is mathematically identical to the full forward pass.
RoPE only needs the token's absolute position, which the state tracks.

## What is here

- `bdh_recurrent.py`: `BDHState`, `step` (one token), `generate_recurrent` (same interface as `BDH.generate`).
- `test_recurrent.py`: checks that logits from the recurrent route match `model(idx)` at every position,
  and that both generate routes give the same text with the same sampling seed.
- `bench_recurrent.py`: time and memory for producing one token at different context lengths.

No existing file is modified.

## Memory

The state holds n_layer x n_head x N x D numbers. For the 25M model (6 layers, 4 heads,
N = 8192, D = 256) that is about 50M numbers (about 200 MB in fp32), and it does not grow with context length.
A cache of past tokens needs about 33,000 numbers per token per layer, so the two cost the same
at roughly 254 tokens; beyond that the state is smaller.

## Benchmark (RTX 3050 4 GB, fp32, random weights, one next token)

| Context T | Full forward ms | Full extra MB | Recurrent ms | Recurrent extra MB |
| --- | --- | --- | --- | --- |
| 256 | TBD | TBD | TBD | TBD |
| 512 | TBD | TBD | TBD | TBD |
| 1024 | TBD | TBD | TBD | TBD |
| 2048 | TBD | TBD | TBD | TBD |

"Full forward" is what `BDH.generate` does today (no cache). A fair comparison with a
token cache is not measured here.

## Limits

- Inference only; training still uses the parallel form.
- The prompt is read one token at a time, so reading a long prompt is slower than one parallel pass.
- The state is kept in fp32; lower precision was not tested.
- Dropout is off (evaluation mode).
- The equivalence test passes on a small randomly initialised BDH built from this repository's code;
  it has not yet been checked on trained weights.
