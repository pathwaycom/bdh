"""Tests for phase-preserving latent-width growth (extend_freqs)."""
import torch
from bdh import get_freqs, extend_freqs


def test_extend_preserves_prefix_exactly():
    f64 = get_freqs(64, theta=2**16, dtype=torch.float32)
    f96 = extend_freqs(f64, 96)
    assert f96.shape[-1] == 96
    # every existing neuron keeps its frequency verbatim
    torch.testing.assert_close(f96[:64], f64)


def test_naive_recompute_changes_prefix():
    # documents the pitfall motivating extend_freqs: recomputing at the new
    # neuron count rewrites the phases of all existing neurons
    f64 = get_freqs(64, theta=2**16, dtype=torch.float32)
    f96_naive = get_freqs(96, theta=2**16, dtype=torch.float32)
    assert not torch.allclose(f96_naive[:64], f64)


def test_attention_builds_with_extended_table():
    from bdh import BDHConfig, Attention
    cfg = BDHConfig(n_head=2, n_embd=8, mlp_internal_dim_multiplier=32)
    attn = Attention(cfg)
    f = extend_freqs(attn.freqs, attn.freqs.shape[-1] + 16)
    assert f.shape[-1] == attn.freqs.shape[-1] + 16


if __name__ == "__main__":
    test_extend_preserves_prefix_exactly()
    test_naive_recompute_changes_prefix()
    test_attention_builds_with_extended_table()
    print("all tests passed")
