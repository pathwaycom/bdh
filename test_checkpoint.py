import torch
import bdh


def run(flag):
    torch.manual_seed(0)
    cfg = bdh.BDHConfig(
        n_layer=2, n_embd=32, n_head=2,
        mlp_internal_dim_multiplier=4, checkpoint_layers=flag,
    )
    model = bdh.BDH(cfg)
    model.train()
    idx = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    torch.manual_seed(1)
    _, loss = model(idx, targets)
    loss.backward()
    return loss.item(), [p.grad.clone() for p in model.parameters()]


def test_checkpointing_matches_plain_forward_backward():
    loss_off, grads_off = run(False)
    loss_on, grads_on = run(True)
    assert abs(loss_off - loss_on) < 1e-6
    for a, b in zip(grads_off, grads_on):
        assert torch.allclose(a, b, atol=1e-6)