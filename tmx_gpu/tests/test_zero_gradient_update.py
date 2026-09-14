"""A selected fallback batch must advance the real Adam update counter."""
import numpy as np
import pytest
import torch
from tmx_gpu import grpo_gpu


@pytest.mark.parametrize("masked", [False, True])
def test_zero_gradient_batch_matches_an_explicit_adam_step(masked):
    model = torch.nn.Linear(1, 1, bias=False)
    reference = torch.nn.Linear(1, 1, bias=False)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, betas=(.9, .95), weight_decay=0)
    expected = torch.optim.AdamW(reference.parameters(), lr=.01, betas=(.9, .95), weight_decay=0)
    for network, opt in ((model, optimizer), (reference, expected)):
        network.weight.grad = torch.ones_like(network.weight)
        opt.step()
        opt.zero_grad(set_to_none=True)
    args = grpo_gpu.build_arg_parser().parse_args(["--output-dir", "/tmp/unused", "--device", "cpu"])
    args.mask_truncated = masked
    rows = [{"prompt_text": "p", "_comp_ids": [1], "clipped": masked}]
    update = grpo_gpu._train_update(model, optimizer, None, rows, np.array([1. if masked else 0.]), args)
    reference.weight.grad = torch.zeros_like(reference.weight)
    expected.step()
    torch.testing.assert_close(model.weight, reference.weight)
    assert int(optimizer.state[model.weight]["step"]) == 2
    assert update["updates"] == 1
    assert update["rows_backward"] == 0
