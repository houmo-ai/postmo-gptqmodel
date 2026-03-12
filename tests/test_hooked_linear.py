import torch

from gptqmodel.nn_modules.hooked_linear import HookedLinear


def test_hooked_linear_accepts_inference_tensor_inputs():
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    linear = torch.nn.Linear(8, 4, bias=False).to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    with torch.no_grad():
        linear.weight.copy_(torch.linspace(-0.5, 0.5, steps=linear.weight.numel(), device=device, dtype=linear.weight.dtype).reshape_as(linear.weight))
    hooked = HookedLinear.from_linear(linear)

    with torch.inference_mode():
        inp = torch.linspace(-1.0, 1.0, steps=16, device=device, dtype=linear.weight.dtype).reshape(2, 8)
        ref = linear(inp)
        out = hooked(inp)

    assert tuple(out.shape) == (2, 4)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref)
