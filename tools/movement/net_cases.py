"""Makes the engine's test of a net (phase 5 step 2): a net of random weights in PyTorch (37
inputs, layers of 64 and 48 with ELU, one of 16 with tanh, 11 outputs, the inputs normalised and
clipped), written as WNET, and 1000 inputs with PyTorch's outputs for them.

    .movement/venv/bin/python tools/movement/net_cases.py

Writes `lib/engine/learned/testdata/small.wnet` and `small.cases`. Seed 1.
"""
import os

import torch

import wnet

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "lib/engine/learned/testdata")


def main():
    g = torch.Generator().manual_seed(1)
    sizes = [37, 64, 48, 16, 11]
    acts = ["elu", "elu", "tanh", "none"]
    linears = [torch.nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])]
    with torch.no_grad():
        for lin in linears:
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) / lin.in_features ** 0.5 * 1.5)
            lin.bias.copy_(torch.randn(lin.bias.shape, generator=g) * 0.3)
    mean = torch.randn(37, generator=g)
    scale = torch.rand(37, generator=g) * 2 + 0.2
    clip = 4.0
    x = torch.randn(1000, 37, generator=g) * 3 + mean
    with torch.no_grad():
        h = ((x - mean) * scale).clamp(-clip, clip)
        for lin, act in zip(linears, acts):
            h = lin(h)
            h = {"elu": torch.nn.functional.elu, "tanh": torch.tanh, "none": lambda t: t}[act](h)
    os.makedirs(OUT, exist_ok=True)
    wnet.write(f"{OUT}/small.wnet", linears, acts, mean, scale, clip)
    wnet.cases(f"{OUT}/small.cases", x, h)
    print(f"net_cases: {sum(l.weight.numel() + l.bias.numel() for l in linears)} weights, "
          f"outputs in [{h.min():.3f}, {h.max():.3f}]")


main()
