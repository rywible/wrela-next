"""Writes a net trained in PyTorch as the engine reads it (`lib/engine/learned/net.wrela`, WNET):

    "WNET", u32 version (1), u32 inputs, u32 layers,
    the inputs' normalisation: f32 mean[inputs], f32 scale[inputs] (1 / spread), f32 clip,
    then each layer: u32 ins, u32 outs, u32 activation (0 none, 1 ELU, 2 tanh),
                     f32 weights[outs][ins] (row after row), f32 bias[outs].

All little-endian. `write(path, linears, activations, mean, scale, clip)` takes the `nn.Linear`s
in order; `cases(path, net, inputs)` writes inputs and the net's outputs for the engine's test.
"""
import struct

import numpy as np

ACTS = {"none": 0, "elu": 1, "tanh": 2}


def encode(linears, activations, mean, scale, clip):
    n_in = linears[0].in_features
    out = [b"WNET", struct.pack("<3I", 1, n_in, len(linears))]
    out.append(np.asarray(mean, dtype="<f4").reshape(n_in).tobytes())
    out.append(np.asarray(scale, dtype="<f4").reshape(n_in).tobytes())
    out.append(struct.pack("<f", clip))
    for lin, act in zip(linears, activations):
        w = lin.weight.detach().cpu().numpy().astype("<f4")
        b = lin.bias.detach().cpu().numpy().astype("<f4")
        out.append(struct.pack("<3I", lin.in_features, lin.out_features, ACTS[act]))
        out.append(np.ascontiguousarray(w).tobytes())
        out.append(b.tobytes())
    return b"".join(out)


def write(path, linears, activations, mean, scale, clip):
    open(path, "wb").write(encode(linears, activations, mean, scale, clip))


def cases(path, inputs, outputs):
    """"WCAS", u32 count, u32 inputs, u32 outputs, then each case's inputs and outputs (f32)."""
    inputs = np.asarray(inputs, dtype="<f4")
    outputs = np.asarray(outputs, dtype="<f4")
    with open(path, "wb") as f:
        f.write(b"WCAS" + struct.pack("<3I", inputs.shape[0], inputs.shape[1], outputs.shape[1]))
        for x, y in zip(inputs, outputs):
            f.write(x.tobytes() + y.tobytes())
