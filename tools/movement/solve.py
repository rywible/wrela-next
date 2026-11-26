"""The solver's training (docs/plans/bodies-and-movement.md, phase 5 steps 3 to 8).

    .movement/venv/bin/python tools/movement/solve.py check SOURCE/CLIP...
        holds `motion.py`'s numbers to the engine's (`tools/movement/solver -- features`)

The poses are `.movement/poses/creature/SOURCE/CLIP.pose` (tools/movement/capture writes them;
the filtered capture of phase 4 is written the same way); the body is
`.movement/solver/creature.body.json` (`tools/movement/solver -- body`).
"""
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import motion  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POSES = os.path.join(ROOT, ".movement/poses/creature")
SOLVER = os.path.join(ROOT, ".movement/solver")


def read_pose(path):
    b = open(path, "rb").read()
    _, d, n = struct.unpack_from("<3I", b, 4)
    rate = struct.unpack_from("<f", b, 16)[0]
    x = np.frombuffer(b, dtype="<f4", offset=20, count=n * (7 + d)).reshape(n, 7 + d)
    return rate, x


def read_features(path):
    b = open(path, "rb").read()
    rows, ns, nt, ng = struct.unpack_from("<4I", b, 4)
    x = np.frombuffer(b, dtype="<f4", offset=20, count=rows * (ns + nt + ng)).reshape(rows, -1)
    return x[:, :ns], x[:, ns:ns + nt], x[:, ns + nt:]


def check(clips):
    body = motion.Body(f"{SOLVER}/creature.body.json")
    worst = {"state": 0.0, "step": 0.0, "goal": 0.0, "after": 0.0}
    for c in clips:
        rate, x = read_pose(f"{POSES}/{c}.pose")
        p = torch.tensor(np.array(x), dtype=torch.float32)
        es, et, eg = (torch.tensor(np.array(a)) for a in read_features(f"{SOLVER}/features/{c}.bin"))
        n = p.shape[0]
        f = torch.arange(1, n - 1)
        s = body.state(p[f - 1], p[f], rate)
        t = body.step(p[f], p[f + 1])
        ahead = int(rate)
        g_at = torch.clamp(f + ahead, max=n - 1)
        seconds = torch.clamp((g_at - f).float() / rate, min=1.0 / rate)
        g = body.goal(p[f], p[g_at], seconds)
        # The step back to the pose (unclamped: the capture is in range already).
        a = body.after(p[f], t, clamp=False)
        da = torch.cat([(a[:, :3] - p[f + 1, :3]).abs(),
                        motion.vector_of_turn(motion.qmul(a[:, 3:7], motion.qinv(p[f + 1, 3:7]))).abs(),
                        (a[:, 7:] - p[f + 1, 7:]).abs()], -1)
        for k, (mine, theirs) in {"state": (s, es), "step": (t, et), "goal": (g, eg)}.items():
            # Each number against the engine's, relative past a size of 1.
            d = ((mine - theirs).abs() / theirs.abs().clamp(min=1.0)).max().item()
            if d > worst[k]:
                worst[k] = d
                if d > 1e-3:
                    i = ((mine - theirs).abs() / theirs.abs().clamp(min=1.0)).max(0).values.argmax()
                    print(f"  {c} {k}: number {int(i)} differs by {d}")
        worst["after"] = max(worst["after"], da.max().item())
        print(f"{c}: {n} frames")
    print("check: the largest differences from the engine: " +
          ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))
    return 0 if max(worst.values()) < 1e-3 else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(check(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)


main()
