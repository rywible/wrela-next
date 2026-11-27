"""The judge's ground forces against the simulator's (docs/plans/bodies-and-movement.md, phase 3's
check of the judge, and phase 4 step 8's filtered capture "with the contacts"): for clips the
tracker filtered, the world's force on the body as the judge works it out from the poses alone
(`tools/movement/judge -- human forces CLIP`) and as the simulator applied it (`filtered_extra.bin`:
the change of the body's momentum over each frame less its weight, the world's force averaged
over the frame), frame by frame, as a share of the body's weight.

    python3 tools/movement/contacts_check.py EXTRA POSES_ROOT [CLIP ...]

EXTRA is a run's `filtered_extra.bin`; POSES_ROOT the same run's filtered capture as poses
(`tools/movement/export -- human unfilter`). With no CLIP, every clip of EXTRA. It prints each
clip's mean and 95th percentile of the force's difference (all three axes, and the vertical) over
the weight, and over all clips. The judge against force plates under walking people: 3.1 to 7.9%
of the weight on average.
"""
import json
import os
import struct
import subprocess
import sys
import zlib

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHT = 72.73 * 9.81


def extras(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[4:8])[0]
    h = json.loads(b[8:8 + n])
    c = h["columns"]
    w = c["action"] + c["servo_force"] + c["contact_force"] + c.get("momentum", 0)
    x = np.frombuffer(zlib.decompress(b[8 + n:]), "<f2").reshape(-1, w).astype(np.float32)
    out, at = {}, 0
    k = c["action"] + c["servo_force"]
    for name, fr in zip(h["clips"], h["frames"]):
        e = x[at:at + fr]
        at += fr
        if "momentum" in c:
            # The world's force averaged over each frame: the momentum's change less the weight
            # (row k's from frame k - 1 to k).
            p = e[:, k + c["contact_force"]:]
            f = np.zeros((fr, 3), dtype=np.float32)
            f[1:] = (p[1:] - p[:-1]) * h["rate"]
            f[1:, 1] += h["mass"] * 9.81
            out[name] = f
        else:
            out[name] = e[:, k:k + c["contact_force"]].reshape(fr, -1, 3).sum(1)
    return out


def judged(poses_root, clip):
    r = subprocess.run([os.path.join(ROOT, "tools", "wrela"), "tool", os.path.join(ROOT, "tools", "movement", "judge"),
                        "--read", f"poses={poses_root}", "--read", "split=tools/movement/split",
                        "--read", "targets=tools/movement/judge/targets", "--", "human", "forces", clip],
                       cwd=ROOT, capture_output=True, text=True)
    rows = [l.split()[1:] for l in r.stdout.splitlines() if l.startswith("forces:")]
    return {int(f): np.array([float(v) for v in xyz]) for f, *xyz in rows}


def main():
    path, root = sys.argv[1], sys.argv[2]
    sim = extras(path)
    clips = sys.argv[3:] or sorted(sim)
    all_d, all_v = [], []
    for c in clips:
        if not os.path.exists(os.path.join(root, "human", f"{c}.pose")):
            continue
        j = judged(root, c)
        # The judge's force at frame f is from frames f - 1, f and f + 1 (a second difference); the
        # simulator's row k is its average over frame k - 1 to k: the two rows about f.
        s = sim[c]
        frames = [f for f in j if 0 < f < len(s) - 1]
        if not frames:
            continue
        mid = {f: 0.5 * (s[f] + s[f + 1]) for f in frames}
        d = np.array([np.linalg.norm(j[f] - mid[f]) for f in frames]) / WEIGHT
        v = np.array([abs(j[f][1] - mid[f][1]) for f in frames]) / WEIGHT
        all_d.append(d)
        all_v.append(v)
        print(f"{c}: {len(frames)} frames, the judge's force from the simulator's {100 * d.mean():.1f}% of the weight "
              f"on average ({100 * np.percentile(d, 95):.1f}% at the 95th percentile), vertical {100 * v.mean():.1f}%")
    if all_d:
        d, v = np.concatenate(all_d), np.concatenate(all_v)
        print(f"all: {len(all_d)} clips, {len(d)} frames: {100 * d.mean():.1f}% of the weight on average, "
              f"{100 * np.percentile(d, 95):.1f}% at the 95th percentile; vertical {100 * v.mean():.1f}%")


if __name__ == "__main__":
    main()
