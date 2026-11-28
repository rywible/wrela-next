"""Packs the states of a set's clips (`.movement/sim/refs/SOURCE/CLIP.ref`, which
tools/movement/export writes) into one file for a training run: their qpos only (the tracker
works out each segment's place with the simulator's own kinematics), compressed.

    python3 tools/movement/pack_refs.py SET [SET ...] [--most N] -o .movement/sim/refs.bin

(a SET is a name of tools/movement/split, or the path of a list of clips ending in .txt)

The file: "WRFS", a u32 length, a JSON head (the clips' names, each one's first frame and length,
the qpos size, the rate, and each clip's set: a training run starts its worlds on training clips
alone, and evaluates the others apart), then zlib's compression of every frame's qpos (float32), clip after
clip.
"""
import argparse
import json
import os
import struct
import zlib

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def longest_clean_span(data, rate, max_root_speed, max_joint_speed, max_start_accel=10.0):
    """Longest contiguous frame slice with no unphysical root teleport or joint jump, with leading
    IK startup transients (impossible initial root acceleration > `max_start_accel` m/s^2) trimmed."""
    n = data.shape[0]
    if n < 2:
        return 0, n
    bad = np.zeros(n - 1, dtype=bool)
    if max_root_speed > 0:
        v_root = np.linalg.norm(data[1:, :3] - data[:-1, :3], axis=-1) * rate
        bad |= v_root > max_root_speed
    if max_joint_speed > 0 and data.shape[1] > 7:
        v_jnt = np.abs(data[1:, 7:] - data[:-1, 7:]).max(axis=-1) * rate
        bad |= v_jnt > max_joint_speed
    if bad.any():
        cuts = [-1] + list(np.nonzero(bad)[0]) + [n - 1]
        best_a, best_b = 0, 0
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            a, b = int(c0 + 1), int(c1 + 1)
            if b - a > best_b - best_a:
                best_a, best_b = a, b
    else:
        best_a, best_b = 0, n
    if max_start_accel > 0 and best_b - best_a > 35:
        seg = data[best_a:best_b]
        v = np.diff(seg[:, :3], axis=0) * rate
        acc = np.linalg.norm(np.diff(v, axis=0) * rate, axis=-1)
        k = 0
        cap = min(10, seg.shape[0] - 35)
        while k < cap:
            if acc[k] > max_start_accel or np.linalg.norm(v[k] - v[k + 2]) > 0.35:
                k += 1
            else:
                break
        best_a += k
    return best_a, best_b


def resampled(q, rate, to):
    """A clip's qpos at `to` frames a second from `rate`: the places and the joints' numbers
    linearly, the root's turn on the sphere (normalised, its sign kept from frame to frame)."""
    n = q.shape[0]
    t = np.arange(0.0, (n - 1) / rate + 1e-9, 1.0 / to) * rate
    i = np.minimum(np.floor(t).astype(int), n - 2)
    u = (t - i)[:, None]
    a, b = q[i], q[i + 1]
    out = a + (b - a) * u
    qa, qb = a[:, 3:7], b[:, 3:7].copy()
    qb[(qa * qb).sum(-1) < 0] *= -1.0
    r = qa * (1 - u) + qb * u
    out[:, 3:7] = r / np.linalg.norm(r, axis=-1, keepdims=True)
    return out.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sets", nargs="+")
    p.add_argument("--most", type=int, default=0, help="at most this many clips of each set")
    p.add_argument("-o", required=True)
    p.add_argument("--sim", default=".movement/sim", help="the export's folder (its refs/)")
    p.add_argument("--max-root-speed", type=float, default=5.0,
                   help="cut clips at frames where root speed exceeds this (m/s); 0 disables")
    p.add_argument("--max-joint-speed", type=float, default=25.0,
                   help="cut clips at frames where any joint speed exceeds this (rad/s); 0 disables")
    p.add_argument("--max-start-accel", type=float, default=10.0,
                   help="trim leading IK startup transients where root accel exceeds this (m/s^2); 0 disables")
    p.add_argument("--rate", type=float, default=0.0,
                   help="resample each clip to this many frames a second (0: as exported); the tracker acts once a "
                        "frame, so 60 halves the delay of its loop")
    p.add_argument("--min-frames", type=int, default=60,
                   help="drop clean spans shorter than this many frames")
    a = p.parse_args()
    names, starts, lengths, chunks, which = [], [], [], [], []
    nq, rate, total, missing, trimmed, dropped = None, None, 0, 0, 0, 0
    for s in a.sets:
        path = s if s.endswith(".txt") else os.path.join(ROOT, "tools/movement/split", f"{s}.txt")
        lines = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
        if a.most:
            lines = lines[:a.most]
        for name in lines:
            path = os.path.join(ROOT, a.sim, "refs", f"{name}.ref")
            if not os.path.exists(path):
                missing += 1
                continue
            b = open(path, "rb").read()
            _, q, segs, n = struct.unpack_from("<4I", b, 4)
            r = struct.unpack_from("<f", b, 20)[0]
            row = q + 7 * segs
            data = np.frombuffer(b, dtype="<f4", offset=24, count=n * row).reshape(n, row)[:, :q]
            assert nq in (None, q) and rate in (None, r)
            nq, rate = q, r
            i0, i1 = longest_clean_span(data, r, a.max_root_speed, a.max_joint_speed, a.max_start_accel)
            if i1 - i0 < a.min_frames:
                dropped += 1
                continue
            if i0 > 0 or i1 < n:
                trimmed += 1
                data = data[i0:i1]
                n = data.shape[0]
            if a.rate and a.rate != r:
                data = resampled(data, r, a.rate)
                n = data.shape[0]
            names.append(name)
            which.append(os.path.basename(s)[:-4] if s.endswith(".txt") else s)
            starts.append(total)
            lengths.append(n)
            total += n
            chunks.append(np.ascontiguousarray(data))
    body = zlib.compress(np.concatenate(chunks).astype("<f4").tobytes(), 6)
    head = json.dumps({"names": names, "starts": starts, "lengths": lengths, "nq": nq,
                       "rate": a.rate or rate, "sets": which}).encode()
    with open(a.o, "wb") as f:
        f.write(b"WRFS" + struct.pack("<I", len(head)) + head + body)
    print(f"pack: {len(names)} clips ({missing} not exported, {trimmed} trimmed at spikes, {dropped} dropped), "
          f"{total} frames ({total / rate / 3600:.2f} h), {os.path.getsize(a.o) / 1e6:.1f} MB")


main()
