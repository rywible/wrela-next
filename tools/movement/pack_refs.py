"""Packs the states of a set's clips (`.movement/sim/refs/SOURCE/CLIP.ref`, which
tools/movement/export writes) into one file for a training run: their qpos only (the tracker
works out each segment's place with the simulator's own kinematics), compressed.

    python3 tools/movement/pack_refs.py SET [SET ...] [--most N] -o .movement/sim/refs.bin

(a SET is a name of tools/movement/split, or the path of a list of clips ending in .txt)

The file: "WRFS", a u32 length, a JSON head (the clips' names, each one's first frame and length,
the qpos size and the rate), then zlib's compression of every frame's qpos (float32), clip after
clip.
"""
import argparse
import json
import os
import struct
import zlib

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sets", nargs="+")
    p.add_argument("--most", type=int, default=0, help="at most this many clips of each set")
    p.add_argument("-o", required=True)
    a = p.parse_args()
    names, starts, lengths, chunks = [], [], [], []
    nq, rate, total, missing = None, None, 0, 0
    for s in a.sets:
        path = s if s.endswith(".txt") else os.path.join(ROOT, "tools/movement/split", f"{s}.txt")
        lines = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
        if a.most:
            lines = lines[:a.most]
        for name in lines:
            path = os.path.join(ROOT, ".movement/sim/refs", f"{name}.ref")
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
            names.append(name)
            starts.append(total)
            lengths.append(n)
            total += n
            chunks.append(np.ascontiguousarray(data))
    body = zlib.compress(np.concatenate(chunks).astype("<f4").tobytes(), 6)
    head = json.dumps({"names": names, "starts": starts, "lengths": lengths, "nq": nq,
                       "rate": rate}).encode()
    with open(a.o, "wb") as f:
        f.write(b"WRFS" + struct.pack("<I", len(head)) + head + body)
    print(f"pack: {len(names)} clips ({missing} not exported), {total} frames ({total / rate / 3600:.2f} h), "
          f"{os.path.getsize(a.o) / 1e6:.1f} MB")


main()
