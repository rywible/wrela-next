"""The solver's training (docs/plans/bodies-and-movement.md, phase 5 steps 3 to 8).

    .movement/venv/bin/python tools/movement/solve.py check SOURCE/CLIP...
        holds `motion.py`'s numbers to the engine's (`tools/movement/solver -- features`)
    .movement/venv/bin/python tools/movement/solve.py pack SET... -o .movement/solver/poses.bin
        the poses of the sets' clips in one file (WPSE: a u32 length, a JSON head of the clips'
        names, sets, first rows and lengths, the rate and D, then zlib of the f32 rows)
    .movement/venv/bin/python tools/movement/solve.py --body human pack SET... -o FILE
        (each command but the run takes `--body NAME` first: human or creature, the default)
    python3 solve.py --out DIR --seed N [--resume CKPT] --poses poses.bin --body creature.body.json ...
        a training run (a job of tools/movement/runlib.py, here or by tools/movement/colab-run)

The net (steps 5 to 8): a conditional variational autoencoder of the step (the method of motion
VAEs, H. Y. Ling et al. 2020, with goals as its condition). The encoder reads the state, the
goal and the true step and gives a code z; the decoder reads the state, the goal (each part
hidden at random, its flag beside it) and z, and gives the step. The game runs the decoder alone,
z drawn from a seed. Training rolls the decoder out over windows of frames and feeds its own
steps back (scheduled sampling: the share fed back rises to all), so its small errors don't grow.

The poses are `.movement/poses/BODY/SOURCE/CLIP.pose` (tools/movement/capture writes them;
the filtered capture of phase 4 is written the same way); the body is
`.movement/solver/BODY.body.json` (`tools/movement/solver -- BODY body`).
"""
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import motion  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOLVER = os.path.join(ROOT, ".movement/solver")
# The body (`--body NAME`: human or creature, the default), as tools/movement/bodies names it.
BODY = "creature"
POSES = os.path.join(ROOT, ".movement/poses", BODY)


def set_body(name, poses_root=None):
    global BODY, POSES
    BODY = name
    root = os.path.join(ROOT, ".movement/poses") if poses_root is None else poses_root
    POSES = os.path.join(root, name)


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
    body = motion.Body(f"{SOLVER}/{BODY}.body.json")
    worst = {"state": 0.0, "step": 0.0, "goal": 0.0, "after": 0.0}
    for c in clips:
        rate, x = read_pose(f"{POSES}/{c}.pose")
        p = torch.tensor(np.array(x), dtype=torch.float32)
        es, et, eg = (torch.tensor(np.array(a)) for a in read_features(f"{SOLVER}/features/{BODY}/{c}.bin"))
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


def read_wnet(path):
    """A WNET file as numpy arrays: (mean, scale, clip, [(W, b, activation)])."""
    b = open(path, "rb").read()
    assert b[:4] == b"WNET"
    _, n_in, count = struct.unpack_from("<3I", b, 4)
    at = 16
    mean = np.frombuffer(b, "<f4", n_in, at)
    scale = np.frombuffer(b, "<f4", n_in, at + 4 * n_in)
    at += 8 * n_in
    clip = struct.unpack_from("<f", b, at)[0]
    at += 4
    layers = []
    for _ in range(count):
        i, o, act = struct.unpack_from("<3I", b, at)
        at += 12
        w = np.frombuffer(b, "<f4", i * o, at).reshape(o, i)
        at += 4 * i * o
        bias = np.frombuffer(b, "<f4", o, at)
        at += 4 * o
        layers.append((torch.tensor(np.array(w)), torch.tensor(np.array(bias)), act))
    return torch.tensor(np.array(mean)), torch.tensor(np.array(scale)), clip, layers


def apply_wnet(net, x):
    mean, scale, clip, layers = net
    h = ((x - mean) * scale).clamp(-clip, clip)
    for w, b, act in layers:
        h = h @ w.T + b
        if act == 1:
            h = torch.nn.functional.elu(h)
        elif act == 2:
            h = torch.tanh(h)
    return h


def pcg(h):
    h = np.uint32(h)
    x = np.uint32((np.uint64((h >> np.uint32((h >> np.uint32(28)) + np.uint32(4))) ^ h) * np.uint64(277803737)) & np.uint64(0xFFFFFFFF))
    return np.uint32(x >> np.uint32(22)) ^ x


def normal(h):
    """`engine::body::motion`'s number of the standard normal distribution from `h`."""
    h = np.uint32(h)
    a = np.uint32((np.uint64(h) * 2 + 1) & 0xFFFFFFFF)
    b = np.uint32((np.uint64(h) * 2 + 2) & 0xFFFFFFFF)
    u = (np.float32(pcg(a) >> np.uint32(8)) + np.float32(0.5)) / np.float32(16777216.0)
    v = np.float32(pcg(b) >> np.uint32(8)) / np.float32(16777216.0)
    return float(np.sqrt(-2.0 * np.log(u)) * np.cos(6.2831853 * v))


CODE_TICKS = 8


def code(seed, tick, k):
    """`engine::body::motion::code`."""
    knot = tick // CODE_TICKS
    u = np.float32(tick % CODE_TICKS) / np.float32(CODE_TICKS)
    ha = pcg(np.uint32((seed + knot * 2654435761) & 0xFFFFFFFF))
    hb = pcg(np.uint32((seed + (knot + 1) * 2654435761) & 0xFFFFFFFF))
    a = normal(np.uint32((int(ha) + k * 7919) & 0xFFFFFFFF))
    b = normal(np.uint32((int(hb) + k * 7919) & 0xFFFFFFFF))
    return float(((1 - u) * a + u * b) / np.sqrt((1 - u) ** 2 + u ** 2))


def play_check(argv):
    """The engine's play (`tools/movement/solver -- play NET SEED SECONDS CLIP`) against the same
    rollout here: the WNET read and run by this script, the code drawn as the engine draws it.
    Prints the largest difference of the root and of a joint over the first second and over all."""
    net_name, seed, clip = argv[0], int(argv[1]), argv[2]
    body = motion.Body(f"{SOLVER}/{BODY}.body.json")
    net = read_wnet(f"{SOLVER}/{net_name}.wnet")
    rate, x = read_pose(f"{POSES}/{clip}.pose")
    _, eng = read_pose(f"{SOLVER}/play/{BODY}/{clip}.pose")
    ref = torch.tensor(np.array(x))
    eng = torch.tensor(np.array(eng))
    n = ref.shape[0]
    ahead = int(rate)
    poses = [ref[0], ref[1]]
    n_code = net[0].shape[0] - (motion.STATE_ROOT + 2 * body.freedom + motion.GOAL + 3)
    first, whole = 0.0, 0.0
    for t in range(1, eng.shape[0] - 1):
        now = poses[t]
        at = min(t + ahead, n - 1)
        secs = max((at - min(t, n - 1)) / rate, 1.0)
        g = body.goal(now[None], ref[at][None], torch.tensor([secs]))[0]
        s = body.state(poses[t - 1][None], now[None], rate)[0]
        z = torch.tensor([code(seed, t, k) for k in range(n_code)])
        inp = torch.cat([s, g, torch.ones(2), torch.zeros(1), z])
        step = apply_wnet(net, inp[None])
        nxt = body.after(now[None], step)[0]
        poses.append(nxt)
        d = (nxt - eng[t + 1]).abs().max().item()
        whole = max(whole, d)
        if t < rate:
            first = max(first, d)
    print(f"play-check {clip}: {len(poses)} frames; the largest difference from the engine's "
          f"poses over the first second {first:.2e}, over all {whole:.2e}")
    return 0 if first < 1e-3 else 1


def fk_check(clips):
    """`motion.Body.places` against the engine's (`Body::places`, which tools/movement/export
    writes in each clip's states, `.movement/sim/refs/CLIP.ref`): the largest distance of a
    segment's origin and of its turn."""
    body = motion.Body(f"{SOLVER}/{BODY}.body.json")
    worst_t, worst_r = 0.0, 0.0
    for c in clips:
        _, x = read_pose(f"{POSES}/{c}.pose")
        sim = ".movement/sim" if BODY == "creature" else f".movement/sim-{BODY}"
        b = open(os.path.join(ROOT, f"{sim}/refs/{c}.ref"), "rb").read()
        _, nq, segs, n = struct.unpack_from("<4I", b, 4)
        row = nq + 7 * segs
        ref = np.frombuffer(b, "<f4", n * row, 24).reshape(n, row)[:, nq:].reshape(n, segs, 7)
        t, q = body.places(torch.tensor(np.array(x)))
        rt = torch.tensor(np.array(ref[:, :, :3]))
        rq = torch.tensor(np.array(ref[:, :, [4, 5, 6, 3]]))
        worst_t = max(worst_t, (t - rt).norm(dim=-1).max().item())
        worst_r = max(worst_r, motion.vector_of_turn(motion.qmul(q, motion.qinv(rq))).norm(dim=-1).max().item())
    print(f"fk-check: the largest distance of a segment from the engine's {worst_t * 1000:.4f} mm, "
          f"of its turn {worst_r * 1000:.4f} mrad")
    return 0 if worst_t < 1e-4 and worst_r < 1e-3 else 1


def copies(argv):
    """Whether the solver gives back clips of its training set (phase 5 step 13): each second of
    its movements (`.movement/solver/goals/BODY/goal/*.pose`) after its first (the training frames
    it starts from), as the joints' numbers at 6 of its frames, against its nearest second of every
    training clip; and, for a sample of the training
    seconds, each one's nearest second in another training clip, which says how near two different
    clips come (standing is alike in every style). A second of the solver's nearer to a training
    second than any two different clips' seconds come (the sample's least) is a clip given back:
    nearer than the 1st percentile is only near, as standing is near every standing."""
    import glob
    train = [l.strip() for l in open(argv[0]) if l.strip()]
    def windows(x, rate, stride):
        q = x[:, 7:]
        n = q.shape[0]
        w = int(rate)
        idx = [i for i in range(0, n - w, stride)]
        return np.stack([q[i:i + w:5].reshape(-1) for i in idx]) if idx else np.zeros((0, 6 * q.shape[1]))
    tw, owner = [], []
    for k, c in enumerate(train):
        rate, x = read_pose(f"{POSES}/{c}.pose")
        w = windows(x, rate, 15)
        tw.append(w)
        owner.extend([k] * len(w))
    tw = torch.tensor(np.concatenate(tw).astype(np.float32))
    owner = torch.tensor(owner)
    tn = (tw * tw).sum(-1)
    def nearest(w, exclude=None):
        w = torch.tensor(w.astype(np.float32)) if not torch.is_tensor(w) else w
        d2 = (w * w).sum(-1)[:, None] + tn[None] - 2 * w @ tw.T
        if exclude is not None:
            d2 = torch.where(owner[None] == exclude[:, None], torch.full_like(d2, 1e9), d2)
        return torch.sqrt(d2.clamp(min=0).min(-1).values).numpy()
    sample = torch.arange(0, tw.shape[0], 20)
    between = np.concatenate([nearest(tw[sample[i:i + 512]], owner[sample[i:i + 512]])
                              for i in range(0, len(sample), 512)])
    floor = np.percentile(between, 1)
    print(f"copies: two different training clips' seconds: the least {between.min():.3f}, the 0.1st "
          f"percentile {np.percentile(between, 0.1):.3f}, the 1st {floor:.3f}")
    mine = []
    for f in sorted(glob.glob(os.path.join(SOLVER, "goals", BODY, "goal", "*.pose"))):
        rate, x = read_pose(f)
        # Its first second is the training frames it starts from: not what the solver made.
        d = nearest(windows(x[int(rate):], rate, 5))
        mine.append(d)
        print(f"copies: {os.path.basename(f)[:-5]}: its nearest training second {d.min():.3f} at the least, "
              f"{np.median(d):.3f} at the median")
    mine = np.concatenate(mine)
    print(f"copies: {tw.shape[0]} training seconds; two different training clips' seconds come "
          f"{floor:.3f} near at the 1st percentile ({np.median(between):.3f} at the median); the "
          f"solver's {mine.min():.3f} at the least ({np.median(mine):.3f} at the median)")
    return 0 if mine.min() >= between.min() else 1


def pack(argv):
    import json
    import zlib
    out = argv[argv.index("-o") + 1]
    sets = [a for a in argv[:argv.index("-o")]]
    names, which, starts, lengths, chunks = [], [], [], [], []
    rate, d, total = None, None, 0
    for given in sets:
        # A set's name (tools/movement/split), or the path of a list of clips named for its set.
        path = given if given.endswith(".txt") else os.path.join(ROOT, "tools/movement/split", f"{given}.txt")
        st = os.path.basename(path)[:-4]
        for line in open(path):
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            path = f"{POSES}/{name}.pose"
            if not os.path.exists(path):
                continue
            r, x = read_pose(path)
            assert rate in (None, r)
            rate, d = r, x.shape[1] - 7
            names.append(name)
            which.append(st)
            starts.append(total)
            lengths.append(x.shape[0])
            total += x.shape[0]
            chunks.append(np.ascontiguousarray(x))
    body = zlib.compress(np.concatenate(chunks).astype("<f4").tobytes(), 6)
    head = json.dumps({"names": names, "sets": which, "starts": starts, "lengths": lengths,
                       "rate": rate, "d": d}).encode()
    with open(out, "wb") as f:
        f.write(b"WPSE" + struct.pack("<I", len(head)) + head + body)
    print(f"pack: {len(names)} clips, {total} frames, {os.path.getsize(out) / 1e6:.1f} MB")
    return 0


def load_packed(path):
    import json
    import zlib
    b = open(path, "rb").read()
    assert b[:4] == b"WPSE"
    n = struct.unpack_from("<I", b, 4)[0]
    head = json.loads(b[8:8 + n])
    x = np.frombuffer(zlib.decompress(b[8 + n:]), dtype="<f4").reshape(-1, 7 + head["d"])
    return head, x


def main():
    poses_root = None
    if "--poses-root" in sys.argv[1:5] and sys.argv[1] != "--out":
        i = sys.argv.index("--poses-root")
        poses_root = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    if "--body" in sys.argv[1:5] and sys.argv[1] != "--out":
        i = sys.argv.index("--body")
        set_body(sys.argv[i + 1], poses_root)
        del sys.argv[i:i + 2]
    elif poses_root is not None:
        set_body(BODY, poses_root)
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(check(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "play-check":
        sys.exit(play_check(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "fk-check":
        sys.exit(fk_check(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "copies":
        sys.exit(copies(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "pack":
        sys.exit(pack(sys.argv[2:]))
    import solve_train
    solve_train.main(load_packed)


if __name__ == "__main__":
    main()
