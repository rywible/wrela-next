"""The prior of style (docs/plans/bodies-and-movement.md, phase 5 step 14): a discriminator that
says how much a short stretch of movement looks like the capture, and a net that moves the body in
the simulator under it with no clip to follow. Run on Colab by the run tool:

    tools/movement/colab-run start NAME --gpu L4 --hours H --script tools/movement/amp.py \
        --file tools/movement/track.py --file .movement/sim-human/human.xml \
        --file .movement/sim-human/refs.bin -- --model human.xml [--steps S] ...

or here, for a smoke test (MuJoCo Warp runs on the CPU too, slowly):

    .movement/venv/bin/python tools/movement/amp.py --out /tmp/x --seed 1 --worlds 4 --steps 2 \
        --model .movement/sim-human/human.xml --refs .movement/sim-human/refs_smoke.bin

and with `--offline`, the discriminator alone against damaged capture (no simulation: the check
of its features), on the CPU here.

The method is published work's (docs/citations.md): X. B. Peng et al., "AMP: Adversarial Motion
Priors for Stylized Physics-Based Character Control" (2021): a least-squares discriminator of
pairs of frames with a penalty on its gradient at the capture, its score as a reward beside the
task's; PPO (J. Schulman et al., 2017). The net is rewarded for its task (`--task`, tools/movement/tasks.py:
the solver's wish of a speed and a direction, a cast, a parry or a dodge) and for the
discriminator's score; each world starts at a random frame of a random training clip. At the end a
goal test plays the task with the net's mean action and writes its first worlds' movement as a
filtered capture (`played.bin`) for the judge.

The discriminator sees one frame's state and its rates to the next, all from the qpos and the
simulator's kinematics in the same way for the capture, its damaged copies and the net's
movement: the root's height and tilt, the joints' numbers, each segment's place from the root,
all in the root's heading frame, their rates, and the root's speed and turn. Its negatives are the
net's movement and damaged copies of training clips (the judge's damages of phase 3: a faster
time, a slide of the root, noise on the joints, a root too high, a body that does not move).

The check (the plan's): each test clip's mean score above each of its damaged copies' (the
judge's 15 damages) in not less than 95% of pairs. It ends with one line: `RESULT {...}`.
"""

import argparse
import io
import json
import math
import os
import struct
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlib  # noqa: E402
import tasks  # noqa: E402
import track  # noqa: E402

torch = mujoco = wp = mjw = None

# The judge's damages (tools/movement/judge, KINDS and AMOUNTS): times as fast, m/s of slide,
# rad of noise, m higher, and the body that does not move.
DAMAGES = [("faster", 1.25), ("faster", 1.5), ("faster", 2.0),
           ("slid", 0.2), ("slid", 0.5), ("slid", 1.0),
           ("noisy", 0.02), ("noisy", 0.05), ("noisy", 0.1),
           ("raised", 0.02), ("raised", 0.05), ("raised", 0.1),
           ("still", 1.0)]


class Kin:
    """The simulator's kinematics on a data of its own: each active segment's place for any qpos."""

    def __init__(self, mjm, batch, active=None):
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        self.m = mjw.put_model(mjm)
        self.d = mjw.put_data(mjm, mjd, nworld=batch, nconmax=64, njmax=256, njmax_nnz=256 * mjm.nv)
        self.n = batch
        self.active = active
        self.qpos = wp.to_torch(self.d.qpos)
        self.xpos = wp.to_torch(self.d.xpos)

    def places(self, q):
        out = []
        for i in range(0, q.shape[0], self.n):
            k = min(self.n, q.shape[0] - i)
            self.qpos[:k] = q[i:i + k]
            mjw.kinematics(self.m, self.d)
            if self.active is not None:
                out.append(self.xpos[:k, self.active].clone())
            else:
                out.append(self.xpos[:k, 1:].clone())
        return torch.cat(out)


def quat_nlerp(a, b, u):
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    q = a * (1 - u) + b * u
    return q / q.norm(dim=-1, keepdim=True)


def rotvec(q):
    """The rotation vector of turn `q` (w, x, y, z), the shorter way round."""
    q = torch.where(q[..., :1] < 0, -q, q)
    s = q[..., 1:].norm(dim=-1, keepdim=True)
    a = 2.0 * torch.atan2(s, q[..., :1])
    return q[..., 1:] * torch.where(s > 1e-8, a / s.clamp(min=1e-8), torch.full_like(s, 2.0))


# The stretch the discriminator reads: this many frames (a third of a second at 30 a second), so
# that it sees the sway of a body that stands, which one pair of frames does not show.
WINDOW = 10
# The fastest time of a damaged training window (times as fast).
FASTEST = 2.5


def features(qs, xs, rate):
    """The discriminator's input for a stretch of frames `qs` (B, K, qpos) and `xs` (B, K, each
    active segment's place, root at index 0): the first frame's state in its root's heading frame,
    then for each frame after it the rates of that state from the frame before, and the root's
    speed and turn, all in the first frame's heading frame."""
    T = track
    B, K = qs.shape[0], qs.shape[1]
    hq = T.quat_conj(T.heading(qs[:, 0, 3:7]))
    hk = hq[:, None].expand(-1, K, -1)
    r = qs[:, :, 3:7]
    local = T.quat_mul(hk, r)
    ex = torch.tensor([1.0, 0.0, 0.0], device=qs.device).expand(B, K, 3)
    ey = torch.tensor([0.0, 1.0, 0.0], device=qs.device).expand(B, K, 3)
    rel = xs[:, :, 1:] - xs[:, :, :1]
    rel = T.quat_rotate(hk[:, :, None].expand(-1, -1, rel.shape[2], -1), rel).reshape(B, K, -1)
    st = torch.cat([qs[:, :, 1:2], T.quat_rotate(local, ex), T.quat_rotate(local, ey), qs[:, :, 7:], rel], -1)
    hk1 = hk[:, 1:]
    lin = T.quat_rotate(hk1, (qs[:, 1:, :3] - qs[:, :-1, :3]) * rate)
    ang = T.quat_rotate(hk1, rotvec(T.quat_mul(r[:, 1:], T.quat_conj(r[:, :-1]))) * rate)
    rates = torch.cat([(st[:, 1:] - st[:, :-1]) * rate * 0.1, lin, ang], -1)
    return torch.cat([st[:, 0], rates.reshape(B, -1)], -1)


def between(q, g, u):
    """The qpos a share `u` of the way from frame g to g + 1 (u in [0, 1))."""
    a, b = q[g], q[g + 1]
    out = a + (b - a) * u[..., None]
    out[..., 3:7] = quat_nlerp(a[..., 3:7], b[..., 3:7], u[..., None])
    return out


class Clips:
    """The clips' frames and the stretches of frames the discriminator reads, as they are and
    damaged."""

    def __init__(self, head, refs, kin, device, gen, q=None, x=None, active=None):
        self.q = torch.tensor(refs, device=device) if q is None else q
        self.starts = torch.tensor(head["starts"], device=device, dtype=torch.long)
        self.lengths = torch.tensor(head["lengths"], device=device, dtype=torch.long)
        self.rate = float(head["rate"])
        self.kin = kin
        self.dev = device
        self.gen = gen
        self.active = active
        self.x = kin.places(self.q) if x is None else x
        sets = head.get("sets") or ["training"] * len(head["names"])
        self.train = [i for i, s in enumerate(sets) if s == "training"] or list(range(len(sets)))
        self.test = [i for i, s in enumerate(sets) if s == "test"]
        self.names = head["names"]
        self.k = torch.arange(WINDOW, device=device)
        # Each training frame that starts a stretch, at the fastest damaged time too.
        need = math.ceil(FASTEST * (WINDOW - 1)) + 2
        frames = [torch.arange(int(self.starts[c]), int(self.starts[c] + self.lengths[c]) - need, device=device)
                  for c in self.train if int(self.lengths[c]) > need]
        self.train_frames = torch.cat(frames)
        # The root's usual height: what a fall is measured against.
        self.root_height = float(self.q[self.train_frames, 1].median())

    def windows(self, g):
        i = g[:, None] + self.k
        return features(self.q[i], self.x[i], self.rate)

    def sample(self, n):
        i = torch.randint(0, self.train_frames.shape[0], (n,), device=self.dev, generator=self.gen)
        return self.train_frames[i]

    def places(self, qs):
        B, K = qs.shape[0], qs.shape[1]
        return self.kin.places(qs.reshape(B * K, -1)).reshape(B, K, -1, 3)

    def damaged(self, g, kind, amount, seed=None):
        """Stretches from frames `g` damaged as the judge damages a clip."""
        i = g[:, None] + self.k
        qs, xs = self.q[i].clone(), self.x[i].clone()
        t = self.k.float() / self.rate
        if kind == "faster":
            # Frame k of the faster clip is frame `amount` k of the clip.
            at = amount * self.k.float()
            j = g[:, None] + at.floor().long()
            qs = between(self.q, j, (at - at.floor()).expand_as(j))
            xs = self.places(qs)
        elif kind == "slid":
            qs[:, :, 0] += amount * t
            xs[:, :, :, 0] += amount * t[None, :, None]
        elif kind == "noisy":
            gen = torch.Generator(device=self.dev).manual_seed(seed or 0)
            qs[:, :, 7:] += amount * torch.randn(qs[:, :, 7:].shape, device=self.dev, generator=gen)
            xs = self.places(qs)
        elif kind == "raised":
            qs[:, :, 1] += amount
            xs[:, :, :, 1] += amount
        elif kind == "still":
            qs = qs[:, :1].expand_as(qs).clone()
            xs = xs[:, :1].expand_as(xs).clone()
        return features(qs, xs, self.rate)

    def damaged_sample(self, n):
        """`n` damaged stretches of random training frames, of a damage at random and an amount at
        random between none and twice the judge's largest."""
        g = self.sample(n)
        kinds = ["faster", "slid", "noisy", "raised", "still"]
        k = int(torch.randint(0, 5, (1,), generator=self.gen, device=self.dev))
        top = {"faster": FASTEST, "slid": 2.0, "noisy": 0.2, "raised": 0.2, "still": 1.0}[kinds[k]]
        u = float(torch.rand(1, generator=self.gen, device=self.dev))
        amount = 1.0 + u * (top - 1.0) if kinds[k] == "faster" else u * top
        seed = int(torch.randint(0, 1 << 30, (1,), generator=self.gen, device=self.dev))
        return self.damaged(g, kinds[k], amount, seed=seed)


class Disc(object):
    """The discriminator: least squares (capture 1, the rest -1), its gradient at the capture held
    small (Peng et al. 2021, weight 5), on normalised inputs."""

    def __init__(self, n_in, hidden, device, lr):
        self.net = track.mlp(n_in, 1, hidden).to(device)
        self.norm = track.Normal(n_in, device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def score(self, f):
        with torch.no_grad():
            return self.net(self.norm(f)).squeeze(-1)

    def reward(self, f):
        d = self.score(f)
        return (1.0 - 0.25 * (d - 1.0) ** 2).clamp(min=0.0)

    def train(self, pos, neg, gp=5.0):
        self.norm.update(pos)
        p = self.norm(pos).requires_grad_(True)
        n = self.norm(neg)
        dp, dn = self.net(p).squeeze(-1), self.net(n).squeeze(-1)
        loss = ((dp - 1.0) ** 2).mean() + ((dn + 1.0) ** 2).mean()
        g = torch.autograd.grad(dp.sum(), p, create_graph=True)[0]
        loss = loss + gp * (g * g).sum(-1).mean()
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        return float(loss.detach()), float(dp.mean().detach()), float(dn.mean().detach())

    def state(self):
        return {"net": {k: v.cpu() for k, v in self.net.state_dict().items()}, "norm": self.norm.state(),
                "opt": self.opt.state_dict()}

    def load(self, s, device):
        self.net.load_state_dict(s["net"])
        self.norm.load(s["norm"], device)
        if "opt" in s:
            self.opt.load_state_dict(s["opt"])


def check(disc, clips):
    """The plan's check: each test clip's mean score against each of its 13 damaged copies (the
    judge's damages; its still body at one amount): the share of pairs where the clip scores
    higher, and per damage."""
    wins, total, per = 0, 0, {}
    for c in clips.test:
        s, n = int(clips.starts[c]), int(clips.lengths[c])
        need = 2 * (WINDOW - 1) + 2
        if n <= need:
            continue
        g = torch.arange(s, s + n - need, device=clips.dev)
        mine = float(disc.score(clips.windows(g)).mean())
        for i, (kind, amount) in enumerate(DAMAGES):
            other = float(disc.score(clips.damaged(g, kind, amount, seed=1000 * c + i)).mean())
            key = f"{kind} {amount}"
            w, t = per.get(key, (0, 0))
            per[key] = (w + int(mine > other), t + 1)
            wins += int(mine > other)
            total += 1
    return {"share": wins / max(total, 1), "pairs": total,
            "per": {k: w / t for k, (w, t) in per.items()}}


# ---- the net that moves under the prior ---------------------------------------------------------


def observe(sim, task):
    """The body's state in its heading's frame (active bodies only, continuous 6D root orientation)
    and what the task gives of its goal (`tasks`), in the same frame."""
    T = track
    root_q = sim.xquat[:, 1]
    root_x = sim.xpos[:, 1]
    hq = T.quat_conj(T.heading(root_q))
    local = T.quat_mul(hq, root_q)
    n = sim.n
    act_x = sim.xpos[:, sim.active]
    rel = act_x[:, 1:] - root_x[:, None]
    return torch.cat([
        root_x[:, 1:2], T.rot6d(local), T.quat_rotate(hq, sim.qvel[:, :3]), T.quat_rotate(local, sim.qvel[:, 3:6]),
        sim.qpos[:, 7:], 0.1 * sim.qvel[:, 6:],
        T.quat_rotate(hq[:, None].expand(-1, rel.shape[1], -1), rel).reshape(n, -1),
        task.observe(),
    ], -1)


def remember(sim, clips, hist_q, hist_x, which):
    """The last frames of the worlds `which`, just started at a frame of a clip: that clip's frames
    before it (its first, before its start), so the discriminator's stretch is whole at once."""
    idx = which.nonzero().flatten()
    if idx.numel() == 0:
        return
    start = sim.starts[sim.clip[idx]]
    g = start + sim.frame[idx]
    back = torch.arange(WINDOW - 1, -1, -1, device=sim.dev)
    i = torch.maximum(g[:, None] - back, start[:, None])
    hist_q[idx] = clips.q[i]
    hist_x[idx] = clips.x[i]


def offline(a, run, dev, gen, head, refs, mjm):
    """The discriminator alone against damaged training clips: the check of its input."""
    active = torch.tensor([b for b in range(1, mjm.nbody) if mjm.body_jntnum[b] > 0], device=dev, dtype=torch.long)
    kin = Kin(mjm, a.batch, active=active)
    clips = Clips(head, refs, kin, dev, gen, active=active)
    n_in = clips.windows(clips.sample(2)).shape[1]
    disc = Disc(n_in, [int(x) for x in a.disc_hidden.split(",")], dev, a.disc_lr)
    print(f"offline: {len(clips.train)} training clips, {len(clips.test)} test, input {n_in}", flush=True)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        loss, dp, dn = disc.train(clips.windows(clips.sample(a.batch)), clips.damaged_sample(a.batch))
        if step % 50 == 0:
            print(f"update {step}: loss {loss:.4f} capture {dp:.3f} damaged {dn:.3f} ({time.time() - t0:.0f} s)",
                  flush=True)
    result = check(disc, clips)
    print(f"check: {json.dumps(result)}", flush=True)
    buf = io.BytesIO()
    torch.save(disc.state(), buf)
    run.output("style.pt", buf.getvalue())
    run.finish({"check": result, "seconds": time.time() - t0})


def main():
    global torch, mujoco, wp, mjw
    run = runlib.Run()
    track.load_packages()
    torch, mujoco, wp, mjw = track.torch, track.mujoco, track.wp, track.mjw
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="human.xml")
    p.add_argument("--refs", default="refs.bin")
    p.add_argument("--worlds", type=int, default=4096)
    p.add_argument("--steps", type=int, default=1000, help="updates of the net")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--hidden", default="1024,512")
    p.add_argument("--disc-hidden", default="1024,512")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--disc-lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=4096, help="pairs of each kind in a step of the discriminator")
    p.add_argument("--damaged", type=float, default=0.25, help="the share of the negatives from damaged capture")
    p.add_argument("--style", type=float, default=0.5, help="the style's share of the reward")
    p.add_argument("--fastest", type=float, default=2.0, help="the fastest wish (m/s)")
    p.add_argument("--episode", type=int, default=300, help="the most frames of a world before it starts again")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--target-kl", type=float, default=0.002)
    p.add_argument("--task", default="walk", choices=["walk", "cast", "parry", "dodge"],
                   help="what the net is rewarded for beside the style (tools/movement/tasks.py)")
    p.add_argument("--test-frames", type=int, default=600, help="the goal test's frames (20 s)")
    p.add_argument("--fall", type=float, default=0.6, help="a fall: the root under this share of its usual height")
    p.add_argument("--offline", action="store_true", help="the discriminator alone against damaged capture")
    a = p.parse_args(run.args)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wp.init()
    torch.manual_seed(run.seed)
    gen = torch.Generator(device=dev).manual_seed(run.seed)
    head, refs = track.load_refs(a.refs)
    with wp.ScopedDevice("cuda:0" if dev == "cuda" else "cpu"):
        mjm = mujoco.MjModel.from_xml_path(a.model)
        if a.offline:
            return offline(a, run, dev, gen, head, refs, mjm)
        train(a, run, dev, gen, head, refs, mjm)


def goal_test(a, sim, agent, norm, clips, head, fall_height, frames=600, kept=16):
    """The task's goal test: every world from a training clip's frame, the net's mean action, for
    `frames` frames, a world that falls not started again; the task's measures over all its
    goals, the share of worlds that fell, and the first `kept` worlds' movement as a filtered
    capture (`WFLT`, as track.py writes it: `tools/movement/export -- human unfilter` makes poses
    of it for the judge and the laws of movement)."""
    gen = torch.Generator(device=sim.dev).manual_seed(12345)
    sim.gen = gen
    kept = min(kept, sim.n)
    sim.start_share = 0.0
    sim.reset(torch.ones(sim.n, dtype=torch.bool, device=sim.dev))
    mjw.kinematics(sim.m, sim.d)
    task = tasks.make(a.task, sim, gen, a.fastest)
    task.measures()
    alive = torch.ones(sim.n, dtype=torch.bool, device=sim.dev)
    fell_at = torch.full((sim.n,), frames, dtype=torch.long, device=sim.dev)
    hist_q = sim.qpos[:, None].expand(-1, WINDOW, -1).clone()
    mid = 0.5 * (sim.jnt_lo + sim.jnt_hi)
    half = 0.5 * (sim.jnt_hi - sim.jnt_lo)
    record = [sim.qpos[:kept].float().cpu().clone()]
    never = torch.zeros(sim.n, dtype=torch.bool, device=sim.dev)
    for t in range(frames):
        with torch.no_grad():
            act = agent.pi(norm(observe(sim, task)))
        # Map [-1, 1] across each joint's actual range (avoids 50% dead zone on asymmetric joints).
        sim.step(mid + act.clamp(-1, 1) * half)
        hist_q = torch.cat([hist_q[:, 1:], sim.qpos[:, None]], 1)
        task.reward(hist_q, clips.rate)
        fell = alive & (track.exploded(sim) | (sim.qpos[:, 1] < fall_height))
        fell_at = torch.where(fell, torch.full_like(fell_at, t), fell_at)
        alive &= ~fell
        task.counted = alive.clone()
        task.after(never)
        record.append(sim.qpos[:kept].float().cpu().clone())
    measures = task.measures()
    measures["fell"] = float((~alive).float().mean())
    chunks, listed = [], []
    for j in range(kept):
        n = int(fell_at[j]) + 1
        chunks.append(torch.stack([f[j] for f in record[:n]]).numpy().astype("<f4"))
        listed.append({"name": f"skill/{a.task}_{j}", "set": "test", "followed": bool(alive[j]),
                       "frames": n, "length": frames + 1, "mean_segment_mm": 0.0})
    body = zlib.compress(np.concatenate(chunks).tobytes(), 6)
    h = json.dumps({"clips": listed, "nq": head["nq"], "rate": head["rate"]}).encode()
    return measures, b"WFLT" + struct.pack("<I", len(h)) + h + body


def train(a, run, dev, gen, head, refs, mjm):
    T = track
    t0 = time.time()
    sim = T.Sim(a.model, head, refs, a.worlds, dev, a.substeps, gen)
    kin = Kin(mjm, a.batch, active=sim.active)
    clips = Clips(head, refs, kin, dev, gen, q=sim.ref_q, x=sim.ref_x, active=sim.active)
    fall_height = a.fall * clips.root_height
    print(f"sim: {a.worlds} worlds, {len(head['names'])} clips, root height {clips.root_height:.3f} m, "
          f"set up in {time.time() - t0:.0f} s", flush=True)
    sim.reset(torch.ones(a.worlds, dtype=torch.bool, device=dev))
    N, H = a.worlds, a.horizon
    hist_q = torch.zeros(N, WINDOW, sim.nq, device=dev)
    hist_x = torch.zeros(N, WINDOW, sim.active.numel(), 3, device=dev)
    remember(sim, clips, hist_q, hist_x, torch.ones(N, dtype=torch.bool, device=dev))
    task = tasks.make(a.task, sim, gen, a.fastest)
    age = torch.zeros(N, dtype=torch.long, device=dev)
    n_obs = observe(sim, task).shape[1]
    n_act = sim.mjm.nu
    mid = 0.5 * (sim.jnt_lo + sim.jnt_hi)
    half = 0.5 * (sim.jnt_hi - sim.jnt_lo)
    hidden = [int(x) for x in a.hidden.split(",")]
    agent = T.make_agent(n_obs, n_act, hidden).to(dev)
    with torch.no_grad():
        agent.log_std.fill_(-1.8)
    opt = torch.optim.Adam(agent.parameters(), lr=a.lr)
    norm = T.Normal(n_obs, dev)
    vnorm = T.ValNorm(dev)
    n_in = clips.windows(clips.sample(2)).shape[1]
    disc = Disc(n_in, [int(x) for x in a.disc_hidden.split(",")], dev, a.disc_lr)
    step, samples = 0, 0
    state = run.resume()
    if state is not None:
        agent.load_state_dict(state["agent"])
        opt.load_state_dict(state["opt"])
        norm.load(state["norm"], dev)
        if "vnorm" in state:
            vnorm.load(state["vnorm"], dev)
        disc.load(state["disc"], dev)
        step, samples = state["step"], state["samples"]
        gen.set_state(state["gen"].cpu())
        torch.set_rng_state(state["torch_rng"])
    obs_buf = torch.zeros(H, N, n_obs, device=dev)
    act_buf = torch.zeros(H, N, n_act, device=dev)
    logp_buf = torch.zeros(H, N, device=dev)
    task_buf = torch.zeros(H, N, device=dev)
    feat_buf = torch.zeros(H, N, n_in, device=dev)
    done_buf = torch.zeros(H, N, device=dev)
    fell_buf = torch.zeros(H, N, device=dev)
    val_buf = torch.zeros(H + 1, N, device=dev)
    trunc_val = torch.zeros(H, N, device=dev)
    last_check = None
    rate_t, rate_s = time.time(), samples
    while step < a.steps:
        ep_task, ep_fall, blown = 0.0, 0.0, 0
        with torch.no_grad():
            for h in range(H):
                o_raw = observe(sim, task)
                norm.update(o_raw)
                o = norm(o_raw)
                dist = agent.dist(o)
                act = dist.sample()
                obs_buf[h], act_buf[h] = o, act
                logp_buf[h] = dist.log_prob(act).sum(-1)
                val_buf[h] = vnorm.denorm(agent.v(o).squeeze(-1))
                # Map [-1, 1] symmetrically across each joint's actual range.
                sim.step(mid + act.clamp(-1, 1) * half)
                bad = T.exploded(sim)
                blown += int(bad.sum())
                hist_q = torch.cat([hist_q[:, 1:], sim.qpos[:, None]], 1)
                hist_x = torch.cat([hist_x[:, 1:], sim.xpos[:, sim.active][:, None]], 1)
                feat_buf[h] = torch.nan_to_num(features(hist_q, hist_x, clips.rate), 0.0, 0.0, 0.0)
                r_task = task.reward(hist_q, clips.rate)
                fell = bad | (sim.qpos[:, 1] < fall_height) | ~torch.isfinite(r_task)
                r_task = torch.where(fell, torch.zeros_like(r_task), r_task)
                age += 1
                ended = age >= a.episode
                trunc_val[h] = 0.0
                if ended.any():
                    trunc_val[h] = torch.where(ended & ~fell, vnorm.denorm(agent.v(norm(observe(sim, task))).squeeze(-1)), 0.0)
                task_buf[h] = r_task
                done_buf[h] = (fell | ended).float()
                fell_buf[h] = fell.float()
                ep_task += float(r_task.mean())
                ep_fall += float(fell.float().mean())
                again = fell | ended
                sim.reset(again)
                remember(sim, clips, hist_q, hist_x, again)
                age[again] = 0
                task.after(again)
            val_buf[H] = vnorm.denorm(agent.v(norm(observe(sim, task))).squeeze(-1))
            style = disc.reward(feat_buf.reshape(H * N, -1)).reshape(H, N)
            # A fall earns nothing.
            style = style * (1.0 - fell_buf)
            rew = (1.0 - a.style) * task_buf + a.style * style
            adv = torch.zeros(H, N, device=dev)
            last = torch.zeros(N, device=dev)
            for h in reversed(range(H)):
                nonterm = 1.0 - done_buf[h]
                nxt = val_buf[h + 1] * nonterm + trunc_val[h]
                delta = rew[h] + 0.99 * nxt - val_buf[h]
                last = delta + 0.99 * 0.95 * nonterm * last
                adv[h] = last
            ret = adv + val_buf[:H]
        # The discriminator: capture against the net's pairs and damaged capture.
        flat = feat_buf.reshape(H * N, -1)
        n_damaged = int(a.batch * a.damaged)
        d_loss = d_pos = d_neg = 0.0
        steps_d = min(max(1, (H * N) // a.batch), 4)
        for _ in range(steps_d):
            j = torch.randint(0, H * N, (a.batch - n_damaged,), device=dev, generator=gen)
            neg = torch.cat([flat[j], clips.damaged_sample(n_damaged)]) if n_damaged else flat[j]
            d_loss, d_pos, d_neg = disc.train(clips.windows(clips.sample(a.batch)), neg)
        b_obs, b_act = obs_buf.reshape(H * N, -1), act_buf.reshape(H * N, -1)
        b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        kl, clipped, vloss, epochs = T.ppo(agent, opt, b_obs, b_act, b_logp, b_adv, b_ret, gen, a.target_kl, vnorm=vnorm)
        skipped = kl[1]
        step += 1
        samples += H * N
        now = time.time()
        sps = (samples - rate_s) / max(now - rate_t, 1e-6)
        run.progress(step, a.steps, f"task {ep_task / H:.3f} style {float(style.mean()):.3f} "
                                    f"falls {ep_fall / H:.4f} at {sps:.0f}/s")
        if step % 10 == 0:
            print(f"update {step}: task {ep_task / H:.3f} style {float(style.mean()):.3f} "
                  f"falls/step {ep_fall / H:.4f}; discriminator loss {d_loss:.3f} capture {d_pos:.3f} "
                  f"other {d_neg:.3f}; samples {samples} ({sps:.0f} a second); blown up {blown}, "
                  f"skipped {skipped}; kl {kl[0]:.4f} clipped {clipped:.3f} value loss {vloss:.3f} "
                  f"epochs {epochs} log std {float(agent.log_std.mean()):.3f} "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        if a.eval_every and step % a.eval_every == 0:
            last_check = check(disc, clips)
            print(f"check at {step}: {json.dumps(last_check)}", flush=True)
            print(f"the task at {step} ({a.task}, sampled actions): {json.dumps(task.measures())}", flush=True)
        if run.due() or step == a.steps:
            run.checkpoint(step, {
                "agent": {k: v.cpu() for k, v in agent.state_dict().items()},
                "opt": opt.state_dict(), "norm": norm.state(), "vnorm": vnorm.state(), "disc": disc.state(), "step": step,
                "samples": samples, "gen": gen.get_state().cpu(), "torch_rng": torch.get_rng_state(),
                "n_obs": n_obs, "n_act": n_act, "hidden": hidden, "args": vars(a),
            })
        if run.should_stop():
            run.abandon()
            return
    final = check(disc, clips)
    print(f"check at the end: {json.dumps(final)}", flush=True)
    test, data = goal_test(a, sim, agent, norm, clips, head, fall_height, frames=a.test_frames)
    print(f"the goal test ({a.task}, the mean action): {json.dumps(test)}", flush=True)
    run.output("played.bin", data)
    buf = io.BytesIO()
    torch.save({"agent": {k: v.cpu() for k, v in agent.state_dict().items()}, "norm": norm.state(),
                "disc": disc.state(), "n_obs": n_obs, "n_act": n_act, "n_in": n_in, "hidden": hidden,
                "seed": run.seed, "args": vars(a), "samples": samples}, buf)
    run.output("style.pt", buf.getvalue())
    run.finish({"step": step, "samples": samples, "check": final, "goal_test": test,
                "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
