"""The tracker (docs/plans/bodies-and-movement.md, phase 4): a net that moves our body in the
simulator so that it follows each clip of capture. Run on Colab by the run tool:

    tools/movement/colab-run start NAME --gpu L4 --hours H --script tools/movement/track.py \
        --file .movement/sim/creature.xml --file .movement/sim/refs.bin -- [--worlds N] [--steps S] ...

or here, for a smoke test (MuJoCo Warp runs on the CPU too, slowly):

    .movement/venv/bin/python tools/movement/track.py --out /tmp/x --seed 1 --worlds 4 --steps 2 \
        --model .movement/sim/creature.xml --refs .movement/sim/refs.bin

The model is the export of the body (tools/movement/export); the clips are its states
(`refs.bin`: tools/movement/pack_refs.py packs the export's `refs/*.ref` into one file).

The method is published work's (docs/citations.md): X. B. Peng et al., "DeepMimic" (2018), with
the reward on the places of all segments of Z. Luo et al., "Perpetual Humanoid Control" (2023);
PPO (J. Schulman et al., 2017). Each world starts at a random frame of a random clip, its state the
clip's there (reference state initialisation) and ends when the body falls away from the clip
(early termination) or the clip ends. The net reads the body's state in its heading's frame and
the clip's next frames, and gives a change of the clip's next joint angles: the targets of the
joints' servos (position servos whose force is held to each joint's strength, in the model).

It writes a checkpoint each `--every` s (the run tool brings it home) and ends with one line:
`RESULT {...}`, the share of clips followed to their end with no fall and the mean distance of a
segment from its place in the clip, by `evaluate`.
"""

import argparse
import json
import math
import os
import struct
import time
import zlib

import subprocess
import sys

import numpy as np

import runlib

mujoco = torch = wp = mjw = None


def load_packages():
    """Imports the simulator and the net's library, installing MuJoCo Warp first on a fresh VM
    (the heartbeat of `runlib` speaks while it installs)."""
    global mujoco, torch, wp, mjw
    try:
        import mujoco_warp  # noqa: F401
    except ImportError:
        t = time.time()
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mujoco-warp"], check=True)
        print(f"installed mujoco-warp in {time.time() - t:.0f} s", flush=True)
    import mujoco as _mujoco
    import torch as _torch
    import warp as _wp
    import mujoco_warp as _mjw
    mujoco, torch, wp, mjw = _mujoco, _torch, _wp, _mjw

# ---- the clips --------------------------------------------------------------------------------


def load_refs(path):
    """The packed clips: names, each clip's first frame and length, its rate, and every frame's
    qpos (float32)."""
    raw = open(path, "rb").read()
    assert raw[:4] == b"WRFS", "not a refs.bin"
    head_len = struct.unpack_from("<I", raw, 4)[0]
    head = json.loads(raw[8:8 + head_len])
    body = zlib.decompress(raw[8 + head_len:])
    q = np.frombuffer(body, dtype="<f4").reshape(-1, head["nq"])
    return head, q


# ---- the body in the simulator ------------------------------------------------------------------


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], -1)


def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


def quat_rotate(q, v):
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def heading(q):
    """The turn about the world's up (y) that the root's turn `q` makes: its left (+x) projected
    on the ground."""
    x = quat_rotate(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand_as(q[..., 1:]))
    yaw = torch.atan2(-x[..., 2], x[..., 0])
    h = torch.zeros_like(q)
    h[..., 0] = torch.cos(0.5 * yaw)
    h[..., 2] = torch.sin(0.5 * yaw)
    return h


def angle_of(q):
    w = q[..., 0].abs().clamp(max=1.0)
    return 2.0 * torch.acos(w)


class Sim:
    """`worlds` copies of the body, stepped together, and the clips they follow."""

    def __init__(self, model_path, head, refs, worlds, device, substeps, gen):
        self.mjm = mujoco.MjModel.from_xml_path(model_path)
        mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, mjd)
        self.m = mjw.put_model(self.mjm)
        # Per world, from the model: contacts (each colliding capsule on the floor, twice over) and
        # constraint rows (four a contact, and each joint at its limit), and the rows' non-zeros
        # (each row a dof's worth). Fixed at 32 and 160 for the creature's 18 parts, they
        # overflowed with the human's (2026-11-26: the simulator dropped contacts and the training
        # fell apart at update 990).
        colliding = int(sum(1 for g in range(self.mjm.ngeom) if self.mjm.geom_contype[g] != 0))
        nconmax = 2 * colliding
        njmax = 4 * nconmax + self.mjm.njnt
        self.d = mjw.put_data(self.mjm, mjd, nworld=worlds, nconmax=nconmax, njmax=njmax,
                              njmax_nnz=njmax * self.mjm.nv)
        self.n = worlds
        self.dev = device
        self.substeps = substeps
        self.gen = gen
        self.qpos = wp.to_torch(self.d.qpos)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.xpos = wp.to_torch(self.d.xpos)
        self.xquat = wp.to_torch(self.d.xquat)
        self.warm = wp.to_torch(self.d.qacc_warmstart)
        self.nq, self.nv, self.nb = self.mjm.nq, self.mjm.nv, self.mjm.nbody
        self.lo = torch.tensor(self.mjm.actuator_ctrlrange[:, 0], device=device)
        self.hi = torch.tensor(self.mjm.actuator_ctrlrange[:, 1], device=device)
        self.rate = head["rate"]
        self.dt = 1.0 / self.rate
        # The clips: their frames, where each starts and how long it is.
        self.ref_q = torch.tensor(refs, device=device)
        self.starts = torch.tensor(head["starts"], device=device, dtype=torch.long)
        self.lengths = torch.tensor(head["lengths"], device=device, dtype=torch.long)
        self.ref_x, self.ref_r = self.frames_of(self.ref_q)
        sets = head.get("sets") or ["training"] * len(head["names"])
        self.train = torch.tensor([i for i, s in enumerate(sets) if s == "training"] or
                                  list(range(len(sets))), device=device, dtype=torch.long)
        self.test = torch.tensor([i for i, s in enumerate(sets) if s == "test"], device=device,
                                 dtype=torch.long)
        # How often each training clip starts a world: by its length (each frame alike), and
        # more for a clip the last evaluation saw the net fail (`weigh`).
        self.weights = self.lengths[self.train].float()
        # Each world's clip and frame.
        self.clip = torch.zeros(worlds, dtype=torch.long, device=device)
        self.frame = torch.zeros(worlds, dtype=torch.long, device=device)
        self.graph = None

    def frames_of(self, q):
        """Each body's place and turn for each of the qpos `q` (by the simulator's kinematics, a
        batch of worlds at a time)."""
        out_x, out_r = [], []
        n = q.shape[0]
        b = self.n
        for i in range(0, n, b):
            k = min(b, n - i)
            self.qpos[:k] = q[i:i + k]
            mjw.kinematics(self.m, self.d)
            out_x.append(self.xpos[:k].clone())
            out_r.append(self.xquat[:k].clone())
        return torch.cat(out_x), torch.cat(out_r)

    def ref_vel(self, g):
        """The qvel of frame `g` of the clips (by differences): the root's in the world, the
        hinges' rates."""
        q0 = self.ref_q[g]
        q1 = self.ref_q[(g + 1).clamp(max=self.ref_q.shape[0] - 1)]
        v = torch.zeros(g.shape[0], self.nv, device=self.dev)
        v[:, :3] = (q1[:, :3] - q0[:, :3]) * self.rate
        dq = quat_mul(q1[:, 3:7], quat_conj(q0[:, 3:7]))
        dq = dq * torch.sign(dq[:, :1] + 1e-9)
        v[:, 3:6] = 2.0 * dq[:, 1:] * self.rate
        # MuJoCo's free joint's angular velocity is in the body's frame.
        v[:, 3:6] = quat_rotate(quat_conj(q0[:, 3:7]), v[:, 3:6])
        v[:, 6:] = (q1[:, 7:] - q0[:, 7:]) * self.rate
        return v

    def reset(self, which):
        """Starts the worlds `which` (a mask) at a random frame of a random clip."""
        idx = which.nonzero().flatten()
        if idx.numel() == 0:
            return
        c = self.train[torch.multinomial(self.weights, idx.numel(), replacement=True,
                                         generator=self.gen)]
        span = (self.lengths[c] - 2).clamp(min=1)
        f = (torch.rand(idx.numel(), device=self.dev, generator=self.gen) * span).long()
        self.clip[idx] = c
        self.frame[idx] = f
        g = self.starts[c] + f
        self.qpos[idx] = self.ref_q[g]
        self.qvel[idx] = self.ref_vel(g)
        self.warm[idx] = 0.0

    def weigh(self, results):
        """Gives each training clip that `results` (of `play`) shows was not followed four times
        its weight by length."""
        failed = {r["clip"] for r in results if not r["followed"]}
        extra = torch.tensor([4.0 if int(c) in failed else 1.0 for c in self.train.tolist()],
                             device=self.dev)
        self.weights = self.lengths[self.train].float() * extra

    def global_frame(self, ahead=0):
        f = torch.minimum(self.frame + ahead, self.lengths[self.clip] - 1)
        return self.starts[self.clip] + f

    def step(self, targets):
        self.ctrl[:] = torch.maximum(torch.minimum(targets, self.hi), self.lo)
        if self.graph is None and self.dev.startswith("cuda"):
            with wp.ScopedCapture() as cap:
                for _ in range(self.substeps):
                    mjw.step(self.m, self.d)
            self.graph = cap.graph
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            for _ in range(self.substeps):
                mjw.step(self.m, self.d)
        mjw.kinematics(self.m, self.d)
        self.frame += 1


# ---- what the net reads, and the reward --------------------------------------------------------

AHEAD = (1, 2, 5, 10)


def observe(sim):
    """The body's state in its heading's frame, and the clip's next frames in it."""
    root_q = sim.xquat[:, 1]
    root_x = sim.xpos[:, 1]
    hq = quat_conj(heading(root_q))
    n = sim.n
    parts = []
    parts.append(root_x[:, 1:2])
    local = quat_mul(hq, root_q)
    parts.append(local)
    parts.append(quat_rotate(hq, sim.qvel[:, :3]))
    parts.append(sim.qvel[:, 3:6])
    parts.append(sim.qpos[:, 7:])
    parts.append(0.1 * sim.qvel[:, 6:])
    rel = sim.xpos[:, 1:] - root_x[:, None]
    parts.append(quat_rotate(hq[:, None].expand(-1, rel.shape[1], -1), rel).reshape(n, -1))
    for a in AHEAD:
        g = sim.global_frame(a)
        tx = sim.ref_x[g][:, 1:] - root_x[:, None]
        parts.append(quat_rotate(hq[:, None].expand(-1, tx.shape[1], -1), tx).reshape(n, -1))
        parts.append(sim.ref_q[g][:, 7:] - sim.qpos[:, 7:])
    return torch.cat(parts, -1)


def errors(sim):
    """Each world's mean distance of a segment from its place in the clip (m), its mean turn
    from it (rad), and its root's distance."""
    g = sim.global_frame(0)
    dx = (sim.xpos[:, 1:] - sim.ref_x[g][:, 1:]).norm(dim=-1)
    dr = angle_of(quat_mul(sim.xquat[:, 1:], quat_conj(sim.ref_r[g][:, 1:])))
    return dx.mean(-1), dr.mean(-1), dx[:, 0], dx.max(-1).values


def reward(sim):
    ex, er, eroot, emax = errors(sim)
    g = sim.global_frame(0)
    dv = sim.qvel[:, 6:] - sim.ref_vel(g)[:, 6:]
    r = 0.45 * torch.exp(-100.0 * ex * ex) + 0.25 * torch.exp(-2.0 * er * er) + \
        0.1 * torch.exp(-0.005 * (dv * dv).sum(-1)) + 0.2 * torch.exp(-20.0 * eroot * eroot)
    fell = (emax > 0.5) | (ex > 0.25)
    return r, fell, ex


# ---- the net and PPO -----------------------------------------------------------------------------


class Normal:
    """A running mean and spread of the observations."""

    def __init__(self, n, device):
        self.mean = torch.zeros(n, device=device)
        self.var = torch.ones(n, device=device)
        self.count = 1e-4

    def update(self, x):
        b = x.shape[0]
        m = x.mean(0)
        v = x.var(0, unbiased=False)
        d = m - self.mean
        tot = self.count + b
        self.mean = self.mean + d * b / tot
        self.var = (self.var * self.count + v * b + d * d * self.count * b / tot) / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-6)).clamp(-5, 5)

    def state(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

    def load(self, s, device):
        self.mean, self.var, self.count = s["mean"].to(device), s["var"].to(device), s["count"]


def mlp(n_in, n_out, hidden):
    layers, last = [], n_in
    for h in hidden:
        layers += [torch.nn.Linear(last, h), torch.nn.ELU()]
        last = h
    layers.append(torch.nn.Linear(last, n_out))
    return torch.nn.Sequential(*layers)


def make_agent(n_obs, n_act, hidden):
    """The actor (a Gaussian about its mean, a learned spread) and the critic, two nets."""

    class Agent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = mlp(n_obs, n_act, hidden)
            self.v = mlp(n_obs, 1, hidden)
            self.log_std = torch.nn.Parameter(torch.full((n_act,), -1.6))
            with torch.no_grad():
                self.pi[-1].weight.mul_(0.01)
                self.pi[-1].bias.zero_()

        def dist(self, o):
            mu = self.pi(o)
            return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    return Agent()


# ---- training ------------------------------------------------------------------------------------


def play(sim, agent, norm, clips, most_frames=None, record=False):
    """The clips `clips` (indices) played by the net from their first frames (its mean action),
    `sim.n` at a time, to their ends (or `most_frames`): for each, whether it was followed to its
    end with no fall, the frame it fell at (or its length), the mean distance of a segment from
    its place while it was followed (m), and with `record`, each frame's qpos up to its fall."""
    out = []
    clips = [int(c) for c in clips]
    for i in range(0, len(clips), sim.n):
        batch = clips[i:i + sim.n]
        k = len(batch)
        c = torch.tensor(batch + [batch[0]] * (sim.n - k), device=sim.dev, dtype=torch.long)
        sim.clip[:] = c
        sim.frame[:] = 0
        g = sim.starts[c]
        sim.qpos[:] = sim.ref_q[g]
        sim.qvel[:] = sim.ref_vel(g)
        sim.warm[:] = 0.0
        mjw.kinematics(sim.m, sim.d)
        alive = torch.ones(sim.n, dtype=torch.bool, device=sim.dev)
        fell_at = sim.lengths[c].clone()
        dist = torch.zeros(sim.n, device=sim.dev)
        count = torch.zeros(sim.n, device=sim.dev)
        steps = int(sim.lengths[c[:k]].max().item()) - 2
        if most_frames:
            steps = min(steps, most_frames)
        frames = [] if record else None
        if record:
            frames.append(sim.qpos[:k].half().cpu())
        for t in range(steps):
            with torch.no_grad():
                a = agent.pi(norm(observe(sim)))
            sim.step(sim.ref_q[sim.global_frame(1)][:, 7:] + 0.5 * a.clamp(-1, 1))
            _, fell, ex = reward(sim)
            live_clip = sim.frame < (sim.lengths[sim.clip] - 1)
            newly = alive & fell & live_clip
            fell_at = torch.where(newly, sim.frame, fell_at)
            alive &= ~(fell & live_clip)
            m = (alive & live_clip).float()
            dist += ex * m
            count += m
            if record:
                frames.append(sim.qpos[:k].half().cpu())
        for j in range(k):
            entry = {"clip": batch[j], "followed": bool(alive[j]), "frames": int(fell_at[j]),
                     "mean_m": float(dist[j] / max(float(count[j]), 1.0))}
            if record:
                n = min(int(fell_at[j]), len(frames))
                entry["qpos"] = torch.stack([f[j] for f in frames[:n]]).float().numpy()
            out.append(entry)
    return out


def summary(results):
    if not results:
        return {"clips": 0}
    followed = sum(r["followed"] for r in results)
    frames = sum(r["frames"] for r in results)
    mean_mm = 1000.0 * sum(r["mean_m"] * r["frames"] for r in results) / max(frames, 1)
    return {"clips": len(results), "followed": followed / len(results), "mean_segment_mm": mean_mm}


def evaluate(sim, agent, norm, most_frames=600):
    """The training clips and the test clips, each played from its start (`play`); the training
    clips' results weigh how often each starts a world after."""
    training = play(sim, agent, norm, sim.train.tolist(), most_frames)
    sim.weigh(training)
    return {"training": summary(training),
            "test": summary(play(sim, agent, norm, sim.test.tolist(), most_frames))}


def filtered(sim, agent, norm, head):
    """Every clip played to its end by the net, as the simulator moved the body (each frame's
    qpos, up to a fall): the filtered capture (`WFLT`: a u32 length and a JSON head of each clip's
    name, set, whether it was followed and its frames; then zlib of the float32 qpos, clip after
    clip)."""
    results = play(sim, agent, norm, list(range(sim.starts.shape[0])), record=True)
    clips, chunks = [], []
    for r in results:
        clips.append({"name": head["names"][r["clip"]], "set": (head.get("sets") or ["training"] * 99999)[r["clip"]],
                      "followed": r["followed"], "frames": int(r["qpos"].shape[0]),
                      "length": int(sim.lengths[r["clip"]]), "mean_segment_mm": 1000.0 * r["mean_m"]})
        chunks.append(r["qpos"].astype("<f4"))
    body = zlib.compress(np.concatenate(chunks).tobytes(), 6)
    h = json.dumps({"clips": clips, "nq": head["nq"], "rate": head["rate"]}).encode()
    return b"WFLT" + struct.pack("<I", len(h)) + h + body, results


def main():
    run = runlib.Run()
    load_packages()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="creature.xml")
    p.add_argument("--refs", default="refs.bin")
    p.add_argument("--worlds", type=int, default=4096)
    p.add_argument("--steps", type=int, default=1000, help="updates of the net")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--hidden", default="1024,512")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--filtered", action="store_true", help="play every clip to its end at the end")
    p.add_argument("--weights", default="",
                   help="a trained tracker (tracker.pt): train nothing, play every clip of the refs to its end")
    a = p.parse_args(run.args)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wp.init()
    torch.manual_seed(run.seed)
    gen = torch.Generator(device=dev).manual_seed(run.seed)
    head, refs = load_refs(a.refs)
    t0 = time.time()
    with wp.ScopedDevice("cuda:0" if dev == "cuda" else "cpu"):
        sim = Sim(a.model, head, refs, a.worlds, dev, a.substeps, gen)
        print(f"sim: {a.worlds} worlds, {len(head['names'])} clips, {refs.shape[0]} frames, "
              f"set up in {time.time() - t0:.0f} s", flush=True)
        sim.reset(torch.ones(a.worlds, dtype=torch.bool, device=dev))
        mjw.kinematics(sim.m, sim.d)
        n_obs = observe(sim).shape[1]
        n_act = sim.mjm.nu
        hidden = [int(x) for x in a.hidden.split(",")]
        agent = make_agent(n_obs, n_act, hidden).to(dev)
        opt = torch.optim.Adam(agent.parameters(), lr=a.lr)
        norm = Normal(n_obs, dev)
        step, samples = 0, 0
        state = run.resume()
        if a.weights:
            # Phase 5 step 12: the tracker as the slow reference of the solver's movements.
            w = torch.load(a.weights, map_location=dev, weights_only=False)
            agent.load_state_dict(w["agent"])
            norm.load(w["norm"], dev)
            results = play(sim, agent, norm, list(range(sim.starts.shape[0])))
            for r in results:
                print(f"played {head['names'][r['clip']]}: followed {r['followed']}, frames {r['frames']} "
                      f"of {int(sim.lengths[r['clip']])}, mean {1000.0 * r['mean_m']:.1f} mm", flush=True)
            full = summary(results)
            print(f"played every clip: {json.dumps(full)}", flush=True)
            run.finish({"played": full, "seconds": time.time() - t0})
            return
        if state is not None:
            agent.load_state_dict(state["agent"])
            opt.load_state_dict(state["opt"])
            norm.load(state["norm"], dev)
            step, samples = state["step"], state["samples"]
            # A generator's state is a CPU byte tensor on any device.
            gen.set_state(state["gen"].cpu())
            torch.set_rng_state(state["torch_rng"])
            if "weights" in state:
                sim.weights = state["weights"].to(dev)
            sim.reset(torch.ones(a.worlds, dtype=torch.bool, device=dev))
        H, N = a.horizon, a.worlds
        obs_buf = torch.zeros(H, N, n_obs, device=dev)
        act_buf = torch.zeros(H, N, n_act, device=dev)
        logp_buf = torch.zeros(H, N, device=dev)
        rew_buf = torch.zeros(H, N, device=dev)
        done_buf = torch.zeros(H, N, device=dev)
        val_buf = torch.zeros(H + 1, N, device=dev)
        trunc_val = torch.zeros(H, N, device=dev)
        last_eval = None
        rate_t, rate_s = time.time(), samples
        while step < a.steps:
            ep_r, ep_fall = 0.0, 0.0
            with torch.no_grad():
                for h in range(H):
                    o_raw = observe(sim)
                    norm.update(o_raw)
                    o = norm(o_raw)
                    dist = agent.dist(o)
                    act = dist.sample()
                    obs_buf[h], act_buf[h] = o, act
                    logp_buf[h] = dist.log_prob(act).sum(-1)
                    val_buf[h] = agent.v(o).squeeze(-1)
                    sim.step(sim.ref_q[sim.global_frame(1)][:, 7:] + 0.5 * act.clamp(-1, 1))
                    r, fell, _ = reward(sim)
                    ended = sim.frame >= (sim.lengths[sim.clip] - 2)
                    # At a clip's end the return goes on (its value); at a fall it stops.
                    trunc_val[h] = 0.0
                    if ended.any():
                        trunc_val[h] = torch.where(ended & ~fell, agent.v(norm(observe(sim))).squeeze(-1), 0.0)
                    rew_buf[h] = r
                    done_buf[h] = (fell | ended).float()
                    ep_r += float(r.mean())
                    ep_fall += float(fell.float().mean())
                    sim.reset(fell | ended)
                val_buf[H] = agent.v(norm(observe(sim))).squeeze(-1)
                # GAE.
                adv = torch.zeros(H, N, device=dev)
                last = torch.zeros(N, device=dev)
                for h in reversed(range(H)):
                    nonterm = 1.0 - done_buf[h]
                    nxt = val_buf[h + 1] * nonterm + trunc_val[h]
                    delta = rew_buf[h] + 0.99 * nxt - val_buf[h]
                    last = delta + 0.99 * 0.95 * nonterm * last
                    adv[h] = last
                ret = adv + val_buf[:H]
            b_obs, b_act = obs_buf.reshape(H * N, -1), act_buf.reshape(H * N, -1)
            b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            mb = max((H * N) // 4, 1)
            for _ in range(5):
                perm = torch.randperm(H * N, device=dev, generator=gen)
                for i in range(0, H * N, mb):
                    j = perm[i:i + mb]
                    dist = agent.dist(b_obs[j])
                    lp = dist.log_prob(b_act[j]).sum(-1)
                    ratio = torch.exp(lp - b_logp[j])
                    pg = -torch.min(ratio * b_adv[j], ratio.clamp(0.8, 1.2) * b_adv[j]).mean()
                    vl = 0.5 * ((agent.v(b_obs[j]).squeeze(-1) - b_ret[j]) ** 2).mean()
                    loss = pg + vl
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    opt.step()
            step += 1
            samples += H * N
            now = time.time()
            sps = (samples - rate_s) / max(now - rate_t, 1e-6)
            run.progress(step, a.steps, f"reward {ep_r / H:.3f} falls {ep_fall / H:.4f} "
                                        f"samples {samples} at {sps:.0f}/s")
            if step % 10 == 0:
                print(f"update {step}: reward {ep_r / H:.3f} falls/step {ep_fall / H:.4f} "
                      f"samples {samples} ({sps:.0f} a second)", flush=True)
            if a.eval_every and step % a.eval_every == 0:
                last_eval = evaluate(sim, agent, norm)
                print(f"evaluate at {step}: {json.dumps(last_eval)}", flush=True)
                sim.reset(torch.ones(a.worlds, dtype=torch.bool, device=dev))
            if run.due() or step == a.steps:
                run.checkpoint(step, {
                    "agent": {k: v.cpu() for k, v in agent.state_dict().items()},
                    "opt": opt.state_dict(), "norm": norm.state(), "step": step,
                    "samples": samples, "gen": gen.get_state().cpu(),
                    "torch_rng": torch.get_rng_state(), "weights": sim.weights.cpu(),
                    "n_obs": n_obs, "n_act": n_act,
                    "hidden": hidden, "args": vars(a),
                })
            if run.should_stop():
                run.abandon()
                return
        final = evaluate(sim, agent, norm)
        print(f"evaluate at the end: {json.dumps(final)}", flush=True)
        import io
        buf = io.BytesIO()
        torch.save({"agent": {k: v.cpu() for k, v in agent.state_dict().items()},
                    "norm": norm.state(), "n_obs": n_obs, "n_act": n_act, "hidden": hidden,
                    "seed": run.seed, "args": vars(a), "samples": samples}, buf)
        run.output("tracker.pt", buf.getvalue())
        full = {}
        if a.filtered:
            data, results = filtered(sim, agent, norm, head)
            run.output("filtered.bin", data)
            full = {"training": summary([r for r in results if r["clip"] in set(sim.train.tolist())]),
                    "test": summary([r for r in results if r["clip"] in set(sim.test.tolist())])}
            print(f"the filtered capture, every clip to its end: {json.dumps(full)}", flush=True)
        run.finish({"step": step, "samples": samples, "evaluation": final, "full": full,
                    "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
