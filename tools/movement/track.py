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
        # The steps to a frame: as many as the model's step goes into a frame of the clips (8 at
        # the export's 1/240 s), or `substeps` where it is given.
        self.substeps = substeps or max(1, int(round(1.0 / (head["rate"] * self.mjm.opt.timestep))))
        self.gen = gen
        self.qpos = wp.to_torch(self.d.qpos)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.xpos = wp.to_torch(self.d.xpos)
        self.xquat = wp.to_torch(self.d.xquat)
        self.xipos = wp.to_torch(self.d.xipos)
        self.xfrc = wp.to_torch(self.d.xfrc_applied)
        self.geom_xpos = wp.to_torch(self.d.geom_xpos)
        self.geom_xmat = wp.to_torch(self.d.geom_xmat)
        self.warm = wp.to_torch(self.d.qacc_warmstart)
        self.nq, self.nv, self.nb = self.mjm.nq, self.mjm.nv, self.mjm.nbody
        # Dynamically active bodies: every body that has a joint (the root freejoint and every
        # articulated segment). Welded segments (such as 30 finger phalanges) are excluded so they
        # cannot dominate observations, rewards or fall distance.
        self.active = torch.tensor(
            [b for b in range(1, self.mjm.nbody) if self.mjm.body_jntnum[b] > 0],
            device=device, dtype=torch.long,
        )
        assert int(self.active[0]) == 1, f"root body 1 must be active, got {self.active[:4]}"
        self.names = list(head["names"])
        # Colliding capsules for computing ground clearance on reset (prevents underground launches).
        # (spheres too: a sphere is a capsule of no length, its half length 0 in the model)
        caps = [g for g in range(self.mjm.ngeom) if self.mjm.geom_contype[g] != 0 and self.mjm.geom_type[g] in (2, 3)]
        self.caps = torch.tensor(caps, device=device, dtype=torch.long)
        self.cap_r = torch.tensor(self.mjm.geom_size[caps, 0], device=device, dtype=torch.float32) if caps else torch.zeros(0, device=device)
        self.cap_h = torch.tensor(self.mjm.geom_size[caps, 1], device=device, dtype=torch.float32) if caps else torch.zeros(0, device=device)
        self.lo = torch.tensor(self.mjm.actuator_ctrlrange[:, 0], device=device, dtype=torch.float32)
        self.hi = torch.tensor(self.mjm.actuator_ctrlrange[:, 1], device=device, dtype=torch.float32)
        # Per-joint action scale from each joint's true range (avoids 3-7x range overshoot on narrow
        # spine/wrist twists while keeping full authority on hips, knees and ankles).
        jids = self.mjm.actuator_trnid[:, 0]
        self.jnt_lo = torch.tensor(self.mjm.jnt_range[jids, 0], device=device, dtype=torch.float32)
        self.jnt_hi = torch.tensor(self.mjm.jnt_range[jids, 1], device=device, dtype=torch.float32)
        self.act_scale = (0.5 * (self.jnt_hi - self.jnt_lo)).clamp(min=0.15, max=1.0)
        self.rate = head["rate"]
        self.dt = 1.0 / self.rate
        # The clips: their frames, where each starts and how long it is.
        self.ref_q = torch.tensor(refs, device=device)
        self.starts = torch.tensor(head["starts"], device=device, dtype=torch.long)
        self.lengths = torch.tensor(head["lengths"], device=device, dtype=torch.long)
        self.ref_x, self.ref_r, self.ref_lift = self.frames_of(self.ref_q)
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
        self.start_share = 0.0
        self.fall_mean = 0.5
        # Each moving ball joint's range as the body holds it (the export's `ball_s{segment}`):
        # its body, its parent's, its frame in the parent and its numbers' lo and hi.
        child, parent, frame, lo, hi = [], [], [], [], []
        for i in range(self.mjm.nnumeric):
            name = self.mjm.numeric(i).name
            if name.startswith("ball_s"):
                b = self.mjm.body(f"s{name[6:]}").id
                data = self.mjm.numeric(i).data
                child.append(b)
                parent.append(int(self.mjm.body_parentid[b]))
                frame.append(self.mjm.body_quat[b].tolist())
                lo.append(data[:3].tolist())
                hi.append(data[3:6].tolist())
        self.ball_child = torch.tensor(child, device=device, dtype=torch.long)
        self.ball_parent = torch.tensor(parent, device=device, dtype=torch.long)
        self.ball_frame = torch.tensor(frame, device=device, dtype=torch.float32).reshape(-1, 4)
        self.ball_lo = torch.tensor(lo, device=device, dtype=torch.float32).reshape(-1, 3)
        self.ball_hi = torch.tensor(hi, device=device, dtype=torch.float32).reshape(-1, 3)
        self.range_weight = 0.0
        # The net's last action (each world's, 0 when it starts), which it reads and pays for
        # changing (`rate_weight`): a net that reads only the body's state shook it at a quarter of
        # its rate (`p4h-12`: 83 to 99% of the joints' speed's power above 6 Hz, the servos alone
        # 2%).
        self.last_act = torch.zeros(worlds, self.mjm.nu, device=device)
        # Each muscle's activation (`activation_ms`): what the servo applies follows the net's
        # command with a lag, as a muscle's force follows its excitation (Zajac 1989; Thelen 2003:
        # some 15 ms rising, 50 falling). 0: none, the command applied at once.
        self.applied = torch.zeros(worlds, self.mjm.nu, device=device)
        self.activation_ms = 0.0
        # The joints' numbers a frame ago and two (`accel_weight`: the reward falls with the
        # joints' acceleration's distance from the clip's, which is what a tremor is made of).
        self.joints_1 = torch.zeros(worlds, self.mjm.nq - 7, device=device)
        self.joints_2 = torch.zeros(worlds, self.mjm.nq - 7, device=device)
        self.accel_weight = 0.0
        self.read_last = False
        self.rate_weight = 0.0
        # The soles: the feet and the toes (segments 9, 10, 13 and 14 of the human), as places in
        # the active bodies, held to the clip's height and stillness where its foot is planted.
        act_list = self.active.tolist()
        sole_ids = [self.mjm.body(f"s{k}").id for k in (9, 10, 13, 14) if f"s{k}" in
                    [self.mjm.body(b).name for b in range(self.mjm.nbody)]]
        self.soles = torch.tensor([act_list.index(b) for b in sole_ids if b in act_list], device=device,
                                  dtype=torch.long)
        self.sole_weight = 0.0

    def command(self, a):
        """The servos' targets for the net's action `a` (in [-1, 1]): the clip's next frame and
        the action, through the muscles' activation (`activation_ms`)."""
        if self.activation_ms > 0.0:
            k = 1.0 - math.exp(-1000.0 / (self.rate * self.activation_ms))
            self.applied = self.applied + k * (a - self.applied)
        else:
            self.applied = a
        return self.ref_q[self.global_frame(1)][:, 7:] + self.act_scale * self.applied

    def tilt(self, g):
        """Each world's own gravity (`g`, worlds by 3): a floor tilted under the body is a slope.
        Set before the first step (the step's graph reads the array it finds then)."""
        self.m.opt.gravity = wp.from_torch(g.contiguous().float(), dtype=wp.vec3)

    def frames_of(self, q):
        """Each body's place and turn for each of the qpos `q`, plus the vertical lift (m) needed
        to clear floor penetration on reset."""
        out_x, out_r, out_l = [], [], []
        n = q.shape[0]
        b = self.n
        for i in range(0, n, b):
            k = min(b, n - i)
            self.qpos[:k] = q[i:i + k]
            mjw.kinematics(self.m, self.d)
            out_x.append(self.xpos[:k, self.active].clone())
            out_r.append(self.xquat[:k, self.active].clone())
            if self.caps.numel():
                gpos = self.geom_xpos[:k, self.caps, 1]
                gmat = self.geom_xmat[:k, self.caps].reshape(k, -1, 3, 3)
                axis_y = gmat[..., 1, 2].abs()
                min_y = (gpos - axis_y * self.cap_h - self.cap_r).amin(dim=-1)
                out_l.append((-min_y).clamp(min=0.0).clone())
            else:
                out_l.append(torch.zeros(k, device=self.dev))
        return torch.cat(out_x), torch.cat(out_r), torch.cat(out_l)

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
        """Starts the worlds `which` (a mask) at a random frame of a random clip, or a share of
        them (`start_share`) at its first frame. Lifts any frame that penetrates the floor and
        updates kinematics immediately so the next observation sees the new state, never the old."""
        idx = which.nonzero().flatten()
        if idx.numel() == 0:
            return
        c = self.train[torch.multinomial(self.weights, idx.numel(), replacement=True,
                                         generator=self.gen)]
        span = (self.lengths[c] - 2).clamp(min=1)
        f = (torch.rand(idx.numel(), device=self.dev, generator=self.gen) * span).long()
        first = torch.rand(idx.numel(), device=self.dev, generator=self.gen) < self.start_share
        f = torch.where(first, torch.zeros_like(f), f)
        self.clip[idx] = c
        self.frame[idx] = f
        g = self.starts[c] + f
        self.qpos[idx] = self.ref_q[g]
        lift = self.ref_lift[g]
        self.qpos[idx, 1] += torch.where(lift > 0.002, lift, torch.zeros_like(lift))
        self.qvel[idx] = self.ref_vel(g)
        self.warm[idx] = 0.0
        self.last_act[idx] = 0.0
        self.applied[idx] = 0.0
        self.joints_1[idx] = self.ref_q[torch.maximum(g - 1, self.starts[c])][:, 7:]
        mjw.kinematics(self.m, self.d)

    def weigh(self, results):
        """Gives each training clip that `results` (of `play`) shows was not followed extra weight
        by length (4x normally, 8x once fewer than 15% of clips fail so hard-example mining stays
        focused on the remaining tail)."""
        failed = {r["clip"] for r in results if not r["followed"]}
        fail_rate = len(failed) / max(len(results), 1)
        boost = 8.0 if fail_rate < 0.15 else 4.0
        extra = torch.tensor([boost if int(c) in failed else 1.0 for c in self.train.tolist()],
                             device=self.dev)
        self.weights = self.lengths[self.train].float() * extra

    def global_frame(self, ahead=0):
        f = torch.minimum(self.frame + ahead, self.lengths[self.clip] - 1)
        return self.starts[self.clip] + f

    def step(self, targets):
        # The soles' places before the step: how far they slide in it (`reward`).
        self.sole_before = self.xpos[:, self.active[self.soles]].clone() if self.soles.numel() else None
        self.joints_2 = self.joints_1
        self.joints_1 = self.qpos[:, 7:].clone()
        self.ctrl[:] = torch.maximum(torch.minimum(targets, self.hi), self.lo)
        if self.graph is None and self.dev.startswith("cuda"):
            with wp.ScopedCapture() as cap:
                for _ in range(self.substeps):
                    mjw.step(self.m, self.d)
                mjw.kinematics(self.m, self.d)
            self.graph = cap.graph
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            for _ in range(self.substeps):
                mjw.step(self.m, self.d)
            mjw.kinematics(self.m, self.d)
        self.frame += 1


# ---- what the net reads, and the reward --------------------------------------------------------

AHEAD = (1, 3, 6, 12)


def rot6d(q):
    """Continuous 6D rotation representation (rotated +x and +y basis vectors): avoids the ±q
    double-cover sign flip when yaw crosses ±pi."""
    ex = torch.zeros_like(q[..., 1:])
    ex[..., 0] = 1.0
    ey = torch.zeros_like(q[..., 1:])
    ey[..., 1] = 1.0
    return torch.cat([quat_rotate(q, ex), quat_rotate(q, ey)], -1)


def observe(sim):
    """The body's state in its heading's frame (active bodies only, continuous 6D root orientation),
    and the clip's next frames as direct per-body tracking deltas, target root orientation, and
    joint angle deltas."""
    root_q = sim.xquat[:, 1]
    root_x = sim.xpos[:, 1]
    hq = quat_conj(heading(root_q))
    n = sim.n
    act = sim.active
    act_x = sim.xpos[:, act]
    parts = []
    parts.append(root_x[:, 1:2])
    local = quat_mul(hq, root_q)
    parts.append(rot6d(local))
    parts.append(quat_rotate(hq, sim.qvel[:, :3]))
    parts.append(quat_rotate(local, sim.qvel[:, 3:6]))
    parts.append(sim.qpos[:, 7:])
    parts.append(0.1 * sim.qvel[:, 6:])
    rel = act_x[:, 1:] - root_x[:, None]
    parts.append(quat_rotate(hq[:, None].expand(-1, rel.shape[1], -1), rel).reshape(n, -1))
    for a in AHEAD:
        g = sim.global_frame(a)
        # Direct per-body error vector to target (includes both root drift and local pose),
        # plus target root orientation, target root linear velocity, and joint angle deltas.
        dx = sim.ref_x[g] - act_x
        parts.append(quat_rotate(hq[:, None].expand(-1, dx.shape[1], -1), dx).reshape(n, -1))
        parts.append(rot6d(quat_mul(hq, sim.ref_r[g][:, 0])))
        parts.append(quat_rotate(hq, sim.ref_vel(g)[:, :3]))
        parts.append(sim.ref_q[g][:, 7:] - sim.qpos[:, 7:])
    if sim.read_last:
        # The muscles' state: what is applied (the last action itself where there is no lag).
        parts.append(sim.applied if sim.activation_ms > 0.0 else sim.last_act)
    return torch.cat(parts, -1)


def errors(sim):
    """Each world's mean distance of an active segment from its place in the clip (m), its root-
    relative local pose distance (m), its mean turn error (rad), its root distance (m), and the
    per-body arrays for multi-scale reward."""
    g = sim.global_frame(0)
    act = sim.active
    sx = sim.xpos[:, act]
    rx = sim.ref_x[g]
    dx = (sx - rx).norm(dim=-1)
    dx_loc = ((sx - sx[:, :1]) - (rx - rx[:, :1])).norm(dim=-1)
    dr = angle_of(quat_mul(sim.xquat[:, act], quat_conj(sim.ref_r[g])))
    return dx, dx_loc, dr


def exploded(sim):
    """The worlds whose simulation blew up: a state not finite, or moving faster than 200 (m/s or
    rad/s)."""
    ok = torch.isfinite(sim.qpos).all(-1) & torch.isfinite(sim.qvel).all(-1)
    ok &= sim.qvel.abs().amax(-1) < 200.0
    return ~ok


def ball_numbers(sim):
    """Each moving ball joint's numbers as the engine has them (`joint::numbers_of`: the twist
    about the joint's y, then the swing left as a vector, its x and z), from the simulator's
    turns: worlds by joints by 3."""
    n, k = sim.n, sim.ball_child.numel()
    pq = sim.xquat[:, sim.ball_parent]
    cq = sim.xquat[:, sim.ball_child]
    fr = sim.ball_frame[None].expand(n, -1, -1)
    r = quat_mul(quat_conj(fr), quat_mul(quat_conj(pq), cq))
    r = torch.where(r[..., :1] < 0, -r, r)
    tw = torch.zeros_like(r)
    tw[..., 0], tw[..., 2] = r[..., 0], r[..., 2]
    tw = tw / tw.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    angle = 2.0 * torch.atan2(tw[..., 2], tw[..., 0])
    sw = quat_mul(r, quat_conj(tw))
    sw = torch.where(sw[..., :1] < 0, -sw, sw)
    sn = sw[..., 1:].norm(dim=-1, keepdim=True)
    f = torch.where(sn < 1e-8, torch.full_like(sn, 2.0), 2.0 * torch.atan2(sn, sw[..., :1]) / sn.clamp(min=1e-12))
    v = sw[..., 1:] * f
    return torch.stack([v[..., 0], v[..., 2], angle], -1)


def range_worst(sim):
    """Each world's worst ball joint's distance past its range (rad, 0 within): what the judge
    holds a movement's worst frame to."""
    return range_excess(sim, worst=True)


def range_excess(sim, worst=False):
    """Each world's ball joints' distances past their ranges, summed (rad): the engine's
    `Joint::range` (the swing's ellipse, scaled by its smaller half axis, and the twist's
    interval), 0 within. The simulator's hinges hold a box of Euler angles that contains each
    cone; this is what is outside the cone."""
    if sim.ball_child.numel() == 0:
        return torch.zeros(sim.n, device=sim.dev)
    q = ball_numbers(sim)
    lo, hi = sim.ball_lo[None], sim.ball_hi[None]
    ax = torch.where(q[..., 0] >= 0, hi[..., 0], -lo[..., 0]).clamp(min=1e-4)
    ay = torch.where(q[..., 1] >= 0, hi[..., 1], -lo[..., 1]).clamp(min=1e-4)
    swing = (torch.sqrt((q[..., 0] / ax) ** 2 + (q[..., 1] / ay) ** 2) - 1.0) * torch.minimum(ax, ay)
    twist = torch.maximum(lo[..., 2] - q[..., 2], q[..., 2] - hi[..., 2])
    past = torch.relu(torch.maximum(swing, twist))
    return past.amax(-1) if worst else past.sum(-1)


def reward(sim, act_clamped=None):
    dx, dx_loc, dr = errors(sim)
    ex = dx.mean(-1)
    eroot = dx[:, 0]
    g = sim.global_frame(0)
    rv = sim.ref_vel(g)
    dv_root = (sim.qvel[:, :3] - rv[:, :3]).norm(dim=-1)
    dv = sim.qvel[:, 6:] - rv[:, 6:]
    # Multi-scale PHC-style reward: sharp fine term (<50 mm accuracy) + wide global term (non-zero
    # gradient all the way to 0.5 m) + root-relative local pose + orientation + root + velocity.
    r_pos_fine = torch.exp(-100.0 * (dx * dx).mean(-1))
    r_pos_wide = torch.exp(-10.0 * (dx * dx).mean(-1))
    r_loc = torch.exp(-40.0 * (dx_loc * dx_loc).mean(-1))
    r_rot = torch.exp(-5.0 * (dr * dr).mean(-1))
    r_root = torch.exp(-10.0 * eroot * eroot)
    r_vel = 0.5 * torch.exp(-0.5 * dv_root * dv_root) + 0.5 * torch.exp(-0.05 * (dv * dv).mean(-1))
    r = (0.25 * r_pos_fine + 0.20 * r_pos_wide + 0.20 * r_loc +
         0.15 * r_rot + 0.10 * r_root + 0.10 * r_vel)
    if sim.sole_weight > 0.0 and sim.soles.numel():
        # The soles where the clip's foot is planted (moving under 0.15 m/s along the ground): at
        # the clip's height (to 1 cm) and still (to 0.1 m/s). The places' terms above cannot see a
        # millimetre: an error of 11 mm costs exp(-100 dx^2) 1%.
        g1 = sim.global_frame(1)
        ref_now, ref_next = sim.ref_x[g][:, sim.soles], sim.ref_x[g1][:, sim.soles]
        planted = ((ref_next - ref_now)[..., [0, 2]].norm(dim=-1) * sim.rate) < 0.15
        mine = sim.xpos[:, sim.active[sim.soles]]
        dy = mine[..., 1] - ref_now[..., 1]
        sliding = (mine - sim.sole_before)[..., [0, 2]].norm(dim=-1) * sim.rate
        each = torch.exp(-(dy / 0.01) ** 2) * torch.exp(-(sliding / 0.1) ** 2)
        k = planted.float().sum(-1)
        r_sole = torch.where(k > 0, (each * planted.float()).sum(-1) / k.clamp(min=1.0), torch.ones_like(k))
        r = (1.0 - sim.sole_weight) * r + sim.sole_weight * r_sole
    if act_clamped is not None:
        r = (r - 0.02 * (act_clamped * act_clamped).mean(-1)).clamp(min=0.0)
        if sim.rate_weight > 0.0:
            # A change of the action from the last costs (the shake's cure).
            d = act_clamped - sim.last_act
            r = (r - sim.rate_weight * (d * d).mean(-1)).clamp(min=0.0)
    if sim.accel_weight > 0.0:
        # The joints' acceleration over the last two frames against the clip's over the same
        # (rad a frame squared): a body that shakes about its clip's path has the first without
        # the second.
        lo = sim.starts[sim.clip]
        mine = sim.qpos[:, 7:] - 2.0 * sim.joints_1 + sim.joints_2
        theirs = sim.ref_q[g][:, 7:] - 2.0 * sim.ref_q[torch.maximum(g - 1, lo)][:, 7:] + \
            sim.ref_q[torch.maximum(g - 2, lo)][:, 7:]
        d = mine - theirs
        r = r * torch.exp(-sim.accel_weight * (d * d).mean(-1))
    if sim.range_weight > 0.0:
        # The body's ranges, which the simulator's hinges do not hold (`range_excess`).
        r = r * torch.exp(-sim.range_weight * range_excess(sim))
    bad = exploded(sim)
    fell = (ex > sim.fall_mean) | bad
    r = torch.where(bad | ~torch.isfinite(r), torch.zeros_like(r), r)
    ex = torch.where(torch.isfinite(ex), ex, torch.zeros_like(ex))
    return r, fell, ex


# ---- the net and PPO -----------------------------------------------------------------------------


class Normal:
    """A running mean and spread of the observations."""

    def __init__(self, n, device):
        self.mean = torch.zeros(n, device=device)
        self.var = torch.ones(n, device=device)
        self.count = 1e-4

    def update(self, x):
        # Only finite rows of sane size teach the mean and the spread.
        keep = torch.isfinite(x).all(-1) & (x.abs().amax(-1) < 1.0e3)
        x = x[keep]
        b = x.shape[0]
        if b == 0:
            return
        m = x.mean(0)
        v = x.var(0, unbiased=False)
        d = m - self.mean
        tot = self.count + b
        self.mean = self.mean + d * b / tot
        self.var = (self.var * self.count + v * b + d * d * self.count * b / tot) / tot
        self.count = tot

    def __call__(self, x):
        y = ((x - self.mean) / torch.sqrt(self.var + 1e-6)).clamp(-5, 5)
        return torch.nan_to_num(y, nan=0.0, posinf=5.0, neginf=-5.0)

    def state(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

    def load(self, s, device):
        self.mean, self.var, self.count = s["mean"].to(device), s["var"].to(device), s["count"]


class ValNorm:
    """Running mean and std of value targets (scalar): keeps critic outputs near unit scale."""

    def __init__(self, device):
        self.mean = torch.tensor(0.0, device=device)
        self.var = torch.tensor(1.0, device=device)
        self.count = 1e-4

    def update(self, x):
        x = x[torch.isfinite(x)]
        b = x.numel()
        if b == 0:
            return
        m = x.mean()
        v = x.var(unbiased=False)
        d = m - self.mean
        tot = self.count + b
        self.mean = self.mean + d * b / tot
        self.var = (self.var * self.count + v * b + d * d * self.count * b / tot) / tot
        self.count = tot

    def norm(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-6)

    def denorm(self, x):
        return x * torch.sqrt(self.var + 1e-6) + self.mean

    def state(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

    def load(self, s, device):
        self.mean, self.var, self.count = s["mean"].to(device), s["var"].to(device), s["count"]


def mlp(n_in, n_out, hidden, layer_norm=False):
    layers, last = [], n_in
    for h in hidden:
        lin = torch.nn.Linear(last, h)
        torch.nn.init.orthogonal_(lin.weight, gain=math.sqrt(2.0))
        torch.nn.init.zeros_(lin.bias)
        layers.append(lin)
        if layer_norm:
            layers.append(torch.nn.LayerNorm(h))
        layers.append(torch.nn.ELU())
        last = h
    head = torch.nn.Linear(last, n_out)
    torch.nn.init.orthogonal_(head.weight, gain=1.0)
    torch.nn.init.zeros_(head.bias)
    layers.append(head)
    return torch.nn.Sequential(*layers)


def make_agent(n_obs, n_act, hidden, log_std=None, layer_norm=True):
    """The actor (a Gaussian about its mean; its spread learned, or fixed at e^`log_std`) and the
    critic, two nets."""

    class Agent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = mlp(n_obs, n_act, hidden, layer_norm=layer_norm)
            self.v = mlp(n_obs, 1, hidden, layer_norm=layer_norm)
            self.log_std = torch.nn.Parameter(torch.full((n_act,), -2.3 if log_std is None else log_std),
                                              requires_grad=log_std is None)
            with torch.no_grad():
                self.pi[-1].weight.mul_(0.05)
                self.pi[-1].bias.zero_()

        def dist(self, o):
            mu = self.pi(o)
            return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    return Agent()


# ---- training ------------------------------------------------------------------------------------

# The least spread of the actions (a log): below it a step of the net changes the action's
# likelihood too much.
LEAST_LOG_STD = -3.0


# The actions' bound: a mean past it is held back (the bounds loss of rl_games). The servos'
# targets take the actions clamped to 1; a mean past that changes nothing the simulator does,
# while the likelihood's gradient can push it on: p4h-3's net after its collapse had 23.5% of its
# means past 1 and 12.6% past 2 (the most 6.05), a healthy net's at most 0.99.
BOUND = 1.1


def ppo(agent, opt, b_obs, b_act, b_logp, b_adv, b_ret, gen, target_kl, bounds=0.01, vnorm=None, smooth=None):
    """PPO's epochs over the rollout (clipped at 0.2, 4 minibatches of 32,768 an epoch), actor and
    critic gradients clipped to 1 apart, value targets normalized by `vnorm` when given, actions'
    means held within `BOUND` (weight `bounds`), and smooth KL-adaptive learning rate using
    per-action-dimension KL so high-DOF bodies (62+ actuators) are not throttled 60x too hard.
    With `smooth` (each sample's next observation, whether it has one, and the weights and the
    noise), the actions' means are held smooth as Mysore et al. 2021 hold them: near the mean of
    the next observation, and near the mean of an observation a little off (a net that changes
    its action much for a little change of what it reads shakes the body it moves)."""
    dev = b_obs.device
    n = b_obs.shape[0]
    n_act = b_act.shape[-1]
    mb = max(n // 4, 1)
    if vnorm is not None:
        vnorm.update(b_ret)
        b_ret_t = vnorm.norm(b_ret)
    else:
        b_ret_t = b_ret
    actor = list(agent.pi.parameters()) + ([agent.log_std] if agent.log_std.requires_grad else [])
    critic = list(agent.v.parameters())
    skipped, kl, clipped, vloss, epochs = 0, 0.0, 0.0, 0.0, 0
    for _ in range(4):
        perm = torch.randperm(n, device=dev, generator=gen)
        kls, clips, vls = [], [], []
        for i in range(0, n, mb):
            j = perm[i:i + mb]
            dist = agent.dist(b_obs[j])
            lp = dist.log_prob(b_act[j]).sum(-1)
            log_ratio = (lp - b_logp[j]).clamp(-20.0, 20.0)
            ratio = torch.exp(log_ratio)
            pg = -torch.min(ratio * b_adv[j], ratio.clamp(0.8, 1.2) * b_adv[j]).mean()
            vl = 0.5 * ((agent.v(b_obs[j]).squeeze(-1) - b_ret_t[j]) ** 2).mean()
            mu = dist.mean
            bl = (torch.relu(mu - BOUND) ** 2 + torch.relu(-BOUND - mu) ** 2).sum(-1).mean()
            loss = pg + vl + bounds * bl
            if smooth is not None:
                nxt, pair, lam_t, lam_s, sigma = smooth
                in_time = (((mu - agent.pi(nxt[j])) ** 2).mean(-1) * pair[j]).mean()
                near = b_obs[j] + sigma * torch.randn(b_obs[j].shape, device=dev, generator=gen)
                in_place = ((mu - agent.pi(near)) ** 2).mean()
                loss = loss + lam_t * in_time + lam_s * in_place
            if not torch.isfinite(loss):
                skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 1.0)
            torch.nn.utils.clip_grad_norm_(critic, 1.0)
            opt.step()
            with torch.no_grad():
                agent.log_std.clamp_(min=LEAST_LOG_STD)
                kls.append(float(((ratio - 1.0) - log_ratio).mean()) / max(n_act, 1))
                clips.append(float(((ratio - 1.0).abs() > 0.2).float().mean()))
                vls.append(float(vl))
        epochs += 1
        if kls:
            kl, clipped, vloss = sum(kls) / len(kls), sum(clips) / len(clips), sum(vls) / len(vls)
        if 0 < target_kl and kl > 4.0 * target_kl:
            break
    if target_kl <= 0:
        return (kl, skipped), clipped, vloss, epochs
    for g in opt.param_groups:
        if kl > 1.5 * target_kl:
            g["lr"] = max(g["lr"] / 1.25, 3e-5)
        elif kl < 0.5 * target_kl:
            g["lr"] = min(g["lr"] * 1.25, 5e-4)
    return (kl, skipped), clipped, vloss, epochs


class Guard:
    """The weights of the last good update, and the test of a collapse: the mean reward of the last
    5 updates below 85% of the best such mean."""

    def __init__(self):
        self.rewards, self.best, self.good, self.rollbacks = [], 0.0, None, 0

    def add(self, r):
        self.rewards = (self.rewards + [r])[-5:]

    def recent(self):
        return sum(self.rewards) / max(len(self.rewards), 1)

    def keep(self, step, agent, opt, norm, vnorm=None):
        if len(self.rewards) < 5:
            return
        r = self.recent()
        if r >= 0.95 * self.best:
            self.best = max(self.best, r)
            self.good = {"step": step, "agent": {k: v.detach().clone() for k, v in agent.state_dict().items()},
                         "opt": __import__("copy").deepcopy(opt.state_dict()), "norm": norm.state(),
                         "vnorm": vnorm.state() if vnorm is not None else None}

    def rebase(self):
        self.rewards, self.best = [], 0.0

    def collapsed(self):
        return self.good is not None and len(self.rewards) == 5 and self.recent() < 0.85 * self.best \
            and self.rollbacks < 6

    def restore(self, agent, opt, norm, vnorm=None):
        agent.load_state_dict(self.good["agent"])
        lr = opt.param_groups[0]["lr"] * 0.5
        opt.load_state_dict(self.good["opt"])
        for g in opt.param_groups:
            g["lr"] = lr
        norm.load(self.good["norm"], agent.log_std.device)
        if vnorm is not None and self.good.get("vnorm") is not None:
            vnorm.load(self.good["vnorm"], agent.log_std.device)
        self.rewards = []
        self.rollbacks += 1



def play(sim, agent, norm, clips, most_frames=None, record=False, push=None):
    """The clips `clips` (indices) played by the net from their first frames (its mean action),
    `sim.n` at a time, to their ends (or `most_frames`): for each, whether it was followed to its
    end with no fall, the frame it fell at (or its length), the mean distance of a segment from
    its place while it was followed (m), and with `record`, each frame's qpos up to its fall. With
    `push` (a force on the root for each world in the world's frame, N, its first frame and how
    many frames it lasts), the worlds of the first batch are pushed."""
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
        lift = sim.ref_lift[g]
        sim.qpos[:, 1] += torch.where(lift > 0.002, lift, torch.zeros_like(lift))
        sim.qvel[:] = sim.ref_vel(g)
        sim.warm[:] = 0.0
        mjw.kinematics(sim.m, sim.d)
        sim.last_act[:] = 0.0
        sim.applied[:] = 0.0
        sim.joints_1[:] = sim.ref_q[g][:, 7:]
        alive = torch.ones(sim.n, dtype=torch.bool, device=sim.dev)
        fell_at = sim.lengths[c].clone()
        dist = torch.zeros(sim.n, device=sim.dev)
        past = torch.zeros(sim.n, device=sim.dev)
        worst = torch.zeros(sim.n, device=sim.dev)
        joints = [sim.qpos[:k, 7:].half().clone()]
        count = torch.zeros(sim.n, device=sim.dev)
        steps = int(sim.lengths[c[:k]].max().item()) - 2
        if most_frames:
            steps = min(steps, most_frames)
        frames = [] if record else None
        extras = [] if record else None
        if record:
            frames.append(sim.qpos[:k].float().cpu().clone())
            extras.append(torch.zeros(k, extra_width(sim), dtype=torch.float16))
        for t in range(steps):
            with torch.no_grad():
                a = agent.pi(norm(observe(sim)))
            if push is not None and i == 0:
                force, first, frames_on = push
                on = (t >= first) & (t < first + frames_on)
                sim.xfrc[:, 1, :3] = torch.where(on[:, None], force, torch.zeros_like(force))
            sim.step(sim.command(a.clamp(-1, 1)))
            _, fell, ex = reward(sim)
            sim.last_act = a.clamp(-1, 1)
            live_clip = sim.frame < (sim.lengths[sim.clip] - 1)
            newly = alive & fell & live_clip
            fell_at = torch.where(newly, sim.frame, fell_at)
            alive &= ~(fell & live_clip)
            m = (alive & live_clip).float()
            dist += ex * m
            past += range_excess(sim) * m
            worst = torch.maximum(worst, range_worst(sim) * m)
            joints.append(sim.qpos[:k, 7:].half().clone())
            count += m
            if record:
                frames.append(sim.qpos[:k].float().cpu().clone())
                extras.append(extra_of(sim, a.clamp(-1, 1))[:k].half().cpu())
        joints = torch.stack(joints)
        for j in range(k):
            entry = {"clip": batch[j], "followed": bool(alive[j]), "frames": int(fell_at[j]),
                     "mean_m": float(dist[j] / max(float(count[j]), 1.0)),
                     "past_range": float(past[j] / max(float(count[j]), 1.0)),
                     "worst_range": float(worst[j]),
                     "tremor": tremor(joints[:min(int(fell_at[j]), joints.shape[0]), j].float(), sim.rate)}
            if record:
                n = min(int(fell_at[j]), len(frames))
                entry["qpos"] = torch.stack([f[j] for f in frames[:n]]).float().numpy()
                entry["extra"] = torch.stack([e[j] for e in extras[:n]]).numpy()
            out.append(entry)
    return out


def widened(mine, theirs, dev):
    """Weights `theirs` for a net shaped as `mine` whose inputs are more (inputs added at the end,
    as the last action is): each input layer's columns for the new inputs 0, so the net does what
    it did until it learns to use them."""
    out = {}
    for k, v in mine.items():
        t = theirs[k].to(dev)
        if t.shape != v.shape:
            assert t.dim() == 2 and t.shape[0] == v.shape[0] and t.shape[1] < v.shape[1], \
                f"{k}: {tuple(t.shape)} does not widen to {tuple(v.shape)}"
            w = torch.zeros_like(v)
            w[:, :t.shape[1]] = t
            t = w
        out[k] = t
    return out


def widened_norm(s, n):
    """An observations' scale for `n` inputs from one for fewer: the new ones mean 0, spread 1."""
    k = s["mean"].shape[0]
    if k == n:
        return s
    mean = torch.zeros(n, dtype=s["mean"].dtype)
    var = torch.ones(n, dtype=s["var"].dtype)
    mean[:k], var[:k] = s["mean"].cpu(), s["var"].cpu()
    return {"mean": mean, "var": var, "count": s["count"]}


def extra_width(sim):
    return 2 * sim.mjm.nu + 3 * (sim.active.numel() - 1) + 3


def extra_of(sim, act):
    """What the simulator knows of a frame beside its poses (the filtered capture's step 8, "with
    the contacts"): the net's action, each servo's force (N m), the world's force on each moving
    segment but the root at the frame's last step (N, the floor's contacts: x, y, z: which
    segments touch), and the body's momentum (kg m/s), whose change over a frame less its weight
    is the world's force averaged over the frame (what poses at 30 a second can give)."""
    mjw.rne_postconstraint(sim.m, sim.d)
    cf = wp.to_torch(sim.d.cfrc_ext)[:, sim.active[1:], 3:6].reshape(sim.n, -1)
    mjw.com_pos(sim.m, sim.d)
    mjw.com_vel(sim.m, sim.d)
    mjw.subtree_vel(sim.m, sim.d)
    p = wp.to_torch(sim.d.subtree_linvel)[:, 1] * float(sim.mjm.body_subtreemass[1])
    return torch.cat([act, wp.to_torch(sim.d.actuator_force), cf, p], -1)


# ---- the generator: a clip under changes that physics permits (phase 5 step 16) ------------------

PUSH_N = (50.0, 300.0)
PUSH_FRAMES = 6
SLOPE_DEG = (2.0, 10.0)
SPEEDS = (0.8, 1.25)


def time_scaled(q, s):
    """A clip's qpos `q` (frames by nq) played `s` times as fast: its frames at times s k."""
    n = q.shape[0]
    at = np.arange(0.0, n - 1, s)
    i = np.floor(at).astype(int)
    u = (at - i)[:, None]
    a, b = q[i], q[np.minimum(i + 1, n - 1)]
    out = a + (b - a) * u
    qa, qb = a[:, 3:7], b[:, 3:7].copy()
    qb[(qa * qb).sum(-1) < 0] *= -1.0
    r = qa * (1 - u) + qb * u
    out[:, 3:7] = r / np.linalg.norm(r, axis=-1, keepdims=True)
    return out.astype(np.float32)


def variants(head, refs, clip, n, seed):
    """`n` variants of the clip named `clip`: a third pushed (a force on the root along the ground,
    `PUSH_N`, for `PUSH_FRAMES` frames at a random time in its first 10 s), a third on a slope
    (gravity tilted `SLOPE_DEG` in a random direction), a third at another speed (`SPEEDS`); the
    first is the clip as it is. Returns the head and the frames of the variants as clips of their
    own, and each one's kind, amount, gravity and push."""
    rng = np.random.default_rng(seed)
    c = head["names"].index(clip)
    s, L = head["starts"][c], head["lengths"][c]
    q = refs[s:s + L]
    names, starts, lengths, chunks, kinds = [], [], [], [], []
    g0 = np.array([0.0, -9.81, 0.0])
    total = 0
    for v in range(n):
        kind = "as it is" if v == 0 else ("push", "slope", "speed")[(v - 1) % 3]
        gravity, push, amount, frames = g0.copy(), (np.zeros(3), 0), 0.0, q
        if kind == "push":
            amount = rng.uniform(*PUSH_N)
            a = rng.uniform(0.0, 2.0 * np.pi)
            push = (amount * np.array([np.cos(a), 0.0, np.sin(a)]), int(rng.integers(30, min(300, L - 10))))
        elif kind == "slope":
            amount = rng.uniform(*SLOPE_DEG)
            a = rng.uniform(0.0, 2.0 * np.pi)
            t = np.radians(amount)
            gravity = 9.81 * np.array([np.sin(t) * np.cos(a), -np.cos(t), np.sin(t) * np.sin(a)])
        elif kind == "speed":
            amount = rng.uniform(*SPEEDS)
            frames = time_scaled(q, amount)
        names.append(f"{clip}~{v}")
        starts.append(total)
        lengths.append(frames.shape[0])
        chunks.append(frames)
        total += frames.shape[0]
        kinds.append({"kind": kind, "amount": float(amount), "gravity": gravity.tolist(),
                      "force": push[0].tolist(), "push_at": push[1]})
    h = dict(head)
    h.update({"names": names, "starts": starts, "lengths": lengths, "sets": ["test"] * n})
    return h, np.concatenate(chunks), kinds


TREMOR_HZ = 6.0
TREMOR_TARGET = 0.05
RANGE_TARGET = 0.0136


def tremor(q, rate):
    """The share of the power of the joints' speeds above `TREMOR_HZ` in the frames `q` (frames
    by joints), as the judge measures it (`engine::judge::movement::tremor`; here by the speeds'
    spectrum, there by a low-pass: the same to a few hundredths). 0 for under a second of frames."""
    if q.shape[0] < 32:
        return 0.0
    v = (q[1:] - q[:-1]) * rate
    v = v - v.mean(0, keepdim=True)
    p = torch.fft.rfft(v, dim=0).abs().pow(2).sum(1)
    f = torch.fft.rfftfreq(v.shape[0], 1.0 / rate).to(p.device)
    return float(p[f > TREMOR_HZ].sum() / p.sum().clamp(min=1e-12))


def summary(results, names=None):
    if not results:
        return {"clips": 0}
    followed = sum(r["followed"] for r in results)
    frames = sum(r["frames"] for r in results)
    mean_mm = 1000.0 * sum(r["mean_m"] * r["frames"] for r in results) / max(frames, 1)
    out = {"clips": len(results), "followed": followed / len(results), "mean_segment_mm": mean_mm}
    if "past_range" in results[0]:
        # The ball joints' distances past the body's ranges, summed, on average over the frames.
        out["past_range_rad"] = sum(r["past_range"] * r["frames"] for r in results) / max(frames, 1)
    if "tremor" in results[0]:
        # The phase's check, as far as the simulator knows it: the share of clips whose worst
        # joint stays in the judge's range target, and whose tremor is in its target.
        out["range_in_target"] = sum(r["worst_range"] <= RANGE_TARGET for r in results) / len(results)
        out["tremor"] = sum(r["tremor"] * r["frames"] for r in results) / max(frames, 1)
        out["tremor_in_target"] = sum(r["tremor"] <= TREMOR_TARGET for r in results) / len(results)
    if names is not None:
        groups = {}
        for r in results:
            nm = names[r["clip"]]
            src = nm.split("/", 1)[0] if "/" in nm else ""
            groups.setdefault(src, []).append(r)
        if len(groups) > 1:
            out["by_source"] = {k: summary(v) for k, v in sorted(groups.items())}
    return out


def evaluate(sim, agent, norm, most_frames=600):
    """The training clips and the test clips, each played from its start (`play`); the training
    clips' results weigh how often each starts a world after."""
    # One play of both sets where the worlds hold them (each play steps every world: two plays of
    # 600 frames cost as much as 40 updates).
    train, test = sim.train.tolist(), sim.test.tolist()
    if len(train) + len(test) <= sim.n:
        both = play(sim, agent, norm, train + test, most_frames)
        training, held = both[:len(train)], both[len(train):]
    else:
        training = play(sim, agent, norm, train, most_frames)
        held = play(sim, agent, norm, test, most_frames)
    sim.weigh(training)
    return {"training": summary(training, sim.names), "test": summary(held, sim.names)}


def filtered(sim, agent, norm, head, push=None):
    """Every clip played to its end by the net, as the simulator moved the body (each frame's
    qpos, up to a fall): the filtered capture (`WFLT`: a u32 length and a JSON head of each clip's
    name, set, whether it was followed and its frames; then zlib of the float32 qpos, clip after
    clip)."""
    results = play(sim, agent, norm, list(range(sim.starts.shape[0])), record=True, push=push)
    clips, chunks = [], []
    for r in results:
        clips.append({"name": head["names"][r["clip"]], "set": (head.get("sets") or ["training"] * 99999)[r["clip"]],
                      "followed": r["followed"], "frames": int(r["qpos"].shape[0]),
                      "length": int(sim.lengths[r["clip"]]), "mean_segment_mm": 1000.0 * r["mean_m"]})
        chunks.append(r["qpos"].astype("<f4"))
    body = zlib.compress(np.concatenate(chunks).tobytes(), 6)
    h = json.dumps({"clips": clips, "nq": head["nq"], "rate": head["rate"]}).encode()
    # Beside it (`WFLX`, `filtered_extra.bin`): each frame's action, servos' forces and contact
    # forces (`extra_of`, float16), the same frames clip after clip.
    xs = zlib.compress(np.concatenate([r["extra"].astype("<f2") for r in results]).tobytes(), 6)
    xh = json.dumps({"clips": [c["name"] for c in clips], "frames": [c["frames"] for c in clips],
                     "columns": {"action": sim.mjm.nu, "servo_force": sim.mjm.nu,
                                 "contact_force": 3 * (sim.active.numel() - 1), "momentum": 3},
                     "mass": float(sim.mjm.body_subtreemass[1]),
                     "segments": [sim.mjm.body(int(b)).name for b in sim.active[1:]],
                     "dtype": "float16", "rate": head["rate"]}).encode()
    extra = b"WFLX" + struct.pack("<I", len(xh)) + xh + xs
    return b"WFLT" + struct.pack("<I", len(h)) + h + body, results, extra


def play_variants(run, sim, agent, norm, head, kinds, t0):
    """Each variant of the clip followed by the tracker to its end (`variants`): the share followed
    with no fall, by kind, and the filtered capture of them (`filtered.bin`) for the judge."""
    dev = sim.dev
    sim.tilt(torch.tensor([k["gravity"] for k in kinds], device=dev))
    force = torch.tensor([k["force"] for k in kinds], device=dev, dtype=torch.float32)
    first = torch.tensor([k["push_at"] for k in kinds], device=dev)
    data, results, extra = filtered(sim, agent, norm, head, push=(force, first, PUSH_FRAMES))
    run.output("filtered_extra.bin", extra)
    by = {}
    for r, k in zip(results, kinds):
        f, n = by.get(k["kind"], (0, 0))
        by[k["kind"]] = (f + int(r["followed"]), n + 1)
        print(f"variant {head['names'][r['clip']]} ({k['kind']} {k['amount']:.2f}): followed {r['followed']}, "
              f"frames {r['frames']} of {int(sim.lengths[r['clip']])}, mean {1000.0 * r['mean_m']:.1f} mm", flush=True)
    out = {"followed": summary(results), "by_kind": {k: {"followed": f / n, "variants": n} for k, (f, n) in by.items()},
           "variants": kinds}
    print(f"the variants: {json.dumps(out['by_kind'])}", flush=True)
    run.output("filtered.bin", data)
    run.finish({"variants": out, "seconds": time.time() - t0})


def main():
    run = runlib.Run()
    load_packages()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="creature.xml")
    p.add_argument("--refs", default="refs.bin")
    p.add_argument("--worlds", type=int, default=4096)
    p.add_argument("--steps", type=int, default=1000, help="updates of the net")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--substeps", type=int, default=0, help="the simulator's steps to a frame (0: from the model)")
    p.add_argument("--hidden", default="1024,512,256")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--bounds", type=float, default=0.01, help="the weight of the bounds loss on the actions' means")
    p.add_argument("--log-std", type=float, default=-1.6,
                   help="starting action log_std (annealed to --log-std-end unless --learn-std)")
    p.add_argument("--log-std-end", type=float, default=-2.5,
                   help="final action log_std reached at 80%% of training steps")
    p.add_argument("--learn-std", action="store_true", help="learn action log_std instead of scheduling it")
    p.add_argument("--target-kl", type=float, default=0.002,
                   help="target mean per-dimension KL per PPO update (adaptive LR); 0 disables")
    p.add_argument("--start-share", type=float, default=0.25,
                   help="the share of worlds started at their clip's first frame (the evaluation plays each "
                        "clip from there)")
    p.add_argument("--smooth-time", type=float, default=0.0,
                   help="the weight of the actions' means' change from an observation to the next (Mysore et al. 2021)")
    p.add_argument("--smooth-place", type=float, default=0.0,
                   help="the weight of the actions' means' change for an observation `--smooth-noise` off")
    p.add_argument("--smooth-noise", type=float, default=0.05)
    p.add_argument("--accel-weight", type=float, default=0.0,
                   help="the reward times e^-(this times the mean square of the joints' acceleration's distance "
                        "from the clip's, rad a frame squared): 1000 halves it at 0.026")
    p.add_argument("--activation-ms", type=float, default=0.0,
                   help="the muscles' activation: the servos apply the net's command through a lag of this "
                        "time constant (ms; 0: at once)")
    p.add_argument("--read-last", action="store_true",
                   help="the net reads its last action (a net that reads only the body shakes it)")
    p.add_argument("--rate-weight", type=float, default=0.0,
                   help="the reward less this times the mean square of the action's change from the last")
    p.add_argument("--sole-weight", type=float, default=0.0,
                   help="this share of the reward for the soles where the clip's foot is planted (height, stillness)")
    p.add_argument("--best-from", type=int, default=0,
                   help="only evaluations from this update on can be the best tracker (a fine-tune toward "
                        "another reward: the weights it starts from are not the best)")
    p.add_argument("--range-weight", type=float, default=0.0,
                   help="the reward times e^-(this times the ball joints' distances past the body's ranges, rad)")
    p.add_argument("--fall-mean", type=float, default=0.5,
                   help="a fall: the segments this far from their places on average (m)")
    p.add_argument("--init", default="",
                   help="start from these weights (a run's checkpoint or a tracker.pt): the net and the "
                        "observations' scale, with a new optimiser")
    p.add_argument("--filtered", action="store_true", help="play every clip to its end at the end")
    p.add_argument("--variants", type=int, default=0,
                   help="with --weights: play this many variants of --clip (pushed, on a slope, at another "
                        "speed) and write them as a filtered capture")
    p.add_argument("--clip", default="100style/Neutral_FW", help="the clip of --variants")
    p.add_argument("--weights", default="",
                   help="a trained tracker (tracker.pt): train nothing, play every clip of the refs to its end")
    a = p.parse_args(run.args)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wp.init()
    torch.manual_seed(run.seed)
    gen = torch.Generator(device=dev).manual_seed(run.seed)
    head, refs = load_refs(a.refs)
    kinds = None
    if a.weights and a.variants:
        head, refs, kinds = variants(head, refs, a.clip, a.variants, run.seed)
        a.worlds = a.variants
    t0 = time.time()
    with wp.ScopedDevice("cuda:0" if dev == "cuda" else "cpu"):
        sim = Sim(a.model, head, refs, a.worlds, dev, a.substeps, gen)
        sim.start_share = a.start_share
        sim.fall_mean = a.fall_mean
        sim.range_weight = a.range_weight
        sim.read_last = a.read_last
        sim.activation_ms = a.activation_ms
        sim.accel_weight = a.accel_weight
        sim.rate_weight = a.rate_weight
        sim.sole_weight = a.sole_weight
        print(f"sim: {a.worlds} worlds, {len(head['names'])} clips, {refs.shape[0]} frames, "
              f"{sim.active.numel()} active bodies, {sim.mjm.nu} actuators, "
              f"set up in {time.time() - t0:.0f} s", flush=True)
        sim.reset(torch.ones(a.worlds, dtype=torch.bool, device=dev))
        n_obs = observe(sim).shape[1]
        n_act = sim.mjm.nu
        hidden = [int(x) for x in a.hidden.split(",")]
        log_std_arg = None if a.learn_std else a.log_std
        agent = make_agent(n_obs, n_act, hidden, log_std_arg).to(dev)
        opt = torch.optim.Adam(agent.parameters(), lr=a.lr)
        norm = Normal(n_obs, dev)
        vnorm = ValNorm(dev)
        print(f"net: n_obs={n_obs}, n_act={n_act}, hidden={hidden}, log_std={'learned' if a.learn_std else a.log_std}", flush=True)
        step, samples = 0, 0
        best = None
        state = run.resume()
        if a.weights:
            # Phase 5 step 12: the tracker as the slow reference of the solver's movements.
            w = torch.load(a.weights, map_location=dev, weights_only=False)
            agent.load_state_dict(w["agent"])
            norm.load(w["norm"], dev)
            if kinds is not None:
                return play_variants(run, sim, agent, norm, head, kinds, t0)
            if a.filtered:
                # The clips as the tracker moves the body: a filtered capture of any states.
                data, results, extra = filtered(sim, agent, norm, head)
                run.output("filtered.bin", data)
                run.output("filtered_extra.bin", extra)
            else:
                results = play(sim, agent, norm, list(range(sim.starts.shape[0])))
            for r in results:
                print(f"played {head['names'][r['clip']]}: followed {r['followed']}, frames {r['frames']} "
                      f"of {int(sim.lengths[r['clip']])}, mean {1000.0 * r['mean_m']:.1f} mm", flush=True)
            full = summary(results)
            print(f"played every clip: {json.dumps(full)}", flush=True)
            run.finish({"played": full, "seconds": time.time() - t0})
            return
        if state is None and a.init:
            try:
                w = torch.load(a.init, map_location=dev, weights_only=False)
            except Exception:
                import io
                import pickle
                import torch.storage as ts
                was = ts._load_from_bytes
                ts._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location=dev, weights_only=False)
                try:
                    with open(a.init, "rb") as fh:
                        w = pickle.load(fh)
                finally:
                    ts._load_from_bytes = was
            agent.load_state_dict(widened(agent.state_dict(), w["agent"], dev))
            norm.load(widened_norm(w["norm"], n_obs), dev)
            if "vnorm" in w:
                vnorm.load(w["vnorm"], dev)
            print(f"init: the net and the observations' scale of {a.init} (its update {w.get('step')})", flush=True)
            init_eval = evaluate(sim, agent, norm)
            print(f"evaluate at 0 (init): {json.dumps(init_eval)}", flush=True)
            if a.best_from == 0:
                best = {"followed": init_eval["training"].get("followed", 0.0), "step": 0, "evaluation": init_eval,
                        "agent": {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()},
                        "norm": {k: (v.cpu().clone() if hasattr(v, "cpu") else v) for k, v in norm.state().items()}}
            sim.reset(torch.ones(a.worlds, dtype=torch.bool, device=dev))
        if state is not None:
            agent.load_state_dict(state["agent"])
            opt.load_state_dict(state["opt"])
            norm.load(state["norm"], dev)
            if "vnorm" in state:
                vnorm.load(state["vnorm"], dev)
            step, samples = state["step"], state["samples"]
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
        guard = Guard()
        if state is not None and state.get("best") is not None:
            best = state["best"]
        rate_t, rate_s = time.time(), samples
        while step < a.steps:
            if not a.learn_std:
                frac = min(step / max(0.8 * a.steps, 1.0), 1.0)
                cur_ls = a.log_std + frac * (a.log_std_end - a.log_std)
                with torch.no_grad():
                    agent.log_std.fill_(cur_ls)
            ep_r, ep_fall, blown, skipped = 0.0, 0.0, 0, 0
            with torch.no_grad():
                for h in range(H):
                    o_raw = observe(sim)
                    norm.update(o_raw)
                    o = norm(o_raw)
                    dist = agent.dist(o)
                    act = dist.sample()
                    obs_buf[h], act_buf[h] = o, act
                    logp_buf[h] = dist.log_prob(act).sum(-1)
                    val_buf[h] = vnorm.denorm(agent.v(o).squeeze(-1))
                    act_c = act.clamp(-1, 1)
                    sim.step(sim.command(act_c))
                    r, fell, _ = reward(sim, act_c)
                    sim.last_act = act_c
                    blown += int(exploded(sim).sum())
                    ended = sim.frame >= (sim.lengths[sim.clip] - 2)
                    # At a clip's end the return goes on (its value); at a fall it stops.
                    trunc_val[h] = 0.0
                    if ended.any():
                        trunc_val[h] = torch.where(ended & ~fell, vnorm.denorm(agent.v(norm(observe(sim))).squeeze(-1)), 0.0)
                    rew_buf[h] = r
                    done_buf[h] = (fell | ended).float()
                    ep_r += float(r.mean())
                    ep_fall += float(fell.float().mean())
                    sim.reset(fell | ended)
                val_buf[H] = vnorm.denorm(agent.v(norm(observe(sim))).squeeze(-1))
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
            with torch.no_grad():
                past = float((agent.pi(b_obs[:4096]).abs() > 1.0).float().mean())
            smooth = None
            if a.smooth_time > 0.0 or a.smooth_place > 0.0:
                with torch.no_grad():
                    nxt = torch.cat([obs_buf[1:], norm(observe(sim))[None]], 0).reshape(H * N, -1)
                smooth = (nxt, (1.0 - done_buf).reshape(-1), a.smooth_time, a.smooth_place, a.smooth_noise)
            kl, clipped, vloss, epochs = ppo(agent, opt, b_obs, b_act, b_logp, b_adv, b_ret, gen, a.target_kl,
                                             a.bounds, vnorm=vnorm, smooth=smooth)
            skipped += kl[1]
            step += 1
            samples += H * N
            now = time.time()
            sps = (samples - rate_s) / max(now - rate_t, 1e-6)
            run.progress(step, a.steps, f"reward {ep_r / H:.3f} falls {ep_fall / H:.4f} "
                                        f"samples {samples} at {sps:.0f}/s")
            if step % 10 == 0:
                print(f"update {step}: reward {ep_r / H:.3f} falls/step {ep_fall / H:.4f} "
                      f"samples {samples} ({sps:.0f} a second); worlds blown up {blown}, "
                      f"minibatches skipped {skipped}; kl {kl[0]:.4f} clipped {clipped:.3f} "
                      f"value loss {vloss:.3f} epochs {epochs} log std {float(agent.log_std.mean()):.3f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} means past 1 {past:.4f}", flush=True)
            # A collapse (on 2026-11-26 three runs on the human fell from a reward of 0.77 to 0.42
            # in 30 updates at update 990, with no world blown up): the weights of the last good
            # update come back with half the rate of learning.
            guard.add(ep_r / H)
            if guard.collapsed():
                print(f"update {step}: collapse (reward {guard.recent():.3f} against {guard.best:.3f}): "
                      f"back to update {guard.good['step']}, rate of learning halved", flush=True)
                guard.restore(agent, opt, norm, vnorm)
                step = guard.good["step"]
            elif step % 25 == 0:
                guard.keep(step, agent, opt, norm, vnorm)
            if a.eval_every and step % a.eval_every == 0:
                last_eval = evaluate(sim, agent, norm)
                print(f"evaluate at {step}: {json.dumps(last_eval)}", flush=True)
                followed = last_eval["training"].get("followed", 0.0)
                if step >= a.best_from and (best is None or followed > best["followed"]):
                    best = {"followed": followed, "step": step, "evaluation": last_eval,
                            "agent": {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()},
                            "norm": {k: (v.cpu().clone() if hasattr(v, "cpu") else v) for k, v in norm.state().items()}}
                sim.reset(torch.ones(a.worlds, dtype=torch.bool, device=dev))
                # The clips' weights changed, and with them the reward a good net gets.
                guard.rebase()
            if run.due() or step == a.steps:
                run.checkpoint(step, {
                    "agent": {k: v.cpu() for k, v in agent.state_dict().items()},
                    "opt": opt.state_dict(), "norm": norm.state(), "vnorm": vnorm.state(), "step": step,
                    "samples": samples, "gen": gen.get_state().cpu(),
                    "torch_rng": torch.get_rng_state(), "weights": sim.weights.cpu(), "best": best,
                    "n_obs": n_obs, "n_act": n_act,
                    "hidden": hidden, "args": vars(a),
                })
            if run.should_stop():
                run.abandon()
                return
        final = evaluate(sim, agent, norm)
        print(f"evaluate at the end: {json.dumps(final)}", flush=True)
        if best is not None and best["followed"] > final["training"].get("followed", 0.0):
            # The best tracker an evaluation saw writes the filtered capture.
            print(f"the best tracker is update {best['step']}'s: followed {best['followed']:.3f}", flush=True)
            agent.load_state_dict({k: v.to(dev) for k, v in best["agent"].items()})
            norm.load(best["norm"], dev)
            final = best["evaluation"]
        import io
        buf = io.BytesIO()
        torch.save({"agent": {k: v.cpu() for k, v in agent.state_dict().items()},
                    "norm": norm.state(), "vnorm": vnorm.state(), "n_obs": n_obs, "n_act": n_act,
                    "hidden": hidden, "seed": run.seed, "args": vars(a), "samples": samples}, buf)
        run.output("tracker.pt", buf.getvalue())
        full = {}
        if a.filtered:
            data, results, extra = filtered(sim, agent, norm, head)
            run.output("filtered.bin", data)
            run.output("filtered_extra.bin", extra)
            full = {"training": summary([r for r in results if r["clip"] in set(sim.train.tolist())], sim.names),
                    "test": summary([r for r in results if r["clip"] in set(sim.test.tolist())], sim.names)}
            print(f"the filtered capture, every clip to its end: {json.dumps(full)}", flush=True)
        run.finish({"step": step, "samples": samples, "evaluation": final, "full": full,
                    "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
