"""The numbers of `engine::body::motion` (lib/engine/body/motion.wrela) in PyTorch, for a batch of
poses at a time: `state`, `step`, `after` and `goal`, as the engine computes them, so a net
trained here reads and writes what the engine gives it. `tools/movement/solve.py check` holds
these to the engine's own numbers (`tools/movement/solver -- features`).

A pose is a tensor [..., 7 + D]: the root's place (x, y, z), its turn (a quaternion x, y, z, w)
and the D numbers of the joints. The body (`tools/movement/solver -- body` writes it) gives the
root's turn at rest and each joint's kind, first number and range.
"""
import json
import math

import torch

STATE_ROOT = 14
STEP_ROOT = 10
GOAL = 4


def qmul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + bw * ax + ay * bz - az * by,
                        aw * by + bw * ay + az * bx - ax * bz,
                        aw * bz + bw * az + ax * by - ay * bx,
                        aw * bw - ax * bx - ay * by - az * bz], -1)


def qinv(q):
    return torch.cat([-q[..., :3], q[..., 3:]], -1)


def qrot(q, p):
    v, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(v, p, dim=-1)
    return p + w * t + torch.cross(v, t, dim=-1)


def yaw(a):
    h = 0.5 * a
    z = torch.zeros_like(a)
    return torch.stack([z, torch.sin(h), z, torch.cos(h)], -1)


def vector_of_turn(q):
    q = torch.where(q[..., 3:] < 0, -q, q)
    s = q[..., :3].norm(dim=-1, keepdim=True)
    small = s < 1e-8
    k = torch.where(small, torch.full_like(s, 2.0), 2.0 * torch.atan2(s, q[..., 3:]) / s.clamp(min=1e-12))
    return q[..., :3] * k


def _about(k, angle):
    h = 0.5 * angle
    z = torch.zeros_like(angle)
    parts = [z, z, z, torch.cos(h)]
    parts[k] = torch.sin(h)
    return torch.stack(parts, -1)


def turn_of_vector(r):
    a = r.norm(dim=-1, keepdim=True)
    small = a < 1e-8
    k = torch.where(small, torch.full_like(a, 0.5), torch.sin(0.5 * a) / a.clamp(min=1e-12))
    q = torch.cat([r * k, torch.cos(0.5 * a)], -1)
    return torch.where(small, q / q.norm(dim=-1, keepdim=True), q)


def from_mat3(x, y, z):
    """The quaternion of the frame whose columns are x, y, z (Shepperd's method, as std::quat)."""
    trace = x[..., 0] + y[..., 1] + z[..., 2]
    s0 = 2.0 * torch.sqrt((trace + 1.0).clamp(min=1e-12))
    q0 = torch.stack([(y[..., 2] - z[..., 1]) / s0, (z[..., 0] - x[..., 2]) / s0,
                      (x[..., 1] - y[..., 0]) / s0, 0.25 * s0], -1)
    s1 = 2.0 * torch.sqrt((1.0 + x[..., 0] - y[..., 1] - z[..., 2]).clamp(min=1e-12))
    q1 = torch.stack([0.25 * s1, (y[..., 0] + x[..., 1]) / s1, (z[..., 0] + x[..., 2]) / s1,
                      (y[..., 2] - z[..., 1]) / s1], -1)
    s2 = 2.0 * torch.sqrt((1.0 + y[..., 1] - x[..., 0] - z[..., 2]).clamp(min=1e-12))
    q2 = torch.stack([(y[..., 0] + x[..., 1]) / s2, 0.25 * s2, (z[..., 1] + y[..., 2]) / s2,
                      (z[..., 0] - x[..., 2]) / s2], -1)
    s3 = 2.0 * torch.sqrt((1.0 + z[..., 2] - x[..., 0] - y[..., 1]).clamp(min=1e-12))
    q3 = torch.stack([(z[..., 0] + x[..., 2]) / s3, (z[..., 1] + y[..., 2]) / s3, 0.25 * s3,
                      (x[..., 1] - y[..., 0]) / s3], -1)
    c0 = (trace > 0)[..., None]
    c1 = ((x[..., 0] > y[..., 1]) & (x[..., 0] > z[..., 2]))[..., None]
    c2 = (y[..., 1] > z[..., 2])[..., None]
    q = torch.where(c0, q0, torch.where(c1, q1, torch.where(c2, q2, q3)))
    return q / q.norm(dim=-1, keepdim=True)


def wrapped(a):
    return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi


class Body:
    def __init__(self, path, device="cpu"):
        b = json.load(open(path))
        self.freedom = b["freedom"]
        self.rest = torch.tensor(b["rest"], dtype=torch.float32, device=device)
        self.joints = b["joints"]
        lo = torch.zeros(self.freedom, device=device)
        hi = torch.zeros(self.freedom, device=device)
        self.balls = []
        for j in self.joints:
            n = {"fixed": 0, "hinge": 1, "universal": 2, "ball": 3}[j["kind"]]
            for k in range(n):
                lo[j["first"] + k] = j["lo"][k]
                hi[j["first"] + k] = j["hi"][k]
            if j["kind"] == "ball":
                self.balls.append(j["first"])
        self.lo, self.hi = lo, hi
        self.balls = torch.tensor(self.balls, dtype=torch.long, device=device)
        self.parent = [j.get("parent", -1) for j in self.joints]
        if "at_q" in self.joints[0]:
            self.at_q = torch.tensor([j["at_q"] for j in self.joints], dtype=torch.float32, device=device)
            self.at_t = torch.tensor([j["at_t"] for j in self.joints], dtype=torch.float32, device=device)
        self.parts = b.get("parts", [])
        if self.parts:
            self.part_a_on = torch.tensor([c["a_on"] for c in self.parts], device=device)
            self.part_b_on = torch.tensor([c["b_on"] for c in self.parts], device=device)
            self.part_a = torch.tensor([c["a"] for c in self.parts], dtype=torch.float32, device=device)
            self.part_b = torch.tensor([c["b"] for c in self.parts], dtype=torch.float32, device=device)
            self.part_ra = torch.tensor([c["ra"] for c in self.parts], dtype=torch.float32, device=device)
            self.part_rb = torch.tensor([c["rb"] for c in self.parts], dtype=torch.float32, device=device)

    def turn(self, s, q):
        """Segment `s`'s joint's turn for the pose numbers `q` (`joint::turn`)."""
        j = self.joints[s]
        f = j["first"]
        if j["kind"] == "hinge":
            return _about(0, q[..., f])
        if j["kind"] == "universal":
            return qmul(_about(0, q[..., f]), _about(1, q[..., f + 1]))
        if j["kind"] == "ball":
            r = torch.stack([q[..., f], torch.zeros_like(q[..., f]), q[..., f + 1]], -1)
            return qmul(turn_of_vector(r), _about(1, q[..., f + 2]))
        shape = q.shape[:-1] + (4,)
        out = torch.zeros(shape, dtype=q.dtype, device=q.device)
        out[..., 3] = 1.0
        return out

    def places(self, pose):
        """Each segment's frame (`Body::places`): its place [..., S, 3] and turn [..., S, 4]."""
        q = pose[..., 7:]
        ts, qs = [pose[..., :3]], [pose[..., 3:7]]
        for s in range(1, len(self.joints)):
            p = self.parent[s]
            at_q = self.at_q[s].expand(q.shape[:-1] + (4,))
            fq = qmul(at_q, self.turn(s, q))
            ts.append(ts[p] + qrot(qs[p], self.at_t[s].expand(q.shape[:-1] + (3,))))
            qs.append(qmul(qs[p], fq))
        return torch.stack(ts, -2), torch.stack(qs, -2)

    def places_of(self, pose, segments):
        """The frames of `segments` and of each segment they hang from (dict: segment -> (place,
        turn)): the kinematics of a part of the body only."""
        need = set()
        for s in segments:
            while s >= 0 and s not in need:
                need.add(s)
                s = self.parent[s]
        q = pose[..., 7:]
        out = {0: (pose[..., :3], pose[..., 3:7])}
        for s in sorted(need):
            if s == 0:
                continue
            p = self.parent[s]
            pt, pq = out[p]
            at_q = self.at_q[s].expand(q.shape[:-1] + (4,))
            fq = qmul(at_q, self.turn(s, q))
            out[s] = (pt + qrot(pq, self.at_t[s].expand(q.shape[:-1] + (3,))), qmul(pq, fq))
        return out

    def ends_of(self, pose, parts):
        """The two ends in the world of each of `parts` (indices) [..., len, 3], and their radii."""
        segs = set(int(self.part_a_on[i]) for i in parts) | set(int(self.part_b_on[i]) for i in parts)
        fr = self.places_of(pose, segs)
        a, b = [], []
        for i in parts:
            ta, qa = fr[int(self.part_a_on[i])]
            tb, qb = fr[int(self.part_b_on[i])]
            a.append(ta + qrot(qa, self.part_a[i].expand(ta.shape)))
            b.append(tb + qrot(qb, self.part_b[i].expand(tb.shape)))
        idx = torch.tensor(parts, device=self.part_ra.device)
        return torch.stack(a, -2), torch.stack(b, -2), self.part_ra[idx], self.part_rb[idx]

    def low_parts(self, below=0.006):
        """The parts that touch the floor at rest (their lowest point within `below` of the
        body's): the feet's, which stand on it."""
        rest = torch.zeros(7 + self.freedom, device=self.part_ra.device)
        rest[3:7] = self.rest
        a, b = self.part_ends(rest)
        low = torch.minimum(a[..., 1] - self.part_ra, b[..., 1] - self.part_rb)
        return [i for i in range(len(self.parts)) if low[i] < low.min() + below]

    def part_ends(self, pose):
        """Each part's two ends in the world [..., P, 3] each, and their radii [P]."""
        t, q = self.places(pose)
        a = t[..., self.part_a_on, :] + qrot(q[..., self.part_a_on, :], self.part_a.expand(t.shape[:-2] + self.part_a.shape))
        b = t[..., self.part_b_on, :] + qrot(q[..., self.part_b_on, :], self.part_b.expand(t.shape[:-2] + self.part_b.shape))
        return a, b

    def heading(self, q):
        f = qrot(qmul(q, qinv(self.rest).expand_as(q)), _z(q))
        return torch.atan2(f[..., 0], f[..., 2])

    def tilt(self, q):
        t = qmul(qmul(yaw(-self.heading(q)), q), qinv(self.rest).expand_as(q))
        return torch.cat([qrot(t, _axis(q, 0)), qrot(t, _axis(q, 1))], -1)

    def clamped(self, q):
        """Each joint's numbers put into its range (`Joint::clamped`)."""
        out = torch.maximum(torch.minimum(q, self.hi), self.lo)
        if self.balls.numel():
            b = self.balls
            sx, sy = q[..., b], q[..., b + 1]
            ax = torch.where(sx >= 0, self.hi[b], -self.lo[b]).clamp(min=1e-4)
            ay = torch.where(sy >= 0, self.hi[b + 1], -self.lo[b + 1]).clamp(min=1e-4)
            k = torch.sqrt((sx / ax) ** 2 + (sy / ay) ** 2)
            f = torch.where(k > 1.0, 1.0 / k.clamp(min=1e-12), torch.ones_like(k))
            out = out.clone()
            out[..., b] = sx * f
            out[..., b + 1] = sy * f
        return out

    def state(self, before, now, rate):
        q0, q1 = before[..., 3:7], now[..., 3:7]
        h = self.heading(q1)
        into = yaw(-h)
        v = qrot(into, now[..., :3] - before[..., :3]) * rate
        dh = wrapped(h - self.heading(q0))[..., None] * rate
        w = qrot(into, vector_of_turn(qmul(q1, qinv(q0)))) * rate
        return torch.cat([now[..., 1:2], self.tilt(q1), v, dh, w, now[..., 7:],
                          (now[..., 7:] - before[..., 7:]) * rate], -1)

    def step(self, now, nxt):
        h = self.heading(now[..., 3:7])
        d = qrot(yaw(-h), nxt[..., :3] - now[..., :3])
        dh = wrapped(self.heading(nxt[..., 3:7]) - h)[..., None]
        return torch.cat([d[..., 0:1], d[..., 2:3], nxt[..., 1:2], dh, self.tilt(nxt[..., 3:7]),
                          nxt[..., 7:] - now[..., 7:]], -1)

    def after(self, now, s, clamp=True):
        h = self.heading(now[..., 3:7])
        z = torch.zeros_like(s[..., 0])
        d = qrot(yaw(h), torch.stack([s[..., 0], z, s[..., 1]], -1))
        t = torch.stack([now[..., 0] + d[..., 0], s[..., 2], now[..., 2] + d[..., 2]], -1)
        a = s[..., 4:7] / s[..., 4:7].norm(dim=-1, keepdim=True).clamp(min=1e-12)
        b0 = s[..., 7:10]
        b = b0 - a * (a * b0).sum(-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        tq = from_mat3(a, b, torch.cross(a, b, dim=-1))
        rq = qmul(qmul(yaw(h + s[..., 3]), tq), self.rest.expand_as(tq))
        q = now[..., 7:] + s[..., STEP_ROOT:]
        if clamp:
            q = self.clamped(q)
        return torch.cat([t, rq, q], -1)

    def goal(self, now, ahead, seconds):
        h = self.heading(now[..., 3:7])
        d = qrot(yaw(-h), ahead[..., :3] - now[..., :3]) / seconds[..., None]
        turn = self.heading(ahead[..., 3:7]) - h
        return torch.stack([d[..., 0], d[..., 2], torch.sin(turn), torch.cos(turn)], -1)


def _z(q):
    z = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype, device=q.device)
    z[..., 2] = 1.0
    return z


def _axis(q, k):
    z = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype, device=q.device)
    z[..., k] = 1.0
    return z
