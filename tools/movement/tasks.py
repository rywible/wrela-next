"""The tasks of the net that moves under the prior of style (docs/plans/bodies-and-movement.md,
phase 5 steps 14 and 15): each gives the net what it must know of its goal, the reward for it each
frame and its measures. `tools/movement/amp.py --task NAME` trains a net on one.

- `walk`: the solver's wish, a speed and a direction on the ground, changed at random every two
  to five seconds (step 14).
- `cast`: the right palm at a point at a tick, the timed goal of `engine::body::motion::reaching`
  as a skill with no capture: a point within the arm's reach in front of the body, 0.4 to 1.5 s
  ahead; then a rest, and the next.
- `parry`: a spell thrown at the body (a point that flies straight, 8 to 15 m/s, from 6 to 10 m
  away), met by either palm at the place and the tick it reaches the body's front.
- `dodge`: a spell thrown at the body's trunk or head, avoided: no part of the body within its
  radius as it passes, the body held near where it was.

A palm is its hand's centre of mass. Every place and velocity the net reads is in the frame of the
root's heading (y up, z ahead, x to the left), as the tracker's are. A goal's tick is a frame of
the simulator (30 a second): each task measures, at its goal's tick, the palm's distance from its
point (`cast`, `parry`) or whether the body was hit (`dodge`).
"""

import math

import torch

import track

T = track

# The goals' ranges, in the heading frame (m) and in frames.
REACH_SIDE = (-0.45, 0.45)
REACH_AHEAD = (0.25, 0.65)
REACH_HEIGHT = (0.8, 1.7)
CAST_TICKS = (12, 45)
REST_TICKS = (10, 40)
SPELL_SPEED = (8.0, 15.0)
SPELL_FROM = (6.0, 10.0)
SPELL_RADIUS = 0.12
# The reward of a timed goal: dense toward the point (sharper as the tick nears), and at the tick
# a bonus by the distance in units of this (m).
AT_TICK = 0.04


def uniform(n, lo_hi, device, gen):
    lo, hi = lo_hi
    return lo + (hi - lo) * torch.rand(n, device=device, generator=gen)


def heading_of(sim):
    return T.heading(sim.xquat[:, 1])


def in_heading(sim, v):
    """World vectors `v` (N, 3) in the root's heading frame."""
    return T.quat_rotate(T.quat_conj(heading_of(sim)), v)


class Walk:
    """The solver's wish: a velocity on the ground."""

    name = "walk"
    n_obs = 2

    def __init__(self, sim, gen, fastest=2.0):
        self.sim, self.gen, self.fastest = sim, gen, fastest
        n, dev = sim.n, sim.dev
        self.wish = self.new(n)
        self.left = torch.randint(60, 150, (n,), device=dev, generator=gen)
        self.since = torch.zeros(n, dtype=torch.long, device=dev)
        self.errors = []
        self.counted = torch.ones(n, dtype=torch.bool, device=dev)

    def new(self, n):
        a = torch.rand(n, device=self.sim.dev, generator=self.gen) * 2.0 * math.pi
        s = torch.rand(n, device=self.sim.dev, generator=self.gen) * self.fastest
        return torch.stack([s * torch.cos(a), s * torch.sin(a)], -1)

    def observe(self):
        w = torch.zeros(self.sim.n, 3, device=self.sim.dev)
        w[:, 0], w[:, 2] = self.wish[:, 0], self.wish[:, 1]
        return in_heading(self.sim, w)[:, [0, 2]]

    def reward(self, hist_q, rate):
        v = (hist_q[:, -1, [0, 2]] - hist_q[:, -2, [0, 2]]) * rate
        e = ((v - self.wish) ** 2).sum(-1)
        # The error a second after the wish changed, on.
        settled = (self.since >= 30) & self.counted
        if settled.any():
            self.errors.append(e[settled].sqrt().detach().cpu())
        return torch.exp(-2.5 * e)

    def after(self, again):
        self.left -= 1
        self.since += 1
        change = (self.left <= 0) | again
        self.since[change] = 0
        k = int(change.sum())
        if k:
            self.wish[change] = self.new(k)
            self.left[change] = torch.randint(60, 150, (k,), device=self.sim.dev, generator=self.gen)

    def measures(self):
        if not self.errors:
            return {}
        e = torch.cat(self.errors)
        self.errors = []
        return {"frames": int(e.numel()), "median_speed_error": float(e.median()),
                "p95_speed_error": float(e.quantile(0.95))}


class Timed:
    """A palm at a point at a tick (`cast`: the right palm, a still point; `parry`: either palm,
    the point a thrown spell's place where it reaches the body's front)."""

    n_obs = 14

    def __init__(self, sim, gen, palms, thrown):
        self.sim, self.gen, self.thrown = sim, gen, thrown
        self.palms = torch.tensor([sim.mjm.body(f"s{s}").id for s in palms], device=sim.dev)
        n, dev = sim.n, sim.dev
        self.point = torch.zeros(n, 3, device=dev)
        self.spell = torch.zeros(n, 3, device=dev)
        self.velocity = torch.zeros(n, 3, device=dev)
        self.left = torch.zeros(n, dtype=torch.long, device=dev)
        self.rest = torch.randint(*REST_TICKS, (n,), device=dev, generator=gen)
        self.active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.errors = []
        self.counted = torch.ones(n, dtype=torch.bool, device=dev)

    def palm_places(self):
        return self.sim.xipos[:, self.palms]

    def start(self, which):
        """New goals for the worlds `which`: a point in reach of the body as it stands now."""
        idx = which.nonzero().flatten()
        k = idx.numel()
        if k == 0:
            return
        sim, dev, gen = self.sim, self.sim.dev, self.gen
        local = torch.stack([uniform(k, REACH_SIDE, dev, gen), torch.zeros(k, device=dev),
                             uniform(k, REACH_AHEAD, dev, gen)], -1)
        h = heading_of(sim)[idx]
        p = sim.xpos[idx, 1] + T.quat_rotate(h, local)
        p[:, 1] = uniform(k, REACH_HEIGHT, dev, gen)
        self.point[idx] = p
        if self.thrown:
            # From ahead of the body (within 60 degrees of its heading), at a speed: the tick is
            # when it reaches the point.
            turn = uniform(k, (-math.pi / 3, math.pi / 3), dev, gen)
            far = uniform(k, SPELL_FROM, dev, gen)
            up = uniform(k, (-0.5, 1.0), dev, gen)
            d = torch.stack([torch.sin(turn) * far, up, torch.cos(turn) * far], -1)
            src = p + T.quat_rotate(h, d)
            speed = uniform(k, SPELL_SPEED, dev, gen)
            way = p - src
            dist = way.norm(dim=-1)
            self.velocity[idx] = way / dist[:, None] * speed[:, None]
            ticks = (dist / speed * sim.rate).round().long().clamp(min=CAST_TICKS[0])
            # The spell starts where it must be to reach the point at its tick.
            self.spell[idx] = p - self.velocity[idx] * (ticks.float() / sim.rate)[:, None]
            self.left[idx] = ticks
        else:
            self.left[idx] = torch.randint(*CAST_TICKS, (k,), device=dev, generator=gen)
        self.active[idx] = True

    def nearest(self):
        """Each world's distance from its point of its nearest palm."""
        d = (self.palm_places() - self.point[:, None]).norm(dim=-1)
        return d.min(-1).values, d.argmin(-1)

    def observe(self):
        sim = self.sim
        root = sim.xpos[:, 1]
        _, which = self.nearest()
        palm = self.palm_places()[torch.arange(sim.n, device=sim.dev), which]
        act = self.active.float()[:, None]
        parts = [in_heading(sim, self.point - palm) * act, in_heading(sim, self.point - root) * act,
                 (self.left.float() / sim.rate)[:, None] * act, act]
        if self.thrown:
            parts += [in_heading(sim, self.spell - root) * act, in_heading(sim, self.velocity) * 0.1 * act]
        else:
            parts += [torch.zeros(sim.n, 6, device=sim.dev)]
        return torch.cat(parts, -1)

    def reward(self, hist_q, rate):
        d, _ = self.nearest()
        soon = torch.exp(-self.left.clamp(min=0).float() / 8.0)
        r = torch.where(self.active, torch.exp(-4.0 * d * d) * (0.2 + 0.8 * soon),
                        torch.full_like(d, 0.5))
        now = self.active & (self.left == 0)
        r = r + torch.where(now, 5.0 * torch.exp(-(d / AT_TICK) ** 2), torch.zeros_like(r))
        if (now & self.counted).any():
            self.errors.append(d[now & self.counted].detach().cpu())
        return r

    def after(self, again):
        sim = self.sim
        self.spell += self.velocity / sim.rate
        due = self.active & (self.left <= 0)
        self.active &= ~due
        self.left -= self.active.long()
        self.rest[due] = torch.randint(*REST_TICKS, (int(due.sum()),), device=sim.dev, generator=self.gen)
        self.rest -= (~self.active).long()
        # A world started again rests first, as one at the end of a goal does.
        self.active &= ~again
        self.rest[again] = torch.randint(*REST_TICKS, (int(again.sum()),), device=sim.dev, generator=self.gen)
        self.start(~self.active & (self.rest <= 0))

    def measures(self):
        if not self.errors:
            return {}
        e = torch.cat(self.errors)
        self.errors = []
        return {"goals": int(e.numel()), "median_mm": 1000.0 * float(e.median()),
                "p95_mm": 1000.0 * float(e.quantile(0.95)), "within_20mm": float((e < 0.02).float().mean())}


class Dodge:
    """A spell thrown at the trunk or the head, avoided."""

    name = "dodge"
    n_obs = 10

    def __init__(self, sim, gen, aims):
        self.sim, self.gen = sim, gen
        self.aims = torch.tensor([sim.mjm.body(f"s{s}").id for s in aims], device=sim.dev)
        m = sim.mjm
        # The body's capsules (a centre, an axis along the geom's z, a half length and a radius).
        caps = [g for g in range(m.ngeom) if m.geom_type[g] == 3 and m.geom_bodyid[g] > 0]
        self.caps = torch.tensor(caps, device=sim.dev)
        self.cap_size = torch.tensor(m.geom_size[caps, :2], device=sim.dev, dtype=torch.float32)
        n, dev = sim.n, sim.dev
        self.spell = torch.zeros(n, 3, device=dev)
        self.velocity = torch.zeros(n, 3, device=dev)
        self.home = torch.zeros(n, 3, device=dev)
        self.left = torch.zeros(n, dtype=torch.long, device=dev)
        self.rest = torch.randint(*REST_TICKS, (n,), device=dev, generator=gen)
        self.active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hit = torch.zeros(n, dtype=torch.bool, device=dev)
        self.outcomes = []
        self.counted = torch.ones(n, dtype=torch.bool, device=dev)

    def clearance(self):
        """Each world's least distance between the spell's surface and a capsule's."""
        sim = self.sim
        c = sim.geom_xpos[:, self.caps]
        axis = sim.geom_xmat[:, self.caps].reshape(sim.n, -1, 3, 3)[..., :, 2]
        half, radius = self.cap_size[:, 1], self.cap_size[:, 0]
        rel = self.spell[:, None] - c
        t = (rel * axis).sum(-1).clamp(-half, half)
        d = (rel - axis * t[..., None]).norm(dim=-1) - radius
        return d.min(-1).values - SPELL_RADIUS

    def start(self, which):
        idx = which.nonzero().flatten()
        k = idx.numel()
        if k == 0:
            return
        sim, dev, gen = self.sim, self.sim.dev, self.gen
        pick = torch.randint(0, self.aims.numel(), (k,), device=dev, generator=gen)
        target = sim.xipos[idx, self.aims[pick]]
        h = heading_of(sim)[idx]
        turn = uniform(k, (-math.pi / 3, math.pi / 3), dev, gen)
        far = uniform(k, SPELL_FROM, dev, gen)
        up = uniform(k, (-0.5, 1.0), dev, gen)
        src = target + T.quat_rotate(h, torch.stack([torch.sin(turn) * far, up, torch.cos(turn) * far], -1))
        speed = uniform(k, SPELL_SPEED, dev, gen)
        way = target - src
        dist = way.norm(dim=-1)
        self.velocity[idx] = way / dist[:, None] * speed[:, None]
        ticks = (dist / speed * sim.rate).round().long().clamp(min=CAST_TICKS[0])
        self.spell[idx] = target - self.velocity[idx] * (ticks.float() / sim.rate)[:, None]
        self.left[idx] = ticks
        self.home[idx] = sim.xpos[idx, 1]
        self.hit[idx] = False
        self.active[idx] = True

    def observe(self):
        sim = self.sim
        root = sim.xpos[:, 1]
        act = self.active.float()[:, None]
        away = in_heading(sim, root - self.home)[:, [0, 2]]
        return torch.cat([in_heading(sim, self.spell - root) * act, in_heading(sim, self.velocity) * 0.1 * act,
                          (self.left.float() / sim.rate)[:, None] * act, act, away], -1)

    def reward(self, hist_q, rate):
        sim = self.sim
        clear = self.clearance()
        self.hit |= self.active & (clear < 0.0)
        away = (sim.xpos[:, 1] - self.home)[:, [0, 2]].norm(dim=-1)
        near = torch.exp(-away * away)
        r = torch.where(self.active, 0.5 * near + 0.5 * (clear.clamp(0.0, 0.3) / 0.3), 0.5 * near + 0.5)
        r = torch.where(self.active & self.hit, torch.zeros_like(r), r)
        # It has passed (three ticks after the one it was thrown to reach), or it hit.
        passed = self.active & ~self.hit & (self.left < -int(0.1 * sim.rate))
        r = r + torch.where(passed, torch.full_like(r, 5.0), torch.zeros_like(r))
        ends = passed | (self.active & self.hit)
        if (ends & self.counted).any():
            self.outcomes.append((~self.hit[ends & self.counted]).float().cpu())
        self._ends = ends
        return r

    def after(self, again):
        sim = self.sim
        self.spell += self.velocity / sim.rate
        done = self._ends
        self.active &= ~done
        self.left -= self.active.long()
        self.rest[done] = torch.randint(*REST_TICKS, (int(done.sum()),), device=sim.dev, generator=self.gen)
        self.rest -= (~self.active).long()
        self.active &= ~again
        self.rest[again] = torch.randint(*REST_TICKS, (int(again.sum()),), device=sim.dev, generator=self.gen)
        self.start(~self.active & (self.rest <= 0))

    def measures(self):
        if not self.outcomes:
            return {}
        o = torch.cat(self.outcomes)
        self.outcomes = []
        return {"spells": int(o.numel()), "avoided": float(o.mean())}


# The human's segments (game/world/human.wrela): the right hand, the left, and the trunk and head.
HAND_LEFT, HAND_RIGHT = 19, 24
TRUNK_AND_HEAD = (0, 2, 3, 4, 6)


def make(name, sim, gen, fastest=2.0):
    if name == "walk":
        return Walk(sim, gen, fastest)
    if name == "cast":
        t = Timed(sim, gen, [HAND_RIGHT], thrown=False)
        t.name = "cast"
        return t
    if name == "parry":
        t = Timed(sim, gen, [HAND_LEFT, HAND_RIGHT], thrown=True)
        t.name = "parry"
        return t
    if name == "dodge":
        return Dodge(sim, gen, TRUNK_AND_HEAD)
    raise ValueError(f"no task {name}")
