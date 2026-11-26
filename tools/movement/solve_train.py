"""The solver's training run (see solve.py's head). A job of tools/movement/runlib.py.

Writes `solver.wnet` (the decoder, as `engine::learned::net` reads it, its inputs' normalisation
and its outputs' scale folded in) and `solver.json` (its inputs' layout and the run's numbers).
"""
import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion  # noqa: E402
import runlib  # noqa: E402
import wnet  # noqa: E402

CLIP = 8.0


def mlp(n_in, n_out, hidden):
    layers, last = [], n_in
    for h in hidden:
        layers += [torch.nn.Linear(last, h), torch.nn.ELU()]
        last = h
    layers.append(torch.nn.Linear(last, n_out))
    return torch.nn.Sequential(*layers)


class Data:
    """The clips of a set on the device, and where a window of `w` frames can start."""

    def __init__(self, head, x, which, w, ahead, device, stride=1):
        self.x = torch.tensor(np.array(x), device=device)
        starts, lengths = head["starts"], head["lengths"]
        firsts, ends, clips = [], [], []
        # Each window has room for the largest stride: a frame that far before it, and its frames.
        for i, s in enumerate(head["sets"]):
            if s != which or lengths[i] < stride * (w + 1) + 3:
                continue
            f = np.arange(starts[i] + stride, starts[i] + lengths[i] - stride * (w + 1) - 1)
            firsts.append(f)
            ends.append(np.full(f.shape, starts[i] + lengths[i] - 1))
            clips.append(i)
        self.first = torch.tensor(np.concatenate(firsts), device=device)
        self.end = torch.tensor(np.concatenate(ends), device=device)
        self.clips = clips
        self.ahead = ahead

    def sample(self, n, gen):
        i = torch.randint(0, self.first.shape[0], (n,), device=self.first.device, generator=gen)
        return self.first[i], self.end[i]


class Norm:
    def __init__(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0).clamp(min=1e-3)

    def __call__(self, x):
        return ((x - self.mean) / self.std).clamp(-CLIP, CLIP)

    def state(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}


def goals(body, data, cur, row, end, rate):
    g_at = torch.minimum(row + data.ahead, end)
    seconds = ((g_at - row).float() / rate).clamp(min=1.0 / rate)
    return body.goal(cur, data.x[g_at], seconds)


def main(load_packed):
    run = runlib.Run()
    p = argparse.ArgumentParser()
    p.add_argument("--poses", default="poses.bin")
    p.add_argument("--body", default="creature.body.json")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--hidden", default="128,128,128")
    p.add_argument("--encoder", default="256,256")
    p.add_argument("--z", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--physics", type=float, default=0.0,
                   help="the weight of the parts' depth and slide on the ground (each over its target)")
    p.add_argument("--prior", type=float, default=0.3,
                   help="the share of windows whose code is drawn from the prior, not the encoder")
    p.add_argument("--goal", type=float, default=0.0,
                   help="the weight of the window's mean velocity against its goal's (per (m/s)^2)")
    p.add_argument("--ground", type=float, default=0.0,
                   help="the weight of each foot's height over the floor against the capture's (per m^2)")
    p.add_argument("--hide", type=float, default=0.3, help="the chance that each goal is hidden")
    p.add_argument("--strides", default="1,2,4",
                   help="the periods (frames) a window's steps may take, each update one, 1 half the time: "
                        "a body far from the eye is solved each few frames (phase 5 step 10)")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args(run.args)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(run.seed)
    gen = torch.Generator(device=dev).manual_seed(run.seed)
    t0 = time.time()
    head, x = load_packed(a.poses)
    rate = head["rate"]
    body = motion.Body(a.body, dev)
    strides = [int(k) for k in a.strides.split(",")]
    train = Data(head, x, "training", a.window, int(rate), dev, max(strides))
    test = Data(head, x, "test", a.window, int(rate), dev, max(strides))
    D = head["d"]
    ns, nt, ng = motion.STATE_ROOT + 2 * D, motion.STEP_ROOT + D, motion.GOAL
    # The normalisation, from 100,000 training frames as the capture has them.
    with torch.no_grad():
        r, e = train.sample(100000, gen)
        s0 = body.state(train.x[r - 1], train.x[r], rate)
        t0_ = body.step(train.x[r], train.x[r + 1])
        g0 = goals(body, train, train.x[r], r, e, rate)
    n_state, n_step, n_goal = Norm(s0), Norm(t0_), Norm(g0)
    n_goal.mean = torch.zeros_like(n_goal.mean)  # a hidden goal is 0, before and after
    n_goal.std = g0.abs().mean(0).clamp(min=1e-3) * 1.5
    hidden = [int(h) for h in a.hidden.split(",")]
    enc = mlp(ns + nt + ng, 2 * a.z, [int(h) for h in a.encoder.split(",")]).to(dev)
    # The decoder reads the state, the goal, its two flags, the period's log2 and the code.
    dec = mlp(ns + ng + 3 + a.z, nt, hidden).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
    print(f"solve: {len(train.clips)} training clips, {len(test.clips)} test clips, "
          f"{train.first.shape[0]} windows; state {ns}, step {nt}, goal {ng}, z {a.z}; "
          f"decoder {hidden} ({sum(q.numel() for q in dec.parameters())} weights); set up in "
          f"{time.time() - t0:.0f} s", flush=True)
    step = 0
    state = run.resume()
    if state is not None:
        enc.load_state_dict(state["enc"])
        dec.load_state_dict(state["dec"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        gen.set_state(state["gen"].to(gen.device) if dev == "cuda" else state["gen"])
        torch.set_rng_state(state["torch_rng"])
        for nm, nn_ in [("n_state", n_state), ("n_step", n_step), ("n_goal", n_goal)]:
            nn_.mean, nn_.std = state[nm]["mean"].to(dev), state[nm]["std"].to(dev)

    def decode(s, g, mask, z, k=1):
        gm = n_goal(g) * mask.repeat_interleave(2, -1)
        period = torch.full_like(mask[:, :1], math.log2(k))
        return dec(torch.cat([n_state(s), gm, mask, period, z], -1))

    def smooth_codes(B, w, g):
        """Codes for `w` ticks as the engine draws them (`engine::body::motion::code`): standard
        normal draws every `CODE_TICKS` ticks, each tick's between its two knots, from a random
        phase."""
        T = 8
        knots = torch.randn(B, (w + T - 1) // T + 2, a.z, device=dev, generator=g)
        phase = torch.randint(0, T, (B,), device=dev, generator=g)
        out = []
        for t in range(w):
            tt = phase + t
            i = tt // T
            u = ((tt % T).float() / T)[:, None]
            ka = knots[torch.arange(B, device=dev), i]
            kb = knots[torch.arange(B, device=dev), i + 1]
            out.append(((1 - u) * ka + u * kb) / torch.sqrt((1 - u) ** 2 + u ** 2))
        return out

    def rollout(data, starts, ends, w, fed_back, train_mode, k=1):
        """A window of `w` steps of `k` frames each from each start: the losses. The net gives a
        step of one frame that `k` frames take `k` times over (`Body.stretched`), from the state
        of poses `k` frames apart."""
        B = starts.shape[0]
        prev, cur = data.x[starts - k], data.x[starts]
        mask = (torch.rand(B, 2, device=dev, generator=gen) > a.hide).float()
        prior_rows = (torch.rand(B, 1, device=dev, generator=gen) < a.prior)
        # The codes at the frames the steps start at, as the engine draws them (a tick a frame).
        code_prior = smooth_codes(B, w * k, gen)[::k]
        rec, kl, phys, ground = 0.0, 0.0, 0.0, 0.0
        ends0 = None
        # The window's mean velocity along the ground (its steps, in each frame's heading frame)
        # and its goals' mean, for the goal's loss.
        v_sum = torch.zeros(B, 2, device=dev)
        g_sum = torch.zeros(B, 2, device=dev)
        for t in range(w):
            row = starts + k * t
            true_next = data.x[row + k]
            s = body.state(prev, cur, rate / k)
            tgt = body.unstretched(cur, body.step(cur, true_next), k)
            g = goals(body, data, cur, row, ends, rate)
            if train_mode:
                h = enc(torch.cat([n_state(s), n_step(tgt), n_goal(g)], -1))
                mu, logvar = h[:, :a.z], h[:, a.z:].clamp(-8, 4)
                z = mu + torch.randn(mu.shape, device=dev, generator=gen) * torch.exp(0.5 * logvar)
                kl = kl + (0.5 * (mu * mu + logvar.exp() - 1.0 - logvar).sum(-1)).mean()
                # A share of the windows take their code from the prior, as the game draws it:
                # then only the goal says where to go, so the decoder learns to follow it.
                z = torch.where(prior_rows, code_prior[t], z)
            else:
                z = torch.randn(B, a.z, device=dev, generator=gen)
            out = decode(s, g, mask, z, k)
            # A Gaussian likelihood of the step (spread 1 in its normalised numbers): summed over
            # the numbers, so the code's KL weighs against the whole step.
            rec = rec + 0.5 * ((out - n_step(tgt)) ** 2).sum(-1).mean()
            pred = n_step.mean + n_step.std * out
            v_sum = v_sum + pred[:, 0:2] * rate
            if k > 1:
                pred = body.stretched(cur, pred, k)
            g_sum = g_sum + g[:, 0:2]
            if (a.physics > 0 or a.ground > 0) and train_mode:
                nxt = body.after(cur, pred)
                if ends0 is None:
                    with torch.no_grad():
                        ends0 = feet_ends(cur)
                ends1 = feet_ends(nxt)
                if a.physics > 0:
                    phys = phys + physics_of(ends0, ends1, k)
                if a.ground > 0:
                    # Each foot's height over the floor, against the capture's same foot's.
                    with torch.no_grad():
                        h_true = foot_heights(feet_ends(true_next))
                    ground = ground + ((foot_heights(ends1) - h_true) ** 2).sum(-1).mean()
                nxt = nxt.detach()
            else:
                nxt = body.after(cur, pred.detach())
            use = (torch.rand(B, 1, device=dev, generator=gen) < fed_back).float()
            if (a.physics > 0 or a.ground > 0) and train_mode:
                # The next step's parts: these where the step was fed back, else the capture's.
                if fed_back >= 1.0:
                    ends0 = (ends1[0].detach(), ends1[1].detach())
                else:
                    ends0 = None
            prev, cur = cur, use * nxt + (1.0 - use) * true_next
            # The root's turn mixed so: renormalise.
            cur = torch.cat([cur[:, :3], cur[:, 3:7] / cur[:, 3:7].norm(dim=-1, keepdim=True),
                             cur[:, 7:]], -1)
        # Only the windows whose speed goal was shown are held to it.
        shown = mask[:, 0:1]
        goal_err = (shown * (v_sum / w - g_sum / w) ** 2).sum(-1).mean()
        return rec / w, kl / w, phys / w, goal_err, ground / w

    # The parts the physics measures: those that stand on the floor at rest (the soles), whose
    # kinematics alone are worked out, as the judge's slide and depth are the feet's in walking.
    feet = body.low_parts()

    def feet_ends(pose):
        a_, b_, ra_, rb_ = body.ends_of(pose, feet)
        return a_, b_

    # Each foot's parts (the soles left of the body's middle at rest, and right of it).
    rest_pose = torch.zeros(7 + body.freedom, device=dev)
    rest_pose[3:7] = body.rest
    ra0, rb0 = body.part_ends(rest_pose)
    side_left = torch.tensor([bool(ra0[i, 0] > 0) for i in feet], device=dev)

    def foot_heights(ends):
        """Each foot's lowest point above the floor [B, 2] (left, right)."""
        a_, b_ = ends
        idx = torch.tensor(feet, device=dev)
        low = torch.minimum(a_[..., 1] - body.part_ra[idx], b_[..., 1] - body.part_rb[idx])
        big = torch.full_like(low, 1e3)
        left = torch.where(side_left, low, big).min(-1).values
        right = torch.where(~side_left, low, big).min(-1).values
        return torch.stack([left, right], -1)

    def physics(cur, nxt):
        return physics_of(feet_ends(cur), feet_ends(nxt))

    def physics_of(ends0, ends1, k=1):
        """What the judge would see wrong in the step from one frame's parts (`ends0`, each
        part's two ends) to the next's, on flat ground at y = 0, each over its target: each part's
        depth under the ground (mm over 2), and the speed along the ground of each part's lowest
        end while it is within 2 cm of the ground in both frames (mm/s over 984)."""
        a0, b0 = ends0
        a1, b1 = ends1
        idx = torch.tensor(feet, device=dev)
        ra, rb = body.part_ra[idx], body.part_rb[idx]
        low0 = torch.stack([a0[..., 1] - ra, b0[..., 1] - rb], -1)
        low1 = torch.stack([a1[..., 1] - ra, b1[..., 1] - rb], -1)
        pen = torch.relu(-low1).sum(-1).mean(-1) * 1000.0 / 2.0
        near = ((low0 < 0.02) & (low1 < 0.02)).float()
        d = torch.stack([(a1 - a0)[..., [0, 2]].norm(dim=-1), (b1 - b0)[..., [0, 2]].norm(dim=-1)], -1)
        slide = (near * d).sum((-1, -2)) / near.sum((-1, -2)).clamp(min=1.0) * rate / k * 1000.0 / 984.0
        return (pen + slide).mean()

    def evaluate(data, n=512, seconds=4.0, k=1):
        """Rollouts from test windows with z drawn, all goals shown, for `seconds`: the mean error
        of the root's speed along the ground from its goal (m/s), the mean of the joints' error
        from the capture's after 1 s (rad), and the share whose root fell below half its height."""
        w = int(seconds * rate) // k
        g2 = torch.Generator(device=dev).manual_seed(12345)
        i = torch.randint(0, data.first.shape[0], (n,), device=dev, generator=g2)
        starts, ends = data.first[i], data.end[i]
        ok = (ends - starts) > k * w + 2
        starts, ends = starts[ok], ends[ok]
        B = starts.shape[0]
        prev, cur = data.x[starts - k], data.x[starts]
        h0 = cur[:, 1].clone()
        mask = torch.ones(B, 2, device=dev)
        codes = smooth_codes(B, w * k, g2)[::k]
        verr, fell, qerr = 0.0, torch.zeros(B, dtype=torch.bool, device=dev), 0.0
        phys_e = 0.0
        with torch.no_grad():
            for t in range(w):
                row = starts + k * t
                s = body.state(prev, cur, rate / k)
                g = goals(body, data, cur, row, ends, rate)
                z = codes[t]
                pred = n_step.mean + n_step.std * decode(s, g, mask, z, k)
                nxt = body.after(cur, body.stretched(cur, pred, k) if k > 1 else pred)
                phys_e += physics_of(feet_ends(cur), feet_ends(nxt), k).item()
                v = motion.qrot(motion.yaw(-body.heading(cur[:, 3:7])), nxt[:, :3] - cur[:, :3]) * rate / k
                verr += (torch.stack([v[:, 0], v[:, 2]], -1) - g[:, :2]).norm(dim=-1).mean().item()
                fell |= nxt[:, 1] < 0.5 * h0
                prev, cur = cur, nxt
                if t == int(rate) // k - 1:
                    qerr = (cur[:, 7:] - data.x[row + k, 7:]).abs().mean().item()
        return {"speed_error": verr / w, "joint_error_1s": qerr, "fell": fell.float().mean().item(),
                "physics": phys_e / w, "rollouts": B}

    last_eval = None
    while step < a.steps:
        # Scheduled sampling: none fed back for the first fifth, then more to all by three fifths.
        fed = min(max((step / a.steps - 0.2) / 0.4, 0.0), 1.0)
        starts, ends = train.sample(a.batch, gen)
        # Each update one stride: 1 half the time, the others the rest alike.
        u = float(torch.rand(1, device=dev, generator=gen))
        k = strides[0] if (u < 0.5 or len(strides) == 1) else strides[1 + int((u - 0.5) * 2 * (len(strides) - 1))]
        rec, kl, phys, goal_err, ground = rollout(train, starts, ends, a.window, fed, True, k)
        loss = rec + a.beta * kl + a.physics * phys + a.goal * goal_err + a.ground * ground
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
        opt.step()
        step += 1
        ph = float(phys.detach()) if not isinstance(phys, float) else phys
        run.progress(step, a.steps, f"rec {rec.item():.4f} kl {kl.item():.2f} phys {ph:.3f} goal {goal_err.item():.3f} fed {fed:.2f}")
        if step % 100 == 0:
            print(f"update {step}: rec {rec.item():.4f} kl {kl.item():.2f} physics {ph:.3f} goal {goal_err.item():.3f} fed back {fed:.2f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        if a.eval_every and step % a.eval_every == 0:
            last_eval = {"test": evaluate(test), "training": evaluate(train)}
            for k in strides[1:]:
                last_eval[f"test_period_{k}"] = evaluate(test, k=k)
            print(f"evaluate at {step}: {json.dumps(last_eval)}", flush=True)
        if run.due() or step == a.steps:
            run.checkpoint(step, {"enc": enc.state_dict(), "dec": dec.state_dict(),
                                  "opt": opt.state_dict(), "step": step, "gen": gen.get_state().cpu(),
                                  "torch_rng": torch.get_rng_state(), "n_state": n_state.state(),
                                  "n_step": n_step.state(), "n_goal": n_goal.state()})
        if run.should_stop():
            run.abandon()
            return
    final = {"test": evaluate(test), "training": evaluate(train)}
    for k in strides[1:]:
        final[f"test_period_{k}"] = evaluate(test, k=k)
    print(f"evaluate at the end: {json.dumps(final)}", flush=True)
    # The decoder as the engine reads it: the inputs' normalisation (state, goal, the goals'
    # flags, z) and the outputs' scale folded into the last layer.
    lin = [m for m in dec if isinstance(m, torch.nn.Linear)]
    last = torch.nn.Linear(lin[-1].in_features, lin[-1].out_features)
    with torch.no_grad():
        last.weight.copy_(lin[-1].weight * n_step.std[:, None])
        last.bias.copy_(lin[-1].bias * n_step.std + n_step.mean)
    lin = lin[:-1] + [last]
    mean = torch.cat([n_state.mean, n_goal.mean, torch.zeros(3 + a.z, device=dev)])
    scale = torch.cat([1.0 / n_state.std, 1.0 / n_goal.std, torch.ones(3 + a.z, device=dev)])
    acts = ["elu"] * (len(lin) - 1) + ["none"]
    run.output("solver.wnet", wnet.encode(lin, acts, mean, scale, CLIP))
    record = {"inputs": {"state": ns, "goal": ng, "goal_flags": 2, "period_log2": 1, "z": a.z}, "outputs": nt,
              "hidden": hidden, "seed": run.seed, "args": vars(a), "steps": step,
              "evaluation": final, "poses": a.poses, "rate": rate,
              "note": "a hidden goal's numbers are 0 and its flag 0 (flag 0: the speed, 1: the heading)"}
    run.output("solver.json", json.dumps(record, indent=1).encode())
    run.finish({"step": step, "evaluation": final, "seconds": time.time() - t0})
