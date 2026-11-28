#!/usr/bin/env python3
"""The slab an affine enclosure gives: the engine's slope against the best multipliers.

Theory.Sheet.Slab proves that for every multiplier `lam` the surface of `c + l.u + eta`
(|eta| <= e, u in [-1, 1]^3) lies at or below

    hi(lam) = -lam c + |lam| e + |wl - lam l|_1

across the plane whose coordinate is `wl.u`, and at or above the mirrored `lo(lam)`. It does not
prove which `lam` is best. This check holds two things by numbers:

1. `hi` is convex and piecewise linear in `lam`, so its least value is at a breakpoint: `lam = 0`
   or `lam = wl[i] / l[i]`. The least over those four is compared with the linear programme's
   true optimum, found by enumerating the vertices of the box cut by the two planes
   `c + l.u = +-e`. (Not proven in Lean: this is the evidence for it.)
2. How much narrower that slab is than `certify_sheet`'s (one slope, `alpha = dot(l / h, n)`,
   cut to the box's span), over cells a surface crosses, for normals tilted from the
   enclosure's gradient.

Run: python3 theory/checks/slab_lp.py
"""
import itertools
import math
import random


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm1(a):
    return sum(abs(x) for x in a)


def hi(lam, c, e, l, wl):
    return -lam * c + abs(lam) * e + norm1([w - lam * x for w, x in zip(wl, l)])


def lo(lam, c, e, l, wl):
    return -lam * c - abs(lam) * e - norm1([w - lam * x for w, x in zip(wl, l)])


def breakpoints(l, wl):
    return [0.0] + [w / x for w, x in zip(wl, l) if abs(x) > 1e-12]


def best_slab(c, e, l, wl):
    bp = breakpoints(l, wl)
    return max(lo(m, c, e, l, wl) for m in bp), min(hi(m, c, e, l, wl) for m in bp)


def engine_slab(c, e, l, wl, n):
    """certify_sheet's slab for a unit box (h_in = 1, so wl = n), without its rounding margins."""
    span = norm1(wl)
    alpha = dot(l, n)
    if abs(alpha) <= 1e-6:
        return -span, span
    r = [x - alpha * w for x, w in zip(l, wl)]
    half = (norm1(r) + e) / abs(alpha)
    ctr = -c / alpha
    return max(ctr - half, -span), min(ctr + half, span)


def lp_slab(c, e, l, wl):
    """The true extent of wl.u over {u in the box, |c + l.u| <= e}, by its vertices."""
    pts = []
    for u in itertools.product((-1.0, 1.0), repeat=3):
        if abs(c + dot(l, u)) <= e + 1e-12:
            pts.append(u)
    for axis in range(3):
        if abs(l[axis]) < 1e-12:
            continue
        others = [i for i in range(3) if i != axis]
        for a, b in itertools.product((-1.0, 1.0), repeat=2):
            for rhs in (e, -e):
                u = [0.0, 0.0, 0.0]
                u[others[0]], u[others[1]] = a, b
                u[axis] = (rhs - c - l[others[0]] * a - l[others[1]] * b) / l[axis]
                if abs(u[axis]) <= 1.0 + 1e-12:
                    pts.append(tuple(u))
    if not pts:
        return None
    vals = [dot(wl, u) for u in pts]
    return min(vals), max(vals)


def unit(rng):
    while True:
        v = [rng.gauss(0, 1) for _ in range(3)]
        m = math.sqrt(dot(v, v))
        if m > 1e-6:
            return [x / m for x in v]


def tilted(g, deg, rng):
    """A unit vector `deg` degrees from the unit vector `g`."""
    while True:
        t = unit(rng)
        d = dot(t, g)
        t = [x - d * y for x, y in zip(t, g)]
        m = math.sqrt(dot(t, t))
        if m > 1e-3:
            t = [x / m for x in t]
            break
    a = math.radians(deg)
    return [math.cos(a) * x + math.sin(a) * y for x, y in zip(g, t)]


def main():
    rng = random.Random(1)

    # 1. The breakpoints against the linear programme.
    worst = 0.0
    held = 0
    for _ in range(20000):
        l = [rng.uniform(-1, 1) for _ in range(3)]
        wl = [rng.uniform(-1, 1) for _ in range(3)]
        e = rng.uniform(0, 0.3)
        c = rng.uniform(-1, 1) * norm1(l)
        true = lp_slab(c, e, l, wl)
        if true is None:
            continue
        b_lo, b_hi = best_slab(c, e, l, wl)
        worst = max(worst, abs(b_lo - true[0]), abs(b_hi - true[1]))
        held += 1
    print(f"breakpoints against the linear programme: {held} enclosures, largest difference {worst:.2e}")

    # 2. The engine's slab against the best, on cells a surface crosses. Unit box (half width 1,
    # so a cell is 2 wide), a field of unit slope, the surface through a point of the box.
    print("tilt   e      engine/best width (median, 90th)   within 0.15 cell: engine, best (of 4000)")
    for deg in (0, 2, 5, 10, 20):
        for e in (0.005, 0.02, 0.05):
            ratios = []
            n_engine = n_best = 0
            for _ in range(4000):
                g = unit(rng)
                n = tilted(g, deg, rng)
                through = [rng.uniform(-1, 1) for _ in range(3)]
                c = -dot(g, through)
                e_lo, e_hi = engine_slab(c, e, g, n, n)
                b_lo, b_hi = best_slab(c, e, g, n)
                we, wb = max(e_hi - e_lo, 0.0), max(b_hi - b_lo, 0.0)
                if wb > 1e-9:
                    ratios.append(we / wb)
                # A slab 0.15 of a cell wide (the watch's 0.375 px at 2.5 px a cell): 0.3 here.
                n_engine += we <= 0.3
                n_best += wb <= 0.3
            ratios.sort()
            med = ratios[len(ratios) // 2]
            p90 = ratios[(9 * len(ratios)) // 10]
            print(f"{deg:3d}°  {e:5.3f}  {med:6.3f} {p90:6.3f}                      {n_engine:5d} {n_best:5d}")


if __name__ == "__main__":
    main()
