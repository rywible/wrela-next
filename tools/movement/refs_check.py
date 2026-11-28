"""The check of the clips a tracker is given (docs/plans/bodies-and-movement.md, phase 2's and
phase 4's gate before any training): each packed clip looked at as a body in the simulator's
world, before a GPU is asked to follow it. A clip the capture's fit got wrong is found here in
seconds, not as a fall after hours.

    python3 tools/movement/refs_check.py MODEL REFS [--poses .movement/poses] [--list]

For each clip of REFS (tools/movement/pack_refs.py), with the model's kinematics:

- `in the floor`: its lowest point more than 10 mm under the floor in its first frame, or more
  than 30 mm under it in any of each tenth frame;
- `unbalanced`: in more than three tenths of its slow frames (the root under 0.3 m/s along the
  ground; each tenth frame looked at, ten or more of them: a body setting off leaves its support) its centre of mass is more than 50 mm, along the ground, outside the
  hull of what is within 30 mm of the floor: no body stands so;
- `far from its capture`: its record's mapped joints that are aimed at places (the pelvis, the
  elbows, the hands, the feet and the toes) more than 60 mm from them on average (`--poses`: the
  import's records).

It prints the share of clips with each fault, by source, and with `--list` each faulty clip, and
ends 1 if more than 5% of the clips of any source have a fault.
"""
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import falls  # noqa: E402

AIMED = ("Hips", "Hand", "Wrist", "ForeArm", "Elbow", "Foot", "Ankle", "ToeBase", "Toe")


def outside(pts, p):
    """How far `p` is outside the convex hull of `pts` (0 inside)."""
    pts = sorted(set(map(tuple, np.round(pts, 5))))
    if not pts:
        return 1.0e9
    if len(pts) < 3:
        a, b = np.array(pts[0]), np.array(pts[-1])
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        return float(np.linalg.norm(p - (a + ab * t)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    h = [np.array(x) for x in lower[:-1] + upper[:-1]]
    inside, near = True, 1.0e9
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        if cross(a, b, p) < 0:
            inside = False
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        near = min(near, float(np.linalg.norm(p - (a + ab * t))))
    return 0.0 if inside else near


def support(m, d):
    """The body's lowest point, and how far its centre of mass is outside what holds it up."""
    low = falls.lowest(m, d)
    pts = []
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 or m.geom_type[g] not in (2, 3):
            continue
        c, r = d.geom_xpos[g], m.geom_size[g][0]
        ends = [c]
        if m.geom_type[g] == 3:
            ax = d.geom_xmat[g].reshape(3, 3)[:, 2]
            ends = [c - ax * m.geom_size[g][1], c + ax * m.geom_size[g][1]]
        for e in ends:
            if e[1] - r < 0.03:
                pts.append([e[0], e[2]])
    com = d.subtree_com[1]
    return low, outside(np.array(pts), np.array([com[0], com[2]]))


def main():
    model, refs = sys.argv[1], sys.argv[2]
    poses = sys.argv[sys.argv.index("--poses") + 1] if "--poses" in sys.argv else ".movement/poses"
    m = mujoco.MjModel.from_xml_path(model)
    d = mujoco.MjData(m)
    h, raw = falls.packed(refs, b"WRFS")
    q = np.frombuffer(raw, "<f4").reshape(-1, h["nq"])
    rate = h["rate"]
    faults, total, rows = {}, {}, []
    for name, s, n in zip(h["names"], h["starts"], h["lengths"]):
        src = name.split("/")[0]
        total[src] = total.get(src, 0) + 1
        slow = unbalanced = 0
        deepest = first_low = 0.0
        for f in range(0, n - 1, 10):
            d.qpos[:] = q[s + f]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            low, out = support(m, d)
            if f == 0:
                first_low = low
            deepest = min(deepest, low)
            if np.linalg.norm((q[s + f + 1, [0, 2]] - q[s + f, [0, 2]]) * rate) < 0.3:
                slow += 1
                unbalanced += int(out > 0.05)
        aimed = -1.0
        rec = os.path.join(poses, "human", f"{name}.json")
        if os.path.exists(rec):
            by = json.load(open(rec)).get("joint_mm_by_bone")
            if by:
                v = [x for k, x in by.items() if k.endswith(AIMED)]
                aimed = float(np.mean(v)) if v else -1.0
        mine = []
        if first_low < -0.01 or deepest < -0.03:
            mine.append("in the floor")
        if slow >= 10 and unbalanced > 0.3 * slow:
            mine.append("unbalanced")
        if aimed > 60.0:
            mine.append("far from its capture")
        for f in mine:
            faults.setdefault(src, {})
            faults[src][f] = faults[src].get(f, 0) + 1
        if mine:
            faults.setdefault(src, {})
            faults[src]["any"] = faults[src].get("any", 0) + 1
            rows.append(f"refs: {name}: {', '.join(mine)} (first frame's lowest {1000 * first_low:.0f} mm, deepest "
                        f"{1000 * deepest:.0f} mm, unbalanced {unbalanced} of {slow} slow frames, aimed joints {aimed:.0f} mm)")
    bad = False
    for src in sorted(total):
        f = faults.get(src, {})
        share = f.get("any", 0) / total[src]
        bad = bad or share > 0.05
        print(f"refs: {src}: {f.get('any', 0)} of {total[src]} clips with a fault ({100 * share:.1f}%): "
              + ", ".join(f"{k} {v}" for k, v in sorted(f.items()) if k != "any"))
    if "--list" in sys.argv:
        for r in rows:
            print(r)
    print("refs: " + ("more than 5% of a source's clips have a fault" if bad else "ok"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
