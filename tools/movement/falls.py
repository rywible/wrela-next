"""Why the tracker lost each clip (docs/plans/bodies-and-movement.md, phase 4 step 7: "find the
clips that the tracker cannot follow"): for each clip a run did not follow to its end, what the
clip itself asks of a body in the second before the fall, and what the simulator's body was doing,
so a fall is put down to the clip (no body could do it: the retarget's fault) or to the tracker
(a body could, this net did not: training's).

    python3 tools/movement/falls.py MODEL REFS RUN_OUTPUTS [--list]

MODEL is the run's `human.xml`, REFS the packed clips it followed, RUN_OUTPUTS its outputs
(`filtered.bin`, and `filtered_extra.bin` where it wrote one). For each fall, over the 30 frames
before it:

- the clip's own demands, from its poses alone (the simulator's kinematics; the body's centre of
  mass differenced twice): the floor's force it needs, F = m (a - g), as a load (F's vertical part
  over the weight) and a grip (its part along the ground over its vertical part: more than the
  floor's friction, 1, cannot be had); how deep its lowest part is in the floor, and how far above
  it while it needs the floor's force; its fastest joint;
- the simulator's body: how far its root and its pose about the root are from the clip's at the
  fall, and the share of its servos at the limit of their strength (`filtered_extra.bin`).

Each fall gets its first cause of: `start` (in its first second), `in the floor` (the clip's
lowest part more than 30 mm under it), `no floor` (it needs more than half its weight from a floor
its lowest part is more than 50 mm above), `grip` (more grip than friction gives, or pulled down),
`strength` (a tenth or more of the servos at their limit), `fast` (a joint past 15 rad/s), or
`tracker`. It prints the counts by cause and by source, and with `--list` each fall.
"""
import json
import struct
import sys
import zlib

import mujoco
import numpy as np

WINDOW = 30
G = 9.81


def packed(path, magic):
    b = open(path, "rb").read()
    assert b[:4] == magic, f"{path}: not {magic}"
    n = struct.unpack("<I", b[4:8])[0]
    return json.loads(b[8:8 + n]), zlib.decompress(b[8 + n:])


def lowest(m, d):
    """The height of the body's lowest point (its colliding capsules and spheres)."""
    low = 1.0e9
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 or m.geom_type[g] not in (2, 3):
            continue
        y = d.geom_xpos[g][1] - m.geom_size[g][0]
        if m.geom_type[g] == 3:
            y -= abs(d.geom_xmat[g].reshape(3, 3)[1, 2]) * m.geom_size[g][1]
        low = min(low, y)
    return low


def demands(m, d, q, rate):
    """Of the frames `q` of a clip: each inner frame's load and grip, each frame's lowest point,
    and the fastest joint (rad/s)."""
    com, low = [], []
    for x in q:
        d.qpos[:] = x
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        com.append(d.subtree_com[1].copy())
        low.append(lowest(m, d))
    com, low = np.array(com), np.array(low)
    acc = (com[2:] - 2 * com[1:-1] + com[:-2]) * rate * rate
    f = acc + np.array([0.0, G, 0.0])
    load = f[:, 1] / G
    grip = np.linalg.norm(f[:, [0, 2]], axis=1) / np.maximum(f[:, 1], 1e-3)
    speed = np.abs(np.diff(q[:, 7:], axis=0)).max() * rate if len(q) > 1 else 0.0
    return load, grip, low, speed


def main():
    model, refs_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    m = mujoco.MjModel.from_xml_path(model)
    d = mujoco.MjData(m)
    rh, rb = packed(refs_path, b"WRFS")
    ref = np.frombuffer(rb, "<f4").reshape(-1, rh["nq"])
    fh, fb = packed(f"{out}/filtered.bin", b"WFLT")
    fil = np.frombuffer(fb, "<f4").reshape(-1, fh["nq"])
    rate = rh["rate"]
    extra = None
    try:
        xh, xb = packed(f"{out}/filtered_extra.bin", b"WFLX")
        c = xh["columns"]
        w = sum(c.values())
        extra = np.frombuffer(xb, "<f2").reshape(-1, w)
        nu = c["action"]
    except FileNotFoundError:
        pass
    strength = np.maximum(np.abs(m.actuator_forcerange[:, 0]), np.abs(m.actuator_forcerange[:, 1]))
    where = {n: (s, l) for n, s, l in zip(rh["names"], rh["starts"], rh["lengths"])}
    causes, by_source, rows, at = {}, {}, [], 0
    total = {}
    for clip in fh["clips"]:
        k, name = clip["frames"], clip["name"]
        src = name.split("/")[0]
        total[src] = total.get(src, 0) + 1
        first = at
        at += k
        if clip["followed"]:
            continue
        s, length = where[name]
        a = max(0, k - WINDOW)
        q = ref[s + a:s + min(k + 2, length)].astype(np.float64)
        load, grip, low, speed = demands(m, d, q, rate)
        # The body at the fall against the clip's.
        root = pose = 0.0
        if k > 0:
            d.qpos[:] = fil[first + k - 1]
            mujoco.mj_kinematics(m, d)
            mine = d.xpos[1:].copy()
            d.qpos[:] = ref[s + k - 1]
            mujoco.mj_kinematics(m, d)
            theirs = d.xpos[1:].copy()
            root = float(np.linalg.norm(mine[0] - theirs[0]))
            pose = float(np.linalg.norm((mine - mine[0]) - (theirs - theirs[0]), axis=1).mean())
        limit = 0.0
        if extra is not None and k > 2:
            sf = np.abs(extra[first + max(1, k - WINDOW):first + k, nu:2 * nu].astype(np.float32))
            limit = float((sf > 0.98 * strength).mean())
        needs = load > 0.5
        hover = low[1:-1][needs].min() if needs.any() and len(low) > 2 else 0.0
        over = bool(((grip > 1.0) & needs).sum() >= 3 or (load < -0.2).sum() >= 3)
        if k < int(rate):
            cause = "start"
        elif low.min() < -0.03:
            cause = "in the floor"
        elif needs.any() and (low[1:-1][needs] > 0.05).sum() >= 3:
            cause = "no floor"
        elif over:
            cause = "grip"
        elif limit >= 0.1:
            cause = "strength"
        elif speed > 15.0:
            cause = "fast"
        else:
            cause = "tracker"
        causes[cause] = causes.get(cause, 0) + 1
        by_source.setdefault(src, {})
        by_source[src][cause] = by_source[src].get(cause, 0) + 1
        rows.append((name, k, length, cause, float(load.max()) if len(load) else 0.0,
                     float(grip[needs].max()) if needs.any() else 0.0, float(low.min()), float(hover), speed,
                     root, pose, limit))
    falls = len(rows)
    print(f"falls: {falls} of {len(fh['clips'])} clips not followed to the end")
    for c, n in sorted(causes.items(), key=lambda x: -x[1]):
        print(f"falls: {c}: {n} ({100.0 * n / max(falls, 1):.0f}%)")
    for src, cs in sorted(by_source.items()):
        n = sum(cs.values())
        print(f"falls: {src}: {n} of {total[src]} clips: " + ", ".join(f"{c} {k}" for c, k in sorted(cs.items(), key=lambda x: -x[1])))
    if "--list" in sys.argv:
        for r in sorted(rows, key=lambda r: (r[3], r[0])):
            print(f"fall: {r[0]}: frame {r[1]} of {r[2]}: {r[3]}; the clip's load {r[4]:.2f}, grip {r[5]:.2f}, "
                  f"lowest {1000 * r[6]:.0f} mm, fastest joint {r[8]:.1f} rad/s; the body's root {1000 * r[9]:.0f} mm "
                  f"and pose {1000 * r[10]:.0f} mm from the clip's, servos at their limit {100 * r[11]:.0f}%")


if __name__ == "__main__":
    main()
