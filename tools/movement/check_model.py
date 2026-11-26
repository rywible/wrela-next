"""Holds the simulator's model of a body to the engine's (docs/plans/bodies-and-movement.md,
phase 4 step 2): each segment's mass, centre of mass and inertia as MuJoCo reads
`.movement/sim/creature.xml`, against the engine's (`creature_masses.txt`, which the export writes
beside it); and, for a clip's states (`refs/CLIP.ref`), each segment's place and turn as MuJoCo's
kinematics put them from the clip's qpos, against the engine's (`Body::places`).

    .movement/venv/bin/python tools/movement/check_model.py [CLIP]

Prints the worst relative difference of each and exits 1 past 0.1% (masses) or 0.1 mm / 3 mrad
(the kinematics: 3 mrad, the f32 angles of a ball joint near a quarter turn).
"""
import struct
import sys

import mujoco
import numpy as np

SIM = ".movement/sim"


def read_ref(path):
    b = open(path, "rb").read()
    assert b[:4] == b"WREF"
    _, nq, segs, n = struct.unpack_from("<4I", b, 4)
    rate = struct.unpack_from("<f", b, 20)[0]
    row = nq + 7 * segs
    data = np.frombuffer(b, dtype="<f4", offset=24, count=n * row).reshape(n, row)
    return rate, data[:, :nq], data[:, nq:].reshape(n, segs, 7)


def main():
    m = mujoco.MjModel.from_xml_path(f"{SIM}/creature.xml")
    d = mujoco.MjData(m)
    rows = [l.split() for l in open(f"{SIM}/creature_masses.txt").read().splitlines()[1:]]
    worst_mass = 0.0
    worst_inertia = 0.0
    for r in rows:
        s = int(r[0])
        mass, c = float(r[1]), np.array(list(map(float, r[2:5])))
        ixx, iyy, izz, ixy, ixz, iyz = map(float, r[5:11])
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"s{s}")
        if mass == 0:
            print(f"segment {s}: the engine weighs it at 0; the model {m.body_mass[b]} kg")
            continue
        worst_mass = max(worst_mass, abs(m.body_mass[b] / mass - 1))
        # The model's inertia in the body's frame: R diag R^T, R from body_iquat.
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.body_iquat[b])
        R = rot.reshape(3, 3)
        I = R @ np.diag(m.body_inertia[b]) @ R.T
        E = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        worst_inertia = max(worst_inertia, np.linalg.norm(I - E) / np.linalg.norm(E))
        assert np.linalg.norm(m.body_ipos[b] - c) < 1e-6, (s, m.body_ipos[b], c)
    print(f"masses: worst {100 * worst_mass:.5f}%, inertia worst {100 * worst_inertia:.5f}%")
    ok = worst_mass < 1e-3 and worst_inertia < 1e-3
    if len(sys.argv) > 1:
        rate, qpos, frames = read_ref(f"{SIM}/refs/{sys.argv[1]}.ref")
        wp, wq = 0.0, 0.0
        for f in range(0, len(qpos), max(1, len(qpos) // 200)):
            d.qpos[:] = qpos[f]
            mujoco.mj_kinematics(m, d)
            for s in range(frames.shape[1]):
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"s{s}")
                wp = max(wp, np.linalg.norm(d.xpos[b] - frames[f, s, :3]))
                qa, qb = d.xquat[b], frames[f, s, 3:]
                dq = abs(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)))
                wq = max(wq, 2 * np.arccos(min(1.0, dq)))
        print(f"kinematics of {sys.argv[1]}: worst place {1000 * wp:.4f} mm, worst turn {1000 * wq:.4f} mrad")
        # A ball joint is three hinges in the model: its angles from its turn in f32, which near
        # a quarter turn of the middle one loses a few bits (1.9 mrad on 2026-11-25).
        ok = ok and wp < 1e-4 and wq < 3e-3
    sys.exit(0 if ok else 1)


main()
