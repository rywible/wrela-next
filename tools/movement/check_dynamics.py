"""Holds the judge's inverse dynamics to the simulator's (docs/plans/bodies-and-movement.md, phase
4 step 3): for a clip, MuJoCo's own inverse dynamics (`mj_inverse`) of the clip's states
(`.movement/sim/refs/CLIP.ref`, which tools/movement/export writes), with the judge's contact
forces applied to each body as outside forces (as generalised forces, `mj_applyFT`) and the simulator's own contacts, servos, damping
and armature off, against the judge's joint moments (`.movement/sim/dyn/CLIP.dyn`, which
`tools/movement/judge -- dump CLIP` writes).

    .movement/venv/bin/python tools/movement/check_dynamics.py CLIP [CLIP ...]

A joint of the model is one to three hinges; their torques are the joint's moment along each
hinge's axis, so the moment is solved back from them and compared as a vector: for each clip, the
median over frames and joints of |M_sim - M_judge| over the median joint's |M_judge|, and the same
for the free joint (the root residual), and the 95th percentile of the joints' differences over
the same median. Prints each, and exits 1 past 5%.
"""
import struct
import sys

import mujoco
import numpy as np

SIM = ".movement/sim"


def read_ref(path):
    b = open(path, "rb").read()
    _, nq, segs, n = struct.unpack_from("<4I", b, 4)
    rate = struct.unpack_from("<f", b, 20)[0]
    row = nq + 7 * segs
    data = np.frombuffer(b, dtype="<f4", offset=24, count=n * row).reshape(n, row)
    return rate, data[:, :nq].astype(np.float64)


def read_dyn(path):
    b = open(path, "rb").read()
    n, segs = struct.unpack_from("<2I", b, 0)
    d = np.frombuffer(b, dtype="<f4", offset=8, count=n * 3 * segs * 3).reshape(n, 3, segs, 3)
    return d[:, 0].astype(np.float64), d[:, 1].astype(np.float64), d[:, 2].astype(np.float64)


def check(m, clip):
    d = mujoco.MjData(m)
    rate, q = read_ref(f"{SIM}/refs/{clip}.ref")
    jm, cf, cm = read_dyn(f"{SIM}/dyn/{clip}.dyn")
    dt = 1.0 / rate
    n = q.shape[0]
    errs, mags, root_errs, root_mags = [], [], [], []
    seg_body = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"s{s}") for s in range(jm.shape[1])]
    for f in range(1, n - 1):
        k = f - 1  # the judge's inner frame
        if k >= jm.shape[0]:
            break
        v0 = np.zeros(m.nv)
        v1 = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, v0, dt, q[f - 1], q[f])
        mujoco.mj_differentiatePos(m, v1, dt, q[f], q[f + 1])
        d.qpos[:] = q[f]
        d.qvel[:] = 0.5 * (v0 + v1)
        d.qacc[:] = (v1 - v0) / dt
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        # The judge's contact forces as generalised forces (each at its segment's origin, with
        # its moment): `mj_inverse` gives what all forces but the outside ones must make, so
        # theirs is taken off.
        ext = np.zeros(m.nv)
        for s, b in enumerate(seg_body):
            mujoco.mj_applyFT(m, d, cf[k, s], cm[k, s], d.xpos[b].copy(), b, ext)
        mujoco.mj_inverse(m, d)
        tau = d.qfrc_inverse.copy() - ext
        # The free joint: MuJoCo's qfrc for it is the force and the moment (in the body's frame)
        # that only a hand at the root could give.
        root = tau[:6]
        root_errs.append(np.linalg.norm(root[:3]))
        # Each joint's hinges: solve its moment back from their torques and their world axes.
        for s, b in enumerate(seg_body):
            if s == 0:
                continue
            jids = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
            if not jids:
                continue
            axes = np.array([d.xaxis[j] for j in jids])
            tq = np.array([tau[m.jnt_dofadr[j]] for j in jids])
            ours = jm[k, s]
            if len(jids) == 3:
                sim_m = np.linalg.solve(axes, tq)
                errs.append(np.linalg.norm(sim_m - ours))
            else:
                # Fewer hinges than three: compare along the hinges' axes alone.
                errs.append(np.linalg.norm(tq - axes @ ours))
            mags.append(np.linalg.norm(ours))
    errs, mags = np.array(errs), np.array(mags)
    rel = np.median(errs) / np.median(mags)
    p95 = np.percentile(errs, 95) / np.median(mags)
    weight = sum(m.body_mass) * 9.81
    print(f"{clip}: joints: median |M_sim - M_judge| {np.median(errs):.3f} N m against a median "
          f"|M_judge| {np.median(mags):.3f} N m: {100 * rel:.2f}% (95th percentile {100 * p95:.2f}%); "
          f"the root's residual force in the "
          f"simulator {np.median(root_errs) / weight:.4f} of the weight (median)")
    return max(rel, p95)


def main():
    m = mujoco.MjModel.from_xml_path(f"{SIM}/creature.xml")
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    m.dof_armature[:] = 0
    m.dof_damping[:] = 0
    m.jnt_limited[:] = 0
    worst = max(check(m, c) for c in sys.argv[1:])
    sys.exit(0 if worst <= 0.05 else 1)


main()
