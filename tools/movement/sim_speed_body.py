"""How fast the simulator steps our body (the export of tools/movement/export) on one Colab GPU,
by the solver's settings and the number of worlds (docs/plans/bodies-and-movement.md, phase 4: the
first small run read 4,800 samples a second, below the estimate of 0.4). Started by the run tool:

    tools/movement/colab-run start NAME --gpu L4 --hours 0.2 --script tools/movement/sim_speed_body.py \
        --file .movement/sim/creature.xml -- [--worlds 1024,4096,8192]

Each variant steps the body standing at rest under gravity (its servos holding the rest), as a CUDA
graph of `SUBSTEPS` steps, for `--seconds` of wall clock after a warm-up; it prints a line
`RESULT_ROW solver=S iterations=I timestep=T worlds=N steps_per_second=X control_steps_per_second=Y`
for each, and ends with the run tool's result.
"""

import argparse
import json
import subprocess
import sys
import time

import runlib


def main():
    run = runlib.Run()
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", default="1024,4096,8192")
    p.add_argument("--seconds", type=float, default=8.0)
    a = p.parse_args(run.args)
    try:
        import mujoco_warp  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mujoco-warp"], check=True)
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    wp.init()
    variants = [
        ("newton", 50, 1 / 240),
        ("newton", 10, 1 / 240),
        ("newton", 4, 1 / 240),
        ("cg", 20, 1 / 240),
        ("newton", 10, 1 / 120),
        ("newton", 4, 1 / 120),
    ]
    rows = []
    total = len(variants) * len(a.worlds.split(","))
    done = 0
    for solver, iters, dt in variants:
        for n in [int(x) for x in a.worlds.split(",")]:
            m = mujoco.MjModel.from_xml_path("creature.xml")
            m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON if solver == "newton" else mujoco.mjtSolver.mjSOL_CG
            m.opt.iterations = iters
            m.opt.timestep = dt
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            d.ctrl[:] = 0.0
            with wp.ScopedDevice("cuda:0"):
                wm = mjw.put_model(m)
                wd = mjw.put_data(m, d, nworld=n, nconmax=32, njmax=160)
                sub = round((1 / 30) / dt)
                mjw.step(wm, wd)
                wp.synchronize()
                with wp.ScopedCapture() as cap:
                    for _ in range(sub):
                        mjw.step(wm, wd)
                g = cap.graph
                for _ in range(5):
                    wp.capture_launch(g)
                wp.synchronize()
                t = time.time()
                k = 0
                while time.time() - t < a.seconds:
                    wp.capture_launch(g)
                    k += 1
                    if k % 10 == 0:
                        wp.synchronize()
                wp.synchronize()
                el = time.time() - t
            sps = k * sub * n / el
            cps = k * n / el
            row = {"solver": solver, "iterations": iters, "timestep": dt, "worlds": n,
                   "steps_per_second": round(sps), "control_steps_per_second": round(cps)}
            rows.append(row)
            print("RESULT_ROW " + " ".join(f"{x}={y}" for x, y in row.items()), flush=True)
            done += 1
            run.progress(done, total, f"{solver} {iters} {n}")
    run.finish({"rows": rows})


main()
