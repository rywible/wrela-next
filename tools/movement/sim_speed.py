"""How fast the simulator steps many bodies on one Colab GPU (docs/plans/bodies-and-movement.md,
phase 0.4). Usage, from this machine:

    colab usage
    colab run --gpu T4 --timeout 900 tools/movement/sim_speed.py [mjx] [WORLDS ...]
    colab usage

It installs MuJoCo Warp, takes the humanoid that comes with its benchmarks (not our body), and
steps WORLDS copies of it (default 256 1024 4096) with the simulator's own speed test. With `mjx`
it steps MJX's test humanoid instead (the model tools/movement/sim_speed_jax.py also steps, which
asks less of the solver). It prints the GPU, each run's steps a second, and one line
`RESULT model=M worlds=N steps_per_second=X` for each size. It keeps nothing: the VM is released
when it ends.
"""

import re
import subprocess
import sys
import time


def sh(cmd, quiet=False):
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not quiet:
        print(p.stdout[-6000:], flush=True)
    return p.returncode, p.stdout


def main():
    mjx = "mjx" in sys.argv[1:]
    worlds = [int(a) for a in sys.argv[1:] if a != "mjx"] or [256, 1024, 4096]
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    t0 = time.time()
    sh("pip install -q mujoco-warp 2>&1 | tail -3")
    sh("rm -rf /content/mujoco_warp && git clone -q --depth 1 "
       "https://github.com/google-deepmind/mujoco_warp.git /content/mujoco_warp")
    print(f"setup {time.time() - t0:.0f} s", flush=True)
    sh("pip show mujoco-warp warp-lang mujoco 2>/dev/null | grep -E '^(Name|Version|License)'")
    sh("cd /content/mujoco_warp && git log -1 --format='%H %ci' && ls benchmarks/humanoid")
    model, name = "benchmarks/humanoid/humanoid.xml", "warp_benchmark"
    if mjx:
        sh("pip install -q mujoco-mjx 2>&1 | tail -3")
        code, out = sh("python -c \"import mujoco.mjx, os; print(os.path.dirname(mujoco.mjx.__file__))\"")
        model, name = out.strip().splitlines()[-1] + "/test_data/humanoid/humanoid.xml", "mjx_test"
    for n in worlds:
        t = time.time()
        code, out = sh(
            f"cd /content/mujoco_warp && mjwarp-testspeed {model} --nworld {n} --nstep 1000 2>&1 | tail -40"
        )
        m = re.search(r"[Tt]otal steps per second:\s*([0-9.,e+]+)", out)
        sps = m.group(1).replace(",", "") if m else "unknown"
        print(f"RESULT model={name} worlds={n} steps_per_second={sps} wall={time.time() - t:.0f}s", flush=True)


main()
