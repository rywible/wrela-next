"""How fast the JAX version of the simulator (MJX-JAX) steps many bodies on one Colab TPU or GPU
(docs/plans/bodies-and-movement.md, phase 0.4). MuJoCo Warp, which tools/movement/sim_speed.py
measures, runs on NVIDIA GPUs only; this is the version a TPU can run. Usage, from this machine:

    colab run --tpu V5E1 --timeout 2400 tools/movement/sim_speed_jax.py [WORLDS ...]
    colab run --gpu L4 --timeout 2400 tools/movement/sim_speed_jax.py [WORLDS ...]

It installs MJX, says what JAX runs on, and steps WORLDS copies (default 1024 4096) of the same
humanoid that sim_speed.py steps (MuJoCo Warp's benchmark model) and of MJX's own test humanoid,
1000 steps each, with MJX's speed test. It prints one line
`RESULT model=M worlds=N steps_per_second=X jit_s=J` for each. It keeps nothing.
"""

import os
import re
import subprocess
import sys
import time


def sh(cmd, tail=3000):
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout[-tail:], flush=True)
    return p.returncode, p.stdout


def main():
    worlds = [int(a) for a in sys.argv[1:]] or [1024, 4096]
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -2")
    sh("env | grep -iE 'tpu|xla|jax|COLAB_(GPU|TPU)' | head -20; nproc; free -g | head -2")
    sh("pip list 2>/dev/null | grep -iE '^(jax|jaxlib|libtpu|libtpu-nightly|mujoco|mujoco-mjx|tensorflow|torch-xla) '")
    sh("pip install -q mujoco-mjx 2>&1 | tail -3")
    sh("pip list 2>/dev/null | grep -iE '^(jax|jaxlib|mujoco|mujoco-mjx) '")
    sh("python -c \"import jax; print('jax', jax.__version__); print('devices', jax.devices()); "
       "print('backend', jax.default_backend())\" 2>&1 | tail -8")
    sh("rm -rf /content/mujoco_warp && git clone -q --depth 1 "
       "https://github.com/google-deepmind/mujoco_warp.git /content/mujoco_warp")
    code, out = sh("python -c \"import mujoco.mjx, os; print(os.path.dirname(mujoco.mjx.__file__))\"")
    mjx_dir = out.strip().splitlines()[-1]
    sh(f"ls {mjx_dir}/test_data/humanoid 2>&1 | head")
    sh("mjx-testspeed --helpshort 2>&1 | head -50")
    models = [
        ("warp_benchmark", "/content/mujoco_warp/benchmarks/humanoid/humanoid.xml", "/"),
        ("mjx_test", "humanoid/humanoid.xml", None),
    ]
    for name, mjcf, base in models:
        for n in worlds:
            t = time.time()
            flags = f"--mjcf={mjcf} --batch_size={n} --nstep=1000"
            if base:
                flags += f" --base_path={base}"
            code, out = sh(f"mjx-testspeed {flags} 2>&1 | tail -25")
            m = re.search(r"[Tt]otal steps per second:\s*([0-9.,e+]+)", out)
            j = re.search(r"JIT time:\s*([0-9.,e+]+)", out)
            sps = m.group(1).replace(",", "") if m else "unknown"
            jit = j.group(1) if j else "unknown"
            print(f"RESULT model={name} worlds={n} steps_per_second={sps} jit_s={jit} "
                  f"wall={time.time() - t:.0f}s", flush=True)


main()
