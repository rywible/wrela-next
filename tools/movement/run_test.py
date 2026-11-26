"""The small run that examines the run tool (docs/plans/bodies-and-movement.md, 0.5, step 5).

    tools/movement/colab-run start NAME --script tools/movement/run_test.py --hours 0.25 \
        [-- --steps 600 --pace 1.0 --warp]

A deterministic training of a small net on the GPU: step k trains on a batch made from the seed
and k alone, so a run that continues from a checkpoint ends with the same weights, to the bit, as
a run that did not stop. Each step is paced to `--pace` seconds, so 600 steps are 10 minutes and
`--every 60` writes a checkpoint each minute. With `--warp` it first installs MuJoCo Warp and
steps its benchmark humanoid, as a run of phase 4 starts (the cold start of 0.5's table).

Its result: the step, the last loss and the SHA-256 of the weights.
"""

import argparse
import hashlib
import os
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import runlib  # noqa: E402
import torch  # noqa: E402


def main():
    run = runlib.Run()
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--pace", type=float, default=1.0)
    p.add_argument("--warp", action="store_true")
    a = p.parse_args(run.args)
    print(f"gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
          flush=True)
    if a.warp:
        t = time.time()
        subprocess.run("pip install -q mujoco-warp 2>&1 | tail -2", shell=True)
        print(f"install {time.time() - t:.0f} s", flush=True)
        t = time.time()
        import mujoco
        import mujoco_warp as mjw
        import warp as wp
        xml = """<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.1"/>
                 </body><geom type="plane" size="5 5 .1"/></worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        wm = mjw.put_model(m)
        wd = mjw.put_data(m, d, nworld=256)
        for _ in range(10):
            mjw.step(wm, wd)
        wp.synchronize()
        print(f"first steps of MuJoCo Warp (compile) {time.time() - t:.0f} s", flush=True)
    torch.use_deterministic_algorithms(True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(run.seed)
    net = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.ELU(), torch.nn.Linear(256, 256),
                              torch.nn.ELU(), torch.nn.Linear(256, 64)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    step = 0
    loss_v = None
    state = run.resume()
    if state is not None:
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        loss_v = state["loss"]
    target = torch.nn.Linear(64, 64).to(dev)
    with torch.no_grad():
        g = torch.Generator().manual_seed(run.seed * 7 + 1)
        target.weight.copy_(torch.randn(64, 64, generator=g))
        target.bias.copy_(torch.randn(64, generator=g))
    while step < a.steps:
        t = time.time()
        g = torch.Generator().manual_seed(run.seed * 1000003 + step)
        x = torch.randn(4096, 64, generator=g).to(dev)
        with torch.no_grad():
            y = torch.sin(target(x))
        loss = torch.nn.functional.mse_loss(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        loss_v = float(loss.detach())
        run.progress(step, a.steps, f"loss {loss_v:.6f}")
        if run.due() or step == a.steps:
            run.checkpoint(step, {"net": {k: v.cpu() for k, v in net.state_dict().items()},
                                  "opt": opt.state_dict(), "step": step, "loss": loss_v})
        if run.should_stop():
            run.abandon()
            return
        rest = a.pace - (time.time() - t)
        if rest > 0:
            time.sleep(rest)
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    run.finish({"step": step, "loss": loss_v, "weights_sha256": h.hexdigest()})


main()
