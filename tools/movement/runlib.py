"""The part of a run that lives in the Colab VM (docs/plans/bodies-and-movement.md, 0.5).

`tools/movement/colab-run` uploads this file beside a job's script and starts the script as

    python3 -u SCRIPT --out DIR --seed N [--resume CHECKPOINT] [ARGS ...]

in a process of its own, so the job goes on when the connection to this machine is lost. A job
uses it so:

    import runlib
    run = runlib.Run()                   # reads --out, --seed, --resume; starts the heartbeat
    state = run.resume()                 # the checkpoint's object (pickled), or None
    ...
    while not done:
        ...
        run.progress(step, total)        # what the heartbeat and the status say
        if run.due():                    # a checkpoint each `--every` seconds (default 60)
            run.checkpoint(step, state)  # written whole, then renamed: never half a file
        if run.should_stop():            # the owner's lease ran out: checkpoint and leave
            run.checkpoint(step, state); run.abandon(); return
    run.finish({"loss": ...})            # the one result line; a run with none failed

It writes `DIR/status.json` (read by the tool at each poll), `DIR/ckpt/<step>.ckpt` (the tool
downloads the newest one) and prints a line at least every 15 s, because a Colab command that
reads a silent process for 30 s takes it for a failure.
"""

import argparse
import json
import os
import pickle
import sys
import threading
import time

HEARTBEAT = 15.0


class Run:
    def __init__(self, argv=None):
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--resume", default=None)
        p.add_argument("--every", type=float, default=60.0)
        p.add_argument("--lease", type=float, default=1800.0)
        self.opts, self.args = p.parse_known_args(argv if argv is not None else sys.argv[1:])
        self.out = self.opts.out
        self.seed = self.opts.seed
        os.makedirs(os.path.join(self.out, "ckpt"), exist_ok=True)
        self.started = time.time()
        self.last_ckpt_time = time.time()
        self.step = 0
        self.total = 0
        self.note = ""
        self.last_ckpt = None
        self.lock = threading.Lock()
        self.state = "running"
        self.result = None
        self._write_status()
        t = threading.Thread(target=self._beat, daemon=True)
        t.start()

    # ---- what the job calls ----

    def resume(self):
        """The object of the checkpoint the run was started from, or None."""
        path = self.opts.resume
        if not path or not os.path.exists(path):
            print(f"runlib: starting fresh, seed {self.seed}", flush=True)
            return None
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"runlib: resumed from {path}", flush=True)
        return obj

    def progress(self, step, total, note=""):
        with self.lock:
            self.step, self.total, self.note = step, total, note

    def due(self):
        return time.time() - self.last_ckpt_time >= self.opts.every

    def checkpoint(self, step, obj):
        """Writes `obj` as the checkpoint of `step` and removes the older ones but the last."""
        d = os.path.join(self.out, "ckpt")
        name = f"{step:012d}.ckpt"
        tmp = os.path.join(d, name + ".part")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(d, name))
        for old in sorted(os.listdir(d)):
            if old.endswith(".ckpt") and old < name and old != self.last_ckpt:
                os.remove(os.path.join(d, old))
        with self.lock:
            self.last_ckpt = name
            self.step = step
        self.last_ckpt_time = time.time()
        self._write_status()
        print(f"runlib: checkpoint {name}", flush=True)

    def should_stop(self):
        """Whether no owner has read the run for `--lease` seconds: the job checkpoints and ends,
        so a session that its owner lost does not work on for nobody."""
        try:
            age = time.time() - os.path.getmtime(os.path.join(self.out, "lease"))
        except OSError:
            age = time.time() - self.started
        return age > self.opts.lease

    def finish(self, result):
        """The run's result: one line `RESULT {json}` and status `done`."""
        with self.lock:
            self.state = "done"
            self.result = result
        self._write_status()
        print("RESULT " + json.dumps(result, sort_keys=True), flush=True)

    def abandon(self):
        with self.lock:
            self.state = "abandoned"
        self._write_status()
        print("runlib: no owner read the run for the lease; it stops at its checkpoint", flush=True)

    # ---- inside ----

    def _write_status(self):
        with self.lock:
            s = {
                "state": self.state,
                "step": self.step,
                "total": self.total,
                "note": self.note,
                "last_ckpt": self.last_ckpt,
                "seed": self.seed,
                "started": self.started,
                "beat": time.time(),
                "pid": os.getpid(),
                "result": self.result,
            }
        tmp = os.path.join(self.out, "status.json.part")
        with open(tmp, "w") as f:
            json.dump(s, f)
        os.replace(tmp, os.path.join(self.out, "status.json"))

    def _beat(self):
        while True:
            time.sleep(HEARTBEAT)
            with self.lock:
                line = f"beat step {self.step}/{self.total} {self.note} " \
                       f"t={time.time() - self.started:.0f}s"
            print(line, flush=True)
            try:
                self._write_status()
            except OSError:
                pass
