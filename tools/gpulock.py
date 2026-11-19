"""The machine's one GPU lock: one GPU job at a time, the others waiting their turn.

The lock is an advisory `flock` on one file for the whole machine, `~/.wrela/gpu.lock`
(`WRELA_GPU_LOCK` names another). The kernel drops it when its holder exits, however it exits, so
a killed job never leaves it held. The holder writes its pid and command into the file, so a
waiter can say what it waits for.
"""

import fcntl
import os
import sys
import time

PATH = os.environ.get("WRELA_GPU_LOCK") or os.path.expanduser("~/.wrela/gpu.lock")


def take(what):
    """Blocks until this process holds the lock; returns the open file, which holds it until it
    is closed or the process exits. `what` is written into the file for waiters to read."""
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    f = open(PATH, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip() or "another job"
        print(f"gpu: waiting for the GPU, held by {holder}", file=sys.stderr, flush=True)
        began = time.monotonic()
        fcntl.flock(f, fcntl.LOCK_EX)
        print(f"gpu: waited {time.monotonic() - began:.1f} s", file=sys.stderr, flush=True)
    f.seek(0)
    f.truncate()
    f.write(f"pid {os.getpid()}: {what}\n")
    f.flush()
    # What this process runs is under the lock: a tool it starts that drives a GPU build
    # (`std::tool::drive`) goes on without taking it again (`wrela-host`'s drive).
    os.environ["WRELA_GPU_LOCK_HELD"] = str(os.getpid())
    return f
