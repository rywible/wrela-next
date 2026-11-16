#!/usr/bin/env python3
"""Median serial time of chosen pass labels over frames A-B (the e-graph spike's measure).

    tools/egg/passtime.py <build> [--size WxH] [--frames A-B] [--labels a,b] [-- env K=V ...]

Runs tools/gpu <build> --timestamps --serial and prints each label's median (ms) over the frames,
summing a label's entries within a frame."""
import os, re, statistics, subprocess, sys
from collections import defaultdict
TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
env = dict(os.environ)
if "--" in args:
    i = args.index("--")
    for kv in args[i + 1:]:
        k, v = kv.split("=", 1)
        env[k] = v
    args = args[:i]
build = args[0]
size, frames, labels = "3840x2160", "300-419", None
k = 1
while k < len(args):
    if args[k] == "--size": size = args[k + 1]; k += 2
    elif args[k] == "--frames": frames = args[k + 1]; k += 2
    elif args[k] == "--labels": labels = args[k + 1].split(","); k += 2
    else: sys.exit(f"unknown {args[k]}")
a, b = map(int, frames.split("-"))
out = subprocess.run([os.path.join(TOOLS, "gpu"), "--timeout", "600", build, "--frames", str(b + 1),
                      "--size", size, "--timestamps", "--serial"], env=env, capture_output=True, text=True)
if out.returncode != 0:
    sys.exit(out.stderr[-3000:])
per = defaultdict(lambda: defaultdict(float))
for line in (out.stdout + out.stderr).splitlines():
    m = re.match(r"frame\s+(\d+)\s+(\S+)\s+([\d.]+) us$", line.strip())
    if m and a <= int(m.group(1)) <= b:
        per[m.group(2)][int(m.group(1))] += float(m.group(3)) / 1000
for lab in sorted(per):
    if labels and lab not in labels: continue
    xs = [per[lab].get(f, 0.0) for f in range(a, b + 1)]
    print(f"{lab:12s} median {statistics.median(xs):.3f} ms  [{min(xs):.3f}, {max(xs):.3f}]")
