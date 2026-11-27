"""The owner's preference (docs/plans/bodies-and-movement.md, phase 5, "The owner's preference,
learned", step 1): two films of the same goal side by side, in an order the owner is not told,
and the owner's choice of the one that looks more alive, kept in git with the two films' records.

    python3 tools/movement/prefer.py make NAME BODY A=ROOT:CLIP B=ROOT:CLIP [--from S] [--seconds S]
        films each movement (tools/movement/film: the body from the side and the front, the
        judge's contacts and residual drawn) and puts them side by side, in a random order, in
        `.movement/prefer/NAME/pair.mp4` with a page to watch it (`index.html`); the order and
        each film's record (its poses' root and clip, the commit) in `.movement/prefer/NAME/pair.json`
    python3 tools/movement/prefer.py choose NAME left|right [--note TEXT]
        the owner's choice: one line in `tools/movement/preferences.jsonl` (git keeps it) with
        both films' records and which one was chosen
    python3 tools/movement/prefer.py count
        the choices so far: how often each kind of film (capture, a solver) was chosen

ROOT is a folder of poses as the capture tool writes them (`.movement/poses`, the filtered
capture's `.movement/poses-filtered`, the solver's `.movement/solver/goals`); CLIP is its
SOURCE/CLIP. The films' pictures are drawn on the CPU (a few seconds each) and joined by ffmpeg.
"""
import argparse
import datetime
import json
import os
import random
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREFER = os.path.join(ROOT, ".movement", "prefer")
CHOICES = os.path.join(ROOT, "tools", "movement", "preferences.jsonl")
FPS = 15


def film(body, root, clip, start, seconds, out):
    """The film tool's pictures of `clip` (every second frame: 15 a second) into `out`."""
    work = os.path.join(out, "frames")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    cmd = [os.path.join(ROOT, "tools", "wrela"), "tool", os.path.join(ROOT, "tools", "movement", "film"),
           "--read", f"poses={root}", "--write", f"out={work}", "--", body, clip, str(start), str(seconds), "2"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    pictures = os.path.join(work, clip)
    if r.returncode != 0 or not os.path.isdir(pictures):
        sys.exit(f"prefer: the film of {root}:{clip} failed:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return pictures


def commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def make(a):
    out = os.path.join(PREFER, a.name)
    os.makedirs(out, exist_ok=True)
    films = []
    for spec in (a.a, a.b):
        key, _, rest = spec.partition("=")
        root, _, clip = rest.rpartition(":")
        films.append({"label": key, "root": root, "clip": clip, "from": a.start, "seconds": a.seconds})
    order = [0, 1]
    random.SystemRandom().shuffle(order)
    sides = []
    for side, i in zip(("left", "right"), order):
        f = films[i]
        pictures = film(a.body, f["root"], f["clip"], a.start, a.seconds, os.path.join(out, side))
        sides.append(pictures)
        f["side"] = side
    # Side by side, one above the other's height: each film 512 by 256 (side and front).
    video = os.path.join(out, "pair.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-framerate", str(FPS), "-i", os.path.join(sides[0], "%04d.png"),
           "-framerate", str(FPS), "-i", os.path.join(sides[1], "%04d.png"),
           "-filter_complex", "[0][1]vstack=inputs=2,pad=iw:ih+8:0:4:white",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", video]
    subprocess.run(cmd, check=True)
    record = {"name": a.name, "body": a.body, "made": datetime.datetime.now().isoformat(timespec="seconds"),
              "commit": commit(), "films": films}
    with open(os.path.join(out, "pair.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    with open(os.path.join(out, "index.html"), "w") as fh:
        fh.write(f"""<!doctype html><meta charset="utf-8"><title>Which looks more alive</title>
<style>body{{font:16px system-ui;margin:24px;background:#f4f4f2;color:#222}}video{{max-width:100%}}</style>
<p>Two movements of the same goal: the upper film and the lower film. Which looks more alive?</p>
<video src="pair.mp4" controls loop autoplay muted playsinline></video>
<p>Answer with <code>tools/movement/prefer.py choose {a.name} left</code> for the upper film or
<code>right</code> for the lower.</p>
""")
    print(f"prefer: {video} (upper: left, lower: right), {os.path.join(out, 'index.html')}")


def choose(a):
    path = os.path.join(PREFER, a.name, "pair.json")
    record = json.load(open(path))
    chosen = [f for f in record["films"] if f["side"] == a.side][0]
    line = dict(record)
    line.update({"chosen": chosen["label"], "side": a.side, "note": a.note,
                 "chosen_at": datetime.datetime.now().isoformat(timespec="seconds")})
    with open(CHOICES, "a") as fh:
        fh.write(json.dumps(line) + "\n")
    print(f"prefer: {a.name}: the owner chose {chosen['label']} ({chosen['root']}:{chosen['clip']})")


def count(_):
    if not os.path.exists(CHOICES):
        print("prefer: no choices yet")
        return
    tally = {}
    for line in open(CHOICES):
        c = json.loads(line)
        for f in c["films"]:
            w, n = tally.get(f["label"], (0, 0))
            tally[f["label"]] = (w + int(f["label"] == c["chosen"]), n + 1)
    for k, (w, n) in sorted(tally.items()):
        print(f"prefer: {k}: chosen {w} of {n}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="what", required=True)
    m = sub.add_parser("make")
    m.add_argument("name")
    m.add_argument("body", choices=["human", "creature"])
    m.add_argument("a", help="LABEL=ROOT:SOURCE/CLIP")
    m.add_argument("b", help="LABEL=ROOT:SOURCE/CLIP")
    m.add_argument("--from", dest="start", type=float, default=0.0)
    m.add_argument("--seconds", type=float, default=8.0)
    c = sub.add_parser("choose")
    c.add_argument("name")
    c.add_argument("side", choices=["left", "right"])
    c.add_argument("--note", default="")
    sub.add_parser("count")
    a = p.parse_args()
    {"make": make, "choose": choose, "count": count}[a.what](a)


if __name__ == "__main__":
    main()
