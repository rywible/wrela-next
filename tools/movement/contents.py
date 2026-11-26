"""What the imported capture holds (docs/plans/bodies-and-movement.md, phase 2 step 6): the hours
of each kind of movement, from each clip's record (`.movement/poses/BODY/SOURCE/CLIP.json`, which
tools/movement/capture writes), by set (tools/movement/split), and how the retarget went.

    python3 tools/movement/contents.py [BODY]

100STYLE names its takes: forward, backward and sideways walks and runs, idling, and turns.
"""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KINDS = {"FW": "walk forward", "FR": "run forward", "BW": "walk backward", "BR": "run backward",
         "SW": "walk sideways", "SR": "run sideways", "ID": "stand (idle)", "TR1": "turns",
         "TR2": "turns", "TR3": "turns"}


def main():
    body = sys.argv[1] if len(sys.argv) > 1 else "creature"
    sets = {}
    for s in ("training", "test", "calibration"):
        for l in open(os.path.join(ROOT, "tools/movement/split", f"{s}.txt")):
            l = l.strip()
            if l and not l.startswith("#"):
                sets[l] = s
    rows = {}
    by_source = {}
    total = 0.0
    out_frames, frames, worst = 0, 0, -1e9
    for path in sorted(glob.glob(os.path.join(ROOT, ".movement/poses", body, "*", "*.json"))):
        r = json.load(open(path))
        src = r["source"]
        name = f"{src}/{os.path.basename(path)[:-5]}"
        kind = KINDS.get(name.rsplit("_", 1)[-1], "other") if src == "100style" else \
            "gestures while standing (stories told)" if src == "embm" else "other"
        h = r["seconds"] / 3600.0
        which = sets.get(name, "none")
        rows.setdefault(kind, {}).setdefault(which, 0.0)
        rows[kind][which] += h
        s = by_source.setdefault(src, {"clips": 0, "hours": 0.0, "mm": 0.0})
        s["clips"] += 1
        s["hours"] += h
        s["mm"] += r["mean_joint_mm"] * r["frames"]
        s["frames"] = s.get("frames", 0) + r["frames"]
        frames += r["frames"]
        out_frames += r["frames_out_of_range"]
        worst = max(worst, r["worst_range"])
        total += h
    print(f"| Movement | Training (h) | Test (h) | Calibration (h) | All (h) |")
    print(f"| --- | --- | --- | --- | --- |")
    for kind in sorted(rows):
        r = rows[kind]
        a = sum(r.values())
        print(f"| {kind} | {r.get('training', 0):.2f} | {r.get('test', 0):.2f} | "
              f"{r.get('calibration', 0):.2f} | {a:.2f} |")
    print(f"| All | | | | {total:.2f} |")
    print()
    for src, s in by_source.items():
        print(f"{src}: {s['clips']} clips, {s['hours']:.2f} h, the mean distance of a mapped joint from "
              f"the scaled capture's {s['mm'] / max(s['frames'], 1):.1f} mm")
    print(f"frames out of range by more than 0.01 rad: {out_frames} of {frames} "
          f"({100 * out_frames / max(frames, 1):.3f}%), the worst {worst:.4f} rad")


main()
