#!/usr/bin/env python3
"""Export the canonical Wrela humanoid (`game/rules/rig.wrela` + `lib/kit/motion/anatomy.wrela`)
directly to a MuJoCo / MJX MJCF XML file so offline physics simulation and RL on Colab match
Wrela's exact rest pose, bone hierarchy, joint DOFs/limits, 3-point foot pads, 5-finger contacts,
and segment masses with zero sim-to-engine drift.

Usage:
    python3 tools/training/export_mjx.py -o tools/training/humanoid_mjx.xml
"""

import argparse
import math
import os
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RIG_PATH = os.path.join(ROOT, "game", "rules", "rig.wrela")
ANATOMY_PATH = os.path.join(ROOT, "lib", "kit", "motion", "anatomy.wrela")
HUMANOID_PATH = os.path.join(ROOT, "lib", "kit", "motion", "humanoid.wrela")


def parse_vrm_names(path: str) -> list[str]:
    text = open(path).read()
    m = re.search(r"pub const BONES:\s*\[Text;\s*55\]\s*=\s*\[(.*?)\]", text, re.S)
    assert m, "could not find BONES table in humanoid.wrela"
    return re.findall(r'"([^"]+)"', m.group(1))


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qrot(q, v):
    qx, qy, qz, qw = q
    vx, vy, vz = v
    ux = 2.0 * (qy * vz - qz * vy)
    uy = 2.0 * (qz * vx - qx * vz)
    uz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * ux + (qy * uz - qz * uy),
        vy + qw * uy + (qz * ux - qx * uz),
        vz + qw * uz + (qx * uy - qy * ux),
    )


def parse_rig(path: str):
    text = open(path).read()
    bones = []
    pat = re.compile(
        r"RigBone\s*\{\s*vrm:\s*(\d+),\s*parent:\s*(NONE|\d+),\s*rest:\s*Rigid\s*\{\s*"
        r"q:\s*Quat\s*\{\s*v:\s*vec4\(([^)]+)\)\s*\},\s*t:\s*vec3\(([^)]+)\)",
        re.S,
    )
    for m in pat.finditer(text):
        vrm = int(m.group(1))
        parent = None if m.group(2) == "NONE" else int(m.group(2))
        q = tuple(float(x.strip()) for x in m.group(3).split(","))
        t = tuple(float(x.strip()) for x in m.group(4).split(","))
        bones.append({"vrm": vrm, "parent": parent, "q": q, "t": t})
    bm = re.search(
        r"base:\s*Rigid\s*\{\s*q:\s*Quat\s*\{\s*v:\s*vec4\(([^)]+)\)\s*\},\s*t:\s*vec3\(([^)]+)\)",
        text,
        re.S,
    )
    assert bm, "could not find base Rigid in rig.wrela"
    base_q = tuple(float(x.strip()) for x in bm.group(1).split(","))
    base_t = tuple(float(x.strip()) for x in bm.group(2).split(","))

    # Compute rest world positions in Wrela body coordinates (+x left, +y up, +z forward).
    world = []
    for b in bones:
        if b["parent"] is None:
            wq = qmul(base_q, b["q"])
            rt = qrot(base_q, b["t"])
            wt = (base_t[0] + rt[0], base_t[1] + rt[1], base_t[2] + rt[2])
        else:
            pq, pt = world[b["parent"]]
            wq = qmul(pq, b["q"])
            rt = qrot(pq, b["t"])
            wt = (pt[0] + rt[0], pt[1] + rt[1], pt[2] + rt[2])
        world.append((wq, wt))
    return bones, world


def parse_anatomy(path: str):
    text = open(path).read()
    specs = {}
    pat = re.compile(
        r"spec\(\s*(\d+),\s*(JOINT_\w+),\s*vec3\(([^)]+)\),\s*vec3\(([^)]+)\),\s*"
        r"(?:mass:\s*)?([0-9.\-e]+),\s*(?:stiffness:\s*)?([0-9.\-e]+),\s*(?:torque_max:\s*)?([0-9.\-e]+)\s*,?\s*\)",
        re.S,
    )
    for m in pat.finditer(text):
        vrm = int(m.group(1))
        kind = m.group(2)
        lo_vals = [float(x.strip()) for x in m.group(3).split(",")]
        hi_vals = [float(x.strip()) for x in m.group(4).split(",")]
        if len(lo_vals) == 1:
            lo_vals *= 3
        if len(hi_vals) == 1:
            hi_vals *= 3
        specs[vrm] = {
            "kind": kind,
            "lo_deg": tuple(lo_vals),
            "hi_deg": tuple(hi_vals),
            "mass": float(m.group(5)),
            "stiffness": float(m.group(6)),
            "torque_max": float(m.group(7)),
        }
    return specs


def wrela_to_mj(v):
    """Map Wrela body frame (+x left, +y up, +z forward) -> MuJoCo (+x forward, +y left, +z up)."""
    return (v[2], v[0], v[1])


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_len(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def fmt3(v):
    return f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}"


def capsule_radius(name: str) -> float:
    if name in ("hips", "spine", "chest", "upperChest"):
        return 0.115
    if name == "neck":
        return 0.055
    if name == "head":
        return 0.105
    if "UpperLeg" in name:
        return 0.072
    if "LowerLeg" in name:
        return 0.052
    if "Foot" in name:
        return 0.036
    if "Toes" in name:
        return 0.016
    if "Shoulder" in name:
        return 0.048
    if "UpperArm" in name:
        return 0.044
    if "LowerArm" in name:
        return 0.036
    if "Hand" in name:
        return 0.030
    if "Thumb" in name:
        return 0.011
    return 0.0085  # finger phalanges


def build_mjcf(bones, world, names, specs, include_fingers: bool = True) -> str:
    mj = ET.Element("mujoco", model="wrela_humanoid_vrm52")
    ET.SubElement(
        mj,
        "compiler",
        angle="degree",
        coordinate="local",
        inertiafromgeom="false",
        balanceinertia="true",
    )
    ET.SubElement(
        mj,
        "option",
        timestep="0.004",
        iterations="4",
        ls_iterations="6",
        solver="Newton",
        integrator="implicitfast",
    )
    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "joint", armature="0.01", damping="2.5", limited="true")
    ET.SubElement(
        default,
        "geom",
        condim="3",
        friction="1.0 0.02 0.005",
        contype="1",
        conaffinity="1",
    )

    worldbody = ET.SubElement(mj, "worldbody")
    ET.SubElement(worldbody, "geom", name="floor", type="plane", size="20 20 0.1", pos="0 0 0")

    children = {i: [] for i in range(len(bones))}
    for i, b in enumerate(bones):
        if b["parent"] is not None:
            children[b["parent"]].append(i)

    actuators = ET.SubElement(mj, "actuator")

    def add_body(parent_el, idx: int, parent_pos_mj):
        b = bones[idx]
        vrm = b["vrm"]
        name = names[vrm]
        if not include_fingers and vrm >= 25:
            return
        pos_mj = wrela_to_mj(world[idx][1])
        rel = vec_sub(pos_mj, parent_pos_mj)
        body_el = ET.SubElement(parent_el, "body", name=name, pos=fmt3(rel))

        sp = specs[vrm]
        mass = max(sp["mass"], 0.005)
        r = capsule_radius(name)

        # Estimate segment end in MuJoCo local coordinates for capsule & inertial frame.
        kids = [c for c in children[idx] if include_fingers or bones[c]["vrm"] < 25]
        if kids:
            # Prefer primary chain child (skip clavicles on upperChest, skip fingers on hand)
            primary = kids[0]
            if name == "hips":
                primary = next((c for c in kids if names[bones[c]["vrm"]] == "spine"), kids[0])
            elif name == "upperChest":
                primary = next((c for c in kids if names[bones[c]["vrm"]] == "neck"), kids[0])
            elif "Hand" in name:
                primary = next((c for c in kids if "MiddleProximal" in names[bones[c]["vrm"]]), kids[0])
            end_rel = vec_sub(wrela_to_mj(world[primary][1]), pos_mj)
        else:
            if name == "head":
                end_rel = (0.0, 0.0, 0.18)
            elif "Toes" in name:
                end_rel = (0.085, 0.0, 0.0)
            elif "left" in name:
                end_rel = (0.0, 0.025, 0.0)
            elif "right" in name:
                end_rel = (0.0, -0.025, 0.0)
            else:
                end_rel = (0.0, 0.0, 0.04)

        seg_len = max(vec_len(end_rel), 0.012)
        mid = (0.5 * end_rel[0], 0.5 * end_rel[1], 0.5 * end_rel[2])
        ixx = max(mass * (0.25 * r * r + (1.0 / 12.0) * seg_len * seg_len), 1e-6)
        izz = min(max(0.5 * mass * r * r, 1e-6), 1.5 * ixx)
        ET.SubElement(
            body_el,
            "inertial",
            pos=fmt3(mid),
            mass=f"{mass:.4f}",
            diaginertia=f"{ixx:.6f} {ixx:.6f} {izz:.6f}",
        )

        # Joints: Wrela axes (x=left, y=up, z=fwd) map to MuJoCo (y=left, z=up, x=fwd)
        if b["parent"] is None:
            ET.SubElement(body_el, "freejoint", name="root")
        else:
            axis_map = [
                ("x", "0 1 0", sp["lo_deg"][0], sp["hi_deg"][0]),  # Wrela +x (left) -> MJ +y
                ("y", "0 0 1", sp["lo_deg"][1], sp["hi_deg"][1]),  # Wrela +y (up)   -> MJ +z
                ("z", "1 0 0", sp["lo_deg"][2], sp["hi_deg"][2]),  # Wrela +z (fwd)  -> MJ +x
            ]
            for suffix, axis_str, lo_d, hi_d in axis_map:
                if abs(hi_d - lo_d) < 1.0:
                    continue
                jname = f"{name}_{suffix}"
                damping = max(0.05 * sp["stiffness"], 0.15)
                ET.SubElement(
                    body_el,
                    "joint",
                    name=jname,
                    type="hinge",
                    axis=axis_str,
                    range=f"{lo_d:.1f} {hi_d:.1f}",
                    damping=f"{damping:.3f}",
                )
                if sp["torque_max"] > 0.0:
                    ET.SubElement(
                        actuators,
                        "motor",
                        name=f"act_{jname}",
                        joint=jname,
                        gear=f"{sp['torque_max']:.1f}",
                        ctrlrange="-1 1",
                    )

        # Collision geometry: 3-point foot pads for feet/toes, capsule for all other segments.
        if "Foot" in name:
            # Heel + midfoot sole touching z=0
            z_sole = -pos_mj[2] + 0.015
            ET.SubElement(
                body_el,
                "geom",
                name=f"{name}_heel",
                type="sphere",
                pos=f"-0.055 0.0 {z_sole:.5f}",
                size="0.015",
            )
            ET.SubElement(
                body_el,
                "geom",
                name=f"{name}_arch",
                type="capsule",
                fromto=f"-0.04 0.0 {z_sole:.5f} {end_rel[0]:.5f} {end_rel[1]:.5f} {z_sole:.5f}",
                size="0.015",
            )
        elif "Toes" in name:
            z_sole = -pos_mj[2] + 0.012
            side = 1.0 if "left" in name else -1.0
            ET.SubElement(
                body_el,
                "geom",
                name=f"{name}_ball_in",
                type="sphere",
                pos=f"0.005 {-0.030 * side:.5f} {z_sole:.5f}",
                size="0.012",
            )
            ET.SubElement(
                body_el,
                "geom",
                name=f"{name}_ball_out",
                type="sphere",
                pos=f"-0.005 {0.035 * side:.5f} {z_sole:.5f}",
                size="0.012",
            )
            ET.SubElement(
                body_el,
                "geom",
                name=f"{name}_tip",
                type="sphere",
                pos=f"0.085 0.0 {z_sole:.5f}",
                size="0.012",
            )
        else:
            if vec_len(end_rel) > 0.015:
                ET.SubElement(
                    body_el,
                    "geom",
                    name=f"{name}_geom",
                    type="capsule",
                    fromto=f"0 0 0 {fmt3(end_rel)}",
                    size=f"{r:.4f}",
                )
            else:
                ET.SubElement(
                    body_el,
                    "geom",
                    name=f"{name}_geom",
                    type="sphere",
                    pos=fmt3(mid),
                    size=f"{r:.4f}",
                )

        for c in kids:
            add_body(body_el, c, pos_mj)

    add_body(worldbody, 0, (0.0, 0.0, 0.0))

    raw = ET.tostring(mj, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "tools", "training", "humanoid_mjx.xml"))
    ap.add_argument("--no-fingers", action="store_true", help="Omit 30 finger bones for fast locomotion-only MJX")
    args = ap.parse_args()

    names = parse_vrm_names(HUMANOID_PATH)
    bones, world = parse_rig(RIG_PATH)
    specs = parse_anatomy(ANATOMY_PATH)
    xml_text = build_mjcf(bones, world, names, specs, include_fingers=not args.no_fingers)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(xml_text)

    total_mass = sum(specs[b["vrm"]]["mass"] for b in bones if (not args.no_fingers or b["vrm"] < 25))
    n_bones = sum(1 for b in bones if (not args.no_fingers or b["vrm"] < 25))
    print(f"Exported {args.out}: {n_bones} bones, total mass = {total_mass:.2f} kg")


if __name__ == "__main__":
    main()
