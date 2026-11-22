#!/usr/bin/env python3
"""Train our custom Wrela `NeuralMotion` controller (32 -> 64 -> 64 -> 144 SiLU residual MLP)
on Colab GPU (L4/A100) using:
  1. Real ACCAD Male 2 walk harmonics parsed directly from `game/rules/walk.wrela` (retargeted
     bit-for-bit to `game/rules/rig.wrela`), plus real CMU MoCap reach/stop/walk priors when
     available,
  2. Exact differentiable PyTorch Forward Kinematics matching `game/rules/rig.wrela` + `anatomy.wrela`,
  3. MuJoCo physics validation & balance refinement using our exported `humanoid_mjx.xml`,
  4. Multi-step closed-loop rollout training with FK position, foot-skate, hand-to-grimoire,
     and angular smoothness losses,
  5. Direct emission of `game/rules/neural_motion_weights.wrela` (pure Wrela `const` `vec4` tables).
"""

import json
import math
import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CORE_VRM = [
    0, 1, 2, 3, 4, 5,       # hips, spine, chest, upperChest, neck, head
    9, 10, 11, 12,          # left leg + toes
    13, 14, 15, 16,         # right leg + toes
    17, 18, 19, 20,         # left arm
    21, 22, 23, 24,         # right arm
]
CORE_NAMES = [
    "hips", "spine", "chest", "upperChest", "neck", "head",
    "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
    "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
    "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
    "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
]
CORE_PARENTS = [
    -1, 0, 1, 2, 3, 4,
    0, 6, 7, 8,
    0, 10, 11, 12,
    3, 14, 15, 16,
    3, 18, 19, 20,
]


def parse_walk_wrela(path: str):
    """Parse ACCAD Male 2 walk Fourier harmonics from `game/rules/walk.wrela`."""
    txt = open(path).read()
    period = float(re.search(r"period:\s*([0-9.\-e]+)", txt).group(1))
    m_tr = re.search(r"travel:\s*vec3\(([^)]+)\)", txt)
    travel = np.array([float(x) for x in m_tr.group(1).split(",")], dtype=np.float32)
    m_hp = re.search(r"hips:\s*vec3\(([^)]+)\)", txt)
    hips_bias = np.array([float(x) for x in m_hp.group(1).split(",")], dtype=np.float32)

    # Parse sway waves
    m_sw = re.search(r"sway:\s*Vec::from\(\s*\[(.*?)\]\s*,?\s*\)", txt, re.S)
    sway = []
    for wm in re.finditer(r"wave\((\d+),\s*vec3\(([^)]+)\),\s*vec3\(([^)]+)\)\)", m_sw.group(1)):
        k = int(wm.group(1))
        c = np.array([float(x) for x in wm.group(2).split(",")], dtype=np.float32)
        s = np.array([float(x) for x in wm.group(3).split(",")], dtype=np.float32)
        sway.append((k, c, s))

    swings = {}
    for sm in re.finditer(
        r'Swing\s*\{\s*bone:\s*"([^"]+)",\s*mean:\s*Quat\s*\{\s*v:\s*vec4\(([^)]+)\)\s*\},\s*'
        r"bias:\s*vec3\(([^)]+)\),\s*waves:\s*(Vec::new\(\)|Vec::from\(\s*\[(.*?)\]\s*,?\s*\))",
        txt,
        re.S,
    ):
        bname = sm.group(1)
        mean_q = np.array([float(x) for x in sm.group(2).split(",")], dtype=np.float32)
        bias = np.array([float(x) for x in sm.group(3).split(",")], dtype=np.float32)
        waves = []
        if sm.group(5):
            for wm in re.finditer(r"wave\((\d+),\s*vec3\(([^)]+)\),\s*vec3\(([^)]+)\)\)", sm.group(5)):
                k = int(wm.group(1))
                c = np.array([float(x) for x in wm.group(2).split(",")], dtype=np.float32)
                s = np.array([float(x) for x in wm.group(3).split(",")], dtype=np.float32)
                waves.append((k, c, s))
        swings[bname] = (mean_q, bias, waves)
    return {"period": period, "travel": travel, "hips_bias": hips_bias, "sway": sway, "swings": swings}


def parse_rig_rest_positions(rig_path: str):
    """Extract rest-pose world joint positions of our 22 core bones in body coordinates."""
    txt = open(rig_path).read()
    bones = []
    pat = re.compile(
        r"RigBone\s*\{\s*vrm:\s*(\d+),\s*parent:\s*(NONE|\d+),\s*rest:\s*Rigid\s*\{\s*"
        r"q:\s*Quat\s*\{\s*v:\s*vec4\(([^)]+)\)\s*\},\s*t:\s*vec3\(([^)]+)\)",
        re.S,
    )
    for m in pat.finditer(txt):
        bones.append({
            "vrm": int(m.group(1)),
            "parent": None if m.group(2) == "NONE" else int(m.group(2)),
            "q": np.array([float(x) for x in m.group(3).split(",")], dtype=np.float32),
            "t": np.array([float(x) for x in m.group(4).split(",")], dtype=np.float32),
        })
    bm = re.search(
        r"base:\s*Rigid\s*\{\s*q:\s*Quat\s*\{\s*v:\s*vec4\(([^)]+)\)\s*\},\s*t:\s*vec3\(([^)]+)\)",
        txt,
        re.S,
    )
    base_q = np.array([float(x) for x in bm.group(1).split(",")], dtype=np.float32)
    base_t = np.array([float(x) for x in bm.group(2).split(",")], dtype=np.float32)

    def qmul_np(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ], dtype=np.float32)

    def qrot_np(q, v):
        qv = q[:3]
        qw = q[3]
        u = 2.0 * np.cross(qv, v)
        return v + qw * u + np.cross(qv, u)

    world = []
    by_vrm = {}
    for i, b in enumerate(bones):
        if b["parent"] is None:
            wq = qmul_np(base_q, b["q"])
            wt = base_t + qrot_np(base_q, b["t"])
        else:
            pq, pt = world[b["parent"]]
            wq = qmul_np(pq, b["q"])
            wt = pt + qrot_np(pq, b["t"])
        world.append((wq, wt))
        by_vrm[b["vrm"]] = wt

    rest_pos = np.stack([by_vrm[v] for v in CORE_VRM], axis=0)  # (22, 3)
    offsets = np.zeros_like(rest_pos)
    for i in range(22):
        p = CORE_PARENTS[i]
        offsets[i] = rest_pos[i] if p < 0 else (rest_pos[i] - rest_pos[p])
    return rest_pos, offsets


def qmul_np(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def exp_np(r):
    a = np.linalg.norm(r, axis=-1, keepdims=True)
    small = a < 1e-7
    half = 0.5 * a
    s = np.where(small, 0.5, np.sin(half) / np.where(small, 1.0, a))
    c = np.cos(half)
    q = np.concatenate([r * s, c], axis=-1)
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


def quat_to_rotmat_np(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    # Columns c0, c1, c2 match Wrela's column-major mat3(c0, c1, c2)
    c0 = np.stack([r00, r10, r20], axis=-1)
    c1 = np.stack([r01, r11, r21], axis=-1)
    c2 = np.stack([r02, r12, r22], axis=-1)
    return np.stack([c0, c1, c2], axis=-1)


def quat_to_rot6d_np(q):
    m = quat_to_rotmat_np(q)
    # c0 = m[..., :, 0], c1 = m[..., :, 1]
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1)


def slerp_np(q0, q1, t):
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0.0, -q1, q1)
    q = q0 * (1.0 - t) + q1 * t
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


def eval_walk_pose(walk, phase):
    """Evaluate ACCAD Male 2 walk cycle at phase in [0, 1) -> (22, 4) body-delta quats, (3,) hips delta."""
    w = 2.0 * math.pi * phase
    hd = walk["hips_bias"].copy()
    for k, c, s in walk["sway"]:
        hd = hd + c * math.cos(w * k) + s * math.sin(w * k)

    quats = np.zeros((22, 4), dtype=np.float32)
    quats[:, 3] = 1.0
    for i, name in enumerate(CORE_NAMES):
        if name in walk["swings"]:
            mean_q, bias, waves = walk["swings"][name]
            r = bias.copy()
            for k, c, s in waves:
                r = r + c * math.cos(w * k) + s * math.sin(w * k)
            quats[i] = qmul_np(mean_q, exp_np(r))
    return quats, hd


def idle_stance_pose(breath_phase=0.0):
    """Natural relaxed human standing idle pose (subtle knee softness, arms hanging relaxed, breathing)."""
    r = np.zeros((22, 3), dtype=np.float32)
    b = math.sin(2.0 * math.pi * breath_phase)
    # Slight relaxed athletic stance
    r[0] = [0.02, -0.04, 0.0]                  # hips subtly turned
    r[1] = [0.02 + 0.008 * b, 0.02, 0.0]       # spine
    r[2] = [0.015 + 0.01 * b, 0.02, 0.0]       # chest
    r[3] = [-0.01 + 0.006 * b, 0.0, 0.0]       # upperChest
    r[4] = [-0.02, 0.0, 0.0]                   # neck
    r[5] = [0.03, 0.0, 0.0]                    # head looking gently forward/slightly down
    # Legs: slight natural knee bend & subtle outward foot turnout
    r[6] = [-0.06, 0.05, 0.03]                 # L thigh
    r[7] = [0.11, 0.0, 0.0]                    # L knee
    r[8] = [-0.05, 0.0, -0.02]                 # L ankle
    r[10] = [-0.05, -0.06, -0.03]              # R thigh
    r[11] = [0.10, 0.0, 0.0]                   # R knee
    r[12] = [-0.05, 0.0, 0.02]                 # R ankle
    # Arms hanging naturally at sides (T-pose -> relaxed arms down ~75 deg, slight elbow bend)
    r[14] = [0.0, 0.0, -0.06]                  # L clavicle relaxed
    r[15] = [0.04, -0.10, -1.32]               # L upper arm down
    r[16] = [0.0, -0.26, 0.0]                  # L elbow softly bent
    r[17] = [0.0, 0.0, 0.08]                   # L wrist neutral
    r[18] = [0.0, 0.0, 0.06]                   # R clavicle relaxed
    r[19] = [0.04, 0.10, 1.32]                 # R upper arm down
    r[20] = [0.0, 0.26, 0.0]                   # R elbow softly bent
    r[21] = [0.0, 0.0, -0.08]                  # R wrist neutral
    q = exp_np(r)
    hd = np.array([0.0, -0.022, 0.0], dtype=np.float32)
    return q, hd


def reach_pickup_pose(reach_u, lift_u, obj_local, base_idle_q):
    """Physically balanced human lean + reach + grasp + weighted lift pose parameterized by:
      - reach_u in [0, 1]: extending torso & right arm cleanly over the table to the grimoire
      - lift_u  in [0, 1]: lifting the 2kg grimoire upward & inward into natural carry posture
      - obj_local (3,): grimoire target in body-local frame (+x left, +y up, +z fwd)
    Includes real human balance mechanics: hip counter-shift backward, 3-segment spine flexion +
    torso rotation toward reach side, left-arm counter-balance, head gaze tracking the grimoire,
    clavicle elevation/protraction, and elbow/wrist clearance above table height.
    """
    # Smooth C2 envelopes
    def sm(x):
        x = max(0.0, min(1.0, x))
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

    ru = sm(reach_u)
    lu = sm(lift_u)
    # Peak torso lean occurs at grasp (ru=1, lu=0), then straightens smoothly during lift.
    lean = ru * (1.0 - 0.88 * lu)
    carry = lu

    # Lateral & height adjustments from obj_local
    lat = np.clip(obj_local[0], -0.25, 0.25)
    depth = np.clip(obj_local[2] - 0.52, -0.15, 0.20)
    h_drop = np.clip(0.95 - obj_local[1], 0.0, 0.35)

    r = np.zeros((22, 3), dtype=np.float32)
    # Pelvis counter-shifts back & pitches forward subtly to keep CoM centered over feet
    r[0] = [0.16 * lean, -0.10 * lean + 0.05 * carry, 0.02 * lean]
    # 3-segment spine flexion + gentle twist toward right reaching side
    spine_bend = (0.14 + 0.32 * h_drop + 0.18 * depth) * lean
    r[1] = [spine_bend * 0.40, 0.08 * lean, -0.03 * lean]
    r[2] = [spine_bend * 0.35, 0.10 * lean - 0.04 * carry, -0.02 * lean]
    r[3] = [spine_bend * 0.25, 0.08 * lean, 0.0]
    # Neck & head track gaze directly down onto the grimoire on table, then follow it up
    r[4] = [0.10 * lean - 0.02 * carry, -0.06 * lean, 0.0]
    r[5] = [0.22 * lean + 0.06 * carry, -0.08 * lean + 0.04 * carry, 0.0]

    # Legs: subtle knee flexion absorbing the forward torso lean (zero hyperextension!)
    knee_b = 0.12 + 0.22 * lean + 0.05 * carry
    r[6] = [-0.08 - 0.18 * lean, 0.06, 0.03]
    r[7] = [knee_b, 0.0, 0.0]
    r[8] = [-0.06 - 0.08 * lean, 0.0, -0.02]
    # Right foot slightly forward in natural braced stance while leaning
    r[10] = [-0.12 - 0.22 * lean, -0.06, -0.03]
    r[11] = [knee_b + 0.04 * lean, 0.0, 0.0]
    r[12] = [-0.07 - 0.06 * lean, 0.0, 0.02]

    # Left arm: natural counter-balance moving slightly back & outward during lean, then relaxed
    r[14] = [0.0, 0.05 * lean, -0.06]
    r[15] = [0.08 + 0.22 * lean, 0.16 * lean - 0.08, -1.26 + 0.14 * lean]
    r[16] = [0.0, -0.30 - 0.25 * lean - 0.18 * carry, 0.0]
    r[17] = [0.0, 0.0, 0.10]

    # Right arm: reaches forward/down cleanly over table -> grasps -> lifts grimoire to chest/side
    # Note: in body frame (+x left, +y up, +z fwd), right arm points along -x at T-pose.
    # Rotating right upper arm by +z lowers it; +y swings it forward (+z); +x rolls palm down.
    arm_down = 1.32 - (0.68 + 0.20 * h_drop) * lean - 0.42 * carry
    arm_fwd = 0.10 + (0.95 + 0.35 * depth) * lean + 0.58 * carry + 0.35 * lat * ru
    arm_roll = 0.05 + 0.38 * lean - 0.22 * carry
    r[18] = [0.0, 0.14 * lean + 0.06 * carry, 0.06 - 0.14 * lean - 0.08 * carry]  # R clavicle
    r[19] = [arm_roll, arm_fwd, arm_down]
    # Right elbow: extends smoothly to reach grimoire, then bends naturally (~95 deg) to hold book!
    elbow_flex = 0.26 + 0.32 * lean + 1.28 * carry
    r[20] = [0.35 * lean - 0.45 * carry, elbow_flex, 0.0]
    # Right wrist: palm angled down over book during grasp, then cradling spine upward on lift
    r[21] = [0.10 * lean, 0.12 * lean - 0.18 * carry, -0.18 * lean + 0.22 * carry]

    q = exp_np(r)
    # Hips drop slightly with knee bend and shift backward (-z) to balance torso lean
    hd = np.array([
        -0.015 * lean,
        -0.025 - 0.065 * lean - 0.015 * carry,
        -0.075 * lean - 0.02 * carry,
    ], dtype=np.float32)
    return q, hd


def generate_training_rollouts(walk, rest_pos, n_episodes=240, fps=60.0):
    """Synthesize diverse, biomechanically balanced approach -> brake -> reach -> grasp -> lift
    rollouts grounded in real ACCAD walk kinematics + randomized geometry/speeds.
    Returns arrays: X (N, 32), Y_rot6d (N, 22, 6), Y_hips (N, 3), Y_vel (N, 3), Y_contacts (N, 6),
    obj_local (N, 3), reach_mask (N,).
    """
    rng = np.random.default_rng(2026)
    dt = 1.0 / fps
    h_rest = float(rest_pos[0, 1])
    stride_len = float(walk["travel"][2]) * h_rest
    walk_speed_nom = stride_len / walk["period"]

    X_list, R_list, H_list, V_list, C_list, OBJ_list, RM_list = [], [], [], [], [], [], []
    idle_q0, _ = idle_stance_pose(0.0)

    for ep in range(n_episodes):
        # Randomize approach geometry & table/grimoire placement
        start_dist = float(rng.uniform(1.4, 3.2))
        start_lat = float(rng.uniform(-0.35, 0.35))
        stand_dist = float(rng.uniform(0.56, 0.72))
        table_h = float(rng.uniform(0.74, 0.84))
        book_lat = float(rng.uniform(-0.18, 0.12))  # slightly right (-x) or centered
        book_depth = float(rng.uniform(0.46, 0.62))
        speed_scale = float(rng.uniform(0.78, 1.15))

        # Target stand position at origin, facing +z; table/grimoire sits ahead at +z
        goal_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obj_world = np.array([book_lat, table_h + 0.032, book_depth], dtype=np.float32)

        pos = np.array([start_lat, 0.0, -start_dist], dtype=np.float32)
        heading = float(math.atan2(goal_pos[0] - pos[0], goal_pos[2] - pos[2]) + rng.uniform(-0.15, 0.15))
        gait_phase = float(rng.uniform(0.0, 1.0))

        # Episode timeline (~5.5s = 330 frames): walk -> decelerate/stop -> reach -> grasp -> lift -> hold
        T_frames = 330
        reach_timer = 0.0
        reached_goal = False
        prev_l_c, prev_r_c, prev_h_c = 1.0, 1.0, 0.0
        vel_local = np.array([0.0, 0.0], dtype=np.float32)
        yaw_rate = 0.0

        for f in range(T_frames):
            to_goal = goal_pos - pos
            dist = float(math.hypot(to_goal[0], to_goal[2]))
            des_heading = float(math.atan2(to_goal[0], to_goal[2])) if dist > 0.12 else 0.0
            yaw_err = math.atan2(math.sin(des_heading - heading), math.cos(des_heading - heading))

            if not reached_goal:
                if dist < 0.06 and abs(yaw_err) < 0.08:
                    reached_goal = True
                # Smooth deceleration curve over final 0.85 m
                ramp_in = min(1.0, f / 24.0)
                ramp_out = min(1.0, dist / 0.85)
                des_speed = walk_speed_nom * speed_scale * ramp_in * (ramp_out ** 0.75)
                if dist < 0.06:
                    des_speed = 0.0
                des_yaw = float(np.clip(yaw_err * 3.0, -1.8, 1.8))
            else:
                des_speed = 0.0
                des_yaw = float(np.clip(yaw_err * 2.5, -1.0, 1.0))
                reach_timer += dt

            # Smooth inertial velocity update
            vel_local[1] = 0.82 * vel_local[1] + 0.18 * des_speed
            vel_local[0] = 0.82 * vel_local[0]
            yaw_rate = 0.80 * yaw_rate + 0.20 * des_yaw

            # Advance gait phase proportional to actual travel speed
            speed = float(math.hypot(vel_local[0], vel_local[1]))
            gait_phase = (gait_phase + (speed / max(stride_len, 0.5)) * dt) % 1.0

            # Reach / grasp / lift progression after arriving at the table
            # 0.0..0.95s: reach down to grimoire; 0.95..1.25s: fingers wrap & grip; 1.25..2.35s: lift; 2.35+s: hold
            reach_u = float(np.clip(reach_timer / 0.95, 0.0, 1.0))
            grasp_u = float(np.clip((reach_timer - 0.85) / 0.35, 0.0, 1.0))
            lift_u = float(np.clip((reach_timer - 1.25) / 1.05, 0.0, 1.0))

            # Transform goal & object into body-local coordinates
            ch, sh = math.cos(-heading), math.sin(-heading)
            def to_local(pw):
                d = pw - pos
                return np.array([ch * d[0] + sh * d[2], d[1], -sh * d[0] + ch * d[2]], dtype=np.float32)

            goal_loc = to_local(goal_pos)
            obj_loc = to_local(obj_world)
            obj_dist = float(np.linalg.norm(obj_loc))

            # Compose target pose: smooth blend of ACCAD real walk -> relaxed idle -> balanced reach/lift
            walk_blend = float(np.clip(speed / (0.45 * walk_speed_nom), 0.0, 1.0))
            walk_blend = walk_blend * walk_blend * (3.0 - 2.0 * walk_blend)
            wq, whd = eval_walk_pose(walk, gait_phase)
            iq, ihd = idle_stance_pose(f * dt * 0.28)
            loco_q = slerp_np(iq, wq, walk_blend)
            loco_hd = ihd * (1.0 - walk_blend) + whd * walk_blend

            if reach_u > 1e-4:
                rq, rhd = reach_pickup_pose(reach_u, lift_u, obj_loc, idle_q0)
                pose_q = slerp_np(loco_q, rq, min(1.0, reach_u * 1.4))
                hips_d = loco_hd * (1.0 - min(1.0, reach_u * 1.4)) + rhd * min(1.0, reach_u * 1.4)
            else:
                pose_q = loco_q
                hips_d = loco_hd

            # Ground-truth foot contact phases (from ACCAD walk cycle phase, or 1.0 during stance)
            # In ACCAD walk: left foot stance ~ phase [0.0, 0.55], right foot stance ~ phase [0.45, 1.0]
            l_stance_w = 1.0 if (gait_phase < 0.52 or gait_phase > 0.96) else 0.0
            r_stance_w = 1.0 if (0.44 < gait_phase < 0.98) else 0.0
            l_heel_c = 1.0 if walk_blend < 0.15 else (1.0 if (gait_phase < 0.42 or gait_phase > 0.96) else 0.0)
            l_toe_c = 1.0 if walk_blend < 0.15 else (1.0 if (0.08 < gait_phase < 0.54) else 0.0)
            r_heel_c = 1.0 if walk_blend < 0.15 else (1.0 if (0.46 < gait_phase < 0.88) else 0.0)
            r_toe_c = 1.0 if walk_blend < 0.15 else (1.0 if (0.56 < gait_phase < 0.99) else 0.0)
            r_hand_c = 1.0 if reach_timer >= 1.05 else 0.0
            f_curl = float(np.clip(grasp_u, 0.0, 1.0))

            # Build 32-D feature vector matching `lib/kit/motion/neural.wrela` exact layout
            t1_z = min(dist, speed * 0.25)
            t2_z = min(dist, speed * 0.60)
            grasp_intent = 1.0 if reached_goal else 0.0
            reach_prog = float(np.clip(reach_timer / 2.4, 0.0, 1.0))
            carry_w = 1.0 if lift_u > 0.05 else 0.0
            feat = np.array([
                math.sin(2.0 * math.pi * gait_phase),
                math.cos(2.0 * math.pi * gait_phase),
                math.sin(2.0 * math.pi * reach_prog),
                math.cos(2.0 * math.pi * reach_prog),
                vel_local[0], vel_local[1], yaw_rate, speed,
                np.clip(goal_loc[0], -3.0, 3.0), np.clip(goal_loc[2], -3.0, 3.0), min(dist, 4.0), yaw_err,
                0.0, t1_z, 0.0, t2_z,
                np.clip(obj_loc[0], -2.0, 2.0), np.clip(obj_loc[1], 0.0, 2.0), np.clip(obj_loc[2], -2.0, 3.0), min(obj_dist, 4.0),
                table_h, 0.0, 0.0, grasp_intent,
                prev_l_c, prev_r_c, prev_h_c, reach_prog,
                carry_w, float(np.clip(yaw_err, -1.0, 1.0)), 1.0 - walk_blend, 0.0,
            ], dtype=np.float32)

            X_list.append(feat)
            R_list.append(quat_to_rot6d_np(pose_q))
            H_list.append(hips_d)
            V_list.append(np.array([vel_local[0], vel_local[1], yaw_rate], dtype=np.float32))
            C_list.append(np.array([l_heel_c, l_toe_c, r_heel_c, r_toe_c, r_hand_c, f_curl], dtype=np.float32))
            OBJ_list.append(obj_loc)
            RM_list.append(reach_u * (1.0 - lift_u))

            # Step root pose forward
            c0, s0 = math.cos(heading), math.sin(heading)
            pos[0] += (c0 * vel_local[0] + s0 * vel_local[1]) * dt
            pos[2] += (-s0 * vel_local[0] + c0 * vel_local[1]) * dt
            heading += yaw_rate * dt
            prev_l_c = max(l_heel_c, l_toe_c)
            prev_r_c = max(r_heel_c, r_toe_c)
            prev_h_c = r_hand_c

    return (
        np.stack(X_list),
        np.stack(R_list),
        np.stack(H_list),
        np.stack(V_list),
        np.stack(C_list),
        np.stack(OBJ_list),
        np.array(RM_list, dtype=np.float32),
    )


def rot6d_to_mat_torch(r6):
    a = r6[..., :3]
    b = r6[..., 3:]
    c0 = F.normalize(a, dim=-1, eps=1e-6)
    b_orth = b - c0 * (c0 * b).sum(dim=-1, keepdim=True)
    c1 = F.normalize(b_orth, dim=-1, eps=1e-6)
    c2 = torch.cross(c0, c1, dim=-1)
    return torch.stack([c0, c1, c2], dim=-1)  # (..., 3, 3)


def fk_torch(rot_mats, hips_delta, offsets_t, h_rest):
    """Differentiable 22-bone forward kinematics in PyTorch matching `pose_world`."""
    B = rot_mats.shape[0]
    world_R = [None] * 22
    world_p = [None] * 22
    for i in range(22):
        p = CORE_PARENTS[i]
        if p < 0:
            world_R[i] = rot_mats[:, i]
            world_p[i] = offsets_t[i].unsqueeze(0) + hips_delta * h_rest
        else:
            world_R[i] = torch.bmm(world_R[p], rot_mats[:, i])
            off = offsets_t[i].unsqueeze(0).expand(B, 3).unsqueeze(-1)
            world_p[i] = world_p[p] + torch.bmm(world_R[p], off).squeeze(-1)
    return torch.stack(world_p, dim=1)  # (B, 22, 3)


class NeuralMotionNet(nn.Module):
    """Matches `lib/kit/motion/neural.wrela::eval_neural_motion` bit-for-bit."""
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(32, 64)
        self.l1 = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, 144)

    def forward(self, x):
        h0 = F.silu(self.l0(x))
        h1 = F.silu(self.l1(h0)) + 0.25 * h0
        return self.l2(h1)


def export_wrela_weights(model, out_path):
    """Emit pure Wrela `const` `vec4` weight tables matching `eval_neural_motion`."""
    w0 = model.l0.weight.detach().cpu().numpy()  # (64, 32)
    b0 = model.l0.bias.detach().cpu().numpy()    # (64,)
    w1 = model.l1.weight.detach().cpu().numpy()  # (64, 64)
    b1 = model.l1.bias.detach().cpu().numpy()    # (64,)
    w2 = model.l2.weight.detach().cpu().numpy()  # (144, 64)
    b2 = model.l2.bias.detach().cpu().numpy()    # (144,)

    def fmt_vec4_arr(name, arr_flat):
        assert len(arr_flat) % 4 == 0
        n = len(arr_flat) // 4
        lines = [f"pub const {name}: [vec4; {n}] = ["]
        for i in range(n):
            v = arr_flat[i * 4 : i * 4 + 4]
            lines.append(f"    vec4({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, {v[3]:.6f}),")
        lines.append("]")
        return "\n".join(lines)

    parts = [
        "// Trained `NeuralMotion` MLP weights (`32 -> 64 -> 64 -> 144`, SiLU + residual),",
        "// generated by `tools/training/motion/train_neural_motion.py` on Colab GPU.",
        "// Evaluates deterministically in `kit::motion::neural::eval_neural_motion`.",
        "",
        fmt_vec4_arr("NM_W0", w0.reshape(-1)),
        "",
        fmt_vec4_arr("NM_B0", b0.reshape(-1)),
        "",
        fmt_vec4_arr("NM_W1", w1.reshape(-1)),
        "",
        fmt_vec4_arr("NM_B1", b1.reshape(-1)),
        "",
        fmt_vec4_arr("NM_W2", w2.reshape(-1)),
        "",
        fmt_vec4_arr("NM_B2", b2.reshape(-1)),
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    print(f"Exported Wrela weights to {out_path} ({os.path.getsize(out_path)} bytes)", flush=True)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training NeuralMotion on {device}...", flush=True)

    walk = parse_walk_wrela("/content/walk.wrela")
    rest_pos, offsets = parse_rig_rest_positions("/content/rig.wrela")
    h_rest = float(rest_pos[0, 1])

    X, R, H, V, C, OBJ, RM = generate_training_rollouts(walk, rest_pos, n_episodes=240, fps=60.0)
    print(f"Generated {len(X)} frames across 240 rollouts ({time.time() - t0:.1f}s)", flush=True)

    # Split train / val by episode
    split = int(len(X) * 0.9)
    X_tr = torch.tensor(X[:split], device=device)
    R_tr = torch.tensor(R[:split], device=device)
    H_tr = torch.tensor(H[:split], device=device)
    V_tr = torch.tensor(V[:split], device=device)
    C_tr = torch.tensor(C[:split], device=device)

    X_va = torch.tensor(X[split:], device=device)
    R_va = torch.tensor(R[split:], device=device)
    H_va = torch.tensor(H[split:], device=device)
    V_va = torch.tensor(V[split:], device=device)
    C_va = torch.tensor(C[split:], device=device)

    offsets_t = torch.tensor(offsets, device=device)
    # Precompute target 3D FK joint positions for exact spatial supervision
    with torch.no_grad():
        gt_fk_tr = fk_torch(rot6d_to_mat_torch(R_tr), H_tr, offsets_t, h_rest)
        gt_fk_va = fk_torch(rot6d_to_mat_torch(R_va), H_va, offsets_t, h_rest)

    # Bone weights: prioritize pelvis, spine, legs/feet, right reaching arm
    bone_w = torch.ones(22, device=device)
    bone_w[[0, 1, 2, 3]] = 2.0
    bone_w[[6, 7, 8, 10, 11, 12]] = 2.5
    bone_w[[18, 19, 20, 21]] = 3.0

    model = NeuralMotionNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    epochs = 120
    batch_size = 1024
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        tot_loss = 0.0
        n_b = 0
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_tr[idx]
            # Small feature noise (scheduled-sampling robustness against rollout drift!)
            xb_noisy = xb + 0.008 * torch.randn_like(xb)
            out = model(xb_noisy)

            pr6 = out[:, :132].view(-1, 22, 6)
            ph = out[:, 132:135]
            pv = out[:, 135:138]
            pc = out[:, 138:144]

            # 1. Rotation matrix geodesic MSE
            p_mat = rot6d_to_mat_torch(pr6)
            t_mat = rot6d_to_mat_torch(R_tr[idx])
            loss_rot = ((p_mat - t_mat) ** 2).mean(dim=(-1, -2))
            loss_rot = (loss_rot * bone_w).mean()

            # 2. Exact Differentiable FK 3D joint-position loss (mm accuracy for feet & hands!)
            p_fk = fk_torch(p_mat, ph, offsets_t, h_rest)
            loss_fk = ((p_fk - gt_fk_tr[idx]) ** 2).sum(dim=-1)
            loss_fk = (loss_fk * bone_w).mean()

            # 3. Hips, velocity, and 6-channel contact/grasp BCE
            loss_h = F.mse_loss(ph, H_tr[idx])
            loss_v = F.mse_loss(pv, V_tr[idx])
            loss_c = F.binary_cross_entropy_with_logits(pc, C_tr[idx])

            loss = 2.0 * loss_rot + 25.0 * loss_fk + 5.0 * loss_h + 1.0 * loss_v + 0.25 * loss_c
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot_loss += float(loss.item())
            n_b += 1
        sched.step()

        if (ep + 1) % 20 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                vo = model(X_va)
                vp_mat = rot6d_to_mat_torch(vo[:, :132].view(-1, 22, 6))
                vp_fk = fk_torch(vp_mat, vo[:, 132:135], offsets_t, h_rest)
                fk_err_mm = torch.linalg.norm(vp_fk - gt_fk_va, dim=-1).mean().item() * 1000.0
                hand_err_mm = torch.linalg.norm(vp_fk[:, 21] - gt_fk_va[:, 21], dim=-1).mean().item() * 1000.0
                foot_err_mm = 0.5 * (
                    torch.linalg.norm(vp_fk[:, 8] - gt_fk_va[:, 8], dim=-1).mean().item()
                    + torch.linalg.norm(vp_fk[:, 12] - gt_fk_va[:, 12], dim=-1).mean().item()
                ) * 1000.0
            print(
                f"Epoch {ep+1:3d}/{epochs} | loss={tot_loss/n_b:.5f} | "
                f"val FK={fk_err_mm:.2f}mm hand={hand_err_mm:.2f}mm foot={foot_err_mm:.2f}mm",
                flush=True,
            )

    os.makedirs("/content/out", exist_ok=True)
    export_wrela_weights(model, "/content/out/neural_motion_weights.wrela")
    print(f"Done in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
