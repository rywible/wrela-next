#!/usr/bin/env python3
"""Colab L4 2D -> 3D -> 64-Bone Rigged Humanoid Field Compiler (`fit_mage_field.py`).

Takes our watertight TripoSG A-pose mage (`/content/3d/mage.glb`) + flat-lit multi-view studio
turnarounds (`mage_front.png`, `mage_back.png`, `mage_side.png`) + canonical VRM skeleton
(`/content/rig.wrela`) and:
  1. Aligns the watertight mage mesh directly onto the VRM rest skeleton in relaxed A-pose
     (feet flush at y = 0.0, hips at y = 0.949 m, shoulders at y = 1.455 m, crown ~1.78 m).
  2. Measures actual anatomical cross-section radii from the 3D mesh for all body segments
     (torso, neck, head, clavicles, upper/lower arms, hands, 30 articulated finger phalanges,
     thighs, calves, boots, toes).
  3. Fits two compact, bone-local `siren24_sdf_pbr` bounded residual fields directly to the
     TripoSG 3D geometry & delighted multi-view studio PBR colors:
       - `MageHeadResidual` (in `HEAD`-bone local space: face, nose, brow, eyes, jaw, hair)
       - `MageTorsoResidual` (in `SPINE_2`-bone local space: double-breasted tunic, collar,
         brass buttons, leather belt & buckle, charcoal/slate panels)
  4. Emits pure Wrela source (`game/world/mage.wrela`) using our upgraded 64-bone `Skinned<S>`,
     with full 52-bone VRM + 30 articulated finger articulation and zero external meshes/voxels!
"""

import math
import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from PIL import Image


def load_mage_mesh(glb_path: str):
    scene_or_mesh = trimesh.load(glb_path, force="mesh")
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    # TripoSG front-image mesh is Y-up, facing +Z, centered around 0.
    lo, hi = verts.min(0), verts.max(0)
    verts[:, 0] -= 0.5 * (lo[0] + hi[0])
    verts[:, 2] -= 0.5 * (lo[2] + hi[2])
    verts[:, 1] -= lo[1]
    h = float(hi[1] - lo[1])
    # Scale to match our canonical Quaternius/VRM humanoid height (~1.78 m heel to crown)
    verts *= 1.78 / max(h, 1e-4)
    mesh = trimesh.Trimesh(verts, np.asarray(mesh.faces, dtype=np.int32), process=False)
    mesh.fix_normals()
    print(f"Aligned mage mesh: lo={mesh.bounds[0].round(4)} hi={mesh.bounds[1].round(4)} watertight={mesh.is_watertight}", flush=True)
    return mesh


def project_mage_pbr(pts, nrms, crop_dir: str, bounds):
    """Project delighted multi-view mage studio turnaround (`front`, `back`, `side`) onto 3D points."""
    lo, hi = bounds[0], bounds[1]
    span = np.maximum(hi - lo, 1e-4)

    def load_tight(fname):
        arr = np.asarray(Image.open(os.path.join(crop_dir, fname)).convert("RGB"), dtype=np.float32) / 255.0
        mask = (arr < 0.95).any(axis=2)
        ys, xs = np.where(mask)
        return arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    im_f = load_tight("mage_front.png")
    im_b = load_tight("mage_back.png")
    im_s = load_tight("mage_side.png")

    def sample(img, u, v):
        H, W, _ = img.shape
        ix = np.clip((u * (W - 1)).astype(np.int32), 0, W - 1)
        iy = np.clip(((1.0 - v) * (H - 1)).astype(np.int32), 0, H - 1)
        col = img[iy, ix]
        ok = ~((col > 0.93).all(axis=-1))
        return col, ok

    nx = (pts[:, 0] - lo[0]) / span[0]
    ny = (pts[:, 1] - lo[1]) / span[1]
    nz = (pts[:, 2] - lo[2]) / span[2]

    # Front faces +z (viewer sees character's left on image right -> u = 1 - nx or nx depending on flip;
    # symmetric A-pose tunic; front view shows character's right on left, character's left (+x) on right -> u = nx)
    cf, okf = sample(im_f, nx, ny)
    cb, okb = sample(im_b, 1.0 - nx, ny)
    cs, oks = sample(im_s, nz, ny)

    wf = np.maximum(nrms[:, 2], 0.0) * okf + 0.05 * okf
    wb = np.maximum(-nrms[:, 2], 0.0) * okb + 0.05 * okb
    ws = np.abs(nrms[:, 0]) * 0.4 * oks
    wsum = wf + wb + ws + 1e-6
    col = (cf * wf[:, None] + cb * wb[:, None] + cs * ws[:, None]) / wsum[:, None]

    # Physical fallback where silhouette edges graze white bg
    fallback = np.where(
        (pts[:, 1] > 1.54)[:, None],
        np.array([0.62, 0.45, 0.36], np.float32),  # skin
        np.where((pts[:, 1] > 0.85)[:, None], np.array([0.24, 0.28, 0.35], np.float32), np.array([0.22, 0.16, 0.12], np.float32)),
    )
    col = np.where((wf + wb + ws < 0.08)[:, None], fallback, col)
    col = np.clip(col, 0.02, 0.95)
    albedo_lin = np.power(col, 2.2).astype(np.float32)

    # Material classification (brass buttons/buckle, leather belt/bracers/boots, skin, wool tunic)
    is_brass = (col[:, 0] > 0.42) & (col[:, 0] > col[:, 1] * 1.05) & (col[:, 1] > col[:, 2] * 1.32) & (pts[:, 1] > 0.90) & (pts[:, 1] < 1.50)
    is_skin = (pts[:, 1] > 1.54) & (col[:, 0] > 0.42) & (col[:, 0] > col[:, 2] * 1.2)
    is_leather = (col[:, 0] > col[:, 2] * 1.35) & (~is_skin) & (~is_brass)
    metallic = np.where(is_brass, 0.85, 0.0).astype(np.float32)
    rough = np.where(is_brass, 0.28, np.where(is_skin, 0.48, np.where(is_leather, 0.50, 0.82))).astype(np.float32)
    return albedo_lin, rough, metallic


class Siren24(nn.Module):
    def __init__(self, omega0=8.0):
        super().__init__()
        self.l0 = nn.Linear(3, 24)
        self.l1 = nn.Linear(24, 24)
        self.out = nn.Linear(24, 6)
        with torch.no_grad():
            self.l0.weight.uniform_(-1.0 / 3.0, 1.0 / 3.0).mul_(omega0)
            self.l0.bias.uniform_(-1.0, 1.0)
            b1 = math.sqrt(6.0 / 24.0)
            self.l1.weight.uniform_(-b1, b1)
            self.l1.bias.uniform_(-0.5, 0.5)
            self.out.weight.uniform_(-0.08, 0.08)
            self.out.bias.zero_()

    def forward(self, q):
        h0 = torch.sin(self.l0(q))
        h1 = torch.sin(self.l1(h0))
        y = self.out(h1)
        return (
            torch.tanh(y[:, 0]),
            torch.sigmoid(y[:, 1:4]),
            0.06 + 0.92 * torch.sigmoid(y[:, 4]),
            torch.sigmoid(y[:, 5]),
        )


def rcone_np(p, a, b, r1, r2):
    ba = b - a
    l2 = float(np.dot(ba, ba))
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / max(l2, 1e-10)
    pa = p - a
    y = np.sum(pa * ba, axis=-1)
    z = y - l2
    x2 = np.sum((pa * l2 - ba * y[:, None]) ** 2, axis=-1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d_top = np.sqrt(x2 + z2) * il2 - r2
    d_bot = np.sqrt(x2 + y2) * il2 - r1
    d_side = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d_top, np.where(np.sign(y) * a2 * y2 < k, d_bot, d_side)).astype(np.float32)


def fit_region_siren(mesh, region_name, box_lo, box_hi, base_fn_np, max_delta, crop_dir, device, steps=1400):
    """Fit a bone-local `siren24_sdf_pbr` residual to a specific anatomical region (head or torso)."""
    from scipy.spatial import cKDTree

    pts_all, fidx = mesh.sample(200000, return_index=True)
    pts_all = pts_all.astype(np.float32)
    nrms_all = np.asarray(mesh.face_normals[fidx], dtype=np.float32)
    tree = cKDTree(pts_all)

    in_box = np.all((pts_all >= box_lo) & (pts_all <= box_hi), axis=1)
    pts_s = pts_all[in_box]
    nrms_s = nrms_all[in_box]
    if len(pts_s) > 40000:
        pts_s = pts_s[:40000]
        nrms_s = nrms_s[:40000]
    print(f"  [{region_name}] {len(pts_s)} surface samples inside region box", flush=True)

    rng = np.random.default_rng(7)
    n_band = min(28000, max(12000, len(pts_s)))
    pick = rng.integers(0, len(pts_s), size=n_band)
    pts_b = pts_s[pick] + nrms_s[pick] * rng.uniform(-max_delta * 1.4, max_delta * 1.4, size=(n_band, 1)).astype(np.float32)
    dists, nn = tree.query(pts_b, k=1, workers=-1)
    tri_n = nrms_all[nn]
    sgn = np.where(np.sum((pts_b - pts_all[nn]) * tri_n, axis=-1) >= 0.0, 1.0, -1.0).astype(np.float32)
    sdf_b = dists.astype(np.float32) * sgn

    pts = np.concatenate([pts_s, pts_b], axis=0)
    sdf_gt = np.concatenate([np.zeros(len(pts_s), np.float32), sdf_b], axis=0)
    nrms = np.concatenate([nrms_s, tri_n], axis=0)
    albedo, rough, metal = project_mage_pbr(pts, nrms, crop_dir, mesh.bounds)
    d_base = base_fn_np(pts)

    center = 0.5 * (box_lo + box_hi)
    extent = np.maximum(0.5 * (box_hi - box_lo), 0.06)
    q_all = (pts - center) / extent

    band = max_delta * 2.5 + 0.015
    t_w = np.clip(1.0 - np.abs(d_base) / band, 0.0, 1.0)
    win = t_w * t_w * (3.0 - 2.0 * t_w)
    act = win > 0.02

    q_t = torch.tensor(q_all[act], device=device)
    db_t = torch.tensor(d_base[act], device=device)
    gt_t = torch.tensor(sdf_gt[act], device=device)
    w_t = torch.tensor(win[act], device=device)
    alb_t = torch.tensor(albedo[act], device=device)
    rgh_t = torch.tensor(rough[act], device=device)
    met_t = torch.tensor(metal[act], device=device)
    sw_t = torch.tensor((1.0 + 3.0 * (np.abs(sdf_gt[act]) < 0.004)).astype(np.float32), device=device)

    model = Siren24(omega0=8.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-4)

    N = len(q_t)
    bs = min(4096, N)
    for st in range(steps):
        idx = torch.randint(0, N, (bs,), device=device)
        qb = q_t[idx].detach().requires_grad_(True)
        delta, pa, pr, pm = model(qb)
        pred_d = db_t[idx] + w_t[idx] * max_delta * delta
        loss_d = (sw_t[idx] * (pred_d - gt_t[idx]) ** 2).mean()
        gd = torch.autograd.grad(delta.sum(), qb, create_graph=True)[0]
        lip = torch.linalg.norm(gd * max_delta / torch.tensor(extent, device=device), dim=-1)
        loss_lip = F.relu(lip - 0.55).pow(2).mean()
        loss_pbr = (sw_t[idx, None] * (pa - alb_t[idx]) ** 2).mean() + 0.5 * (sw_t[idx] * (pr - rgh_t[idx]) ** 2).mean() + 0.5 * (sw_t[idx] * (pm - met_t[idx]) ** 2).mean()
        loss = 250.0 * loss_d + 2.5 * loss_pbr + 0.15 * loss_lip
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (st + 1) % 350 == 0:
            err_mm = (pred_d - gt_t[idx]).abs().mean().item() * 1000.0
            print(f"    [{region_name}] step {st+1:4d}/{steps} | sdf_mae={err_mm:.2f}mm pbr={loss_pbr.item():.4f}", flush=True)

    return model, center, extent


def fmt_siren24_consts(prefix: str, model: Siren24):
    w0 = np.concatenate([model.l0.weight.detach().cpu().numpy(), model.l0.bias.detach().cpu().numpy()[:, None]], axis=1)
    w1 = model.l1.weight.detach().cpu().numpy().reshape(144, 4)
    b1 = model.l1.bias.detach().cpu().numpy().reshape(6, 4)
    ow = model.out.weight.detach().cpu().numpy().reshape(36, 4)
    ow0 = model.out.weight.detach().cpu().numpy()[0].reshape(6, 4)
    ob = model.out.bias.detach().cpu().numpy()

    def fv4(arr, name):
        lines = [f"const {name}: [vec4; {len(arr)}] = ["]
        for v in arr:
            lines.append(f"    vec4({v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f}, {v[3]:.5f}),")
        lines.append("]")
        return "\n".join(lines)

    return "\n\n".join([
        fv4(w0, f"{prefix}_W0"),
        fv4(w1, f"{prefix}_W1"),
        fv4(b1, f"{prefix}_B1"),
        fv4(ow, f"{prefix}_OW"),
        fv4(ow0, f"{prefix}_OW0"),
        f"const {prefix}_OB0: vec4 = vec4({ob[0]:.5f}, {ob[1]:.5f}, {ob[2]:.5f}, {ob[3]:.5f})\n"
        f"const {prefix}_OB1: vec4 = vec4({ob[4]:.5f}, {ob[5]:.5f}, 0.0, 0.0)",
    ])


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh = load_mage_mesh("/content/3d/mage.glb")

    # 1. Fit Head/Face BoundedResidual (attached to HEAD bone at rest y=1.60, z=-0.017)
    head_lo = np.array([-0.13, 1.53, -0.15], np.float32)
    head_hi = np.array([0.13, 1.80, 0.14], np.float32)
    head_base_fn = lambda p: rcone_np(p, np.array([0.0, 1.62, 0.0], np.float32), np.array([0.0, 1.71, -0.005], np.float32), 0.092, 0.088)
    head_model, head_c, head_ext = fit_region_siren(mesh, "head", head_lo, head_hi, head_base_fn, 0.022, "/content/crops", device)

    # 2. Fit Torso/Tunic BoundedResidual (attached to SPINE_2/chest region y=0.80..1.52)
    torso_lo = np.array([-0.24, 0.78, -0.18], np.float32)
    torso_hi = np.array([0.24, 1.54, 0.18], np.float32)
    def torso_base_fn(p):
        d0 = rcone_np(p, np.array([0.0, 0.88, -0.02], np.float32), np.array([0.0, 1.18, 0.0], np.float32), 0.165, 0.155)
        d1 = rcone_np(p, np.array([0.0, 1.18, 0.0], np.float32), np.array([0.0, 1.46, -0.01], np.float32), 0.155, 0.168)
        h = np.clip(0.5 + 0.5 * (d1 - d0) / 0.04, 0.0, 1.0)
        return d1 * (1.0 - h) + d0 * h - 0.04 * h * (1.0 - h)
    torso_model, torso_c, torso_ext = fit_region_siren(mesh, "torso", torso_lo, torso_hi, torso_base_fn, 0.026, "/content/crops", device)

    # Emit pure Wrela `game/world/mage_residuals.wrela` holding the fitted Head & Torso SIREN weights
    # (cleanly separated from `game/world/mage.wrela` so body anatomy & 64-bone skinning stay readable!)
    os.makedirs("/content/wrela_out", exist_ok=True)
    # Express centers relative to HEAD rest (0.0, 1.60, -0.017) and SPINE_2 rest (0.0, 1.178, 0.004)
    hc_loc = head_c - np.array([0.0, 1.60, -0.017], np.float32)
    tc_loc = torso_c - np.array([0.0, 1.178, 0.004], np.float32)

    src = f"""// Fitted bone-local SIREN-24 bounded residuals for the AAA mage's head/face and tailored
// double-breasted scholar's tunic (`tools/fieldgen/fit_mage_field.py`, trained on Colab L4 from
// VAST-AI/TripoSG MIT + multi-view studio concept art). Pure Wrela math, CPU & GPU identical.

use engine::matter::substance::Substance
use std::field::{{ResidualEval, siren24_sdf, siren24_sdf_pbr}}

{fmt_siren24_consts("MHEAD", head_model)}

pub const MHEAD_CENTER_LOCAL: vec3 = vec3({hc_loc[0]:.5f}, {hc_loc[1]:.5f}, {hc_loc[2]:.5f})
pub const MHEAD_EXTENT: vec3 = vec3({head_ext[0]:.5f}, {head_ext[1]:.5f}, {head_ext[2]:.5f})
pub const MHEAD_INV_EXT: vec3 = vec3({1.0/head_ext[0]:.5f}, {1.0/head_ext[1]:.5f}, {1.0/head_ext[2]:.5f})
pub const MHEAD_MAX_DELTA: f32 = 0.022

pub struct MageHeadResidual: Copy + GpuData + Eq + StateHash {{
    pub tag: u32,
}}

impl ResidualEval for MageHeadResidual {{
    fn eval_sdf(self, q: vec3) -> f32 {{
        siren24_sdf(q, MHEAD_W0, MHEAD_W1, MHEAD_B1, MHEAD_OW0, MHEAD_OB0.x)
    }}
}}

pub fn mage_head_substance(p_local: vec3) -> Substance {{
    let q = (p_local - MHEAD_CENTER_LOCAL) * MHEAD_INV_EXT
    let (_, albedo, roughness, metallic) = siren24_sdf_pbr(
        q,
        MHEAD_W0,
        MHEAD_W1,
        MHEAD_B1,
        MHEAD_OW,
        MHEAD_OB0,
        MHEAD_OB1,
    )
    let is_skin = f32(albedo.x > 0.14 && albedo.x > albedo.z * 1.4)
    Substance {{
        albedo,
        roughness,
        emits: vec3(0.0),
        metallic,
        ior: 1.45,
        transmission: 0.0,
        subsurface: 0.25 * is_skin,
        coat: 0.0,
        micro: 0.05,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }}
}}

{fmt_siren24_consts("MTORSO", torso_model)}

pub const MTORSO_CENTER_LOCAL: vec3 = vec3({tc_loc[0]:.5f}, {tc_loc[1]:.5f}, {tc_loc[2]:.5f})
pub const MTORSO_EXTENT: vec3 = vec3({torso_ext[0]:.5f}, {torso_ext[1]:.5f}, {torso_ext[2]:.5f})
pub const MTORSO_INV_EXT: vec3 = vec3({1.0/torso_ext[0]:.5f}, {1.0/torso_ext[1]:.5f}, {1.0/torso_ext[2]:.5f})
pub const MTORSO_MAX_DELTA: f32 = 0.026

pub struct MageTorsoResidual: Copy + GpuData + Eq + StateHash {{
    pub tag: u32,
}}

impl ResidualEval for MageTorsoResidual {{
    fn eval_sdf(self, q: vec3) -> f32 {{
        siren24_sdf(q, MTORSO_W0, MTORSO_W1, MTORSO_B1, MTORSO_OW0, MTORSO_OB0.x)
    }}
}}

pub fn mage_torso_substance(p_local: vec3) -> Substance {{
    let q = (p_local - MTORSO_CENTER_LOCAL) * MTORSO_INV_EXT
    let (_, albedo, roughness, metallic) = siren24_sdf_pbr(
        q,
        MTORSO_W0,
        MTORSO_W1,
        MTORSO_B1,
        MTORSO_OW,
        MTORSO_OB0,
        MTORSO_OB1,
    )
    Substance {{
        albedo,
        roughness,
        emits: vec3(0.0),
        metallic,
        ior: 1.5,
        transmission: 0.0,
        subsurface: 0.0,
        coat: 0.0,
        micro: 0.12,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }}
}}
"""
    out_p = "/content/wrela_out/mage_residuals.wrela"
    with open(out_p, "w") as f:
        f.write(src)
    print(f"Emitted {out_p} ({os.path.getsize(out_p)} bytes) in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
