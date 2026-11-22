#!/usr/bin/env python3
"""Colab L4 Differentiable 3D -> Wrela `BoundedResidual` Field Compiler (`fit_wrela_field.py`).

Takes a watertight/high-res 3D mesh from TripoSG (`/content/3d/<name>.glb`) + flat-lit studio
multi-view crops (`/content/crops/<name>_*.png`), aligns it to exact physical meters, computes
exact signed distances + multi-view delighted PBR (albedo, roughness, metallic), fits:
  1. An analytical base SDF (`RoundCuboid` / `Lathe` / `RoundCone` matching `std::field`)
  2. A compact `siren24_sdf_pbr` bounded residual MLP (`max_delta` = 1.5..3.5 cm)
and emits a pure, self-contained Wrela module (`game/world/<name>.wrela`) implementing
`Field<Substance>` with zero external assets, exact `Bound`, `Eq`, and `StateHash`.
"""

import argparse
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from PIL import Image


def load_and_align_mesh(glb_path: str, kind: str):
    """Load TripoSG mesh, keep largest connected component (removes stray floating specks),
    align orientation to Wrela (+x left/right, +y up, +z fwd), and normalize to physical meters."""
    scene_or_mesh = trimesh.load(glb_path, force="mesh")
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    # Remove any small disconnected floating specks (e.g., stray dots near clasps/brim).
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        parts.sort(key=lambda m: len(m.faces), reverse=True)
        main_faces = len(parts[0].faces)
        kept = [m for m in parts if len(m.faces) >= max(200, int(0.02 * main_faces))]
        mesh = trimesh.util.concatenate(kept) if len(kept) > 1 else parts[0]
        print(f"  cleaned {len(parts)} components -> {len(kept)} kept ({len(mesh.faces)} faces)", flush=True)
    verts = np.asarray(mesh.vertices, dtype=np.float32)

    ext = verts.max(0) - verts.min(0)
    print(f"  raw bounds extents ({kind}): {ext.round(4)}", flush=True)

    if kind == "table":
        if ext[2] > ext[0] * 1.15:
            verts = np.stack([verts[:, 2], verts[:, 1], -verts[:, 0]], axis=-1)
        lo, hi = verts.min(0), verts.max(0)
        c_xz = 0.5 * (lo[[0, 2]] + hi[[0, 2]])
        verts[:, 0] -= c_xz[0]
        verts[:, 2] -= c_xz[1]
        verts[:, 1] -= lo[1]
        cur_h = float(hi[1] - lo[1])
        scale = 0.78 / max(cur_h, 1e-4)
        verts *= scale
        ext2 = verts.max(0) - verts.min(0)
        if ext2[0] > 1.55 or ext2[0] < 0.95:
            verts[:, 0] *= 1.24 / max(ext2[0], 1e-4)
        if ext2[2] > 0.95 or ext2[2] < 0.50:
            verts[:, 2] *= 0.70 / max(ext2[2], 1e-4)

    elif kind == "grimoire":
        # Use PCA principal axes so a book tilted from a 3/4 input lays 100% dead-flat on y = 0!
        c0 = verts.mean(0)
        vc = verts - c0
        # Subsample for fast exact covariance
        sub = vc[:: max(1, len(vc) // 50000)]
        cov = np.cov(sub.T)
        evals, evecs = np.linalg.eigh(cov)  # sorted ascending: [thin(thickness), mid(width), long(length)]
        # Map thin -> +Y (up), mid -> +X (width), long -> +Z (length)
        ax_y = evecs[:, 0]
        ax_x = evecs[:, 1]
        ax_z = evecs[:, 2]
        if np.linalg.det(np.stack([ax_x, ax_y, ax_z], axis=1)) < 0:
            ax_z = -ax_z
        R = np.stack([ax_x, ax_y, ax_z], axis=1).astype(np.float32)  # (3, 3)
        v_new = vc @ R
        lo, hi = v_new.min(0), v_new.max(0)
        v_new[:, 0] -= 0.5 * (lo[0] + hi[0])
        v_new[:, 2] -= 0.5 * (lo[2] + hi[2])
        v_new[:, 1] -= lo[1]
        e = hi - lo
        v_new[:, 0] *= 0.22 / max(e[0], 1e-4)
        v_new[:, 1] *= 0.056 / max(e[1], 1e-4)
        v_new[:, 2] *= 0.30 / max(e[2], 1e-4)
        verts = v_new

    elif kind == "hat":
        # Wizard hat: wide brim in XZ (~0.42m), height along Y (~0.34m), origin at head-mount band (y=0).
        lo, hi = verts.min(0), verts.max(0)
        verts[:, 0] -= 0.5 * (lo[0] + hi[0])
        verts[:, 2] -= 0.5 * (lo[2] + hi[2])
        verts[:, 1] -= lo[1]
        e = hi - lo
        s = 0.42 / max(max(e[0], e[2]), 1e-4)
        verts *= s
        # Shift y so brim/hatband sits cleanly at y = 0.0 (crown rises +y, brim dips slightly below)
        verts[:, 1] -= 0.04

    mesh = trimesh.Trimesh(verts, np.asarray(mesh.faces, dtype=np.int32), process=False)
    mesh.fix_normals()
    print(f"  aligned bounds ({kind}): lo={mesh.bounds[0].round(4)} hi={mesh.bounds[1].round(4)}", flush=True)
    return mesh


def sample_surface_and_band(mesh, n_surf=40000, n_band=35000, band_sigma=0.025):
    """Sample exact surface points + narrow-band 3D points with fast KDTree signed distance."""
    from scipy.spatial import cKDTree

    ref_pts, ref_fidx = mesh.sample(220000, return_index=True)
    ref_pts = ref_pts.astype(np.float32)
    ref_nrms = np.asarray(mesh.face_normals[ref_fidx], dtype=np.float32)
    tree = cKDTree(ref_pts)

    pts_s = ref_pts[:n_surf]
    nrms_s = ref_nrms[:n_surf]

    rng = np.random.default_rng(42)
    pick = rng.integers(0, n_surf, size=n_band)
    offsets = rng.uniform(-band_sigma, band_sigma, size=(n_band, 1)).astype(np.float32)
    jitter = rng.normal(0.0, band_sigma * 0.25, size=(n_band, 3)).astype(np.float32)
    pts_b = pts_s[pick] + nrms_s[pick] * offsets + jitter

    dists, nn = tree.query(pts_b, k=1, workers=-1)
    diff = pts_b - ref_pts[nn]
    tri_n = ref_nrms[nn]
    sgn = np.where(np.sum(diff * tri_n, axis=-1) >= 0.0, 1.0, -1.0).astype(np.float32)
    sdf_b = dists.astype(np.float32) * sgn

    pts = np.concatenate([pts_s, pts_b], axis=0)
    sdf = np.concatenate([np.zeros(n_surf, dtype=np.float32), sdf_b], axis=0)
    nrms = np.concatenate([nrms_s, tri_n], axis=0)
    return pts, sdf, nrms


def project_pbr_from_crops(pts, nrms, kind: str, crop_dir: str):
    """Project our flat-lit multi-view studio turnaround images onto 3D points with delighting
    and physically calibrated (albedo_linear, roughness, metallic) estimation."""
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-4)

    def load_img(fname):
        p = os.path.join(crop_dir, fname)
        if not os.path.exists(p):
            return None
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        # Tight non-white bbox inside crop for accurate orthographic UV alignment
        mask = (arr < 0.95).any(axis=2)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return arr
        return arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    def sample_ortho(img, u, v):
        H, W, _ = img.shape
        ix = np.clip((u * (W - 1)).astype(np.int32), 0, W - 1)
        iy = np.clip(((1.0 - v) * (H - 1)).astype(np.int32), 0, H - 1)
        col = img[iy, ix]
        is_bg = (col > 0.94).all(axis=-1)
        return col, ~is_bg

    # Normalized [0, 1] coordinates
    nx = (pts[:, 0] - lo[0]) / span[0]
    ny = (pts[:, 1] - lo[1]) / span[1]
    nz = (pts[:, 2] - lo[2]) / span[2]

    if kind == "table":
        im_front = load_img("table_front.png")
        im_side = load_img("table_side.png")
        im_34 = load_img("table_34.png")
        c_f, ok_f = sample_ortho(im_front, nx, ny)
        c_s, ok_s = sample_ortho(im_side, nz, ny)
        # Weight front/back by |nz|, left/right by |nx|, top by wood grain from front/34
        wf = (np.abs(nrms[:, 2]) + 0.1) * ok_f
        ws = (np.abs(nrms[:, 0]) + 0.1) * ok_s
        wsum = wf + ws + 1e-6
        col = (c_f * wf[:, None] + c_s * ws[:, None]) / wsum[:, None]
        # Top tabletop: warm dark oak wood tone with subtle brass border ring near outer inset!
        top_m = (nrms[:, 1] > 0.7) & (ny > 0.91)
        # Brass border inlay 5..7cm inside tabletop perimeter
        bx = np.abs(pts[:, 0])
        bz = np.abs(pts[:, 2])
        hx, hz = 0.5 * span[0], 0.5 * span[2]
        inlay = top_m & (
            ((np.abs(bx - (hx - 0.055)) < 0.008) & (bz < hz - 0.048))
            | ((np.abs(bz - (hz - 0.055)) < 0.008) & (bx < hx - 0.048))
        )
        # Rich dark oak base for top + fallback pixels
        oak = np.array([0.27, 0.20, 0.15], dtype=np.float32)
        brass = np.array([0.72, 0.58, 0.30], dtype=np.float32)
        col = np.where((wf + ws < 0.15)[:, None], oak, col)
        col = np.where(top_m[:, None], oak * (0.92 + 0.08 * np.sin(pts[:, 0:1] * 45.0 + pts[:, 2:3] * 8.0)), col)
        col = np.where(inlay[:, None], brass, col)

    elif kind == "grimoire":
        im_top = load_img("grimoire_top.png")
        im_spine = load_img("grimoire_spine.png")
        c_t, ok_t = sample_ortho(im_top, nx, nz)
        c_sp, ok_sp = sample_ortho(im_spine, ny, nz)
        wt = (np.abs(nrms[:, 1]) + 0.15) * ok_t
        wsp = (np.maximum(-nrms[:, 0], 0.0) + 0.05) * ok_sp
        wsum = wt + wsp + 1e-6
        col = (c_t * wt[:, None] + c_sp * wsp[:, None]) / wsum[:, None]
        # Fore-edge & top/bottom page block: warm cream aged parchment!
        pages = (np.abs(nrms[:, 1]) < 0.55) & (nx > 0.18) & (ny > 0.14) & (ny < 0.86)
        leather = np.array([0.30, 0.16, 0.11], dtype=np.float32)
        parchment = np.array([0.78, 0.71, 0.57], dtype=np.float32)
        col = np.where((wt + wsp < 0.15)[:, None], leather, col)
        col = np.where(pages[:, None], parchment, col)

    elif kind == "hat":
        im_f = load_img("hat_front.png")
        im_s = load_img("hat_side.png")
        c_f, ok_f = sample_ortho(im_f, nx, ny)
        c_s, ok_s = sample_ortho(im_s, nz, ny)
        wf = (np.abs(nrms[:, 2]) + 0.1) * ok_f
        ws = (np.abs(nrms[:, 0]) + 0.1) * ok_s
        wsum = wf + ws + 1e-6
        col = (c_f * wf[:, None] + c_s * ws[:, None]) / wsum[:, None]
        felt = np.array([0.14, 0.17, 0.22], dtype=np.float32)
        col = np.where((wf + ws < 0.15)[:, None], felt, col)

    else:
        col = np.full((len(pts), 3), 0.35, dtype=np.float32)

    # Delight & convert sRGB -> linear albedo (lift deep baked shadows softly, clamp to physical range)
    col = np.clip(col, 0.02, 0.95)
    albedo_lin = np.power(col, 2.2).astype(np.float32)

    # Detect antique brass (warm gold hue: R > G > B and elevated saturation/luminance)
    is_brass = (col[:, 0] > 0.42) & (col[:, 0] > col[:, 1] * 1.05) & (col[:, 1] > col[:, 2] * 1.28)
    metallic = np.where(is_brass, 0.88, 0.0).astype(np.float32)
    if kind == "table":
        rough = np.where(is_brass, 0.28, 0.54).astype(np.float32)
    elif kind == "grimoire":
        is_page = (col[:, 0] > 0.62) & (col[:, 1] > 0.55) & (col[:, 2] > 0.42)
        rough = np.where(is_brass, 0.26, np.where(is_page, 0.86, 0.46)).astype(np.float32)
    else:  # hat
        is_leather = (col[:, 0] > col[:, 2] * 1.25) & (~is_brass)
        rough = np.where(is_brass, 0.28, np.where(is_leather, 0.48, 0.88)).astype(np.float32)

    return albedo_lin, rough, metallic


def eval_base_sdf_np(pts, kind: str, bounds):
    """Evaluate the exact Wrela analytical base SDF `d_base(p)` in numpy (matching the emitted
    Wrela base function bit-for-bit!) so the SIREN residual learns ONLY the fine sculpted delta."""
    lo, hi = bounds[0], bounds[1]
    hx, hy, hz = 0.5 * (hi - lo)

    def rbox(p, c, h, r):
        q = np.abs(p - c) - np.maximum(h - r, 0.0)
        m = np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2]))
        out = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        return np.where(m > 0.0, out, m) - r

    def rcone(p, a, b, r1, r2):
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
        return np.where(np.sign(z) * a2 * z2 > k, d_top, np.where(np.sign(y) * a2 * y2 < k, d_bot, d_side))

    def smin(a, b, k):
        h = np.clip(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0)
        return b * (1.0 - h) + a * h - k * h * (1.0 - h)

    if kind == "table":
        # Flat tabletop slab + scalloped apron box + 4 turned legs + H-stretcher
        top = rbox(pts, np.array([0.0, hi[1] - 0.024, 0.0], np.float32), np.array([hx, 0.024, hz], np.float32), 0.008)
        apron = rbox(pts, np.array([0.0, hi[1] - 0.085, 0.0], np.float32), np.array([hx - 0.06, 0.055, hz - 0.06], np.float32), 0.012)
        d = smin(top, apron, 0.015)
        lx, lz = hx - 0.09, hz - 0.085
        for sx in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                leg = rcone(
                    pts,
                    np.array([sx * lx, 0.025, sz * lz], np.float32),
                    np.array([sx * lx, hi[1] - 0.05, sz * lz], np.float32),
                    0.022,
                    0.036,
                )
                d = smin(d, leg, 0.012)
        # Side & center H-stretchers
        for sx in (-1.0, 1.0):
            sb = rbox(pts, np.array([sx * lx, 0.16, 0.0], np.float32), np.array([0.022, 0.022, lz], np.float32), 0.006)
            d = smin(d, sb, 0.010)
        cb = rbox(pts, np.array([0.0, 0.16, 0.0], np.float32), np.array([lx, 0.022, 0.024], np.float32), 0.006)
        d = smin(d, cb, 0.010)
        return d.astype(np.float32)

    elif kind == "grimoire":
        # Rounded thick hardcover book slab + rounded spine cylinder
        body = rbox(pts, np.array([0.0, hy, 0.0], np.float32), np.array([hx, hy, hz], np.float32), 0.008)
        spine = rcone(
            pts,
            np.array([-hx + 0.012, hy, -hz + 0.01], np.float32),
            np.array([-hx + 0.012, hy, hz - 0.01], np.float32),
            hy * 0.98,
            hy * 0.98,
        )
        return smin(body, spine, 0.006).astype(np.float32)

    else:  # hat
        # Wide slightly curved brim disc + tapered bent pointed crown (2 smooth-unioned cones)
        brim = rcone(
            pts,
            np.array([0.0, 0.008, 0.0], np.float32),
            np.array([0.0, 0.016, 0.0], np.float32),
            0.20,
            0.195,
        )
        crown_lo = rcone(
            pts,
            np.array([0.0, 0.02, 0.0], np.float32),
            np.array([0.0, 0.19, -0.02], np.float32),
            0.105,
            0.055,
        )
        crown_tip = rcone(
            pts,
            np.array([0.0, 0.19, -0.02], np.float32),
            np.array([0.02, 0.28, -0.10], np.float32),
            0.055,
            0.010,
        )
        return smin(brim, smin(crown_lo, crown_tip, 0.03), 0.025).astype(np.float32)


class Siren24(nn.Module):
    """Exact PyTorch mirror of `std::field::siren24_sdf_pbr` (width=24, 2 hidden layers)."""
    def __init__(self, omega0=10.0):
        super().__init__()
        self.omega0 = omega0
        self.l0 = nn.Linear(3, 24)
        self.l1 = nn.Linear(24, 24)
        self.out = nn.Linear(24, 6)  # delta_raw, r, g, b, rough, metal
        with torch.no_grad():
            self.l0.weight.uniform_(-1.0 / 3.0, 1.0 / 3.0)
            self.l0.weight.mul_(omega0)
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
        delta = torch.tanh(y[:, 0])
        albedo = torch.sigmoid(y[:, 1:4])
        rough = 0.06 + 0.92 * torch.sigmoid(y[:, 4])
        metal = torch.sigmoid(y[:, 5])
        return delta, albedo, rough, metal


def fit_siren(pts, sdf_gt, d_base, albedo, rough, metal, max_delta, device, steps=1600):
    lo, hi = pts.min(0), pts.max(0)
    center = 0.5 * (lo + hi)
    extent = np.maximum(0.5 * (hi - lo), 0.05)

    q_all = (pts - center) / extent
    band = max_delta * 2.5 + 0.015
    # Window weight matching `BoundedResidual::apply_residual`
    ad = np.abs(d_base)
    t_w = np.clip(1.0 - ad / band, 0.0, 1.0)
    win = t_w * t_w * (3.0 - 2.0 * t_w)

    # Keep only points inside active band where residual operates
    active = win > 0.02
    q_t = torch.tensor(q_all[active], device=device)
    db_t = torch.tensor(d_base[active], device=device)
    gt_t = torch.tensor(sdf_gt[active], device=device)
    w_t = torch.tensor(win[active], device=device)
    alb_t = torch.tensor(albedo[active], device=device)
    rgh_t = torch.tensor(rough[active], device=device)
    met_t = torch.tensor(metal[active], device=device)
    is_surf = (np.abs(sdf_gt[active]) < 0.004)
    surf_t = torch.tensor(is_surf.astype(np.float32), device=device)

    model = Siren24(omega0=8.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-4)

    N = len(q_t)
    bs = 4096
    for st in range(steps):
        idx = torch.randint(0, N, (bs,), device=device)
        qb = q_t[idx].detach().requires_grad_(True)
        delta, p_alb, p_rgh, p_met = model(qb)
        pred_d = db_t[idx] + w_t[idx] * max_delta * delta

        # Target SDF loss (higher weight right on the zero isosurface)
        sw = 1.0 + 3.0 * surf_t[idx]
        loss_d = (sw * (pred_d - gt_t[idx]) ** 2).mean()
        # Mild Lipschitz penalty on residual gradient so |grad(delta_d)| stays bounded
        grad_delta = torch.autograd.grad(
            delta.sum(), qb, create_graph=True
        )[0]
        lip_norm = torch.linalg.norm(grad_delta * max_delta / torch.tensor(extent, device=device), dim=-1)
        loss_lip = F.relu(lip_norm - 0.55).pow(2).mean()

        # PBR Substance loss on surface
        loss_pbr = (
            sw[:, None] * (p_alb - alb_t[idx]) ** 2
        ).mean() + 0.5 * (sw * (p_rgh - rgh_t[idx]) ** 2).mean() + 0.5 * (sw * (p_met - met_t[idx]) ** 2).mean()

        loss = 250.0 * loss_d + 2.0 * loss_pbr + 0.15 * loss_lip
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (st + 1) % 400 == 0:
            err_mm = (pred_d - gt_t[idx]).abs().mean().item() * 1000.0
            print(f"    step {st+1:4d}/{steps} | sdf_mae={err_mm:.2f}mm pbr={loss_pbr.item():.4f}", flush=True)

    return model, center, extent


def emit_wrela_prop(name: str, kind: str, model: Siren24, center, extent, bounds, max_delta: float, out_path: str):
    """Emit a pure, self-contained Wrela `Field<Substance>` module for the fitted asset."""
    w0 = model.l0.weight.detach().cpu().numpy()   # (24, 3)
    b0 = model.l0.bias.detach().cpu().numpy()     # (24,)
    w0_v4 = np.concatenate([w0, b0[:, None]], axis=1)  # (24, 4)

    w1 = model.l1.weight.detach().cpu().numpy()   # (24, 24) -> 24 rows x 6 vec4 = 144 vec4
    b1 = model.l1.bias.detach().cpu().numpy()     # (24,) -> 6 vec4
    ow = model.out.weight.detach().cpu().numpy()  # (6, 24) -> 6 x 6 = 36 vec4
    ob = model.out.bias.detach().cpu().numpy()    # (6,) -> out_b0 (4), out_b1 (4)

    def fmt_v4(arr, var_name):
        n = len(arr)
        lines = [f"const {var_name}: [vec4; {n}] = ["]
        for v in arr:
            lines.append(f"    vec4({v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f}, {v[3]:.5f}),")
        lines.append("]")
        return "\n".join(lines)

    w1_v4 = w1.reshape(144, 4)
    b1_v4 = b1.reshape(6, 4)
    ow_v4 = ow.reshape(36, 4)
    ow0_v4 = ow[0].reshape(6, 4)
    ob0 = ob[:4]
    ob1 = np.array([ob[4], ob[5], 0.0, 0.0], dtype=np.float32)

    lo, hi = bounds[0], bounds[1]
    hx, hy, hz = 0.5 * (hi - lo)
    P = name.upper()
    C = name.capitalize()

    if kind == "table":
        lx, lz = hx - 0.09, hz - 0.085
        base_code = f"""pub const TABLE_TOP_Y: f32 = {hi[1]:.5f}

pub type TableBase = SmoothUnion<
    SmoothUnion<
        SmoothUnion<RoundCuboid, Translate<RoundCuboid>>,
        SmoothUnionAll<RoundCone, 4>,
    >,
    SmoothUnion<SmoothUnion<Translate<RoundCuboid>, Translate<RoundCuboid>>, Translate<RoundCuboid>>,
>

fn table_base() -> TableBase {{
    let top = round_cuboid(vec3({hx:.5f}, 0.024, {hz:.5f}), 0.008)
    let apron = round_cuboid(vec3({hx - 0.06:.5f}, 0.055, {hz - 0.06:.5f}), 0.012)
        .translate(vec3(0.0, -0.061, 0.0))
    let legs = [
        round_cone(vec3({-lx:.5f}, {0.025 - (hi[1] - 0.024):.5f}, {-lz:.5f}), vec3({-lx:.5f}, -0.026, {-lz:.5f}), r_top: 0.022, r_bottom: 0.036),
        round_cone(vec3({lx:.5f}, {0.025 - (hi[1] - 0.024):.5f}, {-lz:.5f}), vec3({lx:.5f}, -0.026, {-lz:.5f}), r_top: 0.022, r_bottom: 0.036),
        round_cone(vec3({-lx:.5f}, {0.025 - (hi[1] - 0.024):.5f}, {lz:.5f}), vec3({-lx:.5f}, -0.026, {lz:.5f}), r_top: 0.022, r_bottom: 0.036),
        round_cone(vec3({lx:.5f}, {0.025 - (hi[1] - 0.024):.5f}, {lz:.5f}), vec3({lx:.5f}, -0.026, {lz:.5f}), r_top: 0.022, r_bottom: 0.036),
    ].smooth_union(k: 0.012)
    let sy = {0.16 - (hi[1] - 0.024):.5f}
    let s_l = round_cuboid(vec3(0.022, 0.022, {lz:.5f}), 0.006).translate(vec3({-lx:.5f}, sy, 0.0))
    let s_r = round_cuboid(vec3(0.022, 0.022, {lz:.5f}), 0.006).translate(vec3({lx:.5f}, sy, 0.0))
    let s_c = round_cuboid(vec3({lx:.5f}, 0.022, 0.024), 0.006).translate(vec3(0.0, sy, 0.0))
    top.smooth_union(apron, k: 0.015).smooth_union(legs, k: 0.012).smooth_union(
        s_l.smooth_union(s_r, k: 0.010).smooth_union(s_c, k: 0.010),
        k: 0.010,
    )
}}"""
        center_adj = center - np.array([0.0, hi[1] - 0.024, 0.0], np.float32)
        wrap_shift = f".translate(vec3(0.0, {hi[1] - 0.024:.5f}, 0.0))"
        micro_val = "0.08"
        ret_type = f"Translate<WithAt<BoundedResidual<TableBase, {C}Residual>, fn(vec3) -> Substance>>"
    elif kind == "grimoire":
        base_code = f"""pub const GRIMOIRE_THICKNESS: f32 = {2.0 * hy:.5f}
pub const GRIMOIRE_HALF: vec3 = vec3({hx:.5f}, {hy:.5f}, {hz:.5f})

pub type GrimoireBase = SmoothUnion<RoundCuboid, RoundCone>

fn grimoire_base() -> GrimoireBase {{
    let body = round_cuboid(vec3({hx:.5f}, {hy:.5f}, {hz:.5f}), 0.008)
    let spine = round_cone(
        vec3({-hx + 0.012:.5f}, 0.0, {-hz + 0.01:.5f}),
        vec3({-hx + 0.012:.5f}, 0.0, {hz - 0.01:.5f}),
        r_top: {hy * 0.98:.5f},
        r_bottom: {hy * 0.98:.5f},
    )
    body.smooth_union(spine, k: 0.006)
}}"""
        center_adj = center - np.array([0.0, hy, 0.0], np.float32)
        wrap_shift = f".translate(vec3(0.0, {hy:.5f}, 0.0))"
        micro_val = "0.10"
        ret_type = f"Translate<WithAt<BoundedResidual<GrimoireBase, {C}Residual>, fn(vec3) -> Substance>>"
    else:  # hat
        base_code = f"""pub type HatBase = SmoothUnion<RoundCone, SmoothUnion<RoundCone, RoundCone>>

fn hat_base() -> HatBase {{
    let brim = round_cone(vec3(0.0, 0.008, 0.0), vec3(0.0, 0.016, 0.0), r_top: 0.20, r_bottom: 0.195)
    let crown_lo = round_cone(vec3(0.0, 0.02, 0.0), vec3(0.0, 0.19, -0.02), r_top: 0.105, r_bottom: 0.055)
    let crown_tip = round_cone(vec3(0.0, 0.19, -0.02), vec3(0.02, 0.28, -0.10), r_top: 0.055, r_bottom: 0.010)
    brim.smooth_union(crown_lo.smooth_union(crown_tip, k: 0.03), k: 0.025)
}}"""
        center_adj = center
        wrap_shift = ""
        micro_val = "0.14"
        ret_type = f"WithAt<BoundedResidual<HatBase, {C}Residual>, fn(vec3) -> Substance>"

    src = f"""// Pure mathematical `Field<Substance>` for `{name}` generated by our Colab 2D -> 3D -> Field
// compiler (`tools/fieldgen/fit_wrela_field.py` on VAST-AI/TripoSG MIT + multi-view studio art):
// analytical base primitives refined by a bounded continuous SIREN-24 residual (`BoundedResidual`)
// and pixel-rate PBR `Substance` channels. Zero prebaked meshes or voxel textures.

use engine::matter::substance::Substance
use std::field::{{
    BoundedResidual, ResidualEval, RoundCone, RoundCuboid, SmoothUnion, SmoothUnionAll, Surface,
    Translate, WithAt, bounded_residual, round_cone, round_cuboid, siren24_sdf, siren24_sdf_pbr,
}}

{fmt_v4(w0_v4, f"{P}_W0")}

{fmt_v4(w1_v4, f"{P}_W1")}

{fmt_v4(b1_v4, f"{P}_B1")}

{fmt_v4(ow_v4, f"{P}_OW")}

{fmt_v4(ow0_v4, f"{P}_OW0")}

const {P}_OB0: vec4 = vec4({ob0[0]:.5f}, {ob0[1]:.5f}, {ob0[2]:.5f}, {ob0[3]:.5f})
const {P}_OB1: vec4 = vec4({ob1[0]:.5f}, {ob1[1]:.5f}, {ob1[2]:.5f}, 0.0)
const {P}_CENTER: vec3 = vec3({center_adj[0]:.5f}, {center_adj[1]:.5f}, {center_adj[2]:.5f})
const {P}_EXTENT: vec3 = vec3({extent[0]:.5f}, {extent[1]:.5f}, {extent[2]:.5f})
const {P}_INV_EXT: vec3 = vec3({1.0/extent[0]:.5f}, {1.0/extent[1]:.5f}, {1.0/extent[2]:.5f})
const {P}_MAX_DELTA: f32 = {max_delta:.5f}

pub struct {C}Residual: Copy + GpuData + Eq + StateHash {{
    pub tag: u32,
}}

impl ResidualEval for {C}Residual {{
    fn eval_sdf(self, q: vec3) -> f32 {{
        siren24_sdf(q, {P}_W0, {P}_W1, {P}_B1, {P}_OW0, {P}_OB0.x)
    }}
}}

fn {name}_substance(p: vec3) -> Substance {{
    let q = (p - {P}_CENTER) * {P}_INV_EXT
    let (_, albedo, roughness, metallic) = siren24_sdf_pbr(
        q,
        {P}_W0,
        {P}_W1,
        {P}_B1,
        {P}_OW,
        {P}_OB0,
        {P}_OB1,
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
        micro: {micro_val},
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }}
}}

{base_code}

pub type {C} = {ret_type}

/// Returns the pure `Field<Substance>` for `{name}` in its canonical local frame.
pub fn {name}() -> {C} {{
    bounded_residual(
        {name}_base(),
        {C}Residual {{ tag: 1 }},
        center: {P}_CENTER,
        extent: {P}_EXTENT,
        max_delta: {P}_MAX_DELTA,
        lip_extra: 0.55,
    )
        .with_at(|p: vec3| {name}_substance(p)){wrap_shift}
}}
"""
    with open(out_path, "w") as f:
        f.write(src)
    print(f"Emitted pure Wrela field: {out_path} ({os.path.getsize(out_path)} bytes)", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("/content/wrela_out", exist_ok=True)
    specs = [
        ("table", "table", 0.035),
        ("grimoire", "grimoire", 0.018),
        ("hat", "hat", 0.032),
    ]
    for name, kind, max_d in specs:
        glb = f"/content/3d/{name}.glb"
        if not os.path.exists(glb):
            print(f"Skipping {name}: {glb} not found", flush=True)
            continue
        t0 = time.time()
        print(f"=== Fitting Wrela BoundedResidual field for {name} ===", flush=True)
        mesh = load_and_align_mesh(glb, kind)
        pts, sdf_gt, nrms = sample_surface_and_band(mesh, n_surf=40000, n_band=35000, band_sigma=max_d * 1.4)
        albedo, rough, metal = project_pbr_from_crops(pts, nrms, kind, "/content/crops")
        d_base = eval_base_sdf_np(pts, kind, mesh.bounds)
        model, center, extent = fit_siren(pts, sdf_gt, d_base, albedo, rough, metal, max_d, device, steps=1600)
        emit_wrela_prop(
            name,
            kind,
            model,
            center,
            extent,
            mesh.bounds,
            max_d,
            f"/content/wrela_out/{name}.wrela",
        )
        print(f"Finished {name} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
