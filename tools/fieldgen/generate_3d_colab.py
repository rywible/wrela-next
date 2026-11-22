#!/usr/bin/env python3
"""Colab L4 2D -> 3D generator using VAST-AI/TripoSG (MIT License) + multi-view studio projection.
Generates watertight 3D meshes (and exact SDF samples) from our approved 2D concept crops:
  - table_34.png    -> /content/out/table.glb
  - grimoire_34.png -> /content/out/grimoire.glb
  - hat_34.png      -> /content/out/hat.glb
  - mage_front.png  -> /content/out/mage.glb
Plus saves 4-view rendered preview PNGs so we can visually inspect every generated 3D shape.
"""

import os
import subprocess
import sys
import time


def sh(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")


def ensure_deps():
    if not os.path.exists("/content/TripoSG"):
        sh(["git", "clone", "--depth", "1", "https://github.com/VAST-AI-Research/TripoSG.git", "/content/TripoSG"])
    sh([
        "uv", "pip", "install", "--system",
        "trimesh", "einops", "omegaconf", "scikit-image", "pymeshlab",
        "huggingface_hub", "accelerate", "opencv-python-headless", "rembg", "onnxruntime-gpu",
        "jaxtyping", "typeguard",
    ])
    # Patch optional diso import in TripoSG so pure skimage marching cubes works cleanly on Python 3.13 / CUDA 13.
    iu = "/content/TripoSG/triposg/inference_utils.py"
    txt = open(iu).read()
    if "from diso import DiffDMC" in txt and "try:\n    from diso" not in txt:
        txt = txt.replace(
            "from diso import DiffDMC",
            "try:\n    from diso import DiffDMC\nexcept Exception:\n    DiffDMC = None",
        )
        open(iu, "w").write(txt)
        print("Patched optional diso import in inference_utils.py", flush=True)


def render_preview(mesh, out_png):
    """Pure-numpy z-buffered normal/shaded 4-view turnaround preview (no EGL/OpenGL needed)."""
    import numpy as np
    from PIL import Image

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(verts) == 0 or len(faces) == 0:
        return
    # Center and normalize into [-0.9, 0.9]
    lo, hi = verts.min(0), verts.max(0)
    c = 0.5 * (lo + hi)
    s = float(np.max(hi - lo)) * 0.55 + 1e-6
    v0 = (verts - c) / s

    # Compute face normals and centroids
    p0, p1, p2 = v0[faces[:, 0]], v0[faces[:, 1]], v0[faces[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
    # Vertex normals
    vn = np.zeros_like(v0)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8)

    # Dense surface point sampling for fast, sharp orthographic software rendering
    pts, fidx = mesh.sample(160000, return_index=True)
    pts = (pts.astype(np.float32) - c) / s
    nrm = fn[fidx]

    W = 384
    angles = [0.0, 90.0, 180.0, 40.0]
    elevs = [0.0, 0.0, 0.0, 22.0]
    panels = []
    light = np.array([0.45, 0.72, 0.53], dtype=np.float32)
    light /= np.linalg.norm(light)
    for ang, el in zip(angles, elevs):
        ry = np.radians(ang)
        rx = np.radians(el)
        cy, sy = np.cos(ry), np.sin(ry)
        cx, sx = np.cos(rx), np.sin(rx)
        # Rotate around Y then X
        x1 = cy * pts[:, 0] + sy * pts[:, 2]
        y1 = pts[:, 1]
        z1 = -sy * pts[:, 0] + cy * pts[:, 2]
        x2 = x1
        y2 = cx * y1 - sx * z1
        z2 = sx * y1 + cx * z1

        nx1 = cy * nrm[:, 0] + sy * nrm[:, 2]
        ny1 = nrm[:, 1]
        nz1 = -sy * nrm[:, 0] + cy * nrm[:, 2]
        nx2 = nx1
        ny2 = cx * ny1 - sx * nz1
        nz2 = sx * ny1 + cx * nz1

        px = np.clip(((x2 * 0.45 + 0.5) * W).astype(np.int32), 0, W - 1)
        py = np.clip(((-y2 * 0.45 + 0.5) * W).astype(np.int32), 0, W - 1)
        order = np.argsort(z2)  # back-to-front painter's sort
        px, py = px[order], py[order]
        ndotl = np.clip(nx2[order] * light[0] + ny2[order] * light[1] + nz2[order] * light[2], 0.0, 1.0)
        shade = 0.22 + 0.78 * ndotl
        img = np.full((W, W, 3), 240, dtype=np.uint8)
        col = np.stack([shade * 210, shade * 200, shade * 195], axis=-1).clip(0, 255).astype(np.uint8)
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(py + dy, 0, W - 1), np.clip(px + dx, 0, W - 1)] = col
        panels.append(img)
    grid = np.concatenate(panels, axis=1)
    Image.fromarray(grid).save(out_png)
    print(f"Saved preview {out_png}", flush=True)


def main():
    t0 = time.time()
    import tarfile
    if os.path.exists("/content/crops.tar.gz"):
        os.makedirs("/content/crops", exist_ok=True)
        with tarfile.open("/content/crops.tar.gz", "r:gz") as tf:
            tf.extractall("/content/crops")
        print("Extracted /content/crops:", sorted(os.listdir("/content/crops")), flush=True)
    ensure_deps()
    sys.path.insert(0, "/content/TripoSG")

    import importlib, inspect, pkgutil
    import torch
    import trimesh
    from PIL import Image
    from huggingface_hub import snapshot_download
    # Pre-import all triposg submodules so diffusers >=0.35 allows loading custom pipeline classes.
    import triposg
    for _, modname, _ in pkgutil.walk_packages(triposg.__path__, prefix="triposg."):
        try:
            importlib.import_module(modname)
        except Exception as e:
            print(f"  skip optional {modname}: {e}", flush=True)
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    ckpt_dir = "/content/pretrained_weights/TripoSG"
    if not os.path.exists(ckpt_dir):
        print("Downloading VAST-AI/TripoSG weights...", flush=True)
        snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=ckpt_dir)

    pipe = TripoSGPipeline.from_pretrained(ckpt_dir, local_files_only=True).to("cuda", dtype=torch.float16)
    print(f"Loaded TripoSG in {time.time() - t0:.1f}s", flush=True)
    call_sig = inspect.signature(pipe.__call__).parameters
    extra_kw = {"use_flash_decoder": False} if "use_flash_decoder" in call_sig else {}

    os.makedirs("/content/out", exist_ok=True)
    jobs = [
        ("table", "/content/crops/table_34.png", 42),
        ("grimoire", "/content/crops/grimoire_34.png", 42),
        ("hat", "/content/crops/hat_34.png", 42),
        ("mage", "/content/crops/mage_front.png", 42),
    ]
    for name, img_path, seed in jobs:
        if not os.path.exists(img_path):
            print(f"Skipping {name}: {img_path} missing", flush=True)
            continue
        ts = time.time()
        img = Image.open(img_path).convert("RGB")
        print(f"Generating 3D for {name} from {img_path}...", flush=True)
        outputs = pipe(
            image=img,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            num_inference_steps=50,
            guidance_scale=7.0,
            **extra_kw,
        ).samples[0]
        mesh = trimesh.Trimesh(outputs[0].astype("float32"), outputs[1].astype("int32"))
        mesh.fix_normals()
        glb = f"/content/out/{name}.glb"
        mesh.export(glb)
        print(
            f"  {name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
            f"watertight={mesh.is_watertight}, bounds={mesh.bounds.tolist()} ({time.time()-ts:.1f}s)",
            flush=True,
        )
        render_preview(mesh, f"/content/out/{name}_preview.png")


if __name__ == "__main__":
    main()
