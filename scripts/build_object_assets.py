"""Convert the downloaded Poly Haven (CC0) glTF models into MuJoCo-ready assets.

Input : assets/third_party/polyhaven_raw/<name>/<name>_1k.gltf (+ .bin, diffuse jpg)
Output: assets/objects/<name>/{<name>.obj, <name>.png}

Each mesh is rotated from glTF Y-up to MuJoCo Z-up, rescaled to a realistic
size, and translated so its bounding-box bottom-center sits at the origin.

Usage:  .venv/bin/python -I scripts/build_object_assets.py
"""
import json
import logging
import os
import sys

import numpy as np
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "assets", "third_party", "polyhaven_raw")
OUT = os.path.join(ROOT, "assets", "objects")

# name -> (poly haven id, geometry node to keep (None = all), target longest extent [m])
OBJECTS = {
    "duck": ("rubber_duck_toy", None, 0.085),
    "apple": ("food_apple_01", None, 0.066),
    "lemon": ("lemon", None, 0.070),
    "banana": ("bananas", "mesh.011", 0.170),
    "lime": ("food_lime_01", None, 0.060),
}
TEX_RES = 512
LAY_FLAT = {"lemon", "lime"}  # rotate so the long axis is horizontal (rests on its side)

CAPSULE_CHAIN = {"banana": 6}  # non-convex objects: collide with a capsule chain along the centreline

logging.getLogger("trimesh").setLevel(logging.ERROR)  # silence the no-scipy fallback notice


def capsule_chain(v, long_ax, n):
    """Capsules through the centroids of `n + 1` slices along the long axis; radius = mean
    half-thickness of the slices (slightly shrunk so the capsules stay inside the mesh)."""
    lo, hi = v[:, long_ax].min(), v[:, long_ax].max()
    edges = np.linspace(lo, hi, n + 2)
    cents, radii = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sl = v[(v[:, long_ax] >= a) & (v[:, long_ax] <= b)]
        c = sl.mean(axis=0)
        cents.append(c)
        other = [i for i in range(3) if i != long_ax]
        radii.append(0.5 * min(np.ptp(sl[:, other[0]]), np.ptp(sl[:, other[1]])))
    caps = []
    for i in range(len(cents) - 1):
        r = 0.85 * min(radii[i], radii[i + 1])
        caps.append([[round(float(x), 4) for x in cents[i]], [round(float(x), 4) for x in cents[i + 1]],
                     round(float(max(r, 0.006)), 4)])
    return caps


def convert(name, src_id, node, target_extent):
    src = os.path.join(RAW, src_id, f"{src_id}_1k.gltf")
    scene = trimesh.load(src)
    if node is not None:
        geom = scene.geometry[node].copy()
    else:
        assert len(scene.geometry) == 1, f"{src_id}: expected a single geometry"
        geom = next(iter(scene.geometry.values())).copy()

    # glTF is Y-up; MuJoCo is Z-up.  Rotate +90 deg about X: (x, y, z) -> (x, -z, y).
    geom.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    if name in LAY_FLAT:
        geom.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    scale = target_extent / float(np.max(geom.extents))
    geom.apply_scale(scale)
    lo, hi = geom.bounds
    geom.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])

    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    tex_src = os.path.join(RAW, src_id, "textures", f"{src_id}_diff_1k.jpg")
    img = Image.open(tex_src).convert("RGB").resize((TEX_RES, TEX_RES), Image.LANCZOS)
    img.save(os.path.join(OUT, name, f"{name}.png"))

    uv = geom.visual.uv if hasattr(geom.visual, "uv") and geom.visual.uv is not None else None
    if uv is None:
        sys.exit(f"{src_id}: mesh has no UVs")
    # Write a minimal OBJ (v / vt / vn / f) without an .mtl -- MuJoCo takes the
    # texture from the <material> element instead.
    path = os.path.join(OUT, name, f"{name}.obj")
    v, n, f = geom.vertices, geom.vertex_normals, geom.faces
    with open(path, "w") as fh:
        fh.write(f"# {name}: converted from Poly Haven '{src_id}' (CC0)\n")
        fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in v)
        fh.writelines(f"vt {a:.6f} {b:.6f}\n" for a, b in uv)
        fh.writelines(f"vn {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in n)
        fh.writelines(f"f {a+1}/{a+1}/{a+1} {b+1}/{b+1}/{b+1} {c+1}/{c+1}/{c+1}\n" for a, b, c in f)
    ext = np.round(geom.extents, 3)
    print(f"{name:7s} <- {src_id:16s} verts={len(v):5d} faces={len(f):5d} extents={ext}")

    # Grasp metadata (object frame): close the fingers across the narrower horizontal axis, at
    # the centroid of the mid cross-section along the longer axis (matters for the curved banana).
    long_ax = 0 if geom.extents[0] >= geom.extents[1] else 1
    mid = np.abs(v[:, long_ax]) < 0.1 * geom.extents[long_ax]
    off = v[mid].mean(axis=0) if mid.any() else np.zeros(3)
    off[long_ax] = 0.0
    width = float(np.ptp(v[mid][:, 1 - long_ax])) if mid.any() else float(geom.extents[1 - long_ax])
    meta = dict(grasp_offset=[round(float(off[0]), 4), round(float(off[1]), 4)],
                close_axis="xy"[1 - long_ax], grasp_width=round(width, 4),
                height=round(float(geom.extents[2]), 4))
    if name in CAPSULE_CHAIN:
        meta["capsules"] = capsule_chain(v, long_ax, CAPSULE_CHAIN[name])
    with open(os.path.join(OUT, name, "grasp.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"        grasp: {meta}")


def main():
    for name, (src_id, node, ext) in OBJECTS.items():
        convert(name, src_id, node, ext)
    with open(os.path.join(OUT, "SOURCES.txt"), "w") as fh:
        fh.write("Object meshes converted from Poly Haven (https://polyhaven.com), license CC0.\n")
        for name, (src_id, _, _) in OBJECTS.items():
            fh.write(f"{name}: https://polyhaven.com/a/{src_id}\n")


if __name__ == "__main__":
    main()
