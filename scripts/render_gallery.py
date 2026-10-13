"""Render a preview of every scenario: the 224x224 policy camera ("agentview") and a larger
overview, a few moments into an expert rollout.  Writes assets/gallery/<scenario>.png and a
contact sheet assets/gallery/all_scenarios.png.  Optionally saves an expert-rollout video.

    MUJOCO_GL=egl .venv/bin/python scripts/render_gallery.py                # all scenarios
    MUJOCO_GL=egl .venv/bin/python scripts/render_gallery.py BeamCup-v0 --video
"""
import argparse

import imageio
import numpy as np
from PIL import Image, ImageDraw

import dynamic_envs
from dynamic_envs.paths import ASSETS

OUT = ASSETS / "gallery"

# Seconds into the expert rollout at which the still is taken (when the action is visible).
SNAP_TIME = {
    "BallisticCatch-v0": 0.9, "MultiBalls-v0": 0.9, "Occlusion-v0": 0.45, "MidFlight-v0": 0.9,
    "MultipleDeflection-v0": 1.0, "OcclusionDeflection-v0": 0.6, "PlinkoDrop-v0": 0.7, "AirHockey-v0": 0.8,
    "RollingBall-v0": 0.5, "RollingBallIncline-v0": 0.4,
}


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 8, 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def render(name, seed, video, size):
    env = dynamic_envs.make(name, action_mode="ee_abs", obs_mode="state")
    env.reset(seed=seed)
    expert = env.make_expert()
    t_snap = SNAP_TIME.get(name, 1.5)
    frames, snap_a, snap_o = [], None, None
    steps = env.max_steps if video else int(t_snap / env.dt) + 1
    for i in range(steps):
        _, _, term, trunc, info = env.step(expert())
        if snap_a is None and env.data.time >= t_snap:
            snap_a = env.render_camera("agentview", 224, 224)
            snap_o = env.render_camera("overview", *size)
        if video:
            frames.append(env.render_camera("overview", *size))
        if term or trunc:
            break
    if snap_a is None:
        snap_a = env.render_camera("agentview", 224, 224)
        snap_o = env.render_camera("overview", *size)
    instr = env.instruction
    env.close()
    return snap_a, snap_o, frames, instr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--video", action="store_true", help="also save an mp4 of the expert rollout")
    args = ap.parse_args()
    names = args.scenarios or dynamic_envs.available()
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for n in names:
        a, o, frames, instr = render(n, args.seed, args.video, (640, 400))
        a_big = np.asarray(Image.fromarray(a).resize((400, 400), Image.NEAREST))
        tile = np.concatenate([label(o, n), label(a_big, "agentview 224x224")], axis=1)
        imageio.imwrite(OUT / f"{n.replace('-v0', '')}.png", tile)
        tiles.append(tile)
        if args.video and frames:
            imageio.mimsave(OUT / f"{n.replace('-v0', '')}.mp4", frames, fps=30)
        print(f"{n:28s} '{instr}'")
    if len(tiles) > 1:
        cols = 2
        h, w, _ = tiles[0].shape
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.full((rows * h, cols * w, 3), 255, np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
        Image.fromarray(sheet).resize((cols * w // 2, rows * h // 2), Image.LANCZOS).save(OUT / "all_scenarios.png")


if __name__ == "__main__":
    main()
