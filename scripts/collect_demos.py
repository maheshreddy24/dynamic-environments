"""Collect scripted-expert demonstrations (paper App. C.3: "scripted rollouts from a
privileged-state planner, 500 per scenario").

One compressed .npz per episode in <out>/<scenario>/ plus manifest.json.  Each .npz holds,
for T control steps (30 Hz):

    images          (T, H, W, 3) uint8    agentview camera *before* the action (omitted with --no-images)
    states          (T, 15) float32       TCP pos(3) + quat(4), gripper width, 7 joint angles
    actions_abs     (T, 8)  float32       expert command: TCP target pos(3) + quat(4), gripper (-1 open / +1 closed)
    actions_delta   (T, 7)  float32       measured TCP motion t -> t+1 in the env's ee_delta units
                                          (dpos / 0.03 m, drot / 0.15 rad, gripper), clipped to [-1, 1]
    object_states   (T, 6)  float32       privileged target pos(3) + linear velocity(3)
    rewards         (T,)    float32
    instruction, seed, success, scenario  (scalars)
    goal (3,), grasp_step, reached_step   (goal-conditioned scenarios only, e.g. ConveyorCupGoal-v0;
                                          episodes there have a fixed length, the robot keeps holding
                                          the object at the goal after `reached_step`)

    MUJOCO_GL=egl .venv/bin/python scripts/collect_demos.py ConveyorCup-v0 --episodes 500
    MUJOCO_GL=egl .venv/bin/python scripts/collect_demos.py ConveyorCup-v0 --episodes 50 --kw speed=0.3 profile=accel
"""
import argparse
import ast
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import dynamic_envs
from dynamic_envs.geometry import quat_error
import numpy as np
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")


def target_body(env):
    """Body whose privileged state is logged (the object the task is about)."""
    t = getattr(env, "target", None)
    if isinstance(t, str):                       # pick tasks: "cup" or an object kind ("duck" -> "obj_duck")
        return t if env.bid(t) >= 0 else f"obj_{t}"
    if isinstance(t, (int, np.integer)):          # catching tasks: index of the instructed ball
        return f"ball{t}"
    for name in ("puck", "ball"):
        if env.bid(name) >= 0:
            return name
    return None


def rollout(name, seed, kw, images, image_size):
    # import dynamic_envs
    # from dynamic_envs.geometry import quat_error

    env = dynamic_envs.make(name, action_mode="ee_abs", obs_mode="rgb_state" if images else "state",
                            render_size=image_size, **kw)
    obs, info = env.reset(seed=seed)
    expert = env.make_expert()
    body = target_body(env)
    buf = {k: [] for k in ("images", "states", "actions_abs", "object_states", "rewards")}
    while True:
        action = expert()
        if images:
            buf["images"].append(obs["image"])
        buf["states"].append(obs["state"])
        buf["actions_abs"].append(action)
        if body is not None:
            buf["object_states"].append(np.concatenate([env.body_pos(body), env.body_linvel(body)]))
        obs, reward, terminated, truncated, info = env.step(action)
        buf["rewards"].append(reward)
        if terminated or truncated:
            break
    states = np.asarray(buf["states"] + [obs["state"]], np.float32)   # T + 1 states for deltas
    dpos = (states[1:, :3] - states[:-1, :3]) / env.pos_scale
    drot = np.array([quat_error(states[i + 1, 3:7], states[i, 3:7]) for i in range(len(states) - 1)]) / env.rot_scale
    grip = np.asarray(buf["actions_abs"])[:, 7:8]
    ep = dict(
        states=states[:-1],
        actions_abs=np.asarray(buf["actions_abs"], np.float32),
        actions_delta=np.clip(np.concatenate([dpos, drot, grip], 1), -1, 1).astype(np.float32),
        rewards=np.asarray(buf["rewards"], np.float32),
        instruction=np.array(env.instruction),
        seed=np.array(seed),
        success=np.array(bool(info["success"])),
        scenario=np.array(name),
    )
    if images:
        ep["images"] = np.asarray(buf["images"], np.uint8)
    if buf["object_states"]:
        ep["object_states"] = np.asarray(buf["object_states"], np.float32)
    meta = dict(seed=seed, success=bool(info["success"]), steps=len(buf["rewards"]),
                instruction=env.instruction, events=dict(env.events))
    # Goal-conditioned scenarios (e.g. ConveyorCupGoal-v0): goal + frame indices of grasp / reach
    # (-1 = never happened).  images[reached_step] is the first frame with the object at the goal.
    if "goal" in info:
        ep["goal"] = np.asarray(info["goal"], np.float32)
        for k in ("grasp_step", "reached_step"):
            ep[k] = np.array(-1 if info[k] is None else int(info[k]))
            meta[k] = int(ep[k])
        meta["goal"] = [round(float(x), 4) for x in info["goal"]]
    env.close()
    return ep, meta


def work(args):
    name, seeds, kw, out, images, image_size, keep_failed = args
    metas = []
    for s in tqdm(seeds):
        ep, meta = rollout(name, s, kw, images, image_size)
        if meta["success"] or keep_failed:
            path = out / f"ep_{s:05d}.npz"
            np.savez_compressed(path, **ep)
            meta["file"] = path.name
        metas.append(meta)
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="ConveyorCup-v0")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="first seed; episode i uses seed + i")
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--kw", nargs="*", default=[], help="scenario config overrides, e.g. speed=0.3")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--no-images", action="store_true", help="state-only (much smaller and faster)")
    ap.add_argument("--keep-failed", action="store_true", help="also save episodes the expert failed")
    args = ap.parse_args()

    kw = {k: ast.literal_eval(v) for k, v in (s.split("=", 1) for s in args.kw)}
    out = Path(args.out) / args.scenario.replace("-v0", "")
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.episodes))
    chunks = [seeds[i::args.workers] for i in range(args.workers) if seeds[i::args.workers]]
    t0 = time.time()
    with ProcessPoolExecutor(len(chunks)) as pool:
        jobs = [(args.scenario, c, kw, out, not args.no_images, args.image_size, args.keep_failed) for c in chunks]
        metas = sorted((m for ms in pool.map(work, jobs) for m in ms), key=lambda m: m["seed"])
    manifest = dict(scenario=args.scenario, config=kw, image_size=None if args.no_images else args.image_size,
                    control_hz=30, episodes=metas)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n_ok = sum(m["success"] for m in metas)
    saved = sum("file" in m for m in metas)
    print(f"{args.scenario}: {n_ok}/{len(metas)} expert successes, {saved} episodes saved to {out} "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
