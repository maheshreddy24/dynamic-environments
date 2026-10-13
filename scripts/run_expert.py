"""Roll out the scripted privileged-state expert on one or more scenarios and report the
success rate (sanity check that every scenario is solvable and its success check fires).

    MUJOCO_GL=egl .venv/bin/python scripts/run_expert.py ConveyorCup-v0 --episodes 20
    MUJOCO_GL=egl .venv/bin/python scripts/run_expert.py all --episodes 10 --workers 8
    MUJOCO_GL=egl .venv/bin/python scripts/run_expert.py PlinkoDrop-v0 --kw window_ms=1600 --video out.mp4
"""
import argparse
import ast
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def parse_kw(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def run(name, episodes, seed0, kw, video=None, verbose=False):
    import dynamic_envs
    env = dynamic_envs.make(name, action_mode="ee_abs", obs_mode="state", **kw)
    results = []
    for ep in range(episodes):
        seed = seed0 + ep
        env.reset(seed=seed)
        expert = env.make_expert()
        frames = []
        info = {}
        for _ in range(env.max_steps):
            _, _, term, trunc, info = env.step(expert())
            if video and ep == 0:
                frames.append(env.render_camera("agentview", 448, 448))
            if term or trunc:
                break
        results.append(dict(seed=seed, success=bool(info.get("success")), time=round(info.get("time", 0), 2),
                            events=dict(env.events)))
        if verbose:
            print(f"  {name} seed={seed} success={info.get('success')} t={info.get('time', 0):.2f} "
                  f"events={env.events}", flush=True)
        if video and ep == 0:
            import imageio
            imageio.mimsave(video, frames, fps=30)
    env.close()
    return name, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="+", help="scenario names or 'all'")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--kw", nargs="*", help="config overrides, e.g. window_ms=800")
    ap.add_argument("--video", help="save the first episode (single scenario only)")
    ap.add_argument("--json", help="write per-episode results here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    import dynamic_envs
    names = dynamic_envs.available() if args.scenarios == ["all"] else args.scenarios
    kw = parse_kw(args.kw)
    t0 = time.time()
    jobs = []
    # Split each scenario's episodes into chunks so all workers stay busy.
    chunk = max(1, args.episodes // max(1, args.workers // max(1, len(names))))
    with ProcessPoolExecutor(args.workers) as pool:
        for n in names:
            for start in range(0, args.episodes, chunk):
                k = min(chunk, args.episodes - start)
                video = args.video if (args.video and start == 0) else None
                jobs.append(pool.submit(run, n, k, args.seed + start, kw, video, args.verbose))
        merged = {}
        for j in jobs:
            n, res = j.result()
            merged.setdefault(n, []).extend(res)
    print(f"\n{'scenario':28s} {'success':>9s}")
    for n in names:
        r = merged[n]
        s = np.mean([x["success"] for x in r])
        print(f"{n:28s} {100 * s:7.1f}%  ({sum(x['success'] for x in r)}/{len(r)})")
    print(f"[{time.time() - t0:.0f}s]")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(merged, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
