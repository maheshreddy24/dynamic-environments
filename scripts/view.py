"""Watch a scenario in the interactive MuJoCo viewer (needs a display).

    .venv/bin/python scripts/view.py ConveyorCup-v0                  # scripted expert drives the robot
    .venv/bin/python scripts/view.py PlinkoDrop-v0 --kw window_ms=1600
    .venv/bin/python scripts/view.py AirHockey-v0 --policy idle       # robot holds still

Press Ctrl+C (or close the window) to quit; a new episode starts when one ends.
"""
import argparse
import ast
import time

import mujoco.viewer
import numpy as np

import dynamic_envs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--policy", choices=["expert", "idle", "random"], default="expert")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kw", nargs="*", default=[], help="config overrides, e.g. speed=0.3")
    ap.add_argument("--camera", default="overview", help="'agentview', 'overview' or 'free'")
    args = ap.parse_args()
    kw = {k: ast.literal_eval(v) for k, v in (s.split("=", 1) for s in args.kw)}

    mode = "ee_abs" if args.policy == "expert" else "ee_delta"
    env = dynamic_envs.make(args.scenario, action_mode=mode, obs_mode="state", **kw)
    seed = args.seed
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        if args.camera != "free":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = env.model.camera(args.camera).id
        while viewer.is_running():
            env.reset(seed=seed)
            print(f"[seed {seed}] {env.instruction!r}")
            expert = env.make_expert() if args.policy == "expert" else None
            rng = np.random.default_rng(seed)
            while viewer.is_running():
                t0 = time.time()
                if expert is not None:
                    a = expert()
                elif args.policy == "random":
                    a = rng.uniform(-1, 1, 7)
                else:
                    a = np.zeros(7)
                _, _, term, trunc, info = env.step(a)
                viewer.sync()
                time.sleep(max(0.0, env.dt - (time.time() - t0)))
                if term or trunc:
                    print(f"  success={info['success']} t={info['time']:.2f}s events={env.events}")
                    time.sleep(1.0)
                    break
            seed += 1


if __name__ == "__main__":
    main()
