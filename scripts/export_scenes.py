"""Export a standalone MJCF snapshot (initial state of one randomised episode) of every
scenario to assets/scenes/<scenario>.xml and check that each file loads.

    .venv/bin/python scripts/export_scenes.py
    .venv/bin/python -m mujoco.viewer --mjcf=assets/scenes/ConveyorCup.xml   # inspect one
"""
import argparse

import mujoco

import dynamic_envs
from dynamic_envs.paths import SCENES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    SCENES.mkdir(parents=True, exist_ok=True)
    for name in args.scenarios or dynamic_envs.available():
        env = dynamic_envs.make(name, obs_mode="state")
        env.reset(seed=args.seed)
        path = SCENES / f"{name.replace('-v0', '')}.xml"
        env.export_xml(str(path))
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        print(f"{name:28s} -> {path.relative_to(SCENES.parent.parent)}  (nbody={m.nbody}, ngeom={m.ngeom})")


if __name__ == "__main__":
    main()
