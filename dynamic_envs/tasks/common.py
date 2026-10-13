"""Shared pieces for the task modules."""
from dataclasses import dataclass

import numpy as np

from ..base_env import TABLE_Z, BaseConfig, DynamicEnv
from ..experts import PickExpert
from ..success import GraspTracker


@dataclass
class MotionConfig(BaseConfig):
    episode_seconds: float = 20.0
    speed_range: tuple = (0.05, 0.25)     # m/s
    speed: float = None                   # fix the (initial) speed, e.g. for the speed sweep
    profile: str = "constant"             # "constant" | "accel" (linear ramp up or down)
    accel_range: tuple = (0.015, 0.04)    # |dv/dt| [m/s^2] for the accel profile
    direction: int = None                 # +1 / -1 travel direction, None = random


class SpeedProfile:
    """v(t) = clip(v0 + a t, v_min, v_max): constant velocity (a = 0) or a linear ramp."""

    def __init__(self, cfg, rng, v_min=0.02, v_max=0.5):
        lo, hi = cfg.speed_range
        self.a = 0.0
        if cfg.profile == "constant":
            self.v0 = cfg.speed if cfg.speed is not None else rng.uniform(lo, hi)
        elif cfg.profile == "accel":
            sign = rng.choice([-1.0, 1.0])
            mag = rng.uniform(*cfg.accel_range)
            if cfg.speed is not None:
                self.v0 = cfg.speed
            else:
                self.v0 = rng.uniform(lo, (lo + hi) / 2) if sign > 0 else rng.uniform((lo + hi) / 2, hi)
            self.a = sign * mag
        else:
            raise ValueError(f"unknown profile {cfg.profile!r}")
        self.v_min, self.v_max = (0.0 if self.v0 == 0 else v_min), v_max

    def __call__(self, t):
        return float(np.clip(self.v0 + self.a * t, self.v_min, self.v_max))

    def as_dict(self):
        return dict(v0=self.v0, accel=self.a)


class PickMovingTask(DynamicEnv):
    """Grasp a moving object, lift >= 10 cm and hold 3 s (App. C.4 pick-and-place criterion)."""

    Config = MotionConfig
    target = "cup"
    exit_y = 0.62           # |y| beyond which an un-grasped target has left the workspace

    def setup(self):
        self.grasp = GraspTracker(self, self.target)
        self.table_geom = self.gid("table_top")

    def trackers(self):
        return [self.grasp]

    def evaluate(self):
        r = self.grasp.update(self.dt)
        if r is not None:
            return r, not r
        p = self.body_pos(self.target)
        if self.grasp.z_grasp is None and (abs(p[1]) > self.exit_y or p[2] < TABLE_Z - 0.2):
            self.events["left_workspace"] = float(self.data.time)
            return False, True
        return False, False

    def target_grasp_z(self):
        from ..components import OBJECT_LIBRARY
        return OBJECT_LIBRARY[self.target_kind()].grasp_z

    def target_kind(self):
        return "cup"

    def motion_yaw(self):
        """Gripper yaw that closes the fingers perpendicular to the object's motion."""
        return np.pi / 2

    def make_expert(self, **kw):
        kw.setdefault("yaw", self.motion_yaw())
        return PickExpert(self, self.target, grasp_z=self.target_grasp_z(), **kw)

    def task_info(self):
        return dict(target=self.target, grasped=self.grasp.z_grasp is not None, hold_time=self.grasp.hold_time)
