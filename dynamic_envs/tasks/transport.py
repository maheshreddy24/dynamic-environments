"""Constant-velocity / acceleration transport scenarios (paper Fig. 4, Table 1):
Conveyor + cup, Beam + cup, Pole push + cup."""
from dataclasses import dataclass

import numpy as np

from ..base_env import TABLE_Z, CameraSpec
from ..components import add_beam, add_conveyor, add_object, add_pole_pusher
from ..geometry import quat_axis_angle
from .common import MotionConfig, PickMovingTask, SpeedProfile


# ============================================================================= conveyor
@dataclass
class ConveyorCupConfig(MotionConfig):
    speed_range: tuple = (0.05, 0.25)            # paper: 5-25 cm/s
    belt_height_range: tuple = (0.0, 0.06)       # belt top above the table top
    spawn_range: tuple = (0.38, 0.62)            # |y| of the cup at t = 0 (upstream side)


class ConveyorCup(PickMovingTask):
    """A conveyor belt transports a red cup across the workspace; the robot must pick the cup
    before it leaves.  Belt speed, cup spawn position along the belt and belt height are
    randomised per episode.  `profile="accel"` ramps the belt speed linearly up or down."""

    Config = ConveyorCupConfig
    family = "constant_velocity"
    table = dict(center=(0.1, 0.0), size=(0.7, 1.1))
    camera = CameraSpec(lookat=(0.45, 0.0, TABLE_Z + 0.08), distance=2.1, azimuth=215, elevation=-24)
    conveyor_x = 0.62

    def build(self, b):
        add_conveyor(b, "conveyor", length=2.6)
        add_object(b, "cup", "cup")

    def setup(self):
        super().setup()
        self.belt = self.gid("conveyor_belt")
        self.conv = self.bid("conveyor")

    def reset_task(self, rng):
        c = self.cfg
        h = rng.uniform(*c.belt_height_range)
        x = self.conveyor_x + rng.uniform(-0.03, 0.03)
        self.model.body_pos[self.conv] = [x, 0.0, TABLE_Z + h]
        self.model.body_quat[self.conv] = quat_axis_angle([0, 0, 1], np.pi / 2)  # belt runs along world y
        self.dir = c.direction if c.direction is not None else rng.choice([-1, 1])
        self.speed = SpeedProfile(c, rng)
        y0 = -self.dir * rng.uniform(*c.spawn_range)
        if self.speed.v0 == 0 and self.speed.a == 0:     # static belt (speed-sweep 0 cm/s point)
            y0 = rng.uniform(-0.25, 0.25)
        self.set_free_pose("cup", [x + rng.uniform(-0.03, 0.03), y0, TABLE_Z + h + 1e-4],
                           linvel=[0, self.dir * self.speed(0.0), 0])
        self.pre_substep(self.data)
        self.instruction = "pick up the red cup"

    def pre_substep(self, data):
        self.model.geom_surfacevel[self.belt, 0] = self.dir * self.speed(data.time)

    def task_info(self):
        return dict(super().task_info(), belt_speed=self.dir * self.speed(self.data.time), **self.speed.as_dict())


# ============================================================================= beam
@dataclass
class BeamCupConfig(MotionConfig):
    speed_range: tuple = (0.05, 0.20)            # paper: 5-20 cm/s
    yaw_range_deg: tuple = (-30.0, 30.0)         # beam orientation w.r.t. the lateral axis
    start_range: tuple = (0.25, 0.40)            # cup start distance from the path centre


class BeamCup(PickMovingTask):
    """A red cup slides along a horizontal post-mounted beam (guided by a hidden track that
    runs alongside it); the robot must grasp the cup from above.  Beam orientation and
    starting position are randomised."""

    Config = BeamCupConfig
    family = "constant_velocity"
    camera = CameraSpec(lookat=(0.45, 0.0, TABLE_Z + 0.08), distance=2.0, azimuth=210, elevation=-26)

    def build(self, b):
        self.beam_info = add_beam(b, "beam")
        add_object(b, "cup", "cup")

    def setup(self):
        super().setup()
        self.track = self.gid("beam_track")
        self.beam = self.bid("beam")

    def reset_task(self, rng):
        c = self.cfg
        bl, off = self.beam_info["beam_len"], self.beam_info["track_offset"]
        phi = np.pi / 2 + np.deg2rad(rng.uniform(*c.yaw_range_deg))   # path direction (~ world y)
        u = np.array([np.cos(phi), np.sin(phi), 0.0])
        n = np.array([-u[1], u[0], 0.0])
        center = np.array([rng.uniform(0.44, 0.54), rng.uniform(-0.05, 0.05), TABLE_Z])
        # Beam body: local x = u, track at local (-bl/2, off) -> path centre.
        self.model.body_pos[self.beam] = center - (-bl / 2) * u - off * n
        self.model.body_quat[self.beam] = quat_axis_angle([0, 0, 1], phi)
        self.dir = c.direction if c.direction is not None else rng.choice([-1, 1])
        self.u = u
        self.speed = SpeedProfile(c, rng)
        s0 = -self.dir * rng.uniform(*c.start_range)
        self.set_free_pose("cup", center + s0 * u + [0, 0, 0.002 + 1e-4], linvel=self.dir * self.speed(0.0) * u)
        self.pre_substep(self.data)
        self.instruction = "pick up the red cup"

    def pre_substep(self, data):
        self.model.geom_surfacevel[self.track, 0] = self.dir * self.speed(data.time)

    def motion_yaw(self):
        return float(np.arctan2(self.u[1], self.u[0]))

    def task_info(self):
        return dict(super().task_info(), **self.speed.as_dict())


# ============================================================================= pole push
@dataclass
class PolePushCupConfig(MotionConfig):
    speed_range: tuple = (0.08, 0.22)            # paper: 8-22 cm/s
    yaw_range_deg: tuple = (-30.0, 30.0)         # push direction w.r.t. the lateral axis
    start_range: tuple = (0.25, 0.38)
    stop_time_range: tuple = (2.5, 6.0)          # "accel" profile: friction-like decel to rest


class PolePushCup(PickMovingTask):
    """A pole pushes a cup laterally across the table; the robot must intercept the cup.
    Push direction and starting offset are randomised.  `profile="accel"` applies frictional
    (constant) deceleration until the pole comes to rest."""

    Config = PolePushCupConfig
    family = "constant_velocity"
    camera = CameraSpec(lookat=(0.45, 0.0, TABLE_Z + 0.08), distance=2.0, azimuth=210, elevation=-26)

    def build(self, b):
        add_pole_pusher(b, "pusher")
        add_object(b, "cup", "cup")

    def setup(self):
        super().setup()
        j = [self.jid(n) for n in ("pusher_x", "pusher_y", "pusher_yaw")]
        self.p_qadr = self.model.jnt_qposadr[j]
        self.p_dadr = self.model.jnt_dofadr[j]
        self.model.body_pos[self.bid("pusher")] = [0, 0, TABLE_Z + 0.002]

    def reset_task(self, rng):
        c = self.cfg
        phi = np.pi / 2 + np.deg2rad(rng.uniform(*c.yaw_range_deg))
        self.dir = c.direction if c.direction is not None else rng.choice([-1, 1])
        if self.dir < 0:
            phi += np.pi
        self.u = np.array([np.cos(phi), np.sin(phi), 0.0])
        self.phi = phi
        center = np.array([rng.uniform(0.44, 0.54), rng.uniform(-0.05, 0.05), 0.0])
        self.start = center - rng.uniform(*c.start_range) * self.u
        self.v0 = c.speed if c.speed is not None else rng.uniform(*c.speed_range)
        self.decel = 0.0
        if c.profile == "accel":
            self.decel = self.v0 / rng.uniform(*c.stop_time_range)
        elif c.profile != "constant":
            raise ValueError(c.profile)
        self.pre_substep(self.data)
        cup_r = 0.03
        self.set_free_pose("cup", self.start + self.u * (cup_r + 0.002) + [0, 0, TABLE_Z + 1e-4],
                           linvel=self.u * self.v0)
        self.instruction = "pick up the red cup"

    def _s(self, t):
        if self.decel <= 0:
            return self.v0 * t, self.v0
        ts = self.v0 / self.decel
        tt = min(t, ts)
        return self.v0 * tt - 0.5 * self.decel * tt * tt, max(self.v0 - self.decel * t, 0.0)

    def pre_substep(self, data):
        s, v = self._s(data.time)
        data.qpos[self.p_qadr] = [self.start[0] + s * self.u[0], self.start[1] + s * self.u[1], self.phi]
        data.qvel[self.p_dadr] = [v * self.u[0], v * self.u[1], 0.0]

    def motion_yaw(self):
        return float(np.arctan2(self.u[1], self.u[0]))

    def task_info(self):
        return dict(super().task_info(), v0=self.v0, decel=self.decel)
