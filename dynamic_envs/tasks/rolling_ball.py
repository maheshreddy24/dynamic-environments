"""Rolling ball (paper Fig. 4d, Table 1): a ball rolls along a track across the table at a
constant horizontal velocity of 10-30 cm/s (or accelerates down an inclined track with a
5-15 degree slope); the robot must stop it."""
from dataclasses import dataclass

import numpy as np

from ..base_env import TABLE_Z, BaseConfig, CameraSpec, DynamicEnv
from ..components import STEEL, add_ball, add_channel, set_geom_box
from ..experts import BlockBallExpert
from ..geometry import quat_axis_angle, quat_mul, quat_to_mat
from ..success import StopTracker

BALL_R = 0.02
UNDER = 0.056   # depth of the track structure below its floor


@dataclass
class RollingBallConfig(BaseConfig):
    episode_seconds: float = 12.0
    speed_range: tuple = (0.10, 0.30)        # constant-velocity variant: 10-30 cm/s
    slope_range_deg: tuple = (5.0, 15.0)     # incline variant
    incline_v0_range: tuple = (0.0, 0.05)
    profile: str = "constant"                # "constant" | "incline"
    yaw_range_deg: tuple = (-25.0, 25.0)


class RollingBall(DynamicEnv):
    Config = RollingBallConfig
    family = "constant_velocity"
    camera = CameraSpec(lookat=(0.45, 0.0, TABLE_Z + 0.08), distance=2.0, azimuth=210, elevation=-26)
    track_len = 0.9

    def build(self, b):
        self.ch = add_channel(b, "channel", length=self.track_len)
        add_ball(b, "ball", BALL_R, rgba=STEEL, mass=0.06)

    def setup(self):
        self.stop = StopTracker(self, "ball")
        self.ch_body = self.bid("channel")
        self.sup_body = self.bid("channel_support_body")

    def trackers(self):
        return [self.stop]

    def reset_task(self, rng):
        c = self.cfg
        L = self.track_len
        theta = np.deg2rad(rng.uniform(*c.slope_range_deg)) if c.profile == "incline" else 0.0
        if c.profile not in ("constant", "incline"):
            raise ValueError(c.profile)
        phi = np.pi / 2 + np.deg2rad(rng.uniform(*c.yaw_range_deg)) + (np.pi if rng.random() < 0.5 else 0.0)
        q = quat_mul(quat_axis_angle([0, 0, 1], phi), quat_axis_angle([0, 1, 0], theta))  # local +x downhill
        R = quat_to_mat(q)
        zc = TABLE_Z + (L / 2) * np.sin(theta) + UNDER * np.cos(theta) + 0.002
        center = np.array([rng.uniform(0.44, 0.52), rng.uniform(-0.04, 0.04), zc])
        self.model.body_pos[self.ch_body] = center
        self.model.body_quat[self.ch_body] = q
        self.center, self.R, self.theta = center, R, theta
        # Support block under the high end.
        hi = center + R @ [-L / 2 + 0.07, 0, 0]
        z_bot = hi[2] - UNDER * np.cos(theta)
        h = max(z_bot - TABLE_Z, 0.004)
        self.model.body_pos[self.sup_body] = [hi[0], hi[1], TABLE_Z]
        self.model.body_quat[self.sup_body] = quat_axis_angle([0, 0, 1], phi)
        set_geom_box(self.model, "channel_support", [0.05, self.ch["inner_w"] / 2 + 0.012, h / 2], [0, 0, h / 2])

        v0 = rng.uniform(*c.speed_range) if c.profile == "constant" else rng.uniform(*c.incline_v0_range)
        x_axis, n = R[:, 0], R[:, 2]
        p0 = center + R @ [-L / 2 + 0.05, 0, BALL_R + 1e-4]
        v = v0 * x_axis
        self.set_free_pose("ball", p0, linvel=v, angvel=np.cross(n, v) / BALL_R)
        self.v0 = v0
        self.instruction = "stop the rolling ball"

    def track_axis(self):
        return self.R[:, 0].copy()

    def floor_height(self, p):
        s = np.dot(np.asarray(p) - self.center, self.R[:, 0])
        return self.center[2] + s * self.R[2, 0]

    def evaluate(self):
        self.stop.substep()
        if self.stop.result is not None:
            return self.stop.result, not self.stop.result
        p = self.body_pos("ball")
        s = np.dot(p - self.center, self.R[:, 0])
        if self.stop.first_contact is None and (s > self.track_len / 2 + 0.02 or p[2] < TABLE_Z - 0.1):
            self.events["left_track"] = float(self.data.time)
            return False, True
        return False, False

    def post_substep(self):
        self.stop.substep()

    def make_expert(self, **kw):
        a = (5.0 / 7.0) * 9.81 * np.sin(self.theta)
        return BlockBallExpert(self, "ball", self.track_axis, self.floor_height, accel=a, **kw)

    def task_info(self):
        return dict(v0=self.v0, slope_deg=float(np.rad2deg(self.theta)),
                    first_contact=self.stop.first_contact, pre_contact_speed=self.stop.v_pre)
