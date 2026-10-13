"""Air Hockey (paper Fig. 5a, Table 3 "AH"): a puck slides on a frictionless air-hockey
surface at a randomised initial velocity; the robot, holding a paddle (mallet), must deflect
it back up the table."""
from dataclasses import dataclass

import numpy as np

from ..base_env import TABLE_Z, BaseConfig, CameraSpec, DynamicEnv
from ..builder import STATIC
from ..components import DARK, add_air_hockey_table, add_puck
from ..experts import AirHockeyExpert


@dataclass
class AirHockeyConfig(BaseConfig):
    episode_seconds: float = 8.0
    speed_range: tuple = (0.5, 1.2)          # m/s initial puck speed
    start_x_range: tuple = (1.45, 1.85)
    start_y_range: tuple = (-0.35, 0.35)
    target_y_range: tuple = (-0.30, 0.30)    # where the puck crosses the defence line
    bank_prob: float = 0.35                  # probability of a bank shot off a side rail
    deflect_dist: float = 0.30               # puck must travel this far back after contact


class AirHockey(DynamicEnv):
    Config = AirHockeyConfig
    family = "complex"
    attachment = "mallet"
    control_site = "mallet_center"
    gripper_fixed = 80.0            # fingers clamp the mallet handle
    table = None
    home_pos = (0.42, 0.0, TABLE_Z + 0.08)
    workspace = ((0.25, -0.5, TABLE_Z + 0.012), (0.85, 0.5, TABLE_Z + 0.5))
    camera = CameraSpec(lookat=(0.95, 0.0, TABLE_Z), distance=2.7, azimuth=230, elevation=-32)
    mallet_height = 0.013           # mallet-disk centre above the surface
    x_def = 0.55                    # defence line used to aim shots / by the expert

    def build(self, b):
        self.ah = add_air_hockey_table(b, "ah", x0=0.18, length=2.0, width=1.1, top_z=TABLE_Z)
        self.puck = add_puck(b, "puck", top_z=TABLE_Z)
        ped = b.body(name="pedestal")
        b.box(ped, [0.12, 0.12, TABLE_Z / 2], pos=[0.0, 0.0, TABLE_Z / 2], rgba=DARK, col=STATIC)

    def setup(self):
        j = [self.jid(n) for n in self.puck["joints"]]
        self.p_qadr = self.model.jnt_qposadr[j]
        self.p_dadr = self.model.jnt_dofadr[j]
        self.model.body_pos[self.bid("puck")] = [0, 0, TABLE_Z + 0.0075]
        self.puck_geoms = self.body_geoms("puck")
        mallet = self.bid("panda/mallet")
        self.mallet_geoms = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == mallet}

    def reset_task(self, rng):
        c = self.cfg
        lim = self.ah["width"] / 2 - self.puck["radius"]
        p0 = np.array([rng.uniform(*c.start_x_range), rng.uniform(*c.start_y_range)])
        y_t = rng.uniform(*c.target_y_range)
        if rng.random() < c.bank_prob:          # aim at the mirror image -> one rail bounce
            y_t = np.sign(rng.uniform(-1, 1)) * 2 * lim - y_t
        d = np.array([self.x_def, y_t]) - p0
        v = rng.uniform(*c.speed_range) * d / np.linalg.norm(d)
        self.data.qpos[self.p_qadr] = [p0[0], p0[1], rng.uniform(-np.pi, np.pi)]
        self.data.qvel[self.p_dadr] = [v[0], v[1], 0.0]
        self.v0 = v
        self.contact_x = None
        self.contact_t = None
        self.instruction = "hit the puck back with the paddle"

    def puck_state(self):
        p = np.array([*self.data.qpos[self.p_qadr[:2]], TABLE_Z])
        v = np.array([*self.data.qvel[self.p_dadr[:2]], 0.0])
        return p, v

    def post_substep(self):
        if self.contact_x is None and self.touching(self.mallet_geoms, self.puck_geoms):
            self.contact_x = float(self.data.qpos[self.p_qadr[0]])
            self.contact_t = float(self.data.time)

    def evaluate(self):
        p, v = self.puck_state()
        if self.contact_x is not None:
            if v[0] > 0.05 and p[0] - self.contact_x >= self.cfg.deflect_dist:
                return True, False
        elif p[0] < self.ah["x0"] + 0.12:
            self.events["puck_passed"] = float(self.data.time)
            return False, True
        return False, False

    def make_expert(self, **kw):
        return AirHockeyExpert(self, x_def=self.x_def, **kw)

    def task_info(self):
        p, v = self.puck_state()
        return dict(puck_speed0=float(np.linalg.norm(self.v0)), contact_time=self.contact_t,
                    puck_vx=float(v[0]))
