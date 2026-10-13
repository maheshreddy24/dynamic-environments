"""Base class shared by all dynamic-manipulation scenarios.

Setup (paper, Appendix C.1): MuJoCo, Franka Panda with a parallel-jaw gripper, a fixed
third-person RGB camera rendered at 224x224, control at 30 Hz, procedural randomisation of
object positions, motion parameters, lighting direction and camera pose.
"""
from dataclasses import dataclass, fields, replace

import gymnasium as gym
import mujoco
import numpy as np

from .builder import SceneBuilder
from .components import add_camera, add_environment, add_panda, add_table
from .geometry import normalize, quat_mul, rotvec_to_quat, spherical_camera, top_down_quat
from .robot import GRIP_CTRL_MAX, MENAGERIE_HOME, PandaArm

TABLE_Z = 0.75


@dataclass
class CameraSpec:
    lookat: tuple = (0.45, 0.0, TABLE_Z + 0.1)
    distance: float = 2.0
    azimuth: float = 210.0
    elevation: float = -24.0
    fovy: float = 45.0


@dataclass
class BaseConfig:
    """Fields shared by every task config (task configs subclass this)."""
    episode_seconds: float = 15.0


class DynamicEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # ---- scenario-level knobs (overridden by subclasses) --------------------------------
    Config = BaseConfig
    family = "base"             # constant_velocity | acceleration | complex | stress
    attachment = None           # None | "net" | "mallet"
    control_site = "tcp"
    gripper_fixed = None        # gripper ctrl held constant when a tool is clamped
    robot_base = (0.0, 0.0, TABLE_Z)
    table = dict(center=(0.25, 0.0), size=(1.0, 1.1))
    home_pos = (0.42, 0.0, TABLE_Z + 0.38)
    home_yaw = 0.0
    workspace = ((0.15, -0.6, TABLE_Z + 0.005), (0.85, 0.6, TABLE_Z + 0.8))
    camera = CameraSpec()
    control_hz = 30
    terminate_on_success = True   # False: keep simulating to the time limit after success (fixed-length episodes)

    # ---- action scaling for the OpenVLA-style delta end-effector action ----------------
    pos_scale = 0.03   # m per control step at |a| = 1  (0.9 m/s)
    rot_scale = 0.15   # rad per control step at |a| = 1

    def __init__(self, render_size=224, action_mode="ee_delta", obs_mode="rgb_state",
                 randomize_lighting=True, randomize_camera=True, render_mode="rgb_array", **config):
        assert action_mode in ("ee_delta", "ee_abs", "joint")
        names = {f.name for f in fields(self.Config)}
        unknown = set(config) - names
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown config keys {sorted(unknown)}; valid: {sorted(names)}")
        self.cfg = replace(self.Config(), **config)
        self.render_size = render_size
        self.action_mode = action_mode
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.randomize_lighting = randomize_lighting
        self.randomize_camera = randomize_camera

        b = SceneBuilder(type(self).__name__)
        add_environment(b)
        if self.table is not None:
            add_table(b, **self.table)
        add_panda(b, self.robot_base, 0.0, self.attachment)
        self.build(b)
        add_camera(b, "agentview", fovy=self.camera.fovy)
        add_camera(b, "overview", fovy=self.camera.fovy)
        self.builder = b
        self.model = b.compile()
        self._compiled = {k: getattr(self.model, k).copy() for k in
                          ("body_pos", "body_quat", "geom_size", "geom_pos", "geom_quat")}
        self.data = mujoco.MjData(self.model)
        self._pre = mujoco.MjData(self.model)
        self.arm = PandaArm(self.model, self.data, control_site=self.control_site)
        self.n_substeps = max(1, int(round(1.0 / (self.control_hz * self.model.opt.timestep))))
        self.dt = self.n_substeps * self.model.opt.timestep
        self.max_steps = int(round(self.cfg.episode_seconds / self.dt))

        self._ids = {}
        self.setup()
        self._home_quat = top_down_quat(self.home_yaw)
        self.arm.q_rest = MENAGERIE_HOME.copy()
        self._home_q = self.arm.ik(self.home_pos, self._home_quat, q0=MENAGERIE_HOME, iters=300)
        self.arm.q_rest = self._home_q.copy()
        self._base_light = {i: (self.model.light_dir[i].copy(), self.model.light_diffuse[i].copy())
                            for i in range(self.model.nlight)}
        self._renderers = {}

        self.action_space = self._make_action_space()
        spaces = {}
        if "rgb" in obs_mode:
            spaces["image"] = gym.spaces.Box(0, 255, (render_size, render_size, 3), np.uint8)
        if "state" in obs_mode:
            spaces["state"] = gym.spaces.Box(-np.inf, np.inf, (15,), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.instruction = ""

    # ================================================================== to override
    def build(self, b):
        """Add task-specific bodies to the SceneBuilder `b`."""

    def setup(self):
        """Resolve ids after compilation."""

    def reset_task(self, rng):
        """Randomise and place task objects, set self.instruction."""

    def pre_substep(self, data):
        """Prescribed kinematics (belts, pushers ...) applied before every mj_step.
        Must only depend on `data` (it is also used for pre-rolls)."""

    def post_substep(self):
        """Contact / event bookkeeping at physics rate."""

    def evaluate(self):
        """Return (success, failure) at control rate."""
        return False, False

    def trackers(self):
        """Success trackers to reset at the start of every episode."""
        return []

    def task_info(self):
        return {}

    def privileged_state(self):
        """Ground-truth pose / linear velocity of every non-robot body that can move (free or
        jointed bodies and mocap fixtures), e.g. for planners or motion-informed baselines."""
        m, d = self.model, self.data
        out = {}
        for b in range(1, m.nbody):
            if b in self.arm.body_ids or (m.body_jntnum[b] == 0 and m.body_mocapid[b] < 0):
                continue
            if d.xpos[b][2] < -0.5:      # parked (unused) object
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            out[name] = dict(pos=d.xpos[b].copy(), quat=d.xquat[b].copy(), linvel=self.body_linvel(name))
        return out

    def make_expert(self):
        raise NotImplementedError

    # ================================================================== id helpers
    def bid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def gid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def jid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def sid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    def body_geoms(self, name):
        b = self.bid(name)
        return {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == b
                and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])}

    def free_qadr(self, body):
        j = self.model.body_jntadr[self.bid(body)]
        return self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]

    def set_free_pose(self, body, pos, quat=(1, 0, 0, 0), linvel=(0, 0, 0), angvel=(0, 0, 0), data=None):
        d = self.data if data is None else data
        qa, da = self.free_qadr(body)
        d.qpos[qa:qa + 3] = pos
        d.qpos[qa + 3:qa + 7] = quat
        d.qvel[da:da + 3] = linvel
        d.qvel[da + 3:da + 6] = angvel

    def body_pos(self, body, data=None):
        d = self.data if data is None else data
        return d.xpos[self.bid(body)].copy()

    def body_linvel(self, body, data=None):
        d = self.data if data is None else data
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.bid(body), v, 0)
        return v[3:].copy()

    def set_mocap(self, body, pos, quat=None):
        mid = self.model.body_mocapid[self.bid(body)]
        self.data.mocap_pos[mid] = pos
        if quat is not None:
            self.data.mocap_quat[mid] = quat

    def contacts(self, data=None):
        d = self.data if data is None else data
        for i in range(d.ncon):
            c = d.contact[i]
            yield c.geom1, c.geom2

    def touching(self, set_a, set_b, data=None):
        for g1, g2 in self.contacts(data):
            if (g1 in set_a and g2 in set_b) or (g2 in set_a and g1 in set_b):
                return True
        return False

    # ================================================================== gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        q0 = self._home_q + rng.uniform(-0.02, 0.02, 7)
        grip = self.gripper_fixed if self.gripper_fixed is not None else GRIP_CTRL_MAX
        self.arm.reset(q0, grip)
        self.arm.frozen = False
        mujoco.mj_forward(m, d)
        self._target_pos, self._target_quat = self.arm.site_pose()
        self._randomize_appearance(rng)
        self.steps = 0
        self.success = False
        self.failure = False
        self.events = {}
        for tracker in self.trackers():
            tracker.reset()
        self.reset_task(rng)
        mujoco.mj_forward(m, d)
        self._target_pos, self._target_quat = self.arm.site_pose()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self._apply_action(np.asarray(action, dtype=float))
        for k in range(self.n_substeps):
            self.arm.apply((k + 1) / self.n_substeps)
            self.pre_substep(self.data)
            mujoco.mj_step(self.model, self.data)
            self.post_substep()
        self.steps += 1
        if not (self.success or self.failure):
            self.success, self.failure = self.evaluate()
        terminated = bool(self.failure or (self.success and self.terminate_on_success))
        truncated = bool(self.steps >= self.max_steps and not terminated)
        return self._get_obs(), float(self.success), terminated, truncated, self._get_info()

    def render(self):
        return self.render_camera("agentview", self.render_size, self.render_size)

    def render_camera(self, camera="agentview", width=None, height=None, supersample=2):
        """RGB image from a named camera.  Anti-aliased by rendering at `supersample` x the
        resolution and box-filtering down, which (unlike MSAA) is bit-exact deterministic."""
        width = width or self.render_size
        height = height or self.render_size
        ss = supersample if max(width, height) * supersample <= 1920 else 1
        key = (width * ss, height * ss)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=key[1], width=key[0])
        r = self._renderers[key]
        r.update_scene(self.data, camera=camera)
        img = r.render()
        if ss > 1:
            img = img.reshape(height, ss, width, ss, 3).mean(axis=(1, 3)).round().astype(np.uint8)
        return img

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    # ================================================================== internals
    def _make_action_space(self):
        if self.action_mode == "ee_delta":
            return gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        if self.action_mode == "ee_abs":
            lo = np.array([-2, -2, 0, -1, -1, -1, -1, -1], np.float32)
            return gym.spaces.Box(lo, -lo + np.array([0, 0, 2, 0, 0, 0, 0, 0], np.float32), (8,), np.float32)
        lo = np.concatenate([self.arm.q_lo, [-1]]).astype(np.float32)
        hi = np.concatenate([self.arm.q_hi, [1]]).astype(np.float32)
        return gym.spaces.Box(lo, hi, (8,), np.float32)

    def _apply_action(self, a):
        self.arm.frozen = self.robot_frozen()
        lo, hi = map(np.asarray, self.workspace)
        if self.action_mode == "ee_delta":
            a = np.clip(a, -1, 1)
            cur, _ = self.arm.site_pose()
            base = self._target_pos
            if np.linalg.norm(base - cur) > 0.12:   # avoid wind-up when the arm lags behind
                base = cur + 0.12 * normalize(base - cur)
            self._target_pos = np.clip(base + a[:3] * self.pos_scale, lo, hi)
            self._target_quat = normalize(quat_mul(rotvec_to_quat(a[3:6] * self.rot_scale), self._target_quat))
            g = a[6]
        elif self.action_mode == "ee_abs":
            self._target_pos = np.clip(a[:3], lo, hi)
            self._target_quat = normalize(a[3:7])
            g = a[7]
        else:
            q = a[:7]
            g = a[7]
        grip = GRIP_CTRL_MAX * (1.0 - np.clip(g, -1, 1)) / 2.0   # -1 open, +1 closed
        if self.gripper_fixed is not None:
            grip = self.gripper_fixed
        if self.action_mode != "joint":
            q = self.arm.ik(self._target_pos, self._target_quat)
        self.arm.command(q, grip, self.dt)

    def robot_frozen(self):
        return False

    def _randomize_appearance(self, rng):
        m = self.model
        for i, (dir0, dif0) in self._base_light.items():
            if self.randomize_lighting:
                az = rng.uniform(0, 2 * np.pi)
                el = rng.uniform(np.deg2rad(50), np.deg2rad(80))
                if i == 0:
                    m.light_dir[i] = normalize([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])
                m.light_diffuse[i] = dif0 * rng.uniform(0.8, 1.2)
            else:
                m.light_dir[i], m.light_diffuse[i] = dir0, dif0
        c = self.camera
        if self.randomize_camera:
            lookat = np.asarray(c.lookat) + rng.uniform(-0.03, 0.03, 3)
            dist = c.distance * rng.uniform(0.95, 1.05)
            az = c.azimuth + rng.uniform(-4, 4)
            el = c.elevation + rng.uniform(-3, 3)
        else:
            lookat, dist, az, el = np.asarray(c.lookat), c.distance, c.azimuth, c.elevation
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
        m.cam_pos[cid], m.cam_quat[cid] = spherical_camera(lookat, dist, az, el)
        oid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        m.cam_pos[oid], m.cam_quat[oid] = spherical_camera(c.lookat, c.distance * 1.15, c.azimuth,
                                                           min(c.elevation + 6.0, -10.0))

    def camera_pose(self, name="agentview"):
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        return self.model.cam_pos[cid].copy(), self.model.cam_quat[cid].copy()

    def proprio(self):
        pos, quat = self.arm.site_pose(self.arm.tcp_site)
        return np.concatenate([pos, quat, [self.arm.gripper_width], self.arm.q]).astype(np.float32)

    def _get_obs(self):
        obs = {}
        if "rgb" in self.obs_mode:
            obs["image"] = self.render()
        if "state" in self.obs_mode:
            obs["state"] = self.proprio()
        return obs

    def _get_info(self):
        info = dict(instruction=self.instruction, success=bool(self.success), failure=bool(self.failure),
                    time=float(self.data.time), family=self.family)
        info.update(self.task_info())
        return info

    # ================================================================== export
    def export_xml(self, path):
        """Write a standalone MJCF snapshot of the *current* episode (randomised body / mocap
        poses, resized geoms, belt speeds, camera and lights) with an 'init' keyframe, viewable
        with `python -m mujoco.viewer --mjcf=<path>`."""
        spec, m, d, c0 = self.builder.spec, self.model, self.data, self._compiled
        changed = lambda field, i: not np.allclose(getattr(m, field)[i], c0[field][i])
        for body in spec.bodies:
            i = body.id
            if i <= 0:
                continue
            mid = m.body_mocapid[i]
            if mid >= 0:
                body.pos, body.quat = d.mocap_pos[mid].copy(), d.mocap_quat[mid].copy()
            elif changed("body_pos", i) or changed("body_quat", i):   # only runtime-randomised bodies
                body.pos, body.quat = m.body_pos[i].copy(), m.body_quat[i].copy()
            body.gravcomp = float(m.body_gravcomp[i])
        for g in spec.geoms:
            i = g.id
            if g.type != mujoco.mjtGeom.mjGEOM_MESH and changed("geom_size", i):
                g.size = m.geom_size[i].copy()
            if changed("geom_pos", i) or changed("geom_quat", i):
                g.pos, g.quat = m.geom_pos[i].copy(), m.geom_quat[i].copy()
            g.contype, g.conaffinity = int(m.geom_contype[g.id]), int(m.geom_conaffinity[g.id])
            if np.any(m.geom_surfacevel[g.id]):
                g.surfacevel = m.geom_surfacevel[g.id].copy()
        for c in spec.cameras:
            c.pos, c.quat = m.cam_pos[c.id].copy(), m.cam_quat[c.id].copy()
        for li in spec.lights:
            li.dir, li.diffuse = m.light_dir[li.id].copy(), m.light_diffuse[li.id].copy()
        for k in list(spec.keys):
            spec.delete(k)
        spec.add_key(name="init", qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(),
                     mpos=d.mocap_pos.flatten(), mquat=d.mocap_quat.flatten(), time=float(d.time))
        return self.builder.to_xml(path)

    # ================================================================== pre-roll
    def preroll(self, duration, bodies, disable_robot=True, until=None):
        """Simulate the passive scene forward (robot collisions off) on a scratch copy of the
        current state and return {body: (T, 3) positions}, plus the time stamps.  Used to
        validate / time chaotic trajectories (deflections, Plinko) at reset."""
        m = self.model
        mujoco.mj_copyData(self._pre, m, self.data)
        saved = None
        if disable_robot:
            ids = np.array(sorted(self.arm.geom_ids))
            saved = (m.geom_contype[ids].copy(), m.geom_conaffinity[ids].copy())
            m.geom_contype[ids] = 0
            m.geom_conaffinity[ids] = 0
        bids = [self.bid(b) for b in bodies]
        n = int(round(duration / m.opt.timestep))
        out = np.zeros((n + 1, len(bids), 3))
        ts = np.zeros(n + 1)
        out[0] = self._pre.xpos[bids]
        ts[0] = self._pre.time
        k = n
        try:
            for i in range(1, n + 1):
                self.pre_substep(self._pre)
                mujoco.mj_step(m, self._pre)
                out[i] = self._pre.xpos[bids]
                ts[i] = self._pre.time
                if until is not None and until(self._pre):
                    k = i
                    break
        finally:
            if saved is not None:
                m.geom_contype[ids], m.geom_conaffinity[ids] = saved
        return {b: out[:k + 1, j] for j, b in enumerate(bodies)}, ts[:k + 1], self._pre
