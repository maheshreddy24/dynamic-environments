"""Ball-catching scenarios with a net mounted on the gripper.

Complex (paper Fig. 5, Table 3):  BallisticCatch, MultiBalls, Occlusion, MidFlight
Stress tests (Fig. 6, Tables 9-11): MultipleDeflection, OcclusionDeflection, PlinkoDrop
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from ..base_env import TABLE_Z, BaseConfig, CameraSpec, DynamicEnv
from ..components import (BALL_COLORS, add_ball, add_deflector, add_launcher, add_occluder, add_plinko,
                          launcher_muzzle, set_occluder)
from ..experts import CatchExpert
from ..geometry import normalize, quat_axis_angle, quat_from_z
from ..success import ContactWindowTracker, NetCatchTracker

G = np.array([0.0, 0.0, -9.81])
BALL_R = 0.025
CATCH_BOX = ((0.36, -0.32, TABLE_Z + 0.18), (0.70, 0.32, TABLE_Z + 0.52))   # reachable net positions


def _near_robot(env, P, clearance):
    """Minimum distance from points P to the robot's body origins, TCP and net centre."""
    if len(P) == 0:
        return False
    pts = [env.data.xpos[b] for b in sorted(env.arm.body_ids)]
    pts += [env.data.site_xpos[env.arm.tcp_site], env.data.site_xpos[env.arm.site]]
    return np.linalg.norm(P[:, None, :] - np.array(pts)[None], axis=-1).min() < clearance


def ballistic(p0, v, t):
    t = np.asarray(t, float)[..., None]
    return p0 + v * t + 0.5 * G * t * t


class Launch:
    """A ball held at a launcher's muzzle until `t`, then fired with velocity `v`."""

    def __init__(self, ball, launcher, t, muzzle, v, yaw, pitch):
        self.ball, self.launcher, self.t, self.muzzle, self.v = ball, launcher, t, muzzle, v
        self.yaw, self.pitch = yaw, pitch


def plan_launch(base_xy, target, T, muzzle_height=0.29, barrel_len=0.12):
    """Launcher yaw/pitch, muzzle position and launch velocity so the ball passes `target`
    after `T` seconds."""
    base = np.array([base_xy[0], base_xy[1], 0.0])
    m = base + [0, 0, muzzle_height]
    for _ in range(4):
        v = (np.asarray(target) - m - 0.5 * G * T * T) / T
        yaw = np.arctan2(v[1], v[0])
        pitch = np.arctan2(v[2], np.hypot(v[0], v[1]))
        m, _ = launcher_muzzle(base, yaw, pitch, muzzle_height, barrel_len)
    v = (np.asarray(target) - m - 0.5 * G * T * T) / T
    return m, v, yaw, pitch


# ============================================================================= occluder planning
class Board:
    """A vertical rectangle used as a visual occluder, planned in the camera's frame."""

    def __init__(self, cam, pts, lam, pad_bottom=0.3, pad_top=2.5, h_margin=0.0):
        C = np.asarray(cam, float)
        M = pts.mean(0)
        d = M - C
        self.n = normalize([d[0], d[1], 0.0])
        self.Q = C + lam * d
        self.h = np.array([-self.n[1], self.n[0], 0.0])
        self.C = C
        X = self._project(pts)
        hc = (X - self.Q) @ self.h
        self.hmin, self.hmax = hc.min() - h_margin, hc.max() + h_margin
        self.z0 = max(0.1, X[:, 2].min() - pad_bottom)
        self.z1 = X[:, 2].max() + pad_top

    def _project(self, P):
        P = np.atleast_2d(P)
        s = (self.n @ (self.Q - self.C)) / ((P - self.C) @ self.n)
        return self.C + s[:, None] * (P - self.C)

    def hides(self, P):
        P = np.atleast_2d(P)
        depth = (P - self.C) @ self.n
        behind = depth > self.n @ (self.Q - self.C) + 1e-3
        X = self._project(P)
        hc = (X - self.Q) @ self.h
        return behind & (hc >= self.hmin) & (hc <= self.hmax) & (X[:, 2] >= self.z0) & (X[:, 2] <= self.z1)

    @property
    def width(self):
        return self.hmax - self.hmin

    def place(self, env, name="occluder"):
        center = self.Q + self.h * (self.hmin + self.hmax) / 2
        env.set_mocap(name, [center[0], center[1], 0.0], quat_axis_angle([0, 0, 1], np.arctan2(self.n[1], self.n[0])))
        set_occluder(env.model, name, self.width, self.z0, self.z1)


def intervals(mask, ts):
    """Contiguous True runs of `mask` as [(t_start, t_end), ...]."""
    mask = np.asarray(mask, bool)
    edges = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
    return [(ts[a], ts[b]) for a, b in zip(starts, ends)]


# ============================================================================= base
@dataclass
class CatchConfig(BaseConfig):
    episode_seconds: float = 4.0
    flight_time_range: tuple = (0.75, 1.1)           # launch -> catch point [s]
    launch_time_range: tuple = (0.25, 0.6)           # ball sits in the launcher until then
    launcher_x_range: tuple = (1.6, 2.2)
    launcher_y_range: tuple = (-0.8, 0.8)
    target_box: tuple = ((0.40, -0.26, TABLE_Z + 0.22), (0.66, 0.26, TABLE_Z + 0.48))


class CatchTask(DynamicEnv):
    Config = CatchConfig
    family = "complex"
    attachment = "net"
    control_site = "net_center"
    gripper_fixed = 0.0
    home_pos = (0.32, 0.0, TABLE_Z + 0.42)
    workspace = ((0.2, -0.5, TABLE_Z + 0.1), (0.8, 0.5, TABLE_Z + 0.75))
    camera = CameraSpec(lookat=(0.95, 0.0, TABLE_Z + 0.1), distance=2.9, azimuth=215, elevation=-22)
    n_balls = 1
    ball_colors = ("yellow",)
    max_tries = 60

    def build(self, b):
        for i in range(self.n_balls):
            add_ball(b, f"ball{i}", BALL_R, rgba=BALL_COLORS[self.ball_colors[i]], mass=0.03)
        for i in range(self.n_launchers()):
            add_launcher(b, f"launcher{i}")
        self.build_extra(b)

    def n_launchers(self):
        return self.n_balls

    def build_extra(self, b):
        pass

    def setup(self):
        ground = {self.gid("floor"), self.gid("floor_infinite"), self.gid("table_top")}
        for i in range(self.n_launchers()):
            ground |= self.body_geoms(f"launcher{i}")
        self.ground = ground
        self.catchers = [NetCatchTracker(self, f"ball{i}", BALL_R, ground) for i in range(self.n_balls)]
        self.target = 0
        self.launches = []

    def trackers(self):
        return self.catchers

    # ---------------------------------------------------------------- launching
    def sample_launch(self, rng, i, target=None, T=None, launcher_xy=None, t_launch=None, descending=True):
        c = self.cfg
        lo, hi = map(np.asarray, c.target_box)
        for _ in range(100):
            P = rng.uniform(lo, hi) if target is None else np.asarray(target)
            TT = rng.uniform(*c.flight_time_range) if T is None else T
            L = (np.array([rng.uniform(*c.launcher_x_range), rng.uniform(*c.launcher_y_range)])
                 if launcher_xy is None else np.asarray(launcher_xy))
            m, v, yaw, pitch = plan_launch(L, P, TT)
            if self._launch_ok(m, v, TT, descending):
                tl = rng.uniform(*c.launch_time_range) if t_launch is None else t_launch
                self.place_launcher(f"launcher{i}", L, yaw, pitch)
                return Launch(f"ball{i}", f"launcher{i}", tl, m, v, yaw, pitch)
        return None

    def _launch_ok(self, m, v, T, descending=True):
        ts = np.linspace(0, T, 60)
        P = ballistic(m, v, ts)
        if descending and v[2] + G[2] * T > -0.8:   # must be descending through the catch point
            return False
        if P[:, 2].max() > 3.0 or np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))) > 80:
            return False
        tbl = self.table
        x0, x1 = tbl["center"][0] - tbl["size"][0] / 2, tbl["center"][0] + tbl["size"][0] / 2
        over = (P[:, 0] > x0 - 0.05) & (P[:, 0] < x1 + 0.05) & (np.abs(P[:, 1]) < tbl["size"][1] / 2 + 0.05)
        return not np.any(over & (P[:, 2] < TABLE_Z + 0.12))

    def place_launcher(self, name, xy, yaw, pitch):
        self.set_mocap(name, [xy[0], xy[1], 0.0], quat_axis_angle([0, 0, 1], yaw))
        gid = self.gid(f"{name}_barrel")
        d = np.array([np.cos(pitch), 0.0, np.sin(pitch)])  # launcher-local barrel direction
        self.model.geom_quat[gid] = quat_from_z(d)
        self.model.geom_pos[gid] = np.array([0, 0, 0.29]) + d * 0.06

    def hide_unused(self, used):
        for i in range(self.n_launchers()):
            if i not in used:
                self.set_mocap(f"launcher{i}", [0, 8 + i, -2])

    def pre_substep(self, data):
        dt = self.model.opt.timestep
        for L in self.launches:
            if data.time < L.t:
                qa, da = self.free_qadr(L.ball)
                data.qpos[qa:qa + 3] = L.muzzle
                data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
                data.qvel[da:da + 6] = 0
            elif data.time < L.t + dt * 0.999:
                qa, da = self.free_qadr(L.ball)
                data.qpos[qa:qa + 3] = L.muzzle
                data.qvel[da:da + 3] = L.v
                data.qvel[da + 3:da + 6] = 0

    def post_substep(self):
        dt = self.model.opt.timestep
        for c in self.catchers:
            c.substep(dt)

    def evaluate(self):
        tgt = self.catchers[self.target]
        for i, c in enumerate(self.catchers):
            if i != self.target and c.result:
                self.events["wrong_ball_caught"] = float(self.data.time)
                return False, True
        if tgt.result is not None:
            return tgt.result, not tgt.result
        return False, False

    def launched(self):
        return any(self.data.time >= L.t for L in self.launches if L.ball == f"ball{self.target}")

    def make_expert(self, **kw):
        return CatchExpert(self, f"ball{self.target}", CATCH_BOX, launched_fn=self.launched, **kw)

    def task_info(self):
        return dict(target_ball=f"ball{self.target}", launch_times=[L.t for L in self.launches],
                    caught=[c.result for c in self.catchers])

    def preroll_balls(self, duration):
        traj, ts, pre = self.preroll(duration, [f"ball{i}" for i in range(self.n_balls)])
        return traj, ts


# ============================================================================= BC
class BallisticCatch(CatchTask):
    """A ball is launched on a parabolic trajectory toward the robot's workspace; the robot
    must catch it (with the gripper-mounted net) before it reaches the floor."""

    def reset_task(self, rng):
        L = None
        while L is None:
            L = self.sample_launch(rng, 0)
        self.launches = [L]
        self.instruction = "catch the ball"


# ============================================================================= MB
@dataclass
class MultiBallsConfig(CatchConfig):
    launch_stagger: float = 0.25
    min_target_sep: float = 0.16


class MultiBalls(CatchTask):
    """Three balls (green, red, yellow) are launched into the workspace from three launchers;
    the robot must intercept the language-specified ball."""

    Config = MultiBallsConfig
    n_balls = 3
    ball_colors = ("green", "red", "yellow")

    def reset_task(self, rng):
        c = self.cfg
        lo, hi = map(np.asarray, c.target_box)
        for _ in range(self.max_tries):
            targets = rng.uniform(lo, hi, size=(3, 3))
            d = np.linalg.norm(targets[:, None] - targets[None], axis=-1) + np.eye(3)
            if d.min() < c.min_target_sep:
                continue
            targets = targets[np.argsort(targets[:, 1])]
            xc = rng.uniform(*c.launcher_x_range)
            ys = np.sort(rng.uniform(-0.15, 0.15) + np.array([-0.55, 0.0, 0.55]) + rng.uniform(-0.05, 0.05, 3))
            order = rng.permutation(3)          # which colour comes from which launcher
            t0 = rng.uniform(*c.launch_time_range)
            self.launches = [self.sample_launch(rng, i, target=targets[k],
                                                launcher_xy=(xc + rng.uniform(-0.1, 0.1), ys[k]),
                                                t_launch=t0 + rng.uniform(0, c.launch_stagger))
                             for k, i in enumerate(order)]
            if all(L is not None for L in self.launches):
                break
        else:
            raise RuntimeError("MultiBalls: could not sample valid launches")
        self.target = int(rng.integers(3))
        self.instruction = f"catch the {self.ball_colors[self.target]} ball"


# ============================================================================= Occ
@dataclass
class OcclusionConfig(CatchConfig):
    flight_time_range: tuple = (1.0, 1.3)
    occlusion_range: tuple = (0.30, 0.60)       # paper: 300-600 ms behind the occluder
    visible_after: float = 0.25                 # ball visible this long before the catch point
    board_lambda: tuple = (0.45, 0.65)          # board position along the camera ray


class Occlusion(CatchTask):
    """A launched ball passes behind a vertical occluder for a randomised 300-600 ms interval
    before reappearing; the robot must catch it after it emerges.  The occluder is placed
    between the (randomised) policy camera and the flight path."""

    Config = OcclusionConfig
    camera = CameraSpec(lookat=(1.0, 0.0, TABLE_Z + 0.15), distance=2.9, azimuth=262, elevation=-14)

    def build_extra(self, b):
        add_occluder(b, "occluder")

    def reset_task(self, rng):
        c = self.cfg
        cam, _ = self.camera_pose()
        for _ in range(self.max_tries):
            L = self.sample_launch(rng, 0)
            if L is None:
                continue
            T = self._flight_time(L)
            delta = rng.uniform(*c.occlusion_range)
            t1_lo, t1_hi = 0.12, T - c.visible_after - delta
            if t1_hi <= t1_lo:
                continue
            t1 = rng.uniform(t1_lo, t1_hi)
            ts = np.arange(0, T + 0.4, 0.004)
            P = ballistic(L.muzzle, L.v, ts)
            seg = P[(ts >= t1) & (ts <= t1 + delta)]
            board = Board(cam, seg, rng.uniform(*c.board_lambda))
            hidden = board.hides(P)
            iv = intervals(hidden, ts)
            if len(iv) != 1 or abs((iv[0][1] - iv[0][0]) - delta) > 0.03:
                continue
            if board.hides(np.array([self.home_pos, P[-1]])).any() or board.hides(self._robot_points()).any():
                continue
            self.launches = [L]
            board.place(self)
            self.occluded = (L.t + iv[0][0], L.t + iv[0][1])
            self.instruction = "catch the ball"
            return
        raise RuntimeError("Occlusion: no valid occluder placement")

    def _flight_time(self, L):
        lo, hi = map(np.asarray, self.cfg.target_box)
        ts = np.arange(0, 2.5, 0.002)
        P = ballistic(L.muzzle, L.v, ts)
        inside = np.all((P >= lo) & (P <= hi), axis=1) & (L.v[2] + G[2] * ts < 0)
        return ts[np.argmax(inside)] if inside.any() else 1.0

    def _robot_points(self):
        return np.array([self.data.xpos[b] for b in sorted(self.arm.body_ids)])

    def evaluate(self):
        s, f = super().evaluate()
        if s and self.catchers[0].time < self.occluded[1]:
            return False, True   # must be caught after re-emerging
        return s, f

    def task_info(self):
        return dict(super().task_info(), occluded_interval=self.occluded)


# ============================================================================= Mid
@dataclass
class DeflectionConfig(CatchConfig):
    first_hit_box: tuple = ((0.95, -0.30, TABLE_Z + 0.45), (1.35, 0.30, TABLE_Z + 0.95))  # 1st contact point
    to_first_hit: tuple = (0.45, 0.80)         # launch -> first contact [s]
    between_hits: tuple = (0.15, 0.30)         # first -> second contact (multiple deflection) [s]
    min_hit_strength: float = 0.3              # |cos| between incoming velocity and contact normal
    min_time_after: float = 0.30               # time from (last) deflection to catch point
    max_time_after: float = 0.90
    restitution: float = 0.68                  # measured for the BOUNCY contact parameters
    preroll_seconds: float = 3.0


class DeflectionMixin:
    """Designed deflections: pick a mid-air contact point in front of the robot, launch the ball
    to reach it, then choose the hanging deflector's contact normal (reflection law with the
    measured restitution) so the outgoing parabola reaches the next contact point or the catch
    zone.  Every draw is verified with a physics pre-roll before it is accepted."""

    n_deflectors = 1
    first_hit_box = None   # override per task (defaults to the config value)

    def build_extra(self, b):
        for k in range(self.n_deflectors):
            add_deflector(b, f"deflector{k}")
        super().build_extra(b)

    def _defl_geoms(self):
        return [self.gid(f"deflector{k}_geom") for k in range(self.n_deflectors)]

    def _choose_normal(self, rng, p, v, final, n_samples=400):
        """Random contact normal (deflector -> ball) giving a useful outgoing velocity."""
        c = self.cfg
        e = c.restitution
        u = normalize(v)
        cands = rng.normal(size=(n_samples, 3))
        cands /= np.linalg.norm(cands, axis=1, keepdims=True)
        cands = cands[cands @ u < -c.min_hit_strength]
        lo, hi = map(np.asarray, CATCH_BOX)
        ts = np.arange(0.02, 1.2, 0.01)
        for n in cands:
            vo = v - (1 + e) * (v @ n) * n
            P = ballistic(p, vo, ts)
            if final:
                vz = vo[2] + G[2] * ts
                ok = np.all((P >= lo + 0.03) & (P <= hi - 0.03), axis=1) & (vz < -0.3)
                ok &= (ts >= c.min_time_after) & (ts <= c.max_time_after)
                if not ok.any():
                    continue
                first = np.argmax(ok)
                pre = P[:first]
                if np.any(pre[:, 2] < TABLE_Z + 0.12):
                    continue
                return n, vo
            else:
                t2 = rng.uniform(*c.between_hits)
                q = ballistic(p, vo, t2)
                if (0.85 <= q[0] <= 1.6 and abs(q[1]) <= 0.45 and TABLE_Z + 0.4 <= q[2] <= TABLE_Z + 1.3
                        and np.linalg.norm(q - p) > 0.25):
                    return n, vo, t2
        return None

    def sample_deflected(self, rng, cam=None):
        c = self.cfg
        R = BALL_R + 0.05
        lo, hi = map(np.asarray, self.first_hit_box or c.first_hit_box)
        for _ in range(self.max_tries * 3):
            D = rng.uniform(lo, hi)
            L = self.sample_launch(rng, 0, target=D, T=rng.uniform(*c.to_first_hit), descending=False)
            if L is None:
                continue
            self.launches = [L]
            for k in range(self.n_deflectors):
                self.set_mocap(f"deflector{k}", [0, 6 + k, 3])
            # Incoming state at D on the planned parabola (the launch was planned through D).
            v = L.v + G * self._time_to(L, D)
            p = D
            ok = True
            for k in range(self.n_deflectors):
                final = k == self.n_deflectors - 1
                res = self._choose_normal(rng, p, v, final)
                if res is None:
                    ok = False
                    break
                self.set_mocap(f"deflector{k}", p - res[0] * R)
                if not final:
                    # Place the next contact on the *simulated* post-bounce path.
                    P, ts, hits, other = self._contacts_in_preroll(c.preroll_seconds)
                    if hits[k] is None:
                        ok = False
                        break
                    j = np.searchsorted(ts, hits[k] + res[2])
                    if j + 2 >= len(ts) or (other is not None and other < ts[j]):
                        ok = False
                        break
                    p = P[j]
                    v = (P[j + 1] - P[j - 1]) / (ts[j + 1] - ts[j - 1])
            if not ok:
                continue
            P, ts, hits, other = self._contacts_in_preroll(c.preroll_seconds)
            if any(h is None for h in hits) or sorted(hits) != hits:
                continue
            t_last = max(hits)
            if other is not None and other < t_last + 0.05:
                continue
            tc = self._catchable(P, ts, t_last + c.min_time_after - 0.05)
            if tc is None:
                continue
            t_exit = self._exit_time(P, ts, t_last)
            if self._near_robot_home(P[ts < t_exit + 0.05]):   # robot must not touch it 'for free'
                continue
            if self.accept_extra(rng, P, ts, hits, tc, cam) is False:
                continue
            self.deflect_times = hits
            self.catch_time = tc
            self.exit_time = t_exit
            self.catchers[0].reset(start_time=t_last + 0.02)
            self.oracle = (P, ts, t_last + 0.02)
            return P, ts
        raise RuntimeError(f"{type(self).__name__}: could not sample a valid deflected launch")

    @staticmethod
    def _time_to(L, D):
        """Flight time from the muzzle to point D on the launch parabola (horizontal motion)."""
        dh = np.hypot(*(np.asarray(D)[:2] - L.muzzle[:2]))
        return dh / np.hypot(L.v[0], L.v[1])

    def accept_extra(self, rng, P, ts, hits, tc, cam):
        return True

    def make_expert(self, **kw):
        kw.setdefault("oracle", self.oracle)
        return CatchExpert(self, "ball0", CATCH_BOX, launched_fn=self.launched, **kw)

    def _contacts_in_preroll(self, duration):
        """Pre-roll and return ball trajectory, times and first contact time with each deflector."""
        m = self.model
        mujoco.mj_copyData(self._pre, m, self.data)
        ids = np.array(sorted(self.arm.geom_ids))
        saved = (m.geom_contype[ids].copy(), m.geom_conaffinity[ids].copy())
        m.geom_contype[ids] = 0
        m.geom_conaffinity[ids] = 0
        ball = self.bid("ball0")
        bgeoms = self.body_geoms("ball0")
        dg = self._defl_geoms()
        hits = [None] * len(dg)
        other = None
        t_fire = max(L.t for L in self.launches)
        n = int(round(duration / m.opt.timestep))
        P = np.zeros((n, 3))
        ts = np.zeros(n)
        try:
            for i in range(n):
                self.pre_substep(self._pre)
                mujoco.mj_step(m, self._pre)
                P[i] = self._pre.xpos[ball]
                ts[i] = self._pre.time
                for g1, g2 in self.contacts(self._pre):
                    g = g2 if g1 in bgeoms else (g1 if g2 in bgeoms else None)
                    if g is None:
                        continue
                    if g in dg:
                        k = dg.index(g)
                        if hits[k] is None:
                            hits[k] = self._pre.time
                    elif other is None:
                        other = self._pre.time
                if self._pre.time > t_fire + 0.1 and P[i, 2] < 0.08:   # landed on the floor
                    P, ts = P[:i + 1], ts[:i + 1]
                    break
        finally:
            m.geom_contype[ids], m.geom_conaffinity[ids] = saved
        return P, ts, hits, other

    def _catchable(self, P, ts, t_after):
        """First time >= t_after when the ball is inside the catch box while descending."""
        lo, hi = map(np.asarray, CATCH_BOX)
        vz = np.gradient(P[:, 2], ts)
        ok = np.all((P >= lo) & (P <= hi), axis=1) & (vz < 0) & (ts >= t_after)
        return ts[np.argmax(ok)] if ok.any() else None

    def _exit_time(self, P, ts, t_after):
        """Time the ball leaves the reachable catch region for good (falls below it)."""
        lo, _ = map(np.asarray, CATCH_BOX)
        below = (P[:, 2] < lo[2]) & (ts > t_after)
        return ts[np.argmax(below)] if below.any() else ts[-1]

    def _near_robot_home(self, P, clearance=0.2):
        """True if the (pre-rolled) ball passes close to the robot in its home pose -- such
        episodes would let a frozen robot touch the ball 'for free'."""
        return _near_robot(self, P, clearance)


class MidFlight(DeflectionMixin, CatchTask):
    """A launched ball on a parabolic trajectory deflects mid-flight off an unexpected contact
    (a hanging sphere), abruptly changing direction; the robot must adjust its intercept.
    Catches completed before the deflection are excluded (reported via info["excluded"])."""

    Config = DeflectionConfig

    def reset_task(self, rng):
        self.sample_deflected(rng)
        self.instruction = "catch the ball"

    def evaluate(self):
        s, f = super().evaluate()
        tgt = self.catchers[0]
        if tgt.result and tgt.time < self.deflect_times[0]:
            self.events["excluded"] = True
            return False, True
        return s, f

    def task_info(self):
        return dict(super().task_info(), deflect_time=self.deflect_times[0], excluded=self.events.get("excluded", False))


# ============================================================================= stress tests
@dataclass
class StressConfig(DeflectionConfig):
    window_ms: float = 800.0       # time-to-exit window: robot may act only in [t_exit - window, t_exit]
    episode_seconds: float = 4.5


class StressMixin:
    """Robot is frozen until `exit_time - window`; success = robot touches the ball within the
    window (App. C.4 stress-test criterion)."""

    family = "stress"

    def setup(self):
        super().setup()
        self.window_tracker = ContactWindowTracker(self, "ball0")

    def trackers(self):
        return super().trackers() + [self.window_tracker]

    def start_window(self):
        w = self.cfg.window_ms / 1000.0
        self.t_release = self.exit_time - w
        self.window_tracker.reset((self.t_release, self.exit_time))

    def robot_frozen(self):
        return self.data.time < self.t_release

    def post_substep(self):
        super().post_substep()
        self.window_tracker.substep()

    def evaluate(self):
        r = self.window_tracker.result
        if r is not None:
            return r, not r
        return False, False

    def task_info(self):
        return dict(window_ms=self.cfg.window_ms, release_time=self.t_release, exit_time=self.exit_time,
                    first_contact=self.window_tracker.first_contact)


class MultipleDeflection(StressMixin, DeflectionMixin, CatchTask):
    """A ball bounces off two intermediate (hanging) surfaces before reaching the workspace."""

    Config = StressConfig
    n_deflectors = 2
    first_hit_box = ((1.25, -0.30, TABLE_Z + 0.6), (1.6, 0.30, TABLE_Z + 1.1))
    camera = CameraSpec(lookat=(1.0, 0.0, TABLE_Z + 0.3), distance=3.1, azimuth=215, elevation=-18)

    def reset_task(self, rng):
        self.sample_deflected(rng)
        self.start_window()
        self.instruction = "catch the ball"


@dataclass
class OcclusionDeflectionConfig(StressConfig):
    occlusion_range: tuple = (0.30, 0.60)
    board_lambda: tuple = (0.45, 0.65)


class OcclusionDeflection(StressMixin, DeflectionMixin, CatchTask):
    """Occlusion combined with a single deflection that happens during the occluded interval."""

    Config = OcclusionDeflectionConfig
    camera = Occlusion.camera
    ball_colors = ("red",)

    def build_extra(self, b):
        add_occluder(b, "occluder")
        super().build_extra(b)

    def reset_task(self, rng):
        cam, _ = self.camera_pose()
        P, ts = self.sample_deflected(rng, cam=cam)
        self.board.place(self)
        self.start_window()
        self.instruction = "catch the ball"

    def accept_extra(self, rng, P, ts, hits, tc, cam):
        c = self.cfg
        td = hits[0]
        delta = rng.uniform(*c.occlusion_range)
        t1 = td - rng.uniform(0.1, delta - 0.12)
        t2 = t1 + delta
        if t2 > tc - 0.15:
            return False
        seg = P[(ts >= t1) & (ts <= t2)]
        dpos = self.data.mocap_pos[self.model.body_mocapid[self.bid("deflector0")]]
        r = 0.05
        board = Board(cam, np.vstack([seg, dpos + [0, 0, r], dpos - [0, 0, r]]), rng.uniform(*c.board_lambda),
                      h_margin=0.06, pad_top=3.0)
        hidden = board.hides(P)
        iv = intervals(hidden, ts)
        if len(iv) != 1 or not (c.occlusion_range[0] - 0.05 <= iv[0][1] - iv[0][0] <= c.occlusion_range[1] + 0.15):
            return False
        if not board.hides(dpos[None])[0] or board.hides(np.array([self.home_pos])).any():
            return False
        if board.hides(np.array([self.data.xpos[b] for b in sorted(self.arm.body_ids)])).any():
            return False
        self.board = board
        self.occluded = iv[0]
        return True

    def task_info(self):
        return dict(super().task_info(), occluded_interval=self.occluded)


# ============================================================================= plinko
@dataclass
class PlinkoConfig(BaseConfig):
    window_ms: float = 800.0
    episode_seconds: float = 4.0
    drop_time_range: tuple = (0.2, 0.4)
    drop_offset: float = 0.05              # lateral release jitter at the dispenser
    drop_speed: float = 0.08               # lateral release speed jitter
    board_center_y: tuple = (-0.06, 0.06)
    board_bottom: float = TABLE_Z + 0.66


class PlinkoDrop(StressMixin, CatchTask):
    """A ball falls through a peg array, each peg contact randomising its trajectory; the robot
    must catch the ball at the bottom (contact within the time-to-exit window)."""

    Config = PlinkoConfig
    n_balls = 1
    ball_colors = ("orange",)
    board_x = 0.58
    camera = CameraSpec(lookat=(0.5, 0.0, TABLE_Z + 0.55), distance=2.6, azimuth=195, elevation=-12)

    def n_launchers(self):
        return 0

    def build_extra(self, b):
        self.pl = add_plinko(b, "plinko")

    def setup(self):
        super().setup()
        self.pl_body = self.bid("plinko")

    def reset_task(self, rng):
        c = self.cfg
        yc = rng.uniform(*c.board_center_y)
        self.model.body_pos[self.pl_body] = [self.board_x, yc, c.board_bottom]
        top = np.array([self.board_x, yc, c.board_bottom + self.pl["height"] + 0.10])
        for _ in range(self.max_tries):
            t = rng.uniform(*c.drop_time_range)
            p = top + [0, rng.uniform(-c.drop_offset, c.drop_offset), 0]
            v = np.array([0, rng.uniform(-c.drop_speed, c.drop_speed), 0])
            self.launches = [Launch("ball0", None, t, p, v, 0, 0)]
            mujoco.mj_forward(self.model, self.data)
            traj, ts, _ = self.preroll(3.0, ["ball0"], until=lambda d: d.xpos[self.bid("ball0")][2] < TABLE_Z)
            P = traj["ball0"]
            lo, hi = map(np.asarray, CATCH_BOX)
            below = (P[:, 2] < lo[2]) & (ts > t)
            if not below.any():
                continue
            self.exit_time = ts[np.argmax(below)]
            self.oracle = (P, ts, t + 0.05)
            ex = P[np.argmax(below)]
            if not (lo[1] <= ex[1] <= hi[1]):
                continue
            if self._near_robot(P[ts < self.exit_time + 0.05]):
                continue
            break
        else:
            raise RuntimeError("PlinkoDrop: could not sample a valid drop")
        self.start_window()
        self.instruction = "catch the ball"

    def _near_robot(self, P):
        return _near_robot(self, P, 0.2)

    def make_expert(self, **kw):
        kw.setdefault("oracle", self.oracle)
        return CatchExpert(self, "ball0", CATCH_BOX, launched_fn=self.launched, **kw)
