"""Scripted privileged-state planners (paper, App. C.3: "Simulation demonstrations are
scripted rollouts from a privileged-state planner").  They read ground-truth object state
from MuJoCo, predict ahead, and emit absolute end-effector actions
`[x, y, z, qw, qx, qy, qz, gripper]` (gripper: -1 open, +1 closed), so the env must be
created with `action_mode="ee_abs"`.

They are deliberately simple (no learning, constant-acceleration / ballistic prediction) and
exist to (a) validate that every scenario is solvable and its success check fires, and
(b) generate demonstrations.
"""
import numpy as np

from .geometry import top_down_quat

G = np.array([0.0, 0.0, -9.81])


class Expert:
    max_speed = 1.2  # m/s, Cartesian speed of the commanded target

    def __init__(self, env):
        assert env.action_mode == "ee_abs", "experts emit absolute EE actions: use action_mode='ee_abs'"
        self.env = env
        self.reset()

    def reset(self):
        self.cmd_pos, _ = self.env.arm.site_pose()
        self.cmd_quat = top_down_quat(self.env.home_yaw)
        self.grip = -1.0
        self.phase = "start"
        self.t_phase = self.env.data.time

    def set_phase(self, name):
        self.phase = name
        self.t_phase = self.env.data.time

    @property
    def phase_time(self):
        return self.env.data.time - self.t_phase

    def move_toward(self, goal, speed=None):
        speed = self.max_speed if speed is None else speed
        d = np.asarray(goal, float) - self.cmd_pos
        n = np.linalg.norm(d)
        step = speed * self.env.dt
        self.cmd_pos = goal.copy() if n <= step else self.cmd_pos + d / n * step

    def action(self):
        return np.concatenate([self.cmd_pos, self.cmd_quat, [self.grip]])

    def __call__(self, obs=None):
        self.plan()
        return self.action()

    def plan(self):
        raise NotImplementedError


class MotionEstimator:
    """Velocity / acceleration of a body from finite differences of privileged positions."""

    def __init__(self, env, body, alpha=0.3):
        self.env, self.body, self.alpha = env, body, alpha
        self.p = None
        self.v = np.zeros(3)
        self.a = np.zeros(3)

    def update(self):
        p = self.env.body_pos(self.body)
        v = self.env.body_linvel(self.body)
        if self.p is not None:
            a = (v - self.v) / self.env.dt
            self.a = (1 - self.alpha) * self.a + self.alpha * a
        self.p, self.v = p, v
        return p, v, self.a

    def predict(self, tau, use_acc=True):
        a = self.a if use_acc else 0.0
        return self.p + self.v * tau + 0.5 * a * tau * tau


# ============================================================================= pick
class PickExpert(Expert):
    """Intercept a moving object, grasp it from above, lift and hold.  Optionally carry it to
    a (possibly moving) box and release it (`place_fn` returns the box centre & velocity)."""

    def __init__(self, env, body, grasp_z, yaw=0.0, lead=0.12, hover=0.10, place_fn=None, place_height=0.14,
                 use_acc=True, lift_height=0.20, offset=(0.0, 0.0), carry_speed=0.8, goal=None,
                 goal_speed=0.4):
        self.body, self.grasp_z, self.yaw, self.lead, self.hover = body, grasp_z, yaw, lead, hover
        self.offset = np.array([offset[0], offset[1], 0.0])   # world-frame grasp offset from the body origin
        self.carry_speed = carry_speed
        self.goal = None if goal is None else np.asarray(goal, float)   # object goal position (body origin)
        self.goal_speed = goal_speed
        self.place_fn, self.place_height, self.use_acc, self.lift_height = place_fn, place_height, use_acc, lift_height
        super().__init__(env)

    def reset(self):
        super().reset()
        self.est = MotionEstimator(self.env, self.body)
        self.cmd_quat = top_down_quat(self.yaw)

    def plan(self):
        e = self.env
        p, v, a = self.est.update()
        p = p + self.offset
        ee, _ = e.arm.site_pose()
        pred = lambda tau: self.est.predict(tau, self.use_acc) + self.offset
        if self.phase == "start":
            self.set_phase("approach")
        if self.phase == "approach":
            tau = self.lead
            for _ in range(3):
                goal = pred(tau) + [0, 0, self.grasp_z + self.hover]
                tau = self.lead + np.linalg.norm(goal - ee) / (0.8 * self.max_speed)
            goal = pred(tau) + [0, 0, self.grasp_z + self.hover]
            self.move_toward(goal)
            near = p + [0, 0, self.grasp_z + self.hover]   # the servo lag ~ cancels the lead
            if np.linalg.norm((ee - near)[:2]) < 0.025 and abs(ee[2] - near[2]) < 0.04:
                self.set_phase("descend")
        elif self.phase == "descend":
            goal = pred(self.lead) + [0, 0, self.grasp_z]
            z = max(goal[2], self.cmd_pos[2] - 0.45 * e.dt)
            self.cmd_pos = np.array([goal[0], goal[1], z])
            if abs(ee[2] - goal[2]) < 0.012 and np.linalg.norm((ee - p)[:2]) < 0.012:
                self.set_phase("close")
        elif self.phase == "close":
            goal = pred(self.lead) + [0, 0, self.grasp_z]
            self.cmd_pos = goal
            self.grip = 1.0
            if self.phase_time > 0.4:
                # With a goal: lift just enough to clear the conveyor rails, then carry.
                self.lift_target = goal[2] + (0.06 if self.goal is not None else self.lift_height)
                self.set_phase("lift")
        elif self.phase == "lift":
            self.grip = 1.0
            goal = np.array([self.cmd_pos[0], self.cmd_pos[1], self.lift_target])
            self.move_toward(goal, speed=0.35)
            if self.cmd_pos[2] >= self.lift_target - 1e-3:
                self.set_phase("to_goal" if self.goal is not None else ("carry" if self.place_fn else "hold"))
        elif self.phase == "to_goal":
            self.grip = 1.0
            obj = e.body_pos(self.body)
            tcp_goal = self.goal - (obj - ee)          # live object-in-hand offset (absorbs slip)
            target = tcp_goal.copy()
            if np.linalg.norm((tcp_goal - self.cmd_pos)[:2]) > 0.02:
                target[2] = max(tcp_goal[2], self.cmd_pos[2])   # move over at height, then descend
            self.move_toward(target, speed=self.goal_speed)
            if np.linalg.norm(self.cmd_pos - tcp_goal) < 1e-3:
                self.set_phase("hold_goal")
        elif self.phase == "hold_goal":
            self.grip = 1.0
            obj = e.body_pos(self.body)
            self.move_toward(self.goal - (obj - ee), speed=0.1)   # small corrections while holding
        elif self.phase == "carry":
            self.grip = 1.0
            c, vb = self.place_fn()
            tau = 0.15 + np.linalg.norm((c - ee)[:2]) / (0.6 * self.max_speed)
            goal = c + vb * tau + [0, 0, self.place_height + self.grasp_z]
            goal[2] = max(goal[2], self.cmd_pos[2] - 0.3 * e.dt)
            self.move_toward(goal, speed=self.carry_speed)
            if np.linalg.norm((ee - (c + vb * self.lead))[:2]) < 0.02:
                self.set_phase("release")
        elif self.phase == "release":
            c, vb = self.place_fn()
            goal = c + vb * self.lead + [0, 0, self.place_height + self.grasp_z]
            self.cmd_pos = np.array([goal[0], goal[1], self.cmd_pos[2]])
            self.grip = -1.0
            if self.phase_time > 0.4:
                self.set_phase("retreat")
        elif self.phase == "retreat":
            self.grip = -1.0
            self.move_toward(self.cmd_pos + [0, 0, 0.1], speed=0.3)


# ============================================================================= catch
class CatchExpert(Expert):
    """Ballistic-prediction net catcher.  Re-plans every step from the true ball state, so it
    reacts to deflections after they happen (it does not anticipate them)."""

    def __init__(self, env, ball_body, catch_box, lead=0.10, below=0.015, launched_fn=None, oracle=None):
        """`oracle` = (P, ts, t_valid): a privileged, pre-simulated future ball trajectory (used for
        deflection / Plinko scenes); intercepts are only planned on samples after `t_valid`."""
        self.ball, self.box, self.lead, self.below = ball_body, catch_box, lead, below
        self.launched_fn = launched_fn
        self.oracle = oracle
        super().__init__(env)

    def reset(self):
        super().reset()
        self.goal = self.cmd_pos.copy()
        self.home = self.cmd_pos.copy()

    def plan(self):
        e = self.env
        p, v = e.body_pos(self.ball), e.body_linvel(self.ball)
        net, _ = e.arm.site_pose()
        if self.oracle is not None:
            P_all, t_all, t_valid = self.oracle
            keep = (t_all > e.data.time) & (t_all >= t_valid)
            P, ts = P_all[keep], t_all[keep] - e.data.time
            VZ = np.gradient(P_all[:, 2], t_all)[keep] if len(P) else np.zeros(0)
        elif (self.launched_fn and not self.launched_fn()) or np.linalg.norm(v) < 1e-3:
            self.move_toward(self.home)
            return
        else:
            ts = np.arange(0.0, 2.5, 0.005)
            P = p + np.outer(ts, v) + 0.5 * np.outer(ts ** 2, G)
            VZ = v[2] + G[2] * ts
        lo, hi = map(np.asarray, self.box)
        ok = np.all((P >= lo) & (P <= hi), axis=1) & (VZ < 0)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            return
        need = np.linalg.norm(P[idx] - net, axis=1) / self.max_speed + self.lead
        feasible = idx[need <= ts[idx]]
        i = feasible[0] if len(feasible) else idx[np.argmin(need - ts[idx])]
        if ts[i] > 0.06:
            self.goal = P[i] - [0, 0, self.below]
        self.move_toward(self.goal, speed=2.0)


# ============================================================================= stop ball
class BlockBallExpert(Expert):
    """Rolling ball: put half-open fingers across the track ahead of the ball and hold."""

    def __init__(self, env, ball_body, axis_fn, floor_fn, accel=0.0, lead=0.1, grip=0.25):
        self.ball, self.axis_fn, self.floor_fn, self.accel = ball_body, axis_fn, floor_fn, accel
        self.lead, self.block_grip = lead, grip
        super().__init__(env)

    def reset(self):
        super().reset()
        self.point = None

    def plan(self):
        e = self.env
        p, v = e.body_pos(self.ball), e.body_linvel(self.ball)
        u = self.axis_fn()
        speed = float(np.dot(v, u))
        tcp, _ = e.arm.site_pose()
        self.grip = self.block_grip
        self.cmd_quat = top_down_quat(np.arctan2(u[1], u[0]))
        if self.point is None:
            lo, hi = map(np.asarray, e.workspace)
            for d in np.arange(0.10, 0.9, 0.01):
                q = p + u * d
                t_ball = self._time_to(d, speed)
                t_arm = np.linalg.norm(q + [0, 0, 0.08] - tcp) / (0.6 * self.max_speed) + 0.6
                if t_ball > t_arm and np.all(q[:2] >= lo[:2]) and np.all(q[:2] <= hi[:2]) and self._reach(q):
                    self.point = q
                    break
            if self.point is None:
                return
            self.set_phase("hover")
        z_floor = self.floor_fn(self.point)
        target = np.array([self.point[0], self.point[1], z_floor + 0.012])
        if self.phase == "hover":
            self.move_toward(target + [0, 0, 0.08])
            if np.linalg.norm(tcp - (target + [0, 0, 0.08])) < 0.02:
                self.set_phase("lower")
        elif self.phase == "lower":
            self.move_toward(target, speed=0.5)

    def _time_to(self, d, v):
        a = self.accel
        if abs(a) < 1e-6:
            return d / max(v, 1e-3)
        disc = v * v + 2 * a * d
        return (-v + np.sqrt(max(disc, 0))) / a

    def _reach(self, q):
        r = np.hypot(q[0] - self.env.robot_base[0], q[1] - self.env.robot_base[1])
        return 0.3 < r < 0.78


# ============================================================================= air hockey
class AirHockeyExpert(Expert):
    """Predict the puck's crossing of a defence line (with side-rail reflections), wait there
    and strike it back up the table."""

    def __init__(self, env, x_def=0.55, strike=0.16, t_strike=0.22):
        self.x_def, self.strike_dist, self.t_strike = x_def, strike, t_strike
        super().__init__(env)

    def reset(self):
        super().reset()
        self.max_speed = 2.0

    def plan(self):
        e = self.env
        p, v = e.puck_state()
        z = e.ah["top_z"] + e.mallet_height
        mallet, _ = e.arm.site_pose()
        y_lim = e.ah["width"] / 2 - e.puck["radius"]
        if v[0] < -1e-3:
            t_hit = (self.x_def - p[0]) / v[0]
            y = _reflect(p[1] + v[1] * max(t_hit, 0), y_lim)
        else:
            t_hit, y = np.inf, mallet[1]
        if self.phase in ("start", "track"):
            self.set_phase("track")
            goal = np.array([self.x_def - 0.10, np.clip(y, -0.4, 0.4), z])
            self.move_toward(goal)
            if 0 < t_hit < self.t_strike:
                self.strike_goal = np.array([self.x_def + self.strike_dist, np.clip(y, -0.4, 0.4), z])
                self.set_phase("strike")
        elif self.phase == "strike":
            self.move_toward(self.strike_goal, speed=1.5)
            if self.phase_time > 0.45 or v[0] > 0.05:
                self.set_phase("recover")
        elif self.phase == "recover":
            self.move_toward(np.array([self.x_def - 0.10, 0.0, z]), speed=0.8)


def _reflect(y, lim):
    """Fold an unbounded coordinate into [-lim, lim] (mirror reflections off both rails)."""
    period = 4 * lim
    y = (y + lim) % period
    return (y if y <= 2 * lim else period - y) - lim
