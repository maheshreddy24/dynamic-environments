"""Success criteria from the paper (Appendix C.4), implemented as small trackers.

Trackers are updated at physics rate from `post_substep` (`substep`) or at control rate
from `evaluate` (`update`), and expose `.result` (None = undecided, True/False) and `.time`.
"""
import numpy as np

from .components import NET_DEPTH, NET_RADIUS


class GraspTracker:
    """Pick-and-place scenarios (conveyor/beam/pole + cup, MO, L+MO): the gripper closes on the
    target with both fingers in contact, lifts it >= 10 cm and holds it 3 s without dropping."""

    def __init__(self, env, body, lift=0.10, hold=3.0):
        self.env, self.body = env, body
        self.geoms = env.body_geoms(body)
        self.lift, self.hold = lift, hold
        self.reset()

    def reset(self):
        self.z_grasp = None
        self.hold_time = 0.0
        self.lifted_once = False
        self.last_grasp = -np.inf
        self.result = None
        self.time = None

    def grasped(self):
        e = self.env
        return e.touching(e.arm.left_finger_geoms, self.geoms) and e.touching(e.arm.right_finger_geoms, self.geoms)

    def update(self, dt, flicker=0.25):
        if self.result is not None:
            return self.result
        e = self.env
        t = e.data.time
        z = e.body_pos(self.body)[2]
        g = self.grasped()
        if g:
            self.last_grasp = t
        if self.z_grasp is None and (g or e.touching(e.arm.left_finger_geoms | e.arm.right_finger_geoms, self.geoms)):
            self.z_grasp = z   # resting height at first finger contact = lift reference
        held = t - self.last_grasp < flicker    # tolerate single-frame contact flicker
        if held and self.z_grasp is not None and z - self.z_grasp >= self.lift:
            self.lifted_once = True
            self.hold_time += dt
            if self.hold_time >= self.hold:
                self.result, self.time = True, t
        else:
            self.hold_time = 0.0
            if self.lifted_once and z < self.z_grasp + 0.5 * self.lift:
                self.result, self.time = False, t   # dropped
        return self.result


def ball_in_net(env, ball_body, ball_radius, margin=0.004):
    """True if the ball centre lies inside the octagonal basket attached to the gripper."""
    nb = env.bid("panda/net")
    p = env.data.xpos[env.bid(ball_body)] - env.data.xpos[nb]
    local = env.data.xmat[nb].reshape(3, 3).T @ p
    r = np.hypot(local[0], local[1])
    return bool(r < NET_RADIUS * np.cos(np.pi / 8) - ball_radius * 0.5 + margin
                and -NET_DEPTH - margin < local[2] < ball_radius + 0.01)


class NetCatchTracker:
    """Ballistic catch / projectile catch / Mid / Occ: the ball must be inside the net at the
    moment its vertical velocity first becomes positive (after it started descending), i.e.
    before it bounces off the gripper or the floor.  A ball resting in the net counts too."""

    def __init__(self, env, ball_body, ball_radius, ground_geoms=()):
        self.env, self.body, self.radius = env, ball_body, ball_radius
        self.ball_geoms = env.body_geoms(ball_body)
        self.ground = set(ground_geoms)
        self.qa, self.da = env.free_qadr(ball_body)
        self.reset()

    def reset(self, start_time=0.0):
        self.start_time = start_time   # ignore everything before (e.g. bounces off deflectors)
        self.descending = False
        self.inside_time = 0.0
        self.result = None
        self.time = None

    def substep(self, dt):
        if self.result is not None or self.env.data.time < self.start_time:
            return
        e = self.env
        vz = e.data.qvel[self.da + 2]
        inside = ball_in_net(e, self.body, self.radius)
        self.inside_time = self.inside_time + dt if inside else 0.0
        if vz < -0.05:
            self.descending = True
        if self.descending and vz > 0.02:
            self.result, self.time = inside, e.data.time
        elif self.inside_time > 0.3:
            self.result, self.time = True, e.data.time
        elif self.ground and e.touching(self.ball_geoms, self.ground):
            self.result, self.time = False, e.data.time


class ContactWindowTracker:
    """Stress tests: the gripper (incl. net) must touch the ball inside [t_start, t_end]."""

    def __init__(self, env, ball_body):
        self.env = env
        self.ball_geoms = env.body_geoms(ball_body)
        self.window = (0.0, np.inf)
        self.reset()

    def reset(self, window=None):
        if window is not None:
            self.window = window
        self.result = None
        self.time = None
        self.first_contact = None

    def substep(self):
        e = self.env
        t = e.data.time
        if self.result is not None:
            return
        if e.touching(e.arm.geom_ids, self.ball_geoms):
            if self.first_contact is None:
                self.first_contact = t
            if self.window[0] <= t <= self.window[1]:
                self.result, self.time = True, t
        elif t > self.window[1]:
            self.result, self.time = False, t


class StopTracker:
    """Rolling ball: the ball speed must fall below 10% of its pre-contact speed within 500 ms
    of the first robot contact."""

    def __init__(self, env, ball_body, frac=0.10, window=0.5):
        self.env = env
        self.ball_geoms = env.body_geoms(ball_body)
        self.qa, self.da = env.free_qadr(ball_body)
        self.frac, self.window = frac, window
        self.reset()

    def reset(self):
        self.prev_speed = 0.0
        self.first_contact = None
        self.v_pre = None
        self.result = None
        self.time = None

    def speed(self):
        return float(np.linalg.norm(self.env.data.qvel[self.da:self.da + 3]))

    def substep(self):
        if self.result is not None:
            return
        e = self.env
        t = e.data.time
        s = self.speed()
        if self.first_contact is None:
            if e.touching(e.arm.geom_ids, self.ball_geoms):
                self.first_contact, self.v_pre = t, max(self.prev_speed, 1e-3)
            else:
                self.prev_speed = s
                return
        if t - self.first_contact <= self.window:
            if s < self.frac * self.v_pre:
                self.result, self.time = True, t
        else:
            self.result, self.time = False, t


def in_box(env, obj_body, box_body, box_half, margin=0.0):
    """Object origin inside the open box's footprint and below its rim."""
    bb = env.bid(box_body)
    p = env.data.xpos[env.bid(obj_body)] - env.data.xpos[bb]
    local = env.data.xmat[bb].reshape(3, 3).T @ p
    hx, hy, hz = box_half
    return bool(abs(local[0]) < hx + margin and abs(local[1]) < hy + margin and -0.01 < local[2] < 2 * hz + 0.03)
