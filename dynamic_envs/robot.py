"""Franka Panda arm interface: IK-based end-effector control on top of the Menagerie
joint-position servos (which stand in for the Panda's internal impedance controller)."""
import numpy as np
import mujoco

from .geometry import mat_to_quat, quat_error

ARM_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # rad/s (Franka datasheet)
MENAGERIE_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
FINGER_MAX = 0.04  # m per finger
GRIP_CTRL_MAX = 255.0


def _id(model, objtype, name):
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise KeyError(f"{objtype.name} '{name}' not found")
    return i


class PandaArm:
    def __init__(self, model, data, prefix="panda/", control_site="tcp"):
        self.m, self.d, self.prefix = model, data, prefix
        J = mujoco.mjtObj.mjOBJ_JOINT
        self.joint_ids = np.array([_id(model, J, f"{prefix}joint{i}") for i in range(1, 8)])
        self.qadr = model.jnt_qposadr[self.joint_ids]
        self.dadr = model.jnt_dofadr[self.joint_ids]
        self.finger_qadr = np.array([model.jnt_qposadr[_id(model, J, f"{prefix}finger_joint{i}")] for i in (1, 2)])
        self.finger_dadr = np.array([model.jnt_dofadr[_id(model, J, f"{prefix}finger_joint{i}")] for i in (1, 2)])
        A = mujoco.mjtObj.mjOBJ_ACTUATOR
        self.act_ids = np.array([_id(model, A, f"{prefix}actuator{i}") for i in range(1, 8)])
        self.grip_act = _id(model, A, f"{prefix}actuator8")
        lo, hi = model.jnt_range[self.joint_ids].T
        self.q_lo, self.q_hi = lo + 0.02, hi - 0.02
        self.tcp_site = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}tcp")
        self.site = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{control_site}")
        self.scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        root = _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}link0")
        self.body_ids = {i for i in range(model.nbody) if self._is_descendant(i, root)}
        self.geom_ids = {g for g in range(model.ngeom) if model.geom_bodyid[g] in self.body_ids
                         and (model.geom_contype[g] or model.geom_conaffinity[g])}
        lf = _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}left_finger")
        rf = _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}right_finger")
        self.left_finger_geoms = {g for g in self.geom_ids if model.geom_bodyid[g] == lf}
        self.right_finger_geoms = {g for g in self.geom_ids if model.geom_bodyid[g] == rf}

        self.q_rest = MENAGERIE_HOME.copy()
        self.q_prev = MENAGERIE_HOME.copy()
        self.q_next = MENAGERIE_HOME.copy()
        self.grip_prev = self.grip_next = GRIP_CTRL_MAX
        self.frozen = False

    def _is_descendant(self, body, root):
        while body > 0:
            if body == root:
                return True
            body = self.m.body_parentid[body]
        return False

    # ------------------------------------------------------------------ state
    @property
    def q(self):
        return self.d.qpos[self.qadr].copy()

    @property
    def qd(self):
        return self.d.qvel[self.dadr].copy()

    @property
    def gripper_width(self):
        return float(self.d.qpos[self.finger_qadr].sum())

    def site_pose(self, site=None, data=None):
        data = data if data is not None else self.d
        sid = self.site if site is None else site
        return data.site_xpos[sid].copy(), mat_to_quat(data.site_xmat[sid])

    def site_velocity(self, site=None):
        sid = self.site if site is None else site
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, sid, vel, 0)
        return vel[3:].copy()  # linear velocity, world frame

    def reset(self, q, grip=GRIP_CTRL_MAX):
        q = np.clip(np.asarray(q, float), self.q_lo, self.q_hi)
        self.d.qpos[self.qadr] = q
        self.d.qvel[self.dadr] = 0
        w = grip / GRIP_CTRL_MAX * FINGER_MAX
        self.d.qpos[self.finger_qadr] = w
        self.d.qvel[self.finger_dadr] = 0
        self.q_prev = q.copy()
        self.q_next = q.copy()
        self.grip_prev = self.grip_next = float(grip)
        self.d.ctrl[self.act_ids] = q
        self.d.ctrl[self.grip_act] = grip

    # ------------------------------------------------------------------ IK
    def ik(self, target_pos, target_quat=None, q0=None, site=None, iters=40, tol=1e-4, rot_weight=0.5,
           damping=1e-3, posture_gain=0.02, max_step=0.25):
        """Damped-least-squares IK for `site` (default: control site) with a nullspace pull
        towards `self.q_rest`.  Runs on a scratch MjData so the simulation is untouched."""
        s, m = self.scratch, self.m
        sid = self.site if site is None else site
        q = (self.q_next if q0 is None else np.asarray(q0, float)).copy()
        s.qpos[:] = self.d.qpos
        for _ in range(iters):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, s)
            mujoco.mj_comPos(m, s)
            err_p = np.asarray(target_pos) - s.site_xpos[sid]
            if target_quat is not None:
                err_r = quat_error(np.asarray(target_quat), mat_to_quat(s.site_xmat[sid])) * rot_weight
                err = np.concatenate([err_p, err_r])
            else:
                err = err_p
            mujoco.mj_jacSite(m, s, self._jacp, self._jacr, sid)
            Jfull = np.vstack([self._jacp, self._jacr * rot_weight]) if target_quat is not None else self._jacp
            Jac = Jfull[:, self.dadr]
            JJt = Jac @ Jac.T + damping * np.eye(Jac.shape[0])
            dq = Jac.T @ np.linalg.solve(JJt, err)
            # Nullspace posture term.
            Jpinv = Jac.T @ np.linalg.inv(JJt)
            dq += (np.eye(7) - Jpinv @ Jac) @ (posture_gain * (self.q_rest - q))
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, self.q_lo, self.q_hi)
            if np.linalg.norm(err) < tol:
                break
        return q

    # ------------------------------------------------------------------ commands
    def command(self, q_target, grip, dt):
        """Set a new joint-space target for the next control interval; the target is limited
        by the Panda joint velocity limits and interpolated across physics substeps."""
        self.q_prev = self.q_next.copy()
        if self.frozen:
            self.grip_prev = self.grip_next
            return
        step = np.clip(np.asarray(q_target) - self.q_prev, -ARM_VEL_LIMITS * dt, ARM_VEL_LIMITS * dt)
        self.q_next = np.clip(self.q_prev + step, self.q_lo, self.q_hi)
        self.grip_prev = self.grip_next
        self.grip_next = float(np.clip(grip, 0.0, GRIP_CTRL_MAX))

    def apply(self, alpha):
        self.d.ctrl[self.act_ids] = self.q_prev + alpha * (self.q_next - self.q_prev)
        self.d.ctrl[self.grip_act] = self.grip_prev + alpha * (self.grip_next - self.grip_prev)
