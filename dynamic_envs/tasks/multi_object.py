"""Multi-object scenarios (paper Fig. 5c/d/h, Table 3):

  MultiObject (MO)        3-5 objects move on the conveyor; pick the language-specified one.
  PlaceInBox (PiB)        a conveyor-mounted box passes at 10-20 cm/s; place a stationary
                          (table) object into the moving box.
  LangMultiObject (L+MO)  3-5 moving objects + a blue and a brown box on the table; pick the
                          instructed object and bin it in the instructed box.
"""
from dataclasses import dataclass

import numpy as np

from ..base_env import TABLE_Z, CameraSpec, DynamicEnv
from ..components import BLUE, CARDBOARD, OBJECT_LIBRARY, add_conveyor, add_object, add_open_box
from ..experts import PickExpert
from ..geometry import quat_axis_angle
from ..success import GraspTracker, in_box
from .common import MotionConfig, SpeedProfile

POOL = ("duck", "apple", "lemon", "banana", "lime", "can", "bottle", "mug", "cup")
BOX_HALF = (0.085, 0.085, 0.05)


def grasp_pose(env, body, kind, default_yaw=np.pi / 2):
    """Gripper yaw (fingers close along the object's narrow axis) and world-frame grasp offset."""
    spec = OBJECT_LIBRARY[kind]
    R = env.data.xmat[env.bid(body)].reshape(3, 3)
    theta = np.arctan2(R[1, 0], R[0, 0])
    if spec.close_axis == "x":
        yaw = theta + np.pi / 2
    elif spec.close_axis == "y":
        yaw = theta
    else:
        yaw = default_yaw
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2          # gripper is symmetric under pi
    off = R[:2, :2] @ np.asarray(spec.grasp_offset)
    return yaw, off


@dataclass
class MultiObjectConfig(MotionConfig):
    episode_seconds: float = 22.0
    speed_range: tuple = (0.05, 0.20)
    n_objects_range: tuple = (3, 5)
    spacing_range: tuple = (0.14, 0.20)
    belt_height_range: tuple = (0.0, 0.04)
    first_offset_range: tuple = (0.25, 0.40)     # |y| of the leading object at t = 0


class ConveyorObjectsScene(DynamicEnv):
    """Conveyor in front of the robot + a pool of objects that are placed or parked per episode."""

    Config = MultiObjectConfig
    family = "complex"
    table = dict(center=(0.1, 0.0), size=(0.7, 1.1))
    camera = CameraSpec(lookat=(0.42, 0.0, TABLE_Z + 0.08), distance=2.0, azimuth=212, elevation=-28)
    conveyor_x = 0.62
    exit_y = 0.62

    def build(self, b):
        add_conveyor(b, "conveyor", length=2.6)
        for k in POOL:
            add_object(b, f"obj_{k}", k)
        self.build_extra(b)

    def build_extra(self, b):
        pass

    def setup(self):
        self.belt = self.gid("conveyor_belt")
        self.conv = self.bid("conveyor")
        self.obj_geoms = {k: sorted(self.body_geoms(f"obj_{k}")) for k in POOL}
        self.geom_bits = {k: (self.model.geom_contype[g].copy(), self.model.geom_conaffinity[g].copy())
                          for k, g in self.obj_geoms.items()}
        self.table_geom = self.gid("table_top")
        self.grasp = None
        self.dir, self.speed = 1, None
        self.setup_extra()

    def setup_extra(self):
        pass

    def trackers(self):
        return [self.grasp] if self.grasp else []

    # ---------------------------------------------------------------- object pool
    def park_all(self):
        for i, k in enumerate(POOL):
            g = self.obj_geoms[k]
            self.model.geom_contype[g] = 0
            self.model.geom_conaffinity[g] = 0
            self.model.body_gravcomp[self.bid(f"obj_{k}")] = 1.0
            self.set_free_pose(f"obj_{k}", [0.3 * i, 0.0, -1.5])
        self.active = []

    def activate(self, kind, pos, yaw, linvel=(0, 0, 0)):
        g = self.obj_geoms[kind]
        self.model.geom_contype[g], self.model.geom_conaffinity[g] = self.geom_bits[kind]
        self.model.body_gravcomp[self.bid(f"obj_{kind}")] = 0.0
        self.set_free_pose(f"obj_{kind}", pos, quat_axis_angle([0, 0, 1], yaw), linvel=linvel)
        self.active.append(kind)

    # ---------------------------------------------------------------- conveyor
    def reset_conveyor(self, rng, profile_cfg):
        h = rng.uniform(*self.cfg.belt_height_range)
        x = self.conveyor_x + rng.uniform(-0.02, 0.02)
        self.model.body_pos[self.conv] = [x, 0.0, TABLE_Z + h]
        self.model.body_quat[self.conv] = quat_axis_angle([0, 0, 1], np.pi / 2)
        self.dir = self.cfg.direction if self.cfg.direction is not None else rng.choice([-1, 1])
        self.speed = SpeedProfile(profile_cfg, rng)
        return x, TABLE_Z + h

    def pre_substep(self, data):
        if self.speed is not None:
            self.model.geom_surfacevel[self.belt, 0] = self.dir * self.speed(data.time)

    def place_on_belt(self, rng, kinds, x, z):
        c = self.cfg
        y = -self.dir * rng.uniform(*c.first_offset_range)
        v = [0, self.dir * self.speed(0.0), 0]
        for k in kinds:
            self.activate(k, [x + rng.uniform(-0.025, 0.025), y, z + 1e-4], rng.uniform(-np.pi, np.pi), v)
            y -= self.dir * rng.uniform(*c.spacing_range)

    def wrong_object_lifted(self, ref_z):
        for k in self.active:
            if k == self.target:
                continue
            g = set(self.obj_geoms[k])
            if (self.touching(self.arm.left_finger_geoms, g) and self.touching(self.arm.right_finger_geoms, g)
                    and self.body_pos(f"obj_{k}")[2] > ref_z + 0.05):
                return True
        return False

    def pick_expert(self, default_yaw=np.pi / 2, **kw):
        body = f"obj_{self.target}"
        yaw, off = grasp_pose(self, body, self.target, default_yaw)
        kw.setdefault("yaw", yaw)
        return PickExpert(self, body, grasp_z=OBJECT_LIBRARY[self.target].grasp_z, offset=off, **kw)


# ============================================================================= MO
class MultiObject(ConveyorObjectsScene):
    """Multiple objects (3-5) move simultaneously on the conveyor; pick the language-specified one."""

    def build_extra(self, b):
        add_open_box(b, "box", BOX_HALF, rgba=CARDBOARD, mass=0.4)

    def reset_task(self, rng):
        self.park_all()
        x, z = self.reset_conveyor(rng, self.cfg)
        n = int(rng.integers(self.cfg.n_objects_range[0], self.cfg.n_objects_range[1] + 1))
        kinds = list(rng.choice(POOL, size=n, replace=False))
        self.place_on_belt(rng, kinds, x, z)
        self.target = kinds[int(rng.integers(n))]
        self.belt_z = z
        self.set_free_pose("box", [rng.uniform(0.25, 0.35), rng.uniform(-0.3, 0.3), TABLE_Z + 1e-4],
                           quat_axis_angle([0, 0, 1], rng.uniform(-0.4, 0.4)))
        self.grasp = GraspTracker(self, f"obj_{self.target}")
        self.instruction = f"pick up the {OBJECT_LIBRARY[self.target].desc}"

    def evaluate(self):
        if self.wrong_object_lifted(self.belt_z):
            self.events["wrong_object"] = float(self.data.time)
            return False, True
        r = self.grasp.update(self.dt)
        if r is not None:
            return r, not r
        p = self.body_pos(f"obj_{self.target}")
        if self.grasp.z_grasp is None and (self.dir * p[1] > self.exit_y or p[2] < TABLE_Z - 0.2):
            self.events["left_workspace"] = float(self.data.time)
            return False, True
        return False, False

    def make_expert(self, **kw):
        return self.pick_expert(**kw)

    def task_info(self):
        return dict(target=self.target, objects=list(self.active), belt_speed=self.dir * self.speed(self.data.time))


# ============================================================================= PiB
@dataclass
class PlaceInBoxConfig(MultiObjectConfig):
    episode_seconds: float = 20.0
    speed_range: tuple = (0.10, 0.20)            # paper: 10-20 cm/s
    n_objects_range: tuple = (3, 5)
    box_start_range: tuple = (0.50, 0.65)


class PlaceInBox(ConveyorObjectsScene):
    """A conveyor-mounted box passes through the workspace; the robot must place a stationary
    table object into the moving box.  Success: the object is inside the box when the box
    leaves the workspace, and it never touched the table again after being picked up."""

    Config = PlaceInBoxConfig

    def build_extra(self, b):
        add_open_box(b, "box", BOX_HALF, rgba=CARDBOARD, mass=0.4)

    def setup_extra(self):
        self.box_body = self.bid("box")

    def reset_task(self, rng):
        c = self.cfg
        self.park_all()
        x, z = self.reset_conveyor(rng, c)
        n = int(rng.integers(c.n_objects_range[0], c.n_objects_range[1] + 1))
        kinds = list(rng.choice(POOL, size=n, replace=False))
        spots = self._table_spots(rng, n)
        for k, (px, py) in zip(kinds, spots):
            self.activate(k, [px, py, TABLE_Z + 1e-4], rng.uniform(-np.pi, np.pi))
        self.target = kinds[int(rng.integers(n))]
        y0 = -self.dir * rng.uniform(*c.box_start_range)
        self.set_free_pose("box", [x, y0, z + 1e-4], linvel=[0, self.dir * self.speed(0.0), 0])
        self.grasp = GraspTracker(self, f"obj_{self.target}")
        self.touched_table_after_lift = False
        self.instruction = f"put the {OBJECT_LIBRARY[self.target].desc} in the moving box"

    def _table_spots(self, rng, n, min_sep=0.12):
        spots = []
        while len(spots) < n:
            p = np.array([rng.uniform(0.28, 0.46), rng.uniform(-0.30, 0.30)])
            if all(np.linalg.norm(p - q) > min_sep for q in spots):
                spots.append(p)
        return spots

    def post_substep(self):
        if self.grasp.lifted_once and not self.touched_table_after_lift:
            if self.touching(set(self.obj_geoms[self.target]), {self.table_geom}):
                self.touched_table_after_lift = True

    def evaluate(self):
        self.grasp.update(self.dt)
        if self.wrong_object_lifted(TABLE_Z):
            self.events["wrong_object"] = float(self.data.time)
            return False, True
        if self.touched_table_after_lift:
            self.events["touched_table"] = float(self.data.time)
            return False, True
        bp = self.data.xpos[self.box_body]
        if self.dir * bp[1] > self.exit_y:
            inside = in_box(self, f"obj_{self.target}", "box", BOX_HALF)
            held = self.touching(self.arm.left_finger_geoms | self.arm.right_finger_geoms,
                                 set(self.obj_geoms[self.target]))
            if not inside:
                self.events["box_left_without_object"] = float(self.data.time)
            return bool(inside and not held), not (inside and not held)
        return False, False

    def box_state(self):
        return self.data.xpos[self.box_body].copy(), self.body_linvel("box")

    def make_expert(self, **kw):
        # Table objects are carried mostly along +x: close the fingers along y and carry gently.
        return self.pick_expert(default_yaw=0.0, place_fn=self.box_state, place_height=0.15, use_acc=False,
                                carry_speed=0.5, **kw)

    def task_info(self):
        return dict(target=self.target, objects=list(self.active), box_speed=self.dir * self.speed(self.data.time))


# ============================================================================= L+MO
class LangMultiObject(ConveyorObjectsScene):
    """3-5 moving objects and two similar boxes ("blue box", "brown box"); pick the
    instruction-specified object and bin it in the instruction-specified box."""

    box_names = {"blue": "box_blue", "brown": "box_brown"}

    def build_extra(self, b):
        add_open_box(b, "box_blue", BOX_HALF, rgba=BLUE, flaps=False, mass=0.5)
        add_open_box(b, "box_brown", BOX_HALF, rgba=CARDBOARD, mass=0.5)

    def reset_task(self, rng):
        c = self.cfg
        self.park_all()
        x, z = self.reset_conveyor(rng, c)
        n = int(rng.integers(c.n_objects_range[0], c.n_objects_range[1] + 1))
        kinds = list(rng.choice(POOL, size=n, replace=False))
        self.place_on_belt(rng, kinds, x, z)
        self.target = kinds[int(rng.integers(n))]
        self.belt_z = z
        side = rng.choice([-1, 1])
        for color, s in (("blue", side), ("brown", -side)):
            self.set_free_pose(self.box_names[color], [rng.uniform(0.27, 0.36), s * rng.uniform(0.17, 0.28),
                                                       TABLE_Z + 1e-4],
                               quat_axis_angle([0, 0, 1], rng.uniform(-0.3, 0.3)))
        self.target_box = str(rng.choice(["blue", "brown"]))
        self.grasp = GraspTracker(self, f"obj_{self.target}")
        self.instruction = (f"put the {OBJECT_LIBRARY[self.target].desc} in the {self.target_box} box")

    def evaluate(self):
        self.grasp.update(self.dt)
        if self.wrong_object_lifted(self.belt_z):
            self.events["wrong_object"] = float(self.data.time)
            return False, True
        body = f"obj_{self.target}"
        g = set(self.obj_geoms[self.target])
        held = self.touching(self.arm.left_finger_geoms | self.arm.right_finger_geoms, g)
        slow = np.linalg.norm(self.body_linvel(body)) < 0.05
        for color, box in self.box_names.items():
            if in_box(self, body, box, BOX_HALF) and not held and slow:
                ok = color == self.target_box
                if not ok:
                    self.events["wrong_box"] = float(self.data.time)
                return ok, not ok
        p = self.body_pos(body)
        if self.grasp.z_grasp is None and (self.dir * p[1] > self.exit_y or p[2] < TABLE_Z - 0.2):
            self.events["left_workspace"] = float(self.data.time)
            return False, True
        return False, False

    def target_box_state(self):
        return self.data.xpos[self.bid(self.box_names[self.target_box])].copy(), np.zeros(3)

    def make_expert(self, **kw):
        return self.pick_expert(place_fn=self.target_box_state, place_height=0.15, **kw)

    def task_info(self):
        return dict(target=self.target, target_box=self.target_box, objects=list(self.active))
