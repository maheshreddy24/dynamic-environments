"""Thin wrapper around `mujoco.MjSpec` used to compose every scenario procedurally.

Collision filtering (contype, conaffinity).  Two geoms collide iff
(c1.contype & c2.conaffinity) or (c2.contype & c1.conaffinity):

    STATIC  (1, 1)   table, conveyor frame, walls, deflectors, pegs
    ROBOT   (2, 1)   Panda links, fingers, net        -> no self-collision
    OBJECT  (1, 7)   cups, balls, fruit, boxes         -> touch everything
    PUSHER  (4, 0)   kinematic pole pusher             -> touches objects only
    PUCK    (8, 8)   air-hockey puck                   -> rails + mallet only
    RAIL    (9, 9)   air-hockey rails                  -> static + puck
    MALLET  (10, 9)  paddle held by the robot          -> static, objects, puck
    VISUAL  (0, 0)   decoration only
"""
import numpy as np
import mujoco

from . import paths
from .geometry import quat_from_z, normalize

STATIC = (1, 1)
ROBOT = (2, 1)
OBJECT = (1, 7)
PUSHER = (4, 0)
PUCK = (8, 8)
RAIL = (9, 9)
MALLET = (10, 9)
VISUAL = (0, 0)

G = mujoco.mjtGeom
BOX, SPHERE, CYLINDER, CAPSULE, MESH, PLANE, ELLIPSOID = (
    G.mjGEOM_BOX, G.mjGEOM_SPHERE, G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE,
    G.mjGEOM_MESH, G.mjGEOM_PLANE, G.mjGEOM_ELLIPSOID)

GROUP_VISIBLE = 0
GROUP_ROBOT = 2
GROUP_HIDDEN = 3


def _size3(size):
    s = list(np.atleast_1d(np.asarray(size, float)))
    return s + [0.0] * (3 - len(s))


class SceneBuilder:
    """Accumulates assets / bodies in an MjSpec and hands out materials by colour."""

    def __init__(self, name, timestep=1.0 / 480.0):
        spec = mujoco.MjSpec()
        spec.modelname = name
        spec.compiler.autolimits = True
        spec.option.timestep = timestep
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        spec.visual.global_.offwidth = 1920
        spec.visual.global_.offheight = 1440
        spec.visual.quality.shadowsize = 4096
        spec.visual.quality.offsamples = 0    # MSAA is GPU-nondeterministic; we supersample instead
        spec.visual.headlight.ambient = [0.32, 0.32, 0.32]
        spec.visual.headlight.diffuse = [0.38, 0.38, 0.38]
        spec.visual.headlight.specular = [0.05, 0.05, 0.05]
        spec.visual.map.znear = 0.01
        spec.visual.map.zfar = 60.0
        spec.visual.map.shadowclip = 4.0      # avoids shadow acne on the floor/table at 224 px
        spec.visual.rgba.haze = [0.33, 0.39, 0.50, 1.0]
        spec.stat.extent = 2.5
        spec.stat.center = [0.4, 0.0, 0.9]
        self.spec = spec
        self.world = spec.worldbody
        self._materials = {}
        self._meshes = set()

    # ------------------------------------------------------------------ assets
    def material(self, rgba, specular=0.3, shininess=0.3, reflectance=0.0, name=None):
        rgba = tuple(float(c) for c in rgba)
        key = name or ("rgba_%.3f_%.3f_%.3f_%.3f_%.2f_%.2f_%.2f" % (rgba + (specular, shininess, reflectance)))
        if key not in self._materials:
            self.spec.add_material(name=key, rgba=list(rgba), specular=specular,
                                   shininess=shininess, reflectance=reflectance)
            self._materials[key] = key
        return key

    def textured_material(self, name, png_path, rgba=(1, 1, 1, 1), specular=0.2, shininess=0.2):
        if name not in self._materials:
            self.spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png_path))
            mat = self.spec.add_material(name=name, rgba=list(rgba), specular=specular, shininess=shininess)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
            self._materials[name] = name
        return name

    def mesh(self, name, obj_path, scale=(1, 1, 1)):
        if name not in self._meshes:
            self.spec.add_mesh(name=name, file=str(obj_path), scale=list(scale))
            self._meshes.add(name)
        return name

    # ------------------------------------------------------------------ geoms
    def geom(self, body, gtype, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), rgba=None, material=None,
             col=STATIC, group=GROUP_VISIBLE, name=None, mass=None, density=None, friction=None,
             solref=None, solimp=None, solmix=None, condim=None, priority=None, meshname=None,
             specular=0.3, shininess=0.3, reflectance=0.0):
        kw = dict(type=gtype, size=_size3(size), pos=list(pos), quat=list(quat),
                  contype=col[0], conaffinity=col[1], group=group)
        if name:
            kw["name"] = name
        if meshname:
            kw["meshname"] = meshname
        if material is None and rgba is not None:
            material = self.material(rgba, specular, shininess, reflectance)
        if material is not None:
            kw["material"] = material
        if rgba is not None and len(rgba) == 4 and rgba[3] == 0:
            kw["rgba"] = list(rgba)
        g = body.add_geom(**kw)
        if mass is not None:
            g.mass = mass
        if density is not None:
            g.density = density
        if friction is not None:
            g.friction = list(friction)
        if solref is not None:
            g.solref = list(solref)
        if solimp is not None:
            g.solimp = list(solimp)
        if solmix is not None:
            g.solmix = solmix
        if condim is not None:
            g.condim = condim
        if priority is not None:
            g.priority = priority
        return g

    def box(self, body, half, pos=(0, 0, 0), **kw):
        return self.geom(body, BOX, half, pos=pos, **kw)

    def sphere(self, body, r, pos=(0, 0, 0), **kw):
        return self.geom(body, SPHERE, [r], pos=pos, **kw)

    def cylinder(self, body, r, half_h, pos=(0, 0, 0), **kw):
        return self.geom(body, CYLINDER, [r, half_h], pos=pos, **kw)

    def capsule(self, body, r, half_h, pos=(0, 0, 0), **kw):
        return self.geom(body, CAPSULE, [r, half_h], pos=pos, **kw)

    def segment(self, body, p0, p1, r, kind=CAPSULE, **kw):
        """Capsule / cylinder spanning p0 -> p1 (in body coordinates)."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = float(np.linalg.norm(d))
        return self.geom(body, kind, [r, L / 2.0], pos=(p0 + p1) / 2.0, quat=quat_from_z(normalize(d)), **kw)

    # ------------------------------------------------------------------ misc
    def body(self, parent=None, name=None, pos=(0, 0, 0), quat=(1, 0, 0, 0), mocap=False):
        parent = parent if parent is not None else self.world
        kw = dict(pos=list(pos), quat=list(quat))
        if name:
            kw["name"] = name
        if mocap:
            kw["mocap"] = True
        return parent.add_body(**kw)

    def compile(self):
        return self.spec.compile()

    def to_xml(self, out_path):
        """Write a standalone MJCF file; absolute asset paths are made relative to `out_path`."""
        import os
        xml = self.spec.to_xml()
        out_dir = os.path.dirname(os.path.abspath(out_path))
        root = str(paths.ROOT)
        rel = os.path.relpath(root, out_dir)
        xml = xml.replace(f'file="{root}/', f'file="{rel}/')
        with open(out_path, "w") as fh:
            fh.write(xml)
        return out_path


def load_panda_spec():
    """Menagerie Panda as an MjSpec with absolute mesh paths, gravity compensation and
    collision bits remapped to the ROBOT class."""
    spec = mujoco.MjSpec.from_file(str(paths.PANDA_XML))
    meshdir = paths.PANDA_XML.parent / spec.meshdir
    for mesh in spec.meshes:
        mesh.file = str(meshdir / mesh.file)
    spec.meshdir = ""
    for key in list(spec.keys):
        spec.delete(key)
    for body in spec.bodies:
        body.gravcomp = 1.0
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = ROBOT
    # Grippy fingertips.
    for geom in spec.geoms:
        if geom.contype:
            geom.friction = [1.5, 0.01, 0.0001]
    return spec
