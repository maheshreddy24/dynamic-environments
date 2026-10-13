"""Reusable scene components (props) mirroring the paper's Figures 4-6.

Every `add_*` function adds bodies/geoms to a `SceneBuilder` and returns the
names that the task code needs at runtime.  Anything randomised per episode is
either a mocap body (moved via `data.mocap_pos/quat`), a static body whose
`model.body_pos/quat` is edited at reset, or a geom whose size is edited at
reset (see `set_geom_box`).
"""
import numpy as np
import mujoco

from . import paths
from .builder import (CYLINDER, MESH, PLANE, GROUP_HIDDEN, GROUP_ROBOT,
                      MALLET, OBJECT, PUCK, PUSHER, RAIL, ROBOT, STATIC, VISUAL, load_panda_spec)
from .geometry import quat_axis_angle

# ----------------------------------------------------------------------------- palette
WHITE = (0.93, 0.93, 0.92, 1)
TABLE_TOP = (0.94, 0.94, 0.93, 1)
LEG_GREY = (0.68, 0.68, 0.70, 1)
DARK = (0.13, 0.13, 0.15, 1)
BELT = (0.16, 0.16, 0.18, 1)
RAIL_GREY = (0.62, 0.63, 0.66, 1)
STEEL = (0.55, 0.57, 0.62, 1)
LAUNCHER_GREY = (0.52, 0.53, 0.58, 1)
CUP_RED = (0.55, 0.09, 0.08, 1)
NET_RED = (0.86, 0.17, 0.10, 1)
NET_MESH = (0.92, 0.92, 0.92, 0.45)
BLUE = (0.18, 0.47, 0.92, 1)
WOOD = (0.58, 0.40, 0.22, 1)
CARDBOARD = (0.74, 0.56, 0.34, 1)
BALL_COLORS = {"red": (0.88, 0.12, 0.10, 1), "green": (0.15, 0.75, 0.22, 1),
               "yellow": (0.98, 0.78, 0.10, 1), "blue": (0.15, 0.40, 0.95, 1),
               "orange": (1.0, 0.5, 0.05, 1)}

HIDDEN = (0.5, 0.5, 0.5, 0)

# Bouncy / absorbing contact parameters (solref = [timeconst, dampratio]).  Time constants below
# ~0.01 s with low damping are unstable at dt = 1/480 s and inject energy, so keep them >= 0.015.
BOUNCY = dict(solref=[0.02, 0.15], solmix=100.0)   # restitution ~0.7 (stable at dt=1/480)
ABSORB = dict(solref=[0.01, 1.2], solimp=[0.9, 0.95, 0.002, 0.5, 2.0], solmix=100.0)


# ----------------------------------------------------------------------------- environment
def add_environment(b):
    """Gradient sky, grey checker floor and two directional lights (Figs. 4-6 look)."""
    s = b.spec
    s.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                  builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                  rgb1=[0.40, 0.47, 0.60], rgb2=[0.16, 0.20, 0.28], width=512, height=3072)
    s.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                  builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                  rgb1=[0.42, 0.43, 0.47], rgb2=[0.31, 0.32, 0.36], markrgb=[0.45, 0.46, 0.50],
                  width=512, height=512)
    mat = s.add_material(name="floor", texrepeat=[3.4, 3.4], texuniform=True, reflectance=0.12)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    # Finite checkered floor (like the paper's renders); an invisible infinite plane below it
    # keeps anything that leaves the floor area from falling forever.
    b.world.add_geom(name="floor", type=PLANE, size=[3.6, 3.6, 0.05], pos=[0.8, 0, 0], material="floor",
                     contype=STATIC[0], conaffinity=STATIC[1])
    b.world.add_geom(name="floor_infinite", type=PLANE, size=[0, 0, 0.05], pos=[0, 0, -0.001],
                     rgba=[0, 0, 0, 0], group=3, contype=STATIC[0], conaffinity=STATIC[1])
    b.world.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 5],
                      dir=[-0.35, 0.25, -1.0], diffuse=[0.62, 0.62, 0.62], specular=[0.2, 0.2, 0.2],
                      castshadow=True)
    b.world.add_light(name="fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 5],
                      dir=[0.5, -0.4, -0.75], diffuse=[0.22, 0.22, 0.22], specular=[0, 0, 0],
                      castshadow=False)


def add_camera(b, name, fovy=45.0):
    return b.world.add_camera(name=name, pos=[2, 0, 2], fovy=fovy)


# ----------------------------------------------------------------------------- furniture
def add_table(b, name="table", center=(0.3, 0.0), size=(1.0, 1.0), top_z=0.75, thickness=0.04,
              leg_r=0.022, inset=0.07):
    body = b.body(name=name, pos=[center[0], center[1], 0.0])
    sx, sy = size
    b.box(body, [sx / 2, sy / 2, thickness / 2], pos=[0, 0, top_z - thickness / 2], rgba=TABLE_TOP,
          name=f"{name}_top", friction=[0.6, 0.005, 0.0001], specular=0.1)
    leg_h = top_z - thickness
    for dx in (-1, 1):
        for dy in (-1, 1):
            b.cylinder(body, leg_r, leg_h / 2, pos=[dx * (sx / 2 - inset), dy * (sy / 2 - inset), leg_h / 2],
                       rgba=LEG_GREY, col=VISUAL, specular=0.6, shininess=0.6)
    return f"{name}_top"


# ----------------------------------------------------------------------------- robot
def add_panda(b, base_pos, base_yaw=0.0, attachment=None):
    """Attach the Menagerie Panda (prefix 'panda/') and an optional end-effector tool."""
    panda = load_panda_spec()
    hand = panda.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.006, 0, 0], group=4)
    frame = b.world.add_frame(pos=list(base_pos), quat=list(quat_axis_angle([0, 0, 1], base_yaw)))
    frame.attach_body(panda.body("link0"), "panda/", "")
    hand = b.spec.body("panda/hand")
    if attachment == "net":
        add_net(b, hand)
    elif attachment == "mallet":
        add_mallet(b, hand)
    return "panda/"


NET_RADIUS = 0.085
NET_DEPTH = 0.10


def add_net(b, hand, n_sides=8):
    """Octagonal catching basket (red rim, black posts, translucent mesh) offset in front of
    the fingers so its opening faces up when the gripper points down (Figs. 5b, 6c)."""
    R, D = NET_RADIUS, NET_DEPTH
    # Arm from the hand to the rim (in hand coordinates; hand +z = finger direction).
    b.segment(hand, [0.035, 0, 0.075], [0.13 - R, 0, 0.10], 0.008, rgba=DARK, col=ROBOT,
              group=GROUP_ROBOT, mass=0.03)
    net = b.body(hand, name="panda/net", pos=[0.13, 0, 0.10], quat=[0, 1, 0, 0])  # net +z = opening
    net.gravcomp = 1.0
    ang = np.arange(n_sides) * 2 * np.pi / n_sides
    rim = np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros(n_sides)], 1)
    side = 2 * R * np.sin(np.pi / n_sides)
    apothem = R * np.cos(np.pi / n_sides)
    for i in range(n_sides):
        p0, p1 = rim[i], rim[(i + 1) % n_sides]
        b.segment(net, p0, p1, 0.006, rgba=NET_RED, col=ROBOT, group=GROUP_ROBOT, mass=0.01, **ABSORB)
        b.segment(net, p0, p0 - [0, 0, D], 0.0035, rgba=DARK, col=VISUAL, group=GROUP_ROBOT)
        mid = ang[i] + np.pi / n_sides
        c = np.array([apothem * np.cos(mid), apothem * np.sin(mid), -D / 2])
        q = quat_axis_angle([0, 0, 1], mid)
        b.box(net, [0.002, side / 2, D / 2], pos=c, quat=q, rgba=NET_MESH, col=ROBOT, group=GROUP_ROBOT,
              mass=0.005, friction=[1.2, 0.02, 0.001], **ABSORB)
    b.cylinder(net, R, 0.003, pos=[0, 0, -D], rgba=NET_MESH, col=ROBOT, group=GROUP_ROBOT, mass=0.03,
               friction=[1.2, 0.02, 0.001], **ABSORB)
    # Site frame = hand frame orientation, so every control site shares the top-down convention.
    net.add_site(name="panda/net_center", pos=[0, 0, -D * 0.45], quat=[0, 1, 0, 0], size=[0.006, 0, 0], group=4)
    return "panda/net_center"


def add_mallet(b, hand):
    """Air-hockey mallet clamped between the fingers; disk axis = finger axis (Fig. 5a)."""
    mallet = b.body(hand, name="panda/mallet", pos=[0, 0, 0.085])
    mallet.gravcomp = 1.0
    b.cylinder(mallet, 0.013, 0.035, pos=[0, 0, 0.0], rgba=BLUE, col=MALLET, group=GROUP_ROBOT, mass=0.05)
    b.cylinder(mallet, 0.045, 0.012, pos=[0, 0, 0.047], rgba=BLUE, col=MALLET, group=GROUP_ROBOT, mass=0.1,
               friction=[0.05, 0.005, 0.0001], solref=[0.02, 0.2], solmix=100.0)
    mallet.add_site(name="panda/mallet_center", pos=[0, 0, 0.047], size=[0.006, 0, 0], group=4)
    return "panda/mallet_center"


# ----------------------------------------------------------------------------- conveyor
def add_conveyor(b, name="conveyor", length=2.2, belt_width=0.18, leg_spacing=0.9):
    """Belt conveyor whose surface drags objects via `geom_surfacevel` (no moving geometry).
    Local frame: x along the belt, belt top surface at z = 0."""
    body = b.body(name=name)
    L, W = length, belt_width
    b.box(body, [L / 2, W / 2, 0.01], pos=[0, 0, -0.01], rgba=BELT, name=f"{name}_belt",
          friction=[1.0, 0.005, 0.0001], specular=0.05)
    body.geoms[-1].surfacevel = [1e-9, 0, 0, 0, 0, 0]  # compiled non-zero so runtime edits work
    for s in (-1, 1):
        b.box(body, [L / 2, 0.012, 0.022], pos=[0, s * (W / 2 + 0.012), 0.008], rgba=RAIL_GREY,
              specular=0.6, shininess=0.5)
        b.box(body, [L / 2, 0.006, 0.05], pos=[0, s * (W / 2 + 0.018), -0.06], rgba=STEEL, col=VISUAL)
    for s in (-1, 1):
        b.geom(body, CYLINDER, [0.03, W / 2 + 0.005], pos=[s * L / 2, 0, -0.03],
               quat=quat_axis_angle([1, 0, 0], np.pi / 2), rgba=DARK, col=VISUAL)
    b.box(body, [L / 2, W / 2, 0.015], pos=[0, 0, -0.05], rgba=DARK, col=VISUAL)
    n_legs = max(2, int(round(L / leg_spacing)) + 1)
    for x in np.linspace(-L / 2 + 0.08, L / 2 - 0.08, n_legs):
        for s in (-1, 1):
            b.cylinder(body, 0.018, 1.5, pos=[x, s * (W / 2 + 0.018), -1.6], rgba=STEEL, col=VISUAL,
                       specular=0.6)
    return dict(body=name, belt=f"{name}_belt", length=L, width=W)


# ----------------------------------------------------------------------------- beam
def add_beam(b, name="beam", beam_len=0.62, height=0.30, track_len=1.1, track_offset=0.11):
    """Post-mounted horizontal beam (Fig. 4b).  The cup glides along a hidden
    surface-velocity track that runs parallel to the beam, `track_offset` to its side, so the
    beam never blocks a top-down grasp.  Local frame: x along the beam, z = table top."""
    body = b.body(name=name)
    b.cylinder(body, 0.06, 0.006, pos=[0, 0, 0.006], rgba=DARK, specular=0.5)
    b.cylinder(body, 0.016, height / 2, pos=[0, 0, height / 2], rgba=(0.38, 0.39, 0.43, 1), specular=0.7)
    b.cylinder(body, 0.024, 0.02, pos=[0, 0, height], rgba=(0.38, 0.39, 0.43, 1), col=VISUAL)
    b.segment(body, [0.02, 0, height], [-beam_len, 0, height], 0.011, rgba=(0.82, 0.80, 0.76, 1),
              specular=0.5)
    b.segment(body, [0.0, 0.0, height + 0.0115], [-beam_len + 0.02, 0.0, height + 0.0115], 0.004,
              rgba=CUP_RED, col=VISUAL)
    b.sphere(body, 0.016, pos=[-beam_len, 0, height], rgba=(0.6, 0.25, 0.22, 1), col=VISUAL)
    b.box(body, [track_len / 2, 0.06, 0.001], pos=[-beam_len / 2, track_offset, 0.001], rgba=HIDDEN,
          group=GROUP_HIDDEN, name=f"{name}_track", friction=[1.0, 0.005, 0.0001])
    body.geoms[-1].surfacevel = [1e-9, 0, 0, 0, 0, 0]
    return dict(body=name, track=f"{name}_track", beam_len=beam_len, track_offset=track_offset)


# ----------------------------------------------------------------------------- pole pusher
def add_pole_pusher(b, name="pusher"):
    """Kinematic L-shaped pole (Fig. 4c) pushing the cup with a flat cross-bar.  Driven by
    prescribed slide-x / slide-y / hinge-z joints.  Local +x = push direction."""
    body = b.body(name=name, pos=[0, 0, 0])
    body.gravcomp = 1.0
    for jn, jt, ax in ((f"{name}_x", mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0]),
                       (f"{name}_y", mujoco.mjtJoint.mjJNT_SLIDE, [0, 1, 0]),
                       (f"{name}_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1])):
        body.add_joint(name=jn, type=jt, axis=ax, armature=10.0, damping=0)
    grey = (0.33, 0.34, 0.38, 1)
    b.box(body, [0.012, 0.055, 0.022], pos=[-0.012, 0, 0.026], rgba=grey, col=PUSHER, mass=1.0,
          friction=[0.3, 0.005, 0.0001])
    b.cylinder(body, 0.012, 0.09, pos=[-0.03, 0, 0.09], rgba=grey, col=VISUAL, mass=0.1)
    b.segment(body, [-0.03, 0, 0.012], [-0.17, 0, 0.012], 0.012, rgba=grey, col=VISUAL, mass=0.1)
    b.sphere(body, 0.02, pos=[-0.17, 0, 0.018], rgba=grey, col=VISUAL, mass=0.05)
    return dict(body=name, joints=[f"{name}_x", f"{name}_y", f"{name}_yaw"])


# ----------------------------------------------------------------------------- rolling channel
def add_channel(b, name="channel", length=0.9, inner_w=0.064, wall_h=0.03, wall_t=0.012):
    """U-shaped grey track for the rolling ball (Fig. 4d).  Local x along the track,
    track floor at z = 0, track centred at the origin.  A support block (`<name>_support`)
    is resized at reset to hold the high end of an inclined track."""
    body = b.body(name=name)
    grey = (0.45, 0.47, 0.53, 1)
    b.box(body, [length / 2, inner_w / 2 + wall_t, 0.008], pos=[0, 0, -0.008], rgba=(0.72, 0.74, 0.8, 1),
          friction=[0.8, 0.005, 0.0001], specular=0.4)
    for s in (-1, 1):
        b.box(body, [length / 2, wall_t / 2, wall_h / 2 + 0.008],
              pos=[0, s * (inner_w / 2 + wall_t / 2), wall_h / 2 - 0.008], rgba=grey, specular=0.4)
    b.box(body, [length / 2 - 0.05, inner_w / 2 + wall_t - 0.01, 0.02], pos=[0, 0, -0.036], rgba=grey,
          col=VISUAL)
    sup = b.body(name=f"{name}_support_body")
    b.box(sup, [0.06, inner_w / 2 + wall_t, 0.05], pos=[0, 0, 0.05], rgba=grey, name=f"{name}_support",
          col=VISUAL)
    return dict(body=name, support_body=f"{name}_support_body", support=f"{name}_support",
                length=length, inner_w=inner_w)


# ----------------------------------------------------------------------------- launcher
def add_launcher(b, name="launcher"):
    """Grey box launcher with a turret and a barrel (Figs. 5b, 5e-g).  Mocap body; the barrel
    geom (`<name>_barrel`) is re-oriented at reset to the launch direction."""
    body = b.body(name=name, mocap=True, pos=[3, 0, 0])
    b.box(body, [0.13, 0.13, 0.12], pos=[0, 0, 0.12], rgba=LAUNCHER_GREY, specular=0.3)
    b.cylinder(body, 0.075, 0.025, pos=[0, 0, 0.265], rgba=(0.45, 0.46, 0.52, 1), col=VISUAL)
    b.geom(body, CYLINDER, [0.028, 0.06], pos=[0, 0, 0.29], rgba=DARK, col=VISUAL, name=f"{name}_barrel")
    return dict(body=name, barrel=f"{name}_barrel", muzzle_height=0.29)


def launcher_muzzle(base_pos, yaw, pitch, height=0.29, barrel_len=0.12):
    d = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    return np.asarray(base_pos, float) + [0, 0, height] + d * barrel_len, d


# ----------------------------------------------------------------------------- hanging deflector
def add_deflector(b, name="deflector", radius=0.05, rgba=BLUE):
    """Blue sphere hanging from a cable (Figs. 5g, 6a); the ball bounces off it."""
    body = b.body(name=name, mocap=True, pos=[0, 5, 3])
    b.sphere(body, radius, rgba=rgba, name=f"{name}_geom", specular=0.6, shininess=0.6,
             friction=[0.05, 0.005, 0.0001], **BOUNCY)
    b.cylinder(body, 0.004, 2.0, pos=[0, 0, radius + 2.0], rgba=DARK, col=VISUAL)
    return dict(body=name, geom=f"{name}_geom", radius=radius)


# ----------------------------------------------------------------------------- occluder
def add_occluder(b, name="occluder"):
    """Tall wooden board on two legs (Figs. 5f, 6b).  Purely visual -- it is placed between
    the camera and the ball's flight path.  Size and pose are set at reset."""
    body = b.body(name=name, mocap=True, pos=[0, 6, 0])
    b.box(body, [0.01, 0.2, 0.5], pos=[0, 0, 1.0], rgba=WOOD, col=VISUAL, name=f"{name}_board", specular=0.1)
    for i, s in enumerate((-1, 1)):
        b.box(body, [0.014, 0.012, 0.5], pos=[0, s * 0.2, 1.0], rgba=(0.18, 0.13, 0.1, 1), col=VISUAL,
              name=f"{name}_side{i}")
    for i, s in enumerate((-1, 1)):
        b.box(body, [0.014, 0.2, 0.012], pos=[0, 0, 1.0 + s * 0.5], rgba=(0.18, 0.13, 0.1, 1), col=VISUAL,
              name=f"{name}_cap{i}")
    for i, s in enumerate((-1, 1)):
        b.box(body, [0.012, 0.012, 0.25], pos=[0, s * 0.2, 0.25], rgba=(0.18, 0.13, 0.1, 1), col=VISUAL,
              name=f"{name}_leg{i}")
    return dict(body=name)


def set_occluder(model, name, width, z_bottom, z_top):
    """Resize the occluder board so it spans `width` and [z_bottom, z_top] (board-local y/z)."""
    h, w = (z_top - z_bottom) / 2, width / 2
    zc = (z_top + z_bottom) / 2
    set_geom_box(model, f"{name}_board", [0.01, w, h], [0, 0, zc])
    for i, s in enumerate((-1, 1)):
        set_geom_box(model, f"{name}_side{i}", [0.014, 0.012, h + 0.012], [0, s * w, zc])
        set_geom_box(model, f"{name}_cap{i}", [0.014, w + 0.012, 0.012], [0, 0, zc + s * h])
        set_geom_box(model, f"{name}_leg{i}", [0.012, 0.012, z_bottom / 2], [0, s * w, z_bottom / 2])


# ----------------------------------------------------------------------------- plinko
def add_plinko(b, name="plinko", width=0.42, height=0.52, gap=0.05, rows=5, peg_dx=0.07, peg_r=0.009):
    """Vertical peg board with side walls, a transparent front pane and a dispenser on top
    (Fig. 6c).  Local frame: board plane = y-z, ball travels between x in [-gap/2, gap/2],
    bottom exit at z = 0."""
    body = b.body(name=name)
    grey = (0.37, 0.37, 0.42, 1)
    b.box(body, [0.008, width / 2, height / 2], pos=[-gap / 2 - 0.008, 0, height / 2], rgba=grey,
          friction=[0.3, 0.005, 0.0001], **BOUNCY)
    b.box(body, [0.002, width / 2, height / 2], pos=[gap / 2 + 0.002, 0, height / 2], rgba=HIDDEN,
          group=GROUP_HIDDEN, friction=[0.3, 0.005, 0.0001])
    for s in (-1, 1):
        b.box(body, [gap / 2 + 0.03, 0.008, height / 2], pos=[0, s * (width / 2 + 0.008), height / 2],
              rgba=grey, **BOUNCY)
    pegs = []
    zs = np.linspace(height - 0.08, 0.10, rows)
    for r, z in enumerate(zs):
        offset = 0 if r % 2 == 0 else peg_dx / 2
        ys = np.arange(-width / 2 + 0.035 + offset, width / 2 - 0.02, peg_dx)
        for y in ys:
            b.geom(body, CYLINDER, [peg_r, gap / 2 + 0.02], pos=[0.01, y, z],
                       quat=quat_axis_angle([0, 1, 0], np.pi / 2), rgba=DARK, specular=0.5, **BOUNCY)
            pegs.append((y, z))
    # Dispenser above the board.
    d = b.body(body, name=f"{name}_dispenser", pos=[0, 0, height + 0.16])
    b.cylinder(d, 0.03, 0.035, pos=[0, 0, 0.0], rgba=(0.45, 0.46, 0.5, 1), col=VISUAL)
    b.cylinder(d, 0.031, 0.012, pos=[0, 0, -0.005], rgba=BALL_COLORS["orange"], col=VISUAL)
    return dict(body=name, dispenser=f"{name}_dispenser", width=width, height=height, gap=gap, pegs=pegs)


# ----------------------------------------------------------------------------- air hockey
def add_air_hockey_table(b, name="ah", x0=0.18, length=2.0, width=1.1, top_z=0.75):
    """Air-hockey table (Fig. 5a): frictionless white surface, dark rails, red centre line,
    blue goal line at the robot end.  The table spans x in [x0, x0+length]."""
    body = b.body(name=name, pos=[x0 + length / 2, 0, 0])
    L, W = length, width
    b.box(body, [L / 2, W / 2, 0.02], pos=[0, 0, top_z - 0.02], rgba=(0.97, 0.97, 0.97, 1),
          name=f"{name}_surface", friction=[0.02, 0.0, 0.0], specular=0.4, shininess=0.6)
    b.box(body, [0.004, W / 2 - 0.01, 0.0004], pos=[0, 0, top_z], rgba=(0.85, 0.15, 0.12, 1), col=VISUAL)
    b.box(body, [0.012, W / 2 - 0.01, 0.0004], pos=[-L / 2 + 0.12, 0, top_z], rgba=BLUE, col=VISUAL)
    rail = (0.13, 0.13, 0.2, 1)
    t, h = 0.04, 0.035
    for s in (-1, 1):
        b.box(body, [L / 2 + t, t / 2, h / 2 + 0.02], pos=[0, s * (W / 2 + t / 2), top_z + h / 2 - 0.02],
              rgba=rail, col=RAIL, solref=[0.03, 0.1], solmix=100.0, friction=[0.0, 0, 0])
        b.box(body, [t / 2, W / 2, h / 2 + 0.02], pos=[s * (L / 2 + t / 2), 0, top_z + h / 2 - 0.02],
              rgba=rail, col=RAIL, solref=[0.03, 0.1], solmix=100.0, friction=[0.0, 0, 0])
    for dx in (-1, 1):
        for dy in (-1, 1):
            b.cylinder(body, 0.03, (top_z - 0.04) / 2, pos=[dx * (L / 2 - 0.1), dy * (W / 2 - 0.08),
                                                            (top_z - 0.04) / 2], rgba=LEG_GREY, col=VISUAL)
    return dict(body=name, surface=f"{name}_surface", x0=x0, length=L, width=W, top_z=top_z)


def add_puck(b, name="puck", top_z=0.75, radius=0.038, half_h=0.007):
    body = b.body(name=name, pos=[1.5, 0, top_z + half_h + 0.0005])
    body.add_joint(name=f"{name}_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    body.add_joint(name=f"{name}_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    body.add_joint(name=f"{name}_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
    b.cylinder(body, radius, half_h, rgba=(0.06, 0.06, 0.07, 1), col=PUCK, name=f"{name}_geom", mass=0.03,
               friction=[0.0, 0.0, 0.0], specular=0.6, shininess=0.7)
    return dict(body=name, geom=f"{name}_geom", joints=[f"{name}_x", f"{name}_y", f"{name}_yaw"],
                radius=radius)


# ----------------------------------------------------------------------------- containers
def add_open_box(b, name, half=(0.085, 0.085, 0.05), wall=0.006, rgba=CARDBOARD, flaps=True, free=True,
                 mass=0.3):
    """Open-top box (cardboard or blue bin).  Body origin = bottom centre."""
    body = b.body(name=name)
    if free:
        body.add_freejoint(name=f"{name}_joint")
    hx, hy, hz = half
    kw = dict(rgba=rgba, col=OBJECT, specular=0.05, friction=[0.8, 0.005, 0.0001])
    m = mass / 5
    b.box(body, [hx, hy, wall / 2], pos=[0, 0, wall / 2], name=f"{name}_bottom", mass=m, **kw)
    b.box(body, [wall / 2, hy, hz], pos=[hx - wall / 2, 0, hz], mass=m, **kw)
    b.box(body, [wall / 2, hy, hz], pos=[-hx + wall / 2, 0, hz], mass=m, **kw)
    b.box(body, [hx, wall / 2, hz], pos=[0, hy - wall / 2, hz], mass=m, **kw)
    b.box(body, [hx, wall / 2, hz], pos=[0, -hy + wall / 2, hz], mass=m, **kw)
    if flaps:
        flap = tuple(np.clip(np.array(rgba[:3]) * 1.08, 0, 1)) + (1,)
        for s in (-1, 1):
            q = quat_axis_angle([0, 1, 0], s * 0.5)
            b.box(body, [0.03, hy * 0.95, wall / 2], pos=[s * (hx + 0.025), 0, 2 * hz + 0.012], quat=q, rgba=flap,
                  col=VISUAL)
    return dict(body=name, half=half)


# ----------------------------------------------------------------------------- objects
def add_ball(b, name, radius=0.02, rgba=STEEL, mass=0.05, free=True, bouncy=False):
    body = b.body(name=name, pos=[0, 0, -5])
    if free:
        body.add_freejoint(name=f"{name}_joint")
    kw = dict(friction=[1.0, 0.005, 0.0001])
    b.sphere(body, radius, rgba=rgba, col=OBJECT, name=f"{name}_geom", mass=mass, specular=0.6,
             shininess=0.7, **kw)
    return dict(body=name, geom=f"{name}_geom", radius=radius)


def _mesh_object(b, body, key, mass):
    """Textured mesh object.  Convex-ish objects collide with the mesh's convex hull; strongly
    non-convex ones (banana) collide with a capsule chain stored in grasp.json."""
    obj = paths.OBJECTS / key / f"{key}.obj"
    png = paths.OBJECTS / key / f"{key}.png"
    b.mesh(f"mesh_{key}", obj)
    mat = b.textured_material(f"tex_{key}", png)
    caps = _grasp_meta(key).get("capsules")
    if not caps:
        b.geom(body, MESH, [0], meshname=f"mesh_{key}", material=mat, col=OBJECT, name=f"{body.name}_geom",
               mass=mass, friction=[1.2, 0.01, 0.0001], condim=4)
        return
    b.geom(body, MESH, [0], meshname=f"mesh_{key}", material=mat, col=VISUAL, mass=0.0)
    for i, (p0, p1, r) in enumerate(caps):
        b.segment(body, p0, p1, r, rgba=HIDDEN, group=GROUP_HIDDEN, col=OBJECT,
                  name=f"{body.name}_geom" if i == 0 else None, mass=mass / len(caps),
                  friction=[1.2, 0.01, 0.0001], condim=4)


def _cup(b, body, rgba=CUP_RED):
    b.cylinder(body, 0.03, 0.05, pos=[0, 0, 0.05], rgba=rgba, col=OBJECT, name=f"{body.name}_geom",
               mass=0.12, friction=[0.4, 0.005, 0.0001], condim=4, specular=0.2)


def _can(b, body):
    b.cylinder(body, 0.032, 0.058, pos=[0, 0, 0.06], rgba=(0.80, 0.08, 0.07, 1), col=OBJECT,
               name=f"{body.name}_geom", mass=0.15, friction=[1.0, 0.005, 0.0001], condim=4, specular=0.6)
    for z in (0.002, 0.118):
        b.cylinder(body, 0.0322, 0.002, pos=[0, 0, z], rgba=(0.8, 0.8, 0.82, 1), col=VISUAL, specular=0.8)


def _bottle(b, body):
    b.cylinder(body, 0.029, 0.055, pos=[0, 0, 0.055], rgba=(0.95, 0.95, 0.95, 1), col=OBJECT,
               name=f"{body.name}_geom", mass=0.12, friction=[1.0, 0.005, 0.0001], condim=4)
    b.cylinder(body, 0.016, 0.015, pos=[0, 0, 0.122], rgba=(0.95, 0.95, 0.95, 1), col=OBJECT, mass=0.01)
    b.cylinder(body, 0.017, 0.01, pos=[0, 0, 0.145], rgba=BLUE, col=OBJECT, mass=0.005)


def _mug(b, body):
    b.cylinder(body, 0.034, 0.045, pos=[0, 0, 0.045], rgba=(0.93, 0.95, 0.98, 1), col=OBJECT,
               name=f"{body.name}_geom", mass=0.18, friction=[1.0, 0.005, 0.0001], condim=4)
    b.cylinder(body, 0.0345, 0.012, pos=[0, 0, 0.06], rgba=(0.25, 0.55, 0.95, 1), col=VISUAL)
    b.segment(body, [0.03, 0, 0.075], [0.055, 0, 0.065], 0.006, rgba=(0.93, 0.95, 0.98, 1), col=OBJECT, mass=0.005)
    b.segment(body, [0.055, 0, 0.065], [0.055, 0, 0.03], 0.006, rgba=(0.93, 0.95, 0.98, 1), col=OBJECT, mass=0.005)
    b.segment(body, [0.055, 0, 0.03], [0.03, 0, 0.022], 0.006, rgba=(0.93, 0.95, 0.98, 1), col=OBJECT, mass=0.005)


PALM_CLEARANCE = 0.032   # the Panda palm sits ~3.7 cm above the TCP: grasp at least this far below the top


def _grasp_meta(key):
    import json
    with open(paths.OBJECTS / key / "grasp.json") as fh:
        return json.load(fh)


class ObjectSpec:
    """Builder + metadata for a graspable object.  Body origin = bottom centre.
    `close_axis` ('x' / 'y' / None) is the object-frame axis the fingers should close along;
    `grasp_offset` is the (x, y) grasp point in the object frame."""

    def __init__(self, build, desc, height, grasp_z=None, close_axis=None, grasp_offset=(0.0, 0.0)):
        self.build, self.desc, self.height = build, desc, height
        self.grasp_z = grasp_z if grasp_z is not None else max(0.5 * height, height - PALM_CLEARANCE)
        self.close_axis, self.grasp_offset = close_axis, tuple(grasp_offset)


def _mesh_spec(key, desc, mass):
    meta = _grasp_meta(key)
    return ObjectSpec(lambda b, body: _mesh_object(b, body, key, mass), desc, meta["height"],
                      close_axis=meta["close_axis"], grasp_offset=meta["grasp_offset"])


OBJECT_LIBRARY = {
    "cup": ObjectSpec(_cup, "red cup", 0.10, grasp_z=0.072),
    "duck": _mesh_spec("duck", "yellow duck", 0.06),
    "apple": _mesh_spec("apple", "red apple", 0.12),
    "lemon": _mesh_spec("lemon", "lemon", 0.08),
    "lime": _mesh_spec("lime", "green lime", 0.07),
    "banana": _mesh_spec("banana", "banana", 0.10),
    "can": ObjectSpec(_can, "red soda can", 0.12),
    "bottle": ObjectSpec(_bottle, "white bottle", 0.155),
    "mug": ObjectSpec(_mug, "blue mug", 0.09, close_axis="y"),
}


def add_object(b, name, kind, free=True):
    body = b.body(name=name, pos=[0, 0, -5])
    if free:
        body.add_freejoint(name=f"{name}_joint")
    spec = OBJECT_LIBRARY[kind]
    spec.build(b, body)
    return dict(body=name, kind=kind, desc=spec.desc, height=spec.height, grasp_z=spec.grasp_z)


# ----------------------------------------------------------------------------- runtime helpers
def set_geom_box(model, name, half, pos=None):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    model.geom_size[gid] = half
    model.geom_rbound[gid] = float(np.linalg.norm(half))
    model.geom_aabb[gid] = [0, 0, 0, *half]
    if pos is not None:
        model.geom_pos[gid] = pos
