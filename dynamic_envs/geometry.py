"""Small rotation / quaternion helpers (MuJoCo convention: quat = [w, x, y, z])."""
import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def quat_axis_angle(axis, angle):
    axis = normalize(axis)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q):
    w, x, y, z = normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R):
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = normalize(q)
    return q if q[0] >= 0 else -q


def euler_to_quat(roll, pitch, yaw):
    """Extrinsic XYZ (= intrinsic ZYX) Euler angles to quaternion."""
    qx = quat_axis_angle([1, 0, 0], roll)
    qy = quat_axis_angle([0, 1, 0], pitch)
    qz = quat_axis_angle([0, 0, 1], yaw)
    return quat_mul(qz, quat_mul(qy, qx))


def quat_error(target, current):
    """Rotation vector (world frame) that rotates `current` onto `target`."""
    dq = quat_mul(target, quat_conj(current))
    if dq[0] < 0:
        dq = -dq
    s = np.linalg.norm(dq[1:])
    if s < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, dq[0])
    return dq[1:] / s * angle


def rotvec_to_quat(rv):
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return np.array([1.0, 0, 0, 0])
    return quat_axis_angle(rv / angle, angle)


def quat_from_z(z_axis):
    """Quaternion whose local +Z axis points along `z_axis` (for capsules/cylinders)."""
    z = normalize(z_axis)
    ref = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(ref, z))
    if c > 1 - 1e-9:
        return np.array([1.0, 0, 0, 0])
    if c < -1 + 1e-9:
        return np.array([0.0, 1, 0, 0])
    axis = np.cross(ref, z)
    return quat_axis_angle(axis, np.arccos(np.clip(c, -1, 1)))


def lookat_quat(cam_pos, target, up=(0, 0, 1)):
    """Quaternion of a MuJoCo camera at `cam_pos` looking at `target` (camera looks along -Z, Y up)."""
    f = normalize(np.asarray(target, float) - np.asarray(cam_pos, float))
    x = normalize(np.cross(f, up))
    y = np.cross(x, f)
    R = np.column_stack([x, y, -f])
    return mat_to_quat(R)


def spherical_camera(lookat, distance, azimuth_deg, elevation_deg):
    """Camera position on a sphere around `lookat` (MuJoCo free-camera convention)."""
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = np.asarray(lookat, float) - distance * forward
    return pos, lookat_quat(pos, lookat)


TOP_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # gripper z-axis pointing at the floor, fingers open along world y


def top_down_quat(yaw=0.0):
    return quat_mul(quat_axis_angle([0, 0, 1], yaw), TOP_DOWN)
