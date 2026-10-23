"""Camera control for Hy-Nomas: WASD/mouse input -> per-frame poses -> the DiT's camera condition.

Follows the reference demo (apps/frontend CameraPathDrawer.tsx and
hymm/models/camera/geometry/demo_precomputed.py ``prepare_turn``):

* controls are integrated at 24 fps in Three.js world axes (Y up, camera looks down -Z), with
  yaw/pitch as an Euler 'YXZ' pair: W/S move along the view direction (pitch included), A/D
  strafe horizontally, the arrows turn at 20 deg/s, a mouse drag turns 0.0024 rad/px;
* poses become OpenCV camera-to-world matrices (``R_cv = R_three @ diag(1,-1,-1)``);
* per turn: relative to the first frame, resampled to ``4k+1`` frames, framewise deltas,
  translations divided by their 99th-percentile step (so only direction and relative speed
  survive), and each step's rotation scaled by 2.5.

The result, ``[4L-3, 4, 4]`` for a turn of L latent frames, is packed into the 64-d camera
tokens by the DiT's own camera MLP (``NomasDiT.camera_tokens``).
"""

import math

import numpy as np

FPS = 24
MOVE_SPEED = 2.8  # units / s (absolute speed is normalised away; only relative speed matters)
TURN_SPEED = math.radians(20.0)  # rad / s, arrow keys
MOUSE_SPEED = 0.0024  # rad / px
ROTATION_GAIN = 2.5
KEYS = ("w", "a", "s", "d", "up", "down", "left", "right")


# -- small SO(3) helpers (no scipy) -----------------------------------------------------------

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def log_so3(r):
    """Rotation matrix -> rotation vector."""
    cos = min(max((np.trace(r) - 1) / 2, -1.0), 1.0)
    angle = math.acos(cos)
    if angle < 1e-8:
        return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2
    if math.pi - angle < 1e-6:
        # near pi: axis from the symmetric part
        m = (r + np.eye(3)) / 2
        axis = np.sqrt(np.clip(np.diag(m), 0, None))
        k = int(np.argmax(axis))
        axis = m[k] / axis[k]
        return axis / np.linalg.norm(axis) * angle
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return w * angle / (2 * math.sin(angle))


def exp_so3(v):
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def slerp_matrices(rots, src, dst):
    """Piecewise SLERP of rotation matrices sampled at ``src`` times, evaluated at ``dst``."""
    out = []
    for t in dst:
        i = min(max(int(np.searchsorted(src, t, side="right")) - 1, 0), len(src) - 2)
        u = 0.0 if src[i + 1] == src[i] else (t - src[i]) / (src[i + 1] - src[i])
        delta = log_so3(rots[i].T @ rots[i + 1])
        out.append(rots[i] @ exp_so3(delta * u))
    return np.stack(out)


# -- controls -> poses ------------------------------------------------------------------------

class Pose:
    """A Three.js-style camera: position and yaw/pitch (Euler 'YXZ', no roll)."""

    def __init__(self, position=(0.0, 1.5, 10.0), yaw=0.0, pitch=0.0):
        self.position = np.array(position, dtype=np.float64)
        self.yaw, self.pitch = float(yaw), float(pitch)

    def copy(self):
        return Pose(self.position.copy(), self.yaw, self.pitch)

    def rotation_three(self):
        return rot_y(self.yaw) @ rot_x(self.pitch)

    def c2w_opencv(self):
        m = np.eye(4)
        m[:3, :3] = self.rotation_three() @ np.diag([1.0, -1.0, -1.0])
        m[:3, 3] = self.position
        return m

    def step(self, keys, mouse_dx=0.0, mouse_dy=0.0, dt=1.0 / FPS, speed=MOVE_SPEED):
        """One control tick, as the demo's animation loop applies it."""
        r = self.rotation_three()
        forward = r @ np.array([0.0, 0.0, -1.0])
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        n = np.linalg.norm(right)
        right = right / n if n > 1e-8 else r @ np.array([1.0, 0.0, 0.0])
        move = forward * (("w" in keys) - ("s" in keys)) + right * (("d" in keys) - ("a" in keys))
        self.position = self.position + move * speed * dt
        self.yaw += TURN_SPEED * dt * (("left" in keys) - ("right" in keys)) - mouse_dx * MOUSE_SPEED
        self.pitch += TURN_SPEED * dt * (("up" in keys) - ("down" in keys)) - mouse_dy * MOUSE_SPEED


def integrate(ticks, start=None):
    """``ticks``: one entry per 24 fps frame, ``(keys, mouse_dx, mouse_dy)``; frame 0 is the start
    pose and tick i moves from frame i to i+1. Returns ``(c2w [len(ticks)+1, 4, 4], end pose)``."""
    pose = (start or Pose()).copy()
    frames = [pose.c2w_opencv()]
    for keys, mdx, mdy in ticks:
        pose.step(set(keys), mdx, mdy)
        frames.append(pose.c2w_opencv())
    return np.stack(frames), pose


def parse_script(script):
    """A camera script -> ticks. One instruction per line or comma: ``<keys> <seconds>``, keys
    joined with ``+`` (w a s d up down left right, or ``wait``), e.g. ``w 2``, ``w+left 1.5``,
    ``right 0.5``, ``wait 1``. ``look <dx> <dy> <seconds>`` spreads a mouse drag (pixels)."""
    ticks = []
    for raw in script.replace(",", "\n").splitlines():
        parts = raw.strip().lower().split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "look":
            dx, dy, sec = float(parts[1]), float(parts[2]), float(parts[3]) if len(parts) > 3 else 1.0
            n = max(1, round(sec * FPS))
            ticks += [((), dx / n, dy / n)] * n
            continue
        keys = () if parts[0] in ("wait", "stay", "none") else tuple(k for k in parts[0].split("+"))
        bad = [k for k in keys if k not in KEYS]
        if bad:
            raise ValueError(f"unknown camera key(s) {bad}; use {', '.join(KEYS)} or wait")
        sec = float(parts[1]) if len(parts) > 1 else 1.0
        ticks += [(keys, 0.0, 0.0)] * max(1, round(sec * FPS))
    return ticks


# -- poses -> the model's camera condition -----------------------------------------------------

def resample(c2ws, count):
    n = len(c2ws)
    if n == count:
        return c2ws.copy()
    if n == 1:
        return np.repeat(c2ws, count, axis=0)
    src, dst = np.linspace(0, 1, n), np.linspace(0, 1, count)
    out = np.repeat(np.eye(4)[None], count, axis=0)
    out[:, :3, :3] = slerp_matrices(c2ws[:, :3, :3], src, dst)
    out[:, :3, 3] = np.stack([np.interp(dst, src, c2ws[:, a, 3]) for a in range(3)], axis=1)
    return out


def normalize(poses):
    """One segment: relative to frame 0, framewise deltas, p99 translation normalisation."""
    first = np.linalg.inv(poses[0])
    rel = first @ poses
    rel[0] = np.eye(4)
    out = np.repeat(np.eye(4)[None], len(poses), axis=0)
    for i in range(1, len(poses)):
        out[i] = np.linalg.inv(rel[i - 1]) @ rel[i]
    if len(poses) > 1:
        steps = np.linalg.norm(out[1:, :3, 3], axis=1)
        out[:, :3, 3] /= max(float(np.quantile(steps, 0.99)), 1e-6)
    return out


def scale_rotations(poses, gain=ROTATION_GAIN):
    out = poses.copy()
    for i in range(len(poses)):
        v = log_so3(poses[i, :3, :3]) * gain
        if np.linalg.norm(v) >= math.pi - 1e-7:
            raise ValueError("a camera step turns too fast (over 72 degrees per frame after the 2.5x gain)")
        out[i, :3, :3] = exp_so3(v)
    return out


def condition(c2ws, latent_frames, rotation_gain=ROTATION_GAIN):
    """Per-frame OpenCV c2w poses of one turn -> ``[4L-3, 4, 4]`` float32 for the DiT."""
    count = (latent_frames - 1) * 4 + 1
    poses = resample(np.asarray(c2ws, dtype=np.float64), count)
    poses = scale_rotations(normalize(poses), rotation_gain)
    return poses.astype(np.float32)
