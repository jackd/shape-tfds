"""Numpy implementations of basic transformations."""
import numpy as np


def normalized(x):
    return x / np.linalg.norm(x, keepdims=True)


def _look_at_nh_helper(eye, center=None, world_up=None, dtype=None, axis=-1):
    """Non-homogeneous eye-to-world coordinate transform."""
    # vector_degeneracy_cutoff = 1e-6
    if dtype is None:
        dtype = eye.dtype
    else:
        eye = np.asanyarray(eye, dtype=dtype)

    center = (
        np.zeros((3,), dtype=dtype)
        if center is None
        else np.asanyarray(center, dtype=dtype)
    )
    world_up = (
        np.array([0, 0, 1], dtype=dtype)
        if world_up is None
        else np.asanyarray(world_up, dtype=dtype)
    )

    # https://web.cs.wpi.edu/~emmanuel/courses/cs543/f13/slides/lecture04_p3.pdf
    n = normalized(eye - center)
    u = normalized(np.cross(world_up, n))
    v = normalized(np.cross(n, u))
    rotation = np.stack([u, v, n], axis=axis)

    translation = eye
    return rotation, translation


def inverse_look_at_nh(eye, center=None, world_up=None, dtype=None):
    R, t = _look_at_nh_helper(eye, center, world_up, dtype, axis=-2)
    t = np.matmul(R, -t)
    return R, t


def look_at_nh(eye, center=None, world_up=None, dtype=None):
    R, t = _look_at_nh_helper(eye, center, world_up, dtype, axis=-1)
    return R, t


def _from_nh(R, t):
    return np.concatenate(
        [np.concatenate([R, np.expand_dims(t, axis=-1)], axis=-1), [[0, 0, 0, 1]]],
        axis=0,
    )


def look_at(eye, center=None, world_up=None, dtype=None):
    return _from_nh(*look_at_nh(eye, center, world_up, dtype))


def inverse_look_at(eye, center=None, world_up=None, dtype=None):
    return _from_nh(*inverse_look_at_nh(eye, center, world_up, dtype))
