from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from shape_tfds.shape import transformations
from collection_utils.mapping import Mapping
import functools
import collections
from shape_tfds.core import random as rand

# default field-of-view in blender
# consistent with R2n2
# looks decent
DEFAULT_FOV = np.degrees(2*np.arctan(32. / 35))


def polar_to_cartesian(dist, theta, phi):
    """
    Convert polar coordinates to cartesian.

    All inputs must be broadcastable to the same shape.

    Args:
        dist: float array, distance from origin (r)
        theta: float angle about z axis, radians
        phi: float angle away from z axis, radians.

    Returns:
        float array of x, y, z coordinates, with 1 additional dimension on the
        end compared to inputs
    """
    z = np.cos(phi)
    s = np.sin(phi)
    x = s * np.cos(theta)
    y = s * np.sin(theta)
    return np.stack((x, y, z), axis=-1) * np.expand_dims(dist, axis=-1)


def get_random(random, value_or_range, size=None):
    if isinstance(value_or_range, tuple):
        return random.uniform(*value_or_range, size=size)
    if size is None:
        return value_or_range
    return np.repeat(
        np.expand_dims(value_or_range, axis=0), size, axis=0)


def get_random_camera_position(
        random=np.random, dist=1.166, theta=(0., 360.),
        phi=(60., 65.), num_views=None):
    """
    Get randomly sampled camera positions.

    Args:
        random: np.random.RandomState instance, or np.random
        dist: float or 2-tuple of floats indicating a range of distances from
            origin
        theta: float or 2-tuple of floats indicating a range of theta values
        phi: float or 2-tuple of floats indicating a range of phi values
        num_views: number of positions ot sample

    Returns:
        [num_views, 3] float array of x, y, z coordinates,
        or [3] if num_views is None.
    """
    return polar_to_cartesian(
        get_random(random, dist, num_views),
        np.deg2rad(get_random(random, theta, num_views)),
        np.deg2rad(get_random(random, phi, num_views)),
    )


def random_view_fn(seed_offset=0, **kwargs):
    """
    Get a mapping from given keys to a dict with camera positions.

    The mapping is deterministic but random, with a new random state used with
    seed given by `(zlib.adler32(key) + seed_offset) % (2 ** 32)`.

    Args:
        keys: iterable of keys in the returned Mapping
        seed_offset: int, offset used in RandomState
        **kwargs: passed to get_random_views (except "random")
    """
    for key in ('random', 'num_views'):
        if key in kwargs:
            raise ValueError('Invalid kwarg key "%s"' % key)

    def get_views(key, num_views=None):
        return get_random_camera_position(
            random=rand.get_random_state(key, seed_offset),  # pylint: disable=no-member
            num_views=num_views,
            **kwargs)

    return get_views


# def get_scene(resolution, position, focal):
#     from trimesh import scene
#     camera = scene.cameras.Camera(
#         resolution=resolution, focal=focal*resolution)
#     return scene.Scene(
#         camera=camera, camera_transform=transformations.look_at(position))


def set_scene_view(scene, resolution, position, fov=(DEFAULT_FOV,)*2):
    resolution = np.array(resolution)
    camera = scene.camera
    camera.resolution = resolution
    scene.camera_transform = transformations.look_at(position)
    camera.fov = fov


_axes_fix_transform = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])


def fix_axes(geometry):
    if hasattr(geometry, 'geometry'):
        for geom in geometry.geometry.values():
            fix_axes(geom)
    else:
        geometry.apply_transform(_axes_fix_transform)
