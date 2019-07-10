from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from shape_tfds.shape import transformations


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


def _get_random(random, value_or_range, size):
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
        _get_random(random, dist, num_views),
        np.deg2rad(_get_random(random, theta, num_views)),
        np.deg2rad(_get_random(random, phi, num_views)),
    )


class SceneMutator(object):
    def __init__(
            self, name='base', seed=152, focal=32./35, dist=1.166,
            theta=(0., 360.), phi=(60., 65.)):
        self._seed = seed
        self._focal = focal
        self._dist = dist
        self._theta = theta
        self._phi = phi
        self._name = name
        self.reset()
    
    def copy(self):
        return SceneMutator(
            name=self._name,
            seed=self._seed,
            focal=self._focal,
            dist=self._dist,
            theta=self._theta,
            phi=self._phi
        )

    @property
    def name(self):
        return self._name

    @property
    def seed(self):
        return self._seed

    @property
    def dist(self):
        return self._dist

    @property
    def theta(self):
        return self._theta

    @property
    def phi(self):
        return self._phi

    def reset(self):
        self._random = np.random.RandomState(self._seed)  # pylint: disable=no-member

    def __call__(self, scene, resolution):
        resolution = np.array(resolution)
        camera = scene.camera
        camera.resolution = resolution
        focal = _get_random(self._random, self._focal, size=None)
        position = get_random_camera_position(
            self._random, self._dist, self._theta, self._phi, num_views=None)
        scene.camera_transform = transformations.look_at(position)
        camera.focal = focal*resolution
        return position, focal


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
        