"""Requires difference trimesh fork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow_datasets as tfds
import shape_tfds as sds
from shape_tfds.shape.shapenet.core import base
from shape_tfds.core.features import BinaryVoxel
from shape_tfds.shape.shapenet.core import views
from shape_tfds.shape.shapenet.core import voxel as voxel_lib
import trimesh

trimesh.util.log.setLevel('ERROR')


# def frustum_matrix(near, far, left, right, bottom, top):
#     # http://www.songho.ca/opengl/gl_projectionmatrix.html
#     return np.array([
#         [2 * near / (right - left), 0, (right + left) / (right - left), 0],
#         [0, 2*near / (top - bottom), (top + bottom) / (top - bottom), 0],
#         [0, 0, (far + near) / (near - far), -near * far / (far - near)],
#         [0, 0, -1, 0],
#     ])


# def symmetric_frustum_matrix(near, far, width, height):
#     dx = width / 2
#     dy = height / 2
#     return frustum_matrix(near, far, -dx, dx, -dy, dy)


def _get_voxel_transform(resolution):
    diag = 1 / (resolution - 1)
    return np.array([
        [diag, 0, 0, -0.5],
        [0, diag, 0, -0.5],
        [0, 0, diag, -0.5],
        [0, 0, 0, 1]
    ])



class ShapenetCoreFrustumVoxelConfig(base.ShapenetCoreConfig):
    def __init__(self, synset_id, resolution=64, seed=0):
        self._seed = seed
        self._resolution = resolution
        super(ShapenetCoreFrustumVoxelConfig, self).__init__(
            synset_id=synset_id,
            name='frustum_voxels-%d-%s-%d' % (resolution, synset_id, seed),
            description='shapenet core frustum voxels',
            version=tfds.core.Version("0.0.1"))

    @property
    def features(self):
        return dict(voxels=BinaryVoxel((self._resolution,)*3))

    @property
    def resolution(self):
        return self._resolution

    def loader_context(self, dl_manager=None):
        view_fn = views.random_view_fn(seed_offset=self._seed)
        transform = _get_voxel_transform(self._resolution)

        def item_map_fn(key, data):
            voxels = trimesh.voxel.VoxelGrid(
                data['voxels'], transform=transform)
            scene = trimesh.primitives.Sphere().scene()
            return dict(voxels=transform_voxels(
                scene, voxels,
                self._resolution, **view_fn(key)))

        return base.get_data_mapping_context(
            config=voxel_lib.ShapenetCoreVoxelConfig(
                synset_id=self.synset_id, resolution=self.resolution),
            dl_manager=dl_manager, item_map_fn=item_map_fn)

        # return base.mesh_loader_context(
        #     synset_id=self.synset_id, dl_manager=dl_manager,
        #     item_map_fn=lambda key, scene: dict(
        #         voxels=load_frustum_voxels_dense(
        #             scene, self._resolution, **view_fn(key))))


def transform_voxels(scene, voxels, resolution, position, focal):
    views.set_scene_view(scene, (resolution,)*2, position, focal)
    dist = np.linalg.norm(position)
    origin, rays = scene.camera_rays()
    rays = rays.reshape((resolution, resolution, 3))
    z = np.linspace(dist - 0.5, dist + 0.5, resolution)
    coords = np.expand_dims(rays, axis=-2) * np.expand_dims(z, axis=-1)
    coords += origin

    frust_vox_dense = voxels.is_filled(coords)
    frust_vox_dense = frust_vox_dense.transpose((1, 0, 2))  # y, x, z  # pylint: disable=no-member
    return frust_vox_dense


def load_frustum_voxels_dense(scene, resolution, position, focal):
    vox = voxel_lib.scene_to_voxels(scene, resolution)
    return transform_voxels(scene, vox, resolution, position, focal)
