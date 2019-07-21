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
from shape_tfds.core.resolver import ZipSubdirResolver
from shape_tfds.core.features import BinaryVoxel
from shape_tfds.shape.shapenet.core import views
from shape_tfds.shape.shapenet.core.voxel import load_voxels
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


class ShapenetCoreFrustumVoxelConfig(base.ShapenetCoreConfig):
    def __init__(self, name, synset_id, view_fn, resolution=64):
        self._view_fn = view_fn
        self._synset_id = synset_id
        self._resolution = resolution
        super(ShapenetCoreFrustumVoxelConfig, self).__init__(
            name=name,
            description='shapenet core frustum voxels',
            version=tfds.core.Version("0.0.1"))

    @property
    def synset_id(self):
        return self._synset_id

    def features(self):
        return dict(voxels=BinaryVoxel((self._resolution,)*3))

    def loader(self, dl_manager=None):
        return base.mesh_loader_context(
            synset_id=self.synset_id, dl_manager=dl_manager,
            item_map_fn=lambda key, scene: dict(
                voxels=load_frustum_voxels_dense(
                    scene, self._resolution, **self._view_fn(key))))


def load_frustum_voxels_dense(scene, resolution, position, focal):
    vox = load_voxels(scene, resolution)
    views.set_scene_view(scene, (resolution,)*2, position, focal)

    dist = np.linalg.norm(position)

    origin, rays = scene.camera_rays()
    rays = rays.reshape((resolution, resolution, 3))
    z = np.linspace(dist - 0.5, dist + 0.5, resolution)
    coords = np.expand_dims(rays, axis=-2) * np.expand_dims(z, axis=-1)
    coords += origin

    frust_vox_dense = vox.is_filled(coords)
    frust_vox_dense = frust_vox_dense.transpose((1, 0, 2))  # y, x, z  # pylint: disable=no-member
    return frust_vox_dense


if __name__ == '__main__':
    ids, names = base.load_synset_ids()

    synset_name = 'suitcase'
    # name = 'watercraft'
    # name = 'aeroplane'
    # name = 'table'
    # name = 'rifle'


    seed_offset = 0
    synset_id = ids[synset_name]
    resolution = 64

    config = ShapenetCoreFrustumVoxelConfig(
        name='frustum_voxels-%s-%03d-%03d' % (
            synset_id, resolution, seed_offset),
        synset_id=synset_id,
        view_fn=views.random_view_fn(seed_offset),
        resolution=resolution)

    builder = base.ShapenetCore(config=config)
    builder.download_and_prepare()
