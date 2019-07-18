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
from shape_tfds.core.features import BinaryRunLengthEncodedFeature
from shape_tfds.shape.shapenet.core.views import SceneMutator
from shape_tfds.shape.shapenet.core.views import fix_axes
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
    def __init__(self, synset_id, resolution=64, scene_mutator=None):
        if scene_mutator is None:
            scene_mutator = SceneMutator()
        self._scene_mutator = scene_mutator
        self._synset_id = synset_id
        self._resolution = resolution
        super(ShapenetCoreFrustumVoxelConfig, self).__init__(
            name='frust_vox-%d-%s-%s' % (
                resolution, scene_mutator.name, synset_id),
            description='shapenet core voxels',
            version=tfds.core.Version("0.0.1"))

    @property
    def synset_id(self):
        return self._synset_id

    def features(self):
        return dict(
            voxels=sds.core.features.BinaryVoxel((self._resolution,)*3))

    def loader(self, dl_manager=None):
        return base.mesh_loader_context(
            synset_id=self.synset_id, dl_manager=dl_manager,
            map_fn=lambda scene: dict(
                voxels=load_frustum_voxels_dense(
                    scene, self._resolution, self._scene_mutator)))


def load_frustum_voxels_dense(scene, resolution, scene_mutator):
    vox = load_voxels(scene, resolution)
    position, _ = scene_mutator(scene, (resolution,)*2)
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


    seed = 0

    config = ShapenetCoreFrustumVoxelConfig(
        synset_id=ids[synset_name],
        scene_mutator=SceneMutator(name='base%03d' % seed, seed=seed),
        resolution=64)

    builder = base.ShapenetCore(config=config)
    builder.download_and_prepare()
