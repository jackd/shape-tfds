"""Requires difference trimesh fork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet.core import base 
from shape_tfds.shape.resolver import ZipSubdirResolver
from shape_tfds.core.features import PaddedTensor
from shape_tfds.core.features import BinaryRunLengthEncodedFeature
from shape_tfds.shape.shapenet.core.views import SceneMutator
from shape_tfds.shape.shapenet.core.views import fix_axes
import trimesh

trimesh.util.log.setLevel('ERROR')


def frustum_matrix(near, far, left, right, bottom, top):
    # http://www.songho.ca/opengl/gl_projectionmatrix.html
    return np.array([
        [2 * near / (right - left), 0, (right + left) / (right - left), 0],
        [0, 2*near / (top - bottom), (top + bottom) / (top - bottom), 0],
        [0, 0, (far + near) / (near - far), -near * far / (far - near)],
        [0, 0, -1, 0],
    ])


def symmetric_frustum_matrix(near, far, width, height):
    dx = width / 2
    dy = height / 2
    return frustum_matrix(near, far, -dx, dx, -dy, dy)


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
        import shape_tfds as sds
        return tfds.core.features.FeaturesDict(dict(
            voxels=sds.core.features.PaddedTensor(
                sds.core.features.ShapedTensor(
                    sds.core.features.BinaryRunLengthEncodedFeature(),
                    (None,)*3),
                padded_shape=(self._resolution,)*3),
        example_id=tfds.core.features.Text()))
    
    def loader(self, archive):
        return VoxelLoader(archive, self._resolution, self._scene_mutator)


class VoxelLoader(base.ExampleLoader):
    def __init__(self, archive, resolution, scene_mutator):
        self._resolution = resolution
        self._scene_mutator = scene_mutator
        super(VoxelLoader, self).__init__(archive=archive)

    def __call__(self, model_path, model_id):
        from shape_tfds.shape.shapenet.core import views
        from shape_tfds.shape.transformations import look_at
        import matplotlib.pyplot as plt
        model_dir, filename = os.path.split(model_path)
        resolver = ZipSubdirResolver(self.archive, model_dir)
        scene = trimesh.load(
            trimesh.util.wrap_as_stream(resolver.get(filename)),
            file_type='obj', resolver=resolver)
        if hasattr(scene, 'geometry'):
            mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene.geometry.values()])
        else:
            mesh = scene

        fix_axes(mesh)
        scene = mesh.scene()
        position, _ = self._scene_mutator(scene, (self._resolution,)*2)
        dist = np.linalg.norm(position)

        camera = scene.camera
        
        vox = mesh.voxelized(
            pitch=1./self._resolution, method='binvox',
            bounding_box=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], exact=True)
        vox.fill(method='orthographic')
        origin, rays = camera.to_rays(scene.camera_transform)
        rays = rays.reshape((self._resolution, self._resolution, 3))
        z = np.linspace(dist - 0.5, dist + 0.5, self._resolution)
        coords = np.expand_dims(rays, axis=-2) * np.expand_dims(z, axis=-1)
        coords += origin

        frust_vox_dense = vox.is_filled(coords)
        frust_vox_dense = frust_vox_dense.transpose((1, 0, 2))  # y, x, z
        frust_vox = trimesh.voxel.VoxelGrid(frust_vox_dense).encoding
        stripped, padding = frust_vox.stripped

        return dict(
            voxels=dict(stripped=stripped.dense, padding=padding),
            example_id=model_id,
        )


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
