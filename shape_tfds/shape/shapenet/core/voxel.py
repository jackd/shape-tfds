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
from shape_tfds.shape.shapenet.core.views import fix_axes
from shape_tfds.core.features import PaddedTensor
from shape_tfds.core.features import BinaryRunLengthEncodedFeature
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


class ShapenetCoreVoxelConfig(base.ShapenetCoreConfig):
    def __init__(self, synset_id, resolution=64):
        self._synset_id = synset_id
        self._resolution = resolution
        super(ShapenetCoreVoxelConfig, self).__init__(
            name='voxel-%s-%d' % (synset_id, resolution),
            description='shapenet core voxels',
            version=tfds.core.Version("0.0.1"))
    
    @property
    def synset_id(self):
        return self._synset_id

    def features(self):
        return tfds.core.features.FeaturesDict(dict(
            voxels=PaddedTensor(
                BinaryRunLengthEncodedFeature((self._resolution,)*3))))
    
    def loader(self, archive):
        return VoxelLoader(archive, self._resolution)


class VoxelLoader(base.ExampleLoader):
    def __init__(self, archive, resolution):
        self._resolution = resolution
        super(VoxelLoader, self).__init__(archive=archive)

    def __call__(self, model_path, model_id):
        from shape_tfds.shape.shapenet.core import views
        from shape_tfds.shape.transformations import look_at
        import matplotlib.pyplot as plt
        model_dir, filename = os.path.split(model_path)
        resolver = ZipSubdirResolver(self.archive, model_dir)
        file_obj = trimesh.util.wrap_as_stream(resolver.get(filename))
        scene = trimesh.load(file_obj, file_type='obj', resolver=resolver)
        if hasattr(scene, 'geometry'):
            mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene.geometry.values()])
        else:
            mesh = scene
        fix_axes(mesh)
        vox = mesh.voxelized(
            pitch=1. / (self._resolution + 1), method='binvox',
            bounding_box=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], exact=True)
        assert(vox.shape == (self._resolution,) * 3)
        stripped, padding = vox.encoding.stripped
        vox = trimesh.voxel.VoxelGrid(stripped)
        vox.fill(method='orthographic')
        return dict(voxels=(vox.encoding.dense, padding))

if __name__ == '__main__':
    ids, names = base.load_synset_ids()

    # name = 'suitcase'
    # name = 'watercraft'
    # name = 'aeroplane'
    # name = 'table'
    name = 'rifle'

    config = ShapenetCoreVoxelConfig(synset_id=ids[name], resolution=32)
    builder = base.ShapenetCore(config=config)
    builder.download_and_prepare()
