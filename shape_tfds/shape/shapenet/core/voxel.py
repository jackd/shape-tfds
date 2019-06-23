"""Requires difference trimesh fork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet import core as c
from shape_tfds.shape.resolver import ZipSubdirResolver
from shape_tfds.core.features import PaddedTensor
from shape_tfds.core.features import BinaryRunLengthEncodedFeature

class ShapenetCoreVoxelConfig(c.ShapenetCoreConfig):
    def __init__(self, synset_id, resolution=128):
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


class VoxelLoader(c.ExampleLoader):
    def __init__(self, archive, resolution):
        self._resolution = resolution
        super(VoxelLoader, self).__init__(archive=archive)

    def __call__(self, model_path, model_id):
        # trimesh = tfds.core.lazy_imports.trimesh
        import trimesh
        model_dir, filename = os.path.split(model_path)
        resolver = ZipSubdirResolver(self.archive, model_dir)
        file_obj = resolver.get(filename)
        scene = trimesh.load(file_obj, file_type='obj', resolver=resolver)
        camera = scene.camera
        camera.resolution = 128, 128
        scene.show()
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene.geometry.values()])
        extents = max(mesh.extents)
        vox = mesh.voxelized(pitch=1./self._resolution, method='binvox')
        vox.fill(method='orthographic')
        origin, rays, angles = camera.to_rays()
        z = np.linspace(0.5, 2, 128)
        print(rays.shape)
        exit()

        # print(extents)
        # vox.show()


if __name__ == '__main__':
    ids, names = c.load_synset_ids()

    # name = 'suitcase'
    name = 'watercraft'
    # name = 'aeroplane'

    config = ShapenetCoreVoxelConfig(synset_id=ids[name], resolution=128)
    builder = c.ShapenetCore(config=config)
    builder.download_and_prepare()
