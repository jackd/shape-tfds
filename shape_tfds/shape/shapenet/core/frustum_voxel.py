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
from shape_tfds.shape.shapenet.core.views import CameraMutator
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


class ShapenetCoreFrustumVoxelConfig(c.ShapenetCoreConfig):
    def __init__(self, synset_id, resolution=64,  camera_mutator=None):
        if camera_mutator is None:
            camera_mutator = CameraMutator()
        self._camera_mutator = camera_mutator
        self._synset_id = synset_id
        self._resolution = resolution
        super(ShapenetCoreFrustumVoxelConfig, self).__init__(
            name='frust_vox-%d-%s-%s' % (
                resolution, camera_mutator.name, synset_id),
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
        return VoxelLoader(archive, self._resolution, self._camera_mutator)


class VoxelLoader(c.ExampleLoader):
    def __init__(self, archive, resolution, camera_mutator):
        self._resolution = resolution
        self._camera_mutator = camera_mutator
        super(VoxelLoader, self).__init__(archive=archive)

    def __call__(self, model_path, model_id):
        from shape_tfds.shape.shapenet.core import views
        from shape_tfds.shape.transformations import look_at
        import matplotlib.pyplot as plt
        model_dir, filename = os.path.split(model_path)
        resolver = ZipSubdirResolver(self.archive, model_dir)
        file_obj = resolver.get(filename)
        scene = trimesh.load(file_obj, file_type='obj', resolver=resolver)
        if hasattr(scene, 'geometry'):
            mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene.geometry.values()])
        else:
            mesh = scene

        fix_axes(mesh)
        scene = mesh.scene()
        for camera in self._camera_mutator(scene.camera):
            scene.show()
            dist = np.linalg.norm(camera.transform[:3, 3])

            vox = mesh.voxelized(
                pitch=1./self._resolution, method='binvox',
                bounding_box=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], exact=True)
            vox.fill(method='orthographic')
            origin, rays, angles = camera.to_rays()
            origin = origin[0]
            rays = rays.reshape((self._resolution, self._resolution, 3))
            # rays = rays[-1::-1, -1::-1]
            del angles
            z = np.linspace(dist - 0.5, dist + 0.5, self._resolution)
            coords = np.expand_dims(rays, axis=-2) * np.expand_dims(z, axis=-1)
            coords += origin
            coords = coords[:, :, -1::-1]

            frust_vox_dense = vox.is_filled(coords)
            frust_vox = trimesh.voxel.VoxelGrid(frust_vox_dense)
            frust_vox.show()
            proj = np.any(frust_vox_dense, axis=-1)
            plt.imshow(proj.T)
            plt.show()

        # print(extents)
        # vox.show()


if __name__ == '__main__':
    ids, names = c.load_synset_ids()

    # name = 'suitcase'
    # name = 'watercraft'
    # name = 'aeroplane'
    # name = 'table'
    name = 'rifle'

    config = ShapenetCoreFrustumVoxelConfig(synset_id=ids[name], resolution=64)
    builder = c.ShapenetCore(config=config)
    builder.download_and_prepare()
