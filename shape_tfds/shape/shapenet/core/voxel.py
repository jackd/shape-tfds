"""Requires difference trimesh fork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet.core import base
from shape_tfds.core.resolver import ZipSubdirResolver
from shape_tfds.shape.shapenet.core.views import fix_axes
from shape_tfds.core.features import PaddedTensor
from shape_tfds.core.features import BinaryRunLengthEncodedFeature
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
        import shape_tfds as sds
        return dict(
            voxels=sds.core.features.BinaryVoxel(shape=(self._resolution,)*3))

    def loader(self, dl_manager=None):
        return base.mesh_loader_context(
            synset_id=self.synset_id, dl_manager=dl_manager,
            item_map_fn=lambda key, scene: dict(
                voxels=load_voxels(scene, self._resolution).encoding.dense))


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh


def voxelize_binvox(mesh, pitch=None, dimension=None, bounds=None, **binvoxer_kwargs):
    """
    Voxelize via binvox tool.

    Parameters
    --------------
    mesh : trimesh.Trimesh
      Mesh to voxelize
    pitch : float
      Side length of each voxel. Ignored if dimension is provided
    dimension: int
      Number of voxels along each dimension. If not provided, this is alculated
        based on pitch and bounds/mesh extents
    bounds: (2, 3) float
      min/max values of the returned `VoxelGrid` in each instance.
    **binvoxer_kwargs:
      Passed to `trimesh.exchange.binvox.Binvoxer`.
      Should not contain `bounding_box` if bounds is not None.

    Returns
    --------------
    `VoxelGrid` instance

    Raises
    --------------
    `ValueError` if both bounds and bounding_box (in binvoxer_kwargs) are given
    """
    from trimesh.exchange import binvox

    if dimension is None:
        # pitch must be provided
        if bounds is None:
            extents = mesh.extents
        else:
            mins, maxs = bounds
            extents = maxs - mins
        dimension = int(np.ceil(np.max(extents) / pitch))
    if bounds is not None:
        if 'bounding_box' in binvoxer_kwargs:
            raise ValueError('Cannot provide both bounds and bounding_box')
        binvoxer_kwargs['bounding_box'] = np.asanyarray(bounds).flatten()

    binvoxer = binvox.Binvoxer(dimension=dimension, **binvoxer_kwargs)
    return binvox.voxelize_mesh(mesh, binvoxer)


def load_voxels(scene, resolution):
    mesh = as_mesh(scene)
    fix_axes(mesh)
    # vox = mesh.voxelized(
    vox = voxelize_binvox(
        mesh,
        pitch=None, dimension=resolution,
        bounds=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], exact=True,
        )
        # method='binvox',
    if vox.shape != (resolution,) * 3:
        raise ValueError(
            'Expected shape %s, got %s'
            % (str((resolution,)*3), str(vox.shape)))
    vox.fill(method='orthographic')
    return vox


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
