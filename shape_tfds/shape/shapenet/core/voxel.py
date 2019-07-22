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
from shape_tfds.core.features import BinaryVoxel
import trimesh

trimesh.util.log.setLevel('ERROR')


class ShapenetCoreVoxelConfig(base.ShapenetCoreConfig):
    def __init__(self, synset_id, resolution=64, from_file_mapping=False):
        self._resolution = resolution
        self._from_file_mapping = from_file_mapping
        super(ShapenetCoreVoxelConfig, self).__init__(
            name='voxel-%s-%d' % (synset_id, resolution),
            description='shapenet core voxels',
            version=tfds.core.Version("0.0.1"),
            synset_id=synset_id)

    @property
    def features(self):
        return dict(voxels=BinaryVoxel(shape=(self._resolution,)*3))

    @property
    def resolution(self):
        return self._resolution


    def base_loader_context(self, dl_manager=None):
        return base.mesh_loader_context(
            synset_id=self.synset_id, dl_manager=dl_manager,
            item_map_fn=lambda key, scene: dict(
                voxels=scene_to_voxels(
                    scene, self._resolution).encoding.dense))

    def mapping_loader_context(self, dl_manager=None):
        return base.get_data_mapping_context(
            config=ShapenetCoreVoxelConfig(
                synset_id=self.synset_id, resolution=self.resolution,
                from_file_mapping=False),
            dl_manager=dl_manager)

    def loader_context(self, dl_manager):
        return (
            self.mapping_loader_context if self._from_file_mapping else
            self.base_loader_context)(dl_manager)


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh


def voxelize_binvox(
        mesh, pitch=None, dimension=None, bounds=None, **binvoxer_kwargs):
    """
    Voxelize via binvox tool.

    Parameters
    --------------
    mesh : trimesh.Trimesh
      Mesh to voxelize
    pitch : float
      Side length of each voxel. Ignored if dimension is provided
    dimension: int
      Number of voxels along each dimension. If not provided, this is
        calculated based on pitch and bounds/mesh extents
    bounds: (2, 3) float
      min/max values of the returned `VoxelGrid` in each instance. Uses
      `mesh.bounds` if not provided.
    **binvoxer_kwargs:
      Passed to `trimesh.exchange.binvox.Binvoxer`.
      Should not contain `bounding_box` if bounds is not None.

    Returns
    --------------
    `VoxelGrid` instance

    Raises
    --------------
    `ValueError` if `bounds is not None and 'bounding_box' in binvoxer_kwargs`.
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


def scene_to_voxels(scene, resolution):
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
