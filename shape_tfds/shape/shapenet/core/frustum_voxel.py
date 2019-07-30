"""Requires difference trimesh fork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
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


class FrustumVoxelConfig(base.ShapenetCoreConfig):
    def __init__(
            self, synset_id, resolution=64, seed=0, use_cached_voxels=True):
        self._seed = seed
        self._resolution = resolution
        self._use_cached_voxels = use_cached_voxels
        super(FrustumVoxelConfig, self).__init__(
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

    @contextlib.contextmanager
    def lazy_mapping(self, dl_manager=None):
        from collection_utils.mapping import ItemMappedMapping
        if dl_manager is None:
            dl_manager = tfds.core.download.DownloadManager(
                download_dir=base.DOWNLOADS_DIR)
        view_fn = views.random_view_fn(seed_offset=self._seed)
        transform = _get_voxel_transform(self._resolution)

        def item_map_fn(key, data):
            voxels = trimesh.voxel.VoxelGrid(
                data['voxels'], transform=transform)
            scene = trimesh.primitives.Sphere().scene()
            return dict(voxels=transform_voxels(
                scene, voxels,
                self._resolution, position=view_fn(key)))

        base_builder = base.ShapenetCore(config=voxel_lib.VoxelConfig(
            synset_id=self.synset_id, resolution=self.resolution))

        # living dangerously
        base_dl_manager = tfds.core.download.DownloadManager(
            download_dir=dl_manager._download_dir,
            extract_dir=dl_manager._extract_dir,
            manual_dir=dl_manager._manual_dir,
            force_download=dl_manager._force_download,
            force_extraction=dl_manager._force_extraction,
            register_checksums=dl_manager._register_checksums,
            dataset_name=base_builder.name, # different
        )
        if self._use_cached_voxels:
            base_builder.create_cache(dl_manager=base_dl_manager)
            context = base_builder.builder_config.cache_mapping(
                base_builder.cache_dir, 'r')
        else:
            context = base_builder.builder_config.lazy_mapping(base_dl_manager)
        with context as src:
            yield ItemMappedMapping(src, item_map_fn)


def transform_voxels(scene, voxels, resolution, position):
    views.fix_axes(voxels)
    views.set_scene_view(scene, (resolution,)*2, position)
    dist = np.linalg.norm(position)
    origin, rays = scene.camera_rays()
    rays = rays.reshape((resolution, resolution, 3))
    z = np.linspace(dist - 0.5, dist + 0.5, resolution)
    coords = np.expand_dims(rays, axis=-2) * np.expand_dims(z, axis=-1)
    coords += origin

    frust_vox_dense = voxels.is_filled(coords)
    frust_vox_dense = frust_vox_dense.transpose((1, 0, 2))  # y, x, z  # pylint: disable=no-member
    return frust_vox_dense
