"""Requires difference trimesh fork."""
import contextlib

import tensorflow as tf
import tensorflow_datasets as tfds
import trimesh

from shape_tfds.core.features.voxel import BinaryVoxel
from shape_tfds.shape.shapenet.core import base

trimesh.util.log.setLevel("ERROR")


class VoxelConfig(base.ShapenetCoreConfig):
    def __init__(self, synset_id, resolution=64):
        self._resolution = resolution
        super(VoxelConfig, self).__init__(
            name="voxel-%s-%d" % (synset_id, resolution),
            description="shapenet core voxels",
            version=tfds.core.Version("0.0.1"),
            synset_id=synset_id,
        )

    @property
    def features(self):
        return dict(voxels=BinaryVoxel(shape=(self._resolution,) * 3))

    @property
    def resolution(self):
        return self._resolution

    @contextlib.contextmanager
    def lazy_mapping(self, dl_manager=None):
        import trimesh

        binvoxer = trimesh.exchange.binvox.Binvoxer(
            exact=True,
            dimension=self._resolution,
            bounding_box=(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5),
        )

        def map_fn(obj_path):
            binvox_path = binvoxer(obj_path)
            with tf.io.gfile.GFile(binvox_path, "rb") as fp:
                vox = trimesh.exchange.binvox.load_binvox(fp)
            tf.io.gfile.remove(binvox_path)
            vox.fill(method="orthographic")
            return dict(voxels=vox.encoding.dense)

        yield base.extracted_mesh_paths(self.synset_id, dl_manager).map(map_fn)
