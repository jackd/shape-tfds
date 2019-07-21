"""
Requires updated obj loader from trimesh (including jackd's edit).

See https://github.com/mikedh/trimesh/pull/436
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import numpy as np
import tensorflow_datasets as tfds
import trimesh
from shape_tfds.shape.shapenet.core import base
from shape_tfds.core.resolver import ZipSubdirResolver
from shape_tfds.shape.shapenet.core import views
import tensorflow_datasets as tfds


class ShapenetCoreRenderingsConfig(tfds.core.BuilderConfig):
    def __init__(self, synset_id, resolution=(128, 128), seed=0, **kwargs):
        self._resolution = resolution
        self._seed = seed
        self._synset_id = synset_id
        ny, nx = resolution
        name = 'renderings-%d-%d-%s-%d' % (ny, nx, synset_id, seed)
        super(ShapenetCoreRenderingsConfig, self).__init__(
            name=name,
            description='shapenet core renderings',
            version=tfds.core.Version("0.0.1"))

    @property
    def resolution(self):
        return self._resolution

    @property
    def seed(self):
        return self._seed

    @property
    def synset_id(self):
        return self._synset_id


class ShapenetCoreRenderings(base.ShapenetCore):
    @property
    def _features(self):
        return dict(image=tfds.core.features.Image(
            shape=self.builder_config.resolution+(3,)))

    def loader_conext(self, dl_manager=None):
        view_fn = views.random_view_fn(self.builder_config.seed)
        return base.mesh_loader_context(
            synset_id=self.synset_id, dl_manager=dl_manager,
            item_map_fn=lambda key, scene: dict(image=render(
                scene, self._resolution, **view_fn(key))))


def render(scene, resolution, position, focal):
    if not isinstance(scene, trimesh.Scene):
        scene = scene.scene()
    views.fix_axes(scene)
    views.set_scene_view(scene, resolution, position, focal)
    image = scene.save_image(resolution=None)
    image = trimesh.util.wrap_as_stream(image)
    return image


if __name__ == '__main__':
    import tensorflow as tf
    import matplotlib.pyplot as plt
    tf.compat.v1.enable_eager_execution()
    ids, names = base.load_synset_ids()

    download_config = tfds.core.download.DownloadConfig(
        register_checksums=True)

    synset_name = 'suitcase'
    # name = 'watercraft'
    # name = 'aeroplane'
    seed_offset = 0
    synset_id = ids[synset_name]

    resolution = (128, 128)
    builder = ShapenetCoreRenderings(config=ShapenetCoreRenderingsConfig(
        synset_id=synset_id, resolution=resolution, seed=seed_offset))
    builder.download_and_prepare(download_config=download_config)

    dataset = builder.as_dataset(split='train')
    for example in dataset:
        image = example['image'].numpy()
        plt.imshow(image)
        plt.show()
