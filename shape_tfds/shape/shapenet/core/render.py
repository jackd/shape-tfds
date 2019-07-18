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
from shape_tfds.shape.shapenet.core.views import SceneMutator
from shape_tfds.shape.shapenet.core.views import fix_axes


class ShapenetCoreRenderConfig(base.ShapenetCoreConfig):
    def __init__(self, synset_id, resolution=(128, 128), scene_mutator=None):
        self._synset_id = synset_id
        self._scene_mutator = (
            SceneMutator() if scene_mutator is None else scene_mutator)
        self._resolution = resolution
        ny, nx = resolution
        super(ShapenetCoreRenderConfig, self).__init__(
            name='render-%dx%d-%s-%s' % (
                ny, nx, self._scene_mutator.name, synset_id),
            description='shapenet core renderings',
            version=tfds.core.Version("0.0.1"))

    def features(self):
        return dict(image=tfds.core.features.Image(shape=self.resolution+(3,)))

    @property
    def synset_id(self):
        return self._synset_id

    @property
    def resolution(self):
        return self._resolution

    def loader(self, dl_manager=None):
        return base.mesh_loader_context(
            synset_id=self.synset_id, dl_manager=dl_manager,
            map_fn=lambda scene: dict(image=render(
                scene, self._scene_mutator, self._resolution)))


def render(scene, scene_mutator, resolution):
    if not isinstance(scene, trimesh.Scene):
        scene = scene.scene()
    fix_axes(scene)
    scene_mutator(scene, resolution)
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
    seed = 0

    config = ShapenetCoreRenderConfig(
        synset_id=ids[synset_name],
        scene_mutator=SceneMutator(name='base%03d' % seed, seed=seed))
    builder = base.ShapenetCore(config=config)
    builder.download_and_prepare(download_config=download_config)

    dataset = builder.as_dataset(split='train')
    for example in dataset:
        image = example['image'].numpy()
        plt.imshow(image)
        plt.show()
