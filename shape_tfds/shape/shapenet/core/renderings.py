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
from shape_tfds.shape.shapenet.core import views
from tensorflow_datasets.core import features
import tensorflow_datasets as tfds


class ShapenetCoreRenderingsConfig(base.ShapenetCoreConfig):
    def __init__(self, synset_id, resolution=(128, 128), seed=0, **kwargs):
        self._resolution = tuple(resolution)
        self._seed = seed
        ny, nx = self._resolution
        name = 'renderings%dx%d-%s-%d' % (ny, nx, synset_id, seed)
        super(ShapenetCoreRenderingsConfig, self).__init__(
            name=name,
            description='shapenet core renderings',
            version=tfds.core.Version("0.0.1"),
            synset_id=synset_id)

    @property
    def resolution(self):
        return self._resolution

    @property
    def seed(self):
        return self._seed

    @property
    def features(self):
        return dict(
            image=features.Image(shape=self.resolution + (3,)))

    def loader(self, dl_manager=None):
        view_fn = views.random_view_fn(self.seed)
        return base.zipped_mesh_loader_context(
            synset_id=self.synset_id, dl_manager=dl_manager,
            item_map_fn=lambda key, scene: dict(image=render(
                key, scene, self.resolution, **view_fn(key))))


def render(key, scene, resolution, position, focal):
    if not isinstance(scene, trimesh.Scene):
        scene = scene.scene()
    views.fix_axes(scene)
    views.set_scene_view(scene, resolution, position, focal)
    # would prefer offscreen rendering - visible=False
    # https://github.com/mikedh/trimesh/issues?q=is%3Aissue+blank+is%3Aclosed
    image = scene.save_image(resolution=None, visible=True)
    image = trimesh.util.wrap_as_stream(image)
    return image
