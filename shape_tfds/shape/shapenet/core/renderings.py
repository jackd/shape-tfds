"""
Requires updated obj loader from trimesh (including jackd's edit).

See https://github.com/mikedh/trimesh/pull/436
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from absl import logging
import six
import contextlib
import collections
import functools
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import trimesh
from shape_tfds.shape.shapenet.core import base
from shape_tfds.shape.shapenet.core import views
from shape_tfds.core import mapping as shape_mapping
from tensorflow_datasets.core import features
import tensorflow_datasets as tfds
import tempfile
import tqdm
from collection_utils import mapping
from PIL import Image


def _string_to_image(string):
    import trimesh
    return Image.open(trimesh.util.BytesIO(string))


def _image_with_background(image, background_color):
    image = np.array(image)
    background = image[:, :, 3] == 0
    image = image[:, :, :3]
    image[background] = background_color
    return Image.fromarray(image)


class Renderer(object):

    def __init__(self, base_name, resolution):
        self._resolution = tuple(resolution)
        self._base_name = base_name

    @staticmethod
    def named(name, resolution):
        return {
            'blender': BlenderRenderer,
            'trimesh': TrimeshRenderer,
        }[name](resolution=resolution)

    @property
    def resolution(self):
        return self._resolution

    @property
    def fov(self):
        return views.DEFAULT_FOV

    @property
    def background_color(self):
        return (255, 255, 255)

    @abc.abstractproperty
    def name(self):
        ry, rx = self.resolution
        return '%s-render%dx%d' % (self._base_name, ry, rx)

    def base_mapping(self, synset_id, dl_manager=None) -> mapping.Mapping:
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, base_value, camera_positions):
        raise NotImplementedError

    def create_multi_cache(self,
                           synset_id,
                           seeds,
                           dl_manager=None,
                           overwrite=False):
        model_ids = shape_mapping.concat_dict_values(
            base.load_split_ids(dl_manager)[synset_id])
        cache_mappings = []
        view_fns = []
        for seed in seeds:
            config = RenderingConfig(synset_id, seed=seed, renderer=self)
            builder = base.ShapenetCore(config=config)
            cache_mappings.append(config._cache_mapping(builder.cache_dir))
            view_fns.append(views.random_view_fn(seed))
        if model_ids is None:
            model_ids = base.load_split_ids(dl_manager=dl_manager)[synset_id]
            model_ids = shape_mapping.concat_dict_values(model_ids)
        if not overwrite:
            model_ids = [
                m for m in model_ids
                if not all(m in cache for cache in cache_mappings)
            ]
        if len(model_ids) == 0:
            return
        with self.base_mapping(synset_id, dl_manager=dl_manager) as base_map:
            for model_id in tqdm.tqdm(model_ids):
                camera_positions = np.stack(
                    [view_fn(model_id) for view_fn in view_fns], axis=0)
                try:
                    renderings = self.render(base_map[model_id],
                                             camera_positions=camera_positions)
                    for cache, rendering in zip(cache_mappings, renderings):
                        cache[model_id] = rendering
                except Exception:
                    logging.warning('Failed to render model %s' % model_id)


class BlenderRenderer(Renderer):

    def __init__(self, resolution=(128, 128)):
        super(BlenderRenderer, self).__init__('blender', resolution)

    @contextlib.contextmanager
    def base_mapping(self, synset_id, dl_manager=None):
        yield base.extracted_mesh_paths(synset_id, dl_manager=dl_manager)

    def render(self, base_value, camera_positions):
        import tempfile
        from shape_tfds.rendering.blender.renderer import render
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = render(obj_path=base_value,
                              output_directory=temp_dir,
                              camera_positions=camera_positions,
                              fov=self.fov,
                              filename_format='{output:s}-{index:03d}.png',
                              include_normals=False,
                              include_albedo=False,
                              include_depth=False,
                              resolution=self.resolution)

            return [
                _image_with_background(
                    Image.open(tmp_path.format(output='render', index=i)),
                    self.background_color) for i in range(len(camera_positions))
            ]


class TrimeshRenderer(Renderer):

    def __init__(self, resolution=(128, 128)):
        super(TrimeshRenderer, self).__init__('trimesh', resolution)

    def base_mapping(self, synset_id, dl_manager=None):
        return base.zipped_mesh_loader_context(synset_id, dl_manager=dl_manager)

    def render(self, base_value, camera_positions):
        if isinstance(base_value, trimesh.scene.Scene):
            scene = base_value
        else:
            scene = base_value.scene()
        views.fix_axes(scene)
        images = []
        for position in camera_positions:
            views.set_scene_view(scene=scene,
                                 position=position,
                                 resolution=self.resolution,
                                 fov=(self.fov,) * 2)
            string = scene.save_image(resolution=None, visible=True)
            images.append(_string_to_image(string))
        return images


_renderer_factories = {'trimesh': TrimeshRenderer, 'blender': BlenderRenderer}


class RenderingConfig(base.ShapenetCoreConfig):

    def __init__(self, synset_id, renderer, seed=0):
        if isinstance(renderer, dict):
            renderer = Renderer.named(**renderer)
        self._renderer = renderer
        self._seed = seed
        renderer_name = self._renderer.name
        name = '%s-%s-%d' % (renderer_name, synset_id, seed)
        super(RenderingConfig, self).__init__(
            name=name,
            description='shapenet core %s renderings' % renderer_name,
            version=tfds.core.Version("0.0.1"),
            synset_id=synset_id)

    @property
    def resolution(self):
        return self._renderer.resolution

    @property
    def seed(self):
        return self._seed

    @property
    def features(self):
        return dict(image=features.Image(shape=self.resolution + (3,)))

    @contextlib.contextmanager
    def lazy_mapping(self, dl_manager=None):
        view_fn = views.random_view_fn(self.seed)

        def map_item(key, base_value):
            camera_positions = np.expand_dims(view_fn(key), axis=0)
            image = self._renderer.render(base_value, camera_positions)[0]
            return dict(image=image)

        with self._renderer.base_mapping(self.synset_id,
                                         dl_manager) as base_mapping:
            yield mapping.ItemMappedMapping(base_mapping, map_item)

    def _cache_mapping(self, cache_dir):
        return shape_mapping.ImageDirectoryMapping(cache_dir)

    @contextlib.contextmanager
    def cache_mapping(self, cache_dir, mode='r'):
        yield mapping.MappedMapping(self._cache_mapping(cache_dir),
                                    lambda image: dict(image=image))

    def create_cache(self, cache_dir, dl_manager=None, overwrite=False):
        # WARNING: slow if you want to create multiple view caches
        # use Renderer.create_multi_cache with multiple seeds
        self._renderer.create_multi_cache(self.synset_id,
                                          seeds=(self.seed,),
                                          dl_manager=None,
                                          overwrite=False)
