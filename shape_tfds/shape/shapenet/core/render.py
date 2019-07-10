"""
Requires updated obj loader from trimesh (including jackd's edit).

See https://github.com/mikedh/trimesh/pull/436
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow_datasets as tfds
import trimesh
from shape_tfds.shape.shapenet.core import base
from shape_tfds.shape.resolver import ZipSubdirResolver
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
        return tfds.core.features.FeaturesDict(dict(
            image=tfds.core.features.Image(shape=self.resolution+(3,)),
            example_id=tfds.core.features.Text()
        ))

    @property
    def synset_id(self):
        return self._synset_id

    @property
    def resolution(self):
        return self._resolution

    def loader(self, archive):
        return RenderLoader(archive, self._scene_mutator, self._resolution)


def fix_visual(visual, nv):
    from PIL import Image
    if isinstance(visual, trimesh.visual.ColorVisuals):
        return
    material = visual.material
    assert(hasattr(material, 'image'))
    if material.image is None and material.diffuse is not None:
        image = np.array(material.diffuse)
        image = (image*255).astype(dtype=np.uint8).reshape((1, 1, 3))
        image = Image.fromarray(image)
        material.image = image
    if visual.uv is None:
        visual.uv = np.zeros((nv, 2))


def fix_geometry_visuals(geometry):
    visual = getattr(geometry, 'visual', [])
    nv = geometry.vertices.shape[0]
    if hasattr(visual, '__iter__'):
        for v in visual:
            fix_visual(v, nv)
    else:
        fix_visual(visual, nv)


def fix_scene_visuals(scene):
    geometry = scene.geometry
    if hasattr(geometry, 'values'):
        for v in geometry.values():
            fix_geometry_visuals(v)
    elif hasattr(geometry, '__iter__'):
        for v in geometry:
            fix_geometry_visuals(v)
    else:
        fix_geometry_visuals(v)


class RenderLoader(base.ExampleLoader):
    def __init__(self, archive, scene_mutator, resolution):
        super(RenderLoader, self).__init__(archive)
        self._scene_mutator = scene_mutator
        self._resolution = resolution

    def __call__(self, model_path, model_id):
        import trimesh
        model_dir, filename = os.path.split(model_path)
        resolver = ZipSubdirResolver(self.archive, model_dir)
        scene = trimesh.load(
            trimesh.util.wrap_as_stream(resolver.get(filename)),
            file_type='obj',
            resolver=resolver)
        fix_scene_visuals(scene)
        fix_axes(scene)
        self._scene_mutator(scene, self._resolution)
        image = scene.save_image(resolution=None)
        image = trimesh.util.wrap_as_stream(image)
        return dict(
            image=image,
            example_id=model_id,
        )
    
    @property
    def mutator(self):
        return self._scene_mutator

if __name__ == '__main__':
    import tensorflow as tf
    import matplotlib.pyplot as plt
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
    image = dataset.make_one_shot_iterator().get_next()['image']
    with tf.Session() as sess:
        try:
            while True:
                plt.imshow(sess.run(image))
                plt.show()
        except BaseException:
            pass