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
from shape_tfds.shape.shapenet import core as c
from shape_tfds.shape.resolver import ZipSubdirResolver
from shape_tfds.shape.shapenet.core.views import CameraMutator
from shape_tfds.shape.shapenet.core.views import fix_axes


class ShapenetCoreRenderConfig(c.ShapenetCoreConfig):
    def __init__(self, synset_id, resolution=(128, 128), camera_mutator=None):
        self._synset_id = synset_id
        self._camera_mutator = (
            CameraMutator() if camera_mutator is None else camera_mutator)
        self._resolution = resolution
        ny, nx = resolution
        super(ShapenetCoreRenderConfig, self).__init__(
            name='render-%dx%d-%s-%s' % (
                ny, nx, self._camera_mutator.name, synset_id),
            description='shapenet core renderings',
            version=tfds.core.Version("0.0.1"))
    
    @property
    def seed(self):
        return self._seed

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
        return RenderLoader(archive, self._camera_mutator, self._resolution)


def fix_visual(visual):
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


def fix_geometry_visuals(geometry):
    visual = getattr(geometry, 'visual', [])
    if hasattr(visual, '__iter__'):
        for v in visual:
            fix_visual(v)
    else:
        fix_visual(visual)


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


class RenderLoader(c.ExampleLoader):
    def __init__(self, archive, camera_mutator, resolution):
        super(RenderLoader, self).__init__(archive)
        self._camera_mutator = camera_mutator
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
        for _ in self._camera_mutator(scene.camera, self._resolution):
            scene.show()
        # raise Exception('still at debug stage')


if __name__ == '__main__':
    ids, names = c.load_synset_ids()

    download_config = tfds.core.download.DownloadConfig(
        register_checksums=True)

    # name = 'suitcase'
    name = 'watercraft'
    # name = 'aeroplane'

    config = ShapenetCoreRenderConfig(synset_id=ids[name])
    builder = c.ShapenetCore(config=config)
    builder.download_and_prepare(download_config=download_config)
