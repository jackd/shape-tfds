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
from ige.vox.data import core as c
from ige.vox.data.transformations import look_at
from ige.vox.data.resolver import ZipSubdirResolver


class ShapenetCoreRenderConfig(c.ShapenetCoreConfig):
    def __init__(self, synset_id, resolution=(128, 128)):
        self._synset_id = synset_id
        if isinstance(resolution, int):
            resolution = (resolution,)*2
        else:
            resolution = tuple(resolution)
            assert(all(isinstance(r, int) for r in resolution))
        self._resolution = resolution

        super(ShapenetCoreRenderConfig, self).__init__(
            name='render-%s-%dx%d' % ((synset_id,) + resolution),
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
        return ExampleLoader(archive)


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


class ExampleLoader(c.ExampleLoader):
    def __call__(self, model_path, model_id):
        import trimesh
        model_dir, filename = os.path.split(model_path)
        print(model_id)
        resolver = ZipSubdirResolver(self.archive, model_dir)
        scene = trimesh.load(
            resolver.get(filename, read=False), file_type='obj',
            resolver=resolver)
        fix_scene_visuals(scene)
        transform = look_at(
            np.array([1, 1, 1], dtype=np.float64), world_up=[0, 1, 0])
        resolution = (1024, 1024)
        camera = trimesh.scene.cameras.Camera(
            fov=(60, 60), resolution=resolution, transform=transform)
        scene.camera = camera
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
