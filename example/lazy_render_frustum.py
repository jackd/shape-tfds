from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

flags.DEFINE_string('synset', default='suitcase', help='synset name')
flags.DEFINE_string('renderer', default='blender', help='renderer name')
flags.DEFINE_integer('resolution', default=128, help='voxel resolution')
FLAGS = flags.FLAGS


def main(_):
    from shape_tfds.shape.shapenet import core
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from trimesh.util import BytesIO
    synset = FLAGS.synset
    resolution = (FLAGS.resolution,)*2
    renderer = core.Renderer.named(FLAGS.renderer, resolution=resolution)

    ids, _ = core.load_synset_ids()
    synset_id = ids[synset]
    resolution = 128

    config = core.RenderingConfig(synset_id, renderer)
    frustum_config = core.FrustumVoxelConfig(
        synset_id, resolution=resolution, use_cached_voxels=False)
    with config.lazy_mapping() as renderings:
        with frustum_config.lazy_mapping() as voxels:
            for k in renderings:
                _, ax = plt.subplots(2, 2)
                ax = ax.reshape((-1,))
                vox = np.any(voxels[k]['voxels'], axis=-1)

                image = np.array(np.array(renderings[k]['image']))
                image = image.astype(np.float32) / np.max(image)
                ax[0].imshow(image)
                ax[1].imshow(vox)
                sil_image = image.copy()
                sil_image[np.logical_not(vox)] = 1
                ax[2].imshow(sil_image)
                sil_image = image.copy()
                sil_image[vox] = 1
                ax[3].imshow(sil_image)
                plt.show()


if __name__ == '__main__':
    app.run(main)
