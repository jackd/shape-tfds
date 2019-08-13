from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf
from shape_tfds.shape.shapenet import core

flags.DEFINE_string('synset', default='suitcase', help='synset name')
flags.DEFINE_integer('resolution', default=32, help='voxel resolution')
flags.DEFINE_boolean('from_cache',
                     default=False,
                     help='create tfrecords data from cache')
flags.DEFINE_boolean('vis', default=False, help='visualize on finish')


def main(_):
    tf.compat.v1.enable_eager_execution()
    FLAGS = flags.FLAGS
    ids, names = core.load_synset_ids()
    name = FLAGS.synset

    synset_id = name if name in names else ids[name]
    if synset_id not in names:
        raise ValueError('Invalid synset_id %s' % synset_id)

    resolution = FLAGS.resolution

    config = core.VoxelConfig(synset_id=ids[name], resolution=resolution)
    builder = core.ShapenetCore(config=config, from_cache=FLAGS.from_cache)
    builder.download_and_prepare()
    dataset = builder.as_dataset(split='train')

    if FLAGS.vis:

        def vis(voxels):
            # visualize a single image/voxel pair
            import matplotlib.pyplot as plt
            # This import registers the 3D projection, but is otherwise unused.
            from mpl_toolkits.mplot3d import Axes3D

            ax = plt.gca(projection="3d")
            ax.voxels(voxels)
            # ax.axis("square")
            plt.show()

        for example in dataset:
            vis(example['voxels'].numpy())


if __name__ == '__main__':
    app.run(main)
