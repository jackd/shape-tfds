from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet import core

import tensorflow as tf
import matplotlib.pyplot as plt

flags.DEFINE_string('name', default='suitcase', help='synset name')
flags.DEFINE_integer('resolution', default=32, help='voxel resolution')
flags.DEFINE_integer('seed', default=0, help='seed to use for random_view_fn')
flags.DEFINE_boolean('vis', default=False, help='visualize on finish')


def main(_):
    tf.compat.v1.enable_eager_execution()
    FLAGS = flags.FLAGS
    ids, names = core.load_synset_ids()
    name = FLAGS.name

    synset_id = name if name in names else ids[name]
    if synset_id not in names:
        raise ValueError('Invalid synset_id %s' % synset_id)

    builder = core.ShapenetCore(
            config=core.ShapenetCoreFrustumVoxelConfig(
        synset_id=synset_id, resolution=FLAGS.resolution, seed=FLAGS.seed))
    builder.download_and_prepare()

    if FLAGS.vis:
        def vis(example):
            import matplotlib.pyplot as plt
            # This import registers the 3D projection, but is otherwise unused.
            from mpl_toolkits.mplot3d import Axes3D

            ax = plt.gca(projection="3d")
            ax.voxels(example['voxels'].numpy())
            # ax.axis("square")
            plt.show()

        dataset = builder.as_dataset(split='train')
        for example in dataset:
            vis(example)


if __name__ == '__main__':
    app.run(main)
