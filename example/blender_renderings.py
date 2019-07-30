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

flags.DEFINE_string('synset', default='suitcase', help='synset name')
flags.DEFINE_integer('resolution', default=128, help='voxel resolution')
flags.DEFINE_integer('seed', default=0, help='seed to use for random_view_fn')
flags.DEFINE_boolean('vis', default=False, help='visualize on finish')
flags.DEFINE_boolean('from_cache', default=False, help='if True, use cache')


def main(_):
    tf.compat.v1.enable_eager_execution()
    FLAGS = flags.FLAGS
    ids, names = core.load_synset_ids()
    synset = FLAGS.synset

    synset_id = synset if synset in names else ids[synset]
    if synset_id not in names:
        raise ValueError('Invalid synset_id %s' % synset_id)

    builder = core.ShapenetCore(config=core.RenderingConfig(
        synset_id=synset_id,
        renderer=core.BlenderRenderer(resolution=(FLAGS.resolution,)*2),
        seed=FLAGS.seed), from_cache=FLAGS.from_cache)
    builder.download_and_prepare()

    if FLAGS.vis:
        def vis(example):
            import matplotlib.pyplot as plt
            plt.imshow(example['image'].numpy())
            plt.show()

        dataset = builder.as_dataset(split='train')
        for example in dataset:
            vis(example)


if __name__ == '__main__':
    app.run(main)
