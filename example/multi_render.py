from absl import app, flags
import tensorflow as tf
import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet import core
from shape_tfds.core.mapping import concat_dict_values

import tensorflow as tf
import matplotlib.pyplot as plt

flags.DEFINE_string('synset', default='suitcase', help='synset name')
flags.DEFINE_string('renderer', default='blender', help='renderer name')
flags.DEFINE_integer('resolution', default=128, help='voxel resolution')
flags.DEFINE_integer('num_seeds',
                     default=24,
                     help='number of seeds to use for random_view_fn')


def main(_):
    tf.compat.v1.enable_eager_execution()
    FLAGS = flags.FLAGS
    ids, names = core.load_synset_ids()
    synset = FLAGS.synset

    synset_id = synset if synset in names else ids[synset]
    if synset_id not in names:
        raise ValueError('Invalid synset_id %s' % synset_id)
    renderer = core.Renderer.named(name=FLAGS.renderer,
                                   resolution=(FLAGS.resolution,) * 2)
    renderer.create_multi_cache(synset_id,
                                seeds=range(FLAGS.num_seeds),
                                keys=concat_dict_values(
                                    core.load_split_ids()[synset_id]))


if __name__ == '__main__':
    app.run(main)
