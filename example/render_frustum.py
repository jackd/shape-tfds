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
import numpy as np

flags.DEFINE_string('synset', default='suitcase', help='synset name')
flags.DEFINE_integer('vox_res', default=32, help='voxel resolution')
flags.DEFINE_integer('image_res', default=128, help='voxel resolution')
flags.DEFINE_integer('seed', default=0, help='seed to use for random_view_fn')
flags.DEFINE_boolean('vis', default=False, help='visualize on finish')


def main(_):
    tf.compat.v1.enable_eager_execution()
    FLAGS = flags.FLAGS
    ids, names = core.load_synset_ids()
    name = FLAGS.synset
    seed = FLAGS.seed

    synset_id = name if name in names else ids[name]
    if synset_id not in names:
        raise ValueError('Invalid synset_id %s' % synset_id)

    configs = dict(
        image=core.TrimeshRenderingConfig(
            synset_id=synset_id, resolution=(FLAGS.image_res,)*2, seed=seed),
        voxels=core.FrustumVoxelConfig(
            synset_id=synset_id, resolution=FLAGS.vox_res, seed=seed))
    builders = {k: core.ShapenetCore(config=config)
                for k, config in configs.items()}
    for b in builders.values():
        b.download_and_prepare()

    if FLAGS.vis:
        def vis(example):
            import matplotlib.pyplot as plt
            image = example['image'].numpy()
            voxels = tf.reduce_any(example['voxels'], axis=-1)
            voxels = tf.image.resize(
                tf.expand_dims(tf.cast(voxels, tf.uint8), axis=-1),
                image.shape[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            voxels = tf.cast(tf.squeeze(voxels, axis=-1), tf.bool).numpy()
            # voxels = voxels.T
            # voxels = voxels[:, -1::-1]
            image[np.logical_not(voxels)] = 0
            plt.imshow(image)
            plt.show()

        datasets = {
            k: b.as_dataset(split='train', shuffle_files=False).map(
                lambda x: x[k]) for k, b in builders.items()}
        dataset = tf.data.Dataset.zip(datasets)
        for example in dataset:
            vis(example)


if __name__ == '__main__':
    app.run(main)
