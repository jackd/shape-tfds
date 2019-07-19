from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from shape_tfds.shape.shapenet import core
tf.compat.v1.enable_eager_execution()

ids, names = core.load_synset_ids()
resolution = 128
seed_offset = 0
synset_name = 'suitcase'
# name = 'watercraft'
# name = 'aeroplane'
# name = 'table'
# name = 'rifle'

synset_id = ids[synset_name]
nx = resolution
ny = resolution
view_fn = core.views.random_view_fn(seed_offset)

configs = dict(
    image=core.ShapenetCoreRenderConfig(
        name='render-%s-%dx%d-%03d' % (synset_id, ny, nx, seed_offset),
        synset_id=synset_id, resolution=(resolution,)*2, view_fn=view_fn),
    voxels=core.ShapenetCoreFrustumVoxelConfig(
        name='frustum_voxels-%s-%03d-%03d' % (
            synset_id, resolution, seed_offset),
        synset_id=synset_id, view_fn=view_fn, resolution=128))
builders = {k: core.ShapenetCore(config=config)
            for k, config in configs.items()}
for b in builders.values():
    b.download_and_prepare()

datasets = {
    k: b.as_dataset(split='train', shuffle_files=False).map(
        lambda x: x[k]) for k, b in builders.items()}

dataset = tf.data.Dataset.zip(datasets)


def vis():
    import matplotlib.pyplot as plt
    for example in dataset:
        image = example['image'].numpy()
        voxels = tf.reduce_any(example['voxels'], axis=-1).numpy()
        image[np.logical_not(voxels)] = 0
        plt.imshow(image)
        plt.show()
        # _, (ax0, ax1) = plt.subplots(1, 2)
        # ax0.imshow(image)
        # ax1.imshow(voxels)
        # plt.show()


vis()
