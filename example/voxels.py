from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from shape_tfds.shape.shapenet import core
tf.compat.v1.enable_eager_execution()

ids, names = core.load_synset_ids()
seed = 0
synset_name = 'suitcase'
# name = 'watercraft'
# name = 'aeroplane'
# name = 'table'
# name = 'rifle'


def vis(image, voxels):
    # visualize a single image/voxel pair
    import matplotlib.pyplot as plt
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D

    # and plot everything
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(image)
    ax.axis("off")
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.voxels(voxels)
    # ax.axis("square")
    plt.show()


synset_id = ids[synset_name]
mutator = core.SceneMutator(seed=seed, name='base%03d' % seed)

configs = dict(
    image=core.ShapenetCoreRenderConfig(synset_id, scene_mutator=mutator),
    voxels=core.ShapenetCoreVoxelConfig(synset_id, resolution=32))
builders = {k: core.ShapenetCore(config=config)
            for k, config in configs.items()}
for b in builders.values():
    b.download_and_prepare()

datasets = {
    k: b.as_dataset(split='train', shuffle_files=False).map(
        lambda x: x[k]) for k, b in builders.items()}

dataset = tf.data.Dataset.zip(datasets)


for example in dataset:
    image = example['image'].numpy()
    voxels = example['voxels'].numpy()
    vis(image, voxels)
