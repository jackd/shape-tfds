from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet import core

import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.enable_eager_execution()
ids, names = core.load_synset_ids()

download_config = tfds.core.download.DownloadConfig(
    register_checksums=True)

synset_name = 'suitcase'
# name = 'watercraft'
# name = 'aeroplane'
seed_offset = 0
synset_id = ids[synset_name]

resolution = (128, 128)
builder = core.ShapenetCore(
        config=core.ShapenetCoreRenderingsConfig(
    synset_id=synset_id, resolution=resolution, seed=seed_offset))
builder.download_and_prepare(download_config=download_config)

def view(example):
    image = example['image'].numpy()
    plt.imshow(image)
    plt.show()


dataset = builder.as_dataset(split='train')
for example in dataset:
    view(example)
