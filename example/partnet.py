from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet.partnet import Partnet

for config in Partnet.BUILDER_CONFIGS:
    Partnet(config=config).download_and_prepare()
