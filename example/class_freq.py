from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow_datasets as tfds

from shape_tfds.shape.modelnet import base
builder = base.Modelnet40(config=base.CloudConfig(2048))

freqs = np.zeros((40,), dtype=np.int64)

for _, label in tfds.as_numpy(
        builder.as_dataset(as_supervised=True, split='train')):
    freqs[label] += 1

print(freqs)
