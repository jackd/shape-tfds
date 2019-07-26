from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tqdm
import os

import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet.core import base
import trimesh
import numpy as np
# import matplotlib.pyplot as plt

ids, _ = base.load_synset_ids()
model_ids = base.load_split_ids(excluded=None)
synset_ids = sorted(model_ids)
synset_id = synset_ids[0]
ids = np.concatenate(tuple(
    model_ids[synset_id][k] for k in ('test', 'train', 'validation')))
paths = base.extracted_mesh_paths(synset_id)
sizes = paths.map(os.path.getsize)
sizes = [sizes[m] for m in ids]
print(np.min(sizes), np.max(sizes), np.mean(sizes))
print(np.mean(sizes) / np.max(sizes))
max_size = np.max(sizes)
n = sum(1 for s in sizes if s > 0.5*max_size)
indices = [i for i, s in enumerate(sizes) if s > 0.5*max_size]
# big_ids = [ids[i] for i in indices]

for i in indices:
    print(sizes[i], paths[ids[i]])

print(len(indices))
# plt.hist(sizes)
# plt.show()

# n = 1595

# sizes = os.path.getsize(path)

# for model_id in ids[n: n+1]:
#     path = paths[model_id]
#     size = os.path.getsize("/path/isa_005.mp3")
#     print(path)
#     # trimesh.load(path)
# # check_core_models(synset_id, model_ids[synset_id][n: n + 2])
