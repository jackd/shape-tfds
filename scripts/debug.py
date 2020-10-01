from absl import app, flags
from absl import logging
import tqdm
import os

import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet import core
from shape_tfds.core.mapping import concat_dict_values
import trimesh
import numpy as np
import matplotlib.pyplot as plt

synset = 'car'
model_index = 1765
ids, names = core.load_synset_ids()
synset_id = ids[synset]
model_ids = core.load_split_ids()[synset_id]
model_ids = concat_dict_values(model_ids)
model_id = model_ids[model_index]

config = core.RenderingConfig(synset_id, renderer=core.BlenderRenderer())
with config.lazy_mapping() as images:
    image = images[model_id]['image']
    image.show()
    print(image)
