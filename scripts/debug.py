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
import matplotlib.pyplot as plt

# synset_id = '02958343'
# model_id = 'e23ae6404dae972093c80fb5c792f223'

# paths = base.extracted_mesh_paths(synset_id)
# path = paths[model_id]
# print('---')
# print(path)
# print(os.stat(path).st_size)
# trimesh.load(path)
# print('done')

failed_ids = {
    '02828884': ('2f0cd28e2f8cdac16121178eafd002fd',),
    '02933112': (),
    '02958343': (),
    '03001627': ('747d2db60ca6b98f3eec26c23f5bc80b', '1de733a48e5607b22d9c1884c92fce12',),
    '03636649': ('3259e491870a8c49eef5d83b671bb264',),
    '03211117': (),
    '02691156': ('6f0ad1fb7917fd9b50577cf04f3bf74a', 'dbab9feed7e936cfa87372b03d6dc78b', 'f9209166fc259d8885e96081cfe0563b', 'c88275e49bc23ee41af5817af570225e',),
    '04090263': ('661ad35c5c3a908c6d1e04c7ae242f3d',),
    '04256520': ('ecf29f5698798a74104d78b9947ee8', '9f5fd43df32187739f2349486c570dd4',),
    '03691459': (),
    '04379243': ('dc0e0beba650c0b78bc6f322a9608b07', '56ea26c10c478555a31cc7b61ec6561', '1846b3533f41ae82f8c4b4cfc2702232',),
    '04401088': (),
    '04530566': ('18761559208a970188d5590328ce0ddf', '80d381a6760185d8c45977b13fbe7645', '587cc1fc65ac4991ff920fdb73e92549',),
    '02747177': (),
    '02773838': (),
    '02801938': (),
    '02808440': (),
    '02818832': (),
    '02843684': (),
    '02871439': (),
    '02876657': ('8099b9d546231dd27b9c6deef486a7d8', '2d1aa4e124c0dd5bf937e5c6aa3f9597'),
    '02880940': (),
    '02924116': (),
    '02942699': (),
    '02946921': (),
    '02954340': (),
    '02992529': (),
    '03046257': (),
    '03085013': (),
    '03207941': (),
    '03261776': (),
    '03325088': (),
    '03337140': (),
    '03467517': ('3c125ee606b03cd263ae8c3a62777578',),
    '03513137': (),
    '03593526': (),
    '03624134': ('67ada28ebc79cc75a056f196c127ed77',),
    '03642806': (),
    '03710193': (),
    '03759954': (),
    '03761084': (),
    '03790512': (),
    '03797390': (),
    '03928116': (),
    '03938244': (),
    '03948459': (),
    '03991062': (),
    '04004475': (),
    '04074963': (),
    '04099429': (),
    '04225987': (),
    '04330267': ('cfa1ad5b5b023fe81d2e2161e7c7075', '993d687908792787672e57a20276f6e9'),
    '04460130': (),
    '04468005': (),
    '04554684': (),
}

synset_ids = sorted(failed_ids)
pairs = []
for synset_id in synset_ids:
    pairs.extend([[synset_id, model_id] for model_id in failed_ids[synset_id]])
# paths = {k: base.extracted_mesh_paths(k) for k in synset_ids}


# for synset_id in synset_ids:
#     paths = base.extracted_mesh_paths(synset_id)
#     for bad_id in failed_ids[synset_id]:
for synset_id, model_id in pairs:
    with base.zipped_mesh_loader_context(synset_id) as loader:
        # print('---')
        # print('%s / %s' % (synset_id, model_id))
        # print(path)
        try:
            loader[model_id]
        except IndexError:
            pass
    # print('***')
print('done')
