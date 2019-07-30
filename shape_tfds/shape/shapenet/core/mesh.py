from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import numpy as np
import tensorflow as tf
from tensorflow_datasets.core import features

from collection_utils import mapping
from shape_tfds.shape.shapenet.core import base


class MeshConfig(base.ShapenetCoreConfig):
    def __init__(self, synset_id, **kwargs):
        super(MeshConfig, self).__init__(
            synset_id=synset_id, name='mesh-%s' % synset_id)

    def lazy_mapping(self, dl_manager=None):
        def item_map_fn(key, mesh):
            return dict(vertices=mesh.vertices, faces=mesh.faces)

        return base.zipped_mesh_loader_context(
            self.synset_id, dl_manager, item_map_fn=item_map_fn)

    def features(self):
        return dict(
            vertices=features.Tensor(shape=(None, 3), dtype=tf.float32),
            faces=features.Tensor(shape=(None, 3), dtype=tf.int64))
