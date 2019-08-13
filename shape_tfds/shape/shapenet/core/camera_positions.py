from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import numpy as np
import tensorflow as tf
from tensorflow_datasets.core import features

from collection_utils import mapping
from shape_tfds.shape.shapenet.core import base
from shape_tfds.shape.shapenet.core import views
from shape_tfds.core.mapping import concat_dict_values


class CameraPositionConfig(base.ShapenetCoreConfig):

    def __init__(self, synset_id, seed=0, **kwargs):
        super(CameraPositionConfig,
              self).__init__(synset_id=synset_id,
                             name='camera-positions%d' % seed)
        self._seed = seed
        self._view_fn = views.random_view_fn(seed)

    @property
    def seed(self):
        return self._seed

    def camera_position(self, model_id):
        return self._view_fn(model_id)

    @contextlib.contextmanager
    def lazy_mapping(self, dl_manager=None):
        model_ids = base.load_split_ids(dl_manager)[self.synset_id]
        model_ids = concat_dict_values(model_ids)
        yield mapping.LazyMapping(
            model_ids, lambda model_id: dict(camera_position=self.
                                             camera_position(model_id)))

    def features(self):
        return dict(
            camera_position=features.Tensor(shape=(3,), dtype=tf.float32))
