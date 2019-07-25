from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zipfile
from shape_tfds.shape.shapenet.core.base import mesh_loader_context
from shape_tfds.shape.shapenet.core.base import get_obj_zip_path
from shape_tfds.shape.shapenet.core.base import load_synset_ids

synset = 'suitcase'

ids, names = load_synset_ids()
with mesh_loader_context(ids[synset]) as loader:
    for k, v in loader.items():
        print(k)
        v.show()
