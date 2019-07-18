from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zipfile
from shape_tfds.shape.shapenet.core.base import mesh_loader
from shape_tfds.shape.shapenet.core.base import get_obj_zip_path
from shape_tfds.shape.shapenet.core.base import load_synset_ids

synset = 'suitcase'

ids, names = load_synset_ids()
meshes = mesh_loader(zipfile.ZipFile(get_obj_zip_path(ids[synset])))
print(tuple(meshes.keys()))
print(len(meshes))
for k, v in meshes.items():
    print(k)
    v.show()
