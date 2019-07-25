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

flags.DEFINE_string('synset', help='synset name or id', default=None)


def check_core_models(synset_id, model_ids):
    model_ids = np.concatenate(tuple(model_ids.values()))
    bad_ids = []
    dl_manager = tfds.core.download.download_manager.DownloadManager(
        download_dir=os.path.join(tfds.core.constants.DATA_DIR, 'downloads'),
        register_checksums=True)
    with base.mesh_loader_context(synset_id, dl_manager=dl_manager) as loader:
        for model_id in tqdm.tqdm(model_ids, desc='%s: ' % synset_id):
            try:
                mesh_or_scene = loader[model_id]
                if (
                        isinstance(mesh_or_scene, trimesh.Scene) and
                        len(tuple(mesh_or_scene.geometry)) == 0):
                    raise Exception('empty mesh')

            except Exception:
                logging.info('Bad id: %s' % model_id)
                bad_ids.append(model_id)
    logging.info('%d bad ids found for synset %s' % (len(bad_ids), synset_id))
    logging.info(bad_ids)



def main(_):
    ids, _ = base.load_synset_ids()
    model_ids = base.load_split_ids(excluded=None)
    synset = flags.FLAGS.synset
    if synset is None:
        synset_ids = sorted(model_ids)
        for synset_id in synset_ids:
            check_core_models(synset_id, model_ids[synset_id])
    else:
        synset_id = ids.get(synset, synset)
        check_core_models(synset_ids, model_ids[synset_id])




if __name__ == '__main__':
    app.run(main)
