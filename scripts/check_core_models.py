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
from shape_tfds.core.mapping import concat_dict_values
import trimesh
import numpy as np

DEFAULT_LOG_PATH = '/tmp/shape-tfds/bad_ids/{synset_id}'

flags.DEFINE_string('synset', help='synset name or id', default=None)
flags.DEFINE_string('log', help='path to log bad ids', default=DEFAULT_LOG_PATH)
flags.DEFINE_string('example_ids',
                    help='comma separated list of example_ids',
                    default=None)


def check_core_models(synset_id, model_ids=None):
    if model_ids is not None:
        if isinstance(model_ids, dict):
            model_ids = concat_dict_values(model_ids)
    bad_ids = []
    dl_manager = None
    # dl_manager = tfds.core.download.download_manager.DownloadManager(
    #     download_dir=os.path.join(tfds.core.constants.DATA_DIR, 'downloads'),
    #     register_checksums=True,
    #     dataset_name='shapenet_core')

    # paths = base.extracted_mesh_paths(synset_id, dl_manager=dl_manager)
    # paths = base.extracted_mesh_paths(synset_id)
    with base.zipped_mesh_loader_context(synset_id,
                                         dl_manager=dl_manager) as loader:
        if model_ids is None:
            model_ids = tuple(loader.keys())
        for model_id in tqdm.tqdm(model_ids, desc='%s: ' % synset_id):
            try:
                mesh_or_scene = loader[model_id]
                if (isinstance(mesh_or_scene, trimesh.Scene) and
                        len(tuple(mesh_or_scene.geometry)) == 0):
                    raise Exception('empty mesh')

            except Exception:
                logging.info('Bad id: %s' % model_id)
                bad_ids.append(model_id)
        logging.info('%d bad ids found for synset %s' %
                     (len(bad_ids), synset_id))
        logging.info(bad_ids)
        path = flags.FLAGS.log
        if path is not None:
            path = path.format(synset_id=synset_id)
            dirname = os.path.dirname(path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            if os.path.isfile(path):
                os.remove(path)
            if len(bad_ids) > 0:
                with open(path, 'w') as fp:
                    fp.write('\n'.join(bad_ids))
                logging.info('wrote bad ids to {}'.format(path))


def main(_):
    try:
        import setproctitle
        setproctitle.setproctitle('shape-tfds/check_core_models')
    except ImportError:
        pass
    ids, _ = base.load_synset_ids()
    synset = flags.FLAGS.synset
    if synset is None:
        # synset_ids = sorted(model_ids)

        # we do a subset by default
        synsets = (
            "bench",
            "cabinet",
            "car",
            "chair",
            "lamp",
            "display",
            "plane",
            "rifle",
            "sofa",
            "speaker",
            "table",
            "telephone",
            "watercraft",
        )
        synset_ids = tuple(ids[synset] for synset in synsets)
    elif ',' in synset:
        synsets = synset.split(',')
        synset_ids = tuple(ids.get(s, s) for s in synsets)
    elif synset.lower() == 'all':
        synset_ids = sorted(ids.values())
    else:
        synset_ids = ids.get(synset, synset),

    for synset_id in synset_ids:
        check_core_models(synset_id, flags.FLAGS.example_ids)


if __name__ == '__main__':
    app.run(main)
