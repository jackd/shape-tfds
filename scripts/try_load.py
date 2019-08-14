from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import os

import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet.core import base
import trimesh
import numpy as np

DEFAULT_LOG_PATH = '/tmp/shape-tfds/bad_ids/{synset_id}'

flags.DEFINE_string('synset', help='synset name or id', default=None)
flags.DEFINE_multi_string('example_ids', default=None, help='ids of examples')
flags.DEFINE_boolean('counts', help='print counts for each model', default=True)
flags.DEFINE_boolean('show', help='should show each model', default=True)


def main(_):
    import numpy as np
    try:
        import setproctitle
        setproctitle.setproctitle('shape-tfds/try_load')
    except ImportError:
        pass
    ids, _ = base.load_synset_ids()
    synset = flags.FLAGS.synset
    synset_id = ids.get(synset, synset)
    example_ids = [eid.split(',') for eid in flags.FLAGS.example_ids]
    example_ids = np.concatenate(example_ids)
    dl_manager = None
    # dl_manager = tfds.core.download.download_manager.DownloadManager(
    #     download_dir=os.path.join(tfds.core.constants.DATA_DIR, 'downloads'),
    #     register_checksums=True,
    #     dataset_name='shapenet_core')
    # print(base.extracted_mesh_paths(synset_id)[example_ids[0]])
    # exit()

    with base.zipped_mesh_loader_context(synset_id,
                                         dl_manager=dl_manager) as loader:
        for example_id in example_ids:
            try:
                mesh_or_scene = loader[example_id]
                logging.info('Sucessfully loaded {}/{}'.format(
                    synset_id, example_id))
                if flags.FLAGS.counts:
                    if isinstance(mesh_or_scene, trimesh.Scene):
                        geoms = list(mesh_or_scene.geometry.values())
                    else:
                        geoms = [mesh_or_scene]
                    nv = sum(len(g.vertices) for g in geoms)
                    nf = sum(len(g.faces) for g in geoms)
                    ng = len(geoms)
                    logging.info('{} geometries, {} vertices, {} faces'.format(
                        ng, nv, nf))
                if flags.FLAGS.show:
                    mesh_or_scene.show()
            except Exception:
                logging.info('Failed to load model {}/{}'.format(
                    synset_id, example_id))
                raise


if __name__ == '__main__':
    app.run(main)
