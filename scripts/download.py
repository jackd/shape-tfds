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
flags.DEFINE_boolean('extract', help='whether or not to extract', default=False)


def download(synset_id, extract):
    dl_manager = tfds.core.download.download_manager.DownloadManager(
        download_dir=os.path.join(tfds.core.constants.DATA_DIR, 'downloads'),
        register_checksums=True,
        dataset_name='shapenet_core')

    dl_url = base.DL_URL.format(synset_id=synset_id)
    if extract:
        dl_manager.download_and_extract(dl_url)
    else:
        dl_manager.download(dl_url)


def main(_):
    extract = flags.FLAGS.extract
    synset = flags.FLAGS.synset
    ids, _ = base.load_synset_ids()
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
        for synset_id in synset_ids:
            download(synset_id, extract)
    else:
        synset_id = ids.get(synset, synset)
        download(synset_ids, extract)


if __name__ == '__main__':
    app.run(main)
