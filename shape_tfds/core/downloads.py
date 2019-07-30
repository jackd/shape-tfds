from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow_datasets as tfds

DOWNLOADS_DIR = os.path.join(tfds.core.constants.DATA_DIR, 'downloads')


def get_dl_manager(download_dir=DOWNLOADS_DIR, **kwargs):
    return tfds.core.download.DownloadManager(
        download_dir=download_dir, **kwargs)
