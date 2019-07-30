from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import contextlib
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.download import resource as resource_lib

from shape_tfds.core import mapping as shape_mapping
from shape_tfds.core.downloads import get_dl_manager
from collection_utils import mapping

_URL_BASE = "http://modelnet.cs.princeton.edu/"

MODELNET_CITATION = """\
@inproceedings{wu20153d,
  title={3d shapenets: A deep representation for volumetric shapes},
  author={Wu, Zhirong and Song, Shuran and Khosla, Aditya and Yu, Fisher and Zhang, Linguang and Tang, Xiaoou and Xiao, Jianxiong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1912--1920},
  year={2015}
}
"""

MODELNET_ALIGNED_CITATION = """\
@InProceedings{SB15,
  author       = "N. Sedaghat and T. Brox",
  title        = "Unsupervised Generation of a Viewpoint Annotated Car Dataset from Videos",
  booktitle    = "IEEE International Conference on Computer Vision (ICCV)",
  year         = "2015",
  url          = "http://lmb.informatik.uni-freiburg.de/Publications/2015/SB15"
}"""


def naive_bounding_sphere(points):
    center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2
    radius = np.max(np.linalg.norm(points - center, axis=-1))
    return radius, center


def get_modelnet10_data_dir(dl_manager=None):
    dl_manager = dl_manager or get_dl_manager(dataset_name='modelnet10')
    folder = dl_manager.download_and_extract(
        "http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip")
    return os.path.join(folder, 'ModelNet10')


def get_modelnet40_data_dir(dl_manager=None):
    dl_manager = dl_manager or get_dl_manager(dataset_name='modelnet40')
    folder = dl_manager.download_and_extract(
        "http://modelnet.cs.princeton.edu/ModelNet40.zip")
    return os.path.join(folder, 'ModelNet40')


def get_modelnet40_aligned_data_dir(dl_manager=None):
    dl_manager = dl_manager or get_dl_manager(
        dataset_name='modelnet40_aligned')
    path = dl_manager.download(
        "https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar")
    folder = dl_manager.extract(resource_lib.Resource(
        path=path, extract_method=resource_lib.ExtractMethod.TAR_GZ))
    return folder


def get_class_names_path(num_classes):
    return os.path.join(
        os.path.dirname(__file__), 'class_names%d.txt' % num_classes)


def load_class_names(num_classes):
    path = get_class_names_path(num_classes)
    with tf.io.gfile.GFile(path, 'rb') as fp:
        return fp.read().decode('utf-8').split('/')[:-1]


def _load_split_paths(root_dir):
    out_paths = {'train': [], 'test': []}
    n = len(root_dir) + 1
    for dirname, _, fns in os.walk(root_dir):
        paths = tuple(os.path.join(dirname[n:], fn)
                      for fn in fns if fn.endswith('.off'))
        if len(paths) > 0:
            out_paths[os.path.split(dirname)[1]].extend(paths)
    return out_paths


class ModelnetConfig(shape_mapping.MappingConfig):
    def __init__(self, data_dir_fn, **kwargs):
        self._data_dir_fn = data_dir_fn
        super(ModelnetConfig, self).__init__(**kwargs)

    def base_mapping(self, dl_manager=None):
        import trimesh
        root_dir = self._data_dir_fn(dl_manager)

        def map_fn(path):
            mesh = trimesh.load(path)
            radius, center = naive_bounding_sphere(mesh.vertices)
            mesh.vertices -= center
            mesh.vertices /= (2*radius)
            return mesh

        return mapping.MappedMapping(
            shape_mapping.DeepDirectoryMapping(root_dir),
            map_fn)



class Modelnet(shape_mapping.MappingBuilder):
    @abc.abstractproperty
    def num_classes(self):
        raise NotImplementedError

    @property
    def key(self):
        return 'subpath'

    @property
    def key_feature(self):
        return tfds.core.features.Text()

    @property
    def citation(self):
        return MODELNET_CITATION

    @property
    def urls(self):
        return [_URL_BASE]

    def _load_split_keys(self, dl_manager):
        return _load_split_paths(self._get_data_dir(dl_manager))

    @abc.abstractmethod
    def _get_data_dir(self, dl_manager):
        raise NotImplementedError


class Modelnet10(Modelnet):
    @property
    def num_classes(self):
        return 10

    def _get_data_dir(self, dl_manager):
        return get_modelnet10_data_dir(dl_manager)

    @property
    def urls(self):
        return [_URL_BASE]


class Modelnet40(Modelnet):
    @property
    def num_classes(self):
        return 40

    def _get_data_dir(self, dl_manager):
        return get_modelnet40_data_dir(dl_manager)


class Modelnet40Aligned(Modelnet):
    @property
    def num_classes(self):
        return 40

    def _get_data_dir(self, dl_manager):
        return get_modelnet40_aligned_data_dir(dl_manager)

    @property
    def urls(self):
        return [_URL_BASE, "https://github.com/lmb-freiburg/orion"]

    @property
    def citation(self):
        return MODELNET_ALIGNED_CITATION


if __name__ == '__main__':
    # dl_manager = get_dl_manager(
    #     dataset_name='modelnet10', register_checksums=True)
    # print('*****')
    # print(get_modelnet10_data_dir(dl_manager))
    # print('*****')

    # dl_manager = get_dl_manager(
    #     dataset_name='modelnet40', register_checksums=True)
    # print('*****')
    # print(get_modelnet40_data_dir(dl_manager))
    # print('*****')

    # dl_manager = get_dl_manager(
    #     dataset_name='modelnet40_aligned', register_checksums=True)
    # print('*****')
    # print(get_modelnet40_aligned_data_dir(dl_manager))
    # print('*****')
    import random
    config = ModelnetConfig(
        get_modelnet40_data_dir, name='test', version='0.0.1')
    base_mapping = config.base_mapping()
    keys = list(base_mapping.keys())
    random.shuffle(keys)
    for k in keys:
        print(k)
        base_mapping[k].show()
