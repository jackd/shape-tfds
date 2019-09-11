from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.download import resource as resource_lib

from shape_tfds.core import mapping as shape_mapping
from shape_tfds.core.downloads import get_dl_manager
from collection_utils import mapping
from shape_tfds.core.random import random_context
from shape_tfds.core.random import get_random_state
from shape_tfds.core.loaders import load_off

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
    """Get the radius and center of an approximate minimum bounding sphere."""
    center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2
    radius = np.max(np.linalg.norm(points - center, axis=-1))
    return radius, center


def normalize_points(points):
    """
    Center points on the origin and inside a half-unit ball.

    Modification is performed in-place.
    """
    radius, center = naive_bounding_sphere(points)
    points -= center
    points /= 2 * radius


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
    dl_manager = dl_manager or get_dl_manager(dataset_name='modelnet40_aligned')
    path = dl_manager.download(
        "https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar"
    )
    folder = dl_manager.extract(
        resource_lib.Resource(path=path,
                              extract_method=resource_lib.ExtractMethod.TAR_GZ))
    return folder


def get_data_dir_fn(name='modelnet40'):
    return {
        'modelnet10': get_modelnet10_data_dir,
        'modelnet40': get_modelnet40_data_dir,
        'modelnet40_aligned': get_modelnet40_aligned_data_dir,
    }[name]


def get_class_names_path(num_classes):
    return os.path.join(os.path.dirname(__file__),
                        'class_names%d.txt' % num_classes)


def load_class_freq(num_classes=40):
    """Class frequency in train split."""
    path = os.path.join(os.path.dirname(__file__), 'class_freq.txt')
    freqs = np.loadtxt(path, dtype=np.int64)
    if num_classes == 40:
        return freqs
    elif num_classes == 10:
        n10, n40 = (load_class_names(n) for n in (10, 40))
        indices40 = {k: i for i, k in enumerate(n40)}
        return freqs[[indices40[k] for k in n10]]
    else:
        raise ValueError(
            'num_classes must be 10 or 40, got {}'.format(num_classes))


def load_class_names(num_classes):
    path = get_class_names_path(num_classes)
    with tf.io.gfile.GFile(path, 'rb') as fp:
        data = fp.read().decode('utf-8')
        return data.split('\n')[:-1]


def _load_split_paths(root_dir):
    out_paths = {'train': [], 'test': []}
    n = len(root_dir) + 1
    for dirname, _, fns in os.walk(root_dir):
        paths = tuple(
            os.path.join(dirname[n:], fn) for fn in fns if fn.endswith('.off'))
        if len(paths) > 0:
            out_paths[os.path.split(dirname)[1]].extend(paths)
    return out_paths


class ModelnetConfig(tfds.core.BuilderConfig):

    @abc.abstractproperty
    def feature_item(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_example(self, off_path):
        raise NotImplementedError


class CloudConfig(ModelnetConfig):

    def __init__(self, num_points, name=None, version="0.0.1", **kwargs):
        self._num_points = num_points
        if name is None:
            name = 'cloud-%d' % num_points
        super(CloudConfig, self).__init__(name=name, version=version, **kwargs)

    @property
    def up_dim(self):
        return 2

    @property
    def num_points(self):
        return self._num_points

    @property
    def feature_item(self):
        return 'positions', tfds.features.Tensor(shape=(self.num_points, 3),
                                                 dtype=tf.float32)

    def load_example(self, path):
        import trimesh
        with tf.io.gfile.GFile(path, 'rb') as fp:
            mesh_kwargs = load_off(fp)
        vertices = mesh_kwargs['vertices']
        normalize_points(vertices)
        mesh = trimesh.Trimesh(**mesh_kwargs)
        with random_context(get_random_state(path)):
            positions, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return positions.astype(np.float32)


class CloudNormalConfig(ModelnetConfig):

    def __init__(self, num_points, name=None, version="0.0.1", **kwargs):
        self._num_points = num_points
        if name is None:
            name = 'cloud_normals-%d' % num_points
        super(CloudNormalConfig, self).__init__(name=name,
                                                version=version,
                                                **kwargs)

    @property
    def num_points(self):
        return self._num_points

    @property
    def feature_item(self):
        return 'cloud', {
            'positions':
                tfds.features.Tensor(shape=(self.num_points, 3),
                                     dtype=tf.float32),
            'normals':
                tfds.features.Tensor(shape=(self.num_points, 3),
                                     dtype=tf.float32)
        }

    def load_example(self, path):
        import trimesh
        with tf.io.gfile.GFile(path, 'rb') as fp:
            mesh_kwargs = load_off(fp)
        vertices = mesh_kwargs['vertices']
        normalize_points(vertices)
        mesh = trimesh.Trimesh(**mesh_kwargs)
        with random_context(get_random_state(path)):
            positions, face_indices = trimesh.sample.sample_surface(
                mesh, self.num_points)
            normals = mesh.face_normals[face_indices]
        return dict(positions=positions.astype(np.float32),
                    normals=normals.astype(np.float32))


class UniformDensityCloudNormalConfig(CloudNormalConfig):
    """Positions and normals of points with roughly uniform point density."""

    def __init__(self, num_points, k=20, r0=0.1, name=None, **kwargs):
        if name is None:
            name = 'uniform_density_cloud_normals-%d-%d-%d' % (k, 100 * r0,
                                                               num_points)
        self._k = k
        self._r0 = r0
        super(UniformDensityCloudNormalConfig,
              self).__init__(name=name, num_points=num_points, **kwargs)

    @property
    def k(self):
        return self._k

    @property
    def r0(self):
        return self._r0

    def load_example(self, path):
        from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
        cloud = super(UniformDensityCloudNormalConfig, self).load_example(path)
        positions = cloud['positions']
        tree = cKDTree(positions)
        dists, _ = tree.query(tree.data, self.k)
        recip_scale_factor = self.r0 / np.mean(dists[:, -1])
        positions *= recip_scale_factor
        return cloud


class Modelnet(tfds.core.GeneratorBasedBuilder):

    @property
    def up_dim(self):
        return self.builder_config.up_dim

    @abc.abstractproperty
    def num_classes(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_data_dir(self, dl_manager):
        raise NotImplementedError

    @property
    def citation(self):
        return MODELNET_CITATION

    @property
    def urls(self):
        return [_URL_BASE]

    def raw_data_dir(self, dl_manager=None):
        if not hasattr(self, '_raw_data_dir'):
            if dl_manager is None:
                dl_manager = self._make_download_manager(
                    download_dir=None,
                    download_config=tfds.core.download.DownloadConfig())
            self._raw_data_dir = self._get_data_dir(dl_manager)
        return self._raw_data_dir

    def paths(self, root_dir, split, suffix='.off'):
        for class_name in tf.io.gfile.listdir(root_dir):
            subdir = os.path.join(root_dir, class_name, split)
            for fn in os.listdir(subdir):
                if fn.endswith(suffix):
                    yield os.path.join(subdir, fn)

    def _info(self):
        feature_key, feature_value = self.builder_config.feature_item
        features = tfds.core.features.FeaturesDict({
            'label':
                tfds.features.ClassLabel(
                    names_file=get_class_names_path(self.num_classes)),
            'example_index':
                tfds.features.Tensor(shape=(), dtype=tf.int64),
            feature_key:
                feature_value
        })
        return tfds.core.DatasetInfo(
            builder=self,
            citation=self.citation,
            supervised_keys=(feature_key, 'label'),
            urls=self.urls,
            features=features,
        )

    def _split_generators(self, dl_manager):
        root_dir = self.raw_data_dir(dl_manager)
        gens = []
        for split in ('train', 'test'):
            paths = tuple(self.paths(root_dir, split, suffix='.off'))
            gen = tfds.core.SplitGenerator(name=split,
                                           num_shards=len(paths) // 500 + 1,
                                           gen_kwargs=dict(paths=paths))
            gen.split_info.statistics.num_examples = len(paths)
            gens.append(gen)
        return gens

    def _generate_examples(self, paths):
        config = self.builder_config
        feature_key = config.feature_item[0]
        assert (self.version.implements(tfds.core.Experiment.S3))
        for path in paths:
            class_name, _, filename = path.split('/')[-3:]
            n = len(class_name)
            example_index = int(filename[n + 1:n + 5])
            feature_values = config.load_example(path)
            key = (class_name, example_index)
            value = {
                'example_index': example_index,
                'label': class_name,
                feature_key: feature_values,
            }
            yield key, value


class Modelnet10(Modelnet):

    @property
    def num_classes(self):
        return 10

    def _get_data_dir(self, dl_manager):
        return get_modelnet10_data_dir(dl_manager)


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
    # import random

    # import tqdm
    # from shape_tfds.core.loaders import load_off
    # config = CloudNormalConfig(
    #     'modelnet40_aligned', name='test', version='0.0.1')
    # base_mapping = config.base_mapping()
    # keys = list(base_mapping.keys())
    # keys.sort()
    # # random.shuffle(keys)
    # bad_ids = []
    # for k in tqdm.tqdm(keys):
    #     try:
    #         path = base_mapping[k]
    #         with tf.io.gfile.GFile(path, 'rb') as fp:
    #             mesh_kwargs = load_off(fp)
    #     except Exception:
    #         print('bad file at %s' % k)
    #         bad_ids.append(k)
    #     # trimesh.Trimesh(**mesh_kwargs).show()

    # if len(bad_ids) == 0:
    #     print('Successfully loaded all ids')
    # else:
    #     print('bad ids:')
    #     for b in bad_ids:
    #         print(b)

    from mayavi import mlab

    _dim = {'x': 0, 'y': 1, 'z': 2}

    def permute_xyz(x, y, z, order='xyz'):
        data = (x, y, z)
        return tuple(data[_dim[k]] for k in order)

    def vis_point_cloud(points, axis_order='xyz', value=None, **kwargs):
        data = permute_xyz(*points.T, order=axis_order)
        if value is not None:
            data = data + (value,)
        mlab.points3d(*data, **kwargs)

    def vis_normals(positions, normals, axis_order='xyz', **kwargs):
        x, y, z = permute_xyz(*positions.T, order=axis_order)
        u, v, w = permute_xyz(*normals.T, order=axis_order)
        mlab.quiver3d(x, y, z, u, v, w, **kwargs)

    # config = CloudNormalConfig(1000)
    # builder = Modelnet40(config=config)
    # data_dir = builder.raw_data_dir()
    # for path in builder.paths(data_dir, 'train'):
    #     print(path)
    #     example = config.load_example(path)
    #     mlab.figure()
    #     vis_normals(example['positions'], example['normals'])
    #     vis_point_cloud(example['positions'], scale_factor=0.01)
    #     mlab.show()

    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    # config = CloudNormalConfig(2048)
    config = UniformDensityCloudNormalConfig(2048)
    builder = Modelnet40(config=config)
    builder.download_and_prepare()
    for example in builder.as_dataset(split='train'):
        cloud = example['cloud']
        positions = cloud['positions'].numpy()
        normals = cloud['normals'].numpy()
        mlab.figure()
        vis_normals(positions, normals)
        vis_point_cloud(positions, scale_factor=0.01)
        mlab.show()
