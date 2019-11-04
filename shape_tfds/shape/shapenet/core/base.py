from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import os
import json
import zipfile

from collection_utils.mapping import Mapping
from collection_utils.iterable import single
from shape_tfds.core import mapping as shape_mapping
from shape_tfds.core.downloads import get_dl_manager

_bad_ids = {
    '04090263': (
        '4a32519f44dc84aabafe26e2eb69ebf4',  # empty
    ),
    # '02958343': (
    #     'e23ae6404dae972093c80fb5c792f223',  # too big
    # )  # resolved in fix/objvert
}

id_sets = shape_mapping.ImmutableMapping(
    {'bad_ids': shape_mapping.ImmutableMapping(_bad_ids)})

SHAPENET_CITATION = """\
@article{chang2015shapenet,
    title={Shapenet: An information-rich 3d model repository},
    author={Chang, Angel X and Funkhouser, Thomas and Guibas, Leonidas and
            Hanrahan, Pat and Huang, Qixing and Li, Zimo and
            Savarese, Silvio and Savva, Manolis and Song, Shuran and
            Su, Hao and others},
    journal={arXiv preprint arXiv:1512.03012},
    year={2015}
}
"""

SHAPENET_URL = "https://www.shapenet.org/"


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, all texture information is lost.
    """
    import trimesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    return mesh


def zipped_mesh_loader(zipfile):
    namelist = zipfile.namelist()
    if len(namelist) == 0:
        raise ValueError('No entries in namelist')
    synset_id = namelist[0].split('/')[0]
    keys = set(n.split('/')[1] for n in namelist if n.endswith('.obj'))

    def load_fn(key):
        from shape_tfds.core.resolver import ZipSubdirResolver
        import trimesh
        subdir = os.path.join(synset_id, key)
        resolver = ZipSubdirResolver(zipfile, subdir)
        obj = os.path.join(subdir, 'model.obj')
        with zipfile.open(obj) as fp:
            return trimesh.load(fp, file_type='obj', resolver=resolver)

    return Mapping.mapped(keys, load_fn)


class Openable(object):
    """
    Base class for classes which can be used as context managers.

    Implement `_open` and `_close`.
    """

    def __init__(self):
        self._is_open = False

    def _open(self):
        pass

    def _close(self):
        pass

    @property
    def is_open(self):
        return self._is_open

    def open(self):
        if self.is_open:
            raise ValueError('Already open')
        self._open()
        self._is_open = True

    def close(self):
        if not self.is_open:
            raise ValueError('Already closed')
        self._close()
        self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class MappedFileContext(object):

    def __init__(self, path, map_fn, on_open=None, on_close=None):
        self._path = path
        self._map_fn = map_fn
        self._fp = None
        self._args = None
        self._on_open = on_open
        self._on_close = on_close

    def __enter__(self):
        self._fp = tf.io.gfile.GFile(self._path, mode='rb')
        out, self._arg = self._map_fn(self._fp)
        if self._on_open is not None:
            self._on_open(self._arg)
        return out

    def __exit__(self, *args, **kwargs):
        if self._on_close is not None:
            self._on_close(self._arg)
        self._fp.close()
        self._mapped = None
        self._fp = None


def zipped_mesh_loader_context(synset_id, dl_manager=None, item_map_fn=None):
    """
    Get a mesh loading context.

    Delays downloading relevant zip files until opened.

    Does not extract files.

    Example usage:
    ```python
    loader_context = zipped_mesh_loader_context(synset_id)
    with loader_context as loader:
        # possible download starts
        for key in loader:
            print(key)
            scene = loader[key]  # trimesh.Scene
            scene.show()
    ```
    """

    def file_map_fn(fp):
        loader = zipped_mesh_loader(zipfile.ZipFile(fp))
        if item_map_fn is not None:
            loader = loader.item_map(item_map_fn)
        return loader, None

    return MappedFileContext(get_obj_zip_path(synset_id, dl_manager),
                             map_fn=file_map_fn)


def extracted_mesh_paths(synset_id, dl_manager=None):
    dl_manager = dl_manager or get_dl_manager()
    zip_path = get_obj_zip_path(synset_id, dl_manager)
    root_dir = dl_manager.extract(zip_path)
    synset_dir = os.path.join(root_dir, synset_id)
    assert (tf.io.gfile.isdir(synset_dir))
    model_ids = tf.io.gfile.listdir(synset_dir)
    return Mapping.mapped(
        tuple(model_ids),
        lambda key: os.path.join(synset_dir, key, 'model.obj'))


def load_synset_ids():
    path = os.path.join(os.path.dirname(__file__), 'core_synset.txt')
    synset_ids = {}
    synset_names = {}
    with tf.io.gfile.GFile(path, "rb") as fp:
        for line in fp.readlines():
            if hasattr(line, 'decode'):
                line = line.decode('utf-8')
            line = line.rstrip()
            if line == '':
                continue
            id_, names = line.split('\t')
            names = tuple(names.split(','))
            synset_names[id_] = names
            for n in names:
                synset_ids[n] = id_
    # repeated synset ids
    ambiguous_ids = {'bench': '02828884'}
    for k, v in ambiguous_ids.items():
        assert (k in synset_names[v])
        synset_ids[k] = v
    return synset_ids, synset_names


BASE_URL = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip'
DL_URL = '%s/ShapeNetCore.v1/{synset_id}.zip' % BASE_URL
SPLIT_URL = '%s/SHREC16/all.csv' % BASE_URL
TAXONOMY_URL = '%s/ShapeNetCore.v1/taxonomy.json' % BASE_URL


def _load_taxonomy(path):
    with tf.io.gfile.GFile(path, 'r') as fp:
        return json.load(fp)


def _load_splits_ids(path, excluded):
    """Get a `dict: synset_id -> (dict: split -> model_ids)`."""
    split_dicts = {}
    with tf.io.gfile.GFile(path, "r") as fp:
        fp.readline()  # header
        for line in fp.readlines():
            line = line.rstrip()
            if line == '':
                pass
            record_id, synset_id, sub_synset_id, model_id, split = \
                line.split(',')
            if excluded and model_id in excluded.get(synset_id, ()):
                continue
            del record_id, sub_synset_id
            split_dicts.setdefault(synset_id,
                                   {}).setdefault(split, []).append(model_id)

    for split_ids in split_dicts.values():
        if 'val' in split_ids:
            if 'validation' in split_ids:
                raise RuntimeError('both "val" and "validation" keys found')
            else:
                split_ids['validation'] = split_ids.pop('val', None)

    return split_dicts


def load_split_ids(dl_manager=None, excluded='bad_ids'):
    dl_manager = dl_manager or get_dl_manager()

    return _load_splits_ids(dl_manager.download(SPLIT_URL),
                            None if excluded is None else id_sets[excluded])


def load_taxonomy(dl_manager=None):
    dl_manager = dl_manager or get_dl_manager()
    return _load_taxonomy(dl_manager.download(TAXONOMY_URL))


def get_obj_zip_path(synset_id, dl_manager=None):
    """Get path of zip file containing obj files."""
    dl_manager = dl_manager or get_dl_manager()
    return dl_manager.download(DL_URL.format(synset_id=synset_id))


class ShapenetCoreConfig(shape_mapping.MappingConfig):

    def __init__(self, synset_id, **kwargs):
        self._synset_id = synset_id
        super(ShapenetCoreConfig, self).__init__(**kwargs)

    @property
    def synset_id(self):
        return self._synset_id


class ShapenetCore(shape_mapping.MappingBuilder):

    @property
    def key(self):
        return 'model_id'

    @property
    def key_feature(self):
        return tfds.core.features.Text()

    def _load_split_keys(self, dl_manager):
        return load_split_ids(dl_manager)[self.builder_config.synset_id]

    @property
    def citation(self):
        return SHAPENET_CITATION

    @property
    def urls(self):
        return [SHAPENET_URL]



def sample_triangle(v, n=None):
    if hasattr(n, 'dtype'):
        n = np.asscalar(n)
    if n is None:
        size = v.shape[:-2] + (2,)
    elif isinstance(n, int):
        size = (n, 2)
    elif isinstance(n, tuple):
        size = n + (2,)
    elif isinstance(n, list):
        size = tuple(n) + (2,)
    else:
        raise TypeError('n must be int, tuple or list, got %s' % str(n))
    assert(v.shape[-2] == 2)
    a = np.random.uniform(size=size)
    mask = np.sum(a, axis=-1) > 1
    a[mask] *= -1
    a[mask] += 1
    a = np.expand_dims(a, axis=-1)
    return np.sum(a*v, axis=-2)


def sample_faces(vertices, faces, n_total):
    if len(faces) == 0:
        raise ValueError('Cannot sample points from zero faces.')
    tris = vertices[faces]
    n_faces = len(faces)
    d0 = tris[..., 0:1, :]
    ds = tris[..., 1:, :] - d0
    assert(ds.shape[1:] == (2, 3))
    areas = 0.5 * np.sqrt(np.sum(np.cross(ds[:, 0], ds[:, 1])**2, axis=-1))
    cum_area = np.cumsum(areas)
    cum_area *= (n_total / cum_area[-1])
    cum_area = np.round(cum_area).astype(np.int32)

    positions = []
    last = 0
    for i in range(n_faces):
        n = cum_area[i] - last
        last = cum_area[i]
        if n > 0:
            positions.append(d0[i] + sample_triangle(ds[i], n))
    return np.concatenate(positions, axis=0)


def cloud_loader_context(synset_id, num_points, dl_manager=None):
    import trimesh

    def map_fn(key, value):
        if not isinstance(value, trimesh.Trimesh):
            meshes = [
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in value.geometry.values()]
            value = trimesh.util.concatenate(meshes)

        # trimesh sample seems to have issues?
        # return sample_surface(value, num_points)[0].astype(np.float32)
        return sample_faces(
            value.vertices, value.faces, num_points).astype(np.float32)

    return zipped_mesh_loader_context(synset_id, dl_manager, map_fn)
