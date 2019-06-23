"""Shapenet renderings/voxelizations based on the work of Choy et al.

3D Reconstruction from images.

`meta` values are those released by the authors (See `META_KEYS` for
interpretation). See
[this discussion](https://github.com/chrischoy/3D-R2N2/issues/39)
regarding the consistency/accuracy of the values.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import six
import contextlib
import numpy as np
import tensorflow as tf
import itertools

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core import lazy_imports
import shape_tfds as sds
import trimesh

VOXEL_SHAPE = (32, 32, 32)
RENDERINGS_PER_EXAMPLE = 24
IMAGE_SHAPE = (137, 137, 3)
BACKGROUND_COLOR = (255, 255, 255)  # white
META_KEYS = (
    "azimuth",
    "elevation",
    "in_plane_rotation",
    "distance",
    "field_of_view",
)


def synset_ids():
  """Ordered list of synset ids."""
  return sorted(_synset_names)


def synset_names():
  """List of synset names in `synset_id` order."""
  return [_synset_names[c] for c in synset_ids()]


def synset_id(name):
  """Get the synset id for the given id.

  If `name` is not a valid name but is a valid id, returns it instead.

  Args:
    name: string name, e.g. "sofa"

  Returns:
    string synset id, e.g. "04256520"

  Raises:
    ValueError if `name` is not a valid synset id or name.
  """
  if name in _synset_ids:
    return _synset_ids[name]
  elif name in _synset_names:
    return name
  else:
    raise ValueError(
        "'%s' is not a valid synset name or id."
        "\nValid synset `id: name`s:\n%s" % (name, _valid_synsets_string()))


def synset_name(id_):
  """Get the synset name for the given synset id.

  If `id_` is not a valid id_ but is a valid name, returns it instead.

  Args:
    id_: string id, e.g. "04256520"

  Returns:
    string synset name, e.g. "sofa"

  Raises:
    ValueError if `id_` is not a valid synset id or name.
  """
  if id_ in _synset_ids:
    return id_
  elif id_ in _synset_names:
    return _synset_names[id_]
  else:
    raise ValueError(
        "'%s' is not a valid synset name or id."
        "\nValid synset `id: name`s:\n%s" % (id_, _valid_synsets_string()))


_CITATION = """
@inproceedings{choy20163d,
  title={3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction},
  author={Choy, Christopher B and Xu, Danfei and Gwak, JunYoung and Chen, Kevin and Savarese, Silvio},
  booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
  year={2016}
}
"""

_synset_ids = {
    "bench": "02828884",
    "cabinet": "02933112",
    "car": "02958343",
    "chair": "03001627",
    "lamp": "03636649",
    "monitor": "03211117",
    "plane": "02691156",
    "rifle": "04090263",
    "sofa": "04256520",
    "speaker": "03691459",
    "table": "04379243",
    "telephone": "04401088",
    "watercraft": "04530566"
}

_synset_names = {v: k for k, v in _synset_ids.items()}


def _valid_synsets_string():
  return "\n".join('%s : %s' % (_synset_ids[k], k) for k in sorted(_synset_ids))


_TRAIN_FRAC = 0.8


def _get_id_split(example_ids):
    example_ids.sort()
    n_examples = len(example_ids)
    split_index = int(_TRAIN_FRAC * n_examples)
    return example_ids[:split_index], example_ids[split_index:]


class ShapenetR2n2Config(tfds.core.BuilderConfig):
  def __init__(self, synset):
    """Create the config object for `ShapenetR2n2` `DatasetBuilder`.

    Args:
      synset: str, synset name or id
    """
    assert(isinstance(synset, six.string_types))
    self.synset_id = synset_id(synset)
    self.synset_name = synset_name(synset)
    super(ShapenetR2n2Config, self).__init__(
        name=self.synset_id,
        version=tfds.core.Version(0, 0, 1),
        description=(
          "Multi-view renderings/voxels for ShapeNet synset %s (%s)"
          % (self.synset_name, self.synset_id))
    )


class ShapenetR2n2(tfds.core.GeneratorBasedBuilder):
  """Builder for rendered/voxelized subset of Shapenet 3D dataset."""

  BUILDER_CONFIGS = [
      ShapenetR2n2Config(synset=synset) for synset in sorted(_synset_ids)]

  def _info(self):
    features = tfds.features.FeaturesDict(dict(
        synset_id=tfds.features.ClassLabel(names=synset_ids()),
        example_id=tfds.features.Text(),
        voxels=sds.features.PaddedTensor(
          sds.features.BinaryRunLengthEncodedFeature(shape=VOXEL_SHAPE)),
        renderings=tfds.features.Sequence(
          dict(
            image=tfds.features.Image(shape=IMAGE_SHAPE),
            meta=tfds.features.Tensor(shape=(5,), dtype=tf.float32)),
          length=RENDERINGS_PER_EXAMPLE)))

    return tfds.core.DatasetInfo(
        builder=self,
        description=(
            "Shapenet is a large collection of 3D CAD models. "
            "This dataset provides renderings and voxelizations "
            "of a subset of 13 categories as used by Choy et al."),
        features=features,
        supervised_keys=("renderings", "voxels"),
        urls=["http://cvgl.stanford.edu/3d-r2n2/", "https://www.shapenet.org/"],
        citation=_CITATION
    )

  def _split_generators(self, dl_manager):
    from tensorflow_datasets.core.download import resource
    # Unfortunately the files at these urls are twice the size they need to be
    # since the archives contain an inner archive containing almost
    # everything in the rest of the outer archive. 
    resources = dict(
        voxels="http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz",
        renderings="http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz")

    data_dirs = dl_manager.download_and_extract(resources)
    base_renderings_dir = os.path.join(
      data_dirs["renderings"], "ShapeNetRendering")
    base_voxels_dir = os.path.join(data_dirs["voxels"], "ShapeNetVox32")

    # We manually delete the inner duplicate archive after extraction
    duplicate_paths = [
      os.path.join(base_renderings_dir, "rendering_only.tgz"),
      os.path.join(base_voxels_dir, "binvox.tgz")
    ]
    for path in duplicate_paths:
      if tf.io.gfile.exists(path):
        tf.io.gfile.remove(path)

    synset_id = self.builder_config.synset_id
    voxels_dir = os.path.join(base_voxels_dir, synset_id)
    example_ids = tf.io.gfile.listdir(voxels_dir)
    train_ids, test_ids = _get_id_split(example_ids)
    kwargs = dict(
      synset_id=synset_id,
      voxels_dir=voxels_dir,
      renderings_dir=os.path.join(base_renderings_dir, synset_id),
    )

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN, num_shards=len(train_ids) // 1000 + 1,
            gen_kwargs=dict(example_ids=train_ids, **kwargs)),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST, num_shards=len(test_ids) // 1000 + 2,
            gen_kwargs=dict(example_ids=test_ids, **kwargs))
    ]

  def _generate_examples(
      self, synset_id, example_ids, voxels_dir, renderings_dir):

    def load_image(example_id, image_index):
      image_path = os.path.join(
          renderings_dir, example_id, "rendering", "%02d.png" % image_index)
      with tf.io.gfile.GFile(image_path, "rb") as fp:
        image = np.array(lazy_imports.PIL_Image.open(fp))  # pylint: disable=no-member
      # tfds image features can't have 4 channels.
      background = (image[..., -1] == 0)
      image = image[..., :3]
      image[background] = BACKGROUND_COLOR
      return image

    def load_meta(example_id):
      meta_path = os.path.join(
          renderings_dir, example_id, "rendering", "rendering_metadata.txt")
      with tf.io.gfile.GFile(meta_path, "rb") as fp:
        meta = np.loadtxt(fp)
      return meta.astype(np.float32)

    def load_voxels(example_id):
      binvox_path = os.path.join(voxels_dir, example_id, "model.binvox")
      with tf.io.gfile.GFile(binvox_path, mode="rb") as fp:
        voxel = trimesh.load(fp, file_type='binvox')
      encoding, padding = voxel.encoding.stripped
      brle = encoding.binary_run_length_data(dtype=np.int64)
      return dict(
        stripped=(encoding.shape, brle),
        padding=padding)

    for example_id in example_ids:
      images = [
          load_image(example_id, i) for i in range(RENDERINGS_PER_EXAMPLE)]
      voxels = load_voxels(example_id)
      meta = load_meta(example_id)
      yield dict(
        voxels=voxels,
        renderings=dict(image=images, meta=meta),
        example_id=example_id,
        synset_id=synset_id)
