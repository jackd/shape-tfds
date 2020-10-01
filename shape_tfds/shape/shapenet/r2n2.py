"""Shapenet renderings/voxelizations based on the work of Choy et al.

3D Reconstruction from images.

`meta` values are those released by the authors (See `META_KEYS` for
interpretation). See
[this discussion](https://github.com/chrischoy/3D-R2N2/issues/39)
regarding the consistency/accuracy of the values.
"""

import contextlib
import os

import numpy as np
import six
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import trimesh
from absl import logging
from tensorflow_datasets.core import lazy_imports

import shape_tfds as sds

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
            "\nValid synset `id: name`s:\n%s" % (name, _valid_synsets_string())
        )


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
            "\nValid synset `id: name`s:\n%s" % (id_, _valid_synsets_string())
        )


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
    "display": "03211117",  # monitor
    "plane": "02691156",
    "rifle": "04090263",
    "sofa": "04256520",
    "speaker": "03691459",
    "table": "04379243",
    "telephone": "04401088",
    "watercraft": "04530566",
}

_synset_names = {v: k for k, v in _synset_ids.items()}


def _valid_synsets_string():
    return "\n".join("%s : %s" % (_synset_ids[k], k) for k in sorted(_synset_ids))


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
        assert isinstance(synset, six.string_types)
        self.synset_id = synset_id(synset)
        self.synset_name = synset_name(synset)
        super(ShapenetR2n2Config, self).__init__(
            name=self.synset_id,
            version=tfds.core.Version("0.0.1"),
            description=(
                "Multi-view renderings/voxels for ShapeNet synset %s (%s)"
                % (self.synset_name, self.synset_id)
            ),
        )


def load_meta(renderings_dir, model_id):
    meta_path = os.path.join(
        renderings_dir, model_id, "rendering", "rendering_metadata.txt"
    )
    with tf.io.gfile.GFile(meta_path, "rb") as fp:
        meta = np.loadtxt(fp)
    return meta.astype(np.float32)


def load_voxels(voxels_dir, model_id):
    binvox_path = os.path.join(voxels_dir, model_id, "model.binvox")
    with tf.io.gfile.GFile(binvox_path, mode="rb") as fp:
        voxel = trimesh.load(fp, file_type="binvox")
    return voxel.encoding.dense


def load_image(renderings_dir, model_id, image_index):
    image_path = os.path.join(
        renderings_dir, model_id, "rendering", "%02d.png" % image_index
    )
    with tf.io.gfile.GFile(image_path, "rb") as fp:
        image = np.array(lazy_imports.PIL_Image.open(fp))  # pylint: disable=no-member
    # tfds image features can't have 4 channels.
    background = image[..., -1] == 0
    image = image[..., :3]
    image[background] = BACKGROUND_COLOR
    return image


class ShapenetR2n2(tfds.core.GeneratorBasedBuilder):
    """Builder for rendered/voxelized subset of Shapenet 3D dataset."""

    BUILDER_CONFIGS = [
        ShapenetR2n2Config(synset=synset) for synset in sorted(_synset_ids)
    ]

    def _info(self):
        features = tfds.features.FeaturesDict(
            dict(
                synset_id=tfds.features.ClassLabel(names=synset_ids()),
                model_id=tfds.features.Text(),
                voxels=sds.core.features.BinaryVoxel(VOXEL_SHAPE),
                renderings=tfds.features.Sequence(
                    dict(
                        image=tfds.features.Image(shape=IMAGE_SHAPE),
                        meta=tfds.features.Tensor(shape=(5,), dtype=tf.float32),
                    ),
                    length=RENDERINGS_PER_EXAMPLE,
                ),
            )
        )

        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Shapenet is a large collection of 3D CAD models. "
                "This dataset provides renderings and voxelizations "
                "of a subset of 13 categories as used by Choy et al."
            ),
            features=features,
            supervised_keys=("renderings", "voxels"),
            urls=["http://cvgl.stanford.edu/3d-r2n2/", "https://www.shapenet.org/"],
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # Unfortunately the files at these urls are twice the size they need to be
        # since the archives contain an inner archive containing almost
        # everything in the rest of the outer archive.
        resources = dict(
            voxels="http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz",
            renderings="http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz",
        )

        data_dirs = dl_manager.download_and_extract(resources)
        base_renderings_dir = os.path.join(data_dirs["renderings"], "ShapeNetRendering")
        base_voxels_dir = os.path.join(data_dirs["voxels"], "ShapeNetVox32")

        # We manually delete the inner duplicate archive after extraction
        duplicate_paths = [
            os.path.join(base_renderings_dir, "rendering_only.tgz"),
            os.path.join(base_voxels_dir, "binvox.tgz"),
        ]
        for path in duplicate_paths:
            if tf.io.gfile.exists(path):
                tf.io.gfile.remove(path)

        synset_id = self.builder_config.synset_id
        voxels_dir = os.path.join(base_voxels_dir, synset_id)
        model_ids = tf.io.gfile.listdir(voxels_dir)
        train_ids, test_ids = _get_id_split(model_ids)
        kwargs = dict(
            synset_id=synset_id,
            voxels_dir=voxels_dir,
            renderings_dir=os.path.join(base_renderings_dir, synset_id),
        )

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN, gen_kwargs=dict(model_ids=train_ids, **kwargs)
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST, gen_kwargs=dict(model_ids=test_ids, **kwargs)
            ),
        ]

    def _generate_examples(self, **kwargs):
        gen = self._generate_example_data(**kwargs)
        return (
            (((v["synset_id"], v["model_id"]), v) for v in gen)
            if self.version.implements(tfds.core.Experiment.S3)
            else gen
        )

    def _generate_example_data(self, synset_id, model_ids, voxels_dir, renderings_dir):

        for model_id in model_ids:
            images = [
                load_image(renderings_dir, model_id, i)
                for i in range(RENDERINGS_PER_EXAMPLE)
            ]
            voxels = load_voxels(voxels_dir, model_id)
            meta = load_meta(renderings_dir, model_id)
            yield dict(
                voxels=voxels,
                renderings=dict(image=images, meta=meta),
                model_id=model_id,
                synset_id=synset_id,
            )


@contextlib.contextmanager
def temp_random_state(seed=123):
    orig_state = np.random.get_state()
    np.random.seed(seed)
    yield
    np.random.set_state(orig_state)


class ShapenetR2n2CloudConfig(tfds.core.BuilderConfig):
    def __init__(self, synset, num_points):
        """Create the config object for `ShapenetR2n2Cloud` `DatasetBuilder`.

        Args:
        synset: str, synset name or id
        """
        assert isinstance(synset, six.string_types)
        assert isinstance(num_points, int) and num_points > 0
        self.synset_id = synset_id(synset)
        self.synset_name = synset_name(synset)
        self.num_points = num_points
        super(ShapenetR2n2CloudConfig, self).__init__(
            name="{}-c{}".format(self.synset_id, num_points),
            version=tfds.core.Version("0.0.1"),
            description=(
                "Multi-view renderings/cloud for ShapeNet synset "
                "{} ({}), {} points".format(
                    self.synset_name, self.synset_id, self.num_points
                )
            ),
        )


class ShapenetR2n2Cloud(ShapenetR2n2):
    def _info(self):
        features = tfds.features.FeaturesDict(
            dict(
                synset_id=tfds.features.ClassLabel(names=synset_ids()),
                model_id=tfds.features.Text(),
                points=tfds.features.Tensor(
                    shape=(self.builder_config.num_points, 3), dtype=tf.float32
                ),
                renderings=tfds.features.Sequence(
                    dict(
                        image=tfds.features.Image(shape=IMAGE_SHAPE),
                        meta=tfds.features.Tensor(shape=(5,), dtype=tf.float32),
                    ),
                    length=RENDERINGS_PER_EXAMPLE,
                ),
            )
        )

        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Shapenet is a large collection of 3D CAD models. "
                "This dataset provides renderings and sampled point "
                "clouds of a subset of 13 categories as used by "
                "Choy et al."
            ),
            features=features,
            supervised_keys=("renderings", "points"),
            urls=["http://cvgl.stanford.edu/3d-r2n2/", "https://www.shapenet.org/"],
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        from shape_tfds.shape.shapenet.core.base import cloud_loader_context

        # Unfortunately the files at these urls are twice the size they need to
        # be since the archives contain an inner archive containing almost
        # everything in the rest of the outer archive.
        resources = dict(
            voxels="http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz",
            renderings="http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz",
        )

        data_dirs = dl_manager.download_and_extract(resources)
        base_renderings_dir = os.path.join(data_dirs["renderings"], "ShapeNetRendering")
        base_voxels_dir = os.path.join(data_dirs["voxels"], "ShapeNetVox32")

        # We manually delete the inner duplicate archive after extraction
        duplicate_paths = [
            os.path.join(base_renderings_dir, "rendering_only.tgz"),
            os.path.join(base_voxels_dir, "binvox.tgz"),
        ]
        for path in duplicate_paths:
            if tf.io.gfile.exists(path):
                tf.io.gfile.remove(path)

        synset_id = self.builder_config.synset_id
        voxels_dir = os.path.join(base_voxels_dir, synset_id)
        model_ids = tf.io.gfile.listdir(voxels_dir)
        train_ids, test_ids = _get_id_split(model_ids)

        num_points = self.builder_config.num_points
        kwargs = dict(
            synset_id=synset_id,
            renderings_dir=os.path.join(base_renderings_dir, synset_id),
            loader_context=cloud_loader_context(synset_id, num_points, dl_manager),
        )

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN, gen_kwargs=dict(model_ids=train_ids, **kwargs)
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST, gen_kwargs=dict(model_ids=test_ids, **kwargs)
            ),
        ]

    def _generate_example_data(
        self, synset_id, model_ids, renderings_dir, loader_context
    ):
        with loader_context as loader:
            with temp_random_state():

                for model_id in model_ids:
                    images = [
                        load_image(renderings_dir, model_id, i)
                        for i in range(RENDERINGS_PER_EXAMPLE)
                    ]
                    try:
                        points = loader[model_id]
                    except:
                        points = None
                    if points is None:
                        logging.warning(
                            "Failed to load model {}, skipping".format(model_id)
                        )
                        continue
                    if np.any(np.isnan(points)):
                        logging.warning(
                            "NaN value detected in model {}, skipping".format(model_id)
                        )
                        continue
                    meta = load_meta(renderings_dir, model_id)
                    yield dict(
                        points=points,
                        renderings=dict(image=images, meta=meta),
                        model_id=model_id,
                        synset_id=synset_id,
                    )
