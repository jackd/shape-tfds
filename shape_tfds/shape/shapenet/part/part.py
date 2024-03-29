import json
import os

import numpy as np
import tensorflow as tf

import tensorflow_datasets.public_api as tfds
from shape_tfds.shape.shapenet.core.base import SHAPENET_URL, load_synset_ids
from tensorflow_datasets.core import utils as core_utils

NUM_OBJECT_CLASSES = 16
NUM_PART_CLASSES = 50

# labels for class i are range(LABEL_SPLITS[i], LABEL_SPLITS[i+1])
LABEL_SPLITS = (
    0,
    4,
    6,
    8,
    12,
    16,
    19,
    22,
    24,
    28,
    30,
    36,
    38,
    41,
    44,
    47,
    50,
)

POINT_CLASS_FREQ = (
    2275419,
    1664032,
    642156,
    460568,
    9715,
    138138,
    75487,
    26726,
    97844,
    131510,
    288766,
    1302657,
    2430900,
    2989559,
    1510814,
    260298,
    74972,
    31482,
    15683,
    109676,
    246091,
    942116,
    298731,
    289889,
    357113,
    1535899,
    30956,
    506672,
    479080,
    415252,
    15960,
    13613,
    80202,
    5217,
    2217,
    223851,
    22325,
    343634,
    359971,
    164593,
    30528,
    76608,
    19519,
    12561,
    29492,
    218704,
    18829,
    7729973,
    2395978,
    317260,
)


def part_class_indices(object_class_index):
    return range(
        LABEL_SPLITS[object_class_index], LABEL_SPLITS[object_class_index + 1],
    )


_CITATION = """\
@article{yi2017large,
    title={Large-scale 3d shape reconstruction and segmentation from shapenet core55},
    author={Yi, Li and Shao, Lin and Savva, Manolis and Huang, Haibin and Zhou, Yang and Wang, Qirui and Graham, Benjamin and Engelcke, Martin and Klokov, Roman and Lempitsky, Victor and others},
    journal={arXiv preprint arXiv:1710.06104},
    year={2017}
}"""

ICCV2017_URL = "https://shapenet.cs.stanford.edu/iccv17/"

PART_SYNSET_IDS = (
    "02691156",
    "02773838",
    "02954340",
    "02958343",
    "03001627",
    "03261776",
    "03467517",
    "03624134",
    "03636649",
    "03642806",
    "03790512",
    "03797390",
    "03948459",
    "04099429",
    "04225987",
    "04379243",
)

_part_synset_index = {k: i for i, k in enumerate(PART_SYNSET_IDS)}


class ShapenetPart2017Config(tfds.core.BuilderConfig):
    def __init__(
        self,
        name_prefix="base",
        version=core_utils.Version("0.0.1"),
        description=("point cloud segmentation dataset for iccv2017 challenge"),
        synset=None,
    ):
        if synset is None:
            synset_index = None
            synset_id = None
            synset = None
            name = name_prefix
        else:
            ids, names = load_synset_ids()
            if isinstance(synset, int):
                synset_index = synset
                synset_id = _part_synset_index[synset_index]
                synset = names[synset_id]
            else:
                synset = synset.lower()
                if synset in ids:
                    synset_id = ids[synset]
                    synset_index = _part_synset_index[synset_id]
                elif synset in names:
                    synset_id = synset
                    synset = names[synset]
                    synset_index = _part_synset_index[synset_id]
                else:
                    raise ValueError(f"Unrecognized synset {synset}")

            name = f"{name_prefix}-{synset_id}"
        self.synset = synset
        self.synset_id = synset_id
        self.synset_index = synset_index
        description = f"{description} ({synset})"
        super(ShapenetPart2017Config, self).__init__(
            name=name, version=version, description=description
        )

    @property
    def cloud_features(self):
        if self.synset_index is None:
            num_part_classes = NUM_PART_CLASSES
        else:
            num_part_classes = (
                LABEL_SPLITS[self.synset_index + 1] - LABEL_SPLITS[self.synset_index]
            )
        return tfds.features.Sequence(
            {
                "labels": tfds.features.ClassLabel(num_classes=num_part_classes),
                "positions": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                "normals": tfds.features.Tensor(shape=(3,), dtype=tf.float32),
            }
        )

    def map_cloud(self, cloud):
        return cloud


base_config = ShapenetPart2017Config()


class ShapenetPart2017(tfds.core.GeneratorBasedBuilder):
    URLS = [SHAPENET_URL, ICCV2017_URL]
    _DL_URL = (
        "https://shapenet.cs.stanford.edu/media/"
        "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
    )

    BUILDER_CONFIGS = [base_config]

    def _info(self):
        cloud = self.builder_config.cloud_features
        label = tfds.features.ClassLabel(names=PART_SYNSET_IDS)
        example_id = tfds.features.Text()
        features = tfds.features.FeaturesDict(
            {"label": label, "example_id": example_id, "cloud": cloud,}
        )

        description = "ICCV2017 point cloud segmentation challenge"

        return tfds.core.DatasetInfo(
            builder=self,
            description=description,
            features=features,
            citation=_CITATION,
            supervised_keys=("cloud", "label"),
            homepage=ICCV2017_URL,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(self._DL_URL)
        data_dir = os.path.join(
            data_dir, "shapenetcore_partanno_segmentation_benchmark_v0_normal"
        )
        split_dir = os.path.join(data_dir, "train_test_split")

        out = {}
        for split, key in (
            (tfds.Split.TRAIN, "train"),
            (tfds.Split.VALIDATION, "val"),
            (tfds.Split.TEST, "test"),
        ):
            split_path = os.path.join(split_dir, f"shuffled_{key}_file_list.json")
            out[split] = self._generate_examples(
                split_path=split_path, data_dir=data_dir
            )
        return out

    def _generate_examples(self, split_path, data_dir):
        config_synset_id = self.builder_config.synset_id
        config_synset_index = self.builder_config.synset_index
        label_offset = (
            0 if config_synset_index is None else LABEL_SPLITS[config_synset_index]
        )

        with tf.io.gfile.GFile(split_path, "rb") as fp:
            subpaths = json.load(fp)
        for subpath in subpaths:
            synset_id, example_id = subpath.split("/")[1:]
            if config_synset_id is not None and synset_id != config_synset_id:
                continue
            path = os.path.join(data_dir, synset_id, f"{example_id}.txt")
            with tf.io.gfile.GFile(path, "rb") as fp:
                data = np.loadtxt(fp, dtype=np.float32)
            # pylint: disable=unbalanced-tuple-unpacking
            (positions, normals, labels) = np.split(data, (3, 6), axis=1)
            # pylint: enable=unbalanced-tuple-unpacking
            labels = np.squeeze(labels, axis=-1)
            yield f"{synset_id}-{example_id}", dict(
                cloud=self.builder_config.map_cloud(
                    dict(
                        positions=positions,
                        normals=normals,
                        labels=labels.astype(np.int64) - label_offset,
                    )
                ),
                label=synset_id,
                example_id=example_id,
            )


if __name__ == "__main__":
    config = tfds.core.download.DownloadConfig(verify_ssl=False)
    builder = ShapenetPart2017()
    builder.download_and_prepare(download_config=config)

    ds: tf.data.Dataset = builder.as_dataset(split="train").map(
        lambda kwargs: tf.shape(kwargs["cloud"]["positions"])[0]
    )
    import matplotlib.pyplot as plt

    plt.hist(list(ds), bins=20)
    plt.show()
    # for el in ds:
    #     print(el.numpy())

    # print(
    #     ds.reduce(
    #         (tf.constant(int(1e6)), tf.constant(-int(1e6))),
    #         lambda old_state, element: (
    #             tf.minimum(old_state[0], element),
    #             tf.maximum(old_state[1], element),
    #         ),
    #     )
    # )
