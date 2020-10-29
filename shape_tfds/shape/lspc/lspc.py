"""Large-Scale Point Cloud Classification Benchmark (semantic segmentation)."""
import os
from tempfile import TemporaryDirectory
from typing import Mapping

import numpy as np
import pandas  # pandas.read_csv is much faster than np.loadtxt
import py7zr
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """\
Large-Scale Point Cloud Classification Benchmark is a point-cloud semantic-segmentation
dataset from ETH Zurich, mapping (x, y, z, intensity, R, G, B) values to one of 8
classes (plus an optional "unlabeled" class which should not be used for training).

The full training and test sets have 15 scenes each with tens of millions of points in
each.
"""

_HOMEPAGE_URL = "http://www.semantic3d.net/"

_CITATION = """\
@inproceedings{hackel2017isprs,
   title={{SEMANTIC3D.NET: A new large-scale point cloud classification benchmark}},
   author={Timo Hackel and N. Savinov and L. Ladicky and Jan D. Wegner and K. Schindler and M. Pollefeys},
   booktitle={ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
   year = {2017},
   volume = {IV-1-W1},
   pages = {91--98}
 }
"""

# original labels have "unlabeled" first, but we strip these out
CLASS_LABELS = (
    # "unlabeled",
    "man-made terrain",
    "natural terrain",
    "high vegetation",
    "low vegetation",
    "buildings",
    "hard scape",
    "scanning artefacts",
    "cars",
)

_TRAIN_SCENES = (
    "bildstein_station1_xyz",
    "bildstein_station3_xyz",
    "bildstein_station5_xyz",
    "domfountain_station1_xyz",
    "domfountain_station2_xyz",
    "domfountain_station3_xyz",
    "neugasse_station1_xyz",
    "sg27_station1",
    "sg27_station2",
    "sg27_station4",
    "sg27_station5",
    "sg27_station9",
    "sg28_station4",
    "untermaederbrunnen_station1_xyz",
    "untermaederbrunnen_station3_xyz",
)

_TEST_SCENES = (
    "birdfountain_station1_xyz",
    "castleblatten_station1",
    "castleblatten_station5_xyz",
    "marketplacefeldkirch_station1",
    "marketplacefeldkirch_station4",
    "marketplacefeldkirch_station7",
    "sg27_station10",
    "sg27_station3",
    "sg27_station6",
    "sg27_station8",
    "sg28_station2",
    "sg28_station5_xyz",
    "stgallencathedral_station1",
    "stgallencathedral_station3",
    "stgallencathedral_station6",
)

_REDUCED_TEST_SCENES = (
    "MarketplaceFeldkirch_Station4",
    "StGallenCathedral_station6",
    "sg27_station10",
    "sg28_Station2",
)


class LSPCFull(tfds.core.BuilderConfig):
    def __init__(self):
        description = (
            f"{_DESCRIPTION}\n\nThis config provides the full clouds for all examples."
        )
        super().__init__(
            name="full", version=tfds.core.Version("0.0.1"), description=description
        )


full = LSPCFull()


class LSPC(tfds.core.GeneratorBasedBuilder):
    """
    Full Large Scale Point Cloud Classification Benchmark datasets.

    Note there are very few examples in each split:
        * train: 15
        * test: 15
        * reduced_test: 4

    However, each example has many millions - up to 250 million - points. Effort has
    been made to reduce RAM usage during serialization though it still requires 20+gb.
    """

    BUILDER_CONFIGS = [full]

    def _info(self):

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "positions": tfds.features.Tensor(
                        shape=(None, 3), dtype=tf.float32
                    ),
                    "intensities": tfds.features.Tensor(shape=(None,), dtype=tf.int64),
                    "colors": tfds.features.Tensor(shape=(None, 3), dtype=tf.uint8),
                    # "labels": tfds.features.ClassLabel(names=_CLASS_LABELS),
                    "labels": tfds.features.Tensor(shape=(None,), dtype=tf.int64),
                    "scene": tfds.features.Text(),
                    "limits": tfds.features.Tensor(shape=(2, 3), dtype=tf.float32),
                }
            ),
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_url = "http://www.semantic3d.net/data"
        labels = os.path.join(data_url, "sem8_labels_training.7z")
        download_url = os.path.join(
            data_url, "point-clouds/{split}/{scene}_intensity_rgb.7z"
        )
        reduced_download_url = os.path.join(
            data_url, "point-clouds/testing2/{}_rgb_intensity-reduced.txt.7z"
        )
        train_features = {
            scene: download_url.format(split="training1", scene=scene)
            for scene in _TRAIN_SCENES
        }
        test_features = {
            scene: download_url.format(split="testing1", scene=scene)
            for scene in _TEST_SCENES
        }
        reduced_test_features = {
            scene: reduced_download_url.format(scene) for scene in _REDUCED_TEST_SCENES
        }

        labels = "http://www.semantic3d.net/data/sem8_labels_training.7z"

        (
            train_features,
            labels,
            test_features,
            reduced_test_features,
        ) = dl_manager.download(
            (train_features, labels, test_features, reduced_test_features)
        )

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(feature_files=train_features, labels_path=labels),
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST, gen_kwargs=dict(feature_files=test_features),
            ),
            tfds.core.SplitGenerator(
                name="reduced_test",
                gen_kwargs=dict(feature_files=reduced_test_features),
            ),
        ]

    def _generate_examples(self, feature_files: Mapping[str, str], labels_path=None):
        if self.builder_config is not full:
            raise NotImplementedError()
        # TODO: resolve na_filter issues?
        # read_kwargs = dict(na_filter=False, header=None)
        read_kwargs = dict(header=None)

        def load_labels(k):
            # repeatedly opens and closes archive, but uses less memory
            if labels_path is None:
                return np.zeros((0,), dtype=np.int64)
            with py7zr.SevenZipFile(labels_path, "r") as fp:
                # (data,) = fp.read([f"{k}_intensity_rgb.labels"])
                data = fp.readall()[f"{k}_intensity_rgb.labels"]
                labels = pandas.read_csv(
                    data, names=("labels",), dtype=np.int64, **read_kwargs
                ).labels
            return labels

        def load_features(k, mask=None):
            def split(x):
                return x[:, :3], x[:, 3], x[:, 4:7]

            # extract to disk to reduce memory usage
            chunksize = int(1e6)
            with py7zr.SevenZipFile(feature_files[k], "r") as fp:
                with TemporaryDirectory() as tmp_dir:
                    fp.extractall(tmp_dir)
                    path = os.path.join(tmp_dir, os.listdir(tmp_dir)[0])
                    # pandas.read_csv is much faster than np.loadtxt
                    if mask is None:
                        features = pandas.read_csv(
                            path, sep=" ", dtype=np.float32, **read_kwargs
                        ).to_numpy()
                        positions, intensities, colors = split(features)
                        del features
                        intensities = intensities.astype(np.int64)
                        colors = colors.astype(np.uint8)
                    else:
                        # read in chunks so we don't accumulate invalid points
                        assert mask.dtype == np.bool
                        num_valid = np.count_nonzero(mask)
                        positions = np.full((num_valid, 3), np.nan, dtype=np.float32)
                        intensities = np.empty((num_valid,), dtype=np.int64)
                        colors = np.empty((num_valid, 3), dtype=np.uint8)
                        start = 0
                        for i, chunk in enumerate(
                            pandas.read_csv(
                                path,
                                sep=" ",
                                dtype=np.float32,
                                chunksize=chunksize,
                                **read_kwargs,
                            )
                        ):
                            chunk = chunk.to_numpy()[
                                mask[i * chunksize : (i + 1) * chunksize]
                            ]
                            end = start + chunk.shape[0]
                            (
                                positions[start:end],
                                intensities[start:end],
                                colors[start:end],
                            ) = split(chunk)
                            start = end
                        assert end == num_valid
                    assert not np.any(np.isnan(positions))
            return (positions, intensities, colors)

        for k in sorted(feature_files.keys()):
            labels = load_labels(k)
            mask = None if labels.size == 0 else labels != 0
            positions, intensities, colors = load_features(k, mask)
            # bbox = tfds.features.BBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)
            limits = np.array([np.min(positions, axis=0), np.max(positions, axis=0)])
            labels -= 1
            ex = dict(
                positions=positions,
                intensities=intensities,
                colors=colors,
                labels=labels,
                scene=k,
                limits=limits,
            )
            yield (k, tuple(float(x) for x in limits.flatten())), ex


# TODO: add non-full configs

if __name__ == "__main__":
    builder = LSPC()
    dl_config = tfds.core.download.DownloadConfig(register_checksums=True)
    builder.download_and_prepare(download_config=dl_config)
