"""
Dataset discussed in https://hkust-vgd.github.io/scanobjectnn/

Note the dataset is currently undergoing some cleaning and may be different to
the final released version. See project page for details on downloading the
data.
"""
import os
import tempfile

import tensorflow as tf
import tensorflow_datasets as tfds

NUM_POINTS = 2048

CLASS_NAMES = (
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
)

CITATION = """@inproceedings{uy-scanobjectnn-iccv19,
    title = {Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data},
    author = {Mikaela Angelina Uy and Quang-Hieu Pham and Binh-Son Hua and Duc Thanh Nguyen and Sai-Kit Yeung},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year = {2019}
}
"""

URL = "https://hkust-vgd.github.io/scanobjectnn/"


class Variant(object):
    OBJ = "obj"
    PB_T25 = "pb_t25"
    PB_T25_R = "pb_t25_r"
    PB_T50_R = "pb_t50_r"
    PB_T50_RS = "pb_t50_rs"

    @classmethod
    def all(cls):
        return (
            Variant.OBJ,
            Variant.PB_T25,
            Variant.PB_T25_R,
            Variant.PB_T50_R,
            Variant.PB_T50_RS,
        )

    @classmethod
    def validate(cls, x):
        if x not in cls.all():
            raise ValueError(
                "Invalid {}: must be in {}".format(cls.__name__, cls.all())
            )


_suffixes = {
    "obj": "",
    "pb_t25": "_augmented25_norot",
    "pb_t25_r": "_augmented25rot",
    "pb_t50_r": "_augmentedrot",
    "pb_t50_rs": "_augmentedrot_scale75",
}


class ScanObjectNNConfig(tfds.core.BuilderConfig):
    def __init__(self, variant=Variant.OBJ, background=True, split_key="main"):
        Variant.validate(variant)
        self.variant = variant
        self.suffix = _suffixes[variant]
        self.background = background
        self.split = split_key
        if split_key == "main":
            subdir = "main_split"
        elif split_key in (1, 2, 3, 4):
            subdir = "split{}".format(split_key)
        else:
            raise ValueError(
                'split must be one of ("main", 1, 2, 3, 4), got {}'.format(split_key)
            )
        if not background:
            subdir = "{}_nobg".format(subdir)
        self.subdir = subdir
        name = "{}-{}".format(variant, subdir)
        super(ScanObjectNNConfig, self).__init__(
            name=name, version=tfds.core.Version("0.0.1")
        )

    def filename(self, split):
        if split == "train":
            prefix = "training"
        else:
            prefix = "test"
        return "{}_objectdataset{}.h5".format(prefix, self.suffix)

    def data_path(self, split):
        return os.path.join("h5_files", self.subdir, self.filename(split))


class ScanObjectNN(tfds.core.GeneratorBasedBuilder):
    def _info(self):
        features = tfds.core.features.FeaturesDict(
            dict(
                positions=tfds.core.features.Tensor(
                    shape=(NUM_POINTS, 3), dtype=tf.float32
                ),
                label=tfds.core.features.ClassLabel(names=CLASS_NAMES),
                mask=tfds.core.features.Tensor(shape=(NUM_POINTS,), dtype=tf.bool),
            )
        )
        return tfds.core.DatasetInfo(
            builder=self,
            features=features,
            citation=CITATION,
            supervised_keys=("positions", "label"),
            urls=[URL],
        )

    @property
    def up_dim(self):
        return 1

    def _split_generators(self, dl_manager):
        manual_dir = dl_manager.manual_dir
        path = os.path.join(manual_dir, "h5_files.zip")
        config = self.builder_config
        return [
            tfds.core.SplitGenerator(
                name=split, gen_kwargs=dict(path=path, filename=config.data_path(split))
            )
            for split in ("train", "test")
        ]

    def _generate_examples(self, path, filename):
        import zipfile

        import h5py

        with tf.io.gfile.GFile(path, "rb") as fp:
            zf = zipfile.ZipFile(fp)
            with tempfile.TemporaryDirectory() as temp_dir:
                assert isinstance(temp_dir, str)
                zf.extract(filename, temp_dir)
                dst = os.path.join(temp_dir, filename)
                print("extracted")
                h5 = h5py.File(dst, "r")
                positions = h5["data"]
                labels = h5["label"]
                masks = h5["mask"]
                n = h5["data"].shape[0]
                for i in range(n):
                    yield i, dict(
                        positions=positions[i], label=labels[i], mask=masks[i] != -1
                    )


if __name__ == "__main__":
    for variant in Variant.all():
        ScanObjectNN(config=ScanObjectNNConfig(variant)).download_and_prepare()
