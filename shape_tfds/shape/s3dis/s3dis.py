"""s3dis dataset."""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow_datasets as tfds

_DESCRIPTION = """\
We collected and annotated five large-scale indoor parts in three buildings of mainly
educational and office use. Each area covers approximately 1900, 450, 1700, 870 and
1100 square meters (total of 6020 square meters). Conference rooms, personal offices,
auditoriums, restrooms, open spaces, lobbies, stairways and hallways are commonly
found. The areas show diverse properties in architectural style and appearance. We
fully annotated our dataset for 12 semantic elements which pertain in the categories of
structural building elements (ceiling, floor, wall, beam, column, window and door) and
commonly found furniture (table, chair, sofa, bookcase and board). A clutter class
exists as well for all other elements
"""

_CITATION = """\
@InProceedings{armeni_cvpr16,
  title={3D Semantic Parsing of Large-Scale Indoor Spaces},
  author= {Iro Armeni and Ozan Sener and Amir R. Zamir and Helen Jiang and Ioannis Brilakis and Martin Fischer and Silvio Savarese},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition},
  year={2016},
}
"""

FORM_URL = (
    "https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_"
    "rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1"
)
ALIGNED_FILENAME = "Stanford3dDataset_v1.2_Aligned_Version.zip"


NUM_AREAS = 6

AREAS = tuple((f"Area_{i}" for i in range(1, NUM_AREAS + 1)))

CLASS_NAMES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "chair",
    "table",
    "bookcase",
    "sofa",
    "board",
    "clutter",
)
NUM_CLASSES = len(CLASS_NAMES)

CLASS_COLORS = np.asarray(
    [
        [233, 229, 107],  # 'ceiling' .-> .yellow
        [95, 156, 196],  # 'floor' .-> . blue
        [179, 116, 81],  # 'wall'  ->  brown
        [241, 149, 131],  # 'beam'  ->  salmon
        [81, 163, 148],  # 'column'  ->  bluegreen
        [77, 174, 84],  # 'window'  ->  bright green
        [108, 135, 75],  # 'door'   ->  dark green
        [41, 49, 101],  # 'chair'  ->  darkblue
        [79, 79, 76],  # 'table'  ->  dark grey
        [223, 52, 52],  # 'bookcase'  ->  red
        [89, 47, 95],  # 'sofa'  ->  purple
        [81, 109, 114],  # 'board'   ->  grey
        [233, 233, 229],  # 'clutter'  ->  light grey
        [0, 0, 0],  # unlabelled .->. black
    ]
)


ROOM_TYPES = (
    "conferenceRoom",
    "copyRoom",
    "hallway",
    "office",
    "pantry",
    "WC",
    "auditorium",
    "storage",
    "lounge",
    "lobby",
    "openspace",
)


def _parse_path(archive_path: str):
    if not archive_path.endswith(".txt"):
        return None
    split = archive_path.split("/")
    if len(split) != 5:
        return None
    _, area, room, _, filename = split
    cls_name = filename.split("_")[0]
    if cls_name == "stairs":
        cls_name = "clutter"
    return area, room, cls_name


class S3dis(tfds.core.GeneratorBasedBuilder):
    """
    DatasetBuilder for s3dis dataset.

    Each split corresponds to a different area.

    e.g.
    ```python
    import tensorflow_datasets as tfds
    import shape_tfds.shape.s3dis

    train_ds, test_ds = tfds.load(
        's3dis', split=('Area_1+Area2+Area3+Area_4+Area_5', 'Area_6')
    )
    ```
    """

    VERSION = tfds.core.Version("1.2.0")

    RELEASE_NOTES = {
        "1.2.0": "Initial release.",
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = f"""\
    Fill out form at {FORM_URL} and download {ALIGNED_FILENAME} TO `manual_dir/`
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "coords": tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
                    "colors": tfds.features.Tensor(shape=(None, 3), dtype=tf.uint8),
                    "instances": tfds.features.Sequence(
                        {
                            "size": tfds.features.Tensor(shape=(), dtype=tf.int64),
                            "label": tfds.features.ClassLabel(names=CLASS_NAMES),
                        }
                    ),
                    "room_type": tfds.features.ClassLabel(names=ROOM_TYPES),
                    "room_idx": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            homepage="https://cvgl.stanford.edu/papers/iro_cvpr16.pdf",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = os.path.join(dl_manager.manual_dir, ALIGNED_FILENAME)
        if not tf.io.gfile.exists(path):
            raise Exception(
                f"Requires manual download. Download {ALIGNED_FILENAME} from "
                f"{FORM_URL} and move to {dl_manager.manual_dir}"
            )

        return {
            split: self._generate_examples(dl_manager, path, split=split)
            for split in AREAS
        }

    def _generate_examples(
        self, dl_manager: tfds.download.DownloadManager, archive_path: str, split: str,
    ):
        def split_data():
            for filename, fobj in dl_manager.iter_archive(archive_path):
                parsed = _parse_path(filename)
                if parsed is None:
                    continue
                area, room, cls_name = parsed
                if area != split:
                    continue
                yield fobj, room, cls_name

        instance_counts = defaultdict(lambda: 0)
        for fobj, room, cls_name in split_data():
            instance_counts[room] += 1

        instance_data = defaultdict(lambda: [])
        for fobj, room, cls_name in split_data():
            data = np.asarray(
                pd.read_csv(fobj, delimiter=" ", header=None, dtype=np.float32)
            )
            coords = data[:, :3]
            colors = data[:, 3:].astype(np.uint8)
            instance_data[room].append(
                {"coords": coords, "colors": colors, "label": cls_name}
            )
            instance_counts[room] -= 1
            if instance_counts[room] == 0:
                del instance_counts[room]
                data = instance_data.pop(room)
                coords = [d["coords"] for d in data]
                colors = [d["colors"] for d in data]
                room_type, room_idx = room.split("_")
                room_idx = int(room_idx)
                yield room, {
                    "instances": [
                        {"size": c.shape[0], "label": d["label"]}
                        for c, d in zip(coords, data)
                    ],
                    "coords": np.concatenate(coords, axis=0),
                    "colors": np.concatenate(colors, axis=0),
                    "room_type": room_type,
                    "room_idx": room_idx,
                }
        assert not instance_counts, list(instance_counts)
        assert not instance_data, list(instance_data)


if __name__ == "__main__":
    ds: tf.data.Dataset = tfds.load("s3dis", split="+".join(AREAS))
    sizes = ds.map(lambda inputs: tf.shape(inputs["coords"], tf.int64)[0])
    min_size = sizes.reduce(tf.constant(int(1e6), dtype=tf.int64), tf.minimum)
    max_size = sizes.reduce(tf.zeros((), dtype=tf.int64), tf.maximum)
    print(min_size, max_size)
