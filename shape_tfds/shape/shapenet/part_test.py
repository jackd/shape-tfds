"""Tests for shapenet_part dataset module."""

from tensorflow_datasets import testing

from shape_tfds.shape.shapenet import part

splits = {"train": 4, "test": 2, "validation": 2}


class ShapenetPart2017Test(testing.DatasetBuilderTestCase):
    DATASET_CLASS = part.ShapenetPart2017
    SPLITS = splits


if __name__ == "__main__":
    testing.test_main()
