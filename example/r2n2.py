from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet.r2n2 import ShapenetR2n2Config
from shape_tfds.shape.shapenet.r2n2 import synset_id

tf.compat.v1.enable_eager_execution()
FLAGS = flags.FLAGS

flags.DEFINE_string("synset", "telephone", "category name or id")
flags.DEFINE_bool(
    "download", True, "setting to false after initial run makes things faster")
flags.DEFINE_integer("image_index", 0, "image index to use")


def vis(image, voxels):
  """visualize a single image/voxel pair."""
  import matplotlib.pyplot as plt
  # This import registers the 3D projection, but is otherwise unused.
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.gca()
  ax.imshow(image)
  ax.axis("off")
  fig = plt.figure()
  ax = fig.gca(projection="3d")
  ax.voxels(voxels)
  plt.show()


def main(argv):
  builder = tfds.builder('shapenet_r2n2/%s' % synset_id(FLAGS.synset))
  download_config = tfds.download.DownloadConfig(register_checksums=True)
  builder.download_and_prepare(download_config=download_config)
  dataset = builder.as_dataset(split='train')

  for example in dataset:
    voxels = example["voxels"]
    image = example["renderings"]["image"][FLAGS.image_index]
    vis(image.numpy(), voxels.numpy())


if __name__ == "__main__":
  app.run(main)
