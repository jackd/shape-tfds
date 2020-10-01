import tensorflow as tf
from absl import app, flags

from shape_tfds.shape.shapenet import core

flags.DEFINE_string("synset", default="suitcase", help="synset name")
flags.DEFINE_integer("resolution", default=32, help="voxel resolution")
flags.DEFINE_integer("seed", default=0, help="seed to use for random_view_fn")
flags.DEFINE_boolean("vis", default=False, help="visualize on finish")


def main(_):
    tf.compat.v1.enable_eager_execution()
    FLAGS = flags.FLAGS
    ids, names = core.load_synset_ids()
    name = FLAGS.synset

    synset_id = name if name in names else ids[name]
    if synset_id not in names:
        raise ValueError("Invalid synset_id %s" % synset_id)

    builder = core.ShapenetCore(
        config=core.FrustumVoxelConfig(
            synset_id=synset_id, resolution=FLAGS.resolution, seed=FLAGS.seed
        )
    )
    builder.download_and_prepare()

    if FLAGS.vis:

        def vis(example):
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

            ax = plt.gca(projection="3d")
            ax.voxels(example["voxels"].numpy())
            # ax.axis("square")
            plt.show()

        dataset = builder.as_dataset(split="train")
        for example in dataset:
            vis(example)


if __name__ == "__main__":
    app.run(main)
