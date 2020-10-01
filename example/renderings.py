from absl import app, flags

flags.DEFINE_string("synset", default="suitcase", help="synset name")
flags.DEFINE_string("renderer", default="blender", help="renderer name")
flags.DEFINE_integer("resolution", default=128, help="voxel resolution")
flags.DEFINE_integer("seed", default=0, help="seed to use for random_view_fn")
flags.DEFINE_boolean("vis", default=False, help="visualize on finish")
flags.DEFINE_boolean("from_cache", default=False, help="generate from cache data")


def main(_):
    import tensorflow as tf

    from shape_tfds.shape.shapenet import core

    tf.compat.v1.enable_eager_execution()
    FLAGS = flags.FLAGS
    ids, names = core.load_synset_ids()
    name = FLAGS.synset

    synset_id = name if name in names else ids[name]
    if synset_id not in names:
        raise ValueError("Invalid synset_id %s" % synset_id)

    renderer = core.Renderer(FLAGS.renderer, resolution=(FLAGS.resolution,) * 2,)

    builder = core.ShapenetCore(
        config=core.RenderingConfig(
            synset_id=synset_id, renderer=renderer, seed=FLAGS.seed
        ),
        from_cache=FLAGS.from_cache,
    )
    builder.download_and_prepare()

    if FLAGS.vis:

        def vis(example):
            import matplotlib.pyplot as plt

            plt.imshow(example["image"].numpy())
            plt.show()

        dataset = builder.as_dataset(split="train")
        for example in dataset:
            vis(example)


if __name__ == "__main__":
    app.run(main)
