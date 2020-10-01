from absl import app, flags

flags.DEFINE_string('version',
                    help='pointnet version: 1, 2 or 2h',
                    default='2h')
flags.DEFINE_boolean('vis',
                     help='visualize after preparing. Requires trimesh',
                     default=False)


def main(_):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    import tensorflow_datasets as tfds
    FLAGS = flags.FLAGS

    if FLAGS.version == '2':
        from shape_tfds.shape.modelnet import Pointnet2
        from shape_tfds.shape.modelnet import get_pointnet2_config
        builder = Pointnet2(config=get_pointnet2_config(num_classes=40))
    elif FLAGS.version == '2h':
        from shape_tfds.shape.modelnet import Pointnet2H5
        builder = Pointnet2H5()
    elif FLAGS.version == '1':
        from shape_tfds.shape.modelnet import Pointnet
        builder = Pointnet()
    else:
        raise ValueError(
            'Invalid choice of version {} - must be "1", "2", or "2h"'.format(
                FLAGS.version))
    download_config = tfds.core.download.DownloadConfig(register_checksums=True)
    # download_config = None
    builder.download_and_prepare(download_config=download_config)

    if FLAGS.vis:
        import numpy as np
        try:
            import trimesh
        except ImportError:
            raise ImportError('visualizing requires trimesh')
        for cloud, _ in builder.as_dataset(split='train', as_supervised=True):
            pos = cloud['positions'].numpy()
            print(np.min(pos, axis=0))
            print(np.max(pos, axis=0))
            print(np.max(np.linalg.norm(pos, axis=-1)))
            trimesh.PointCloud(pos).show()


if __name__ == '__main__':
    app.run(main)
