import tensorflow as tf
from shape_tfds.shape.shapenet import core
import os

tf.compat.v1.enable_eager_execution()

resolution = 32
split = 'train'
names = ('suitcase', 'telephone', 'table')
ids, _ = core.load_synset_ids()

configs = (core.VoxelConfig(ids[n], resolution) for n in names)
builders = tuple(core.ShapenetCore(config=config) for config in configs)

all_fns = []
for builder in builders:
    builder.download_and_prepare()
    prefix = '%s-%s' % (builder.name.split('/')[0], split)
    data_dir = builder.data_dir
    record_fns = tuple(
        os.path.join(data_dir, fn)
        for fn in os.listdir(data_dir)
        if fn.startswith(prefix))
    all_fns.extend(record_fns)

print(all_fns)
builder = builders[0]
dataset = tf.data.TFRecordDataset(all_fns, num_parallel_reads=len(all_fns))
dataset = dataset.map(builder._file_format_adapter._parser.parse_example)

# dataset = builder._file_format_adapter.dataset_from_filename(all_fns)
dataset = dataset.map(builder.info.features.decode_example)


def vis(example):
    import trimesh
    print(example['model_id'].numpy())
    trimesh.voxel.VoxelGrid(example['voxels'].numpy()).show()


for example in dataset.take(20):
    vis(example)

print('---')
for example in dataset.take(20):
    vis(example)
