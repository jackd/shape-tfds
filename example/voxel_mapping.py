from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shape_tfds.shape.shapenet import core


ids, names = core.load_synset_ids()

name = 'suitcase'
# name = 'watercraft'
# name = 'aeroplane'
# name = 'table'
# name = 'rifle'

config = core.VoxelConfig(synset_id=ids[name], resolution=32)
mapping_context = core.get_data_mapping_context(config)


def vis(voxels):
    # visualize a single image/voxel pair
    import matplotlib.pyplot as plt
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D

    ax = plt.gca(projection="3d")
    ax.voxels(voxels)
    # ax.axis("square")
    plt.show()


with mapping_context as mapping:
    for k, v in mapping.items():
        vis(v['voxels'])
