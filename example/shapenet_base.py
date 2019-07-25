from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zipfile
from shape_tfds.shape.shapenet.core.base import mesh_loader_context
from shape_tfds.shape.shapenet.core.base import get_obj_zip_path
from shape_tfds.shape.shapenet.core.base import load_synset_ids

import pyrender
import trimesh
import numpy as np
import os
import matplotlib.pyplot as plt


synset = 'suitcase'


def vis(mesh_or_scene):
    """
    based on https://colab.research.google.com/drive/1Z71mHIc-Sqval92nK290vAsHZRUkCjUx

    setup:
    ```bash
    sudo apt update
    sudo wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
    sudo dpkg -i ./mesa_18.3.3-0.deb || true
    sudo apt install -f
    git clone https://github.com/mmatl/pyopengl.git
    pip install ./pyopengl
    ```
    """
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    resolution = (512, 512)

    scene = pyrender.Scene()
    if isinstance(mesh_or_scene, trimesh.Scene):
        for mesh in mesh_or_scene.geometry.values():
            scene.add(pyrender.Mesh.from_trimesh(mesh))
        tm_scene = mesh_or_scene
    else:
        scene.add(pyrender.Mesh.from_trimesh(mesh_or_scene))
        tm_scene = mesh_or_scene.scene()
    tm_scene.camera.resolution = resolution
    camera = pyrender.PerspectiveCamera(
        yfov=np.pi / 180 * tm_scene.camera.fov[1])
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi/16.0)

    camera_transform = tm_scene.camera_transform

    scene.add(camera, camera_transform)
    scene.add(light, camera_transform)

    r = pyrender.OffscreenRenderer(*resolution)
    color, depth = r.render(scene)

    # Show the images
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(color)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(depth, cmap=plt.cm.gray_r)
    plt.show()



ids, names = load_synset_ids()
with mesh_loader_context(ids[synset]) as loader:
    for k, v in loader.items():
        print(k)
        vis(v)
