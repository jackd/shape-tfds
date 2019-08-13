from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import zipfile
from shape_tfds.shape.shapenet.core.base import mesh_loader_context
from shape_tfds.shape.shapenet.core.base import get_obj_zip_path
from shape_tfds.shape.shapenet.core.base import load_synset_ids
from shape_tfds.shape.shapenet.core import views

import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt

synset = 'suitcase'
resolution = (512, 512)
view_fn = views.random_view_fn(0)


def vis(model_id, mesh_or_scene):
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

    # scene = pyrender.Scene()
    if isinstance(mesh_or_scene, trimesh.Scene):
        # for mesh in mesh_or_scene.geometry.values():
        #     scene.add(pyrender.Mesh.from_trimesh(mesh))
        # tm_scene = mesh_or_scene
        meshes = mesh_or_scene.geometry.values()
    else:
        # scene.add(pyrender.Mesh.from_trimesh(mesh_or_scene))
        # tm_scene = mesh_or_scene.scene()
        meshes = [mesh_or_scene]

    camera_transform = np.array([
        [
            0.58991882,
            -0.3372407,
            0.73366511,
            0.86073076,
        ],
        [
            0.04221561,
            0.92024442,
            0.38906047,
            0.45644301,
        ],
        [
            -0.80635825,
            -0.19854197,
            0.55710632,
            0.65359322,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    ])
    trimesh.scene.cameras.Camera(fov=(45., 45.))
    light = trimesh.scene.lighting.SpotLight(color=0.5 * np.ones(3),
                                             intensity=0.0,
                                             innerConeAngle=np.pi / 16.0)
    tm_scene = trimesh.scene.Scene(meshes, lights=[light])
    for name in [light.name, tm_scene.camera.name]:
        tm_scene.graph[name] = camera_transform

    # tm_scene = trimesh.scene.Scene(meshes)

    # views.set_scene_view(tm_scene, resolution, **view_fn(model_id))
    tm_scene.camera.resolution = resolution
    tm_scene.camera_transform = camera_transform
    tm_scene.show()
    # dist = np.linalg.norm(tm_scene.camera_transform[:3, 3])

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 180 *
                                        tm_scene.camera.fov[1],
                                        # znear=dist-0.5, zfar=dist+0.5
                                       )

    scene = pyrender.Scene.from_trimesh_scene(tm_scene)
    scene.add(camera, pose=camera_transform)

    light = pyrender.SpotLight(color=np.ones(3),
                               intensity=10.0,
                               innerConeAngle=np.pi / 16.0)
    scene.add(light, pose=camera_transform)

    # viewer = pyrender.Viewer(scene, use_direct_lighting=True)

    r = pyrender.OffscreenRenderer(*resolution)
    color, depth = r.render(scene)

    # Show the images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(color)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(depth, cmap=plt.cm.gray_r)  # pylint: disable=no-member
    plt.show()


ids, names = load_synset_ids()
with mesh_loader_context(ids[synset]) as loader:
    model_ids = sorted(loader)
    for model_id in model_ids:
        print(model_id)
        vis(model_id, loader[model_id])
