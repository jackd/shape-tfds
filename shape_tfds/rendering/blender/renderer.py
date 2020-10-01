import os
import subprocess
import tempfile
from distutils.spawn import find_executable

import numpy as np
from absl import logging

BLENDER_RENDER_SCRIPT = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "_blender_render.py"
)
BLENDER_PATH = find_executable("blender")


def render(
    obj_path,
    output_directory,
    camera_positions,
    fov,
    resolution=(128, 128),
    filename_format="{output:s}-{index:03d}.png",
    script_path=BLENDER_RENDER_SCRIPT,
    blender_path=BLENDER_PATH,
    include_depth=True,
    include_normals=True,
    include_albedo=True,
    verbose=False,
):
    """
    Render the given obj file with blender.

    Args:
        obj_path: path to .obj file
        output_directory: directory into which outputs are saved.
        camera_positions: (num_views, 3) float array of locations of camera.
            Renderings will be from this position, pointing towards the origin.
        fov: field of view in degrees. Calculated based on focal and resolution
            if not provided.
        filename_format: format of output files. Filenames will be formated
            using view index (int) and data type (str).
        script_path: path to python script used in blender. The file at this
            path should have the same CLI as the default.
        blender_path: path to blender executable. Will check on path as well.
        verbose: if True, verbose output will be printed

    Returns:
        output file with `index` and `output` parameters. Files will be at
        `index in range(len(camera_positions))` and `output in
        ('render', 'normals', 'depth', 'albedo)`
    """
    if blender_path is None:
        if BLENDER_PATH is None:
            raise RuntimeError(
                "`blender_path` must be provided if `blender` executable not "
                "found on path"
            )
        else:
            raise ValueError("`blender_path` cannot be `None`")
    if not os.path.isfile(blender_path):
        raise ValueError("No file at `blender_path` %s" % blender_path)
    if len(camera_positions.shape) != 2:
        raise ValueError("camera_positions must be 2D")
    with tempfile.TemporaryDirectory() as temp_dir:
        camera_positions_path = os.path.join(str(temp_dir), "camera_positions.npy")
        np.save(camera_positions_path, camera_positions)
        ry, rx = resolution
        args = [
            blender_path,
            "--background",
            "--python",
            script_path,
            "--",
            "--obj",
            obj_path,
            "--out_dir",
            output_directory,
            # '--filename_format', filename_format,
            "--camera_positions",
            camera_positions_path,
            "--fov",
            str(fov),
            "--resolution",
            str(ry),
            str(rx),
        ]
        if include_depth:
            args.append("-d")
        if include_normals:
            args.append("-n")
        if include_albedo:
            args.append("-a")
        output = subprocess.check_output(args)
        if verbose:
            logging.info(output)
    return os.path.join(output_directory, "{output}-{index:03d}.png")


if __name__ == "__main__":
    import shutil

    from shape_tfds.shape.shapenet.core import base, views

    synset = "airplane"
    split = "train"
    index = 0
    num_views = 5
    include_normals = False
    include_depth = False
    include_albedo = False

    ids, _ = base.load_synset_ids()
    model_ids = base.load_split_ids(excluded=None)
    synset_id = ids[synset]
    model_id = model_ids[synset_id][split][index]

    paths = base.extracted_mesh_paths(synset_id)
    obj_path = paths[model_id]

    view_fn = views.random_view_fn()
    positions = view_fn(model_id, num_views=num_views)
    output_directory = os.path.join("/tmp/shapenet_renderings/%s" % model_id)
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    rendered_path = render(
        obj_path,
        output_directory,
        positions,
        fov=views.DEFAULT_FOV,
        include_normals=include_normals,
        include_depth=include_depth,
        include_albedo=include_albedo,
    )
