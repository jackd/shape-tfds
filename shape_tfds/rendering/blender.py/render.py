from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import numpy as np
import subprocess
import tempfile

BLENDER_RENDER_SCRIPT = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), '_blender_render.py')


def _as_clarg_value(value):
    if isinstance(value, (list, tuple)):
        return ' '.join(str(v) for v in value)
    else:
        return str(value)


def render(
        obj_path, camera_positions, output_directory,
        filename_format='r%03d%s',
        script_path=BLENDER_RENDER_SCRIPT, blender_path='blender',
        **render_params):
    """
    Render the given obj file with blender.

    Args:
        obj_path: path to .obj file
        camera_positions: (num_views, 3) float array of locations of camera.
            Renderings will be from this position, pointing towards the origin.
        output_directory: directory into which outputs are saved.
        filename_format: format of output files. Filenames will be formated
            using view index (int) and data type (str).
        script_path: path to python script used in blender. The file at this
            path should have the same CLI as the default.
        blender_path: path to blender executable. Will check on path as well.
        verbose: if True, verbose output will be printed
        **render_params: additional parameters passed to script via a json
            file. These are the same for

    Returns:
        output of subprocess.check_outputs
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        camera_positions_path = os.path.join(temp_dir, 'camera_positions.npy')
        np.save(camera_positions_path, camera_positions)
        args = [
                blender_path, '--background',
                '--python', script_path, '--',
                '--obj', obj_path,
                '--out_dir', output_directory,
                '--filename_format', filename_format,
                '--camera_positions', camera_positions_path]
        for k, v in render_params.items():
            args.extend(['--%s' % k, _as_clarg_value(v)])
        return subprocess.check_output(args)
