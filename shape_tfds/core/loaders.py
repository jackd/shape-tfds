from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


def load_off(file_obj):
    """
    Load an OFF file into the kwargs for a Trimesh constructor


    Parameters
    ----------
    file_obj : file object
      Contains an OFF file

    Returns
    ----------
    loaded : dict of vertices, faces

    Variant of trimesh.exchange.misc.load_off that allows for the first line
    to be OFFX Y Z
    """
    header_string = file_obj.readline()
    if hasattr(header_string, 'decode'):
        header_string = header_string.decode('utf-8')
    header_string = header_string.strip().upper()

    if not header_string.startswith('OFF'):
        raise NameError('Not an OFF file! Header was ' +
                        header_string)
    if len(header_string) == 3:
        header_string = file_obj.readline().strip()
    else:
        header_string = header_string[3:]
    header = np.array(header_string.split()).astype(np.int64)

    vertex_count, face_count = header[:2]

    # read the rest of the file
    blob = np.array(file_obj.read().strip().split())
    # there should be 3 points per vertex
    # and 3 indexes + 1 count per face
    data_ok = np.sum(header * [3, 4, 0]) == len(blob)
    if not data_ok:
        raise NameError('Incorrect number of vertices or faces!')

    vertices = blob[:(vertex_count * 3)].astype(
        np.float64).reshape((-1, 3))
    # strip the first column which is a per- face count
    faces = blob[(vertex_count * 3):].astype(
        np.int64).reshape((-1, 4))[:, 1:]

    if face_count != len(faces):
        raise ValueError(
            'Inconsistent number of faces vs header spec, %d vs %d'
            % (len(faces, face_count)))

    return {'vertices': vertices,
            'faces': faces}
