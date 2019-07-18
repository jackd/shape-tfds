from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import trimesh
import os


class ZipSubdirResolver(trimesh.visual.resolvers.Resolver):
    def __init__(self, archive, subdir):
        assert(hasattr(archive, 'read'))
        self.archive = archive
        self.subdir = subdir

    def get(self, name):
        name = name.lstrip()
        if name.startswith('./'):
            name = name[2:]
        fp = self.archive.open(os.path.join(self.subdir, name))
        return fp.read()
