import trimesh
import os


class ZipSubdirResolver(trimesh.visual.resolvers.Resolver):

    def __init__(self, archive, subdir):
        assert (hasattr(archive, 'read'))
        self.archive = archive
        self.subdir = subdir

    def get(self, name):
        name = name.lstrip()
        if name.startswith('./'):
            name = name[2:]
        with self.archive.open(os.path.join(self.subdir, name)) as fp:
            return fp.read()
