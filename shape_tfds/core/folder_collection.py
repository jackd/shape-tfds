import collections
import os


class FolderCollection(collections.Collection):
    def __init__(self, directory):
        self._directory = directory

    def __len__(self):
        return len(os.listdir(self._directory))

    def __iter__(self):
        return iter(os.listdir(self._directory))

    def __contains__(self, key):
        return os.path.exists(os.path.join(self._directory, key))

    def path(self, subpath):
        return os.path.join(self._directory, subpath)

    @property
    def directory(self):
        return self._directory
