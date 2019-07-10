# shape-tfds: shape datasets for tensorflow

[Tensorflow datasets][tfds] (`tfds`) implementations of various shape datasets.

## Setup

```bash
git clone https://github.com/jackd/shape-tfds
cd shape-tfds
pip install -r requirements.txt
pip install -e .
```

## Supported Datasets

* shapenet
  * [r2n2](shape_tfds/shapenet/r2n2.py)
  * [core](shape_tfds/shapenet/core.py)
* [modelnet](shape_tfds/modelnet/__init__.py) (coming soon)

## Motivation for a separate repository

Since the release of [tensorflow datasets][tfds] there has been an understandable surge in pull requests to the main repository. Due to this, and the high bar for quality set by the maintainers, there is a considerable backlog of PRs awaiting review (including several of my own). Having multiple PRs sitting in queue for months on end causes two main issues:

1. Poor visibility: Hacking datasets together is not particularly fun and nets you no research kudos. If I can save someone else the bother of finding download links, loading and transforming data, all the better. The prospect of people finding an obscure branch or fork and digging into the detail is quite low.
2. Difficult/confusing packaging: for separate packages, I'd rather not have dependencies on my own branch of such a large package like tfds. For people wanting to quickly download my research projects and run them without much bother (and whom have a lax attitude towards virtual environments etc.), having `pip` install a custom fork can go unnoticed at first and lead to much confusion later on.

## Converting functionality to PRs

Eventually, we would like to see this work merged into [tfds][tfds]. Due to size, this will likely have to be made in multiple separate pull requests.

To make this process as simple as possible, we keep the directory structure identical. Creating pull requests should involve:

* changing `import shape_tfds.REST` to `import tensorflow_datasets.REST`
* removing `import trimesh` and replacing `trimesh.blah` calls with `lazy_imports.trimesh.blah`
* adding relevant `dataset_files` to [tfds setup.py](https://github.com/tensorflow_datasets/blob/master/tensorflow_datasets/setup.py)
* copying `url_checksums` across

## Status

Under heavy development. Expect untested functionality, bugs and breaking changes.

TODO:

* add `PaddedTensor` tests
* shapenet core
* modelnet
* [abc](https://deep-geometry.github.io/abc-dataset/)

[tfds]: https://github.com/tensorflow/datasets
