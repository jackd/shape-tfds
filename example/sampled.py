import tensorflow_datasets as tfds

from shape_tfds.shape.modelnet import sampled

builder = sampled.ModelnetSampled(config=sampled.get_config(10))
config = tfds.download.DownloadConfig(register_checksums=True)
builder.download_and_prepare(download_config=config)
