from setuptools import find_packages, setup

with open("requirements.txt") as fp:
    install_requires = fp.read().split("\n")

DATASET_FILES = [
    "url_checksums/*",
    "shape/shapenet/core/core_synset.txt",
    "shape/modelnet/class_names10.txt",
    "shape/modelnet/class_names40.txt",
    "shape/modelnet/class_freq.txt",
    "shape/shapenet/part2019/after_merging_label_ids/*",
]

setup(
    name="shape-tfds",
    version="0.1",
    description="tensorflow_datasets implementations for shape datasets",
    url="http://github.com/jackd/shape-tfds",
    author="Dominic Jack",
    author_email="thedomjack@gmail.com",
    license="MIT",
    packages=find_packages(),
    requirements=install_requires,
    include_package_data=True,
    package_data={"shape_tfds": DATASET_FILES}
    # zip_safe=False
)
