from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as fp:
      install_requires = fp.read().split('\n')


setup(name='shape-tfds',
      version='0.1',
      description='tensorflow_datasets implementations for shape datasets',
      url='http://github.com/jackd/shape-tfds',
      author='Dominic Jack',
      author_email='thedomjack@gmail.com',
      license='MIT',
      # packages=['shape_tfds'],
      packages=find_packages(),
      requirements=install_requires,
      # include_package_data=True,
      # zip_safe=False
)
