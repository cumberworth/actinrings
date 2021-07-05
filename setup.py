import glob
from setuptools import setup

setup(
    name='actinrings',
    packages=['actinrings'],
    scripts=glob.glob('scripts/*.py')
)
