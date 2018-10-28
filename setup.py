#   encoding: utf-8
#   filename: setup.py

from setuptools import find_packages, setup

setup(name='svae',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'svae = svae.cli:main',
          ],
      })
