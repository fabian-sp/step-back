import stepback
from setuptools import setup, find_packages

# Package meta-data.
NAME = 'stepback'
DESCRIPTION = 'Deep Learning Optimization Experiments in Pytorch'
URL = 'https://github.com/fabian-sp/step-back'
EMAIL = 'fabian.schaipp@gmail.com'
AUTHOR = 'Fabian Schaipp'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = stepback.__version__

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      url=URL,
      author=AUTHOR,
      author_email=EMAIL,
      license='MIT',
      packages=find_packages(exclude=["tests"]),
      zip_safe=False)