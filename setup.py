from setuptools import setup
import re
import os
import codecs
from setuptools import find_packages

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

with open('test_requirements.txt', 'r') as f:
    test_required = f.read().splitlines()

setup(
    name='nglui',
    version=find_version('src', 'nglui', '__init__.py'),
    description='Framework for data-driven generation of neuroglancer states.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Derrick Brittain, Casey Schneider-Mizell, Forrest Collman',
    author_email='caseysm@gmail.com',
    url="https://github.com/seung-lab/NeuroglancerAnnotationUI",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    setup_requires=['pytest-runner'],
    tests_require=test_required,
    include_package_data=True,
    install_requires=[required],  # external packages as dependencies
)
