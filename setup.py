from setuptools import setup, find_packages
import re
import os
import codecs

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
    raise RuntimeError()

with open('requirements.txt','r') as f:
    required = f.read().splitlines()

with open('test_requirements.txt', 'r') as f:
    test_required = f.read().splitlines()

setup(
    version=find_version("neuroglancer_annotation_ui", "__init__.py"),
    name='neuroglancer_annotation_ui',
    description='python tools for interacting with annotations in neuroglancer',
    author='Casey Schneider-Mizell',
    author_email='caseys@alleninstitute.org',
    url='https://github.com/seung-lab/NeuroglancerAnnotationUI',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    setup_requires=['pytest-runner'],
    tests_require=test_required
)