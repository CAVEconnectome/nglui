from setuptools import setup, find_packages

setup(name="neuroglancer_annotation_ui", packages=find_packages())

setup(
   name='neuroglancer_annotation_ui',
   version='0.1',
   description='Python tools for managing neuroglancer viewers, their states, and python annotations.',
   long_description=open('README.md').read(),
   author='Casey Schneider-Mizell',
   author_email='caseysm@gmail.com',
   url="https://github.com/seung-lab/NeuroglancerAnnotationUI",
   packages=['neuroglancer_annotation_ui'],
   install_requires=[
                'neuroglancer',
                'numpy',
                'pandas',
                'numpy',
                'emannotationschemas',
                'annotationengine',
                'cloud_volume',
                'PyChunkedGraph',
                'grpcio',
                'pytest',
                'requests',
                'marshmallow',
                'protobuf',
                'grpc',
                'tornado==4.5.3',
                ],  # external packages as dependencies
)
