from setuptools import setup, find_packages

# setup(name="neuroglancer_annotation_ui", packages=find_packages())


setup(
   name='neuroglancer_annotation_ui',
   version='0.1',
   description='Neuroglancer Annotation UI.',
   long_description=open('README.md').read(),
   author='Derrick Brittain',
   author_email='dlbrittain@gmail.com',
   url="https://github.com/dlbrittain/neuroglancer_analysis",
   packages=['neuroglancer_annotation_ui'],
   install_requires=[
                'neuroglancer',
                'numpy',
                'tornado==4.5.3',
                ],  # external packages as dependencies
)
