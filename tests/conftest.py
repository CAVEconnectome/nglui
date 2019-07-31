import pytest
from neuroglancer_annotation_ui import EasyViewer

@pytest.fixture(scope='session')
def viewer():
    return EasyViewer()

@pytest.fixture(scope='session')
def img_layer():
    return 'precomputed://gs://neuroglancer-public-data/flyem_fib-25/image'

@pytest.fixture(scope='session')
def seg_layer_precomputed():
    return 'precomputed://gs://neuroglancer-public-data/flyem_fib-25/ground_truth'

@pytest.fixture(scope='session')
def seg_layer_graphene():
    return 'graphene://https://dev12.dynamicannotationframework.com/segmentation/1.0/pinky100_neo1'