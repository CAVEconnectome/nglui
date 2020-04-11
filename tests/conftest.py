import pytest
from neuroglancer_annotation_ui import EasyViewer, set_static_content_source
import pandas as pd
import numpy as np


@pytest.fixture(scope='session')
def viewer():
    return EasyViewer()


@pytest.fixture(scope='session')
def img_path():
    return 'precomputed://gs://pathtoimagery'


@pytest.fixture(scope='session')
def seg_path_precomputed():
    return 'precomputed://gs://pathtosegmentation/seg'


@pytest.fixture(scope='session')
def seg_path_graphene():
    return 'graphene://gs://pathtosegmentation/seg'


@pytest.fixture(scope='session')
def anno_layer_name():
    return 'test_anno_layer'


@pytest.fixture(scope='function')
def soma_df():
    return pd.read_hdf('tests/testdata/test_data.h5', 'soma').head(5)


@pytest.fixture(scope='function')
def pre_syn_df():
    return pd.read_hdf('tests/testdata/test_data.h5', 'presyn').head(5)


@pytest.fixture(scope='function')
def post_syn_df():
    return pd.read_hdf('tests/testdata/test_data.h5', 'postsyn').head(5)
