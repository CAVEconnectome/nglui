import pytest
from nglui import EasyViewer
import pandas as pd
import numpy as np
import json


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
def soma_df_Int64():
    df = pd.read_hdf('tests/testdata/test_data.h5', 'soma').head(5)
    df['pt_root_id'] = df['pt_root_id'].astype('Int64')
    df['pt_root_id'].iloc[0] = pd.NA
    return df.head()


@pytest.fixture(scope='function')
def pre_syn_df():
    return pd.read_hdf('tests/testdata/test_data.h5', 'presyn').head(5)


@pytest.fixture(scope='function')
def post_syn_df():
    return pd.read_hdf('tests/testdata/test_data.h5', 'postsyn').head(5)


@pytest.fixture(scope='session')
def test_state():
    with open('tests/testdata/test_state.json', 'r') as f:
        state = json.load(f)
    return state
