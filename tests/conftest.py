import pytest
from neuroglancer_annotation_ui import EasyViewer
import pandas as pd
import numpy as np

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

@pytest.fixture(scope='session')
def anno_layer():
    return 'test_anno_layer'

@pytest.fixture(scope='session')
def df():
    datalen = 10
    single_inds = np.arange(0,datalen)
    multi_inds_array = [100+np.arange(ii, ii+2) for ii in np.arange(0, 2*datalen, 2)]
    multi_inds_list = [(200+np.arange(ii, ii+2)).tolist() for ii in np.arange(0, 2*datalen, 2)]

    single_pts = [np.random.randint(0,10000,(3,)) for i in single_inds]
    multi_pts_array = [np.random.randint(0,10000,(2,3)) for i in single_inds]
    multi_pts_list_array = [[np.random.randint(0,10000,(3,)) for j in range(2)] for i in single_inds]
    multi_pts_list_list = [np.random.randint(0,10000,(2,3)).tolist() for i in single_inds]

    return pd.DataFrame({'single_inds':single_inds,
                         'multi_inds_array':multi_inds_array,
                         'multi_inds_list': multi_inds_list,
                         'single_pts': single_pts,
                         'multi_pts_array': multi_pts_array,
                         'multi_pts_list_array': multi_pts_list_array,
                         'multi_pts_list_list': multi_pts_list_list})
