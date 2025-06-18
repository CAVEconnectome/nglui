import json

import numpy as np
import pandas as pd
import pytest
from caveclient.tools.testing import CAVEclientMock


@pytest.fixture(scope="session")
def client_simple():
    return CAVEclientMock(json_service=True)


@pytest.fixture(scope="session")
def client_full():
    client = CAVEclientMock(
        chunkedgraph=True,
        materialization=True,
        json_service=True,
        available_materialization_versions=[1],
        set_version=1,
    )
    print(client.materialize.version)
    return client


@pytest.fixture(scope="session")
def img_path():
    return "precomputed://gs://pathtoimagery"


@pytest.fixture(scope="session")
def seg_path_precomputed():
    return "precomputed://gs://pathtosegmentation/seg"


@pytest.fixture(scope="session")
def seg_path_graphene():
    return "graphene://gs://pathtosegmentation/seg"


@pytest.fixture(scope="session")
def anno_layer_name():
    return "test_anno_layer"


@pytest.fixture(scope="session")
def test_state():
    with open("tests/testdata/test_state.json", "r") as f:
        state = json.load(f)
    return state


@pytest.fixture(scope="function")
def soma_df():
    return pd.read_feather("tests/testdata/soma_data.feather")


@pytest.fixture(scope="function")
def soma_df_Int64(soma_df):
    df = soma_df.copy()
    df["pt_root_id"] = df["pt_root_id"].astype("Int64")
    df.loc[0, "pt_root_id"] = np.nan
    return df.head()


@pytest.fixture(scope="function")
def pre_syn_df():
    return pd.read_feather("tests/testdata/pre_syn_data.feather")


@pytest.fixture(scope="function")
def post_syn_df():
    return pd.read_feather("tests/testdata/post_syn_data.feather")
