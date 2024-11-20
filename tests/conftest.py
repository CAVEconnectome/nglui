import json

import numpy as np
import pandas as pd
import pytest
from caveclient.tools.testing import CAVEclientMock

from nglui import EasyViewer


@pytest.fixture(scope="session")
def client_simple():
    return CAVEclientMock(json_service=True)


@pytest.fixture(scope="function")
def viewer_seunglab():
    return EasyViewer(target_site="seunglab")


@pytest.fixture(scope="function")
def viewer_cave_explorer():
    return EasyViewer(target_site="spelunker")


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


@pytest.fixture(scope="session")
def test_state():
    with open("tests/testdata/test_state.json") as f:
        state = json.load(f)
    return state


@pytest.fixture(scope="function")
def split_point_df():
    seg_id = 864691135293185292

    red_pts = [[182983, 172432, 20264], [182987, 172436, 20264]]

    blue_pts = [[182546, 172262, 20338], [182483, 171930, 20316]]

    red_df = pd.DataFrame(
        {
            "pts": np.array(red_pts).tolist(),
            "team": "red",
            "seg_id": seg_id,
        }
    )

    blue_df = pd.DataFrame(
        {
            "pts": np.array(blue_pts).tolist(),
            "team": "blue",
            "seg_id": seg_id,
        }
    )

    return pd.concat((red_df, blue_df))
