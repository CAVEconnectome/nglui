import pytest
from nglui import EasyViewer
import pandas as pd
import numpy as np
import json


@pytest.fixture(scope="session")
def viewer():
    return EasyViewer(target_site='seunglab')


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
    return pd.read_hdf("tests/testdata/test_data.h5", "soma").head(5)


@pytest.fixture(scope="function")
def soma_df_Int64():
    df = pd.read_hdf("tests/testdata/test_data.h5", "soma").head(5)
    df["pt_root_id"] = df["pt_root_id"].astype("Int64")
    df["pt_root_id"].iloc[0] = np.nan
    return df.head()


@pytest.fixture(scope="function")
def pre_syn_df():
    return pd.read_hdf("tests/testdata/test_data.h5", "presyn").head(5)


@pytest.fixture(scope="function")
def post_syn_df():
    return pd.read_hdf("tests/testdata/test_data.h5", "postsyn").head(5)


@pytest.fixture(scope="session")
def test_state():
    with open("tests/testdata/test_state.json", "r") as f:
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