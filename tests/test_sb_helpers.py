import pandas as pd
import pytest
from pytest_mock import mocker

from nglui import statebuilder


def query_sideeffect(**kwargs):
    print(kwargs)
    if "pre_ids" in kwargs:
        df = pd.read_feather("tests/testdata/pre_df_helper.feather")
    elif "post_ids" in kwargs:
        df = pd.read_feather("tests/testdata/post_df_helper.feather")
    df.attrs["dataframe_resolution"] = [4, 4, 40]
    return df


def test_neuron_helper(client_full, mocker):
    root_id = 864691136137805181
    client_full.materialize.synapse_query = mocker.Mock()
    client_full.materialize.synapse_query.side_effect = query_sideeffect

    print(client_full.info.get_datastack_info())

    state_dict_spelunker = statebuilder.helpers.make_neuron_neuroglancer_link(
        client_full,
        root_id,
        return_as="dict",
        shorten="never",
        show_inputs=True,
        show_outputs=True,
        infer_coordinates=False,
    )
    assert "dimensions" in state_dict_spelunker.keys()
