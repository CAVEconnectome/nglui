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
    state_dict = statebuilder.helpers.make_neuron_neuroglancer_link(
        client_full,
        root_id,
        return_as="dict",
        shorten="never",
        show_inputs=True,
        show_outputs=True,
    )
    df = pd.read_feather("tests/testdata/post_df_helper.feather")
    assert len(state_dict["layers"][2]["annotations"]) == len(df)
    assert "dimensions" not in state_dict.keys()

    statebuilder.site_utils.set_default_config(target_site="spelunker")
    state_dict_spelunker = statebuilder.helpers.make_neuron_neuroglancer_link(
        client_full,
        root_id,
        return_as="dict",
        shorten="never",
        show_inputs=True,
        show_outputs=True,
    )
    assert "dimensions" in state_dict_spelunker.keys()
