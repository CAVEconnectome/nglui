import pytest
import pandas as pd
import numpy as np
from nglui.segmentprops import SegmentProperties


@pytest.fixture
def test_df():
    return pd.DataFrame(
        {
            "seg_id": np.arange(0, 100),
            "cell_type": 30 * ["ct_a"] + 30 * ["ct_b"] + 40 * ["ct_c"],
            "category": 50 * ["cat_1"] + 40 * ["cat_2"] + 10 * [None],
            "number_int": np.arange(300, 400),
            "number_float": np.arange(300, 400) + 0.1,
            "tag_a": 90 * [False] + 10 * [True],
            "tag_b": 95 * [True] + 5 * [False],
        }
    )


def test_segment_props(test_df):
    props = SegmentProperties.from_dataframe(
        test_df,
        id_col="seg_id",
        label_col="seg_id",
        number_cols=["number_int", "number_float"],
        tag_value_cols="cell_type",
        tag_bool_cols=["tag_a", "tag_b"],
        tag_descriptions={"tag_a": "The first tag", "tag_b": "The second tag"},
    )

    assert len(props) == 100
    assert len(props.property_description()) == 4
    p_dict = props.to_dict()
    assert p_dict["inline"]["properties"][2]["data_type"] == "int32"

    rh_props = SegmentProperties.from_dict(p_dict)
    assert len(rh_props) == 100
