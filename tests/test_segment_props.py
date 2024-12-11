import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture
def test_categorical_df():
    df = pd.DataFrame(
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
    df["cell_type"] = df["cell_type"].astype(
        pd.CategoricalDtype(categories=["ct_d", "ct_c", "ct_b", "ct_a"], ordered=True)
    )
    return df


@pytest.fixture
def test_null_df():
    return pd.DataFrame(
        {
            "seg_id": np.arange(0, 100),
            "cell_type": 30 * ["ct_a"] + 30 * [""] + 40 * [None],
            "category": np.nan,
            "number_int": np.arange(300, 400),
            "number_float": np.arange(300, 400) + 0.1,
            "tag_a": 90 * [False] + 10 * [True],
            "tag_b": 50 * [True] + 50 * [False],
        }
    )


@pytest.fixture
def test_all_null_df():
    return pd.DataFrame(
        {
            "seg_id": np.arange(0, 100),
            "cell_type": 30 * [None] + 30 * [""] + 40 * [None],
            "category": np.nan,
            "number_int": np.arange(300, 400),
            "number_float": np.arange(300, 400) + 0.1,
            "tag_a": 90 * [False] + 10 * [False],
            "tag_b": 50 * [False] + 50 * [False],
        }
    )


@pytest.fixture
def test_segprops():
    return {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": ["44", "70", "77", "38", "68", "2", "19", "41", "67", "3"],
            "properties": [
                {
                    "id": "label",
                    "type": "label",
                    "values": [
                        "44",
                        "70",
                        "77",
                        "38",
                        "68",
                        "2",
                        "19",
                        "41",
                        "67",
                        "3",
                    ],
                },
                {
                    "id": "tags",
                    "type": "tags",
                    "tags": ["ct_a", "ct_b", "ct_c", "tag_a", "tag_b"],
                    "tag_descriptions": [
                        "ct_a",
                        "ct_b",
                        "ct_c",
                        "The first tag",
                        "The second tag",
                    ],
                    "values": [
                        [1, 4],
                        [2, 4],
                        [2, 4],
                        [1, 4],
                        [2, 4],
                        [0, 4],
                        [0, 4],
                        [1, 4],
                        [2, 4],
                        [0, 4],
                    ],
                },
                {
                    "id": "number_int",
                    "type": "number",
                    "values": [344, 370, 377, 338, 368, 302, 319, 341, 367, 303],
                    "data_type": "int32",
                },
                {
                    "id": "number_float",
                    "type": "number",
                    "values": [
                        344.1000061035156,
                        370.1000061035156,
                        377.1000061035156,
                        338.1000061035156,
                        368.1000061035156,
                        302.1000061035156,
                        319.1000061035156,
                        341.1000061035156,
                        367.1000061035156,
                        303.1000061035156,
                    ],
                    "data_type": "float32",
                },
            ],
        },
    }


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
    assert p_dict["inline"]["properties"][1]["tags"][2] == "ct_c"
    rh_props = SegmentProperties.from_dict(p_dict)
    assert len(rh_props) == 100


def test_categorical_props(test_categorical_df):
    props = SegmentProperties.from_dataframe(
        test_categorical_df,
        id_col="seg_id",
        label_col="seg_id",
        number_cols=["number_int", "number_float"],
        tag_value_cols="cell_type",
        tag_bool_cols=["tag_a", "tag_b"],
        tag_descriptions={"tag_a": "The first tag", "tag_b": "The second tag"},
    )

    assert len(props) == 100
    p_dict = props.to_dict()
    ct_a_idx = p_dict["inline"]["properties"][1]["tags"].index("ct_a")
    assert ct_a_idx in p_dict["inline"]["properties"][1]["values"][0]
    assert "ct_a" in p_dict["inline"]["properties"][1]["tags"]
    assert p_dict["inline"]["properties"][2]["data_type"] == "int32"


def test_prepend_col_name(test_categorical_df):
    props = SegmentProperties.from_dataframe(
        test_categorical_df,
        id_col="seg_id",
        label_col="seg_id",
        number_cols=["number_int", "number_float"],
        tag_value_cols="cell_type",
        tag_bool_cols=["tag_a", "tag_b"],
        prepend_col_name=True,
    )

    assert len(props) == 100
    p_dict = props.to_dict()
    assert "cell_type:ct_a" in p_dict["inline"]["properties"][1]["tags"]


def test_segment_props_nulls(test_null_df):
    props = SegmentProperties.from_dataframe(
        test_null_df,
        id_col="seg_id",
        tag_value_cols="cell_type",
        tag_bool_cols=["tag_a", "tag_b"],
        tag_descriptions={"tag_a": "The first tag", "tag_b": "The second tag"},
    )

    assert tuple(props.tag_properties.tags) == ("ct_a", "tag_a", "tag_b")
    p_dict = props.to_dict()
    assert 2 in p_dict["inline"]["properties"][0]["values"][0]
    assert len(p_dict["inline"]["properties"][0]["values"][50]) == 0


def test_segment_props_all_null(test_all_null_df):
    props = SegmentProperties.from_dataframe(
        test_all_null_df,
        id_col="seg_id",
        tag_value_cols="cell_type",
        tag_bool_cols=["tag_a", "tag_b"],
        tag_descriptions={"tag_a": "The first tag", "tag_b": "The second tag"},
    )

    assert tuple(props.tag_properties.tags) == ("tag_a", "tag_b")
    p_dict = props.to_dict()
    assert len(p_dict["inline"]["properties"][0]["values"][0]) == 0


def test_property_conversion(test_segprops):
    props = SegmentProperties.from_dict(test_segprops)
    assert len(props) == 10
    prop_df = props.to_dataframe()
    assert len(prop_df.columns) == 9


def test_multicolumn_label(test_df):
    props = SegmentProperties.from_dataframe(
        test_df,
        id_col="seg_id",
        label_col=["seg_id", "cell_type", "category"],
    )
    assert len(props) == 100
    p_dict = props.to_dict()
    assert p_dict["inline"]["properties"][0]["values"][1] == "1_ct_a_cat_1"

    props_fmt_map = SegmentProperties.from_dataframe(
        test_df,
        id_col="seg_id",
        label_format_map="{seg_id}_{cell_type}_{category}",
    )

    p_dict_fmt = props_fmt_map.to_dict()
    assert (
        p_dict_fmt["inline"]["properties"][0]["values"][1]
        == p_dict["inline"]["properties"][0]["values"][1]
    )


def test_random_sampler(test_df):
    props = SegmentProperties.from_dataframe(
        test_df,
        id_col="seg_id",
        label_col=["seg_id", "cell_type", "category"],
        random_columns=1,
        random_column_prefix="random_sample_test",
    )
    assert "random_sample_test" in props.to_dataframe().columns

    props = SegmentProperties.from_dataframe(
        test_df,
        id_col="seg_id",
        label_col=["seg_id", "cell_type", "category"],
        random_columns=3,
        random_column_prefix="random_sample_test",
    )
    assert "random_sample_test_2" in props.to_dataframe().columns
