import numpy as np
import pandas as pd
import pytest

from nglui import parser


def test_layer_names(test_state):
    layer_names = parser.layer_names(test_state)
    assert len(layer_names) == 3
    assert "synapses" in layer_names


def test_layers(test_state):
    img_layer = parser.image_layers(test_state)
    assert len(img_layer) == 1
    assert img_layer[0] == "imagery"

    seg_layers = parser.segmentation_layers(test_state)
    assert len(seg_layers) == 1
    assert seg_layers[0] == "segments"

    anno_layers = parser.annotation_layers(test_state)
    assert len(anno_layers) == 1
    assert anno_layers[0] == "synapses"

    lyr = parser.get_layer(test_state, anno_layers[0])
    assert isinstance(lyr, dict)
    assert lyr["name"] == anno_layers[0]


def test_tag_dictionary(test_state):
    tags = parser.tag_dictionary(test_state, "synapses")
    assert tags[1] == "ChC"


def test_view_settings(test_state):
    view = parser.view_settings(test_state)
    assert view["perspectiveZoom"] == 1500


def test_annotation_parsing(test_state):
    points = parser.point_annotations(test_state, "synapses")
    assert len(points) == 6
    assert points[2][2] == 1440

    points, desc, tags = parser.point_annotations(
        test_state, "synapses", description=True, tags=True
    )
    assert len(desc) == len(tags)
    assert desc[1] is None
    assert tags[4][0] == 1


def test_dataframe(test_state):
    df = parser.annotation_dataframe(test_state, expand_tags=True)
    assert isinstance(df, pd.DataFrame)
    assert np.all(df["ChC"])
