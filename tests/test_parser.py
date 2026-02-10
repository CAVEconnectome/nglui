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


def test_polyline_annotation_parsing():
    """Test that polyline annotations are parsed into the annotation dataframe."""
    state = {
        "layers": [
            {
                "type": "annotation",
                "annotations": [
                    {
                        "type": "polyline",
                        "points": [
                            [100, 200, 30],
                            [110, 210, 31],
                            [120, 220, 32],
                        ],
                        "id": "polyline_1",
                        "description": "path_a",
                        "segments": ["12345"],
                    },
                    {
                        "type": "polyline",
                        "points": [
                            [400, 500, 60],
                            [410, 510, 61],
                        ],
                        "id": "polyline_2",
                        "segments": ["67890"],
                    },
                    {
                        "type": "point",
                        "point": [50, 60, 10],
                        "id": "point_1",
                    },
                ],
                "voxelSize": [4, 4, 40],
                "name": "my_annotations",
            }
        ],
        "navigation": {
            "pose": {
                "position": {
                    "voxelSize": [4, 4, 40],
                    "voxelCoordinates": [100, 200, 30],
                }
            },
            "zoomFactor": 8,
        },
        "perspectiveZoom": 1500,
        "layout": "xy-3d",
    }

    # Test polyline_annotations directly
    pl_pts = parser.base.polyline_annotations(state, "my_annotations")
    assert len(pl_pts) == 2
    assert len(pl_pts[0]) == 3  # first polyline has 3 points
    assert len(pl_pts[1]) == 2  # second polyline has 2 points
    assert pl_pts[0][0] == [100, 200, 30]

    # Test polyline_annotations with description
    pl_pts, pl_desc, pl_seg = parser.base.polyline_annotations(
        state, "my_annotations", description=True, linked_segmentations=True
    )
    assert pl_desc[0] == "path_a"
    assert pl_desc[1] is None
    assert pl_seg[0] == [12345]

    # Test annotation_dataframe includes polylines
    df = parser.annotation_dataframe(state)
    assert isinstance(df, pd.DataFrame)
    polyline_rows = df[df["anno_type"] == "polyline"]
    assert len(polyline_rows) == 2
    point_rows = df[df["anno_type"] == "point"]
    assert len(point_rows) == 1

    # Check polyline point data is a list of points
    first_polyline = polyline_rows.iloc[0]
    assert len(first_polyline["point"]) == 3
    assert first_polyline["point"][0] == [100.0, 200.0, 30.0]

    # Test anno_id column in dataframe
    assert "anno_id" in df.columns
    assert polyline_rows.iloc[0]["anno_id"] == "polyline_1"
    assert polyline_rows.iloc[1]["anno_id"] == "polyline_2"
    assert point_rows.iloc[0]["anno_id"] == "point_1"

    # Test anno_id flag on individual annotation functions
    pl_pts, pl_ids = parser.base.polyline_annotations(
        state, "my_annotations", anno_id=True
    )
    assert pl_ids == ["polyline_1", "polyline_2"]

    # Test with point_resolution scaling
    df_scaled = parser.annotation_dataframe(state, point_resolution=[4, 4, 40])
    polyline_scaled = df_scaled[df_scaled["anno_type"] == "polyline"]
    assert len(polyline_scaled) == 2
