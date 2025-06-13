import pytest
from pytest_mock import mocker

from nglui.statebuilder import *


def test_basic_viewer(client_full, mocker):
    viewer = (
        ViewerState(dimensions=[4, 4, 40])
        .add_layer(
            ImageLayer(
                source=client_full.info.image_source(),
                name="TestImageLayer",
            )
        )
        .add_layer(
            SegmentationLayer(
                source=client_full.info.segmentation_source(),
                name="TestSegmentationLayer",
            )
        )
        .add_layer(LocalAnnotationLayer(name="some_annotations"))
        .add_annotation_layer(
            name="other_annotations",
            linked_segmentation="TestSegmentationLayer",
        )
        .set_viewer_properties(
            position=[1, 1, 1],
            scale_imagery=10,
            scale_3d=1000.4,
            selected_layer="some_annotations",
            layout="xy",
        )
    )
    assert viewer.layers[0].name == "TestImageLayer"
    assert isinstance(viewer.to_dict(), dict)
    assert viewer.to_json_string() is not None
    assert "https" in viewer.to_url()
    viewer.to_link()

    mocker.patch.object(
        client_full.state,
        "upload_state_json",
        return_value=12345,
    )

    assert "12345" in viewer.to_link_shortener(client=client_full)


def test_adding_points(client_full, soma_df):
    pass
