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
                pick=False,
            )
        )
        .add_segments(name="TestSegmentationLayer", segments=[1, 2, 3])
        .add_layer(AnnotationLayer(name="some_annotations"))
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
        .add_layer(
            AnnotationLayer(
                name="TestAnnotationLayer",
                source="precomputed://gs://pathtoannotations",
            )
        )
    )
    assert viewer.layers[0].name == "TestImageLayer"
    assert viewer.layers["TestImageLayer"].source == client_full.info.image_source()
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

    site_utils.add_neuroglancer_site(
        site_name="test_site",
        site_url="https://test.neuroglancer.com",
    )
    assert "https://test.neuroglancer.com" in viewer.to_url(target_site="test_site")


def test_adding_points(client_full, soma_df):
    viewer = (
        ViewerState(infer_coordinates=False)
        .add_layers_from_client(
            client_full,
        )
        .add_segments_from_data(
            data=soma_df.head(2),
            segment_column="pt_root_id",
        )
        .add_points(
            data=soma_df,
            name="soma_points",
            point_column="pt_position",
            segment_column="pt_root_id",
            tag_column="cell_type",
        )
    )
    assert len(viewer.layers["segmentation"].segments) == 2
    state = viewer.to_dict()
    assert len(state["layers"][2]["annotations"]) == len(soma_df)


def test_adding_lines(client_full, pre_syn_df):
    viewer = (
        ViewerState(infer_coordinates=False)
        .add_layers_from_client(
            client_full,
        )
        .add_lines(
            data=pre_syn_df,
            name="pre_syn_lines",
            point_a_column="pre_pt_position",
            point_b_column="post_pt_position",
            segment_column="post_pt_root_id",
        )
    )
    state = viewer.to_dict()
    assert len(state["layers"][2]["annotations"]) == len(pre_syn_df)


def test_translated_source(img_path):
    img35_translate = Source(
        url=img_path,
        transform=CoordSpaceTransform(
            output_dimensions=[4, 4, 40],
            matrix=[
                [1, 0, 0.5, 1000],
                [0, 1, 0, 2000.0],
                [0, 0, 1, -1000.0],
            ],
        ),
    )

    img_layer_translated = ImageLayer("imagery_translated", img35_translate)

    img_layer_translated.to_dict()
