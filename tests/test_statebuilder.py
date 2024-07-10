from collections import OrderedDict

import numpy as np
import pytest

from nglui.statebuilder import (
    AnnotationLayerConfig,
    ChainedStateBuilder,
    ImageLayerConfig,
    LineMapper,
    PointMapper,
    SegmentationLayerConfig,
    SphereMapper,
    SplitPointMapper,
    StateBuilder,
)


@pytest.fixture
def image_layer(img_path):
    img_layer = ImageLayerConfig(
        name="img",
        source=img_path,
    )
    return img_layer


@pytest.fixture
def seg_layer_basic(seg_path_precomputed):
    seg_layer = SegmentationLayerConfig(name="seg", source=seg_path_precomputed)
    return seg_layer


@pytest.fixture
def anno_layer_basic():
    anno_layer = AnnotationLayerConfig(name="anno")
    return anno_layer


def test_basic(image_layer, seg_layer_basic, anno_layer_basic):
    sb = StateBuilder(
        [image_layer, seg_layer_basic, anno_layer_basic], target_site="seunglab"
    )
    state_url = sb.render_state()
    assert state_url[:4] == "http"

    state = sb.render_state(return_as="html")
    # padding for link html plus text
    assert state.data == f'<a href="{state_url}" target="_blank">Neuroglancer Link</a>'

    state = sb.render_state(return_as="dict")
    assert isinstance(state, OrderedDict)

    state = sb.render_state(return_as="json")
    assert type(state) is str

    state = sb.render_state(return_as="viewer")
    assert len(state.layer_names) == 3


def test_basic_cave_explorer(image_layer, seg_layer_basic, anno_layer_basic):
    sb = StateBuilder(
        [image_layer, seg_layer_basic, anno_layer_basic], target_site="cave-explorer"
    )
    state_url = sb.render_state()
    assert state_url[:4] == "http"

    state = sb.render_state(return_as="html")
    # padding for link html plus text
    assert state.data == f'<a href="{state_url}" target="_blank">Neuroglancer Link</a>'

    state = sb.render_state(return_as="dict")
    assert isinstance(state, (OrderedDict, dict))  # Cave-explorer uses dict

    state = sb.render_state(return_as="json")
    assert type(state) is str

    state = sb.render_state(return_as="viewer")
    assert len(state.layer_names) == 3


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_segmentation_layer(soma_df, seg_path_precomputed, target_site):
    seg_layer = SegmentationLayerConfig(
        name="seg",
        source=seg_path_precomputed,
        fixed_ids=[1000],
        selected_ids_column="pt_root_id",
    )

    sb = StateBuilder(layers=[seg_layer], target_site=target_site)
    state = sb.render_state(soma_df, return_as="dict")
    print(state["layers"])
    assert (
        648518346349538466 in state["layers"][0]["segments"]
        or "648518346349538466"
        in state["layers"][0]["segments"]  # Cave-explorer uses strings for ids
    )
    assert (
        1000 in state["layers"][0]["segments"]
        or "1000" in state["layers"][0]["segments"]
    )


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_segmentation_layer_color(soma_df, seg_path_precomputed, target_site):
    soma_df["color"] = ["#fdd4c2", "#fca082", "#fb694a", "#e32f27", "#b11218"]
    seg_layer = SegmentationLayerConfig(
        name="seg",
        source=seg_path_precomputed,
        selected_ids_column="pt_root_id",
        color_column="color",
    )
    sb = StateBuilder(layers=[seg_layer], target_site=target_site)
    state = sb.render_state(soma_df, return_as="dict")
    assert state["layers"][0]["segmentColors"]["648518346349538715"] == "#e32f27"


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_segmentation_layer_options(soma_df, seg_path_precomputed, target_site):
    segmentation_view_options = {"alpha_selected": 0.6, "alpha_3d": 0.2}
    seg_layer = SegmentationLayerConfig(
        seg_path_precomputed, view_kws=segmentation_view_options
    )
    sb = StateBuilder([seg_layer], target_site=target_site)
    state = sb.render_state(return_as="dict")
    assert state["layers"][0]["selectedAlpha"] == 0.6


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_annotations(pre_syn_df, target_site):
    points = PointMapper(point_column="pre_pt_position")
    points_2 = PointMapper(point_column="ctr_pt_position")
    anno_layer = AnnotationLayerConfig(name="annos", mapping_rules=[points, points_2])

    sb = StateBuilder([anno_layer], target_site=target_site)
    state = sb.render_state(pre_syn_df, return_as="dict")
    assert len(state["layers"][0]["annotations"]) == 10


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_array_annotations(target_site):
    data = np.array([[1, 2, 3], [3, 4, 5], [6, 5, 3], [1, 2, 1]])
    anno_layer = AnnotationLayerConfig(
        name="annos", array_data=True, mapping_rules=PointMapper()
    )
    sb = StateBuilder([anno_layer], target_site=target_site)
    state = sb.render_state(data, return_as="dict")
    assert len(state["layers"][0]["annotations"]) == 4


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_annotations_line(pre_syn_df, target_site):
    lines = LineMapper(
        point_column_a="pre_pt_position", point_column_b="post_pt_position"
    )
    anno_layer = AnnotationLayerConfig(name="annos", mapping_rules=lines)

    sb = StateBuilder([anno_layer], target_site=target_site)
    state = sb.render_state(pre_syn_df, return_as="dict")
    assert len(state["layers"][0]["annotations"]) == 5


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_annotations_sphere(soma_df, target_site):
    soma_df["radius"] = 5000
    spheres = SphereMapper(center_column="pt_position", radius_column="radius")
    anno_layer = AnnotationLayerConfig(name="annos", mapping_rules=spheres)
    sb = StateBuilder([anno_layer], target_site=target_site)
    state = sb.render_state(soma_df, return_as="dict")
    assert len(state["layers"][0]["annotations"]) == 5


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_annotations_description(soma_df, target_site):
    soma_df["desc"] = ["a", "b", "c", "d", "e"]
    points = PointMapper("pt_position", description_column="desc")
    anno_layer = AnnotationLayerConfig(name="annos", mapping_rules=points)
    sb = StateBuilder([anno_layer], target_site=target_site)
    state = sb.render_state(soma_df, return_as="dict")
    assert state["layers"][0]["annotations"][2]["description"] == "c"


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_annotations_linked(soma_df, soma_df_Int64, seg_path_precomputed, target_site):
    seg_layer = SegmentationLayerConfig(seg_path_precomputed, name="seg")

    points = PointMapper("pt_position", linked_segmentation_column="pt_root_id")
    anno_layer = AnnotationLayerConfig(
        mapping_rules=points, linked_segmentation_layer="seg"
    )

    sb = StateBuilder([seg_layer, anno_layer], target_site=target_site)
    state = sb.render_state(soma_df, return_as="dict")
    assert 648518346349538715 in np.squeeze(
        state["layers"][1]["annotations"][3]["segments"]
    )

    state = sb.render_state(soma_df_Int64, return_as="dict")
    print("TYPE IS", state["layers"][1]["annotations"][0]["segments"])
    assert len(np.squeeze(state["layers"][1]["annotations"][0]["segments"])) == 0


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_annotation_tags(soma_df, target_site):
    tag_list = ["i", "e"]

    points = PointMapper("pt_position", tag_column="cell_type")
    anno_layer = AnnotationLayerConfig(mapping_rules=points, tags=tag_list)

    sb = StateBuilder([anno_layer], target_site=target_site)
    state = sb.render_state(soma_df, return_as="dict")
    if target_site == "seunglab":
        assert 2 in state["layers"][0]["annotations"][1]["tagIds"]
    else:
        # Not implemented yet in cave-explorer.
        assert True


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_annotation_groups(pre_syn_df, target_site):
    df = pre_syn_df.copy()
    df["group"] = [1, 1, np.nan, 2.0, 2.0]
    points = PointMapper("ctr_pt_position", group_column="group")
    anno_layer = AnnotationLayerConfig(mapping_rules=[points])
    sb = StateBuilder([anno_layer], target_site=target_site)
    state = sb.render_state(df, return_as="dict")
    if target_site == "seunglab" or target_site is None:
        assert len(state["layers"][0]["annotations"]) == 7
    else:
        assert len(state["layers"][0]["annotations"]) == 5


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_view_options(soma_df, image_layer, target_site):
    view_options = {
        "layout": "4panel",
        "show_slices": True,
        "zoom_3d": 500,
        "position": [71832, 54120, 1089],
    }

    sb = StateBuilder([image_layer], view_kws=view_options, target_site=target_site)
    state = sb.render_state(return_as="dict")
    if target_site == "seunglab":
        assert state["navigation"]["pose"]["position"]["voxelCoordinates"] == [
            71832,
            54120,
            1089,
        ]
    elif target_site == "cave-explorer":
        assert state["position"] == [
            71832,
            54120,
            1089,
        ]

    points = PointMapper("pt_position", set_position=True)
    anno_layer = AnnotationLayerConfig(mapping_rules=points)

    sb = StateBuilder([image_layer, anno_layer], target_site=target_site)
    state = sb.render_state(soma_df, return_as="dict")
    if target_site == "seunglab":
        assert state["navigation"]["pose"]["position"]["voxelCoordinates"] == list(
            soma_df["pt_position"].loc[0]
        )
    elif target_site == "cave-explorer":
        assert state["position"] == list(soma_df["pt_position"].loc[0])


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_chained(pre_syn_df, post_syn_df, image_layer, target_site):
    # First state builder

    postsyn_mapper = LineMapper(
        point_column_a="pre_pt_position", point_column_b="ctr_pt_position"
    )
    postsyn_annos = AnnotationLayerConfig(
        "post", color="#00CCCC", mapping_rules=postsyn_mapper
    )

    postsyn_sb = StateBuilder(
        layers=[image_layer, postsyn_annos], target_site=target_site
    )

    # Second state builder
    presyn_mapper = LineMapper(
        point_column_a="ctr_pt_position", point_column_b="post_pt_position"
    )
    presyn_annos = AnnotationLayerConfig(
        "pre", color="#CC1111", mapping_rules=presyn_mapper
    )

    presyn_sb = StateBuilder(layers=[presyn_annos], target_site=target_site)

    # Chained state builder
    chained_sb = ChainedStateBuilder([postsyn_sb, presyn_sb])
    state = chained_sb.render_state([post_syn_df, pre_syn_df], return_as="dict")
    assert len(state["layers"]) == 3


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_mapping_sets(pre_syn_df, post_syn_df, image_layer, target_site):
    postsyn_mapper = LineMapper(
        point_column_a="pre_pt_position",
        point_column_b="ctr_pt_position",
        mapping_set="post",
    )
    postsyn_annos = AnnotationLayerConfig(
        "post", color="#00CCCC", mapping_rules=postsyn_mapper
    )

    presyn_mapper = LineMapper(
        point_column_a="ctr_pt_position",
        point_column_b="post_pt_position",
        mapping_set="pre",
    )
    presyn_annos = AnnotationLayerConfig(
        "pre", color="#CC1111", mapping_rules=presyn_mapper
    )

    sb = StateBuilder(
        [image_layer, postsyn_annos, presyn_annos], target_site=target_site
    )
    state = sb.render_state({"pre": pre_syn_df, "post": post_syn_df}, return_as="dict")
    assert len(state["layers"][1]["annotations"]) > 0
    assert len(state["layers"][2]["annotations"]) > 0


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_split_points(split_point_df, seg_path_graphene, target_site):
    split_mapper = SplitPointMapper(
        id_column="seg_id", point_column="pts", team_column="team", focus=True
    )
    seg = SegmentationLayerConfig(seg_path_graphene, split_point_map=split_mapper)
    sb = StateBuilder([seg], target_site=target_site)
    state = sb.render_state(split_point_df, return_as="dict")
    seg_layer = state["layers"][0]
    if target_site == "seunglab" or target_site is None:
        assert len(seg_layer["graphOperationMarker"][1]["annotations"]) == 2
    else:
        # Not implemented in cave-explorer
        assert True


@pytest.mark.parametrize("target_site", [None, "seunglab", "cave-explorer"])
def test_timestamp(seg_path_graphene, target_site):
    ts = 12345
    seg = SegmentationLayerConfig(seg_path_graphene, timestamp=ts)
    sb = StateBuilder([seg], target_site=target_site)
    state = sb.render_state(return_as="dict")
    seg_layer = state["layers"][0]
    if target_site == "seunglab":
        assert int(seg_layer["timestamp"]) == 12345
    else:
        # Not implemented yet in cave-explorer
        assert True
