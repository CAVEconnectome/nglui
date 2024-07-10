# from nglui import annotation
import numpy as np


def test_viewer_layers(viewer, img_path, seg_path_precomputed, seg_path_graphene):
    viewer.add_image_layer("img", img_path)
    assert "img" in viewer.layer_names

    viewer.add_segmentation_layer("seg_pre", seg_path_precomputed)
    assert viewer.state.layers["seg_pre"].type == "segmentation"

    viewer.add_segmentation_layer("seg_graph", seg_path_graphene)
    assert viewer.state.layers["seg_graph"].type == "segmentation_with_graph"

    id_val = 1000
    viewer.add_selected_objects("seg_graph", [id_val])
    assert id_val in viewer.state.layers["seg_graph"].segments


def test_viewer_layers_cave_explorer(viewer_cave_explorer, img_path, seg_path_precomputed, seg_path_graphene):
    viewer_cave_explorer.add_image_layer("img", img_path)
    assert "img" in viewer_cave_explorer.layer_names

    viewer_cave_explorer.add_segmentation_layer("seg_pre", seg_path_precomputed)
    assert viewer_cave_explorer.state.layers["seg_pre"].type == "segmentation"

    id_val = 1000
    viewer_cave_explorer.add_selected_objects("seg_graph", [id_val])
    assert id_val in viewer_cave_explorer.state.layers["seg_graph"].segments


def test_pose_and_navigation(viewer):
    viewer.set_resolution([8, 8, 40])
    assert np.array_equal(viewer.state.voxel_size, [8, 8, 40])

    new_pos = [3926, 3528, 4070]
    viewer.set_view_options(position=new_pos, zoom_image=4)
    assert all(
        new_pos[ii] == int(val)
        for ii, val in enumerate(viewer.state.position.voxel_coordinates)
    )

def test_pose_and_navigation_cave_explorer(viewer_cave_explorer):
    viewer_cave_explorer.set_resolution([8, 8, 40])
    assert np.array_equal(viewer_cave_explorer.state.dimensions.scales * 10**9, [8, 8, 40])

    new_pos = [3926, 3528, 4070]
    viewer_cave_explorer.set_view_options(position=new_pos, zoom_image=4)
    assert all(
        new_pos[ii] == int(val)
        for ii, val in enumerate(viewer_cave_explorer.state.position)
    )


def test_annotations(viewer, anno_layer_name):
    anno_ln = anno_layer_name
    viewer.add_annotation_layer(layer_name=anno_ln, color="#00bb33")
    viewer.set_annotation_layer_color(anno_ln, color="#aabbcc")
    assert viewer.state.layers[anno_ln].annotationColor == "#aabbcc"

    pt_anno = viewer.point_annotation([1, 2, 3])
    line_anno = viewer.line_annotation([1, 2, 3], [2, 3, 4])
    sphere_anno = viewer.sphere_annotation([1, 2, 3], 4, 0.1)
    bb_anno = viewer.bounding_box_annotation([1, 2, 3], [2, 3, 4])
    viewer.add_annotations(anno_ln, [pt_anno, line_anno, sphere_anno, bb_anno])
    assert len(viewer.state.layers[anno_ln].annotations) == 4

    # viewer.update_description({anno_ln: [pt_anno.id]}, "test_description")
    # assert viewer.state.layers[anno_ln].annotations[0].description == "test_description"

    viewer.remove_annotations(anno_ln, [pt_anno.id])
    assert len(viewer.state.layers[anno_ln].annotations) == 3

    viewer.clear_annotation_layers([anno_ln])
    assert len(viewer.state.layers[anno_ln].annotations) == 0

def test_annotations_cave_explorer(viewer_cave_explorer, anno_layer_name):
    anno_ln = anno_layer_name
    viewer_cave_explorer.add_annotation_layer(layer_name=anno_ln, color="#00bb33")
    viewer_cave_explorer.set_annotation_layer_color(anno_ln, color="#aabbcc")
    assert viewer_cave_explorer.state.layers[anno_ln].annotationColor == "#aabbcc"

    pt_anno = viewer_cave_explorer.point_annotation([1, 2, 3])
    line_anno = viewer_cave_explorer.line_annotation([1, 2, 3], [2, 3, 4])
    sphere_anno = viewer_cave_explorer.sphere_annotation([1, 2, 3], 4, 0.1)
    bb_anno = viewer_cave_explorer.bounding_box_annotation([1, 2, 3], [2, 3, 4])
    viewer_cave_explorer.add_annotations(anno_ln, [pt_anno, line_anno, sphere_anno, bb_anno])
    assert len(viewer_cave_explorer.state.layers[anno_ln].annotations) == 4

    # viewer.update_description({anno_ln: [pt_anno.id]}, "test_description")
    # assert viewer.state.layers[anno_ln].annotations[0].description == "test_description"

    viewer_cave_explorer.remove_annotations(anno_ln, [pt_anno.id])
    assert len(viewer_cave_explorer.state.layers[anno_ln].annotations) == 3

    viewer_cave_explorer.clear_annotation_layers([anno_ln])
    assert len(viewer_cave_explorer.state.layers[anno_ln].annotations) == 0

# Cave Explorer does not have the below features yet

def test_grouped_annotations(viewer, anno_layer_name):
    pt_anno_1 = viewer.point_annotation([1, 2, 3])
    pt_anno_2 = viewer.point_annotation([4, 5, 6])
    grp = viewer.group_annotations([pt_anno_1, pt_anno_2], return_all=False)
    assert grp.id == pt_anno_1.parent_id


def test_annotation_tags(viewer, anno_layer_name):
    anno_ln = anno_layer_name
    tags = ["tagA", "tagB"]
    tag_dict = {str(ii + 1): tag for ii, tag in zip(range(len(tags)), tags)}
    anno_A = viewer.point_annotation([1, 2, 3], tag_ids=[2])
    viewer.add_annotation_tags(anno_ln, tags)
    viewer.add_annotations(anno_ln, [anno_A])
    anno_a_ids = viewer.state.layers[anno_ln].annotations[0].tag_ids
    assert tag_dict[str(anno_a_ids[0])] == tags[1]
