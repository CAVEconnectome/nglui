import pytest
from neuroglancer_annotation_ui import annotation
import numpy as np

def test_viewer_layers(viewer, img_layer, seg_layer_precomputed, seg_layer_graphene):
    viewer.add_image_layer('img', img_layer)
    assert 'img' in viewer.layer_names

    viewer.add_segmentation_layer('seg_pre', seg_layer_precomputed)
    assert viewer.state.layers['seg_pre'].type == 'segmentation'

    viewer.add_segmentation_layer('seg_graph', seg_layer_graphene)
    assert viewer.state.layers['seg_graph'].type == 'segmentation_with_graph'

    id_val = 1000
    viewer.add_selected_objects('seg_graph', [id_val])
    assert id_val in viewer.state.layers['seg_graph'].segments

def test_pose_and_navigation(viewer, img_layer):
    viewer.add_image_layer('img', img_layer)

    viewer.set_resolution([8,8,40])
    assert np.array_equal(viewer.state.voxel_size, [8,8,40])

    new_pos = [3926, 3528, 4070]
    viewer.set_position(new_pos, zoom_factor=4)
    assert all(new_pos[ii]==int(val) for ii, val in enumerate(viewer.state.position.voxel_coordinates))

    viewer.set_view_options()

def test_annotations(viewer, seg_layer_graphene):
    anno_ln = 'test_anno_layer'
    viewer.add_annotation_layer(layer_name=anno_ln,
                                color='#00bb33')
    viewer.set_annotation_layer_color(anno_ln, color='#aabbcc')
    assert viewer.state.layers[anno_ln].annotationColor == '#aabbcc'

    pt_anno = annotation.point_annotation([1,2,3])
    line_anno = annotation.line_annotation([1,2,3],[2,3,4])
    sphere_anno = annotation.sphere_annotation([1,2,3], 4, 0.1)
    bb_anno = annotation.bounding_box_annotation([1,2,3],[2,3,4])
    viewer.add_annotations(anno_ln, [pt_anno, line_anno, sphere_anno, bb_anno])
    assert len(viewer.state.layers[anno_ln].annotations) == 4

    viewer.update_description({anno_ln: [pt_anno.id]}, 'test_description')
    assert viewer.state.layers[anno_ln].annotations[0].description == 'test_description'

    viewer.remove_annotations(anno_ln, [pt_anno.id])
    assert len(viewer.state.layers[anno_ln].annotations) == 3


def test_annotation_tags(viewer, img_layer):
    viewer.add_image_layer('img', img_layer)
    anno_ln = 'test_anno_layer'
    viewer.add_annotation_layer(layer_name=anno_ln,
                                color='#00bb33')
