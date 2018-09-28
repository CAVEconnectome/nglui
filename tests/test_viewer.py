import pytest
from neuroglancer_annotation_ui.base import EasyViewer, AnnotationManager
from neuroglancer_annotation_ui.synapse_extension import SynapseExtension
from neuroglancer_annotation_ui.annotation import point_annotation

def test_viewer(img_layer, seg_layer):
    viewer = EasyViewer()
    viewer.add_image_layer('test_image_layer', img_layer)
    assert 'test_image_layer' in viewer.layer_names

    viewer.add_segmentation_layer('test_seg_layer', seg_layer)
    assert viewer.state.layers['test_seg_layer'].type == 'segmentation'

    viewer.add_selected_objects('test_seg_layer', 22060)
    assert 22060 in viewer.state.layers['test_seg_layer'].segments

    new_pos = [3926, 3528, 4070]
    viewer.set_position(new_pos)
    assert all(new_pos[ii]==int(val) for ii, val in enumerate(viewer.state.position.voxel_coordinates))

    anno_ln = 'test_anno_layer'
    viewer.add_annotation_layer(layer_name=anno_ln,
                                color='#00bb33')
    viewer.set_annotation_layer_color(anno_ln, color='#aabbcc')
    assert viewer.state.layers[anno_ln].annotationColor == '#aabbcc'

    new_anno = point_annotation([1,2,3])

    viewer.add_annotation(anno_ln, new_anno)
    assert len(viewer.state.layers[anno_ln].annotations) == 1

    viewer.update_description({anno_ln: [new_anno.id]}, 'test_description')
    assert viewer.state.layers[anno_ln].annotations[0].description == 'test_description'

    viewer.remove_annotation(anno_ln, new_anno.id)
    assert len(viewer.state.layers[anno_ln].annotations) == 0


def test_manager(annotation_client):
    manager = AnnotationManager(annotation_client=annotation_client)

    assert "backspace" in manager.key_bindings
    assert "shift+keyc" in manager.key_bindings
    assert "shift+control+keyu" in manager.key_bindings
    assert "shift+control+keyr" in manager.key_bindings

    manager.add_extension('synapse',SynapseExtension.set_db_tables('SynapseExtensionTest',
                                                         {'synapse':'synapse'}))

    assert 'synapse' in manager.extensions

    # Should fail due to overlapping key bindings
    manager.add_extension('conflict',SynapseExtension.set_db_tables('ConflictExtension',
                                                         {'synapse':'synapse'}))
    assert 'conflict' not in manager.extensions



