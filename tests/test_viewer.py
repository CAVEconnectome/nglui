import pytest
import numpy as np
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



def test_synapse_annotations(annotation_client, img_layer, seg_layer, s1, s2, s3):
    manager = AnnotationManager(annotation_client=annotation_client)
    manager.add_image_layer('image', img_layer)
    manager.add_segmentation_layer('seg', seg_layer)
    manager.add_extension('synapse', SynapseExtension.set_db_tables('SynapseExtensionTest',
                                                                     {'synapse':'synapse'}))

    # Use the synapses as an example for using an annotation
    syn_ext = manager.extensions['synapse']

    # Test adding a synapse correctly
    syn_ext.update_presynaptic_point( s1 )
    assert syn_ext.points.points['pre_pt'] is not None
    syn_ext.update_postsynaptic_point( s3 )
    syn_ext.update_center_synapse_point( s2 )

    # Is the annotation now in the internal record?
    assert len(syn_ext.annotation_df) == 3
    
    # Is the annotation in the viewer state?
    assert len(manager.viewer.state.layers['synapses'].annotations) == 1
    assert len(manager.viewer.state.layers['synapses_post'].annotations) == 1
    assert len(manager.viewer.state.layers['synapses_pre'].annotations) == 1

    # Is the annotation in the annotation server?
    anno_id = syn_ext.annotation_df.anno_id.values[0]
    a_type, a_id = syn_ext.parse_anno_id(anno_id)
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['type'] == 'synapse'
  
    # Can we update the synapse?
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['ctr_pt']['position'] != [4,4,4]
    with manager.viewer.txn() as s:
        s.layers['synapses'].annotations[0].point = [4.,4.,4.]
    manager.viewer.select_annotation('synapses', manager.viewer.state.layers['synapses'].annotations[0].id)
    manager.update_annotation(None)
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['ctr_pt']['position'] == [4,4,4]

    # Test reloading a synapse, make sure we have as many annotations as we started with
    with manager.viewer.txn() as s:
        s.layers['synapses'].annotations[0].point = [1.,1.,1.]
    manager.reload_all_annotations(None)
    assert np.all(manager.viewer.state.layers['synapses'].annotations[0].point == np.array([4,4,4]))
    assert len(syn_ext.annotation_df) == 3
    assert len(manager.viewer.state.layers['synapses'].annotations) == 1

    # Test re-assigning pre/post points then adding the final point
    syn_ext.update_presynaptic_point( s1 )
    manager.cancel_annotation(None)
    assert syn_ext.points.points['pre_pt'] == None

    # Test trying to add a synapse on the incorrect layer
    manager.viewer.set_selected_layer('seg')
    syn_ext.update_presynaptic_point( s1 )
    assert syn_ext.points.points['pre_pt'] == None
    manager.viewer.set_selected_layer('synapses')

    # Test deleting a synapse
    anno_id = syn_ext.annotation_df.anno_id.values[0]
    a_type, a_id = syn_ext.parse_anno_id(anno_id)
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['type'] == 'synapse'

    manager.viewer.select_annotation('synapses', manager.viewer.state.layers['synapses'].annotations[0].id)

    manager.delete_annotation(None)
    assert len(manager.viewer.state.layers['synapses'].annotations) == 1
    manager.delete_annotation(None)

    assert len(manager.viewer.state.layers['synapses'].annotations) == 0
    assert len(manager.viewer.state.layers['synapses_pre'].annotations) == 0
    assert len(manager.viewer.state.layers['synapses_post'].annotations) == 0

    assert len(syn_ext.annotation_df) == 0
    try:
        d = manager.annotation_client.get_annotation(a_type, a_id)
        assert False
    except:
        assert True

    ### Same things, without annotation client