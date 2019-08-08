import pytest
import numpy as np
import sys
sys.path.append('./src/')
from neuroglancer_annotation_ui.base import EasyViewer, AnnotationManager
from neuroglancer_annotation_ui.synapse_extension import SynapseExtension

def test_synapse_annotations(annotation_client, img_layer, seg_layer, s1, s2, s3):
    manager = AnnotationManager(annotation_client=annotation_client)
    manager.add_image_layer('image', img_layer)
    manager.add_segmentation_layer('seg', seg_layer)

    # Use synapses as an example for using an annotation
    manager.add_extension('synapse', SynapseExtension.set_db_tables('SynapseExtensionTest',
                                                                    {'synapse':'synapse'}))
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
  
    # Can we update the synapse using local information?
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['ctr_pt']['position'] != [4,4,4]
    with manager.viewer.txn() as s:
        s.layers['synapses'].annotations[0].point = [4.,4.,4.]
    manager.viewer.select_annotation('synapses', manager.viewer.state.layers['synapses'].annotations[0].id)
    manager.update_annotation(None)
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['ctr_pt']['position'] == [4,4,4]

    # Can we reload a synapse from the server and
    # get as many annotations as we started with
    with manager.viewer.txn() as s:
        s.layers['synapses'].annotations[0].point = [1.,1.,1.]
    manager.reload_all_annotations(None)
    assert np.all(manager.viewer.state.layers['synapses'].annotations[0].point == np.array([4,4,4]))
    assert len(syn_ext.annotation_df) == 3
    assert len(manager.viewer.state.layers['synapses'].annotations) == 1

    # Can we cancel an annotation?
    syn_ext.update_presynaptic_point( s1 )
    manager.cancel_annotation(None)
    assert syn_ext.points.points['pre_pt'] == None

    # Does synapse addition fail if we aren't on the right layer?
    manager.viewer.set_selected_layer('seg')
    syn_ext.update_presynaptic_point( s1 )
    assert syn_ext.points.points['pre_pt'] == None
    manager.viewer.set_selected_layer('synapses')

    # Can we delete a synapse and have it propagate across all locations?
    # (viewer state, annotation extension bucket, server)
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
def test_extension_without_client(img_layer, seg_layer, s1, s2, s3):
    manager = AnnotationManager()
    manager.add_image_layer('image', img_layer)
    manager.add_segmentation_layer('seg', seg_layer)

    # Use synapses as an example for using an annotation
    manager.add_extension('synapse', SynapseExtension.set_db_tables('SynapseExtensionTest',
                                                                    {'synapse':'synapse'}))
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
  
    # Can we cancel an annotation?
    syn_ext.update_presynaptic_point( s1 )
    manager.cancel_annotation(None)
    assert syn_ext.points.points['pre_pt'] == None

    # Does synapse addition fail if we aren't on the right layer?
    manager.viewer.set_selected_layer('seg')
    syn_ext.update_presynaptic_point( s1 )
    assert syn_ext.points.points['pre_pt'] == None
    manager.viewer.set_selected_layer('synapses')

    # Can we delete a synapse and have it propagate across all locations?
    # (viewer state, annotation extension bucket, server)

    manager.viewer.select_annotation('synapses', manager.viewer.state.layers['synapses'].annotations[0].id)

    manager.delete_annotation(None)
    assert len(manager.viewer.state.layers['synapses'].annotations) == 1
    manager.delete_annotation(None)

    assert len(manager.viewer.state.layers['synapses'].annotations) == 0
    assert len(manager.viewer.state.layers['synapses_pre'].annotations) == 0
    assert len(manager.viewer.state.layers['synapses_post'].annotations) == 0

    assert len(syn_ext.annotation_df) == 0
