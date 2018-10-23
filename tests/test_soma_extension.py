import pytest
import numpy as np
sys.path.append('./src/')
from src.neuroglancer_annotation_ui.base import EasyViewer, AnnotationManager
from src.neuroglancer_annotation_ui.soma_extension import SomaExtension
from src.neuroglancer_annotation_ui.annotation import point_annotation


def test_soma_extension(annotation_client, img_layer, seg_layer, s1, s2):
    manager = AnnotationManager(annotation_client=annotation_client)
    manager.add_image_layer('image', img_layer)
    manager.add_segmentation_layer('seg', seg_layer)

    # Use synapses as an example for using an annotation
    manager.add_extension('somata', SomaExtension.set_db_tables('SomaExtensionTest',
                                                                 {'sphere':'soma_ai_manual'}))
    soma_ext = manager.extensions['somata']

    # Test adding a synapse correctly
    soma_ext.update_center_point( s1 )
    assert soma_ext.points.points['ctr_pt'] is not None
    soma_ext.update_radius_point( s2 )

    # Is the annotation now in the internal record?
    assert len(soma_ext.annotation_df) == 1
    
    # Is the annotation in the viewer state?
    assert len(manager.viewer.state.layers['somata'].annotations) == 1
    assert manager.viewer.state.layers['somata'].annotations[0].type == 'ellipsoid'

    # Is the annotation in the annotation server?
    anno_id = soma_ext.annotation_df.anno_id.values[0]
    a_type, a_id = soma_ext.parse_anno_id(anno_id)
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['type'] == 'sphere'
  
    # Can we update the synapse using local information?
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['ctr_pt']['position'] != [3,3,3]
    with manager.viewer.txn() as s:
        s.layers['somata'].annotations[0].center = [3,3,3]
    manager.viewer.select_annotation('somata', manager.viewer.state.layers['somata'].annotations[0].id)
    manager.update_annotation(None)
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['ctr_pt']['position'] == [3,3,3]

    # Can we reload a synapse from the server and
    # get as many annotations as we started with
    with manager.viewer.txn() as s:
        s.layers['somata'].annotations[0].center = [1.,1.,1.]
    manager.reload_all_annotations(None)
    assert np.all(manager.viewer.state.layers['somata'].annotations[0].center == np.array([3,3,3]))
    assert len(soma_ext.annotation_df) == 1
    assert len(manager.viewer.state.layers['somata'].annotations) == 1

    # Can we cancel an annotation?
    soma_ext.update_center_point( s1 )
    manager.cancel_annotation(None)
    assert soma_ext.points.points['ctr_pt'] == None

    # Does synapse addition fail if we aren't on the right layer?
    manager.viewer.set_selected_layer('seg')
    soma_ext.update_center_point( s1 )
    assert soma_ext.points.points['ctr_pt'] == None
    manager.viewer.set_selected_layer('somata')

    # Can we delete a synapse and have it propagate across all locations?
    # (viewer state, annotation extension bucket, server)
    anno_id = soma_ext.annotation_df.anno_id.values[0]
    a_type, a_id = soma_ext.parse_anno_id(anno_id)
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['type'] == 'sphere'

    manager.viewer.select_annotation('somata', manager.viewer.state.layers['somata'].annotations[0].id)

    manager.delete_annotation(None)
    assert len(manager.viewer.state.layers['somata'].annotations) == 1
    manager.delete_annotation(None)

    assert len(manager.viewer.state.layers['somata'].annotations) == 0

    assert len(soma_ext.annotation_df) == 0
    try:
        d = manager.annotation_client.get_annotation(a_type, a_id)
        assert False
    except:
        assert True

def test_soma_extension_no_annotation_client(img_layer, seg_layer, s1, s2):
    manager = AnnotationManager(annotation_client=None)
    manager.add_image_layer('image', img_layer)
    manager.add_segmentation_layer('seg', seg_layer)

    # Use synapses as an example for using an annotation
    manager.add_extension('somata', SomaExtension.set_db_tables('SomaExtensionTest',
                                                                 {'sphere':'soma_ai_manual'}))
    soma_ext = manager.extensions['somata']

    # Test adding a synapse correctly
    soma_ext.update_center_point( s1 )
    assert soma_ext.points.points['ctr_pt'] is not None
    soma_ext.update_radius_point( s2 )

    # Is the annotation now in the internal record?
    assert len(soma_ext.annotation_df) == 1
    
    # Is the annotation in the viewer state?
    assert len(manager.viewer.state.layers['somata'].annotations) == 1
    assert manager.viewer.state.layers['somata'].annotations[0].type == 'ellipsoid'

    # Can we cancel an annotation?
    soma_ext.update_center_point( s1 )
    manager.cancel_annotation(None)
    assert soma_ext.points.points['ctr_pt'] == None

    # Does synapse addition fail if we aren't on the right layer?
    manager.viewer.set_selected_layer('seg')
    soma_ext.update_center_point( s1 )
    assert soma_ext.points.points['ctr_pt'] == None
    manager.viewer.set_selected_layer('somata')

    # Can we delete a synapse and have it propagate across all locations?
    # (viewer state, annotation extension bucket, server)
    manager.viewer.select_annotation('somata', manager.viewer.state.layers['somata'].annotations[0].id)

    manager.delete_annotation(None)
    assert len(manager.viewer.state.layers['somata'].annotations) == 1
    manager.delete_annotation(None)

    assert len(manager.viewer.state.layers['somata'].annotations) == 0

    assert len(soma_ext.annotation_df) == 0
