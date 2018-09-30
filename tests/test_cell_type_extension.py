import pytest
import numpy as np
import re
from neuroglancer_annotation_ui.base import EasyViewer, AnnotationManager
from neuroglancer_annotation_ui.cell_type_point_extension import CellTypeExtension
from neuroglancer_annotation_ui.annotation import point_annotation

def test_cell_type_point_extension(annotation_client, img_layer, seg_layer, s1, s2):
    manager = AnnotationManager(annotation_client=annotation_client)
    manager.add_image_layer('image', img_layer)
    manager.add_segmentation_layer('seg', seg_layer)

    manager.add_extension('cell_type', CellTypeExtension.set_db_tables('CellTypeExtensionTest',
                                                                       {'cell_type':'cell_type_ai_manual'}))
    ct_ext = manager.extensions['cell_type']

    # Add a cell type correctly with a correct description field
    ct_ext.update_center_point('spiny_4', s1)
    assert ct_ext.points.points['ctr_pt'] is not None
    assert len(manager.viewer.state.layers['cell_type_tool'].annotations) == 1
    ct_ext.trigger_upload(s2)

    # Internal record?
    assert len(ct_ext.annotation_df) == 1

    # Annotation in the viewer state?
    assert len(manager.viewer.state.layers['cell_types'].annotations) == 1
    assert len(manager.viewer.state.layers['cell_type_tool'].annotations) == 0

    # Is the annotation in the database?
    anno_id = ct_ext.annotation_df.anno_id.values[0]
    a_type, a_id = ct_ext.parse_anno_id(anno_id)
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['type'] == 'cell_type_local'

    # Can we update the cell type info?
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['cell_type'] == 'spiny_4'
    with manager.viewer.txn() as s:
        s.layers['cell_types'].annotations[0].description = 'aspiny_s_1'

    manager.viewer.set_selected_layer('cell_types')
    manager.viewer.select_annotation('cell_types',
                    manager.viewer.state.layers['cell_types'].annotations[0].id)
    manager.update_annotation(None)

    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['cell_type'] == 'aspiny_s_1'

    manager.viewer.set_selected_layer('cell_type_tool')

    # Can we reload the annotation from the server?
    with manager.viewer.txn() as s:
        s.layers['cell_types'].annotations[0].point = [2., 2., 2.]

    assert np.all(manager.viewer.state.layers['cell_types'].annotations[0].point 
        == np.array([2.,2.,2.]))
    manager.reload_all_annotations(None)
    assert np.all(manager.viewer.state.layers['cell_types'].annotations[0].point 
        == np.array([1.,1.,1.]))

    # Can we cancel an annotation in media res?
    ct_ext.update_center_point( 'spiny_4', s2 )
    manager.cancel_annotation(None)
    assert ct_ext.points.points['ctr_pt'] == None

    # Does annotation fail on an incorrect layer?
    manager.viewer.set_selected_layer('cell_types')
    ct_ext.update_center_point( 'spiny_4', s2 )
    assert ct_ext.points.points['ctr_pt'] == None

    manager.viewer.set_selected_layer('cell_type_tool')

    # Can we delete an annotation?
    anno_id = ct_ext.annotation_df.anno_id.values[0]
    a_type, a_id = ct_ext.parse_anno_id(anno_id)
    d = manager.annotation_client.get_annotation(a_type, a_id)
    assert d['type'] == 'cell_type_local'

    manager.viewer.select_annotation('cell_types',
                                     manager.viewer.state.layers['cell_types'].annotations[0].id)

    manager.delete_annotation(None)
    assert len(manager.viewer.state.layers['cell_types'].annotations) == 1
    manager.delete_annotation(None)

    assert len(manager.viewer.state.layers['cell_types'].annotations) == 0

    assert len(ct_ext.annotation_df) == 0

    try:
        d = manager.annotation_client.get_annotation(a_type, a_id)
        assert False
    except:
        assert True

    # Does the ivscc_m a_spiny_s_ work?
    manager.viewer.set_selected_layer('cell_type_tool')
    ct_ext.update_center_point_spiny(s1)
    with manager.viewer.txn() as s:
        s.layers['cell_type_tool'].annotations[0].description = '{}4'.format(
            s.layers['cell_type_tool'].annotations[0].description)
    ct_ext.trigger_upload(s2)
    assert len(ct_ext.annotation_df) == 1
    print(manager.viewer.state.layers['cell_types'].annotations[0].description)
    assert re.search('spiny_4', manager.viewer.state.layers['cell_types'].annotations[0].description) != None

    manager.viewer.select_annotation('cell_types',
                                     manager.viewer.state.layers['cell_types'].annotations[0].id)
    manager.delete_annotation(None)
    manager.delete_annotation(None)

    # Does the ivscc_m a_spiny_d_ work?
    manager.viewer.set_selected_layer('cell_type_tool')
    ct_ext.update_center_point_aspiny(s1)
    with manager.viewer.txn() as s:
        s.layers['cell_type_tool'].annotations[0].description = '{}7'.format(
            s.layers['cell_type_tool'].annotations[0].description)
    ct_ext.trigger_upload(s2)
    assert len(ct_ext.annotation_df) == 1
    print(manager.viewer.state.layers['cell_types'].annotations[0].description)
    assert re.search('aspiny_s_7', manager.viewer.state.layers['cell_types'].annotations[0].description) != None

    manager.viewer.select_annotation('cell_types',
                                     manager.viewer.state.layers['cell_types'].annotations[0].id)
    manager.delete_annotation(None)
    manager.delete_annotation(None)

    # Does the valence:e work?
    manager.viewer.set_selected_layer('cell_type_tool')
    ct_ext.update_center_point_e(s1)
    ct_ext.trigger_upload(s2)
    assert len(ct_ext.annotation_df) == 1
    print(manager.viewer.state.layers['cell_types'].annotations[0].description)
    assert re.search('e\n', manager.viewer.state.layers['cell_types'].annotations[0].description) != None

    manager.viewer.select_annotation('cell_types',
                                     manager.viewer.state.layers['cell_types'].annotations[0].id)
    manager.delete_annotation(None)
    manager.delete_annotation(None)

    # Does the valence:i work?
    manager.viewer.set_selected_layer('cell_type_tool')
    ct_ext.update_center_point_i(s1)
    ct_ext.trigger_upload(s2)
    assert len(ct_ext.annotation_df) == 1
    print(manager.viewer.state.layers['cell_types'].annotations[0].description)
    assert re.search('i\n', manager.viewer.state.layers['cell_types'].annotations[0].description) != None

    manager.viewer.select_annotation('cell_types',
                                     manager.viewer.state.layers['cell_types'].annotations[0].id)
    manager.delete_annotation(None)
    manager.delete_annotation(None)

    # Does the uncertain work?
    manager.viewer.set_selected_layer('cell_type_tool')
    ct_ext.update_center_point_uncertain(s1)
    ct_ext.trigger_upload(s2)
    assert len(ct_ext.annotation_df) == 1
    print(manager.viewer.state.layers['cell_types'].annotations[0].description)
    assert re.search('uncertain\n', manager.viewer.state.layers['cell_types'].annotations[0].description) != None

    manager.viewer.select_annotation('cell_types',
                                     manager.viewer.state.layers['cell_types'].annotations[0].id)
    manager.delete_annotation(None)
    manager.delete_annotation(None)


    # Does nonsense fail validation?



    # Does updating also validate?
 