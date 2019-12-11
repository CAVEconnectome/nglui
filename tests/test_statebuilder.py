import pytest
import numpy as np
import pandas as pd
from neuroglancer_annotation_ui.statebuilder import StateBuilder, build_state_direct

def test_statebuilder_basic(img_layer, seg_layer_graphene):
    sb = StateBuilder(image_sources={'img':img_layer},
                      seg_sources={'seg':seg_layer_graphene})
    viewer = sb.render_state(return_as='viewer')
    assert viewer.state.layers['img'].type == 'image'
    assert viewer.state.layers['seg'].type == 'segmentation_with_graph'


def test_statebuilder_selections(img_layer, seg_layer_graphene, df):
    selection_instruction_single = {'seg': ['single_inds']}
    sb = StateBuilder(image_sources={'img':img_layer},
                      seg_sources={'seg':seg_layer_graphene},
                      selected_ids=selection_instruction_single)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][1]['segments']) == 10

    selection_instruction_multi_a = {'seg': ['multi_inds_array']}
    sb = StateBuilder(image_sources={'img':img_layer},
                      seg_sources={'seg':seg_layer_graphene},
                      selected_ids=selection_instruction_multi_a)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][1]['segments']) == 20

    selection_instruction_multi_b = {'seg': ['multi_inds_list']}
    sb = StateBuilder(image_sources={'img':img_layer},
                      seg_sources={'seg':seg_layer_graphene},
                      selected_ids=selection_instruction_multi_b)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][1]['segments']) == 20

    selection_instruction_multicol = {'seg': ['single_inds', 'multi_inds_array']}
    sb = StateBuilder(image_sources={'img':img_layer},
                      seg_sources={'seg':seg_layer_graphene},
                      selected_ids=selection_instruction_multicol)
    statejson = sb.render_state(data=df, return_as='json')
    print(statejson['layers'][1]['segments'])
    assert len(statejson['layers'][1]['segments']) == 30


def test_statebuilder_annotation_types(img_layer, seg_layer_graphene, df):
    point_annotations_single = {'annos': ['single_pts']}
    sb = StateBuilder(image_sources={'img':img_layer},
                      seg_sources={'seg':seg_layer_graphene},
                      point_annotations=point_annotations_single)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 10

    anno_instructions_array = {'annos': ['multi_pts_array']}
    sb = StateBuilder(image_sources={'img':img_layer},
                      seg_sources={'seg':seg_layer_graphene},
                      point_annotations=anno_instructions_array)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 20

    anno_instructions_la = {'annos': ['multi_pts_list_array']}
    sb = StateBuilder(image_sources={'img':img_layer},
                  seg_sources={'seg':seg_layer_graphene},
                  point_annotations=anno_instructions_la)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 20

    anno_instructions_ll = {'annos': ['multi_pts_list_list']}
    sb = StateBuilder(image_sources={'img':img_layer},
                  seg_sources={'seg':seg_layer_graphene},
                  point_annotations=anno_instructions_ll)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 20

    anno_instructions_multicolumn = {'annos': ['single_pts', 'multi_pts_array']}
    sb = StateBuilder(image_sources={'img':img_layer},
                  seg_sources={'seg':seg_layer_graphene},
                  point_annotations=anno_instructions_multicolumn)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 30


def test_statebuilder_complex_annotations(img_layer, seg_layer_graphene, df):
    line_instructions = {'annos': [['line_a', 'line_b']]}
    sb = StateBuilder(image_sources={'img': img_layer},
                      seg_sources={'seg': seg_layer_graphene},
                      line_annotations=line_instructions)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 10


    sphere_instructions = {'annos': [['line_a', 'radius']]}
    sb = StateBuilder(image_sources={'img': img_layer},
                      seg_sources={'seg': seg_layer_graphene},
                      sphere_annotations=sphere_instructions)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 10

    anno_instructions_multicolumn = {'annos': ['single_pts', 'multi_pts_array']}
    sb = StateBuilder(image_sources={'img': img_layer},
                      seg_sources={'seg': seg_layer_graphene},
                      point_annotations=anno_instructions_multicolumn,
                      line_annotations=line_instructions,
                      sphere_annotations=sphere_instructions)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 50

    sphere_instructions = {'sphere_annos': [['line_a', 'radius']]}
    line_instructions = {'line_annos': [['line_a', 'line_b']],
                         'line_annos_2': [['line_a', 'line_b']]}
    sb = StateBuilder(image_sources={'img': img_layer},
                      seg_sources={'seg': seg_layer_graphene},
                      line_annotations=line_instructions,
                      sphere_annotations=sphere_instructions)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers']) == 5

def test_statebuilder_direct(img_layer, seg_layer_graphene, df):
    img_sources = {'img': img_layer}
    seg_sources = {'seg': seg_layer_graphene}
    statejson = build_state_direct(point_annotations=np.vstack(df['single_pts'].values),
                                   state_kws={'image_sources': img_sources, 'seg_sources': seg_sources},
                                   return_as='json')
    assert len(statejson['layers'][2]['annotations'])==10
