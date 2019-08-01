import pytest
import numpy as np
import pandas as pd
from neuroglancer_annotation_ui.statebuilder import StateBuilder, FilteredDataStateBuilder

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


def test_statebuilder_annotations(img_layer, seg_layer_graphene, df):
    anno_instructions_single = {'annos': {'points': ['single_pts']}}
    sb = StateBuilder(image_sources={'img':img_layer},
                  seg_sources={'seg':seg_layer_graphene},
                  annotation_layers=anno_instructions_single)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 10

    anno_instructions_array = {'annos': {'points': ['multi_pts_array']}}
    sb = StateBuilder(image_sources={'img':img_layer},
                  seg_sources={'seg':seg_layer_graphene},
                  annotation_layers=anno_instructions_array)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 20

    anno_instructions_la = {'annos': {'points': ['multi_pts_list_array']}}
    sb = StateBuilder(image_sources={'img':img_layer},
                  seg_sources={'seg':seg_layer_graphene},
                  annotation_layers=anno_instructions_la)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 20

    anno_instructions_ll = {'annos': {'points': ['multi_pts_list_list']}}
    sb = StateBuilder(image_sources={'img':img_layer},
                  seg_sources={'seg':seg_layer_graphene},
                  annotation_layers=anno_instructions_ll)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 20

    anno_instructions_multicolumn = {'annos': {'points': ['single_pts', 'multi_pts_array']}}
    sb = StateBuilder(image_sources={'img':img_layer},
                  seg_sources={'seg':seg_layer_graphene},
                  annotation_layers=anno_instructions_multicolumn)
    statejson = sb.render_state(data=df, return_as='json')
    assert len(statejson['layers'][2]['annotations']) == 30
