from collections.abc import Iterable
from annotationframeworkclient.infoservice import InfoServiceClient
import numpy as np
import pandas as pd


def bucket_of_values(col, data, item_is_array=False, array_width=3):
    '''
    Use to get a flat array of items when you don't know if it's already a collection or a collection of iterables.
    Parameters:
        dataseries: Pandas dataseries with either all items or all a collection of items.
                    If the item is expected to be an nd-array, use item_shape to define what an element is.
    '''

    if len(data) == 0:
        return []

    dataseries = data[col]
    dataseries = dataseries[~dataseries.isnull()]

    if item_is_array:
        # If already an m x n array, just vstack. Else, need to stack every element first.
        if type(dataseries.iloc[0]) is np.ndarray:
            if len(data) > 1:
                return np.vstack(dataseries.values)
            else:
                return dataseries.values[0].reshape(1, -1)
        else:
            if len(data) > 1:
                return np.vstack(dataseries.map(np.vstack)).reshape(-1, array_width)
            else:
                return np.vstack(dataseries).reshape(1, -1)
    else:
        if isinstance(dataseries.iloc[0], Iterable):
            return np.concatenate(dataseries.values)
        else:
            return dataseries.values


def make_basic_dataframe(point_annotations={}, line_annotations={}, sphere_annotations={}):
    """Makes a dataframe containing pre-computed data to be read by a StateBuilder object.
    """
    dfs = []

    pals = {}
    for ii, (ln, data) in enumerate(point_annotations.items()):
        col_name = f'point_{ii}'
        if isinstance(data, np.ndarray):
            data = data.tolist()
        df = pd.DataFrame()
        df[col_name] = data
        dfs.append(df)
        pals[ln] = [col_name]

    lals = {}
    for ii, (ln, (data_a, data_b)) in enumerate(line_annotations.items()):
        col_name_a = f'line_{ii}_a'
        col_name_b = f'line_{ii}_b'
        if isinstance(data_a, np.ndarray):
            data_a = data_a.tolist()
        if isinstance(data_b, np.ndarray):
            data_b = data_b.tolist()
        df = pd.DataFrame()
        df[col_name_a] = data_a
        df[col_name_b] = data_b
        dfs.append(df)
        lals[ln] = [[col_name_a, col_name_b]]

    sals = {}
    for ii, (ln, (data_c, data_r)) in enumerate(sphere_annotations.items()):
        col_name_c = f'sphere_{ii}_c'
        col_name_r = f'sphere_{ii}_r'
        if isinstance(data_c, np.ndarray):
            data_c = data_c.tolist()
        if isinstance(data_r, np.ndarray):
            data_r = data_r.tolist()
        df = pd.DataFrame()
        df[col_name_c] = data_c
        df[col_name_r] = data_r
        dfs.append(df)
        sals[ln] = [[col_name_c, col_name_r]]

    if len(dfs) > 0:
        df_out = pd.concat(dfs, sort=False)
    else:
        df_out = pd.DataFrame()
    return df_out, pals, lals, sals


def sources_from_infoclient(dataset_name, segmentation_type='default', image_layer_name='img', seg_layer_name='seg'):
    """Generate an img_source and seg_source dict from the info service. Will default to graphene and fall back to flat segmentation, unless otherwise specified. 

    Parameters
    ----------
    dataset_name : str
        InfoService dataset name
    segmentation_type : 'default', 'graphene' or 'flat', optional
        Choose which type of segmentation to use. 'default' will try graphene first and fall back to flat. Graphene or flat will
        only use the specified type or give nothing. By default 'default'. 
    image_layer_name : str, optional
        Layer name for the imagery, by default 'img'
    seg_layer_name : str, optional
        Layer name for the segmentation, by default 'seg'
    """
    info_client = InfoServiceClient(dataset_name=dataset_name)
    image_source = {image_layer_name: info_client.image_source(
        format_for='neuroglancer')}

    if segmentation_type == 'default':
        if info_client.pychunkedgraph_segmentation_source() is None:
            segmentation_type = 'flat'
        else:
            segmentation_type = 'graphene'

    if segmentation_type == 'graphene':
        seg_source = {seg_layer_name: info_client.pychunkedgraph_segmentation_source(
            format_for='neuroglancer')}
    elif segmentation_type == 'flat':
        seg_source = {seg_layer_name: info_client.flat_segmentation_source(
            format_for='neuroglancer')}
    else:
        seg_source = {}
    return image_source, seg_source
