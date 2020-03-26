from collections.abc import Iterable
from annotationframeworkclient import FrameworkClient
from .statebuilder import ImageLayerConfig, SegmentationLayerConfig
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


def sources_from_infoclient(dataset_name, segmentation_type='graphene', image_kws={}, segmentation_kws={}, client=None):
    """Generate an Image and Segmentation source from a dataset name.

    Parameters
    ----------
    dataset_name : str
        Dataset name for client
    segmentation_type : 'graphene' or 'flat', optional
        Which segmentation type for try first, graphene or flat. It will fall back to the other if not found. By default 'graphene'
    image_kws : dict, optional
        Keyword arguments for an ImageLayerConfig (other than source), by default {}
    segmentation_kws : dict, optional
        Keyword arguments for a SegmentationLayerConfig (other than source), by default {}
    client : InfoClient or None, optional
        Predefined info client for lookup

    Returns
    -------
    ImageLayerConfig
        Config for an image layer in the statebuilder 
    SegmentationLayerConfig
        Config for a segmentation layer in the statebuilder
    """

    if client is None:
        client = FrameworkClient(dataset_name=dataset_name)

    image_source = client.info.image_source(format_for='neuroglancer')
    if segmentation_type == "graphene":
        seg_source = client.info.graphene_source(format_for='neuroglancer')
        if seg_source is None:
            seg_source = client.info.flat_segmentation_source(
                format_for='neuroglancer')
    else:
        seg_source = client.info.flat_segmentation_source(
            format_for='neuroglancer')
        if seg_source is None:
            seg_source = client.info.graphene_source(format_for='neuroglancer')

    image_layer = ImageLayerConfig(image_source, *image_kws)
    seg_layer = SegmentationLayerConfig(seg_source, *segmentation_kws)
    return image_layer, seg_layer
