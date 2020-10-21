import numpy as np
from ..easyviewer.base import SEGMENTATION_LAYER_TYPES


def layer_names(state):
    """Get all layer names in the state

    Parameters
    ----------
    state : dict
        Neuroglancer state as a JSON dict

    Returns
    -------
    names : list
        List of layer names
    """
    return [l['name'] for l in state['layers']]


def image_layers(state):
    """Get all image layer names in the state

    Parameters
    ----------
    state : dict
        Neuroglancer state as a JSON dict

    Returns
    -------
    names : list
        List of layer names
    """
    return [l['name'] for l in state['layers'] if l['type'] == 'image']


def segmentation_layers(state):
    """Get all segmentation layer names in the state

    Parameters
    ----------
    state : dict
        Neuroglancer state as a JSON dict

    Returns
    -------
    names : list
        List of layer names
    """
    return [l['name'] for l in state['layers'] if l['type'] in SEGMENTATION_LAYER_TYPES]


def annotation_layers(state):
    """Get all annotation layer names in the state

    Parameters
    ----------
    state : dict
        Neuroglancer state as a JSON dict

    Returns
    -------
    names : list
        List of layer names
    """
    return [l['name'] for l in state['layers'] if l['type'] == 'annotation']


def tag_dictionary(state, layer_name):
    """Get the tag id to string dictionary for a layer

    Parameters
    ----------
    state : dict

    layer_name : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    l = get_layer(state, layer_name)
    taginfo = l.get('annotationTags', [])
    tags = {}
    for t in taginfo:
        tags[int(t['id'])] = t['label']
    return tags


def get_layer(state, layer_name):
    """Gets the contents of the layer based on the layer name.

    Parameters
    ----------
    state : dict
        Neuroglancer state as a JSON dict
    layer_name : str
        Name of layer

    Returns
    -------
    layer : dict
        Layer data contents
    """
    layer_ind = layer_names(state).index(layer_name)
    return state['layers'][layer_ind]


def view_settings(state):
    """Get all data about the view state in neuroglancer: position,
    image zoom, orientation and zoom of the 3d view, and voxel size.

    Parameters
    ----------
    state : dict
        Neuroglancer state as JSON dict

    Returns
    -------
    view : dict
        Dictionary with keys: position, zoomFactor, perspectiveOrientation,
        perspectiveZoom, and voxelSize
    """
    view = {}
    view['position'] = state['navigation']['pose']['position']['voxelCoordinates']
    view['zoomFactor'] = state['navigation'].get('zoomFactor', None)
    view['perspectiveOrientation'] = state.get('perspectiveOrientation', None)
    view['perspectiveZoom'] = state.get('perspectiveZoom', None)
    view['voxelSize'] = state['navigation']['pose']['position']['voxelSize']
    return view


def _get_type_annotations(state, layer_name, type):
    l = get_layer(state, layer_name)
    annos = l.get('annotations', [])
    return [anno for anno in annos if anno['type'] == type]


def _get_point_annotations(state, layer_name):
    return _get_type_annotations(state, layer_name, 'point')


def _get_sphere_annotations(state, layer_name):
    return _get_type_annotations(state, layer_name, 'ellipsoid')


def _get_line_annotations(state, layer_name):
    return _get_type_annotations(state, layer_name, 'line')


def _extract_point_data(annos):
    return [[anno.get('point') for anno in annos]]


def _extract_sphere_data(annos):
    pt = [anno.get('center') for anno in annos]
    rad = [anno.get('radii') for anno in annos]
    return [pt, rad]


def _extract_line_data(annos):
    ptA = [anno.get('pointA') for anno in annos]
    ptB = [anno.get('pointB') for anno in annos]
    return [ptA, ptB]


_get_map = {
    'point': _get_point_annotations,
    'line': _get_line_annotations,
    'sphere': _get_sphere_annotations,
}

_extraction_map = {
    'point': _extract_point_data,
    'line': _extract_line_data,
    'sphere': _extract_sphere_data
}


def _generic_annotations(state, layer_name, description, linked_segmentations, tags, type):
    annos = _get_map[type](state, layer_name)
    out = _extraction_map[type](annos)
    if description:
        desc = [anno.get('description', None) for anno in annos]
        out.append(desc)
    if linked_segmentations:
        linked_seg = [[np.uint64(x) for x in anno.get(
            'segments', [])] for anno in annos]
        out.append(linked_seg)
    if tags:
        tag_list = [anno.get('tagIds', []) for anno in annos]
        out.append(tag_list)
    if len(out) == 1:
        return out[0]
    else:
        return out


def point_annotations(state, layer_name, description=False, linked_segmentations=False, tags=False):
    """Get all point annotation points and other info from a layer.

    Parameters
    ----------
    state : dict
        Neuroglancer state as JSON dict
    layer_name : str
        Layer name
    description : bool, optional
        If True, also returns descriptions as well. By default False
    linked_segmentations : bool, optional
        If True, also returns list of linked segmentations, by default False
    tags : bool, optional
        If True, also returns list of tags, by default False

    Returns
    -------
    anno_points : list
        List of N 3-element points (as list)
    anno_descriptions : list
        List of N strings (or None), only returned if description=True.
    anno_linked_segmentations : list
        List of N lists of object ids. Only returned if linked_segmentations=True.
    anno_tags : list
        List of N lists of tag ids. Only returned if tags=True.
    """
    return _generic_annotations(state, layer_name, description, linked_segmentations, tags, 'point')


def line_annotations(state, layer_name, description=False, linked_segmentations=False, tags=False):
    """Get all line annotation points and other info from a layer.

    Parameters
    ----------
    state : dict
        Neuroglancer state as JSON dict
    layer_name : str
        Layer name
    description : bool, optional
        If True, also returns descriptions as well. By default False
    linked_segmentations : bool, optional
        If True, also returns list of linked segmentations, by default False
    tags : bool, optional
        If True, also returns list of tags, by default False

    Returns
    -------
    anno_points_A : list
        List of N 3-element points (as list) of the first point in each line.
    anno_points_B : list
        List of N 3-element points (as list) of the second point in each line.
    anno_descriptions : list
        List of N strings (or None), only returned if description=True.
    anno_linked_segmentations : list
        List of N lists of object ids. Only returned if linked_segmentations=True.
    anno_tags : list
        List of N lists of tag ids. Only returned if tags=True.
    """
    return _generic_annotations(state, layer_name, description, linked_segmentations, tags, 'line')


def sphere_annotations(state, layer_name, description=False, linked_segmentations=False, tags=False):
    """Get all sphere annotation points and other info from a layer.

    Parameters
    ----------
    state : dict
        Neuroglancer state as JSON dict
    layer_name : str
        Layer name
    description : bool, optional
        If True, also returns descriptions as well. By default False
    linked_segmentations : bool, optional
        If True, also returns list of linked segmentations, by default False
    tags : bool, optional
        If True, also returns list of tags, by default False

    Returns
    -------
    anno_points : list
        List of N 3-element center points (as list)
    radius_points : list
        List of N 3-element radii for each axis of the ellipsoid.
    anno_descriptions : list
        List of N strings (or None), only returned if description=True.
    anno_linked_segmentations : list
        List of N lists of object ids. Only returned if linked_segmentations=True.
    anno_tags : list
        List of N lists of tag ids. Only returned if tags=True.
    """
    return _generic_annotations(state, layer_name, description, linked_segmentations, tags, 'sphere')
