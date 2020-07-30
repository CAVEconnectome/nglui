import numpy as npj


def layer_names(state):
    """ Get layer names from a neuroglancer json state """
    return [l['name'] for l in state['layers']]


def image_layers(state):
    return [l['name'] for l in state['layers'] if l['type'] == 'image']


def segmentation_layers(state):
    return [l['name'] for l in state['layers'] if l['type'] == 'segmentation']


def annotation_layers(state):
    return [l['name'] for l in state['layers'] if l['type'] == 'annotation']


def view_settings(state):
    view = {}
    view['position'] = state['navigation']['pose']['position']['voxelCoordinates']
    view['zoomFactor'] = state['zoomFactor']
    view['perspectiveOrientation'] = state['perspectiveOrientation']
    view['perspectiveZoom'] = state['perspectiveZoom']
    view['voxelSize'] = state['navigation']['pose']['position']['voxelSize']
    return view


def get_annotations(state, layers=None, linked_segmentations=False, descriptions=False, tags=False, collapse_single=True, annotation_types='point'):
    if layers is None:
        layers = annotation_layers(state)

    if isinstance(layers, str):
        layers = [layers]

    if len(layers) == 1 and collapse_single:
        return_single = True
    else:
        return_single = False

    if isinstance(annotation_types, str):
        annotation_types = [annotation_types]
        single_anno = True
    else:
        single_anno = False

    annotations = {}
    for ln in layers:
        annotations[ln] = {}

        for l in state['layers']:
            if l.name == ln:
                break
        else:
            continue

        annos = l['annotations']
        for at in annotation_types:

    return out


def _layer_from_name(state, layer_name):
    for l in state['layers']:
        if l['name'] == layer_name:
            return l
    else:
        return None


def _get_type_annotations(state, layer_name, type):
    l = _layer_from_name(state, layer_name)
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
    return _generic_annotations(state, layer_name, description, linked_segmentations, tags, 'point')


def line_annotations(state, layer_name, description=False, linked_segmentations=False, tags=False):
    return _generic_annotations(state, layer_name, description, linked_segmentations, tags, 'line')


def sphere_annotations(state, layer_name, description=False, linked_segmentations=False, tags=False):
    return _generic_annotations(state, layer_name, description, linked_segmentations, tags, 'sphere')
