
import numpy as np
import pandas as pd

from ..easyviewer.ev_base.base import SEGMENTATION_LAYER_TYPES


def _is_spelunker_state(state):
    """Check if a state is a spelunker state or not."""
    return "dimension" in state.keys()


def layer_names(state, include_archived = True):
    """Get all layer names in the state

    Parameters
    ----------
    state : dict
        Neuroglancer state as a JSON dict
    include_archived : bool
        Whether to include archived layers

    Returns
    -------
    names : list
        List of layer names
    """
    return [l["name"] for l in state["layers"] if \
        (include_archived or l.get('archived', False) == False)]


def image_layers(state, include_archived = True):
    """Get all image layer names in the state

    Parameters
    ----------
    state : dict
        Neuroglancer state as a JSON dict
    include_archived : bool
        Whether to include archived layers

    Returns
    -------
    names : list
        List of layer names
    """
    return [l["name"] for l in state["layers"] if l["type"] == "image" \
        and (include_archived or l.get('archived', False) == False)]


def segmentation_layers(state, include_archived = True):
    """Get all segmentation layer names in the state

    Parameters
    ----------
    state : dict
        Neuroglancer state as a JSON dict
    include_archived : bool
        Whether to include archived layers

    Returns
    -------
    names : list
        List of layer names
    """
    return [l["name"] for l in state["layers"] if l["type"] in SEGMENTATION_LAYER_TYPES \
            and (include_archived or l.get('archived', False) == False)]


def annotation_layers(state, include_archived = True):
    """Get all annotation layer names in the state

    Parameters
    ----------
    state : dict
        Neuroglancer state as a JSON dict
    include_archived : bool
        Whether to include archived layers

    Returns
    -------
    names : list
        List of layer names
    """
    return [l["name"] for l in state["layers"] if l["type"] == "annotation" \
            and (include_archived or l.get('archived', False) == False)]


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
    taginfo = l.get("annotationTags", [])
    tags = {}
    for t in taginfo:
        tags[int(t["id"])] = t["label"]
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
    return state["layers"][layer_ind]


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
    view["position"] = state["navigation"]["pose"]["position"]["voxelCoordinates"]
    view["zoomFactor"] = state["navigation"].get("zoomFactor", None)
    view["perspectiveOrientation"] = state.get("perspectiveOrientation", None)
    view["perspectiveZoom"] = state.get("perspectiveZoom", None)
    view["voxelSize"] = state["navigation"]["pose"]["position"]["voxelSize"]
    return view


def get_selected_ids(state, layer=None):
    """Get a list of selected ids in a segmentation layer

    Parameters
    ----------
    state : dict
        State dict
    layer : str, optional
        Segmentation layer name, if needed. If None and only one segmentation layer
        is present, default to it. By default None

    Returns
    -------
    list
        List of root ids.
    """
    seg_layers = segmentation_layers(state)
    if len(seg_layers) == 1 and layer is None:
        layer = seg_layers[0]
    else:
        raise ValueError(
            "Segmentation layer must be specified since the state has more than one"
        )
    return [int(s) for s in get_layer(state, layer)["segments"]]


def _get_type_annotations(state, layer_name, type):
    l = get_layer(state, layer_name)
    annos = l.get("annotations", [])
    return [anno for anno in annos if anno["type"] == type]


def _get_point_annotations(state, layer_name):
    return _get_type_annotations(state, layer_name, "point")


def _get_sphere_annotations(state, layer_name):
    return _get_type_annotations(state, layer_name, "ellipsoid")


def _get_line_annotations(state, layer_name):
    return _get_type_annotations(state, layer_name, "line")


def _get_bbox_annotations(state, layer_name):
    return _get_type_annotations(state, layer_name, "axis_aligned_bounding_box")


def _get_group_annotations(state, layer_name):
    return _get_type_annotations(state, layer_name, "collection")


def _extract_point_data(annos):
    return [[anno.get("point") for anno in annos]]


def _extract_sphere_data(annos):
    pt = [anno.get("center") for anno in annos]
    rad = [anno.get("radii") for anno in annos]
    return [pt, rad]


def _extract_line_data(annos):
    ptA = [anno.get("pointA") for anno in annos]
    ptB = [anno.get("pointB") for anno in annos]
    return [ptA, ptB]


def _extract_bbox_data(annos):
    ptA = [anno.get("pointA") for anno in annos]
    ptB = [anno.get("pointB") for anno in annos]
    return [ptA, ptB]


def _extract_group_data(annos):
    pt = [anno.get("source") for anno in annos]
    anno_id = [anno.get("id") for anno in annos]
    return [pt, anno_id]


_get_map = {
    "point": _get_point_annotations,
    "line": _get_line_annotations,
    "sphere": _get_sphere_annotations,
    "bbox": _get_bbox_annotations,
    "group": _get_group_annotations,
}

_extraction_map = {
    "point": _extract_point_data,
    "line": _extract_line_data,
    "sphere": _extract_sphere_data,
    "bbox": _extract_bbox_data,
    "group": _extract_group_data,
}


def _flatten_list_of_strings(l):
    out = []
    for x in l:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def _extract_segments(anno):
    seg_data = anno.get("segments", [])
    seg_list = _flatten_list_of_strings(seg_data)
    return [int(x) for x in seg_list]


def _generic_annotations(
    state, layer_name, description, linked_segmentations, tags, group, anno_type
):
    annos = _get_map[anno_type](state, layer_name)
    out = _extraction_map[anno_type](annos)
    if description:
        desc = [anno.get("description", None) for anno in annos]
        out.append(desc)
    if linked_segmentations:
        out.append([_extract_segments(anno) for anno in annos])
    if tags:
        tag_list = [anno.get("tagIds", []) for anno in annos]
        out.append(tag_list)
    if group:
        group_ids = [anno.get("parentId") for anno in annos]
        out.append(group_ids)
    if len(out) == 1:
        return out[0]
    else:
        return out


def point_annotations(
    state,
    layer_name,
    description=False,
    linked_segmentations=False,
    tags=False,
    group=False,
):
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
    anno_group : list
        List of group ids (as string) or None for annotations. Only returned if group=True
    """
    return _generic_annotations(
        state,
        layer_name,
        description=description,
        linked_segmentations=linked_segmentations,
        tags=tags,
        group=group,
        anno_type="point",
    )


def line_annotations(
    state,
    layer_name,
    description=False,
    linked_segmentations=False,
    tags=False,
    group=False,
):
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
    anno_group : list
        List of group ids (as string) or None for annotations. Only returned if group=True
    """
    return _generic_annotations(
        state, layer_name, description, linked_segmentations, tags, group, "line"
    )


def bbox_annotations(
    state,
    layer_name,
    description=False,
    linked_segmentations=False,
    tags=False,
    group=False,
):
    """Get all bounding box annotation points and other info from a layer.

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
        List of N 3-element points (as list) of the first point in each bbox.
    anno_points_B : list
        List of N 3-element points (as list) of the second point in each bbox.
    anno_descriptions : list
        List of N strings (or None), only returned if description=True.
    anno_linked_segmentations : list
        List of N lists of object ids. Only returned if linked_segmentations=True.
    anno_tags : list
        List of N lists of tag ids. Only returned if tags=True.
    anno_group : list
        List of group ids (as string) or None for annotations. Only returned if group=True
    """
    return _generic_annotations(
        state, layer_name, description, linked_segmentations, tags, group, "bbox"
    )


def sphere_annotations(
    state,
    layer_name,
    description=False,
    linked_segmentations=False,
    tags=False,
    group=False,
):
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
    anno_group : list
        List of group ids (as string) or None for annotations. Only returned if group=True
    """
    return _generic_annotations(
        state, layer_name, description, linked_segmentations, tags, group, "sphere"
    )


def group_annotations(
    state, layer_name, description=False, linked_segmentations=False, tags=False
):
    """All group annotations and their associated points

    Parameters
    ----------
    state : dict
        Neuroglancer state as JSON dict
    layer_name : str
        Annotation layer name
    description : bool, optional
        If True, also returns descriptions as well. By default False
    linked_segmentations : bool, optional
        If True, also returns list of linked segmentations, by default False
    tags : bool, optional
        If True, also returns list of tags, by default False

    Returns
    -------
    group_points : list
        List of N 3-element points
    group_id: list
        List of N id strings for groups.
    anno_descriptions : list
        List of N strings (or None), only returned if description=True.
    anno_linked_segmentations : list
        List of N lists of object ids. Only returned if linked_segmentations=True.
    anno_tags : list
        List of N lists of tag ids. Only returned if tags=True.
    """
    return _generic_annotations(
        state,
        layer_name,
        description=description,
        linked_segmentations=linked_segmentations,
        tags=tags,
        group=False,
        anno_type="group",
    )


def extract_multicut(state, seg_layer=None):
    """Extract information entered into the multicut graph operation

    Parameters
    ----------
    state : dict
        Neuroglancer state
    seg_layer : str, optional
        Name of a segmentation layer or None. If None, the function will check how many segmentation
        layers there are and, if only one exits, choose it. If more than one segmentation layer is present,
        it errors. By default None

    Returns
    -------
    pts: np.array
        Nx3 array of points selected
    side: np.array
        N array with 'source' or 'sink', depending on which side the point is on.
    svids: np.array
        N array with selected supervoxel. If only points are selected (e.g. via clicking on the mesh),
        the value will be NaN.
    root_id: int
        Root id of the object to split
    """
    if seg_layer is None:
        seg_layers = segmentation_layers(state)
        if len(seg_layers) == 1:
            seg_layer = seg_layers[0]
        else:
            raise ValueError(
                "State has multiple segmentation layers. Please specify the layer to use."
            )
    l = get_layer(state, "seg")
    pts = []
    svids = []
    side = []
    root_id = None

    source_data = l["graphOperationMarker"][0]
    for anno in source_data["annotations"]:
        pts.append(anno["point"])
        side.append("source")
        if root_id is None:
            root_id = int(anno["segments"][1])
        svid = int(anno["segments"][0])
        if svid != root_id:
            svids.append(svid)
        else:
            # This means that no supervoxel was directly selected
            svids.append(np.nan)

    sink_data = l["graphOperationMarker"][1]
    for anno in sink_data["annotations"]:
        pts.append(anno["point"])
        side.append("sink")
        svid = int(anno["segments"][0])
        if root_id is None:
            root_id = int(anno["segments"][1])
        if svid != root_id:
            svids.append(svid)
        else:
            # This means that no supervoxel was directly selected
            svids.append(np.nan)

    return np.array(pts), np.array(side), np.array(svids), root_id


def _concat_list(d):
    d_out = []
    for x in d:
        for y in x:
            d_out.append(y)
    return d_out


def _parse_layer_dataframe(state, ln, expand_tags):
    lns = []
    points = []
    anno_types = []
    pointBs = []
    linked_segs = []
    tags = []
    group_ids = []
    descs = []

    p_pt, p_desc, p_seg, p_tag, p_grp = point_annotations(
        state,
        ln,
        description=True,
        linked_segmentations=True,
        tags=True,
        group=True,
    )
    n_p_pts = len(p_pt)
    p_type = ["point"] * n_p_pts
    p_ln = [ln] * n_p_pts
    p_ptB = [np.nan] * n_p_pts

    lns.append(p_ln)
    points.append(p_pt)
    anno_types.append(p_type)
    pointBs.append(p_ptB)
    linked_segs.append(p_seg)
    tags.append(p_tag)
    group_ids.append(p_grp)
    descs.append(p_desc)

    # Lines
    l_ptA, l_ptB, l_desc, l_seg, l_tag, l_grp = line_annotations(
        state,
        ln,
        description=True,
        linked_segmentations=True,
        tags=True,
        group=True,
    )
    n_l_pts = len(l_ptA)
    l_type = ["line"] * n_l_pts
    l_ln = [ln] * n_l_pts

    lns.append(l_ln)
    points.append(l_ptA)
    anno_types.append(l_type)
    pointBs.append(l_ptB)
    linked_segs.append(l_seg)
    tags.append(l_tag)
    group_ids.append(l_grp)
    descs.append(l_desc)

    # Spheres
    s_ptA, s_ptB, s_desc, s_seg, s_tag, s_grp = sphere_annotations(
        state,
        ln,
        description=True,
        linked_segmentations=True,
        tags=True,
        group=True,
    )
    n_s_pts = len(s_ptA)
    s_type = ["sphere"] * n_s_pts
    s_ln = [ln] * n_s_pts

    lns.append(s_ln)
    points.append(s_ptA)
    anno_types.append(s_type)
    pointBs.append(s_ptB)
    linked_segs.append(s_seg)
    tags.append(s_tag)
    group_ids.append(s_grp)
    descs.append(s_desc)

    # Bboxes
    b_ptA, b_ptB, b_desc, b_seg, b_tag, b_grp = bbox_annotations(
        state,
        ln,
        description=True,
        linked_segmentations=True,
        tags=True,
        group=True,
    )
    n_b_pts = len(b_ptA)
    b_type = ["bbox"] * n_b_pts
    b_ln = [ln] * n_b_pts

    lns.append(b_ln)
    points.append(b_ptA)
    anno_types.append(b_type)
    pointBs.append(b_ptB)
    linked_segs.append(b_seg)
    tags.append(b_tag)
    group_ids.append(b_grp)
    descs.append(b_desc)

    df = pd.DataFrame(
        {
            "layer": _concat_list(lns),
            "anno_type": _concat_list(anno_types),
            "point": _concat_list(points),
            "pointB": _concat_list(pointBs),
            "linked_segmentation": _concat_list(linked_segs),
            "tags": _concat_list(tags),
            "group_id": _concat_list(group_ids),
            "description": _concat_list(descs),
        }
    )
    if expand_tags:
        tag_dict = tag_dictionary(state, ln)
        for k, v in tag_dict.items():
            df[v] = df["tags"].apply(lambda x: k in x)
    return df


def annotation_dataframe(state, expand_tags=False):
    """Return all annotations across all annotation layers in the state.

    Parameters
    ----------
    state : dict
        Neuroglancer state dictionary
    expand_tags : bool, optional
        If True, expand tags into separate boolean columns named by the tag label. By default False.
        Note that if tag labels are duplicated in multiple layers, the values will appear in the same column.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns layer, anno_type, point, pointB, linked_segmentation, tags, anno_id, group_id, description.
        If expand_tags is True, an additional column will be added for each tag.
    """
    dfs = [
        _parse_layer_dataframe(state, ln, expand_tags)
        for ln in annotation_layers(state)
    ]
    return pd.concat(dfs, ignore_index=True)
