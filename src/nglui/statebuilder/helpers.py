from typing import TYPE_CHECKING, Iterable, Literal, Optional, Union

from caveclient import CAVEclient

from ..segmentprops import SegmentProperties
from .base import DataMap, ViewerState
from .ngl_components import AnnotationLayer
from .shaders import DEFAULT_SHADER_MAP, simple_point_shader

if TYPE_CHECKING:
    import datetime

    import pandas as pd

DEFAULT_POSTSYN_COLOR = "turquoise"
DEFAULT_PRESYN_COLOR = "tomato"

DEFAULT_SEGMENTATION_LAYER = "segmentation"

MAX_URL_LENGTH = 1_750_000

ReturnAs = Literal["link", "url", "clipboard", "browser", "dict", "json", "viewer"]
Shorten = Union[bool, Literal["never", "always", "if_long"]]


def _as_segment_list(root_ids: Union[int, str, Iterable[int]]) -> Union[list, dict]:
    """Coerce a single root id or a collection of root ids into something `add_segments` accepts.

    A dict of `{root_id: visible}` is passed through unchanged, since that form carries
    visibility information that a list cannot.
    """
    if root_ids is None:
        return []
    if isinstance(root_ids, dict):
        return root_ids
    if isinstance(root_ids, str) or not isinstance(root_ids, Iterable):
        return [root_ids]
    return list(root_ids)


def _normalize_shorten(shorten: Shorten) -> Union[bool, Literal["if_long"]]:
    "Map the human-friendly shorten words onto the values `ViewerState.to_url` expects."
    match shorten:
        case "never":
            return False
        case "always":
            return True
        case _:
            return shorten


def add_segment_properties_source(
    viewer: ViewerState,
    segment_properties: Union[SegmentProperties, dict],
    client: CAVEclient,
    name: str = DEFAULT_SEGMENTATION_LAYER,
) -> ViewerState:
    """Upload an already-built segment property and add it as a source on a segmentation layer.

    Use this when you already have a [SegmentProperties][nglui.segmentprops.SegmentProperties]
    object (or the property JSON it produces). To build segment properties from a dataframe
    as part of a viewer pipeline, use
    [ViewerState.add_segment_properties][nglui.statebuilder.ViewerState.add_segment_properties]
    instead.

    Parameters
    ----------
    viewer : ViewerState
        The viewer state holding the segmentation layer to add the property source to.
    segment_properties : Union[SegmentProperties, dict]
        Either an nglui `SegmentProperties` object or the property JSON dictionary it
        serializes to.
    client : CAVEclient
        CAVEclient used to upload the property JSON to the state service.
    name : str, optional
        Name of the segmentation layer to add the source to, by default "segmentation".

    Returns
    -------
    ViewerState
        The viewer state, with the segment property source added.
    """
    if isinstance(segment_properties, SegmentProperties):
        property_json = segment_properties.to_dict()
    elif isinstance(segment_properties, dict):
        property_json = segment_properties
    else:
        raise TypeError(
            "segment_properties must be a SegmentProperties object or a property JSON dictionary, "
            f"not {type(segment_properties)}."
        )
    property_id = client.state.upload_property_json(property_json)
    property_url = client.state.build_neuroglancer_url(
        property_id, format_properties=True
    )
    viewer.layers[name].add_source(property_url)
    return viewer


def _render_viewer_state(
    viewer: ViewerState,
    return_as: ReturnAs,
    shorten: Shorten = "if_long",
    target_url: Optional[str] = None,
    target_site: Optional[str] = None,
    client: Optional[CAVEclient] = None,
    link_text: str = "Neuroglancer Link",
):
    "Render a viewer state into whichever representation the caller asked for."
    shorten = _normalize_shorten(shorten)
    match return_as:
        case "viewer":
            return viewer
        case "dict":
            return viewer.to_dict()
        case "json":
            return viewer.to_json_string()
        case "url":
            return viewer.to_url(
                shorten=shorten,
                target_url=target_url,
                target_site=target_site,
                client=client,
            )
        case "link":
            return viewer.to_link(
                shorten=shorten,
                target_url=target_url,
                target_site=target_site,
                client=client,
                link_text=link_text,
            )
        case "clipboard":
            return viewer.to_clipboard(
                shorten=shorten,
                target_url=target_url,
                target_site=target_site,
                client=client,
            )
        case "browser":
            return viewer.to_browser(
                shorten=shorten,
                target_url=target_url,
                target_site=target_site,
                client=client,
            )
        case _:
            raise ValueError(
                f"Invalid return_as value: {return_as}. Must be one of 'link', 'url', "
                "'clipboard', 'browser', 'dict', 'json', or 'viewer'."
            )


def sort_dataframe_by_root_id(
    df, root_id_column, ascending=False, num_column="n_times", drop=False
) -> "pd.DataFrame":
    """Sort a dataframe so that rows belonging to the same root id are together, ordered by how many times the root id appears.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to sort
    root_id_column : str
        Column name to use for sorting root ids
    ascending : bool, optional
        Whether to sort ascending (lowest count to highest) or not, by default False
    num_column : str, optional
        Temporary column name to use for count information, by default 'n_times'
    drop : bool, optional
        If True, drop the additional column when returning.

    Returns
    -------
    pd.DataFrame
    """
    cols = df.columns[df.columns != root_id_column]
    if len(cols) == 0:
        df = df.copy()
        df[num_column] = df[root_id_column]
        use_column = num_column
    else:
        use_column = cols[0]

    df[num_column] = df.groupby(root_id_column).transform("count")[use_column]
    if drop:
        return df.sort_values(
            by=[num_column, root_id_column], ascending=ascending
        ).drop(columns=[num_column])
    else:
        return df.sort_values(by=[num_column, root_id_column], ascending=ascending)


def make_point_state(
    client: CAVEclient,
    data: Optional["pd.DataFrame"] = None,
    point_column: str = "pt_position",
    segment_column: str = "pt_root_id",
    description_column: Optional[str] = None,
    tag_column: Optional[str] = None,
    data_resolution: Optional[list] = None,
    tags: Optional[list] = None,
    layer_name: str = "Points",
    shader: Optional[Union[bool, str]] = True,
    selected_alpha: Optional[float] = None,
    alpha_3d: Optional[float] = None,
    mesh_silhouette: Optional[float] = None,
) -> ViewerState:
    """Generate a state builder that puts points on a single column with a linked segmentaton id

    Parameters
    ----------
    client : CAVEclient
        CAVEclient configured for the datastack desired
    data : Optional[pd.DataFrame], optional
        Dataframe to use for points, by default None.
        If None, a simple DataMap will be used with no key.
        Fill in the resulting state with viewer.map(df).
    point_column : str, optional
        Column in the dataframe to use for point positions, by default "pt_position"
    segment_column : str, optional
        Column in the dataframe to use for segment ids, by default "pt_root_id"
    description_column : Optional[str], optional
        Column in the dataframe to use for point descriptions, by default None
    tag_column : Optional[str], optional
        Column in the dataframe to use for tags, by default None
    data_resolution : Optional[list], optional
        Resolution of the data, by default None
    tags : Optional[list], optional
        List of tags to apply to the points, by default None
    layer_name : str, optional
        Name of the layer to create, by default "Points"
    shader : Optional[Union[bool, str]], optional
        Shader to use for the points, by default True (uses default shader)
    selected_alpha : Optional[float], optional
        Alpha value for selected segments in the 2D view, by default None (uses default value)
    alpha_3d : Optional[float], optional
        Alpha value for meshes, by default None (uses default value)
    mesh_silhouette : Optional[float], optional
        Mesh silhouette value, by default None (uses default value)

    Returns
    -------
    ViewerState:
        A viewerstate to make points with linked segmentations
    """
    if shader is True:
        shader = DEFAULT_SHADER_MAP.get("points")
    if data is None:
        data = DataMap()
    return (
        ViewerState(infer_coordinates=True)
        .add_layers_from_client(
            client,
            selected_alpha=selected_alpha,
            alpha_3d=alpha_3d,
            mesh_silhouette=mesh_silhouette,
        )
        .add_points(
            data=data,
            name=layer_name,
            point_column=point_column,
            segment_column=segment_column,
            description_column=description_column,
            tag_column=tag_column,
            data_resolution=data_resolution,
            tags=tags,
            shader=shader,
        )
    )


def make_line_state(
    client: CAVEclient,
    data: Optional["pd.DataFrame"] = None,
    point_a_column: str = "pre_pt_position",
    point_b_column: str = "post_pt_position",
    segment_column: str = "pt_root_id",
    description_column: Optional[str] = None,
    tag_column: Optional[str] = None,
    data_resolution: Optional[list] = None,
    tags: Optional[list] = None,
    layer_name: str = "lines",
    shader: Optional[Union[bool, str]] = True,
    selected_alpha: Optional[float] = None,
    alpha_3d: Optional[float] = None,
    mesh_silhouette: Optional[float] = None,
) -> ViewerState:
    """Generate a state builder that puts line segments from two columns with a linked segmentaton id

    Parameters
    ----------
    client : CAVEclient
        CAVEclient configured for the datastack desired
    data : Optional[pd.DataFrame], optional
        Dataframe to use for points, by default None.
        If None, a simple DataMap will be used with no key.
        Fill in the resulting state with viewer.map(df).
    point_a_column : str, optional
        Column in the dataframe to use for line start positions, by default "pre_pt_position"
    point_b_column : str, optional
        Column in the dataframe to use for line end, by default "post_pt_position"
    segment_column : str, optional
        Column in the dataframe to use for segment ids, by default "pt_root_id"
    description_column : Optional[str], optional
        Column in the dataframe to use for point descriptions, by default None
    tag_column : Optional[str], optional
        Column in the dataframe to use for tags, by default None
    data_resolution : Optional[list], optional
        Resolution of the data, by default None
    tags : Optional[list], optional
        List of tags to apply to the points, by default None
    layer_name : str, optional
        Name of the layer to create, by default "Points"
    shader : Optional[Union[bool, str]], optional
        Shader to use for the points, by default True (uses default shader)
    selected_alpha : Optional[float], optional
        Alpha value for selected segments in the 2D view, by default None (uses default value)
    alpha_3d : Optional[float], optional
        Alpha value for meshes, by default None (uses default value)
    mesh_silhouette : Optional[float], optional
        Mesh silhouette value, by default None (uses default value)

    Returns
    -------
    ViewerState:
        A viewerstate to make points with linked segmentations
    """
    if shader is True:
        shader = DEFAULT_SHADER_MAP.get("lines")
    if data is None:
        data = DataMap()
    return (
        ViewerState(infer_coordinates=True)
        .add_layers_from_client(
            client,
            selected_alpha=selected_alpha,
            alpha_3d=alpha_3d,
            mesh_silhouette=mesh_silhouette,
        )
        .add_lines(
            data=data,
            name=layer_name,
            point_a_column=point_a_column,
            point_b_column=point_b_column,
            segment_column=segment_column,
            description_column=description_column,
            tag_column=tag_column,
            data_resolution=data_resolution,
            tags=tags,
            shader=shader,
        )
    )


def make_connectivity_state_map(
    client: CAVEclient,
    show_inputs: bool = True,
    show_outputs: bool = True,
    point_column: str = "ctr_pt_position",
    input_root_id_col: str = "pre_pt_root_id",
    output_root_id_col: str = "post_pt_root_id",
    data_resolution_input: Optional[list] = None,
    data_resolution_output: Optional[list] = None,
    input_layer_name: str = "syns_in",
    output_layer_name: str = "syns_out",
    input_layer_color: Union[tuple, str] = DEFAULT_POSTSYN_COLOR,
    output_layer_color: Union[tuple, str] = DEFAULT_PRESYN_COLOR,
    input_shader: Optional[Union[bool, str]] = True,
    output_shader: Optional[Union[bool, str]] = True,
    selected_alpha: Optional[float] = None,
    alpha_3d: Optional[float] = None,
    mesh_silhouette: Optional[float] = None,
    **kwargs,
) -> ViewerState:
    """Create a Neuroglancer state with input and output synapses.
    Parameters
    ----------
    client : CAVEclient
        CAVEclient configured for the datastack desired
    show_inputs : bool, optional
        Whether to show input synapses, by default True
    show_outputs : bool, optional
        Whether to show output synapses, by default True
    point_column : str, optional
        Column in the dataframe to use for point positions, by default "ctr_pt_position"
    input_root_id_col : str, optional
        Column in the dataframe to use for input synapse root ids, by default "pre_pt_root_id"
    output_root_id_col : str, optional
        Column in the dataframe to use for output synapse root ids, by default "post_pt_root_id"
    dataframe_resolution_input : Optional[list], optional
        Resolution of the input dataframe, by default None
    dataframe_resolution_output : Optional[list], optional
        Resolution of the output dataframe, by default None
    input_layer_name : str, optional
        Name of the input layer, by default "syns_in"
    output_layer_name : str, optional
        Name of the output layer, by default "syns_out"
    input_layer_color : Union[tuple, str], optional
        Color of the input layer, by default "turquoise"
    output_layer_color : Union[tuple, str], optional
        Color of the output layer, by default "tomato"
    input_shader : Optional[Union[bool, str]], optional
        Shader to use for input synapses, by default None.
    output_shader : Optional[Union[bool, str]], optional
        Shader to use for output synapses, by default None (uses default shader)
    selected_alpha : Optional[float], optional
        Alpha value for selected segments in the 2d view, by default None (uses default value)
    alpha_3d : Optional[float], optional
        Alpha value for meshes, by default None (uses default value)
    mesh_silhouette : Optional[float], optional
        Mesh silhouette value, by default None (uses default value)

    Returns
    -------
    ViewerState
        A Neuroglancer ViewerState with input and output synapses configured.
    """
    if "dataframe_resolution_input" in kwargs:
        data_resolution_input = kwargs["dataframe_resolution_input"]
        raise DeprecationWarning(
            "The argument 'dataframe_resolution_input' is deprecated and will be removed in a future version. Please use 'data_resolution_input' instead."
        )
    if "dataframe_resolution_output" in kwargs:
        data_resolution_output = kwargs["dataframe_resolution_output"]
        raise DeprecationWarning(
            "The argument 'dataframe_resolution_output' is deprecated and will be removed in a future version. Please use 'data_resolution_output' instead."
        )
    ngl = ViewerState(infer_coordinates=True).add_layers_from_client(
        client,
        selected_alpha=selected_alpha,
        alpha_3d=alpha_3d,
        mesh_silhouette=mesh_silhouette,
    )
    if show_inputs:
        if input_shader is True:
            input_shader = DEFAULT_SHADER_MAP.get("points")
        ngl = ngl.add_layer(
            AnnotationLayer(
                name=input_layer_name,
                linked_segmentation="segmentation",
                shader=input_shader,
                color=input_layer_color,
            ).add_points(
                DataMap("inputs"),
                point_column=point_column,
                segment_column=input_root_id_col,
                data_resolution=data_resolution_input,
            )
        )
    if show_outputs:
        if output_shader is True:
            output_shader = DEFAULT_SHADER_MAP.get("points")
        ngl = ngl.add_layer(
            AnnotationLayer(
                name=output_layer_name,
                linked_segmentation="segmentation",
                shader=output_shader,
                color=output_layer_color,
            ).add_points(
                DataMap("outputs"),
                point_column=point_column,
                segment_column=output_root_id_col,
                data_resolution=data_resolution_output,
            )
        )
    return ngl


def make_segment_state(
    client: CAVEclient,
    root_ids: Optional[Union[int, Iterable[int], dict]] = None,
    segment_properties: Optional[Union[SegmentProperties, dict]] = None,
    segmentation_layer_name: str = DEFAULT_SEGMENTATION_LAYER,
    imagery: Union[bool, str] = True,
    selected_alpha: Optional[float] = None,
    alpha_3d: Optional[float] = None,
    mesh_silhouette: Optional[float] = None,
    infer_coordinates: bool = True,
) -> ViewerState:
    """Build a viewer state showing one or more segments from a datastack.

    This is the quickest way to go from a CAVEclient and a set of root ids to a
    Neuroglancer state: imagery and segmentation layers come from the client, the root ids
    are selected in the segmentation layer, and segment properties (if given) are uploaded
    and attached as a source on that same layer.

    Parameters
    ----------
    client : CAVEclient
        CAVEclient configured for the datastack desired.
    root_ids : Optional[Union[int, Iterable[int], dict]], optional
        Root id or root ids to select in the segmentation layer, by default None.
        If None, no segments are selected. A dict of `{root_id: visible}` can be used
        to control per-segment visibility.
    segment_properties : Optional[Union[SegmentProperties, dict]], optional
        Segment properties to attach to the segmentation layer, by default None.
        Either a [SegmentProperties][nglui.segmentprops.SegmentProperties] object or the
        property JSON dictionary it serializes to. The property is uploaded to the state
        service via the client and added as a source on the segmentation layer.
    segmentation_layer_name : str, optional
        Name of the segmentation layer, by default "segmentation".
    imagery : Union[bool, str], optional
        Whether to add an imagery layer, by default True.
        If a string is provided, it is used as the name of the imagery layer.
    selected_alpha : Optional[float], optional
        Alpha value for selected segments in the 2D view, by default None (uses default value).
    alpha_3d : Optional[float], optional
        Alpha value for meshes, by default None (uses default value).
    mesh_silhouette : Optional[float], optional
        Mesh silhouette value, by default None (uses default value).
    infer_coordinates : bool, optional
        Whether to infer the viewer position from the layer sources, by default True.

    Returns
    -------
    ViewerState
        A viewer state with the requested segments selected.

    Examples
    --------
    >>> from nglui import statebuilder
    >>> vs = statebuilder.helpers.make_segment_state(client, root_ids=[864691135474648896])
    >>> vs.to_url()
    """
    viewer = ViewerState(
        client=client,
        infer_coordinates=infer_coordinates,
    ).add_layers_from_client(
        client,
        imagery=imagery,
        segmentation=segmentation_layer_name,
        selected_alpha=selected_alpha,
        alpha_3d=alpha_3d,
        mesh_silhouette=mesh_silhouette,
    )

    segments = _as_segment_list(root_ids)
    if len(segments) > 0:
        viewer.add_segments(segments, name=segmentation_layer_name)

    if segment_properties is not None:
        add_segment_properties_source(
            viewer,
            segment_properties,
            client=client,
            name=segmentation_layer_name,
        )
    return viewer


def make_segment_link(
    client: CAVEclient,
    root_ids: Optional[Union[int, Iterable[int], dict]] = None,
    segment_properties: Optional[Union[SegmentProperties, dict]] = None,
    return_as: ReturnAs = "link",
    shorten: Shorten = "if_long",
    target_url: Optional[str] = None,
    target_site: Optional[str] = None,
    segmentation_layer_name: str = DEFAULT_SEGMENTATION_LAYER,
    imagery: Union[bool, str] = True,
    selected_alpha: Optional[float] = None,
    alpha_3d: Optional[float] = None,
    mesh_silhouette: Optional[float] = None,
    infer_coordinates: bool = True,
    link_text: str = "Neuroglancer Link",
):
    """Make a Neuroglancer link showing one or more segments from a datastack.

    A one-line convenience wrapping [make_segment_state][nglui.statebuilder.helpers.make_segment_state]
    that renders the resulting state as a link, URL, or other representation.

    Parameters
    ----------
    client : CAVEclient
        CAVEclient configured for the datastack desired.
    root_ids : Optional[Union[int, Iterable[int], dict]], optional
        Root id or root ids to select in the segmentation layer, by default None.
        If None, no segments are selected. A dict of `{root_id: visible}` can be used
        to control per-segment visibility.
    segment_properties : Optional[Union[SegmentProperties, dict]], optional
        Segment properties to attach to the segmentation layer, by default None.
        Either a [SegmentProperties][nglui.segmentprops.SegmentProperties] object or the
        property JSON dictionary it serializes to.
    return_as : ReturnAs, optional
        How to return the state, by default "link".
        Options are "link" (an HTML link for notebooks), "url" (a URL string),
        "clipboard" (copies the URL to the system clipboard and returns it),
        "browser" (opens the URL in a web browser), "dict", "json", or "viewer"
        (the [ViewerState][nglui.statebuilder.ViewerState] itself).
    shorten : Shorten, optional
        Whether to shorten the URL with the CAVE state service, by default "if_long",
        which only shortens URLs past a length threshold. Also accepts True/"always"
        and False/"never".
    target_url : Optional[str], optional
        Base Neuroglancer URL to use, by default None (uses the configured default).
    target_site : Optional[str], optional
        Target site to use, based on the keys in site_utils.NEUROGLANCER_SITES,
        by default None (uses the configured default).
    segmentation_layer_name : str, optional
        Name of the segmentation layer, by default "segmentation".
    imagery : Union[bool, str], optional
        Whether to add an imagery layer, by default True.
        If a string is provided, it is used as the name of the imagery layer.
    selected_alpha : Optional[float], optional
        Alpha value for selected segments in the 2D view, by default None (uses default value).
    alpha_3d : Optional[float], optional
        Alpha value for meshes, by default None (uses default value).
    mesh_silhouette : Optional[float], optional
        Mesh silhouette value, by default None (uses default value).
    infer_coordinates : bool, optional
        Whether to infer the viewer position from the layer sources, by default True.
    link_text : str, optional
        Text to display for the HTML link when `return_as` is "link", by default "Neuroglancer Link".

    Returns
    -------
    The Neuroglancer state in the format specified by `return_as`.

    Examples
    --------
    >>> from nglui import statebuilder
    >>> statebuilder.helpers.make_segment_link(client, root_ids=[864691135474648896])
    """
    viewer = make_segment_state(
        client,
        root_ids=root_ids,
        segment_properties=segment_properties,
        segmentation_layer_name=segmentation_layer_name,
        imagery=imagery,
        selected_alpha=selected_alpha,
        alpha_3d=alpha_3d,
        mesh_silhouette=mesh_silhouette,
        infer_coordinates=infer_coordinates,
    )
    return _render_viewer_state(
        viewer,
        return_as=return_as,
        shorten=shorten,
        target_url=target_url,
        target_site=target_site,
        client=client,
        link_text=link_text,
    )


def make_neuron_neuroglancer_link(
    client: CAVEclient,
    root_ids: Union[int, list[int]],
    return_as: ReturnAs = "link",
    shorten: Shorten = "if_long",
    show_inputs: bool = True,
    show_outputs: bool = True,
    point_column: str = "ctr_pt_position",
    target_url: Optional[str] = None,
    target_site: Optional[str] = None,
    timestamp: Optional["datetime.datetime"] = None,
    infer_coordinates: bool = True,
    segment_properties: Optional[Union[SegmentProperties, dict]] = None,
    segmentation_layer_name: str = DEFAULT_SEGMENTATION_LAYER,
    link_text: str = "Neuroglancer Link",
):
    """Create a Neuroglancer state with a neuron and optionally its inputs and outputs.
    Parameters
    ----------
    client : CAVEclient
        CAVEclient configured for the datastack desired
    root_ids : Union[int, list[int]]
        Root IDs of the neuron to visualize. Can be a single ID or a list of IDs.
    return_as : ReturnAs, optional
        Format to return the Neuroglancer state, by default "link".
        Options are "link", "url", "clipboard", "browser", "dict", "json", or "viewer".
    shorten : Shorten, optional
        Whether to shorten the URL if it is long, by default "if_long".
        Options are "never" (or False), "always" (or True), or "if_long".
    show_inputs : bool, optional
        Whether to show input synapses, by default True
    show_outputs : bool, optional
        Whether to show output synapses, by default True
    point_column : str, optional
        Column in the dataframe to use for point positions, by default "ctr_pt_position"
    target_url : Optional[str], optional
        Target URL to use for the Neuroglancer link, by default None.
        If None, the default CAVEclient URL will be used.
    target_site : Optional[str], optional
        Target site to use for the Neuroglancer link, by default None.
        If None, the default CAVEclient site will be used.
    timestamp : Optional[datetime.datetime], optional
        Timestamp to use for the query, by default None.
        If None, the current time will be used.
    infer_coordinates : bool, optional
        Whether to infer coordinates from the data, by default True.
    segment_properties : Optional[Union[SegmentProperties, dict]], optional
        Segment properties to attach to the segmentation layer, by default None.
        Either a [SegmentProperties][nglui.segmentprops.SegmentProperties] object or the
        property JSON dictionary it serializes to.
    segmentation_layer_name : str, optional
        Name of the segmentation layer, by default "segmentation".
    link_text : str, optional
        Text to display for the HTML link when `return_as` is "link", by default "Neuroglancer Link".

    Returns
    -------
    Neuroglancer state in the specified format.
    """

    root_ids = _as_segment_list(root_ids)
    query_ids = list(root_ids)

    viewer = make_segment_state(
        client,
        root_ids=root_ids,
        segment_properties=segment_properties,
        segmentation_layer_name=segmentation_layer_name,
        infer_coordinates=infer_coordinates,
    )
    if show_inputs:
        post_df = client.materialize.synapse_query(
            post_ids=query_ids,
            desired_resolution=[1, 1, 1],
            split_positions=True,
            timestamp=timestamp,
        )
        viewer.add_points(
            data=post_df,
            name="inputs",
            point_column=point_column,
            segment_column="post_pt_root_id",
            data_resolution=[1, 1, 1],
            shader=simple_point_shader(color=DEFAULT_POSTSYN_COLOR),
        )
    if show_outputs:
        pre_df = client.materialize.synapse_query(
            pre_ids=query_ids,
            desired_resolution=[1, 1, 1],
            split_positions=True,
            timestamp=timestamp,
        )
        viewer.add_points(
            data=pre_df,
            name="outputs",
            point_column=point_column,
            segment_column="pre_pt_root_id",
            data_resolution=[1, 1, 1],
            shader=simple_point_shader(color=DEFAULT_PRESYN_COLOR),
        )
    return _render_viewer_state(
        viewer,
        return_as=return_as,
        shorten=shorten,
        target_url=target_url,
        target_site=target_site,
        client=client,
        link_text=link_text,
    )
