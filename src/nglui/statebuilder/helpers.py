from typing import TYPE_CHECKING, Literal, Optional, Union

from caveclient import CAVEclient

from ..site_utils import neuroglancer_url
from .base import *
from .ngl_components import *
from .shaders import DEFAULT_SHADER_MAP

if TYPE_CHECKING:
    from nglui.segmentprops import SegmentProperties


DEFAULT_POSTSYN_COLOR = "turquoise"
DEFAULT_PRESYN_COLOR = "tomato"

MAX_URL_LENGTH = 1_750_000


def sort_dataframe_by_root_id(
    df, root_id_column, ascending=False, num_column="n_times", drop=False
) -> pd.DataFrame:
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
    data: Optional[pd.DataFrame] = None,
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
    data: Optional[pd.DataFrame] = None,
    point_a_column: str = "pre_pt_position",
    point_b_column: str = "post_pt_position",
    segment_column: str = "pt_root_id",
    description_column: Optional[str] = None,
    tag_column: Optional[str] = None,
    data_resolution: Optional[list] = None,
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
    dataframe_resolution_input: Optional[list] = None,
    dataframe_resolution_output: Optional[list] = None,
    input_layer_name: str = "syns_in",
    output_layer_name: str = "syns_out",
    input_layer_color: Union[tuple, str] = DEFAULT_POSTSYN_COLOR,
    output_layer_color: Union[tuple, str] = DEFAULT_PRESYN_COLOR,
    input_shader: Optional[Union[bool, str]] = True,
    output_shader: Optional[Union[bool, str]] = True,
    selected_alpha: Optional[float] = None,
    alpha_3d: Optional[float] = None,
    mesh_silhouette: Optional[float] = None,
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
            LocalAnnotationLayer(
                name=input_layer_name,
                linked_segmentation="segmentation",
                shader=input_shader,
                color=input_layer_color,
            ).add_points(
                DataMap("inputs"),
                point_column=point_column,
                segment_column=input_root_id_col,
                dataframe_resolution=dataframe_resolution_input,
            )
        )
    if show_outputs:
        if output_shader is True:
            output_shader = DEFAULT_SHADER_MAP.get("points")
        ngl = ngl.add_layer(
            LocalAnnotationLayer(
                name=output_layer_name,
                linked_segmentation="segmentation",
                shader=output_shader,
                color=output_layer_color,
            ).add_points(
                DataMap("outputs"),
                point_column=point_column,
                segment_column=output_root_id_col,
                dataframe_resolution=dataframe_resolution_output,
            )
        )
    return ngl
