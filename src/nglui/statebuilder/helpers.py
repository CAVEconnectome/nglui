from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Optional, Union
from warnings import warn

import pandas as pd
from caveclient import CAVEclient
from caveclient.endpoints import fallback_ngl_endpoint
from IPython.display import HTML

from ..site_utils import neuroglancer_url
from .layers import (
    AnnotationLayerConfig,
    ImageLayerConfig,
    SegmentationLayerConfig,
)
from .mappers import LineMapper, PointMapper
from .statebuilder import ChainedStateBuilder, StateBuilder

if TYPE_CHECKING:
    from nglui.segmentprops import SegmentProperties

DEFAULT_POSTSYN_COLOR = (0.25098039, 0.87843137, 0.81568627)  # CSS3 color turquise
DEFAULT_PRESYN_COLOR = (1.0, 0.38823529, 0.27843137)  # CSS3 color tomato

CONTRAST_CONFIG = {
    "minnie65_phase3_v1": {
        "contrast_controls": True,
        "black": 0.35,
        "white": 0.70,
    },
    "minnie65_public": {
        "contrast_controls": True,
        "black": 0.35,
        "white": 0.70,
    },
}
MAX_URL_LENGTH = 1_750_000
DEFAULT_NGL = fallback_ngl_endpoint


def sort_dataframe_by_root_id(
    df, root_id_column, ascending=False, num_column="n_times", drop=False
):
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


def make_line_statebuilder(
    client: CAVEclient,
    point_column_a="pre_pt_position",
    point_column_b="post_pt_position",
    linked_seg_column="pt_root_id",
    description_column=None,
    tag_column=None,
    data_resolution=None,
    group_column=None,
    contrast=None,
    view_kws=None,
    point_layer_name="lines",
    color=None,
    split_positions=False,
):
    """Generate a state builder that puts points on a single column with a linked segmentaton id

    Parameters
    ----------
    client : CAVEclient
        CAVEclient configured for the datastack desired
    point_column_a : str, optional
        column in dataframe to pull points from. Defaults to "pre_pt_position".
    point_column_b : str, optional
        column in dataframe to pull points from. Defaults to "post_pt_position".
    linked_seg_column : str, optional
        column to link to segmentation, None for no column. Defaults to "pt_root_id".
    group_column : str, or list, optional
        column(s) to group annotations by, None for no grouping (default=None)
    tag_column : str, optional
        column to use for tags, None for no tags (default=None)
    description_column : str, optional
        column to use for descriptions, None for no descriptions (default=None)
    contrast : list, optional
        Two elements specifying the black level and white level as
        floats between 0 and 1, by default None. If None, no contrast
        is set.
    view_kws : dict, optional
        dictionary of view keywords to configure neuroglancer view
    split_positions : bool, optional
        whether the position column into x,y,z columns. Defaults to False.
    Returns
    -------
    StateBuilder:
        A statebuilder to make points with linked segmentations
    """
    img_layer, seg_layer = from_client(client, contrast=contrast)
    line_mapper = LineMapper(
        point_column_a=point_column_a,
        point_column_b=point_column_b,
        linked_segmentation_column=linked_seg_column,
        tag_column=tag_column,
        description_column=description_column,
        group_column=group_column,
        split_positions=split_positions,
    )
    ann_layer = AnnotationLayerConfig(
        point_layer_name,
        mapping_rules=[line_mapper],
        linked_segmentation_layer=seg_layer.name,
        data_resolution=data_resolution,
        color=color,
    )
    if view_kws is None:
        view_kws = {}
    return StateBuilder(
        [img_layer, seg_layer, ann_layer],
        client=client,
        view_kws=view_kws,
    )


def make_point_statebuilder(
    client: CAVEclient,
    point_column="pt_position",
    linked_seg_column="pt_root_id",
    data_resolution=None,
    group_column=None,
    tag_column=None,
    description_column=None,
    contrast=None,
    view_kws=None,
    point_layer_name="pts",
    color=None,
    split_positions=False,
):
    """Generate a state builder that puts points on a single column with a linked segmentaton id

    Parameters
    -----------
    client : CAVEclient
        CAVEclient configured for the datastack desired
    point_column : str, optional
        Column in dataframe to pull points from. Defaults to "pt_position".
    linked_seg_column : str, optional
        column to link to segmentation, None for no column. Defaults to "pt_root_id".
    group_column : str, or list, optional
        column(s) to group annotations by, None for no grouping (default=None)
    tag_column : str, optional)
        column to use for tags, None for no tags (default=None)
    description_column : str, optional
        column to use for descriptions, None for no descriptions (default=None)
    contrast : list, optional
        Two elements specifying the black level and white level as
        floats between 0 and 1, by default None. If None, no contrast
        is set.
    view_kws : dict, optional
        dictionary of view keywords to configure neuroglancer view
    split_positions : bool, optional
        whether the position column into x,y,z columns. Defaults to False.

    Returns
    -------
    StateBuilder:
        A statebuilder to make points with linked segmentations
    """
    img_layer, seg_layer = from_client(client, contrast=contrast)
    point_mapper = PointMapper(
        point_column=point_column,
        linked_segmentation_column=linked_seg_column,
        tag_column=tag_column,
        description_column=description_column,
        group_column=group_column,
        split_positions=split_positions,
    )
    ann_layer = AnnotationLayerConfig(
        point_layer_name,
        mapping_rules=[point_mapper],
        linked_segmentation_layer=seg_layer.name,
        data_resolution=data_resolution,
        color=color,
    )
    if view_kws is None:
        view_kws = {}
    return StateBuilder(
        [img_layer, seg_layer, ann_layer],
        client=client,
        view_kws=view_kws,
    )


def make_pre_post_statebuilder(
    client: CAVEclient,
    show_inputs: bool = False,
    show_outputs: bool = False,
    contrast: list = None,
    view_kws: dict = None,
    point_column="ctr_pt_position",
    pre_pt_root_id_col="pre_pt_root_id",
    post_pt_root_id_col="post_pt_root_id",
    dataframe_resolution_pre=None,
    dataframe_resolution_post=None,
    input_layer_name="syns_in",
    output_layer_name="syns_out",
    input_layer_color=DEFAULT_POSTSYN_COLOR,
    output_layer_color=DEFAULT_PRESYN_COLOR,
    split_positions=False,
):
    """Function to generate ChainedStateBuilder with optional pre and post synaptic
    annotation layers

    Parameters
    ----------
    client : CAVEclient
        a CAVEclient configured for datastack to visualize
    show_inputs : bool, optional
        whether to show input synapses. Defaults to False.
    show_outputs : bool, optional
        whether to show output synapses.. Defaults to False.
    contrast : list, optional
        Two elements specifying the black level and white level as
        floats between 0 and 1, by default None. If None, no contrast
        is set.
    view_kws : dict, optional
        view_kws to configure statebuilder, see nglui.StateBuilder. Defaults to None.
        keys are:
            show_slices: Boolean
                sets if slices are shown in the 3d view. Defaults to False.
            layout: str
                `xy-3d`/`xz-3d`/`yz-3d` (sections plus 3d pane), `xy`/`yz`/`xz`/`3d` (only one pane), or `4panel` (all panes). Default is `xy-3d`.
            show_axis_lines: Boolean
                determines if the axis lines are shown in the middle of each view.
            show_scale_bar: Boolean
                toggles showing the scale bar.
            orthographic : Boolean
                toggles orthographic view in the 3d pane.
            position* : 3-element vector
                determines the centered location.
            zoom_image : float
                Zoom level for the imagery in units of nm per voxel. Defaults to 8.
            zoom_3d : float
                Zoom level for the 3d pane. Defaults to 2000. Smaller numbers are more zoomed in.
    point_column : str, optional
        column to pull points for synapses from. Defaults to "ctr_pt_position".
    pre_pt_root_id_col : str, optional
        column to pull pre synaptic ids for synapses from. Defaults to "pre_pt_root_id".
    post_pt_root_id_col : str, optional
        column to pull post synaptic ids for synapses from. Defaults to "post_pt_root_id".
    input_layer_name : str, optional
        name of layer for inputs. Defaults to "syns_in".
    output_layer_name : str, optional
        name of layer for outputs. Defaults to "syns_out".
    split_positions : bool, optional
        whether the position column is split into x,y,z columns. Defaults to False.

    Returns
    -------
    ChainedStateBuilder:
        An instance of a ChainedStateBuilder configured to accept
        a list  starting with None followed by optionally synapse input dataframe
        followed by optionally synapse output dataframe.
    """

    img_layer, seg_layer = from_client(client, contrast=contrast)
    seg_layer.add_selection_map(selected_ids_column="root_id")

    if view_kws is None:
        view_kws = {}

    sb1 = StateBuilder(layers=[img_layer, seg_layer], client=client, view_kws=view_kws)
    state_builders = [sb1]
    if show_inputs:
        # First state builder
        input_point_mapper = PointMapper(
            point_column=point_column,
            linked_segmentation_column=pre_pt_root_id_col,
            split_positions=split_positions,
        )
        inputs_lay = AnnotationLayerConfig(
            input_layer_name,
            mapping_rules=[input_point_mapper],
            linked_segmentation_layer=seg_layer.name,
            data_resolution=dataframe_resolution_post,
            color=input_layer_color,
        )
        sb_in = StateBuilder([inputs_lay], client=client)
        state_builders.append(sb_in)
    if show_outputs:
        output_point_mapper = PointMapper(
            point_column=point_column,
            linked_segmentation_column=post_pt_root_id_col,
            split_positions=split_positions,
        )
        outputs_lay = AnnotationLayerConfig(
            output_layer_name,
            mapping_rules=[output_point_mapper],
            linked_segmentation_layer=seg_layer.name,
            data_resolution=dataframe_resolution_pre,
            color=output_layer_color,
        )
        sb_out = StateBuilder([outputs_lay], client=client)
        state_builders.append(sb_out)
    return ChainedStateBuilder(state_builders)


def make_state_url(df, sb, client, ngl_url=None, target_site=None):
    """Generate a url from a neuroglancer state via a state server.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to pass through statebuilder
    sb : statebuilder.StateBuilder
        Statebuilder to use to render data for link
    client : CAVEclient
        CAVEclient configured with a state server
    ngl_url : str, optional
        Neuroglancer deployment URL, by default None
    target_site : str, optional
        Type of neuroglancer deployment to build link for, by default None.
        This value overrides automatic checking based on the provided url.
        Use `seunglab` for a Seung-lab branch and either `mainline` or `cave-explorer` for the Cave Explorer or main Google branch.

    Returns
    -------
    str
        Url to the uploaded neuroglancer state.
    """
    state = sb.render_state(df, return_as="dict", target_site=target_site)
    state_id = client.state.upload_state_json(state)
    if ngl_url is None:
        ngl_url = client.info.viewer_site()
        if ngl_url is None:
            ngl_url = DEFAULT_NGL
    url = client.state.build_neuroglancer_url(
        state_id, ngl_url=ngl_url, target_site=target_site
    )
    return url


def make_url_robust(
    df: pd.DataFrame,
    sb: StateBuilder,
    client: CAVEclient,
    shorten: str = "if_long",
    ngl_url: str = None,
    max_url_length=MAX_URL_LENGTH,
    target_site=None,
):
    """Generate a url from a neuroglancer state. If too long, return through state server,
    othewise return a url containing the data directly.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to pass through statebuilder
    sb : statebuilder.StateBuilder
        Statebuilder to use to render data for link
    client : CAVEclient
        CAVEclient configured with a state server
    shorten : str, optional
        How to shorten link. one of 'if_long', 'always', 'never'.
        'if_long' will use the state server to shorten links longer than nglui.statebuilder.MAX_URL_LENGTH
        (set to 1,750,000).
        'always' will always use the state server to shorten the url
        'never' will always return the full url.  Defaults to "if_long".
    ngl_url : str, optional
        Neuroglancer deployment URL, by default None
    max_url_length : int, optional
        Maximum length of url to return directly, by default 1_750_000
    target_site : str, optional
        Type of neuroglancer deployment to build link for, by default None.
        This value overrides automatic checking based on the provided url.
        Use `seunglab` for a Seung-lab branch and either `mainline` or `cave-explorer` for the Cave Explorer or main Google branch.

    Returns
    -------
    str
        URL containing the state created by the statebuilder.
    """

    if shorten == "if_long":
        url = sb.render_state(
            df, return_as="url", url_prefix=ngl_url, target_site=target_site
        )
        if len(url) > max_url_length:
            url = make_state_url(
                df, sb, client, ngl_url=ngl_url, target_site=target_site
            )
    elif shorten == "always":
        url = make_state_url(df, sb, client, ngl_url=ngl_url, target_site=target_site)
    elif shorten == "never":
        url = sb.render_state(
            df, return_as="url", url_prefix=ngl_url, target_site=target_site
        )
    else:
        raise (ValueError('shorten should be one of ["if_long", "always", "never"]'))
    return url


def package_state(
    df: pd.DataFrame,
    sb: StateBuilder,
    client: CAVEclient,
    shorten: str = "if_long",
    return_as: str = "url",
    ngl_url: str = None,
    link_text: str = "Neuroglancer Link",
    target_site: str = None,
):
    """Automate creating a state from a statebuilder and
    a dataframe, return it in the desired format, shortening if desired.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to pass through statebuilder
    sb : statebuilder.StateBuilder
        Statebuilder to use to render data for link
    client : CAVEclient
        CAVEclient configured with a state server
    shorten : str, optional
        How to shorten link. one of 'if_long', 'always', 'never'.
        'if_long' will use the state server to shorten links longer than nglui.statebuilder.MAX_URL_LENGTH
        (set to 1,750,000).
        'always' will always use the state server to shorten the url
        'never' will always return the full url.  Defaults to "if_long".
    return_as : str, optional
        How to return the state. one of 'html', 'url', 'json'.
        'html' will return an ipython clickable link
        'url' will return a string with the url
        'json' will return the state as a dictionary
    ngl_url : str, optional
        Neuroglancer deployment URL, by default None
    link_text : str, optional
        Text to use for the link, by default "Neuroglancer Link"
    target_site : str, optional
        Type of neuroglancer deployment to build link for, by default None.
        This value overrides automatic checking based on the provided url.
        Use `seunglab` for a Seung-lab branch and either `mainline` or `cave-explorer` for the Cave Explorer or main Google branch.

    Returns
    -------
    HTML, str or dict
        state in format specified by return_as
    """
    if ngl_url is None:
        ngl_url = client.info.viewer_site()
        if ngl_url is None:
            ngl_url = DEFAULT_NGL

    if (return_as == "html") or (return_as == "url"):
        url = make_url_robust(
            df, sb, client, shorten=shorten, ngl_url=ngl_url, target_site=target_site
        )
        if return_as == "html":
            return HTML(f'<a href="{url}">{link_text}</a>')
        else:
            return url
    elif return_as == "json":
        return sb.render_state(
            df, return_as=return_as, ngl_url=ngl_url, target_site=target_site
        )
    else:
        raise (
            ValueError(
                f'return_as={return_as} not a valid choice, choose one of "html", "url", or "json")'
            )
        )


def make_synapse_neuroglancer_link(
    synapse_df,
    client,
    return_as="html",
    shorten="always",
    contrast=None,
    point_column="ctr_pt_position",
    dataframe_resolution=None,
    group_connections=True,
    link_pre_and_post=True,
    ngl_url=None,
    view_kws=None,
    pre_post_columns=None,
    neuroglancer_link_text="Neuroglancer Link",
    color=None,
    split_positions=False,
    target_site=None,
):
    """Generate a neuroglancer link from a synapse dataframe as returned from CAVEclient.materialize.synapse_query.

    Parameters
    ----------
    synapse_df : pandas.DataFrame
        DataFrame where each row represents a synapse.
    client : caveclient.CAVEclient
        A CAVEclient instance.
    return_as : str, optional
        How to return the URL. Valid options are:
        - 'html': Returns an IPython HTML element.
        - 'url': Returns a URL string.
        - 'json': Returns a dictionary representing the Neuroglancer state.
        Defaults to 'html'.
    shorten : str, optional
        Whether to shorten the link. Valid options are:
        - 'always': Always shorten the link.
        - 'if_long': Shorten the link if it exceeds MAX_URL_LENGTH.
        - 'never': Never shorten the link.
        Defaults to 'always'.
    contrast : list, optional
        List of two floats between 0 and 1, specifying the black and white levels.
        If None, no contrast is set.
    point_column : str, optional
        Column in the DataFrame containing synapse positions. Defaults to 'ctr_pt_position'.
    dataframe_resolution : list, optional
        List of length 3, specifying the resolution units of the position column.
        If None, attempts to get the resolution from DataFrame metadata or client.info.viewer_resolution().
    group_connections : bool, optional
        Whether to group synapses within the same connection (between the same neurons).
        Defaults to True.
    link_pre_and_post : bool, optional
        Whether to link the synapse annotations to the pre- and post-synaptic partners.
        Defaults to True.
    ngl_url : str, optional
        URL of the Neuroglancer instance to use. Defaults to client.info.viewer_site() or DEFAULT_NGL.
    view_kws : dict, optional
        Dictionary containing viewer settings for Neuroglancer. Available keys:
        - show_slices: Boolean, sets if slices are shown in the 3D view.
        - layout: str, specifies the viewer layout (e.g., 'xy-3d', '4panel').
        - show_axis_lines: Boolean, determines if axis lines are shown.
        - show_scale_bar: Boolean, toggles the scale bar.
        - orthographic: Boolean, toggles orthographic view in the 3D pane.
        - position: 3-element vector, sets the centered location.
        - zoom_image: float, zoom level for the imagery in nm per voxel.
        - zoom_3d: float, zoom level for the 3D pane.
    pre_post_columns : list, optional
        List of two strings, specifying the column names for pre- and post-synaptic root_ids.
        Defaults to ['pre_pt_root_id', 'post_pt_root_id'].
    neuroglancer_link_text : str, optional
        Text to use in the returned HTML link. Defaults to 'Neuroglancer Link'.
    color : list(float) or str, optional
        Color of synapse points as an RGB list [0, 1], hex string, or common name.
    split_positions : bool, optional
        Whether the position column is split into x, y, and z columns.
    target_site : str, optional
        Type of neuroglancer deployment to build link for, by default None.
        This value overrides automatic checking based on the provided url.
        Use `seunglab` for a Seung-lab branch and either `mainline` or `cave-explorer` for the Cave Explorer or main Google branch.

    Returns
    -------
    IPython.HTML, str, or dict
        Representation of the Neuroglancer state, depending on the `return_as` parameter.

    Raises
    ------
    ValueError
        If the specified `point_column` is not found in the DataFrame.
    """
    if point_column not in synapse_df.columns:
        raise ValueError(f"point_column={point_column} not in dataframe")
    if pre_post_columns is None:
        pre_post_columns = ["pre_pt_root_id", "post_pt_root_id"]
    if dataframe_resolution is None:
        dataframe_resolution = synapse_df.attrs.get("dataframe_resolution", None)

    if group_connections:
        group_column = pre_post_columns
    else:
        group_column = None
    if link_pre_and_post:
        linked_columns = pre_post_columns
    else:
        linked_columns = None

    sb = make_point_statebuilder(
        client,
        point_column=point_column,
        linked_seg_column=linked_columns,
        group_column=group_column,
        data_resolution=dataframe_resolution,
        contrast=contrast,
        view_kws=view_kws,
        point_layer_name="synapses",
        color=color,
        split_positions=split_positions,
    )
    return package_state(
        synapse_df,
        sb,
        client,
        shorten,
        return_as,
        ngl_url,
        neuroglancer_link_text,
        target_site=target_site,
    )


def make_neuron_neuroglancer_link(
    client,
    root_ids,
    return_as="html",
    shorten="always",
    show_inputs=False,
    show_outputs=False,
    sort_inputs=True,
    sort_outputs=True,
    sort_ascending=False,
    input_color=DEFAULT_POSTSYN_COLOR,
    output_color=DEFAULT_PRESYN_COLOR,
    contrast=None,
    timestamp=None,
    view_kws=None,
    point_column="ctr_pt_position",
    pre_pt_root_id_col="pre_pt_root_id",
    post_pt_root_id_col="post_pt_root_id",
    input_layer_name="syns_in",
    output_layer_name="syns_out",
    ngl_url=None,
    link_text="Neuroglancer Link",
    target_site=None,
):
    """function to create a neuroglancer link view of a neuron, optionally including inputs and outputs

    Parameters
    ----------
    client : CAVEclient
        A CAVEclient configured for the datastack to visualize.
    root_ids : Iterable[int]
        The root_ids to build the visualization around.
    return_as : str, optional
        How to return the URL or state. Valid options are:
        - 'html': Returns an IPython HTML element.
        - 'json': Returns a dictionary representing the Neuroglancer state.
        - 'url': Returns a URL string.
        Defaults to 'html'.
    shorten : str, optional
        Whether to shorten the link. Valid options are:
        - 'always': Always shorten the link.
        - 'if_long': Shorten the link if it exceeds MAX_URL_LENGTH.
        - 'never': Never shorten the link.
        Defaults to 'if_long'.
    show_inputs : bool, optional
        Whether to include input synapses. Defaults to False.
    show_outputs : bool, optional
        Whether to include output synapses. Defaults to False.
    sort_inputs : bool, optional
        Whether to sort input synapses by presynaptic root id, ordered by synapse count.
        Defaults to True.
    sort_outputs : bool, optional
        Whether to sort output synapses by presynaptic root id, ordered by postsynaptic synapse count.
        Defaults to True.
    sort_ascending : bool, optional
        If sorting, whether to sort ascending (lowest synapse count to highest).
        Defaults to False.
    input_color : list(float) or str, optional
        Color of input synapse points as an RGB list [0, 1], hex string, or common name.
    output_color : list(float) or str, optional
        Color of output synapse points as an RGB list [0, 1], hex string, or common name.
    contrast : list, optional
        List of two floats between 0 and 1, specifying the black and white levels.
        If None, no contrast is set.
    timestamp : datetime.datetime, optional
        Timestamp to use for the query. Defaults to None, which will use the materialized version.
    view_kws : dict, optional
        Dictionary containing viewer settings for Neuroglancer. See nglui.StateBuilder for options.
        See the previous docstring for details on available keys.
    point_column : str, optional
        Column to pull synapse positions from. Defaults to "ctr_pt_position".
    pre_pt_root_id_col : str, optional
        Column to pull presynaptic IDs for synapses from. Defaults to "pre_pt_root_id".
    post_pt_root_id_col : str, optional
        Column to pull postsynaptic IDs for synapses from. Defaults to "post_pt_root_id".
    input_layer_name : str, optional
        Name of the layer for input synapses. Defaults to "syns_in".
    output_layer_name : str, optional
        Name of the layer for output synapses. Defaults to "syns_out".
    ngl_url : str, optional
        URL of the Neuroglancer instance to use. Defaults to the default viewer set in the datastack.
    link_text : str, optional
        Text to use for the HTML return. Defaults to 'Neuroglancer Link'.

    Raises
    ------
    ValueError
        If the specified point column is not present in the synapse table.

    Returns
    -------
    IPython.HTML, str, or dict
        Representation of the Neuroglancer state, depending on the `return_as` parameter.
    """
    if not isinstance(root_ids, Iterable):
        root_ids = [root_ids]
    df1 = pd.DataFrame({"root_id": root_ids})
    dataframes = [df1]
    data_resolution_pre = None
    data_resolution_post = None
    if show_inputs:
        syn_in_df = client.materialize.synapse_query(
            post_ids=root_ids,
            timestamp=timestamp,
            desired_resolution=client.info.viewer_resolution(),
            split_positions=True,
        )
        data_resolution_pre = syn_in_df.attrs["dataframe_resolution"]
        if sort_inputs:
            syn_in_df = sort_dataframe_by_root_id(
                syn_in_df, pre_pt_root_id_col, ascending=sort_ascending, drop=True
            )
        dataframes.append(syn_in_df)
    if show_outputs:
        syn_out_df = client.materialize.synapse_query(
            pre_ids=root_ids,
            timestamp=timestamp,
            desired_resolution=client.info.viewer_resolution(),
            split_positions=True,
        )
        data_resolution_post = syn_out_df.attrs["dataframe_resolution"]
        if sort_outputs:
            syn_out_df = sort_dataframe_by_root_id(
                syn_out_df, post_pt_root_id_col, ascending=sort_ascending, drop=True
            )
        dataframes.append(syn_out_df)
    sb = make_pre_post_statebuilder(
        client,
        show_inputs=show_inputs,
        show_outputs=show_outputs,
        contrast=contrast,
        point_column=point_column,
        view_kws=view_kws,
        pre_pt_root_id_col=pre_pt_root_id_col,
        post_pt_root_id_col=post_pt_root_id_col,
        input_layer_name=input_layer_name,
        output_layer_name=output_layer_name,
        input_layer_color=input_color,
        output_layer_color=output_color,
        dataframe_resolution_pre=data_resolution_pre,
        dataframe_resolution_post=data_resolution_post,
        split_positions=True,
    )
    return package_state(
        dataframes,
        sb,
        client,
        shorten,
        return_as,
        ngl_url,
        link_text,
        target_site=target_site,
    )


def from_client(
    client,
    image_name=None,
    segmentation_name=None,
    contrast=None,
    use_skeleton_service=False,
):
    """Generate basic image and segmentation layers from a FrameworkClient

    Parameters
    ----------
    client : caveclient.CAVEclient
        A CAVEclient with a specified datastack
    image_name : str, optional
        Name for the image layer, by default None.
    segmentation_name : str, optional
        Name for the segmentation layer, by default None
    contrast : list-like or False, optional
        Two elements specifying the black level and white level as
        floats between 0 and 1, by default None. If None, no contrast
        is set.
    use_skeleton_service : bool, optional
        If True, uses a skeleton service, if advertised, with the segmentation. Defaults to False.

    Returns
    -------
    image_layer : ImageLayerConfig
        Image layer with default values from the client
    seg_layer : ImageLayerConfig
        Segmentation layer with default values from the client
    """
    if contrast is None:
        config = CONTRAST_CONFIG.get(
            client.datastack_name, {"contrast_controls": True, "black": 0, "white": 1}
        )
    elif contrast is False:
        config = {}
    else:
        config = {"contrast_controls": True, "black": contrast[0], "white": contrast[1]}

    if image_name is not False:
        img_layer = ImageLayerConfig(
            client.info.image_source(), name=image_name, **config
        )
    else:
        img_layer = None
    if segmentation_name is not False:
        if use_skeleton_service:
            skeleton_source = client.info.get_datastack_info().get(
                "skeleton_source", None
            )
            if skeleton_source is None:
                warn(
                    "Skeleton source requested but no skeleton source found in datastack info."
                )
        else:
            skeleton_source = None
        seg_layer = SegmentationLayerConfig(
            client.info.segmentation_source(),
            name=segmentation_name,
            skeleton_source=skeleton_source,
        )
    else:
        seg_layer = None
    return img_layer, seg_layer


def segment_property_link(
    seg_props: Union[str, SegmentProperties],
    client: CAVEclient,
    ngl_url: Optional[str] = None,
    return_as: Literal["html", "url", "viewer", "json", "dict", "short"] = "html",
):
    """Returns a basic link to a default neuroglancer state and segment properties.

    Parameters
    ----------
    seg_props : segment_properties.SegmentProperties or str
        A segment propeties object or a URL to a segment properties.
    client : caveclient.CAVEclient
        A caveclient object.
    ngl_url : str, optional
        URL for a neuroglancer deployment, by default None which uses a default Spelunker deployment.
    return_as : str, optional
        Select how the data comes back from statebuilder, by default "html".
        See statebuilder.render_state for more information.

    Returns
    -------
    string or neuroglancer.Viewer
    """
    img, seg = from_client(client)
    if isinstance(seg_props, str):
        seg_prop_url = seg_props
    else:
        prop_id = client.state.upload_property_json(seg_props.to_dict())
        seg_prop_url = client.state.build_neuroglancer_url(
            prop_id,
            target_site="cave-explorer",
            format_properties=True,
        )
    ngl_url = neuroglancer_url(ngl_url, target_site="spelunker")
    seg.add_segment_propeties(seg_prop_url)
    sb = StateBuilder(
        [img, seg], client=client, url_prefix=ngl_url, target_site="spelunker"
    )
    return sb.render_state(return_as=return_as)
