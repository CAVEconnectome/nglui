from multiprocessing.sharedctypes import Value
from typing import Iterable
from .layers import (
    ImageLayerConfig,
    SegmentationLayerConfig,
    AnnotationLayerConfig,
    PointMapper,
    LineMapper,
)
from .statebuilder import ChainedStateBuilder, StateBuilder
from caveclient import CAVEclient
import pandas as pd
from IPython.display import HTML

DEFAULT_POSTSYN_COLOR = (0.25098039, 0.87843137, 0.81568627)  # CSS3 color turquise
DEFAULT_PRESYN_COLOR = (1.0, 0.38823529, 0.27843137)  # CSS3 color tomato

CONTRAST_CONFIG = {
    "minnie65_phase3_v1": {
        "contrast_controls": True,
        "black": 0.35,
        "white": 0.70,
    }
}
MAX_URL_LENGTH = 1_750_000
DEFAULT_NGL = "https://neuromancer-seung-import.appspot.com/"


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
    """make a state builder that puts points on a single column with a linked segmentaton id

    Args:
        client (CAVEclient): CAVEclient configured for the datastack desired
        point_column (str, optional): column in dataframe to pull points from. Defaults to "pt_position".
        linked_seg_column (str, optional): column to link to segmentation, None for no column. Defaults to "pt_root_id".
        group_columns (str, or list, optional): column(s) to group annotations by, None for no grouping (default=None)
        tag_column (str, optional): column to use for tags, None for no tags (default=None)
        description_column (str, optional): column to use for descriptions, None for no descriptions (default=None)
        contrast (list, optional):  list-like, optional
            Two elements specifying the black level and white level as
            floats between 0 and 1, by default None. If None, no contrast
            is set.
        view_kws (dict, optional): dict, optional
            dictionary of view keywords to configure neuroglancer view
        split_positions (bool, optional): whether the position column into x,y,z columns. Defaults to False.
    Returns:
        StateBuilder: a statebuilder to make points with linked segmentations
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
    """make a state builder that puts points on a single column with a linked segmentaton id

    Args:
        client (CAVEclient): CAVEclient configured for the datastack desired
        point_column (str, optional): column in dataframe to pull points from. Defaults to "pt_position".
        linked_seg_column (str, optional): column to link to segmentation, None for no column. Defaults to "pt_root_id".
        group_columns (str, or list, optional): column(s) to group annotations by, None for no grouping (default=None)
        tag_column (str, optional): column to use for tags, None for no tags (default=None)
        description_column (str, optional): column to use for descriptions, None for no descriptions (default=None)
        contrast (list, optional):  list-like, optional
            Two elements specifying the black level and white level as
            floats between 0 and 1, by default None. If None, no contrast
            is set.
        view_kws (dict, optional): dict, optional
            dictionary of view keywords to configure neuroglancer view
        split_positions (bool, optional): whether the position column into x,y,z columns. Defaults to False.
    Returns:
        StateBuilder: a statebuilder to make points with linked segmentations
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

    Args:
        client (CAVEclient): a CAVEclient configured for datastack to visualize
        root_id (int): a rootID to build around
        show_inputs (bool, optional): whether to show input synapses. Defaults to False.
        show_outputs (bool, optional): whether to show output synapses.. Defaults to False.
        contrast (list, optional):  list-like, optional
            Two elements specifying the black level and white level as
            floats between 0 and 1, by default None. If None, no contrast
            is set.
        view_kws (dict, optional): view_kws to configure statebuilder, see nglui.StateBuilder . Defaults to None.
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
        point_column (str, optional): column to pull points for synapses from. Defaults to "ctr_pt_position".
        pre_pt_root_id_col (str, optional): column to pull pre synaptic ids for synapses from. Defaults to "pre_pt_root_id".
        post_pt_root_id_col (str, optional): column to pull post synaptic ids for synapses from. Defaults to "post_pt_root_id".
        input_layer_name (str, optional): name of layer for inputs. Defaults to "syns_in".
        output_layer_name (str, optional): name of layer for outputs. Defaults to "syns_out".
        split_positions (bool, optional): whether the position column is split into x,y,z columns. Defaults to False.
    Returns:
        ChainedStateBuilder: An instance of a ChainedStateBuilder configured to accept
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


def make_state_url(df, sb, client, ngl_url=None):
    state = sb.render_state(df, return_as="dict")
    state_id = client.state.upload_state_json(state)
    if ngl_url is None:
        ngl_url = client.info.viewer_site()
        if ngl_url is None:
            ngl_url = DEFAULT_NGL
    url = client.state.build_neuroglancer_url(state_id, ngl_url=ngl_url)
    return url


def make_url_robust(
    df: pd.DataFrame,
    sb: StateBuilder,
    client: CAVEclient,
    shorten: str = "if_long",
    ngl_url: str = None,
    max_url_length = MAX_URL_LENGTH,
):
    """Generate a url from a neuroglancer state. If too long, return through state server

    Args:
        df (pandas.DataFrame): dataframe to pass through statebuilder
        sb (nglui.statebuilder.StateBuilder): statebuilder to generate link with
        client (caveclient.CAVEclient): client to interact with state server with and get defaults from
        shorten (str, optional): How to shorten link. one of 'if_long', 'always', 'never'.
            'if_long' will use the state server to shorten links longer than nglui.statebuilder.MAX_URL_LENGTH
            (set to 1,750,000).
            'always' will always use the state server to shorten the url
            'never' will always return the full url.  Defaults to "if_long".
        ngl_url (str, optional): neuroglancer deployment to make url with.
            Defaults to None, which will use the default in the passed sb StateBuilder

    Returns:
        str: a url containing the state created by the statebuilder.
    """

    if shorten == "if_long":
        url = sb.render_state(df, return_as="url", url_prefix=ngl_url)
        if len(url) > max_url_length:
            url = make_state_url(df, sb, client, ngl_url=ngl_url)
    elif shorten == "always":
        url = make_state_url(df, sb, client)
    elif shorten == "never":
        url = sb.render_state(df, return_as="url", url_prefix=ngl_url)
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
):
    """a function to automate creating a state from a statebuilder and
    a dataframe, return it in the desired format, shortening if desired.

    Args:
        df (pd.DataFrame): dataframe to pass to the sb:StateBuilder
        sb (StateBuilder): StateBuilder to generate links with
        client (CAVEclient): caveclient to get default ngl_url and iteract with state viewer
        shorten (str): one of ["if_long", "always", "never"]
            'if_long' will use the state server to shorten links longer than nglui.statebuilder.MAX_URL_LENGTH
            (set to 1,750,000).
            'always' will always use the state server to shorten the url
            'never' will always return the full url.  Defaults to "if_long".
        return_as (str): one of ['html', 'url', 'json']
            'html' will return an ipython clickable link
            'url' will return a string with the url
            'json' will return the state as a dictionary
        ngl_url (str): neuroglancer url to use, if None will use client.info.viewer_site()
        link_text (str): if returning as html, what text to show for the link.

    Returns:
       HTML, str or dict : state in format specified by return_as
    """
    if ngl_url is None:
        ngl_url = client.info.viewer_site()
        if ngl_url is None:
            ngl_url = DEFAULT_NGL

    if (return_as == "html") or (return_as == "url"):
        url = make_url_robust(df, sb, client, shorten=shorten, ngl_url=ngl_url)
        if return_as == "html":
            return HTML(f'<a href="{url}">{link_text}</a>')
        else:
            return url
    elif return_as == "json":
        return sb.render_state(df, return_as=return_as)
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
):
    """_summary_

    Args:
        synapse_df (pandas.DataFrame): dataframe where each row is a synapse
        client (caveclient.CAVEclient): a caveclient
        return_as (str, optional): how to return url.
            'html' as a ipython html element
            'url' as a url string
            'json' as dictionary state.
            Defaults to "html".
        shorten (str, optional): whether to shorten link, 'always' will always shorten,
             'if_long' will shorten it if it's beyond a size MAX_URL_LENGTH.
             'never' will never shorten. Defaults to "always".
        contrast (list, optional): Two elements specifying the black level and white level as
            floats between 0 and 1, by default None. If None, no contrast is set.
        point_column (str, optional): column in dataframe to take synapse position from. Defaults to "ctr_pt_position".
        dataframe_resolution (list, optional): list of length 3 specifying resolution units of position column.
            If None, will attempt to get resolution from dataframe metadata. If no metadata exists,
            will assume it's in client.info.viewer_resolution(). Defaults to None.
        group_connections (bool, optional): whether to group synapses in the same connection (between the same neurons).
            Defaults to True.
        link_pre_and_post (bool, optional): whether to make link the synapse annotations
            to the pre and post synaptic partners. Defaults to True.
        ngl_url (_type_, optional): what neuroglancer to use. Defaults to None.
            If None will default to client.info.viewer_site().
            If that is None will default to DEFAULT_NGL.
        view_kws (dict, optional): viewer dictionary to control neuroglancer viewer. Defaults to None.
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
        pre_post_columns (list, optional): [pre,post] column names for pre and post synaptic root_ids. Defaults to None.
            If None will assume ['pre_pt_root_id', 'post_pt_root_id']
        neuroglancer_link_text (str, optional): Text to use in returning html link. Defaults to "Neuroglancer Link".
        color (list(float) or str, optional): color of synapse points as rgb list [0,1],
            or hex string, or common name (see webcolors documentation)
        split_positions (bool, optional): whether the position column are splits into x,y,z columns.

    Raises:
        ValueError: If the point_column is not in the dataframe

    Returns:
        Ipython.HTML, str, or json: a representation of the neuroglancer state.Type depends on return_as
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
        synapse_df, sb, client, shorten, return_as, ngl_url, neuroglancer_link_text
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
):
    """function to create a neuroglancer link view of a neuron, optionally including inputs and outputs

    Args:
        client (_type_): a CAVEclient configured for datastack to visualize
        root_ids (Iterable[int]): root_ids to build around
        return_as (str, optional): one of 'html', 'json', 'url'. (default 'html')
        shorten (str, optional): if 'always' make a state link always
                             'if_long' make a state link if the json is too long (default)
                             'never' don't shorten link
        show_inputs (bool, optional): whether to include input synapses. Defaults to False.
        show_outputs (bool, optional): whether to include output synapses. Defaults to False.
        sort_inputs (bool, optional): whether to sort inputs by presynaptic root id, ordered by synapse count.
            Defaults to True.
        sort_outputs (bool, optional): whether to sort inputs by presynaptic root id, ordered by postsynaptic synapse count.
            Defaults to True.
        sort_ascending (bool, optional): If sorting, whether to sort ascending (lowest synapse count to highest).
            Defaults to False.
        input_color (list(float) or str, optional): color of input points as rgb list [0,1],
            or hex string, or common name (see webcolors documentation)
        output_color (list(float) or str, optional): color of output points as rgb list [0,1],
            or hex string, or common name (see webcolors documentation)
        contrast (list, optional): Two elements specifying the black level and white level as
            floats between 0 and 1, by default None. If None, no contrast is set.
        timestamp (datetime.datetime, optional): timestamp to do query. Defaults to None, will use materialized version.
        view_kws (dict, optional): view_kws to configure statebuilder, see nglui.StateBuilder.
            Defaults to None.
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
        point_column (str, optional): column to pull points for synapses from. Defaults to "ctr_pt_position".
        pre_pt_root_id_col (str, optional): column to pull pre synaptic ids for synapses from.
            Defaults to "pre_pt_root_id".
        post_pt_root_id_col (str, optional): column to pull post synaptic ids for synapses from.
            Defaults to "post_pt_root_id".
        input_layer_name (str, optional): name of layer for inputs. Defaults to "syns_in".
        output_layer_name (str, optional): name of layer for outputs. Defaults to "syns_out".
        ngl_url (str, optional): url to use for neuroglancer.
            Defaults to None (will use default viewer set in datastack)
        link_text (str, optional): text to use for html return.
            Defaults to 'Neuroglancer Link'
    Raises:
        ValueError: If the point column is not present in the synapse table

    Returns:
        Ipython.HTML, str, or json: a representation of the neuroglancer state.Type depends on return_as
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
        if point_column not in syn_in_df.columns:
            raise ValueError("column pt_column={pt_column} not in synapse table")
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
    return package_state(dataframes, sb, client, shorten, return_as, ngl_url, link_text)


def from_client(client, image_name=None, segmentation_name=None, contrast=None):
    """Generate basic image and segmentation layers from a FrameworkClient

    Parameters
    ----------
    client : annotationframeworkclient.FrameworkClient
        A FrameworkClient with a specified datastack
    image_name : str, optional
        Name for the image layer, by default None.
    segmentation_name : str, optional
        Name for the segmentation layer, by default None
    contrast : list-like, optional
        Two elements specifying the black level and white level as
        floats between 0 and 1, by default None. If None, no contrast
        is set.

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
    else:
        config = {"contrast_controls": True, "black": contrast[0], "white": contrast[1]}
    img_layer = ImageLayerConfig(client.info.image_source(), name=image_name, **config)
    seg_layer = SegmentationLayerConfig(
        client.info.segmentation_source(), name=segmentation_name
    )
    return img_layer, seg_layer
