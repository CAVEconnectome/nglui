from multiprocessing.sharedctypes import Value
from typing import Iterable
from .layers import (
    ImageLayerConfig,
    SegmentationLayerConfig,
    AnnotationLayerConfig,
    PointMapper,
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


def make_point_statebuilder(
    client: CAVEclient,
    point_column="pt_position",
    linked_seg_column="pt_root_id",
    data_resolution=None,
    group_column=None,
    contrast=None,
    view_kws=None,
    point_layer_name="pts",
    color=None,
):
    """make a state builder that puts points on a single column with a linked segmentaton id

    Args:
        client (CAVEclient): CAVEclient configured for the datastack desired
        point_column (str, optional): column in dataframe to pull points from. Defaults to "pt_position".
        linked_seg_column (str, optional): column to link to segmentation, None for no column. Defaults to "pt_root_id".
        group_columns (str, or list, optional): column(s) to group annotations by, None for no grouping (default=None)
        contrast (list, optional):  list-like, optional
            Two elements specifying the black level and white level as
            floats between 0 and 1, by default None. If None, no contrast
            is set.
        view_kws (dict, optional): dict, optional
            dictionary of view keywords to configure neuroglancer view
    Returns:
        StateBuilder: a statebuilder to make points with linked segmentations
    """
    img_layer, seg_layer = from_client(client, contrast=contrast)
    point_mapper = PointMapper(
        point_column=point_column,
        linked_segmentation_column=linked_seg_column,
        group_column=group_column,
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
        point_column (str, optional): column to pull points for synapses from. Defaults to "ctr_pt_position".
        pre_pt_root_id_col (str, optional): column to pull pre synaptic ids for synapses from. Defaults to "pre_pt_root_id".
        post_pt_root_id_col (str, optional): column to pull post synaptic ids for synapses from. Defaults to "post_pt_root_id".
        input_layer_name (str, optional): name of layer for inputs. Defaults to "syns_in".
        output_layer_name (str, optional): name of layer for outputs. Defaults to "syns_out".

    Returns:
        ChainedStateBuilder: An instance of a ChainedStateBuilder configured to accept
        a list  starting with None followed by optionally synapse input dataframe
        followed by optionally synapse output dataframe.
    """

    img_layer, seg_layer = from_client(client, contrast=contrast)
    seg_layer.add_selection_map(selected_ids_column="root_id")

    if view_kws is None:
        view_kws = {}
    sb1 = StateBuilder(
        layers=[img_layer, seg_layer],
        client=client,
    )

    state_builders = [sb1]
    if show_inputs:
        # First state builder
        input_point_mapper = PointMapper(
            point_column=point_column, linked_segmentation_column=pre_pt_root_id_col
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


def make_url_robust(df, sb, client, shorten="if_long", ngl_url=None):
    """Generate a url from a neuroglancer state. If too long, return through state server"""
    if shorten == "if_long":
        url = sb.render_state(df, return_as="url", url_prefix=ngl_url)
        if len(url) > MAX_URL_LENGTH:
            url = make_state_url(df, sb, client, ngl_url=ngl_url)
    elif shorten == "always":
        url = make_state_url(df, sb, client)
    elif shorten == "never":
        url = sb.render_state(df, return_as="url", url_prefix=ngl_url)
    else:
        raise (ValueError('shorten should be one of ["if_long", "always", "never"]'))
    return url


def package_state(df, sb, client, shorten, return_as, ngl_url, link_text):
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
):
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
        sort_inputs (bool, optional): whether to sort inputs by presynaptic root id, ordered by synapse count. Defaults to True.
        sort_outputs (bool, optional): whether to sort inputs by presynaptic root id, ordered by postsynaptic synapse count. Defaults to True.
        sort_ascending (bool, optional): If sorting, whether to sort ascending (lowest synapse count to highest). Defaults to False.
        contrast (list, optional): Two elements specifying the black level and white level as
            floats between 0 and 1, by default None. If None, no contrast
            is set.
        timestamp (datetime.datetime, optional): timestamp to do query. Defaults to None, will use materialized version.
        view_kws (dict, optional): view_kws to configure statebuilder, see nglui.StateBuilder . Defaults to None.
        point_column (str, optional): column to pull points for synapses from. Defaults to "ctr_pt_position".
        pre_pt_root_id_col (str, optional): column to pull pre synaptic ids for synapses from. Defaults to "pre_pt_root_id".
        post_pt_root_id_col (str, optional): column to pull post synaptic ids for synapses from. Defaults to "post_pt_root_id".
        input_layer_name (str, optional): name of layer for inputs. Defaults to "syns_in".
        output_layer_name (str, optional): name of layer for outputs. Defaults to "syns_out".
        ngl_url (str, optional): url to use for neuroglancer. Defaults to None (will use default viewer set in datastack)
        link_text (str, optional): text to use for html return. Defaults to Neuroglancer Link
    Raises:
        ValueError: If the point column is not present in the synapse table

    Returns:
        str: url of neuroglancer link with saved state
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
