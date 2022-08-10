from .layers import (
    ImageLayerConfig,
    SegmentationLayerConfig,
    AnnotationLayerConfig,
    PointMapper,
)
from .statebuilder import ChainedStateBuilder, StateBuilder
from caveclient import CAVEclient

CONTRAST_CONFIG = {
    "minnie65_phase3_v1": {
        "contrast_controls": True,
        "black": 0.35,
        "white": 0.70,
    }
}


def make_point_statebuilder(
    client: CAVEclient,
    point_column="pt_position",
    linked_seg_column="pt_root_id",
    contrast=None,
):
    """make a state builder that puts points on a single column with a linked segmentaton id

    Args:
        client (CAVEclient): CAVEclient configured for the datastack desired
        point_column (str, optional): column in dataframe to pull points from. Defaults to "pt_position".
        linked_seg_column (str, optional): column to link to segmentation, None for no column. Defaults to "pt_root_id".
        contrast (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    img_layer, seg_layer = from_client(client, contrast=contrast)
    point_mapper = PointMapper(
        point_column=point_column, linked_segmentation_column=linked_seg_column
    )
    ann_layer = AnnotationLayerConfig("pts", mapping_rules=[point_mapper])
    return StateBuilder(
        [img_layer, seg_layer, ann_layer],
        state_server=client.state.client.state._server_address,
        resolution=client.info.viewer_resolution(),
        client=client,
    )


def make_pre_post_statebuilder(
    client: CAVEclient,
    root_id: int,
    show_inputs: bool = False,
    show_outputs: bool = False,
    contrast: dict = None,
    view_kws: dict = None,
    point_column="ctr_pt_position",
    pre_pt_root_id_col="pre_pt_root_id",
    post_pt_root_id_col="post_pt_root_id",
    input_layer_name="syns_in",
    output_layer_name="syns_out",
):
    """Function to generate ChainedStateBuilder with optional pre and post synaptic
    annotation layers

    Args:
        client (CAVEclient): a CAVEclient configured for datastack to visualize
        root_id (_type_): a rootID to build around
        show_inputs (bool, optional): whether to show input synapses. Defaults to False.
        show_outputs (bool, optional): whether to show output synapses.. Defaults to False.
        contrast (list, optional):  list-like, optional
            Two elements specifying the black level and white level as
            floats between 0 and 1, by default None. If None, no contrast
            is set.
        view_kws (_type_, optional): _description_. Defaults to None.
        point_column (str, optional): _description_. Defaults to "ctr_pt_position".
        pre_pt_root_id_col (str, optional): _description_. Defaults to "pre_pt_root_id".
        post_pt_root_id_col (str, optional): _description_. Defaults to "post_pt_root_id".
        input_layer_name (str, optional): _description_. Defaults to "syns_in".
        output_layer_name (str, optional): _description_. Defaults to "syns_out".

    Returns:
        ChainedStateBuilder: An instance of a ChainedStateBuilder configured to accept
        a list  starting with None followed by optionally synapse input dataframe
        followed by optionally synapse output dataframe.
    """

    img_layer, seg_layer = from_client(client, contrast=contrast)
    seg_layer.add_selection_map(fixed_ids=root_id)
    seg_layer.color
    sb1 = StateBuilder(
        layers=[img_layer, seg_layer],
        resolution=client.info.viewer_resolution(),
        view_kws=view_kws,
    )

    state_builders = [sb1]
    if show_inputs:
        # First state builder

        input_point_mapper = PointMapper(
            point_column=point_column, linked_segmentation_column=pre_pt_root_id_col
        )
        inputs_lay = AnnotationLayerConfig(
            input_layer_name, mapping_rules=[input_point_mapper]
        )
        sb_in = StateBuilder([inputs_lay])
        state_builders.append(sb_in)
    if show_outputs:
        output_point_mapper = PointMapper(
            point_column=point_column, linked_segmentation_column=post_pt_root_id_col
        )
        outputs_lay = AnnotationLayerConfig(
            output_layer_name, mapping_rules=[output_point_mapper]
        )
        sb_out = StateBuilder([outputs_lay])
        state_builders.append(sb_out)
    return ChainedStateBuilder(state_builders)


def make_neuron_neuroglancer_link(
    client,
    root_id,
    show_inputs=False,
    show_outputs=False,
    contrast=None,
    timestamp=None,
    view_kws=None,
    point_column="ctr_pt_position",
    pre_pt_root_id_col="pre_pt_root_id",
    post_pt_root_id_col="post_pt_root_id",
    input_layer_name="syns_in",
    output_layer_name="syns_out",
    ngl_url=None,
):
    """function to create a neuroglancer link view of a neuron, optionally including inputs and outputs

    Args:
        client (_type_): a CAVEclient configured for datastack to visualize
        root_id (_type_): root_id to build around
        show_inputs (bool, optional): whether to include input synapses. Defaults to False.
        show_outputs (bool, optional): whether to include output synapses. Defaults to False.
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

    Raises:
        ValueError: If the point column is not present in the synapse table

    Returns:
        str: url of neuroglancer link with saved state
    """
    dataframes = [None]
    if show_inputs:
        syn_in_df = client.materialize.synapse_query(
            post_ids=root_id,
            timestamp=timestamp,
            desired_resolution=client.info.viewer_resolution,
        )
        if point_column not in syn_in_df.columns:
            raise ValueError("column pt_column={pt_column} not in synapse table")
        dataframes.append(syn_in_df)
    if show_outputs:
        syn_out_df = client.materialize.synapse_query(
            pre_ids=root_id,
            timestamp=timestamp,
            desired_resolution=client.info.viewer_resolution,
        )
        dataframes.append(syn_out_df)
    sb = make_pre_post_statebuilder(
        client,
        root_id,
        show_inputs=show_inputs,
        show_outputs=show_outputs,
        contrast=contrast,
        point_column=point_column,
        view_kws=view_kws,
        pre_pt_root_id_col=pre_pt_root_id_col,
        post_pt_root_id_col=post_pt_root_id_col,
        input_layer_name=input_layer_name,
        output_layer_name=output_layer_name,
    )
    state_d = sb.render_state(dataframes, return_as="json")
    link_id = client.state.upload_state_json(state_d)
    url = client.state.build_neuroglancer_url(link_id, ngl_url=ngl_url)
    return url


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
