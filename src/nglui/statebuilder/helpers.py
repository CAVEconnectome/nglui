from .layers import (
    ImageLayerConfig,
    SegmentationLayerConfig,
    AnnotationLayerConfig,
    PointMapper,
)
from .statebuilder import ChainedStateBuilder, StateBuilder

CONTRAST_CONFIG = {
    "minnie65_phase3_v1": {
        "contrast_controls": True,
        "black": 0.35,
        "white": 0.70,
    }
}


def make_point_statebuilder(
    client, point_column="pt_position", linked_seg_column="pt_root_id", contrast=None
):
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


seg_layer = SegmentationLayerConfig(seg_source, selected_ids_column="post_pt_root_id")

postsyn_mapper = LineMapper(
    point_column_a="pre_pt_position", point_column_b="ctr_pt_position"
)
postsyn_annos = AnnotationLayerConfig(
    "post", color="#00CCCC", mapping_rules=postsyn_mapper
)

postsyn_sb = StateBuilder(layers=[img_layer, seg_layer, postsyn_annos])

# Second state builder
presyn_mapper = LineMapper(
    point_column_a="ctr_pt_position", point_column_b="post_pt_position"
)
presyn_annos = AnnotationLayerConfig(
    "pre", color="#CC1111", mapping_rules=presyn_mapper
)

presyn_sb = StateBuilder(layers=[presyn_annos])


def make_pre_post_statebuilder(
    client,
    root_id,
    show_inputs=False,
    show_outputs=False,
    contrast=None,
    point_column="ctr_pt_position",
    view_kws=None,
):

    img_layer, seg_layer = from_client(client, contrast=contrast)
    seg_layer.add_selection_map(fixed_ids=root_id)
    sb1 = StateBuilder(
        layers=[img_layer, seg_layer],
        resolution=client.info.viewer_resolution(),
        view_kws=view_kws,
    )

    state_builders = [sb1]
    if show_inputs:
        # First state builder

        input_point_mapper = PointMapper(
            point_column=point_column, linked_segmentation_column="pre_pt_root_id"
        )
        inputs_lay = AnnotationLayerConfig(
            "syns_in", mapping_rules=[input_point_mapper]
        )
        sb_in = StateBuilder([inputs_lay])
        state_builders.append(sb_in)
    if show_outputs:
        output_point_mapper = PointMapper(
            point_column=point_column, linked_segmentation_column="post_pt_root_id"
        )
        outputs_lay = AnnotationLayerConfig(
            "syns_in", mapping_rules=[output_point_mapper]
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
    point_column="ctr_pt_position",
    view_kws=None,
    ngl_url=None,
):
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
    )
    state_d = sb.render_state(dataframes, return_as="json")
    link_id = client.state.upload_state_json(state_d)
    url = client.state.build_neuroglancer_url(link_id, ng_url=ng_url)
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
