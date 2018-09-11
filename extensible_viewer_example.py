import neuroglancer_annotation_ui.extensible_viewer as anno_ui
from neuroglancer_annotation_ui.connector_extension import ConnectorExtension
from neuroglancer_annotation_ui.cell_type_extension import CellTypeExtension

# Building a viewer
img_src = "your_favorite_image_source"
seg_src = "your_favorite_segmentation_source"

example_viewer = anno_ui.ExtensibleViewer()
example_viewer.add_image_layer('imagery', img_src)
example_viewer.add_segmentation_layer('segmentation', seg_src)


# Extending a viewer with just one function function
example_viewer.add_action('clear_all', 'shift+keyc', example_viewer.clear_all)


# Extending a viewer with a class and a bevy of actions
bindings = {
    'update_presynaptic_point': 'shift+keyq',
    'update_synapse': 'shift+keyw',
    'update_postsynaptic_point': 'shift+keye',
    'create_synapse_layer': 'shift+control+keys',
    'clear_segment': 'shift-keyv',
}
example_viewer.add_extension('synapses', ConnectorExtension, bindings )

# Or or we could do the same with with default actions placed in the class:
example_viewer.add_extension('synapses', ConnectorExtension, ConnectorExtension.default_bindings() )
example_viewer.add_extension('cell_types', CellTypeExtension, CellTypeExtension.default_bindings() )

print(syn_anno_viewer.url)