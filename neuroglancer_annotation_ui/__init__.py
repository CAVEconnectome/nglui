from neuroglancer_annotation_ui.synapse_extension import SynapseExtension

extension_mapping = {
    'synapse': SynapseExtension
}

def get_extensions():
    return [k for k in extension_mapping.keys()]
