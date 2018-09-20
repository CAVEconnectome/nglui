from neuroglancer_annotation_ui.synapse_extension import SynapseExtension

class SynapseExtensionAIManual(SynapseExtension):
    """
    Subclasses the extension to configure the tables.
    This could also be done via a class factory, if necessary.
    """
    def __init__(self, easy_viewer, annotation_client=None):
        super(SynapseExtensionAIManual, self).__init__(easy_viewer, annotation_client)
        self.tables = {'synapse':'synapse_ai_manual'}


extension_mapping = {
    'synapse_ai_manual': SynapseExtensionAIManual,
}


def get_extensions():
    return [k for k in extension_mapping.keys()]
