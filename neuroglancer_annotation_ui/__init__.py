from neuroglancer_annotation_ui.synapse_extension import SynapseExtension
from neuroglancer_annotation_ui.cell_type_extension import CellTypeExtension

class SynapseExtensionAIManual(SynapseExtension):
    """
    Subclasses the extension to configure the tables.
    This could also be done via a class factory, if necessary.
    """
    def __init__(self, easy_viewer, annotation_client=None):
        super(SynapseExtensionAIManual, self).__init__(easy_viewer, annotation_client)
        self.db_tables = {'synapse':'synapse_ai_manual'}

class CellTypeExtensionAIManual(CellTypeExtension):
    def __init__(self, easy_viewer, annotation_client=None):
        super(CellTypeExtensionAIManual,self).__init__(easy_viewer, annotation_client)
        self.db_tables = {'cell_type': 'cell_type_ai_manual',
                          'sphere': 'soma_ai_manual'}


extension_mapping = {
    'synapse_ai_manual': SynapseExtensionAIManual,
    'cell_types_ai_manual': CellTypeExtensionAIManual,
    }


def get_extensions():
    return [k for k in extension_mapping.keys()]
