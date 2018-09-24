from neuroglancer_annotation_ui.synapse_extension import SynapseExtension
from neuroglancer_annotation_ui.cell_type_extension import CellTypeExtension


extension_mapping = {
    'synapse_ai_manual': SynapseExtension.set_db_tables('SynapseAIManual',
                                                       {'synapse':'synapse_ai_manual'}
                                                       ),
    'cell_type_ai_manual': CellTypeExtension.set_db_tables('CellTypeAIManual',
                                                          {'cell_type': 'cell_type_ai_manual',
                                                           'sphere': 'soma_ai_manual'
                                                           }
                                                          )
    }

def get_extensions():
    return [k for k in extension_mapping.keys()]
