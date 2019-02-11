from neuroglancer_annotation_ui.base import EasyViewer, AnnotationManager
from neuroglancer_annotation_ui.synapse_extension import SynapseExtension
from neuroglancer_annotation_ui.cell_type_point_extension import CellTypeExtension
from neuroglancer_annotation_ui.synapse_getter import SynapseGetterFactory
from neuroglancer_annotation_ui.cell_type_getter import CellTypeGetterFactory
from collections import namedtuple
# from neuroglancer_annotation_ui.soma_extension import SomaExtension

__version__ = "0.1.2"

dummy_db_config = {
    'sqlalchemy_database_uri': None,
    'materialization_version': None,
    'dataset_name': None,
    'annotation_endpoint': None,
}

cell_type_config = {'table_name': 'soma_valence',
                    'db_config': dummy_db_config,
                    }

synapse_config = {'table_name': 'pni_synapses_i3',
                  'db_config': dummy_db_config,
                  }

extension_mapping = {
    'AutomaticSynapseViewer': SynapseGetterFactory,
    'CellTypeViewer': CellTypeGetterFactory,
    }

def get_extensions():
    return [k for k in extension_mapping.keys()]

def get_extension_configs():
    return {'CellTypeViewer': cell_type_config,
            'AutomaticSynapseViewer': synapse_config,
            }
