from neuroglancer_annotation_ui.base import EasyViewer, AnnotationManager
from neuroglancer_annotation_ui.synapse_extension import SynapseExtension
from neuroglancer_annotation_ui.cell_type_point_extension import CellTypeExtension
from neuroglancer_annotation_ui.synapse_getter import SynapseGetterFactory
from neuroglancer_annotation_ui.cell_type_getter import CellTypeGetterFactory
from collections import namedtuple
# from neuroglancer_annotation_ui.soma_extension import SomaExtension

db_config = dict(uri="postgresql://postgres:welcometothematrix@35.196.105.34/postgres",
              data_version=36,
              dataset='pinky100',
              )

cell_type_config = {'table_name': 'soma_valence',
                    'schema_name': 'cell_type_local',
                    'db_config': db_config}

synapse_config = {'table_name': 'pni_synapses_i2',
                  'schema_name': 'synapse',
                  'db_config': db_config}

__version__ = "0.0.3"

extension_mapping = {
    # 'SynapsePlacer': SynapseExtension.set_config,
    # 'CellTypePlacer': CellTypeExtension.set_config,
    'AutomaticSynapseViewer': SynapseGetterFactory(**synapse_config),
    'CellTypeViewer': CellTypeGetterFactory(**cell_type_config),
    }

def get_extensions():
    return [k for k in extension_mapping.keys()]
