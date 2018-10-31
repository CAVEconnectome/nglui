from neuroglancer_annotation_ui.synapse_extension import SynapseExtension
from neuroglancer_annotation_ui.cell_type_point_extension import CellTypeExtension
from collections import namedtuple
# from neuroglancer_annotation_ui.soma_extension import SomaExtension

__version__ = "0.0.3"

extension_mapping = {
    'synapse_ai_manual': SynapseExtension.set_db_tables('SynapseAIManual',
                                                        {'synapse':'synapse_ai_manual'}
                                                        ),
    'cell_type_ai_manual': CellTypeExtension.set_db_tables('CellTypeAIManual',
                                                           {'cell_type': 'cell_type_ai_manual'}
                                                           ),
    }

def get_extensions():
    return [k for k in extension_mapping.keys()]

"""
Thoughts below for how to make this more generic going forward

ExtensionInfo = namedtuple('ExtensionInfo',['class', 'schema_map'])
extension_map = {'synapse': ExtensionInfo(SynapseExtension,
                                           SynapseExtension._schema_map()
                                          ),
                 }

# Here we need a function that given a schema finds the tables that can go in that schema.
# then a schema_table_map swaps out {annotation_name: schema} from, say _required_schema()
# and replaces it with {annotation_name:table}

def configure_extension( Extension, class_name=None, anno_schema_map={}, schema_table_map={}):
    if class_name == None:
        if len(schema_table_map) == 0:
            class_name == Extension.__name__
        else:
            class_name = Extension.__name__ + '_'.join(schema_table_map.values())
    anno_table_map = {}
    if len(anno_schema_map) > 0:
        anno_table_map = {anno:schema_table_map[schema_name] for anno, schema_name in anno_schema_map.items()}
    return Extension.set_db_tables(class_name=class_name,
                                   db_tables=anno_table_map)
"""
