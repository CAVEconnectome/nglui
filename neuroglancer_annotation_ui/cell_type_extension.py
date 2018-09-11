import neuroglancer
from neuroglancer_annotation_ui import annotation
from neuroglancer_annotation_ui.extensible_viewer import check_layer
from emannotationschemas.cell_type_local import allowed_types

class_sys_map = {}
for class_system, item_list in allowed_types.items():
    for item in item_list:
        class_sys_map[item] = class_system

cell_types = ['chandelier', 'pyramidal']

class CellTypeExtension():
    def __init__(self, extensible_viewer ):
        self.data = []
        self.viewer = extensible_viewer

        self.cell_types = cell_types
        self.generate_cell_type_layers(None)

    @staticmethod
    def default_bindings( ):
        bindings = {
        'generate_cell_type_layers':'shift+control+keyt',
        'add_cell_type_point':'shift+keyt',
        }
        return bindings

    def generate_cell_type_layers(self, s):
        for cell_type in cell_types:
            self.viewer.add_annotation_layer(cell_type)

    @check_layer(cell_types)
    def add_cell_type_point( self, s ):
        point = self.viewer.add_point(s)
        curr_layer = self.viewer.get_selected_layer()

        self.data.append( self.format_cell_type_annotation(point, curr_layer) )
        self.viewer.add_annotation( curr_layer, point )
        self.viewer.update_message('Added point annotating cell type {}'.format(curr_layer))


    def format_cell_type_annotation(self, point, curr_layer ):
        return {'type':'cell_type_local',
                'pt':{'position': point.point.tolist() },
                'cell_type':curr_layer,
                'classification_system':class_sys_map[curr_layer]}
    #self.annotation_client = annotation_client
