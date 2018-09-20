from neuroglancer_annotation_ui.base import check_layer
from neuroglancer_annotation_ui.ngl_rendering import SchemaRenderer
from emannotationschemas.cell_type_local import CellTypeLocal, allowed_types

class_sys_map = {}
for class_system, item_list in allowed_types.items():
    for item in item_list:
        class_sys_map[item] = class_system

cell_types = ['chandelier', 'pyramidal']

class CellTypeLocalWithRule( CellTypeLocal ):
    @staticmethod
    def render_rule():
        return {'point':{'cell_type':['pt']}}

class CellTypeExtension():
    def __init__(self, easy_viewer, annotation_client=None ):
        # General
        self.viewer = easy_viewer
        self.annotation_client = annotation_client
        self.ngl_renderer = SchemaRenderer(CellTypeLocalWithRule)
        self.allowed_layers = cell_types

        # Specific
        self.data = []
        self.generate_cell_type_layers(None)

    @staticmethod
    def _default_key_bindings( ):
        bindings = {
        'generate_cell_type_layers':'shift+control+keyt',
        'add_cell_type_point':'shift+keyt',
        }
        return bindings

    @staticmethod
    def _defined_layers():
        return cell_types

    def generate_cell_type_layers(self, s):
        for cell_type in cell_types:
            self.viewer.add_annotation_layer(cell_type)

    @check_layer()
    def add_cell_type_point( self, s ):
        # 1. Retrieve data
        xyz = self.viewer.get_mouse_coordinates(s)
        if xyz is not None:
            curr_layer = self.viewer.get_selected_layer()
            new_point = xyz
            new_data = self.format_cell_type_annotation(new_point, curr_layer)
            self.data.append(new_data)
        else:
            self.viewer.update_message('Mouse position not well defined!')
            return
        # 2. Render to neuroglancer
        self.ngl_renderer(self.viewer, new_data, layermap={'cell_type':curr_layer})
        self.viewer.update_message('Added point annotating cell type {}'.format(curr_layer))

        # 3. Send to annotation client and update neuroglancer annotation.
        self.post_data( new_data )

    def format_cell_type_annotation(self, point, curr_layer ):
        """
        Should match the schema in dict form
        """
        return {'type':'cell_type_local',
                'pt':{'position': [int(x) for x in point] },
                'cell_type':curr_layer,
                'classification_system':class_sys_map[curr_layer]}

    def post_data(self, data, update_id=True):
        return