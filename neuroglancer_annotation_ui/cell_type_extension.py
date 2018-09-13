import neuroglancer
from neuroglancer_annotation_ui.annotation import point_annotation
from neuroglancer_annotation_ui.base import check_layer
from neuroglancer_annotation_ui.ngl_rendering import NeuroglancerRenderer
from emannotationschemas.cell_type_local import CellTypeLocal, allowed_types

class_sys_map = {}
for class_system, item_list in allowed_types.items():
    for item in item_list:
        class_sys_map[item] = class_system

cell_types = ['chandelier', 'pyramidal']

class CellTypeExtension():
    def __init__(self, easy_viewer, annotation_client=None ):
        # General
        self.data = []
        self.viewer = easy_viewer
        self.defined_layers = cell_types
        self.annotation_client = annotation_client
        self.allowed_layers = cell_types

        # Specific
        self.generate_cell_type_layers(None)
        self.ngl_renderer = NeuroglancerRenderer(CellTypeLocal)

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

    @check_layer()
    def add_cell_type_point( self, s ):
        # 1. Retrieve data
        xyz = self.viewer.get_mouse_coordinates(s)
        if xyz is not None:
            curr_layer = self.viewer.get_selected_layer()
            new_point = point_annotation(xyz)
            new_data = self.format_cell_type_annotation(new_point, curr_layer)
            self.data.append(new_data)
        else:
            self.viewer.update_message('Mouse position not well defined!')
            return
        # 2. Render to neuroglancer
        self.ngl_renderer(new_data)
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

    # def post_data(self, data, update_id=True):
    #     if self.annotation_client is not None:
    #         response = self.annotation_client.post( data )
    #         if update_id:

    #     else:
    #         return