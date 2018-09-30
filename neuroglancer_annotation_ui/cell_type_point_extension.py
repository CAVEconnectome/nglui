from neuroglancer_annotation_ui.extension_core import check_layer, AnnotationExtensionBase, PointHolder
from neuroglancer_annotation_ui.ngl_rendering import SchemaRenderer
from emannotationschemas.cell_type_local import CellTypeLocal, allowed_types
import copy
import re

CELL_TYPE_TOOL_LAYER = 'cell_type_tool'
CELL_TYPE_DISPLAY_LAYER = 'cell_types'

class CellTypeLocalWithRule( CellTypeLocal ):
    @staticmethod
    def render_rule():
        return {'point': {'cell_type': ['pt']},
                'description_field': ['cell_type']}

class CellTypeExtension(AnnotationExtensionBase):
    def __init__(self, easy_viewer, annotation_client=None):
        super(CellTypeExtension, self).__init__(easy_viewer, annotation_client)

        self.color_map = {CELL_TYPE_TOOL_LAYER: '#cccccc',
                          CELL_TYPE_DISPLAY_LAYER: '#2255cc'
                          }

        self.create_layers(None)
        self.allowed_layers = [CELL_TYPE_TOOL_LAYER]

        self.ngl_renderer = {'cell_type':SchemaRenderer(CellTypeLocalWithRule),
                             }
        # Which annotation goes to which layer
        self.anno_layer_dict = {'cell_type': CELL_TYPE_DISPLAY_LAYER}

        # Which point goes to which layer
        self.point_layer_dict = {'ctr_pt': CELL_TYPE_TOOL_LAYER,
                                 'trigger_pt': CELL_TYPE_TOOL_LAYER}

        self.points = PointHolder(viewer=self.viewer,
                                  pt_types=['ctr_pt', 'trigger_pt'],
                                  trigger='trigger_pt',
                                  layer_dict=self.point_layer_dict)


    @staticmethod
    def _default_key_bindings():
        bindings = {
            'update_center_point_spiny': 'keyk',
            'update_center_point_aspiny': 'keyj',
            'update_center_point_e': 'shift+keyk',
            'update_center_point_i': 'shift+keyj',
            'update_center_point_blank': 'keyi',
            'update_center_point_uncertain': 'shift+keyi',
            'trigger_upload': 'keyu'}
        return bindings

    @staticmethod
    def _defined_layers():
        return [CELL_TYPE_DISPLAY_LAYER, CELL_TYPE_TOOL_LAYER]

    def create_layers(self, s):
        for ln in self._defined_layers():
            self.viewer.add_annotation_layer(ln,
                                             self.color_map[ln])
    @check_layer()
    def update_center_point( self, description, s):
        pos = self.viewer.get_mouse_coordinates(s)
        self.points.update_point(pos, 'ctr_pt', message_type='cell type center point')
        new_id = self.points.points['ctr_pt'].id
        self.viewer.update_description({self.point_layer_dict['ctr_pt']:[new_id]}, description)
        self.viewer.select_annotation(self.point_layer_dict['ctr_pt'], new_id)

    @check_layer()
    def trigger_upload( self, s):
        pos = self.viewer.get_mouse_coordinates(s)
        anno_done = self.points.update_point(pos, 'trigger_pt', message_type='confirmation')

        self.points.points['ctr_pt'] = self.viewer.get_annotation(self.point_layer_dict['ctr_pt'],
                                                self.points.points['ctr_pt'].id
                                                )

        cell_type = self.validate_cell_type_annotation( self.points() )
        if cell_type is None:
            self.points.reset_points(pts_to_reset=['trigger_pt'])
            self.viewer.update_message('Please change the description to a valid cell type')
            return

        if anno_done:
            self.render_and_post_annotation(self.format_cell_type_data,
                                            'cell_type',
                                            self.anno_layer_dict,
                                            'cell_type')
            self.points.reset_points()


    @check_layer()
    def update_center_point_spiny(self, s):
        self.update_center_point(description='spiny_', s=s)

    @check_layer()
    def update_center_point_aspiny(self, s):
        self.update_center_point(description='aspiny_s_', s=s)

    @check_layer()
    def update_center_point_blank(self, s):
        self.update_center_point(description='', s=s)

    @check_layer()
    def update_center_point_e(self, s):
        self.update_center_point(description='valence:e', s=s)

    @check_layer()
    def update_center_point_i(self, s):
        self.update_center_point(description='valence:i', s=s)

    @check_layer()
    def update_center_point_uncertain(self, s):
        self.update_center_point(description='uncertain', s=s)


    def validate_cell_type_annotation(self, points):
        ct_anno = self.format_cell_type_data(points)
        schema = CellTypeLocal()
        d = schema.load(ct_anno)
        if d.data.get('valid', False):
            return ct_anno['cell_type']
        else:
            return None


    @staticmethod
    def parse_cell_type_description( description ):
        cell_type = ''
        class_system = ''
        qry_ivscc = re.search('(?P<cell_type>aspiny_d_[\d]+|aspiny_s_[\d]+|spiny_[\d]+|uncertain)', description)
        if qry_ivscc is not None:
            cell_type = qry_ivscc.group()
            class_system = 'ivscc_m'
        else:
            qry_valence = re.search('valence\:(?P<cell_type>[eiEI]|uncertain)', description)
            if qry_valence is not None:
                cell_type = qry_valence.groupdict()['cell_type']
                class_system = 'valence'
        return cell_type, class_system


    def format_cell_type_data(self, points, cell_type=None, class_system=None):
        anno_point = self.points.points['ctr_pt']
        if (cell_type is None) or (class_system is None):
            if anno_point.description is None:
                cell_type, class_system = self.parse_cell_type_description('')
            else:
                cell_type, class_system = self.parse_cell_type_description(anno_point.description)

        datum = {'type':'cell_type_local',
                 'pt':{'position':[int(x) for x in points['ctr_pt'].point]},
                 'cell_type':cell_type.lower(),
                 'classification_system':class_system,
                 }
        return datum

    def _update_annotation(self, ngl_id):
        self.update_cell_type_annotation( ngl_id )

    def update_cell_type_annotation(self, ngl_id):
        # Read new position, read new description
        ln = self.viewer.get_selected_layer()

        # Format into the schema
        self.points.reset_points()
        self.points.points['ctr_pt'] = self.viewer.get_annotation(ln,
                                                                  ngl_id
                                                                  )
        print(self.points())
        # print(self.parse_cell_type_description(self.points.points['ctr_pt'].description))

        if self.validate_cell_type_annotation(self.points()) is not None:
            # print(self.points())
            new_datum = self.format_cell_type_data(self.points())
            # print(new_datum)
            # Upload to the server as an update.
            ae_type, ae_id = self.parse_anno_id(self.get_anno_id(ngl_id))
            self.annotation_client.update_annotation(ae_type, ae_id, new_datum)
            self.viewer.update_message('Updated annotation')

        else:
            self.viewer.update_message('Updated cell type not valid, please change or reload annotations')
        self.points.reset_points()
