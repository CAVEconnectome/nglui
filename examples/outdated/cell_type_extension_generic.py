from neuroglancer_annotation_ui.extension_core import check_layer, AnnotationExtensionBase, PointHolder
from neuroglancer_annotation_ui.ngl_rendering import SchemaRenderer
from emannotationschemas.cell_type_local import CellTypeLocal, allowed_types, allowed_classification_systems
from itertools import chain
import copy
import re

CELL_TYPE_TOOL_LAYER = 'cell_type_tool'
CELL_TYPE_DISPLAY_LAYER = 'cell_types'

ANNO_NAME = 'cell_type'
CTR_PT, RADIUS_PT = 'pt', 'trigger_pt'
TYPE_RENDERER='cell_type'

def CellTypeExtensionFactory(CellTypeSchema):
    render_rule =  {'point': {ANNO_NAME: [PT_NAME]},
                    'description_field': ['classification_system','cell_type']}
    class CellTypeExtension(CellTypeExtensionUnspecified):
        def __init__ (easy_viewer, annotation_client=None):
            super(CellTypeExtension, self).__init__(easy_viewer, annotation_client=None)
            self.ngl_renderer = {TYPE_RENDERER: SchemaRenderer(CellTypeSchema, render_rule=render_rule)}
            self.schema = CellTypeSchema

    return CellTypeExtension

class CellTypeExtensionUnspecified(AnnotationExtensionBase):
    def __init__(self, easy_viewer, annotation_client=None):
        super(CellTypeExtension, self).__init__(easy_viewer, annotation_client)

        self.color_map = {CELL_TYPE_TOOL_LAYER: '#cccccc',
                          CELL_TYPE_DISPLAY_LAYER: '#2255cc'
                          }

        self.create_layers(None)
        self.allowed_layers = [CELL_TYPE_TOOL_LAYER]

        # Which annotation goes to which layer from a finished annotation
        self.anno_layer_dict = {ANNO_NAME: CELL_TYPE_DISPLAY_LAYER}

        # Which point goes to which layer while building annotation
        self.point_layer_dict = {CTR_PT: CELL_TYPE_TOOL_LAYER,
                                 RADIUS_PT: CELL_TYPE_TOOL_LAYER}

        self.points = PointHolder(viewer=self.viewer,
                                  pt_types=[CTR_PT, RADIUS_PT],
                                  trigger=RADIUS_PT,
                                  layer_dict=self.point_layer_dict)

    @staticmethod
    def _default_key_bindings():
        bindings = {
            'update_center_point_spiny': 'keyq',
            'update_center_point_aspiny': 'keyw',
            'update_center_point_uncertain': 'keyu',
            'trigger_upload': 'keyt'}
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
        self.points.update_point(pos, CTR_PT, message_type='cell type center point')
        new_id = self.points.points[CTR_PT].id
        self.viewer.update_description({self.point_layer_dict[CTR_PT]:[new_id]}, description)
        self.viewer.select_annotation(self.point_layer_dict[CTR_PT], new_id)

    @check_layer()
    def trigger_upload( self, s):
        pos = self.viewer.get_mouse_coordinates(s)
        anno_done = self.points.update_point(pos, RADIUS_PT, message_type='confirmation')

        self.points.points[CTR_PT] = self.viewer.get_annotation(self.point_layer_dict[CTR_PT],
                                                self.points.points[CTR_PT].id
                                                )

        cell_type = self.validate_cell_type_annotation( self.points() )
        if cell_type is None:
            self.points.reset_points(pts_to_reset=[RADIUS_PT])
            self.viewer.update_message('Please change the description to a valid cell type')
            return

        if anno_done:
            self.render_and_post_annotation(self.format_cell_type_data,
                                            TYPE_RENDERER,
                                            self.anno_layer_dict,
                                            ANNO_NAME)
            self.points.reset_points()


    @check_layer()
    def update_center_point_spiny(self, s):
        self.update_center_point(description='ivscc_m:spiny_', s=s)

    @check_layer()
    def update_center_point_aspiny(self, s):
        self.update_center_point(description='ivscc_m:aspiny_s_', s=s)

    @check_layer()
    def update_center_point_uncertain(self, s):
        self.update_center_point(description='ivscc_m:uncertain', s=s)

    def validate_cell_type_annotation(self, points):
        ct_anno = self.format_cell_type_data(points)
        schema = CellTypeLocal()
        d = schema.load(ct_anno)
        if d.data.get('valid', False):
            return ct_anno['cell_type']
        else:
            return None


    @staticmethod
    def parse_cell_type_description( description, binding_char='\:' ):
        class_sys_qry_str = '|'.join(allowed_classification_systems)
        cell_type_qry_str = '|'.join(['{}$'.format(x) for x in chain.from_iterable(allowed_types.values())])
        qry = re.search('(?P<csys>{}){}(?P<ct>{})'.format(class_sys_qry_str,
                                                          binding_char,
                                                          cell_type_qry_str),
                        description,
                        re.MULTILINE)
        if qry is not None:
            class_system = qry.groupdict()['csys']
            cell_type = qry.groupdict()['ct']
        else:
            cell_type = ''
            class_system = ''
        return cell_type, class_system


    def format_cell_type_data(self, points, cell_type=None, class_system=None):
        anno_point = points['ctr_pt']
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
        new_data_points = dict(ctr_pt=self.viewer.get_annotation(ln,ngl_id),
                               trigger_pt=None)

        if self.validate_cell_type_annotation(new_data_points) is not None:
            new_datum = self.format_cell_type_data(new_data_points)
            ae_type, ae_id = self.parse_anno_id(self.get_anno_id(ngl_id))
            self.annotation_client.update_annotation(ae_type, ae_id, new_datum)
            self.viewer.update_message('Updated annotation')

        else:
            self.viewer.update_message('Updated cell type not valid, please change or reload annotations')
