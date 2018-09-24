
from neuroglancer_annotation_ui.extension_core import check_layer, AnnotationExtensionBase, PointHolder
from neuroglancer_annotation_ui.ngl_rendering import SchemaRenderer
from emannotationschemas.cell_type_local import CellTypeLocal, allowed_types
from emannotationschemas.bound_sphere import BoundSphere
import re


class CellTypeLocalWithRule( CellTypeLocal ):
    @staticmethod
    def render_rule():
        return {'point': {'cell_type': ['pt']}}

def MakeBoundSphereWithRule( z_multiplier ):
    class BoundSphereWithRule( BoundSphere ):
        @staticmethod
        def render_rule():
            return {'sphere': {'sphere':[ ('ctr_pt', 'radius', z_multiplier) ] } } 
    return BoundSphereWithRule


class CellTypeExtension(AnnotationExtensionBase):
    def __init__(self, easy_viewer, annotation_client=None):
        super(CellTypeExtension, self).__init__(easy_viewer, annotation_client)
        self.ngl_renderer = {'cell_type':SchemaRenderer(CellTypeLocalWithRule),
                             'sphere':SchemaRenderer(MakeBoundSphereWithRule(0.1))
                             }
        self.allowed_layers = ['ivscc_cell_type']

        self.color_map = {'ivscc_cell_type': '#2222ff',
                          'ivscc_cell_type_point': '#cccccc'}

        # Which point goes to which layer
        self.point_layer_dict = {'ctr_pt': 'ivscc_cell_type',
                                 'radius': 'ivscc_cell_type'}

        # Which annotation goes to which layer
        self.anno_layer_dict = {'cell_type': 'ivscc_cell_type',
                                'sphere': 'ivscc_cell_type_point'}

        self.create_layers(None)

        self.points = PointHolder(pt_types=['ctr_pt', 'radius'],
                                  trigger='radius',
                                  layer_dict=self.point_layer_dict,
                                  viewer=self.viewer)


    @staticmethod
    def _default_key_bindings():
        bindings = {
            'update_center_point': 'keyk',
            'update_radius_point': 'keyj'}
        return bindings

    @staticmethod
    def _defined_layers():
        return ['ivscc_cell_type_point', 'ivscc_cell_type']

    def create_layers(self, s):
        for ln in self._defined_layers():
            self.viewer.add_annotation_layer(ln,
                                             self.color_map[ln])


    @check_layer()
    def update_center_point( self, s):
        pos = self.viewer.get_mouse_coordinates(s)
        new_id = self.points.update_point(pos, 'ctr_pt', message_type='cell type center point')
        self.viewer.select_annotation(self.point_layer_dict['ctr_pt'], new_id)

    @check_layer()
    def update_radius_point( self, s):
        pos = self.viewer.get_mouse_coordinates(s)
        anno_done = self.points.update_point(pos, 'radius', message_type='radius point')

        cell_type = self.validate_cell_type_annotation( self.points() )
        if cell_type is None:
            self.points.reset_points(pts_to_reset=['radius'])
            self.update_message('Please change the description to a valid cell type')
            return 

        if anno_done:
            vids_p = self.render_and_post_annotation(self.format_cell_type_data,
                                            'cell_type',
                                            self.anno_layer_dict,
                                            'cell_type')
            self.viewer.update_description(vids_p, cell_type)

            print('\n')
            print(self.format_sphere_data(self.points()))
            vids_s = self.render_and_post_annotation(self.format_sphere_data,
                                            'sphere',
                                            self.anno_layer_dict,
                                            'sphere')
            self.viewer.update_description(vids_s, cell_type)
            self.points.reset_points()


    def validate_cell_type_annotation(self, points):
        ct_anno = self.format_cell_type_data(points)
        schema = CellTypeLocal()
        d = schema.load(ct_anno)
        print(d)
        return ct_anno['cell_type']


    def format_cell_type_data(self, points):
        anno_point = self.viewer.get_annotation(self.point_layer_dict['ctr_pt'],
                                                points['ctr_pt'].id
                                                )
        if anno_point.description is None:
            cell_type = self.parse_cell_type_description('')
        else:
            cell_type = self.parse_cell_type_description(anno_point.description)

        datum = {'type':'cell_type_local',
                 'pt':{'position':[int(x) for x in points['ctr_pt'].point]},
                 'cell_type':cell_type,
                 'classification_system':'ivscc_m',
                 }
        return datum


    @staticmethod
    def parse_cell_type_description( description ):
        qry = re.search('(?P<cell_type>aspiny_d_[\d]+|aspiny_s_[\d]+|spiny_[\d]+)', description)
        if qry is not None:
            cell_type = qry.group()
        else:
            cell_type = ''
        return cell_type


    def format_sphere_data(self, points):
        rsq = 0
        for i in range(0,3):
            rsq += (points['ctr_pt'].point[i] - points['radius'].point[i])**2
        datum = {'type':'sphere',
                 'ctr_pt':{'position':[float(x) for x in points['ctr_pt'].point]},
                 'radius': rsq**0.5
                 }
        return datum

