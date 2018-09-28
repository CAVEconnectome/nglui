from neuroglancer_annotation_ui.extension_core import check_layer, AnnotationExtensionBase, PointHolder
from neuroglancer_annotation_ui.ngl_rendering import SchemaRenderer
from emannotationschemas.bound_sphere import BoundSphere
import copy
import re

def MakeBoundSphereWithRule( z_multiplier ):
    class BoundSphereWithRule( BoundSphere ):
        @staticmethod
        def render_rule():
            return {'sphere': {'sphere':[ ('ctr_pt', 'radius', z_multiplier) ] } } 
    return BoundSphereWithRule


class SomaExtension(AnnotationExtensionBase):
    def __init__(self, easy_viewer, annotation_client=None):
        super(SomaExtension, self).__init__(easy_viewer, annotation_client)
        self.ngl_renderer = {'sphere':SchemaRenderer(MakeBoundSphereWithRule(0.1))}
        self.allowed_layers = ['somata']

        self.color_map = {'somata': '#2222cc'}
        self.point_layer_dict = {'ctr_pt': 'somata',
                                 'radius': 'somata'}
        self.anno_layer_dict = {'sphere': 'somata'}
        self.create_layers(None)
        self.points = PointHolder(self.viewer,
                                  ['ctr_pt', 'radius'],
                                  'radius',
                                  self.point_layer_dict)


    @staticmethod
    def _default_key_bindings():
        bindings = {'update_center_point': 'keyi',
                    'update_radius_point': 'keyu'}
        return bindings


    @staticmethod
    def _defined_layers():
        return ['somata']


    def create_layers(self, s):
        for ln in self._defined_layers():
            self.viewer.add_annotation_layer(ln,
                                             self.color_map[ln])


    @check_layer()
    def update_center_point( self, s):
        pos = self.viewer.get_mouse_coordinates(s)
        new_id = self.points.update_point(pos, 'ctr_pt', message_type='soma center')
        self.viewer.select_annotation(self.point_layer_dict['ctr_pt'], new_id)


    @check_layer()
    def update_radius_point( self, s):
        pos = self.viewer.get_mouse_coordinates(s)
        anno_done = self.points.update_point(pos, 'radius', message_type='radius point')
        if anno_done:
            vid_s = self.render_and_post_annotation( self.format_sphere_data,
                                                     'sphere',
                                                     self.anno_layer_dict,
                                                     'sphere')
            self.viewer.update_message('Annotated soma')
            self.points.reset_points()


    def format_sphere_data(self, points):
        rsq = 0
        for i in range(0,3):
            rsq += (points['ctr_pt'].point[i] - points['radius'].point[i])**2
        datum = {'type':'sphere',
                 'ctr_pt':{'position':[float(x) for x in points['ctr_pt'].point]},
                 'radius': rsq**0.5
                 }
        return datum


    def _update_annotation(self, ngl_id):
        ln = self.viewer.get_selected_layer()
        for anno in self.viewer.state.layers[ln].annotations:
            if anno.id == ngl_id:
                pos = anno.center
                rad = copy.copy(anno.radii)
                break

        #Format into schema
        self.points.reset_points()
        self.points.update_point(pos, 'ctr_pt')
        rad[1:]=0
        rad_pt = pos+rad
        self.points.update_point(rad_pt, 'radius')
        new_datum = self.format_sphere_data( self.points() )
        self.points.reset_points()

        # Upload to the server as an update
        ae_type, ae_id = self.parse_anno_id(self.get_anno_id(ngl_id))
        self.annotation_client.update_annotation(ae_type, ae_id, new_datum)
        self.viewer.update_message('Updated soma annotation!')