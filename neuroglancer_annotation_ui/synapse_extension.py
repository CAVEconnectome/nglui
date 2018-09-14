from neuroglancer_annotation_ui.base import check_layer
from neuroglancer_annotation_ui.ngl_rendering import SchemaRenderer
from neuroglancer_annotation_ui.annotation import point_annotation, line_annotation
from emannotationschemas.synapse import SynapseSchema

class SynapseExtension():
    def __init__(self, easy_viewer, annotation_client=None):
        self.viewer = easy_viewer
        self.annotation_client = annotation_client
        self.ngl_renderer = SchemaRenderer(SynapseSchema)
        self.allowed_layers = ['synapses']
        self.create_synapse_layers(None)

        self.data = []
        self.synapse_points = {'pre_pt':None, 'post_pt':None, 'ctr_pt':None}

        self.pre_lines = []
        self.post_lines = []
        self.color_map = {'pre_line':'#ff0000',
                          'post_line':'#00ffff',
                          'pre_pt':'#ff0000',
                          'post_pt':'#00ffff',
                          }

    @staticmethod
    def _default_key_bindings():
        bindings = {
            'update_presynaptic_point': 'shift+keyq',
            'update_synapse': 'shift+keyw',
            'update_postsynaptic_point': 'shift+keye',
            'create_synapse_layers': 'shift+control+keys',
            'clear_segment': 'shift-keyv',
            }
        return bindings

    @staticmethod
    def _defined_layers():
        return ['synapses','pre_line','post_line','pre_point','post_point']

    def create_synapse_layers( self, s):
        for layer in self._defined_layers():
            self.viewer.add_annotation_layer(layer)

    @check_layer()
    def update_presynaptic_point( self, s):
        self.update_synapse_points( 'pre_pt', s)

    @check_layer()
    def update_postsynaptic_point( self, s):
        self.update_synapse_points( 'post_pt', s)

    def update_synapse_points( self, point_type, s):
        message_dict = {'pre_pt':'presynaptic point',
                        'post_pt':'postsynaptic point',
                        'ctr_pt': 'synapse'}
        layer_dict = {'pre_pt':'pre_point',
                      'post_pt':'post_point'}

        if (point_type == 'pre_pt') or (point_type=='post_pt'):
            if self.synapse_points[point_type] is None:
                message = 'Assigned {}'.format(message_dict[point_type])
            else:
                self.ngl_renderer.remove_annotations_from_viewer(layer_dict[point_type],
                                                                 self.synapse_points[point_type].id )
                message = 'Re-assigned {}'.format(message_dict[point_type])

            self.synapse_points[point_type] = self.make_synapse_point(s)
            self.viewer.add_annotation( layer_dict[point_type],
                                 [self.synapse_points[point_type]],
                                 self.color_map[point_type] )
            self.viewer.update_message( message)
        else:
            message = 'Assigned {}'.format(message_dict[point_type])
            self.synapse_points[point_type] = self.make_synapse_point(s)
            self.viewer.update_message( message)

    @check_layer()
    def update_synapse( self, s ):
        if (self.synapse_points['pre_pt'] is None) or \
                    (self.synapse_points['post_pt'] is None):
            self.viewer.update_message("Pre and Post targets must be defined before \
                                        adding a synapse!")
            return

        self.update_synapse_points( 'ctr_pt', s )
        synapse_data = self.format_synapse_data()
        self.ngl_renderer(self.viewer,
                          synapse_data,
                          layermap={'pre':'pre_line',
                                    'post':'post_line'}
                          )
        
        if self.annotation_client is not None:
            self.annotation_client.post_data(synapse_data)
        self.clear_segment(None)

    def make_synapse_point( self, s):
        pos = self.viewer.get_mouse_coordinates(s)
        if pos is not None:
            return point_annotation(pos)
        else:
            return None

    def format_synapse_data(self):
        return {'type':'synapse',
                'pre_pt':{'position':self.synapse_points['pre_pt'].point},
                'ctr_pt':{'position':self.synapse_points['ctr_pt'].point},
                'post_pt':{'position':self.synapse_points['post_pt'].point}}

    @check_layer()
    def clear_segment(self, s):
        self.synapse_points = {'pre_pt':None, 'post_pt':None, 'ctr_pt':None}
        self.viewer.update_message('Starting new synapse...')
