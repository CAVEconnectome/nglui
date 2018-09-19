from neuroglancer_annotation_ui.extension_core import check_layer, AnnotationExtensionBase
from neuroglancer_annotation_ui.ngl_rendering import SchemaRenderer
from neuroglancer_annotation_ui.annotation import point_annotation, \
                                                  line_annotation
from emannotationschemas.synapse import SynapseSchema

class SynapseSchemaWithRule(SynapseSchema):
    @staticmethod
    def render_rule():
        return {'line': {'pre': [('pre_pt', 'ctr_pt')],
                         'post': [('post_pt', 'ctr_pt')]},
                'point': {'syn': ['ctr_pt']}
                }

class SynapseExtension(AnnotationExtensionBase):
    def __init__(self, easy_viewer, annotation_client=None):
        super(SynapseExtension, self).__init__(easy_viewer, annotation_client)
        self.ngl_renderer = SchemaRenderer(SynapseSchemaWithRule)
        self.allowed_layers = ['synapses']

        self.color_map = {'synapses': '#000000',
                          'synapses_pre': '#ff0000',
                          'synapses_post': '#00ffff',
                          }
        self.create_synapse_layers(None)
        self.synapse_points = {'pre_pt': None, 'post_pt': None, 'ctr_pt': None}

    @staticmethod
    def _default_key_bindings():
        bindings = {
            'update_presynaptic_point': 'shift+keyq',
            'update_synapse': 'shift+keyw',
            'update_postsynaptic_point': 'shift+keye',
            'create_synapse_layers': 'shift+control+keys',
            'clear_segment': 'shift-keyv'}
        return bindings

    @staticmethod
    def _defined_layers():
        return ['synapses_pre', 'synapses_post', 'synapses']

    def create_synapse_layers(self, s):
        for layer in self._defined_layers():
            self.viewer.add_annotation_layer(layer,
                                             color=self.color_map[layer])

    @check_layer()
    def update_presynaptic_point(self, s):
        self.update_synapse_points('pre_pt', s)

    @check_layer()
    def update_postsynaptic_point(self, s):
        self.update_synapse_points('post_pt', s)

    def update_synapse_points(self, point_type, s):
        message_dict = {'pre_pt': 'presynaptic point',
                        'post_pt': 'postsynaptic point',
                        'ctr_pt': 'synapse'}
        layer_dict = {'pre_pt': 'synapses_pre',
                      'post_pt': 'synapses_post'}

        if (point_type == 'pre_pt') or (point_type == 'post_pt'):
            if self.synapse_points[point_type] is None:
                message = 'Assigned {}'.format(message_dict[point_type])
            else:
                self.viewer.remove_annotation(layer_dict[point_type],
                                              self.synapse_points[point_type].id)
                message = 'Re-assigned {}'.format(message_dict[point_type])

            self.synapse_points[point_type] = self.make_synapse_point(s)
            self.viewer.add_annotation( layer_dict[point_type],
                                 [self.synapse_points[point_type]] )
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

        viewer_ids = self.ngl_renderer(self.viewer,
                                      synapse_data,
                                      layermap={'pre':'synapses_pre',
                                                'post':'synapses_post',
                                                'syn':'synapses'},
                                      replace_annotations={'synapses_pre':self.synapse_points['pre_pt'].id,
                                                           'synapses_post':self.synapse_points['post_pt'].id}
                                      )
        self.clear_segment()

        if self.annotation_client is not None:
            annotation_id = self._post_data(synapse_data)
            id_description = 'synapse_{}'.format(annotation_id[0])
            self.viewer.update_description(viewer_ids, id_description)
            self._update_map_id( viewer_ids, id_description )

    def _post_data(self, synapse_data):
        response = self.annotation_client.post_annotation('synapse', [synapse_data])
        return response

    def make_synapse_point( self, s):
        pos = self.viewer.get_mouse_coordinates(s)
        if pos is not None:
            return point_annotation(pos)
        else:
            return None

    def format_synapse_data(self):
        return {'type':'synapse',
                'pre_pt':{'position':[int(x) for x in self.synapse_points['pre_pt'].point]},
                'ctr_pt':{'position':[int(x) for x in self.synapse_points['ctr_pt'].point]},
                'post_pt':{'position':[int(x) for x in self.synapse_points['post_pt'].point]}}

    @check_layer()
    def clear_segment(self, s=None):
        self.synapse_points = {'pre_pt':None, 'post_pt':None, 'ctr_pt':None}
        self.viewer.update_message('Starting new synapse...')

    @check_layer()
    def _delete_annotation(self, ngl_id ):
        anno_id = self.get_anno_id(ngl_id)
        ae_type, ae_id = self.parse_anno_id(anno_id)
        #try:
        self.annotation_client.delete_annotation(annotation_type=ae_type,
                                                     oid=ae_id)
        # except:
        #     self.viewer.update_message('Could not delete annotation!')
        #     return

        for _, row in self._annotation_filtered_iterrows(anno_id=anno_id):
            self.viewer.remove_annotation(row['layer'], row['ngl_id'])

        self._remove_map_id(anno_id)
