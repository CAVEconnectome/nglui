from neuroglancer_annotation_ui.extension_core import check_layer, \
                                                      AnnotationExtensionStateResponsive, \
                                                      OrderedPointHolder
from neuroglancer_annotation_ui.ngl_rendering import SchemaRenderer
from neuroglancer_annotation_ui.annotation import point_annotation, \
                                                  line_annotation
from emannotationschemas.synapse import SynapseSchema

PRE_LAYER, POST_LAYER, SYN_LAYER = 'synapses_pre', 'synapses_post', 'synapses'
PRE_PT, POST_PT, SYN_PT = 'pre_pt', 'post_pt', 'ctr_pt' 
PRE_ANNO, POST_ANNO, SYN_ANNO = 'pre', 'post', 'syn'

SYNAPSE_RENDERER = 'synapse'

# Key in db_tables, most useful if multiple tables come from the same 
DB_TABLE_KEY = 'synapse'

# Sets the order of points created by clicking
POINT_TYPE_ORDER = {0: PRE_PT,
                    1: POST_PT,
                    2: SYN_PT}

# Assigns names in the message field on creation of point types
MESSAGE_DICT = {PRE_PT: 'presynaptic point',
                POST_PT: 'postsynaptic point',
                SYN_PT: 'synapse'}

# Assigns rendered annotation names (from render_rule) to layers
ANNO_LAYER_DICT = {PRE_ANNO: PRE_LAYER,
                   POST_ANNO: POST_LAYER,
                   SYN_ANNO: SYN_LAYER}

# Assigns point types to different layers
POINT_LAYER_MAP = {PRE_PT: PRE_LAYER,
                   POST_PT: POST_LAYER,
                   SYN_PT: SYN_LAYER}

# Assigns colors to layers
COLOR_MAP = {SYN_LAYER: '#cccccc',
             PRE_LAYER: '#ff0000',
             POST_LAYER: '#00ffff'}


class SynapseSchemaWithRule(SynapseSchema):
    @staticmethod
    def render_rule():
        return {'line': {PRE_ANNO: [(PRE_PT, SYN_PT)],
                         POST_ANNO: [(POST_PT, SYN_PT)]},
                'point': {SYN_ANNO: [SYN_PT]}
                }


class SynapseExtension(AnnotationExtensionStateResponsive):
    def __init__(self, easy_viewer, annotation_client=None):
        super(SynapseExtension, self).__init__(easy_viewer, annotation_client)
        self.ngl_renderer = {SYNAPSE_RENDERER: SchemaRenderer(SynapseSchemaWithRule)}
        self.allowed_layers = [SYN_LAYER]

        self.color_map = COLOR_MAP
        self.point_layer_dict = POINT_LAYER_MAP
        self.anno_layer_dict = ANNO_LAYER_DICT

        self.points = OrderedPointHolder(viewer=self.viewer,
                                         pt_type_dict=POINT_TYPE_ORDER,
                                         trigger=SYN_PT,
                                         layer_dict=self.point_layer_dict,
                                         message_dict=MESSAGE_DICT)
        self.create_synapse_layers()
        # TODO Select layer and activate node tool

    @staticmethod
    def _default_key_bindings():
        bindings = {
            }
        return bindings

    @staticmethod
    def _defined_layers():
        return [PRE_LAYER, POST_LAYER, SYN_LAYER]

    def create_synapse_layers(self):
        for layer in self._defined_layers():
            self.viewer.add_annotation_layer(layer,
                                             color=self.color_map[layer])


    def on_changed_annotations(self, new_annos, changed_annos, removed_ids):
        '''
            This is the function deployed when the annotation state changes. 
            new_annos : list of (layer_name, annotation)
            changed_annos : list of (layer_name, annotation)
            removed_ids : set of ngl_ids
        '''
        for row in new_annos:
            # This will ensure that anything not managed is cleaned up
            self.viewer.remove_annotation(row[0],row[1].id)
            if row[0] in self.allowed_layers:
                if row[1].type=='point':
                    self.update_synapse_points(pos=row[1].point)
            

    def update_synapse_points(self, pos):

        anno_done = self.points.add_next_point(pos)
        if anno_done:
            self.render_and_post_annotation(data_formatter=self.format_synapse_data,
                                            render_name=SYNAPSE_RENDERER,
                                            anno_layer_dict=self.anno_layer_dict,
                                            table_name=DB_TABLE_KEY)
            self.points.reset_points()


    def _cancel_annotation( self ):
        self.points.reset_points()
        self._reset_point_type()
        self.viewer.update_message('Canceled annotation! No active annotations.')


    def format_synapse_data(self, points):
        return {'type':'synapse',
                'pre_pt':{'position':[int(x) for x in points[PRE_PT].point]},
                'ctr_pt':{'position':[int(x) for x in points[SYN_PT].point]},
                'post_pt':{'position':[int(x) for x in points[POST_PT].point]}}

    def _update_annotation(self, ngl_id):
        anno_id = self.get_anno_id(ngl_id)
        d_syn = self.annotation_df[self.annotation_df.anno_id==anno_id]
        ngl_id_ctr = d_syn[d_syn.layer == SYN_LAYER].ngl_id.values[0]
        ngl_id_pre = d_syn[d_syn.layer == PRE_LAYER].ngl_id.values[0]
        ngl_id_post = d_syn[d_syn.layer == PRE_LAYER].ngl_id.values[0]

        self.points.reset_points()
        self.points.update_point(self._get_pt_pos(PRE_LAYER, ngl_id_pre), PRE_PT)
        self.points.update_point(self._get_pt_pos(POST_LAYER, ngl_id_post), POST_PT)
        self.points.update_point(self._get_pt_pos(SYN_LAYER, ngl_id_ctr), SYN_PT)

        new_datum = self.format_synapse_data( self.points() )
        self.points.reset_points()

        ae_type, ae_id = self.parse_anno_id(anno_id)
        self.annotation_client.update_annotation(ae_type, ae_id, new_datum)
        self.viewer.update_message('Updated synapse annotation!')

    def _get_pt_pos(self, ln, ngl_id):
        for anno in self.viewer.state.layers[ln].annotations:
            if anno.id == ngl_id:
                if anno.type == 'point':
                    return anno.point
                elif anno.type == 'line':
                    return anno.pointA
        else:
            return None
