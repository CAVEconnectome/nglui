from .extension_core import AnnotationExtensionBase
from .synapse_extension import SynapseSchemaWithRule
from collections import defaultdict
from neuroglancer import random_token
from pandas import DataFrame

PRESYN_LN = 'asyn_pre'
POSTSYN_LN = 'asyn_pre'
CTRSYN_LN = 'synapses'

DB_ANNO_NAME = 'synapse'
RENDER_ANNO_NAME = 'synapse'

class SynapseGetterExtension(AnnotationExtensionBase):
    def __init__(self, easy_viewer, annotation_client=None):
        super(SynapseGetterExtension, self).__init__(easy_viewer, annotation_client)
        self.ngl_renderer = {RENDER_ANNO_NAME: SchemaRenderer(SynapseSchemaWithRule)}
        self.allowed_layers = [ln.name for ln in easy_viewer.layers if ln.type == 'segmentation']
        self.color_map = {PRESYN_LN:'#ff0000',
                          POSTSYN_LN:'#00ffff'}
        self.anno_layer_dict = {'pre':PRESYN_LN,
                                'post':POSTSYN_LN}
        self.create_synapse_layers(None)


    @staticmethod
    def _defined_layers():
        return [PRESYN_LN, POSTSYN_LN, CTRSYN_LN]


    @staticmethod
    def _default_key_bindings():
        bindings = {
            'get_synapses_for_selected': 'shift+keys',
            'clear_synapses': 'shift+control+keys',
            }
        return bindings


    def create_synapse_layers(self, s):
        for layer in self._defined_layers():
            self.viewer.add_annotation_layer(layer,
                                             color=self.color_map[layer])


    @check_layer()
    def get_synapses_for_selected(self, s):
        ln = self.viewer.get_selected_layer()
        oids = self.viewer.selected_objects(ln)
        annos = list()
        for oid in oids:
            synapse_anno_ids = self.annotation_client.get_annotations_of_root_id(self.db_tables[DB_ANNO_NAME],
                                                                                 oid)
            for aid in synapse_anno_ids:
                annos.append(aid)
        for anno in annos:
            self.load_annotation_by_aid(self.db_tables[DB_ANNO_NAME], anno, RENDER_ANNO_NAME)
        return            


    def _update_annotation(self, ngl_id):
        self.viewer.update_message('Manual editing of these synapses is not permitted')
        return

    # def update_synapse_annotation(self, ngl_id):
    #     anno_id = self.get_anno_id(ngl_id)
    #     d_syn = self.annotation_df[self.annotation_df.anno_id==anno_id]
    #     ngl_id_ctr = d_syn[d_syn.layer==CTRSYN_LN].ngl_id.values[0]
    #     ngl_id_pre = d_syn[d_syn.layer==PRESYN_LN].ngl_id.values[0]
    #     ngl_id_post = d_syn[d_syn.layer==POSTSYN_LN].ngl_id.values[0]

    #     new_pt = dict(pre_pt=self._get_pt_pos('synapses_pre', ngl_id_pre),
    #               post_pt=self._get_pt_pos('synapses_post', ngl_id_post),
    #               ctr_pt=self._get_pt_pos('synapses', ngl_id_ctr)
    #               )
    #     new_datum = self.format_synapse_data( new_pt )

    #     ae_type, ae_id = self.parse_anno_id(anno_id)
    #     self.annotation_client.update_annotation(ae_type, ae_id, new_datum)
    #     self.viewer.update_message('Updated synapse annotation!')

    def _delete_annotation(self, ngl_id):
        self.viewer.update_message('Manual editing of these synapses is not permitted')
        return 

    def format_synapse_data(self, points):
        return {'type':'synapse',
                'pre_pt':{'position':[int(x) for x in points['pre_pt'].point]},
                'ctr_pt':{'position':[int(x) for x in points['ctr_pt'].point]},
                'post_pt':{'position':[int(x) for x in points['post_pt'].point]}}


    def _get_pt_pos(self, ln, ngl_id):
        for anno in self.viewer.state.layers[ln].annotations:
            if anno.id == ngl_id:
                if anno.type == 'point':
                    return anno.point
                elif anno.type == 'line':
                    return anno.pointA
        else:
            return None