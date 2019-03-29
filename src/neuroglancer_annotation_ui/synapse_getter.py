from neuroglancer_annotation_ui.extension_core import check_layer, \
                                                      AnnotationExtensionBase, \
                                                      OrderedPointHolder
from neuroglancer_annotation_ui.ngl_rendering import SchemaRenderer
from neuroglancer_annotation_ui.annotation import line_annotation, point_annotation
from emannotationschemas.synapse import SynapseSchema
from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink
from collections import defaultdict

PRE_LAYER, POST_LAYER, SYN_LAYER = 'synapses_pre', 'synapses_post', 'synapses'
PRE_PT, POST_PT, SYN_PT = 'pre_pt', 'post_pt', 'ctr_pt' 
PRE_ANNO, POST_ANNO, SYN_ANNO = 'pre', 'post', 'syn'

SYNAPSE_RENDERER = 'synapse'

synapse_render_rule = {'line': {PRE_ANNO: [(PRE_PT, SYN_PT)],
                                POST_ANNO: [(POST_PT, SYN_PT)]},
                       'point': {SYN_ANNO: [SYN_PT]},
                       }

# Sets the order of points created by clicking
POINT_TYPE_ORDER = {0: PRE_PT,
                    1: POST_PT,
                    2: SYN_PT,
                    }

# Assigns names in the message field on creation of point types
MESSAGE_DICT = {PRE_PT: 'presynaptic point',
                POST_PT: 'postsynaptic point',
                SYN_PT: 'synapse',
                }

# Assigns rendered annotation names (from render_rule) to layers
ANNO_LAYER_DICT = {PRE_ANNO: PRE_LAYER,
                   POST_ANNO: POST_LAYER,
                   SYN_ANNO: SYN_LAYER,
                   }

# Assigns point types to different layers
POINT_LAYER_MAP = {PRE_PT: PRE_LAYER,
                   POST_PT: POST_LAYER,
                   SYN_PT: SYN_LAYER,
                   }

# Assigns colors to layers
DEFAULT_COLOR_MAP = {PRE_LAYER: '#ff0000',
                     POST_LAYER: '#00ffff',
                     }


def SynapseGetterFactory(table_name, db_config):
    """
    Builds an extension that retrieves synapses from a given data table.
    """
    database_uri = db_config.get('sqlalchemy_database_uri', None)
    materialization_version = db_config.get('materialization_version', None)
    annotation_endpoint = db_config.get('annotation_endpoint', None)
    dataset_name = db_config.get('dataset_name', None)

    class SynapseGetterExtension(AnnotationExtensionBase):
        def __init__(self, easy_viewer, annotation_client=None):
            super(SynapseGetterExtension, self).__init__(easy_viewer, None)
            self.ngl_renderer = {SYNAPSE_RENDERER: SchemaRenderer(SynapseSchema, synapse_render_rule)}
            self.allowed_layers = []
            self.db_tables = {}
            self.color_map = DEFAULT_COLOR_MAP
            self.point_layer_dict = POINT_LAYER_MAP
            self.anno_layer_dict = ANNO_LAYER_DICT
            self.selection_mode = 'all'
            self.watched_layer = None
            self.anno_type = 'line'

            self.points = OrderedPointHolder(viewer=self.viewer,
                                             pt_type_dict=POINT_TYPE_ORDER,
                                             trigger=SYN_PT,
                                             layer_dict=self.point_layer_dict,
                                             message_dict=MESSAGE_DICT
                                             )
            self.create_synapse_layers()

            self.dl = AnalysisDataLink(dataset_name=dataset_name,
                                       sqlalchemy_database_uri=database_uri,
                                       materialization_version=materialization_version,
                                       verbose=False,
                                       annotation_endpoint=annotation_endpoint,
                                       )
            self.synapse_table = table_name

        def change_selection_mode(self, s):
            if self.selection_mode == 'all':
                self.selection_mode = 'restricted'
                self.viewer.update_message('Synapses restricted to those between selected neurons')
                self._restricted_synapses()

            elif self.selection_mode == 'restricted':
                self.selection_mode = 'all'
                self.viewer.update_message('Synapse restriction off')
                self.reset_all()


        def reset_all(self):
            self.viewer.clear_annotation_layers(self._defined_layers())
            selected_oids = self.viewer.selected_objects(self.watched_layer)
            selected_oids = [int(oid) for oid in selected_oids]
            self._unrestricted_synapses(selected_oids, [])


        @staticmethod
        def _default_key_bindings():
            bindings = {
                'change_selection_mode': 'shift+keyr',
                }
            return bindings

        @staticmethod
        def _defined_layers():
            return [PRE_LAYER, POST_LAYER]

        def create_synapse_layers(self):
            linked_seg_layer = None
            for layer in self.viewer.state.layers:
                if layer.type == 'segmentation':
                    linked_seg_layer = layer.name
                    break
            else:
                print('No segmentation layer found for linking')
            self.watched_layer = linked_seg_layer
            for layer in self._defined_layers():
                self.viewer.add_annotation_layer(layer,
                                                 color=self.color_map[layer],
                                                 linked_segmentation_layer=linked_seg_layer)


        def _on_selection_change(self, added_ids, removed_ids):
            added_ids = [int(oid) for oid in added_ids]
            removed_ids = [int(oid) for oid in removed_ids]

            if self.selection_mode == 'all':
                self._unrestricted_synapses(added_ids, removed_ids)
            elif self.selection_mode == 'restricted':
                self._restricted_synapses()


        def _unrestricted_synapses(self, added_ids, removed_ids):
            if len(added_ids) > 0:
                new_pre_annos = self._get_presynaptic_synapses(added_ids)
                new_post_annos = self._get_postsynaptic_synapses(added_ids)
                self.viewer.add_annotation_one_shot({PRE_LAYER: new_pre_annos['pre']+new_post_annos['pre'],
                                                     POST_LAYER: new_pre_annos['post']+new_post_annos['post']})
            
            if len(removed_ids)>0:
                self._remove_synapse_annotations(removed_ids)


        def _get_presynaptic_synapses( self, oids ):
            pre_synapses_df = self.dl.query_synapses(self.synapse_table, pre_ids=oids)
            annos = self.synapse_annotations_from_df(pre_synapses_df, linked_side='pre', half_synapse=True)
            return annos


        def _get_postsynaptic_synapses( self, oids ):
            post_synapses_df = self.dl.query_synapses(self.synapse_table, post_ids=oids)
            annos = self.synapse_annotations_from_df(post_synapses_df, linked_side='post', half_synapse=True)
            return annos


        def _remove_synapse_annotations( self, oids):
            self.viewer.remove_annotation_by_linked_oids_one_shot(self._defined_layers(), oids)


        def _restricted_synapses( self ):
            selected_oids = self.viewer.selected_objects(self.watched_layer)
            selected_oids = [int(oid) for oid in selected_oids]

            common_synapses_df = self.dl.query_synapses(self.synapse_table, pre_ids=selected_oids, post_ids=selected_oids)
            annos = self.synapse_annotations_from_df(common_synapses_df, linked_side='both')
            
            self.viewer.set_annotation_one_shot({PRE_LAYER:annos['pre'],
                                                 POST_LAYER:annos['post']})

        def synapse_annotations_from_df( self, syn_df, linked_side, half_synapse=False ):
            annos = {'pre':[], 'post':[]}
            if linked_side == 'pre':
                link_ind = [3]
            elif linked_side == 'post':
                link_ind = [4]
            if linked_side == 'both':
                link_ind = [3,4]

            if half_synapse:
                do_sides = {'pre':False, 'post':False}
                do_sides[linked_side] = True
            else:
                do_sides = {'pre':True, 'post':True}

            for row in syn_df[['pre_pt_position', 'ctr_pt_position', 'post_pt_position', 'pre_pt_root_id', 'post_pt_root_id']].values:
                linked_seg = row[link_ind]
                if do_sides['pre']:
                    if self.anno_type=='line':
                        new_anno = line_annotation(a=row[2], b=row[1], linked_segmentation=linked_seg)
                    elif self.anno_type=='point':
                        new_anno = point_annotation(point=row[1], linked_segmentation=linked_seg)
                    annos['pre'].append( new_anno )
                if do_sides['post']:
                    if self.anno_type=='line':
                        new_anno = line_annotation(a=row[0], b=row[1], linked_segmentation=linked_seg)
                    elif self.anno_type=='point':
                        new_anno = point_annotation(point=row[1], linked_segmentation=linked_seg)
                    annos['post'].append( new_anno )
            return annos

    return SynapseGetterExtension