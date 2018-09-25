import re
from collections import defaultdict
from neuroglancer_annotation_ui.annotation import point_annotation
from functools import wraps
from pandas import DataFrame

def check_layer(allowed_layer_key=None):
    def specific_layer_wrapper( func ):
        @wraps(func)
        def layer_wrapper(self, *args, **kwargs):
            if allowed_layer_key == None:
                allowed_layers = self.allowed_layers
            else:
                allowed_layers = self.allowed_layers[allowed_layer_key]

            curr_layer = self.viewer.get_selected_layer()
            if curr_layer in allowed_layers:
                func(self, *args, **kwargs)
            else:
                self.viewer.update_message( 'Select layer from amongst \"{}\"" to do that action!'.format(allowed_layers) )
        return layer_wrapper
    return specific_layer_wrapper


class PointHolder():
    def __init__(self, viewer, pt_types=None, trigger=None, layer_dict=None):
        if pt_types is not None:
            self.points = {k:None for k in pt_types}
            self.trigger = trigger
            self.layer_dict = layer_dict
            self.viewer = viewer
            if trigger not in pt_types:
                raise Exception
        else:
            self.points = {}
            self.trigger = ''
            self.layer_dict = {}
            self.viewer = viewer 


    def __call__(self):
        return self.points

    def reset_points( self, pts_to_reset=None):
        if pts_to_reset is None:
            pts_to_reset = list(self.points.keys())
        for pt_type in pts_to_reset:
            if pt_type is not self.trigger:
                self.viewer.remove_annotation(self.layer_dict[pt_type],
                                              self.points[pt_type].id
                                              )
            self.points[pt_type] = None

    def _make_point( self, pos):
        if pos is not None:
            return point_annotation(pos)
        else:
            return None

    def update_point( self, pos, pt_type, message_type=None):
        if pt_type == self.trigger:
            if any([v==None for k, v in self.points.items() if k != self.trigger]):
                self.viewer.update_message('Cannot finish annotation until all other points are set')
                return False

        if message_type is None:
            message_type = 'annotation'

        if self.points[pt_type] is None:
            message = 'Assigned {}'.format(message_type)
        else:
            self.viewer.remove_annotation(self.layer_dict[pt_type],
                                     self.points[pt_type].id)
            message = 'Re-assigned {}'.format(message_type)

        self.points[pt_type] = self._make_point(pos)
        if pt_type != self.trigger:
            self.viewer.add_annotation(self.layer_dict[pt_type], [self.points[pt_type]])
        self.viewer.update_message(message)
        return self.points[pt_type].id


class ExtensionBase():
    """
    Basic class that contains all of the objects that are expected
    by the Extension Manager, but won't actually do anything.
    """
    def __init__(self, easy_viewer, annotation_client=None):
        self.viewer = easy_viewer
        self.annotation_client = annotation_client
        self.allowed_layers = []

    @staticmethod
    def _default_key_bindings():
        bindings = {}
        return bindings

    @staticmethod
    def _defined_layers():
        return []

    @check_layer()
    def _delete_annotation( ngl_id ):
        pass


class AnnotationExtensionBase(ExtensionBase):
    """
    Adds framework to interact with a mapping between layer, ngl_id, and anno_id on the
    annotation engine side.
    Note that `db_tables` must be configured. This object is intended to relate annotations to
    database tables, and can be set through the set_db_tables class method.
    """
    def __init__(self, easy_viewer, annotation_client=None):
        super(AnnotationExtensionBase, self).__init__(easy_viewer, annotation_client)

        self.db_tables = 'MUST_BE_CONFIGURED'

        self.render_map = dict()
        self.ngl_renderer = defaultdict(lambda *args, **kwargs: self.viewer.update_message('No renderer configured'))

        self.annotation_df = DataFrame(columns=['ngl_id',
                                                'layer',
                                                'anno_id'])

        self.points = PointHolder(viewer=easy_viewer, pt_types=None)

        self.linked_annotations = dict()

    @classmethod
    def set_db_tables(cls, class_name, db_tables):
        class ClassNew(cls):
            def __init__(self, easy_viewer, annotation_client=None):
                super(ClassNew, self).__init__(easy_viewer, annotation_client)
                self.db_tables = db_tables
        ClassNew.__name__ = class_name
        ClassNew.__qualname__ = class_name
        return ClassNew

    def get_anno_id(self, ngl_id):
        return self.annotation_df[self.annotation_df.ngl_id==ngl_id].anno_id.values[0]

    def parse_anno_id(self, anno_id_description, regex_type_match=None ):
        if regex_type_match is None:
            anno_parser = re.search('(?P<type>\w*)_(?P<id>\d.*)$', anno_id_description)
        else:
            anno_parser = re.search('(?P<type>{})_(?P<id>\d.*)'.format(regex_type_match), anno_id_description)

        ae_type = anno_parser.groupdict()['type']
        ae_id = anno_parser.groupdict()['id']
        return ae_type, ae_id

    def _remove_map_id(self, anno_id):
        self.annotation_df.drop(index=self.annotation_df[self.annotation_df.anno_id==anno_id].index,
                                inplace=True)

    def _update_map_id(self, viewer_ids, id_description ):
        for layer, id_list in viewer_ids.items():
            for ngl_id in id_list:
                self.annotation_df = self.annotation_df.append({'ngl_id': ngl_id,
                                                                'layer': layer,
                                                                'anno_id': id_description
                                                                },
                                                                ignore_index=True)

    def _annotation_filtered_iterrows(self, anno_id=None, ngl_id=None, layer=None):
        arg1 = True if anno_id is None else (self.annotation_df.anno_id == anno_id)
        arg2 = True if ngl_id is None else (self.annotation_df.ngl_id == ngl_id)
        arg3 = True if layer is None else (self.annotation_df.layer == layer)
        return self.annotation_df[arg1 & arg2 & arg3].iterrows()

    def _delete_annotation( self, base_ngl_id ):
        if base_ngl_id in self.linked_annotations:
            rel_ngl_ids = self.linked_annotations[base_ngl_id]
        else:
            rel_ngl_ids = [base_ngl_id]
            
        for ngl_id in rel_ngl_ids:
            anno_id = self.get_anno_id(ngl_id)
            ae_type, ae_id = self.parse_anno_id(anno_id)
            try:
                self.annotation_client.delete_annotation(annotation_type=ae_type,
                                                         oid=ae_id)
                if ngl_id in self.linked_annotations:
                    del self.linked_annotations[ngl_id]
            except:
                self.viewer.update_message('Annotation client could not delete annotation!')

            self.remove_associated_annotations(anno_id)
            self.viewer.update_message('Successfully deleted annotation')


    def _cancel_annotation( self ):
        self.points.reset_points()
        self.viewer.update_message('Canceled annotation! No active annotations.')


    def _update_annotation(self, ngl_id, new_annotation):
        self.viewer.update_message('No update function is configured for this extension')


    def load_annotation_by_aid(self, a_type, a_id, render_type):
        if self.annotation_client is not None:
            anno_dat = self.annotation_client.get(a_type, a_id)
            viewer_ids = self.ngl_renderer[render_type](self.viewer,
                                                        anno_dat,
                                                        layermap=self.render_map)
            annotation_id = anno_dat['id']
            self._update_map_id(viewer_ids, '{}_{}'.format(a_type, a_id)) 


    def remove_associated_annotations(self, anno_id ):
        for _, row in self._annotation_filtered_iterrows(anno_id=anno_id):
            self.viewer.remove_annotation(row['layer'], row['ngl_id'])
        self._remove_map_id(anno_id)


    def reload_annotation(self, ngl_id):
        anno_id = self.get_anno_id(ngl_id)
        a_type, a_id = self.parse_anno_id(anno_id)
        self.load_annotation_by_aid(a_type, a_id)
        self.remove_associated_annotations(a_id)


    def _post_data(self, data, table_name):
        response = self.annotation_client.post_annotation(self.db_tables[table_name],
                                                          data)
        return response


    def render_and_post_annotation(self, data_formatter, render_name, anno_layer_dict, table_name):
        data = data_formatter( self.points() )
        viewer_ids = self.ngl_renderer[render_name](self.viewer,
                                                    data,
                                                    layermap=anno_layer_dict)

        if self.annotation_client is not None:
            aid = self._post_data([data], table_name)
            id_description = '{}_{}'.format(self.db_tables[table_name], aid[0])
            self.viewer.update_description(viewer_ids, id_description)
            self._update_map_id(viewer_ids, id_description)

        return viewer_ids

