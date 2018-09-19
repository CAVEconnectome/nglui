import re
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

class ExtensionBase():
    """
    Basic class that contains all of the objects that are expected
    by the Extension Manager, but won't actually do anything.
    """
    def __init__(self, easy_viewer, annotation_client=None):
        self.viewer = easy_viewer
        self.annotation_client = annotation_client
        self.ngl_renderer = None
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
    """
    def __init__(self, easy_viewer, annotation_client=None):
        super(AnnotationExtensionBase, self).__init__(easy_viewer, annotation_client)
        self.annotation_df = DataFrame(columns=['ngl_id',
                                                'layer',
                                                'anno_id'])

    def get_anno_id(self, ngl_id):
        return self.annotation_df[self.annotation_df.ngl_id==ngl_id].anno_id.values[0]

    def parse_anno_id(self, anno_id_description ):
        anno_parser = re.search('(?P<type>\w*)_(?P<id>\d.*)$', anno_id_description)
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