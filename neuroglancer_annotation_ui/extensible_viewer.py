import neuroglancer
from collections import OrderedDict
from neuroglancer_annotation_ui import connections 
from neuroglancer_annotation_ui import annotation
from inspect import getmembers, ismethod
from functools import wraps

def check_layer( layer_list ):
    def specific_layer_wrapper( func ):
        @wraps(func)
        def layer_wrapper(self, *args, **kwargs):
            curr_layer = self.viewer.get_selected_layer()
            if curr_layer in layer_list:
                func(self, *args, **kwargs)
            else:
                self.viewer.update_message( 'Select layer \"{}\"" to do that action!'.format(layer_name) )
        return layer_wrapper
    return specific_layer_wrapper 

class ExtensibleViewer( neuroglancer.Viewer ):

    def __init__(self):
        super(ExtensibleViewer, self).__init__()
        self.viewer = neuroglancer.Viewer()
        self.extensions = {}
        self.annotation_server = None

        
    def __repr__(self):
        return self.viewer.get_viewer_url()


    def _repr_html_(self):
        return '<a href="%s" target="_blank">Viewer</a>' % self.viewer.get_viewer_url()


    def set_source_url(self, ngl_url):
        self.ngl_url = neuroglancer.set_server_bind_address(ngl_url)


    def load_url(self, url):
        """Load neuroglancer compatible url and updates viewer state

        Attributes:
        url (str): neuroglancer url

        """
        state = neuroglancer.parse_url(url)
        self.viewer.set_state(state)


    def add_segmentation_layer(self, layer_name, segmentation_source):
        """Add segmentation layer to viewer instance.

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            segment_source (str): source of neuroglancer segment layer
                e.g. :'precomputed://gs://neuroglancer/pinky40_v11/watershed_mst_trimmed_sem_remap'
        """
        try:
            with self.viewer.txn() as s:
                s.layers[layer_name] = neuroglancer.SegmentationLayer(
                    source=segmentation_source)
                self.segment_layer_name = layer_name
                self.segment_source = segmentation_source
        except Exception as e:
            raise e


    def add_image_layer(self, layer_name, image_source):
        """Add segmentation layer to viewer instance.

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            image_source (str): source of neuroglancer image layer
        """
        if layer_name is None:
            layer_name = 'ImageLayer'
        try:
            with self.viewer.txn() as s:
                s.layers[layer_name] = neuroglancer.ImageLayer(
                    source=image_source)
        except Exception as e:
            raise e


    def add_annotation_layer(self, layer_name, color=None ):
        """Add annotation layer to the viewer instance.

        Attributes:
            layer_name (str): name of layer to be created
        """
        if layer_name in [l.name for l in self.viewer.state.layers]:
            pass
        else:
            with self.viewer.txn() as s:
                s.layers.append( name=layer_name,
                                 layer=neuroglancer.AnnotationLayer() )
                if color is not None:
                    s.layers[layer_name].annotationColor = color


    def set_annotation_layer_color( self, layer_name, color ):
        """Set the color for the annotation layer

        """
        if layer_name in [l.name for l in self.viewer.state.layers]:
            with self.viewer.txn() as s:
                s.layers[layer_name].annotationColor = color
        else:
            pass


    def _clear_annotation_layers( self, s ):
        all_layers = neuroglancer.json_wrappers.to_json( self.viewer.state.layers )
        new_layers = OrderedDict()
        for layer in all_layers:
            if all_layers[layer]['type'] != 'annotation':
                new_layers[layer] = all_layers[layer]
        
        with self.viewer.txn() as s:
            s.layers = neuroglancer.viewer_state.Layers(new_layers)


    def clear_all(self, s):
        self._clear_annotation_layers(s)
        self.pre_id = None
        self.post_id = None
        self.post_point = None
        self.pre_point = None
        self.synapse = []


    def add_annotation(self, layer_name, annotation, color=None):
        """Add annotations to a viewer instance, the type is specified.
           If layer name does not exist, add the layer

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            layer_type (str): can be: 'points, ellipse or line' only
        """
        if layer_name is None:
            layer_name = 'Annotations'
        if issubclass(type(annotation), neuroglancer.viewer_state.AnnotationBase):
            annotation = [ annotation ]
        try:
            self.add_annotation_layer(layer_name, color)
            with self.viewer.txn() as s:
                for anno in annotation:
                    s.layers[layer_name].annotations.append( anno )
            self.current_state = self.viewer.state
        except Exception as e:
            raise e

    def _remove_annotation(self, layer_name, aid):
        try:
            with self.viewer.txn() as s:
                for ind, anno in enumerate( s.layers[layer_name].annotations ):
                    if anno.id == aid:
                        s.layers[layer_name].annotations.pop(ind)
                        break
                else:
                    raise Exception
        except:
            self.update_message('Could not remove annotation')

    @property
    def state(self):
        return self.viewer.state

    def set_state(self, new_state):
        return self.viewer.set_state(new_state)

    @property
    def url(self):
        return self.viewer.get_viewer_url()

    def add_action( self, action_name, key_command, method ):
        self.viewer.actions.add( action_name, method )
        with self.viewer.config_state.txn() as s:
            s.input_event_bindings.viewer[key_command] = action_name
        return self.viewer.config_state

    def update_message(self, message):
        with self.viewer.config_state.txn() as s:
            if message is not None:
                s.status_messages['status'] = message


    def add_extension( self, extension_name, ExtensionClass, bindings ):
        self.extensions[extension_name] = ExtensionClass( self )
        bound_methods = {method_row[0]:method_row[1] \
                        for method_row in getmembers(self.extensions[extension_name], ismethod)}
        for method_name, key_command in bindings.items():
            self.add_action(method_name,
                            key_command,
                            bound_methods[method_name])

    def get_selected_layer( self ):
        state_json = self.viewer.state.to_json()
        try:    
            selected_layer = state_json['selectedLayer']['layer']
        except:
            selected_layer = None
        return selected_layer