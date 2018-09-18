import neuroglancer
from collections import OrderedDict
from neuroglancer_annotation_ui import annotation
from inspect import getmembers, ismethod
from functools import wraps
import json
import os

base_dir=os.path.dirname(os.path.dirname(__file__))
with open(base_dir+"/data/default_key_bindings.json") as fid:
    default_key_bindings = json.load(fid)

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

class EasyViewer( neuroglancer.Viewer ):
    """
    Extends the neuroglancer Viewer object to make simple operations simple.
    """
    def __init__(self):
        super(EasyViewer, self).__init__()

    def __repr__(self):
        return self.get_viewer_url()

    def _repr_html_(self):
        return '<a href="%s" target="_blank">Viewer</a>' % self.get_viewer_url()

    def set_source_url(self, ngl_url):
        self.ngl_url = neuroglancer.set_server_bind_address(ngl_url)

    def load_url(self, url):
        """Load neuroglancer compatible url and updates viewer state

        Attributes:
        url (str): neuroglancer url

        """
        state = neuroglancer.parse_url(url)
        self.set_state(state)

    def add_segmentation_layer(self, layer_name, segmentation_source):
        """Add segmentation layer to viewer instance.

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            segment_source (str): source of neuroglancer segment layer
                e.g. :'precomputed://gs://neuroglancer/pinky40_v11/watershed_mst_trimmed_sem_remap'
        """
        try:
            with self.txn() as s:
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
            with self.txn() as s:
                s.layers[layer_name] = neuroglancer.ImageLayer(
                    source=image_source)
        except Exception as e:
            raise e


    def add_annotation_layer(self, layer_name=None, color=None ):
        """Add annotation layer to the viewer instance.

        Attributes:
            layer_name (str): name of layer to be created
        """
        if layer_name is None:
            layer_name = 'new_annotation_layer'
        if layer_name in [l.name for l in self.state.layers]:
            return

        with self.txn() as s:
            s.layers.append( name=layer_name,
                             layer=neuroglancer.AnnotationLayer() )
            if color is not None:
                s.layers[layer_name].annotationColor = color


    def set_annotation_layer_color( self, layer_name, color ):
        """Set the color for the annotation layer

        """
        if layer_name in [l.name for l in self.layers]:
            with self.txn() as s:
                s.layers[layer_name].annotationColor = color
        else:
            pass

    def clear_annotation_layers( self, s ):
        all_layers = neuroglancer.json_wrappers.to_json( self.state.layers )
        new_layers = OrderedDict()
        for layer in all_layers:
            if all_layers[layer]['type'] != 'annotation':
                new_layers[layer] = all_layers[layer]
        
        with self.txn() as s2:
            s2.layers = neuroglancer.viewer_state.Layers(new_layers)

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
            if layer_name not in self.state.layers:
                self.add_annotation_layer(layer_name, color)
            with self.txn() as s:
                for anno in annotation:
                    s.layers[layer_name].annotations.append( anno )
            self.current_state = self.state
        except Exception as e:
            raise e

    def remove_annotation(self, layer_name, aids):
        if isinstance(aids, str):
            aids = [aids]
        try:
            with self.txn() as s:
                for ind, anno in enumerate( s.layers[layer_name].annotations ):
                    if anno.id in aids:
                        s.layers[layer_name].annotations.pop(ind)
                        break
                else:
                    raise Exception
        except:
            self.update_message('Could not remove annotation')

    def update_description(self, layer_id_dict, new_description):
        try:
            with self.txn() as s:
                for layer_name, id_list in layer_id_dict.items():
                    for anno in s.layers[layer_name].annotations:
                        if anno.id in id_list:
                            if anno.description is None:
                                anno.description = new_description
                            else:
                                anno.description = "{}; {}".format(anno.description, new_description)
                            id_list.remove(anno.id)
                            if len(id_list)==0:
                                break                            
        except:
            self.update_message('Could not update annotations!')


    def set_state(self, new_state):
        return self.set_state(new_state)

    @property
    def url(self):
        return self.get_viewer_url()

    def _add_action( self, action_name, key_command, method ):
        self.actions.add( action_name, method )
        with self.config_state.txn() as s:
            s.input_event_bindings.viewer[key_command] = action_name
        return self.config_state

    def update_message(self, message):
        with self.config_state.txn() as s:
            if message is not None:
                s.status_messages['status'] = message

    def get_selected_layer( self ):
        state_json = self.state.to_json()
        try:    
            selected_layer = state_json['selectedLayer']['layer']
        except:
            selected_layer = None
        return selected_layer

    def get_mouse_coordinates(self, s):
        pos = s.mouse_voxel_coordinates
        if (pos is None) or ( len(pos)!= 3):
            return None
        else:  # FIXME: bad hack need to revisit
            return pos


class AnnotationManager( ):
    def __init__(self, easy_viewer=None, annotation_client=None):
        if easy_viewer is None:
            self.viewer = EasyViewer()
        else:
            self.viewer = easy_viewer
        self.annotation_client = annotation_client
        self.extensions = {}

        self.key_bindings = default_key_bindings
        self.extension_layers = []

    def __repr__(self):
        return self.viewer.get_viewer_url()

    def _repr_html_(self):
        return '<a href="%s" target="_blank">Viewer</a>' % self.viewer.get_viewer_url()

    @property
    def url(self):
        return self.viewer.get_viewer_url()

    def add_image_layer(self, layer_name, image_source):
        self.viewer.add_image_layer(layer_name, image_source)

    def add_segmentation_layer(self, layer_name, seg_source):
        self.viewer.add_segmentation_layer(layer_name, seg_source)

    def add_annotation_layer(self, layer_name=None, layer_color=None):
        self.viewer.add_annotation_layer(layer_name, layer_color)

    def add_extension( self, extension_name, ExtensionClass, bindings=None ):
        if not self.validate_extension( ExtensionClass ):
            print("Note: {} was not added to annotation manager!".format(ExtensionClass))
            return

        if bindings is None:
            try:
                bindings = ExtensionClass._default_key_bindings()
            except:
                raise Exception('No bindings provided and no default bindings in {}!'.format(ExtensionClass)) 

        self.extensions[extension_name] = ExtensionClass( self.viewer, self.annotation_client )
        
        bound_methods = {method_row[0]:method_row[1] \
                        for method_row in getmembers(self.extensions[extension_name], ismethod)}

        for method_name, key_command in bindings.items():
            self.viewer._add_action(method_name,
                                   key_command,
                                   bound_methods[method_name])
            self.key_bindings.append(key_command)

    def list_extensions(self):
        return list(self.extensions.keys())

    def validate_extension( self, ExtensionClass ):
        validity = True
        if len( set(self.extension_layers).intersection(set(ExtensionClass._defined_layers())) ) > 0:
            print('{} contains layers that conflict with the current ExtensionManager'.format(ExtensionClass))
            validity=False
        if len( set(self.key_bindings).intersection(set(ExtensionClass._default_key_bindings().values())) ) > 0:
            print('{} contains key bindings that conflict with the current ExtensionManager'.format(ExtensionClass))
            validity=False
        return validity
