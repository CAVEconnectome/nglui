import neuroglancer
from collections import OrderedDict
from neuroglancer_annotation_ui import annotation
from neuroglancer_annotation_ui.extension_core import AnnotationExtensionBase
from inspect import getmembers, ismethod
from numpy import issubdtype, integer
import copy
import json
import os

base_dir=os.path.dirname(__file__)
with open(base_dir+"/data/default_key_bindings.json") as fid:
    default_key_bindings = json.load(fid)

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
        """
        try:
            with self.txn() as s:
                s.layers[layer_name] = neuroglancer.SegmentationLayer(
                    source=segmentation_source)
                self.segment_layer_name = layer_name
                self.segment_source = segmentation_source
        except Exception as e:
            raise e
        self.set_view_options(segmentation_layer=layer_name,
                      show_slices=self.state.showSlices,
                      layout=self.state.layout.type,
                      orthographic_projection=self.state.layout.orthographic_projection,
                      show_axis_lines=self.state.showAxisLines,
                      show_scale_bar=self.state.showScaleBar)


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
        if layer_name in [l.name for l in self.state.layers]:
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
            layer_name = 'New Annotation'
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
                        aids.remove(anno.id)
                        s.layers[layer_name].annotations.pop(ind)
                        if len(aids) == 0:
                            break
                else:
                    raise Exception
        except Exception:
            self.update_message('Could not remove annotation')

    def update_description(self, layer_id_dict, new_description):
        layer_id_dict = copy.deepcopy(layer_id_dict)
        try:
            with self.txn() as s:
                for layer_name, id_list in layer_id_dict.items():
                    for anno in s.layers[layer_name].annotations:
                        if anno.id in id_list:
                            if anno.description is None:
                                anno.description = new_description
                            else:
                                anno.description = "{}\n{}".format(anno.description, new_description)
                            id_list.remove(anno.id)
                            if len(id_list)==0:
                                break                            
        except:
            self.update_message('Could not update annotations!')


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


    def set_selected_layer(self, layer_name):
        if layer_name in self.layer_names:
            with self.txn() as s:
                s._json_data['selectedLayer'] = OrderedDict(layer=layer_name,visible=True)


    def get_selected_layer( self ):
        state_json = self.state.to_json()
        try:    
            selected_layer = state_json['selectedLayer']['layer']
        except:
            selected_layer = None
        return selected_layer


    def get_annotation( self, layer_name, aid ):
        if self.state.layers[layer_name].type == 'annotation':
            for anno in self.state.layers[layer_name].annotations:
                if anno.id == aid:
                    return anno
            else:
                return None
        else:
            return None

    def get_selected_annotation_id( self ):
        layer_name = self.get_selected_layer()
        try:
            aid = self.state.layers[layer_name]._json_data['selectedAnnotation']
        except:
            aid = None
        return aid


    def select_annotation( self, layer_name, aid):
        if layer_name in self.layer_names:
            with self.txn() as s:
                s.layers[layer_name]._json_data['selectedAnnotation'] = aid
        self.set_selected_layer(layer_name)


    @property
    def layer_names(self):
        return [l.name for l in self.state.layers]


    def add_selected_objects(self, segmentation_layer, oids):
        if issubdtype(type(oids), integer):
            oids = [oids]

        with self.txn() as s:
            for oid in oids:
                s.layers[segmentation_layer].segments.add(oid)


    def get_mouse_coordinates(self, s):
        pos = s.mouse_voxel_coordinates
        if (pos is None) or ( len(pos)!= 3):
            return None
        else:
            return pos


    def set_position(self, xyz, zoom_factor=None):
        if zoom_factor is None:
            zoom_factor = self.state.navigation.zoom_factor
        with self.txn() as s:
            s.position.voxelCoordinates = xyz
            s.navigation.zoomFactor = zoom_factor


    def set_view_options( self,
                          segmentation_layer=None,
                          show_slices = False,
                          layout='xy-3d',
                          show_axis_lines=True,
                          show_scale_bar=False,
                          orthographic_projection=False,
                          selected_alpha=0.3,
                          not_selected_alpha=0,
                          perspective_alpha=0.8):
        if segmentation_layer is None:
            layers = [l for l in self.state.layers if l.type == 'segmentation']
        else:
            layers = [segmentation_layer]

        with self.txn() as s:
            s.showSlices = show_slices
            s.layout.type = layout
            s.layout.orthographic_projection = orthographic_projection
            s.show_axis_lines = show_axis_lines
            s.show_scale_bar = show_scale_bar
            for ln in layers:
                s.layers[ln].selectedAlpha = selected_alpha
                s.layers[ln].objectAlpha = perspective_alpha
                s.layers[ln].notSelectedAlpha = not_selected_alpha


class AnnotationManager( ):
    def __init__(self,
                 easy_viewer=None,
                 annotation_client=None,
                 global_delete=True,
                 global_cancel=True,
                 global_update=True, 
                 global_reload=True):
        if easy_viewer is None:
            self.viewer = EasyViewer()
            self.viewer.set_view_options()
        else:
            self.viewer = easy_viewer
        self.annotation_client = annotation_client
        self.extensions = {}

        self.key_bindings = copy.copy(default_key_bindings)
        self.extension_layers = {}

        if global_delete is True:
            self.initialize_delete_action()

        if global_cancel is True:
            self.initialize_cancel_action() 

        if global_update is True:
            self.initialize_update_action()

        if global_reload is True:
            self.initialize_reload_action()

    def initialize_delete_action(self, delete_binding=None):
        if delete_binding == None:
            delete_binding = 'backspace'

        if delete_binding not in self.key_bindings:
            self.annotation_rubbish_bin = None
            self.viewer._add_action('Delete annotation (2x to confirm)',
                                    delete_binding,
                                    self.delete_annotation)
            self.key_bindings.append(delete_binding)
        else:
            print('Could not add the delete action due to a key binding conflict.')


    def initialize_cancel_action(self, cancel_binding="shift+keyc"):
        self._add_bound_action(cancel_binding,
                               self.cancel_annotation,
                               'Cancel current annotation')


    def initialize_update_action(self, update_binding='shift+enter'):
        self._add_bound_action(update_binding,
                               self.update_annotation,
                               'Update selected annotation')


    def initialize_reload_action( self, reload_binding='shift+control+keyr'):
        self._add_bound_action(reload_binding,
                               self.reload_all_annotations,
                               'Reload all annotations from server')


    def _add_bound_action(self, binding, method, method_name ):
        if binding not in self.key_bindings:
            self.viewer._add_action(method_name,
                                    binding,
                                    method)
            self.key_bindings.append(binding)
        else:
            print('Could not add method {} due to key binding conflict'.format(method))

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
        for layer in ExtensionClass._defined_layers():
            self.extension_layers[layer] = extension_name

        for method_name, key_command in bindings.items():
            self._add_bound_action(key_command, bound_methods[method_name], method_name)
            # self.viewer._add_action(method_name,
            #                        key_command,
            #                        bound_methods[method_name])
            print("added {}".format(method_name))
            # self.key_bindings.append(key_command)

        if issubclass(ExtensionClass, AnnotationExtensionBase):
            if self.extensions[extension_name].db_tables == 'MUST_BE_CONFIGURED':
                raise Exception('Table map must be configured for an annotation extension')

        if len(self.extensions[extension_name].allowed_layers) > 0:
            self.viewer.set_selected_layer(self.extensions[extension_name].allowed_layers[0])

        pass


    def list_extensions(self):
        return list(self.extensions.keys())


    def validate_extension( self, ExtensionClass ):
        validity = True
        if len( set( self.extension_layers.keys() ).intersection(set(ExtensionClass._defined_layers())) ) > 0:
            print('{} contains layers that conflict with the current ExtensionManager'.format(ExtensionClass))
            print(set( self.extension_layers.keys() ).intersection(set(ExtensionClass._defined_layers())) )
            validity=False
        if len( set(self.key_bindings).intersection(set(ExtensionClass._default_key_bindings().values())) ) > 0:
            print('{} contains key bindings that conflict with the current ExtensionManager'.format(ExtensionClass))
            print(set(self.key_bindings).intersection(set(ExtensionClass._default_key_bindings().values())))
            validity=False
        return validity


    def delete_annotation( self, s):
        """
        A manager for deleting annotations.
        """
        selected_layer = self.viewer.get_selected_layer()
        if (selected_layer is None) or (self.viewer.state.layers[selected_layer].type != 'annotation'):
            self.viewer.update_message('Please select an annotation layer to delete an annotation')
            return

        ngl_id = self.viewer.get_selected_annotation_id()
        if ngl_id is not None:
            delete_confirmed = self.check_rubbish_bin(ngl_id)
        else:
            curr_pos = self.viewer.state.position.voxel_coordinates
            for annotation in self.viewer.state.layers[selected_layer].annotations:
                process_ngl_id = False
                if annotation.type == 'point':
                    if all(annotation.point==curr_pos):
                        process_ngl_id = True
                elif annotation.type == 'line':
                    if all(annotation.pointA==curr_pos) or all(annotation.pointB==curr_pos):
                        process_ngl_id = True
                elif annotation.type == 'ellipsoid':
                    if all(annotation.center==curr_pos):
                        process_ngl_id = True

                if process_ngl_id:
                    ngl_id = annotation.id
                    delete_confirmed = self.check_rubbish_bin( ngl_id )
                    break
            else:
                delete_confirmed = False
                self.viewer.update_message('Nothing to delete! No annotation selected or targeted!')
                return


        if delete_confirmed:
            bound_extension = self.extensions[ self.extension_layers[selected_layer] ]
            try:
                bound_extension._delete_annotation( ngl_id )
            except Exception as err:
               print(err)
               self.viewer.update_message('Extension could not not delete annotation!')
        pass

    def check_rubbish_bin( self, ngl_id ):
        if self.annotation_rubbish_bin is None:
            self.annotation_rubbish_bin = ngl_id
            self.viewer.update_message( 'Confirm deletion!')
            return False
        elif ngl_id == self.annotation_rubbish_bin:
            #Return True and reset rubbish bin
            self.annotation_rubbish_bin = None
            return True
        else:
            #Return False and reset rubbish bin with notice
            self.annotation_rubbish_bin = None
            self.viewer.update_message( 'Canceled deletion')
            return False

    def cancel_annotation( self, s):
        """
        A manager for canceling annotations in media res.
        """
        selected_layer = self.viewer.get_selected_layer()
        if (selected_layer is None) or (self.viewer.state.layers[selected_layer].type != 'annotation'):
            self.viewer.update_message('Please select relevent layer to cancel annotation')
            return
        
        if issubclass(type(self.extensions[self.extension_layers[selected_layer]]),
                      AnnotationExtensionBase):
            self.extensions[self.extension_layers[selected_layer]]._cancel_annotation()
        return

    def update_annotation(self, s):
        """
            Manages updating a selected annotation.
        """ 
        selected_ngl_id = self.viewer.get_selected_annotation_id()
        if selected_ngl_id is not None:
            selected_layer = self.viewer.get_selected_layer()
            self.extensions[self.extension_layers[selected_layer]]._update_annotation(selected_ngl_id)
        else:
            self.viewer.update_message('Please select an annotation to update it.')
            return

    def reload_all_annotations(self, s):
        for ext_name, ext_class in self.extensions.items():
            if issubclass(type(ext_class), AnnotationExtensionBase):
                ext_class._reload_all_annotations()
        self.viewer.update_message('Reloaded all annotations')

