from neuroglancer_annotation_ui import annotation, EasyViewer 
from .extension_core import AnnotationExtensionBase, OneShotHolder
from inspect import getmembers, ismethod
from numpy import issubdtype, integer, uint64
import copy 
import json
import os
import re

class EasyViewerInteractive(EasyViewer):
    def __init__(self, state=None):
        super(EasyViewerInteractive, self).__init__(state=state)
        self._expected_ids = OneShotHolder()
        if state is not None:
            self.set_state(state)

    def track_expected_annotations(self):
        self._expected_ids.make_active()

    def ignore_expected_annotations(self):
        self._expected_ids.make_inactive()

    def _add_action( self, action_name, key_command, method ):
        if self.is_interactive:
            self.actions.add( action_name, method )
            with self.config_state.txn() as s:
                s.input_event_bindings.viewer[key_command] = action_name
            return self.config_state

    def set_annotation_one_shot( self, ln_anno_dict, ignore=True):
        '''
        ln_anno_dict is a layer_name to annotation list dict.
        '''
        if ignore is True:
            for ln, annos in ln_anno_dict.items():
                for anno in annos:
                    self._expected_ids.add(anno.id)
        super(EasyViewerInteractive, self).set_annotation_one_shot(ln_anno_dict)

    def add_annotations(self, layer_name, annotations, ignore=True):
        """Add annotations to a viewer instance, the type is specified.
           If layer name does not exist, add the layer
           If ignore is True, add to expected_ngl_ids

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            layer_type (str): can be: 'points, ellipse or line' only
        """
        if ignore:
            for anno in annotations:
                self._expected_ids.add(anno.id)
        super(EasyViewerInteractive, self).add_annotations(layer_name, annotations)

    def add_annotation(self, layer_name, annotation, color=None, ignore=True):
        raise DeprecationWarning('This function is depreciated. Use ''add_annotation'' instead.')
        self.add_annotations(layer_name, annotation, color=color, ignore=ignore)
    
    def remove_annotations(self, layer_name, anno_ids, ignore=True):
        if isinstance(anno_ids, str):
            anno_ids = [anno_ids]
        if ignore:
            for anno_id in anno_ids:
                self._expected_ids.add(anno_id)
        super(EasyViewerInteractive, self).remove_annotations(layer_name, anno_ids)
    
    def remove_annotation(self, layer_name, aids, ignore=True):
        raise DeprecationWarning('This function is depreciated. Use ''remove_annotations'' instead.')
        self.remove_annotations(self, layer_name, aids, ignore=ignore)

    def update_description(self, layer_id_dict, new_description, ignore=True):
        layer_id_dict = copy.deepcopy(layer_id_dict)
        if ignore:
            for layer_name, id_list in layer_id_dict.items():
                for anno in s.layers[layer_name].annotations:
                    if anno.id in id_list:
                            self._expected_ids.add(anno.id)
        super(EasyViewerInteractive, self).update_description(layer_id_dict, new_description)



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

        self._key_bindings = copy.copy(default_key_bindings)
        self.extension_layers = {}

        self._watched_segmentation_layer = None
        self._selected_segments = frozenset()
        self.viewer.shared_state.add_changed_callback(
            lambda: self.viewer.defer_callback(self.on_selection_change))

        if global_delete is True:
            self.initialize_delete_action()

        if global_cancel is True:
            self.initialize_cancel_action() 

        if global_update is True:
            self.initialize_update_action()

        if global_reload is True:
            self.initialize_reload_action()

    @property
    def key_bindings(self):
        return copy.copy(self._key_bindings)

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


    def initialize_cancel_action(self, cancel_binding="escape"):
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

    def add_layers(self, image_layers={}, segmentation_layers={}, annotation_layers={}, resolution=None):
        self.viewer.add_layers(image_layers, segmentation_layers, annotation_layers, resolution)

    def add_image_layer(self, layer_name, image_source):
        self.viewer.add_image_layer(layer_name, image_source)


    def add_segmentation_layer(self, layer_name, segmentation_source, watched=False):
        self.viewer.add_segmentation_layer(layer_name, segmentation_source)
        if watched:
            self.watched_segmentation_layer = layer_name

    def add_annotation_layer(self, layer_name=None, layer_color=None, linked_annotation_layer=None):
        self.viewer.add_annotation_layer(layer_name, layer_color, linked_annotation_layer=linked_annotation_layer)


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
            #print("added {}".format(method_name))

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

    @property
    def watched_segmentation_layer(self):
        return copy.copy(self._watched_segmentation_layer)
    
    @watched_segmentation_layer.setter
    def watched_segmentation_layer(self, watched_layer):
        if watched_layer in [l.name for l in self.viewer.state.layers if l.type == 'segmentation']:
            self._watched_segmentation_layer=watched_layer

    def on_selection_change(self):
        if self.watched_segmentation_layer in self.viewer.layer_names:
            curr_segments = self.viewer.state.layers[self.watched_segmentation_layer].segments
            if curr_segments != self._selected_segments:
                added_ids = list(curr_segments.difference(self._selected_segments))
                removed_ids = list(self._selected_segments.difference(curr_segments))
                for _,ext in self.extensions.items():
                    #try:
                    ext._on_selection_change(added_ids, removed_ids)
                    # except Exception as e:
                    #     print(e)
                    #     continue
                self._selected_segments = curr_segments

    @staticmethod
    def set_static_content_source(url=default_static_content_source):
        set_static_content_source(url)



annotation_function_map = {'point': annotation.point_annotation,
                           'line': annotation.line_annotation,
                           'ellipsoid': annotation.ellipsoid_annotation,
                           'sphere': annotation.sphere_annotation,
                           'bounding_box': annotation.bounding_box_annotation}

class SchemaRenderer():
    def __init__(self, EMSchema, render_rule=None):
        if render_rule is None:
            self.render_rule = RenderRule(EMSchema)
            # Todo: Introduce a default point render rule
        else:
            self.render_rule = RenderRule(EMSchema, render_rule=render_rule)

        self.apply_description_rule = self.render_rule.make_description_rule()

        self.annotations = {}
        self.render_functions = self.render_rule.generate_processors()

        self.reset_annotations()


    def __call__(self, 
                 viewer,
                 data,
                 anno_id=None,
                 layermap=None,
                 colormap=None,
                 replace_annotations=None):
        viewer_ids = self.render_data(viewer, data, anno_id=anno_id, layermap=layermap, colormap=colormap, replace_annotations=replace_annotations)
        return viewer_ids

    def render_data(self,
                    viewer,
                    data,
                    anno_id=None,
                    layermap=None,
                    colormap=None,
                    replace_annotations=None ):
        """
        Takes a formatted data point and returns annotation layers based on the schema's RenderRule
        """
        if layermap is None:
            layermap = {layer:layer for layer in self.all_layers() }

        self.apply_render_rules(data, anno_id=anno_id)
        viewer_ids = self.send_annotations_to_viewer(viewer, layermap=layermap, colormap=colormap)
        self.apply_description_rule(data, viewer_ids, viewer)

        if replace_annotations is not None:
            for layer, ngl_id in replace_annotations.items():
                viewer.remove_annotation(layer, ngl_id)
        self.reset_annotations()
        return viewer_ids

    def apply_render_rules(self, data, anno_id=None):
        for func in self.render_functions:
            func(self, data, anno_id=anno_id)

    def send_annotations_to_viewer(self, viewer, layermap=None, colormap=None):
        if colormap is None:
            colormap={layermap[layer]:None for layer in self.annotations}
            
        viewer_ids = defaultdict(list)
        for layer, anno_list in self.annotations.items():
            nl = layermap[layer]
            for anno in anno_list:
                viewer.add_annotation(nl,anno,color=colormap[nl])
                viewer_ids[nl].append(anno.id)

        return viewer_ids

    def all_fields(self):
        return self.render_rule.fields

    def all_layers(self):
        return self.render_rule.layers

    def reset_annotations(self):
        self.annotations = {layer:[] for layer in self.all_layers()}


class RenderRule():
    def __init__(self, EMSchema, render_rule=None):
        # Should improve the validation here
        self.schema_fields = EMSchema().fields
        if render_rule is None:
            try:
                self.render_rule = EMSchema.render_rule()
            except:
                raise Exception('No render rule defined for {}!'.format(EMSchema))
        else:
            self.render_rule=render_rule

    def make_description_rule(self, spacing_character=':'):
        description_keys = self.render_rule.get('description_field', [])
        if len(description_keys) > 0:
            def dr(data, viewer_ids, viewer):
                added_description = spacing_character.join(data[f] for f in description_keys)
                viewer.update_description(viewer_ids, added_description)
        else:
            def dr(data, viewer_ids, viewer):
                pass
        return dr

    @property
    def layers( self ):
        all_layers = set()
        for anno_type, type_rule in self.render_rule.items():
            if anno_type == 'description_field':
                continue
            for layer in type_rule.keys():
                all_layers.add(layer)
        return list(all_layers)

    @property
    def fields( self ):
        all_fields = set()
        for anno_type, type_rule in self.render_rule.items():
            if anno_type == 'description_field':
                continue
            for _, rule_list in type_rule.items():
                for rule in rule_list:
                    for f in [*rule]:
                        all_fields.add(f)
        return list(all_fields)

    def generate_processors( self ):
        annotation_processor_list = []
        for anno_type in self.render_rule.keys():
            if anno_type == 'description_field':
                continue
            annotation_processor_list.append(
                self._annotation_processor_factory(anno_type, annotation_function_map[anno_type]))
        return annotation_processor_list

    def _annotation_processor_factory(self, anno_type, annotation_function):
        if anno_type in self.render_rule:
            rule_category = self.render_rule[anno_type]
            def annotation_processor(ngr, data, anno_id=None):
                for layer, rule_list in rule_category.items():
                    for rule in rule_list:
                        if isinstance(rule,str):
                            rule_fields = [rule]
                        else:
                            rule_fields = [*rule]
                        anno_args = []
                        for field in rule_fields:
                            if field in self.schema_fields:
                                if isinstance(self.schema_fields[field], Nested):
                                    anno_args.append( data[field]['position'] )
                                else:
                                    anno_args.append( data[field] )
                            else:
                                anno_args.append(field)
                        ngr.annotations[layer].append(
                            annotation_function(*anno_args, description=anno_id))
        else:
            def annotation_processor(ngr, data, anno_id=None):
                return
        return annotation_processor


    @classmethod
    def default_render_rule(EMSchema):
        render_rule = {'point':{'annotations':[]}}
        schema_fields = EMSchema().fields
        for field_name, field in schema_fields.items():
            if issubclass(type(field), SpatialPoint):
                render_rule['point']['annotations'].append(field_name)
            elif isinstance(type(field), Nested):
                is_spatial = [issubtype(subfield) for _x, subfield in field.nested._declared_fields.items()]
                if any(is_spatial):
                    render_rule['point']['annotations'].append(field_name)
        return cls(render_rule)
