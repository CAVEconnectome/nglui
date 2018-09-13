import neuroglancer
from collections import OrderedDict
from neuroglancer_annotation_ui import connections 
from neuroglancer_annotation_ui import annotation
from inspect import getmembers, ismethod
from functools import wraps
from collections import defaultdict
import json
import os

base_dir=os.path.dirname(os.path.dirname(__file__))
with open(base_dir+"/data/default_key_bindings.json") as fid:
    default_key_bindings = json.load(fid)

def check_layer( context_specifier=None ):
    def specific_layer_wrapper( func ):
        @wraps(func)
        def layer_wrapper(self, *args, **kwargs):
            if context_specier is not None:
                layer_list = self.allowed_layers[context_specifier]
            else:
                layer_list = self.allowed_layers
            curr_layer = self.get_selected_layer()
            if curr_layer in layer_list:
                func(self, *args, **kwargs)
            else:
                self.update_message( 'Select layer in \"{}\"" to do that action!'.format(layer_list) )
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

    def _remove_annotation(self, layer_name, aid):
        try:
            with self.txn() as s:
                for ind, anno in enumerate( s.layers[layer_name].annotations ):
                    if anno.id == aid:
                        s.layers[layer_name].annotations.pop(ind)
                        break
                else:
                    raise Exception
        except:
            self.update_message('Could not remove annotation')

    def set_state(self, new_state):
        return self.set_state(new_state)

    @property
    def url(self):
        return self.get_viewer_url()

    def add_action( self, action_name, key_command, method ):
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
        return s.mouse_voxel_coordinates

    def add_point(self, s, description=None ):
        pos = s.mouse_voxel_coordinates
        if pos is None:
            return
        if len(pos) is 3:  # FIXME: bad hack need to revisit
            id = neuroglancer.random_token.make_random_token()
            point = annotation.point_annotation(pos, id, description)
            return point
        else:
            return

    def add_line(self, a, b, description=None):
        id = neuroglancer.random_token.make_random_token()
        line = annotation.line_annotation(a, b, id)
        return line


class ViewerManager():
    def __init__(self, easy_viewer, annotation_client=None):
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

    def add_extension( self, extension_name, ExtensionClass, bindings ):
        if not self.validate_extension( ExtensionClass ):
            print("Note: {} was not added to ExtensionManager!".format(ExtensionClass))
            return

        self.extensions[extension_name] = ExtensionClass( self.viewer, self.annotation_client )
        
        bound_methods = {method_row[0]:method_row[1] \
                        for method_row in getmembers(self.extensions[extension_name], ismethod)}

        for method_name, key_command in bindings.items():
            self.viewer.add_action(method_name,
                                   key_command,
                                   bound_methods[method_name])
            self.key_bindings.append(key_command)

        for layer in self.ExtensionClass.extension_layers():
            self.extension_layers.append(layer)

    def list_extensions(self):
        return list(self.extensions.keys())

    def validate_extension( self, ExtensionClass ):
        validity = True
        if len( set(self.extension_layers).intersection(set(ExtensionClass.extension_layers)) ) > 0:
            print('{} contains layers that conflict with the current ExtensionManager'.format(ExtensionClass))
            validity=False
        if len( set(self.key_bindings).intersection(set(ExtensionClass.default_key_bindings.values())) ) > 0:
            print('{} contains key bindings that conflict with the current ExtensionManager'.format(ExtensionClass))
            validity=False
        return validity


class NeuroglancerRenderer():
    def __init__(self, EMSchema, render_rule=None, layer_map=None):
        self.schema = EMSchema()
        if render_rule is None:
            self.render_rule = RenderRule( self.schema.render_rule )
            # Todo: Introduce a default point render rule
        else:
            self.render_rule = RenderRule( render_rule )
        if layer_map is not None:
            self.render_rule.apply_layer_map(layer_map)

        self.reset_annotations()

    def render_data(self, viewer, data, anno_id=None, colormap=None ):
        """
        Takes a formatted data point and returns annotation layers based on the schema's RenderRule
        """
        self.apply(data, anno_id=anno_id)
        self.send_annotations_to_viewer(viewer, colormap=colormap)
        self.reset_annotations()

    def send_annotations_to_viewer(self, viewer, colormap=None):
        if colormap is None:
            colormap={layer:None for layer in self.annotations}
        for layer, anno_list in self.annotations.items():
            for anno in anno_list:
                viewer.add_annotation(layer,anno,color=colormap[layer])

    def all_fields(self):
        return self.render_rule.fields

    def all_layers(self):
        return self.render_rule.layers

    def reset_annotations(self):
        self.annotations = {layer:[] for layer in self.all_layers()}

    def apply(self, data, anno_id=None):
        self._process_points(data)
        self._process_lines(data)
        self._process_ellipsoids(data)
        self._process_bounding_boxes(data)

    def _process_points(self, data, anno_id=None ):
        for layer, rule_list in self.render_rule.points.items():
            for point in rule_list:
                xyz = data[point]['position']
                self.annotations[layer].append(
                    annotation.point_annotation(xyz, description=anno_id))

    def _process_lines(self, data, anno_id=None ):
        for layer, rule_list in self.render_rule.lines.items():
            for line in rule_list:
                xyzA = data[line[0]]['position']
                xyzB = data[line[1]]['position']
                self.annotations[layer].append(
                    annotation.line_annotation(xyzA, xyzB, description=anno_id))

    def _process_ellipsoids(self, data, anno_id=None):
        for layer, rule_list in self.render_rule.ellipsoids.items():
            for ellipsoid in rule_list:
                center = data[ellipsoid[0]]['position']
                radii = data[ellipsoid[1]]
                self.annotations[layer].append(
                    annotation.ellipsoid_annotation(center,radii,description=anno_id))

    def _process_bounding_boxes(self, data, anno_id=None):
        for layer, rule_list in self.render_rule.bounding_boxes.items():
            for bounding_box in rule_list:
                xyzA = data[bounding_box[0]]['position']
                xyzB = data[bounding_box[1]]['position']
                self.annotations[layer].append(
                    annotation.bounding_box(xyzA,xyzB,description=anno_id))


class RenderRule():
    def __init__(self, render_rules, layer_map=None):
        # Should improve the validation here
        self.points = defaultdict(list)
        if 'points' in render_rules:
            for layer, point_list in render_rules['points'].items():
                for point in point_list:
                    self.points[layer].append(point)

        self.lines = defaultdict(list)
        if 'lines' in render_rules:
            for layer, line_list in render_rules['lines'].items():
                for line in line_list:
                    if len(line) == 2:
                        self.lines[layer].append(line)

        self.ellipsoids = defaultdict(list)
        if 'ellipsoids' in render_rules:
            for layer, ellipsoid_list in render_rules['ellipsoids'].items():
                for ellipsoid in ellipsoid_list:
                    if len(ellipsoid) == 2:
                        self.ellipsoids[layer].append(ellipsoid)

        self.bounding_boxes = defaultdict(list)
        if 'bounding_boxes' in render_rules:
            for layer, bounding_box_list in render_rules['bounding_boxes'].items():
                for bounding_box in bounding_box_list:
                    if len(bounding_box) == 2:
                        self.bounding_boxes[layer].append(bounding_box)

    @classmethod
    def default_render_rule(EMSchema):
        #to implement
        return

    @property
    def layers( self ):
        all_layers = set()
        for layer in self.points:
            all_layers.add(layer)
        for layer in self.lines:
            all_layers.add(layer)
        for layer in self.ellipsoids:
            all_layers.add(layer)
        for layer in self.bounding_boxes:
            all_layers.add(layer)
        return list(all_layers)

    @property
    def fields( self ):
        all_fields = set()
        for layer, point_list in self.points.items():
            for point in point_list:
                all_fields.add(point)
        for layer, line_list in self.lines.items():
            for line in line_list:
                all_fields.add(line[0])
                all_fields.add(line[1])
        for layer, ell_list in self.ellipsoids.items():
            for ell in ell_list:
                all_fields.add(ell[0])
                all_fields.add(ell[1])
        for layer, bb_list in self.bounding_boxes.items():
            for bb in bb_list:
                all_fields.add(bb[0])
                all_fields.add(bb[1])
        return list(all_fields)

    def apply_layer_map(self, layer_map):
        self.points = self._apply_layer_map_to_anno_type(self.points, layer_map)
        self.lines = self._apply_layer_map_to_anno_type(self.lines, layer_map)
        self.ellipsoids = self._apply_layer_map_to_anno_type(self.ellipsoids, layer_map)
        self.bounding_boxes = self._apply_layer_map_to_anno_type(self.bounding_boxes, layer_map)

    def _apply_layer_map_to_anno_type(self, anno_type, layer_map):
        new_type = dict()
        for layer in self.points:
            new_type[layer_map[layer]] = anno_type[layer]
        return new_type
