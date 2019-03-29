import json
from collections import defaultdict
from neuroglancer_annotation_ui.extension_core import check_layer,\
                                                      ExtensionBase
from neuroglancer_annotation_ui.annotation import sphere_annotation

def SphereDropperFactory(radius,
                         layer_names=['annos'],
                         annotation_layer_colors=['#2dfee7'],
                         filename=None):
    color_map = dict(zip(layer_names, annotation_layer_colors))

    class SphereDropperExtension(ExtensionBase):
        def __init__(self, easy_viewer, annotation_client=None):
            super(SphereDropperExtension, self).__init__(easy_viewer, None)
            # No annotation client will be needed, so we might as well None this out now.
            # allowed_layers sets which layers are permitted to work with functions decorated by check_layer

            self.allowed_layers = layer_names
            self.make_layers()
            self._filename = filename
            self.viewer.set_selected_layer(layer_names[0])
            self.viewer.update_message('Press q to drop points and control-s to save points')

        @staticmethod
        def _defined_layers():
            # Defines all layers the extension creates and manages
            return layer_names

        def make_layers(self):
            for layer_name in self._defined_layers():
                self.viewer.add_annotation_layer(layer_name,
                                                 color=color_map[layer_name])


        @staticmethod
        def _default_key_bindings():
            # The functions named as keys in _default_key_bindings get added as callbacks
            # when the extension is added to the Annotation Manager
            bindings = {
                'place_point': 'keyq',
                'dump_points': 'control+keys'
            }
            return bindings

        def add_sphere(self, pos, layer_name):
            new_anno = self.make_sphere_annotation(pos)
            self.viewer.add_annotation(layer_name=layer_name,
                                       annotation=new_anno)

        def make_sphere_annotation(self, pos):
            return sphere_annotation(pos, radius=radius/4, z_multiplier=0.1)

        @check_layer()
        def place_point(self, s):
            # The first calllback, to be called when the key binding is pressed
            pos = self.viewer.get_mouse_coordinates(s)
            if pos is None:
                self.viewer.update_message('Cursor must be in the image space')
                return
            self.add_sphere(pos, self.viewer.get_selected_layer())
            # self.viewer.update_description({ANNOTATION_LAYER: [new_anno.id]}, 'A new dropped point')

        def dump_points(self, s):
            points = defaultdict(list)
            if self._filename is not None:
                layers = self.viewer.state.layers
                for ln in self.viewer.layer_names:
                    lr = layers[ln]
                    if lr.type == 'annotation':
                        for anno in lr.annotations:
                            if anno.type == 'ellipsoid':
                                points[ln].append(anno.center.tolist())
                with open(filename, 'w') as f:
                    json.dump(points, f)
                self.viewer.update_message('Saved points file!')
            else:
                self.viewer.update_message('Did not save, no filename specified!')

    return SphereDropperExtension