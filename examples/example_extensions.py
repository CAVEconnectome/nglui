import numpy as np
from neuroglancer_annotation_ui.extension_core import check_layer,\
                                                      ExtensionBase
# ExtensionBase is a minimal base class for extensions that does not interact with formatted annotations, but has definitions for other expected functions.
# check_layer is a decorator that enforces

from neuroglancer_annotation_ui.annotation import point_annotation
# Helper function for a neuroglancer annotation object.

# We define layers that the extension manages
ANNOTATION_LAYER = 'annos'
color_map = {ANNOTATION_LAYER: '#CC1111'}

class PointDropperExtension(ExtensionBase):
    def __init__(self, easy_viewer, annotation_client=None):
        super(PointDropperExtension, self).__init__(easy_viewer, None)
        # No annotation client will be needed, so we might as well None this out now.
        # allowed_layers sets which layers are permitted to work with functions decorated by check_layer
        self.allowed_layers = [ANNOTATION_LAYER]
        self.make_layers()
        self.viewer.set_selected_layer(ANNOTATION_LAYER)
        self.viewer.update_message('Press q to drop points and control-s to save points')


    @staticmethod
    def _defined_layers():
        # Defines all layers the extension creates and manages
        return [ANNOTATION_LAYER]


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
            'save_points': 'control+keys',
        }
        return bindings


    @check_layer()
    def place_point(self, s):
        # The first calllback, to be called when the key binding is pressed
        pos = self.viewer.get_mouse_coordinates(s)
        if pos is None:
            self.viewer.update_message('Cursor must be in the image space')
            return

        new_anno = point_annotation(pos)
        self.viewer.add_annotation(layer_name=ANNOTATION_LAYER,
                                   annotation=new_anno)
        self.viewer.update_description({ANNOTATION_LAYER: [new_anno.id]}, 'A new dropped point')

    @check_layer()
    def save_points(self, s):
        # Note that callbacks always get s as an input, even if unused.
        points = []
        anno_layer = self.viewer.state.layers[ANNOTATION_LAYER]
        for anno in anno_layer.annotations:
            points.append(anno.point)

        with open('example_text_data.csv', 'w') as f:
            np.savetxt(f, points, fmt='%i', delimiter=',')

        self.viewer.update_message('Saved points!')