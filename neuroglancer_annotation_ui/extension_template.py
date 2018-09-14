from neuroglancer_annotation_ui.base import check_layer
from neuroglancer_annotation_ui.ngl_rendering import SchemaRenderer
from emannotationschemas.my_favorite_schema import MyFavoriteSchema

##########
# This part goes in emannotationschemas, not here!
# One could subclass a schema to rewrite the render_rule, however, or just pass it as a raw dict.
class MyFavoriteSchema():
    # All the usual schema functions. Note that the import is not correct
    pt = mm.fields.Nested( BoundSpatialPoint )

    @staticmethod
    def render_rule():
    """
    This dict defines how schema points are mapped into annotations.
    The first keys must correspond to a neuroglancer annotation (e.g. point, line, ellipsoid, bounding_box)
    The value dicts give a layer name (can be remapped later) and the schema field names that populate the annotation.
    Points can be a list of fields, all others come in tuples of field names (generally 2 elements).
    """
        return {'point':{'annotation_layer':['pt']}}
###########


class DummyExtension():
    def __init__(self, easy_viewer, annotation_client=None):
        # Template properties, these are expected to exist under the following names
        self.viewer = easy_viewer   # A viewer of the neuroglancer_annotation_ui.base.EasyViewer class.
        self.annotation_client = annotation_client

        # Establish the rendering rules, can be named anything.
        self.ngl_renderer = SchemaRenderer(MyFavoriteSchema)

        # This is also a good time to produce your core annotation layers. Could be named anything.
        self.create_annotation_layers(None)

    @staticmethod
    def _default_key_bindings():
    """
    Returns the dict of key bindings expected by a neuroglancer viewer.
    See neuroglancer examples for details.
    """
        bindings = {
        'create_annotation_layers':'keyx',
        'add_dummy_annotation':'shift+keyx',
        }
        return bindings

    @staticmethod
    def _defined_layers():
    """
    Returns a list of layer names that are produced and controlled by the extension.
    These can't conflict with other current extensions.
    """
        return ['dummy_widget']
    
    @staticmethod
    def _allowed_layers():
    """
    Returns a list of layers in which key commands work.
    By accepting an argument and returning different lists, this could be different
    for different functions.
    Otherwise, just give a list to @check_layer decorator.
    """
        return ['dummy_widget']

    def create_annotation_layers(self, s):
    """
    Initializing layers is useful, otherwise it's not clear how the user can do much.
    Note that a function with a key binding needs to accept 's' as a parameter.
    """
        for layer in self._defined_layers():
            self.viewer.add_annotation_layer(layer)

    @check_layer()
    def add_dummy_annotation(self, s):
    """
    A function that lets you interact with the data in some way and is bound to a key command.
    The check_layer decorator means that it will only be active if the user has selected a layer in the list
    given. Selected layers can also be used as an input into the data generation.

    This function should do three things, shown below:
    """
        # 1. Retrieve data from viewer and format it for the schema
        xyz_position = self.viewer.get_mouse_coordinates()
        new_data = self.format_annotation( xyz_position ) # Where self.format_annotation formats the data
                                                          # into the schema style

        # 2. Render the new data into neuroglancer following the schema
        self.ngl_renderer(self.viewer, new_data)
        self.viewer.update_message('You annotated a point!')

        # 3. Post the new data to the annotation server, if present.
        if self.annotation_client is not None:
            self.annotation_client.post_annotation( new_data )

    def format_annotation(self, point):
        return {'type':'dummy_annotation',
                'pt':{'position':[int(x) for x in point]}}

