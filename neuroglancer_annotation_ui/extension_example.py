import neuroglancer
from neuroglancer_annotation_ui import annotation
from neuroglancer_annotation_ui.extensible_viewer import check_layer


class SimpleExtension():
    def __init__(self, extensibleViewer, annotationClient ):
        self.data = []
        self.viewer = extensibleViewer
        self.allowed_layers = ['new_annotation_layer']
        self.create_annotation_layer(None)
        self.annotationClient = annotationClient


    def create_annotation_layer(self, s):
        self.viewer.add_annotation_layer('new_annotation_layer')

    @staticmethod
    def default_bindings():
        bindings = {
            'add_point_data': 'shift+keyd',
            }
        return bindings

    @check_layer(None)
    def add_point_data(self, s):
        pt = self.viewer.add_point(s)
        schematized_data = {'data': pt.point.tolist()}
        self.data.append( schematized_data )
        self.annotationClient.post_annotation(schematized_data)