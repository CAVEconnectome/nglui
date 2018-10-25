import neuroglancer
import webbrowser
import time
import json
import urllib
import copy


class Interface(neuroglancer.Viewer):
    """Abstraction layer of neuroglancer.Viewer to quickly create
    custom ngl interfaces.
    """

    def __init__(self, ngl_url):
        super(Interface, self).__init__()

        self._ngl_url = ngl_url

        self._current_state = None
        self._base_state = None
        self._states = {}
        self._state = None

    def __repr__(self):
        return self.get_viewer_url()

    @property
    def ngl_url(self):
        return self.ngl_url

    def _repr_html_(self):
        return '<a href="%s" target="_blank">Viewer</a>' % self.get_viewer_url()

    def add_state(self, state_name):
        self.states[state_name] = self.state
        return self.states

    def set_state(self, state_name):
        return self.set_state(self.states[state_name])

    def remove_state(self, state_name):
        del self.states[state_name]
        return self.states

    def update_message(self, message):
        with self.config_state.txn() as s:
            if message is not None:
                s.status_messages['status'] = message

    def load_url(self, url):
        """Load neuroglancer compatible url and updates viewer state

        Attributes:
        url (str): neuroglancer url

        """
        state = neuroglancer.parse_url(url)
        self.set_state(state)
        self.base_state = copy.deepcopy(self.state)

        self.segment_layer_name = self.state.layers[1].type
        return self

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
            # self._update_state()
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

    def add_annotation_layer(self, layer_name, color=None):
        """Add annotation layer to the viewer instance.

        Attributes:
            layer_name (str): name of layer to be created
        """
        if layer_name in [l.name for l in self.state.layers]:
            pass
        else:
            with self.txn() as s:
                s.layers.append(name=layer_name,
                                layer=neuroglancer.AnnotationLayer())
                if color is not None:
                    s.layers[layer_name].annotationColor = color

    def set_annotation_layer_color(self, layer_name, color):
        """Set the color for the annotation layer

        """
        if layer_name in [l.name for l in self.state.layers]:
            with self.txn() as s:
                s.layers[layer_name].annotationColor = color
        else:
            pass

    def add_annotation(self, layer_name, annotation, color=None):
        """Add annotations to a viewer instance, the type is specified.
           If layer name does not exist, add the layer

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            layer_type (str): can be: 'points, ellipse or line' only
        """
        if layer_name is None:
            layer_name = 'Annotations'

        if issubclass(type(annotation),
                      neuroglancer.viewer_state.AnnotationBase):
            annotation = [annotation]

        try:
            self.add_annotation_layer(layer_name, color)
            with self.txn() as s:
                for anno in annotation:
                    s.layers[layer_name].annotations.append(anno)

            # self.current_state = self.state

        except Exception as e:
            raise e

    def add_action(self, action_name, key_command, method):
        """Add an action that is linked to an external method via key bindings

        Attributes:
            action_name (str): name of action.
            key_command (str): key bindings that provoke action
            method (method): method that is called when key bindings are pressed
        """
        self.actions.add(action_name, method)
        with self.config_state.txn() as s:
            s.input_event_bindings.viewer[key_command] = action_name
        return self.config_state

    def load_state(self, state_name, state):
        """Load neuroglancer json state to current ngl viewer

        Attributes:
            state_name (str): name of state that is added to dict.
            state (json): neuroglancer formated json state
        """
        with open(state, 'r') as f:
            self.state = neuroglancer.decode_json(f.read())
        self.set_state(self.state)
        self.states[state_name] = state
        self._update_state()
        return self

    def save_json(self, file_name):
        try:
            current_state = self.state.to_json()
            timestr = time.strftime("%Y%m%d-%H%M%S")
            with open('{}_{}.json'.format(file_name, timestr), 'w') as outfile:
                json.dump(current_state, outfile)
        except Exception as e:
            print(e)
            self._update_message(e)
        url = neuroglancer.to_url(current_state)
        return urllib.unquote(url).decode('utf8')

    def show(self):
        webbrowser.open_new_tab(self.get_viewer_url())
