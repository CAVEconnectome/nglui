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
    def __init__(self):
        super(Interface, self).__init__()
        self.viewer = neuroglancer.Viewer()
        self.current_state = None
        self._base_state = None
        self.states = {}
        self.segment_source = None
        self.segmet_layer_name = None

    def set_source_url(self, ngl_url):
        self.ngl_url = neuroglancer.set_server_bind_address(ngl_url)

    def load_url(self, url):
        """Load neuroglancer compatible url and updates viewer state

        Attributes:
        url (str): neuroglancer url

        """
        state = neuroglancer.parse_url(url)
        self.viewer.set_state(state)
        self._base_state = copy.deepcopy(self.viewer.state)
        self.segment_layer_name = self.viewer.state.layers[1].type
        return self.viewer

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
                self.segment_source = segmentation_source
                self.segment_layer_name = layer_name
            self._update_state()
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
            self._update_state()
        except Exception as e:
            raise e

    def add_annotation(self, layer_name, annotation, color):
        """Add annotations layer to viewer instance, the type is specified.

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            layer_type (str): can be: 'points, ellipse or line' only
        """
        if layer_name is None:
            layer_name = 'Annotations'
        try:
            with self.viewer.txn() as s:
                s.layers[layer_name] = neuroglancer.AnnotationLayer(
                    annotations=annotation,
                    annotation_color=color)
            self.current_state = self.viewer.state
        except Exception as e:
            raise e

    def add_action(self, action_name, key_command, method):
        """Add an action that is linked to an external method via key bindings

        Attributes:
            action_name (str): name of action.
            key_command (str): key bindings that provoke action
            method (method): method that is called when key bindings are pressed
        """
        self.viewer.actions.add(action_name, method)
        with self.viewer.config_state.txn() as s:
            s.input_event_bindings.viewer[key_command] = action_name
        return self.viewer.config_state

    @property
    def state(self):
        return self._base_state

    def _update_state(self):
        self._base_state = self.viewer.state
        return self.viewer

    def add_state(self, state_name):
        self.states[state_name] = self.viewer.state
        return self.states

    def set_state(self, state_name):
        return self.viewer.set_state(self.states[state_name])

    def remove_state(self, state_name):
        del self.states[state_name]
        return self.states

    def update_message(self, message):
        with self.viewer.config_state.txn() as s:
            if message is not None:
                s.status_messages['status'] = message

    def load_state(self, state_name, state):
        """Load neuroglancer json state to current ngl viewer

        Attributes:
            state_name (str): name of state that is added to dict.
            state (json): neuroglancer formated json state
        """
        with open(state, 'r') as f:
            self.state = neuroglancer.decode_json(f.read())
        self.viewer.set_state(self.state)
        self.states[state_name] = state
        self._update_state()
        return self.viewer

    def save_json(self, file_name):
        try:
            current_state = self.viewer.state.to_json()
            timestr = time.strftime("%Y%m%d-%H%M%S")
            with open('{}_{}.json'.format(file_name, timestr), 'w') as outfile:
                json.dump(current_state, outfile)
        except Exception as e:
            print(e)
            self._update_message(e)
        url = neuroglancer.to_url(current_state)
        return urllib.unquote(url).decode('utf8')

    def show(self):
        webbrowser.open_new_tab(self.viewer.get_viewer_url())
