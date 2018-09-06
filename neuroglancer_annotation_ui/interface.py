import neuroglancer
import webbrowser
import time
import json
import urllib
import copy
from collections import OrderedDict
from neuroglancer_annotation_ui import connections 
from neuroglancer_annotation_ui import annotation


class Interface(neuroglancer.Viewer):
    """Abstraction layer of neuroglancer.Viewer to quickly create
    custom ngl interfaces.
    """
    def __init__(self):
        super(Interface, self).__init__()
        self.viewer = neuroglancer.Viewer()
        self.current_state = None
        self.base_state = None
        self.states = {}
        self.segment_source = None
        self.segmet_layer_name = None

    def __repr__(self):
        return self.viewer.get_viewer_url()

    def _repr_html_(self):
        return '<a href="%s" target="_blank">Viewer</a>' % self.viewer.get_viewer_url()

    def set_source_url(self, ngl_url):
        self.ngl_url = neuroglancer.set_server_bind_address(ngl_url)

    def load_url(self, url):
        """Load neuroglancer compatible url and updates viewer state

        Attributes:
        url (str): neuroglancer url

        """
        state = neuroglancer.parse_url(url)
        self.viewer.set_state(state)
        self.base_state = copy.deepcopy(self.viewer.state)
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


    def add_annotation_layer(self, layer_name, color=None ):
        """Add annotation layer to the viewer instance.

        Attributes:
            layer_name (str): name of layer to be created
        """
        if layer_name in [l.name for l in self.viewer.state.layers]:
            pass
        else:
            with self.viewer.txn() as s:
                s.layers.append( name=layer_name,
                                 layer=neuroglancer.AnnotationLayer() )
                if color is not None:
                    s.layers[layer_name].annotationColor = color

    def set_annotation_layer_color( self, layer_name, color ):
        """Set the color for the annotation layer

        """
        if layer_name in [l.name for l in self.viewer.state.layers]:
            with self.viewer.txn() as s:
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
        if issubclass(type(annotation), neuroglancer.viewer_state.AnnotationBase):
            annotation = [ annotation ]
        try:
            self.add_annotation_layer(layer_name, color)
            with self.viewer.txn() as s:
                for anno in annotation:
                    s.layers[layer_name].annotations.append( anno )
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
        return self.state

    def _update_state(self):
        self.base_state = self.viewer.state
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


class Connector(Interface):
    def __init__(self):
        super(Connector, self).__init__()
        self.data = connections.Connections()
        self.post_id = None
        self.pre_id = None
        self.synapse = []
        self.index = None
        self.annotation_layer_name = None
        self.post_point = None
        self.pre_point = None
        self.pre_lines = []
        self.post_lines = []

    def __repr__(self):
        return self.viewer.get_viewer_url()

    def _repr_html_(self):
        return '<a href="%s" target="_blank">Viewer</a>' % self.viewer.get_viewer_url()

    def select_post_process(self, s):
        if self.post_id is not None:
            self.update_message("Press 'Shift + C' to clear so a new \
            postsynaptic process can be assigned.")
            return
        self.post_id = self.get_segment(s)
        if self.post_id == self.pre_id:
            self.update_message("Cannot assign new postsynaptic process \
            as current presynaptic process.")
            return
        self._update_index()
        self.post_point = self.add_point(s, 2)
        self.update_message("Current postsynaptic process \
         is {}".format(self.post_id))
        self.add_annotation('Post Synaptic Process', [self.post_point], '#ff0000')

    def add_point(self, s, description=None):
        pos = s.mouse_voxel_coordinates
        if pos is None:
            return
        if len(pos) is 3:  # FIXME: bad hack need to revisit
            id = annotation.generate_id()
            point = annotation.point_annotation(pos, id, description)
            return point
        else:
            return

    def add_line(self, a, b, description=None):
        id = annotation.generate_id()
        line = annotation.line_annotation(a, b, id)
        return line

    def add_synapse(self, s):
        if self.post_id is None and self.pre_id is None:
            self.update_message("Pre and Post targets must be defined before \
            adding a synapse!!!")
            return
        self.synapse = self.add_point(s, 1)
        self.annotation_layer_name = 'Synapse'
        pre_line = self.add_line(self.pre_point.point, self.synapse.point)
        post_line = self.add_line(self.post_point.point, self.synapse.point)
        self.pre_lines.append(pre_line)
        self.post_lines.append(post_line)
        self.add_annotation('Post_connection', self.post_lines, '#ff0000')
        self.add_annotation('Pre_connection', self.pre_lines, '#00ff24')
        # append connection to datastruct class...
        self.data.set_active_pair(self.pre_id, self.post_id)
        self.data.add_connection(self.pre_point.point.tolist(),
                                 self.post_point.point.tolist(),
                                 self.synapse.point.tolist())
        self.clear_segment(None)

    def select_pre_process(self, s):
        if self.pre_id is not None:
            self.update_message("Press 'Shift + V' to clear so a new \
            presynaptic process can be assigned.")
            return
        self.pre_id = self.get_segment(s)
        if self.pre_id == self.post_id:
            self.update_message("Cannot assign presynaptic process as current \
             postsynaptic process.")
            self.pre_id = None
            return
        self._update_index()
        self.pre_point = self.add_point(s, 0)
        self.update_message("Current presynaptic is {}".format(self.pre_id))
        self.add_annotation('Pre Synaptic Process', [self.pre_point], '#00ff24')

    def delete_synapse(self, s):
        """ TODO
        Find nearest X,Y,Z point in radius of mouse position and remove index
        from list import scipy.spatial.KDTree ?? for lookup of xzy pos
        """
        self.update_message('Delete key pressed')

    def undo_last_point(self, s):
        try:
            with self.viewer.txn() as s:
                point_layer = s.layers[self.annotation_layer_name]
                point_layer.annotations = point_layer.annotations[:-1]
            del self.data.dataset[self.index]['synapses'][-1]
            self.synapse = self.synapse[:-1]
            self.update_message("Last Synapse removed!!!")
        except Exception as e:
            raise e

    def _update_view(self, pos):
        with self.viewer.txn() as s:
            s.voxel_coordinates = pos

    def clear_segment(self, s):
        self.pre_id = None
        self.post_id = None
        self.post_point = None
        self.pre_point = None
        self._update_index()

    def _update_index(self):
        if self.pre_id and self.post_id:
            self.index = self.data.set_active_pair(self.pre_id, self.post_id)
        return self.index

    def _clear_annotation_layers( self, s ):
        all_layers = neuroglancer.json_wrappers.to_json( self.viewer.state.layers )
        new_layers = OrderedDict()
        for layer in all_layers:
            if all_layers[layer]['type'] != 'annotation':
                new_layers[layer] = all_layers[layer]
        
        with self.viewer.txn() as s:
            s.layers = neuroglancer.viewer_state.Layers(new_layers)


    def clear_all(self, s):
        self._clear_annotation_layers(s)
        self.pre_id = None
        self.post_id = None
        self.post_point = None
        self.pre_point = None
        self._update_index()
        self.synapse = []

    def get_segment(self, s):
        try:
            return s.selected_values[self.segment_layer_name]
        except Exception as e:
            raise e

    def activate_existing_state(self, index):
        if self.pre_id and self.post_id:
            return self.set_state_index(index)
        else:
            return self.viewer.state

    def set_state_index(self, index):
        if index in self.states:
            return self.viewer.set_state(self.states[index])
        else:
            return self.viewer.set_state(self.viewer.state)

    # def get_point_list(self, index):
    #     synapses = []
    #     synapses = self.states[index].layers['Synapse'].points
    #     synapses = [np.asarray(l) for l in synapses]
    #     return synapses

    def save_json(self, s):
        """ Please delete this """
        self.data.save_json('example.json')
        self.update_message("JSON saved")
