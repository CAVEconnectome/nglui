from neuroglancer_annotation_ui import interface
from neuroglancer_annotation_ui import annotation
from neuroglancer_annotation_ui import connections


class Connector(interface.Interface):
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

    def clear_all(self, s):
        pos = self.viewer.state.navigation.position.voxel_coordinates
        print(self.base_state)
        self.viewer.set_state(self.base_state)
        self._update_view(pos)
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


if __name__ == '__main__':
    # intialize annotation class
    ngl_url = 'https://nkem-rebase-dot-neuromancer-seung-import.appspot.com/'

    example = Connector()
    example.set_source_url(ngl_url)
    # dict of actions
    actions = {
        'select_pre': ['shift+keyq', example.select_pre_process],
        'add_synapse': ['shift+keyw', example.add_synapse],
        'select_post': ['shift+keye', example.select_post_process],
        'save_json': ['shift+keys', example.save_json],
        'clear_all': ['shift+keyc', example.clear_all],
        'undo_last_point': ['shift+keyz', example.undo_last_point],
        'delete_synapse': ['delete', example.delete_synapse],
        'clear_segment': ['shift+keyv', example.clear_segment],
    }
    # bind actions to interface
    for action, bindings in actions.items():
        example.add_action(action, bindings[0], bindings[1])
    segment_source = 'precomputed://gs://neuroglancer/pinky40_v11/watershed_mst_trimmed_sem_remap'
    image_source = 'precomputed://gs://neuroglancer/pinky40_v11/image_rechunked'
    example.add_segmentation_layer('Segmentation', segment_source)
    example.add_image_layer('Image', image_source)
    example.show()
