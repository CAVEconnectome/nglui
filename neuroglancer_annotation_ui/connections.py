import json


class Connections:
    """
    Simple datastructure to append pre-post synaptic connections
    """
    def __init__(self):
        self.dataset = []
        self.points = {}
        self.info = {}
        self.pre_point = None
        self.synapse_point = None
        self.post_point = None

    def add_connection(self, pre, post, synapse, obj_type=None, description=None):
        """"Add segmentation layer to viewer instance.

        Attributes:
            pre (list): name of layer to be displayed in neuroglancer ui.
            post (list): source of neuroglancer image layer
            synapse (list): name of layer to be displayed in neuroglancer ui.
            obj_type (string): type of anatomy (e.g. synapse)
            description (string): optional description tag for annotation
        """
        if description is not None:
            self.points['description'] = description
        if obj_type is not None:
            self.points['type'] = obj_type
        self.points['pre_pt'] = {'position': [int(x) for x in pre]}
        self.points['ctr_pt'] = {'position': [int(x) for x in synapse]}
        self.points['post_pt'] = {'position': [int(x) for x in post]}
        self.dataset.append(self.points)
        self._reset_points()
        return self.dataset

    def _reset_points(self):
        self.points = {}

    def save_json(self, json_file):
        with open(json_file, "w") as f:
            json.dump(self.dataset, f, sort_keys=True, indent=4)
