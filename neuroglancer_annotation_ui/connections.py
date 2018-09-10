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
        self.points['pre_pt'] = {'position': pre}
        self.points['ctr_pt'] = {'position': synapse}
        self.points['post_pt'] = {'position': post}
        self.dataset.append(self.points)
        self._reset_points()
        return self.dataset

    def _reset_points(self):
        self.points = {}

    def save_json(self, json_file):
        with open(json_file, "w") as f:
            json.dump(self.dataset, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    import annotation
    import random

    # {'pre_post': {0:{'pre': [], 'post': [], 'syn': []}}}
    def generate_connections():
        points = []
        pre_pt = annotation.point_annotation(random.sample(range(1, 100), 3),
                        annotation.generate_id(), description={'order':0})
        syn_pt = annotation.point_annotation(random.sample(range(1, 100), 3),
                        annotation.generate_id(), description={'order':1,'detail':'pancake'})
        post_pt = annotation.point_annotation(random.sample(range(1, 100), 3),
                        annotation.generate_id(), description={'order':2})
        points.append(pre_pt.point.tolist())
        points.append(syn_pt.point.tolist())
        points.append(post_pt.point.tolist())
        return points

    pt1 = generate_connections()
    print("POINTS 1: ", pt1)
    # print(pt1)

    # print(pre_point.point, synapse.point, post_point.description)
    # print(pre_point)
    data = Connections()
    pre_id1 = '0987'
    post_id1 = '1234'
    data.set_active_pair(pre_id1, post_id1)
    data.add_connection(pt1[0], pt1[1], pt1[2])
    pre_id2 = '6789'
    post_id2 = '4321'
    pt2 = generate_connections()
    print("POINTS 2: ", pt2)
    data.set_active_pair(pre_id2, post_id2)
    data.add_connection(pt2[0], pt2[1], pt2[2])
    data.set_active_pair(pre_id1, post_id1)
    pt3 = generate_connections()
    print("POINTS 3: ", pt3)
    data.add_connection(pt3[0], pt3[1], pt3[2])
    # print("Dataset:", data.dataset)
    # rmv = 'pre_d1'+'_'+'post_id1'
    # data.remove_connection(rmv)
    # print("Removed:", data.dataset)
    data.save_json('test.txt')
