from neuroglancer_annotation_ui import annotation
from neuroglancer_annotation_ui.connections import Connections
import random
import pytest


def generate_connections():
    points = []
    pre_pt = annotation.point_annotation(random.sample(range(1, 100), 3),
                              annotation.generate_id(), description=0)
    syn_pt = annotation.point_annotation(random.sample(range(1, 100), 3),
                              annotation.generate_id(), description=1)
    post_pt = annotation.point_annotation(random.sample(range(1, 100), 3),
                               annotation.generate_id(), description=2)
    points.append(pre_pt.point.tolist())
    points.append(syn_pt.point.tolist())
    points.append(post_pt.point.tolist())
    return points


if __name__ == '__main__':
    pt1 = generate_connections()
    data = Connections()
    # add pre and post segment ids
    pre_id1 = '0987'
    post_id1 = '1234'
    data.set_active_pair(pre_id1, post_id1)
    data.add_connection(pt1[0], pt1[1], pt1[2])
    print("Add new ids: ", data.dataset)
    # add another pre and post segment ids
    pre_id2 = '6789'
    post_id2 = '4321'
    pt2 = generate_connections()
    d1 = data.set_active_pair(pre_id2, post_id2)
    data.add_connection(pt2[0], pt2[1], pt2[2])
    data.save_json('test.txt')
    rmv = 'pre_d1'+'_'+'post_id1'
    data.remove_connection(rmv)
    print("Removed:", data.dataset)
