import numpy as np
import gzip
import os

from neuroglancer_annotation_ui import annotation


def parse_skeleton(path):
    """ Reads skeleton from zip file

    Format: https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Skeletons-and-Point-Clouds

    :param path: str
    :return:
        vertices, edges, radii, vertex_types
    """
    f = gzip.open(path, mode="rb")
    data_b = f.read()
    f.close()

    vertex_count, edge_count = np.frombuffer(data_b[:8], dtype=np.uint32)

    offset = 8
    vertices = np.frombuffer(data_b[offset: offset + vertex_count * 12],
                             dtype=np.float32).reshape(-1, 3)

    offset += vertex_count * 12
    edges = np.frombuffer(data_b[offset: offset + edge_count * 8],
                          dtype=np.uint32).reshape(-1, 2)

    offset += edge_count * 8
    radii = np.frombuffer(data_b[offset: offset + vertex_count * 4],
                          dtype=np.float32)

    offset += vertex_count * 4
    vertex_types = np.frombuffer(data_b[offset: offset + vertex_count * 1],
                                 dtype=np.uint8)

    return vertices, edges, radii, vertex_types


class SkeletonMeta(object):
    def __init__(self, name, color=None, scaling=(8, 8, 40)):
        """

        :param name: str
            = annotation layer name
        :param color: Hex or RGB
        """
        self._skeletons = []
        self._name = name
        self._color = color
        self._scaling = np.array(list(scaling))

    @property
    def skeletons(self):
        return self._skeletons

    @property
    def name(self):
        return self._name

    @property
    def color(self):
        return self._color

    def add_skeleton_from_file(self, path, skeleton_id=None):
        if skeleton_id is None:
            skeleton_id = os.path.basename(path).split(".")[0]

        vertices, edges, radii, vertex_types = parse_skeleton(path)
        self.add_skeleton(skeleton_id, vertices, edges)

    def add_skeleton(self, skeleton_id, vertices, edges):
        skeleton = Skeleton(skeleton_id, vertices, edges, scaling=self._scaling)
        self._skeletons.append(skeleton)

    def add_to_ngl(self, viewer):
        viewer.add_annotation_layer(self.name, color=self.color)

        for skeleton in self.skeletons:
            skeleton.add_to_ngl(viewer, self.name, color=self.color)


class Skeleton(object):
    def __init__(self, skeleton_id, vertices, edges, scaling=(8, 8, 40)):
        """

        :param skeleton_id: int or str
        :param vertices: np.float32 (n x 3)
        :param edges: np.uint32 (m x 2)
        """
        self._skeleton_id = skeleton_id
        self._vertices = vertices
        self._edges = edges
        self._scaling = np.array(list(scaling))

    @property
    def skeleton_id(self):
        return self._skeleton_id

    @property
    def edges(self):
        return self._edges

    @property
    def vertices(self):
        return self._vertices

    @property
    def scaling(self):
        return self._scaling

    @property
    def scaled_vertices(self):
        scaled_vertices = self.vertices / self._scaling
        return scaled_vertices.astype(np.int)

    @property
    def edge_nodes(self):
        return self.vertices[self.edges]

    @property
    def scaled_edge_nodes(self):
        return self.scaled_vertices[self.edges]

    def add_to_ngl(self, viewer, layer_name, color=None):
        lines = []
        for i_nodes, nodes in enumerate(self.scaled_edge_nodes):
            lines.append(annotation.line_annotation(nodes[0], nodes[1],
                                                    annotation.generate_id(),
                                                    description=str(self.skeleton_id)))


        viewer.add_annotation(layer_name, lines, color=color)

