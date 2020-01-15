import neuroglancer
import copy
import numpy as np
from neuroglancer.viewer_state import SegmentationLayer

default_static_content_source='https://neuromancer-seung-import.appspot.com/'

def set_static_content_source(source=default_static_content_source):
    neuroglancer.set_static_content_source(url=source)

def stop_ngl_server():
    """
    Shuts down the neuroglancer tornado server
    """
    neuroglancer.server.stop()

class ChunkedgraphSegmentationLayer(SegmentationLayer):
    def __init__(self, *args, **kwargs):
        super(SegmentationLayer, self).__init__(*args, type='segmentation_with_graph', **kwargs)