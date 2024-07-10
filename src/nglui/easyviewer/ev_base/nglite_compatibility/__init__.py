from neuroglancer.url_state import *
from neuroglancer.viewer_base import UnsynchronizedViewerBase
from neuroglancer.viewer_state import *


class Viewer(UnsynchronizedViewerBase):
    def __init__(self, **kwargs):
        super(UnsynchronizedViewerBase, self).__init__(**kwargs)

    def get_viewer_url(self):
        return "No server implemented"
