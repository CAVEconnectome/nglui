import nglui.nglite as neuroglancer
from nglui.nglite.viewer_state import SegmentationLayer

default_static_content_source = 'https://neuromancer-seung-import.appspot.com/'

graphene_version_content_sources = {0: 'https://graphene-v0-dot-neuromancer-seung-import.appspot.com/',
                                    1: 'https://neuromancer-seung-import.appspot.com/'}


def set_static_content_source(source=default_static_content_source, graphene_version=None):
    if graphene_version is None:
        neuroglancer.set_static_content_source(url=source)
    else:
        neuroglancer.set_static_content_source(
            url=graphene_version_content_sources.get(graphene_version, 0))


def stop_ngl_server():
    """
    Shuts down the neuroglancer tornado server
    """
    neuroglancer.stop()


class ChunkedgraphSegmentationLayer(SegmentationLayer):
    def __init__(self, *args, **kwargs):
        super(SegmentationLayer, self).__init__(
            *args, type='segmentation_with_graph', **kwargs)


def omit_nones(seg_list):
    if seg_list is None:
        return []

    seg_list = list(filter(lambda x: x is not None, seg_list))
    if len(seg_list) == 0:
        return []
    else:
        return seg_list
