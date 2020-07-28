from .. import nglite as neuroglancer

SegmentationLayer = neuroglancer.viewer_state.SegmentationLayer

default_neuroglancer_base = "https://neuromancer-seung-import.appspot.com/"


def omit_nones(seg_list):
    if seg_list is None:
        return []

    seg_list = list(filter(lambda x: x is not None, seg_list))
    if len(seg_list) == 0:
        return []
    else:
        return seg_list
