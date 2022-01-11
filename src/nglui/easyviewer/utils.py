from .. import nglite as neuroglancer
import numpy as np
import pandas as pd

SegmentationLayer = neuroglancer.viewer_state.SegmentationLayer

default_neuroglancer_base = "https://neuromancer-seung-import.appspot.com/"


def omit_nones(seg_list):
    if seg_list is None or np.all(pd.isna(seg_list)):
        return []
    seg_list = np.atleast_1d(seg_list)
    seg_list = list(filter(lambda x: x is not None, seg_list))
    if len(seg_list) == 0:
        return []
    else:
        return seg_list
