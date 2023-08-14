from . import nglite as neuroglancer
import numpy as np
import pandas as pd
import webcolors
import re
import numbers

SegmentationLayer = neuroglancer.viewer_state.SegmentationLayer

default_neuroglancer_base = "https://neuromancer-seung-import.appspot.com/"
default_mainline_neuroglancer_base = "https://ngl.cave-explorer.org/"

def omit_nones(seg_list):
    if seg_list is None or np.all(pd.isna(seg_list)):
        return []
    seg_list = np.atleast_1d(seg_list)
    seg_list = list(filter(lambda x: x is not None, seg_list))
    if len(seg_list) == 0:
        return []
    else:
        return seg_list

def parse_color(clr):
    if clr is None:
        return None

    if isinstance(clr, numbers.Number):
        clr = (clr, clr, clr)

    if isinstance(clr, str):
        hex_match = "\#[0123456789abcdef]{6}"
        if re.match(hex_match, clr.lower()):
            return clr
        else:
            return webcolors.name_to_hex(clr)
    else:
        return webcolors.rgb_to_hex([int(255 * x) for x in clr])
