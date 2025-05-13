import numbers
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import webcolors

from ..site_utils import (
    is_mainline,
)


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
        hex_match = r"\#[0123456789abcdef]{6}"
        if re.match(hex_match, clr.lower()):
            return clr
        else:
            return webcolors.name_to_hex(clr)
    else:
        return webcolors.rgb_to_hex([int(255 * x) for x in clr])


def parse_graphene_header(source, target):
    qry = urlparse(source)
    if qry.scheme == "graphene":
        return _parse_to_mainline(qry)
    else:
        return source


def parse_graphene_image_url(source, target):
    qry = urlparse(source)
    if qry.scheme == "graphene":
        if is_mainline(target):
            return _parse_to_mainline_imagery(qry)
    else:
        return source


def _parse_to_seunglab(qry):
    return f"{qry.scheme}://https:{qry.path}"


def _parse_to_mainline(qry):
    if "https" in qry.netloc and "middleauth":
        return f"{qry.scheme}://middleauth+https:{qry.path}"
    else:
        return f"{qry.scheme}://middleauth+http:{qry.path}"


def _parse_to_mainline_imagery(qry):
    if qry.scheme == "graphene":
        if "https" in qry.netloc:
            return f"precomputed://middleauth+https:{qry.path}"
        else:
            return f"precomputed://middleauth+http:{qry.path}"
    else:
        return qry.geturl()
