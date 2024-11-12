from .ev_base import EasyViewerMainline, EasyViewerSeunglab
from ..site_utils import is_mainline


def EasyViewer(
    target_site="seunglab",
):
    if not is_mainline(target_site) or target_site is None:
        return EasyViewerSeunglab()
    elif is_mainline(target_site):
        return EasyViewerMainline()
    else:
        raise ValueError(
            f'Unknown target: {target_site}. Must be one of "seunglab" or "spelunker/mainline"/"cave-explorer"'
        )
