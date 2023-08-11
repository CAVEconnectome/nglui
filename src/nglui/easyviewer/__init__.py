from .ev_base import EasyViewerSeunglab, EasyViewerMainline
from . import annotation

def EasyViewer(
    target_site='seunglab',
):
    if target_site == 'seunglab' or target_site is None:
        return EasyViewerSeunglab()
    elif target_site == 'mainline':
        return EasyViewerMainline()
    else:
        raise ValueError(f'Unknown target: {target_site}. Must be one of "seunglab" or "mainline"')
