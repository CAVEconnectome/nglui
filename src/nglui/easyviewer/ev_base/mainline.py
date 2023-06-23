try:
    import neuroglancer
except ImportError:
    Warning(
        """
        Making states of this type requires the google neuroglancer python, but it is not installed.
        Please install it with `pip install neuroglancer`
        """
    )
    use_ngl = False
else:
    use_ngl = True

from . import utils
from .base import EasyViewerBase
from typing import Union, List, Dict, Tuple, Optional
from numpy import issubdtype, integer, uint64, vstack
from collections import OrderedDict
import re

class EasyViewerMainline(neuroglancer.UnsynchronizedViewer, EasyViewerBase):
    def __init__(self, **kwargs):
        super(neuroglancer.UnsynchronizedViewer, self).__init__(**kwargs)
        super(EasyViewerBase, self).__init__(**kwargs)

    def load_url(self, url) -> None:
        pass

    def _ImageLayer(self, source, **kwargs):
        return neuroglancer.viewer_state.ImageLayer(source=source, **kwargs)
    
    def _SegmentationLayer(self, source, **kwargs):
        return neuroglancer.viewer_state.SegmentationLayer(source, **kwargs)
    
    def _AnnotationLayer(self, **kwargs):
        return neuroglancer.viewer_state.LocalAnnotationLayer(**kwargs)

    def set_resolution(self, resolution) -> None:
        with self.txn() as s:
            dims = neuroglancer.CoordinateSpace(
                names=["x", "y", "z"],
                scales=resolution,
                units="nm",
            )
            s.dimensions = dims


    def set_state_server(self, state_server) -> None:
        Warning('State server is set by neuroglancer deployment for this viewer type.')
        pass

    def add_annotation_layer(
        self,
        layer_name=None,
        color=None,
        linked_segmentation_layer=None,
        filter_by_segmentation=False,
        brackets_show_segmentation=True,
        selection_shows_segmentation=True,
        tags=None,
    ) -> None:
        pass

    def _convert_annotations(
        self,
        annotations: List) -> List:
        """Pass through annotations, currently defaulting to seung lab format already"""
        pass


    def add_annotation_tags(self, layer_name, tags):
        Warning('Annotation tags are not supported by this viewer type.')
        pass