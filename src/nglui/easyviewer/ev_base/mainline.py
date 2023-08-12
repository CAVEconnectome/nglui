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
from . import utils, annotation_compatibility

from .base import EasyViewerBase, SEGMENTATION_LAYER_TYPES
from typing import Union, List, Dict, Tuple, Optional
from numpy import issubdtype, integer, uint64, vstack
from collections import OrderedDict
import re

def nanometer_dimension(resolution):
    return neuroglancer.CoordinateSpace(
        names=["x", "y", "z"],
        scales=resolution,
        units="nm",
    )

class EasyViewerMainline(neuroglancer.UnsynchronizedViewer, EasyViewerBase):
    def __init__(self, **kwargs):
        super(neuroglancer.UnsynchronizedViewer, self).__init__(**kwargs)
        super(EasyViewerBase, self).__init__(**kwargs)

    def load_url(self, url) -> None:
        "Parse a neuroglancer state based on URL and load it into the state"
        state = neuroglancer.parse_url(url)
        self.set_state(state)

    def _ImageLayer(self, source, **kwargs):
        return neuroglancer.viewer_state.ImageLayer(source=source, **kwargs)
    
    def _SegmentationLayer(self, source, **kwargs):
        return neuroglancer.viewer_state.SegmentationLayer(source=source, **kwargs)
    
    def _AnnotationLayer(self, *args, **kwargs):
        return neuroglancer.viewer_state.LocalAnnotationLayer(*args, **kwargs)

    def set_resolution(self, resolution) -> None:
        with self.txn() as s:
            s.dimensions = nanometer_dimension(resolution)

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
        if layer_name is None:
            layer_name = 'annos'
        if layer_name in [l.name for l in self.state.layers]:
            raise ValueError("Layer name already exists")        

        if filter_by_segmentation:
            filter_by_segmentation = ['segments']
        else:
            filter_by_segmentation = []

        if linked_segmentation_layer is True:
            linked_segmentation_layer = {'segments': linked_segmentation_layer}
        else:
            linked_segmentation_layer = {}

        with self.txn() as s:
            new_layer = self._AnnotationLayer(
                linked_segmentation_layer=linked_segmentation_layer,
                filter_by_segmentation=filter_by_segmentation,
                dimensions=self.state.dimensions,
            )
            s.layers.append(name=layer_name, layer=new_layer)
            if color is not None:
                s.layers[layer_name].annotationColor = utils.parse_color(color)

        if tags is not None:
            raise Warning('Tags are not supported by this viewer type.')
        
    def _convert_annotations(
        self,
        annotations: List) -> List:
        return [
            annotation_compatibility.convert_annotation(anno) for anno in annotations
        ]

    def add_annotation_tags(self, layer_name, tags):
        Warning('Annotation tags are not supported by this viewer type.')
        pass

    def as_url(
        self,
        prefix: Optional[str] = None,
        as_html: Optional[bool] = False,
        link_text: Optional[str] = "Neuroglancer Link",
    ) -> str:
        if prefix is None:
            prefix = utils.default_mainline_neuroglancer_base
        ngl_url = neuroglancer.to_url(self.state, prefix=prefix)
        if as_html:
            return '<a href="{}" target="_blank">{}</a>'.format(ngl_url, link_text)
        else:
            return ngl_url


    def select_annotation(self, layer_name, annotation_id):
        Warning('Annotation selection is not supported by this viewer type.')
        pass

    def add_selected_objects(
            self,
            segmentation_layer: str,
            oids: List[int],
            colors: List | Dict | None = None
        ) -> None:
        pass

    def set_view_options(
        self,
        show_slices: Optional[bool] = None,
        layout: Optional[str]=None,
        show_axis_lines: Optional[bool] = None,
        show_scale_bar: Optional[bool] = None,
        orthographic: Optional[bool] = None,
        position: Optional[Tuple[float]] = None,
        zoom_image: Optional[float] = None,
        zoom_3d: Optional[float] = None,
        background_color: Optional[Tuple[float]] = None,
    )->None:
        with self.txn() as s:
            if show_slices is not None:
                s.showSlices = show_slices
            if layout is not None:
                s.layout = layout
            if show_axis_lines is not None:
                s.showAxisLines = show_axis_lines
            if show_scale_bar is not None:
                s.showScaleBar  = show_scale_bar
            if orthographic is not None:
                s.layout.orthographic_projection = orthographic 
            if position is not None:
                s.position = position
            if zoom_image is not None:
                s.crossSectionScale = zoom_image
            if zoom_3d is not None:
                s.projectionScale = zoom_3d
            if background_color is not None:
                s.perspectiveViewBackgroundColor = utils.parse_color(background_color)


    def set_segmentation_view_options(
        self,
        layer_name: str,
        alpha_selected: Optional[float] = None,
        alpha_3d: Optional[float] =None,
        alpha_unselected: Optional[float] = None,
        **kwargs,
    ):
        if self.state.layers[layer_name].type not in SEGMENTATION_LAYER_TYPES:
            raise ValueError("Layer is not a segmentation layer")
        pass

    def set_timestamp(
        self,
        layer_name,
        timestamp: Optional[int] = None,
    ):
        pass

    def set_multicut_points(
        self,
        layer_name,
        seg_id,
        points_red,
        points_blue,
        supervoxels_red=None,
        supervoxels_blue=None,
        focus=True,
    ):
        pass
    