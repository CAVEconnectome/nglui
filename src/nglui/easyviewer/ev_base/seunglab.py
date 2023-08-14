# Seung lab branch of easyviewer

from .base import EasyViewerBase, SEGMENTATION_LAYER_TYPES
from . import utils
import re
from . import nglite as neuroglancer
from typing import Union, List, Dict, Tuple, Optional
from numpy import issubdtype, integer, uint64, vstack
from collections import OrderedDict

class EasyViewerSeunglab(neuroglancer.UnsynchronizedViewer, EasyViewerBase):
    def __init__(self, **kwargs):
        super(neuroglancer.UnsynchronizedViewer, self).__init__(**kwargs)
        super(EasyViewerBase, self).__init__(**kwargs)

    def load_url(self, url) -> None:
        "Parse a neuroglancer state based on URL and load it into the state"
        state = neuroglancer.parse_url(url)
        self.set_state(state)

    def _ImageLayer(self, source, **kwargs):
        return neuroglancer.ImageLayer(source=source, **kwargs)

    def _AnnotationLayer(self, **kwargs):
        return neuroglancer.AnnotationLayer(
            **kwargs,
        )

    def _SegmentationLayer(self, source, **kwargs):
        if re.search(r"^graphene://", source) is not None:
            return neuroglancer.ChunkedgraphSegmentationLayer(
                source=source, **kwargs
            )
        elif re.search(r"^precomputed://", source) is not None:
            return neuroglancer.SegmentationLayer(
                source=source, **kwargs
            )
        else:
            raise ValueError('Source must be either graphene:// or precomputed://')

    def set_resolution(self, resolution) -> None:
        with self.txn() as s:
            s.voxel_size = resolution

    def set_state_server(self, state_server) -> None:
        with self.txn() as s:
            s._json_data["jsonStateServer"] = state_server

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
            layer_name = "annos"
        if layer_name in [l.name for l in self.state.layers]:
            raise ValueError("Layer name already exists")
        if linked_segmentation_layer is None:
            filter_by_segmentation = None

        with self.txn() as s:
            new_layer = self._AnnotationLayer(
                linked_segmentation_layer=linked_segmentation_layer,
                filter_by_segmentation=filter_by_segmentation,
                brackets_show_segmentation=brackets_show_segmentation,
                selection_shows_segmentation=selection_shows_segmentation,
            )
            s.layers.append(name=layer_name, layer=new_layer)
            if color is not None:
                s.layers[layer_name].annotationColor = utils.parse_color(color)

        if tags is not None:
            self.add_annotation_tags(layer_name=layer_name, tags=tags)

    def _convert_annotations(
        self,
        annotations: List) -> List:
        """Pass through annotations, currently defaulting to seung lab format already"""
        return annotations

    def add_annotation_tags(self, layer_name, tags):
        if layer_name not in self.layer_names:
            raise ValueError("Layer is not an annotation layer")
        tag_list = [
            OrderedDict({"id": tag_id + 1, "label": label})
            for tag_id, label in enumerate(tags)
        ]
        with self.txn() as s:
            s.layers[layer_name]._json_data["annotationTags"] = tag_list

    def as_url(
        self,
        prefix: Optional[str] = None,
        as_html: Optional[bool] = False,
        link_text: Optional[str] = "Neuroglancer Link",
    ) -> str:
        if prefix is None:
            prefix = utils.default_neuroglancer_base
        ngl_url = neuroglancer.to_url(self.state, prefix=prefix)
        if as_html:
            return '<a href="{}" target="_blank">{}</a>'.format(ngl_url, link_text)
        else:
            return ngl_url


    def select_annotation(self, layer_name, anno_id):
        if layer_name in self.layer_names:
            with self.txn() as s:
                s.layers[layer_name]._json_data["selectedAnnotation"] = id
        self.set_selected_layer(layer_name)

    def set_selected_layer(self, layer_name):
        with self.txn() as s:
            s.selectedLayer.layer = layer_name
        

    def add_selected_objects(
        self, 
        segmentation_layer: str,
        oids: List[int],
        colors: Optional[Union[List, Dict]] = None,
    ) -> None:
        if issubdtype(type(oids), integer):
            oids = [oids]

        with self.txn() as s:
            for oid in oids:
                s.layers[segmentation_layer].segments.add(uint64(oid))
            if s.layers[segmentation_layer].type == "segmentation_with_graph":
                s.layers[segmentation_layer].segmentQuery = ", ".join(
                    [str(x) for x in oids]
                )
        if colors is not None:
            if isinstance(colors, dict):
                self.assign_colors(segmentation_layer, colors)
            elif len(colors) == len(oids):
                seg_colors = {str(oid): clr for oid, clr in zip(oids, colors)}
                self.assign_colors(segmentation_layer, seg_colors)


    def assign_colors(self, layer_name, seg_colors):
        """Assign colors to root ids in a segmentation layer

        Parameters
        ----------
        layer_name : str,
            Segmentation layer name
        seg_colors : dict
            dict with root ids as keys and colors as values.
        """
        with self.txn() as s:
            if seg_colors is not None:
                seg_colors = {
                    str(oid): utils.parse_color(k)
                    for oid, k in seg_colors.items()
                    if k is not None
                }
                s.layers[layer_name]._json_data["segmentColors"] = seg_colors

 

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
        """Set options relating to the neuroglancer view state. Only changes the values of the parameters provided.

        Parameters
        ----------
        show_slices : bool, optional
            Show slice cutout in the 3d view, by default None
        layout : str, optional
            Change the layout type ('xy', 'yz', 'zx', '3d', 'xy-3d', 'yz-3d', 'zx-3d', or '4panel'), by default None
        show_axis_lines : bool, optional
            Show the red/blue/green lines indicating the axis directions, by default None
        show_scale_bar : bool, optional
            Controls showing of scale bar, by default None
        orthographic : bool, optional
            Controls whether the 3d perspective view is orthographic or not, by default None
        position : list, optional
            Sets the location of center point of the view in Neuroglancer coordinates, by default None
        zoom_image : int, optional
            Sets the zoom factor for the imagery, by default None
        zoom_3d : int, optional
            Sets the zoom factor for the 3d view, by default None
        background_color : list or str, optional
            hex, rgb, or named color for the background of the 3d viewer, by default None
        """
        with self.txn() as s:
            if show_slices is not None:
                s.showSlices = show_slices
            if layout is not None:
                s.layout.type = layout
            if show_axis_lines is not None:
                s.show_axis_lines = show_axis_lines
            if show_scale_bar is not None:
                s.show_scale_bar = show_scale_bar
            if orthographic is not None:
                s.layout.orthographic_projection = orthographic
            if position is not None:
                s.position.voxelCoordinates = position
            if zoom_image is not None:
                s.navigation.zoomFactor = zoom_image
            if zoom_3d is not None:
                s.perspectiveZoom = zoom_3d
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
        with self.txn() as s:
            l = s.layers[layer_name]
            if alpha_selected is not None:
                l.selectedAlpha = alpha_selected
            if alpha_3d is not None:
                l.objectAlpha = alpha_3d
            if alpha_unselected is not None:
                l.notSelectedAlpha = alpha_unselected

    def set_timestamp(
        self,
        layer_name,
        timestamp: Optional[int] = None,
    ):
        """Set timestamp of a segmentation layer

        Parameters
        ----------
        layer_name : str
            Name of a segmentation layer
        timestamp : float, optional
            Timestamp in unix epoch time (e.g. `time.time.now()` in python), by default None
        """
        if self.state.layers[layer_name].type != "segmentation_with_graph":
            return
        with self.txn() as s:
            l = s.layers[layer_name]
            if timestamp is not None:
                l.timestamp = int(timestamp)
            else:
                l.timestamp = None


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
        """Configures multicut points in the neuroglancer state. Note that points need to be in mesh units (e.g. nanometers), not voxels!

        Parameters
        ----------
        layer_name : str
            Segmentation layer name
        seg_id : np.uint64
            Segmentation id of the object in question
        points_red : np.array
            Nx3 array of locations in voxel space for side 1 of the cut.
        points_blue : np.array
            Mx3 array of locations in voxel space for side 2 of the cut.
        supervoxels_red : np.array or None, optional
            N-length array of supervoxel ids associated with locations in points_red or None.
            If None, supervoxels lookup occurs based on the mesh. By default None
        supervoxels_blue : np.array or None, optional
            M-length array of supervoxel ids associated with locations in points_blue or None.
            If None, supervoxels lookup occurs based on the mesh. By default None
        focus : bool, optional
            If True, makes the layer and graph tool focused. By default True
        """

        def _multicut_annotation(pt, oid, sv_id):
            if sv_id is None:
                sv_id = oid
            return neuroglancer.annotation.point_annotation(
                pt, description=str(sv_id), linked_segmentation=[sv_id, oid]
            )

        if supervoxels_red is None:
            supervoxels_red = [None for x in points_red]
        if supervoxels_blue is None:
            supervoxels_blue = [None for x in points_blue]

        if self.state.voxel_size is None:
            voxel_size = [4, 4, 40]
        else:
            voxel_size = self.state.voxel_size

        annos_red = neuroglancer.annotationHolder()
        for pt, sv_id in zip(points_red, supervoxels_red):
            annos_red.annotations.append(
                _multicut_annotation(pt * voxel_size, seg_id, sv_id)
            )

        annos_blue = neuroglancer.annotationHolder()
        for pt, sv_id in zip(points_blue, supervoxels_blue):
            annos_blue.annotations.append(
                _multicut_annotation(pt * voxel_size, seg_id, sv_id)
            )

        self.add_selected_objects(layer_name, [seg_id])

        with self.txn() as s:
            l = s.layers[layer_name]
            l.tab = "graph"
            l.graphOperationMarker.append(annos_red)
            l.graphOperationMarker.append(annos_blue)

        if focus:
            self.set_selected_layer(layer_name)
            ctr_pt = vstack([points_red, points_blue]).mean(axis=0)
            self.set_view_options(position=ctr_pt, zoom_3d=100)

    @staticmethod
    def point_annotation(
        point,
        id=None,
        description=None,
        linked_segmentation=None,
        tag_ids=None,
    ):
        return neuroglancer.point_annotation(
            point=point,
            id=id,
            description=description,
            linked_segmentation=linked_segmentation,
            tag_ids=tag_ids,
        )
    
    @staticmethod
    def line_annotation(
        pointA,
        pointB,
        id=None,
        description=None,
        linked_segmentation=None,
        tag_ids=None,
    ):
        return neuroglancer.line_annotation(
            a=pointA,
            b=pointB,
            id=id,
            description=description,
            linked_segmentation=linked_segmentation,
            tag_ids=tag_ids,
        )

    @staticmethod
    def ellipsoid_annotation(
        center,
        radii,
        id=None,
        description=None,
        linked_segmentation=None,
        tag_ids=None,
    ):
        return neuroglancer.ellipsoid_annotation(
            center=center,
            radii=radii,
            id=id,
            description=description,
            linked_segmentation=linked_segmentation,
            tag_ids=tag_ids,
        )

    @staticmethod
    def bounding_box_annotation(
        pointA,
        pointB,
        id=None,
        description=None,
        linked_segmentation=None,
        tag_ids=None,
    ):
        return neuroglancer.bounding_box_annotation(
            pointA=pointA,
            pointB=pointB,
            id=id,
            description=description,
            linked_segmentation=linked_segmentation,
            tag_ids=tag_ids,
        )
    
    @staticmethod
    def group_annotations(
        annotations,
        source=None,
        id=None,
        return_all=True,
        gather_linked_segmentations=True,
        share_linked_segmentations=False,
        children_visible=True,
    ):
        return neuroglancer.group_annotations(
            annotations,
            source=source,
            id=id,
            return_all=return_all,
            gather_linked_segmentations=gather_linked_segmentations,
            share_linked_segmentations=share_linked_segmentations,
            children_visible=children_visible,
        )