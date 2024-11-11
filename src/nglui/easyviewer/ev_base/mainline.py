import re
from typing import Dict, List, Optional, Tuple, Union
from warnings import warn

import attrs
import neuroglancer
from numpy import integer, issubdtype

from . import utils
from .base import SEGMENTATION_LAYER_TYPES, EasyViewerBase

DEFAULT_SKELETON_SHADER = """void main() {
  emitDefault();
}
"""


def TagToolFactory(number_tags: int):
    for ii in range(number_tags):

        @neuroglancer.viewer_state.export_tool
        class TagTool(neuroglancer.viewer_state.Tool):
            __slots__ = ()
            TOOL_TYPE = f"tagTool_tag{ii}"


TagToolFactory(24)


@attrs.define
class AnnotationTag:
    id: str = None
    type: str = attrs.field(default="uint8")
    tag: str = None


def _make_annotation_properties(annotations, tag_base_number=0):
    "Take a list of annotations and build up the dictionary needed"
    properties = []
    for ii, an in enumerate(annotations):
        properties.append(
            attrs.asdict(AnnotationTag(id=f"tag{ii+tag_base_number}", tag=an))
        )
    return properties


def _make_bindings(properties, bindings=None):
    "Make a dictinary describing key bindings for tags"
    if bindings is None:
        bindings = ["Q", "W", "E", "R", "T", "A", "S", "D", "F", "G"]
    if len(properties) > len(bindings):
        raise ValueError("Too many properties for bindings")
    tool_bindings = {}
    for bind, prop in zip(bindings, properties):
        tool_bindings[bind] = f"tagTool_{prop['id']}"
    return tool_bindings


def nanometer_dimension(resolution):
    return neuroglancer.CoordinateSpace(
        names=["x", "y", "z"],
        scales=resolution,
        units="nm",
    )


class UnservedViewer(neuroglancer.viewer_base.UnsynchronizedViewerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default_viewer_url = kwargs.pop(
            "default_viewer_url", utils.default_mainline_neuroglancer_base
        )

    def get_server_url(self):
        return self._default_viewer_url


class EasyViewerMainline(UnservedViewer, EasyViewerBase):
    def __init__(self, **kwargs):
        super(UnservedViewer, self).__init__(**kwargs)
        super(EasyViewerBase, self).__init__(**kwargs)

    def load_url(self, url) -> None:
        "Parse a neuroglancer state based on URL and load it into the state"
        state = neuroglancer.parse_url(url)
        self.set_state(state)

    def _ImageLayer(self, source, **kwargs):
        if isinstance(source, str):
            source = utils.parse_graphene_header(source, target="mainline")
        return neuroglancer.viewer_state.ImageLayer(source=source, **kwargs)

    def _SegmentationLayer(self, source, **kwargs):
        if isinstance(source, str):
            source = utils.parse_graphene_header(source, target="mainline")
        elif isinstance(source, list):
            source = [utils.parse_graphene_header(s, target="mainline") for s in source]
        return neuroglancer.viewer_state.SegmentationLayer(source=source, **kwargs)

    def _AnnotationLayer(self, *args, **kwargs):
        return neuroglancer.viewer_state.LocalAnnotationLayer(*args, **kwargs)

    def set_resolution(self, resolution) -> None:
        with self.txn() as s:
            s.dimensions = nanometer_dimension(resolution)

    def set_state_server(self, state_server) -> None:
        Warning("State server is set by neuroglancer deployment for this viewer type.")

    def add_annotation_layer(
        self,
        layer_name=None,
        color=None,
        linked_segmentation_layer=None,
        filter_by_segmentation=False,
        brackets_show_segmentation=True,
        selection_shows_segmentation=True,
        tags=None,
        key_bindings=None,
    ) -> None:
        if layer_name is None:
            layer_name = "annos"
        if layer_name in [l.name for l in self.state.layers]:
            raise ValueError("Layer name already exists")

        if filter_by_segmentation:
            filter_by_segmentation = ["segments"]
        else:
            filter_by_segmentation = []

        if linked_segmentation_layer is not None:
            linked_segmentation_layer = {"segments": linked_segmentation_layer}
        else:
            linked_segmentation_layer = {}

        with self.txn() as s:
            new_layer = self._AnnotationLayer(
                dimensions=self.state.dimensions,
                linked_segmentation_layer=linked_segmentation_layer,
                filter_by_segmentation=filter_by_segmentation,
            )
            s.layers.append(name=layer_name, layer=new_layer)
            if color is not None:
                s.layers[layer_name].annotationColor = utils.parse_color(color)

        if tags:
            self.add_annotation_tags(
                layer_name=layer_name, tags=tags, key_bindings=key_bindings
            )

    def _get_number_of_tags(self, layer_name):
        anno_props = self.state.layers[layer_name].annotationProperties
        num_tags = 0
        for prop in anno_props:
            if re.match(r"^tag\d+$", prop.get("id", "")):
                num_tags += 1
        return num_tags

    def add_annotation_tags(self, layer_name, tags, key_bindings=None):
        tag_base_number = self._get_number_of_tags(layer_name)
        props = _make_annotation_properties(tags, tag_base_number)
        bindings = _make_bindings(props, key_bindings)
        with self.txn() as s:
            s.layers[layer_name].annotationProperties = props
            s.layers[layer_name].tool_bindings = bindings

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
            return f'<a href="{ngl_url}" target="_blank">{link_text}</a>'
        else:
            return ngl_url

    def set_selected_layer(self, layer_name):
        with self.txn() as s:
            s.selected_layer = neuroglancer.SelectedLayerState(
                {"visible": True, "layer": layer_name}
            )

    def add_contrast_shader(self, layer_name, black=0, white=255):
        if isinstance(black, float) and black < 1:
            black = int(black * 255)
        if isinstance(white, float) and white < 1:
            white = int(white * 255)
        with self.txn() as s:
            s.layers[
                layer_name
            ].shader_controls = neuroglancer.viewer_state.ShaderControls(
                {"normalized": {"range": [black, white]}}
            )

    def select_annotation(self, layer_name, annotation_id):
        # self.set_selected_layer(layer_name)
        raise Warning("Annotation selection is not supported by this viewer type.")

    def assign_colors(self, layer_name, seg_colors):
        with self.txn() as s:
            s.layers[layer_name].segmentColors = seg_colors

    def add_selected_objects(
        self,
        segmentation_layer: str,
        oids: List[int],
        colors: Optional[Union[List, Dict]] = None,
        invisible_oids: Optional[List[int]] = None,
    ) -> None:
        if issubdtype(type(oids), integer):
            oids = [oids]
        if invisible_oids is None:
            invisible_oids = []
        with self.txn() as s:
            for oid in oids:
                s.layers[segmentation_layer].segments.add(str(oid))
            for oid in invisible_oids:
                s.layers[segmentation_layer].segments.add(f"!{oid}")
        if colors is not None:
            if isinstance(colors, dict):
                self.assign_colors(segmentation_layer, colors)
            elif len(colors) == len(oids):
                seg_colors = {oid: clr for oid, clr in zip(oids, colors)}
                self.assign_colors(segmentation_layer, seg_colors)

    def add_segmentation_layer(self, layer_name, source, **kwargs):
        """Add segmentation layer to viewer instance.

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            source (str): source of neuroglancer segment layer
        """
        with self.txn() as s:
            s.layers[layer_name] = self._SegmentationLayer(source=source, **kwargs)

    def append_source_to_segmentation_layer(self, layer_name, source):
        """Append source or sources to an existing segmentation layer.

        Parameters
        ----------
        layer_name : str
            Name of an existing layer
        source : str or list of str
            Source or sources to add to the layer. Can be segment properties or skeleton sources, for example.
        """
        if isinstance(source, str):
            source = [source]
        source = [utils.parse_graphene_header(s, target="mainline") for s in source]

        with self.txn() as s:
            for src in source:
                s.layers[layer_name].source.append({"url": src})

    def add_skeleton_source(self, layer_name, source, shader_text=None):
        if shader_text is None:
            shader_text = DEFAULT_SKELETON_SHADER
        with self.txn() as s:
            s.layers[layer_name].source.append({"url": source})
            s.layers[layer_name].skeletonShader = shader_text

    def set_skeleton_shader(self, layer_name, shader_text=None):
        if shader_text is None:
            shader_text = DEFAULT_SKELETON_SHADER
        with self.txn() as s:
            s.layers[layer_name].skeletonShader = shader_text

    def set_view_options(
        self,
        show_slices: Optional[bool] = None,
        layout: Optional[str] = None,
        show_axis_lines: Optional[bool] = None,
        show_scale_bar: Optional[bool] = None,
        orthographic: Optional[bool] = None,
        position: Optional[Tuple[float]] = None,
        zoom_image: Optional[float] = None,
        zoom_3d: Optional[float] = None,
        background_color: Optional[Tuple[float]] = None,
    ) -> None:
        with self.txn() as s:
            if show_slices is not None:
                s.showSlices = show_slices
            if layout is not None:
                s.layout = layout
            if show_axis_lines is not None:
                s.showAxisLines = show_axis_lines
            if show_scale_bar is not None:
                s.showScaleBar = show_scale_bar
            if orthographic is not None:
                s.layout.orthographic_projection = orthographic
            if position is not None:
                s.position = position
            if zoom_image is not None:
                s.crossSectionScale = zoom_image
            if zoom_3d is not None:
                s.projectionScale = zoom_3d
            if background_color is not None:
                s.projectionBackgroundColor = utils.parse_color(background_color)

    def set_segmentation_view_options(
        self,
        layer_name: str,
        alpha_selected: Optional[float] = None,
        alpha_3d: Optional[float] = None,
        alpha_unselected: Optional[float] = None,
        silhouette_value: Optional[float] = None,
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
            if silhouette_value is not None:
                l.meshSilhouetteRendering = silhouette_value

    def set_timestamp(
        self,
        layer_name,
        timestamp: Optional[int] = None,
    ):
        if timestamp is not None:
            warn("Timestamp setting is not yet enabled for this viewer type.")

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
        Warning("Setting multicut is not yet enabled for this viewer type")

    @staticmethod
    def point_annotation(
        point,
        id=None,
        description=None,
        linked_segmentation=None,
        annotation_properties=None,
        **kwargs,
    ):
        segments = utils.omit_nones(linked_segmentation)
        if id is None:
            id = neuroglancer.random_token.make_random_token()
        return neuroglancer.PointAnnotation(
            point=list(point),
            id=id,
            description=description,
            segments=[[int(x) for x in segments]],
            props=annotation_properties,
        )

    @staticmethod
    def line_annotation(
        pointA,
        pointB,
        id=None,
        description=None,
        linked_segmentation=None,
        annotation_properties=None,
        **kwargs,
    ):
        segments = utils.omit_nones(linked_segmentation)
        if id is None:
            id = neuroglancer.random_token.make_random_token()
        return neuroglancer.LineAnnotation(
            point_a=list(pointA),
            point_b=list(pointB),
            id=id,
            description=description,
            segments=[[int(x) for x in segments]],
            props=annotation_properties,
        )

    @staticmethod
    def ellipsoid_annotation(
        center,
        radii,
        id=None,
        description=None,
        linked_segmentation=None,
        annotation_properties=None,
        **kwargs,
    ):
        segments = utils.omit_nones(linked_segmentation)
        if id is None:
            id = neuroglancer.random_token.make_random_token()
        return neuroglancer.EllipsoidAnnotation(
            center=list(center),
            radii=list(radii),
            id=id,
            description=description,
            segments=[[int(x) for x in segments]],
            props=annotation_properties,
        )

    @staticmethod
    def bounding_box_annotation(
        pointA,
        pointB,
        id=None,
        description=None,
        linked_segmentation=None,
        annotation_properties=None,
        **kwargs,
    ):
        segments = utils.omit_nones(linked_segmentation)
        if id is None:
            id = neuroglancer.random_token.make_random_token()
        return neuroglancer.AxisAlignedBoundingBoxAnnotation(
            point_a=list(pointA),
            point_b=list(pointB),
            id=id,
            description=description,
            segments=[[int(x) for x in segments]],
            props=annotation_properties,
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
        **kwargs,
    ):
        Warning("Annotation groups are not yet supported by this viewer type.")
