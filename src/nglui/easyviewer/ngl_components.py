from __future__ import annotations

import json
from typing import Optional, Self, Union

import caveclient
import numpy as np
import pandas as pd
from attrs import asdict, define, field
from neuroglancer import Viewer, viewer_state
from neuroglancer.coordinate_space import CoordinateSpace, parse_unit
from neuroglancer.random_token import make_random_token

from ..segmentprops import SegmentProperties
from .ngl_annotations import *
from .shaders import DEFAULT_SHADER_MAP
from .utils import (
    is_dict_like,
    is_list_like,
    parse_color,
    parse_graphene_header,
    parse_graphene_image_url,
    split_point_columns,
)


@define
class CoordSpace:
    resolution = field(default=None, type=list[int])
    units = field(default="nm", type=str)
    names = field(factory=lambda: ["x", "y", "z"], type=list[str])

    def __attrs_post_init__(self):
        if isinstance(self.units, str):
            self.units = [self.units] * len(self.resolution)
            if len(self.names) != len(self.units):
                raise ValueError("Length of names and unit must match")
            if len(self.resolution) != len(self.names):
                raise ValueError("Length of resolution and names must match")

    def to_neuroglancer(self) -> CoordinateSpace:
        if self.resolution is None:
            raise ValueError("Resolution must be set before converting to Neuroglancer")
        return CoordinateSpace(
            units=self.units, scales=self.resolution, names=self.names
        )


@define
class CoordSpaceTransform:
    output_dimensions = field(default=None, type=list[int])
    input_dimensions = field(default=None, type=list[int])
    matrix = field(default=None, type=list[list[float]])

    def __attrs_post_init__(self):
        if self.output_dimensions is None:
            self.output_dimensions = None
        elif not isinstance(self.output_dimensions, CoordSpace):
            self.output_dimensions = CoordSpace(resolution=self.output_dimensions)

    def to_neuroglancer(self):
        if self.output_dimensions is None:
            return None
        if self.input_dimensions is not None:
            input_dims = self.input_dimensions.to_neuroglancer()
        else:
            input_dims = None
        return viewer_state.CoordinateSpaceTransform(
            output_dimensions=self.output_dimensions.to_neuroglancer(),
            input_dimensions=input_dims,
            matrix=self.matrix,
        )


@define
class Source:
    url = field(type=str)
    resolution = field(default=None, type=list)
    transform = field(default=None, type=CoordSpaceTransform)
    subsources = field(default=None, type=dict)
    enable_default_subsources = field(default=True, type=bool)

    def to_neuroglancer(self) -> dict:
        if self.transform is None:
            self.transform = CoordSpaceTransform(output_dimensions=self.resolution)
        return viewer_state.LayerDataSource(
            url=self.url,
            transform=self.transform.to_neuroglancer(),
            subsources=self.subsources,
            enable_default_subsources=self.enable_default_subsources,
        )


@define
class _Layer:
    name = field(type=str)
    visible = field(default=True, type=bool, kw_only=True)
    archived = field(default=False, type=bool, kw_only=True)

    def add_source(
        self,
        source: Union[str, list, Source],
        resolution: Optional[list] = None,
    ) -> Self:
        """Add a source to the layer.

        Parameters
        ----------
        source : str or Source
            The source to add.
        resolution : list or np.typing.ArrayLike, optional
            The resolution of the source. Default is None.

        """
        if self.source is None:
            self.source = []
        if isinstance(self.source, Source) or isinstance(self.source, str):
            self.source = [self.source]
        if isinstance(source, str):
            self.source.append(Source(url=source, resolution=resolution))
        elif isinstance(source, Source):
            self.source.append(source)
        elif is_list_like(source):
            for src in source:
                if isinstance(src, str):
                    self.source.append(Source(url=src, resolution=resolution))
                elif isinstance(src, Source):
                    self.source.append(src)
                else:
                    raise ValueError("Invalid source type. Must be str or Source.")
        else:
            raise ValueError("Invalid source type. Must be str or Source.")
        return self

    def to_neuroglancer_layer(self):
        raise NotImplementedError(
            "to_neuroglancer_layer() must be implemented in subclasses"
        )

    def to_dict(self, with_name: bool = True) -> dict:
        """Convert the layer to a dictionary.
        Parameters
        ----------
        with_name : bool, optional
            Whether to include the name, visibility, and archived states of the layer in the dictionary, by default True.
            These are not typically included until the layer is part of a state, but adding them in allows the resulting data to be passed to a viewer state string.

        Returns
        -------
        dict
            The layer as a dictionary.
        """

        layer_dict = self.to_neuroglancer_layer().to_json()
        if with_name:
            layer_dict["name"] = self.name
            layer_dict["visible"] = self.visible
            layer_dict["archived"] = self.archived
        return layer_dict

    def to_json(self, with_name: bool = True, indent=2):
        return json.dumps(self.to_dict(with_name=with_name), indent=indent)

    def _to_neuroglancer_state(self, s):
        if self.name in s.layers:
            raise ValueError(
                f"Layer {self.name} already exists in the viewer. Please use a different name."
            )
        s.layers[self.name] = self.to_neuroglancer_layer()
        ll = s.layers[self.name]
        ll.visible = self.visible
        ll.archived = self.archived

    def _to_neuroglancer(self, viewer):
        # Opens context or not depending on if the object is a viewer or a (presumed within-context) state
        if isinstance(viewer, Viewer):
            with viewer.txn() as s:
                self._to_neuroglancer_state(s)
        elif isinstance(viewer, viewer_state.ViewerState):
            self._to_neuroglancer_state(viewer)


def _handle_source(source, resolution=None, image_layer=False):
    "Convert one or more sources to a Source object."
    if image_layer:
        auth_parse_func = parse_graphene_image_url
    else:
        auth_parse_func = parse_graphene_header
    if isinstance(source, str):
        return Source(url=auth_parse_func(source), resolution=resolution)
    elif isinstance(source, Source):
        source.url = auth_parse_func(source.url)
        return source
    elif is_list_like(source):
        return [
            _handle_source(src, resolution, image_layer=image_layer) for src in source
        ]
    else:
        raise ValueError("Invalid source type. Must be str or Source.")


def _handle_annotations(annos):
    "Convert a multi-url source to a Source object."
    if annos is None:
        return []
    elif len(annos) == 0:
        return []
    return [
        anno.to_neuroglancer() if issubclass(type(anno), AnnotationBase) else anno
        for anno in annos
    ]


def source_to_neuroglancer(source, resolution=None, image_layer=False):
    "Convert a possibly multi-url source to a Neuroglancer-compatible format."

    source = _handle_source(source, resolution=resolution, image_layer=image_layer)
    if isinstance(source, list):
        return [src.to_neuroglancer() for src in source]
    else:
        return source.to_neuroglancer()


def segments_to_neuroglancer(segments):
    "Convert a flat or mixed-visibility segment list to Neuroglancer."
    if segments is None:
        return viewer_state.StarredSegments()
    if is_list_like(segments) or is_dict_like(segments):
        # Annoying processing to avoid np.True_/np.False_ types
        return viewer_state.StarredSegments(strip_numpy_types(segments))
    else:
        return segments


def _handle_linked_segmentation(segmentation_layer):
    if segmentation_layer is None:
        return None
    elif isinstance(segmentation_layer, SegmentationLayer):
        return {"segments": segmentation_layer.name}
    elif isinstance(segmentation_layer, str):
        return ({"segments": segmentation_layer},)
    else:
        raise ValueError(
            "Invalid linked segmentation layer type. Must be str or SegmentationLayer."
        )


@define
class ImageLayer(_Layer):
    name = field(default="img", type=str)
    source = field(factory=list, type=Union[list, Source])
    resolution = field(default=None, type=list, kw_only=True)
    shader = field(default=None, type=Optional[str], kw_only=True)
    color = field(default=None, type=list, kw_only=True)
    opacity = field(default=1.0, type=float, kw_only=True)
    blend = field(default=None, type=str, kw_only=True)
    volume_rendering_mode = field(default=None, type=str, kw_only=True)
    volume_rendering_gain = field(default=None, type=float, kw_only=True)
    volume_rendering_depth_samples = field(default=None, type=int, kw_only=True)
    cross_section_render_scale = field(default=None, type=float, kw_only=True)

    def __attrs_post_init__(self):
        self.color = parse_color(self.color)

    def to_neuroglancer_layer(self) -> viewer_state.ImageLayer:
        return viewer_state.ImageLayer(
            source=source_to_neuroglancer(
                self.source, resolution=self.resolution, image_layer=True
            ),
            shader=self.shader,
            annotation_color=self.color,
        )

    def to_neuroglancer(self, viewer: Viewer) -> viewer_state.ImageLayer:
        "Can be a viewer or a viewer.txn()-context state"
        self._to_neuroglancer(viewer)
        return self

    def from_client(
        self,
        client: caveclient.CAVEclient,
    ) -> Self:
        """Add an image layer source from caveclient.

        Parameters
        ----------
        client : CAVEclient
            The CAVEclient object.

        Returns
        -------
        ImageLayer
            An ImageLayer object.

        """
        return self.add_source(
            source=client.info.image_source(),
            resolution=client.info.viewer_resolution(),
        )

    def add_shader(self, shader: str) -> Self:
        """Add a shader to the layer.

            Parameters
            ----------
            shader : str
                The shader to add.

        #"""
        self.shader = shader
        return self


@define
class SegmentationLayer(_Layer):
    name = field(default="seg", type=str)
    source = field(default=None, type=Source)
    resolution = field(default=None, type=list, kw_only=True)
    segments = field(
        factory=list, type=Union[list, dict, viewer_state.VisibleSegments], kw_only=True
    )
    color = field(default=None, type=list, kw_only=True)
    hide_segment_zero = field(default=True, type=bool, kw_only=True)
    selected_alpha = field(default=0.2, type=float, kw_only=True)
    not_selected_alpha = field(default=0.0, type=float, kw_only=True)
    alpha_3d = field(default=0.9, type=float, kw_only=True)
    mesh_silhouette = field(default=0.0, type=float, kw_only=True)
    segment_colors = field(default=None, type=dict, kw_only=True)
    shader = field(default=None, type=str, kw_only=True)

    def __attrs_post_init__(self):
        self.color = parse_color(self.color)

    def to_neuroglancer_layer(self) -> viewer_state.SegmentationLayer:
        return viewer_state.SegmentationLayer(
            source=source_to_neuroglancer(self.source, resolution=self.resolution),
            visible_segments=segments_to_neuroglancer(self.segments),
            annotation_color=self.color,
            selected_alpha=self.selected_alpha,
            not_selected_alpha=self.not_selected_alpha,
            object_alpha=self.alpha_3d,
            segment_colors=self.segment_colors,
            skeleton_shader=self.shader,
        )

    def to_neuroglancer(self, viewer) -> viewer_state.SegmentationLayer:
        "Can be a viewer or a viewer.txn()-context state"
        self._to_neuroglancer(viewer)
        return self

    def from_client(
        self,
        client: caveclient.CAVEclient,
        add_skeleton_source: bool = True,
    ) -> Self:
        """Add a segmentation layer source from caveclient.

        Parameters
        ----------
        client : CAVEclient
            The CAVEclient object.

        Returns
        -------
        SegmentationLayer
            A SegmentationLayer object.

        """
        self.add_source(
            source=client.info.segmentation_source(),
            resolution=client.info.viewer_resolution(),
        )
        if (
            add_skeleton_source
            and client.info.get_datastack_info().get("skeleton_source") is not None
        ):
            self.add_source(
                source=client.info.get_datastack_info().get("skeleton_source")
            )
        return self

    def add_segments(
        self,
        segments: Union[list, dict, viewer_state.VisibleSegments],
        visible: Optional[list] = None,
    ) -> Self:
        """Add segment ids to the layer.

        Parameters
        ----------
        segments : list or dict or VisibleSegments
            The segments to add. If a dict, the keys are the segment IDs and the values are the boolean visibility.
        visible: list, optional
            The visibility of the segments, assumed to be True if not provided.
            Should be the same length as segments, and segments should be a list of the same length.

        Returns
        -------
        SegmentationLayer
            The SegmentationLayer object.
        """
        old_segments = dict(segments_to_neuroglancer(self.segments))
        if visible is not None:
            segments = {s: v for s, v in zip(segments, visible)}
        elif not is_dict_like(segments):
            segments = {s: True for s in segments}
        new_segments = dict(segments_to_neuroglancer(segments))
        self.segments = old_segments | new_segments
        return self

    def add_segment_colors(
        self,
        segment_colors: dict,
    ) -> Self:
        """Add segment colors to the layer.

        Parameters
        ----------
        segment_colors : dict
            The segment colors to add.

        Returns
        -------
        SegmentationLayer
            The SegmentationLayer object with the added segment colors.
        """
        if self.segment_colors is None:
            self.segment_colors = {}
        parsed_colors = {k: parse_color(v) for k, v in segment_colors.items()}
        self.segment_colors.update(parsed_colors)
        return self

    def segments_from_dataframe(
        self,
        df: pd.DataFrame,
        segment_column: str,
        visible_column: str = None,
        color_column: str = None,
    ) -> Self:
        """Create a SegmentationLayer from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the segmentation data.
        name : str, optional
            The name of the layer. Default is "seg".
        color : str, optional
            The color to use.
        shader : str, optional
            The shader to use.

        Returns
        -------
        SegmentationLayer
            A SegmentationLayer object.
        """
        segments = df[segment_column].values
        if visible_column is not None:
            segments = {s: v for s, v in zip(segments, df[visible_column].values)}
        if color_column is not None:
            self.add_segment_colors(
                {s: parse_color(c) for s, c in zip(segments, df[color_column].values)}
            )

        self.add_segments(segments)
        return self

    def segment_properties(
        self,
        df: pd.DataFrame,
        client: "caveclient.CAVEclient",
        id_column: str = "pt_root_id",
        label_column: Optional[Union[str, list]] = None,
        description_column: Optional[str] = None,
        string_columns: Optional[list] = None,
        number_columns: Optional[list] = None,
        tag_value_columns: Optional[list] = None,
        tag_bool_columns: Optional[list] = None,
        tag_descriptions: Optional[list] = None,
        allow_disambiguation: bool = True,
        label_separator: str = "_",
        label_format_map: Optional[str] = None,
        prepend_col_name: bool = False,
        random_columns: Optional[int] = None,
        random_column_prefix: str = None,
        dry_run: bool = False,
    ) -> Self:
        """Upload segment properties and add to the layer.
        If you already have a segment properties cloud path, use `add_source` to add it to the layer.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the segment properties.
        client : CAVEclient
            The CAVEclient object needed to upload .
        id_column : str, optional
            The column name for the segment IDs. Default is 'pt_root_id'.
        label_column : str or list, optional
            The column name(s) for the segment labels. Default is None.
        description_column : str, optional
            The column name for the segment descriptions. Default is None.
        string_columns : list, optional
            The column names for string properties. Default is None.
        number_columns : list, optional
            The column names for number properties. Default is None.
        tag_value_columns : list, optional
            The column names for tag value properties. Default is None.
        tag_bool_columns : list, optional
            The column names for tag boolean properties. Default is None.
        tag_descriptions : list, optional
            The descriptions for the tags. Default is None.
        allow_disambiguation : bool, optional
            Whether to allow disambiguation of segment IDs. Default is True.
        label_separator : str, optional
            The separator for label formatting. Default is "_".
        label_format_map : str, optional
            The format map for labels. Default is None.
        prepend_col_name: bool, optional
            Whether to prepend the column name to the label. Default is False.
        random_columns: int, optional
            Number of random columns to add. Default is None.
        random_column_prefix: str, optional
            Name prefix of the random columns. Default is None.
        dry_run: bool, optional
            If dry run is true, build but do not actually upload and instead use a placeholder source text. Default is False.

        Returns
        -------
        SegmentationLayer
            A SegmentationLayer object with added segment properties.

        """
        segprops = SegmentProperties.from_dataframe(
            df=df,
            id_col=id_column,
            label_col=label_column,
            description_col=description_column,
            string_cols=string_columns,
            number_cols=number_columns,
            tag_value_cols=tag_value_columns,
            tag_bool_cols=tag_bool_columns,
            tag_descriptions=tag_descriptions,
            allow_disambiguation=allow_disambiguation,
            label_separator=label_separator,
            label_format_map=label_format_map,
            prepend_col_name=prepend_col_name,
            random_columns=random_columns,
            random_column_prefix=random_column_prefix,
        )
        if not dry_run:
            prop_id = client.state.upload_property_json(segprops.to_dict())
            prop_url = client.state.build_neuroglancer_url(
                prop_id, format_properties=True
            )
        else:
            prop_url = "DRYRUN_SEGMENT_PROPERTIES"
        self.add_source(prop_url)
        return self

    def set_view_options(
        self,
        selected_alpha: Optional[float] = None,
        not_selected_alpha: Optional[float] = None,
        alpha_3d: Optional[float] = None,
        mesh_silhouette: Optional[float] = None,
    ) -> Self:
        """Set the view options for the layer.

        Parameters
        ----------
        selected_alpha : float, optional
            The alpha value for selected segments. Default is 0.2.
        not_selected_alpha : float, optional
            The alpha value for not selected segments. Default is 0.0.
        alpha_3d : float, optional
            The alpha value for 3D segments. Default is 0.9.
        mesh_silhouette : float, optional
            The silhouette value for the mesh. Default is 0.0.

        Returns
        -------
        SegmentationLayer
            A SegmentationLayer object with updated view options.

        """
        if selected_alpha:
            self.selected_alpha = selected_alpha
        if not_selected_alpha:
            self.not_selected_alpha = not_selected_alpha
        if alpha_3d:
            self.alpha_3d = alpha_3d
        if mesh_silhouette:
            self.mesh_silhouette = mesh_silhouette
        return self

    def add_shader(self, shader: str) -> Self:
        """Add a shader to the layer.

        Parameters
        ----------
        shader : str
            The shader to add.

        """
        self.shader = shader
        return self

    def add_default_skeleton_shader(self) -> Self:
        """Add a default skeleton shader with desaturated axons to the layer."""
        return self.add_shader(DEFAULT_SHADER_MAP["skeleton_compartments"])


@define
class AnnotationLayer(_Layer):
    name = field(default="anno", type=str)
    source = field(default=None, type=Source)
    color = field(default=None, type=list, kw_only=True)
    linked_segmentation_layer = field(default=None, type=Union[str, dict], kw_only=True)
    shader = field(default=None, type=str, kw_only=True)

    def __attrs_post_init__(self):
        self.color = parse_color(self.color)

    def to_neuroglancer_layer(self) -> viewer_state.AnnotationLayer:
        return viewer_state.AnnotationLayer(
            source=source_to_neuroglancer(self.source),
            annotation_color=self.color,
            linked_segmentation_layer=self.linked_segmentation_layer,
            shader=self.shader,
        )

    def add_shader(self, shader: str) -> Self:
        """Add a shader to the layer.

        Parameters
        ----------
        shader : str
            The shader to add.

        """
        self.shader = shader
        return self


@define
class LocalAnnotationLayer(_Layer):
    name = field(default="anno", type=str)
    resolution = field(default=None, type=list, kw_only=True)
    color = field(default=None, type=list, kw_only=True)
    annotations = field(default=None, type=list, kw_only=True)
    linked_segmentation_layer = field(default=None, type=Union[str, dict], kw_only=True)
    shader = field(default=None, type=str, kw_only=True)
    tags = field(factory=list, type=list, kw_only=True)

    def __attrs_post_init__(self):
        self.color = parse_color(self.color)

    def to_neuroglancer_layer(self) -> viewer_state.LocalAnnotationLayer:
        props = make_annotation_properties(self.tags, tag_base_number=0)
        bindings = make_bindings(props)
        if not isinstance(self.resolution, CoordSpace):
            self.resolution = CoordSpace(resolution=self.resolution)
        return viewer_state.LocalAnnotationLayer(
            dimensions=self.resolution.to_neuroglancer(),
            annotation_color=self.color,
            annotations=_handle_annotations(self.annotations),
            linked_segmentation_layer=_handle_linked_segmentation(
                self.linked_segmentation_layer
            ),
            shader=self.shader,
            annotation_properties=props,
            tool_bindings=bindings,
        )

    def to_neuroglancer(self, viewer) -> viewer_state.AnnotationLayer:
        "Can be a viewer or a viewer.txn()-context state"
        if self.resolution is None:
            raise ValueError(
                f"Resolution for annotation layer '{self.name}' must be set before converting to Neuroglancer"
            )
        self._to_neuroglancer(viewer)

    def set_linked_segmentation_layer(
        self,
        layer: Union[str, SegmentationLayer],
    ) -> Self:
        """Add a linked segmentation layer to the annotation layer.

        Parameters
        ----------
        layer : str or SegmentationLayer
            The linked segmentation layer or the string value of its name.

        """
        if isinstance(layer, SegmentationLayer):
            layer = layer.name
        self.linked_segmentation_layer = layer
        return self

    def add_annotations(
        self,
        annotations: list,
    ) -> Self:
        """Add annotations to the layer.

        Parameters
        ----------
        annotations : list
            The annotations to add.

        Returns
        -------
        self
            The AnnotationLayer object with added annotations.
        """
        if self.annotations is None:
            self.annotations = []
        for annotation in annotations:
            if issubclass(type(annotation), AnnotationBase):
                self.annotations.append(annotation)
            else:
                raise ValueError(
                    "Invalid annotation type. Must be a PointAnnotation, LineAnnotation, Ellipse/SphereAnnotation, or BoundingBoxAnnotation."
                )
        return self

    def add_point_annotations(
        self,
        df: pd.DataFrame,
        point_column: str,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        data_resolution: Optional[list] = None,
        tag_column: Optional[str] = None,
    ) -> Self:
        """Add point annotations to the layer.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the point annotations.
        point_column : str
            The column name for the point coordinates.
            Can be a prefix for a split column name with _x, _y, _z as suffixes.
        segment_column : str, optional
            The column name for the segment IDs. Default is None.
        description_column : str, optional
            The column name for the segment descriptions. Default is None.
        tag_column : str, optional
            The column name for segment descripion. Default is None.
        data_resolution : list, optional
            The resolution of the data. If None, follows from layer resolution.
            Default is None.

        Returns
        -------
        SegmentationLayer
            The SegmentationLayer object with added point annotations.
        """
        point_column = split_point_columns(point_column, df.columns)
        points = df[point_column].values
        if data_resolution is not None:
            points = np.array(points) / np.array(data_resolution)
        if segment_column is not None:
            segments = df[segment_column].values
        else:
            segments = [None] * len(points)
        if description_column is not None:
            descriptions = df[description_column].values
        else:
            descriptions = [None] * len(points)

        self.add_annotations(
            [
                PointAnnotation(
                    point=p,
                    segments=seg,
                    description=d,
                )
                for p, seg, d in zip(points, segments, descriptions)
            ]
        )
        return self

    def add_line_annotations(
        self,
        df: pd.DataFrame,
        point_a_column: str,
        point_b_column: str,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        data_resolution: Optional[list] = None,
        tag_column: Optional[str] = None,
    ) -> Self:
        """Add point annotations to the layer.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the point annotations.
        point_a_column : str
            The column name for the start point coordinates.
            Can be a prefix for a split column name with _x, _y, _z as suffixes.
        point_b_column : str
            The column name for the end point coordinates.
            Can be a prefix for a split column name with _x, _y, _z as suffixes.
        segment_column : str, optional
            The column name for the segment IDs. Default is None.
        description_column : str, optional
            The column name for the segment descriptions. Default is None.
        data_resolution : list, optional
            The resolution of the data. Default is None.

        Returns
        -------
        SegmentationLayer
            The SegmentationLayer object with added point annotations.
        """
        point_a_column = split_point_columns(point_a_column, df.columns)
        point_b_column = split_point_columns(point_b_column, df.columns)

        points_a = df[point_a_column].values
        points_b = df[point_b_column].values

        if segment_column is not None:
            segments = df[segment_column].values
        else:
            segments = [None] * len(points_a)
        if description_column is not None:
            descriptions = df[description_column].values
        else:
            descriptions = [None] * len(points_a)

        self.add_annotations(
            [
                LineAnnotation(
                    point_a=p_a,
                    point_b=p_b,
                    segments=seg,
                    description=d,
                )
                for p_a, p_b, seg, d in zip(points_a, points_b, segments, descriptions)
            ]
        )
        return self

    def add_ellipsoid_annotations(
        self,
        df: pd.DataFrame,
        center_column: str,
        radii_column: str,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        data_resolution: Optional[list] = None,
        tag_column: Optional[str] = None,
    ) -> Self:
        """Add point annotations to the layer.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the point annotations.
        center_column : str
            The column name for the center point coordinates.
            Can be a prefix for a split column name with _x, _y, _z as suffixes.
        radii_column : str
            The column name for the radius values.
            Can be a prefix for a split column name with _x, _y, _z as suffixes.
        segment_column : str, optional
            The column name for the segment IDs. Default is None.
        description_column : str, optional
            The column name for the segment descriptions. Default is None.
        data_resolution : list, optional
            The resolution of the data. Default is None.

        Returns
        -------
        SegmentationLayer
            The SegmentationLayer object with added point annotations.
        """
        center_column = split_point_columns(center_column, df.columns)
        radii_column = split_point_columns(radii_column, df.columns)

        centers = df[center_column].values
        radii_vals = df[radii_column].values

        if segment_column is not None:
            segments = df[segment_column].values
        else:
            segments = [None] * len(center_column)
        if description_column is not None:
            descriptions = df[description_column].values
        else:
            descriptions = [None] * len(center_column)

        self.add_annotations(
            [
                EllipsoidAnnotation(
                    center=p_a,
                    radii=p_b,
                    segments=seg,
                    description=d,
                )
                for p_a, p_b, seg, d in zip(
                    center_column, radii_column, segments, descriptions
                )
            ]
        )
        return self

    def add_sphere_annotations(
        self,
        df: pd.DataFrame,
        center_column: str,
        radius_column: str,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        data_resolution: Optional[list] = None,
        tag_column: Optional[str] = None,
    ) -> Self:
        """Add point annotations to the layer.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the point annotations.
        center_column : str
            The column name for the center point coordinates.
            Can be a prefix for a split column name with _x, _y, _z as suffixes.
        radii_column : str
            The column name for the radius values.
            Can be a prefix for a split column name with _x, _y, _z as suffixes.
        segment_column : str, optional
            The column name for the segment IDs. Default is None.
        description_column : str, optional
            The column name for the segment descriptions. Default is None.
        data_resolution : list, optional
            The resolution of the data. Default is None.

        Returns
        -------
        SegmentationLayer
            The SegmentationLayer object with added point annotations.
        """
        center_column = split_point_columns(center_column, df.columns)
        radius_column = radius_column

        centers = df[center_column].values
        radius_vals = df[radius_column].values

        if segment_column is not None:
            segments = df[segment_column].values
        else:
            segments = [None] * len(center_column)
        if description_column is not None:
            descriptions = df[description_column].values
        else:
            descriptions = [None] * len(center_column)

        self.add_annotations(
            [
                SphereAnnotation(
                    center=p_a,
                    radius=rad,
                    segments=seg,
                    description=d,
                )
                for p_a, rad, seg, d in zip(
                    center_column, radius_vals, segments, descriptions
                )
            ]
        )
        return self

    def add_line_annotations(
        self,
        df: pd.DataFrame,
        point_a_column: str,
        point_b_column: str,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        data_resolution: Optional[list] = None,
        tag_column: Optional[str] = None,
    ) -> Self:
        """Add point annotations to the layer.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the point annotations.
        point_a_column : str
            The column name for the start point coordinates of the box.
            Can be a prefix for a split column name with _x, _y, _z as suffixes.
        point_b_column : str
            The column name for the end point coordinates of the box
            Can be a prefix for a split column name with _x, _y, _z as suffixes.
        segment_column : str, optional
            The column name for the segment IDs. Default is None.
        description_column : str, optional
            The column name for the segment descriptions. Default is None.
        data_resolution : list, optional
            The resolution of the data. Default is None.

        Returns
        -------
        SegmentationLayer
            The SegmentationLayer object with added point annotations.
        """
        point_a_column = split_point_columns(point_a_column, df.columns)
        point_b_column = split_point_columns(point_b_column, df.columns)

        points_a = df[point_a_column].values
        points_b = df[point_b_column].values

        if segment_column is not None:
            segments = df[segment_column].values
        else:
            segments = [None] * len(points_a)
        if description_column is not None:
            descriptions = df[description_column].values
        else:
            descriptions = [None] * len(points_a)

        self.add_annotations(
            [
                BoundingBoxAnnotation(
                    point_a=p_a,
                    point_b=p_b,
                    segments=seg,
                    description=d,
                )
                for p_a, p_b, seg, d in zip(points_a, points_b, segments, descriptions)
            ]
        )
        return self

    @property
    def tag_map(self) -> dict:
        return {tag: f"tagTool_{ii}" for ii, tag in enumerate(self.tags)}

    # def add_tags(
    #     self,
    #     tags: list,
    # ) -> Self:
    #     """Add tags to the layer.

    #     Parameters
    #     ----------
    #     tags : list
    #         The tags to add.

    #     """
    #     if self.tags is None:
    #         self.tags = []
    #     for tag in tags:
    #         if tag not in self.tags:
    #             self.tags.append(tags)
    #     return self
