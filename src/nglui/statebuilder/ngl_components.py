from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Optional, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import caveclient
import numpy as np
import pandas as pd
from attrs import define, field
from neuroglancer import Viewer, viewer_state
from neuroglancer.coordinate_space import CoordinateSpace
from neuroglancer.json_wrappers import optional, wrapped_property

from ..segmentprops import SegmentProperties
from .ngl_annotations import (
    MAX_TAG_COUNT,
    AnnotationBase,
    BoundingBoxAnnotation,
    EllipsoidAnnotation,
    LineAnnotation,
    PointAnnotation,
    make_annotation_properties,
    make_bindings,
    strip_numpy_types,
)
from .shaders import DEFAULT_SHADER_MAP
from .utils import (
    is_dict_like,
    is_list_like,
    parse_color,
    parse_graphene_header,
    parse_graphene_image_url,
    split_point_columns,
)

# Monkey-patch AnnotationLayer to add swap_visible_segments_on_move property
viewer_state.AnnotationLayer.swap_visible_segments_on_move = (
    viewer_state.AnnotationLayer.swapVisibleSegmentsOnMove
) = wrapped_property("swapVisbleSegmentsOnMove", optional(bool, True))


class UnmappedDataError(Exception):
    """Exception raised when a layer is not fully mapped to a datamap."""


@define
class DataMap:
    """Class to defer mapping of dataframes to a ViewerState or Layer.

    Parameters
    ----------
    key : str, optional
        The key to use for the datamap, which is used to label data when using the `map` or `with_map` classes.
        If no key is provided, a default key of "None" is used and assumed if no dictionary is provided later.
    """

    key = field(default=None, type=Optional[str])
    priority = field(default=10, type=int, init=False, repr=False)

    def _adjust_priority(self, value: int) -> Self:
        """Make the datamap high priority."""
        self.priority = value
        return self


@define
class CoordSpace:
    """Coordinate space for Neuroglancer.
    Parameters
    ----------
    resolution : list[int], optional
        The resolution of the coordinate space. Default is None, which will raise an error if not set.
    units : str or list[str], optional
        The units of the coordinate space. Default is "nm". If a single string is provided, it will be repeated for each dimension.
    names : list[str], optional
        The names of the dimensions. Default is ["x", "y", "z"]. If the length of names does not match the length of resolution, an error will be raised.
    """

    resolution = field(default=None, type=list[int])
    units = field(default="nm", type=str)
    names = field(factory=lambda: ["x", "y", "z"], type=list[str])

    def __attrs_post_init__(self):
        if self.resolution is None:
            self.resolution = []
            self.units = []
            self.names = []
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
    """Coordinate space transform for Neuroglancer.

    Parameters
    ----------
    output_dimensions : list[int] or CoordSpace, optional
        The output dimensions of the transform. If a list, it will be converted to a CoordSpace.
        Default is None, which will create a transform with no output dimensions.
    input_dimensions : list[int] or CoordSpace, optional
        The input dimensions of the transform. If a list, it will be converted to a CoordSpace.
        Default is None, which will create a transform with no input dimensions.
    matrix : list[list[float]], optional
        The transformation matrix. Default is None, which will create an identity matrix.
        The matrix should be a 4x4 list of lists, where the last row is [0, 0, 0, 1].
        If None, an identity matrix will be created based on the output dimensions.
    """

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
    """
    Configuration for a Neuroglancer data source.

    Parameters
    ----------
    url : str
        The URL of the data source.
    resolution : list, optional
        The resolution of the data source. Default is None.
    transform : CoordSpaceTransform, optional
        The coordinate space transform for the data source. Default is None, which will create a transform based on the resolution.
    subsources : dict, optional
        A dictionary of subsources for the data source (e.g. meshes, skeletons, etc). Default is None, which includes all subsources.
    enable_default_subsources : bool, optional
        Whether to enable default subsources. Default is True, which includes all default subsources.
    """

    url = field(type=str)
    resolution = field(default=None, type=list)
    transform = field(default=None, type=CoordSpaceTransform)
    subsources = field(default=None, type=dict)
    enable_default_subsources = field(default=True, type=bool)

    def to_neuroglancer(self) -> dict:
        """Convert the Source to a neuroglancer-python object."""
        if self.transform is None:
            self.transform = CoordSpaceTransform(output_dimensions=self.resolution)
        return viewer_state.LayerDataSource(
            url=self.url,
            transform=self.transform.to_neuroglancer(),
            subsources=self.subsources,
            enable_default_subsources=self.enable_default_subsources,
        )


@define
class Layer(ABC):
    name = field(type=str)
    resolution = field(default=None, type=list, kw_only=True, repr=False)
    visible = field(default=True, type=bool, kw_only=True, repr=False)
    archived = field(default=False, type=bool, kw_only=True, repr=False)
    pick = field(default=True, type=bool, kw_only=True, repr=False)
    _datamaps = field(factory=dict, type=dict, init=False, repr=False)
    _datamap_priority = field(factory=dict, type=dict, init=False, repr=False)

    @contextmanager
    def with_datamap(self, datamap: dict):
        if datamap is None:
            yield self
        else:
            if not isinstance(datamap, dict):
                datamap = {None: datamap}

            layer_copy = copy.deepcopy(self)
            dm_keys = list(layer_copy._datamap_priority.keys())
            dm_priorities = list(layer_copy._datamap_priority.values())
            sorted_keys = [
                x for x, _ in sorted(zip(dm_keys, dm_priorities), key=lambda x: x[1])
            ]

            for k in sorted_keys:
                if k in datamap:
                    func = layer_copy._datamaps.pop(k)
                    func(datamap.pop(k))
            yield layer_copy

    @property
    def is_static(self) -> bool:
        """Check if the layer is static."""
        return len(self._datamaps) == 0

    def _check_fully_mapped(self):
        """Check if the layer is fully mapped."""
        if not self.is_static:
            raise UnmappedDataError(
                f"Layer '{self.name}' has datamaps registered but no datamap provided: {list(self._datamaps.keys())}"
            )

    def _register_datamap(self, key, func, **kwargs):
        """Register a function to be called when the datamap is set.

        Parameters
        ----------
        key: DataMap
            Object with a key and priority for the datamap.
        func : Callable
            The function to register.
        kwargs:
            Additional keyword arguments to pass to the function.
        """
        self._datamaps[key.key] = partial(func, **kwargs)
        self._datamap_priority[key.key] = key.priority

    def _apply_datamaps(self, datamap: dict):
        """Map"""
        for k, v in datamap.items():
            if k in self._datamaps:
                self._datamaps.pop(k)(v)  # Apply the function with the remaining values

    def map(self, datamap: dict, inplace: bool = False) -> Self:
        """Map the layer to a datamap.

        Parameters
        ----------
        datamap : dict
            The datamap to map the layer to.
        inplace: bool, optional
            Whether to modify the layer in place or return a new layer. Default is False.
        """
        if inplace:
            self._apply_datamaps(datamap)
            return self
        else:
            with self.with_datamap(datamap) as layer:
                return layer

    @abstractmethod
    def to_neuroglancer_layer(self):
        self._check_fully_mapped()

    def to_dict(self, with_name: bool = True) -> dict:
        """Convert the layer to a dictionary.
        Parameters
        ----------
        with_name : bool, optional
            Whether to include the name, visibility, and archived states of the layer in the dictionary, by default True.
            These are not typically included until the layer is part of a state, but adding them in allows the resulting data to be passed to a viewer state string.
        datamap: dict, optional
            A dictionary with keys being names of DataMap parameters and values being the data to be passed.
            Must be provided if the layer has any datamaps registered.

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

    def to_json(self, with_name: bool = True, indent: int = 2):
        return json.dumps(self.to_dict(with_name=with_name), indent=indent)

    def _apply_to_neuroglancer_state(self, s):
        if self.name in s.layers:
            raise ValueError(
                f"Layer {self.name} already exists in the viewer. Please use a different name."
            )
        s.layers[self.name] = self.to_neuroglancer_layer()
        ll = s.layers[self.name]
        ll.visible = self.visible
        ll.archived = self.archived

    def _apply_to_neuroglancer(self, viewer):
        # Opens context or not depending on if the object is a viewer or a (presumed within-context) state
        self._check_fully_mapped()
        if isinstance(viewer, Viewer):
            with viewer.txn() as s:
                self._apply_to_neuroglancer_state(s)
        elif isinstance(viewer, viewer_state.ViewerState):
            self._apply_to_neuroglancer_state(viewer)


@define
class LayerWithSource(Layer):
    source = field(factory=list, type=Union[list, Source])

    def __attrs_post_init__(self):
        if isinstance(self.source, DataMap):
            self.source._adjust_priority(1)
            self._register_datamap(
                key=self.source,
                func=self.add_source,
                resolution=self.resolution,
            )
            self.source = []

    def add_source(
        self,
        source: Union[str, list, Source, DataMap],
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
        if isinstance(source, DataMap):
            if len(self.source) > 0:
                raise ValueError(
                    "Cannot add a DataMap source to a layer that already has sources."
                )
            self._register_datamap(
                key=source._adjust_priority(1),
                func=self.add_source,
                resolution=resolution,
            )
            return self  # Return early for DataMap
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
                    raise ValueError(
                        "Invalid source type. Must be str, list, Source, or DataMap."
                    )
        else:
            raise ValueError(
                "Invalid source type. Must be str, list, Source, or DataMap."
            )
        return self


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


def _handle_annotations(annos, tag_map=None, resolution=None):
    "Convert a multi-url source to a Source object."
    if annos is None:
        return []
    elif len(annos) == 0:
        return []
    return [
        anno.to_neuroglancer(tag_map=tag_map, layer_resolution=resolution)
        if issubclass(type(anno), AnnotationBase)
        else anno
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


def _handle_linked_segmentation(segmentation_layer) -> dict:
    if segmentation_layer is None:
        return None
    elif isinstance(segmentation_layer, dict):
        return segmentation_layer
    elif isinstance(segmentation_layer, SegmentationLayer):
        return {"segments": segmentation_layer.name}
    elif isinstance(segmentation_layer, str):
        return {"segments": segmentation_layer}
    else:
        raise ValueError(
            "Invalid linked segmentation layer type. Must be str or SegmentationLayer."
        )


def _handle_filter_by_segmentation(
    filter: Optional[Union[list, bool, str]],
    linked_segmentation: Optional[Union[list, bool, str]],
):
    linked_seg = _handle_linked_segmentation(linked_segmentation)
    if filter is True:
        return [x for x in linked_seg.keys()]
    elif filter is False:
        return []
    elif is_list_like(filter):
        return linked_seg


@define
class ImageLayer(LayerWithSource):
    """Configuration for a Neuroglancer image layer.

    Parameters
    ----------
    name : str, optional
        The name of the layer. Default is "img".
    source : Source or list of Source, optional
        The source of the image data. Can be a Source object or a list of Source objects.
    shader : str, optional
        The shader to use for rendering the image. Default is None, which will use the default shader.
    color : list, optional
        The color to use for the image layer. Default is None, which will use the default color.
    opacity : float, optional
        The opacity of the image layer. Default is 1.0.
    blend : str, optional
        The blending mode for the image layer. Default is None, which will use the default blend mode.
    volume_rendering_mode : str, optional
        The volume rendering mode for the image layer. Default is None, which will use the default mode.
    volume_rendering_gain : float, optional
        The gain for volume rendering. Default is None, which will use the default gain.
    volume_rendering_depth_samples : int, optional
        The number of depth samples for volume rendering. Default is None, which will use the default number of samples.
    cross_section_render_scale : float, optional
        The scale for cross-section rendering. Default is None, which will use the default scale.
    pick: bool, optional
        Whether to allow cursor interaction with meshes and skeletons. Default is True.
    """

    name = field(default="img", type=str)
    source = field(factory=list, type=Union[list, Source])
    shader = field(default=None, type=Optional[str], kw_only=True, repr=False)
    color = field(default=None, type=list, kw_only=True, repr=False)
    opacity = field(default=1.0, type=float, kw_only=True, repr=False)
    blend = field(default=None, type=str, kw_only=True, repr=False)
    volume_rendering_mode = field(default=None, type=str, kw_only=True, repr=False)
    volume_rendering_gain = field(default=None, type=float, kw_only=True, repr=False)
    volume_rendering_depth_samples = field(
        default=None, type=int, kw_only=True, repr=False
    )
    cross_section_render_scale = field(
        default=None, type=float, kw_only=True, repr=False
    )

    def __attrs_post_init__(self):
        self.color = parse_color(self.color)
        super().__attrs_post_init__()

    def to_neuroglancer_layer(self) -> viewer_state.ImageLayer:
        super().to_neuroglancer_layer()
        if self.shader is None:
            return viewer_state.ImageLayer(
                source=source_to_neuroglancer(
                    self.source, resolution=self.resolution, image_layer=True
                ),
                annotation_color=self.color,
            )
        else:
            return viewer_state.ImageLayer(
                source=source_to_neuroglancer(
                    self.source, resolution=self.resolution, image_layer=True
                ),
                shader=self.shader,
                annotation_color=self.color,
            )

    def apply_to_neuroglancer(self, viewer: Viewer) -> viewer_state.ImageLayer:
        "Can be a viewer or a viewer.txn()-context state"
        self._apply_to_neuroglancer(viewer)

    def add_from_client(
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
class SegmentationLayer(LayerWithSource):
    """Configuration for a Neuroglancer segmentation layer.

    Parameters
    ----------
    name : str, optional
        The name of the layer. Default is "seg".
    source : Source or str, optional
        The source of the segmentation data. Can be a Source object or a string URL.
    segments : list or dict or VisibleSegments, optional
        The segments to display in the layer. Can be a list of segment IDs, a dictionary with segment IDs as keys and visibility as values, or a VisibleSegments object.
    color : list, optional
        The color to use for the segments. Default is None, which will use the default color.
    hide_segment_zero : bool, optional
        Whether to hide segment zero, which is typically treated as "no segmentation". Default is True.
    selected_alpha : float, optional
        The transparency value for selected segments in the 2d views. Default is 0.2.
    not_selected_alpha : float, optional
        The transparency value for unselected segments in the 2d views. Default is 0.0.
    alpha_3d : float, optional
        The transparency value for segments in the 3d view. Default is 0.9.
    mesh_silhouette : float, optional
        The silhouette rendering value for the mesh. Default is 0.0.
    segment_colors : dict, optional
        A dictionary mapping segment IDs to colors. Default is None, which will use the default colors.
    shader : str, optional
        The shader to use for rendering the skeletons if a skeleton source is provided. Default is None, which will use the default shader.
    pick: bool, optional
        Whether to allow cursor interaction with meshes and skeletons. Default is True.

    """

    name = field(default="seg", type=str)
    source = field(default=None, type=Union[str, Source])
    segments = field(
        factory=list,
        type=Union[list, dict, viewer_state.VisibleSegments],
        kw_only=True,
        repr=False,
    )
    color = field(default=None, type=list, kw_only=True, repr=False)
    hide_segment_zero = field(default=True, type=bool, kw_only=True, repr=False)
    selected_alpha = field(default=0.2, type=float, kw_only=True, repr=False)
    not_selected_alpha = field(default=0.0, type=float, kw_only=True, repr=False)
    alpha_3d = field(default=0.9, type=float, kw_only=True, repr=False)
    mesh_silhouette = field(default=0.0, type=float, kw_only=True, repr=False)
    segment_colors = field(default=None, type=dict, kw_only=True, repr=False)
    shader = field(default=None, type=str, kw_only=True, repr=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if isinstance(self.segments, DataMap):
            self._register_datamap(
                key=self.segments,
                priority=self.segments.priority,
                func=self.add_segments,
            )
            self.segments = list()
        self.color = parse_color(self.color)

    def to_neuroglancer_layer(self) -> viewer_state.SegmentationLayer:
        super().to_neuroglancer_layer()
        if self.shader is None:
            return viewer_state.SegmentationLayer(
                source=source_to_neuroglancer(self.source, resolution=self.resolution),
                visible_segments=segments_to_neuroglancer(self.segments),
                annotation_color=self.color,
                selected_alpha=self.selected_alpha,
                not_selected_alpha=self.not_selected_alpha,
                object_alpha=self.alpha_3d,
                segment_colors=self.segment_colors,
                mesh_silhouette_rendering=self.mesh_silhouette,
                pick=self.pick,
            )
        else:
            return viewer_state.SegmentationLayer(
                source=source_to_neuroglancer(self.source, resolution=self.resolution),
                visible_segments=segments_to_neuroglancer(self.segments),
                annotation_color=self.color,
                selected_alpha=self.selected_alpha,
                not_selected_alpha=self.not_selected_alpha,
                object_alpha=self.alpha_3d,
                segment_colors=self.segment_colors,
                mesh_silhouette_rendering=self.mesh_silhouette,
                skeleton_shader=self.shader,
                pick=self.pick,
            )

    def apply_to_neuroglancer(self, viewer) -> viewer_state.SegmentationLayer:
        "Can be a viewer or a viewer.txn()-context state"
        return self._apply_to_neuroglancer(viewer)

    def add_from_client(
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
        if isinstance(segments, DataMap):
            self._register_datamap(
                key=segments,
                func=self.add_segments,
                visible=visible,
            )
            return self
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

    def add_segments_from_data(
        self,
        data: Union[pd.DataFrame, DataMap],
        segment_column: str,
        visible_column: str = None,
        color_column: str = None,
    ) -> Self:
        """Create a SegmentationLayer from a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
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
        if isinstance(data, DataMap):
            self._register_datamap(
                key=data,
                func=self.segments_from_dataframe,
                segment_column=segment_column,
                visible_column=visible_column,
                color_column=color_column,
            )
            return self
        segments = data[segment_column].values
        if visible_column is not None:
            segments = {s: v for s, v in zip(segments, data[visible_column].values)}
        if color_column is not None:
            self.add_segment_colors(
                {s: parse_color(c) for s, c in zip(segments, data[color_column].values)}
            )

        self.add_segments(segments)
        return self

    def add_segment_properties(
        self,
        data: Union[pd.DataFrame, DataMap],
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
        data : pd.DataFrame
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
        if isinstance(data, DataMap):
            self._register_datamap(
                key=data,
                func=self.segment_properties,
                client=client,
                id_column=id_column,
                label_column=label_column,
                description_column=description_column,
                string_columns=string_columns,
                number_columns=number_columns,
                tag_value_columns=tag_value_columns,
                tag_bool_columns=tag_bool_columns,
                tag_descriptions=tag_descriptions,
                allow_disambiguation=allow_disambiguation,
                label_separator=label_separator,
                label_format_map=label_format_map,
                prepend_col_name=prepend_col_name,
                random_columns=random_columns,
                random_column_prefix=random_column_prefix,
            )
        segprops = SegmentProperties.from_dataframe(
            df=data,
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
                prop_id, format_properties=True, target_site="spelunker"
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
class AnnotationLayer(LayerWithSource):
    name = field(default="anno", type=str)
    source = field(default=None, type=Union[str, Source])
    resolution = field(default=None, type=list, kw_only=True, repr=False)
    color = field(default=None, type=list, kw_only=True, repr=False)
    annotations = field(factory=list, type=list, kw_only=True, repr=False)
    shader = field(default=None, type=str, kw_only=True, repr=False)
    tags = field(factory=list, type=list, kw_only=True, repr=False)
    filter_by_segmentation = field(
        default=False, type=Union[str, bool, list], kw_only=True, repr=False
    )
    linked_segmentation = field(
        default=None, type=Union[str, dict], kw_only=True, repr=False
    )
    set_position = field(default=True, type=bool, kw_only=True, repr=False)
    swap_visible_segments_on_move = field(
        default=True, type=bool, kw_only=True, repr=False
    )

    def __attrs_post_init__(self):
        if self.source is not None:
            super().__attrs_post_init__()
        if self.name is None:
            self.name = "anno"
        if self.tags is None:
            self.tags = []

    def _to_neuroglancer_layer_local(self) -> viewer_state.LocalAnnotationLayer:
        if len(self.tags) > MAX_TAG_COUNT:
            raise ValueError(
                f"Too many tags. Only {MAX_TAG_COUNT} distinct tags are allowed and {len(self.tags)} have been provided."
            )

        tag_map = {t: i for i, t in enumerate(self.tags)}
        props = make_annotation_properties(self.tags, tag_base_number=0)
        bindings = make_bindings(props)
        if not isinstance(self.resolution, CoordSpace):
            self.resolution = CoordSpace(resolution=self.resolution)
        if self.shader is None:
            return viewer_state.LocalAnnotationLayer(
                dimensions=self.resolution.to_neuroglancer(),
                annotation_color=self.color,
                annotations=_handle_annotations(
                    self.annotations, tag_map, self.resolution.resolution
                ),
                linked_segmentation_layer=_handle_linked_segmentation(
                    self.linked_segmentation
                ),
                annotation_properties=props,
                tool_bindings=bindings,
                swap_visible_segments_on_move=self.swap_visible_segments_on_move,
            )
        else:
            return viewer_state.LocalAnnotationLayer(
                dimensions=self.resolution.to_neuroglancer(),
                annotation_color=self.color,
                annotations=_handle_annotations(
                    self.annotations, tag_map, self.resolution.resolution
                ),
                linked_segmentation_layer=_handle_linked_segmentation(
                    self.linked_segmentation
                ),
                shader=self.shader,
                annotation_properties=props,
                tool_bindings=bindings,
                swap_visible_segments_on_move=self.swap_visible_segments_on_move,
            )

    def _to_neuroglancer_layer_cloud(self) -> viewer_state.AnnotationLayer:
        return viewer_state.AnnotationLayer(
            source=source_to_neuroglancer(self.source),
            annotation_color=self.color,
            linked_segmentation_layer=_handle_linked_segmentation(
                self.linked_segmentation
            ),
            filter_by_segmentation=_handle_filter_by_segmentation(
                self.filter_by_segmentation,
                self.linked_segmentation,
            ),
            shader=self.shader,
            swap_visible_segments_on_move=self.swap_visible_segments_on_move,
        )

    def to_neuroglancer_layer(
        self,
    ) -> Union[viewer_state.AnnotationLayer, viewer_state.LocalAnnotationLayer]:
        super().to_neuroglancer_layer()
        if self.source is None:
            return self._to_neuroglancer_layer_local()
        else:
            return self._to_neuroglancer_layer_cloud()

    def apply_to_neuroglancer(
        self,
        viewer,
    ):
        "Can be a viewer or a viewer.txn()-context state"
        if self.source is None:
            if self.resolution is None:
                raise ValueError(
                    f"Resolution for annotation layer '{self.name}' must be set before converting to Neuroglancer"
                )
            if self.linked_segmentation is True:
                for l in viewer.layers:
                    if l.type == "segmentation":
                        self.linked_segmentation = l.name
                        break
                else:
                    raise ValueError(
                        "No SegmentationLayer found in viewer to link to. Please set linked_segmentation manually."
                    )
            if self.set_position:
                if len(self.annotations) > 0:
                    if not isinstance(self.resolution, CoordSpace):
                        self.resolution = CoordSpace(resolution=self.resolution)
                    first_anno = self.annotations[0]
                    scale = viewer.dimensions.scales * 10**9
                    if isinstance(first_anno, PointAnnotation):
                        new_position = _handle_annotations([first_anno], {}, scale)[
                            0
                        ].point
                    elif isinstance(first_anno, LineAnnotation) or isinstance(
                        first_anno, BoundingBoxAnnotation
                    ):
                        new_position = _handle_annotations(
                            [self.annotations[0]],
                            {},
                            scale,
                        )[0].point_a
                    elif isinstance(self.annotations[0], EllipsoidAnnotation):
                        new_position = _handle_annotations(
                            [self.annotations[0]],
                            {},
                            scale,
                        )[0].center
                    viewer.position = new_position
        self._apply_to_neuroglancer(viewer)

    def set_linked_segmentation(
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
        self.linked_segmentation = layer
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

    def _handle_annotation_details(
        self,
        df,
        segment_column,
        description_column,
        tag_column,
        tag_bools,
    ):
        if segment_column is not None:
            segments = df[segment_column].values
        else:
            segments = [None] * len(df)
        if description_column is not None:
            descriptions = df[description_column].values
        else:
            descriptions = [None] * len(df)

        # Handle tags for annotations
        tag_list = [[] for _ in range(len(df))]
        if tag_column is not None:
            for ii, t in enumerate(df[tag_column].values):
                tag_list[ii].extend([t])
        if tag_bools is not None:
            for tag_ in tag_bools:
                row_to_add = np.arange(len(df))[df[tag_]]
                for i in row_to_add:
                    tag_list[i].extend([tag_])

        # Update annotation layer tag list with new unique values
        all_tags = []
        if tag_column:
            all_tags.extend(df[tag_column].unique().tolist())
        if tag_bools:
            all_tags.extend(tag_bools)
        self.tags.extend(sorted(set([t for t in all_tags if t not in self.tags])))

        return segments, descriptions, tag_list

    def add_points(
        self,
        data: Union[pd.DataFrame, np.ndarray, DataMap],
        point_column: Optional[str] = None,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        tag_bools: Optional[list] = None,
        data_resolution: Optional[list] = None,
    ) -> Self:
        """Add point annotations to the layer.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            The DataFrame containing the point annotations or an Nx3 array.
            If using an array, there is no way to set segment IDs, descriptions, or tags.
        point_column : str
            The column name for the point coordinates.
            Can be a prefix for a split column name with _x, _y, _z as suffixes.
        segment_column : str, optional
            The column name for the segment IDs. Default is None.
        description_column : str, optional
            The column name for the segment descriptions. Default is None.
        tag_column : str, optional
            The column name for segment descripion. Default is None.
        tag_column: str, optional
            The column name for the tags. Default is None.
            Row values should be strings. Nones and NAs will be skipped.
        tag_bools: list, optional
            List of column names to treat as a tag, with values being booleans
            indicating whether the tag is present or not. Default is None.
        data_resolution : list, optional
            The resolution of the data. If None, follows from layer resolution.
            Default is None.

        Returns
        -------
        SegmentationLayer
            The SegmentationLayer object with added point annotations.
        """
        if isinstance(data, DataMap):
            self._register_datamap(
                key=data,
                func=self.add_points,
                point_column=point_column,
                segment_column=segment_column,
                description_column=description_column,
                tag_column=tag_column,
                tag_bools=tag_bools,
                data_resolution=data_resolution,
            )
            return self
        if isinstance(data, pd.DataFrame):
            point_column = split_point_columns(point_column, data.columns)
            points = data[point_column].values

            segments, descriptions, tag_list = self._handle_annotation_details(
                data,
                segment_column,
                description_column,
                tag_column,
                tag_bools,
            )
        else:
            points = np.array(data).reshape(-1, 3)
            segments = [None] * len(points)
            descriptions = [None] * len(points)
            tag_list = [None] * len(points)

        self.add_annotations(
            [
                PointAnnotation(
                    point=p,
                    segments=seg,
                    description=d,
                    tags=t,
                    resolution=data_resolution,
                )
                for p, seg, d, t in zip(points, segments, descriptions, tag_list)
            ]
        )
        return self

    def add_lines(
        self,
        data: pd.DataFrame,
        point_a_column: str,
        point_b_column: str,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        tag_bools: Optional[list] = None,
        data_resolution: Optional[list] = None,
    ) -> Self:
        """Add point annotations to the layer.

        Parameters
        ----------
        data : pd.DataFrame
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
        tag_column: str, optional
            The column name for the tags. Default is None.
            Row values should be strings. Nones and NAs will be skipped.
        tag_bools: list, optional
            List of column names to treat as a tag, with values being booleans
            indicating whether the tag is present or not. Default is None.
        data_resolution : list, optional
            The resolution of the data. If None, follows from layer resolution.
            Default is None.

        Returns
        -------
        SegmentationLayer
            The SegmentationLayer object with added point annotations.
        """
        if isinstance(data, DataMap):
            self._register_datamap(
                key=data,
                func=self.add_lines,
                point_a_column=point_a_column,
                point_b_column=point_b_column,
                segment_column=segment_column,
                description_column=description_column,
                tag_column=tag_column,
                tag_bools=tag_bools,
                data_resolution=data_resolution,
            )
            return self
        if isinstance(data, pd.DataFrame):
            point_a_column = split_point_columns(point_a_column, data.columns)
            point_b_column = split_point_columns(point_b_column, data.columns)

            points_a = data[point_a_column].values
            points_b = data[point_b_column].values

            segments, descriptions, tag_list = self._handle_annotation_details(
                data,
                segment_column,
                description_column,
                tag_column,
                tag_bools,
            )
        else:
            data_a, data_b = data
            p_a = np.array(data_a).reshape(-1, 3)
            p_b = np.array(data_b).reshape(-1, 3)
            if len(p_a) != len(p_b):
                raise ValueError(
                    "Point A and Point B arrays must have the same length."
                )
            segments = [None] * len(p_a)
            descriptions = [None] * len(p_a)
            tag_list = [None] * len(p_a)

        self.add_annotations(
            [
                LineAnnotation(
                    pointA=p_a,
                    pointB=p_b,
                    segments=seg,
                    description=d,
                    tags=t,
                    resolution=data_resolution,
                )
                for p_a, p_b, seg, d, t in zip(
                    points_a, points_b, segments, descriptions, tag_list
                )
            ]
        )
        return self

    def add_ellipsoids(
        self,
        data: Union[pd.DataFrame, DataMap],
        center_column: str,
        radii_column: str,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        tag_bools: Optional[list] = None,
        data_resolution: Optional[list] = None,
    ) -> Self:
        """Add point annotations to the layer.

        Parameters
        ----------
        data : pd.DataFrame
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
        tag_column: str, optional
            The column name for the tags. Default is None.
            Row values should be strings. Nones and NAs will be skipped.
        tag_bools: list, optional
            List of column names to treat as a tag, with values being booleans
            indicating whether the tag is present or not. Default is None.
        data_resolution : list, optional
            The resolution of the data. If None, follows from layer resolution.
            Default is None.

        Returns
        -------
        SegmentationLayer
            The SegmentationLayer object with added point annotations.
        """
        if isinstance(data, DataMap):
            self._register_datamap(
                key=data,
                func=self.add_ellipsoids,
                center_column=center_column,
                radii_column=radii_column,
                segment_column=segment_column,
                description_column=description_column,
                tag_column=tag_column,
                tag_bools=tag_bools,
                data_resolution=data_resolution,
            )
            return self

        center_column = split_point_columns(center_column, data.columns)
        radii_column = split_point_columns(radii_column, data.columns)

        centers = data[center_column].values
        radii_vals = data[radii_column].values

        segments, descriptions, tag_list = self._handle_annotation_details(
            data,
            segment_column,
            description_column,
            tag_column,
            tag_bools,
        )

        self.add_annotations(
            [
                EllipsoidAnnotation(
                    center=c,
                    radii=r,
                    segments=seg,
                    description=d,
                    tags=t,
                    resolution=data_resolution,
                )
                for c, r, seg, d, t in zip(
                    centers, radii_vals, segments, descriptions, tag_list
                )
            ]
        )
        return self

    def add_boxes(
        self,
        data: pd.DataFrame,
        point_a_column: str,
        point_b_column: str,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        tag_bools: Optional[list] = None,
        data_resolution: Optional[list] = None,
    ) -> Self:
        """Add point annotations to the layer.

        Parameters
        ----------
        data : pd.DataFrame
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
        tag_column: str, optional
            The column name for the tags. Default is None.
            Row values should be strings. Nones and NAs will be skipped.
        tag_bools: list, optional
            List of column names to treat as a tag, with values being booleans
            indicating whether the tag is present or not. Default is None.
        data_resolution : list, optional
            The resolution of the data. If None, follows from layer resolution.
            Default is None.

        Returns
        -------
        SegmentationLayer
            The SegmentationLayer object with added point annotations.
        """
        if isinstance(data, DataMap):
            self._register_datamap(
                key=data,
                func=self.add_boxes,
                point_a_column=point_a_column,
                point_b_column=point_b_column,
                segment_column=segment_column,
                description_column=description_column,
                tag_column=tag_column,
                tag_bools=tag_bools,
                data_resolution=data_resolution,
            )
            return self

        point_a_column = split_point_columns(point_a_column, data.columns)
        point_b_column = split_point_columns(point_b_column, data.columns)

        points_a = data[point_a_column].values
        points_b = data[point_b_column].values

        segments, descriptions, tag_list = self._handle_annotation_details(
            data,
            segment_column,
            description_column,
            tag_column,
            tag_bools,
        )
        self.add_annotations(
            [
                BoundingBoxAnnotation(
                    point_a=p_a,
                    point_b=p_b,
                    segments=seg,
                    description=d,
                    tags=t,
                    resolution=data_resolution,
                )
                for p_a, p_b, seg, d, t in zip(
                    points_a, points_b, segments, descriptions, tag_list
                )
            ]
        )
        return self

    @property
    def tag_map(self) -> dict:
        return {tag: f"tagTool_{ii}" for ii, tag in enumerate(self.tags)}

    def add_shader(self, shader: str) -> Self:
        """Add a shader to the layer.

        Parameters
        ----------
        shader : str
            The shader to add.

        """
        self.shader = shader
        return self
