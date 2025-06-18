from __future__ import annotations

import copy

import attrs
import numpy as np
from attrs import asdict, define, field
from neuroglancer import viewer_state
from neuroglancer.random_token import make_random_token

from .utils import list_of_lists, list_of_strings, none_or_array, strip_numpy_types

LOCAL_ANNOTATION_SOURCE = "local://annotations"
MAX_TAG_COUNT = 10


@define
class TagTool:
    tag_num = field(type=int)

    def initialize_neuroglancer(self) -> None:
        @viewer_state.export_tool
        class TagTool(viewer_state.Tool):
            __slots__ = ()
            TOOL_TYPE = f"tagTool_tag{self.tag_num}"


def TagToolFactory(number_tags: int):
    for ii in range(number_tags):
        TagTool(ii).initialize_neuroglancer()


# Initialize tags tool
TagToolFactory(MAX_TAG_COUNT)


@define
class AnnotationProperty:
    id: str = None
    type: str = field(default="uint8")
    tag: str = None

    def to_neuroglancer(self) -> dict:
        return asdict(self)


@define
class AnnotationTag:
    id: int
    tag: str

    def to_neuroglancer(self, tag_base_number) -> dict:
        return AnnotationProperty(
            id=f"tag{self.id + tag_base_number}",
            tag=self.tag,
        ).to_neuroglancer()


def make_annotation_properties(annotations, tag_base_number=0):
    "Take a list of annotations and build up the dictionary needed"
    properties = []
    for ii, an in enumerate(annotations):
        properties.append(
            attrs.asdict(AnnotationProperty(id=f"tag{ii + tag_base_number}", tag=an))
        )
    return properties


def make_bindings(properties, bindings=None):
    "Make a dictinary describing key bindings for tags"
    if bindings is None:
        bindings = ["Q", "W", "E", "R", "T", "A", "S", "D", "F", "G"]
    if len(properties) > len(bindings):
        raise ValueError("Too many properties for bindings")
    tool_bindings = {}
    for bind, prop in zip(bindings, properties):
        tool_bindings[bind] = f"tagTool_{prop['id']}"
    return tool_bindings


def _annotation_filter(a, v):
    """
    Filter function to exclude certain attributes from the Neuroglancer serialization.
    This is used to prevent serialization of attributes that are not needed in the Neuroglancer viewer.
    """
    if a.name in ["tags", "resolution"]:
        return False
    return True


@define
class AnnotationBase:
    id = field(default=None, type=str, kw_only=True, converter=strip_numpy_types)
    description = field(default=None, type=str, kw_only=True)
    segments = field(
        default=None, type=list[int], kw_only=True, converter=list_of_lists
    )
    tags = field(factory=list, type=list[str], kw_only=True, converter=list_of_strings)
    resolution = field(default=None, type=list, kw_only=True, converter=none_or_array)
    props = field(factory=list, type=list, kw_only=True)

    def initialize_property_list(self, n_properties: int):
        self.props = [0] * n_properties

    def set_tag_id(self, tag_id: int):
        self.props[tag_id] = 1

    def set_tags(self, tag_map):
        self.initialize_property_list(len(tag_map))
        for tag in self.tags:
            if tag in tag_map:
                self.set_tag_id(tag_map[tag])

    def _scale_points(self, scale):
        # Implement per annotation type
        pass

    def scale_points(self, layer_resolution):
        scale = np.array(self.resolution) / np.array(layer_resolution)
        self._scale_points(scale)

    def __attrs_post_init__(self):
        if self.id is None:
            self.id = make_random_token()

    def _to_neuroglancer(
        self, NglAnnotation, tag_map=dict(), layer_resolution=None
    ) -> dict:
        anno = copy.deepcopy(self)
        if tag_map:
            anno.set_tags(tag_map)
        if layer_resolution is not None and self.resolution is not None:
            anno.scale_points(layer_resolution)
        return NglAnnotation(**asdict(anno, filter=_annotation_filter))


@define
class PointAnnotation(AnnotationBase):
    point = field(type=list, converter=strip_numpy_types)

    def _scale_points(self, scale):
        self.point = strip_numpy_types(self.point * scale)

    def to_neuroglancer(self, tag_map=dict(), layer_resolution=None) -> dict:
        return self._to_neuroglancer(
            viewer_state.PointAnnotation,
            tag_map=tag_map,
            layer_resolution=layer_resolution,
        )


@define
class LineAnnotation(AnnotationBase):
    pointA = field(type=list, converter=strip_numpy_types)
    pointB = field(type=list, converter=strip_numpy_types)

    def _scale_points(self, scale):
        self.pointA = strip_numpy_types(self.pointA * scale)
        self.pointB = strip_numpy_types(self.pointB * scale)

    def to_neuroglancer(self, tag_map=dict, layer_resolution=None) -> dict:
        return self._to_neuroglancer(
            viewer_state.LineAnnotation,
            tag_map=tag_map,
            layer_resolution=layer_resolution,
        )


@define
class EllipsoidAnnotation:
    center = field(type=list, converter=strip_numpy_types)
    radii = field(type=list, converter=strip_numpy_types)

    def _scale_points(self, scale):
        self.center = strip_numpy_types(self.center * scale)
        self.radii = strip_numpy_types(self.radii * scale)

    def to_neuroglancer(self, tag_map=dict, layer_resolution=None) -> dict:
        return self._to_neuroglancer(
            viewer_state.EllipsoidAnnotation,
            tag_map=tag_map,
            layer_resolution=layer_resolution,
        )


@define
class BoundingBoxAnnotation(AnnotationBase):
    pointA = field(type=list, converter=strip_numpy_types)
    pointB = field(type=list, converter=strip_numpy_types)

    def _scale_points(self, scale):
        self.pointA = strip_numpy_types(self.pointA * scale)
        self.pointB = strip_numpy_types(self.pointB * scale)

    def to_neuroglancer(self, tag_map=dict, layer_resolution=None) -> dict:
        return self._to_neuroglancer(
            viewer_state.AxisAlignedBoundingBoxAnnotation,
            tag_map=tag_map,
            layer_resolution=layer_resolution,
        )
