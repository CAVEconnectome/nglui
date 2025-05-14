from __future__ import annotations

import json
from typing import Iterable, Literal, Optional, Self, Union

import attrs
import caveclient
import numpy as np
import pandas as pd
from attrs import asdict, define, field
from neuroglancer import Viewer, viewer_state
from neuroglancer.coordinate_space import CoordinateArray, CoordinateSpace, parse_unit
from neuroglancer.random_token import make_random_token

from ..segmentprops import SegmentProperties
from .utils import parse_color


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
TagToolFactory(24)


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
            attrs.asdict(AnnotationTag(id=f"tag{ii + tag_base_number}", tag=an))
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


@define
class AnnotationBase:
    id = field(default=None, type=str, kw_only=True)
    description = field(default=None, type=str, kw_only=True)
    linked_segmentation = field(default=None, type=list[int], kw_only=True)
    annotation_properties = field(default=None, type=list, kw_only=True)

    def __attrs_post_init__(self):
        if self.id is not None:
            self.id = make_random_token()
        self.linked_segmentation = [int(seg) for seg in self.linked_segmentation]


@define
class PointAnnotation(AnnotationBase):
    point = field(type=list)

    def __attrs_post_init__(self):
        self.point = list(self.point)

    def to_neuroglancer(self) -> dict:
        return viewer_state.PointAnnotation(**asdict(self))


@define
class LineAnnotation(AnnotationBase):
    pointA = field(type=list)
    pointB = field(type=list)

    def __attrs_post_init__(self):
        self.pointA = list(self.pointA)
        self.pointB = list(self.pointB)
        super().__attrs_post_init__()

    def to_neuroglancer(self) -> dict:
        return viewer_state.LineAnnotation(**asdict(self))


@define
class EllipsoidAnnotation:
    center = field(type=list)
    radii = field(type=list)

    def __attrs_post_init__(self):
        self.center = list(self.center)
        self.radii = list(self.radii)
        super().__attrs_post_init__()

    def to_neuroglancer(self) -> dict:
        return viewer_state.EllipsoidAnnotation(**asdict(self))


@define
class SphereAnnotation(AnnotationBase):
    center = field(type=list)
    radius = field(default=None, type=float)

    def __attrs_post_init__(self):
        self.center = list(self.center)
        self.radii = [self.radius] * len(self.center)
        super().__attrs_post_init__()

    def to_neuroglancer(self) -> dict:
        return viewer_state.EllipsoidAnnotation(**asdict(self))


@define
class BoundingBoxAnnotation(AnnotationBase):
    pointA = field(type=list)
    pointB = field(type=list)

    def __attrs_post_init__(self):
        self.min = list(self.min)
        self.max = list(self.max)
        super().__attrs_post_init__()

    def to_neuroglancer(self) -> dict:
        return viewer_state.AxisAlignedBoundingBoxAnnotation(**asdict(self))
