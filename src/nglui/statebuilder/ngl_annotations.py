from __future__ import annotations

import copy
import re
import warnings

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
        tool_type = f"tagTool_tag{self.tag_num}"
        if tool_type in viewer_state.tool_types:
            return

        @viewer_state.export_tool
        class TagTool(viewer_state.Tool):
            __slots__ = ()
            TOOL_TYPE = tool_type


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
        self,
        NglAnnotation,
        tag_map=dict(),
        layer_resolution=None,
        tags=None,
        strategy=None,
    ) -> dict:
        anno = copy.deepcopy(self)
        if strategy is not None and tags is not None:
            anno.props = strategy.encode(anno, tags)
        elif tag_map:
            anno.set_tags(tag_map)
        if layer_resolution is not None and self.resolution is not None:
            anno.scale_points(layer_resolution)
        return NglAnnotation(**asdict(anno, filter=_annotation_filter))


@define
class PointAnnotation(AnnotationBase):
    point = field(type=list, converter=strip_numpy_types)

    def _scale_points(self, scale):
        self.point = strip_numpy_types(np.array(self.point) * scale)

    def to_neuroglancer(
        self, tag_map=dict(), layer_resolution=None, tags=None, strategy=None
    ) -> dict:
        return self._to_neuroglancer(
            viewer_state.PointAnnotation,
            tag_map=tag_map,
            layer_resolution=layer_resolution,
            tags=tags,
            strategy=strategy,
        )


@define
class LineAnnotation(AnnotationBase):
    pointA = field(type=list, converter=strip_numpy_types)
    pointB = field(type=list, converter=strip_numpy_types)

    def _scale_points(self, scale):
        self.pointA = strip_numpy_types(np.array(self.pointA) * scale)
        self.pointB = strip_numpy_types(np.array(self.pointB) * scale)

    def to_neuroglancer(
        self, tag_map=dict(), layer_resolution=None, tags=None, strategy=None
    ) -> dict:
        return self._to_neuroglancer(
            viewer_state.LineAnnotation,
            tag_map=tag_map,
            layer_resolution=layer_resolution,
            tags=tags,
            strategy=strategy,
        )


@define
class EllipsoidAnnotation(AnnotationBase):
    center = field(type=list, converter=strip_numpy_types)
    radii = field(type=list, converter=strip_numpy_types)

    def _scale_points(self, scale):
        self.center = strip_numpy_types(np.array(self.center) * scale)
        self.radii = strip_numpy_types(np.array(self.radii) * scale)

    def to_neuroglancer(
        self, tag_map=dict(), layer_resolution=None, tags=None, strategy=None
    ) -> dict:
        return self._to_neuroglancer(
            viewer_state.EllipsoidAnnotation,
            tag_map=tag_map,
            layer_resolution=layer_resolution,
            tags=tags,
            strategy=strategy,
        )


@define
class BoundingBoxAnnotation(AnnotationBase):
    pointA = field(type=list, converter=strip_numpy_types)
    pointB = field(type=list, converter=strip_numpy_types)

    def _scale_points(self, scale):
        self.pointA = strip_numpy_types(np.array(self.pointA) * scale)
        self.pointB = strip_numpy_types(np.array(self.pointB) * scale)

    def to_neuroglancer(
        self, tag_map=dict(), layer_resolution=None, tags=None, strategy=None
    ) -> dict:
        return self._to_neuroglancer(
            viewer_state.AxisAlignedBoundingBoxAnnotation,
            tag_map=tag_map,
            layer_resolution=layer_resolution,
            tags=tags,
            strategy=strategy,
        )


@define
class PolylineAnnotation(AnnotationBase):
    points = field(type=list, converter=list_of_lists)

    def _scale_points(self, scale):
        self.points = strip_numpy_types(np.array(self.points) * scale)

    def to_neuroglancer(
        self, tag_map=dict(), layer_resolution=None, tags=None, strategy=None
    ) -> dict:
        return self._to_neuroglancer(
            viewer_state.PolyLineAnnotation,
            tag_map=tag_map,
            layer_resolution=layer_resolution,
            tags=tags,
            strategy=strategy,
        )


# --- Annotation property tools (main Neuroglancer) -----------------------------

TOGGLE_BOOL_PROPERTY_TOOL = "toggleBoolProperty"
ANNOTATE_ENUM_PROPERTY_TOOL = "annotateEnumProperty"
ANNOTATE_NUMBER_PROPERTY_TOOL = "annotateNumberProperty"
SELECT_PREVIOUS_ANNOTATION_TOOL = "selectPreviousAnnotation"
SELECT_NEXT_ANNOTATION_TOOL = "selectNextAnnotation"

_PROPERTY_TOOL_TYPES = (
    TOGGLE_BOOL_PROPERTY_TOOL,
    ANNOTATE_ENUM_PROPERTY_TOOL,
    ANNOTATE_NUMBER_PROPERTY_TOOL,
)
_SELECTION_TOOL_TYPES = (
    SELECT_PREVIOUS_ANNOTATION_TOOL,
    SELECT_NEXT_ANNOTATION_TOOL,
)


def register_property_tools() -> None:
    """Register main Neuroglancer's annotation property tools with the local library.

    ``viewer_state`` validates tool bindings against a registry and raises KeyError on
    an unknown type, so these must exist before they can be bound. Registration is
    guarded, so it becomes a no-op once a released neuroglancer package ships the real
    classes and the library's own definitions take precedence.
    """
    for tool_type in _PROPERTY_TOOL_TYPES:
        if tool_type in viewer_state.tool_types:
            continue
        viewer_state.export_tool(
            type(
                f"_{tool_type}Tool",
                (viewer_state.Tool,),
                {
                    "__slots__": (),
                    "TOOL_TYPE": tool_type,
                    "property": viewer_state.wrapped_property("property", str),
                },
            )
        )
    for tool_type in _SELECTION_TOOL_TYPES:
        if tool_type in viewer_state.tool_types:
            continue
        viewer_state.export_tool(
            type(
                f"_{tool_type}Tool",
                (viewer_state.Tool,),
                {"__slots__": (), "TOOL_TYPE": tool_type},
            )
        )


register_property_tools()


# --- Property identifiers ------------------------------------------------------

PROPERTY_ID_PATTERN = re.compile(r"^[a-z][a-zA-Z0-9_]*$")
"""Neuroglancer's `parseAnnotationPropertyId`, which throws on anything else.

A rejected identifier fails the whole layer, not just the property, so nglui has to
enforce this even though the Python library does not.
"""


def sanitize_property_id(label: str, fallback_index: int = 0) -> str:
    """Convert a tag label into a valid Neuroglancer annotation property identifier.

    Mirrors the transformation the Neuroglancer annotation schema tab applies to
    typed input, so an nglui-generated identifier matches what a user would get by
    entering the same string in the viewer. It then adds the step the viewer omits:
    requiring a leading letter, without which an id like ``1st_pass`` is accepted on
    entry but rejected when the state is reloaded.

    The identifier is what a user actually sees -- the schema tab labels each property
    by its id -- so this stays readable rather than producing an opaque slug.

    Parameters
    ----------
    label : str
        Human-readable tag label.
    fallback_index : int, optional
        Index used to build a placeholder if the label has no usable characters.

    Returns
    -------
    str
        An identifier matching `PROPERTY_ID_PATTERN`.

    Examples
    --------
    >>> sanitize_property_id("Cell Body")
    'cell_body'
    >>> sanitize_property_id("post-synaptic")
    'post_synaptic'
    >>> sanitize_property_id("1st pass")
    'tag_1st_pass'
    """
    cleaned = re.sub(r"[-\s]+", "_", str(label)).lower()
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    cleaned = cleaned.strip("_")
    if not cleaned:
        return f"tag_{fallback_index}"
    if not cleaned[0].isalpha():
        cleaned = f"tag_{cleaned}"
    return cleaned


def unique_tags(tags: list) -> list:
    """Drop repeated tags, keeping first-seen order.

    A repeated tag cannot be represented on the wire under either encoding: it would
    mean two properties with the same identifier, which Neuroglancer rejects outright,
    and the per-annotation value vector has one slot per distinct tag regardless.
    Tags discovered from a dataframe are already distinct; this guards the case where
    a caller supplies `tags` directly.

    Parameters
    ----------
    tags : list
        Tag labels, possibly with repeats.

    Returns
    -------
    list
        The same labels, first occurrence only.

    Examples
    --------
    >>> unique_tags(["axon", "axon", "soma"])
    ['axon', 'soma']
    """
    seen = {}
    for tag in tags or []:
        seen.setdefault(tag, None)
    return list(seen)


def build_property_ids(
    tags: list,
    tag_ids: dict = None,
    strict: bool = False,
) -> dict:
    """Map tag labels to unique, valid annotation property identifiers.

    Parameters
    ----------
    tags : list of str
        Ordered tag labels.
    tag_ids : dict, optional
        Explicit ``{label: property_id}`` overrides. Identifiers given here are
        validated but never rewritten.
    strict : bool, optional
        If True, raise instead of sanitizing a label that is not already a valid
        identifier. Default is False.

    Returns
    -------
    dict
        Mapping of tag label to property identifier, in the order of ``tags``.

    Raises
    ------
    ValueError
        If ``strict`` is set and a label is not a valid identifier, or if an
        explicitly supplied identifier is invalid.
    """
    tag_ids = tag_ids or {}
    assigned = {}
    renamed = {}
    used = set()
    for index, tag in enumerate(tags):
        if tag in tag_ids:
            candidate = str(tag_ids[tag])
            if not PROPERTY_ID_PATTERN.match(candidate):
                raise ValueError(
                    f"Property id {candidate!r} for tag {tag!r} is not valid. "
                    f"Ids must match {PROPERTY_ID_PATTERN.pattern}."
                )
        else:
            candidate = sanitize_property_id(tag, fallback_index=index)
            if candidate != tag:
                if strict:
                    raise ValueError(
                        f"Tag {tag!r} is not a valid annotation property id "
                        f"(must match {PROPERTY_ID_PATTERN.pattern}). Pass "
                        f"strict_property_ids=False to sanitize automatically, or "
                        f"supply an explicit id."
                    )
                renamed[tag] = candidate
        # Neuroglancer rejects duplicate property ids outright.
        unique = candidate
        suffix = 2
        while unique in used:
            unique = f"{candidate}_{suffix}"
            suffix += 1
        if unique != candidate:
            renamed[tag] = unique
        used.add(unique)
        assigned[tag] = unique
    if renamed:
        # One warning per layer, not one per tag: the whole rename map at once is
        # what makes it actionable.
        warnings.warn(
            "Some tag labels are not valid Neuroglancer annotation property ids and "
            f"were renamed: {renamed}. The id is what the viewer displays, so pass "
            "explicit ids via `tag_ids` if these are not what you want.",
            stacklevel=3,
        )
    return assigned


# --- Tag encoding strategies ---------------------------------------------------

DEFAULT_TAG_BINDINGS = ["Q", "W", "E", "R", "T", "A", "S", "D", "F", "G"]
"""Keys bound to tag tools, in order. Neuroglancer requires single capital letters."""


@define
class TagStrategy:
    """How a set of tag labels becomes Neuroglancer annotation properties.

    The two dialects differ in every part of the encoding -- the property spec, the
    tool binding, and the per-annotation value -- so they are kept as whole strategies
    rather than as branches inside the serializer.
    """

    def property_specs(self, tags: list, **kwargs) -> list:
        """Build the layer's ``annotationProperties`` list."""
        raise NotImplementedError

    def tool_bindings(self, specs: list) -> dict:
        """Build the layer's ``toolBindings`` map."""
        raise NotImplementedError

    def encode(self, annotation, tags: list) -> list:
        """Build one annotation's ``props`` array.

        Must return exactly one entry per tag: Neuroglancer reads ``props``
        positionally against ``annotationProperties`` and rejects an array of any
        other length.
        """
        raise NotImplementedError


@define
class LegacyTagStrategy(TagStrategy):
    """seung-lab / Spelunker encoding.

    Tags are ``uint8`` properties named ``tag0..tagN`` carrying a non-standard ``tag``
    key that holds the label, bound to ``tagTool_tagN`` tool types. Main Neuroglancer
    parses these without error but ignores the ``tag`` key, so the tags survive a
    modern viewer as unlabeled columns.
    """

    def property_specs(self, tags: list, **kwargs) -> list:
        if len(tags) > MAX_TAG_COUNT:
            raise ValueError(
                f"The Spelunker tag encoding supports at most {MAX_TAG_COUNT} tags per "
                f"layer and {len(tags)} were provided, because it binds each tag to one "
                f"of {MAX_TAG_COUNT} dedicated tools. Deployments tracking Neuroglancer "
                "main encode tags as boolean annotation properties and have no such "
                "limit; pass capabilities='main' if the target supports them."
            )
        return make_annotation_properties(tags, tag_base_number=0)

    def tool_bindings(self, specs: list) -> dict:
        return make_bindings(specs)

    def encode(self, annotation, tags: list) -> list:
        annotation.set_tags({tag: ii for ii, tag in enumerate(tags)})
        return annotation.props


@define
class BoolPropertyStrategy(TagStrategy):
    """Main Neuroglancer encoding.

    Tags are annotation properties of ``type: "bool"``, bound to the generic
    ``toggleBoolProperty`` tool. Unlike the legacy encoding this is not backward
    compatible: a viewer that predates bool properties fails to parse the entire
    layer, which is why it is only used when the target is known to support it.
    """

    def property_specs(
        self,
        tags: list,
        tag_ids: dict = None,
        strict_property_ids: bool = False,
        **kwargs,
    ) -> list:
        ids = build_property_ids(tags, tag_ids=tag_ids, strict=strict_property_ids)
        specs = []
        for tag in tags:
            spec = {"id": ids[tag], "type": "bool"}
            if ids[tag] != tag:
                # The schema tab labels properties by id and shows the description
                # only as a tooltip, so this preserves the original without relying
                # on it for display.
                spec["description"] = str(tag)
            specs.append(spec)
        return specs

    def tool_bindings(self, specs: list) -> dict:
        if len(specs) > len(DEFAULT_TAG_BINDINGS):
            warnings.warn(
                f"Only the first {len(DEFAULT_TAG_BINDINGS)} of {len(specs)} tags get "
                "keyboard shortcuts; the rest remain usable from the annotation tab.",
                stacklevel=2,
            )
        return {
            key: {"type": TOGGLE_BOOL_PROPERTY_TOOL, "property": spec["id"]}
            for key, spec in zip(DEFAULT_TAG_BINDINGS, specs)
        }

    def encode(self, annotation, tags: list) -> list:
        labels = set(annotation.tags or [])
        # Full-length vector: Neuroglancer requires len(props) == len(properties).
        return [tag in labels for tag in tags]


LEGACY_TAG_STRATEGY = LegacyTagStrategy()
BOOL_PROPERTY_TAG_STRATEGY = BoolPropertyStrategy()


def strategy_for_capabilities(capabilities) -> TagStrategy:
    """Choose a tag encoding for a deployment's capabilities.

    Parameters
    ----------
    capabilities : Capabilities
        Capabilities of the target Neuroglancer deployment.

    Returns
    -------
    TagStrategy
        `BOOL_PROPERTY_TAG_STRATEGY` if bool annotation properties are supported,
        otherwise `LEGACY_TAG_STRATEGY`.
    """
    if capabilities is None:
        from .capabilities import get_default_capabilities

        capabilities = get_default_capabilities()
    if capabilities.annotation_bool_properties:
        return BOOL_PROPERTY_TAG_STRATEGY
    return LEGACY_TAG_STRATEGY
