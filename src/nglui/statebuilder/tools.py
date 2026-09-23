"""Neuroglancer tools: key bindings and tool palettes.

A tool is an action a user activates with a key (``shift`` plus the bound letter) or
from a tool palette: placing annotations, selecting segments, or adjusting a layer
setting by dragging. Tools are bound when a `ViewerState` is built, so keys can be
allocated across the whole viewer: Neuroglancer keeps one key namespace for every
layer, and a key bound twice silently loses its earlier tool when the state loads.

Examples
--------
>>> from nglui.statebuilder import tools
>>> vs.add_tools(
...     tools.AnnotatePoint(layer="synapses", key="P"),
...     tools.ShaderControl(layer="img", control="normalized"),  # key auto-assigned
...     tools.SelectSegments(layer="seg"),
...     palette=tools.ToolPalette("Proofreading", side="right"),
... )
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, ClassVar, Optional, Union

import attrs
from neuroglancer import viewer_state

from .ngl_annotations import TOGGLE_BOOL_PROPERTY_TOOL

if TYPE_CHECKING:
    from .ngl_components import Layer

__all__ = [
    "Tool",
    "AnnotatePoint",
    "AnnotateLine",
    "AnnotateBoundingBox",
    "AnnotateEllipsoid",
    "AnnotatePolyline",
    "SelectSegments",
    "MergeSegments",
    "SplitSegments",
    "ShaderControl",
    "LayerSetting",
    "Dimension",
    "ToolPalette",
    "LAYER_SETTINGS",
]

_KEY_PATTERN = re.compile(r"^[A-Z]$")

# Keys handed out when a tool does not name one. Tag tools take Q W E R T A S D F G,
# so these start from the other rows to leave tags on the keys users expect.
_AUTO_KEY_ORDER = "ZXCVBNMYUIOPHJKLQWERTASDFG"
# Where a tag tool goes when its usual key is taken: on after the tag keys, so a
# second tagged layer's tags sit next to the first's.
_TAG_KEY_ORDER = "QWERTASDFGYUIOPHJKLZXCVBNM"

#: Layer settings Neuroglancer can bind to a key, adjusted by dragging while held.
LAYER_SETTINGS = (
    "opacity",
    "blend",
    "volumeRendering",
    "volumeRenderingGain",
    "volumeRenderingDepthSamples",
    "crossSectionRenderScale",
    "selectedAlpha",
    "notSelectedAlpha",
    "objectAlpha",
    "hideSegmentZero",
    "hoverHighlight",
    "baseSegmentColoring",
    "ignoreNullVisibleSet",
    "colorSeed",
    "segmentDefaultColor",
    "meshRenderScale",
    "meshSilhouetteRendering",
    "saturation",
    "skeletonRendering.mode2d",
    "skeletonRendering.mode3d",
    "skeletonRendering.lineWidth2d",
    "skeletonRendering.lineWidth3d",
)


def _validate_key(instance, attribute, value):
    if value is None or value is False:
        return
    if not isinstance(value, str) or not _KEY_PATTERN.match(value):
        raise ValueError(
            f"Tool key must be a single capital letter, got {value!r}. "
            "Neuroglancer activates it with shift plus that letter."
        )


def _layer_name(layer) -> Optional[str]:
    if layer is None or isinstance(layer, str):
        return layer
    return layer.name


@attrs.define
class Tool:
    """Base class for tools.

    Parameters
    ----------
    layer : str or Layer, optional
        The layer the tool acts on. Required for layer tools added with
        `ViewerState.add_tools`; omitted when the tool is given to a layer's own
        ``tools`` list.
    key : str or False, optional
        A single capital letter to bind the tool to. None assigns a free key;
        False binds no key, for a tool that should only appear in a palette.
    """

    layer: Optional[Union[str, "Layer"]] = attrs.field(
        default=None, kw_only=True, converter=_layer_name
    )
    key: Optional[Union[str, bool]] = attrs.field(
        default=None, kw_only=True, validator=_validate_key
    )

    #: Neuroglancer tool type
    tool_type: ClassVar[str] = ""
    #: Layer types this tool works on, or None for any layer
    layer_types: ClassVar[Optional[tuple[str, ...]]] = None
    #: Whether the tool is bound on a layer (True) or on the viewer (False)
    is_layer_tool: ClassVar[bool] = True

    def _options(self) -> dict:
        """Tool-specific JSON fields beyond ``type``."""
        return {}

    def to_json(self, layer_name: Optional[str] = None) -> dict:
        """The tool's JSON, with a ``layer`` field when given one (as palettes need)."""
        out = {"type": self.tool_type, **self._options()}
        if layer_name is not None:
            out["layer"] = layer_name
        return out


@attrs.define
class _AnnotationTool(Tool):
    layer_types: ClassVar = ("annotation",)


@attrs.define
class AnnotatePoint(_AnnotationTool):
    """Place point annotations."""

    tool_type: ClassVar[str] = "annotatePoint"


@attrs.define
class AnnotateLine(_AnnotationTool):
    """Place line annotations."""

    tool_type: ClassVar[str] = "annotateLine"


@attrs.define
class AnnotateBoundingBox(_AnnotationTool):
    """Place axis-aligned bounding box annotations."""

    tool_type: ClassVar[str] = "annotateBoundingBox"


@attrs.define
class AnnotateEllipsoid(_AnnotationTool):
    """Place ellipsoid annotations (Neuroglancer's ``annotateSphere`` tool)."""

    tool_type: ClassVar[str] = "annotateSphere"


@attrs.define
class AnnotatePolyline(_AnnotationTool):
    """Place polyline annotations.

    Newer than the other annotation tools; a deployment built before polyline
    support rejects the tool, which can drop the layer it is bound on.
    """

    tool_type: ClassVar[str] = "annotatePolyline"


@attrs.define
class _SegmentationTool(Tool):
    layer_types: ClassVar = ("segmentation",)


@attrs.define
class SelectSegments(_SegmentationTool):
    """Select or deselect segments by painting over them."""

    tool_type: ClassVar[str] = "selectSegments"


@attrs.define
class MergeSegments(_SegmentationTool):
    """Merge two segments (graphene/proofreading-enabled segmentations)."""

    tool_type: ClassVar[str] = "mergeSegments"


@attrs.define
class SplitSegments(_SegmentationTool):
    """Split a segment (graphene/proofreading-enabled segmentations)."""

    tool_type: ClassVar[str] = "splitSegments"


@attrs.define
class ShaderControl(Tool):
    """Adjust one of a layer's shader controls, e.g. image contrast.

    Parameters
    ----------
    control : str
        Name of the ``#uicontrol``; ``"normalized"`` is the default image shader's
        contrast control.
    """

    control: str = attrs.field(default="normalized", kw_only=True)
    tool_type: ClassVar[str] = "shaderControl"

    def _options(self) -> dict:
        return {"control": self.control}


@attrs.define
class LayerSetting(Tool):
    """Adjust a layer setting, such as opacity or mesh alpha, while the key is held.

    Parameters
    ----------
    setting : str
        One of `LAYER_SETTINGS`, e.g. ``"opacity"``, ``"objectAlpha"``, or
        ``"skeletonRendering.lineWidth3d"``.
    """

    setting: str = attrs.field(
        kw_only=True, validator=attrs.validators.in_(LAYER_SETTINGS)
    )

    @property
    def tool_type(self) -> str:  # type: ignore[override]
        return self.setting


@attrs.define
class Dimension(Tool):
    """Step through one dimension, e.g. z, while the key is held.

    A viewer-level tool: it acts on the viewer position, not on a layer.

    Parameters
    ----------
    dimension : str
        Name of the dimension, e.g. ``"z"``.
    """

    dimension: str = attrs.field(kw_only=True)
    tool_type: ClassVar[str] = "dimension"
    is_layer_tool: ClassVar[bool] = False

    def _options(self) -> dict:
        return {"dimension": self.dimension}


@attrs.define
class ToolPalette:
    """A named panel of tool buttons.

    Parameters
    ----------
    name : str
        The palette's title.
    tools : list of Tool, optional
        Tools shown in the palette. Tools can also be added with
        ``vs.add_tools(..., palette=...)``. Layer tools need their `layer` set.
    side : {"left", "right", "top", "bottom"}, optional
        Which edge of the viewer the palette docks to.
    size : int, optional
        Width (or height, for top/bottom) in pixels.
    visible : bool, optional
        Whether the palette is open. Default is True.
    query : str, optional
        A query that fills the palette with matching tools, in Neuroglancer's
        palette query syntax.

    Examples
    --------
    >>> ToolPalette("Proofreading", side="right")
    """

    name: str
    tools: list = attrs.field(factory=list, converter=list)
    side: Optional[str] = attrs.field(
        default=None,
        kw_only=True,
        validator=attrs.validators.optional(
            attrs.validators.in_(("left", "right", "top", "bottom"))
        ),
    )
    size: Optional[int] = attrs.field(default=None, kw_only=True)
    visible: Optional[bool] = attrs.field(default=True, kw_only=True)
    query: Optional[str] = attrs.field(default=None, kw_only=True)

    def to_json(self, state) -> dict:
        """The palette's JSON, checking each tool's layer against `state`."""
        entries = []
        for tool in self.tools:
            layer_name = tool.layer if tool.is_layer_tool else None
            _check_target(state, tool, layer_name)
            entries.append(tool.to_json(layer_name=layer_name))
        spec = {"tools": entries}
        for field in ("side", "size", "visible", "query"):
            value = getattr(self, field)
            if value is not None:
                spec[field] = value
        return spec


def bind_tools(state, requests: list, tag_layers: tuple = ()) -> None:
    """Bind tools onto a built neuroglancer state, allocating keys viewer-wide.

    Keys are claimed in order of how deliberately they were chosen:

    1. Bindings the state already has from a raw layer or base state, which are
       kept as they are. Duplicates among them are warned about, since the viewer
       will drop one.
    2. Tools with an explicit `key`, which must be free, or this raises.
    3. Tag tools that nglui generated on `tag_layers`, which keep their usual key
       when it is free and otherwise move to the next free letter.
    4. Tools with ``key=None``, which take the next free letter.

    Parameters
    ----------
    state : neuroglancer.ViewerState
        The state, with every layer already added.
    requests : list of (Tool, str or None)
        Each tool with the name of the layer it belongs to, if it came from a
        layer's own ``tools`` list.
    tag_layers : tuple of str
        Names of nglui annotation layers whose tag tool bindings may be moved.
    """
    fixed, tag_bindings, kept = _existing_bindings(state, tag_layers)
    used = dict(fixed)
    resolved = []
    for tool, owner in requests:
        layer_name = tool.layer or owner
        _check_target(state, tool, layer_name)
        resolved.append((tool, layer_name))

    # Final bindings for each layer (None is the viewer) that nglui rewrites
    new_bindings: dict = {}

    def claim(key, layer_name, value, description):
        used[key] = description
        new_bindings.setdefault(layer_name, {})[key] = value

    for tool, layer_name in resolved:
        if isinstance(tool.key, str):
            if tool.key in used:
                raise ValueError(
                    f"Tool key {tool.key!r} for {tool.tool_type} is already bound to "
                    f"{used[tool.key]}. Neuroglancer has one key namespace for the "
                    "whole viewer; choose another key or leave key=None to assign one."
                )
            target = layer_name if tool.is_layer_tool else None
            claim(tool.key, target, tool.to_json(), _describe(tool, layer_name))

    for layer_name, key, value in tag_bindings:
        if key in used:
            key = _next_free(used, _TAG_KEY_ORDER, "tag tools")
        claim(key, layer_name, value, f"a tag tool on layer '{layer_name}'")

    for tool, layer_name in resolved:
        if tool.key is None:
            key = _next_free(used, _AUTO_KEY_ORDER, "tools")
            target = layer_name if tool.is_layer_tool else None
            claim(key, target, tool.to_json(), _describe(tool, layer_name))

    # Rewrite each touched map in one assignment: assigning a key at a time turns
    # a legacy "tagTool_*" string into object form, changing the wire format.
    for layer_name, bindings in new_bindings.items():
        merged = {**kept.get(layer_name, {}), **bindings}
        if layer_name is None:
            state.tool_bindings = merged
        else:
            state.layers[layer_name].tool_bindings = merged
    for layer_name in tag_layers:
        # A tagged layer whose tag keys were all taken still needs its old ones cleared
        if layer_name not in new_bindings and layer_name in kept:
            state.layers[layer_name].tool_bindings = kept[layer_name]


def add_palettes(state, palettes: dict) -> None:
    """Write tool palettes onto a built neuroglancer state."""
    for name, palette in palettes.items():
        state.tool_palettes[name] = palette.to_json(state)


def _is_tag_tool(value) -> bool:
    tool_type = value if isinstance(value, str) else value.get("type", "")
    return tool_type.startswith("tagTool_") or tool_type == TOGGLE_BOOL_PROPERTY_TOOL


def _existing_bindings(state, tag_layers) -> tuple[dict, list, dict]:
    """Split the state's bindings into fixed keys, movable tag bindings, and the rest.

    Returns
    -------
    fixed : dict
        Key -> description of what holds it, for bindings that must not move.
    tag_bindings : list of (layer name, key, value)
        Tag tool bindings nglui generated, in layer order.
    kept : dict
        Layer name (None for the viewer) -> its non-tag bindings, as raw JSON.
    """
    # Read through JSON: touching the wrapped `tool_bindings` property creates an
    # empty map that would then be emitted as "toolBindings": {}.
    fixed, tag_bindings, kept = {}, [], {}
    viewer_bindings = state.to_json().get("toolBindings", {})
    if viewer_bindings:
        kept[None] = dict(viewer_bindings)
    for key in viewer_bindings:
        fixed[key] = "the viewer"
    for layer in state.layers:
        bindings = layer.to_json().get("toolBindings", {})
        movable = layer.name in tag_layers
        for key, value in bindings.items():
            if movable and _is_tag_tool(value):
                tag_bindings.append((layer.name, key, value))
                continue
            kept.setdefault(layer.name, {})[key] = value
            if key in fixed:
                warnings.warn(
                    f"Key {key!r} is bound both on {fixed[key]} and on layer "
                    f"'{layer.name}'. Neuroglancer keeps one binding per key, so one "
                    "of these tools will be unbound when the state loads.",
                    stacklevel=4,
                )
            fixed[key] = f"layer '{layer.name}'"
        if movable:
            kept.setdefault(layer.name, {})
    return fixed, tag_bindings, kept


def _next_free(used: dict, order: str, what: str) -> str:
    key = next((k for k in order if k not in used), None)
    if key is None:
        raise ValueError(
            f"Every key A-Z is in use, so no key is left for the {what}. "
            "Set key=False on some tools, or use fewer tags."
        )
    return key


def _check_target(state, tool: Tool, layer_name: Optional[str]) -> None:
    if not tool.is_layer_tool:
        return
    if layer_name is None:
        raise ValueError(
            f"{type(tool).__name__} acts on a layer; pass layer=... to say which."
        )
    if layer_name not in state.layers:
        raise ValueError(
            f"{type(tool).__name__} refers to layer '{layer_name}', which is not in "
            f"the viewer. Layers: {[layer.name for layer in state.layers]}"
        )
    layer_type = state.layers[layer_name].type
    if tool.layer_types is not None and layer_type not in tool.layer_types:
        raise ValueError(
            f"{type(tool).__name__} works on {' or '.join(tool.layer_types)} layers, "
            f"but '{layer_name}' is a {layer_type} layer."
        )


def _bind(state, tool: Tool, layer_name: Optional[str], key: str) -> None:
    if tool.is_layer_tool:
        state.layers[layer_name].tool_bindings[key] = tool.to_json()
    else:
        state.tool_bindings[key] = tool.to_json()


def _describe(tool: Tool, layer_name: Optional[str]) -> str:
    where = f"layer '{layer_name}'" if tool.is_layer_tool else "the viewer"
    return f"{tool.tool_type} on {where}"


# The tools here are all in neuroglancer-python's registry; make sure of it, since
# binding an unregistered type raises inside the library.
_missing = {
    cls.tool_type
    for cls in (
        AnnotatePoint,
        AnnotateLine,
        AnnotateBoundingBox,
        AnnotateEllipsoid,
        AnnotatePolyline,
        SelectSegments,
        MergeSegments,
        SplitSegments,
        ShaderControl,
        Dimension,
    )
} | set(LAYER_SETTINGS)
_missing -= set(viewer_state.tool_types)
if _missing:  # pragma: no cover - depends on the installed neuroglancer
    warnings.warn(
        f"The installed neuroglancer package does not know tool types {sorted(_missing)}; "
        "binding them will fail."
    )
