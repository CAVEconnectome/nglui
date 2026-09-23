"""Neuroglancer tools: key bindings and tool palettes.

A tool is an action a user activates with a key (``shift`` plus the bound letter) or
from a tool palette: selecting segments, adjusting a shader control or layer setting
by dragging, or stepping through a dimension. Tools are bound when a `ViewerState` is built, so keys can be
allocated across the whole viewer: Neuroglancer keeps one key namespace for every
layer, and a key bound twice silently loses its earlier tool when the state loads.

Annotation-placing tools are not here: Neuroglancer restores them only as a layer's
active tool, never from a key binding or palette -- and an unrestorable binding makes
it drop every binding after it on that layer. Use ``AnnotationLayer(active_tool=...)``.

Examples
--------
>>> from nglui.statebuilder import tools
>>> vs.add_tools(
...     tools.SelectSegments(layer="seg", key="S"),
...     tools.ShaderControl(layer="img", control="normalized"),  # key auto-assigned
...     tools.MeshSilhouette(layer="seg"),
...     tools.Dimension(dimension="z"),
...     palette=tools.ToolPalette("Proofreading", side="right"),
... )
"""

from __future__ import annotations

import difflib
import re
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar, Optional, Union

import attrs
from neuroglancer import viewer_state

from .ngl_annotations import TOGGLE_BOOL_PROPERTY_TOOL
from .utils import one_of

if TYPE_CHECKING:
    from .ngl_components import Layer

__all__ = [
    "Tool",
    "SelectSegments",
    "MergeSegments",
    "SplitSegments",
    "ShaderControl",
    "LayerSetting",
    "Opacity",
    "Blend",
    "VolumeRendering",
    "VolumeRenderingGain",
    "VolumeRenderingDepthSamples",
    "CrossSectionRenderScale",
    "SelectedAlpha",
    "NotSelectedAlpha",
    "Alpha3d",
    "MeshSilhouette",
    "MeshRenderScale",
    "Saturation",
    "HideSegmentZero",
    "HoverHighlight",
    "BaseSegmentColoring",
    "IgnoreNullVisibleSet",
    "ColorSeed",
    "SegmentDefaultColor",
    "SkeletonMode2d",
    "SkeletonMode3d",
    "SkeletonLineWidth2d",
    "SkeletonLineWidth3d",
    "Dimension",
    "ToolPalette",
    "TOOL_TYPES",
    "coerce_tool",
    "coerce_layer_tools",
]

_KEY_PATTERN = re.compile(r"^[A-Z]$")

# Keys handed out when a tool does not name one. Tag tools take Q W E R T A S D F G,
# so these start from the other rows to leave tags on the keys users expect.
_AUTO_KEY_ORDER = "ZXCVBNMYUIOPHJKLQWERTASDFG"
# Where a tag tool goes when its usual key is taken: on after the tag keys, so a
# second tagged layer's tags sit next to the first's.
_TAG_KEY_ORDER = "QWERTASDFGYUIOPHJKLZXCVBNM"


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


@attrs.frozen
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


@attrs.frozen
class _SegmentationTool(Tool):
    layer_types: ClassVar = ("segmentation",)


@attrs.frozen
class SelectSegments(_SegmentationTool):
    """Select or deselect segments by painting over them."""

    tool_type: ClassVar[str] = "selectSegments"


@attrs.frozen
class MergeSegments(_SegmentationTool):
    """Merge two segments (graphene/proofreading-enabled segmentations)."""

    tool_type: ClassVar[str] = "mergeSegments"


@attrs.frozen
class SplitSegments(_SegmentationTool):
    """Split a segment (graphene/proofreading-enabled segmentations)."""

    tool_type: ClassVar[str] = "splitSegments"


@attrs.frozen
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


@attrs.frozen
class LayerSetting(Tool):
    """Base class for tools that adjust one of a layer's display settings.

    Activating one (shift plus its key) shows that setting's control at the bottom
    of the viewer. Use the named subclasses, e.g. `MeshSilhouette` or `Alpha3d`, so
    a misspelled setting is an attribute error your editor or type checker catches.
    """

    #: The setting's label in Neuroglancer's layer panel
    ui_label: ClassVar[str] = ""


@attrs.frozen
class Opacity(LayerSetting):
    """Image opacity in the 2d views. Hold the key and scroll to adjust.

    The "Opacity (slice)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "opacity"
    layer_types: ClassVar = ("image",)
    ui_label: ClassVar[str] = "Opacity (slice)"


@attrs.frozen
class Blend(LayerSetting):
    """How the image combines with the layers beneath it. Hold the key and scroll to cycle through the options.

    The "Blending (slice)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "blend"
    layer_types: ClassVar = ("image",)
    ui_label: ClassVar[str] = "Blending (slice)"


@attrs.frozen
class VolumeRendering(LayerSetting):
    """Volume rendering mode in the 3d view. Hold the key and scroll to cycle through the options.

    The "Volume rendering (experimental)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "volumeRendering"
    layer_types: ClassVar = ("image",)
    ui_label: ClassVar[str] = "Volume rendering (experimental)"


@attrs.frozen
class VolumeRenderingGain(LayerSetting):
    """Brightness of the volume rendering. Hold the key and scroll to adjust.

    The "Gain (3D)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "volumeRenderingGain"
    layer_types: ClassVar = ("image",)
    ui_label: ClassVar[str] = "Gain (3D)"


@attrs.frozen
class VolumeRenderingDepthSamples(LayerSetting):
    """Depth samples for the volume rendering. Hold the key and scroll to adjust.

    The "Resolution (3D)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "volumeRenderingDepthSamples"
    layer_types: ClassVar = ("image",)
    ui_label: ClassVar[str] = "Resolution (3D)"


@attrs.frozen
class CrossSectionRenderScale(LayerSetting):
    """Resolution of the 2d rendering. Hold the key and scroll to adjust.

    The "Resolution (slice)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "crossSectionRenderScale"
    layer_types: ClassVar = ("image", "segmentation")
    ui_label: ClassVar[str] = "Resolution (slice)"


@attrs.frozen
class SelectedAlpha(LayerSetting):
    """2d opacity of selected segments. Hold the key and scroll to adjust.

    The "Opacity (on)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "selectedAlpha"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Opacity (on)"


@attrs.frozen
class NotSelectedAlpha(LayerSetting):
    """2d opacity of unselected segments. Hold the key and scroll to adjust.

    The "Opacity (off)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "notSelectedAlpha"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Opacity (off)"


@attrs.frozen
class Alpha3d(LayerSetting):
    """Opacity of meshes in the 3d view. Hold the key and scroll to adjust.

    The "Opacity (3d)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "objectAlpha"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Opacity (3d)"


@attrs.frozen
class MeshSilhouette(LayerSetting):
    """Silhouette shading of meshes in the 3d view. Hold the key and scroll to adjust.

    The "Silhouette (3d)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "meshSilhouetteRendering"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Silhouette (3d)"


@attrs.frozen
class MeshRenderScale(LayerSetting):
    """Mesh level of detail. Hold the key and scroll to adjust.

    The "Resolution (mesh)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "meshRenderScale"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Resolution (mesh)"


@attrs.frozen
class Saturation(LayerSetting):
    """Saturation of segment colors. Hold the key and scroll to adjust.

    The "Saturation" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "saturation"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Saturation"


@attrs.frozen
class HideSegmentZero(LayerSetting):
    """Whether segment 0 is hidden. Press to toggle.

    The "Hide segment ID 0" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "hideSegmentZero"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Hide segment ID 0"


@attrs.frozen
class HoverHighlight(LayerSetting):
    """Whether the segment under the mouse is highlighted. Press to toggle.

    The "Highlight on hover" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "hoverHighlight"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Highlight on hover"


@attrs.frozen
class BaseSegmentColoring(LayerSetting):
    """Whether supervoxels are colored individually. Press to toggle.

    The "Base segment coloring" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "baseSegmentColoring"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Base segment coloring"


@attrs.frozen
class IgnoreNullVisibleSet(LayerSetting):
    """Whether an empty selection shows every segment. Press to toggle.

    The "Show all by default" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "ignoreNullVisibleSet"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Show all by default"


@attrs.frozen
class ColorSeed(LayerSetting):
    """The random segment color palette. Press to pick new random colors.

    The "Color seed" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "colorSeed"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Color seed"


@attrs.frozen
class SegmentDefaultColor(LayerSetting):
    """One color for all segments without an explicit color. Press to switch to a fixed color, then scroll to adjust it.

    The "Fixed color" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "segmentDefaultColor"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Fixed color"


@attrs.frozen
class SkeletonMode2d(LayerSetting):
    """Skeletons as lines, or lines and points, in the 2d views. Hold the key and scroll to cycle through the options.

    The "Skeleton mode (2d)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "skeletonRendering.mode2d"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Skeleton mode (2d)"


@attrs.frozen
class SkeletonMode3d(LayerSetting):
    """Skeletons as lines, or lines and points, in the 3d view. Hold the key and scroll to cycle through the options.

    The "Skeleton mode (3d)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "skeletonRendering.mode3d"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Skeleton mode (3d)"


@attrs.frozen
class SkeletonLineWidth2d(LayerSetting):
    """Skeleton line width in the 2d views. Hold the key and scroll to adjust.

    The "Line width (2d)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "skeletonRendering.lineWidth2d"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Line width (2d)"


@attrs.frozen
class SkeletonLineWidth3d(LayerSetting):
    """Skeleton line width in the 3d view. Hold the key and scroll to adjust.

    The "Line width (3d)" control in Neuroglancer's layer panel.
    """

    tool_type: ClassVar[str] = "skeletonRendering.lineWidth3d"
    layer_types: ClassVar = ("segmentation",)
    ui_label: ClassVar[str] = "Line width (3d)"


@attrs.frozen
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


def _concrete_tools(cls) -> list:
    found = []
    for sub in cls.__subclasses__():
        if isinstance(sub.__dict__.get("tool_type"), str) and sub.tool_type:
            found.append(sub)
        found.extend(_concrete_tools(sub))
    return found


#: Neuroglancer tool type -> nglui tool class, for every bindable tool
TOOL_TYPES: dict = {cls.tool_type: cls for cls in _concrete_tools(Tool)}


def coerce_tool(value) -> Tool:
    """Make a Tool from a Tool, a Tool class, or a Neuroglancer tool type name.

    Raises
    ------
    ValueError
        If a name is not a bindable tool type, or names a tool that needs options.
    """
    if isinstance(value, Tool):
        return value
    if isinstance(value, type) and issubclass(value, Tool):
        cls = value
    elif isinstance(value, str):
        cls = TOOL_TYPES.get(value)
        if cls is None:
            close = difflib.get_close_matches(value, TOOL_TYPES, n=3)
            hint = f" Did you mean {close}?" if close else ""
            raise ValueError(
                f"{value!r} is not a bindable Neuroglancer tool type.{hint} "
                "Annotation-placing tools cannot be bound; use active_tool."
            )
    else:
        raise TypeError(
            f"Expected a Tool, Tool class, or tool type name, got {value!r}."
        )
    try:
        return cls()
    except TypeError as err:
        raise ValueError(
            f"{cls.__name__} needs options; pass tools.{cls.__name__}(...) instead."
        ) from err


def coerce_layer_tools(value) -> list:
    """Normalize a layer's ``tools``: a list of tools, or a mapping of key -> tool.

    Values may be Tool instances, Tool classes, or Neuroglancer tool type names.

    Examples
    --------
    >>> coerce_layer_tools({"H": "meshSilhouetteRendering", "S": tools.SelectSegments})
    [MeshSilhouette(layer=None, key='H'), SelectSegments(layer=None, key='S')]
    """
    if value is None:
        return []
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            tool = coerce_tool(item)
            if tool.key is not None and tool.key != key:
                raise ValueError(
                    f"{type(tool).__name__} is mapped to key {key!r} but has "
                    f"key={tool.key!r}; give the key in one place."
                )
            out.append(attrs.evolve(tool, key=key))
        return out
    return [coerce_tool(item) for item in value]


def check_layer_type(tool: Tool, layer_type: str, layer_name: str) -> None:
    """Raise if `tool` cannot act on a layer of `layer_type`."""
    if tool.layer_types is not None and layer_type not in tool.layer_types:
        raise ValueError(
            f"{type(tool).__name__} works on {' or '.join(tool.layer_types)} layers, "
            f"but layer '{layer_name}' has type {layer_type!r}."
        )


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
        validator=one_of("left", "right", "top", "bottom", optional=True),
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


def bind_tools(
    state,
    requests: list[tuple[Tool, Optional[str]]],
    tag_layers: tuple[str, ...] = (),
    reserved: Iterable[str] = (),
) -> None:
    """Bind tools onto a built neuroglancer state, allocating keys viewer-wide.

    Keys are claimed in order of how deliberately they were chosen:

    1. Bindings the state already has from a raw layer or base state, which are
       kept as they are. Duplicates among them are warned about, since the viewer
       will drop one.
    2. Tools with an explicit `key`, which must be free, or this raises.
    3. Tag tools that nglui generated on `tag_layers`. Each keeps its usual key
       when that is free; the rest then move to the next free letters, with a
       warning, and any left without a key are dropped, with a warning -- the tag
       stays usable from the layer's annotation panel.
    4. Tools with ``key=None``, which take the next free letter, or this raises.

    Parameters
    ----------
    state : neuroglancer.ViewerState
        The state, with every layer already added.
    requests : list of (Tool, str or None)
        Each tool with the name of the layer it belongs to, if it came from a
        layer's own ``tools`` list.
    tag_layers : tuple of str
        Names of nglui annotation layers whose tag tool bindings may be moved.
    reserved : iterable of str
        Keys bound elsewhere, e.g. by the viewer's ``extra`` JSON, which is merged
        in after binding and would otherwise silently replace what lands there.
    """
    fixed, tag_bindings, kept = _existing_bindings(state, tag_layers)
    used = dict(fixed)
    for key in reserved:
        used.setdefault(key, "the viewer's extra toolBindings")
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

    # Tags whose usual key is free keep it, so one conflict moves only that tag
    displaced = []
    for layer_name, key, value in tag_bindings:
        if key in used:
            displaced.append((layer_name, key, value))
        else:
            claim(key, layer_name, value, f"a tag tool on layer '{layer_name}'")
    moved, dropped = {}, {}
    for layer_name, old_key, value in displaced:
        key = next((k for k in _TAG_KEY_ORDER if k not in used), None)
        if key is None:
            dropped.setdefault(layer_name, []).append(_tag_label(value))
            continue
        claim(key, layer_name, value, f"a tag tool on layer '{layer_name}'")
        moved.setdefault(layer_name, []).append(f"{_tag_label(value)} {old_key}->{key}")
    if moved:
        warnings.warn(
            "Tag tools moved to free keys, since Neuroglancer has one set of keys "
            "for the whole viewer: "
            + "; ".join(f"layer '{n}': {', '.join(m)}" for n, m in moved.items()),
            stacklevel=4,
        )
    for layer_name, labels in dropped.items():
        warnings.warn(
            f"No free key is left for the tag tools {', '.join(labels)} on layer "
            f"'{layer_name}'; they are left unbound and remain usable from the "
            "layer's annotation panel.",
            stacklevel=4,
        )

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


#: Tool types Neuroglancer restores only as a layer's active tool, not from bindings
LEGACY_ONLY_TOOLS = frozenset(
    {
        "annotatePoint",
        "annotateLine",
        "annotateBoundingBox",
        "annotateSphere",
        "annotatePolyline",
    }
)


def _tool_type(value) -> str:
    return value if isinstance(value, str) else value.get("type", "")


def _is_tag_tool(value) -> bool:
    tool_type = _tool_type(value)
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
            if _tool_type(value) in LEGACY_ONLY_TOOLS:
                warnings.warn(
                    f"Layer '{layer.name}' binds {_tool_type(value)!r} to key {key!r}. "
                    "Neuroglancer cannot restore annotation tools from key bindings: "
                    "it drops this binding and every binding after it on the layer. "
                    "Use AnnotationLayer(active_tool=...) instead.",
                    stacklevel=4,
                )
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
            "Set key=False on some tools to show them only in a palette."
        )
    return key


def _tag_label(value) -> str:
    """A readable name for a tag tool binding, for warnings."""
    if isinstance(value, dict) and "property" in value:
        return repr(value["property"])
    return repr(_tool_type(value).removeprefix("tagTool_"))


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
    check_layer_type(tool, state.layers[layer_name].type, layer_name)


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
_missing = set(TOOL_TYPES) - set(viewer_state.tool_types)
if _missing:  # pragma: no cover - depends on the installed neuroglancer
    warnings.warn(
        f"The installed neuroglancer package does not know tool types {sorted(_missing)}; "
        "binding them will fail."
    )
