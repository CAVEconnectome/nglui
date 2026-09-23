"""Multi-panel layouts: rows and columns of panels, each showing its own layers.

Neuroglancer's preset layouts ("xy-3d", "4panel", ...) show every layer in every
view. For side-by-side comparisons -- imagery beside a 3d mesh view, or one neuron
beside another -- a layout can instead be a tree of rows and columns whose leaves
are `Panel`s, each with its own layers, view type, and optionally its own camera.

Examples
--------
>>> from nglui.statebuilder import Panel, row
>>> vs.set_layout(
...     row(
...         Panel(["img", "seg"], layout="xy"),
...         Panel(["seg"], layout="3d", camera=Camera(projection_orientation="xz")),
...     )
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

import attrs

from .viewer_config import Camera, _set_fields

if TYPE_CHECKING:
    from .ngl_components import Layer

__all__ = ["Panel", "StackLayout", "row", "column", "PRESET_LAYOUTS"]

#: Neuroglancer's built-in arrangements of 2d and 3d views
PRESET_LAYOUTS = (
    "xy",
    "yz",
    "xz",
    "xy-3d",
    "xz-3d",
    "yz-3d",
    "4panel",
    "3d",
    "4panel-alt",
)

# Camera field -> the LayerGroupViewer JSON key holding its linked value
_CAMERA_KEYS = {
    "cross_section_scale": "crossSectionScale",
    "cross_section_orientation": "crossSectionOrientation",
    "cross_section_depth": "crossSectionDepth",
    "projection_scale": "projectionScale",
    "projection_orientation": "projectionOrientation",
    "projection_depth": "projectionDepth",
}


def _layer_names(layers) -> Optional[list[str]]:
    if layers is None:
        return None
    if isinstance(layers, str) or not isinstance(layers, (list, tuple)):
        layers = [layers]
    return [layer if isinstance(layer, str) else layer.name for layer in layers]


def _positive_flex(instance, attribute, value):
    if value is not None and value <= 0:
        raise ValueError(f"flex must be positive, got {value}.")


@attrs.frozen
class Panel:
    """One view in a multi-panel layout, showing a chosen set of layers.

    Parameters
    ----------
    layers : list of str or Layer, optional
        Layers shown in this panel, by name or object. None shows every layer.
    layout : str, optional
        Which views the panel has, as one of Neuroglancer's presets (e.g. "xy",
        "3d", "xy-3d"). Default is "xy".
    camera : Camera, optional
        A camera for this panel. Fields that are set are decoupled from the rest of
        the viewer (see `camera_link`); unset fields follow the global camera.
    camera_link : {"unlinked", "relative"}, optional
        How the set camera fields relate to the global camera: "unlinked" holds them
        fixed, "relative" applies them as an offset that follows the global view.
        Default is "unlinked".
    flex : float, optional
        Share of the enclosing row or column this panel takes. Default is 1.

    Examples
    --------
    >>> Panel([img, seg], layout="xy")
    >>> Panel(["seg"], layout="3d", camera=Camera(projection_orientation="xz"))
    """

    layers: Optional[list] = attrs.field(default=None, converter=_layer_names)
    layout: str = attrs.field(
        default="xy", kw_only=True, validator=attrs.validators.in_(PRESET_LAYOUTS)
    )
    camera: Optional[Camera] = attrs.field(default=None, kw_only=True)
    camera_link: Literal["unlinked", "relative"] = attrs.field(
        default="unlinked",
        kw_only=True,
        validator=attrs.validators.in_(("unlinked", "relative")),
    )
    flex: Optional[float] = attrs.field(
        default=None, kw_only=True, validator=_positive_flex
    )

    def layer_names(self) -> list[str]:
        """Names of the layers this panel shows, or [] if it shows all of them."""
        return list(self.layers or [])

    def to_json(self, all_layers: list[str]) -> dict:
        """The panel's ``viewer`` layout JSON."""
        out = {
            "type": "viewer",
            "layers": list(self.layers)
            if self.layers is not None
            else list(all_layers),
            "layout": self.layout,
        }
        if self.flex is not None:
            out["flex"] = self.flex
        if self.camera is not None:
            for name, value in _set_fields(self.camera).items():
                out[_CAMERA_KEYS[name]] = {"link": self.camera_link, "value": value}
        return out


@attrs.frozen
class StackLayout:
    """A row or column of panels and nested layouts. Build with `row` or `column`."""

    direction: Literal["row", "column"] = attrs.field(
        validator=attrs.validators.in_(("row", "column"))
    )
    children: list = attrs.field(converter=list)
    flex: Optional[float] = attrs.field(
        default=None, kw_only=True, validator=_positive_flex
    )

    @children.validator
    def _check_children(self, attribute, value):
        if not value:
            raise ValueError(f"A {self.direction} layout needs at least one child.")
        for child in value:
            if not isinstance(child, (Panel, StackLayout)):
                raise TypeError(
                    f"Children of a {self.direction} must be Panels, rows, or columns; "
                    f"got {child!r}. Wrap a preset in Panel(layout=...)."
                )

    def layer_names(self) -> list[str]:
        """Names of every layer a panel in this layout names explicitly."""
        return [name for child in self.children for name in child.layer_names()]

    def to_json(self, all_layers: list[str]) -> dict:
        """The layout's JSON."""
        out = {
            "type": self.direction,
            "children": [child.to_json(all_layers) for child in self.children],
        }
        if self.flex is not None:
            out["flex"] = self.flex
        return out


def row(
    *children: Union[Panel, StackLayout], flex: Optional[float] = None
) -> StackLayout:
    """Lay panels out side by side, left to right.

    Parameters
    ----------
    *children : Panel or StackLayout
        The panels and nested rows or columns.
    flex : float, optional
        Share of an enclosing row or column this row takes.
    """
    return StackLayout("row", list(children), flex=flex)


def column(
    *children: Union[Panel, StackLayout], flex: Optional[float] = None
) -> StackLayout:
    """Lay panels out stacked, top to bottom.

    Parameters
    ----------
    *children : Panel or StackLayout
        The panels and nested rows or columns.
    flex : float, optional
        Share of an enclosing row or column this column takes.
    """
    return StackLayout("column", list(children), flex=flex)


Layout = Union[str, Panel, StackLayout]


def validate_layout(value) -> None:
    """Raise if `value` is not a preset name, Panel, or StackLayout."""
    if isinstance(value, (Panel, StackLayout)):
        return
    if value not in PRESET_LAYOUTS:
        raise ValueError(
            f"Invalid layout: {value!r}. Use a preset ({', '.join(PRESET_LAYOUTS)}) "
            "or a Panel, row(...), or column(...)."
        )


def layout_to_json(value: Layout, all_layers: list[str]):
    """Convert a layout to the value of the state's ``layout`` field.

    Raises
    ------
    ValueError
        If a panel names a layer that is not in the state.
    """
    if isinstance(value, str):
        return value
    missing = sorted(set(value.layer_names()) - set(all_layers))
    if missing:
        raise ValueError(
            f"The layout names layers {missing} that are not in the viewer. "
            f"Layers: {all_layers}"
        )
    return value.to_json(all_layers)
