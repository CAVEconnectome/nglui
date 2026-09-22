"""Config objects for the parts of a Neuroglancer state that are nested structures.

Neuroglancer models the camera and each side panel as a sub-object of the viewer
state, and the same shapes reappear elsewhere (a layer-group panel has its own
camera; tool palettes are placed like any other side panel). Each class here maps
onto one of those sub-objects so it can be built once and reused.

Every field defaults to None, meaning "not set": unset fields are left out of the
emitted state, so Neuroglancer's own defaults -- or a base state's values -- apply.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Sequence, Union

import attrs
import numpy as np

__all__ = [
    "Camera",
    "SidePanel",
    "SkeletonRendering",
    "NAMED_ORIENTATIONS",
    "parse_orientation",
]

_HALF_SQRT2 = math.sqrt(0.5)

#: Quaternions ``[x, y, z, w]`` for the orientations of Neuroglancer's own named
#: panels. "xz" and "yz" are the quarter-turns Neuroglancer uses for its xz and yz
#: cross-section panels, so a named orientation looks the way that panel does.
NAMED_ORIENTATIONS: dict[str, list[float]] = {
    "xy": [0.0, 0.0, 0.0, 1.0],
    "xz": [_HALF_SQRT2, 0.0, 0.0, _HALF_SQRT2],
    "yz": [0.0, _HALF_SQRT2, 0.0, _HALF_SQRT2],
}

Orientation = Union[Literal["xy", "xz", "yz"], Sequence[float], np.ndarray]


def parse_orientation(value: Optional[Orientation]) -> Optional[list[float]]:
    """Convert an orientation to a normalized ``[x, y, z, w]`` quaternion.

    Parameters
    ----------
    value : str, sequence of 4 floats, or None
        Either the name of a plane (``"xy"``, ``"xz"``, ``"yz"``), matching the
        orientation of Neuroglancer's panel of that name, or a quaternion in
        Neuroglancer's ``[x, y, z, w]`` order. Quaternions are normalized.

    Returns
    -------
    list of float or None
        The quaternion, or None if `value` is None.

    Examples
    --------
    >>> parse_orientation("xy")
    [0.0, 0.0, 0.0, 1.0]
    >>> parse_orientation([0, 0, 0, 2])
    [0.0, 0.0, 0.0, 1.0]
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return list(NAMED_ORIENTATIONS[value])
        except KeyError:
            raise ValueError(
                f"Unknown orientation {value!r}. Use one of "
                f"{sorted(NAMED_ORIENTATIONS)} or an [x, y, z, w] quaternion."
            ) from None
    quat = np.asarray(value, dtype=float)
    if quat.shape != (4,):
        raise ValueError(
            f"Orientation must be a plane name or an [x, y, z, w] quaternion, got {value!r}."
        )
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("Orientation quaternion must be nonzero.")
    return [float(x) for x in quat / norm]


def _optional_float(value) -> Optional[float]:
    return None if value is None else float(value)


def _set_fields(obj) -> dict:
    """The fields of an attrs config object that have been set."""
    return {k: v for k, v in attrs.asdict(obj, recurse=False).items() if v is not None}


@attrs.define
class Camera:
    """Where the viewer looks, in both the 2d cross-section and the 3d projection.

    Neuroglancer keeps separate zoom, orientation, and depth for the cross-section
    (2d) and projection (3d) views. Field names follow Neuroglancer's own.

    Parameters
    ----------
    cross_section_scale : float, optional
        Zoom of the 2d views, in voxels per screen pixel. Smaller is closer.
    cross_section_orientation : str or sequence of float, optional
        Orientation of the 2d views, as a plane name (``"xy"``, ``"xz"``, ``"yz"``)
        or an ``[x, y, z, w]`` quaternion. See `parse_orientation`.
    cross_section_depth : float, optional
        Depth of field of the 2d views, used when rendering annotations off-plane.
    projection_scale : float, optional
        Zoom of the 3d view. Smaller is closer.
    projection_orientation : str or sequence of float, optional
        Orientation of the 3d view, as a plane name or quaternion.
    projection_depth : float, optional
        Depth of the 3d view's clipping volume.

    Examples
    --------
    >>> vs.set_camera(projection_orientation="xz", projection_scale=5000)
    >>> side_view = Camera(projection_orientation=[0.5, 0.5, 0.5, 0.5])
    >>> vs.set_camera(side_view)
    """

    cross_section_scale: Optional[float] = attrs.field(
        default=None, kw_only=True, converter=_optional_float
    )
    cross_section_orientation: Optional[list[float]] = attrs.field(
        default=None, kw_only=True, converter=parse_orientation
    )
    cross_section_depth: Optional[float] = attrs.field(
        default=None, kw_only=True, converter=_optional_float
    )
    projection_scale: Optional[float] = attrs.field(
        default=None, kw_only=True, converter=_optional_float
    )
    projection_orientation: Optional[list[float]] = attrs.field(
        default=None, kw_only=True, converter=parse_orientation
    )
    projection_depth: Optional[float] = attrs.field(
        default=None, kw_only=True, converter=_optional_float
    )

    def merge(self, other: Camera) -> Camera:
        """Return a camera with `other`'s set fields laid over this one's."""
        return attrs.evolve(self, **_set_fields(other))

    def apply_to(self, state) -> None:
        """Write the set fields onto a neuroglancer state or layer-group panel."""
        for name, value in _set_fields(self).items():
            setattr(state, name, value)


@attrs.define
class SidePanel:
    """Placement and visibility of one of Neuroglancer's side panels.

    Used for the layer list, statistics, help, and selected-layer panels.

    Parameters
    ----------
    visible : bool, optional
        Whether the panel is open.
    side : {"left", "right", "top", "bottom"}, optional
        Which edge of the viewer the panel docks to.
    size : int, optional
        Width (or height, for top/bottom panels) in pixels.
    flex : float, optional
        Share of the edge's length when several panels dock to the same side.
    row, col : int, optional
        Position among other panels docked to the same side.

    Examples
    --------
    >>> vs.set_panels(layer_list=True, selected_layer=SidePanel(side="right", size=500))
    """

    visible: Optional[bool] = attrs.field(default=None, kw_only=True)
    side: Optional[Literal["left", "right", "top", "bottom"]] = attrs.field(
        default=None,
        kw_only=True,
        validator=attrs.validators.optional(
            attrs.validators.in_(("left", "right", "top", "bottom"))
        ),
    )
    size: Optional[int] = attrs.field(default=None, kw_only=True)
    flex: Optional[float] = attrs.field(default=None, kw_only=True)
    row: Optional[int] = attrs.field(default=None, kw_only=True)
    col: Optional[int] = attrs.field(default=None, kw_only=True)

    @classmethod
    def coerce(cls, value: Union[bool, SidePanel, dict]) -> SidePanel:
        """Accept a SidePanel, a dict of its fields, or a bool meaning `visible`."""
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(visible=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(
            f"Expected a SidePanel, dict, or bool for a side panel, got {value!r}."
        )

    def merge(self, other: SidePanel) -> SidePanel:
        """Return a panel with `other`'s set fields laid over this one's."""
        return attrs.evolve(self, **_set_fields(other))

    def apply_to(self, panel_state) -> None:
        """Write the set fields onto a neuroglancer side panel state."""
        for name, value in _set_fields(self).items():
            setattr(panel_state, name, value)


@attrs.define
class SkeletonRendering:
    """How a segmentation layer draws skeletons.

    Parameters
    ----------
    shader : str, optional
        GLSL skeleton shader. See `nglui.statebuilder.shaders` for ready-made ones.
    shader_controls : dict, optional
        Values for the ``#uicontrol`` controls the shader declares, keyed by name.
    mode_2d, mode_3d : {"lines", "lines_and_points"}, optional
        Whether to draw vertices as well as edges, in the 2d and 3d views.
    line_width_2d, line_width_3d : float, optional
        Edge width in pixels, in the 2d and 3d views.

    Examples
    --------
    >>> SegmentationLayer(
    ...     source=...,
    ...     skeleton_rendering=SkeletonRendering(line_width_3d=3, mode_3d="lines_and_points"),
    ... )
    """

    shader: Optional[str] = attrs.field(default=None, kw_only=True)
    shader_controls: Optional[dict] = attrs.field(default=None, kw_only=True)
    mode_2d: Optional[Literal["lines", "lines_and_points"]] = attrs.field(
        default=None,
        kw_only=True,
        validator=attrs.validators.optional(
            attrs.validators.in_(("lines", "lines_and_points"))
        ),
    )
    mode_3d: Optional[Literal["lines", "lines_and_points"]] = attrs.field(
        default=None,
        kw_only=True,
        validator=attrs.validators.optional(
            attrs.validators.in_(("lines", "lines_and_points"))
        ),
    )
    line_width_2d: Optional[float] = attrs.field(
        default=None, kw_only=True, converter=_optional_float
    )
    line_width_3d: Optional[float] = attrs.field(
        default=None, kw_only=True, converter=_optional_float
    )

    _JSON_KEYS = {
        "shader": "shader",
        "shader_controls": "shaderControls",
        "mode_2d": "mode2d",
        "mode_3d": "mode3d",
        "line_width_2d": "lineWidth2d",
        "line_width_3d": "lineWidth3d",
    }

    @classmethod
    def coerce(cls, value: Union[SkeletonRendering, dict, None]):
        """Accept a SkeletonRendering or a dict of its fields."""
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Expected a SkeletonRendering or dict, got {value!r}.")

    def to_json(self) -> dict:
        """The set fields, under Neuroglancer's ``skeletonRendering`` JSON keys."""
        return {self._JSON_KEYS[k]: v for k, v in _set_fields(self).items()}
