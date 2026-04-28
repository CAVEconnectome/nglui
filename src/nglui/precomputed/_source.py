"""Resolve segmentation sources to neuroglancer CoordinateSpace.

Follows the same pattern as SkeletonManager: accepts a CAVEclient,
a segmentation source URL string, or a CloudVolume instance, and
extracts the coordinate space (resolution, axis names, units).
"""

from typing import Optional, Sequence, Union

import numpy as np
from neuroglancer.coordinate_space import CoordinateSpace

try:
    import caveclient

    _has_caveclient = True
except ImportError:
    _has_caveclient = False

try:
    import cloudvolume as cv_module

    _has_cloudvolume = True
except ImportError:
    _has_cloudvolume = False


def resolve_coordinate_space(
    segmentation_source=None,
    coordinate_space: Optional[CoordinateSpace] = None,
    resolution: Optional[Sequence[float]] = None,
    units: Union[str, Sequence[str]] = "nm",
    names: Sequence[str] = ("x", "y", "z"),
) -> CoordinateSpace:
    """Resolve a coordinate space from various input types.

    Exactly one of ``segmentation_source``, ``coordinate_space``, or
    ``resolution`` must be provided.

    Parameters
    ----------
    segmentation_source : CAVEclient, CloudVolume, or str, optional
        A CAVE client (uses its segmentation source), a CloudVolume
        instance, or a segmentation source URL string. Resolution and
        units are extracted automatically.
    coordinate_space : CoordinateSpace, optional
        An explicit neuroglancer CoordinateSpace.
    resolution : sequence of float, optional
        Explicit resolution per axis (e.g., ``[8, 8, 40]``). Used with
        ``units`` and ``names`` to build a CoordinateSpace.
    units : str or sequence of str
        Units for each axis (default ``"nm"``). Only used with
        ``resolution``.
    names : sequence of str
        Axis names (default ``("x", "y", "z")``). Only used with
        ``resolution``.

    Returns
    -------
    CoordinateSpace
        The resolved neuroglancer coordinate space.

    Raises
    ------
    ValueError
        If none or more than one source of coordinate space info is given.
    """
    provided = sum(
        x is not None for x in [segmentation_source, coordinate_space, resolution]
    )
    if provided == 0:
        raise ValueError(
            "Must provide one of: segmentation_source, coordinate_space, or resolution."
        )
    if provided > 1:
        raise ValueError(
            "Provide only one of: segmentation_source, coordinate_space, or resolution."
        )

    if coordinate_space is not None:
        return coordinate_space

    if resolution is not None:
        if isinstance(units, str):
            units = [units] * len(resolution)
        return CoordinateSpace(
            names=list(names), scales=list(resolution), units=list(units)
        )

    # segmentation_source path
    return _coordinate_space_from_source(segmentation_source)


def _coordinate_space_from_source(segmentation_source) -> CoordinateSpace:
    """Extract CoordinateSpace from a CAVEclient, CloudVolume, or URL string."""
    cv = _source_to_cloudvolume(segmentation_source)
    res = np.array(cv.resolution, dtype=np.float64)
    rank = len(res)
    names = ["x", "y", "z"][:rank]
    units = ["nm"] * rank
    return CoordinateSpace(names=names, scales=res.tolist(), units=units)


def _source_to_cloudvolume(segmentation_source):
    """Convert a segmentation source to a CloudVolume instance."""
    # CAVEclient
    if _has_caveclient and isinstance(
        segmentation_source, caveclient.frameworkclient.CAVEclientFull
    ):
        if not _has_cloudvolume:
            raise ImportError(
                "cloud-volume is required to resolve coordinate space from a "
                "CAVEclient. Install with: pip install cloud-volume"
            )
        return segmentation_source.info.segmentation_cloudvolume()

    # CloudVolume instance
    if _has_cloudvolume and isinstance(segmentation_source, cv_module.CloudVolume):
        return segmentation_source

    # URL string
    if isinstance(segmentation_source, str):
        if not _has_cloudvolume:
            raise ImportError(
                "cloud-volume is required to resolve coordinate space from a "
                "source URL. Install with: pip install cloud-volume"
            )
        return cv_module.CloudVolume(segmentation_source)

    raise TypeError(
        f"segmentation_source must be a CAVEclient, CloudVolume, or URL string, "
        f"got {type(segmentation_source)}"
    )
