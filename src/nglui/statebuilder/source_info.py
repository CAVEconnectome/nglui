try:
    import cloudvolume

    HAS_CLOUDVOLUME = True
except ImportError:
    HAS_CLOUDVOLUME = False
    cloudvolume = None

import warnings

import numpy as np
from cachetools import LRUCache, cached

from .ngl_components import AnnotationLayer, Layer


class NoCloudvolumeError(Exception):
    pass


@cached(cache=LRUCache(maxsize=100))
def get_source_info(source):
    """
    Get source info from a Neuroglancer source.

    Parameters
    ----------
    source : neuroglancer.source.Source
        Neuroglancer source object.

    Returns
    -------
    dict
        Source info dictionary.
    """
    if not HAS_CLOUDVOLUME:
        raise NoCloudvolumeError
    cv = cloudvolume.CloudVolume(source)

    return {
        "name": cv.dataset_name,
        "type": cv.info.get("type"),
        "resolution": np.array(cv.resolution),
        "data_type": cv.data_type,
        "bounds": np.array(cv.bounds.to_list()).reshape(-1, 3),
        "num_channels": cv.num_channels,
    }


def populate_info(layers):
    info_dict = {}
    for layer in layers:
        if issubclass(type(layer), Layer) and not isinstance(layer, AnnotationLayer):
            if not hasattr(layer, "source"):
                continue
            source = layer.source
            if not isinstance(source, list):
                source = [source]
            for s in source:
                if hasattr(s, "url"):
                    s = s.url
                    # Just use the path part of the URL
                try:
                    source_info = get_source_info(s)
                    info_dict[s] = source_info
                except NoCloudvolumeError:
                    info_dict[s] = {}
                    warnings.warn(
                        f"CloudVolume is not available. Cannot get source information for {s}.",
                    )
                except:
                    info_dict[s] = {}
                # This is a hack to avoid long-delays for non-functional URLs that I can't avoid,
                # But it assumes that the first source is the true segmentation which is not necessarily true.
                break
    return info_dict


def _get_mean_position(info, resolution):
    bounds = info.get("bounds")
    if bounds is not None:
        position = np.mean(bounds, axis=0)
        pos_resolution = info.get("resolution")
        if pos_resolution is not None and resolution is not None:
            scaling = np.array(pos_resolution) / np.array(resolution)
            position = position * scaling
    return position


def suggest_position(
    info_cache,
    resolution=None,
):
    """Get a suggested position based on the bounds in the source info."""
    for _, info in info_cache.items():
        if info.get("type") == "segmentation":
            position = _get_mean_position(info, resolution)
            if position is not None:
                return position
    for _, info in info_cache.items():
        if info.get("type") == "image":
            position = _get_mean_position(info, resolution)
            if position is not None:
                return position
    else:
        return None


def suggest_resolution(
    info_cache,
):
    """Get a suggested resolution based on the source info.
    Returns the resolution of the first image or segmentation source found.
    """
    for _, info in info_cache.items():
        if info.get("type") == "image":
            resolution = info.get("resolution")
            if resolution is not None:
                return resolution
    for _, info in info_cache.items():
        if info.get("type") == "segmentation":
            resolution = info.get("resolution")
            if resolution is not None:
                return resolution
    else:
        return None
