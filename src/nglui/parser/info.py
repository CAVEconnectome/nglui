import requests

ngl_info_endpoint = "{neuroglancer_endpoint}/version.json"

ANNOTATION_INFO_TYPE = "neuroglancer_annotations_v1"
SKELETON_INFO_TYPE = "neuroglancer_skeletons"


def get_ngl_info(ngl_url):
    """
    Get the version of neuroglancer running at a given endpoint.
    """
    try:
        r = requests.get(ngl_info_endpoint.format(neuroglancer_endpoint=ngl_url))
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error getting neuroglancer version: {e}")
        return None


def get_annotation_info(url: str) -> dict:
    """Fetch and validate a Neuroglancer precomputed annotation `info` file.

    Parameters
    ----------
    url : str
        Base URL of the precomputed annotation source. The `info` file is
        fetched from ``{url}/info``.

    Returns
    -------
    dict
        Parsed JSON contents of the info file.

    Raises
    ------
    requests.HTTPError
        If the HTTP request fails.
    ValueError
        If the response is not a valid annotation info file (wrong ``@type``
        or missing/invalid ``properties``).

    See Also
    --------
    https://neuroglancer-docs.web.app/datasource/precomputed/annotation.html
    """
    info_url = f"{url.rstrip('/')}/info"
    r = requests.get(info_url)
    r.raise_for_status()
    info = r.json()

    if info.get("@type") != ANNOTATION_INFO_TYPE:
        raise ValueError(
            f"Expected '@type' == {ANNOTATION_INFO_TYPE!r}, got {info.get('@type')!r} "
            f"from {info_url}"
        )
    if not isinstance(info.get("properties"), list):
        raise ValueError(
            f"Annotation info file at {info_url} is missing a 'properties' list."
        )
    return info


def get_skeleton_info(url: str) -> dict:
    """Fetch and validate a Neuroglancer precomputed skeleton ``info`` file.

    Parameters
    ----------
    url : str
        Base URL of the precomputed skeleton source. The ``info`` file is
        fetched from ``{url}/info``.

    Returns
    -------
    dict
        Parsed JSON contents of the info file. ``"vertex_attributes"`` is
        normalised to a list (default ``[]``) so callers can iterate without
        a ``None`` check.

    Raises
    ------
    requests.HTTPError
        If the HTTP request fails.
    ValueError
        If the ``@type`` field is not ``"neuroglancer_skeletons"``.

    See Also
    --------
    https://neuroglancer-docs.web.app/datasource/precomputed/skeleton.html
    """
    info_url = f"{url.rstrip('/')}/info"
    r = requests.get(info_url)
    r.raise_for_status()
    info = r.json()

    if info.get("@type") != SKELETON_INFO_TYPE:
        raise ValueError(
            f"Expected '@type' == {SKELETON_INFO_TYPE!r}, got {info.get('@type')!r} "
            f"from {info_url}"
        )
    # vertex_attributes is documented as optional; normalise so callers can
    # iterate it directly.
    if info.get("vertex_attributes") is None:
        info["vertex_attributes"] = []
    elif not isinstance(info["vertex_attributes"], list):
        raise ValueError(
            f"Skeleton info file at {info_url} has a non-list 'vertex_attributes'."
        )
    return info
