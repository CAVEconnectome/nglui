"""Deployment version lookup.

Retained for backward compatibility. The implementation now lives in
`nglui.statebuilder.capabilities`, which parses payloads that are not strictly JSON
(the Neuroglancer demo serves a Python repr) and caches results, including failures.
"""

from ..statebuilder.capabilities import get_version_info

__all__ = ["get_ngl_info", "get_version_info"]


def get_ngl_info(ngl_url: str) -> dict | None:
    """Get the version information of the Neuroglancer deployment at an endpoint.

    Parameters
    ----------
    ngl_url : str
        URL of the Neuroglancer deployment.

    Returns
    -------
    dict or None
        Parsed ``version.json`` contents, or None if it could not be read.
    """
    return get_version_info(ngl_url)
