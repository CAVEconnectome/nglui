import os
from typing import Optional

NEUROGLANCER_SITES = {
    "spelunker": "https://spelunker.cave-explorer.org/",
    "google": "https://neuroglancer-demo.appspot.com/",
}

DEFAULT_KEY = "DEFAULT_"
if "NGLUI_TARGET_SITE" in os.environ and "NGLUI_TARGET_URL" in os.environ:
    NEUROGLANCER_SITES[os.environ["NGLUI_TARGET_SITE"]] = os.environ["NGLUI_TARGET_URL"]
    DEFAULT_TARGET_SITE = os.environ["NGLUI_TARGET_SITE"]
else:
    DEFAULT_TARGET_SITE = "spelunker"
DEFAULT_TARGET_URL = NEUROGLANCER_SITES[DEFAULT_TARGET_SITE]
NEUROGLANCER_SITES[DEFAULT_KEY] = NEUROGLANCER_SITES[DEFAULT_TARGET_SITE]

MAX_URL_LENGTH = 1_750_000


def add_neuroglancer_site(
    site_name: str,
    site_url: str,
    set_default: bool = False,
) -> None:
    """
    Add a neuroglancer site to the list of available sites.

    Parameters
    ----------
    site_name : str
        Name of the neuroglancer site.
    site_url : str
        URL of the neuroglancer site.
    """
    if site_name in NEUROGLANCER_SITES:
        raise ValueError(f"Neuroglancer site {site_name} already exists")
    NEUROGLANCER_SITES[site_name] = site_url

    if set_default:
        set_default_neuroglancer_site(site_name)


def set_default_neuroglancer_site(
    site_name: str,
) -> None:
    """
    Set the default neuroglancer site.

    Parameters
    ----------
    site_name : str
        Name of the neuroglancer site to set as default.
    """
    if site_name not in NEUROGLANCER_SITES:
        raise ValueError(f"Neuroglancer site {site_name} does not exist")
    NEUROGLANCER_SITES[DEFAULT_KEY] = NEUROGLANCER_SITES[site_name]


def get_default_neuroglancer_site() -> str:
    """
    Get the default neuroglancer site URL.

    Returns
    -------
    dict
        Single element dictionary with the site and URL of the default neuroglancer site.
    """
    for key, value in NEUROGLANCER_SITES.items():
        if key == DEFAULT_KEY:
            continue
        if value == NEUROGLANCER_SITES[DEFAULT_KEY]:
            break
    return {key: NEUROGLANCER_SITES[DEFAULT_KEY]}


def neuroglancer_url(
    url: Optional[str] = None,
    target_site: Optional[str] = None,
) -> str:
    """
    Check neuroglancer info to determine which kind of site a neuroglancer URL is.
    If either url or target_site are provided, it will use these values, looking up target site
    from the fallback values in the config. Otherwise, it falls back to the value of
    "target_url" in the config.

    Parameters
    ----------
    url : str, optional
        URL to check, by default None
    target_site : str, optional
        Target site to check, by default None

    Returns
    -------
    str
        URL of the neuroglancer viewer
    """
    if url is not None:
        return url
    if target_site is None:
        target_site = DEFAULT_KEY
    if target_site in NEUROGLANCER_SITES:
        return NEUROGLANCER_SITES[target_site]
    else:
        raise ValueError(
            f"Neuroglancer site {target_site} not found. Available sites: {list(NEUROGLANCER_SITES.keys()).remove(DEFAULT_KEY)}"
        )


def get_neuroglancer_sites() -> dict:
    """
    Get the list of available neuroglancer sites.

    Returns
    -------
    dict
        List of available neuroglancer URLs and their names.
    """
    ngl_sites = NEUROGLANCER_SITES.copy()
    ngl_sites.pop(DEFAULT_KEY, None)
    return ngl_sites
