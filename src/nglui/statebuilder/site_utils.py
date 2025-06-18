from typing import Optional

NEUROGLANCER_SITES = {
    "spelunker": "https://spelunker.cave-explorer.org/",
    "google": "https://neuroglancer-demo.appspot.com/",
}

DEFAULT_TARGET_SITE = "spelunker"

MAX_URL_LENGTH = 1_750_000


def add_neuroglancer_site(
    site_name: str,
    site_url: str,
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
        target_site = DEFAULT_TARGET_SITE
    if target_site in NEUROGLANCER_SITES:
        return NEUROGLANCER_SITES[target_site]
    else:
        raise ValueError(
            f"Neuroglancer site {target_site} not found. Available sites: {list(NEUROGLANCER_SITES.keys())}"
        )


def target_sites() -> dict:
    """
    Get the list of available neuroglancer sites.

    Returns
    -------
    dict
        List of available neuroglancer URLs and their names.
    """
    return NEUROGLANCER_SITES.copy()
