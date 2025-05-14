import logging
import os
from typing import Literal, Optional
from warnings import warn

import attrs
import caveclient

logger = logging.getLogger(__name__)

NEUROGLANCER_SITES = {
    "spelunker": "https://spelunker.cave-explorer.org/",
    "google": "https://neuroglancer-demo.appspot.com/",
}

DEFAULT_TARGET_SITE = "spelunker"


def is_mainline(ngl_url, client):
    """
    Check neuroglancer info to determine which kind of site a neuroglancer URL is.
    Returns False if the URL is not a modern neuroglancer URL.
    """
    ngl_info = client.state.get_neuroglancer_info(ngl_url)
    return len(ngl_info) > 0


@attrs.define
class NGLUIConfig:
    target_site = attrs.field(type=str)
    target_url = attrs.field(type=str)
    caveclient = attrs.field(default=None)
    datastack_name = attrs.field(default=None, type=str)

    def __attrs_post_init__(self):
        # validate target site
        if self.target_site not in NEUROGLANCER_SITES and self.target_site is not None:
            raise ValueError(
                f"target_site must be one of {list(NEUROGLANCER_SITES.keys())}"
            )
        # Set and validate caveclient if info is provided
        if self.caveclient is None and self.datastack_name is not None:
            try:
                self.caveclient = caveclient.CAVEclient(self.datastack_name)
            except Exception as e:
                msg = f"Could not create CAVEclient with datastack_name {self.datastack_name}: {e}"
                logger.error(msg)
        if self.caveclient is not None and self.datastack_name is None:
            self.datastack_name = self.caveclient.datastack_name
        if self.datastack_name is not None and self.caveclient is not None:
            if self.caveclient.datastack_name != self.datastack_name:
                raise ValueError(
                    f"CAVEclient datastack_name {self.caveclient.datastack_name} does not match provided datastack_name {self.datastack_name}"
                )

        # Set and validate target_url if target_site is provided
        if self.target_site is not None and self.target_url is None:
            self.target_url = NEUROGLANCER_SITES.get(self.target_site)


default_config = NGLUIConfig(
    target_site=DEFAULT_TARGET_SITE,
    target_url=NEUROGLANCER_SITES[DEFAULT_TARGET_SITE],
)
default_key = "default"

NGL_CONFIG = {default_key: default_config}  # type: dict[str, NGLUIConfig]


def get_ngl_config(config_key: str = None) -> dict:
    """Get the current configuration for nglui viewers and statebuilders.

    Parameters
    ----------
    config_key : str, optional
        Key for the configuration setting, by default "default"
    """
    if config_key is None:
        config_key = default_key
    return attrs.asdict(NGL_CONFIG[config_key])


def set_ngl_config(
    target_site: str = None,
    target_url: str = None,
    datastack_name: str = None,
    caveclient: "caveclient.CAVEclient" = None,
    url_from_client: bool = False,
    config_key: str = "default",
) -> None:
    """Set default configuration for nglui viewers and statebuilders.

    Parameters
    ----------
    target_site : str, optional
        Target site for the neuroglancer viewer, by default None
    target_url : str, optional
        Target URL for the neuroglancer viewer, by default None
    datastack_name : str, optional
        Name of the datastack, by default None
    caveclient : caveclient.CAVEclient, optional
        CAVEclient object, by default None.
    """
    if config_key in NGL_CONFIG:
        curr_config = get_ngl_config(config_key)
    else:
        curr_config = attrs.asdict(default_config)
    if target_site is not None or target_url is not None:
        curr_config["target_site"] = target_site
        curr_config["target_url"] = target_url
    if caveclient is not None or datastack_name is not None:
        curr_config["caveclient"] = caveclient
        curr_config["datastack_name"] = datastack_name
        if url_from_client:
            curr_config["target_site"] = None
            curr_config["target_url"] = caveclient.info.viewer_site()
    NGL_CONFIG[config_key] = NGLUIConfig(**curr_config)


def reset_config(
    config_key: str = "default",
) -> None:
    """Reset the configuration for nglui viewers and statebuilders to the default.

    Parameters
    ----------
    config_key : str, optional
        Key for the configuration setting, by default "default"
    """
    NGL_CONFIG[config_key] = default_config


def neuroglancer_url(
    url: Optional[str] = None,
    target_site: Optional[str] = None,
    config_key: str = "default",
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
        url = get_ngl_config(config_key)["target_url"]
    else:
        url = NEUROGLANCER_SITES[target_site]
    return url


def reset_default_config():
    NGL_CONFIG[default_key] = default_config


def assign_site_parameters(
    url: Optional[str] = None,
    target_site: Optional[str] = None,
    client: Optional["caveclient.CAVEclient"] = None,
    config_key: Optional[str] = None,
) -> tuple[str, str, caveclient.CAVEclient]:
    if config_key is None:
        config_key = default_key
    if client is None:
        client = get_ngl_config(config_key)["caveclient"]
    url = neuroglancer_url(url, target_site, config_key)
    return url, client
