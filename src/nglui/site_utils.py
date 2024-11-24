import logging
import os
from typing import Literal, Optional

import attrs
import caveclient

logger = logging.getLogger(__name__)

MAINLINE_NAMES = ["spelunker", "mainline", "cave-explorer"]
SEUNGLAB_NAMES = ["seunglab"]
default_seunglab_neuroglancer_base = "https://neuromancer-seung-import.appspot.com/"
default_mainline_neuroglancer_base = "https://spelunker.cave-explorer.org/"

DEFAULT_TARGET_SITE = os.environ.get("NGLUI_DEFAULT_TARGET_SITE", "seunglab")
DEFAULT_URL = os.environ.get("NGLUI_DEFAULT_TARGET_URL", None)
DEFAULT_DATASTACK_NAME = os.environ.get("NGLUI_DEFAULT_DATASTACK_NAME", None)

__all__ = [
    "is_mainline",
    "is_seunglab",
    "check_target_site",
    "set_default_config",
    "get_default_config",
    "reset_default_config",
    "assign_site_parameters",
    MAINLINE_NAMES,
    SEUNGLAB_NAMES,
]


def is_mainline(target_name):
    return target_name in MAINLINE_NAMES


def is_seunglab(target_name):
    return target_name in SEUNGLAB_NAMES


def check_target_site(ngl_url, client):
    """
    Check neuroglancer info to determine which kind of site a neuroglancer URL is.
    """
    ngl_info = client.state.get_neuroglancer_info(ngl_url)
    if len(ngl_info) == 0:
        return SEUNGLAB_NAMES[0]
    else:
        return MAINLINE_NAMES[0]


@attrs.define
class NGLUIConfig:
    target_site = attrs.field(type=str)
    target_url = attrs.field(type=str)
    seunglab_fallback_url = attrs.field(
        default=default_seunglab_neuroglancer_base, type=str
    )
    mainline_fallback_url = attrs.field(
        default=default_mainline_neuroglancer_base, type=str
    )
    caveclient = attrs.field(default=None)
    datastack_name = attrs.field(default=DEFAULT_DATASTACK_NAME, type=str)

    def __attrs_post_init__(self):
        # validate target site
        if (
            self.target_site not in MAINLINE_NAMES + SEUNGLAB_NAMES
            and self.target_site is not None
        ):
            raise ValueError(
                f"target_site must be one of {MAINLINE_NAMES + SEUNGLAB_NAMES}"
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
        if self.target_url is not None and self.target_site is None:
            self.target_site = check_target_site(self.target_url, self.caveclient)
        if self.target_site is not None and self.target_url is None:
            if is_seunglab(self.target_site):
                self.target_url = self.seunglab_fallback_url
            else:
                self.target_url = self.mainline_fallback_url
        if self.target_site is not None and self.target_url is not None:
            if self.caveclient is not None:
                url_based_site = check_target_site(self.target_url, self.caveclient)
                if is_mainline(url_based_site) != is_mainline(self.target_site):
                    logging.warning(
                        f"target_site {self.target_site} does not match target_url {self.target_url}. Assuming manual setting is correct."
                    )


default_config = NGLUIConfig(
    target_site=DEFAULT_TARGET_SITE,
    target_url=DEFAULT_URL,
)
default_key = "default"
NGL_CONFIG = {default_key: default_config}  # type: dict[str, NGLUIConfig]


def get_default_config(config_key: str = None) -> dict:
    """Get the current configuration for nglui viewers and statebuilders.

    Parameters
    ----------
    config_key : str, optional
        Key for the configuration setting, by default "default"
    """
    if config_key is None:
        config_key = default_key
    return attrs.asdict(NGL_CONFIG[config_key])


def set_default_config(
    target_site: Optional[
        Literal["seunglab", "mainline", "cave-explorer", "spelunker"]
    ] = None,
    target_url: str = None,
    seunglab_fallback_url: str = None,
    mainline_fallback_url: str = None,
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
    seunglab_fallback_url : str, optional
        URL for the seunglab neuroglancer viewer, by default None
    mainline_fallback_url : str, optional
        URL for the mainline neuroglancer viewer, by default None
    datastack_name : str, optional
        Name of the datastack, by default None
    caveclient : caveclient.CAVEclient, optional
        CAVEclient object, by default None.
    """
    if config_key in NGL_CONFIG:
        curr_config = get_default_config(config_key)
    else:
        curr_config = attrs.asdict(default_config)
    if target_site is not None or target_url is not None:
        curr_config["target_site"] = target_site
        curr_config["target_url"] = target_url
    if seunglab_fallback_url is not None:
        curr_config["seunglab_fallback_url"] = seunglab_fallback_url
    if mainline_fallback_url is not None:
        curr_config["mainline_fallback_url"] = mainline_fallback_url
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
        url = get_default_config(config_key)["target_url"]
    elif is_seunglab(target_site):
        url = get_default_config(config_key)["seunglab_fallback_url"]
    else:
        url = get_default_config(config_key)["mainline_fallback_url"]
    return url


def get_target_site(
    url: Optional[str] = None,
    target_site: Optional[str] = None,
    config_key: Optional[str] = None,
    client: Optional["caveclient.CAVEclient"] = None,
) -> str:
    """Check target site based on parameters provided.

    Parameters
    ----------
    url : Optional[str], optional
        URL to a neuroglancer deployment, by default None
    target_site : Optional[str], optional
        Categorical neuroglancer target site, by default None
    config_key : Optional[str], optional
        Dictionary key for config, by default None
    client : Optional[caveclient.CAVEclient], optional
        Initialized CAVEclient, by default None

    Returns
    -------

        _description_
    """
    if target_site is not None:
        return target_site
    if url is not None:
        if client is not None:
            return check_target_site(url, client)
        else:
            msg = f"No client provided to check target site from URL. Assuming default value of \"{get_default_config(config_key)['target_site']}\""
            logging.warning(msg)
    return get_default_config(config_key)["target_site"]


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
        client = get_default_config(config_key)["caveclient"]
    url = neuroglancer_url(url, target_site, config_key)
    target_site = get_target_site(url, target_site, config_key, client)
    return url, target_site, client
