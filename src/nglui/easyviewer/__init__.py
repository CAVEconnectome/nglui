from typing import Optional
from .ev_base import EasyViewerMainline, EasyViewerSeunglab
from ..site_utils import is_mainline, get_default_config


def EasyViewer(
    target_site: Optional[str] = None,
    config_key: str = "default",
):
    """Factory function to return the appropriate EasyViewer object based on the target site.

    Parameters
    ----------
    target_site : Optional[str], optional
        Type of target site, either "seunglab" or ""spelunker/mainline/"cave-explorer", by default None.
        If None, uses the configuration setting.
    config_key : str, optional
        Name of the configuration to use, by default "default"

    Returns
    -------
    EasyViewer
        EasyViewer object based on the target site to aid in building states.
    """
    if target_site is None:
        target_site = get_default_config(config_key)["target_site"]
    if not is_mainline(target_site):
        return EasyViewerSeunglab()
    elif is_mainline(target_site):
        return EasyViewerMainline()
    else:
        raise ValueError(
            f'Unknown target: {target_site}. Must be one of "seunglab" or "spelunker/mainline"/"cave-explorer"'
        )
