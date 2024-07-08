from .ev_base import EasyViewerSeunglab, EasyViewerMainline


def EasyViewer(
    target_site="seunglab",
):
    if target_site == "seunglab" or target_site is None:
        return EasyViewerSeunglab()
    elif (
        target_site == "mainline"
        or target_site == "cave-explorer"
        or target_site == "spelunker"
    ):
        return EasyViewerMainline()
    else:
        raise ValueError(
            f'Unknown target: {target_site}. Must be one of "seunglab" or "spelunker/mainline"/"cave-explorer"'
        )
