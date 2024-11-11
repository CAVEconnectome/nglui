MAINLINE_NAMES = ["mainline", "spelunker", "cave-explorer"]
SEUNGLAB_NAMES = ["seunglab"]
default_seunglab_neuroglancer_base = "https://neuromancer-seung-import.appspot.com/"
default_mainline_neuroglancer_base = "https://spelunker.cave-explorer.org/"


def is_mainline(target_name):
    return target_name in MAINLINE_NAMES


def is_seunglab(target_name):
    return target_name in SEUNGLAB_NAMES
