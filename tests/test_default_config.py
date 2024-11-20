from urllib.parse import urlparse

from nglui.site_utils import (
    get_default_config,
    neuroglancer_url,
    reset_default_config,
    set_default_config,
)
from nglui.statebuilder import (
    ImageLayerConfig,
    SegmentationLayerConfig,
    StateBuilder,
)


def test_simple_config():
    config = get_default_config()
    assert "target_site" in config

    set_default_config(target_site="spelunker")
    assert get_default_config()["target_url"] == config["mainline_fallback_url"]

    set_default_config(target_site="seunglab")
    assert get_default_config()["target_url"] == config["seunglab_fallback_url"]

    assert neuroglancer_url(None, "spelunker") == config["mainline_fallback_url"]
    assert neuroglancer_url(None, "seunglab") == config["seunglab_fallback_url"]


def test_client_config(client_simple):
    set_default_config(caveclient=client_simple)
    assert get_default_config()["datastack_name"] == client_simple.datastack_name


def test_statebuilder_integration(client_simple):
    set_default_config(caveclient=client_simple)
    img = ImageLayerConfig(name="img", source="precomputed://gs://fakebucket")
    seg = SegmentationLayerConfig(
        name="seg", source="precomputed://gs://fakebucket/seg"
    )
    sb = StateBuilder([img, seg])
    assert sb._client == client_simple

    set_default_config(target_site="spelunker")
    sb = StateBuilder([img, seg])
    url = sb.render_state(return_as="url")
    assert (
        urlparse(url).netloc
        == urlparse(get_default_config()["mainline_fallback_url"]).netloc
    )
    reset_default_config()
