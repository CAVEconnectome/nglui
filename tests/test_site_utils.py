from nglui.statebuilder import site_utils


def test_site_utils():
    assert site_utils.neuroglancer_url() == site_utils.NEUROGLANCER_SITES.get(
        site_utils.DEFAULT_TARGET_SITE
    )
    assert site_utils.neuroglancer_url(
        target_site="spelunker"
    ) == site_utils.NEUROGLANCER_SITES.get("spelunker")
    assert (
        site_utils.neuroglancer_url(url="https://example.com") == "https://example.com"
    )

    site_utils.add_neuroglancer_site("testsite", "https://testsite.example.com")
    assert (
        site_utils.neuroglancer_url(target_site="testsite")
        == "https://testsite.example.com"
    )
