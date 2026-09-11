import pandas as pd
import pytest
from pytest_mock import mocker

from nglui import statebuilder
from nglui.segmentprops import SegmentProperties
from nglui.statebuilder import ViewerState

ROOT_IDS = [864691136137805181, 864691135474648896]

PROPERTY_ID = "test_property_id"
PROPERTY_URL = "precomputed://middleauth+https://local.cave.com/nglstate/api/v1/property/test_property_id"


def source_urls(vs, name="segmentation"):
    """Rendered source URLs for a layer, in order."""
    layer = [l for l in vs.to_dict()["layers"] if l["name"] == name][0]
    return [src["url"] for src in layer["source"]]


def selected_segments(vs, name="segmentation"):
    """Root ids selected in a segmentation layer, as ints."""
    return [int(seg) for seg in vs.layers[name].segments]


def query_sideeffect(**kwargs):
    print(kwargs)
    if "pre_ids" in kwargs:
        df = pd.read_feather("tests/testdata/pre_df_helper.feather")
    elif "post_ids" in kwargs:
        df = pd.read_feather("tests/testdata/post_df_helper.feather")
    df.attrs["dataframe_resolution"] = [4, 4, 40]
    return df


@pytest.fixture
def mock_property_upload(client_full, mocker):
    """Patch the state service so segment property uploads do not hit the network."""
    upload = mocker.patch.object(
        client_full.state, "upload_property_json", return_value=PROPERTY_ID
    )
    build = mocker.patch.object(
        client_full.state, "build_neuroglancer_url", return_value=PROPERTY_URL
    )
    return upload, build


@pytest.fixture
def segment_properties():
    return SegmentProperties.from_dataframe(
        pd.DataFrame({"pt_root_id": ROOT_IDS, "cell_type": ["pyr", "bc"]}),
        id_col="pt_root_id",
        label_col="cell_type",
    )


def test_neuron_helper(client_full, mocker):
    root_id = 864691136137805181
    client_full.materialize.synapse_query = mocker.Mock()
    client_full.materialize.synapse_query.side_effect = query_sideeffect

    print(client_full.info.get_datastack_info())

    state_dict_spelunker = statebuilder.helpers.make_neuron_neuroglancer_link(
        client_full,
        root_id,
        return_as="dict",
        shorten="never",
        show_inputs=True,
        show_outputs=True,
        infer_coordinates=False,
    )
    assert "dimensions" in state_dict_spelunker.keys()


class TestMakeSegmentState:
    def test_layers_and_segments(self, client_full):
        vs = statebuilder.helpers.make_segment_state(
            client_full,
            root_ids=ROOT_IDS,
            infer_coordinates=False,
        )
        assert isinstance(vs, ViewerState)
        assert vs.layer_names == ["imagery", "segmentation"]
        assert selected_segments(vs) == ROOT_IDS

    def test_single_root_id(self, client_full):
        vs = statebuilder.helpers.make_segment_state(
            client_full,
            root_ids=ROOT_IDS[0],
            infer_coordinates=False,
        )
        assert selected_segments(vs) == [ROOT_IDS[0]]

    def test_root_ids_from_series(self, client_full):
        vs = statebuilder.helpers.make_segment_state(
            client_full,
            root_ids=pd.Series(ROOT_IDS),
            infer_coordinates=False,
        )
        assert selected_segments(vs) == ROOT_IDS

    def test_root_ids_dict_preserves_visibility(self, client_full):
        vs = statebuilder.helpers.make_segment_state(
            client_full,
            root_ids={ROOT_IDS[0]: True, ROOT_IDS[1]: False},
            infer_coordinates=False,
        )
        state = vs.to_dict()["layers"][1]
        assert state["segments"] == [str(ROOT_IDS[0]), f"!{ROOT_IDS[1]}"]

    def test_no_root_ids(self, client_full):
        vs = statebuilder.helpers.make_segment_state(
            client_full,
            infer_coordinates=False,
        )
        state = vs.to_dict()
        assert len(state["layers"]) == 2
        assert state["layers"][1]["segments"] == []

    def test_custom_layer_names(self, client_full):
        vs = statebuilder.helpers.make_segment_state(
            client_full,
            root_ids=ROOT_IDS,
            segmentation_layer_name="seg",
            imagery="em",
            infer_coordinates=False,
        )
        assert vs.layer_names == ["em", "seg"]
        assert selected_segments(vs, "seg") == ROOT_IDS

    def test_no_imagery(self, client_full):
        vs = statebuilder.helpers.make_segment_state(
            client_full,
            root_ids=ROOT_IDS,
            imagery=False,
            infer_coordinates=False,
        )
        assert vs.layer_names == ["segmentation"]
        assert selected_segments(vs) == ROOT_IDS

    def test_view_options(self, client_full):
        vs = statebuilder.helpers.make_segment_state(
            client_full,
            root_ids=ROOT_IDS,
            selected_alpha=0.5,
            alpha_3d=0.7,
            mesh_silhouette=0.3,
            infer_coordinates=False,
        )
        seg_layer = vs.layers["segmentation"].to_dict()
        assert seg_layer["selectedAlpha"] == 0.5
        assert seg_layer["objectAlpha"] == 0.7
        assert seg_layer["meshSilhouetteRendering"] == 0.3


class TestSegmentProperties:
    def test_segment_properties_object(
        self, client_full, segment_properties, mock_property_upload
    ):
        upload, build = mock_property_upload
        vs = statebuilder.helpers.make_segment_state(
            client_full,
            root_ids=ROOT_IDS,
            segment_properties=segment_properties,
            infer_coordinates=False,
        )
        upload.assert_called_once_with(segment_properties.to_dict())
        build.assert_called_once_with(PROPERTY_ID, format_properties=True)
        assert PROPERTY_URL in source_urls(vs)

    def test_segment_properties_dict(
        self, client_full, segment_properties, mock_property_upload
    ):
        upload, _ = mock_property_upload
        prop_json = segment_properties.to_dict()
        vs = statebuilder.helpers.make_segment_state(
            client_full,
            root_ids=ROOT_IDS,
            segment_properties=prop_json,
            infer_coordinates=False,
        )
        upload.assert_called_once_with(prop_json)
        assert PROPERTY_URL in source_urls(vs)

    def test_segment_properties_custom_layer(
        self, client_full, segment_properties, mock_property_upload
    ):
        vs = statebuilder.helpers.make_segment_state(
            client_full,
            root_ids=ROOT_IDS,
            segment_properties=segment_properties,
            segmentation_layer_name="seg",
            infer_coordinates=False,
        )
        assert PROPERTY_URL in source_urls(vs, "seg")

    def test_missing_layer_reports_clearly_and_uploads_nothing(
        self, client_full, mock_property_upload, segment_properties
    ):
        """The upload is a side effect on the state service, so validate first."""
        upload, _ = mock_property_upload
        vs = statebuilder.helpers.make_segment_state(
            client_full, root_ids=ROOT_IDS, infer_coordinates=False
        )
        with pytest.raises(ValueError, match="No layer named 'typo'"):
            statebuilder.helpers.add_segment_properties_source(
                vs, segment_properties, client=client_full, name="typo"
            )
        upload.assert_not_called()

    def test_segment_properties_bad_type(self, client_full, mock_property_upload):
        with pytest.raises(TypeError):
            statebuilder.helpers.make_segment_state(
                client_full,
                root_ids=ROOT_IDS,
                segment_properties="not-a-property",
                infer_coordinates=False,
            )

    def test_neuron_link_accepts_segment_properties(
        self, client_full, segment_properties, mock_property_upload, mocker
    ):
        upload, _ = mock_property_upload
        mocker.patch.object(
            client_full.materialize,
            "synapse_query",
            side_effect=query_sideeffect,
        )
        vs = statebuilder.helpers.make_neuron_neuroglancer_link(
            client_full,
            ROOT_IDS[0],
            return_as="viewer",
            segment_properties=segment_properties,
            infer_coordinates=False,
        )
        upload.assert_called_once()
        assert PROPERTY_URL in source_urls(vs)


class TestNeuronLinkShortening:
    """The shortener bug was in make_neuron_neuroglancer_link specifically.

    Its sibling make_segment_link covers the shared rendering path, but nothing
    pinned that the neuron helper passes its own client through to the shortener --
    which is exactly what it failed to do.
    """

    def test_shorten_always_passes_the_client(self, client_full, mocker):
        client_full.materialize.synapse_query = mocker.Mock(
            side_effect=query_sideeffect
        )
        upload = mocker.patch.object(
            client_full.state, "upload_state_json", return_value=999
        )
        mocker.patch.object(
            client_full.state,
            "build_neuroglancer_url",
            return_value="https://short.url/999",
        )
        url = statebuilder.helpers.make_neuron_neuroglancer_link(
            client_full,
            ROOT_IDS[0],
            return_as="url",
            shorten="always",
            show_inputs=False,
            show_outputs=False,
            infer_coordinates=False,
        )
        upload.assert_called_once()
        assert url == "https://short.url/999"


class TestMakeSegmentLink:
    def test_return_as_viewer(self, client_full):
        vs = statebuilder.helpers.make_segment_link(
            client_full,
            root_ids=ROOT_IDS,
            return_as="viewer",
            infer_coordinates=False,
        )
        assert isinstance(vs, ViewerState)

    def test_return_as_dict(self, client_full):
        state = statebuilder.helpers.make_segment_link(
            client_full,
            root_ids=ROOT_IDS,
            return_as="dict",
            infer_coordinates=False,
        )
        assert "dimensions" in state
        assert state["layers"][1]["segments"] == [str(r) for r in ROOT_IDS]

    def test_return_as_json(self, client_full):
        state_json = statebuilder.helpers.make_segment_link(
            client_full,
            root_ids=ROOT_IDS,
            return_as="json",
            infer_coordinates=False,
        )
        assert isinstance(state_json, str)
        assert str(ROOT_IDS[0]) in state_json

    def test_return_as_url(self, client_full):
        url = statebuilder.helpers.make_segment_link(
            client_full,
            root_ids=ROOT_IDS,
            return_as="url",
            shorten="never",
            target_site="spelunker",
            infer_coordinates=False,
        )
        assert isinstance(url, str)
        assert url.startswith("http")

    def test_return_as_link(self, client_full):
        link = statebuilder.helpers.make_segment_link(
            client_full,
            root_ids=ROOT_IDS,
            return_as="link",
            shorten=False,
            target_site="spelunker",
            link_text="My Neurons",
            infer_coordinates=False,
        )
        assert "My Neurons" in link.data

    def test_shorten_always(self, client_full, mocker):
        upload = mocker.patch.object(
            client_full.state, "upload_state_json", return_value=12345
        )
        mocker.patch.object(
            client_full.state,
            "build_neuroglancer_url",
            return_value="https://short.url/12345",
        )
        url = statebuilder.helpers.make_segment_link(
            client_full,
            root_ids=ROOT_IDS,
            return_as="url",
            shorten="always",
            target_site="spelunker",
            infer_coordinates=False,
        )
        upload.assert_called_once()
        assert url == "https://short.url/12345"

    def test_return_as_clipboard(self, client_full, mocker):
        copy = mocker.patch("nglui.statebuilder.base.pyperclip.copy")
        url = statebuilder.helpers.make_segment_link(
            client_full,
            root_ids=ROOT_IDS,
            return_as="clipboard",
            shorten=False,
            target_site="spelunker",
            infer_coordinates=False,
        )
        copy.assert_called_once_with(url)

    def test_return_as_browser(self, client_full, mocker):
        open_browser = mocker.patch("webbrowser.open", return_value=True)
        statebuilder.helpers.make_segment_link(
            client_full,
            root_ids=ROOT_IDS,
            return_as="browser",
            shorten=False,
            target_site="spelunker",
            infer_coordinates=False,
        )
        open_browser.assert_called_once()

    def test_invalid_return_as(self, client_full):
        with pytest.raises(ValueError, match="Invalid return_as value"):
            statebuilder.helpers.make_segment_link(
                client_full,
                root_ids=ROOT_IDS,
                return_as="nonsense",
                infer_coordinates=False,
            )
