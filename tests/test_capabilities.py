"""Tests for capability detection of Neuroglancer deployments.

The probe is a convenience layered over state building, so the contract under test is
as much about *not* failing -- offline, on garbage, on an unknown fork -- as it is
about correctly identifying known deployments.
"""

import warnings

import pandas as pd
import pytest
import requests

from nglui.statebuilder import capabilities as caps

# Verbatim payloads from the real deployments, including the google demo's
# single-quoted body, which is a Python repr rather than valid JSON.
SPELUNKER_PAYLOAD = (
    '{"tag":"v2.37-347-g78c701ed", '
    '"url":"https://github.com/seung-lab/neuroglancer/commit/78c701ed", '
    '"timestamp":"Tue May 5 20:12:17 UTC 2026", '
    '"branch":"seung-lab/neuroglancer/spelunker"}'
)
GOOGLE_DEMO_PAYLOAD = (
    "{'tag':'v2.41.2-110-g3598da30', "
    "'url':'https://github.com/google/neuroglancer/commit/3598da30', "
    "'timestamp':'Sat Sep 5 09:21:38 UTC 2026'}"
)
OLD_GOOGLE_PAYLOAD = (
    '{"tag":"v2.41.2", '
    '"url":"https://github.com/google/neuroglancer/commit/deadbeef", '
    '"timestamp":"Tue Sep 23 12:00:00 UTC 2025"}'
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Capability results are process-cached; isolate every test from that."""
    caps.clear_capability_cache()
    yield
    caps.clear_capability_cache()


class TestParseCapabilities:
    @pytest.mark.parametrize(
        "alias,bool_props",
        [
            ("main", True),
            ("google", True),
            ("legacy", False),
            ("spelunker", False),
        ],
    )
    def test_string_aliases(self, alias, bool_props):
        assert caps.parse_capabilities(alias).annotation_bool_properties is bool_props

    def test_alias_is_case_insensitive(self):
        assert caps.parse_capabilities("Main") == caps.MAIN_CAPABILITIES

    def test_none_passes_through(self):
        assert caps.parse_capabilities(None) is None

    def test_instance_passes_through(self):
        assert (
            caps.parse_capabilities(caps.LEGACY_CAPABILITIES)
            is caps.LEGACY_CAPABILITIES
        )

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown capability alias"):
            caps.parse_capabilities("spelunkr")

    def test_bool_properties_is_not_an_alias(self):
        """It named one capability but enabled two; `from_names` is the precise way."""
        with pytest.raises(ValueError, match="Unknown capability alias"):
            caps.parse_capabilities("bool_properties")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            caps.parse_capabilities(42)


class TestCapabilitiesObject:
    def test_supports_named_capability(self):
        assert caps.MAIN_CAPABILITIES.supports(caps.ANNOTATION_BOOL_PROPERTIES)
        assert not caps.LEGACY_CAPABILITIES.supports(caps.ANNOTATION_BOOL_PROPERTIES)
        assert caps.LEGACY_CAPABILITIES.supports(caps.SPELUNKER_TAG_TOOLS)

    def test_unknown_capability_raises(self):
        with pytest.raises(ValueError, match="Unknown capability"):
            caps.MAIN_CAPABILITIES.supports("teleportation")

    def test_default_tracks_main(self):
        """Deployments overwhelmingly descend from main; the ones that do not are probed."""
        assert caps.get_default_capabilities() == caps.MAIN_CAPABILITIES

    def test_set_default_round_trip(self):
        original = caps.get_default_capabilities()
        try:
            caps.set_default_capabilities("main")
            assert caps.get_default_capabilities().annotation_bool_properties
        finally:
            caps.set_default_capabilities(original)


class TestVersionPayloadParsing:
    def test_parses_standard_json(self):
        info = caps._parse_version_payload(SPELUNKER_PAYLOAD)
        assert "seung-lab" in info["url"]

    def test_parses_python_repr_payload(self):
        """The google demo serves single quotes, which json.loads rejects."""
        info = caps._parse_version_payload(GOOGLE_DEMO_PAYLOAD)
        assert "google/neuroglancer" in info["url"]

    @pytest.mark.parametrize(
        "body", ["", "   ", "not json at all", "[1, 2, 3]", "null"]
    )
    def test_unparseable_bodies_return_none(self, body):
        assert caps._parse_version_payload(body) is None


class TestProbe:
    def _mock_response(self, mocker, body, status=200):
        response = mocker.Mock()
        response.text = body
        response.content = body.encode()
        response.status_code = status
        response.raise_for_status = mocker.Mock()
        return mocker.patch("requests.get", return_value=response)

    def test_probe_reads_origin_version_json(self, mocker):
        get = self._mock_response(mocker, SPELUNKER_PAYLOAD)
        info = caps.get_version_info("https://example.org/some/path?query=1")
        assert info["branch"].endswith("spelunker")
        assert get.call_args[0][0] == "https://example.org/version.json"

    def test_probe_result_is_cached(self, mocker):
        get = self._mock_response(mocker, SPELUNKER_PAYLOAD)
        caps.get_version_info("https://example.org/")
        caps.get_version_info("https://example.org/")
        assert get.call_count == 1

    def test_failures_are_cached_too(self, mocker):
        """An offline user pays the timeout once per process, not once per call."""
        get = mocker.patch("requests.get", side_effect=requests.Timeout("offline"))
        assert caps.get_version_info("https://unreachable.example/") is None
        assert caps.get_version_info("https://unreachable.example/") is None
        assert get.call_count == 1

    @pytest.mark.parametrize(
        "failure",
        [
            requests.Timeout("timed out"),
            requests.ConnectionError("no route"),
            requests.HTTPError("404"),
            ValueError("something odd"),
        ],
    )
    def test_all_failures_return_none(self, mocker, failure):
        mocker.patch("requests.get", side_effect=failure)
        assert caps.get_version_info("https://example.org/") is None

    def test_oversized_body_rejected(self, mocker):
        self._mock_response(mocker, "x" * (caps.MAX_VERSION_INFO_BYTES + 1))
        assert caps.get_version_info("https://example.org/") is None

    def test_env_var_disables_probe(self, mocker, monkeypatch):
        get = self._mock_response(mocker, SPELUNKER_PAYLOAD)
        monkeypatch.setenv(caps.DISABLE_PROBE_ENV_VAR, "1")
        assert caps.get_version_info("https://example.org/") is None
        assert get.call_count == 0

    def test_uses_short_timeout(self, mocker):
        get = self._mock_response(mocker, SPELUNKER_PAYLOAD)
        caps.get_version_info("https://example.org/")
        assert get.call_args[1]["timeout"] == caps.PROBE_TIMEOUT


class TestCapabilitiesForUrl:
    def test_identified_deployment_needs_no_warning(self, mocker):
        def get(url, **kwargs):
            response = mocker.Mock()
            response.text = (
                TestBundleProbe.INDEX if url.endswith("/") else TestBundleProbe.MODERN
            )
            response.content = response.text.encode()
            response.raise_for_status = mocker.Mock()
            return response

        mocker.patch("requests.get", side_effect=get)
        with warnings_as_errors():
            resolved = caps.capabilities_for_url("https://example.org/")
        assert resolved.annotation_bool_properties

    def test_unreachable_falls_back_to_default_with_warning(self, mocker):
        mocker.patch("requests.get", side_effect=requests.Timeout("offline"))
        with pytest.warns(UserWarning, match="Could not determine"):
            resolved = caps.capabilities_for_url("https://unreachable.example/")
        assert resolved == caps.get_default_capabilities()

    def test_fallback_warns_only_once_per_url(self, mocker):
        mocker.patch("requests.get", side_effect=requests.Timeout("offline"))
        with pytest.warns(UserWarning):
            caps.capabilities_for_url("https://unreachable.example/")
        with warnings_as_errors():
            caps.capabilities_for_url("https://unreachable.example/")

    def test_no_url_returns_default_silently(self):
        with warnings_as_errors():
            assert caps.capabilities_for_url(None) == caps.get_default_capabilities()

    def test_warning_can_be_suppressed(self, mocker):
        mocker.patch("requests.get", side_effect=requests.Timeout("offline"))
        with warnings_as_errors():
            caps.capabilities_for_url(
                "https://unreachable.example/", warn_on_fallback=False
            )


def warnings_as_errors():
    """Context manager asserting that no warning is raised."""
    import warnings as _w

    class _NoWarnings:
        def __enter__(self):
            self._ctx = _w.catch_warnings()
            self._ctx.__enter__()
            _w.simplefilter("error")
            return self

        def __exit__(self, *exc):
            return self._ctx.__exit__(*exc)

    return _NoWarnings()


class TestParserInfoShim:
    """`parser.info.get_ngl_info` predates this module and is kept working."""

    def test_delegates_to_get_version_info(self, mocker):
        from nglui.parser.info import get_ngl_info

        response = mocker.Mock()
        response.text = SPELUNKER_PAYLOAD
        response.content = SPELUNKER_PAYLOAD.encode()
        response.raise_for_status = mocker.Mock()
        mocker.patch("requests.get", return_value=response)
        assert get_ngl_info("https://example.org/")["branch"].endswith("spelunker")

    def test_returns_none_on_failure_rather_than_raising(self, mocker):
        from nglui.parser.info import get_ngl_info

        mocker.patch("requests.get", side_effect=requests.Timeout("offline"))
        assert get_ngl_info("https://unreachable.example/") is None


class TestCapabilityEncoding:
    """Capabilities are a named set, not a label; strings are one way in, not the model."""

    def test_provenance_does_not_affect_equality(self):
        """Two deployments with the same abilities are equivalent however each was learned."""
        probed = caps.Capabilities(
            annotation_bool_properties=True,
            annotation_property_tools=True,
            source="probe:https://example.org/",
        )
        assert probed == caps.parse_capabilities("main")
        assert probed.source != caps.parse_capabilities("main").source

    def test_equal_capabilities_hash_alike(self):
        probed = caps.Capabilities(
            annotation_bool_properties=True,
            annotation_property_tools=True,
            source="probe:somewhere",
        )
        assert hash(probed) == hash(caps.MAIN_CAPABILITIES)
        assert len({probed, caps.MAIN_CAPABILITIES}) == 1

    @pytest.mark.parametrize("not_a_capability", ["source", "uses_bool_tags", "names"])
    def test_supports_rejects_non_capability_attributes(self, not_a_capability):
        """`hasattr` would wave these through; they are not capabilities."""
        with pytest.raises(ValueError, match="Unknown capability"):
            caps.MAIN_CAPABILITIES.supports(not_a_capability)

    def test_membership_reads_as_a_set(self):
        assert caps.ANNOTATION_BOOL_PROPERTIES in caps.MAIN_CAPABILITIES
        assert caps.ANNOTATION_BOOL_PROPERTIES not in caps.LEGACY_CAPABILITIES
        assert caps.SPELUNKER_TAG_TOOLS in caps.LEGACY_CAPABILITIES

    def test_enabled_reports_only_capabilities(self):
        assert caps.MAIN_CAPABILITIES.enabled == frozenset(
            {"annotation_bool_properties", "annotation_property_tools"}
        )
        assert "source" not in caps.MAIN_CAPABILITIES.enabled

    def test_names_excludes_metadata_fields(self):
        assert "source" not in caps.Capabilities.names()
        assert caps.ANNOTATION_BOOL_PROPERTIES in caps.Capabilities.names()

    def test_from_names_round_trips(self):
        built = caps.Capabilities.from_names(caps.MAIN_CAPABILITIES.enabled)
        assert built == caps.MAIN_CAPABILITIES

    def test_from_names_reaches_sets_no_alias_covers(self, mocker):
        """A real build between the two landing dates has properties but not tools."""
        bundle = "x={rgb:void 0,int8:e.INT8,bool:e.UINT8};// no property tools yet"

        def get(url, **kwargs):
            response = mocker.Mock()
            response.text = TestBundleProbe.INDEX if url.endswith("/") else bundle
            response.content = response.text.encode()
            response.raise_for_status = mocker.Mock()
            return response

        mocker.patch("requests.get", side_effect=get)
        assert caps.Capabilities.from_names(
            [caps.ANNOTATION_BOOL_PROPERTIES]
        ) == caps.probe_capabilities("https://example.org/")

    def test_from_names_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown capabilities"):
            caps.Capabilities.from_names(["annotation_bool_properties", "warp_drive"])

    def test_a_capabilities_instance_passes_through_parse(self):
        built = caps.Capabilities.from_names([caps.ANNOTATION_BOOL_PROPERTIES])
        assert caps.parse_capabilities(built) is built


class TestDeploymentWithoutVersionJson:
    """Many deployments do not serve version.json at all.

    Detection is a convenience layered over state building, so every one of these
    has to fall back rather than fail. The fallback is the module default, which
    tracks main; what matters here is that nothing raises and a warning names the
    argument to set, since an unidentified deployment predating bool properties
    would drop the annotation layer.
    """

    def _get(self, mocker, body, status=200):
        response = mocker.Mock()
        response.text = body
        response.content = body.encode()
        response.raise_for_status = mocker.Mock(
            side_effect=requests.HTTPError(str(status)) if status >= 400 else None
        )
        return mocker.patch("requests.get", return_value=response)

    def test_404(self, mocker):
        self._get(mocker, "Not Found", status=404)
        with pytest.warns(UserWarning, match="Could not determine"):
            assert (
                caps.capabilities_for_url("https://example.org/")
                == caps.get_default_capabilities()
            )

    def test_static_host_serves_index_html_with_200(self, mocker):
        """A single-page host answers any path with the app, not a 404."""
        self._get(mocker, "<!doctype html><html><body>neuroglancer</body></html>")
        with pytest.warns(UserWarning, match="Could not determine"):
            assert (
                caps.capabilities_for_url("https://example.org/")
                == caps.get_default_capabilities()
            )

    def test_index_html_larger_than_the_read_cap(self, mocker):
        self._get(
            mocker, "<html>" + "x" * (caps.MAX_VERSION_INFO_BYTES + 1) + "</html>"
        )
        with pytest.warns(UserWarning):
            assert (
                caps.capabilities_for_url("https://example.org/")
                == caps.get_default_capabilities()
            )

    def test_json_that_is_not_version_info(self, mocker):
        self._get(mocker, '{"hello": "world"}')
        with pytest.warns(UserWarning):
            assert (
                caps.capabilities_for_url("https://example.org/")
                == caps.get_default_capabilities()
            )

    def test_version_json_without_a_tag_falls_back(self, mocker):
        """Without a describe string there is no trustworthy signal.

        The build timestamp is deliberately not used as a substitute: it records when
        a build was cut, not what is in it, so a fresh rebuild of an old branch would
        look capable. Claiming capability wrongly is the expensive direction.
        """
        self._get(
            mocker,
            '{"url": "https://github.com/google/neuroglancer/commit/abc",'
            ' "timestamp": "Sat Sep 5 09:21:38 UTC 2026"}',
        )
        with pytest.warns(UserWarning):
            assert (
                caps.capabilities_for_url("https://example.org/")
                == caps.get_default_capabilities()
            )

    def test_a_tagged_state_still_builds_offline(self, mocker):
        """The point of all of the above: state building must not fail."""
        import pandas as pd

        from nglui.statebuilder import ViewerState

        mocker.patch("requests.get", side_effect=requests.ConnectionError("refused"))
        vs = ViewerState(
            dimensions=[1, 1, 1], target_url="https://unreachable.example/"
        )
        with pytest.warns(UserWarning):
            vs.add_points(
                pd.DataFrame({"x": [1], "y": [2], "z": [3], "ct": ["axon"]}),
                point_column=["x", "y", "z"],
                tag_column="ct",
                linked_segmentation=None,
            )
            layer = vs.to_dict()["layers"][0]
        assert layer["annotationProperties"] == [{"id": "axon", "type": "bool"}]


class TestBundleProbe:
    """Capabilities are read from the deployed client rather than inferred.

    A fork describes against the last tag in its own repository, so its release says
    nothing about which upstream features it merged -- Spelunker reported `v2.37` both
    before and after gaining the whole bool-property system. Asking the bundle is the
    only answer that survives a backport.
    """

    INDEX = '<html><body><script src="main.abc123.js"></script></body></html>'
    # Shapes taken from real bundles: the property type table lists `bool` beside the
    # numeric types, and tool type names travel in state JSON so they survive minifying.
    MODERN = 'x={rgb:void 0,int8:e.INT8,bool:e.UINT8};let d="toggleBoolProperty";'
    SPELUNKER_OLD = 'x={rgb:void 0,int8:e.INT8};class a{static TOOL_ID="tagTool";}'
    SPELUNKER_NEW = MODERN + 'class a{static TOOL_ID="tagTool";}'
    ANCIENT = "x={rgb:void 0,int8:e.INT8};// no tags at all"

    def _serve(self, mocker, bundle):
        def get(url, **kwargs):
            response = mocker.Mock()
            response.text = self.INDEX if url.endswith("/") else bundle
            response.content = response.text.encode()
            response.raise_for_status = mocker.Mock()
            return response

        return mocker.patch("requests.get", side_effect=get)

    def test_modern_deployment(self, mocker):
        self._serve(mocker, self.MODERN)
        probed = caps.probe_capabilities("https://example.org/")
        assert probed.enabled == frozenset(
            {"annotation_bool_properties", "annotation_property_tools"}
        )

    def test_spelunker_before_the_backport(self, mocker):
        """v2.37-347: tag tools only."""
        self._serve(mocker, self.SPELUNKER_OLD)
        probed = caps.probe_capabilities("https://example.org/")
        assert probed.enabled == frozenset({"spelunker_tag_tools"})

    def test_spelunker_after_the_backport(self, mocker):
        """v2.37-398: the same release, now carrying everything."""
        self._serve(mocker, self.SPELUNKER_NEW)
        probed = caps.probe_capabilities("https://example.org/")
        assert probed.enabled == frozenset(
            {
                "annotation_bool_properties",
                "annotation_property_tools",
                "spelunker_tag_tools",
            }
        )

    def test_deployment_with_neither_system(self, mocker):
        self._serve(mocker, self.ANCIENT)
        assert caps.probe_capabilities("https://example.org/").enabled == frozenset()

    def test_probe_is_cached(self, mocker):
        get = self._serve(mocker, self.MODERN)
        caps.probe_capabilities("https://example.org/")
        caps.probe_capabilities("https://example.org/")
        assert get.call_count == 2  # index + bundle, once

    def test_index_without_a_bundle_is_a_non_answer(self, mocker):
        response = mocker.Mock()
        response.text = "<html><body>no script here</body></html>"
        response.content = response.text.encode()
        response.raise_for_status = mocker.Mock()
        mocker.patch("requests.get", return_value=response)
        assert caps.probe_capabilities("https://example.org/") is None

    @pytest.mark.parametrize(
        "failure",
        [
            requests.Timeout("t"),
            requests.ConnectionError("c"),
            requests.HTTPError("404"),
        ],
    )
    def test_failures_are_non_answers(self, mocker, failure):
        mocker.patch("requests.get", side_effect=failure)
        assert caps.probe_capabilities("https://example.org/") is None

    def test_env_var_disables_the_probe(self, mocker, monkeypatch):
        get = self._serve(mocker, self.MODERN)
        monkeypatch.setenv(caps.DISABLE_PROBE_ENV_VAR, "1")
        assert caps.probe_capabilities("https://example.org/") is None
        assert get.call_count == 0


class TestOfflineStateBuilding:
    """Building a state must not depend on reaching the network.

    The probe is a convenience; every way it can fail has to leave a usable state.
    """

    @pytest.fixture
    def tagged_df(self):
        return pd.DataFrame({"x": [1, 2], "y": [1, 2], "z": [1, 2], "ct": ["a", "b"]})

    def _build(self, tagged_df, n=1):
        from nglui.statebuilder import ViewerState

        layers = []
        for _ in range(n):
            vs = ViewerState(dimensions=[1, 1, 1])
            vs.add_points(
                tagged_df,
                point_column=["x", "y", "z"],
                tag_column="ct",
                linked_segmentation=None,
            )
            layers.append(vs.to_dict()["layers"][0])
        return layers

    def test_offline_still_produces_a_valid_state(self, mocker, tagged_df):
        mocker.patch("requests.get", side_effect=requests.ConnectionError("offline"))
        with pytest.warns(UserWarning, match="Could not determine"):
            layers = self._build(tagged_df)
        assert len(layers[0]["annotationProperties"]) == 2
        assert all(len(a["props"]) == 2 for a in layers[0]["annotations"])

    def test_offline_costs_one_attempt_per_process(self, mocker, tagged_df):
        get = mocker.patch(
            "requests.get", side_effect=requests.ConnectionError("offline")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._build(tagged_df, n=5)
        assert get.call_count == 1

    def test_a_tagless_state_never_reaches_the_network(self, mocker):
        from nglui.statebuilder import ViewerState

        get = mocker.patch("requests.get", side_effect=AssertionError("probed!"))
        vs = ViewerState(dimensions=[1, 1, 1])
        vs.add_points(
            pd.DataFrame({"x": [1], "y": [1], "z": [1]}),
            point_column=["x", "y", "z"],
            linked_segmentation=None,
        )
        vs.to_dict()
        assert get.call_count == 0

    def test_opting_out_is_silent(self, mocker, monkeypatch, tagged_df):
        """Disabling the probe is a choice, not a failure to determine anything."""
        monkeypatch.setenv(caps.DISABLE_PROBE_ENV_VAR, "1")
        get = mocker.patch("requests.get", side_effect=AssertionError("probed!"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._build(tagged_df)
        assert get.call_count == 0

    def test_pinned_capabilities_never_reach_the_network(self, mocker, tagged_df):
        from nglui.statebuilder import ViewerState

        get = mocker.patch("requests.get", side_effect=AssertionError("probed!"))
        vs = ViewerState(dimensions=[1, 1, 1], capabilities="legacy")
        vs.add_points(
            tagged_df,
            point_column=["x", "y", "z"],
            tag_column="ct",
            linked_segmentation=None,
        )
        vs.to_dict()
        assert get.call_count == 0

    def test_the_bundle_read_is_bounded(self):
        """A misbehaving deployment must not stall a state build indefinitely."""
        assert caps.BUNDLE_TIMEOUT[1] <= 5.0
        assert caps.PROBE_TIMEOUT[1] <= 2.0


class TestCacheKeying:
    """One deployment is one cache entry, however its URL is spelled.

    Keying on the raw argument meant `https://x.org/`, `https://x.org` and a URL with
    a state fragment were three separate probes of the same deployment -- and with a
    64-entry cache, a session working with many links could evict real answers.
    """

    BUNDLE = 'x={int8:e.INT8,bool:e.UINT8};d="toggleBoolProperty";'

    def _serve(self, mocker):
        def get(url, **kwargs):
            response = mocker.Mock()
            response.text = (
                '<script src="main.a.js"></script>'
                if url.endswith("/")
                else self.BUNDLE
            )
            response.content = response.text.encode()
            response.raise_for_status = mocker.Mock()
            return response

        return mocker.patch("requests.get", side_effect=get)

    @pytest.mark.parametrize(
        "url",
        [
            "https://x.org",
            "https://x.org/",
            "https://x.org/ngl",
            "https://x.org/ngl?query=1",
            "https://x.org/#!%7B%22layers%22:%5B%5D%7D",
        ],
    )
    def test_every_spelling_is_one_entry(self, mocker, url):
        get = self._serve(mocker)
        caps.probe_capabilities("https://x.org/")
        before = get.call_count
        caps.probe_capabilities(url)
        assert get.call_count == before
        assert len(caps._capability_cache) == 1

    def test_distinct_origins_are_distinct_entries(self, mocker):
        self._serve(mocker)
        caps.probe_capabilities("https://a.example/")
        caps.probe_capabilities("https://b.example/")
        assert len(caps._capability_cache) == 2

    def test_a_state_url_does_not_evict_the_plain_one(self, mocker):
        """State URLs are long and varied; they must not each take a cache slot."""
        self._serve(mocker)
        for i in range(100):
            caps.probe_capabilities(f"https://x.org/#!%7B%22n%22:{i}%7D")
        assert len(caps._capability_cache) == 1

    def test_warnings_are_also_keyed_on_origin(self, mocker):
        mocker.patch("requests.get", side_effect=requests.ConnectionError("offline"))
        with pytest.warns(UserWarning, match="Could not determine"):
            caps.capabilities_for_url("https://x.org/")
        with warnings_as_errors():
            caps.capabilities_for_url("https://x.org/some/other/path")

    def test_clear_resets_everything(self, mocker):
        self._serve(mocker)
        caps.probe_capabilities("https://x.org/")
        caps._warned_origins.add("https://x.org/")
        caps.clear_capability_cache()
        assert len(caps._capability_cache) == 0
        assert len(caps._warned_origins) == 0

    def test_entries_expire(self):
        """A long-lived session must eventually see a redeployment."""
        assert caps._capability_cache.ttl == caps.CACHE_TTL_SECONDS
        assert caps.CACHE_TTL_SECONDS <= 3600
