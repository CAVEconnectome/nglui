"""Tests for capability detection of Neuroglancer deployments.

The probe is a convenience layered over state building, so the contract under test is
as much about *not* failing -- offline, on garbage, on an unknown fork -- as it is
about correctly identifying known deployments.
"""

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


class TestCapabilitiesFromVersionInfo:
    def test_spelunker_is_tag_tools(self):
        info = caps._parse_version_payload(SPELUNKER_PAYLOAD)
        resolved = caps.capabilities_from_version_info(info)
        assert resolved.spelunker_tag_tools
        assert not resolved.annotation_bool_properties

    def test_recent_google_supports_bool_properties(self):
        info = caps._parse_version_payload(GOOGLE_DEMO_PAYLOAD)
        resolved = caps.capabilities_from_version_info(info)
        assert resolved.annotation_bool_properties
        assert resolved.annotation_property_tools

    def test_old_google_does_not(self):
        info = caps._parse_version_payload(OLD_GOOGLE_PAYLOAD)
        resolved = caps.capabilities_from_version_info(info)
        assert not resolved.annotation_bool_properties

    def test_bool_properties_precede_property_tools(self):
        """Between the two landing dates, properties work but the tools do not."""
        info = {
            "url": "https://github.com/google/neuroglancer/commit/abc",
            "tag": "v2.41.2-40-gabc1234",
            "timestamp": "Wed Jul 1 00:00:00 UTC 2026",
        }
        resolved = caps.capabilities_from_version_info(info)
        assert resolved.annotation_bool_properties
        assert not resolved.annotation_property_tools

    def test_old_release_is_legacy_whatever_the_build_date(self):
        """A fresh rebuild of an old branch is still an old branch."""
        info = {
            "url": "https://github.com/seung-lab/neuroglancer/commit/abc",
            "tag": "v2.37-347-gabc1234",
            "branch": "seung-lab/neuroglancer/spelunker",
            "timestamp": "Sat Sep 5 09:21:38 UTC 2026",
        }
        resolved = caps.capabilities_from_version_info(info)
        assert not resolved.annotation_bool_properties
        assert resolved.spelunker_tag_tools

    def test_tag_tools_come_from_the_branch_not_the_repository(self):
        """base-cave and spelunker share a repository but are different deployments."""
        base = {
            "url": "https://github.com/seung-lab/neuroglancer/commit/383b1cfe",
            "tag": "v2.37-32-g383b1cfe",
            "branch": "seung-lab/neuroglancer/base-cave",
        }
        spelunker = dict(base, branch="seung-lab/neuroglancer/spelunker")
        assert not caps.capabilities_from_version_info(base).spelunker_tag_tools
        assert caps.capabilities_from_version_info(spelunker).spelunker_tag_tools

    def test_missing_branch_is_not_tag_tooled(self):
        """The Google demo serves no branch field."""
        info = {
            "url": "https://github.com/google/neuroglancer/commit/3598da30",
            "tag": "v2.41.2-110-g3598da30",
            "timestamp": "Sat Sep 5 09:21:38 UTC 2026",
        }
        assert not caps.capabilities_from_version_info(info).spelunker_tag_tools

    def test_rebased_fork_is_recognized_without_an_allowlist(self):
        """A fork that rebases past the feature release gains it automatically.

        This is the point of reading the describe tag: no table of known deployments
        has to be edited when Spelunker moves onto newer Neuroglancer.
        """
        info = {
            "url": "https://github.com/seung-lab/neuroglancer/commit/ffff",
            "tag": "v2.41.2-400-gffffaaa",
            "branch": "seung-lab/neuroglancer/spelunker",
            "timestamp": "Wed Sep 9 00:00:00 UTC 2026",
        }
        resolved = caps.capabilities_from_version_info(info)
        assert resolved.annotation_bool_properties
        assert resolved.annotation_property_tools
        # Tag tooling is an independent axis -- a rebased fork keeps both.
        assert resolved.spelunker_tag_tools

    def test_unknown_fork_is_classified_by_its_release(self):
        info = {
            "url": "https://github.com/someone/neuroglancer-fork/commit/abc",
            "tag": "v2.41.2-200-gabc1234",
            "timestamp": "Sat Sep 5 09:21:38 UTC 2026",
        }
        resolved = caps.capabilities_from_version_info(info)
        assert resolved.annotation_bool_properties
        assert not resolved.spelunker_tag_tools

    def test_later_release_needs_no_timestamp(self):
        info = {
            "url": "https://github.com/google/neuroglancer/commit/zzz",
            "tag": "v2.42.0",
        }
        resolved = caps.capabilities_from_version_info(info)
        assert resolved.annotation_bool_properties
        assert resolved.annotation_property_tools

    def test_exactly_the_feature_base_release_predates_the_features(self):
        """v2.41.2 was tagged before either feature landed."""
        info = {
            "url": "https://github.com/google/neuroglancer/commit/yyy",
            "tag": "v2.41.2",
            "timestamp": "Tue Sep 23 12:00:00 UTC 2025",
        }
        resolved = caps.capabilities_from_version_info(info)
        assert not resolved.annotation_bool_properties

    def test_unparseable_tag_is_undeterminable(self):
        info = {
            "url": "https://github.com/google/neuroglancer/commit/abc",
            "tag": "some-custom-build",
            "timestamp": "Sat Sep 5 09:21:38 UTC 2026",
        }
        assert caps.capabilities_from_version_info(info) is None

    def test_unparseable_timestamp_is_undeterminable_at_the_boundary(self):
        """Only builds cut from the feature release need the timestamp at all."""
        info = {
            "url": "https://github.com/google/neuroglancer/commit/abc",
            "tag": "v2.41.2-40-gabc1234",
            "timestamp": "yesterday",
        }
        assert caps.capabilities_from_version_info(info) is None

    def test_empty_info_is_undeterminable(self):
        assert caps.capabilities_from_version_info(None) is None
        assert caps.capabilities_from_version_info({}) is None


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
        response = mocker.Mock()
        response.text = GOOGLE_DEMO_PAYLOAD
        response.content = GOOGLE_DEMO_PAYLOAD.encode()
        response.raise_for_status = mocker.Mock()
        mocker.patch("requests.get", return_value=response)
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


class TestParseDescribeTag:
    """version.json carries `git describe` output: release, commits since, sha."""

    @pytest.mark.parametrize(
        "tag,expected",
        [
            ("v2.37-347-g78c701ed", ((2, 37), 347)),
            ("v2.41.2-110-g3598da30", ((2, 41, 2), 110)),
            ("v2.41.2", ((2, 41, 2), 0)),
            ("2.41.2-1-gdeadbee", ((2, 41, 2), 1)),
            ("v3.0-5-gabc1234", ((3, 0), 5)),
            ("v2.41.2-110-g3598da30-dirty", ((2, 41, 2), 110)),
        ],
    )
    def test_parses_real_forms(self, tag, expected):
        assert caps.parse_describe_tag(tag) == expected

    @pytest.mark.parametrize("tag", ["", "garbage", "release-7", "v", None])
    def test_unparseable_returns_none(self, tag):
        assert caps.parse_describe_tag(tag) is None

    def test_release_ordering_is_numeric_not_lexical(self):
        """(2, 37) must sort below (2, 41, 2); '2.37' > '2.41' as strings."""
        older, _ = caps.parse_describe_tag("v2.37-347-gabc1234")
        newer, _ = caps.parse_describe_tag("v2.41.2-1-gabc1234")
        assert older < newer


class TestCapabilityEncoding:
    """Capabilities are a named set, not a label; strings are one way in, not the model."""

    def test_provenance_does_not_affect_equality(self):
        """Two deployments with the same abilities are equivalent however each was learned."""
        probed = caps.capabilities_from_version_info(
            {
                "url": "https://github.com/google/neuroglancer/commit/abc",
                "tag": "v2.41.2-110-gabc1234",
                "timestamp": "Sat Sep 5 09:21:38 UTC 2026",
            }
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

    def test_from_names_reaches_sets_no_alias_covers(self):
        """A real build between the two landing dates has properties but not tools."""
        partial = caps.Capabilities.from_names([caps.ANNOTATION_BOOL_PROPERTIES])
        probed = caps.capabilities_from_version_info(
            {
                "url": "https://github.com/google/neuroglancer/commit/abc",
                "tag": "v2.41.2-40-gabc1234",
                "timestamp": "Wed Jul 1 00:00:00 UTC 2026",
            }
        )
        assert partial == probed

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
