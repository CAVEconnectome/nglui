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
            ("modern", True),
            ("google", True),
            ("bool_properties", True),
            ("legacy", False),
            ("spelunker", False),
            ("seung-lab", False),
        ],
    )
    def test_string_aliases(self, alias, bool_props):
        assert caps.parse_capabilities(alias).annotation_bool_properties is bool_props

    def test_alias_is_case_insensitive(self):
        assert caps.parse_capabilities("Modern") == caps.MODERN_CAPABILITIES

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

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            caps.parse_capabilities(42)


class TestCapabilitiesObject:
    def test_supports_named_capability(self):
        assert caps.MODERN_CAPABILITIES.supports(caps.ANNOTATION_BOOL_PROPERTIES)
        assert not caps.LEGACY_CAPABILITIES.supports(caps.ANNOTATION_BOOL_PROPERTIES)
        assert caps.LEGACY_CAPABILITIES.supports(caps.SEUNG_LAB_TAG_TOOLS)

    def test_unknown_capability_raises(self):
        with pytest.raises(ValueError, match="Unknown capability"):
            caps.MODERN_CAPABILITIES.supports("teleportation")

    def test_default_is_conservative(self):
        """A bool property breaks an old viewer outright; legacy only degrades."""
        assert caps.get_default_capabilities().annotation_bool_properties is False

    def test_set_default_round_trip(self):
        original = caps.get_default_capabilities()
        try:
            caps.set_default_capabilities("modern")
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
        assert resolved.seung_lab_tag_tools
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
            "timestamp": "Wed Jul 1 00:00:00 UTC 2026",
        }
        resolved = caps.capabilities_from_version_info(info)
        assert resolved.annotation_bool_properties
        assert not resolved.annotation_property_tools

    def test_seung_lab_timestamp_is_ignored(self):
        """A fresh seung-lab build is still a seung-lab build."""
        info = {
            "url": "https://github.com/seung-lab/neuroglancer/commit/abc",
            "timestamp": "Sat Sep 5 09:21:38 UTC 2026",
        }
        resolved = caps.capabilities_from_version_info(info)
        assert not resolved.annotation_bool_properties
        assert resolved.seung_lab_tag_tools

    def test_unknown_fork_is_undeterminable(self):
        info = {
            "url": "https://github.com/someone/neuroglancer-fork/commit/abc",
            "timestamp": "Sat Sep 5 09:21:38 UTC 2026",
        }
        assert caps.capabilities_from_version_info(info) is None

    def test_unparseable_timestamp_is_undeterminable(self):
        info = {
            "url": "https://github.com/google/neuroglancer/commit/abc",
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
