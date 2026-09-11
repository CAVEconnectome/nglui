"""Capability-based feature detection for Neuroglancer deployments.

Neuroglancer is not semantically versioned and is deployed by many groups from many
commits, so nglui cannot choose an output format from a version number. Instead it
describes what a deployment *can do* and picks an encoding that the target understands.

The capability that currently matters is how annotation tags are encoded:

- Main Neuroglancer has no "tag" concept. A tag is an annotation property of
  ``type: "bool"``, and hotkeys bind to generic property tools such as
  ``toggleBoolProperty``.
- seung-lab forks (Spelunker) instead use ``uint8`` properties carrying a non-standard
  ``tag`` key, bound to ``tagTool_tagN`` tool types.

These encodings fail asymmetrically, which is why the conservative default is the older
one. A ``type: "bool"`` property sent to a viewer that predates bool properties makes
``parseAnnotationPropertyType`` throw, and the **entire layer** fails to load. The
reverse is merely lossy: a modern viewer parses a legacy tag property fine, ignoring
the unknown ``tag`` key, so the state still opens -- just without tag labels or
bindings. When nglui does not know what the target supports, it must pick the encoding
that degrades rather than the one that breaks.
"""

from __future__ import annotations

import ast
import json
import os
import warnings
from datetime import datetime, timezone
from typing import Optional, Union
from urllib.parse import urljoin, urlparse

import attrs
import requests
from cachetools import TTLCache, cached

__all__ = [
    "ANNOTATION_BOOL_PROPERTIES",
    "ANNOTATION_PROPERTY_TOOLS",
    "SEUNG_LAB_TAG_TOOLS",
    "Capabilities",
    "LEGACY_CAPABILITIES",
    "MODERN_CAPABILITIES",
    "get_default_capabilities",
    "set_default_capabilities",
    "parse_capabilities",
    "get_version_info",
    "capabilities_from_version_info",
    "capabilities_for_url",
    "prefetch",
    "clear_capability_cache",
]

# Named behaviors a deployment may support.
ANNOTATION_BOOL_PROPERTIES = "annotation_bool_properties"
"""Understands annotation properties of ``type: "bool"``."""

ANNOTATION_PROPERTY_TOOLS = "annotation_property_tools"
"""Understands object-form ``toolBindings`` such as ``toggleBoolProperty``."""

SEUNG_LAB_TAG_TOOLS = "seung_lab_tag_tools"
"""Implements ``tagTool_tagN`` *and* renders a property's ``tag`` key as its label.

Deliberately not named "legacy tag properties": every viewer happily parses a uint8
property with an unknown ``tag`` key. What is actually separable is whether the
deployment gives that key any meaning, which only seung-lab forks do.
"""


@attrs.define(frozen=True)
class Capabilities:
    """What a Neuroglancer deployment can do.

    Attributes
    ----------
    annotation_bool_properties : bool
        Whether ``type: "bool"`` annotation properties are understood.
    annotation_property_tools : bool
        Whether object-form property tool bindings are understood.
    seung_lab_tag_tools : bool
        Whether ``tagTool_tagN`` bindings and the ``tag`` property key are rendered.
    source : str
        Human-readable note on where this determination came from, for diagnostics.
    """

    annotation_bool_properties: bool = False
    annotation_property_tools: bool = False
    seung_lab_tag_tools: bool = False
    source: str = "unspecified"

    def supports(self, capability: str) -> bool:
        """Return whether a named capability is available.

        Parameters
        ----------
        capability : str
            One of the module-level capability constants.

        Returns
        -------
        bool
            True if the deployment supports the capability.

        Examples
        --------
        >>> MODERN_CAPABILITIES.supports(ANNOTATION_BOOL_PROPERTIES)
        True
        """
        if not hasattr(self, capability):
            raise ValueError(
                f"Unknown capability {capability!r}. Known capabilities: "
                f"{ANNOTATION_BOOL_PROPERTIES}, {ANNOTATION_PROPERTY_TOOLS}, "
                f"{SEUNG_LAB_TAG_TOOLS}."
            )
        return bool(getattr(self, capability))

    @property
    def uses_bool_tags(self) -> bool:
        """Whether tags should be encoded as boolean annotation properties."""
        return self.annotation_bool_properties


LEGACY_CAPABILITIES = Capabilities(
    seung_lab_tag_tools=True,
    source="legacy",
)
"""seung-lab / Spelunker style: uint8 tag properties and ``tagTool_*`` bindings."""

MODERN_CAPABILITIES = Capabilities(
    annotation_bool_properties=True,
    annotation_property_tools=True,
    source="modern",
)
"""Main Neuroglancer style: boolean annotation properties and property tools."""

_CAPABILITY_ALIASES = {
    "legacy": LEGACY_CAPABILITIES,
    "spelunker": LEGACY_CAPABILITIES,
    "seung-lab": LEGACY_CAPABILITIES,
    "modern": MODERN_CAPABILITIES,
    "bool_properties": MODERN_CAPABILITIES,
    "google": MODERN_CAPABILITIES,
}

# Conservative on purpose -- see the module docstring on asymmetric failure.
_DEFAULT_CAPABILITIES = LEGACY_CAPABILITIES


def parse_capabilities(
    capabilities: Union[str, Capabilities, None],
) -> Optional[Capabilities]:
    """Coerce a user-supplied capability specification to a `Capabilities`.

    Parameters
    ----------
    capabilities : str or Capabilities or None
        A `Capabilities` instance, a string alias (``"legacy"``, ``"modern"``,
        ``"spelunker"``, ``"google"``, ``"bool_properties"``, ``"seung-lab"``), or
        None to defer the decision to the caller.

    Returns
    -------
    Capabilities or None
        The resolved capabilities, or None if ``capabilities`` was None.

    Raises
    ------
    ValueError
        If a string alias is not recognized.

    Examples
    --------
    >>> parse_capabilities("modern").annotation_bool_properties
    True
    """
    if capabilities is None:
        return None
    if isinstance(capabilities, Capabilities):
        return capabilities
    if isinstance(capabilities, str):
        try:
            return _CAPABILITY_ALIASES[capabilities.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown capability alias {capabilities!r}. "
                f"Options: {sorted(_CAPABILITY_ALIASES)}."
            ) from None
    raise TypeError(
        f"capabilities must be a Capabilities, a string alias, or None, "
        f"not {type(capabilities).__name__}."
    )


def get_default_capabilities() -> Capabilities:
    """Return the capabilities assumed when the target deployment is unknown.

    Returns
    -------
    Capabilities
        The module-level default, conservative unless changed.
    """
    return _DEFAULT_CAPABILITIES


def set_default_capabilities(
    capabilities: Union[str, Capabilities],
) -> Capabilities:
    """Set the capabilities assumed when the target deployment is unknown.

    Use this when you consistently target one deployment and would rather not have
    nglui probe for it.

    Parameters
    ----------
    capabilities : str or Capabilities
        Capabilities or a string alias, as accepted by `parse_capabilities`.

    Returns
    -------
    Capabilities
        The newly set default.

    Examples
    --------
    >>> _ = set_default_capabilities("modern")
    >>> get_default_capabilities().annotation_bool_properties
    True
    >>> _ = set_default_capabilities("legacy")
    """
    global _DEFAULT_CAPABILITIES
    resolved = parse_capabilities(capabilities)
    if resolved is None:
        raise ValueError("capabilities must not be None")
    _DEFAULT_CAPABILITIES = resolved
    return _DEFAULT_CAPABILITIES


# --- Probing a live deployment -------------------------------------------------

VERSION_INFO_PATH = "version.json"
PROBE_TIMEOUT = (1.0, 2.0)
"""(connect, read) timeout. Short on purpose: a state build must not hang offline."""

MAX_VERSION_INFO_BYTES = 64 * 1024
DISABLE_PROBE_ENV_VAR = "NGLUI_DISABLE_CAPABILITY_PROBE"

# Dates the relevant features landed upstream, used to interpret a build timestamp.
# Bool annotation properties: google/neuroglancer PR #809, 2026-05-11.
# Annotation property tools:  google/neuroglancer PR #1063, 2026-08-13.
_BOOL_PROPERTIES_LANDED = datetime(2026, 5, 11, tzinfo=timezone.utc)
_PROPERTY_TOOLS_LANDED = datetime(2026, 8, 13, tzinfo=timezone.utc)

# Repositories whose history the landing dates above actually describe. seung-lab
# forks are deliberately absent: their branches carry the tag tooling instead, and a
# recent build timestamp there says nothing about bool property support.
_LINEAR_HISTORY_REPOS = ("google/neuroglancer", "alleninstitute/neuroglancer")
_TAG_TOOL_REPOS = ("seung-lab/neuroglancer",)

# Cached including failures: an offline user should pay the timeout once per process,
# not once per layer per call.
_version_info_cache = TTLCache(maxsize=64, ttl=3600)
_warned_urls = set()


def _origin(url: str) -> str:
    """Normalize a site URL to a scheme://netloc origin."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url.rstrip("/") + "/"
    return f"{parsed.scheme}://{parsed.netloc}/"


def _parse_version_payload(text: str) -> Optional[dict]:
    """Parse a version.json body, tolerating non-JSON Python-repr payloads.

    neuroglancer-demo.appspot.com serves single-quoted output, which is a valid Python
    literal but not valid JSON, so ``requests.Response.json`` raises on it.
    """
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return None
    return parsed if isinstance(parsed, dict) else None


@cached(cache=_version_info_cache)
def get_version_info(url: str) -> Optional[dict]:
    """Fetch a deployment's ``version.json``, or None if it cannot be read.

    Results -- including failures -- are cached per origin so that an unreachable or
    non-Neuroglancer URL costs one timeout per process rather than one per call.

    Parameters
    ----------
    url : str
        Any URL on the deployment; only its origin is used.

    Returns
    -------
    dict or None
        The parsed version info, typically with ``tag``, ``url``, ``timestamp`` and
        sometimes ``branch`` keys. None if unreachable or unparseable.

    Examples
    --------
    >>> info = get_version_info("https://neuroglancer-demo.appspot.com")  # doctest: +SKIP
    >>> info["url"]  # doctest: +SKIP
    'https://github.com/google/neuroglancer/commit/...'
    """
    if os.environ.get(DISABLE_PROBE_ENV_VAR):
        return None
    try:
        response = requests.get(
            urljoin(_origin(url), VERSION_INFO_PATH), timeout=PROBE_TIMEOUT
        )
        response.raise_for_status()
        if len(response.content) > MAX_VERSION_INFO_BYTES:
            return None
        return _parse_version_payload(response.text)
    except Exception:
        # Any failure is a non-answer, never an error: capability detection is a
        # convenience and must not break state building.
        return None


def _parse_build_timestamp(timestamp: str) -> Optional[datetime]:
    """Parse the git-style timestamp in version.json, e.g. 'Sat Sep 5 09:21:38 UTC 2026'."""
    for fmt in ("%a %b %d %H:%M:%S %Z %Y", "%a %b %d %H:%M:%S %Y"):
        try:
            parsed = datetime.strptime(" ".join(timestamp.split()), fmt)
        except (ValueError, TypeError):
            continue
        return parsed.replace(tzinfo=timezone.utc)
    return None


def capabilities_from_version_info(info: Optional[dict]) -> Optional[Capabilities]:
    """Infer capabilities from a parsed ``version.json`` payload.

    Keyed primarily on the repository, only secondarily on the build timestamp. A
    timestamp records when a build was cut, not which features it contains -- a fresh
    rebuild of an old branch carries a recent stamp -- so it is only trusted for
    repositories whose history the known landing dates actually describe.

    Parameters
    ----------
    info : dict or None
        A parsed version.json payload, as returned by `get_version_info`.

    Returns
    -------
    Capabilities or None
        Inferred capabilities, or None if the payload is unrecognizable.
    """
    if not info:
        return None
    commit_url = str(info.get("url", "")).lower()
    repo = "/".join(urlparse(commit_url).path.strip("/").split("/")[:2])

    if any(repo == known for known in _TAG_TOOL_REPOS):
        return attrs.evolve(LEGACY_CAPABILITIES, source=f"probe:{repo}")

    if any(repo == known for known in _LINEAR_HISTORY_REPOS):
        built = _parse_build_timestamp(str(info.get("timestamp", "")))
        if built is None:
            return None
        return Capabilities(
            annotation_bool_properties=built >= _BOOL_PROPERTIES_LANDED,
            annotation_property_tools=built >= _PROPERTY_TOOLS_LANDED,
            seung_lab_tag_tools=False,
            source=f"probe:{repo}@{built.date()}",
        )
    return None


def capabilities_for_url(
    url: Optional[str],
    warn_on_fallback: bool = True,
) -> Capabilities:
    """Determine what a deployment supports, falling back to the default.

    Parameters
    ----------
    url : str or None
        URL of the target Neuroglancer deployment.
    warn_on_fallback : bool, optional
        Whether to warn (once per URL) when the deployment could not be identified
        and the default is used instead. Default is True.

    Returns
    -------
    Capabilities
        The deployment's capabilities, or the module default if undeterminable.

    Examples
    --------
    >>> caps = capabilities_for_url("https://spelunker.cave-explorer.org")  # doctest: +SKIP
    >>> caps.seung_lab_tag_tools  # doctest: +SKIP
    True
    """
    if url:
        resolved = capabilities_from_version_info(get_version_info(url))
        if resolved is not None:
            return resolved
    if warn_on_fallback and url and url not in _warned_urls:
        _warned_urls.add(url)
        default = get_default_capabilities()
        warnings.warn(
            f"Could not determine Neuroglancer capabilities for {url}; assuming "
            f"'{default.source}' annotation tags. Pass capabilities='modern' or "
            f"capabilities='legacy' to choose explicitly, or call "
            f"set_default_capabilities() to change the fallback.",
            stacklevel=2,
        )
    return get_default_capabilities()


def prefetch(url: str) -> Optional[Capabilities]:
    """Warm the capability cache for a deployment ahead of time.

    Useful in notebooks so that the first ``to_url`` call does not pay a network
    round trip, and so an offline session fails fast and visibly.

    Parameters
    ----------
    url : str
        URL of the target Neuroglancer deployment.

    Returns
    -------
    Capabilities or None
        The detected capabilities, or None if the deployment was unreachable.
    """
    return capabilities_from_version_info(get_version_info(url))


def clear_capability_cache() -> None:
    """Forget all probed capabilities and fallback warnings."""
    _version_info_cache.clear()
    _warned_urls.clear()
