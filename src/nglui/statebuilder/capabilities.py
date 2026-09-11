"""Capability-based feature detection for Neuroglancer deployments.

Neuroglancer is not semantically versioned and is deployed by many groups from many
commits, so nglui cannot choose an output format from a version number. Instead it
describes what a deployment *can do* and picks an encoding that the target understands.

The capability that currently matters is how annotation tags are encoded:

- Main Neuroglancer has no "tag" concept. A tag is an annotation property of
  ``type: "bool"``, and hotkeys bind to generic property tools such as
  ``toggleBoolProperty``.
- Spelunker instead uses ``uint8`` properties carrying a non-standard
  ``tag`` key, bound to ``tagTool_tagN`` tool types.

The default when a deployment cannot be identified is ``main``, because deployments
overwhelmingly descend from it and the ones that do not -- Spelunker in particular --
are identified by probing rather than left to the default.

Be aware of what that costs when it is wrong, because the two encodings fail very
differently. A ``type: "bool"`` property sent to a viewer predating bool properties
makes ``parseAnnotationPropertyType`` throw, and Neuroglancer drops the whole
annotation layer from the state and rewrites its URL without it -- so the annotations
are gone, not merely unrendered. The reverse is only lossy: a current viewer parses a
Spelunker tag property fine, ignoring the unknown ``tag`` key, so the state still opens
without tag labels or bindings.

Detection covers the deployments that need the older encoding, so the default is
reached only for a deployment nglui cannot identify at all. That case warns, naming the
argument to set. Pin `capabilities` or call `set_default_capabilities` when targeting a
deployment that does not serve a usable ``version.json``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import warnings
from datetime import datetime, timezone
from typing import Iterable, Optional, Union
from urllib.parse import urljoin, urlparse

import attrs
import requests
from cachetools import TTLCache, cached

__all__ = [
    "ANNOTATION_BOOL_PROPERTIES",
    "ANNOTATION_PROPERTY_TOOLS",
    "SPELUNKER_TAG_TOOLS",
    "Capabilities",
    "LEGACY_CAPABILITIES",
    "MAIN_CAPABILITIES",
    "get_default_capabilities",
    "set_default_capabilities",
    "parse_capabilities",
    "get_version_info",
    "probe_capabilities",
    "capabilities_for_url",
    "prefetch",
    "clear_capability_cache",
]

# Named behaviors a deployment may support.
ANNOTATION_BOOL_PROPERTIES = "annotation_bool_properties"
"""Understands annotation properties of ``type: "bool"``."""

ANNOTATION_PROPERTY_TOOLS = "annotation_property_tools"
"""Understands object-form ``toolBindings`` such as ``toggleBoolProperty``."""

SPELUNKER_TAG_TOOLS = "spelunker_tag_tools"
"""Implements ``tagTool_tagN`` *and* renders a property's ``tag`` key as its label.

Deliberately not named "legacy tag properties": every viewer happily parses a uint8
property with an unknown ``tag`` key. What is actually separable is whether the
deployment gives that key any meaning, which is a Spelunker behavior.
"""


def _capability_field(default: bool = False):
    """A field that is part of the capability set, as opposed to metadata about it."""
    return attrs.field(default=default, metadata={"capability": True})


@attrs.define(frozen=True)
class Capabilities:
    """What a Neuroglancer deployment can do.

    A set of named behaviors, not a version and not a single label. Members are typed
    fields so they are discoverable and checkable, and the same set can also be handled
    by name -- ``"annotation_bool_properties" in capabilities`` -- for code that works
    over capabilities generically.

    Attributes
    ----------
    annotation_bool_properties : bool
        Whether ``type: "bool"`` annotation properties are understood.
    annotation_property_tools : bool
        Whether object-form property tool bindings are understood.
    spelunker_tag_tools : bool
        Whether ``tagTool_tagN`` bindings and the ``tag`` property key are rendered.
    source : str
        Where this determination came from, for diagnostics. Deliberately excluded
        from equality: two deployments with the same abilities are equivalent whether
        one was probed and the other named.

    Examples
    --------
    >>> MAIN_CAPABILITIES.enabled == frozenset(
    ...     {"annotation_bool_properties", "annotation_property_tools"}
    ... )
    True
    >>> Capabilities.from_names(["annotation_bool_properties"]).annotation_bool_properties
    True
    """

    annotation_bool_properties: bool = _capability_field()
    annotation_property_tools: bool = _capability_field()
    spelunker_tag_tools: bool = _capability_field()
    source: str = attrs.field(default="unspecified", eq=False)

    @classmethod
    def names(cls) -> tuple:
        """Every capability name this version of nglui knows about."""
        return tuple(
            field.name
            for field in attrs.fields(cls)
            if field.metadata.get("capability")
        )

    @classmethod
    def from_names(cls, names: Iterable, source: str = "explicit") -> "Capabilities":
        """Build a capability set from the names that should be enabled.

        Parameters
        ----------
        names : iterable of str
            Capability names to enable. Anything omitted is disabled.
        source : str, optional
            Provenance note, by default "explicit".

        Returns
        -------
        Capabilities
            The described capability set.

        Raises
        ------
        ValueError
            If any name is not a known capability.
        """
        requested = set(names)
        known = set(cls.names())
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(
                f"Unknown capabilities {unknown}. Known capabilities: {sorted(known)}."
            )
        return cls(source=source, **{name: name in requested for name in known})

    @property
    def enabled(self) -> frozenset:
        """The names of the capabilities that are available."""
        return frozenset(name for name in self.names() if getattr(self, name))

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

        Raises
        ------
        ValueError
            If the name is not a known capability. Metadata fields such as ``source``
            are not capabilities and are rejected here.

        Examples
        --------
        >>> MAIN_CAPABILITIES.supports(ANNOTATION_BOOL_PROPERTIES)
        True
        """
        if capability not in self.names():
            raise ValueError(
                f"Unknown capability {capability!r}. Known capabilities: "
                f"{sorted(self.names())}."
            )
        return bool(getattr(self, capability))

    def __contains__(self, capability: str) -> bool:
        return self.supports(capability)


LEGACY_CAPABILITIES = Capabilities(
    spelunker_tag_tools=True,
    source="legacy",
)
"""Spelunker style: uint8 tag properties and ``tagTool_*`` bindings."""

MAIN_CAPABILITIES = Capabilities(
    annotation_bool_properties=True,
    annotation_property_tools=True,
    source="main",
)
"""Everything on google/neuroglancer's main branch: bool properties and property tools.

Tracks main rather than naming a fixed set, so it widens as capabilities are added.
Build an explicit `Capabilities` if you need a set that will not move.
"""

# Shorthand for the two sets that matter in practice, each naming the deployment it
# tracks rather than a fixed list of abilities: "main" follows google/neuroglancer's
# main branch and widens as capabilities land there, "spelunker" follows the
# seung-lab deployment. These are conveniences, not the encoding -- `Capabilities` is
# the encoding, and `Capabilities.from_names` builds any set an alias does not cover.
_CAPABILITY_ALIASES = {
    "legacy": LEGACY_CAPABILITIES,
    "spelunker": LEGACY_CAPABILITIES,
    "main": MAIN_CAPABILITIES,
    "google": MAIN_CAPABILITIES,
}

# Deployments overwhelmingly descend from main, and the ones that do not are
# identified by probing rather than left to this. See the module docstring for what
# being wrong costs in each direction.
_DEFAULT_CAPABILITIES = MAIN_CAPABILITIES


def parse_capabilities(
    capabilities: Union[str, Capabilities, None],
) -> Optional[Capabilities]:
    """Coerce a user-supplied capability specification to a `Capabilities`.

    Parameters
    ----------
    capabilities : str or Capabilities or None
        A `Capabilities` instance, a string alias (``"main"``, ``"google"``,
        ``"legacy"``, ``"spelunker"``), or
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
    >>> parse_capabilities("main").annotation_bool_properties
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
    >>> _ = set_default_capabilities("main")
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

# Cached including failures: an offline user should pay the timeout once per process,
# not once per layer per call.
_version_info_cache = TTLCache(maxsize=64, ttl=3600)
_capability_cache = TTLCache(maxsize=64, ttl=3600)
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


BUNDLE_TIMEOUT = (2.0, 10.0)
"""(connect, read) for the client bundle, which is larger than version.json."""

MAX_BUNDLE_BYTES = 32 * 1024 * 1024

_MAIN_BUNDLE_RE = re.compile(r"""src=["'](main[.\w-]*\.js)["']""")

# Markers read out of the deployed client. Each is a string the viewer must keep
# verbatim -- tool type names travel in state JSON, and `bool` is a key in the
# annotation property type table -- so minification does not rename them.
_BOOL_PROPERTY_MARKER = re.compile(r"(?:int16|uint8|int8|float32):[^{}]{0,80}?bool:")
_PROPERTY_TOOLS_MARKER = "toggleBoolProperty"
_TAG_TOOLS_MARKER = "tagTool"


def _fetch_client_bundle(url: str) -> Optional[str]:
    """Fetch a deployment's main client bundle, or None if it cannot be read."""
    origin = _origin(url)
    index = requests.get(origin, timeout=PROBE_TIMEOUT)
    index.raise_for_status()
    match = _MAIN_BUNDLE_RE.search(index.text)
    if match is None:
        return None
    bundle = requests.get(urljoin(origin, match.group(1)), timeout=BUNDLE_TIMEOUT)
    bundle.raise_for_status()
    if len(bundle.content) > MAX_BUNDLE_BYTES:
        return None
    return bundle.text


@cached(cache=_capability_cache)
def probe_capabilities(url: str) -> Optional[Capabilities]:
    """Determine what a deployment supports by reading its client bundle.

    This asks the deployed viewer directly rather than inferring from a version,
    because for a fork there is nothing reliable to infer from. Neuroglancer stamps a
    ``git describe`` string in ``version.json``, but a fork describes against the last
    tag *in its own repository*: Spelunker reports ``v2.37`` while containing commits
    from well past ``v2.41.2``, so the release says nothing about which upstream
    features were merged, and backports defeat the ordering entirely. Its build at
    ``v2.37-347`` had no bool property support and the one at ``v2.37-398`` has the
    full system, with the same release either way.

    Results, including failures, are cached per origin -- an unreachable or
    non-Neuroglancer URL costs one timeout per process rather than one per call.

    Parameters
    ----------
    url : str
        Any URL on the deployment; only its origin is used.

    Returns
    -------
    Capabilities or None
        What the deployment supports, or None if its bundle could not be read.
    """
    if os.environ.get(DISABLE_PROBE_ENV_VAR):
        return None
    try:
        bundle = _fetch_client_bundle(url)
    except Exception:
        # Any failure is a non-answer, never an error: capability detection is a
        # convenience and must not break state building.
        return None
    if not bundle:
        return None
    return Capabilities(
        annotation_bool_properties=bool(_BOOL_PROPERTY_MARKER.search(bundle)),
        annotation_property_tools=_PROPERTY_TOOLS_MARKER in bundle,
        spelunker_tag_tools=_TAG_TOOLS_MARKER in bundle,
        source=f"probe:{_origin(url)}",
    )


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
    >>> caps.spelunker_tag_tools  # doctest: +SKIP
    True
    """
    if url:
        resolved = probe_capabilities(url)
        if resolved is not None:
            return resolved
    if warn_on_fallback and url and url not in _warned_urls:
        _warned_urls.add(url)
        default = get_default_capabilities()
        warnings.warn(
            f"Could not determine Neuroglancer capabilities for {url}; assuming "
            f"'{default.source}' annotation tags. If that deployment is older than "
            "bool annotation properties, it will drop the annotation layer rather "
            "than render it. Pass capabilities='legacy' for a Spelunker-style "
            "deployment, capabilities='main' to silence this, or call "
            "set_default_capabilities() to change the fallback.",
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
    return probe_capabilities(url)


def clear_capability_cache() -> None:
    """Forget all probed capabilities and fallback warnings."""
    _version_info_cache.clear()
    _capability_cache.clear()
    _warned_urls.clear()
