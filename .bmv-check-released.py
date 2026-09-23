"""bump-my-version setup hook: refuse to bump past a version that was never released.

A bump commits, tags and pushes a new version, but publishing is a separate step
(creating a GitHub Release). Bumping twice without publishing in between burns a
version number that nobody can install. This check makes that mistake loud.

Set BUMP_ALLOW_UNRELEASED=1 to bump anyway.
"""

import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PACKAGE = "nglui"


def current_version() -> str:
    text = Path("pyproject.toml").read_text()
    match = re.search(r'^current_version = "([^"]+)"', text, re.MULTILINE)
    if match is None:
        sys.exit("Could not find [tool.bumpversion] current_version in pyproject.toml")
    return match.group(1)


def is_on_pypi(version: str) -> bool:
    url = f"https://pypi.org/pypi/{PACKAGE}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=10):
            return True
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        raise


def main() -> None:
    if os.environ.get("BUMP_ALLOW_UNRELEASED") == "1":
        return
    version = current_version()
    try:
        released = is_on_pypi(version)
    except (urllib.error.URLError, TimeoutError) as err:
        sys.exit(
            f"Could not reach PyPI to check whether {version} was released ({err}).\n"
            "Set BUMP_ALLOW_UNRELEASED=1 to bump without checking."
        )
    if not released:
        sys.exit(
            f"{PACKAGE} {version} is not on PyPI yet, so bumping would skip it.\n"
            f"Publish it by creating a GitHub Release for v{version}, or reuse it.\n"
            "Set BUMP_ALLOW_UNRELEASED=1 to bump anyway."
        )


if __name__ == "__main__":
    main()
