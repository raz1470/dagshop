"""Sanity tests that the package is correctly installed and importable."""

from __future__ import annotations

import dagshop


def test_package_is_importable() -> None:
    """The package itself should import without errors."""
    assert dagshop is not None


def test_version_is_set() -> None:
    """__version__ should be a non-empty string."""
    assert isinstance(dagshop.__version__, str)
    assert dagshop.__version__ != ""


def test_version_follows_semver() -> None:
    """Version should follow MAJOR.MINOR.PATCH (semantic versioning)."""
    parts = dagshop.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
