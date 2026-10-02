"""Unit tests for the marketrisk package."""

import marketrisk


def test_package_imports():
    """The package is installed in the environment and importable."""
    assert marketrisk.__version__ == "0.1.0"
