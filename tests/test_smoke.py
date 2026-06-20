"""Trivial smoke test — proves the toolchain (pytest + package import) is wired up.

Real logic lands in phase 1 (scheduler + SimWorker), tested first.
"""

from inference_demo import __version__


def test_package_imports() -> None:
    assert __version__ == "0.1.0"
