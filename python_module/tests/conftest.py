import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: runs a simulation for more than a few steps; skipped on GitHub CI"
    )
    config.addinivalue_line(
        "markers", "needs_extras: needs packages CI does not install; skipped on GitHub CI"
    )


def pytest_collection_modifyitems(config, items):
    # GPU compute on the CI runners (software Vulkan) is too slow for long runs
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    skip_slow = pytest.mark.skip(reason="slow GPU simulation, skipped on GitHub CI")
    skip_extras = pytest.mark.skip(reason="needs packages CI does not install")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
        if "needs_extras" in item.keywords:
            item.add_marker(skip_extras)
