"""Pytest configuration for barebone tests."""

import pytest


# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path
