import pytest


pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path
