import os

import pytest


@pytest.fixture
def data_path():
    return os.path.join(os.path.split(os.path.split(__file__)[0])[0], "data")
