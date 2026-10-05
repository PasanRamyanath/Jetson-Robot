"""Tests run on any Python >= 3.10 dev machine (no Jetson/GPU): shared + agent logic + brain with --stub models."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("shared", "kaggle", os.path.join("jetson", "agent")):
    sys.path.insert(0, os.path.join(ROOT, p))

import pytest  # noqa: E402

from beni_common.memory import MemoryStore  # noqa: E402


@pytest.fixture
def store_pair(tmp_path):
    """(jetson_store, kaggle_store) on disk, as in production."""
    a = MemoryStore(str(tmp_path / "jetson.db"), node="jetson")
    b = MemoryStore(str(tmp_path / "kaggle.db"), node="kaggle")
    yield a, b
    a.close()
    b.close()
