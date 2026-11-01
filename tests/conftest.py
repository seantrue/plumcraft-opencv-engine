"""Shared by the tests: this engine, one instance, and its demo workflow."""

import json
from pathlib import Path

import pytest

from opencv_engine import OpencvEngine

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def engine() -> OpencvEngine:
    return OpencvEngine()


@pytest.fixture(scope="session")
def demo() -> dict:
    return json.loads((ROOT / "examples" / "demo.json").read_text("utf-8"))
