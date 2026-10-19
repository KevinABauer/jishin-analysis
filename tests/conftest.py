import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

CASE = ROOT / "cases" / "2026-10-16"
SPOOF = ROOT / "cases" / "spoof-demo"


@pytest.fixture
def case():
    return CASE


@pytest.fixture
def spoof():
    return SPOOF
