"""Shared fixtures. Run from the project root:  python -m pytest"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.database import Database  # noqa: E402
from src.face_engine import Face  # noqa: E402

TEST_SET = ROOT / "data" / "samples" / "test_set"
needs_models = pytest.mark.skipif(not (ROOT / "models" / "det_10g.onnx").exists(),
                                  reason="face models not downloaded")
needs_samples = pytest.mark.skipif(not (TEST_SET / "obama.jpg").exists(),
                                   reason="run scripts/get_sample_video.py --test-set")


@pytest.fixture
def cfg(tmp_path):
    """The real config.yaml with every data path redirected into a temp folder."""
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {"db": str(tmp_path / "t.db"), "faces": str(tmp_path / "faces"),
                    "timetable": str(tmp_path / "timetable.csv"), "students": str(tmp_path / "students.csv"),
                    "cameras": str(tmp_path / "cameras.csv"), "snapshots": str(tmp_path / "snapshots"),
                    "reports": str(tmp_path / "reports")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


@pytest.fixture
def db(cfg):
    return Database(cfg["paths"]["db"])


@pytest.fixture(scope="session")
def engine():
    from src.face_engine import FaceEngine
    return FaceEngine(load_config(ROOT / "config.yaml"))


def unit(seed):
    """A deterministic random unit vector: stands in for one person's face embedding."""
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def fake_face(person, x=100, blur=200.0, noise=0.0, seed=0, score=0.9):
    """A Face whose embedding is person's identity vector plus optional noise."""
    emb = unit(person)
    if noise:
        emb = emb + noise * unit(10_000 + seed)
        emb = emb / np.linalg.norm(emb)
    box = np.array([x, 50, x + 60, 110], np.float32)
    return Face(box, np.zeros((5, 2), np.float32), score, emb.astype(np.float32),
                np.zeros((112, 112, 3), np.uint8), blur, 60.0)


class FakeEngine:
    """Replays a script: frame k -> list of Faces. Lets the attendance rules be tested exactly."""

    def __init__(self, script):
        self.script = script
        self.calls = 0
        self.device = "fake"

    def analyze(self, img, tiled=None, reject_blurry=True, embed_from=None):
        faces = self.script(self.calls)
        self.calls += 1
        return copy.deepcopy(faces)


SLOT = {"day": "Thu", "start": "10:00", "end": "11:00", "room": "R1", "batch": "B1",
        "subject": "Test", "faculty": "T"}
