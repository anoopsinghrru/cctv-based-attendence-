"""Attendance decision logic, tested exactly with a scripted fake face engine."""
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.attendance import SessionProcessor, export_session, run_video
from src.face_engine import Face
from tests.conftest import SLOT, FakeEngine, fake_face, unit

START = datetime(2026, 11, 12, 10, 0)
FRAME = np.full((240, 400, 3), 120, np.uint8)


def with_sims(sims, blur=200.0, x=100, seed=0):
    """A face whose cosine similarity to given people is (approximately) as requested."""
    emb = sum(s * unit(p) for p, s in sims.items())
    rest = unit(50_000 + seed)
    for p in sims:
        rest = rest - (rest @ unit(p)) * unit(p)
    rest /= np.linalg.norm(rest)
    emb = emb + np.sqrt(max(0.0, 1 - float(emb @ emb))) * rest
    f = fake_face(0, x=x, blur=blur)
    f.embedding = (emb / np.linalg.norm(emb)).astype(np.float32)
    return f


@pytest.fixture
def setup(cfg, db):
    for i in (1, 2, 3):
        db.upsert_student(f"S{i}", f"Student {i}", "B1")
        db.add_embedding(f"S{i}", unit(i))
    db.upsert_student("X9", "Other batch", "B2")
    db.add_embedding("X9", unit(9))
    return cfg, db


def run(setup, script, n_frames, step_sec):
    cfg, db = setup
    proc = SessionProcessor(FakeEngine(script), db, cfg, SLOT, START, "test")
    for k in range(n_frames):
        proc.process_frame(FRAME, START + timedelta(seconds=k * step_sec))
    return proc, proc.finalize().set_index("roll_no")


def test_present_absent_and_snapshot(setup):
    proc, res = run(setup, lambda k: [fake_face(1)], 20, 3)
    assert res.loc["S1", "status"] == "Present" and res.loc["S1", "sightings"] == 20
    assert res.loc["S1", "buckets"] == 10 and res.loc["S1", "best_score"] > 0.99
    assert Path(res.loc["S1", "snapshot"]).exists()
    assert res.loc["S2", "status"] == "Absent" and res.loc["S2", "sightings"] == 0
    assert len(res) == 3                                # the other batch's student is not listed


def test_fewer_than_three_sightings_is_absent(setup):
    _, res = run(setup, lambda k: [fake_face(1)] if k in (0, 10) else [], 20, 3)
    assert res.loc["S1", "status"] == "Absent" and res.loc["S1", "sightings"] == 2


def test_many_sightings_in_one_short_burst_is_absent(setup):
    # one frame per minute for an hour; seen in minutes 0-3 only => 1 of 10 time buckets
    _, res = run(setup, lambda k: [fake_face(1)] if k < 4 else [], 60, 60)
    assert res.loc["S1", "sightings"] == 4 and res.loc["S1", "buckets"] == 1
    assert res.loc["S1", "status"] == "Absent"


@pytest.mark.parametrize("arrives_min, expected", [(0, "Present"), (9, "Present"), (11, "Late"),
                                                   (30, "Late"), (50, "Absent")])
def test_late_rule_at_default_ten_minutes(setup, arrives_min, expected):
    _, res = run(setup, lambda k: [fake_face(1)] if k >= arrives_min else [], 60, 60)
    assert res.loc["S1", "status"] == expected


def test_late_threshold_is_configurable(setup):
    setup[0]["attendance"]["late_after_min"] = 1
    _, res = run(setup, lambda k: [fake_face(1)] if k >= 5 else [], 60, 60)
    assert res.loc["S1", "status"] == "Late"


def test_short_clip_is_judged_over_its_own_length(setup):
    # a 60 s clip inside a 60 min slot: buckets span the clip, not the whole hour
    _, res = run(setup, lambda k: [fake_face(1)], 20, 3)
    assert res.loc["S1", "buckets"] == 10


def test_threshold_is_lower_for_blurry_crops(setup):
    cfg = setup[0]
    assert (cfg["recognition"]["blurry_threshold"], cfg["recognition"]["match_threshold"]) == (0.35, 0.40)
    _, sharp = run(setup, lambda k: [with_sims({1: 0.37}, blur=200)], 10, 3)
    assert sharp.loc["S1", "sightings"] == 0             # 0.37 < 0.40
    _, blurry = run(setup, lambda k: [with_sims({1: 0.37}, blur=20)], 10, 3)
    assert blurry.loc["S1", "sightings"] == 10           # 0.37 >= 0.35 for a blurry crop
    _, low = run(setup, lambda k: [with_sims({1: 0.30}, blur=20)], 10, 3)
    assert low.loc["S1", "sightings"] == 0


def test_ambiguous_face_fails_the_margin(setup):
    _, res = run(setup, lambda k: [with_sims({1: 0.60, 2: 0.58})], 10, 3)
    assert res["sightings"].sum() == 0                   # best - second < 0.06
    _, res = run(setup, lambda k: [with_sims({1: 0.60, 2: 0.40})], 10, 3)
    assert res.loc["S1", "sightings"] == 10 and res.loc["S2", "sightings"] == 0


def test_a_student_is_given_to_at_most_one_face_per_frame(setup):
    script = lambda k: [with_sims({1: 0.70}, x=50), with_sims({1: 0.90}, x=250, seed=1)]
    proc, res = run(setup, script, 10, 3)
    assert res.loc["S1", "sightings"] == 10              # not 20
    assert abs(res.loc["S1", "best_score"] - 0.90) < 0.02   # the stronger face won


def test_two_students_in_one_frame_both_counted(setup):
    _, res = run(setup, lambda k: [fake_face(1, x=50), fake_face(2, x=250)], 10, 3)
    assert res.loc["S1", "status"] == res.loc["S2", "status"] == "Present"


def test_student_of_another_batch_is_unknown(setup):
    proc, res = run(setup, lambda k: [fake_face(9)], 10, 3)
    assert res["sightings"].sum() == 0
    assert len(proc.unknown) == 1 and proc.unknown[0]["sightings"] == 10


def test_unknowns_are_merged_per_person_and_one_offs_dropped(setup):
    def script(k):
        faces = [fake_face(70, noise=0.5, seed=k), fake_face(71, x=250, noise=0.5, seed=100 + k)]
        if k == 4:
            faces.append(fake_face(72, x=320))           # a single stray detection
        return faces
    proc, _ = run(setup, script, 12, 3)
    assert sorted(u["sightings"] for u in proc.unknown) == [12, 12]
    files = list((proc.snap_dir / "unknown").glob("*.jpg"))
    assert len(files) == 2                               # dropped / merged snapshots are deleted


def test_weak_detections_are_not_saved_as_unknown(setup):
    proc, _ = run(setup, lambda k: [fake_face(70, score=0.55)], 10, 3)
    assert proc.unknown == []


def test_empty_batch_and_empty_frames(cfg, db):
    proc = SessionProcessor(FakeEngine(lambda k: [fake_face(1)] if k else []), db, cfg, SLOT, START, "t")
    for k in range(6):
        proc.process_frame(FRAME, START + timedelta(seconds=k))
    res = proc.finalize()
    assert len(res) == 0 and proc.frames == 6            # nobody enrolled: runs, marks no one


def test_results_are_stored_and_exported(setup):
    cfg, db = setup
    proc, _ = run(setup, lambda k: [fake_face(1), fake_face(70, x=250)], 10, 3)
    xlsx, csv = export_session(db, cfg, proc.session_id)
    sheets = pd.read_excel(xlsx, sheet_name=None)
    assert list(sheets) == ["Attendance", "Summary", "Unknown faces"]
    assert list(sheets["Attendance"]["status"]) == ["Present", "Absent", "Absent"]
    summary = dict(zip(sheets["Summary"]["field"], sheets["Summary"]["value"]))
    assert summary["Present"] == 1 and summary["Absent"] == 2 and float(summary["attendance %"]) == 33.3
    assert len(sheets["Unknown faces"]) == 1
    assert len(pd.read_csv(csv)) == 3
    assert db.sessions().iloc[0]["present"] == 1


def make_video(path, frames=10, fps=5):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (400, 240))
    for _ in range(frames):
        w.write(FRAME)
    w.release()


def test_run_video_virtual_clock_and_sampling(setup, tmp_path):
    cfg, db = setup
    make_video(tmp_path / "v.mp4", frames=50, fps=5)     # 10 s of video
    proc = SessionProcessor(FakeEngine(lambda k: [fake_face(1)]), db, cfg, SLOT, START, "v")
    seen = []
    res = run_video(proc, tmp_path / "v.mp4", START, every=2, on_frame=lambda img, p, t: seen.append(t))
    assert proc.frames == 5                              # one frame every 2 s
    assert seen[0] == START and seen[1] - seen[0] == timedelta(seconds=2)
    assert res.iloc[0]["status"] == "Present"


def test_run_video_timelapse_and_stop_at_slot_end(setup, tmp_path):
    cfg, db = setup
    make_video(tmp_path / "v.mp4", frames=10)
    proc = SessionProcessor(FakeEngine(lambda k: [fake_face(1)] if k >= 6 else []), db, cfg, SLOT, START, "v")
    res = run_video(proc, tmp_path / "v.mp4", START, frame_seconds=120).set_index("roll_no")
    assert proc.frames == 10 and proc.last_t == START + timedelta(minutes=18)
    assert res.loc["S1", "first_seen"] == "10:12:00" and res.loc["S1", "status"] == "Late"

    proc = SessionProcessor(FakeEngine(lambda k: []), db, cfg, SLOT, START, "v")
    run_video(proc, tmp_path / "v.mp4", START, frame_seconds=600)
    assert proc.frames == 6                              # 10:00 .. 10:50, stops when the slot ends


def test_run_video_missing_file_raises(setup, tmp_path):
    cfg, db = setup
    proc = SessionProcessor(FakeEngine(lambda k: []), db, cfg, SLOT, START, "v")
    with pytest.raises(IOError):
        run_video(proc, tmp_path / "missing.mp4", START)
