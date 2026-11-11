"""The real ONNX models: detection, alignment, embedding, and the full pipeline on real images."""
from datetime import datetime

import cv2
import numpy as np
import pytest

from src import timetable as tt
from src.attendance import SessionProcessor, enroll_folder, enroll_photo, export_session, run_video
from src.face_engine import ARCFACE_DST, nms, similarity_transform
from tests.conftest import TEST_SET, needs_models, needs_samples


def test_similarity_transform_recovers_a_known_transform():
    ang, s, t = np.deg2rad(25), 1.7, np.array([12.0, -30.0])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    src = ARCFACE_DST.astype(np.float64)
    dst = (s * src @ rot.T) + t
    m = similarity_transform(src, dst)
    assert np.allclose(m[:, :2], s * rot, atol=1e-4) and np.allclose(m[:, 2], t, atol=1e-3)
    assert np.allclose(similarity_transform(src, src), [[1, 0, 0], [0, 1, 0]], atol=1e-5)


def test_nms_keeps_best_of_overlapping_and_all_separate_boxes():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [300, 300, 400, 400], [20, 20, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6], np.float32)
    keep = set(nms(boxes, scores, 0.4).tolist())
    assert keep == {0, 2}             # 1 overlaps 0; 3 lies inside 0 (tile / full-frame duplicate)


pytestmark_models = [needs_models, needs_samples]


def img(name):
    return cv2.imread(str(TEST_SET / name))


@needs_models
@needs_samples
def test_detects_the_right_number_of_faces(engine):
    assert len(engine.analyze(img("obama.jpg"))) == 1
    two = engine.analyze(img("two_people.jpg"))
    assert len(two) == 2
    for f in two:
        assert f.score > 0.5 and f.kps.shape == (5, 2) and f.crop.shape == (112, 112, 3)
        x1, y1, x2, y2 = f.bbox
        assert x2 > x1 and y2 > y1
        assert x1 <= f.kps[:, 0].mean() <= x2 and y1 <= f.kps[:, 1].mean() <= y2   # landmarks inside box
        assert f.kps[0, 0] < f.kps[1, 0]          # left eye is left of right eye
        assert abs(np.linalg.norm(f.embedding) - 1) < 1e-4


@needs_models
def test_no_faces_in_a_blank_image(engine):
    assert engine.analyze(np.full((480, 640, 3), 128, np.uint8)) == []
    assert engine.largest_face(np.zeros((200, 200, 3), np.uint8)) is None


@needs_models
@needs_samples
def test_same_person_scores_high_and_different_people_low(engine):
    a, a2, b = (engine.largest_face(img(n)).embedding for n in ("obama.jpg", "obama2.jpg", "biden.jpg"))
    assert float(a @ a2) > 0.5
    assert float(a @ b) < 0.25 and float(a2 @ b) < 0.25


@needs_models
@needs_samples
def test_two_people_photo_matches_the_two_portraits(engine):
    gallery = {n: engine.largest_face(img(n + ".jpg")).embedding for n in ("obama", "biden")}
    found = set()
    for f in engine.analyze(img("two_people.jpg")):
        sims = {n: float(e @ f.embedding) for n, e in gallery.items()}
        best = max(sims, key=sims.get)
        assert sims[best] > 0.40
        found.add(best)
    assert found == {"obama", "biden"}


@needs_models
@needs_samples
def test_small_faces_survive_downscaling_thanks_to_lanczos_upscale(engine):
    """A face shrunk to CCTV size must still match its full-size portrait."""
    full = img("obama.jpg")
    ref = engine.largest_face(full).embedding
    h, w = full.shape[:2]
    for target in (40, 24):
        face = engine.largest_face(full)
        scale = target / face.size
        small = cv2.resize(full, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        canvas = np.full((480, 640, 3), 110, np.uint8)
        sh, sw = small.shape[:2]
        canvas[100:100 + sh, 200:200 + sw] = small
        got = engine.largest_face(canvas)
        assert got is not None and got.size < target * 1.4
        assert float(got.embedding @ ref) > 0.40, f"face {target}px no longer matches"


@needs_models
@needs_samples
def test_tiled_detection_finds_small_faces_the_full_frame_pass_misses(engine):
    """Paste shrunken portraits into a 1920x1080 frame: tiling must find at least as many."""
    small = cv2.resize(img("two_people.jpg"), None, fx=0.11, fy=0.11, interpolation=cv2.INTER_AREA)
    frame = np.full((1080, 1920, 3), 120, np.uint8)
    sh, sw = small.shape[:2]
    spots = [(60, 80), (60, 900), (60, 1650), (500, 400), (500, 1300), (900, 150), (900, 1000), (900, 1700)]
    for y, x in spots:
        frame[y:y + sh, x:x + sw] = small
    full = engine.detect(frame, tiled=False)[0]
    tiled = engine.detect(frame, tiled=True)[0]
    assert len(tiled) >= len(full)
    assert len(tiled) >= 12                      # 16 faces pasted, most must be found
    # no duplicates from overlapping tiles: every pair of boxes is clearly separate
    for i in range(len(tiled)):
        for j in range(i + 1, len(tiled)):
            a, b = tiled[i], tiled[j]
            iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            assert iw * ih < 0.4 * min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


@needs_models
@needs_samples
def test_embed_from_uses_the_other_image_for_the_crop(engine):
    frame = img("obama.jpg")
    dark = (frame * 0.3).astype(np.uint8)
    f = engine.analyze(frame, embed_from=dark)[0]
    assert f.crop.mean() < engine.analyze(frame)[0].crop.mean() * 0.5


@needs_models
@needs_samples
def test_enrollment_from_photos(engine, cfg, db):
    import shutil
    from pathlib import Path
    db.upsert_student("T1", "Obama", "B1")
    folder = Path(cfg["paths"]["faces"]) / "T1"
    folder.mkdir(parents=True)
    shutil.copy(TEST_SET / "obama.jpg", folder / "a.jpg")
    shutil.copy(TEST_SET / "obama2.jpg", folder / "b.jpg")
    cv2.imwrite(str(folder / "blank.jpg"), np.full((200, 200, 3), 128, np.uint8))
    assert enroll_folder(engine, db, cfg["paths"]["faces"], "T1") == (2, 3)     # blank photo skipped
    assert enroll_folder(engine, db, cfg["paths"]["faces"], "T1") == (2, 3)     # re-enrol replaces
    assert db.students().iloc[0]["photos"] == 2
    assert enroll_photo(engine, db, "T1", np.zeros((100, 100, 3), np.uint8)) is None


@needs_models
@needs_samples
def test_end_to_end_recorded_session(engine, cfg, db):
    """Enrol from portraits, run a real clip through the timetable, check the result and files."""
    clip = TEST_SET / "short_hamilton_clip.mp4"
    people = {"T3": ("Lin-Manuel Miranda", "lin-manuel-miranda.png"), "T2": ("Joe Biden", "biden.jpg")}
    for roll, (name, photo) in people.items():
        db.upsert_student(roll, name, "DEMO")
        assert enroll_photo(engine, db, roll, img(photo), photo) is not None
    open(cfg["paths"]["timetable"], "w").write(
        "day,start,end,room,batch,subject,faculty\nThu,11:00,12:00,R102,DEMO,Lab,Dr. S\n")
    when = datetime(2026, 11, 12, 11, 0)
    slot = tt.find_slot(tt.load_timetable(cfg["paths"]["timetable"]), "R102", when)
    assert slot["batch"] == "DEMO"
    proc = SessionProcessor(engine, db, cfg, slot, when, str(clip))
    res = run_video(proc, clip, when, every=0.5).set_index("roll_no")
    assert proc.frames >= 6
    assert res.loc["T3", "status"] == "Present" and res.loc["T3", "best_score"] > 0.5
    assert res.loc["T2", "status"] == "Absent" and res.loc["T2", "sightings"] == 0   # not in the clip
    xlsx, csv = export_session(db, cfg, proc.session_id)
    assert all(__import__("os").path.getsize(p) > 0 for p in (xlsx, csv))
    assert (proc.snap_dir / "T3.jpg").exists()
