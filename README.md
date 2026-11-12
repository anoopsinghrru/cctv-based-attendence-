# CCTV Face-Recognition Attendance (DIP project)

The timetable says which batch should be in a room at a given time. The class video or camera
stream is processed, the faces seen are matched against that batch's enrolled photos, and
attendance is marked Present / Late / Absent automatically.

Results, figures and measurements are in [docs/REPORT.md](docs/REPORT.md).

## Setup

```
python -m pip install -r requirements.txt
```

Put `det_10g.onnx` and `w600k_r50.onnx` (from InsightFace `buffalo_l.zip`, release v0.7) in
`models/`. They are run directly with onnxruntime; the `insightface` package is not needed.

## Run it

```
python -m streamlit run app.py                 # dashboard
python -m pytest                               # 74 automated tests (about a minute)
python scripts/rebuild_demo_sessions.py        # re-run every demo session from scratch
python scripts/evaluate.py                     # verification accuracy on the LFW benchmark
python scripts/evaluate_open_set.py            # threshold choice for a class-sized gallery
python scripts/evaluate_dataset.py             # detection / identification against the dataset's labels
```

## Real use

1. Edit `data/timetable.csv` (day,start,end,room,batch,subject,faculty) and `data/cameras.csv`
   (room,source,description; source is a webcam index or an RTSP url).
2. Add students to `data/students.csv` (roll_no,name,batch) and 3-5 photos each in
   `data/faces/<roll_no>/`, then `python scripts/enroll.py` - or use the Students tab.
3. Recorded class:
   `python scripts/run_session.py --room R101 --start "2026-11-12 10:00" --video class.mp4`
4. Live class (runs until the slot ends): `python scripts/run_session.py --room R101 --live`
5. Reports are written to `outputs/reports/`, snapshots to `outputs/snapshots/session_<id>/`.

Useful `run_session.py` options: `--every N` (seconds between sampled frames), `--save-video`
(annotated output), `--timelapse N` (the file is a time-lapse, N real seconds per frame),
`--late-after M` (minutes), `--duration S` (stop a live run early), `--batch X` (run ad hoc
when no class is scheduled), `--no-enhance`.

## Demo data

| Batch | Room | Source | What it shows |
|---|---|---|---|
| BSCS-A, BSCS-B | R201, R202 | Kaggle "Classroom Student Engagement Dataset" (Ghulam Mustafa, CC BY-NC 4.0) | Enrol from one class recording, mark attendance on other recordings of the same class |
| BSCS-B | R202 | The same dataset, three recordings joined as a 40-minute time-lapse | Late at the real 10-minute threshold (late arrival simulated) |
| EXAM | R103 | Altima IP-camera exam-hall clip (YouTube, 720p) | Real overhead CCTV, small faces looking down |
| CSE-A | R101 | Seminar audience, cropped from the 4K YouTube video "Virtual Reality (VR) Public Speaking Practice Stage: Audience of 100" | 40 people, dim room, tiled detection; Late is simulated |
| DEMO | R102 | ageitgey/face_recognition example portraits and clip | Enrolment from separate portraits |

Rebuild from the raw downloads:

```
python scripts/get_sample_video.py --test-set
python scripts/build_clips_from_frames.py --frames data/samples/kaggle/engagement/dataset/images --out data/samples/kaggle_clips
python scripts/auto_enroll_from_video.py --video data/samples/kaggle_clips/owais-class-1.mp4 --batch BSCS-A --prefix A --every 1 --min-face 30 --min-samples 6 --photos 10
python scripts/auto_enroll_from_video.py --video data/samples/kaggle_clips/taimoor-class-2.mp4 --batch BSCS-B --prefix B --every 1 --min-face 30 --min-samples 6 --photos 10
python scripts/apply_names.py
python scripts/simulate_late_arrival.py --video data/samples/kaggle_clips/composite_long_class.mp4 --roll B002 --until 20
python scripts/rebuild_demo_sessions.py --yes
```

`auto_enroll_from_video.py` clusters the faces of a video into students (DBSCAN on embeddings).
It exists so the pipeline can be shown on footage of people who were never photographed; real
students are enrolled from photos. The frame-dataset clips are 1 frame per second; files at
2 fps or less are recognised as sparse and every frame is analysed automatically.

Staged demo effects, to be presented as such: both "Late" students are simulated with
`simulate_late_arrival.py`, which hides a face for the first part of the clip (CSE-A: first 75 s,
run with `--late-after 1`; BSCS-B: first 13 minutes of a 40-minute time-lapse, at the real
10-minute threshold). The BSCS-B long class is three separate recordings played as one.

Student names in the demo are invented (`data/demo_names.csv`, applied with
`scripts/apply_names.py`), because the people in the public footage are not identified. To use a
real class list, pass your own CSV of `roll_no,name` to that script or edit names on the Students page.

## DIP Lab

The dashboard's **DIP Lab** page demonstrates each image-processing fundamental on a chosen frame
of the footage: intensity transformations, histogram processing, noise and spatial filtering,
sharpening and the blur measure, the frequency domain, colour models, geometric transforms and
interpolation, multi-scale tiling, and the complete pipeline. Each topic shows the formula,
adjustable parameters, before/after images, charts and measurements, and states whether it is a
pipeline stage or shown for analysis. `scripts/dip_demo.py` writes the same comparisons as
figures for the report.

## Layout

| File | Role |
|---|---|
| `src/preprocess.py` | DIP stage: histogram -> auto gamma, noise estimate, bilateral / NLM, CLAHE, unsharp, Lanczos, blur score, tiles |
| `src/face_engine.py` | SCRFD decoding (strides 8/16/32, 2 anchors), NMS, tiled detection, 5-point alignment, ArcFace embedding |
| `src/timetable.py` | (room, datetime) -> slot, including slots that cross midnight |
| `src/database.py` | SQLite: students, embeddings, sessions, attendance, unknown faces |
| `src/attendance.py` | matching, per-frame processing, Present / Late / Absent rules, recorded and live runners, export |
| `app.py` | Dashboard shell ("AttendVision"): sidebar navigation, theme, page routing |
| `ui/` | One module per page — Overview, Take attendance, Students, Timetable, Reports, DIP Lab, System — plus `theme.py` (styling, charts) and `common.py` (shared data helpers, session result panel) |
| `.streamlit/config.toml`, `assets/` | Colour theme and logo |
| `scripts/` | enroll, run_session, auto_enroll_from_video, dip_demo, evaluate, get_sample_video, build_clips_from_frames, simulate_late_arrival, rebuild_demo_sessions |
| `tests/` | unit tests, real-model tests, and headless dashboard tests |
| `config.yaml` | every threshold and setting |

## How a frame is processed

1. Luminance histogram -> mean brightness; if dark, gamma is chosen to bring the mean to the target.
2. Noise level is estimated; bilateral (or NLM) denoising strength follows it.
3. CLAHE on L of LAB, then unsharp masking (both eased off on noisy frames).
4. SCRFD runs on the full frame and on overlapping tiles; results are merged with NMS.
5. Each face is aligned to 112x112 (small faces are Lanczos-upscaled first); crops with a very
   low Laplacian variance are rejected.
6. Detection uses the enhanced frame; the ArcFace embedding is taken from the original pixels,
   because measured match scores are lower on contrast-stretched crops.
7. A face matches a student when similarity >= 0.35 (0.32 for blurry crops) and beats the
   second-best student by the margin. One student is assigned to at most one face per frame.
   These thresholds come from the open-set measurement in the report (section 4.1).
8. Faces that match nobody are grouped per person and saved as unknown faces.

## Attendance rules (`config.yaml -> attendance`)

- Present: seen >= 3 times and in >= 30% of the time buckets.
- Late: as above, but first seen more than 10 minutes after the start.
- Absent: otherwise.

The buckets split the observed class time (slot start to slot end, or to the end of the
recording if it is shorter) into 10 equal parts, so a short clip is judged over its own length.

## Speed and GPU

Measured on a 1080p frame with tiling (7 detector passes + 9 faces embedded): GTX 1050 about
0.36 s, CPU about 2.0 s. `detection.tiled: false` is roughly 5x faster again.

The engine uses CUDA automatically when `onnxruntime-gpu` is installed and falls back to CPU
otherwise (`models.use_gpu: false` forces CPU). For a GTX 10xx card the versions in
`requirements.txt` matter: `onnxruntime-gpu` 1.24.4 (CUDA 12) with `nvidia-cudnn-cu12` 9.10.2.21.
Newer releases target CUDA 13 or ship a cuDNN without Pascal kernels and fail at the first
convolution. Do not install `onnxruntime` and `onnxruntime-gpu` side by side.

## Privacy

Face photos, embeddings, snapshots and the database are personal data. They stay on this
machine and are excluded from version control (`.gitignore`). Enrol people only with their
consent, and cite the footage sources above when presenting the demo.
