"""System: what the software is running on and the settings in force."""
from pathlib import Path

import onnxruntime as ort
import streamlit as st

from src.config import ROOT
from ui import common as c
from ui import theme


def _size(path):
    p = Path(path)
    return f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "missing"


def render():
    conf, d = c.cfg(), c.db()
    theme.header("System", "The inference device, model files, storage and the thresholds currently in force. "
                           "Settings are changed in config.yaml.")

    gpu = "CUDAExecutionProvider" in ort.get_available_providers() and conf["models"].get("use_gpu")
    mdir = Path(conf["models"]["dir"])
    models_ok = all((mdir / conf["models"][k]).exists() for k in ("detector", "recognizer"))
    k = st.columns(4)
    k[0].metric("Inference device", "GPU (CUDA)" if gpu else "CPU")
    k[1].metric("Face models", "Ready" if models_ok else "Missing")
    k[2].metric("Database", _size(conf["paths"]["db"]))
    k[3].metric("Face embeddings", int(d.df("SELECT COUNT(*) AS n FROM embeddings").iloc[0]["n"]))

    left, right = st.columns(2)
    with left:
        with theme.card("models"):
            st.subheader("Models")
            theme.kv([("Detector", f"SCRFD-10G · {conf['models']['detector']} · {_size(mdir / conf['models']['detector'])}"),
                      ("Recogniser", f"ArcFace R50 · {conf['models']['recognizer']} · {_size(mdir / conf['models']['recognizer'])}"),
                      ("Embedding", "512 dimensions, cosine similarity"),
                      ("Runtime", f"onnxruntime {ort.__version__}"),
                      ("GPU memory cap", f"{conf['models'].get('gpu_mem_gb', 3)} GB")])
        with theme.card("storage"):
            st.subheader("Storage")
            rel = lambda p: str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else str(p)
            theme.kv([("Database", rel(conf["paths"]["db"])), ("Student photos", rel(conf["paths"]["faces"])),
                      ("Timetable", rel(conf["paths"]["timetable"])), ("Reports", rel(conf["paths"]["reports"])),
                      ("Snapshots", rel(conf["paths"]["snapshots"]))])
            st.markdown("<div class='av-note' style='margin-top:.6rem'>Face photos, embeddings and snapshots are "
                        "personal data. They stay on this computer and are excluded from version control.</div>",
                        unsafe_allow_html=True)
    with right:
        with theme.card("rules"):
            st.subheader("Recognition and attendance rules")
            r, a, det = conf["recognition"], conf["attendance"], conf["detection"]
            note = lambda value, meaning: (f"{value}<br><span style='font-weight:400;color:{theme.MUTED};"
                                           f"font-size:.8rem'>{meaning}</span>")
            theme.kv([
                ("Match threshold", note(r["match_threshold"], "minimum similarity for a sharp face")),
                ("Match threshold, blurry face", note(r["blurry_threshold"], "used when the crop is blurry")),
                ("Margin over second best", note(r["margin"], "rejects faces that resemble two students")),
                ("Frame sampling", note(f"every {a['sample_every_sec']} s", "one frame is analysed per interval")),
                ("Present", note(f"≥ {a['min_sightings']} sightings",
                                 f"in ≥ {a['min_bucket_fraction']:.0%} of {a['time_buckets']} time buckets")),
                ("Late", note(f"after {a['late_after_min']} min", "first seen later than this after the start")),
                ("Tiled detection", note("On" if det["tiled"] else "Off",
                                         f"{det['tile_grid'][0]}×{det['tile_grid'][1]} tiles, {det['tile_overlap']:.0%} overlap")),
                ("Smallest face", note(f"{det['min_face_px']} px", "smaller detections are ignored")),
            ])
        with theme.card("pipeline"):
            st.subheader("Image-processing pipeline")
            p = conf["preprocess"]
            theme.kv([("Status", "Enabled" if p.get("enabled", True) else "Disabled"),
                      ("Gamma correction", f"when mean brightness < {p['dark_mean']}"),
                      ("Denoising", f"{p['denoise']}, strength follows measured noise"),
                      ("Contrast", f"CLAHE on LAB-L, clip {p['clahe_clip']}, {p['clahe_grid']}×{p['clahe_grid']} grid"),
                      ("Sharpening", f"unsharp mask, amount {p['unsharp_amount']}")])
