"""Face engine: detect (SCRFD) -> align (5-point similarity) -> embed (ArcFace).

Runs the InsightFace buffalo_l ONNX models directly with onnxruntime, so the
`insightface` package (and its C++ build tools) is not needed.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from . import preprocess as pp

# ArcFace reference landmarks for a 112x112 crop: eyes, nose tip, mouth corners
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
STRIDES = (8, 16, 32)
NUM_ANCHORS = 2


@dataclass
class Face:
    bbox: np.ndarray        # x1, y1, x2, y2 in frame pixels
    kps: np.ndarray         # (5, 2) landmarks
    score: float            # detector confidence
    embedding: np.ndarray   # (512,) L2-normalised
    crop: np.ndarray        # aligned 112x112 BGR
    blur: float             # Laplacian variance of the aligned crop
    size: float             # shorter side of the bbox in pixels


def similarity_transform(src, dst):
    """Least-squares similarity (rotation + uniform scale + shift), Umeyama's method."""
    src, dst = src.astype(np.float64), dst.astype(np.float64)
    sm, dm = src.mean(0), dst.mean(0)
    s, d = src - sm, dst - dm
    u, sing, vt = np.linalg.svd(d.T @ s / len(src))
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rot = u @ sign @ vt
    scale = (sing * np.diag(sign)).sum() / ((s ** 2).sum() / len(src))
    shift = dm - scale * rot @ sm
    return np.hstack([scale * rot, shift[:, None]]).astype(np.float32)


def nms(boxes, scores, thr):
    order = scores.argsort()[::-1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        # also drop a box that is mostly inside a stronger one (tile vs full-frame duplicates)
        inside = inter / (np.minimum(area[i], area[rest]) + 1e-9)
        order = rest[(iou <= thr) & (inside <= 0.7)]
    return np.array(keep, dtype=int)


def _add_nvidia_dll_dirs():
    """cuDNN loads its sub-libraries lazily by name, so the folders of the pip-installed
    nvidia-* packages must be on the DLL search path (Windows does not look there itself)."""
    import os
    try:
        import nvidia
    except ImportError:
        return
    for base in nvidia.__path__:
        for bin_dir in Path(base).glob("*/bin"):
            if str(bin_dir) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(bin_dir))


class FaceEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dcfg = cfg["detection"]
        self.rcfg = cfg["recognition"]
        mdir = Path(cfg["models"]["dir"])
        det_path, rec_path = mdir / cfg["models"]["detector"], mdir / cfg["models"]["recognizer"]
        for p in (det_path, rec_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing model {p}. Download buffalo_l.zip from the InsightFace v0.7 "
                    f"release and put det_10g.onnx and w600k_r50.onnx in {mdir}.")
        providers = ["CPUExecutionProvider"]
        if cfg["models"].get("use_gpu") and "CUDAExecutionProvider" in ort.get_available_providers():
            _add_nvidia_dll_dirs()
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()      # load CUDA / cuDNN from the pip-installed nvidia-* packages
            # settings sized for a 4 GB card: no exhaustive cuDNN search, capped memory arena
            providers.insert(0, ("CUDAExecutionProvider", {
                "cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": int(cfg["models"].get("gpu_mem_gb", 3) * 1024 ** 3)}))
        opts = ort.SessionOptions()
        opts.log_severity_level = 3     # hide the harmless batch-shape warnings of w600k_r50
        self.det = ort.InferenceSession(str(det_path), opts, providers=providers)
        self.rec = ort.InferenceSession(str(rec_path), opts, providers=providers)
        self.device = self.det.get_providers()[0].replace("ExecutionProvider", "")
        self.det_in = self.det.get_inputs()[0].name
        self.rec_in = self.rec.get_inputs()[0].name
        self.size = int(self.dcfg["input_size"])
        self._anchors = {}
        for s in STRIDES:
            n = self.size // s
            ys, xs = np.mgrid[:n, :n]
            centers = np.stack([xs, ys], -1).reshape(-1, 2).astype(np.float32) * s
            self._anchors[s] = np.repeat(centers, NUM_ANCHORS, axis=0)
        if self.device == "CUDA":       # first GPU call initialises cuDNN (several seconds): do it now
            self._detect_once(np.zeros((self.size, self.size, 3), np.uint8))
            self.embed_crops([np.zeros((112, 112, 3), np.uint8)])

    # ------------------------------------------------------------ detection
    def _detect_once(self, img):
        """One SCRFD pass over `img` (letterboxed to input_size). Coordinates in img pixels."""
        h, w = img.shape[:2]
        scale = self.size / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=interp)
        blob = (canvas[:, :, ::-1].astype(np.float32) - 127.5) / 128.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        outs = self.det.run(None, {self.det_in: blob})

        # group the 9 outputs by their last dimension, then order by stride (most rows = stride 8)
        groups = {1: [], 4: [], 10: []}
        for o in outs:
            o = o.reshape(-1, o.shape[-1])
            groups[o.shape[1]].append(o)
        for g in groups.values():
            g.sort(key=len, reverse=True)

        boxes, scores, kpss = [], [], []
        for i, s in enumerate(STRIDES):
            sc = groups[1][i].ravel()
            keep = sc >= self.dcfg["score_threshold"]
            if not keep.any():
                continue
            c = self._anchors[s][keep]
            d = groups[4][i][keep] * s
            k = groups[10][i][keep] * s
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1],
                                   c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], 1))
            kpss.append(k.reshape(-1, 5, 2) + c[:, None, :])
            scores.append(sc[keep])
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 5, 2), np.float32)
        return np.concatenate(boxes) / scale, np.concatenate(scores), np.concatenate(kpss) / scale

    def detect(self, img, tiled=None):
        """Full-frame pass plus (optionally) overlapping tiles, merged with NMS."""
        tiled = self.dcfg["tiled"] if tiled is None else tiled
        h, w = img.shape[:2]
        b, s, k = self._detect_once(img)
        boxes, scores, kpss = [b], [s], [k]
        # tiling only pays off when the frame is bigger than the detector input
        if tiled and max(h, w) > self.size:
            for x0, y0, x1, y1 in pp.make_tiles(w, h, self.dcfg["tile_grid"], self.dcfg["tile_overlap"]):
                b, s, k = self._detect_once(img[y0:y1, x0:x1])
                if not len(b):
                    continue
                b = b + np.array([x0, y0, x0, y0], np.float32)
                k = k + np.array([x0, y0], np.float32)
                # a face cut by an interior tile border is seen whole in the neighbouring tile
                e = 3
                ok = ((b[:, 0] > x0 + e) | (x0 == 0)) & ((b[:, 1] > y0 + e) | (y0 == 0)) & \
                     ((b[:, 2] < x1 - e) | (x1 == w)) & ((b[:, 3] < y1 - e) | (y1 == h))
                boxes.append(b[ok]); scores.append(s[ok]); kpss.append(k[ok])
        boxes, scores, kpss = np.concatenate(boxes), np.concatenate(scores), np.concatenate(kpss)
        if not len(boxes):
            return boxes, scores, kpss
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
        side = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        big = side >= self.dcfg["min_face_px"]
        boxes, scores, kpss = boxes[big], scores[big], kpss[big]
        if not len(boxes):
            return boxes, scores, kpss
        keep = nms(boxes, scores, self.dcfg["nms_threshold"])
        return boxes[keep], scores[keep], kpss[keep]

    # ------------------------------------------------------------ alignment
    def align(self, img, bbox, kps):
        """Warp the face to the 112x112 ArcFace template.

        Small faces are first cut out and Lanczos-upscaled, so the final warp
        interpolates from a cleaner, larger source instead of a handful of pixels.
        """
        side = float(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        target = self.rcfg["upscale_below_px"]
        src, pts = img, kps
        if 0 < side < target:
            scale = min(4.0, target / side)
            pad = 0.6 * max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            h, w = img.shape[:2]
            x0, y0 = int(max(0, bbox[0] - pad)), int(max(0, bbox[1] - pad))
            x1, y1 = int(min(w, bbox[2] + pad)), int(min(h, bbox[3] + pad))
            if x1 - x0 > 2 and y1 - y0 > 2:
                src = pp.lanczos_upscale(img[y0:y1, x0:x1], scale)
                pts = (kps - np.array([x0, y0], np.float32)) * scale
        m = similarity_transform(pts, ARCFACE_DST)
        return cv2.warpAffine(src, m, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0)

    # ------------------------------------------------------------ embedding
    def embed_crops(self, crops):
        """ArcFace embeddings for aligned 112x112 BGR crops -> (N, 512), L2-normalised."""
        if not len(crops):
            return np.zeros((0, 512), np.float32)
        blob = (np.stack(crops)[:, :, :, ::-1].astype(np.float32) - 127.5) / 127.5
        blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2))
        out = []
        for i in range(0, len(blob), 16):
            chunk = blob[i:i + 16]
            try:
                out.append(self.rec.run(None, {self.rec_in: chunk})[0])
            except Exception:   # model exported with a fixed batch of 1
                out.extend(self.rec.run(None, {self.rec_in: c[None]})[0] for c in chunk)
        emb = np.concatenate(out).astype(np.float32)
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)

    # ------------------------------------------------------------ everything
    def analyze(self, img, tiled=None, reject_blurry=True, embed_from=None):
        """Detect, align and embed every face in a BGR image.

        `embed_from`: same-sized image the face crops are cut from. The attendance run
        detects on the enhanced frame (more faces are found) but embeds from the original
        pixels, because ArcFace scores drop slightly on contrast-stretched / sharpened crops.
        """
        boxes, scores, kpss = self.detect(img, tiled)
        src = img if embed_from is None else embed_from
        crops, meta = [], []
        for b, s, k in zip(boxes, scores, kpss):
            crop = self.align(src, b, k)
            blur = pp.blur_score(crop)
            if reject_blurry and blur < self.rcfg["reject_below"]:
                continue
            crops.append(crop)
            meta.append((b, k, float(s), blur))
        embs = self.embed_crops(crops)
        return [Face(b, k, s, e, c, blur, float(min(b[2] - b[0], b[3] - b[1])))
                for (b, k, s, blur), e, c in zip(meta, embs, crops)]

    def largest_face(self, img):
        """The biggest face in an enrollment photo (None if no face)."""
        faces = self.analyze(img, tiled=False, reject_blurry=False)
        if not faces:
            # A photo cropped tightly to the face leaves the detector no context: retry with a border.
            h, w = img.shape[:2]
            pad = max(h, w) // 2
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(124, 116, 104))
            faces = self.analyze(padded, tiled=False, reject_blurry=False)
        return max(faces, key=lambda f: f.size) if faces else None


def draw_faces(img, faces, labels=None, colors=None):
    """Draw boxes (+ optional labels) on a copy of the image."""
    out = img.copy()
    t = max(1, round(max(out.shape[:2]) / 1000))
    for i, f in enumerate(faces):
        color = colors[i] if colors else (0, 255, 0)
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
        if labels and labels[i]:
            # label size follows the face size, so back-row labels do not bury their neighbours
            fs = float(np.clip((x2 - x1) / 110, 0.38, 0.8))
            lt = 2 if fs > 0.6 else 1
            (tw, th), _ = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, fs, lt)
            cv2.rectangle(out, (x1, y1 - th - 5), (x1 + tw + 3, y1), color, -1)
            cv2.putText(out, labels[i], (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lt,
                        cv2.LINE_AA)
    return out
