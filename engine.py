"""
Unified YOLO (.onnx) + PP-OCR (.onnx) Inference Engine for CAD PDF Testing.
Supports dynamic ONNX model selection, automatic class/dictionary detection,
and multi-pass CAD image preprocessing.
"""
from __future__ import annotations

import ast
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from preprocessor import (
    build_ocr_passes,
    clean_leader_lines,
    extract_crop_with_context,
    pad_and_upscale,
    pil_to_base64_png,
    render_pdf_page,
    split_stacked_tolerance_projection,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
YOLO_ONNX_DIR = MODELS_DIR / "onnx_yolo"
PPOCR_ONNX_DIR = MODELS_DIR / "onnx_ppocr"
PT_LIB_DIR = MODELS_DIR / "pt_library"
DEFAULT_PPOCR_MODEL = "PP-OCRv6_small_rec_041026 v2.onnx"
DEFAULT_1CLASS_MODEL = "yolo_best_1class071026.onnx"
DEFAULT_3REGION_MODEL = "yolo_3regions_31026.onnx"

DEFAULT_CLASS_NAMES = {
    0: "nominal",
    1: "tol_up",
    2: "tol_lower",
}


_DECIMAL_LOOKALIKES = str.maketrans({
    "o": "0", "O": "0", "Q": "0", "q": "0", "D": "0",
    "l": "1", "I": "1", "|": "1", "i": "1",
    "z": "2", "Z": "2",
    "s": "5", "S": "5",
    "b": "8", "B": "8",
})


def _sanitize_decimal_match(m: re.Match) -> str:
    int_part = m.group(1)
    rest = m.group(2)

    # Preserve valid trailing CAD suffixes (only allowed at the end AFTER at least one decimal digit)
    suffix = ""
    m_kw = re.search(r"(MAX|MIN|DP|THRU|TYP|REF|PLC|PCS|FLT|rad|deg)$", rest, flags=re.IGNORECASE)
    if m_kw and len(rest) > len(m_kw.group(0)):
        suffix = m_kw.group(0)
        rest = rest[: -len(suffix)]
    else:
        # Preserve valid ISO fit class at the end (e.g. '115.3k6', '6.0H7', '10.0js6') when preceded by a digit
        m_fit = re.search(r"(?<=\d)(js|JS|[a-hA-Hj-nJ-NpPrRsStTuUxXzZ])[4-9]$", rest)
        if m_fit:
            suffix = m_fit.group(0)
            rest = rest[: -len(suffix)]

    # Group 1: Fix intruder letters immediately after '.' before digits (e.g. '0.d008' -> '0.008', '0.o05' -> '0.005', '1.l25' -> '1.125')
    m_lead = re.match(r"^([a-zA-Z|]+)(\d+.*)$", rest)
    if m_lead:
        letters, tail_digits = m_lead.group(1), m_lead.group(2)
        if all(ch in "lI|" for ch in letters):
            rest = ("1" * len(letters)) + tail_digits
        elif all(ch in "oOqQ" for ch in letters) and len(tail_digits) <= 2:
            rest = ("0" * len(letters)) + tail_digits
        elif len(tail_digits) >= 2:
            # Intruder noise letter like 'd' in '0.d008', 'c' in '0.c005', 'a' in '12.a25' -> drop the letter
            rest = tail_digits
        else:
            rest = "0" + tail_digits

    # Convert any remaining lookalike characters inside the fractional part and strip stray letters
    cleaned_frac = "".join(ch for ch in rest.translate(_DECIMAL_LOOKALIKES) if ch.isdigit())
    if not cleaned_frac:
        return m.group(0)
    return f"{int_part}.{cleaned_frac}{suffix}"


def clean_cad_text(text: str, is_sub_crop: bool = False) -> str:
    if not text:
        return ""
    text = text.strip()

    # Normalize triangle revision symbol variants (including 'A' before decimal dimension when ◬ is read as A)
    text = re.sub(r"^[∆▲]", "△", text)
    text = re.sub(r"^A(?=\d+\.\d)", "△", text)

    # Group 1 & 4: Replace accented o/O variants and prefix/integer lookalikes before decimal point
    text = re.sub(r"[óòôöõōŏÓÒÔÖÕŌŎ]", "0", text)
    text = re.sub(r"\b(DP|[CRØøϕM])[oOqQdD](?=\.?\d)", r"\g<1>0", text)
    text = re.sub(r"(^|(?<=[\s+\-±(]))[oOqQdD](?=\.\d)", r"\g<1>0", text)
    text = re.sub(r"(^|(?<=[\s+\-±(]))[lI|](?=\.\d)", r"\g<1>1", text)

    # Group 3: Clean leading/trailing leader-line & extension-line noise ('=C0.5', '_R0.9', '4-R0.05MAX-')
    text = re.sub(r"^[=_~]+(?=[CRDØøϕM△A0-9(])", "", text)
    text = re.sub(r"^-+(?=[CRDØøϕM△A(])", "", text)
    text = re.sub(r"(?<=[0-9A-Za-z°'\"%)])[-_=~]+$", "", text)

    # Remove spaces immediately around decimal points
    text = re.sub(r"(?<=\d)\s+\.\s*(?=\d)", ".", text)
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)

    # Fix '±q.' or '±q' OCR confusion when horizontal line touches '0' after '±'
    text = re.sub(r"±\s*[qQoO](?=\.?\d)", "±0", text)

    # Group 1: Sanitize fractional digits so intruder letters like '0.d008', '0.o05', '1.l25', '0.00B' are cleaned
    text = re.sub(r"(\d+)\.([0-9a-zA-Z|]+)", _sanitize_decimal_match, text)

    # Group 3: Remove spurious trailing dot on a number that already has a decimal point (e.g. '32.27.' -> '32.27')
    text = re.sub(r"(\d+\.\d+)\.+$", r"\1", text)

    # If there is a symmetric tolerance '±', collapse kerning spaces inside the nominal and tolerance parts
    if "±" in text:
        left, right = text.split("±", 1)
        left = re.sub(r"(?<=\d)\s*\|\s*(?=\s*$)", "1", left)
        left = re.sub(r"(?<=[\d.])\s+(?=[\d.])", "", left.strip())
        # Group 2: Nominal before '±' is never negative (strip false leading '-' from dimension line)
        left = re.sub(r"^-+(?=\d|[CRØøϕM(])", "", left)
        right = re.sub(r"(?<=\d)\s*\|\s*(?=\s*$)", "1", right.strip())
        right = re.sub(r"(?<=[\d.])\s+(?=[\d.])", "", right.strip())
        right = re.sub(r"(\d+)\.([0-9a-zA-Z|]+)", _sanitize_decimal_match, right)
        right = re.sub(r"(\d+\.\d+)\.+$", r"\1", right)
        if right.startswith("."):
            right = "0" + right
        text = f"{left} ±{right}".strip()
    elif not is_sub_crop:
        # Group 2: Full-box nominal dimensions are never negative (e.g. '-35.7' -> '35.7', '-38.7' -> '38.7')
        text = re.sub(r"^-+(?=\d+(?:\.\d+)?(?:°|\b|$))", "", text)

    # Group 4: Normalize DMS quotes and compact spaces around °, ', "
    text = re.sub(r"[\u2032\u2018\u2019\u02b9]", "'", text)
    text = re.sub(r"[\u2033\u201c\u201d\u02ba]", '"', text)
    text = re.sub(r"(\d+)\s*°\s*", r"\1°", text)
    text = re.sub(r"(\d+)\s*'\s*", r"\1'", text)
    text = re.sub(r'(\d+)\s*"\s*', r'\1"', text)

    # Group 5: Auto-balance single parenthesis on reference dimensions/angles like '19.060)' or '(0.581°'
    if not is_sub_crop and " " not in text and "/" not in text:
        if text.startswith("(") and ")" not in text and re.match(r"^\([0-9CRØøϕM.+±°'\"rad-]+$", text):
            text = text + ")"
        elif text.endswith(")") and "(" not in text and re.match(r"^[0-9CRØøϕM.+±°'\"rad-]+\)$", text):
            text = "(" + text

    return text



def parse_dimension_semantics(text: str) -> Dict[str, str]:
    """Parse CAD callout text into nominal, upper_tol, lower_tol."""
    clean = clean_cad_text(text)
    if not clean:
        return {"nominal": "", "upperTol": "", "lowerTol": "", "type": "empty"}

    # Symmetric: 24.5 ± 0.05
    m_sym = re.search(r"^([^\s±]+)\s*±\s*([0-9.]+)$", clean)
    if m_sym:
        return {
            "nominal": m_sym.group(1),
            "upperTol": f"+{m_sym.group(2)}",
            "lowerTol": f"-{m_sym.group(2)}",
            "type": "symmetric",
        }

    # Stacked / slash separated: 12.5 +0.02/-0.01 or 12.5 0/-0.02
    m_stack = re.search(r"^([^\s]+)\s+([+-]?[0-9.]+)\s*/\s*([+-]?[0-9.]+)$", clean)
    if m_stack:
        return {
            "nominal": m_stack.group(1),
            "upperTol": m_stack.group(2),
            "lowerTol": m_stack.group(3),
            "type": "stacked",
        }

    return {"nominal": clean, "upperTol": "", "lowerTol": "", "type": "single"}


class YOLODetectorONNX:
    """Dynamic ONNX detector supporting 1-class, 2-class, 3-class or custom Ultralytics YOLO ONNX models."""

    _cache: Dict[Tuple[str, int, int], Tuple[ort.InferenceSession, Dict[int, str], Tuple[int, int], Any]] = {}
    _lock = threading.Lock()

    @classmethod
    def load(cls, model_path: Path, default_imgsz: int = 1024):
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy YOLO ONNX model: {model_path}")
        stat = model_path.stat()
        key = (str(model_path), stat.st_mtime_ns, stat.st_size)

        with cls._lock:
            if key not in cls._cache:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.log_severity_level = 3
                session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
                meta = session.get_modelmeta().custom_metadata_map or {}
                inp = session.get_inputs()[0]
                out = session.get_outputs()[0]

                ih = int(inp.shape[2]) if len(inp.shape) >= 4 and isinstance(inp.shape[2], int) and inp.shape[2] > 0 else default_imgsz
                iw = int(inp.shape[3]) if len(inp.shape) >= 4 and isinstance(inp.shape[3], int) and inp.shape[3] > 0 else default_imgsz
                dtype = np.float16 if inp.type == "tensor(float16)" else np.float32

                # Parse class names from metadata or infer from output shape
                names: Dict[int, str] = {}
                if "names" in meta:
                    try:
                        parsed = ast.literal_eval(meta["names"])
                        if isinstance(parsed, dict):
                            names = {int(k): str(v) for k, v in parsed.items()}
                        elif isinstance(parsed, list):
                            names = {i: str(v) for i, v in enumerate(parsed)}
                    except Exception:
                        names = {}

                if not names:
                    # Infer channel count from output shape [1, 4+C, N] or [1, N, 4+C]
                    oshape = out.shape
                    c_dim = 1
                    if len(oshape) == 3:
                        s1 = oshape[1] if isinstance(oshape[1], int) else 0
                        s2 = oshape[2] if isinstance(oshape[2], int) else 0
                        if 5 <= s1 <= 100:
                            c_dim = s1 - 4
                        elif 5 <= s2 <= 100:
                            c_dim = s2 - 4
                    names = {i: DEFAULT_CLASS_NAMES.get(i, f"class_{i}") for i in range(max(1, c_dim))}

                cls._cache[key] = (session, names, (ih, iw), dtype)
            return cls._cache[key]

    @classmethod
    def inspect_model(cls, model_path: Path) -> Dict[str, Any]:
        try:
            session, names, (ih, iw), dtype = cls.load(model_path)
            return {
                "name": model_path.name,
                "path": str(model_path),
                "sizeMb": round(model_path.stat().st_size / (1024 * 1024), 2),
                "inputSize": [iw, ih],
                "numClasses": len(names),
                "classNames": names,
                "dtype": "fp16" if dtype == np.float16 else "fp32",
            }
        except Exception as e:
            return {
                "name": model_path.name,
                "path": str(model_path),
                "sizeMb": round(model_path.stat().st_size / (1024 * 1024), 2),
                "error": str(e),
            }

    @staticmethod
    def _restore_box(row: np.ndarray, width: int, height: int, input_w: int, input_h: int) -> List[int]:
        ratio = min(input_w / width, input_h / height)
        left = round((input_w - round(width * ratio)) / 2 - 0.1)
        top = round((input_h - round(height * ratio)) / 2 - 0.1)
        cx, cy, w, h = map(float, row)
        x1 = max(0.0, min(float(width), (cx - w / 2 - left) / ratio))
        y1 = max(0.0, min(float(height), (cy - h / 2 - top) / ratio))
        x2 = max(0.0, min(float(width), (cx + w / 2 - left) / ratio))
        y2 = max(0.0, min(float(height), (cy + h / 2 - top) / ratio))
        x, y = round(x1), round(y1)
        return [x, y, max(0, round(x2) - x), max(0, round(y2) - y)]

    @staticmethod
    def _class_nms(boxes: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        class_ids = sorted({b["classId"] for b in boxes})
        for cid in class_ids:
            cands = sorted((b for b in boxes if b["classId"] == cid), key=lambda b: b["confidence"], reverse=True)
            if not cands:
                continue
            rects = np.array([[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]] for b in cands], dtype=np.float64)
            areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
            indices = np.arange(len(cands))
            while len(indices) > 0:
                first, rest = indices[0], indices[1:]
                kept.append(cands[int(first)])
                overlap = np.maximum(
                    0.0,
                    np.minimum(rects[first, 2:], rects[rest, 2:]) - np.maximum(rects[first, :2], rects[rest, :2]),
                )
                inter = overlap[:, 0] * overlap[:, 1]
                iou = inter / np.maximum(areas[first] + areas[rest] - inter, 1e-9)
                indices = rest[iou <= iou_threshold]
        return kept

    @classmethod
    def detect(
        cls,
        image: Image.Image,
        model_path: Path,
        conf_thres: float = 0.30,
        iou_thres: float = 0.45,
        imgsz_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        session, names, (ih, iw), dtype = cls.load(model_path, default_imgsz=imgsz_override or 1024)
        inp = session.get_inputs()[0]
        if imgsz_override and (not isinstance(inp.shape[2], int) or inp.shape[2] <= 0):
            ih = iw = int(imgsz_override)

        num_classes = len(names)
        w, h = image.size
        ratio = min(iw / w, ih / h)
        rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
        left, top = round((iw - rw) / 2 - 0.1), round((ih - rh) / 2 - 0.1)

        canvas = Image.new("RGB", (iw, ih), (114, 114, 114))
        canvas.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), (left, top))
        tensor = np.ascontiguousarray(np.asarray(canvas).transpose(2, 0, 1)[None], dtype=dtype) / dtype(255.0)
        preds = np.asarray(session.run(None, {inp.name: tensor})[0])

        channels = 4 + num_classes
        if preds.ndim != 3 or preds.shape[0] != 1:
            raise ValueError(f"YOLO ONNX output shape không hợp lệ: {preds.shape}")
        if preds.shape[1] == channels:
            rows = preds[0].T
        elif preds.shape[2] == channels:
            rows = preds[0]
        else:
            # Dynamic fallback if metadata count differed from tensor channels
            if preds.shape[1] > 4 and preds.shape[1] < preds.shape[2]:
                rows = preds[0].T
                num_classes = rows.shape[1] - 4
            else:
                rows = preds[0]
                num_classes = rows.shape[1] - 4
            names = {i: names.get(i, DEFAULT_CLASS_NAMES.get(i, f"class_{i}")) for i in range(num_classes)}

        rows = rows[np.isfinite(rows).all(axis=1)]
        raw_boxes: List[Dict[str, Any]] = []
        for row in rows:
            scores = row[4 : 4 + num_classes]
            cid = int(np.argmax(scores))
            score = float(scores[cid])
            if score < conf_thres or row[2] <= 0 or row[3] <= 0:
                continue
            bx, by, bw, bh = cls._restore_box(row[:4], w, h, iw, ih)
            if bw < 3 or bh < 3:
                continue
            raw_boxes.append(
                {
                    "x": bx,
                    "y": by,
                    "w": bw,
                    "h": bh,
                    "confidence": round(score, 4),
                    "classId": cid,
                    "className": names.get(cid, f"class_{cid}"),
                }
            )

        kept = cls._class_nms(raw_boxes, iou_threshold=iou_thres)
        # Sort in natural reading order (top-to-bottom bands, then left-to-right)
        kept.sort(key=lambda b: (b["y"] // 90, b["x"]))
        for idx, b in enumerate(kept, 1):
            b["id"] = idx

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "boxes": kept,
            "modelName": model_path.name,
            "numClasses": num_classes,
            "classNames": names,
            "inputSize": [iw, ih],
            "elapsedMs": elapsed_ms,
        }


class PPOCRRecognizerONNX:
    """
    Pure Pillow + NumPy + ONNXRuntime CTC Recognizer for PP-OCRv4 / PP-OCRv6 models.
    Automatically detects output vocabulary size and selects matching dictionary if needed.
    """

    _cache: Dict[Tuple[Any, ...], Tuple[ort.InferenceSession, List[str], Path]] = {}
    _lock = threading.Lock()

    @classmethod
    def _resolve_dict_for_session(cls, session: ort.InferenceSession, requested_dict: Optional[Path]) -> Path:
        out_shape = session.get_outputs()[0].shape
        vocab_size = out_shape[-1] if len(out_shape) >= 3 and isinstance(out_shape[-1], int) else None

        candidates: List[Path] = []
        if requested_dict:
            if not requested_dict.is_file():
                raise FileNotFoundError(f"Không tìm thấy từ điển OCR: {requested_dict}")
            candidates.append(requested_dict)
        for p in sorted(PPOCR_ONNX_DIR.glob("*.txt")):
            if p not in candidates:
                candidates.append(p)

        if vocab_size is not None:
            for cand in candidates:
                lines = [l.strip("\r\n") for l in cand.read_text(encoding="utf-8").splitlines() if l.strip("\r\n")]
                # PP-OCR CTC vocab = blank (1) + len(chars) + optional space (1)
                if vocab_size in (len(lines) + 1, len(lines) + 2):
                    return cand
                if cand == requested_dict:
                    raise ValueError(f"Từ điển {cand.name} không khớp vocabulary {vocab_size} của model")
            raise ValueError(f"Không có từ điển khớp vocabulary {vocab_size} của model")

        if candidates:
            return candidates[0]
        raise FileNotFoundError("Không tìm thấy file từ điển (.txt) cho PP-OCR trong models/onnx_ppocr/")

    @classmethod
    def load(cls, model_path: Path, dict_path: Optional[Path] = None) -> Tuple[ort.InferenceSession, List[str], Path]:
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy PP-OCR ONNX model: {model_path}")
        stat = model_path.stat()
        req_dict_str = str(dict_path.resolve()) if dict_path else "auto"
        dictionaries = set(PPOCR_ONNX_DIR.glob("*.txt"))
        if dict_path:
            dictionaries.add(dict_path)
        dict_versions = tuple((str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(dictionaries) if p.is_file())
        key = (str(model_path), stat.st_mtime_ns, stat.st_size, req_dict_str, dict_versions)

        with cls._lock:
            if key not in cls._cache:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.inter_op_num_threads = 2
                opts.log_severity_level = 3
                session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
                resolved_dict = cls._resolve_dict_for_session(session, dict_path)
                chars = [
                    l.strip("\r\n")
                    for l in resolved_dict.read_text(encoding="utf-8").splitlines()
                    if l.strip("\r\n")
                ]
                char_list = ["blank"] + chars
                if session.get_outputs()[0].shape[-1] != len(char_list):
                    char_list.append(" ")
                cls._cache[key] = (session, char_list, resolved_dict)
            return cls._cache[key]

    @classmethod
    def predict_single(
        cls,
        img: Image.Image,
        model_path: Path,
        dict_path: Optional[Path] = None,
        is_sub_crop: bool = False,
    ) -> Tuple[str, float]:
        session, char_list, _ = cls.load(model_path, dict_path)
        rgb = img.convert("RGB")
        w, h = rgb.size
        if w <= 1 or h <= 1:
            return "", 0.0

        img_h = 48
        ratio = w / float(max(1, h))
        resized_w = max(16, int(math.ceil(img_h * ratio)))
        max_w = max(320, int(math.ceil(resized_w / 32.0) * 32))

        resized_pil = rgb.resize((resized_w, img_h), Image.Resampling.BICUBIC)
        # Convert RGB -> BGR to match OpenCV convention used during PP-OCR training
        arr_bgr = np.asarray(resized_pil, dtype=np.float32)[:, :, ::-1]
        norm = arr_bgr.transpose((2, 0, 1)) / 255.0
        norm = (norm - 0.5) / 0.5

        padded = np.zeros((1, 3, img_h, max_w), dtype=np.float32)
        padded[0, :, :, :resized_w] = norm

        inp_name = session.get_inputs()[0].name
        out = session.run(None, {inp_name: padded})[0]
        probs = out[0]
        preds = np.argmax(probs, axis=-1)
        confs = np.max(probs, axis=-1)

        res_chars: List[str] = []
        res_confs: List[float] = []
        prev = -1
        for idx, conf in zip(preds, confs):
            idx_i = int(idx)
            if idx_i != prev:
                if idx_i != 0 and idx_i < len(char_list):
                    res_chars.append(char_list[idx_i])
                    res_confs.append(float(conf))
                prev = idx_i

        text = clean_cad_text("".join(res_chars), is_sub_crop=is_sub_crop)
        mean_conf = float(np.mean(res_confs)) if res_confs else 0.0
        return text, round(mean_conf, 4)

    @staticmethod
    def _detect_leading_sign_from_ink(sub_img: Image.Image) -> Optional[str]:
        """
        Inspect the leftmost ink component of a tolerance sub-crop to determine if there is
        a leading '-' (thin horizontal bar) or '+' (cross with vertical + horizontal bar)
        that PP-OCR might have skipped.
        """
        gray = np.asarray(sub_img.convert("L"), dtype=np.uint8)
        ink = gray < 185
        h, w = ink.shape
        if h < 10 or w < 16:
            return None
        col_has_ink = np.any(ink, axis=0)
        if not np.any(col_has_ink):
            return None
        padded = np.concatenate(([False], col_has_ink, [False]))
        starts = np.where(padded[1:] & ~padded[:-1])[0]
        ends = np.where(~padded[1:] & padded[:-1])[0]
        if len(starts) < 2:
            return None
        # Inspect the first ink blob on the left
        s0, e0 = int(starts[0]), int(ends[0])
        blob_w = e0 - s0
        if blob_w < 4 or blob_w > int(w * 0.35):
            return None
        ys = np.where(ink[:, s0:e0])[0]
        if len(ys) == 0:
            return None
        blob_h = int(ys.max() - ys.min() + 1)
        # A digit like '0' has blob_h >= 0.55 * h. A sign ('+' or '-') has smaller height than a full digit.
        second_ys = np.where(ink[:, int(starts[1]) : int(ends[1])])[0]
        digit_h = int(second_ys.max() - second_ys.min() + 1) if len(second_ys) > 0 else h
        if blob_h <= max(5, int(digit_h * 0.32)) and (blob_h / float(max(1, blob_w))) <= 0.48:
            return "-"
        if blob_h <= int(digit_h * 0.82) and (blob_h / float(max(1, blob_w))) >= 0.55:
            return "+"
        return None

    @classmethod
    def recognize_crop(
        cls,
        raw_crop: Image.Image,
        model_path: Path,
        dict_path: Optional[Path] = None,
        preprocess_mode: str = "raw",
        enable_stack_split: bool = False,
        allow_rotate: bool = True,
        is_sub_crop: bool = False,
    ) -> Dict[str, Any]:
        """
        Recognize an oriented crop using the requested preprocessing mode.
        Raw is the default; other modes evaluate candidate passes by OCR confidence.
        """
        w, h = raw_crop.size
        rot_deg = 0
        oriented_img = raw_crop
        if allow_rotate and h > w * 1.22:
            rot_deg = 90
            oriented_img = raw_crop.rotate(-90, expand=True)

        candidates = build_ocr_passes(oriented_img, mode=preprocess_mode, is_sub_crop=is_sub_crop)
        pass_details = []
        best = None
        for pass_name, candidate in candidates:
            txt, conf = cls.predict_single(candidate, model_path, dict_path, is_sub_crop=is_sub_crop)
            pass_details.append({"passName": pass_name, "text": txt, "confidence": round(conf, 4)})
            if best is None or (bool(txt), conf) > (bool(best[1]), best[2]):
                best = (pass_name, txt, conf, candidate)
        pass_name, txt, conf, processed_img = best
        return {
            "text": txt,
            "confidence": round(conf, 4),
            "passUsed": pass_name,
            "rotationUsed": rot_deg,
            "processedPreview": pil_to_base64_png(processed_img),
            "passDetails": pass_details,
            "semantics": parse_dimension_semantics(txt),
        }

    @classmethod
    def recognize_bluebox_3regions(
        cls,
        raw_crop: Image.Image,
        sub_yolo_path: Path,
        ppocr_path: Path,
        dict_path: Optional[Path] = None,
        preprocess_mode: str = "raw",
        sub_conf_thres: float = 0.45,
    ) -> Optional[Dict[str, Any]]:
        """
        PURE RAW 3-Region YOLO mode (NO image processing, NO ink-projection coordinate tweaking):
          1. Takes the raw Blue-Box crop (only rotated 90° CW if vertical h > w * 1.22).
          2. Runs the 3-class YOLO ONNX model directly on the raw Blue-Box.
          3. Only accepts the 3-region split when ALL 3 classes (0, 1, 2) are genuinely present
             as a stacked tolerance (never chops Nominal-Only boxes).
        """
        if not sub_yolo_path.is_file():
            return None

        w, h = raw_crop.size
        best_rot = 0
        oriented_img = raw_crop
        if h > w * 1.22:
            best_rot = 90
            oriented_img = raw_crop.rotate(-90, expand=True)

        sub_det = YOLODetectorONNX.detect(
            oriented_img, sub_yolo_path, conf_thres=sub_conf_thres, imgsz_override=320
        )
        if sub_det.get("numClasses", 1) < 3 or not sub_det.get("boxes"):
            return None

        ow, oh = oriented_img.size
        # Pick highest-confidence raw YOLO box per class (0: nominal, 1: tol_up, 2: tol_lower)
        by_cid: Dict[int, Dict[str, Any]] = {}
        for b in sorted(sub_det["boxes"], key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
            cid = int(b.get("classId", 0))
            if cid in (0, 1, 2) and cid not in by_cid:
                by_cid[cid] = dict(b)

        # ONLY split when ALL 3 regions (0: nominal, 1: tol_up, 2: tol_lower) are present!
        # Never chop Nominal-Only or single-line boxes (which only have 1 or 2 boxes).
        if not (0 in by_cid and 1 in by_cid and 2 in by_cid):
            return None

        nb = by_cid[0]
        b1 = by_cid[1]
        b2 = by_cid[2]

        # Geometric sanity check for a true 3-region Stack Tolerance:
        # 1) Both tolerance boxes must be tall enough to be real digits (>= 8px), not thin underline/tick fragments
        if b1["h"] < 8 or b2["h"] < 8 or b1["w"] < 6 or b2["w"] < 6:
            return None

        # 2) tol_up (1) must be vertically above tol_lower (2)
        cy1 = b1["y"] + b1["h"] * 0.5
        cy2 = b2["y"] + b2["h"] * 0.5
        if cy2 - cy1 < max(6.0, min(b1["h"], b2["h"]) * 0.50):
            return None

        # 3) tol_up and tol_lower must be to the right of nominal's left edge and shorter than full-height nominal
        if b1["x"] < nb["x"] + nb["w"] * 0.35 or b2["x"] < nb["x"] + nb["w"] * 0.35:
            return None
        if b1["h"] >= nb["h"] * 0.85 or b2["h"] >= nb["h"] * 0.85:
            return None
        # Reject if nominal box only covers the first digit on the far left with a huge unboxed gap in the middle (e.g. '0' ... '80'/'0U' in '0.180')
        tol_left = min(b1["x"], b2["x"])
        if nb["w"] < ow * 0.28 and (tol_left - (nb["x"] + nb["w"])) > ow * 0.20:
            return None

        # Draw exact raw YOLO bounding boxes on a copy of oriented_img for visual inspection
        from PIL import ImageDraw
        overlay_img = oriented_img.copy()
        draw = ImageDraw.Draw(overlay_img)
        palette = {0: "#0284c7", 1: "#9333ea", 2: "#d97706"}

        region_results: List[Dict[str, Any]] = []
        pass_details: List[Dict[str, Any]] = []
        texts_by_cid: Dict[int, str] = {0: "", 1: "", 2: ""}

        for cid in (0, 1, 2):
            sb = by_cid[cid]
            rx0 = max(0, min(ow - 1, int(round(sb["x"]))))
            ry0 = max(0, min(oh - 1, int(round(sb["y"]))))
            rx1 = max(rx0 + 1, min(ow, int(round(sb["x"] + sb["w"]))))
            ry1 = max(ry0 + 1, min(oh, int(round(sb["y"] + sb["h"]))))

            draw.rectangle([rx0, ry0, rx1 - 1, ry1 - 1], outline=palette.get(cid, "#16a34a"), width=2)

            raw_sub_img = oriented_img.crop((rx0, ry0, rx1, ry1))
            sub_result = cls.recognize_crop(raw_sub_img, ppocr_path, dict_path,
                                            preprocess_mode=preprocess_mode, allow_rotate=False, is_sub_crop=(cid in (1, 2)))
            sub_txt, sub_conf = sub_result["text"], sub_result["confidence"]
            texts_by_cid[cid] = sub_txt.strip()

            cls_label = DEFAULT_CLASS_NAMES.get(cid, f"class_{cid}")
            region_results.append(
                {
                    "classId": cid,
                    "className": cls_label,
                    "box": [rx0, ry0, rx1 - rx0, ry1 - ry0],
                    "yoloConf": round(float(sb.get("confidence", 0.0)), 3),
                    "text": texts_by_cid[cid],
                    "confidence": round(sub_conf, 4),
                    "passUsed": sub_result["passUsed"],
                    "passDetails": sub_result["passDetails"],
                    "preview": pil_to_base64_png(raw_sub_img),
                }
            )
            pass_details.append(
                {
                    "passName": f"raw_yolo[{cls_label}] box=[{rx0},{ry0},{rx1-rx0},{ry1-ry0}]",
                    "text": texts_by_cid[cid],
                    "confidence": round(sub_conf, 4),
                }
            )

        nom_txt = texts_by_cid[0]
        up_txt = texts_by_cid[1]
        low_txt = texts_by_cid[2]

        def _is_valid_tol_text(t: str) -> bool:
            t = t.strip()
            if not t or not any(c.isdigit() for c in t):
                return False
            # A sliced decimal fragment like '.0' or '.8' (from '3.8') starts with a bare dot — never a valid CAD tolerance
            if t.startswith("."):
                return False
            if any(c.isalpha() and c not in "oOiI°'\"rad" for c in t):
                return False
            if t in ("0", "+0", "-0"):
                return True
            return t[0] in "+-±" or "." in t or "°" in t or "'" in t

        # Validate that nominal has digits, both upper/lower tolerances are valid, and at least one tolerance has a non-zero digit (1-9)
        has_nonzero_tol = any(c in "123456789" for c in (up_txt + low_txt))
        if not (any(c.isdigit() for c in nom_txt) and has_nonzero_tol and _is_valid_tol_text(up_txt) and _is_valid_tol_text(low_txt)):
            return None

        combined = f"{nom_txt} {up_txt}/{low_txt}".strip()

        confs = [r["confidence"] for r in region_results if r["text"]]
        avg_conf = round(float(np.mean(confs)), 4) if confs else 0.0

        return {
            "text": combined,
            "confidence": avg_conf,
            "passUsed": f"{preprocess_mode}_yolo_3regions",
            "rotationUsed": best_rot,
            "processedPreview": pil_to_base64_png(overlay_img),
            "passDetails": pass_details,
            "regionCrops": region_results,
            "semantics": {
                "nominal": nom_txt,
                "upperTol": up_txt,
                "lowerTol": low_txt,
                "type": "yolo_3regions",
            },
        }


def list_available_models() -> Dict[str, Any]:
    """List all YOLO (.onnx), PP-OCR (.onnx + .txt), and Fine-tune (.pt) models in models/."""
    YOLO_ONNX_DIR.mkdir(parents=True, exist_ok=True)
    PPOCR_ONNX_DIR.mkdir(parents=True, exist_ok=True)
    PT_LIB_DIR.mkdir(parents=True, exist_ok=True)

    yolo_models = [
        YOLODetectorONNX.inspect_model(p)
        for p in sorted(YOLO_ONNX_DIR.glob("*.onnx"), key=lambda x: x.stat().st_mtime, reverse=True)
    ]

    ppocr_models = []
    for p in sorted(
        PPOCR_ONNX_DIR.glob("*.onnx"),
        key=lambda x: (0 if x.name == DEFAULT_PPOCR_MODEL else 1, -x.stat().st_mtime),
    ):
        if "det" in p.stem.lower():
            continue
        ppocr_models.append(
            {
                "name": p.name,
                "path": str(p),
                "sizeMb": round(p.stat().st_size / (1024 * 1024), 2),
                "recommendedDict": "ppocrv6_dict.txt",
            }
        )

    dicts = [
        {
            "name": p.name,
            "path": str(p),
            "charCount": len([l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]),
        }
        for p in sorted(PPOCR_ONNX_DIR.glob("*.txt"))
    ]

    pt_models = [
        {
            "name": p.name,
            "path": str(p),
            "sizeMb": round(p.stat().st_size / (1024 * 1024), 2),
        }
        for p in sorted(PT_LIB_DIR.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    ]

    workspace_root = BASE_DIR.parent
    datasets = []
    synth_3reg = BASE_DIR / "synthetic_3regions_dataset"
    if synth_3reg.is_dir() and (synth_3reg / "data.yaml").is_file():
        datasets.append({"name": "synthetic_3regions_dataset (3 vùng: nominal, tol_up, tol_lower)", "path": str(synth_3reg)})

    for ds_name in ("ready-to-train-priority", "ready-to-train", "synthetic-ocr-regions"):
        ds_path = workspace_root / ds_name
        if ds_path.is_dir() and (ds_path / "data.yaml").is_file():
            datasets.append({"name": ds_name, "path": str(ds_path)})

    sample_pdfs = [
        {"name": p.name, "path": str(p), "sizeKb": round(p.stat().st_size / 1024, 1)}
        for p in sorted((BASE_DIR / "sample_pdfs").glob("*.pdf"))
    ]
    uploads_dir = BASE_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    uploaded_pdfs = [
        {"name": p.name, "path": str(p), "sizeKb": round(p.stat().st_size / 1024, 1)}
        for p in sorted(uploads_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if p.is_file() and p.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg")
    ]

    default_1cls = (
        DEFAULT_1CLASS_MODEL
        if (YOLO_ONNX_DIR / DEFAULT_1CLASS_MODEL).is_file()
        else "yolo_best_1class.onnx"
    )
    return {
        "defaults": {
            "yoloModel": default_1cls,
            "subRegionModel": DEFAULT_3REGION_MODEL,
            "ppocrModel": DEFAULT_PPOCR_MODEL,
        },
        "yoloOnnx": yolo_models,
        "ppocrOnnx": ppocr_models,
        "ppocrDicts": dicts,
        "ptLibrary": pt_models,
        "datasets": datasets,
        "samplePdfs": sample_pdfs,
        "uploadedPdfs": uploaded_pdfs,
    }


def _find_3region_sub_yolo(preferred_name: Optional[str] = None) -> Optional[Path]:
    """Return the requested 3-class YOLO ONNX model, or the newest 3-class model if not specified."""
    if preferred_name:
        cand = YOLO_ONNX_DIR / Path(preferred_name).name
        if cand.is_file():
            info = YOLODetectorONNX.inspect_model(cand)
            if info.get("numClasses") == 3:
                return cand
    for p in sorted(YOLO_ONNX_DIR.glob("*.onnx"), key=lambda x: x.stat().st_mtime, reverse=True):
        info = YOLODetectorONNX.inspect_model(p)
        if info.get("numClasses") == 3:
            return p
    return None


def _find_fullpage_bluebox_yolo(requested_yolo: Path) -> Path:
    """
    Use the user's selected 1-class full-page model (e.g., yolo_best_1class071026.onnx or yolo_best_1class.onnx)
    to detect Blue Boxes on the full PDF page. If a 320x320 3-region model was passed, fall back to
    yolo_best_1class071026.onnx (or yolo_best_1class.onnx).
    """
    info = YOLODetectorONNX.inspect_model(requested_yolo)
    inp_size = info.get("inputSize") or [1024, 1024]
    if info.get("numClasses") == 3 and max(inp_size) <= 416:
        for cand_name in (DEFAULT_1CLASS_MODEL, "yolo_best_1class.onnx"):
            cand_path = YOLO_ONNX_DIR / cand_name
            if cand_path.is_file():
                return cand_path
    return requested_yolo


def run_full_pipeline(
    pdf_or_img_path: str,
    yolo_model_name: str,
    ppocr_model_name: str,
    dict_name: Optional[str] = None,
    page_index: int = 0,
    dpi: int = 250,
    rotation: int = 0,
    conf_thres: float = 0.30,
    preprocess_mode: str = "raw",
    clean_bg: bool = True,
    enable_stack_split: bool = True,
) -> Dict[str, Any]:
    """Run PDF render -> 1-class Full-Page YOLO ONNX -> Always run 3-region YOLO ONNX -> PP-OCR ONNX recognition."""
    t_start = time.perf_counter()
    src_path = Path(pdf_or_img_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {src_path}")

    if src_path.suffix.lower() == ".pdf":
        page_img, page_meta = render_pdf_page(
            str(src_path), page_index=page_index, dpi=dpi, rotation=rotation, clean_bg=clean_bg
        )
    else:
        with Image.open(src_path) as im:
            page_img = im.convert("RGB")
        if rotation in (90, 180, 270):
            page_img = page_img.rotate(-rotation, expand=True)
        page_meta = {
            "pageIndex": 0,
            "totalPages": 1,
            "width": page_img.width,
            "height": page_img.height,
            "dpi": dpi,
            "rotation": rotation,
        }

    requested_yolo_path = YOLO_ONNX_DIR / Path(yolo_model_name).name
    page_yolo_path = _find_fullpage_bluebox_yolo(requested_yolo_path)
    ppocr_path = PPOCR_ONNX_DIR / Path(ppocr_model_name).name
    dict_path = (PPOCR_ONNX_DIR / Path(dict_name).name) if dict_name else None

    det_result = YOLODetectorONNX.detect(page_img, page_yolo_path, conf_thres=conf_thres)
    boxes = det_result["boxes"]
    if src_path.suffix.lower() != ".pdf" and (not boxes or (page_img.width <= 600 and page_img.height <= 600)):
        boxes = [
            {
                "id": 1,
                "classId": 0,
                "className": "dimension_crop",
                "confidence": 1.0,
                "x": 0,
                "y": 0,
                "w": page_img.width,
                "h": page_img.height,
            }
        ]

    t_ocr_start = time.perf_counter()
    _, _, resolved_dict = PPOCRRecognizerONNX.load(ppocr_path, dict_path)

    # Always use the 3-region YOLO model (yolo_3regions_31026.onnx) inside each Blue-Box crop
    req_info = YOLODetectorONNX.inspect_model(page_yolo_path)
    sub_yolo_3reg = _find_3region_sub_yolo(DEFAULT_3REGION_MODEL) if enable_stack_split else None

    for b in boxes:
        raw_crop = extract_crop_with_context(page_img, b["x"], b["y"], b["w"], b["h"], context_pad=4)
        ocr_out = None
        if sub_yolo_3reg is not None:
            ocr_out = PPOCRRecognizerONNX.recognize_bluebox_3regions(
                raw_crop,
                sub_yolo_path=sub_yolo_3reg,
                ppocr_path=ppocr_path,
                dict_path=resolved_dict,
                preprocess_mode=preprocess_mode,
            )
        if ocr_out is None:
            ocr_out = PPOCRRecognizerONNX.recognize_crop(
                raw_crop,
                ppocr_path,
                dict_path=resolved_dict,
                preprocess_mode=preprocess_mode,
                enable_stack_split=False,
            )
        b["rawCropPreview"] = pil_to_base64_png(raw_crop)
        b["processedPreview"] = ocr_out["processedPreview"]
        b["ocrText"] = ocr_out["text"]
        b["ocrConfidence"] = ocr_out["confidence"]
        b["passUsed"] = ocr_out["passUsed"]
        b["rotationUsed"] = ocr_out["rotationUsed"]
        b["passDetails"] = ocr_out.get("passDetails", [])
        b["regionCrops"] = ocr_out.get("regionCrops", [])
        b["semantics"] = ocr_out["semantics"]

    ocr_ms = round((time.perf_counter() - t_ocr_start) * 1000, 1)
    total_ms = round((time.perf_counter() - t_start) * 1000, 1)

    read_boxes = [b for b in boxes if b.get("ocrText")]
    avg_conf = (
        round(float(np.mean([b["ocrConfidence"] for b in read_boxes])), 4)
        if read_boxes
        else 0.0
    )
    stacked_count = sum(
        1
        for b in boxes
        if (b.get("semantics") or {}).get("type") in ("stacked", "yolo_3regions", "symmetric")
        or b.get("classId", 0) in (1, 2)
    )

    return {
        "fileName": src_path.name,
        "page": page_meta,
        "pageImage": pil_to_base64_png(page_img),
        "yolo": {
            "model": requested_yolo_path.name,
            "pageDetectorModel": det_result["modelName"],
            "subRegionModel": sub_yolo_3reg.name if sub_yolo_3reg else None,
            "numClasses": req_info.get("numClasses", det_result["numClasses"]),
            "classNames": req_info.get("classNames", det_result["classNames"]),
            "inputSize": req_info.get("inputSize", det_result["inputSize"]),
            "elapsedMs": det_result["elapsedMs"],
        },
        "ppocr": {
            "model": ppocr_path.name,
            "dict": resolved_dict.name,
            "preprocessMode": preprocess_mode,
            "elapsedMs": ocr_ms,
            "readCount": len(read_boxes),
            "avgConfidence": avg_conf,
            "stackedCount": stacked_count,
        },
        "totalElapsedMs": total_ms,
        "boxCount": len(boxes),
        "boxes": boxes,
    }


def run_ppocr_on_uploaded_result(
    result_data: Dict[str, Any],
    pdf_or_img_path: Optional[str],
    ppocr_model_name: str = DEFAULT_PPOCR_MODEL,
    dict_name: Optional[str] = None,
    preprocess_mode: str = "raw",
    enable_stack_split: bool = True,
    yolo_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    When a user uploads a JSON result (boxes with or without OCR/crop previews),
    ensure PP-OCR runs on every box (using either embedded rawCropPreview or the current PDF page)
    and return the full payload ready for the UI table and side-by-side PP-OCR panel.
    """
    import base64
    import io

    t0 = time.perf_counter()
    ppocr_path = PPOCR_ONNX_DIR / Path(ppocr_model_name).name
    dict_path = (PPOCR_ONNX_DIR / Path(dict_name).name) if dict_name else None
    _, _, resolved_dict = PPOCRRecognizerONNX.load(ppocr_path, dict_path)

    page_img: Optional[Image.Image] = None
    page_b64 = result_data.get("pageImage")
    if page_b64 and isinstance(page_b64, str) and "," in page_b64:
        raw = base64.b64decode(page_b64.split(",", 1)[1])
        page_img = Image.open(io.BytesIO(raw)).convert("RGB")
    elif pdf_or_img_path and Path(pdf_or_img_path).is_file():
        src = Path(pdf_or_img_path)
        p_idx = int((result_data.get("page") or {}).get("pageIndex", 0))
        dpi = int((result_data.get("page") or {}).get("dpi", 250))
        rot = int((result_data.get("page") or {}).get("rotation", 0))
        if src.suffix.lower() == ".pdf":
            page_img, page_meta = render_pdf_page(str(src), page_index=p_idx, dpi=dpi, rotation=rot, clean_bg=True)
            result_data.setdefault("page", page_meta)
        else:
            with Image.open(src) as im:
                page_img = im.convert("RGB")
        result_data["pageImage"] = pil_to_base64_png(page_img)

    sub_yolo_3reg = _find_3region_sub_yolo(yolo_model_name) if enable_stack_split else None
    boxes = result_data.get("boxes") or []
    for idx, b in enumerate(boxes, 1):
        b.setdefault("id", idx)
        b.setdefault("classId", 0)
        b.setdefault("className", DEFAULT_CLASS_NAMES.get(b["classId"], f"class_{b['classId']}"))
        b.setdefault("confidence", b.get("yoloConf", 1.0))

        raw_crop: Optional[Image.Image] = None
        if b.get("rawCropPreview") and "," in str(b["rawCropPreview"]):
            raw_bytes = base64.b64decode(str(b["rawCropPreview"]).split(",", 1)[1])
            raw_crop = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        elif page_img is not None:
            raw_crop = extract_crop_with_context(page_img, b.get("x", 0), b.get("y", 0), b.get("w", 32), b.get("h", 24))

        if raw_crop is not None:
            ocr_out = None
            if sub_yolo_3reg is not None:
                ocr_out = PPOCRRecognizerONNX.recognize_bluebox_3regions(
                    raw_crop,
                    sub_yolo_path=sub_yolo_3reg,
                    ppocr_path=ppocr_path,
                    dict_path=resolved_dict,
                    preprocess_mode=preprocess_mode,
                )
            if ocr_out is None:
                ocr_out = PPOCRRecognizerONNX.recognize_crop(
                    raw_crop,
                    ppocr_path,
                    dict_path=resolved_dict,
                    preprocess_mode=preprocess_mode,
                    enable_stack_split=False,
                )
            b["rawCropPreview"] = pil_to_base64_png(raw_crop)
            b["processedPreview"] = ocr_out["processedPreview"]
            b["ocrText"] = ocr_out["text"]
            b["ocrConfidence"] = ocr_out["confidence"]
            b["passUsed"] = ocr_out["passUsed"]
            b["rotationUsed"] = ocr_out["rotationUsed"]
            b["passDetails"] = ocr_out.get("passDetails", [])
            b["regionCrops"] = ocr_out.get("regionCrops", [])
            b["semantics"] = ocr_out["semantics"]
        else:
            b.setdefault("ocrText", "")
            b.setdefault("ocrConfidence", b.get("ocrConf", 0.0))
            b.setdefault("passUsed", "uploaded_json")
            b.setdefault("semantics", parse_dimension_semantics(b.get("ocrText", "")))

    ocr_ms = round((time.perf_counter() - t0) * 1000, 1)
    read_boxes = [b for b in boxes if b.get("ocrText")]
    avg_conf = round(float(np.mean([b["ocrConfidence"] for b in read_boxes])), 4) if read_boxes else 0.0
    stacked_count = sum(
        1
        for b in boxes
        if (b.get("semantics") or {}).get("type") in ("stacked", "yolo_3regions", "symmetric")
        or b.get("classId", 0) in (1, 2)
    )

    result_data["boxCount"] = len(boxes)
    result_data["boxes"] = boxes
    result_data.setdefault("yolo", {"model": "uploaded_result", "elapsedMs": 0})
    result_data["ppocr"] = {
        "model": ppocr_path.name,
        "dict": resolved_dict.name,
        "preprocessMode": preprocess_mode,
        "elapsedMs": ocr_ms,
        "readCount": len(read_boxes),
        "avgConfidence": avg_conf,
        "stackedCount": stacked_count,
    }
    return result_data
