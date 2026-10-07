"""
Advanced CAD Image Preprocessor for PDF -> YOLO -> PP-OCR Pipeline.
Pure NumPy + Pillow implementation (zero OpenCV binary dependency required).

Key capabilities:
1. High-DPI PDF page rendering with optional background normalization.
2. Context-aware crop extraction + white canvas padding (protects +/-/./Ø at edges).
3. Auto-orientation for vertical CAD callouts (ISO/ASME 90° CCW & 90° CW).
4. Stroke-preserving leader/extension line removal (smart thickness-aware morphology).
5. Projection-profile deskewing for slightly tilted callouts.
6. Multi-pass enhancement (Standard Lanczos, Leader-Cleaned, Autocontrast+Unsharp, Otsu Binary).
7. Projection-profile 2-tier stacked tolerance splitting (Nominal | Upper Tol / Lower Tol).
"""
from __future__ import annotations

import base64
import io
import math
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_pdf_page(
    pdf_path: str,
    page_index: int = 0,
    dpi: int = 250,
    rotation: int = 0,
    clean_bg: bool = True,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Render a single PDF page to a crisp RGB PIL Image."""
    doc = fitz.open(pdf_path)
    try:
        total_pages = len(doc)
        if page_index < 0 or page_index >= total_pages:
            raise ValueError(f"Page index {page_index} out of range (0..{total_pages - 1})")
        page = doc[page_index]
        zoom = max(72, min(600, int(dpi))) / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()

    rotation = int(rotation) % 360
    if rotation in (90, 180, 270):
        img = img.rotate(-rotation, expand=True)

    if clean_bg:
        arr = np.asarray(img, dtype=np.uint8).copy()
        # Push near-white paper noise (>238) to pure white (255) and darken ink (<190) slightly
        gray = arr.mean(axis=2)
        arr[gray > 238] = 255
        ink_mask = gray < 185
        if np.any(ink_mask):
            arr[ink_mask] = (arr[ink_mask].astype(np.float32) * 0.82).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")

    meta = {
        "pageIndex": page_index,
        "totalPages": total_pages,
        "width": img.width,
        "height": img.height,
        "dpi": dpi,
        "rotation": rotation,
    }
    return img, meta


def extract_crop_with_context(
    page_img: Image.Image,
    x: float,
    y: float,
    w: float,
    h: float,
    context_pad: int = 4,
) -> Image.Image:
    """Extract bounding box from page with slight context margin clamped to image bounds."""
    pw, ph = page_img.size
    x0 = max(0, int(math.floor(x - context_pad)))
    y0 = max(0, int(math.floor(y - context_pad)))
    x1 = min(pw, int(math.ceil(x + w + context_pad)))
    y1 = min(ph, int(math.ceil(y + h + context_pad)))
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGB", (16, 16), (255, 255, 255))
    return page_img.crop((x0, y0, x1, y1)).convert("RGB")


def _run_lengths_1d(mask_1d: np.ndarray) -> np.ndarray:
    """Return an array of the same length with the contiguous True run-length at each position."""
    n = len(mask_1d)
    if n == 0 or not np.any(mask_1d):
        return np.zeros(n, dtype=np.int32)
    padded = np.concatenate(([False], mask_1d, [False]))
    diffs = np.diff(padded.astype(np.int32))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    out = np.zeros(n, dtype=np.int32)
    for s, e in zip(starts, ends):
        out[s:e] = e - s
    return out


def clean_leader_lines(img: Image.Image, is_sub_crop: bool = False) -> Image.Image:
    """
    Stroke-preserving CAD leader/dimension line and border noise remover.
    - Removes long horizontal (>=60% width) or vertical (>=70% height) lines where perpendicular stroke thickness <= 4px.
    - Removes horizontal leader/divider bars that touch the left or right border of a Blue-Box crop
      (e.g. dimension line extending between tol_up and tol_lower to the right edge, or underline under ±tol).
    - Removes short vertical/horizontal ticks hugging the outer border of the crop.
    Never erases interior minus signs or character strokes.
    """
    w, h = img.size
    if w < 16 or h < 12:
        return img

    arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    gray = (
        0.299 * arr[:, :, 0].astype(np.float32)
        + 0.587 * arr[:, :, 1].astype(np.float32)
        + 0.114 * arr[:, :, 2].astype(np.float32)
    )
    dark = gray < 200

    v_thickness = np.zeros((h, w), dtype=np.int32)
    for col in range(w):
        v_thickness[:, col] = _run_lengths_1d(dark[:, col])

    h_thickness = np.zeros((h, w), dtype=np.int32)
    for row in range(h):
        h_thickness[row, :] = _run_lengths_1d(dark[row, :])

    min_h_len = max(18, int(w * 0.58))
    min_v_len = max(16, int(h * 0.72))

    erase_mask = np.zeros((h, w), dtype=bool)
    # 1. Long horizontal leader lines where vertical thickness is thin (<= 4 px)
    erase_mask |= (h_thickness >= min_h_len) & (v_thickness <= 4)

    # 2. Long vertical extension lines where horizontal thickness is thin (<= 4 px)
    erase_mask |= (v_thickness >= min_v_len) & (h_thickness <= 4)

    # 3. Horizontal leader lines, parallel top/bottom frame lines, or divider bars touching the right/left border
    if not is_sub_crop and w >= 40 and h >= 20:
        right_span_x0 = int(w * 0.56)
        right_span_len = max(1, w - right_span_x0)
        for r in range(h):
            row = dark[r, :]
            if not np.any(row):
                continue
            padded = np.concatenate(([False], row, [False]))
            starts = np.where(padded[1:] & ~padded[:-1])[0]
            ends = np.where(~padded[1:] & padded[:-1])[0]
            is_edge_band = (r <= int(h * 0.23)) or (r >= int(h * 0.77))

            # Top/bottom parallel dimension lines that may have small gaps around ticks (e.g. 108-M1063 #1, #10, #15)
            if is_edge_band and np.sum(row & (v_thickness[r, :] <= 4)) >= max(22, int(w * 0.45)):
                erase_mask[r, :] |= row & (v_thickness[r, :] <= 3)

            # Right-half horizontal divider line running through tolerance stack to right border (e.g. 105-M0800 #23..#29)
            if (
                np.any(row[max(0, w - 6) :])
                and np.sum(row[right_span_x0:] & (v_thickness[r, right_span_x0:] <= 4)) >= int(right_span_len * 0.62)
            ):
                erase_mask[r, right_span_x0:] |= row[right_span_x0:] & (v_thickness[r, right_span_x0:] <= 3)

            for s, e in zip(starts, ends):
                seg_len = e - s
                touches_right = e >= w - 6
                touches_left = s <= 5
                # Right-border touching divider/underline or top/bottom border line
                if (touches_right and seg_len >= max(16, int(w * 0.20))) or (
                    is_edge_band and seg_len >= max(18, int(w * 0.36))
                ):
                    thin_cols = v_thickness[r, s:e] <= 4
                    erase_mask[r, s:e] |= thin_cols
                # Short left-border leader stub (like '-3.410' or '-35.770' where leader line touches x=0..5)
                elif touches_left and seg_len <= max(14, int(w * 0.08)) and (e < w - 4):
                    if np.all(v_thickness[r, s:e] <= 4):
                        erase_mask[r, s:e] = True

        # 4. Short vertical ticks touching top or bottom border band (e.g. dimension ticks at y=0..5 or y=h-6..h-1)
        top_band_y = max(4, int(h * 0.20))
        bot_band_y = min(h - 4, int(h * 0.80))
        for c in range(w):
            col = dark[:, c]
            if not np.any(col):
                continue
            padded = np.concatenate(([False], col, [False]))
            starts = np.where(padded[1:] & ~padded[:-1])[0]
            ends = np.where(~padded[1:] & padded[:-1])[0]
            for s, e in zip(starts, ends):
                seg_len = e - s
                if s <= 5 and seg_len <= max(16, int(h * 0.25)):
                    erase_mask[s : min(e, top_band_y), c] |= h_thickness[s : min(e, top_band_y), c] <= 4
                elif e >= h - 5 and seg_len <= max(16, int(h * 0.25)):
                    erase_mask[max(s, bot_band_y) : e, c] |= h_thickness[max(s, bot_band_y) : e, c] <= 4

    # 5. Border frame artifacts (1-2px dark lines hugging the top/bottom/left/right border)
    if h >= 16 and w >= 24:
        for r in (0, 1, 2, h - 3, h - 2, h - 1):
            if np.mean(dark[r, :]) > 0.50:
                erase_mask[r, :] |= v_thickness[r, :] <= 4
        for c in (0, 1, 2, w - 3, w - 2, w - 1):
            if np.mean(dark[:, c]) > 0.50:
                erase_mask[:, c] |= h_thickness[:, c] <= 4

    if np.any(erase_mask):
        arr[erase_mask] = 255
    return Image.fromarray(arr, mode="RGB")



def deskew_small_angle(img: Image.Image, max_angle: float = 12.0) -> Image.Image:
    """
    Deskew slightly tilted text using horizontal projection variance maximization.
    Only applies rotation if an angle in [-max_angle, +max_angle] significantly sharpens
    row-level ink alignment.
    """
    w, h = img.size
    if w < 28 or h < 14:
        return img

    gray = np.asarray(img.convert("L"), dtype=np.float32)
    ink = np.clip(255.0 - gray, 0.0, 255.0)
    if np.sum(ink > 40) < 20:
        return img

    base_row_sums = np.sum(ink, axis=1)
    base_score = float(np.var(base_row_sums))
    if base_score <= 1e-3:
        return img

    best_angle = 0.0
    best_score = base_score
    for angle in (-10.0, -7.0, -4.0, -2.0, 2.0, 4.0, 7.0, 10.0):
        if abs(angle) > max_angle:
            continue
        rot = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(255, 255, 255))
        r_gray = np.asarray(rot.convert("L"), dtype=np.float32)
        r_ink = np.clip(255.0 - r_gray, 0.0, 255.0)
        score = float(np.var(np.sum(r_ink, axis=1)))
        if score > best_score * 1.18:
            best_score = score
            best_angle = angle

    if abs(best_angle) >= 1.5:
        return img.rotate(best_angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    return img


def pad_and_upscale(img: Image.Image, pad: int = 8, scale: float = 2.0) -> Image.Image:
    """Add clean white border and Lanczos upscale for small CAD crops."""
    padded = ImageOps.expand(img.convert("RGB"), border=pad, fill=(255, 255, 255))
    w, h = padded.size
    eff_scale = scale if min(w, h) < 96 else max(1.0, scale * 0.75)
    if abs(eff_scale - 1.0) > 0.05:
        nw = max(16, int(round(w * eff_scale)))
        nh = max(16, int(round(h * eff_scale)))
        return padded.resize((nw, nh), Image.Resampling.LANCZOS)
    return padded


def otsu_binarize(img: Image.Image) -> Image.Image:
    """Compute Otsu threshold in pure NumPy and return high-contrast RGB image."""
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    if total == 0:
        return img.convert("RGB")

    sum_total = np.dot(np.arange(256), hist)
    weight_bg = 0.0
    sum_bg = 0.0
    max_var = -1.0
    thresh = 180

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t

    thresh = min(225, max(110, int(thresh)))
    binary = np.where(gray <= thresh, 0, 255).astype(np.uint8)
    rgb = np.stack([binary, binary, binary], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def build_ocr_passes(img: Image.Image, mode: str = "auto", is_sub_crop: bool = False) -> List[Tuple[str, Image.Image]]:
    """
    Build preprocessed image candidates for PP-OCR recognition.
    Modes:
      - 'fast': Only leader-cleaned padded + standard padded
      - 'clean': Leader-cleaned + autocontrast sharpen
      - 'auto' (default): Full multi-pass (leader_cleaned, standard, contrast_sharpen, otsu)
    """
    if mode not in ("raw", "fast", "clean", "auto"):
        raise ValueError(f"Chế độ preprocessing không hợp lệ: {mode}")
    if mode == "raw":
        return [("raw", img)]
    cleaned_base = clean_leader_lines(img, is_sub_crop=is_sub_crop)
    deskewed = deskew_small_angle(cleaned_base)

    pass_cleaned = pad_and_upscale(deskewed, pad=8, scale=2.0)
    pass_standard = pad_and_upscale(img, pad=8, scale=2.0)

    if mode == "fast":
        if not is_sub_crop:
            return [("leader_cleaned_tight", cleaned_base), ("leader_cleaned", pass_cleaned), ("standard", pass_standard)]
        return [("leader_cleaned", pass_cleaned), ("standard", pass_standard)]

    # Contrast + Unsharp mask pass (enhances faint decimal points and +/- signs)
    auto_c = ImageOps.autocontrast(pass_cleaned.convert("L"), cutoff=1)
    sharpened = auto_c.filter(ImageFilter.UnsharpMask(radius=1.5, percent=170, threshold=2)).convert("RGB")

    # Otsu binarized pass
    otsu_img = otsu_binarize(pass_cleaned)

    if mode == "clean":
        if not is_sub_crop:
            return [("leader_cleaned_tight", cleaned_base), ("leader_cleaned", pass_cleaned), ("contrast_sharpen", sharpened)]
        return [("leader_cleaned", pass_cleaned), ("contrast_sharpen", sharpened)]

    if not is_sub_crop:
        return [
            ("leader_cleaned_tight", cleaned_base),
            ("leader_cleaned", pass_cleaned),
            ("contrast_sharpen", sharpened),
            ("standard", pass_standard),
            ("otsu_binary", otsu_img),
        ]

    return [
        ("leader_cleaned", pass_cleaned),
        ("contrast_sharpen", sharpened),
        ("standard", pass_standard),
        ("otsu_binary", otsu_img),
    ]


def split_stacked_tolerance_projection(img: Image.Image) -> Dict[str, Any]:
    """
    Detect and split a 2-tier stacked tolerance box into 3 sub-regions:
      [nominal_box, upper_tol_box, lower_tol_box]
    using Vertical & Horizontal Projection Profiles.
    """
    w, h = img.size
    oriented = img
    rot_applied = 0
    if h > w * 1.22:
        # Standard vertical CAD dimension reads bottom-to-top (rotate 90 CCW to make horizontal)
        oriented = img.rotate(90, expand=True)
        rot_applied = 270
        w, h = oriented.size

    # A stacked tolerance box is relatively tall compared to a single-line number
    if (h / float(max(1, w)) <= 0.38) or w < 34 or h < 18:
        return {"is_stacked": False, "oriented": oriented, "rotation": rot_applied}

    cleaned = clean_leader_lines(oriented)
    gray = np.asarray(cleaned.convert("L"), dtype=np.uint8)
    dark = gray < 205

    # Trim top/bottom empty rows first
    row_counts = np.sum(dark, axis=1)
    active_rows = np.where(row_counts > 0)[0]
    if len(active_rows) < 10:
        return {"is_stacked": False, "oriented": oriented, "rotation": rot_applied}

    top_r, bot_r = int(active_rows[0]), int(active_rows[-1])
    active_h = bot_r - top_r + 1
    if active_h < 14:
        return {"is_stacked": False, "oriented": oriented, "rotation": rot_applied}

    # 1. Vertical projection profile in [35%..82%] of width to separate Nominal (left) and Tolerance (right)
    v_proj = np.sum(dark[top_r : bot_r + 1, :], axis=0)
    scan_x0 = int(w * 0.35)
    scan_x1 = int(w * 0.82)
    if scan_x1 <= scan_x0 + 4:
        return {"is_stacked": False, "oriented": oriented, "rotation": rot_applied}

    sub_v = v_proj[scan_x0:scan_x1]
    min_v = int(np.min(sub_v))
    max_left = int(np.max(v_proj[:scan_x0])) if scan_x0 > 0 else 1
    if min_v > max(3, int(max_left * 0.35)):
        return {"is_stacked": False, "oriented": oriented, "rotation": rot_applied}

    valleys_x = np.where(sub_v <= min_v + 1)[0]
    valley_x = scan_x0 + int(valleys_x[len(valleys_x) // 2])

    nom_crop = cleaned.crop((0, top_r, valley_x, bot_r + 1))
    tol_crop = cleaned.crop((valley_x, top_r, w, bot_r + 1))
    tw, th = tol_crop.size
    if tw < 12 or th < 14 or nom_crop.width < 10:
        return {"is_stacked": False, "oriented": oriented, "rotation": rot_applied}

    # 2. Horizontal projection profile on tol_crop in [30%..70%] height to separate Upper and Lower tolerance
    tol_dark = dark[top_r : bot_r + 1, valley_x:]
    h_proj = np.sum(tol_dark, axis=1)
    scan_y0 = int(th * 0.30)
    scan_y1 = int(th * 0.70)
    if scan_y1 <= scan_y0 + 2:
        return {"is_stacked": False, "oriented": oriented, "rotation": rot_applied}

    sub_h = h_proj[scan_y0:scan_y1]
    min_h = int(np.min(sub_h))
    if min_h > max(2, int(np.max(h_proj) * 0.30)):
        return {"is_stacked": False, "oriented": oriented, "rotation": rot_applied}

    valleys_y = np.where(sub_h <= min_h)[0]
    valley_y = scan_y0 + int(valleys_y[len(valleys_y) // 2])

    upper_dark = int(np.sum(tol_dark[:valley_y, :]))
    lower_dark = int(np.sum(tol_dark[valley_y:, :]))
    if upper_dark < 8 or lower_dark < 8:
        return {"is_stacked": False, "oriented": oriented, "rotation": rot_applied}

    upper_crop = tol_crop.crop((0, 0, tw, valley_y))
    lower_crop = tol_crop.crop((0, valley_y, tw, th))

    return {
        "is_stacked": True,
        "rotation": rot_applied,
        "valleyX": valley_x,
        "valleyY": top_r + valley_y,
        "nominal": pad_and_upscale(nom_crop, pad=6, scale=2.0),
        "upper": pad_and_upscale(upper_crop, pad=6, scale=2.2),
        "lower": pad_and_upscale(lower_crop, pad=6, scale=2.2),
    }
