"""
Generate an upgraded CAD Blue-Box Synthetic + Real Hard-Case Dataset v3
(3 classes: 0: nominal, 1: tol_up, 2: tol_lower) for fine-tuning best_3regions.pt / best.pt.

Specifically emphasizes all hard cases across 105-M0800, 108-M1063, and 133-MF00307:
1. Vertically touching / 0px-gap stacked tolerances ('6.120 +0.001/-0.001' #4, '4.810 +0.001/-0.001' #14,
   '2.600 +0.005/0' #6, '1.400 0/-0.005' #15, '3.410 0/-0.002' #16, '3.50 -0.01/-0.02' #23).
2. Wide post-decimal kerning in nominal & tolerances ('◬32. 27 0/-0. 1' #1, '4. 000 0/-0. 005' #6,
   '◬3. 450 0/-0. 002' #15, '◬3. 500 0/-0. 002' #19, '35. 770 0/-0. 002' #30, '25. 00 0/-0. 02' #37).
3. Triangle revision prefix ('◬1' inside nominal box) and Trapezoid feature tag ('[1]' / '[⊥]' above nominal, excluded from box).
4. Subscript / lowered single-line symmetric tolerance ('6. 250 ±0. 001' #10, '13. 880 ±0. 001' #39) with dimension line & end ticks (+───+),
   labeled with class 0 (nominal) + class 1 (tol_up) only.
5. Single-digit nominals ('5', '6', '7') + stacked tolerances and horizontal fraction divider bars (#15, #35, #39, #45, #57, #58).
6. 120 verified real CAD hard-case crops from 105-M0800, 108-M1063, and 133-MF00307 with multi-scale/padding/cleaned variants.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import yaml

from engine import PPOCRRecognizerONNX, PPOCR_ONNX_DIR
from preprocessor import pad_and_upscale

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parent
DATASET_DIR = BASE_DIR / "synthetic_3regions_dataset"

CLASSES = {
    0: "nominal",
    1: "tol_up",
    2: "tol_lower",
}

SINGLE_DIGIT_NOMINALS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "1", "3", "5", "6"]

MULTI_NOMINAL_POOL = [
    "0.555", "128.555", "1.00", "2.00", "4.94", "5.4", "6.120", "4.810", "2.600", "6.250",
    "32.27", "4.000", "3.450", "3.500", "35.770", "25.00", "13.880", "1.400", "3.410", "3.480",
    "3.50", "17.20", "2.460", "1.500", "0.80", "4.00", "5.31", "7.05", "2.3", "2.306", "0.24",
    "0.29", "41.42", "7.5", "12.5", "16", "18.2", "0.84", "3.06", "6.07", "25.92", "1.25",
    "2.5", "4.5", "10.5", "12.04", "14.8", "18.25", "22.4", "30.05", "35.6", "42.08", "57.63",
    "0.125", "0.250", "0.750", "3.00", "5.00", "8.00", "10.00", "15.00", "20.00",
    "DP7.0", "DP 7.0", "DP8.0", "Ø7", "Ø7.5", "Ø12", "Ø4.5",
    "R0.02", "C0.02", "R0.49", "C0.5", "R0.5", "R0.05", "C0.05", "C0.1", "R2",
]

ANGLED_RC_POOL = ["R0.02", "C0.02", "R0.49", "C0.5", "R0.05", "C0.05", "R0.5", "C0.1", "R0.2", "C0.2"]

TOL_POOL = [
    ("+0.001", "+0.002"),
    ("+0.001", "-0.002"),
    ("+0.001", "-0.001"),
    ("+0.002", "-0.002"),
    ("-0.1", "+0"),
    ("+0", "-0.1"),
    ("0", "-0.002"),
    ("-0.003", "-0.008"),
    ("+0.01", "-0.02"),
    ("+0", "-0.05"),
    ("+0.005", "0"),
    ("0", "-0.005"),
    ("0", "-0.001"),
    ("0", "-0.1"),
    ("0", "-0.02"),
    ("0", "-0.01"),
    ("-0.01", "-0.02"),
    ("-0.002", "-0.005"),
    ("+0.01", "0"),
    ("+0.02", "0"),
    ("+0.05", "0"),
    ("+0.002", "0"),
    ("+0.004", "+0.001"),
    ("+0.003", "+0.001"),
    ("+0.003", "-0.008"),
    ("+0.003", "0"),
    ("+0.002", "-0.000"),
    ("-0.001", "-0.004"),
    ("+0.02", "-0.01"),
    ("+0.015", "0"),
]

SYM_TOL_POOL = ["±0.001", "±0.002", "±0.005", "±0.01", "±0.02", "±0.05"]


def discover_fonts() -> List[Path]:
    candidates = [
        WORKSPACE_ROOT / "dataset-fonts" / "osifont.ttf",
        WORKSPACE_ROOT / "dataset-fonts" / "osifont-italic.ttf",
        Path(r"C:\Windows\Fonts\bahnschrift.ttf"),
        Path(r"C:\Windows\Fonts\GOTHIC.TTF"),
        Path(r"C:\Windows\Fonts\consola.ttf"),
        Path(r"C:\Windows\Fonts\lucon.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
    ]
    found = [p for p in candidates if p.is_file()]
    if not found:
        raise FileNotFoundError("Không tìm thấy font chữ nào trong dataset-fonts hoặc Windows Fonts")
    return found


def draw_triangle_prefix_mask(height: int, bold: bool = False) -> Image.Image:
    """Draw a CAD revision triangle '◬' with a '1' inside it."""
    h = max(20, int(height))
    w = max(20, int(round(h * 0.92)))
    img = Image.new("L", (w + 4, h + 4), 0)
    draw = ImageDraw.Draw(img)
    lw = 2 if bold or h >= 28 else 1
    top_pt = (w // 2 + 2, 2)
    left_pt = (2, h)
    right_pt = (w + 1, h)
    draw.line([top_pt, left_pt, right_pt, top_pt], fill=255, width=lw)
    cx = w // 2 + 2
    y0 = max(6, int(h * 0.38))
    y1 = max(y0 + 4, int(h * 0.84))
    draw.line([(cx, y0), (cx, y1)], fill=255, width=lw)
    draw.line([(cx - 3, y1), (cx + 3, y1)], fill=255, width=1)
    return img


def draw_trapezoid_top_tag(canvas: Image.Image, center_x: int, top_y: int, ink_val: int = 25) -> None:
    """Draw a small CAD trapezoid tag '[1]' / '[⊥]' above the nominal."""
    draw = ImageDraw.Draw(canvas)
    col = (ink_val, ink_val, ink_val)
    tw_top = 18
    tw_bot = 28
    th = 10
    x0_t = center_x - tw_top // 2
    x1_t = center_x + tw_top // 2
    x0_b = center_x - tw_bot // 2
    x1_b = center_x + tw_bot // 2
    y0 = max(0, top_y)
    y1 = y0 + th
    draw.line([(x0_t, y0), (x1_t, y0), (x1_b, y1), (x0_b, y1), (x0_t, y0)], fill=col, width=1)
    draw.line([(center_x, y0 + 2), (center_x, y1 - 2)], fill=col, width=1)


def render_tight_glyph_mask(
    text: str,
    font_path: Path,
    font_size: int,
    stretch_x: float = 1.0,
    bold: bool = False,
    char_spacing: int = 0,
    post_dot_space: int = 0,
    has_triangle_prefix: bool = False,
) -> Image.Image:
    """Render text to an L-mode mask (255 = ink, 0 = background) cropped tightly to ink pixels."""
    face = ImageFont.truetype(str(font_path), max(11, int(font_size)))
    clean_text = text.lstrip("△◬∆ ") if has_triangle_prefix else text

    if char_spacing > 0 or post_dot_space > 0 or has_triangle_prefix:
        ch_masks = []
        max_h = max(18, int(font_size * 1.45))
        total_w = 12
        for ch in clean_text:
            bb = face.getbbox(ch)
            cw = max(4, (bb[2] - bb[0]) if bb else int(font_size * 0.4))
            extra_after = post_dot_space if ch in (".", ",") else char_spacing
            ch_masks.append((ch, cw, extra_after))
            total_w += cw + extra_after

        tri_mask = None
        tri_w = 0
        if has_triangle_prefix:
            tri_mask = draw_triangle_prefix_mask(int(font_size * 0.88), bold=bold)
            tri_w = tri_mask.width + max(3, char_spacing + 2)
            total_w += tri_w

        canvas = Image.new("L", (total_w + 16, max_h + 14), 0)
        draw = ImageDraw.Draw(canvas)
        cx = 6
        if tri_mask is not None:
            canvas.paste(tri_mask, (cx, 6))
            cx += tri_w

        for ch, cw, extra_after in ch_masks:
            if ch != " ":
                draw.text((cx, 4), ch, fill=255, font=face)
            cx += cw + extra_after
    else:
        bbox = face.getbbox(clean_text)
        gw = max(4, bbox[2] - bbox[0] + 12)
        gh = max(4, bbox[3] - bbox[1] + 12)
        canvas = Image.new("L", (gw, gh), 0)
        draw = ImageDraw.Draw(canvas)
        draw.text((6 - bbox[0], 6 - bbox[1]), clean_text, fill=255, font=face)

    arr = np.asarray(canvas, dtype=np.uint8)
    if bold:
        arr = np.maximum(arr, np.pad(arr[:, :-1], ((0, 0), (1, 0))))

    ys, xs = np.where(arr > 25)
    if len(xs) == 0:
        return Image.new("L", (8, 8), 255)
    cropped = Image.fromarray(arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], mode="L")
    if abs(stretch_x - 1.0) > 0.03:
        nw = max(4, int(round(cropped.width * stretch_x)))
        cropped = cropped.resize((nw, cropped.height), Image.Resampling.BICUBIC)
    return cropped


def paste_ink_mask(canvas: Image.Image, mask: Image.Image, x: int, y: int, ink_val: int = 20) -> Tuple[int, int, int, int]:
    """Paste an L-mode ink mask onto an RGB canvas and return exact ink bounding box [x, y, w, h]."""
    arr_mask = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.where(arr_mask > 20)
    if len(xs) == 0:
        return (x, y, mask.width, mask.height)

    color_layer = Image.new("RGB", mask.size, (ink_val, ink_val, ink_val))
    canvas.paste(color_layer, (x, y), mask)

    bx0 = x + int(xs.min())
    by0 = y + int(ys.min())
    bx1 = x + int(xs.max()) + 1
    by1 = y + int(ys.max()) + 1
    return (bx0, by0, bx1 - bx0, by1 - by0)


def expand_box_safely(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_x: int = 2,
    pad_y: int = 2,
    y_min_limit: int = 0,
    y_max_limit: Optional[int] = None,
) -> List[int]:
    """Expand bounding box by safe margin so no digit stroke is ever clipped."""
    x, y, w, h = box
    y_hi = img_h if y_max_limit is None else min(img_h, y_max_limit)
    x0 = max(0, x - pad_x)
    y0 = max(y_min_limit, y - pad_y)
    x1 = min(img_w, x + w + pad_x)
    y1 = min(y_hi, y + h + pad_y)
    return [int(x0), int(y0), max(2, int(x1 - x0)), max(2, int(y1 - y0))]


def apply_cad_noise_and_blur(
    canvas: Image.Image,
    rng: random.Random,
    mode: str = "auto",
) -> Tuple[Image.Image, str]:
    """
    Apply realistic CAD scan degradations (noise, blur, low-res 1-bit fax/scan pixelation,
    ink bleed, pepper specks, and JPEG mosquito ringing) without shifting pixel coordinates.
    """
    w, h = canvas.size
    if mode == "auto":
        r = rng.random()
        if r < 0.24:
            mode = "pixelated_scan"
        elif r < 0.44:
            mode = "ink_bleed_blur"
        elif r < 0.62:
            mode = "gaussian_blur_noise"
        elif r < 0.72:
            mode = "light_blur_jpeg"
        else:
            mode = "clean"

    if mode == "clean":
        if rng.random() < 0.35:
            return canvas.filter(ImageFilter.GaussianBlur(radius=0.30)), "clean"
        return canvas, "clean"

    out = canvas.copy()

    if mode == "pixelated_scan":
        # Simulate low-DPI scan (0.45x..0.64x), 1-bit/low-bit thresholding with staircase blocks,
        # nearest-neighbor upscale back to exact (w, h), and optional dashed scan border noise.
        scale = rng.uniform(0.45, 0.64)
        sw = max(16, int(round(w * scale)))
        sh = max(12, int(round(h * scale)))
        small = out.convert("L").resize((sw, sh), Image.Resampling.BILINEAR)
        s_arr = np.asarray(small, dtype=np.float32)
        # Add slight sensor jitter before thresholding so strokes have realistic jagged teeth
        jitter = np.array(
            [[rng.uniform(-14.0, 14.0) for _ in range(sw)] for _ in range(sh)],
            dtype=np.float32,
        )
        thresh = rng.uniform(148.0, 188.0)
        bin_arr = np.where((s_arr + jitter) < thresh, rng.randint(5, 28), rng.randint(245, 255)).astype(np.uint8)
        up_l = Image.fromarray(bin_arr, mode="L").resize(
            (w, h),
            Image.Resampling.NEAREST if rng.random() < 0.70 else Image.Resampling.BILINEAR,
        )
        out = up_l.convert("RGB")
        # Add JPEG ringing around blocky pixels
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=rng.randint(35, 68))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")

    elif mode == "ink_bleed_blur":
        # Simulate heavy ink bleed / thick blurred strokes where stacked '0's touch vertically
        gray = out.convert("L")
        if rng.random() < 0.65:
            gray = gray.filter(ImageFilter.MinFilter(size=3))
            # Blend original and dilated ink so strokes thicken by ~0.6..1.0px
            g_orig = np.asarray(out.convert("L"), dtype=np.float32)
            g_min = np.asarray(gray, dtype=np.float32)
            alpha = rng.uniform(0.45, 0.78)
            blended = np.clip(g_orig * (1.0 - alpha) + g_min * alpha, 0, 255).astype(np.uint8)
            gray = Image.fromarray(blended, mode="L")
        blur_r = rng.uniform(0.65, 1.35)
        gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_r))
        arr = np.asarray(gray, dtype=np.float32)
        noise = np.array(
            [[rng.gauss(0.0, rng.uniform(4.0, 10.0)) for _ in range(w)] for _ in range(h)],
            dtype=np.float32,
        )
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr, mode="L").convert("RGB")

    elif mode == "gaussian_blur_noise":
        # Out-of-focus optical blur + lower contrast + paper grain + pepper specks
        blur_r = rng.uniform(0.85, 1.65)
        blurred = out.convert("L").filter(ImageFilter.GaussianBlur(radius=blur_r))
        arr = np.asarray(blurred, dtype=np.float32)
        # Soften contrast slightly like a faded scan
        contrast_ink = rng.uniform(25.0, 75.0)
        contrast_bg = rng.uniform(226.0, 248.0)
        arr = contrast_ink + (arr / 255.0) * (contrast_bg - contrast_ink)
        sigma = rng.uniform(6.0, 14.0)
        noise = np.array(
            [[rng.gauss(0.0, sigma) for _ in range(w)] for _ in range(h)],
            dtype=np.float32,
        )
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        # Occasional tiny pepper specks near edges
        for _ in range(rng.randint(2, 8)):
            px = rng.randint(0, max(0, w - 2))
            py = rng.choice([rng.randint(0, min(4, h - 1)), rng.randint(max(0, h - 5), h - 1)])
            arr[py : min(h, py + rng.randint(1, 2)), px : min(w, px + rng.randint(1, 2))] = rng.randint(20, 90)
        out = Image.fromarray(arr, mode="L").convert("RGB")

    elif mode == "light_blur_jpeg":
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.45, 0.85)))
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=rng.randint(42, 75))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")

    return out, mode


def generate_one_sample(rng: random.Random, fonts: List[Path], idx: int) -> Dict[str, Any]:
    """
    Generate one CAD Blue-Box image and its exact 3-region (or 2-region / 1-region) YOLO labels (v5).
    Includes clean, noisy, blurred, ink-bleed, and pixelated 1-bit scan variations.
    """
    import math

    font_path = rng.choice(fonts)
    thin_candidates = [p for p in fonts if p.name.lower() in ("gothic.ttf", "osifont.ttf", "calibri.ttf")]
    tol_font_path = rng.choice(thin_candidates) if thin_candidates else font_path

    # Deterministic showcase at the top of preview_crops.html (idx 1..26):
    # (nom_text, up_text, low_text, div_mode, gap_x_override, gap_y, up_dx, low_dx, tol_dy, ul_mode, tilt_deg, sample_kind, noise_mode)
    forced_cases = [
        ("3", "+0.005", "0", "none", 6, -1, 0, 11, -3, "full_touch", 0.0, "stacked", "pixelated_scan"),
        ("1", "+0.1", "0", "none", 6, -1, 0, 11, -2, "none", 0.0, "stacked", "pixelated_scan"),
        ("0.845", "+0.005", "0", "none", 5, -1, 0, 11, -2, "full_touch", 0.0, "stacked", "pixelated_scan"),
        ("4", "0", "-0.005", "none", 6, -1, 11, 0, 0, "none", 0.0, "stacked", "ink_bleed_blur"),
        ("2.92", "0", "-0.001", "none", 5, -1, 11, 0, 0, "double_leader", 0.0, "stacked", "pixelated_scan"),
        ("0.65", "+0.005", "0", "none", 5, -1, 0, 11, -2, "full_touch", 0.0, "stacked", "gaussian_blur_noise"),
        ("0.555", "+0.001", "+0.002", "none", 8, 3, 0, 0, 0, "none", 0.0, "stacked", "clean"),
        ("0.555", "+0.001", "+0.002", "none", 0, 0, 0, 0, 0, "none", 0.0, "stacked", "ink_bleed_blur"),
        ("1", "-0.1", "+0", "none", 7, 3, 0, 0, 0, "none", 0.0, "stacked", "gaussian_blur_noise"),
        ("1", "-0.1", "+0", "none", 0, 0, 0, 0, 0, "none", 0.0, "stacked", "pixelated_scan"),
        ("0.555", "±0.001", "", "none", 0, 0, 0, 0, 0, "none", 0.0, "symmetric", "gaussian_blur_noise"),
        ("128.555", "+0.001", "-0.002", "none", 0, 0, 0, 0, 0, "none", 0.0, "stacked", "ink_bleed_blur"),
        ("0.555", "+0.001", "+0.002", "none", 46, 3, 0, 0, 0, "none", 0.0, "stacked", "gaussian_blur_noise"),
        ("1", "-0.1", "+0", "none", 50, 3, 0, 0, 0, "none", 0.0, "stacked", "clean"),
        ("4.94", "-0.003", "-0.008", "none", 8, 3, 0, 0, 0, "double_leader", 0.0, "stacked", "ink_bleed_blur"),
        ("1", "-0.1", "+0", "none", 0, 0, 0, 0, 0, "full_touch", 0.0, "stacked", "pixelated_scan"),
        ("3", "0", "-0.002", "none", 6, 3, 12, 0, -9, "tol_only", 0.0, "stacked", "gaussian_blur_noise"),
        ("5.4", "0", "-0.002", "none", 8, 3, 13, 0, +8, "full_touch", 0.0, "stacked", "ink_bleed_blur"),
        ("0.555", "+0.001", "-0.002", "none", 1, 3, 18, 0, 0, "none", 0.0, "stacked", "clean"),
        ("1", "-0.1", "+0", "none", 0, 3, 0, 20, -6, "none", 0.0, "stacked", "pixelated_scan"),
        ("0.555", "+0.001", "-0.002", "thick", 6, 6, 0, 0, 0, "none", 0.0, "stacked", "gaussian_blur_noise"),
        ("0.555", "+0.001", "-0.02", "thick_ext", 2, 6, 16, 0, 0, "none", 0.0, "stacked", "ink_bleed_blur"),
        ("1.00", "+0.01", "-0.02", "thick", 0, 5, 0, 12, 0, "none", 0.0, "stacked", "gaussian_blur_noise"),
        ("1.00", "+0", "-0.05", "none", 0, 0, 14, 0, +5, "full_touch", 0.0, "stacked", "pixelated_scan"),
        ("R0.02", "", "", "none", 0, 0, 0, 0, 0, "parallel_diag", 38.0, "single_angled", "gaussian_blur_noise"),
        ("C0.02", "", "", "none", 0, 0, 0, 0, 0, "parallel_diag", -45.0, "single_angled", "pixelated_scan"),
    ]

    nom_size = rng.randint(30, 42)
    # Small, thin-stroke tolerance font (38%..48% of nominal height)
    tol_scale = rng.uniform(0.38, 0.48) if rng.random() < 0.78 else rng.uniform(0.48, 0.56)
    tol_size = max(13, int(round(nom_size * tol_scale)))
    stretch_x = rng.uniform(0.92, 1.08)
    bold = rng.random() < 0.25
    char_spacing = rng.choice([0, 0, 1, 2])
    ink_val = rng.randint(10, 45)
    bg_val = rng.randint(248, 255)

    margin_x = rng.randint(10, 22)
    margin_y = rng.randint(10, 20)

    post_dot_px = 0
    tri_prefix = False
    top_trapezoid = False

    noise_mode = "auto"
    if idx <= len(forced_cases):
        (
            nom_text,
            up_text,
            low_text,
            div_mode,
            gap_x_override,
            gap_y,
            up_dx,
            low_dx,
            tol_dy,
            ul_mode,
            tilt_deg,
            sample_kind,
            noise_mode,
        ) = forced_cases[idx - 1]
    else:
        r_val = rng.random()
        post_dot_px = rng.randint(6, 15) if rng.random() < 0.22 else rng.choice([0, 0, 1, 2])
        tri_prefix = rng.random() < 0.10
        top_trapezoid = (not tri_prefix) and (rng.random() < 0.08)

        if r_val < 0.78:
            sample_kind = "stacked"
            if rng.random() < 0.30 and not tri_prefix:
                nom_text = rng.choice(SINGLE_DIGIT_NOMINALS)
            else:
                nom_text = rng.choice(MULTI_NOMINAL_POOL)
            up_text, low_text = rng.choice(TOL_POOL)

            # Divider bar between tol_up and tol_lower (including thick 3-4px middle underline)
            div_roll = rng.random()
            if div_roll < 0.16:
                div_mode = "thick"
            elif div_roll < 0.26:
                div_mode = "thick_ext"
            elif div_roll < 0.36:
                div_mode = "thin"
            else:
                div_mode = "none"

            if div_mode in ("thick", "thick_ext"):
                gap_y = rng.randint(5, 7)
            elif rng.random() < 0.45:
                gap_y = rng.choice([-1, 0, 0, 1])
            else:
                gap_y = rng.randint(2, 5)

            # Horizontal gap: 32% ultra-tight/touching (0..2px), 43% normal (3..14px), 25% wide (26..54px)
            gx_roll = rng.random()
            if gx_roll < 0.32:
                gap_x_override = rng.choice([0, 0, 1, 2])
            elif gx_roll < 0.75:
                gap_x_override = rng.randint(3, 14)
            else:
                gap_x_override = rng.randint(26, 54)

            # Staggered X-axis ("cái thụt cái thò")
            stagger_x_roll = rng.random()
            if up_text == "0" and len(low_text) >= 3:
                up_dx = rng.randint(8, 16)
                low_dx = 0
            elif low_text == "0" and len(up_text) >= 3:
                up_dx = 0
                low_dx = rng.randint(8, 16)
            elif stagger_x_roll < 0.22:
                up_dx = rng.randint(8, 22)
                low_dx = 0
            elif stagger_x_roll < 0.44:
                up_dx = 0
                low_dx = rng.randint(8, 22)
            else:
                up_dx = rng.randint(0, 2)
                low_dx = rng.randint(0, 2)

            # Staggered Y-axis ("không cùng trục Y")
            stagger_y_roll = rng.random()
            if stagger_y_roll < 0.24:
                tol_dy = rng.randint(-11, -5)  # Tol stack shifted higher than nominal center
            elif stagger_y_roll < 0.48:
                tol_dy = rng.randint(5, 11)    # Tol stack shifted lower than nominal center
            else:
                tol_dy = rng.randint(-2, 2)

            # Underline mode
            ul_roll = rng.random()
            if ul_roll < 0.16:
                ul_mode = "nom_touch"
            elif ul_roll < 0.34:
                ul_mode = "full_touch"
            elif ul_roll < 0.48:
                ul_mode = "double_leader"
            elif ul_roll < 0.58:
                ul_mode = "tol_only"
            else:
                ul_mode = "none"
            tilt_deg = 0.0

        elif r_val < 0.89:
            sample_kind = "symmetric"
            nom_text = rng.choice(MULTI_NOMINAL_POOL)
            up_text = rng.choice(SYM_TOL_POOL)
            low_text = ""
            div_mode = "none"
            gap_y = 0
            gx_roll = rng.random()
            if gx_roll < 0.40:
                gap_x_override = rng.choice([0, 0, 1, 2])
            elif gx_roll < 0.80:
                gap_x_override = rng.randint(3, 12)
            else:
                gap_x_override = rng.randint(24, 46)
            up_dx = 0
            low_dx = 0
            tol_dy = rng.randint(-5, 5)
            ul_mode = rng.choice(["none", "full_touch", "double_leader", "nom_touch"])
            tilt_deg = 0.0

        else:
            sample_kind = "single_angled"
            nom_text = rng.choice(ANGLED_RC_POOL)
            up_text = ""
            low_text = ""
            div_mode = "none"
            gap_x_override = 0
            gap_y = 0
            up_dx = 0
            low_dx = 0
            tol_dy = 0
            ul_mode = "parallel_diag"
            tilt_deg = rng.choice([rng.uniform(26.0, 48.0), rng.uniform(-48.0, -26.0)])

    nom_mask = render_tight_glyph_mask(
        nom_text,
        font_path,
        nom_size,
        stretch_x=stretch_x,
        bold=bold,
        char_spacing=char_spacing,
        post_dot_space=post_dot_px,
        has_triangle_prefix=tri_prefix,
    )
    display_nom_text = f"△{nom_text}" if tri_prefix and not nom_text.startswith("△") else nom_text
    top_extra = 14 if top_trapezoid else 0

    if sample_kind == "stacked":
        tol_post_dot = max(0, int(post_dot_px * 0.65))
        up_mask = render_tight_glyph_mask(
            up_text, tol_font_path, tol_size, stretch_x=stretch_x, bold=False, post_dot_space=tol_post_dot
        )
        low_mask = render_tight_glyph_mask(
            low_text, tol_font_path, tol_size, stretch_x=stretch_x, bold=False, post_dot_space=tol_post_dot
        )

        gap_x = int(gap_x_override)
        stack_w = max(up_dx + up_mask.width, low_dx + low_mask.width)
        stack_h = max(up_mask.height + low_mask.height - 1, up_mask.height + gap_y + low_mask.height)

        extra_y_span = abs(tol_dy) * 2 + 8
        content_h = max(nom_mask.height, stack_h) + extra_y_span
        right_extra = rng.randint(10, 24)
        canvas_w = margin_x * 2 + nom_mask.width + max(0, gap_x) + stack_w + right_extra
        canvas_h = margin_y * 2 + content_h + top_extra

        canvas = Image.new("RGB", (canvas_w, canvas_h), (bg_val, bg_val, bg_val))

        nom_x = margin_x
        center_y = margin_y + top_extra + content_h // 2
        nom_y = center_y - nom_mask.height // 2
        nom_y = max(3 + top_extra, min(canvas_h - nom_mask.height - 4, nom_y))

        if top_trapezoid:
            draw_trapezoid_top_tag(canvas, nom_x + nom_mask.width // 2, max(1, nom_y - 13), ink_val=ink_val)

        stack_x = nom_x + nom_mask.width + gap_x
        up_x = max(nom_x + 4, stack_x + up_dx)
        low_x = max(nom_x + 4, stack_x + low_dx)

        up_y = center_y - stack_h // 2 + tol_dy
        up_y = max(2 + top_extra, min(canvas_h - stack_h - 4, up_y))
        low_y = up_y + up_mask.height + gap_y

        raw_nom_box = paste_ink_mask(canvas, nom_mask, nom_x, nom_y, ink_val=ink_val)
        raw_up_box = paste_ink_mask(canvas, up_mask, up_x, up_y, ink_val=ink_val)
        raw_low_box = paste_ink_mask(canvas, low_mask, low_x, low_y, ink_val=ink_val)

        mid_y_split = (raw_up_box[1] + raw_up_box[3] + raw_low_box[1]) // 2
        draw = ImageDraw.Draw(canvas)

        if div_mode in ("thin", "thick", "thick_ext"):
            bar_y = mid_y_split
            bar_w = rng.choice([3, 4]) if "thick" in div_mode else rng.choice([1, 2])
            bar_x0 = (nom_x + nom_mask.width - 2) if div_mode == "thick_ext" else (min(up_x, low_x) - rng.randint(2, 5))
            bar_x1 = max(raw_up_box[0] + raw_up_box[2], raw_low_box[0] + raw_low_box[2]) + rng.randint(4, 16)
            draw.line([(bar_x0, bar_y), (min(canvas_w - 2, bar_x1), bar_y)], fill=(ink_val, ink_val, ink_val), width=bar_w)

        nom_box = expand_box_safely(
            raw_nom_box, canvas_w, canvas_h, pad_x=2, pad_y=2, y_min_limit=max(0, raw_nom_box[1] - 2)
        )
        # Ensure clean non-overlapping seam when gap_x == 0
        min_tol_left = min(raw_up_box[0], raw_low_box[0])
        if nom_box[0] + nom_box[2] > min_tol_left:
            nom_box[2] = max(4, min_tol_left - nom_box[0])

        up_box = expand_box_safely(
            raw_up_box, canvas_w, canvas_h, pad_x=2, pad_y=1, y_min_limit=0, y_max_limit=mid_y_split
        )
        low_box = expand_box_safely(
            raw_low_box, canvas_w, canvas_h, pad_x=2, pad_y=1, y_min_limit=mid_y_split, y_max_limit=canvas_h
        )

        # Draw requested underline / dimension line (strictly outside or touching the boundary, excluded from GT boxes)
        if ul_mode == "nom_touch":
            uy = min(canvas_h - 2, nom_box[1] + nom_box[3] + rng.randint(0, 1))
            draw.line([(max(1, nom_box[0] - 5), uy), (min(canvas_w - 2, nom_box[0] + nom_box[2] + 6), uy)], fill=(ink_val, ink_val, ink_val), width=rng.choice([2, 3]))
        elif ul_mode == "tol_only":
            uy = min(canvas_h - 2, low_box[1] + low_box[3] + rng.randint(0, 2))
            draw.line([(max(1, low_box[0] - 2), uy), (min(canvas_w - 2, low_box[0] + low_box[2] + 8), uy)], fill=(ink_val, ink_val, ink_val), width=2)
        elif ul_mode == "full_touch":
            uy = min(canvas_h - 2, max(nom_box[1] + nom_box[3], low_box[1] + low_box[3]) + rng.randint(0, 2))
            rx1 = max(up_box[0] + up_box[2], low_box[0] + low_box[2])
            draw.line([(max(1, nom_box[0] - 6), uy), (min(canvas_w - 2, rx1 + 8), uy)], fill=(ink_val, ink_val, ink_val), width=rng.choice([2, 3]))
        elif ul_mode == "double_leader":
            uy = min(canvas_h - 3, max(nom_box[1] + nom_box[3], low_box[1] + low_box[3]) + rng.randint(1, 3))
            draw.line([(2, uy), (canvas_w - 2, uy)], fill=(ink_val, ink_val, ink_val), width=2)
            ax_l = max(14, nom_box[0] - 4)
            ax_r = min(canvas_w - 14, low_box[0] + 10)
            draw.line([(ax_l - 10, max(0, uy - 4)), (ax_l, uy), (ax_l - 10, min(canvas_h - 1, uy + 4))], fill=(ink_val, ink_val, ink_val), width=2)
            draw.line([(ax_r + 10, max(0, uy - 4)), (ax_r, uy), (ax_r + 10, min(canvas_h - 1, uy + 4))], fill=(ink_val, ink_val, ink_val), width=2)

        regions = [
            {"classId": 0, "className": "nominal", "box": nom_box, "text": display_nom_text},
            {"classId": 1, "className": "tol_up", "box": up_box, "text": up_text},
            {"classId": 2, "className": "tol_lower", "box": low_box, "text": low_text},
        ]

    elif sample_kind == "symmetric":
        tol_post_dot = max(0, int(post_dot_px * 0.65))
        sym_mask = render_tight_glyph_mask(
            up_text, tol_font_path, max(16, int(nom_size * 0.56)), stretch_x=stretch_x, bold=False, post_dot_space=tol_post_dot
        )
        gap_x = int(gap_x_override)
        content_h = max(nom_mask.height, sym_mask.height + abs(tol_dy) * 2 + 8)
        canvas_w = margin_x * 2 + nom_mask.width + max(0, gap_x) + sym_mask.width + rng.randint(8, 18)
        canvas_h = margin_y * 2 + content_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), (bg_val, bg_val, bg_val))

        nom_x = margin_x
        nom_y = margin_y + (content_h - nom_mask.height) // 2
        sym_x = nom_x + nom_mask.width + gap_x
        sym_y = max(2, min(canvas_h - sym_mask.height - 3, nom_y + (nom_mask.height - sym_mask.height) // 2 + tol_dy))

        raw_nom_box = paste_ink_mask(canvas, nom_mask, nom_x, nom_y, ink_val=ink_val)
        raw_sym_box = paste_ink_mask(canvas, sym_mask, sym_x, sym_y, ink_val=ink_val)
        draw = ImageDraw.Draw(canvas)

        nom_box = expand_box_safely(raw_nom_box, canvas_w, canvas_h, pad_x=2, pad_y=2)
        if nom_box[0] + nom_box[2] > raw_sym_box[0]:
            nom_box[2] = max(4, raw_sym_box[0] - nom_box[0])
        sym_box = expand_box_safely(raw_sym_box, canvas_w, canvas_h, pad_x=2, pad_y=2)

        if ul_mode in ("full_touch", "double_leader", "nom_touch"):
            ul_y = min(canvas_h - 3, max(nom_box[1] + nom_box[3], sym_box[1] + sym_box[3]) + rng.randint(1, 3))
            draw.line([(2, ul_y), (canvas_w - 2, ul_y)], fill=(ink_val, ink_val, ink_val), width=2)

        regions = [
            {"classId": 0, "className": "nominal", "box": nom_box, "text": display_nom_text},
            {"classId": 1, "className": "tol_up", "box": sym_box, "text": up_text},
        ]

    else:
        # single_angled: R0.02 / C0.02 / R0.49 / C0.5 rotated at ±26°..48° with parallel diagonal leader line
        pad_diag = max(nom_mask.width, nom_mask.height) + 32
        canvas_w = pad_diag + 24
        canvas_h = pad_diag + 16
        text_layer = Image.new("L", (canvas_w, canvas_h), 0)
        line_layer = Image.new("L", (canvas_w, canvas_h), 0)

        nx = (canvas_w - nom_mask.width) // 2
        ny = (canvas_h - nom_mask.height) // 2 - 4
        text_layer.paste(nom_mask, (nx, ny), nom_mask)

        ld = ImageDraw.Draw(line_layer)
        uy = ny + nom_mask.height + rng.randint(4, 7)
        ld.line([(nx - 18, uy), (nx + nom_mask.width + 18, uy)], fill=255, width=rng.choice([2, 3]))
        if rng.random() < 0.55:
            ld.line([(nx - 14, ny - 12), (nx - 1, ny + 2)], fill=255, width=2)

        cx_rot = canvas_w / 2.0
        cy_rot = canvas_h / 2.0
        text_rot = text_layer.rotate(tilt_deg, resample=Image.Resampling.BICUBIC, center=(cx_rot, cy_rot))
        line_rot = line_layer.rotate(tilt_deg, resample=Image.Resampling.BICUBIC, center=(cx_rot, cy_rot))

        t_arr = np.asarray(text_rot, dtype=np.uint8)
        ys_t, xs_t = np.where(t_arr > 28)
        if len(xs_t) > 0:
            bx0, by0 = max(0, int(xs_t.min()) - 2), max(0, int(ys_t.min()) - 2)
            bx1, by1 = min(canvas_w, int(xs_t.max()) + 3), min(canvas_h, int(ys_t.max()) + 3)
            nom_box = [bx0, by0, max(4, bx1 - bx0), max(4, by1 - by0)]
        else:
            nom_box = [nx, ny, nom_mask.width, nom_mask.height]

        combined_l = np.maximum(t_arr, np.asarray(line_rot, dtype=np.uint8))
        canvas = Image.new("RGB", (canvas_w, canvas_h), (bg_val, bg_val, bg_val))
        c_arr = np.asarray(canvas, dtype=np.uint8).copy()
        alpha = (combined_l.astype(np.float32) / 255.0) * ((255 - ink_val) / 255.0)
        for ch_i in range(3):
            c_arr[:, :, ch_i] = np.clip(c_arr[:, :, ch_i] * (1.0 - alpha), 0, 255).astype(np.uint8)
        canvas = Image.fromarray(c_arr)

        regions = [
            {"classId": 0, "className": "nominal", "box": nom_box, "text": display_nom_text},
        ]

    canvas, applied_noise = apply_cad_noise_and_blur(canvas, rng, mode=noise_mode)

    return {
        "id": idx,
        "name": f"cad_box_{idx:04d}",
        "isStacked": sample_kind == "stacked",
        "font": f"{font_path.name} [{applied_noise}]",
        "image": canvas,
        "width": canvas_w,
        "height": canvas_h,
        "regions": regions,
    }


def _resolve_ref_page(page_key: str, cache_folder_prefix: str) -> Optional[Path]:
    ref_p = BASE_DIR / "uploads" / "ref_pages" / f"{page_key}.jpg"
    if ref_p.is_file():
        return ref_p
    cache_dir = WORKSPACE_ROOT.parent / "ocrtool-pdfscan-render-cache"
    if cache_dir.is_dir():
        for d in sorted(cache_dir.glob(f"{cache_folder_prefix}*")):
            cand = d / "page-001.jpg"
            if cand.is_file():
                return cand
    return None


def extract_real_hard_samples(start_idx: int) -> List[Dict[str, Any]]:
    """
    Extract real CAD hard crops across:
      - 9 real noisy/pixelated scan crops from real_noisy_box_*.png (with 8x variations = 72 samples)
      - 5 real user crops in uploads/real_hard_v4/ (with 8x variations = 40 samples)
      - 30 real CAD hard crops across 105-M0800, 108-M1063, and 133-MF00307 (with 5x variations = 150 samples)
    """
    from engine import YOLODetectorONNX, extract_crop_with_context, YOLO_ONNX_DIR
    from preprocessor import clean_leader_lines

    out_samples: List[Dict[str, Any]] = []
    cur_idx = start_idx
    rng_real = random.Random(20261005)

    # 1. Include the 9 real noisy/pixelated scan crops + 5 real user hard crops in uploads/real_hard_v4/
    v4_dir = BASE_DIR / "uploads" / "real_hard_v4"
    v4_specs = [
        ("real_noisy_box_44.png", False, [[14, 17, 15, 29], [31, 13, 61, 20], [42, 29, 12, 17]], ["3", "+0.005", "0"]),
        ("real_noisy_box_45.png", False, [[11, 14, 9, 28], [25, 10, 39, 20], [36, 26, 12, 17]], ["1", "+0.1", "0"]),
        ("real_noisy_box_6.png", False, [[22, 20, 86, 30], [112, 17, 59, 20], [122, 33, 12, 17]], ["0.845", "+0.005", "0"]),
        ("real_noisy_box_10.png", False, [[14, 12, 18, 29], [45, 9, 12, 16], [34, 24, 60, 21]], ["4", "0", "-0.005"]),
        ("real_noisy_box_18.png", False, [[13, 21, 42, 29], [58, 18, 61, 20], [69, 34, 12, 17]], ["1.6", "+0.005", "0"]),
        ("real_noisy_box_21.png", False, [[11, 17, 67, 30], [92, 15, 13, 16], [80, 30, 60, 20]], ["2.92", "0", "-0.001"]),
        ("real_noisy_box_22.png", False, [[8, 15, 66, 30], [88, 12, 12, 16], [77, 27, 58, 20]], ["2.82", "0", "-0.001"]),
        ("real_noisy_box_24.png", False, [[10, 14, 65, 29], [78, 10, 61, 20], [89, 26, 12, 17]], ["0.34", "+0.005", "0"]),
        ("real_noisy_box_26.png", False, [[12, 16, 66, 30], [80, 13, 60, 20], [90, 29, 12, 17]], ["0.65", "+0.005", "0"]),
        ("real_5_4_0_-0002.jpg", False, [[33, 22, 52, 27], [110, 19, 10, 14], [97, 36, 62, 14]], ["5.4", "0", "-0.002"]),
        ("real_4_94_-0003_-0008.jpg", False, [[28, 21, 81, 25], [121, 17, 61, 15], [121, 35, 61, 11]], ["4.94", "-0.003", "-0.008"]),
        ("real_3_0_-0002_vert.jpg", True, [[9, 15, 17, 27], [44, 12, 10, 13], [32, 29, 61, 15]], ["3", "0", "-0.002"]),
        ("real_C0_5_tilted.jpg", False, [[28, 14, 70, 68]], ["C0.5"]),
        ("real_R0_49_tilted.jpg", False, [[28, 20, 93, 84]], ["R0.49"]),
    ]
    for fname, rot_cw, box_coords, box_texts in v4_specs:
        fp = v4_dir / fname
        if not fp.is_file():
            continue
        base_crop = Image.open(fp).convert("RGB")
        if rot_cw:
            base_crop = base_crop.rotate(-90, expand=True)
        cleaned_crop = clean_leader_lines(base_crop, is_sub_crop=False)
        blurred_crop = base_crop.filter(ImageFilter.GaussianBlur(radius=0.85))
        noisy_crop, _ = apply_cad_noise_and_blur(base_crop, rng_real, mode="gaussian_blur_noise")
        variants = [
            (base_crop, 0),
            (cleaned_crop, 0),
            (blurred_crop, 0),
            (noisy_crop, 0),
            (ImageOps.expand(base_crop, border=4, fill=(255, 255, 255)), 4),
            (ImageOps.expand(blurred_crop, border=4, fill=(255, 255, 255)), 4),
            (ImageOps.expand(base_crop, border=8, fill=(255, 255, 255)), 8),
            (ImageOps.expand(noisy_crop, border=10, fill=(252, 252, 252)), 10),
        ]
        for v_img, pad_off in variants:
            vw, vh = v_img.size
            v_regs = []
            for cid_idx, (bx, by, bw, bh) in enumerate(box_coords):
                v_regs.append(
                    {
                        "classId": cid_idx,
                        "className": CLASSES[cid_idx],
                        "box": [bx + pad_off, by + pad_off, bw, bh],
                        "text": box_texts[cid_idx],
                    }
                )
            out_samples.append(
                {
                    "id": cur_idx,
                    "name": f"real_cad_v5_{Path(fname).stem}_{cur_idx:04d}",
                    "isStacked": len(box_coords) == 3,
                    "font": "real_noisy_v5" if "noisy" in fname else "real_user_v5",
                    "image": v_img,
                    "width": vw,
                    "height": vh,
                    "regions": v_regs,
                }
            )
            cur_idx += 1

    y1 = YOLO_ONNX_DIR / "yolo_best_1class.onnx"
    if not y1.is_file():
        return out_samples

    page_groups = [
        (
            "105",
            _resolve_ref_page("page_105", "105-M0800"),
            [
                (4, False, [[8, 20, 118, 35], [128, 20, 80, 18], [128, 38, 80, 19]], ["6.120", "+0.001", "-0.001"]),
                (6, False, [[6, 17, 119, 34], [127, 17, 80, 17], [142, 33, 24, 19]], ["2.600", "+0.005", "0"]),
                (13, True, [[12, 18, 85, 34], [102, 18, 36, 17], [102, 34, 55, 19]], ["5.31", "+0", "-0.1"]),
                (14, True, [[11, 16, 112, 33], [127, 16, 80, 17], [127, 32, 80, 19]], ["4.810", "+0.001", "-0.001"]),
                (15, False, [[4, 17, 110, 34], [132, 18, 22, 16], [118, 32, 82, 18]], ["1.400", "0", "-0.005"]),
                (16, False, [[10, 15, 110, 32], [138, 16, 22, 16], [124, 30, 80, 18]], ["3.410", "0", "-0.002"]),
                (20, False, [[11, 9, 110, 31], [140, 10, 22, 15], [126, 23, 80, 17]], ["3.480", "0", "-0.002"]),
                (23, True, [[12, 18, 86, 31], [103, 17, 68, 16], [103, 33, 68, 16]], ["3.50", "-0.01", "-0.02"]),
                (25, True, [[8, 16, 110, 32], [123, 16, 68, 17], [137, 31, 22, 17]], ["17.20", "+0.02", "0"]),
                (26, True, [[11, 17, 112, 32], [140, 18, 22, 15], [126, 31, 84, 18]], ["2.460", "0", "-0.002"]),
                (27, True, [[4, 20, 110, 32], [118, 20, 82, 18], [133, 36, 22, 17]], ["1.500", "+0.002", "0"]),
                (28, True, [[10, 19, 112, 32], [125, 19, 82, 18], [140, 36, 22, 16]], ["1.500", "+0.002", "0"]),
                (29, True, [[9, 15, 112, 32], [139, 16, 22, 15], [125, 29, 84, 18]], ["2.460", "0", "-0.002"]),
                (34, True, [[10, 17, 88, 33], [103, 17, 66, 17], [117, 33, 22, 16]], ["0.80", "+0.05", "0"]),
                (36, False, [[14, 18, 86, 33], [105, 18, 66, 17], [119, 34, 22, 16]], ["4.00", "+0.02", "0"]),
            ],
        ),
        (
            "108",
            _resolve_ref_page("page_108", "108-M1063"),
            [
                (1, False, [[10, 19, 155, 35], [176, 17, 22, 20], [162, 37, 65, 19]], ["△32.27", "0", "-0.1"]),
                (6, False, [[12, 11, 122, 34], [151, 9, 18, 19], [135, 28, 100, 20]], ["4.000", "0", "-0.005"]),
                (10, True, [[8, 18, 124, 32], [133, 27, 94, 22]], ["6.250", "±0.001"]),
                (12, True, [[8, 17, 97, 35], [123, 15, 20, 20], [107, 35, 66, 20]], ["7.05", "0", "-0.1"]),
                (15, True, [[2, 22, 141, 34], [160, 19, 22, 20], [144, 38, 104, 21]], ["△3.450", "0", "-0.002"]),
                (19, True, [[6, 22, 153, 35], [177, 21, 22, 20], [160, 40, 104, 21]], ["△3.500", "0", "-0.002"]),
                (27, True, [[10, 21, 123, 34], [135, 19, 108, 21], [151, 39, 22, 20]], ["4.000", "+0.005", "0"]),
                (30, False, [[10, 16, 147, 35], [174, 15, 22, 20], [158, 34, 103, 21]], ["35.770", "0", "-0.002"]),
                (37, False, [[0, 13, 116, 34], [135, 11, 19, 20], [118, 30, 83, 20]], ["25.00", "0", "-0.02"]),
                (39, False, [[4, 18, 147, 35], [152, 29, 98, 22]], ["13.880", "±0.001"]),
            ],
        ),
        (
            "133",
            _resolve_ref_page("page_133", "133-MF00307"),
            [
                (57, False, [[16, 26, 38, 49], [47, 23, 106, 27], [68, 51, 30, 26]], ["5", "+0.005", "0"]),
                (58, False, [[8, 30, 26, 43], [40, 25, 102, 26], [40, 51, 102, 25]], ["6", "-0.002", "-0.005"]),
                (45, True, [[12, 30, 38, 49], [46, 27, 106, 27], [63, 55, 30, 27]], ["6", "+0.005", "0"]),
                (39, False, [[6, 27, 152, 50], [158, 23, 104, 27], [176, 51, 28, 26]], ["DP7.0", "+0.005", "0"]),
                (35, True, [[5, 28, 128, 50], [154, 25, 30, 27], [138, 52, 100, 26]], ["18.2", "0", "-0.001"]),
            ],
        ),
    ]

    for tag, img_path, specs in page_groups:
        if img_path is None or not img_path.is_file():
            continue
        page_img = Image.open(img_path).convert("RGB")
        det = YOLODetectorONNX.detect(page_img, y1, conf_thres=0.30)
        by_id = {b["id"]: b for b in det["boxes"]}

        for bid, rot_cw, box_coords, box_texts in specs:
            if bid not in by_id:
                continue
            b = by_id[bid]
            base_crop = extract_crop_with_context(page_img, b["x"], b["y"], b["w"], b["h"], context_pad=4)
            if rot_cw:
                base_crop = base_crop.rotate(-90, expand=True)

            cleaned_crop = clean_leader_lines(base_crop, is_sub_crop=False)
            degraded_crop, _ = apply_cad_noise_and_blur(base_crop, rng_real, mode="pixelated_scan" if (bid % 2 == 0) else "ink_bleed_blur")
            variants = [
                (cleaned_crop, 0),
                (base_crop, 0),
                (degraded_crop, 0),
                (ImageOps.expand(cleaned_crop, border=4, fill=(255, 255, 255)), 4),
                (ImageOps.expand(degraded_crop, border=6, fill=(255, 255, 255)), 6),
            ]
            for v_img, pad_off in variants:
                vw, vh = v_img.size
                v_regs = []
                for cid_idx, (bx, by, bw, bh) in enumerate(box_coords):
                    v_regs.append(
                        {
                            "classId": cid_idx,
                            "className": CLASSES[cid_idx],
                            "box": [bx + pad_off, by + pad_off, bw, bh],
                            "text": box_texts[cid_idx],
                        }
                    )
                out_samples.append(
                    {
                        "id": cur_idx,
                        "name": f"real_cad_{tag}_{bid:02d}_{cur_idx:04d}",
                        "isStacked": len(box_coords) == 3,
                        "font": f"real_cad_{tag}",
                        "image": v_img,
                        "width": vw,
                        "height": vh,
                        "regions": v_regs,
                    }
                )
                cur_idx += 1

    return out_samples


def box_to_yolo_line(class_id: int, box: List[int], img_w: int, img_h: int) -> str:
    x, y, w, h = box
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / float(img_w)
    nh = h / float(img_h)
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def yolo_line_to_pixel_box(line: str, img_w: int, img_h: int) -> Tuple[int, List[int]]:
    parts = line.strip().split()
    cid = int(parts[0])
    cx, cy, nw, nh = map(float, parts[1:5])
    w = nw * img_w
    h = nh * img_h
    x = cx * img_w - w / 2.0
    y = cy * img_h - h / 2.0
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(img_w, int(round(x + w)))
    y1 = min(img_h, int(round(y + h)))
    return cid, [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def sharpen_crop_only(crop: Image.Image) -> Image.Image:
    upscaled = pad_and_upscale(crop, pad=6, scale=2.0)
    gray = ImageOps.autocontrast(upscaled.convert("L"), cutoff=0)
    sharp = gray.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=2)).convert("RGB")
    return sharp


def build_html_preview(
    records: List[Dict[str, Any]],
    out_html_path: Path,
    max_preview_items: int = 90,
) -> None:
    ppocr_v6_small = PPOCR_ONNX_DIR / "PP-OCRv6_small_rec.onnx"
    ppocr_v6_dict = PPOCR_ONNX_DIR / "ppocrv6_dict.txt"
    use_model = ppocr_v6_small if ppocr_v6_small.is_file() else (PPOCR_ONNX_DIR / "rec_cad_v6.onnx")
    use_dict = ppocr_v6_dict if ppocr_v6_small.is_file() else (PPOCR_ONNX_DIR / "cad_v6_dict.txt")

    color_map = {
        0: ("#38bdf8", "rgba(56, 189, 248, 0.18)", "0: nominal"),
        1: ("#c084fc", "rgba(192, 132, 252, 0.18)", "1: tol_up"),
        2: ("#f59e0b", "rgba(245, 158, 11, 0.18)", "2: tol_lower"),
    }

    # Show 1 cleaned variant per real CAD hard crop + first synthetic showcase items at top of preview_crops.html
    real_recs = [r for r in records if r["name"].startswith("real_cad_")][::4]
    synth_recs = [r for r in records if not r["name"].startswith("real_cad_")]
    preview_list = (real_recs + synth_recs)[:max_preview_items]

    cards_html: List[str] = []
    for rec in preview_list:
        img_path = DATASET_DIR / rec["imageRel"]
        lbl_path = DATASET_DIR / rec["labelRel"]
        clean_img = Image.open(img_path).convert("RGB")
        iw, ih = clean_img.size
        clean_uri = pil_to_data_uri(clean_img)

        txt_lines = [l.strip() for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        parsed_regions = []
        overlay_boxes_html = []

        for line in txt_lines:
            cid, (bx, by, bw, bh) = yolo_line_to_pixel_box(line, iw, ih)
            stroke_col, fill_col, cls_label = color_map.get(cid, ("#22c55e", "rgba(34,197,94,0.15)", f"{cid}"))

            left_pct = (bx / iw) * 100.0
            top_pct = (by / ih) * 100.0
            w_pct = (bw / iw) * 100.0
            h_pct = (bh / ih) * 100.0
            overlay_boxes_html.append(
                f'<div class="ov-box" style="left:{left_pct:.2f}%;top:{top_pct:.2f}%;width:{w_pct:.2f}%;height:{h_pct:.2f}%;'
                f'border-color:{stroke_col};background:{fill_col};" title="{cls_label}: [{bx},{by},{bw},{bh}]">'
                f'<span class="ov-tag" style="background:{stroke_col}">{cls_label}</span></div>'
            )

            sub_crop = clean_img.crop((bx, by, bx + bw, by + bh))
            sharp_sub = sharpen_crop_only(sub_crop)
            ocr_txt, ocr_conf = PPOCRRecognizerONNX.predict_single(sharp_sub, use_model, use_dict)

            gt_match = next((r["text"] for r in rec["regions"] if r["classId"] == cid), "")
            parsed_regions.append(
                {
                    "cid": cid,
                    "label": cls_label,
                    "color": stroke_col,
                    "box": [bx, by, bw, bh],
                    "cropUri": pil_to_data_uri(sharp_sub),
                    "gt": gt_match,
                    "ocr": ocr_txt,
                    "conf": round(ocr_conf * 100),
                }
            )

        sub_rows_html = []
        for pr in parsed_regions:
            sub_rows_html.append(
                f"""
                <div class="sub-crop-item" style="border-left: 3px solid {pr['color']}">
                  <div class="sub-meta">
                    <b style="color:{pr['color']}">{pr['label']}</b>
                    <span>Box: {pr['box']}</span>
                  </div>
                  <img class="sub-thumb" src="{pr['cropUri']}" alt="{pr['label']}" />
                  <div class="sub-ocr">
                    <div>GT: <b>{pr['gt']}</b></div>
                    <div>PP-OCRv6: <b style="color:#22c55e">{pr['ocr'] or '(trống)'}</b> <span class="conf">({pr['conf']}%)</span></div>
                  </div>
                </div>
                """
            )

        cards_html.append(
            f"""
            <div class="sample-card" data-kind="{'stacked' if rec['isStacked'] else 'single'}">
              <div class="sample-head">
                <b>#{rec['id']:03d} — {rec['name']} ({rec['font']})</b>
                <span class="split-pill">{rec['split'].upper()} | {len(parsed_regions)} vùng</span>
              </div>
              <div class="sample-body">
                <div class="stage-col">
                  <div class="stage-lbl">Ảnh gốc kiểm tra tọa độ (.txt overlay):</div>
                  <div class="stage-box" style="width:{iw * 2}px;height:{ih * 2}px">
                    <img src="{clean_uri}" style="width:100%;height:100%;display:block;image-rendering:auto;" />
                    {''.join(overlay_boxes_html)}
                  </div>
                  <div class="stage-lbl" style="margin-top:6px">Ảnh sạch đưa vào Train:</div>
                  <img src="{clean_uri}" style="max-height:48px;border:1px solid #334155;border-radius:4px;background:#fff;" />
                </div>
                <div class="crops-col">
                  <div class="stage-lbl">3 Vùng cắt rời & Quét riêng bằng {use_model.name}:</div>
                  {''.join(sub_rows_html)}
                </div>
              </div>
            </div>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <title>Preview Tọa Độ Chia 3 Vùng YOLO v3 (Dung Sai Dính Sát + Khoảng Cách Sau Dấu Chấm + Ký Hiệu ◬)</title>
  <style>
    body {{
      margin: 0;
      padding: 20px;
      background: #0f172a;
      color: #f1f5f9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    header {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      align-items: center;
      font-size: 13px;
      font-weight: 700;
    }}
    .dot {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      margin-right: 5px;
      vertical-align: middle;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
      gap: 16px;
    }}
    .sample-card {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 10px;
      overflow: hidden;
    }}
    .sample-head {{
      background: #263349;
      padding: 8px 14px;
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      border-bottom: 1px solid #334155;
    }}
    .split-pill {{
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
    }}
    .sample-body {{
      padding: 14px;
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 16px;
      align-items: start;
    }}
    .stage-lbl {{
      font-size: 11px;
      color: #94a3b8;
      margin-bottom: 6px;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .stage-box {{
      position: relative;
      border: 1px solid #475569;
      border-radius: 4px;
      background: #fff;
      overflow: visible;
    }}
    .ov-box {{
      position: absolute;
      border: 2px solid;
      box-sizing: border-box;
      pointer-events: auto;
    }}
    .ov-tag {{
      position: absolute;
      top: -15px;
      left: -1px;
      color: #0f172a;
      font-size: 9px;
      font-weight: 800;
      padding: 0 4px;
      border-radius: 2px;
      white-space: nowrap;
    }}
    .crops-col {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .sub-crop-item {{
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 6px 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    .sub-meta {{
      display: flex;
      flex-direction: column;
      font-size: 11px;
      min-width: 115px;
    }}
    .sub-meta span {{ color: #64748b; font-size: 10px; }}
    .sub-thumb {{
      height: 36px;
      max-width: 120px;
      background: #fff;
      border: 1px solid #334155;
      border-radius: 4px;
      padding: 2px;
      object-fit: contain;
    }}
    .sub-ocr {{
      font-family: Consolas, monospace;
      font-size: 12px;
      text-align: right;
      min-width: 150px;
    }}
    .conf {{ color: #94a3b8; font-size: 11px; }}
    button {{
      background: #334155;
      color: #fff;
      border: 1px solid #475569;
      border-radius: 6px;
      padding: 6px 12px;
      cursor: pointer;
      font-weight: 600;
    }}
    button.active {{ background: #38bdf8; color: #0f172a; border-color: #38bdf8; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h2 style="margin:0 0 6px 0;color:#38bdf8;">🎯 Kiểm Tra Tọa Độ Chia 3 Vùng Blue-Box v3 (Hard Cases 105-M0800 & 108-M1063)</h2>
      <div style="font-size:13px;color:#94a3b8;">
        Tổng số mẫu đã sinh: <b>{len(records)}</b> (Bao gồm 120 mẫu thật từ 105-M0800, 108-M1063, 133-MF00307 + mẫu Synthetic dung sai sát nhau & khoảng trắng sau dấu chấm)
      </div>
    </div>
    <div class="legend">
      <span><i class="dot" style="background:#38bdf8"></i>0: nominal</span>
      <span><i class="dot" style="background:#c084fc"></i>1: tol_up</span>
      <span><i class="dot" style="background:#f59e0b"></i>2: tol_lower</span>
      <button class="active" onclick="filterCards('all', this)">Tất cả</button>
      <button onclick="filterCards('stacked', this)">3 Vùng (Có dung sai)</button>
      <button onclick="filterCards('single', this)">1-2 Vùng (Nominal / ±)</button>
    </div>
  </header>
  <div class="grid" id="cardGrid">
    {''.join(cards_html)}
  </div>
  <script>
    function filterCards(kind, btn) {{
      document.querySelectorAll('header button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.sample-card').forEach(c => {{
        c.style.display = (kind === 'all' || c.dataset.kind === kind) ? '' : 'none';
      }});
    }}
  </script>
</body>
</html>
"""
    out_html_path.write_text(html, encoding="utf-8")


def generate_dataset(total_samples: int = 1240, seed: int = 20261005) -> Dict[str, Any]:
    rng = random.Random(seed)
    fonts = discover_fonts()

    # Preserve Export/ directory inside DATASET_DIR! Only clean train/ and val/
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        split_dir = DATASET_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    counts = {"train": 0, "val": 0, "stacked": 0, "single": 0, "realHard": 0}

    all_samples: List[Dict[str, Any]] = []
    for i in range(1, total_samples + 1):
        all_samples.append(generate_one_sample(rng, fonts, i))

    real_samples = extract_real_hard_samples(total_samples + 1)
    counts["realHard"] = len(real_samples)
    all_samples.extend(real_samples)

    for idx_s, sample in enumerate(all_samples, 1):
        split = "val" if (idx_s % 5 == 0) else "train"
        counts[split] += 1
        counts["stacked" if sample["isStacked"] else "single"] += 1

        img_rel = f"{split}/images/{sample['name']}.png"
        lbl_rel = f"{split}/labels/{sample['name']}.txt"

        sample["image"].save(DATASET_DIR / img_rel, format="PNG")
        yolo_lines = [
            box_to_yolo_line(r["classId"], r["box"], sample["width"], sample["height"])
            for r in sample["regions"]
        ]
        (DATASET_DIR / lbl_rel).write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        records.append(
            {
                "id": sample["id"],
                "name": sample["name"],
                "split": split,
                "isStacked": sample["isStacked"],
                "font": sample["font"],
                "width": sample["width"],
                "height": sample["height"],
                "imageRel": img_rel,
                "labelRel": lbl_rel,
                "regions": sample["regions"],
            }
        )

    data_yaml = {
        "path": DATASET_DIR.as_posix(),
        "train": "train/images",
        "val": "val/images",
        "names": CLASSES,
    }
    (DATASET_DIR / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (DATASET_DIR / "manifest.json").write_text(
        json.dumps({"counts": counts, "classes": CLASSES, "samples": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_html = DATASET_DIR / "preview_crops.html"
    build_html_preview(records, preview_html, max_preview_items=90)

    return {
        "datasetDir": str(DATASET_DIR),
        "previewHtml": str(preview_html),
        "counts": counts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1240)
    args = parser.parse_args()
    summary = generate_dataset(total_samples=args.count)
    print(json.dumps(summary, ensure_ascii=True))
