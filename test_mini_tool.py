"""
Automated verification tests for Mini PDF + YOLO (.onnx) + PP-OCR (.onnx) Tester.
Tests:
1. Model catalog discovery (two YOLO models, one OCR model and its dictionary).
2. CAD Image Preprocessor (leader-line cleanup, multi-pass generation, Otsu binarization, stacked tolerance split).
3. Full End-to-End PDF -> YOLO (.onnx) -> Multi-Pass Preprocessor -> PP-OCR (.onnx) pipeline.
"""
from __future__ import annotations

from pathlib import Path
import io
import json
import unittest
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from engine import (
    BASE_DIR,
    DEFAULT_PPOCR_MODEL,
    DEFAULT_3REGION_MODEL,
    PPOCR_ONNX_DIR,
    PPOCRRecognizerONNX,
    clean_cad_text,
    list_available_models,
    run_full_pipeline,
    run_ppocr_on_uploaded_result,
)
from training.finetune_yolo import build_finetune_command
from preprocessor import (
    build_ocr_passes,
    clean_leader_lines,
    split_stacked_tolerance_projection,
)


class TestMiniPdfOcrTool(unittest.TestCase):
    def test_01_model_catalog(self):
        catalog = list_available_models()
        self.assertEqual(
            {m["name"] for m in catalog["yoloOnnx"]},
            {DEFAULT_3REGION_MODEL, "yolo_best_1class071026.onnx"},
        )
        self.assertEqual(catalog["defaults"]["yoloModel"], "yolo_best_1class071026.onnx")
        self.assertEqual(catalog["defaults"]["subRegionModel"], DEFAULT_3REGION_MODEL)
        self.assertEqual([m["name"] for m in catalog["ppocrOnnx"]], [DEFAULT_PPOCR_MODEL])
        self.assertTrue(all("error" not in m for m in catalog["yoloOnnx"]))
        self.assertGreaterEqual(len(catalog["samplePdfs"]), 1)

    def test_02_preprocessor_leader_line_removal_and_passes(self):
        # Create a synthetic crop with a 1px horizontal leader line crossing a thick vertical stroke
        img = Image.new("RGB", (100, 40), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Thin 1px horizontal leader line across the entire width at y=20
        draw.line([(0, 20), (99, 20)], fill=(0, 0, 0), width=1)
        # Thick character block in the center
        draw.rectangle([42, 8, 56, 32], fill=(0, 0, 0))

        cleaned = clean_leader_lines(img)
        arr = np.asarray(cleaned.convert("L"))
        # The thin horizontal leader line at x=10, y=20 should be erased (255)
        self.assertEqual(int(arr[20, 10]), 255)
        # The thick character block at x=48, y=20 should be preserved (0)
        self.assertEqual(int(arr[20, 48]), 0)

        passes = build_ocr_passes(img, mode="auto")
        pass_names = [name for name, _ in passes]
        self.assertEqual(
            pass_names,
            ["leader_cleaned_tight", "leader_cleaned", "contrast_sharpen", "standard", "otsu_binary"],
        )

    def test_03_full_pipeline_on_sample_pdf(self):
        sample_pdf = BASE_DIR / "sample_pdfs" / "107-M1457.pdf"
        self.assertTrue(sample_pdf.is_file())

        for yolo_name in ("yolo_best_1class071026.onnx",):
            model_name = DEFAULT_PPOCR_MODEL
            result = run_full_pipeline(
                pdf_or_img_path=str(sample_pdf),
                yolo_model_name=yolo_name,
                ppocr_model_name=model_name,
                dict_name=None,  # test auto dictionary matching
                page_index=0,
                dpi=200,
                rotation=0,
                conf_thres=0.35,
                preprocess_mode="raw",
                clean_bg=True,
                enable_stack_split=True,
            )
            self.assertGreater(result["boxCount"], 5)
            self.assertEqual(result["yolo"]["model"], yolo_name)
            self.assertEqual(result["yolo"]["pageDetectorModel"], yolo_name)
            self.assertEqual(result["yolo"]["subRegionModel"], DEFAULT_3REGION_MODEL)
            self.assertEqual(result["ppocr"]["model"], DEFAULT_PPOCR_MODEL)
            non_empty_ocr = [b for b in result["boxes"] if b.get("ocrText")]
            self.assertGreater(len(non_empty_ocr), 5)
            print(
                f"\n[OK] Tested {sample_pdf.name} with {yolo_name} + {DEFAULT_3REGION_MODEL}: "
                f"{result['boxCount']} YOLO boxes, {len(non_empty_ocr)} OCR texts "
                f"(YOLO {result['yolo']['elapsedMs']}ms, PP-OCR {result['ppocr']['elapsedMs']}ms)."
            )
            for sample_box in non_empty_ocr[:6]:
                safe_txt = sample_box["ocrText"].encode("ascii", "replace").decode("ascii")
                print(
                    f"   #{sample_box['id']:02d}: '{safe_txt}' "
                    f"(conf={sample_box['ocrConfidence']:.2f}, pass={sample_box['passUsed']})"
                )

    def test_04_finetune_command_builder(self):
        cmd = build_finetune_command("best.pt", "ready-to-train-priority", epochs=50, imgsz=1024, batch=4, smoke_only=True)
        self.assertIn("--weights \"best.pt\"", cmd)
        self.assertIn("--smoke-only", cmd)

    def test_05_upload_result_ppocr_enrichment(self):
        sample_pdf = BASE_DIR / "sample_pdfs" / "107-M1457.pdf"
        mock_uploaded = {
            "fileName": "107-M1457.pdf",
            "page": {"pageIndex": 0, "dpi": 200, "rotation": 0},
            "boxes": [
                {"id": 1, "x": 1144, "y": 270, "w": 110, "h": 34},
            ],
        }
        enriched = run_ppocr_on_uploaded_result(
            result_data=mock_uploaded,
            pdf_or_img_path=str(sample_pdf),
            preprocess_mode="fast",
        )
        self.assertEqual(enriched["boxCount"], 1)
        self.assertEqual(enriched["ppocr"]["model"], DEFAULT_PPOCR_MODEL)
        self.assertIn("passDetails", enriched["boxes"][0])
        self.assertTrue(len(enriched["boxes"][0]["passDetails"]) >= 1)

    def test_06_no_digit_clipping_and_synthetic_3regions(self):
        # Use a labeled, fixed crop rather than the changing order of page detections.
        with zipfile.ZipFile(BASE_DIR / "training" / "yolo_3regions_finetune_pack.zip") as archive:
            prefix = "synthetic_3regions_dataset/"
            manifest = json.loads(archive.read(prefix + "manifest.json"))
            sample = next(s for s in manifest["samples"] if s["name"] == "cad_box_0105")
            region = next(r for r in sample["regions"] if r["classId"] == 0)
            self.assertEqual(region["text"], "2.3")
            x, y, w, h = region["box"]
            with Image.open(io.BytesIO(archive.read(prefix + sample["imageRel"]))) as image:
                result = PPOCRRecognizerONNX.recognize_crop(
                    image.crop((x, y, x + w, y + h)), PPOCR_ONNX_DIR / DEFAULT_PPOCR_MODEL,
                    preprocess_mode="raw", allow_rotate=False,
                )
            self.assertEqual(result["text"], region["text"])
            self.assertEqual(result["semantics"]["nominal"], region["text"])

    def test_07_symmetric_tolerance_whitespace(self):
        self.assertEqual(clean_cad_text("±0.005"), "±0.005")
        self.assertEqual(clean_cad_text("12.5 ±0.005"), "12.5 ±0.005")


if __name__ == "__main__":
    unittest.main()
