"""
Lightweight HTTP Server for the Mini PDF + YOLO (.onnx) + PP-OCR (.onnx) Tester
and .pt Fine-Tune Library Manager.
"""
from __future__ import annotations

import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import sys
import traceback

import fitz

from engine import (
    BASE_DIR,
    DEFAULT_PPOCR_MODEL,
    PPOCR_ONNX_DIR,
    PT_LIB_DIR,
    YOLO_ONNX_DIR,
    list_available_models,
    run_full_pipeline,
    run_ppocr_on_uploaded_result,
)
from training.finetune_yolo import build_finetune_command, run_finetune

TRAINING_DIR = BASE_DIR / "training"
UPLOADS_DIR = BASE_DIR / "uploads"
SAMPLE_PDFS_DIR = BASE_DIR / "sample_pdfs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._\-() ]+", "_", Path(name).name).strip()
    return cleaned or "uploaded_file"


class MiniToolHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write(f"[MiniTool] {self.address_string()} - {fmt % args}\n")

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith("/assets/"):
            assets_root = (BASE_DIR / "assets").resolve()
            asset_path = (BASE_DIR / path.lstrip("/")).resolve()
            if not asset_path.is_relative_to(assets_root) or not asset_path.is_file():
                self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return
            data = asset_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
            return
        if path in ("/", "/index.html"):
            html_path = BASE_DIR / "index.html"
            data = html_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path in ("/checkval", "/checkval_review.html", "/CheckVal/review_stack_tol.html"):
            review_path = TRAINING_DIR / "CheckVal" / "review_stack_tol.html"
            if not review_path.is_file():
                self._send_json({"error": "Chưa tạo review_stack_tol.html"}, status=HTTPStatus.NOT_FOUND)
                return
            data = review_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/preview_crops.html":
            preview_path = TRAINING_DIR / "synthetic_3regions_dataset" / "preview_crops.html"
            if not preview_path.is_file():
                self._send_json({"error": "Chưa tạo preview_crops.html"}, status=HTTPStatus.NOT_FOUND)
                return
            data = preview_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/yolo_3regions_finetune_pack.zip":
            zip_path = TRAINING_DIR / "yolo_3regions_finetune_pack.zip"
            if not zip_path.is_file():
                self._send_json({"error": "Chưa tạo yolo_3regions_finetune_pack.zip"}, status=HTTPStatus.NOT_FOUND)
                return
            data = zip_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="yolo_3regions_finetune_pack.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/train_3regions_finetune.ipynb":
            nb_path = TRAINING_DIR / "train_3regions_finetune.ipynb"
            if not nb_path.is_file():
                self._send_json({"error": "Chưa tạo train_3regions_finetune.ipynb"}, status=HTTPStatus.NOT_FOUND)
                return
            data = nb_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ipynb+json; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="train_3regions_finetune.ipynb"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/ppocrv6_small_rec_cad_pack.zip":
            zip_path = TRAINING_DIR / "ppocrv6_small_rec_cad_pack.zip"
            if not zip_path.is_file():
                self._send_json({"error": "Chưa tạo ppocrv6_small_rec_cad_pack.zip"}, status=HTTPStatus.NOT_FOUND)
                return
            data = zip_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="ppocrv6_small_rec_cad_pack.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/Train_PPOCRv6_Small_Rec_CAD.ipynb":
            nb_path = TRAINING_DIR / "Train_PPOCRv6_Small_Rec_CAD.ipynb"
            if not nb_path.is_file():
                self._send_json({"error": "Chưa tạo Train_PPOCRv6_Small_Rec_CAD.ipynb"}, status=HTTPStatus.NOT_FOUND)
                return
            data = nb_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ipynb+json; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="Train_PPOCRv6_Small_Rec_CAD.ipynb"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/preview_ppocrv6_small_cad.png":
            img_path = TRAINING_DIR / "preview_ppocrv6_small_cad.png"
            if not img_path.is_file():
                self._send_json({"error": "Chưa tạo preview_ppocrv6_small_cad.png"}, status=HTTPStatus.NOT_FOUND)
                return
            data = img_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/api/models":
            self._send_json(list_available_models())
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_json_body()

            if path == "/api/upload-pdf":
                filename = safe_filename(body.get("filename") or "drawing.pdf")
                b64_data = body.get("dataBase64") or ""
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64_data)
                dest = UPLOADS_DIR / filename
                dest.write_bytes(raw_bytes)

                total_pages = 1
                if dest.suffix.lower() == ".pdf":
                    doc = fitz.open(str(dest))
                    total_pages = len(doc)
                    doc.close()

                self._send_json(
                    {
                        "status": "ok",
                        "filename": dest.name,
                        "sourceType": "upload",
                        "totalPages": total_pages,
                        "sizeKb": round(len(raw_bytes) / 1024, 1),
                    }
                )
                return

            if path == "/api/upload-model":
                target_slot = body.get("slot")  # 'yolo_onnx' | 'ppocr_onnx' | 'ppocr_dict' | 'pt_model'
                filename = safe_filename(body.get("filename") or "")
                b64_data = body.get("dataBase64") or ""
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64_data)

                if target_slot == "yolo_onnx" and filename.lower().endswith(".onnx"):
                    dest = YOLO_ONNX_DIR / filename
                elif target_slot == "ppocr_onnx" and filename.lower().endswith(".onnx"):
                    dest = PPOCR_ONNX_DIR / filename
                elif target_slot == "ppocr_dict" and filename.lower().endswith(".txt"):
                    dest = PPOCR_ONNX_DIR / filename
                elif target_slot == "pt_model" and filename.lower().endswith(".pt"):
                    dest = PT_LIB_DIR / filename
                else:
                    raise ValueError(f"Slot '{target_slot}' hoặc đuôi file '{filename}' không hợp lệ.")

                dest.write_bytes(raw_bytes)
                self._send_json(
                    {
                        "status": "ok",
                        "savedAs": dest.name,
                        "slot": target_slot,
                        "models": list_available_models(),
                    }
                )
                return

            if path == "/api/run":
                source_type = body.get("sourceType", "sample")
                filename = Path(body.get("filename") or "").name
                if not filename:
                    raise ValueError("Hãy chọn hoặc upload một bản vẽ PDF trước khi chạy.")

                file_path = (UPLOADS_DIR / filename) if source_type == "upload" else (SAMPLE_PDFS_DIR / filename)
                if not file_path.is_file():
                    # Try fallback in either directory
                    alt = SAMPLE_PDFS_DIR / filename if source_type == "upload" else UPLOADS_DIR / filename
                    if alt.is_file():
                        file_path = alt
                    else:
                        raise FileNotFoundError(f"Không tìm thấy bản vẽ: {filename}")

                result = run_full_pipeline(
                    pdf_or_img_path=str(file_path),
                    yolo_model_name=str(body.get("yoloModel") or "yolo_best_1class.onnx"),
                    ppocr_model_name=str(body.get("ppocrModel") or DEFAULT_PPOCR_MODEL),
                    dict_name=body.get("dictName") or None,
                    page_index=int(body.get("pageIndex", 0)),
                    dpi=int(body.get("dpi", 250)),
                    rotation=int(body.get("rotation", 0)),
                    conf_thres=float(body.get("confThres", 0.30)),
                    preprocess_mode=str(body.get("preprocessMode", "raw")),
                    clean_bg=bool(body.get("cleanBg", True)),
                    enable_stack_split=bool(body.get("enableStackSplit", True)),
                )
                self._send_json(result)
                return

            if path == "/api/upload-result":
                result_data = body.get("resultData") or {}
                source_type = body.get("sourceType", "sample")
                filename = Path(body.get("filename") or result_data.get("fileName") or "").name
                file_path = None
                if filename:
                    cand1 = (UPLOADS_DIR / filename) if source_type == "upload" else (SAMPLE_PDFS_DIR / filename)
                    cand2 = (SAMPLE_PDFS_DIR / filename) if source_type == "upload" else (UPLOADS_DIR / filename)
                    if cand1.is_file():
                        file_path = str(cand1)
                    elif cand2.is_file():
                        file_path = str(cand2)

                enriched = run_ppocr_on_uploaded_result(
                    result_data=result_data,
                    pdf_or_img_path=file_path,
                    ppocr_model_name=str(body.get("ppocrModel") or DEFAULT_PPOCR_MODEL),
                    dict_name=body.get("dictName") or None,
                    preprocess_mode=str(body.get("preprocessMode", "raw")),
                    enable_stack_split=bool(body.get("enableStackSplit", True)),
                    yolo_model_name=body.get("yoloModel"),
                )
                self._send_json(enriched)
                return

            if path == "/api/finetune":
                pt_name = str(body.get("ptModel") or "best.pt")
                dataset = str(body.get("dataset") or (BASE_DIR.parent / "ready-to-train-priority"))
                epochs = int(body.get("epochs", 80))
                imgsz = int(body.get("imgsz", 1024))
                batch = int(body.get("batch", 4))
                smoke_only = bool(body.get("smokeOnly", True))
                execute_now = bool(body.get("executeNow", False))

                cli_cmd = build_finetune_command(pt_name, dataset, epochs, imgsz, batch, smoke_only)
                if not execute_now:
                    self._send_json({"status": "command_ready", "command": cli_cmd})
                    return

                train_res = run_finetune(pt_name, dataset, epochs, imgsz, batch, smoke_only)
                train_res["command"] = cli_cmd
                train_res["models"] = list_available_models()
                self._send_json(train_res)
                return

            if path == "/api/checkval/save":
                from build_checkval_review import save_checkval_label
                res = save_checkval_label(
                    crop_name=str(body.get("cropName") or "crop_custom.jpg"),
                    image_data_uri=str(body.get("imageDataUri") or ""),
                    boxes=list(body.get("boxes") or []),
                    rotation=int(body.get("rotation", 0)),
                )
                self._send_json(res)
                return

            if path == "/api/checkval/delete":
                from build_checkval_review import delete_checkval_label
                res = delete_checkval_label(str(body.get("cropName") or ""))
                self._send_json(res)
                return

            if path == "/api/checkval/rebuild-zip":
                from build_training_zip import build_zip
                res = build_zip()
                self._send_json(res)
                return

            self._send_json({"error": f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def run_server(port: int = 58080) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), MiniToolHandler)
    print(f"============================================================")
    print(f" Mini PDF + YOLO (.onnx) + PP-OCR (.onnx) Tester")
    print(f" URL: http://127.0.0.1:{port}")
    print(f"============================================================")
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=58080)
    args = parser.parse_args()
    run_server(args.port)
