"""
Fine-tune YOLO (.pt) checkpoint from models/pt_library/ on CAD datasets
and automatically export the resulting best.onnx (nms=False) into models/onnx_yolo/.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, Dict

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
PT_LIB_DIR = BASE_DIR / "models" / "pt_library"
YOLO_ONNX_DIR = BASE_DIR / "models" / "onnx_yolo"


def build_finetune_command(
    pt_name: str,
    dataset_path: str,
    epochs: int = 80,
    imgsz: int = 1024,
    batch: int = 4,
    smoke_only: bool = False,
) -> str:
    cmd = (
        f'python finetune_yolo.py --weights "{pt_name}" '
        f'--dataset "{dataset_path}" --epochs {int(epochs)} '
        f'--imgsz {int(imgsz)} --batch {int(batch)}'
    )
    if smoke_only:
        cmd += " --smoke-only"
    return cmd


def run_finetune(
    weights_name_or_path: str,
    dataset_dir: str,
    epochs: int = 80,
    imgsz: int = 1024,
    batch: int = 4,
    smoke_only: bool = False,
) -> Dict[str, Any]:
    weights_path = Path(weights_name_or_path)
    if not weights_path.is_file():
        weights_path = PT_LIB_DIR / weights_path.name
    if not weights_path.is_file() or weights_path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"Không tìm thấy file checkpoint .pt: {weights_path}")

    ds_root = Path(dataset_dir).resolve()
    yaml_path = ds_root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy data.yaml trong dataset: {ds_root}")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Môi trường hiện tại chưa cài torch + ultralytics. "
            "Hãy chạy lệnh: pip install torch ultralytics hoặc dùng lệnh CLI trên máy có GPU/Colab."
        ) from exc

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg["path"] = ds_root.as_posix()
    runtime_yaml = ds_root / "training-runtime.yaml"
    runtime_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    device = 0 if torch.cuda.is_available() else "cpu"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = BASE_DIR / "finetune_runs"

    train_kwargs = dict(
        data=str(runtime_yaml),
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
        workers=2,
        project=str(runs_dir),
        resume=False,
        single_cls=False,
        agnostic_nms=False,
        # CAD drawing safe augmentations: never flip text upside down or mirror digits
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
    )

    actual_epochs = 1 if smoke_only else int(epochs)
    run_name = f"{'smoke' if smoke_only else 'finetune'}_{weights_path.stem}_{run_id}"

    model = YOLO(str(weights_path))
    model.train(epochs=actual_epochs, patience=25, name=run_name, **train_kwargs)

    best_pt = Path(model.trainer.best)
    exported_pt = PT_LIB_DIR / f"finetuned_{run_id}.pt"
    shutil.copy2(best_pt, exported_pt)

    best_model = YOLO(str(best_pt))
    onnx_out = Path(best_model.export(format="onnx", imgsz=int(imgsz), dynamic=False, nms=False, device="cpu"))
    target_onnx = YOLO_ONNX_DIR / f"yolo_finetuned_{run_id}.onnx"
    shutil.copy2(onnx_out, target_onnx)

    return {
        "status": "ok",
        "device": str(device),
        "epochs": actual_epochs,
        "savedPt": exported_pt.name,
        "savedOnnx": target_onnx.name,
        "runDir": str(runs_dir / run_name),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="best.pt", help="Name in models/pt_library or path to .pt")
    parser.add_argument("--dataset", default=str(BASE_DIR.parent / "ready-to-train-priority"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    res = run_finetune(
        args.weights,
        args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        smoke_only=args.smoke_only,
    )
    print(res)


if __name__ == "__main__":
    main()
