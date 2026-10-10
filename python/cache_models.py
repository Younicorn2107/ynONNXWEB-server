"""A local model store for the browser libraries and their tests.

    python -B cache_models.py                # YOLO26 n, every task, + the common Hub models
    python -B cache_models.py --sizes n,s    # more YOLO sizes
    python -B cache_models.py --no-hub       # YOLO exports only

Two halves:

1. **YOLO**: Ultralytics' own weights exported to ONNX for every task —
   detect, segment, semantic, depth, classify, pose, obb — in float32 and
   float16 — any input size (dynamic height and width), the raw head (NMS
   on the page) — written as entries of the store `@youni/onnxweb-server` serves at
   `/models/` (`<name>_<precision>/model.onnx` + `ynonnx_manifest.json`),
   which is the layout `ynonnx_core` and `serverResolver` read. The Hub
   publishes ONNX for detect and pose only; this is how the other tasks get
   tested at all.
2. **Hub**: the published repositories the libraries load, fetched into this
   machine's Hugging Face cache — the server's `/hf/` mirror (a page's
   `endpoint` pointing at it) serves them from there instead of downloading
   them again.

Needs `ultralytics`, `onnx`, `onnxconverter_common`, `huggingface_hub`
(the `yolo` requirement set). Ultralytics and its weights are AGPL-3.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(os.environ.get("YNONNXWEB_MODELS", Path.home() / ".cache" / "ynonnxweb" / "models"))

# Task suffix → input size the model was trained at.
TASKS = {"": 640, "-seg": 640, "-sem": 640, "-depth": 640, "-cls": 224, "-pose": 640, "-obb": 1024}

# What the libraries load from the Hub (lightest useful precisions).
HUB = {
    "onnx-community/yolo26n-ONNX": ["onnx/model_fp16.onnx", "onnx/model.onnx", "*.json"],
    "onnx-community/yolo26n-pose-ONNX": ["onnx/model_fp16.onnx", "onnx/model.onnx", "*.json"],
    "PaddlePaddle/PP-OCRv6_tiny_det_onnx": ["*"],
    "PaddlePaddle/PP-OCRv6_tiny_rec_onnx": ["*"],
    "PaddlePaddle/PP-OCRv6_small_det_onnx": ["*"],
    "PaddlePaddle/PP-OCRv6_small_rec_onnx": ["*"],
    "Xenova/opus-mt-en-fr": ["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "onnx/*quantized*", "*.json", "*.spm"],
    "Xenova/opus-mt-fr-en": ["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "onnx/*quantized*", "*.json", "*.spm"],
    "onnx-community/distilbart-cnn-12-6-ONNX": ["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "onnx/*quantized*", "*.json", "*.txt"],
    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase": ["*.onnx", "*.yaml", "*.model", "*.json"],
    "Xenova/bge-small-en-v1.5": ["onnx/model.onnx", "onnx/model_quantized.onnx", "*.json"],
    "Xenova/bge-reranker-base": ["onnx/model.onnx", "onnx/model_quantized.onnx", "*.json"],
    "onnx-community/Qwen3-0.6B-ONNX": ["onnx/model_q4f16*", "onnx/model_q4.*", "onnx/model_q4.onnx_data*", "*.json", "*.jinja"],
    "Qdrant/clip-ViT-B-32-text": ["*"],
    "onnx-community/whisper-tiny": ["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "*.json"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_entry(name: str, precision: str, onnx_path: Path, architecture: str = "yolo") -> Path:
    """One cache entry: the graph and the manifest that describes it."""
    import onnx

    entry = ROOT / f"{name.replace('/', '--').replace('_', '-')}_{precision}"
    entry.mkdir(parents=True, exist_ok=True)
    target = entry / "model.onnx"
    target.write_bytes(onnx_path.read_bytes())
    model = onnx.load(str(target), load_external_data=False)
    initializers = {i.name for i in model.graph.initializer}
    manifest = {
        "model_key": f"{name}_{precision}",
        "compute_type": precision,
        "architecture": architecture,
        "graphs": [
            {
                "filename": "model.onnx",
                "role": "single",
                "sha256": sha256(target),
                "size_mb": round(target.stat().st_size / 1e6, 3),
                "inputs": [i.name for i in model.graph.input if i.name not in initializers],
                "outputs": [o.name for o in model.graph.output],
                "external_data": [],
            }
        ],
        "tokenizer_file": None,
        "logits_vocab_size": None,
        "kv": None,
        "exposes_cross_qk": False,
        "extra": {"source": "ultralytics", "metadata": {p.key: p.value for p in model.metadata_props}},
        "version": 2,
    }
    (entry / "ynonnx_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return entry


def export_yolo(sizes: list[str], force: bool, only: set[str] | None = None) -> None:
    import onnx
    from onnxconverter_common import float16
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as tmp:
        for size in sizes:
            for suffix, imgsz in TASKS.items():
                name = f"yolo26{size}{suffix}"
                if only is not None and name not in only:
                    continue
                if not force and (ROOT / f"{name}_float16" / "ynonnx_manifest.json").exists():
                    print(f"{name}: cached")
                    continue
                try:
                    model = YOLO(str(Path(tmp) / f"{name}.pt"))
                    # Dynamic height and width: the page picks the input size per call
                    # (`imgsz`, `rect`). The raw head, NMS done by the page: the
                    # end-to-end head ends in a TopK, which onnxruntime-web runs on the
                    # CPU in mid-graph — a round trip per frame, and no graph capture.
                    exported = Path(model.export(format="onnx", imgsz=imgsz, device="cpu", simplify=True, dynamic=True, opset=18))
                except Exception as error:  # noqa: BLE001 - one task failing must not stop the others
                    print(f"{name}: export failed — {type(error).__name__}: {error}")
                    continue
                write_entry(name, "float32", exported)
                # float16 weights and activations, float32 inputs and outputs (what the page feeds).
                half_path = Path(tmp) / f"{name}_fp16.onnx"
                try:
                    # ONNX Runtime's converter first: onnxconverter_common's leaves outputs
                    # whose declared type no longer matches (sessions refuse them).
                    try:
                        from onnxruntime.transformers.float16 import convert_float_to_float16

                        from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST

                        # Split stays float32: WebGPU has no float16 Split, and one CPU
                        # node made the whole graph 4x slower (and uncapturable).
                        half = convert_float_to_float16(onnx.load(str(exported)), keep_io_types=True, op_block_list=[*DEFAULT_OP_BLOCK_LIST, "Split"])
                    except Exception:  # noqa: BLE001
                        half = float16.convert_float_to_float16(onnx.load(str(exported)), keep_io_types=True)
                    onnx.save(half, str(half_path))
                    write_entry(name, "float16", half_path)
                    print(f"{name}: float32 {exported.stat().st_size / 1e6:.1f} MB, float16 {half_path.stat().st_size / 1e6:.1f} MB")
                except Exception as error:  # noqa: BLE001
                    print(f"{name}: float32 {exported.stat().st_size / 1e6:.1f} MB, float16 failed — {type(error).__name__}: {error}")


def export_reid(size: str, force: bool) -> None:
    """
    An appearance model for re-identification: the classifier cut before its
    last layer, its pooled features (1280 values) as the embedding — what
    Ultralytics' BoT-SORT uses when given a `-cls` model.
    """
    import onnx

    name = f"yolo26{size}-reid"
    source = ROOT / f"yolo26{size}-cls_float32" / "model.onnx"
    if not source.exists():
        print(f"{name}: needs yolo26{size}-cls exported first")
        return
    if not force and (ROOT / f"{name}_float32" / "ynonnx_manifest.json").exists():
        print(f"{name}: cached")
        return
    model = onnx.load(str(source))
    gemm = next(n for n in reversed(model.graph.node) if n.op_type == "Gemm")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{name}.onnx"
        full = Path(tmp) / "full.onnx"
        onnx.save(model, str(full))
        onnx.utils.extract_model(str(full), str(out), [model.graph.input[0].name], [gemm.input[0]])
        cut = onnx.load(str(out))
        meta = {p.key: p.value for p in model.metadata_props}
        for key in ("imgsz", "stride", "author", "license", "version"):
            if key in meta:
                cut.metadata_props.add(key=key, value=meta[key])
        cut.metadata_props.add(key="task", value="reid")
        onnx.save(cut, str(out))
        write_entry(name, "float32", out, architecture="reid")
        print(f"{name}: float32 {out.stat().st_size / 1e6:.1f} MB")


def fetch_hub() -> None:
    from huggingface_hub import snapshot_download

    for repo, patterns in HUB.items():
        try:
            path = snapshot_download(repo, allow_patterns=patterns)
            print(f"{repo}: {path}")
        except Exception as error:  # noqa: BLE001
            print(f"{repo}: failed — {type(error).__name__}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="n", help="YOLO26 sizes, comma-separated (n, s, m, l, x)")
    parser.add_argument("--no-hub", action="store_true", help="skip the Hub downloads")
    parser.add_argument("--force", action="store_true", help="export again even when cached")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"cache: {ROOT}")
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    export_yolo(sizes, args.force)
    for size in sizes:
        export_reid(size, args.force)
    if not args.no_hub:
        fetch_hub()


if __name__ == "__main__":
    main()
