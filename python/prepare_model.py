"""Make one model the server does not have yet, into its store.

    python -B prepare_model.py yolo26s-seg
    python -B prepare_model.py llmlingua2-small --precision int8
    python -B prepare_model.py mt5-xlsum

Run by `@youni/onnxweb-server` when a page asks for a model its cache lacks
(`/models/prepare`); prints one line per step, the last one `ready <entry>`
or `error <why>`. Recipes:

- YOLO26 (`yolo26{n,s,m,l,x}[-seg|-sem|-depth|-cls|-pose|-obb|-reid]`): the
  exports of `cache_models.py` (dynamic size, raw head, float32 and float16).
- Models with no published ONNX the libraries know by name (the pruner's
  LLMLingua-2, summarizers, correctors): exported with Optimum from their
  Hugging Face checkpoint, float32, and int8 by dynamic quantization; the
  tokenizer and configs beside; the manifest written with every graph's
  digest and the decoder's KV geometry.

- A language model published by onnx-community (`onnx-community/<repo>`,
  precision `int4_float16`, `int4`, `int8` or `float16` — the variants
  `q4f16`, `q4`, `quantized`, `fp16`): its graph, tokenizer, configs and chat
  template, taken from the Hugging Face cache (downloaded once otherwise).

Every entry has its external weights written in files of at most
`--chunk-mb` (1024) MB: a browser then never holds one buffer of several GB
for a model (some refuse an ArrayBuffer past 2 GB) and the weights reach the
runtime file by file.

Needs `ultralytics` for YOLO, `optimum-onnx` and `transformers` for the
Optimum recipes, `huggingface_hub` and `onnx` for the language models: the
requirement sets in `requirements/`, which `@youni/onnxweb-server` installs
into an environment of its own on first need.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))

import cache_models  # noqa: E402  (the YOLO exports and the cache layout)

ROOT = cache_models.ROOT

# Catalogue name → Hugging Face checkpoint, Optimum task, architecture.
HF_RECIPES: dict[str, tuple[str, str, str]] = {
    "llmlingua2-small": ("microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", "token-classification", "bert"),
    "llmlingua2-large": ("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", "token-classification", "xlm-roberta"),
    "mt5-xlsum": ("csebuetnlp/mT5_multilingual_XLSum", "text2text-generation-with-past", "mt5"),
    "barthez-orangesum": ("moussaKam/barthez-orangesum-abstract", "text2text-generation-with-past", "bart"),
    "kredor-punctuate-all": ("kredor/punctuate-all", "token-classification", "xlm-roberta"),
    "flan-t5-large-grammar": ("pszemraj/flan-t5-large-grammar-synthesis", "text2text-generation-with-past", "t5"),
    # The embedder's French model: the encoder's hidden states, mean-pooled by the page.
    "solon-embeddings-large": ("OrdalieTech/Solon-embeddings-large-0.1", "feature-extraction", "xlm-roberta"),
    # The seq2seq reranker: one decoder step from the start token, so no KV cache.
    "monot5-base": ("castorini/monot5-base-msmarco", "text2text-generation", "t5"),
}

ROLES = [
    ("encoder_model", "encoder"),
    ("decoder_model_merged", "decoder_merged"),
    ("decoder_with_past_model", "decoder_with_past"),
    ("decoder_model", "decoder"),
    ("model", "single"),
]
CHUNK_MB = 1024
SIDE_FILES = ["tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json", "special_tokens_map.json", "spiece.model", "sentencepiece.bpe.model"]


def say(line: str) -> None:
    print(line, flush=True)


def entry_dir(name: str, precision: str) -> Path:
    return ROOT / f"{name.replace('/', '--').replace('_', '-')}_{precision}"


def kv_geometry(graph) -> dict | None:
    """Heads, head size and layers of a decoder's `past_key_values.*` inputs."""
    pasts = [i for i in graph.graph.input if i.name.startswith("past_key_values.")]
    if not pasts:
        return None
    dims = [d.dim_value or None for d in pasts[0].type.tensor_type.shape.dim]
    layers = len({re.match(r"past_key_values\.(\d+)\.", i.name).group(1) for i in pasts if re.match(r"past_key_values\.(\d+)\.", i.name)})
    cross = any(".encoder." in i.name for i in pasts)
    return {"num_layers": layers, "num_heads": dims[1] or 1, "head_dim": dims[3] or 1, "has_cross_attention": cross, "dtype": "float32"}


def write_manifest(directory: Path, name: str, precision: str, architecture: str) -> None:
    import onnx

    graphs = []
    kv = None
    for stem, role in ROLES:
        file = directory / f"{stem}.onnx"
        if not file.exists():
            continue
        if role == "decoder" and (directory / "decoder_model_merged.onnx").exists():
            file.unlink()  # the merged decoder replaces the pair
            (directory / "decoder_with_past_model.onnx").unlink(missing_ok=True)
            continue
        model = onnx.load(str(file), load_external_data=False)
        initializers = {i.name for i in model.graph.initializer}
        sidecars = sorted({e.value for t in model.graph.initializer for e in t.external_data if e.key == "location"})
        graphs.append(
            {
                "filename": file.name,
                "role": role,
                "sha256": cache_models.sha256(file),
                "size_mb": round(file.stat().st_size / 1e6, 3),
                "inputs": [i.name for i in model.graph.input if i.name not in initializers],
                "outputs": [o.name for o in model.graph.output],
                "external_data": sidecars,
            }
        )
        if role.startswith("decoder") and kv is None:
            kv = kv_geometry(model)
    manifest = {
        "model_key": f"{name}_{precision}",
        "compute_type": precision,
        "architecture": architecture,
        "graphs": graphs,
        "tokenizer_file": "tokenizer.json" if (directory / "tokenizer.json").exists() else None,
        "logits_vocab_size": None,
        "kv": kv,
        "exposes_cross_qk": False,
        "extra": {"source": "optimum"},
        "version": 2,
    }
    (directory / "ynonnx_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


# The LLM variants as onnx-community names its graphs, by compute type.
LLM_VARIANTS = {"int4_float16": "q4f16", "int4": "q4", "int8": "quantized", "float16": "fp16"}
LLM_SIDE_FILES = ["tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json", "special_tokens_map.json", "chat_template.jinja"]


def split_external(directory: Path, chunk_mb: int) -> None:
    """Every graph's external weights rewritten in files of at most `chunk_mb` MB."""
    import onnx
    from onnx.external_data_helper import set_external_data

    for graph_file in directory.glob("*.onnx"):
        head = onnx.load(str(graph_file), load_external_data=False)
        sidecars = sorted({e.value for t in head.graph.initializer for e in t.external_data if e.key == "location"})
        sizes = [(directory / s).stat().st_size for s in sidecars if (directory / s).exists()] or [0]
        # Already in pieces small enough, or a graph small enough to stay whole.
        if max(sizes) <= chunk_mb * 2**20 and graph_file.stat().st_size <= chunk_mb * 2**20:
            continue
        say(f"step splitting the weights of {graph_file.name} into files of {chunk_mb} MB at most")
        model = onnx.load(str(graph_file), load_external_data=True)
        limit = chunk_mb * 2**20
        index, written, handle = 0, 0, None
        names = []
        for tensor in model.graph.initializer:
            raw = tensor.raw_data
            if len(raw) < 1024:
                continue  # small tensors stay in the graph
            if handle is None or (written > 0 and written + len(raw) > limit):
                if handle:
                    handle.close()
                name = f"{graph_file.name}_data_{index}"
                names.append(name)
                handle = open(directory / f"{name}.tmp", "wb")
                index, written = index + 1, 0
            set_external_data(tensor, location=names[-1], offset=written, length=len(raw))
            handle.write(raw)
            written += len(raw)
            tensor.ClearField("raw_data")
            tensor.data_location = onnx.TensorProto.EXTERNAL
        if handle:
            handle.close()
        for old in sidecars:
            (directory / old).unlink(missing_ok=True)
        for name in names:
            (directory / f"{name}.tmp").replace(directory / name)
        onnx.save(model, str(graph_file))
        say(f"step {graph_file.name}: {len(names)} weight files")


def export_llm(repo: str, precision: str) -> Path:
    """An onnx-community language model at one precision, its weights in chunks."""
    from huggingface_hub import hf_hub_download, list_repo_files

    variant = LLM_VARIANTS.get(precision)
    if not variant:
        raise SystemExit(f"error precision '{precision}' is not one of {', '.join(LLM_VARIANTS)}")
    stem = f"model_{variant}"
    files = list_repo_files(repo)
    graph = f"onnx/{stem}.onnx"
    if graph not in files:
        raise SystemExit(f"error {repo} publishes no {graph}")
    target = entry_dir(repo, precision)
    target.mkdir(parents=True, exist_ok=True)
    for name in [graph, *[f for f in files if f.startswith(f"onnx/{stem}.onnx_data")]]:
        say(f"step fetching {name}")
        shutil.copy(hf_hub_download(repo, name), target / Path(name).name)
    for name in LLM_SIDE_FILES:
        if name in files:
            shutil.copy(hf_hub_download(repo, name), target / name)
    (target / f"{stem}.onnx").replace(target / "model.onnx")
    for sidecar in target.glob(f"{stem}.onnx_data*"):
        sidecar.replace(target / sidecar.name.replace(stem, "model"))
    import onnx

    # The graph names its weight files: point it at the renamed ones.
    model = onnx.load(str(target / "model.onnx"), load_external_data=False)
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = entry.value.replace(stem, "model")
    onnx.save(model, str(target / "model.onnx"))
    return target


def export_hf(name: str, precision: str) -> Path:
    hf_id, task, architecture = HF_RECIPES[name]
    from optimum.exporters.onnx import main_export
    from transformers import AutoTokenizer

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "export"
        say(f"step exporting {hf_id} ({task}) with Optimum")
        main_export(hf_id, output=out, task=task, opset=17, do_validation=False)
        # A fast tokenizer's tokenizer.json, which the page reads.
        try:
            AutoTokenizer.from_pretrained(hf_id).save_pretrained(out)
        except Exception as error:  # noqa: BLE001 - some checkpoints ship one already
            say(f"step tokenizer kept as exported ({type(error).__name__})")
        for precision_out in ("float32", "int8"):
            target = entry_dir(name, precision_out)
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            for file in out.iterdir():
                if file.suffix == ".onnx" or file.name.endswith(".onnx_data"):
                    continue
                if file.name in SIDE_FILES or file.suffix in (".json", ".model"):
                    shutil.copy(file, target / file.name)
            for graph in out.glob("*.onnx"):
                if precision_out == "float32":
                    shutil.copy(graph, target / graph.name)
                    for sidecar in out.glob(f"{graph.name}_data"):
                        shutil.copy(sidecar, target / sidecar.name)
                else:
                    say(f"step quantizing {graph.name} to int8")
                    from onnxruntime.quantization import QuantType, quantize_dynamic

                    quantize_dynamic(str(graph), str(target / graph.name), weight_type=QuantType.QInt8)
            split_external(target, CHUNK_MB)
            write_manifest(target, name, precision_out, architecture)
            say(f"step wrote {target.name}")
    return entry_dir(name, precision)


def prepare(name: str, precision: str) -> Path:
    yolo = re.fullmatch(r"yolo26([nslmx])(-(?:seg|sem|depth|cls|pose|obb|reid))?", name)
    if yolo:
        size, suffix = yolo.group(1), yolo.group(2) or ""
        if suffix == "-reid":
            cache_models.export_yolo([size], False, only={f"yolo26{size}-cls"})
            cache_models.export_reid(size, False)
        else:
            cache_models.export_yolo([size], False, only={name})
        return entry_dir(name, precision)
    if name in HF_RECIPES:
        return export_hf(name, precision)
    if name.startswith("onnx-community/"):
        target = export_llm(name, precision)
        split_external(target, CHUNK_MB)
        write_manifest(target, name, precision, "decoder-only")
        return target
    raise SystemExit(f"error no recipe for '{name}' (YOLO26 names, onnx-community/<repo>, and {', '.join(HF_RECIPES)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--chunk-mb", type=int, default=1024, help="the largest weight file written (MB)")
    args = parser.parse_args()
    global CHUNK_MB
    CHUNK_MB = args.chunk_mb
    try:
        entry = prepare(args.name, args.precision)
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - reported to the page as it is
        say(f"error {type(error).__name__}: {error}")
        raise SystemExit(1) from error
    if not (entry / "ynonnx_manifest.json").exists():
        say(f"error {entry.name} was not produced")
        raise SystemExit(1)
    say(f"ready {entry.name}")


if __name__ == "__main__":
    main()
